"""T6 SAM 3D smoke test (spec Section 7): input export for the GPU run, and the analysis.

Usage:
    uv run python -m video3d.data.generation.t6 export --split eval --num-clips 20 --out t6/inputs
    uv run python -m video3d.data.generation.t6 analyze --outputs t6/outputs --report t6/report.json

The export holds, per clip, frame 0's RGB, one visible mask per object and a GT pointmap in
SAM 3D's input convention, so the GPU run (scripts/t6/run_sam3d.py) needs no code from this
package. The analysis follows the paper's protocol (App. D.3) with 100k instead of 1M points.
"""

from __future__ import annotations
import typing

import fire
import itertools
import json
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh
from PIL import Image
from scipy.spatial import ConvexHull, cKDTree

from .clipio import Clip
from .conventions import object_pose_in_camera
from .sam3d import PYTORCH3D_FROM_OPENCV, quat_wxyz_to_matrix, sam3d_posed_points
from .validate import load_canonical_mesh, sample_surface

N_POINTS = 100_000
VARIANTS = ("gt_pointmap", "moge")
AXIS_ROTATIONS = [
    M for M in (np.diag(sg) @ np.eye(3)[list(p)] for p in itertools.permutations(range(3))
                for sg in itertools.product([1, -1], repeat=3))
    if np.isclose(np.linalg.det(M), 1)
]
IDENTITY = next(i for i, M in enumerate(AXIS_ROTATIONS) if np.allclose(M, np.eye(3)))


def pointmap_opencv(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """(H, W, 3) camera-frame points from z-depth; NaN where depth is NaN."""
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    x = (u - K[0, 2]) / K[0, 0] * depth
    y = (v - K[1, 2]) / K[1, 1] * depth
    return np.stack([x, y, depth], axis=-1)


def export_inputs(clips: list[Clip], out: Path, t: int = 0) -> None:
    out.mkdir(parents=True, exist_ok=True)
    index = []
    for clip in clips:
        d = out / clip.path.name
        d.mkdir(exist_ok=True)
        Image.fromarray(clip.rgb(t)).save(d / "image.png")
        visible = clip.visible(t)
        objects = []
        for k in range(1, clip.num_objects + 1):
            mask = visible == k
            Image.fromarray(mask.astype(np.uint8) * 255).save(d / f"mask_{k}.png")
            objects.append({"k": k, "visible_px": int(mask.sum())})
        points = pointmap_opencv(clip.depth(t), clip.K) @ PYTORCH3D_FROM_OPENCV.T
        np.savez_compressed(d / "pointmap.npz", pointmap=points.astype(np.float32))
        index.append({"clip_id": clip.path.name, "frame": t, "objects": objects})
    (out / "index.json").write_text(json.dumps(index, indent=1))


def nn_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance from each point of a to its nearest neighbour in b."""
    return cKDTree(b).query(a, workers=-1)[0]


def to_unit(points: np.ndarray) -> np.ndarray:
    """Center the bounding box and scale its largest side to 2, i.e. into [-1, 1]."""
    lo, hi = points.min(0), points.max(0)
    return (points - (lo + hi) / 2) * (2 / (hi - lo).max())


def icp(src: np.ndarray, dst: np.ndarray, max_dist: float, init: np.ndarray = np.eye(4)) -> np.ndarray:
    """Rigid point-to-point ICP; returns the 4x4 transform taking src onto dst."""
    pc = lambda p: o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p))  # noqa: E731
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
    return o3d.pipelines.registration.registration_icp(
        pc(src), pc(dst), max_dist, init, o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria,
    ).transformation


def apply(T: np.ndarray, p: np.ndarray) -> np.ndarray:
    return p @ T[:3, :3].T + T[:3, 3]


def rotation_angle_deg(R: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def voxel_iou(a: np.ndarray, b: np.ndarray, lo: float, hi: float, res: int = 64) -> float:
    def occ(p: typing.Any) -> typing.Any:
        g = np.zeros((res,) * 3, bool)
        g[tuple(np.clip(((p - lo) / (hi - lo) * res).astype(int), 0, res - 1).T)] = True
        return g
    ga, gb = occ(a), occ(b)
    return float((ga & gb).sum() / (ga | gb).sum())


def shape_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Paper App. D.3.1: both shapes normalized to [-1, 1] independently, ICP-aligned, then compared.

    ICP starts from the best of the 24 axis-aligned rotations, because SAM 3D's canonical yaw is
    its own choice for objects without a clear front; the metric is about shape, not orientation.
    """
    pred, gt = to_unit(pred), to_unit(gt)
    sub = np.random.default_rng(0).choice(len(pred), 20_000, replace=False)
    chamfer = lambda a, b: (nn_dist(a, b).mean() + nn_dist(b, a).mean()) / 2  # noqa: E731
    best = min(AXIS_ROTATIONS, key=lambda M: chamfer(pred[sub] @ M.T, gt[sub]))
    init = np.eye(4)
    init[:3, :3] = best
    pred = apply(icp(pred, gt, 0.2, init), pred)
    d_pg, d_gp = nn_dist(pred, gt), nn_dist(gt, pred)
    precision, recall = (d_pg < 0.01).mean(), (d_gp < 0.01).mean()
    return {
        "f1@0.01": float(2 * precision * recall / max(precision + recall, 1e-12)),
        "chamfer": float((d_pg.mean() + d_gp.mean()) / 2),
        "viou": voxel_iou(pred, gt, -1, 1),
    }


def diameter(points: np.ndarray) -> float:
    hull = points[ConvexHull(points).vertices]
    return float(max(np.linalg.norm(hull - p, axis=1).max() for p in hull))


def layout_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Paper App. D.3.2 on posed points in the same camera frame."""
    d = diameter(gt[np.random.default_rng(0).choice(len(gt), 5000, replace=False)])
    adds = (nn_dist(pred, gt).mean() + nn_dist(gt, pred).mean()) / (2 * d)
    lo, hi = np.maximum(pred.min(0), gt.min(0)), np.minimum(pred.max(0), gt.max(0))
    inter = np.prod(np.clip(hi - lo, 0, None))
    union = np.prod(np.ptp(pred, 0)) + np.prod(np.ptp(gt, 0)) - inter
    T = icp(pred, gt, 0.1 * d)
    return {
        "add_s": float(adds),
        "add_s@0.1": float(adds < 0.1),
        "iou3d": float(inter / union),
        "icp_rot_deg": rotation_angle_deg(T[:3, :3]),
    }


def canonical_check(coords: np.ndarray, voxels: np.ndarray) -> list[float]:
    """IoU of SAM 3D's canonical voxels with ours under each of the 24 axis-aligned rotations,
    without any alignment. Both are dilated by one voxel, since 1-voxel-thick shells barely
    overlap under sub-voxel offsets."""
    from scipy.ndimage import binary_dilation

    sam = np.zeros((64,) * 3, bool)
    sam[tuple(coords.T)] = True
    sam = binary_dilation(sam)
    centers = (np.argwhere(voxels) + 0.5) / 64 - 0.5
    ious = []
    for M in AXIS_ROTATIONS:
        ours = np.zeros((64,) * 3, bool)
        ours[tuple(np.clip(np.floor((centers @ M.T + 0.5) * 64).astype(int), 0, 63).T)] = True
        ours = binary_dilation(ours)
        ious.append(float((sam & ours).sum() / (sam | ours).sum()))
    return ious


def analyze_object(clip: Clip, k: int, out: dict, inputs: Path) -> dict:
    R, t, s = object_pose_in_camera(clip, k, 0)
    gt_mesh = load_canonical_mesh(clip.asset_path(k, "mesh"))
    gt_local, _ = sample_surface(gt_mesh, N_POINTS, seed=0)
    pred_mesh = trimesh.Trimesh(out["mesh_vertices"], out["mesh_faces"], process=False)
    pred_local, _ = sample_surface(pred_mesh, N_POINTS, seed=0)

    q, t_sam, s_sam = out["rotation"][0], out["translation"][0], out["scale"][0]
    gt_cam = s[0] * gt_local @ R.T + t
    pred_cam = sam3d_posed_points(pred_local, q, t_sam, s_sam) @ PYTORCH3D_FROM_OPENCV.T
    rec = {
        "shape": shape_metrics(pred_local, gt_local),
        "layout": layout_metrics(pred_cam, gt_cam),
        "scale_ratio": float(s_sam[0] / s[0]),
        "translation_err_rel": float(np.linalg.norm(PYTORCH3D_FROM_OPENCV @ t_sam - t) / np.linalg.norm(t)),
    }
    if "pointmap_used" in out:
        # MoGe's pointmap has its own (metric) scale; align it to ours by the median depth over
        # the object's visible pixels, which uses GT depth but no GT pose.
        mask = np.asarray(Image.open(inputs / clip.path.name / f"mask_{k}.png")) > 0
        z_gt = clip.depth(0)[mask]
        z_moge = out["pointmap_used"][..., 2].astype(np.float32)[mask]
        ok = np.isfinite(z_gt) & np.isfinite(z_moge) & (z_moge > 0)
        rec["moge_depth_scale"] = float(np.median(z_gt[ok]) / np.median(z_moge[ok]))
        rec["layout_scale_aligned"] = layout_metrics(pred_cam * rec["moge_depth_scale"], gt_cam)

    # Rotation taking our canonical frame into SAM 3D's predicted one; identity up to the yaw
    # SAM 3D chooses itself if the converter's object-frame part is right.
    M = quat_wxyz_to_matrix(q) @ PYTORCH3D_FROM_OPENCV @ R
    rec["sam_local_from_ours_up"] = (M @ [0, 0, 1.0]).tolist()
    ious = canonical_check(out["coords_original"][:, 1:], np.load(clip.asset_path(k, "voxels")))
    rec["canonical_iou_identity"] = ious[IDENTITY]
    rec["canonical_iou_rank_identity"] = int(sorted(ious, reverse=True).index(ious[IDENTITY]) + 1)
    rec["canonical_ious"] = ious
    return rec


def summarize(records: list[dict]) -> dict:
    summary = {}
    for variant in VARIANTS:
        rs = [r for r in records if r["variant"] == variant]
        mean = lambda key, sub: float(np.mean([r[key][sub] for r in rs]))  # noqa: E731
        s = {
            "n": len(rs),
            "shape": {m: mean("shape", m) for m in ("f1@0.01", "chamfer", "viou")},
            "layout": {m: mean("layout", m) for m in ("add_s", "add_s@0.1", "iou3d", "icp_rot_deg")},
            "scale_ratio_median": float(np.median([r["scale_ratio"] for r in rs])),
            "translation_err_rel_median": float(np.median([r["translation_err_rel"] for r in rs])),
        }
        if variant == "moge":
            s["layout_scale_aligned"] = {m: mean("layout_scale_aligned", m)
                                         for m in ("add_s", "add_s@0.1", "iou3d", "icp_rot_deg")}
        ious = np.array([r["canonical_ious"] for r in rs])
        s["canonical"] = {
            "mean_iou_identity": float(ious[:, IDENTITY].mean()),
            "mean_iou_best_other": float(np.delete(ious.mean(0), IDENTITY).max()),
            "identity_rank_of_mean": int(np.argsort(-ious.mean(0)).tolist().index(IDENTITY) + 1),
            "up_axis_is_z": float(np.mean([r["sam_local_from_ours_up"][2] > 0.9 for r in rs])),
        }
        summary[variant] = s
    return summary


def analyze(outputs: Path, dataset: Path, inputs: Path, split: str = "eval") -> dict:
    records = []
    for path in sorted(outputs.glob("*/obj*_*.npz")):
        clip_id = path.parent.name
        k, variant = path.stem[3:].split("_", 1)
        clip = Clip(dataset / "clips" / split / clip_id)
        with np.load(path) as f:
            out = {key: f[key] for key in f.files}
        rec = analyze_object(clip, int(k), out, inputs)
        records.append({"clip_id": clip_id, "k": int(k), "variant": variant, **rec})
        print(f"{clip_id} obj{k} {variant}: F1 {rec['shape']['f1@0.01']:.3f} ADD-S {rec['layout']['add_s']:.3f}",
              flush=True)
    return {"summary": summarize(records), "objects": records}


def run(command: str, dataset: str = "dataset", split: str = "eval", num_clips: int = 20,
        out: str = "t6/inputs", inputs: str = "t6/inputs", outputs: str = "t6/outputs",
        report: str = "t6/report.json") -> None:
    if command == "export":
        paths = sorted((Path(dataset) / "clips" / split).iterdir())[:num_clips]
        export_inputs([Clip(p) for p in paths], Path(out))
    elif command == "analyze":
        result = analyze(Path(outputs), Path(dataset), Path(inputs))
        Path(report).write_text(json.dumps(result, indent=1))
        print(json.dumps(result["summary"], indent=1))
    else:
        raise ValueError("Choose export or analyze")


def main() -> None:
    fire.Fire(run)


if __name__ == "__main__":
    main()
