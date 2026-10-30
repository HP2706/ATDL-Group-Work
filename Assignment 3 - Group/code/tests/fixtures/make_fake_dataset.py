"""Write a tiny dataset for CPU tests, packed by the data generator's own exporter.

Clips are first written in the generator's full per-clip format (synthetic-data/docs/
data_format.md), then packed with `video3d.data.generation.export` from the sibling `synthetic-data/src`
(or $SAM3DVID_SRC) into <root>/<split>/, exactly as the training data will arrive. The full
format is deleted afterwards, so the tests read nothing else.

Per split, even seeds hold one box, odd seeds a box and a sphere that occludes it for part of
the orbit. Frames are rendered by splatting the solid voxel centres of each object with a
z-buffer, so masks, cameras and voxels are mutually consistent. The geometry is computed through
world coordinates (x_w = R_wo (s x_o) + t_wo, then x_c = R_cw x_w + t_cw), independently of the
reader's composed R_co. `mesh.glb` files are empty placeholders.

Usage: python tests/fixtures/make_fake_dataset.py <out_dir>
"""

from __future__ import annotations
import typing

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

RES = 64          # voxel grid
NUM_FRAMES = 32
SPLIT_SEEDS = {"train": [0, 1, 2, 3], "eval": [100, 101]}
COLORS = {"box": (220, 60, 40), "sphere": (40, 120, 230)}
def generator_export() -> typing.Any:
    from video3d.data.generation.export import export
    return export


def solid_shape(name: str) -> np.ndarray:
    """Solid occupancy of a canonical shape whose longest side is 1 (TRELLIS normalisation)."""
    c = (np.arange(RES) + 0.5) / RES - 0.5
    x, y, z = np.meshgrid(c, c, c, indexing="ij")
    if name == "box":                       # 1.0 x 0.6 x 0.4, so all 24 cube rotations differ
        return (np.abs(x) < 0.5) & (np.abs(y) < 0.3) & (np.abs(z) < 0.2)
    if name == "sphere":
        return x**2 + y**2 + z**2 < 0.25
    raise ValueError(name)


def surface(solid: np.ndarray) -> np.ndarray:
    """Surface occupancy: solid voxels with at least one empty 6-neighbour."""
    p = np.pad(solid, 1)
    interior = solid.copy()
    for axis in range(3):
        for shift in (-1, 1):
            interior &= np.roll(p, shift, axis)[1:-1, 1:-1, 1:-1]
    return solid & ~interior


def rotation(axis_angles: tuple[float, float, float]) -> np.ndarray:
    """Rotation matrix Rz @ Ry @ Rx from angles in degrees."""
    ax, ay, az = np.deg2rad(axis_angles)
    rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
    ry = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
    rz = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]])
    return rz @ ry @ rx


def look_at(cam_pos: np.ndarray, target: np.ndarray) -> np.ndarray:
    """World->camera 4x4 (OpenCV axes: x right, y down, z forward), world Z up, no roll."""
    f = target - cam_pos
    f /= np.linalg.norm(f)
    r = np.cross(f, [0.0, 0.0, 1.0])
    r /= np.linalg.norm(r)
    d = np.cross(f, r)
    T = np.eye(4)
    T[:3, :3] = np.stack([r, d, f])
    T[:3, 3] = -T[:3, :3] @ cam_pos
    return T


def camera_path(arc: tuple[float, float], elev_deg: float, radius: float) -> list[np.ndarray]:
    """Eased orbit u(t) = (1 - cos(pi t / 31)) / 2 from azimuth arc[0] to arc[1]."""
    out = []
    for t in range(NUM_FRAMES):
        u = (1 - np.cos(np.pi * t / (NUM_FRAMES - 1))) / 2
        az, el = np.deg2rad(arc[0] + u * (arc[1] - arc[0])), np.deg2rad(elev_deg)
        pos = radius * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        out.append(look_at(pos, np.zeros(3)))
    return out


def clip_specs(seed: int) -> dict:
    """Scene description of the clip with this seed (two fixed layouts)."""
    box = dict(shape="box", R=rotation((10, 20, 30)), t=[0.0, -0.55, 0.0], s=0.7)
    if seed % 2 == 0:
        return dict(objects=[dict(box, t=[0.0, 0.0, 0.0])], arc=(20.0, 50.0), elev=20.0)
    sphere = dict(shape="sphere", R=rotation((0, 0, 45)), t=[0.0, 0.55, 0.05], s=0.5)
    return dict(objects=[box, sphere], arc=(60.0, 120.0), elev=15.0)


def render(objects: list[dict], solids: dict, T_cw: np.ndarray, K: np.ndarray, size: int) -> typing.Any:
    """Z-buffer splat of every object's solid voxel centres; returns depth, instance map, amodal."""
    depth = np.full((size, size), np.inf)
    inst = np.zeros((size, size), np.uint8)
    amodal = []
    for k, o in enumerate(objects, start=1):
        x_o = (np.argwhere(solids[o["shape"]]) + 0.5) / RES - 0.5
        x_w = (x_o * o["s"]) @ o["R"].T + np.asarray(o["t"])
        x_c = x_w @ T_cw[:3, :3].T + T_cw[:3, 3]
        uv = np.round(x_c[:, :2] / x_c[:, 2:] @ K[:2, :2].T + K[:2, 2]).astype(int)
        ok = (uv >= 0).all(1) & (uv < size).all(1)
        uv, z = uv[ok], x_c[ok, 2]
        a = np.zeros((size, size), bool)
        a[uv[:, 1], uv[:, 0]] = True
        amodal.append(a)
        zk = np.full((size, size), np.inf)
        np.minimum.at(zk, (uv[:, 1], uv[:, 0]), z)
        closer = zk < depth
        depth[closer], inst[closer] = zk[closer], k
    depth[np.isinf(depth)] = np.nan
    return depth.astype(np.float32), inst, amodal


def write_clip(root: Path, split: str, seed: int, size: int, solids: dict) -> dict:
    """Render and write one clip; return its manifest line."""
    spec = clip_specs(seed)
    clip_id = f"{seed:08d}"
    clip_dir = root / "clips" / split / clip_id
    for sub in ("rgb", "rgba_fg", "mask_visible", "depth"):
        (clip_dir / sub).mkdir(parents=True, exist_ok=True)
    f = (size / 2) / np.tan(np.deg2rad(25))
    K = np.array([[f, 0, (size - 1) / 2], [0, f, (size - 1) / 2], [0, 0, 1]])
    cams = camera_path(spec["arc"], spec["elev"], radius=3.2)
    rng = np.random.default_rng(seed)
    n = len(spec["objects"])
    vis_px, amo_px = np.zeros((NUM_FRAMES, n), int), np.zeros((NUM_FRAMES, n), int)
    for t, T_cw in enumerate(cams):
        depth, inst, amodal = render(spec["objects"], solids, T_cw, K, size)
        bg = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
        fg = np.zeros((size, size, 4), np.uint8)
        for k, o in enumerate(spec["objects"], start=1):
            fg[inst == k] = (*COLORS[o["shape"]], 255)
            (clip_dir / "mask_amodal" / f"obj{k}").mkdir(parents=True, exist_ok=True)
            Image.fromarray(amodal[k - 1].astype(np.uint8) * 255).save(
                clip_dir / "mask_amodal" / f"obj{k}" / f"{t:03d}.png")
            vis_px[t, k - 1], amo_px[t, k - 1] = (inst == k).sum(), amodal[k - 1].sum()
        rgb = np.where(inst[..., None] > 0, fg[..., :3], bg)
        Image.fromarray(rgb).save(clip_dir / "rgb" / f"{t:03d}.png")
        Image.fromarray(fg, "RGBA").save(clip_dir / "rgba_fg" / f"{t:03d}.png")
        Image.fromarray(inst).save(clip_dir / "mask_visible" / f"{t:03d}.png")
        np.savez_compressed(clip_dir / "depth" / f"{t:03d}.npz", depth=depth)
    assert (vis_px > 0).all(), "fixture: every object must be visible in every frame"

    cameras = {"convention": "opencv", "width": size, "height": size, "K": K.tolist(),
               "frames": [{"t": t, "T_cw": T.tolist()} for t, T in enumerate(cams)]}
    objects = [{"k": k, "asset_id": o["shape"], "source": "gso",
                "mesh": f"assets/{split}/{o['shape']}/mesh.glb",
                "voxels": f"assets/{split}/{o['shape']}/voxels.npy",
                "R_wo": o["R"].tolist(), "t_wo": list(o["t"]), "s_wo": [o["s"]] * 3}
               for k, o in enumerate(spec["objects"], start=1)]
    arc_span = abs(spec["arc"][1] - spec["arc"][0])
    meta = {"clip_id": clip_id, "split": split, "seed": seed, "attempt": 0, "num_objects": n,
            "generator_version": "fake", "git_commit": None,
            "arc_span_deg": arc_span, "arc_direction": 1, "hfov_deg": 50,
            "start_azimuth_deg": spec["arc"][0], "start_elevation_deg": spec["elev"],
            "elevation_drift_deg": 0.0, "orbit_radius": 3.2, "scene_center": [0, 0, 0],
            "visibility": {"visible_px": vis_px.tolist(), "amodal_px": amo_px.tolist(),
                           "visible_frac": (vis_px / np.maximum(amo_px, 1)).tolist()}}
    for name, obj in (("cameras", cameras), ("objects", objects), ("meta", meta)):
        (clip_dir / f"{name}.json").write_text(json.dumps(obj, indent=1))
    return {"clip_id": clip_id, "split": split, "N": n, "arc_span_deg": arc_span, "seed": seed,
            "path": f"clips/{split}/{clip_id}"}


def write_full_format(root: Path, size: int, splits: typing.Any, seeds: dict[str, list[int]]) -> None:
    """The generator's per-clip folder format: assets/, clips/, manifest.jsonl."""
    solids = {name: solid_shape(name) for name in COLORS}
    lines = []
    for split in splits:
        for name, solid in solids.items():
            asset = root / "assets" / split / name
            asset.mkdir(parents=True, exist_ok=True)
            np.save(asset / "voxels.npy", surface(solid))
            (asset / "mesh.glb").write_bytes(b"")
        lines += [write_clip(root, split, seed, size, solids) for seed in seeds[split]]
    (root / "manifest.jsonl").write_text("".join(json.dumps(line) + "\n" for line in lines))


def make_fake_dataset(root: str | Path, size: int = 64, splits: typing.Any=("train", "eval"),
                      seeds: dict[str, list[int]] | None = None, clips_per_shard: int = 2) -> Path:
    """Write <root>/<split>/ training exports and return `root` (`seeds` overrides SPLIT_SEEDS)."""
    export = generator_export()
    root = Path(root)
    with tempfile.TemporaryDirectory() as full:
        write_full_format(Path(full), size, splits, seeds or SPLIT_SEEDS)
        for split in splits:
            export(Path(full), split, root / split, clips_per_shard=clips_per_shard)
    return root


if __name__ == "__main__":
    print(make_fake_dataset(sys.argv[1]))
