"""Read the synthetic video dataset, format v0.1 (docs/data_format.md). numpy + PIL only.

Conventions: x_a = R_ab x_b + t_ab; T_cw is world->camera with OpenCV axes;
x_w = R_wo (s_wo * x_o) + t_wo; voxel centre = (index + 0.5) / 64 - 0.5.
Ground-truth-only data (amodal masks, depth) is loaded only through `load_*_gt`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

# glTF is Y-up: (x, y, z)_gltf -> (x, -z, y)_canonical. Blender applies this on import; trimesh,
# Open3D and PyTorch3D do not.
CANONICAL_FROM_GLTF = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)


@dataclass(frozen=True)
class Clip:
    """Parsed JSON files of one clip plus its location."""
    root: Path        # dataset/
    dir: Path         # dataset/clips/<split>/<clip_id>
    clip_id: str
    split: str
    cameras: dict
    objects: list[dict]
    meta: dict

    @property
    def num_frames(self) -> int:
        return len(self.cameras["frames"])

    @property
    def num_objects(self) -> int:
        return len(self.objects)

    @property
    def image_size(self) -> tuple[int, int]:
        """(height, width) from cameras.json."""
        return int(self.cameras["height"]), int(self.cameras["width"])


def _read_json(path: Path):
    """Load one JSON file."""
    with open(path) as f:
        return json.load(f)


def read_manifest(data_root: str | Path, split: str | None = None) -> list[dict]:
    """Read manifest.jsonl, optionally keeping only one split."""
    lines = Path(data_root, "manifest.jsonl").read_text().splitlines()
    entries = [json.loads(line) for line in lines if line.strip()]
    return [e for e in entries if split is None or e["split"] == split]


def load_clip(data_root: str | Path, entry: dict) -> Clip:
    """Load cameras.json, objects.json and meta.json of a manifest entry."""
    root = Path(data_root)
    clip_dir = root / entry["path"]
    return Clip(root=root, dir=clip_dir, clip_id=entry["clip_id"], split=entry["split"],
                cameras=_read_json(clip_dir / "cameras.json"),
                objects=_read_json(clip_dir / "objects.json"),
                meta=_read_json(clip_dir / "meta.json"))


def object_entry(clip: Clip, k: int) -> dict:
    """objects.json entry of object k (1-based, matching the mask ids)."""
    o = clip.objects[k - 1]
    assert o["k"] == k, f"{clip.clip_id}: objects.json entry {k - 1} has k={o['k']}"
    return o


# ---- images and masks (model inputs) ----

def load_rgb(clip: Clip, t: int) -> np.ndarray:
    """uint8 (H, W, 3) composited frame t."""
    return np.asarray(Image.open(clip.dir / "rgb" / f"{t:03d}.png").convert("RGB"))


def load_instance_mask(clip: Clip, t: int) -> np.ndarray:
    """uint8 (H, W) visible instance map of frame t: 0 = background, k = object k."""
    return np.asarray(Image.open(clip.dir / "mask_visible" / f"{t:03d}.png"))


def load_visible_mask(clip: Clip, k: int, t: int) -> np.ndarray:
    """bool (H, W) visible mask of object k in frame t."""
    return load_instance_mask(clip, t) == k


# ---- ground truth only (targets, evaluation, analysis) ----

def load_amodal_mask_gt(clip: Clip, k: int, t: int) -> np.ndarray:
    """bool (H, W) amodal silhouette of object k in frame t. Ground truth only."""
    return np.asarray(Image.open(clip.dir / "mask_amodal" / f"obj{k}" / f"{t:03d}.png")) > 0


def load_depth_gt(clip: Clip, t: int) -> np.ndarray:
    """float32 (H, W) z-depth of frame t, NaN on background. Ground truth only."""
    with np.load(clip.dir / "depth" / f"{t:03d}.npz") as f:
        return f["depth"]


def load_voxels(data_root: str | Path, rel_path: str) -> np.ndarray:
    """bool (64, 64, 64) surface occupancy, index [i, j, k] along (x, y, z)."""
    vox = np.load(Path(data_root) / rel_path)
    assert vox.dtype == bool and vox.ndim == 3, f"{rel_path}: {vox.dtype} {vox.shape}"
    return vox


def load_object_voxels(clip: Clip, k: int) -> np.ndarray:
    """Surface voxels of object k's asset."""
    return load_voxels(clip.root, object_entry(clip, k)["voxels"])


def voxel_centers(vox: np.ndarray) -> np.ndarray:
    """(M, 3) canonical coordinates of the occupied voxel centres."""
    return (np.argwhere(vox) + 0.5) / vox.shape[0] - 0.5


# ---- cameras and poses ----

def intrinsics(clip: Clip) -> np.ndarray:
    """3x3 K (identical for all frames)."""
    return np.array(clip.cameras["K"], dtype=np.float64)


def camera_extrinsics(clip: Clip, t: int) -> tuple[np.ndarray, np.ndarray]:
    """(R_cw, t_cw) of frame t: world -> camera."""
    frame = clip.cameras["frames"][t]
    assert frame["t"] == t, f"{clip.clip_id}: cameras.json frame {t} has t={frame['t']}"
    T_cw = np.array(frame["T_cw"], dtype=np.float64)
    return T_cw[:3, :3], T_cw[:3, 3]


def object_world_pose(clip: Clip, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(R_wo, t_wo, s_wo) of object k: canonical object -> world."""
    o = object_entry(clip, k)
    return (np.array(o["R_wo"], dtype=np.float64), np.array(o["t_wo"], dtype=np.float64),
            np.array(o["s_wo"], dtype=np.float64))


def camera_frame_pose(clip: Clip, k: int, t: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(R_co, t_co, s_co) of object k in frame t, so that x_c = R_co (s_co * x_o) + t_co."""
    R_cw, t_cw = camera_extrinsics(clip, t)
    R_wo, t_wo, s_wo = object_world_pose(clip, k)
    return R_cw @ R_wo, R_cw @ t_wo + t_cw, s_wo


def project(x_c: np.ndarray, K: np.ndarray) -> np.ndarray:
    """(M, 2) pixel coordinates (u, v) of camera points; pixel centres are integers."""
    return x_c[:, :2] / x_c[:, 2:] @ K[:2, :2].T + K[:2, 2]


# ---- per-clip statistics ----

def visibility(clip: Clip, key: str = "visible_frac") -> np.ndarray:
    """(T, N) array from meta.json visibility: visible_px | amodal_px | visible_frac."""
    return np.asarray(clip.meta["visibility"][key])


def voxel_projection_check(clip: Clip, k: int, t: int) -> float:
    """Data-format §8 sanity check: fraction of projected voxel centres on the amodal mask."""
    R_co, t_co, s_co = camera_frame_pose(clip, k, t)
    x_c = (voxel_centers(load_object_voxels(clip, k)) * s_co) @ R_co.T + t_co
    uv = np.round(project(x_c, intrinsics(clip))).astype(int)
    H, W = clip.image_size
    inside = (uv >= 0).all(1) & (uv[:, 0] < W) & (uv[:, 1] < H)
    amodal = load_amodal_mask_gt(clip, k, t)
    return float(amodal[uv[inside, 1], uv[inside, 0]].mean())
