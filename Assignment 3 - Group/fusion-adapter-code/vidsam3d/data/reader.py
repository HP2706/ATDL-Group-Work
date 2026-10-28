"""Read the training export of the synthetic video dataset (numpy + PIL only).

The format is defined in `synthetic-data/docs/data_format.md` (section 9, built by
`python -m sam3dvid.export`). One directory per split:

    <split>/index.json              shards with their clip ids and byte sizes
    <split>/<split>-NNNNNN.tar      per clip: <id>.rgb.TTT.jpg, <id>.visible.TTT.png, <id>.json
    <split>/voxels_<split>.npz      asset_id -> np.packbits of the bool (64, 64, 64) grid

Conventions (section 3): x_a = R_ab x_b + t_ab; T_cw is world->camera with OpenCV axes;
x_c = R_co (s_co * x_o) + t_co; voxel centre = (index + 0.5) / 64 - 0.5; pixel centres at
integers, so the principal point is ((W - 1) / 2, (H - 1) / 2).
Clips are read by random access: the member offsets of every shard are scanned once and cached.
"""

from __future__ import annotations

import io
import json
import tarfile
import zlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

VOXEL_RES = 64


@dataclass(frozen=True)
class ClipRef:
    """Where a clip lives in its shard, and which assets it shows (objects k = 1..N)."""
    clip_id: str
    shard: Path
    members: dict          # extension (e.g. "rgb.000.jpg", "json") -> (offset, size)
    asset_ids: tuple


@dataclass(frozen=True)
class Clip:
    """One clip: its JSON (camera, per-frame object poses, visibility) and shard location."""
    ref: ClipRef
    info: dict

    @property
    def clip_id(self) -> str:
        return self.ref.clip_id

    @property
    def num_frames(self) -> int:
        return len(self.info["T_cw"])

    @property
    def num_objects(self) -> int:
        return len(self.info["objects"])

    @property
    def image_size(self) -> tuple[int, int]:
        """(height, width) from the principal point at the image centre."""
        K = self.info["K"]
        return round(2 * K[1][2] + 1), round(2 * K[0][2] + 1)


@dataclass(frozen=True)
class Split:
    """One export directory: its index and every clip's location."""
    dir: Path
    index: dict
    clips: dict            # clip_id -> ClipRef, sorted by clip id

    @property
    def voxels_file(self) -> Path:
        return self.dir / self.index["voxels_file"]

    def load_clip(self, clip_id: str) -> Clip:
        """Read the clip's JSON from its shard."""
        ref = self.clips[clip_id]
        return Clip(ref, json.loads(read_member(ref, "json")))


# ---- shards ----

def scan_shard(path: Path) -> dict[str, dict]:
    """{clip_id: {"members": {ext: (offset, size)}, "asset_ids": [...]}} of one shard."""
    clips: dict[str, dict] = {}
    with tarfile.open(path, "r:") as tar:
        for m in tar:
            if not m.isfile():
                continue
            clip_id, _, ext = m.name.rsplit("/", 1)[-1].partition(".")
            clips.setdefault(clip_id, {"members": {}})["members"][ext] = (m.offset_data, m.size)
            if ext == "json":
                info = json.loads(tar.extractfile(m).read())
                clips[clip_id]["asset_ids"] = [o["asset_id"] for o in info["objects"]]
    return clips


@lru_cache(maxsize=None)
def open_split(split_dir: str | Path, cache_dir: str | Path | None = None) -> Split:
    """Index a split directory; with `cache_dir`, shard scans are reused while shard sizes match."""
    split_dir = Path(split_dir)
    index_path = split_dir / "index.json"
    assert index_path.is_file(), f"{index_path} missing: expected a training export (sam3dvid.export)"
    index = json.loads(index_path.read_text())
    cache_path = None
    cached = {}
    if cache_dir is not None:
        key = zlib.crc32(str(split_dir.resolve()).encode())
        cache_path = Path(cache_dir) / "index" / f"{split_dir.name}-{key:08x}.json"
        if cache_path.is_file():
            cached = json.loads(cache_path.read_text())
    scans, changed = {}, False
    for shard in index["shards"]:
        entry = cached.get(shard["name"])
        if entry is None or entry["bytes"] != shard["bytes"]:
            entry, changed = {"bytes": shard["bytes"], "clips": scan_shard(split_dir / shard["name"])}, True
        assert sorted(entry["clips"]) == sorted(shard["clip_ids"]), f"{shard['name']}: clips differ from index.json"
        scans[shard["name"]] = entry
    if cache_path is not None and changed:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(scans))
    clips = {}
    for name, entry in scans.items():
        for clip_id, c in entry["clips"].items():
            members = {ext: tuple(v) for ext, v in c["members"].items()}
            clips[clip_id] = ClipRef(clip_id, split_dir / name, members, tuple(c["asset_ids"]))
    return Split(split_dir, index, dict(sorted(clips.items())))


def read_member(ref: ClipRef, ext: str) -> bytes:
    """Bytes of one member of a clip, e.g. ext = "rgb.000.jpg"."""
    offset, size = ref.members[ext]
    with open(ref.shard, "rb") as f:
        f.seek(offset)
        return f.read(size)


# ---- images and masks (model inputs) ----

def load_rgb(clip: Clip, t: int) -> np.ndarray:
    """uint8 (H, W, 3) frame t (JPEG q95, 4:4:4)."""
    return np.asarray(Image.open(io.BytesIO(read_member(clip.ref, f"rgb.{t:03d}.jpg"))).convert("RGB"))


def load_instance_mask(clip: Clip, t: int) -> np.ndarray:
    """uint8 (H, W) visible instance map of frame t: 0 = background, k = object k."""
    return np.asarray(Image.open(io.BytesIO(read_member(clip.ref, f"visible.{t:03d}.png"))))


def load_visible_mask(clip: Clip, k: int, t: int) -> np.ndarray:
    """bool (H, W) visible mask of object k in frame t."""
    return load_instance_mask(clip, t) == k


# ---- voxels (targets) ----

@lru_cache(maxsize=4)
def voxel_table(voxels_file: str | Path) -> dict[str, np.ndarray]:
    """asset_id -> packed voxels of a whole split (a few KB per asset), read once per process."""
    with np.load(voxels_file) as f:
        return {asset_id: f[asset_id] for asset_id in f.files}


def load_voxels(voxels_file: str | Path, asset_id: str) -> np.ndarray:
    """bool (64, 64, 64) surface occupancy, index [i, j, k] along (x, y, z)."""
    bits = np.unpackbits(voxel_table(str(voxels_file))[asset_id], count=VOXEL_RES**3)
    return bits.reshape((VOXEL_RES,) * 3).astype(bool)


def voxel_centers(vox: np.ndarray) -> np.ndarray:
    """(M, 3) canonical coordinates of the occupied voxel centres."""
    return (np.argwhere(vox) + 0.5) / vox.shape[0] - 0.5


# ---- cameras and poses ----

def object_entry(clip: Clip, k: int) -> dict:
    """JSON entry of object k (1-based, matching the mask ids)."""
    o = clip.info["objects"][k - 1]
    assert o["k"] == k, f"{clip.clip_id}: objects[{k - 1}] has k={o['k']}"
    return o


def intrinsics(clip: Clip) -> np.ndarray:
    """3x3 K (identical for all frames)."""
    return np.array(clip.info["K"], dtype=np.float64)


def camera_extrinsics(clip: Clip, t: int) -> tuple[np.ndarray, np.ndarray]:
    """(R_cw, t_cw) of frame t: world -> camera."""
    T_cw = np.array(clip.info["T_cw"][t], dtype=np.float64)
    return T_cw[:3, :3], T_cw[:3, 3]


def object_world_pose(clip: Clip, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(R_wo, t_wo, s_wo) of object k: canonical object -> world."""
    o = object_entry(clip, k)
    return (np.array(o["R_wo"], dtype=np.float64), np.array(o["t_wo"], dtype=np.float64),
            np.array(o["s_wo"], dtype=np.float64))


def camera_frame_pose(clip: Clip, k: int, t: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(R_co, t_co, s_co) of object k in frame t, composed from the world poses.

    The export also stores this per frame; `validate-data` checks that the two agree."""
    R_cw, t_cw = camera_extrinsics(clip, t)
    R_wo, t_wo, s_wo = object_world_pose(clip, k)
    return R_cw @ R_wo, R_cw @ t_wo + t_cw, s_wo


def project(x_c: np.ndarray, K: np.ndarray) -> np.ndarray:
    """(M, 2) pixel coordinates (u, v) of camera points; pixel centres are integers."""
    return x_c[:, :2] / x_c[:, 2:] @ K[:2, :2].T + K[:2, 2]


# ---- per-clip statistics ----

def visibility(clip: Clip, key: str = "visible_frac") -> np.ndarray:
    """(T, N) array: visible_px | amodal_px | visible_frac."""
    return np.asarray(clip.info["visibility"][key])


def arc_span_deg(clip: Clip) -> float:
    """Orbit angle of the clip. The camera looks at a fixed point with world Z up and no roll,
    so the yaw of its viewing direction turns by exactly the orbit's azimuth change."""
    R_cw = np.array(clip.info["T_cw"])[:, :3, :3]
    forward = R_cw[:, 2, :]                       # camera +z in world coordinates
    yaw = np.unwrap(np.arctan2(forward[:, 1], forward[:, 0]))
    return float(np.degrees(abs(yaw[-1] - yaw[0])))


def voxel_coverage(clip: Clip, k: int, t: int, vox: np.ndarray) -> float:
    """Fraction of object k's visible pixels covered by its projected voxels.

    Each voxel centre is splatted as a square of half a projected voxel (at least 1 px), i.e. the
    voxel's footprint. A convention check: the right pose covers ~all pixels, a wrong one far fewer."""
    R_co, t_co, s_co = camera_frame_pose(clip, k, t)
    x_c = (voxel_centers(vox) * s_co) @ R_co.T + t_co
    K = intrinsics(clip)
    uv = np.round(project(x_c, K)).astype(int)
    dilate = max(1, int(np.ceil(0.5 * K[0, 0] * s_co[0] / (vox.shape[0] * np.median(x_c[:, 2])))))
    H, W = clip.image_size
    hit = np.zeros((H + 2 * dilate, W + 2 * dilate), bool)
    uv = uv[(uv[:, 0] >= -dilate) & (uv[:, 0] < W + dilate) & (uv[:, 1] >= -dilate) & (uv[:, 1] < H + dilate)]
    for dy in range(-dilate, dilate + 1):
        for dx in range(-dilate, dilate + 1):
            hit[np.clip(uv[:, 1] + dy + dilate, 0, H + 2 * dilate - 1),
                np.clip(uv[:, 0] + dx + dilate, 0, W + 2 * dilate - 1)] = True
    visible = load_visible_mask(clip, k, t)
    return float(hit[dilate:H + dilate, dilate:W + dilate][visible].mean())
