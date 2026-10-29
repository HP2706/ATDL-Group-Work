"""Reader for the training export (numpy, PIL and the stdlib only).

    from video3d.data.generation.export_reader import iter_samples, load_voxels

    for clip_id, rgb, visible, info in iter_samples("export/train/train-000000.tar"):
        # rgb (32,512,512,3) uint8, visible (32,512,512) uint8 (0 = background, k = object k)
        for obj in info["objects"]:
            R_co = np.array(obj["R_co"])  # (32,3,3), camera-frame pose per frame, our convention
            t_co, s_co = np.array(obj["t_co"]), np.array(obj["s_co"])  # (32,3), (32,3)
            voxels = load_voxels("export/train/voxels_train.npz", obj["asset_id"])  # (64,64,64) bool

Samples come out in tar order. Pass `load_voxels` an open `np.load` result to avoid reopening
the file per object.
"""

from __future__ import annotations
import typing

import io
import json
import tarfile
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from PIL import Image

VOXEL_RES = 64


def decode_rgb(data: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))


def decode_mask(data: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(data)))


def decode(group: dict[str, bytes]) -> tuple[str, np.ndarray, np.ndarray, dict]:
    rgb, visible = {}, {}
    for ext, data in group.items():
        kind, _, rest = ext.partition(".")
        if kind == "rgb":
            rgb[int(rest.split(".")[0])] = decode_rgb(data)
        elif kind == "visible":
            visible[int(rest.split(".")[0])] = decode_mask(data)
    info = json.loads(group["json"])
    return (
        info["clip_id"],
        np.stack([rgb[t] for t in sorted(rgb)]),
        np.stack([visible[t] for t in sorted(visible)]),
        info,
    )


def iter_samples(shard: str | Path) -> Iterator[tuple[str, np.ndarray, np.ndarray, dict]]:
    """Yield (clip_id, rgb, visible, info) for every clip in a shard."""
    key, group = None, {}
    with tarfile.open(shard, "r|") as tar:
        for member in tar:
            if not member.isfile():
                continue
            name = member.name.rsplit("/", 1)[-1]
            member_key, _, ext = name.partition(".")
            if member_key != key and group:
                yield decode(group)
                group = {}
            key = member_key
            group[ext] = tar.extractfile(member).read()
    if group:
        yield decode(group)


def load_voxels(voxels_npz: typing.Any, asset_id: str) -> np.ndarray:
    """Unpack one asset's bool (64,64,64) grid from `voxels_<split>.npz` (a path or np.load result)."""
    if isinstance(voxels_npz, (str, Path)):
        with np.load(voxels_npz) as f:
            return load_voxels(f, asset_id)
    bits = np.unpackbits(voxels_npz[asset_id], count=VOXEL_RES**3)
    return bits.reshape(VOXEL_RES, VOXEL_RES, VOXEL_RES).astype(bool)
