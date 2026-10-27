"""Training export: pack full-format clips into WebDataset-compatible tar shards.

A shard holds one sample per clip (RGB as JPEG, visible label maps as PNG, one JSON with
camera, per-frame object poses and clip meta). Voxels are written once per asset to
`voxels_<split>.npz`. Depth, amodal masks and rgba_fg are not exported. Poses stay in our
convention (docs/data_format.md section 3). Read the result with `sam3dvid.export_reader`.

    uv run python -m sam3dvid.export --dataset dataset --split train --out export/train \
        --clips-per-shard 100 --workers 8

One `--out` directory holds one split: `index.json` has a fixed name.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from .clipio import Clip
from .conventions import object_pose_in_camera

FORMAT_VERSION = 1
JPEG_QUALITY = 95
# 4:4:4 chroma: PIL's default 4:2:0 at q95 gives only ~38-42 dB PSNR on these saturated,
# sharp-edged renders, for about 20% fewer bytes.
JPEG_SUBSAMPLING = 0
VOXEL_RES = 64

CONVENTIONS = {
    "camera": "opencv (x right, y down, z forward)",
    "world": "right-handed, Z up",
    "T_cw": "world -> camera, 4x4, row-major",
    "pose": "x_c = R_co @ (s_co * x_o) + t_co, with R_co = R_cw @ R_wo and t_co = R_cw @ t_wo + t_cw",
    "arrays": "row-major, matrices as nested lists",
    "pixel_centers": "integer coordinates",
    "units": "arbitrary scene units, no metric scale",
    "voxels": "bool 64^3 surface occupancy, np.packbits (C order, big-endian bits) of the flattened grid",
    "visible": "uint8, 0 = background, k = object k (1-based, matches objects[k-1])",
}


def _encode(image: Image.Image, fmt: str, **kwargs) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format=fmt, **kwargs)
    return buf.getvalue()


def clip_json(clip: Clip) -> dict:
    meta = clip.meta
    objects = []
    for o in clip.objects:
        k = o["k"]
        poses = [object_pose_in_camera(clip, k, t) for t in range(clip.num_frames)]
        objects.append(
            {
                "k": k,
                "asset_id": o["asset_id"],
                "R_wo": o["R_wo"].tolist(),
                "t_wo": o["t_wo"].tolist(),
                "s_wo": o["s_wo"].tolist(),
                "R_co": [p[0].tolist() for p in poses],
                "t_co": [p[1].tolist() for p in poses],
                "s_co": [p[2].tolist() for p in poses],
            }
        )
    out = {
        "clip_id": meta["clip_id"],
        "split": meta["split"],
        "seed": meta["seed"],
        "num_objects": meta["num_objects"],
        "generator_version": meta["generator_version"],
        "git_commit": meta["git_commit"],
        "K": clip.K.tolist(),
        "T_cw": clip.T_cw.tolist(),
        "objects": objects,
        "visibility": meta["visibility"],
    }
    blender = meta.get("blender_version") or meta.get("render", {}).get("blender_version")
    if blender:
        out["blender_version"] = blender
    return out


def clip_members(clip_path: Path) -> list[tuple[str, bytes]]:
    """Sample files (extension, bytes) of one clip, in a fixed order."""
    clip = Clip(clip_path)
    cid = clip.meta["clip_id"]
    files = []
    for t in range(clip.num_frames):
        rgb = Image.fromarray(clip.rgb(t))
        files.append((f"rgb.{t:03d}.jpg", _encode(rgb, "JPEG", quality=JPEG_QUALITY, subsampling=JPEG_SUBSAMPLING)))
    for t in range(clip.num_frames):
        vis = Image.fromarray(np.ascontiguousarray(clip.visible(t), dtype=np.uint8))
        files.append((f"visible.{t:03d}.png", _encode(vis, "PNG", compress_level=9)))
    files.append(("json", json.dumps(clip_json(clip)).encode()))
    return [(f"{cid}.{ext}", data) for ext, data in files]


def _add(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    # Fixed metadata so identical input gives byte-identical shards.
    info = tarfile.TarInfo(name)
    info.size, info.mtime, info.mode = len(data), 0, 0o644
    tar.addfile(info, io.BytesIO(data))


def write_shard(path: Path, clip_paths: list[Path]) -> int:
    tmp = path.with_name(path.name + ".tmp")
    with tarfile.open(tmp, "w", format=tarfile.USTAR_FORMAT) as tar:
        for cp in clip_paths:
            for name, data in clip_members(cp):
                _add(tar, name, data)
    os.replace(tmp, path)
    return path.stat().st_size


def _shard_job(args: tuple[Path, list[Path]]) -> tuple[str, int, float]:
    path, clip_paths = args
    start = time.perf_counter()
    size = write_shard(path, clip_paths)
    return path.name, size, time.perf_counter() - start


def list_clips(dataset: Path, split: str) -> list[Path]:
    """Clip folders of a split, sorted by clip id. The manifest decides when present."""
    manifest = dataset / "manifest.jsonl"
    if manifest.exists():
        entries = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
        paths = [dataset / e["path"] for e in entries if e["split"] == split]
    else:
        paths = list((dataset / "clips" / split).iterdir())
    return sorted(paths, key=lambda p: p.name)


def pack_voxels(dataset: Path, clips: list[Path], out_path: Path) -> int:
    """Write asset_id -> packbits(voxels) for the assets the clips use; returns the count."""
    packed: dict[str, np.ndarray] = {}
    for cp in clips:
        for o in json.loads((cp / "objects.json").read_text()):
            if o["asset_id"] not in packed:
                grid = np.load(dataset / o["voxels"])
                assert grid.dtype == bool and grid.shape == (VOXEL_RES,) * 3, o["voxels"]
                packed[o["asset_id"]] = np.packbits(grid.reshape(-1))
    tmp = out_path.with_name(out_path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **dict(sorted(packed.items())))
    os.replace(tmp, out_path)
    return len(packed)


def export(
    dataset: Path, split: str, out: Path, clips_per_shard: int = 100, workers: int = 1
) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    clips = list_clips(dataset, split)
    chunks = [clips[i : i + clips_per_shard] for i in range(0, len(clips), clips_per_shard)]
    shards = [(out / f"{split}-{i:06d}.tar", chunk) for i, chunk in enumerate(chunks)]
    # A shard that exists is finished: shards only appear via the final rename.
    todo = [s for s in shards if not s[0].exists()]
    print(f"{split}: {len(clips)} clips, {len(shards)} shards, {len(todo)} to write", flush=True)

    start = time.perf_counter()
    if workers > 1 and len(todo) > 1:
        with ProcessPoolExecutor(workers) as pool:
            results = pool.map(_shard_job, todo)
            for name, size, secs in results:
                print(f"  {name}: {size / 1e6:.1f} MB in {secs:.1f} s", flush=True)
    else:
        for job in todo:
            name, size, secs = _shard_job(job)
            print(f"  {name}: {size / 1e6:.1f} MB in {secs:.1f} s", flush=True)
    elapsed = time.perf_counter() - start

    voxels_name = f"voxels_{split}.npz"
    num_assets = pack_voxels(dataset, clips, out / voxels_name)
    index = {
        "format_version": FORMAT_VERSION,
        "split": split,
        "clips_per_shard": clips_per_shard,
        "num_clips": len(clips),
        "num_frames": 32,
        "image_size": [512, 512],
        "rgb_encoding": f"JPEG quality {JPEG_QUALITY}, 4:4:4",
        "voxels_file": voxels_name,
        "num_assets": num_assets,
        "conventions": CONVENTIONS,
        "shards": [
            {
                "name": path.name,
                "num_clips": len(chunk),
                "bytes": path.stat().st_size,
                "clip_ids": [p.name for p in chunk],
            }
            for path, chunk in shards
        ],
    }
    tmp = out / "index.json.tmp"
    tmp.write_text(json.dumps(index, indent=1))
    os.replace(tmp, out / "index.json")
    index["_elapsed_s"] = elapsed
    return index


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dataset", type=Path, default=Path("dataset"))
    p.add_argument("--split", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--clips-per-shard", type=int, default=100)
    p.add_argument("--workers", type=int, default=1)
    args = p.parse_args()
    index = export(args.dataset, args.split, args.out, args.clips_per_shard, args.workers)
    total = sum(s["bytes"] for s in index["shards"])
    print(f"{index['num_clips']} clips, {total / 1e9:.2f} GB, {index['num_assets']} assets")


if __name__ == "__main__":
    main()
