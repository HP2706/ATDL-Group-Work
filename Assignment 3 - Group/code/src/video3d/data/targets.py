"""Training targets: GT voxels -> SAM 3D shape latents (cached per asset); pose conversion (Phase B)."""

from __future__ import annotations
import typing

import itertools
from functools import lru_cache
import chz
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm
from video3d.config import ExperimentConfig
from scipy import ndimage

from video3d.data import reader
from video3d.data.samples import split_folder
from video3d.metrics import voxel_iou
from video3d.data.generation.sam3d import PYTORCH3D_FROM_OPENCV, ours_to_sam3d as convert_pose_to_sam3d

__all__ = ["PYTORCH3D_FROM_OPENCV", "convert_pose_to_sam3d", "apply_canonical_permutation",
           "cube_rotations", "fill_voxels", "latent_path", "load_latent", "target_voxels",
           "cache_latents", "cache_identity"]

log = logging.getLogger("video3d")


@lru_cache(maxsize=8)
def scanned_source_signature(root_name: str, manifest_mtime: int, manifest_size: int) -> tuple:
    """Hash prepared immutable geometry once, rather than reading it per batch."""
    from video3d.data.scanned import records
    root = Path(root_name)
    result = [('manifest', hashlib.sha256((root / 'manifest.jsonl').read_bytes()).hexdigest())]
    result.extend((row.object_id, hashlib.sha256((root / row.voxels).read_bytes()).hexdigest())
                  for row in records(root))
    return tuple(result)


def apply_canonical_permutation(vox: np.ndarray, P: typing.Any) -> np.ndarray:
    """Rotate a voxel grid by a signed permutation matrix P (x_new = P x on voxel centres)."""
    P = np.asarray(P)
    assert sorted(np.abs(P).sum(0).tolist()) == [1, 1, 1] and np.allclose(np.abs(P).sum(1), 1),\
        f"not a signed permutation matrix: {P.tolist()}"
    order = [int(np.flatnonzero(row)[0]) for row in P]        # output axis i <- input axis order[i]
    out = vox.transpose(order)
    for i, row in enumerate(P):
        if row[order[i]] < 0:
            out = np.flip(out, axis=i)
    return np.ascontiguousarray(out)


def cube_rotations() -> list[np.ndarray]:
    """The 24 rotations of the cube as signed permutation matrices (det = +1), identity first."""
    mats = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            P = np.zeros((3, 3), dtype=int)
            P[range(3), perm] = signs
            if round(np.linalg.det(P)) == 1:
                mats.append(P)
    return mats


def fill_voxels(vox: np.ndarray) -> np.ndarray:
    """Solid occupancy from surface occupancy (fills enclosed cavities)."""
    return ndimage.binary_fill_holes(vox)


def cache_identity(cfg: ExperimentConfig) -> str:
    sources = []
    root = Path(cfg.paths.data_root)
    if cfg.data.source in {"omniobject3d", "gso"}:
        stat = (root / 'manifest.jsonl').stat()
        sources.extend(scanned_source_signature(str(root.resolve()), stat.st_mtime_ns, stat.st_size))
    for split in sorted(root.glob("*/index.json")):
        sources.append((str(split.resolve()), hashlib.sha256(split.read_bytes()).hexdigest()))
        index = json.loads(split.read_text())
        voxels = split.parent / index["voxels_file"]
        sources.append((str(voxels.resolve()), voxels.stat().st_size, voxels.stat().st_mtime_ns))
    signature = {"schema_version":1, "source":cfg.data.source, "target":chz.asdict(cfg.target),
                 "sam_revision":cfg.sam3d.checkpoint_revision, "sam_commit":cfg.provenance.sam_commit,
                 "fake_sam3d":cfg.fake_sam3d, "sources":sources}
    return hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()[:20]


def latent_path(cfg: ExperimentConfig, asset_id: str) -> Path:
    if Path(asset_id).name != asset_id or asset_id in {".",".."}:
        raise ValueError("asset_id must be a filename-safe identifier")
    return Path(cfg.paths.cache_root) / "latents" / cache_identity(cfg) / f"{asset_id}.pt"


def load_latent(cfg: ExperimentConfig, asset_id: str) -> torch.Tensor:
    path = latent_path(cfg, asset_id)
    if not path.is_file():
        raise FileNotFoundError(f"{path} missing: run cache-latents with this configuration")
    return torch.load(path, map_location="cpu", weights_only=True)


def target_voxels(voxels_file: str | Path, asset_id: str, cfg: ExperimentConfig) -> np.ndarray:
    """GT voxels as SAM 3D sees them: canonical_permutation applied, filled if voxel_fill=solid."""
    vox = apply_canonical_permutation(reader.load_voxels(voxels_file, asset_id),
                                      cfg.target.canonical_permutation)
    if cfg.target.voxel_fill == "solid":
        return fill_voxels(vox)
    assert cfg.target.voxel_fill == "surface", f"unknown target.voxel_fill {cfg.target.voxel_fill!r}"
    return vox


@torch.no_grad()
def cache_latents(cfg: ExperimentConfig, split: str, sam3d: typing.Any, device: torch.device,
                  batch_size: int = 8) -> dict:
    """Encode every asset of a split's export folder (train covers val) with the VAE encoder;
    save the latents and the round-trip IoU (T5)."""
    if cfg.data.source in {"omniobject3d", "gso"}:
        from video3d.data.scanned import ScannedDataset
        dataset = ScannedDataset(cfg, split, train=False)
        assets = sorted(row.object_id for row in dataset.rows)
        load_voxels = dataset.target_voxels
    else:
        voxels_file = reader.open_split(Path(cfg.paths.data_root) / split_folder(cfg.data, split),
                                        str(cfg.paths.cache_root)).voxels_file
        assets = sorted(reader.voxel_table(str(voxels_file)))
        def load_voxels(asset_id: str) -> np.ndarray:
            return target_voxels(voxels_file, asset_id, cfg)
    assert assets, 'no target assets'
    ious = {}
    for i in tqdm(range(0, len(assets), batch_size), desc=f"Cache targets ({split})", unit="batch"):
        chunk = assets[i:i + batch_size]
        vox = torch.from_numpy(np.stack([load_voxels(a) for a in chunk])).to(device)
        z = sam3d.encode_voxels(vox)
        recon = sam3d.decode_latent(z)
        for j, asset_id in enumerate(chunk):
            path = latent_path(cfg, asset_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                from video3d.storage import atomic_torch
                atomic_torch(path, z[j].float().cpu())
            ious[asset_id] = voxel_iou(recon[j], vox[j])
    values = np.array(list(ious.values()))
    summary = {"split": split, "num_assets": len(values), "mean_iou": float(values.mean()),
               "median_iou": float(np.median(values)), "min_iou": float(values.min()),
               "frac_iou_ge_0.9": float((values >= 0.9).mean())}
    out = Path(cfg.paths.cache_root) / "latents" / f"roundtrip_iou_{split}.json"
    out.write_text(json.dumps({"summary": summary, "per_asset": ious}, indent=1))
    log.info("round-trip IoU %s", summary)
    return summary


