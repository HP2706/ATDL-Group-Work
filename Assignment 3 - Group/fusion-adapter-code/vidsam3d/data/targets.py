"""Training targets: GT voxels -> SAM 3D shape latents (cached per asset); pose conversion (Phase B)."""

from __future__ import annotations

import itertools
import json
import logging
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig
from scipy import ndimage
from scipy.spatial.transform import Rotation

from vidsam3d.data import reader
from vidsam3d.data.samples import split_folder
from vidsam3d.metrics import voxel_iou

log = logging.getLogger("vidsam3d")


def apply_canonical_permutation(vox: np.ndarray, P) -> np.ndarray:
    """Rotate a voxel grid by a signed permutation matrix P (x_new = P x on voxel centres)."""
    P = np.asarray(P)
    assert sorted(np.abs(P).sum(0).tolist()) == [1, 1, 1] and np.allclose(np.abs(P).sum(1), 1), \
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


def latent_path(cache_root: str | Path, asset_id: str) -> Path:
    """cache_root/latents/<asset_id>.pt"""
    return Path(cache_root) / "latents" / f"{asset_id}.pt"


def load_latent(cache_root: str | Path, asset_id: str) -> torch.Tensor:
    """Cached shape latent x1 of one asset."""
    path = latent_path(cache_root, asset_id)
    assert path.is_file(), f"{path} missing: run `python -m vidsam3d cache-latents --split ...`"
    return torch.load(path, map_location="cpu")


def target_voxels(voxels_file: str | Path, asset_id: str, cfg: DictConfig) -> np.ndarray:
    """GT voxels as SAM 3D sees them: canonical_permutation applied, filled if voxel_fill=solid."""
    vox = apply_canonical_permutation(reader.load_voxels(voxels_file, asset_id),
                                      cfg.target.canonical_permutation)
    if cfg.target.voxel_fill == "solid":
        return fill_voxels(vox)
    assert cfg.target.voxel_fill == "surface", f"unknown target.voxel_fill {cfg.target.voxel_fill!r}"
    return vox


@torch.no_grad()
def cache_latents(cfg: DictConfig, split: str, sam3d, device: torch.device,
                  batch_size: int = 8) -> dict:
    """Encode every asset of a split's export folder (train covers val) with the VAE encoder;
    save the latents and the round-trip IoU (T5)."""
    voxels_file = reader.open_split(Path(cfg.paths.data_root) / split_folder(cfg.data, split),
                                    str(cfg.paths.cache_root)).voxels_file
    assets = sorted(reader.voxel_table(str(voxels_file)))
    assert assets, f"no assets in {voxels_file}"
    ious = {}
    for i in range(0, len(assets), batch_size):
        chunk = assets[i:i + batch_size]
        vox = torch.from_numpy(np.stack([target_voxels(voxels_file, a, cfg) for a in chunk])).to(device)
        z = sam3d.encode_voxels(vox)
        recon = sam3d.decode_latent(z)
        for j, asset_id in enumerate(chunk):
            path = latent_path(cfg.paths.cache_root, asset_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(z[j].float().cpu(), path)
            ious[asset_id] = voxel_iou(recon[j], vox[j])
    values = np.array(list(ious.values()))
    summary = {"split": split, "num_assets": len(values), "mean_iou": float(values.mean()),
               "median_iou": float(np.median(values)), "min_iou": float(values.min()),
               "frac_iou_ge_0.9": float((values >= 0.9).mean())}
    out = Path(cfg.paths.cache_root) / "latents" / f"roundtrip_iou_{split}.json"
    out.write_text(json.dumps({"summary": summary, "per_asset": ious}, indent=1))
    log.info("round-trip IoU %s", summary)
    return summary


PYTORCH3D_FROM_OPENCV = np.diag([-1.0, -1.0, 1.0])


def convert_pose_to_sam3d(R_co, t_co, s_co) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Our camera-frame pose -> SAM 3D layout (q wxyz with w >= 0, t, s) in PyTorch3D camera axes.

    Port of synthetic-data's `sam3dvid.sam3d.ours_to_sam3d`, verified against SAM 3D's outputs in
    its test T6: SAM 3D poses canonical points as x_cam = (s * x) @ R(q) + t (row vectors), and
    its object frame and scale are ours, so only the camera axes and the rotation format change."""
    F = PYTORCH3D_FROM_OPENCV
    x, y, z, w = Rotation.from_matrix((F @ np.asarray(R_co, float)).T).as_quat()
    q = np.array([w, x, y, z])
    return (q if q[0] >= 0 else -q), F @ np.asarray(t_co, float), np.asarray(s_co, float)
