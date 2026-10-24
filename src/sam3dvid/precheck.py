"""Cheap pre-render visibility estimate: z-buffered splats of posed voxel centers at low resolution."""

from __future__ import annotations

import numpy as np

from .conventions import object_matrix, project, transform_points

VOXEL_RES = 64


def voxel_centers(voxels: np.ndarray) -> np.ndarray:
    """Canonical-frame centers of occupied voxels, matching TRELLIS's (i + 0.5) / 64 - 0.5."""
    return (np.argwhere(voxels) + 0.5) / VOXEL_RES - 0.5


def estimate_visibility(
    K: np.ndarray,
    T_cw: np.ndarray,
    objects: list,
    centers: list[np.ndarray],
    width: int,
    height: int,
    res: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimated (visible_px, amodal_px) per object at full resolution for one frame.

    Each voxel is splatted as a square covering its projected diagonal, so masks are slightly
    fat; the caller compares against thresholds with slack.
    """
    scale = res / width
    res_y = int(round(height * scale))
    pix, depth, owner = [], [], []
    amodal = np.zeros(len(objects), np.int64)
    for i, (obj, x_o) in enumerate(zip(objects, centers)):
        x_c = transform_points(T_cw @ object_matrix(obj.R_wo, obj.t_wo, obj.s_wo), x_o)
        uv, z = project(K, x_c)
        uv = (uv + 0.5) * scale - 0.5
        half = np.ceil(0.5 * np.sqrt(3) * obj.s_wo[0] / VOXEL_RES * K[0, 0] * scale / z.min()).astype(int)
        offs = np.arange(-half, half + 1)
        du, dv = np.meshgrid(offs, offs)
        cu = np.round(uv[:, 0])[:, None].astype(int) + du.ravel()[None]
        cv = np.round(uv[:, 1])[:, None].astype(int) + dv.ravel()[None]
        inside = (cu >= 0) & (cu < res) & (cv >= 0) & (cv < res_y)
        flat = (cv * res + cu)[inside]
        amodal[i] = np.unique(flat).size
        pix.append(flat)
        depth.append(np.broadcast_to(z[:, None], cu.shape)[inside])
        owner.append(np.full(flat.size, i))
    pix, depth, owner = np.concatenate(pix), np.concatenate(depth), np.concatenate(owner)
    order = np.lexsort((depth, pix))
    _, first = np.unique(pix[order], return_index=True)
    visible = np.bincount(owner[order][first], minlength=len(objects))
    to_full = (width / res) * (height / res_y)
    return visible * to_full, amodal * to_full
