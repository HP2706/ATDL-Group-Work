"""Checks shared by the generator and the acceptance tests (spec Sections 4.7 and 7)."""

from __future__ import annotations

import cv2
import numpy as np
import trimesh

from .config import Config
from .conventions import CANONICAL_FROM_GLTF, object_matrix, project, transform_points


def visibility_stats(visible: np.ndarray, amodal: np.ndarray) -> dict[str, np.ndarray]:
    """Per (frame, object) pixel counts from a (T,H,W) instance map and (N,T,H,W) amodal masks."""
    n = amodal.shape[0]
    vis = np.stack([(visible == k + 1).sum(axis=(1, 2)) for k in range(n)], axis=1)
    amo = amodal.sum(axis=(2, 3)).T
    frac = np.divide(vis, amo, out=np.zeros(vis.shape, float), where=amo > 0)
    return {"visible_px": vis, "amodal_px": amo, "visible_frac": frac}


def visibility_failures(stats: dict[str, np.ndarray], cfg: Config, slack: float = 1.0) -> list[str]:
    """Human-readable violations of the per-object, per-frame visibility rules."""
    failures = []
    vis, frac = stats["visible_px"], stats["visible_frac"]
    for t, k in zip(*np.nonzero(vis < slack * cfg.min_visible_px)):
        failures.append(f"t={t} k={k + 1} visible_px={vis[t, k]:.0f}")
    for t, k in zip(*np.nonzero(frac < slack * cfg.min_visible_over_amodal)):
        failures.append(f"t={t} k={k + 1} visible_frac={frac[t, k]:.3f}")
    return failures


def load_canonical_mesh(path) -> trimesh.Trimesh:
    """mesh.glb as one Trimesh with vertices in the canonical (Blender Z-up) object frame."""
    mesh = trimesh.load(path, force="mesh", process=False)
    mesh.vertices = mesh.vertices @ CANONICAL_FROM_GLTF.T
    return mesh


def sample_surface(mesh: trimesh.Trimesh, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Area-uniform surface points and their face normals, in the mesh's frame."""
    points, faces = trimesh.sample.sample_surface(mesh, n, seed=seed)
    return np.asarray(points), np.asarray(mesh.face_normals[faces])


def projection_consistency(
    points_o: np.ndarray,
    normals_o: np.ndarray,
    obj: dict,
    K: np.ndarray,
    T_cw: np.ndarray,
    amodal: np.ndarray,
    visible_k: np.ndarray,
    depth: np.ndarray,
    dilate_px: int = 3,
) -> dict[str, float]:
    """T1 metrics for one object in one frame.

    Returns the fraction of in-image points inside the dilated amodal mask, the fraction of
    amodal pixels covered by the dilated point image, and the median |z-buffered point depth -
    rendered depth| over pixels where the object is visible.
    """
    h, w = amodal.shape
    T_co = T_cw @ object_matrix(obj["R_wo"], obj["t_wo"], obj["s_wo"])
    x_c = transform_points(T_co, points_o)
    # Normals transform with the inverse transpose; the scale is isotropic, so R_co suffices.
    n_c = normals_o @ T_co[:3, :3].T
    front = np.einsum("ij,ij->i", n_c, x_c) < 0
    uv, z = project(K, x_c)
    px = np.round(uv).astype(int)
    inb = (px[:, 0] >= 0) & (px[:, 0] < w) & (px[:, 1] >= 0) & (px[:, 1] < h)
    px, z, front = px[inb], z[inb], front[inb]

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
    amodal_dil = cv2.dilate(amodal.astype(np.uint8), kernel) > 0
    in_mask = amodal_dil[px[:, 1], px[:, 0]].mean() if len(px) else 0.0

    hit = np.zeros((h, w), np.uint8)
    hit[px[:, 1], px[:, 0]] = 1
    hit_dil = cv2.dilate(hit, kernel) > 0
    coverage = hit_dil[amodal].mean() if amodal.any() else 1.0

    # 10k points leave most pixels with a single point, so a far-side point would often win the
    # z-buffer unopposed; back-facing points are dropped first.
    zbuf = np.full(h * w, np.inf)
    np.minimum.at(zbuf, px[front, 1] * w + px[front, 0], z[front])
    zbuf = zbuf.reshape(h, w)
    sel = visible_k & np.isfinite(zbuf) & np.isfinite(depth)
    depth_err = float(np.median(np.abs(zbuf[sel] - depth[sel]))) if sel.any() else float("nan")
    return {
        "points_in_amodal": float(in_mask),
        "amodal_coverage": float(coverage),
        "median_depth_err": depth_err,
        "depth_pixels": int(sel.sum()),
    }
