"""Checks shared by the generator and the acceptance tests (spec Sections 4.7 and 7)."""

from __future__ import annotations

import cv2
import numpy as np
import trimesh
from scipy.ndimage import minimum_filter

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

    # 10k points leave most pixels with a single point, so a hidden point would often win the
    # z-buffer unopposed. Back-facing points are dropped, and a pixel's point is also dropped when
    # a point in its 3x3 neighbourhood is clearly nearer (hidden front faces in concavities).
    zbuf = np.full(h * w, np.inf)
    np.minimum.at(zbuf, px[front, 1] * w + px[front, 0], z[front])
    zbuf = zbuf.reshape(h, w)
    occluder = minimum_filter(zbuf, size=3, mode="constant", cval=np.inf)
    unoccluded = zbuf <= occluder + 0.02 * obj["s_wo"][0]
    sel = visible_k & np.isfinite(zbuf) & np.isfinite(depth) & unoccluded
    depth_err = float(np.median(np.abs(zbuf[sel] - depth[sel]))) if sel.any() else float("nan")
    return {
        "points_in_amodal": float(in_mask),
        "amodal_coverage": float(coverage),
        "median_depth_err": depth_err,
        "depth_pixels": int(sel.sum()),
    }


def raycast_object(
    mesh_o: trimesh.Trimesh, obj: dict, K: np.ndarray, T_cw: np.ndarray, width: int, height: int
) -> tuple[np.ndarray, np.ndarray]:
    """Exact silhouette (bool HxW) and z-depth (NaN off-object) of one posed mesh, without Blender.

    One ray per pixel center, cast in the canonical object frame so the mesh BVH is built once.
    Only pixels inside the projected vertex bounding box (plus 2 px) are traced.
    """
    T_co = T_cw @ object_matrix(obj["R_wo"], obj["t_wo"], obj["s_wo"])
    uv, _ = project(K, transform_points(T_co, mesh_o.vertices))
    u0, v0 = np.maximum(np.floor(uv.min(0)).astype(int) - 2, 0)
    u1, v1 = np.minimum(np.ceil(uv.max(0)).astype(int) + 2, [width - 1, height - 1])
    silhouette = np.zeros((height, width), bool)
    depth = np.full((height, width), np.nan)
    if u0 > u1 or v0 > v1:
        return silhouette, depth
    vv, uu = np.mgrid[v0 : v1 + 1, u0 : u1 + 1]
    pix = np.stack([uu.ravel(), vv.ravel(), np.ones(uu.size)], axis=1)
    dirs_c = pix @ np.linalg.inv(K).T
    T_oc = np.linalg.inv(T_co)
    origins = np.broadcast_to(T_oc[:3, 3], dirs_c.shape)
    dirs_o = dirs_c @ T_oc[:3, :3].T
    locations, index_ray, _ = mesh_o.ray.intersects_location(origins, dirs_o, multiple_hits=False)
    z = transform_points(T_co, locations)[:, 2]
    rows, cols = vv.ravel()[index_ray], uu.ravel()[index_ray]
    silhouette[rows, cols] = True
    depth[rows, cols] = z
    return silhouette, depth


def raycast_consistency(
    mesh_o: trimesh.Trimesh, obj: dict, K: np.ndarray, T_cw: np.ndarray,
    amodal: np.ndarray, visible_k: np.ndarray, depth: np.ndarray,
) -> dict[str, float]:
    """T1b metrics: silhouette agreement with the amodal mask, and depth error relative to object scale.

    `boundary_mismatch` is the number of disagreeing pixels per amodal boundary pixel. Unlike IoU
    it doesn't shrink with object size, so a sub-pixel offset is caught on large objects too.
    """
    h, w = amodal.shape
    sil, z = raycast_object(mesh_o, obj, K, T_cw, w, h)
    union = np.count_nonzero(sil | amodal)
    iou = np.count_nonzero(sil & amodal) / union if union else 1.0
    a = amodal.astype(np.uint8)
    boundary = np.count_nonzero(a - cv2.erode(a, np.ones((3, 3), np.uint8)))
    mismatch = np.count_nonzero(sil ^ amodal) / max(boundary, 1)
    sel = visible_k & np.isfinite(z) & np.isfinite(depth)
    err = np.abs(z[sel] - depth[sel]) / obj["s_wo"][0]
    return {
        "iou": float(iou),
        "boundary_mismatch": float(mismatch),
        "rel_depth_err_median": float(np.median(err)) if sel.any() else float("nan"),
        "amodal_px": int(amodal.sum()),
    }
