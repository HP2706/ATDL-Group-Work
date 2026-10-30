"""Background at infinity: virtual background camera, per-frame homography, coverage (spec 4.6)."""

from __future__ import annotations

import cv2
import numpy as np

from .conventions import intrinsics


def background_K(bg_width: int, bg_height: int, hfov_deg: float) -> np.ndarray:
    return intrinsics(bg_width, bg_height, hfov_deg)


def relative_rotation(T_cw: np.ndarray, T_cw_ref: np.ndarray) -> np.ndarray:
    return T_cw[:3, :3] @ T_cw_ref[:3, :3].T


def homography(K: np.ndarray, R_rel: np.ndarray, K_bg: np.ndarray) -> np.ndarray:
    """Maps background-image pixels to frame pixels."""
    return K @ R_rel @ np.linalg.inv(K_bg)


def corners_in_background(
    K: np.ndarray, R_rel: np.ndarray, K_bg: np.ndarray, width: int, height: int
) -> tuple[np.ndarray, np.ndarray]:
    """Background-image positions and ray depths of the four outer frame corners."""
    lo, hx, hy = -0.5, width - 0.5, height - 0.5
    corners = np.array([[lo, lo, 1], [hx, lo, 1], [lo, hy, 1], [hx, hy, 1]], float)
    rays_ref = (R_rel.T @ np.linalg.inv(K) @ corners.T).T
    z = rays_ref[:, 2]
    p = (K_bg @ (rays_ref / z[:, None]).T).T[:, :2]
    return p, z


def covers(
    K: np.ndarray,
    R_rel: np.ndarray,
    K_bg: np.ndarray,
    bg_width: int,
    bg_height: int,
    width: int,
    height: int,
    margin: float,
) -> bool:
    """True if the whole frame maps inside the background image, `margin` px from its border.

    The set of frame rays that land inside the background rectangle is convex, so checking
    the four corners suffices.
    """
    p, z = corners_in_background(K, R_rel, K_bg, width, height)
    return bool(
        np.all(z > 0)
        and np.all(p[:, 0] >= margin)
        and np.all(p[:, 0] <= bg_width - 1 - margin)
        and np.all(p[:, 1] >= margin)
        and np.all(p[:, 1] <= bg_height - 1 - margin)
    )


def warp(image: np.ndarray, H: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Warp the background onto the frame; also returns a bool map of fully defined pixels."""
    out = cv2.warpPerspective(image, H, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    ones = np.ones(image.shape[:2], np.float32)
    weight = cv2.warpPerspective(ones, H, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return out, weight >= 1 - 1e-4
