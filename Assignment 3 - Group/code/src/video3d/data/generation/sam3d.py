"""Conversion between our object poses and SAM 3D's layout output (determined in test T6).

SAM 3D's layout maps an object's canonical points into the camera as
    x_cam = (s * x_local) @ R(q) + t        (row vectors, PyTorch3D style)
with `q` a wxyz quaternion and the camera in PyTorch3D axes (x left, y up, z forward). Its
canonical frame is ours: Z-up, normalized into the unit cube [-0.5, 0.5]^3, so no object-frame
rotation or scale factor is needed. Our poses map canonical points into the OpenCV camera as
x_cam = s * R x + t (column vectors).
"""

from __future__ import annotations

import numpy as np

PYTORCH3D_FROM_OPENCV = np.diag([-1.0, -1.0, 1.0])


def quat_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def matrix_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """Unit quaternion (wxyz, w >= 0) of a rotation matrix."""
    m = np.asarray(R, float)
    # Shepperd's method: branch on the largest diagonal term for numerical stability.
    tr = np.trace(m)
    if tr > 0:
        S = 2 * np.sqrt(tr + 1)
        q = [S / 4, (m[2, 1] - m[1, 2]) / S, (m[0, 2] - m[2, 0]) / S, (m[1, 0] - m[0, 1]) / S]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        S = 2 * np.sqrt(1 + m[0, 0] - m[1, 1] - m[2, 2])
        q = [(m[2, 1] - m[1, 2]) / S, S / 4, (m[0, 1] + m[1, 0]) / S, (m[0, 2] + m[2, 0]) / S]
    elif m[1, 1] > m[2, 2]:
        S = 2 * np.sqrt(1 + m[1, 1] - m[0, 0] - m[2, 2])
        q = [(m[0, 2] - m[2, 0]) / S, (m[0, 1] + m[1, 0]) / S, S / 4, (m[1, 2] + m[2, 1]) / S]
    else:
        S = 2 * np.sqrt(1 + m[2, 2] - m[0, 0] - m[1, 1])
        q = [(m[1, 0] - m[0, 1]) / S, (m[0, 2] + m[2, 0]) / S, (m[1, 2] + m[2, 1]) / S, S / 4]
    q = np.array(q)
    return q if q[0] >= 0 else -q


def ours_to_sam3d(R: np.ndarray, t: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Object -> OpenCV-camera pose (R, t, s) to SAM 3D layout (q wxyz, t, s), PyTorch3D camera."""
    F = PYTORCH3D_FROM_OPENCV
    return matrix_to_quat_wxyz((F @ R).T), F @ t, np.asarray(s, float)


def sam3d_to_ours(q: np.ndarray, t: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inverse of `ours_to_sam3d`."""
    F = PYTORCH3D_FROM_OPENCV
    return F @ quat_wxyz_to_matrix(q).T, F @ np.asarray(t, float), np.asarray(s, float)


def sam3d_posed_points(x_local: np.ndarray, q: np.ndarray, t: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Canonical points posed by a SAM 3D layout, in SAM 3D's (PyTorch3D) camera frame."""
    return (np.asarray(s, float) * x_local) @ quat_wxyz_to_matrix(q) + t
