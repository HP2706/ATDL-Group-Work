"""Coordinate conventions: world Z-up, OpenCV cameras, Blender cameras, object poses.

Subscripts follow `R_ab`: maps frame-b coordinates into frame a.
"""

from __future__ import annotations

import numpy as np

# Blender cameras look down -Z with +Y up; OpenCV cameras look down +Z with +Y down.
# Both are right-handed, so the change of camera axes is a 180 degree turn about x.
OPENCV_FROM_BLENDER_CAM = np.diag([1.0, -1.0, -1.0])

# Blender's glTF importer maps glTF (Y-up) coordinates (x, y, z) to Blender (Z-up) (x, -z, y).
# The canonical object frame is the Blender one, so meshes loaded from mesh.glb outside
# Blender go through this matrix.
CANONICAL_FROM_GLTF = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


def intrinsics(width: int, height: int, hfov_deg: float) -> np.ndarray:
    """Pinhole K with square pixels and the principal point at the image center.

    OpenCV convention: pixel centers at integer coordinates, so the center is (W-1)/2.
    """
    f = (width / 2) / np.tan(np.radians(hfov_deg) / 2)
    return np.array([[f, 0.0, (width - 1) / 2], [0.0, f, (height - 1) / 2], [0.0, 0.0, 1.0]])


def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def invert_T(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    return make_T(R.T, -R.T @ t)


def look_at_opencv(eye: np.ndarray, target: np.ndarray, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """World->camera T_cw for an OpenCV camera at `eye` looking at `target`, with no roll."""
    z = np.asarray(target, float) - np.asarray(eye, float)
    z /= np.linalg.norm(z)
    x = np.cross(z, np.asarray(up, float))
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R_wc = np.stack([x, y, z], axis=1)
    return invert_T(make_T(R_wc, np.asarray(eye, float)))


def blender_cam_matrix_from_T_cw(T_cw: np.ndarray) -> np.ndarray:
    """Blender camera `matrix_world` (camera->world, Blender camera axes) from OpenCV T_cw."""
    T_wc = invert_T(T_cw)
    T_wc[:3, :3] = T_wc[:3, :3] @ OPENCV_FROM_BLENDER_CAM
    return T_wc


def T_cw_from_blender_cam_matrix(M: np.ndarray) -> np.ndarray:
    """Inverse of `blender_cam_matrix_from_T_cw`."""
    T_wc = np.array(M, float)
    T_wc[:3, :3] = T_wc[:3, :3] @ OPENCV_FROM_BLENDER_CAM
    return invert_T(T_wc)


def object_matrix(R_wo: np.ndarray, t_wo: np.ndarray, s_wo: np.ndarray) -> np.ndarray:
    """4x4 object->world matrix for x_w = R_wo (s_wo * x_o) + t_wo."""
    return make_T(np.asarray(R_wo) @ np.diag(s_wo), t_wo)


def transform_points(T: np.ndarray, x: np.ndarray) -> np.ndarray:
    return x @ T[:3, :3].T + T[:3, 3]


def project(K: np.ndarray, x_c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pixel coordinates (pixel centers at integers) and z-depth of camera-frame points."""
    z = x_c[:, 2]
    return (x_c[:, :2] / z[:, None]) @ K[:2, :2].T + K[:2, 2], z


def pose_in_camera(R_wo, t_wo, s_wo, T_cw) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Camera-frame target (R_co, t_co, s_co) that SAM 3D predicts."""
    R_cw, t_cw = T_cw[:3, :3], T_cw[:3, 3]
    return R_cw @ np.asarray(R_wo), R_cw @ np.asarray(t_wo) + t_cw, np.asarray(s_wo, float)


def object_pose_in_camera(clip, k: int, t: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(R_co, t_co, s_co) of object k (1-based) in frame t of a loaded `Clip`."""
    obj = clip.objects[k - 1]
    assert obj["k"] == k
    return pose_in_camera(obj["R_wo"], obj["t_wo"], obj["s_wo"], clip.T_cw[t])


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    """Uniform sample on SO(3) via a normalized Gaussian quaternion."""
    q = rng.standard_normal(4)
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )
