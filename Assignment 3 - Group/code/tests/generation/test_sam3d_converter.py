"""T6 converter: round trips and agreement with SAM 3D's own posing convention."""
import typing

import numpy as np

from video3d.data.generation.conventions import random_rotation
from video3d.data.generation.sam3d import (
    PYTORCH3D_FROM_OPENCV,
    matrix_to_quat_wxyz,
    ours_to_sam3d,
    quat_wxyz_to_matrix,
    sam3d_posed_points,
    sam3d_to_ours,
)


def test_quaternion_round_trip() -> typing.Any:
    rng = np.random.default_rng(0)
    for _ in range(200):
        R = random_rotation(rng)
        np.testing.assert_allclose(quat_wxyz_to_matrix(matrix_to_quat_wxyz(R)), R, atol=1e-12)
    # Known value: 90 degrees about z.
    np.testing.assert_allclose(quat_wxyz_to_matrix([np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)]),
                               [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-12)


def test_ours_to_sam3d_round_trip() -> typing.Any:
    rng = np.random.default_rng(1)
    R, t, s = random_rotation(rng), rng.normal(size=3), np.full(3, 0.7)
    R2, t2, s2 = sam3d_to_ours(*ours_to_sam3d(R, t, s))
    np.testing.assert_allclose(R2, R, atol=1e-12)
    np.testing.assert_allclose(t2, t)
    np.testing.assert_allclose(s2, s)


def test_converted_layout_poses_points_like_ours() -> typing.Any:
    rng = np.random.default_rng(2)
    R, t, s = random_rotation(rng), rng.normal(size=3) + [0, 0, 4], np.full(3, 0.6)
    x = rng.uniform(-0.5, 0.5, (100, 3))
    ours_cam_opencv = s[0] * x @ R.T + t
    sam_cam_p3d = sam3d_posed_points(x, *ours_to_sam3d(R, t, s))
    np.testing.assert_allclose(sam_cam_p3d @ PYTORCH3D_FROM_OPENCV.T, ours_cam_opencv, atol=1e-12)
