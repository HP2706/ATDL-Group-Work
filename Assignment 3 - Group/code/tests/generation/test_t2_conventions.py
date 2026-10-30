"""T2: hand-constructed poses with known projections and convention round trips."""
import typing

import numpy as np
import pytest

from video3d.data.generation.background import background_K, covers, homography, warp
from video3d.data.generation.config import Config
from video3d.data.generation.conventions import (
    T_cw_from_blender_cam_matrix,
    blender_cam_matrix_from_T_cw,
    intrinsics,
    invert_T,
    look_at_opencv,
    make_T,
    object_matrix,
    pose_in_camera,
    project,
    random_rotation,
    transform_points,
)
from video3d.data.generation.sampling import Trajectory, sample_layout

K = intrinsics(512, 512, 50.0)


def test_intrinsics_principal_point_and_fov() -> typing.Any:
    assert K[0, 2] == K[1, 2] == 255.5
    # A point on the horizontal FOV boundary lands on the outer edge of the image (-0.5 / 511.5).
    x = np.tan(np.radians(25.0))
    uv, _ = project(K, np.array([[x, 0.0, 1.0], [-x, 0.0, 1.0]]))
    np.testing.assert_allclose(uv[:, 0], [511.5, -0.5])


def test_look_at_known_projection() -> typing.Any:
    # Camera on the -X axis looking at the origin with Z up: world +Y is image left, +Z is image up.
    T_cw = look_at_opencv(np.array([-5.0, 0, 0]), np.zeros(3))
    np.testing.assert_allclose(T_cw[:3, :3], [[0, -1, 0], [0, 0, -1], [1, 0, 0]], atol=1e-12)
    pts = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1]], float)
    uv, z = project(K, transform_points(T_cw, pts))
    np.testing.assert_allclose(z, 5.0)
    np.testing.assert_allclose(uv[0], [255.5, 255.5])
    assert uv[1, 0] < 255.5 and abs(uv[1, 1] - 255.5) < 1e-9
    assert uv[2, 1] < 255.5 and abs(uv[2, 0] - 255.5) < 1e-9


def test_look_at_has_no_roll() -> typing.Any:
    rng = np.random.default_rng(0)
    for _ in range(50):
        eye = rng.normal(size=3) * 4
        eye[2] = abs(eye[2]) + 0.5
        R_cw = look_at_opencv(eye, np.zeros(3))[:3, :3]
        x_cam_in_world = R_cw.T[:, 0]
        assert abs(x_cam_in_world[2]) < 1e-12
        assert R_cw.T[2, 1] < 0  # camera y (down) has a negative world-Z component


def test_blender_opencv_roundtrip() -> typing.Any:
    rng = np.random.default_rng(1)
    for _ in range(20):
        T_cw = make_T(random_rotation(rng), rng.normal(size=3))
        M = blender_cam_matrix_from_T_cw(T_cw)
        np.testing.assert_allclose(T_cw_from_blender_cam_matrix(M), T_cw, atol=1e-12)
        # Blender camera looks down its local -Z; OpenCV down +Z. Same viewing direction in world.
        np.testing.assert_allclose(-M[:3, 2], invert_T(T_cw)[:3, 2], atol=1e-12)
        # Blender local +Y (image up) is OpenCV -y.
        np.testing.assert_allclose(M[:3, 1], -invert_T(T_cw)[:3, 1], atol=1e-12)
        np.testing.assert_allclose(M[:3, 3], invert_T(T_cw)[:3, 3], atol=1e-12)


def test_object_pose_in_camera_matches_direct_transform() -> typing.Any:
    rng = np.random.default_rng(2)
    R_wo, t_wo, s = random_rotation(rng), rng.normal(size=3), np.full(3, 0.7)
    T_cw = look_at_opencv(np.array([3.0, 1.0, 2.0]), np.zeros(3))
    x_o = rng.uniform(-0.5, 0.5, size=(100, 3))
    direct = transform_points(T_cw @ object_matrix(R_wo, t_wo, s), x_o)
    R_co, t_co, s_co = pose_in_camera(R_wo, t_wo, s, T_cw)
    np.testing.assert_allclose(x_o * s_co @ R_co.T + t_co, direct, atol=1e-12)


def test_random_rotation_is_rotation_and_uniform() -> typing.Any:
    rng = np.random.default_rng(3)
    Rs = np.stack([random_rotation(rng) for _ in range(20000)])
    np.testing.assert_allclose(Rs @ Rs.transpose(0, 2, 1), np.broadcast_to(np.eye(3), Rs.shape), atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(Rs), 1.0, atol=1e-12)
    # Haar measure: E[R] = 0 and the rotation angle has density (1 - cos a) / pi.
    np.testing.assert_allclose(Rs.mean(0), 0, atol=0.02)
    angle = np.arccos(np.clip((np.trace(Rs, axis1=1, axis2=2) - 1) / 2, -1, 1))
    assert abs(angle.mean() - (np.pi / 2 + 2 / np.pi)) < 0.02


def test_layout_no_overlap_and_scale_unbiased() -> typing.Any:
    cfg = Config()
    rng = np.random.default_rng(4)
    scales = []
    for _ in range(300):
        objs = sample_layout(rng, cfg, ["a", "b", "c", "d"])
        for i in range(4):
            for j in range(i + 1, 4):
                d = np.linalg.norm(objs[i].t_wo - objs[j].t_wo)
                assert d > objs[i].bounding_radius + objs[j].bounding_radius
        scales += [o.s_wo[0] for o in objs]
    # Positions-only resampling must leave s ~ U(0.4, 1.0): mean 0.7, sd 0.173.
    assert abs(np.mean(scales) - 0.7) < 0.02


def test_trajectory_reference_is_arc_midpoint_and_max_rotation() -> typing.Any:
    traj = Trajectory(center=np.zeros(3), orbit_radius=5.0, start_azimuth_deg=30.0, start_elevation_deg=20.0, arc_span_deg=70.0, elevation_drift_deg=10.0, num_frames=32)
    frames, ref = traj.frames(), traj.reference()
    angles = [np.degrees(np.arccos(np.clip((np.trace(T[:3, :3] @ ref[:3, :3].T) - 1) / 2, -1, 1))) for T in frames]
    assert max(angles) < 36.0
    assert abs(angles[0] - angles[-1]) < 1e-6
    eased = traj.eased()
    assert eased[0] == 0 and eased[-1] == pytest.approx(1)


def test_background_homography_identity_at_reference_and_coverage() -> typing.Any:
    bg_w, bg_h = 2250, 1500
    K_bg = background_K(bg_w, bg_h, 130.0)
    H = homography(K, np.eye(3), K_bg)
    # At the reference, the frame center sees the background center.
    p = H @ np.array([(bg_w - 1) / 2, (bg_h - 1) / 2, 1.0])
    np.testing.assert_allclose(p[:2] / p[2], [255.5, 255.5])
    assert covers(K, np.eye(3), K_bg, bg_w, bg_h, 512, 512, 2.0)
    img = np.full((bg_h, bg_w, 3), 7, np.uint8)
    out, defined = warp(img, H, 512, 512)
    assert defined.all() and (out == 7).all()


def test_coverage_check_agrees_with_warp() -> typing.Any:
    """The analytic corner check must never accept a frame the warp leaves partly undefined."""
    bg_w, bg_h = 1500, 1000
    K_bg = background_K(bg_w, bg_h, 130.0)
    rng = np.random.default_rng(5)
    accepted = 0
    for _ in range(200):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        a = np.radians(rng.uniform(0, 60))
        Kx = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        R = np.eye(3) + np.sin(a) * Kx + (1 - np.cos(a)) * Kx @ Kx
        ok = covers(K, R, K_bg, bg_w, bg_h, 512, 512, 2.0)
        _, defined = warp(np.zeros((bg_h, bg_w), np.uint8), homography(K, R, K_bg), 512, 512)
        if ok:
            accepted += 1
            assert defined.all()
    assert accepted > 20
