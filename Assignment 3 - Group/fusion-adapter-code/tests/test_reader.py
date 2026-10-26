"""Reader: data format v0.1 conventions on the fake dataset."""

import numpy as np
import pytest

from vidsam3d.data import reader


def clips(root, split=None):
    return [reader.load_clip(root, e) for e in reader.read_manifest(root, split)]


def test_manifest_and_splits(fake_root):
    assert len(reader.read_manifest(fake_root)) == 4
    assert {e["split"] for e in reader.read_manifest(fake_root, "eval")} == {"eval"}


def test_image_size_comes_from_cameras_json(fake_root):
    clip = clips(fake_root, "eval")[0]
    assert clip.image_size == (64, 64)
    assert reader.load_rgb(clip, 0).shape == (64, 64, 3)
    assert clip.num_frames == 32


def test_voxel_projection_sanity_check(fake_root):
    """Data format §8: voxel centres project onto the amodal mask (> 0.9)."""
    for clip in clips(fake_root):
        for k in range(1, clip.num_objects + 1):
            for t in (0, 16, 31):
                assert reader.voxel_projection_check(clip, k, t) > 0.9


def test_projection_check_catches_a_wrong_convention(fake_root):
    """Reading voxel indices as (z, y, x) instead of (x, y, z) must fail the check."""
    clip = clips(fake_root, "eval")[0]
    R_co, t_co, s_co = reader.camera_frame_pose(clip, 1, 16)
    x_o = reader.voxel_centers(reader.load_object_voxels(clip, 1))[:, ::-1]
    x_c = (x_o * s_co) @ R_co.T + t_co
    uv = np.round(reader.project(x_c, reader.intrinsics(clip))).astype(int)
    ok = (uv >= 0).all(1) & (uv < 64).all(1)
    amodal = reader.load_amodal_mask_gt(clip, 1, 16)
    assert amodal[uv[ok, 1], uv[ok, 0]].mean() < 0.9


def test_camera_frame_pose_matches_world_composition(fake_root):
    clip = clips(fake_root, "eval")[1]
    R_co, t_co, s_co = reader.camera_frame_pose(clip, 2, 7)
    R_cw, t_cw = reader.camera_extrinsics(clip, 7)
    R_wo, t_wo, s_wo = reader.object_world_pose(clip, 2)
    x_o = np.random.default_rng(0).uniform(-0.5, 0.5, (10, 3))
    x_c_direct = (R_cw @ ((x_o * s_wo) @ R_wo.T + t_wo).T).T + t_cw
    np.testing.assert_allclose((x_o * s_co) @ R_co.T + t_co, x_c_direct, atol=1e-12)
    np.testing.assert_allclose(R_co @ R_co.T, np.eye(3), atol=1e-9)
    assert np.linalg.det(R_co) == pytest.approx(1.0)


def test_masks_and_depth_are_consistent(fake_root):
    for clip in clips(fake_root, "eval"):
        depth, inst = reader.load_depth_gt(clip, 5), reader.load_instance_mask(clip, 5)
        assert np.array_equal(np.isfinite(depth), inst > 0)
        for k in range(1, clip.num_objects + 1):
            vis, amodal = reader.load_visible_mask(clip, k, 5), reader.load_amodal_mask_gt(clip, k, 5)
            assert vis.any() and not (vis & ~amodal).any()   # visible ⊆ amodal


def test_visibility_stats_shape(fake_root):
    clip = clips(fake_root, "eval")[1]
    vf = reader.visibility(clip)
    assert vf.shape == (32, 2) and (vf > 0).all() and (vf <= 1).all()


def test_voxel_centers_and_gltf_matrix():
    vox = np.zeros((64, 64, 64), bool)
    vox[0, 0, 0] = vox[63, 63, 63] = True
    np.testing.assert_allclose(reader.voxel_centers(vox), [[-0.5 + 1 / 128] * 3, [0.5 - 1 / 128] * 3])
    np.testing.assert_array_equal(reader.CANONICAL_FROM_GLTF @ [0, 1, 0], [0, 0, 1])  # glTF Y-up -> Z-up
