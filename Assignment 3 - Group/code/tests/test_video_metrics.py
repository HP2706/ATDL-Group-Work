from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from video3d.data.video_metrics import coverage_summary, project_voxels, video_metrics, voxel_self_visibility


def test_equal_occlusion_can_have_different_complementarity() -> None:
    repeated = np.array([[1, 1, 0, 0], [1, 1, 0, 0]], dtype=bool)
    complementary = np.array([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=bool)
    alone = np.ones_like(repeated)
    first = coverage_summary(repeated, alone, 0)
    second = coverage_summary(complementary, alone, 0)
    assert first["additional_fraction"] == 0
    assert second["additional_fraction"] == 0.5
    assert second["best_frame_gain_fraction"] == 0.5
    assert second["external_occlusion_recovery_fraction"] == 1
    assert first["external_occlusion_recovery_fraction"] == 0


def test_unoccluded_reference_has_undefined_external_recovery() -> None:
    observed = np.array([[1, 0], [0, 1]], dtype=bool)
    result = coverage_summary(observed, observed, 0)
    assert result["external_occlusion_recovery_fraction"] is None
    assert result["additional_fraction"] == 0.5


def test_best_frame_gain_distinguishes_one_good_frame() -> None:
    observed = np.array([[1, 0, 0], [1, 1, 1]], dtype=bool)
    result = coverage_summary(observed, np.ones_like(observed), 0)
    assert result["additional_fraction"] == pytest.approx(2 / 3)
    assert result["best_frame_gain_fraction"] == 0


def test_ray_traversal_rejects_back_surface_and_reverses_with_viewpoint() -> None:
    grid = np.zeros((4, 4, 4), bool)
    grid[1, 2, 2] = grid[3, 2, 2] = True
    points = (np.argwhere(grid) + 0.5) / 4 - 0.5
    np.testing.assert_array_equal(voxel_self_visibility(grid, np.array([2., .125, .125]), points), [0, 1])
    np.testing.assert_array_equal(voxel_self_visibility(grid, np.array([-2., .125, .125]), points), [1, 0])


def test_projection_rejects_behind_camera_and_outside_frame() -> None:
    points = np.array([[0., 0., 1.], [0., 0., -1.], [10., 0., 1.]])
    intrinsics = np.array([[4., 0., 3.5], [0., 4., 3.5], [0., 0., 1.]])
    _, _, inside = project_voxels(points, np.eye(3), np.zeros(3), np.ones(3), intrinsics, (8, 8), 8)
    np.testing.assert_array_equal(inside, [1, 0, 0])


def test_axis_parallel_ray_and_single_voxel_are_visible() -> None:
    grid = np.ones((1, 1, 1), bool)
    np.testing.assert_array_equal(voxel_self_visibility(grid, np.array([0., 0., 2.]), np.zeros((1, 3))), [1])


def test_export_report_measures_every_frame(fake_root: Path, tmp_path: Path) -> None:
    out = tmp_path / "metrics.json"
    result = video_metrics(str(fake_root / "train"), str(out))
    assert result["clips"] == 4
    payload = json.loads(out.read_text())
    assert payload["geometry_method"] == "surface_voxel_centre_ray_traversal_proxy"
    for clip in payload["clips"]:
        assert clip["reference_frame"] == 16
        for obj in clip["objects"]:
            assert len(obj["per_frame"]) == 32
            assert obj["min_occlusion"] >= 0
            assert obj["max_occlusion"] <= 1
            proxy = obj["voxel_proxy"]
            assert 0 <= proxy["best_frame_gain_fraction"] <= proxy["additional_fraction"] + 1e-12
            assert 0 <= proxy["total_observed_fraction"] <= 1
            assert len(proxy["per_reference_additional_fraction"]) == 32
            if obj["reference_occlusion"] == 0:
                assert proxy["external_occlusion_recovery_fraction"] is None
    assert out.with_suffix(".clips.csv").is_file()
    assert out.with_suffix(".objects.csv").is_file()
