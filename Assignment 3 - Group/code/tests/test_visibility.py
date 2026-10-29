from __future__ import annotations

import numpy as np
import trimesh

from video3d.data.visibility import visible_surface_points, exposure_summary


def test_hidden_back_face_is_not_counted_when_projection_hits_the_mask() -> None:
    mesh = trimesh.creation.box(extents=(1, 1, 1))
    points = np.array([[0.0, 0.0, -0.5], [0.0, 0.0, 0.5]])
    transform = np.eye(4)
    transform[2, 3] = 3
    intrinsics = np.array([[100, 0, 63.5], [0, 100, 63.5], [0, 0, 1]])
    instance = np.ones((128, 128), dtype=np.uint8)
    assert visible_surface_points(mesh, points, transform, intrinsics, instance, 1).tolist() == [True, False]
    assert not visible_surface_points(mesh, points, transform, intrinsics, instance * 0, 1).any()


def test_union_and_frame_counts_are_distinct_metrics() -> None:
    result = exposure_summary(np.array([0, 1, 2, 32]), np.array([0, 0, 1, 16]))
    assert result["surface_fraction_seen"] == 0.75
    assert result["mean_frames_seen"] == 8.75
    assert result["surface_fraction_seen_in_16_even_frames"] == 0.5
    assert sum(result["frame_count_histogram"]) == 4
