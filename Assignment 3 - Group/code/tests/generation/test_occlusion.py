from __future__ import annotations

import chz
import numpy as np
import pytest

from video3d.data.generation.config import Config
from video3d.data.generation.occlusion import occlusion_failures, sample_occlusion_scene


def test_random_mode_acceptance_is_unchanged() -> None:
    assert occlusion_failures(np.ones((32, 2)), Config()) == []


def test_recoverable_mode_rejects_clear_and_permanently_hidden_clips() -> None:
    cfg = chz.replace(Config(), occlusion_mode="recoverable")
    assert occlusion_failures(np.ones((32, 2)), cfg)
    assert occlusion_failures(np.full((32, 2), 0.5), cfg)
    fractions = np.ones((32, 2))
    fractions[8:24, 0] = 0.5
    assert occlusion_failures(fractions, cfg) == []


def test_occlusion_is_required_in_reference_not_only_somewhere() -> None:
    cfg = chz.replace(Config(), occlusion_mode="recoverable")
    fractions = np.ones((32, 2))
    fractions[:10, 0] = 0.5
    assert any("reference" in text for text in occlusion_failures(fractions, cfg))


def test_pair_sampler_preserves_nonintersection_and_determinism() -> None:
    cfg = chz.replace(Config(), occlusion_mode="recoverable", n_objects_choices=[2],
                      n_objects_probs=[1.], precheck_resolution=32, num_frames=8)
    coords = np.linspace(-0.45, 0.45, 8)
    points = np.stack(np.meshgrid(coords, coords, coords), axis=-1).reshape(-1, 3)
    first, trajectory = sample_occlusion_scene(np.random.default_rng(42), cfg, ["a", "b"], [points, points])
    again, second = sample_occlusion_scene(np.random.default_rng(42), cfg, ["a", "b"], [points, points])
    distance = np.linalg.norm(first[0].t_wo - first[1].t_wo)
    assert distance > first[0].bounding_radius + first[1].bounding_radius
    for a, b in zip(first, again):
        np.testing.assert_array_equal(a.t_wo, b.t_wo)
    np.testing.assert_array_equal(trajectory.frames(), second.frames())


def test_pair_mode_requires_two_objects() -> None:
    with pytest.raises(ValueError, match="exactly two"):
        sample_occlusion_scene(np.random.default_rng(0), Config(), ["a"], [np.zeros((1, 3))])
