"""Shape metrics and the statistics used by analyze."""

import numpy as np
import pytest
import torch

from vidsam3d.analyze import cluster_bootstrap_ci, paired, wilcoxon_p
from vidsam3d.config import load_config
from vidsam3d.metrics import normalize_points, shape_metrics, voxel_iou, voxel_points, voxelize

MCFG = load_config("c3", "local", ["eval.metrics.max_points=1500", "eval.metrics.icp_iterations=30"]).eval.metrics  # CPU-sized


def box(rot_deg: float = 0.0) -> torch.Tensor:
    """Solid 1.0 x 0.5 x 0.3 box at 64^3, rotated about z (voxelised from dense samples)."""
    g = torch.linspace(-0.5, 0.5, 160)
    p = torch.stack(torch.meshgrid(g * 0.9, g * 0.45, g * 0.27, indexing="ij"), -1).reshape(-1, 3)
    a = np.deg2rad(rot_deg)
    R = torch.tensor([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]], dtype=torch.float32)
    return voxelize((p @ R.T) * 2, 64)                        # canonical [-0.5, 0.5] -> [-1, 1]


def test_voxel_iou_basics():
    a = torch.zeros(4, 4, 4, dtype=torch.bool)
    a[:2] = True
    assert voxel_iou(a, a) == 1.0 and voxel_iou(a, ~a) == 0.0
    assert voxel_iou(torch.zeros_like(a), torch.zeros_like(a)) == 1.0


def test_normalize_points_centres_and_scales_longest_side():
    p = normalize_points(voxel_points(box()) * 0.3 + 0.1)        # shrink and shift first
    lo, hi = p.min(0).values, p.max(0).values
    torch.testing.assert_close((lo + hi) / 2, torch.zeros(3), atol=1e-6, rtol=0)
    assert float((hi - lo).max()) == pytest.approx(2.0)


def test_identical_shapes_are_perfect():
    pytest.importorskip("pytorch3d")
    m = shape_metrics(box(), box(), MCFG)
    assert m["viou"] == 1.0 and m["viou_aligned"] == 1.0 and m["chamfer"] == pytest.approx(0, abs=1e-6)
    assert m["fscore@0.02"] == 1.0 and m["fscore@0.05"] == 1.0 and not m["empty_pred"]


def test_rotated_copy_is_recovered_by_icp():
    pytest.importorskip("pytorch3d")
    m = shape_metrics(box(15), box(), MCFG)
    assert m["viou_aligned"] > 0.9 and m["viou_aligned"] > m["viou"] + 0.1
    assert m["fscore@0.05"] > 0.95


def test_empty_prediction():
    m = shape_metrics(torch.zeros(64, 64, 64, dtype=torch.bool), box(), MCFG)
    assert m["empty_pred"] and m["viou"] == 0.0 and m["chamfer"] is None


def test_paired_deltas_use_shared_samples_only():
    base = [{"sample_id": "a", "clip_id": "1", "viou": 0.5}, {"sample_id": "b", "clip_id": "1", "viou": None}]
    other = [{"sample_id": "a", "clip_id": "1", "viou": 0.7}, {"sample_id": "b", "clip_id": "1", "viou": 0.9},
             {"sample_id": "c", "clip_id": "2", "viou": 0.1}]
    d, clips = paired(base, other, "viou")
    np.testing.assert_allclose(d, [0.2]) and clips.tolist() == ["1"]


def test_cluster_bootstrap_is_seeded_and_resamples_clusters():
    rng = np.random.default_rng(0)
    values = rng.normal(0.1, 1.0, 300)
    clusters = np.repeat(np.arange(100), 3).astype(str)
    a = cluster_bootstrap_ci(values, clusters, 2000, seed=0)
    assert a == cluster_bootstrap_ci(values, clusters, 2000, seed=0)
    assert a != cluster_bootstrap_ci(values, clusters, 2000, seed=1)
    assert a[0] < values.mean() < a[1]
    # one cluster: every resample is the same cluster -> a zero-width interval
    lo, hi = cluster_bootstrap_ci(values[:3], np.array(["x"] * 3), 100, seed=0)
    assert lo == pytest.approx(hi)


def test_wilcoxon():
    assert wilcoxon_p(np.zeros(10)) == 1.0
    assert wilcoxon_p(np.linspace(0.1, 1.0, 30)) < 0.001
