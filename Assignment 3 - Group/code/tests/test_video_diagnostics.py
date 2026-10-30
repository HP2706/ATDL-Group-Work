"""Video routing, causality controls and fixed unseen-surface metric contracts."""
from pathlib import Path
from typing import Any

import chz
import numpy as np
import pytest
import torch

from video3d.diagnostics import attention_statistics, evaluate_video, surface_region_metrics
from video3d.metrics import voxel_points
from video3d.models.adapter import VideoFusionAdapter


def test_uniform_attention_enrichment_excludes_padding() -> None:
    weights = torch.tensor([[[[0.25, 0.25, 0.5, 0.0, 0.0]]]])
    coverage = torch.tensor([[1., 0., 0.5, 1., 1.]])
    dt = torch.tensor([[-1, -1, 0, 0, 0]])
    padding = torch.tensor([[False, False, False, True, True]])
    stats = attention_statistics(weights, coverage, dt, padding)
    assert stats["object_attention_mass"] == 0.5
    assert stats["object_enrichment"] == 1.0
    assert stats["background_enrichment"] == 1.0
    assert stats["temporal_groups"][0]["mass"] == 0.5
    assert stats["temporal_groups"][0]["enrichment"] == pytest.approx(0.75)
    assert stats["temporal_groups"][1]["enrichment"] == pytest.approx(1.5)


def test_gates_and_diagnostic_path_preserve_output() -> None:
    torch.manual_seed(12)
    adapter = VideoFusionAdapter(d_cond=8, d_vid=8, heads=2)
    condition, video = torch.randn(1, 3, 8), torch.randn(1, 4, 8)
    coverage, offsets = torch.ones(1, 4), torch.tensor([[-1, -1, 0, 0]])
    stats: dict[str, Any] = {}
    adapter(condition, video, coverage, offsets, diagnostics=stats)
    assert stats["block0"]["video_update_ratio"] == 0.0
    assert stats["conditioning_change_ratio"] == 0.0
    for block in adapter.blocks:
        block.g_attn.data.fill_(0.4)
        block.g_ff.data.fill_(0.2)
    expected = adapter(condition, video, coverage, offsets)
    actual = adapter(condition, video, coverage, offsets, diagnostics=stats)
    torch.testing.assert_close(actual, expected)
    assert stats["block0"]["video_update_ratio"] > 0
    disabled = adapter(condition, video, coverage, offsets, video_disabled=True)
    torch.testing.assert_close(disabled, adapter(condition, video*20, coverage, offsets, video_disabled=True))
    assert not torch.allclose(actual, disabled)


def test_visibility_regions_use_fixed_reference_and_correct_frames(tmp_path: Path) -> None:
    prediction = torch.zeros(8, 8, 8, dtype=torch.bool)
    prediction[1, 1, 1] = True
    prediction[6, 6, 6] = True
    points = voxel_points(prediction).numpy()
    points = np.concatenate([points, np.array([[0., 0., 0.]], dtype=np.float32)])
    exposure = tmp_path / "exposure.npz"
    np.savez(exposure, points=points, visible=np.array([[1, 0, 0], [0, 1, 0]], dtype=bool))
    metrics = surface_region_metrics(prediction, exposure, 0, [0, 1])
    assert metrics["surface/reference_visible/recall_002"] == 1.0
    assert metrics["surface/newly_revealed/recall_002"] == 1.0
    assert metrics["surface/never_observed/recall_002"] == 0.0
    assert metrics["surface/reference_unseen/recall_002"] == 0.5
    empty = surface_region_metrics(torch.zeros_like(prediction), exposure, 0, [0, 1])
    assert empty["surface/newly_revealed/recall_002"] == 0.0
    assert empty["surface/newly_revealed/completeness_distance"] is None
    no_region = surface_region_metrics(prediction, exposure, 0, [0])
    assert no_region["surface/newly_revealed/points"] == 0
    assert no_region["surface/newly_revealed/recall_002"] is None


def test_paired_controls_keep_reference_noise_and_report_deltas(fake_root: Path, tmp_path: Path) -> None:
    from tests.fixtures.fake_config import fake_train_cfg
    from video3d.data import build_dataset, ground_truth
    from video3d.evaluate import wrong_video_partner
    from video3d.models.adapter import build_adapter
    from video3d.models.encoders import build_kv_encoder
    from video3d.models.sam3d import build_sam3d
    cfg = fake_train_cfg(fake_root, tmp_path)
    cfg = chz.replace(cfg, eval=chz.replace(cfg.eval, video_diagnostics=True))
    device = torch.device("cpu")
    sam = build_sam3d(cfg, device)
    encoder = build_kv_encoder(cfg, sam, device)
    adapter = build_adapter(cfg.adapter).eval()
    for block in adapter.blocks:
        block.g_attn.data.fill_(0.4)
    dataset = build_dataset(cfg, cfg.data.eval_split, train=False, clip_len=cfg.eval.k)
    sample = dataset[0]
    reference = sample["inputs"]["ref_rgb"].clone()
    other = dataset[wrong_video_partner(dataset, 0)]
    truth = torch.from_numpy(ground_truth(dataset, sample["info"]["asset_id"], cfg))
    result = evaluate_video(sample, other, truth, sam, encoder, adapter, cfg, device)
    assert set(result.controls) == {"correct", "repeated", "wrong", "disabled"}
    assert len({item["sampling_seed"] for item in result.controls.values()}) == 1
    torch.testing.assert_close(sample["inputs"]["ref_rgb"], reference)
    correct = result.controls["correct"]["viou"]
    for mode in ["repeated", "wrong", "disabled"]:
        record = result.controls[mode]
        assert record["delta_correct_minus_control"]["viou"] == pytest.approx(correct-record["viou"])
    assert result.diagnostics["block0"]["video_update_ratio"] > 0
    assert sum(group["mass"] for group in result.diagnostics["block0"]["temporal_groups"]) == pytest.approx(1.0)


def test_summary_keeps_paired_counts_and_ignores_absent_regions() -> None:
    from video3d.diagnostics import summarize_video_records
    records = [{"video_diagnostics": {}, "video_controls": {
        "correct": {"viou": 0.8, "sampling_seed": 99},
        "disabled": {"viou": 0.5, "delta_correct_minus_control": {"viou": 0.3}}}},
               {"video_diagnostics": {}, "video_controls": {
        "correct": {"viou": 0.6},
        "disabled": {"viou": 0.5, "delta_correct_minus_control": {"viou": 0.1}}}}]
    summary = summarize_video_records(records)
    assert summary["video_benefit/disabled/viou"] == pytest.approx(0.2)
    assert summary["video_benefit/disabled/viou/sample_count"] == 2
    assert summary["video_diagnostic_samples"] == 2
    assert all("sampling_seed" not in key for key in summary)


def test_shared_evaluator_persists_diagnostic_summary(fake_root: Path, tmp_path: Path,
                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    import json
    from tests.fixtures.fake_config import fake_train_cfg
    from video3d.config import save_config
    from video3d.evaluate import evaluate
    from video3d.models.adapter import build_adapter
    cfg = fake_train_cfg(fake_root, tmp_path, **{"data.val_asset_frac": 0})
    cfg = chz.replace(cfg, eval=chz.replace(cfg.eval, video_diagnostics=True),
                      logging=chz.replace(cfg.logging, media_samples=0))
    run = Path(cfg.paths.output_root)/cfg.run_name
    save_config(cfg, run)
    state = build_adapter(cfg.adapter).state_dict()
    torch.save({"adapter": state, "ema": state}, run/"best.pt")
    from video3d.logging import RunLogger
    logged_steps: list[int | None] = []

    def capture_log(self: RunLogger, values: dict[str, Any], step: int | None = None) -> None:
        logged_steps.append(step)

    monkeypatch.setattr(RunLogger, "log", capture_log)
    path = evaluate(cfg, str(run), split="train", limit=2)
    assert logged_steps == [None, None]
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(records) == 2
    assert all(set(record["video_controls"]) == {"correct", "repeated", "wrong", "disabled"} for record in records)
    summary = json.loads(path.with_suffix(".summary.json").read_text())
    assert summary["video_diagnostic_samples"] == 2
    assert summary["video_benefit/disabled/viou"] == 0.0
