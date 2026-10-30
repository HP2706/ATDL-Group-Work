"""Matched baseline loss uses identical noise, targets and reference frames."""
from pathlib import Path
from typing import Any
import json

import pytest
import torch

from tests.fixtures.fake_config import fake_train_cfg
from video3d.config import replace_config
from video3d.data.samples import ClipDataset
from video3d.evaluate import evaluate, load_run
from video3d.models.adapter import build_adapter
from video3d.models.encoders import build_kv_encoder
from video3d.models.fake_sam3d import FakeSam3D
from video3d.models.sam3d import build_sam3d
from video3d.train import batch_loss, sample_flow_loss, val_flow_loss


def test_fixed_noise_matches_zero_gated_adapter(fake_root: Path, tmp_path: Path) -> None:
    cfg = fake_train_cfg(fake_root, tmp_path, **{"data.val_asset_frac": 0})
    device = torch.device("cpu")
    sam = build_sam3d(cfg, device)
    encoder = build_kv_encoder(cfg, sam, device)
    adapter = build_adapter(cfg.adapter).eval()
    ds = ClipDataset(cfg, "train", train=False)
    baseline = sample_flow_loss(sam, None, None, ds[0], cfg, device)
    torch.randn(11)
    assert sample_flow_loss(sam, None, None, ds[0], cfg, device) == baseline
    assert sample_flow_loss(sam, encoder, adapter, ds[0], cfg, device) == baseline
    expected = sum(sample_flow_loss(sam, None, None, ds[i], cfg, device)
                   for i in range(len(ds))) / len(ds)
    assert val_flow_loss(sam, None, None, ds, cfg, device) == expected
    batch = [ds[0], ds[1]]
    baseline_batch = batch_loss(sam, None, None, batch, cfg, device,
                                torch.Generator().manual_seed(123))
    adapted_batch = batch_loss(sam, encoder, adapter, batch, cfg, device,
                               torch.Generator().manual_seed(123))
    assert torch.equal(baseline_batch, adapted_batch)


def test_loss_only_baseline_exports_and_resumes(fake_root: Path, tmp_path: Path,
                                               monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = fake_train_cfg(fake_root, tmp_path, **{"data.val_asset_frac": 0, "sam3d.lora": True})
    baseline_cfg, adapter, _ = load_run(cfg, "zero-shot", torch.device("cpu"))
    assert not baseline_cfg.sam3d.lora and adapter is None

    def no_sampling(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Loss-only evaluation must not reconstruct meshes")

    monkeypatch.setattr(FakeSam3D, "sample", no_sampling)
    path = evaluate(cfg, "zero-shot", split="train", limit=2, loss_only=True)
    first = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(first) == 2 and all(row["flow_loss"] > 0 for row in first)
    evaluate(cfg, "zero-shot", split="train", loss_only=True)
    final = [json.loads(line) for line in path.read_text().splitlines()]
    assert final[:2] == first
    assert len({row["sample_id"] for row in final}) == len(final)
    summary = json.loads(path.with_suffix(".summary.json").read_text())
    assert summary["samples"] == len(final)
    assert summary["mean_flow_loss"] == sum(row["flow_loss"] for row in final) / len(final)


def test_raw_and_ema_checkpoint_selection(fake_root: Path, tmp_path: Path) -> None:
    cfg = fake_train_cfg(fake_root, tmp_path)
    directory = tmp_path / "checkpoint"
    directory.mkdir()
    import chz
    (directory / "config.json").write_text(json.dumps(chz.asdict(cfg)))
    adapter = build_adapter(cfg.adapter)
    raw = {key: value.clone() for key, value in adapter.state_dict().items()}
    with torch.no_grad():
        adapter.blocks[0].g_attn.fill_(0.4)
    torch.save({"adapter": raw, "ema": adapter.state_dict(), "extra": {}, "extra_ema": {}},
               directory / "best.pt")
    _, raw_adapter, _ = load_run(replace_config(cfg, **{"eval.checkpoint_weights": "current"}), str(directory), torch.device("cpu"))
    _, ema_adapter, _ = load_run(cfg, str(directory), torch.device("cpu"))
    assert raw_adapter.state_dict()["blocks.0.g_attn"].item() == 0
    assert ema_adapter.state_dict()["blocks.0.g_attn"].item() == pytest.approx(0.4)
