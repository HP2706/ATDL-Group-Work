from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import torch

from video3d.epochs import EpochBatchSampler, update_sizes, epoch_previews


def test_all_examples_once_per_epoch_with_partial_batches() -> None:
    sampler = EpochBatchSampler(101, 32, 8, 10, 17)
    batches = list(sampler)
    counts = Counter(item for batch in batches for item in batch)
    assert len(counts) == 1010 and set(counts.values()) == {1}
    assert set(counts) == {(index, epoch) for index in range(101) for epoch in range(10)}
    assert all(len({epoch for _, epoch in batch}) == 1 for batch in batches)
    assert [len(batch) for batch in batches[:13]] == [8] * 12 + [5]


def test_resume_starts_at_optimizer_update_boundary() -> None:
    full = list(EpochBatchSampler(101, 32, 8, 3, 17))
    resumed = list(EpochBatchSampler(101, 32, 8, 3, 17, start_step=5))
    assert resumed == full[17:]
    assert update_sizes(101, 32, 8) == [[8, 8, 8, 8]] * 3 + [[5]]


def test_exact_epoch_count_and_checkpoint_resume(fake_root: Path, tmp_path: Path) -> None:
    from tests.fixtures.fake_config import fake_train_cfg
    from video3d.train import train

    cfg = fake_train_cfg(fake_root, tmp_path, **{
        "train.epochs":3, "train.batch_size":2, "train.effective_batch_size":4,
        "data.val_asset_frac":0, "val.epoch_media_samples":0})
    assert train(cfg, stop_at=3)["step"] == 3
    assert train(cfg)["step"] == 6
    output: Path = Path(cfg.paths.output_root) / cfg.run_name
    epochs: list[dict] = json.loads((output / "epochs.json").read_text())
    assert [row["epoch"] for row in epochs] == [1, 2, 3]
    assert [row["epoch/examples"] for row in epochs] == [6, 6, 6]
    assert all(torch.isfinite(torch.tensor(row["epoch/loss"])) for row in epochs)
    assert len(list(output.glob("epoch_*.pt"))) == 3


def test_epoch_previews_preserve_fixed_inputs(fake_root: Path, tmp_path: Path) -> None:
    from tests.fixtures.fake_config import fake_train_cfg
    from video3d.models.sam3d import build_sam3d
    from video3d.models.encoders import build_kv_encoder
    from video3d.models.adapter import build_adapter
    from video3d.epochs import epoch_previews

    cfg = fake_train_cfg(fake_root, tmp_path, **{
        "data.val_asset_frac":0, "val.epoch_media_samples":2})
    device = torch.device("cpu")
    sam = build_sam3d(cfg, device)
    encoder = build_kv_encoder(cfg, sam, device)
    adapter = build_adapter(cfg.adapter)
    for epoch in (1, 2):
        result = epoch_previews(cfg, sam, encoder, adapter, device, tmp_path / "media", epoch)
        assert result["epoch/reconstruction_count"] == 2
    first = json.loads((tmp_path / "media/epoch-01/reconstructions.json").read_text())
    second = json.loads((tmp_path / "media/epoch-02/reconstructions.json").read_text())
    assert [(row["clip_id"], row["object"], row["reference_frame"]) for row in first] == [
        (row["clip_id"], row["object"], row["reference_frame"]) for row in second]


def test_epoch_table_serializes_video_masks_and_meshes(fake_root: Path, tmp_path: Path) -> None:
    import wandb
    from tests.fixtures.fake_config import fake_train_cfg
    from video3d.models.sam3d import build_sam3d
    from video3d.models.encoders import build_kv_encoder
    from video3d.models.adapter import build_adapter

    cfg = fake_train_cfg(fake_root, tmp_path, **{
        "data.val_asset_frac":0, "val.epoch_media_samples":2})
    device = torch.device("cpu")
    sam = build_sam3d(cfg, device)
    run = wandb.init(project="epoch-media-tests", mode="offline", dir=str(tmp_path))
    media = epoch_previews(
        cfg, sam, build_kv_encoder(cfg, sam, device), build_adapter(cfg.adapter),
        device, tmp_path / "media", 1, run)
    run.log(media, step=1)
    run.finish()
    assert len(media["epoch/reconstructions"].data) == 2
