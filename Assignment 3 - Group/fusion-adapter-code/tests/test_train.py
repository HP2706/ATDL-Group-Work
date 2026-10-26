"""Training loop with FakeSam3D: loss falls, exact resume, benchmark, helpers."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tests.fixtures.make_fake_dataset import make_fake_dataset
from vidsam3d.config import load_config
from vidsam3d.data.samples import ClipDataset
from vidsam3d.data.targets import apply_canonical_permutation, cache_latents, load_latent
from vidsam3d.models.adapter import VideoFusionAdapter, build_adapter
from vidsam3d.models.encoders import build_kv_encoder
from vidsam3d.models.sam3d import FakeSam3D, apply_adapter, build_sam3d
from vidsam3d.train import StreamBatchSampler, lr_factor, param_groups, train, val_flow_loss
from vidsam3d.utils import read_jsonl

D = 32


@pytest.fixture(scope="module")
def two_sample_root(tmp_path_factory):
    """Train split = one clip with two objects -> exactly 2 training samples."""
    return make_fake_dataset(tmp_path_factory.mktemp("ds2"), splits=("train",), seeds={"train": [1]})


def fake_train_cfg(root, tmp, condition="c3", **overrides):
    """Small fake-SAM 3D config writing everything under `tmp`."""
    base = {"paths.data_root": root, "paths.output_root": tmp / "out",
            "paths.scratch_root": tmp / "scratch", "paths.cache_root": tmp / "cache",
            "fake_sam3d": "true", "device": "cpu", "adapter.d_cond": D, "adapter.d_vid": D,
            "adapter.heads": 4, "adapter.query_sets": "[crop,full]",
            "sam3d.phase_a_layout": "noise", "sampler.steps": 4, "sampler.cfg_scale": 1.0,
            "data.val_split": "train", "data.crop.min_side": 16, "data.crop.size": 32,
            "train.bf16": "false", "train.batch_size": 2, "train.effective_batch_size": 2,
            "train.max_steps": 50, "train.log_every": 1000, "train.ckpt_every": 1000,
            "val.loss_every": 1000, "val.sample_every": 1000, "optim.warmup_steps": 5,
            "optim.lr": 3e-3, "optim.gate_lr_mult": 10, "optim.ema_decay": 0.0}
    base.update(overrides)
    cfg = load_config(condition, "local", [f"{k}={v}" for k, v in base.items()])
    cache_latents(cfg, "train", build_sam3d(cfg, torch.device("cpu")), torch.device("cpu"))
    return cfg


def adapter_state(cfg):
    """(latest checkpoint, output run dir) of a run."""
    out = Path(cfg.paths.output_root) / cfg.run_name
    latest = json.loads((out / "latest.json").read_text())
    return torch.load(out / latest["checkpoint"], weights_only=False), out


def test_loss_decreases_over_50_steps(two_sample_root, tmp_path):
    cfg = fake_train_cfg(two_sample_root, tmp_path)
    device = torch.device("cpu")
    sam3d, val_ds = build_sam3d(cfg, device), ClipDataset(cfg, "train", train=False)
    encoder = build_kv_encoder(cfg, sam3d, device)
    before = val_flow_loss(sam3d, encoder, build_adapter(cfg.adapter), val_ds, cfg, device)
    train(cfg)
    ckpt, out = adapter_state(cfg)
    adapter = build_adapter(cfg.adapter)
    adapter.load_state_dict(ckpt["adapter"])
    after = val_flow_loss(sam3d, encoder, adapter, val_ds, cfg, device)
    print(f"val flow loss {before:.4f} -> {after:.4f}")
    assert after < 0.9 * before, (before, after)
    records = read_jsonl(out / "metrics.jsonl")
    assert [r["step"] for r in records] == list(range(1, 51))
    assert np.mean([r["loss"] for r in records[-10:]]) < np.mean([r["loss"] for r in records[:10]])
    assert (out / "config.yaml").is_file()


@pytest.mark.parametrize("condition", ["c1", "c2", "c3_nomask"])
def test_other_conditions_train(fake_root, tmp_path, condition):
    cfg = fake_train_cfg(fake_root, tmp_path, condition, **{"train.max_steps": 3})
    assert train(cfg)["step"] == 3


def test_resume_matches_uninterrupted_training(fake_root, tmp_path):
    over = {"train.max_steps": 8, "train.effective_batch_size": 4, "val.loss_every": 4,
            "optim.ema_decay": 0.9}
    full = fake_train_cfg(fake_root, tmp_path / "a", **over)
    train(full)
    parts = fake_train_cfg(fake_root, tmp_path / "b", **over)
    assert train(parts, stop_at=4)["step"] == 4                       # "pre-empted" job
    assert json.loads((tmp_path / "b/out/c3_seed0/latest.json").read_text())["step"] == 4
    assert train(parts)["step"] == 8                                  # resume: auto
    a, out_a = adapter_state(full)
    b, out_b = adapter_state(parts)
    for key in ("adapter", "ema"):
        for name, tensor in a[key].items():
            assert torch.equal(tensor, b[key][name]), (key, name)
    rec_a, rec_b = read_jsonl(out_a / "metrics.jsonl"), read_jsonl(out_b / "metrics.jsonl")
    assert [r["step"] for r in rec_b] == list(range(1, 9))
    for ra, rb in zip(rec_a, rec_b):
        assert ra["loss"] == rb["loss"] and ra.get("val_loss") == rb.get("val_loss")


def test_validation_sampling_and_best_checkpoint(fake_root, tmp_path):
    cfg = fake_train_cfg(fake_root, tmp_path, **{"train.max_steps": 4, "val.sample_every": 2,
                                                 "val.loss_every": 2})
    record = train(cfg)
    assert 0 <= record["val_viou"] <= 1 and "val_loss" in record
    best = json.loads((tmp_path / "out/c3_seed0/best.json").read_text())
    assert (tmp_path / "out/c3_seed0/best.pt").is_file() and best["step"] in (2, 4)


def test_benchmark_mode(fake_root, tmp_path):
    cfg = fake_train_cfg(fake_root, tmp_path, **{"train.benchmark_steps": 5})
    report = train(cfg, benchmark=True)
    assert report["s_per_step"] > 0 and report["projected_hours"] is not None
    assert (tmp_path / "out/c3_seed0_benchmark/benchmark.json").is_file()
    assert not (tmp_path / "out/c3_seed0/latest.json").exists()     # no checkpoints


def test_zero_shot_layout_mode_and_phase_b_guard(fake_root, tmp_path):
    cfg = fake_train_cfg(fake_root, tmp_path, **{"train.max_steps": 2,
                                                 "sam3d.phase_a_layout": "zero_shot"})
    assert train(cfg)["step"] == 2
    cfg.loss.rotation_weight = 0.1
    with pytest.raises(NotImplementedError, match="ours_to_sam3d"):
        train(cfg)


# ---- pieces ----

def test_gradients_reach_conditioning_tokens_through_checkpointed_blocks():
    sam3d = FakeSam3D(d=D, activation_checkpointing=True)
    rgb = torch.randint(0, 255, (2, 64, 64, 3), dtype=torch.uint8)
    mask = torch.zeros(2, 64, 64, dtype=torch.bool)
    mask[:, 20:40, 10:30] = True
    cond = {k: v.clone().requires_grad_(True) for k, v in sam3d.condition(rgb, mask).items()}
    x = torch.randn(2, *sam3d.latent_shape)
    v, _ = sam3d.velocity(x, torch.rand(2), cond, torch.randn(2, *sam3d.layout_shape))
    v.sum().backward()
    assert all(c.grad.abs().sum() > 0 for c in cond.values())
    assert all(p.grad is None for p in sam3d.parameters())


def test_apply_adapter_only_changes_query_sets():
    adapter = VideoFusionAdapter(d_cond=D, d_vid=D, heads=4)
    with torch.no_grad():
        for blk in adapter.blocks:
            blk.g_attn.fill_(0.5)
    cond = {name: torch.randn(1, 4, D) for name in ("crop", "full", "mask")}
    f, m, dt = torch.randn(1, 6, D), torch.rand(1, 6), torch.zeros(1, 6, dtype=torch.long)
    out = apply_adapter(cond, adapter, f, m, dt, ["crop", "full"])
    assert out["mask"] is cond["mask"]
    assert not torch.equal(out["crop"], cond["crop"]) and out["full"].shape == cond["full"].shape
    with pytest.raises(AssertionError, match="not in conditioning sets"):
        apply_adapter(cond, adapter, f, m, dt, ["pointmap"])


def test_param_groups_and_schedule():
    cfg = load_config("c3", "local", ["optim.gate_lr_mult=10"])
    adapter = VideoFusionAdapter(d_cond=D, d_vid=D, heads=4)
    decay, no_decay, gates = param_groups(adapter, cfg.optim)
    assert gates["lr"] == pytest.approx(1e-3) and len(gates["params"]) == 4
    assert any(p is adapter.e_obj for p in no_decay["params"])
    assert all(p is not adapter.blocks[0].ln_q.weight for p in decay["params"])
    assert all(p.ndim == 2 for p in decay["params"])                      # no biases
    assert any(p is adapter.in_proj.bias for p in no_decay["params"])
    assert lr_factor(0, 10, 100, 0.01) == pytest.approx(0.1)
    assert lr_factor(10, 10, 100, 0.01) == pytest.approx(1.0)
    assert lr_factor(100, 10, 100, 0.01) == pytest.approx(0.01)


def test_stream_sampler_is_seeded_and_restartable():
    s = StreamBatchSampler(n=5, batch_size=2, seed=3)
    first = [b for b, _ in zip(iter(s), range(6))]
    later = [b for b, _ in zip(iter(StreamBatchSampler(5, 2, 3, start=2)), range(4))]
    assert first[2:] == later
    assert sorted(i for i, _ in sum(first[:2], []) + [first[2][0]]) == [0, 1, 2, 3, 4]   # epoch 0


def test_cache_latents_and_canonical_permutation(fake_root, tmp_path):
    cfg = fake_train_cfg(fake_root, tmp_path)
    assert load_latent(cfg.paths.cache_root, "box").shape == (8, 4, 4, 4)
    summary = json.loads((tmp_path / "cache/latents/roundtrip_iou_train.json").read_text())["summary"]
    assert summary["num_assets"] == 2
    vox = np.zeros((64, 64, 64), bool)
    vox[1, 2, 3] = True
    P = [[0, 1, 0], [0, 0, -1], [1, 0, 0]]                 # x' = y, y' = -z, z' = x
    out = apply_canonical_permutation(vox, P)
    np.testing.assert_array_equal(np.argwhere(out)[0], [2, 63 - 3, 1])
    with pytest.raises(AssertionError):
        apply_canonical_permutation(vox, [[1, 1, 0], [0, 0, 1], [0, 0, 1]])
