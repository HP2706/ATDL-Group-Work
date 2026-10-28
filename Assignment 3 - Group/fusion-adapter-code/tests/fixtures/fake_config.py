"""Small fake-SAM 3D training config used by the train and eval tests."""

import torch

from vidsam3d.config import load_config
from vidsam3d.data.targets import cache_latents
from vidsam3d.models.sam3d import build_sam3d

D = 32


# Everything except paths: tiny fake model, CPU-sized crops, batches and metric point counts.
FAKE_OVERRIDES = {
    "fake_sam3d": "true", "device": "cpu", "adapter.d_cond": D, "adapter.d_vid": D,
    "adapter.heads": 4, "adapter.query_sets": "[crop,full]", "sam3d.phase_a_layout": "noise",
    "sampler.steps": 4, "sampler.cfg_scale": 1.0, "data.val_split": "train",
    "data.crop.min_side": 16, "data.crop.size": 32, "train.bf16": "false", "train.batch_size": 2,
    "train.effective_batch_size": 2, "train.max_steps": 50, "train.log_every": 1000,
    "train.ckpt_every": 1000, "val.loss_every": 1000, "val.sample_every": 1000,
    "optim.warmup_steps": 5, "optim.lr": 3e-3, "optim.gate_lr_mult": 10, "optim.ema_decay": 0.0,
    "eval.metrics.max_points": 500, "eval.metrics.icp_iterations": 20}


def fake_train_cfg(root, tmp, condition="c3", **overrides):
    """Small fake-SAM 3D config writing everything under `tmp`; caches the train latents."""
    base = {"paths.data_root": root, "paths.output_root": tmp / "out",
            "paths.scratch_root": tmp / "scratch", "paths.cache_root": tmp / "cache",
            **FAKE_OVERRIDES, **overrides}
    cfg = load_config(condition, "local", [f"{k}={v}" for k, v in base.items()])
    cache_latents(cfg, "train", build_sam3d(cfg, torch.device("cpu")), torch.device("cpu"))
    return cfg
