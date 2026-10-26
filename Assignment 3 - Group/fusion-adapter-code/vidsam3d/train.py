"""Train the fusion adapter with rectified flow matching on the cached GT shape latent.

Only the adapter trains; SAM 3D and the key/value encoder are frozen. All randomness is a
function of (seed, step, micro-batch) or (seed, epoch, item), so resuming reproduces the
uninterrupted run exactly.
"""

from __future__ import annotations

import copy
import json
import logging
import math
import time

import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Sampler

from vidsam3d.config import save_config
from vidsam3d.data.samples import ClipDataset
from vidsam3d.data.targets import convert_pose_to_sam3d, load_latent, target_voxels
from vidsam3d.metrics import voxel_iou
from vidsam3d.models.adapter import build_adapter, count_parameters, pad_video_batch
from vidsam3d.models.encoders import build_kv_encoder, kv_tokens
from vidsam3d.models.sam3d import apply_adapter, build_sam3d
from vidsam3d.utils import (append_jsonl, copy_to_output, find_resume_checkpoint, pick_device,
                            read_jsonl, rng_state, run_dirs, sample_seed, save_checkpoint,
                            seed_everything, set_rng_state, setup_logging)

log = logging.getLogger("vidsam3d")
GATE_NAMES = ("g_attn", "g_ff")


# ---- data stream ----

class StreamBatchSampler(Sampler):
    """Endless seeded stream of (index, epoch) batches, starting at micro-batch `start`."""

    def __init__(self, n: int, batch_size: int, seed: int, start: int = 0):
        self.n, self.batch_size, self.seed, self.start = n, batch_size, seed, start
        self._perms: dict[int, list[int]] = {}

    def item(self, position: int) -> tuple[int, int]:
        """(dataset index, epoch) at a position of the stream."""
        epoch, pos = divmod(position, self.n)
        if epoch not in self._perms:
            g = torch.Generator().manual_seed(sample_seed(self.seed, "epoch", epoch))
            self._perms[epoch] = torch.randperm(self.n, generator=g).tolist()
        return self._perms[epoch][pos], epoch

    def __iter__(self):
        b = self.start
        while True:
            yield [self.item(b * self.batch_size + j) for j in range(self.batch_size)]
            b += 1


def collate_list(batch: list[dict]) -> list[dict]:
    """Keep samples as a list (clip lengths differ; padding happens on tokens)."""
    return batch


# ---- optimisation ----

def param_groups(adapter: torch.nn.Module, ocfg: DictConfig) -> list[dict]:
    """AdamW groups: decay / no decay (biases, embeddings, LayerNorm) / gates (no decay, lr x mult)."""
    decay, no_decay, gates = [], [], []
    for name, p in adapter.named_parameters():
        if name.split(".")[-1] in GATE_NAMES:
            gates.append(p)
        elif p.ndim < 2 or name.startswith(("e_obj", "e_time")) or ".ln_" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    return [{"params": decay, "weight_decay": ocfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
            {"params": gates, "weight_decay": 0.0, "lr": ocfg.lr * ocfg.gate_lr_mult}]


def lr_factor(step: int, warmup: int, max_steps: int, min_ratio: float) -> float:
    """Linear warm-up, then cosine decay from 1 to min_ratio at max_steps."""
    if step < warmup:
        return (step + 1) / warmup
    progress = min((step - warmup) / max(max_steps - warmup, 1), 1.0)
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def ema_update(ema: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    """ema <- decay * ema + (1 - decay) * model."""
    for e, p in zip(ema.parameters(), model.parameters()):
        e.lerp_(p, 1 - decay)


def sample_tau(n: int, lcfg: DictConfig, gen: torch.Generator, device) -> torch.Tensor:
    """Flow time τ in (0, 1): logit-normal(mean, std) or uniform."""
    z = torch.randn(n, generator=gen, device=device)
    if lcfg.tau_dist == "logit_normal":
        return torch.sigmoid(lcfg.tau_mean + lcfg.tau_std * z)
    if lcfg.tau_dist == "uniform":
        return torch.rand(n, generator=gen, device=device)
    raise ValueError(f"unknown loss.tau_dist {lcfg.tau_dist!r}")


# ---- forward ----

def to_device(inputs: dict, device) -> dict:
    """Move the tensors of a model-input dict to the device."""
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}


def reference_images(batch: list[dict], device) -> tuple[torch.Tensor, torch.Tensor]:
    """Stacked full-resolution reference frames (B, H, W, 3) and visible masks (B, H, W)."""
    return (torch.stack([s["inputs"]["ref_rgb"] for s in batch]).to(device),
            torch.stack([s["inputs"]["ref_mask"] for s in batch]).to(device))


def adapted_condition(sam3d, encoder, adapter, batch: list[dict], cfg: DictConfig, device) -> dict:
    """SAM 3D conditioning tokens of the reference frames with the adapter applied."""
    rgb, mask = reference_images(batch, device)
    with torch.no_grad():
        cond = sam3d.condition(rgb, mask)
        kv = [kv_tokens(encoder, to_device(s["inputs"], device), cfg.encoder.kv_source)
              for s in batch]
    v = pad_video_batch(*map(list, zip(*kv)))
    return apply_adapter(cond, adapter, v["tokens"], v["coverage"], v["dt"],
                         list(cfg.adapter.query_sets), v["kv_padding_mask"])


def layout_input(sam3d, mode: str, rgb, mask, tau, gen) -> torch.Tensor:
    """Phase A layout tokens: pure noise, or SAM 3D's zero-shot layout noised with the same τ."""
    noise = torch.randn((len(rgb), *sam3d.layout_shape), generator=gen, device=rgb.device)
    if mode == "noise":
        return noise
    if mode == "zero_shot":
        l1 = sam3d.zero_shot_layout(rgb, mask)
        t = tau.view(-1, *[1] * (noise.dim() - 1))
        return t * l1 + (1 - t) * noise
    raise ValueError(f"unknown sam3d.phase_a_layout {mode!r}")


def flow_loss(sam3d, cond: dict, x1, tau, x0, layout) -> torch.Tensor:
    """Rectified flow matching: || (x1 - x0) - v(τ x1 + (1 - τ) x0) ||^2 (mean)."""
    t = tau.view(-1, *[1] * (x1.dim() - 1))
    v_shape, _ = sam3d.velocity(t * x1 + (1 - t) * x0, tau, cond, layout)
    return ((x1 - x0) - v_shape.float()).pow(2).mean()


def batch_loss(sam3d, encoder, adapter, batch, cfg, device, gen) -> torch.Tensor:
    """Training loss of one micro-batch; τ, noise and layout noise come from `gen`."""
    cond = adapted_condition(sam3d, encoder, adapter, batch, cfg, device)
    x1 = torch.stack([load_latent(cfg.paths.cache_root, s["info"]["asset_id"]) for s in batch])
    x1 = x1.to(device)
    tau = sample_tau(len(batch), cfg.loss, gen, device)
    x0 = torch.randn(x1.shape, generator=gen, device=device)
    rgb, mask = reference_images(batch, device)
    layout = layout_input(sam3d, cfg.sam3d.phase_a_layout, rgb, mask, tau, gen)
    return flow_loss(sam3d, cond, x1, tau, x0, layout)


# ---- validation ----

@torch.no_grad()
def val_flow_loss(sam3d, encoder, adapter, val_ds, cfg, device) -> float:
    """Mean flow loss on the first val samples at a fixed τ grid with fixed noise."""
    taus = torch.tensor(list(cfg.val.tau_grid), device=device)
    losses = []
    for i in range(min(len(val_ds), cfg.val.loss_num_samples)):
        s = val_ds[i]
        cond = adapted_condition(sam3d, encoder, adapter, [s], cfg, device)
        cond = {k: v.expand(len(taus), *v.shape[1:]) for k, v in cond.items()}
        x1 = load_latent(cfg.paths.cache_root, s["info"]["asset_id"]).to(device)
        x1 = x1.expand(len(taus), *x1.shape)
        info = s["info"]
        gen = torch.Generator(device).manual_seed(sample_seed("val", info["clip_id"], info["k"], info["r"]))
        x0 = torch.randn(x1.shape, generator=gen, device=device)
        rgb, mask = (x.expand(len(taus), *x.shape[1:]) for x in reference_images([s], device))
        layout = layout_input(sam3d, cfg.sam3d.phase_a_layout, rgb, mask, taus, gen)
        losses.append(flow_loss(sam3d, cond, x1, taus, x0, layout).item())
    return sum(losses) / len(losses)


@torch.no_grad()
def val_viou(sam3d, encoder, adapter, val_ds, cfg, device) -> float:
    """Mean canonical vIoU of sampled shapes for the first val objects (middle reference)."""
    items = [i for i, x in enumerate(val_ds.index) if "middle" in x["rules"]]
    ious = []
    for i in items[:cfg.val.sample_num_objects]:
        s = val_ds[i]
        info = s["info"]
        cond = adapted_condition(sam3d, encoder, adapter, [s], cfg, device)
        z, _ = sam3d.sample(cond, [sample_seed(info["clip_id"], info["k"], info["r"])], cfg.sampler)
        gt = target_voxels(cfg.paths.data_root, info["voxels"], cfg)
        ious.append(voxel_iou(sam3d.decode_latent(z)[0].cpu(), torch.from_numpy(gt)))
    return sum(ious) / len(ious)


# ---- loop ----

def restore_metrics(scratch_dir, output_dir, step: int):
    """metrics.jsonl on scratch = the persistent copy truncated to `step` (fresh on step 0)."""
    path = scratch_dir / "metrics.jsonl"
    records = [r for r in read_jsonl(output_dir / "metrics.jsonl") if r["step"] <= step]
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return path


def train(cfg: DictConfig, benchmark: bool = False, stop_at: int | None = None) -> dict:
    """Train (or benchmark) an adapter; `stop_at` simulates a pre-empted job in tests."""
    setup_logging()
    if cfg.loss.rotation_weight > 0:
        convert_pose_to_sam3d(None, None, None)            # Phase B: fails until the converter exists
    if benchmark:                                           # never touch the real run's files
        cfg = cfg.copy()
        cfg.run_name = f"{cfg.run_name}_benchmark"
    device = pick_device(cfg.device)
    seed_everything(cfg.seed)
    tcfg, ocfg = cfg.train, cfg.optim
    scratch_dir, output_dir = run_dirs(cfg)
    save_config(cfg, output_dir)

    sam3d = build_sam3d(cfg, device)
    encoder = build_kv_encoder(cfg, sam3d, device)
    adapter = build_adapter(cfg.adapter).to(device)
    ema = copy.deepcopy(adapter).requires_grad_(False)
    log.info("adapter: %.2fM trainable parameters", count_parameters(adapter) / 1e6)

    max_steps = int(tcfg.benchmark_steps if benchmark else tcfg.max_steps)
    accum = tcfg.effective_batch_size // tcfg.batch_size
    assert accum * tcfg.batch_size == tcfg.effective_batch_size, "effective batch % batch size != 0"
    opt = torch.optim.AdamW(param_groups(adapter, ocfg), lr=ocfg.lr, betas=tuple(ocfg.betas))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(
        s, ocfg.warmup_steps, max_steps, ocfg.min_lr / ocfg.lr))

    step, best = 0, -1.0
    ckpt_path = None if benchmark else find_resume_checkpoint(cfg)
    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        adapter.load_state_dict(ckpt["adapter"])
        ema.load_state_dict(ckpt["ema"])
        opt.load_state_dict(ckpt["optimizer"])
        sched.load_state_dict(ckpt["scheduler"])
        set_rng_state(ckpt["rng"])
        step, best = ckpt["step"], ckpt["best_val_viou"]
        log.info("resumed from %s at step %d", ckpt_path, step)
    metrics_path = restore_metrics(scratch_dir, output_dir, step)

    train_ds = ClipDataset(cfg, cfg.data.train_split, train=True)
    loader = DataLoader(train_ds, collate_fn=collate_list, num_workers=cfg.data.num_workers,
                        batch_sampler=StreamBatchSampler(len(train_ds), tcfg.batch_size, cfg.seed,
                                                         start=step * accum))
    val_ds = None if benchmark else ClipDataset(cfg, cfg.data.val_split, train=False)
    autocast = torch.autocast(device.type, dtype=torch.bfloat16, enabled=bool(tcfg.bf16))

    def checkpoint_state() -> dict:
        return {"adapter": adapter.state_dict(), "ema": ema.state_dict(),
                "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                "rng": rng_state(), "step": step, "best_val_viou": best}

    batches, step_times, record = iter(loader), [], {}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    while step < max_steps:
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for micro in range(accum):
            gen = torch.Generator(device).manual_seed(sample_seed(cfg.seed, "train", step, micro))
            with autocast:
                loss = batch_loss(sam3d, encoder, adapter, next(batches), cfg, device, gen) / accum
            loss.backward()
            loss_sum += loss.item()
        grad_norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), ocfg.grad_clip).item()
        opt.step()
        sched.step()
        ema_update(ema, adapter, ocfg.ema_decay)
        step += 1
        step_times.append(time.perf_counter() - t0)

        record = {"step": step, "loss": loss_sum, "lr": sched.get_last_lr()[0],
                  "grad_norm": grad_norm, "s_per_step": step_times[-1], **adapter.gates()}
        if device.type == "cuda":
            record["gpu_mem_gb"] = torch.cuda.max_memory_allocated(device) / 1024**3
        if val_ds is not None and step % cfg.val.loss_every == 0:
            with autocast:
                record["val_loss"] = val_flow_loss(sam3d, encoder, ema, val_ds, cfg, device)
        if val_ds is not None and step % cfg.val.sample_every == 0:
            with autocast:
                record["val_viou"] = val_viou(sam3d, encoder, ema, val_ds, cfg, device)
            if record["val_viou"] > best:
                best = record["val_viou"]
                save_checkpoint({"ema": ema.state_dict(), "step": step, "val_viou": best},
                                cfg, "best.pt", pointer=None)
                (output_dir / "best.json").write_text(json.dumps({"step": step, "val_viou": best}))
        append_jsonl(metrics_path, record)
        if step % tcfg.log_every == 0 or step == max_steps or "val_loss" in record:
            log.info(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                              for k, v in record.items()))
        stopping = stop_at is not None and step >= stop_at
        if not benchmark and (step % tcfg.ckpt_every == 0 or step == max_steps or stopping):
            save_checkpoint(checkpoint_state(), cfg, f"ckpt_{step}.pt", keep=tcfg.keep_ckpts)
            copy_to_output(metrics_path, output_dir)
        if stopping:
            break

    if benchmark:
        return report_benchmark(cfg, step_times, device, output_dir)
    copy_to_output(metrics_path, output_dir)
    return record


def report_benchmark(cfg: DictConfig, step_times: list[float], device, output_dir) -> dict:
    """Print and save s/step (after 3 warm-up steps), peak memory and projected run time."""
    times = step_times[3:] or step_times
    s_per_step = sum(times) / len(times)
    planned = None if OmegaConf.is_missing(cfg.train, "max_steps") else int(cfg.train.max_steps)
    report = {"s_per_step": s_per_step, "batch_size": cfg.train.batch_size,
              "effective_batch_size": cfg.train.effective_batch_size,
              "peak_mem_gb": (torch.cuda.max_memory_allocated(device) / 1024**3
                              if device.type == "cuda" else None),
              "hours_per_10k_steps": s_per_step * 10_000 / 3600,
              "projected_hours": s_per_step * planned / 3600 if planned else None}
    (output_dir / "benchmark.json").write_text(json.dumps(report, indent=1))
    log.info("benchmark %s", report)
    return report
