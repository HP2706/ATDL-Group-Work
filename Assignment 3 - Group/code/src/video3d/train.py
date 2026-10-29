"""Train a conditioning strategy with rectified flow matching on the cached GT shape latent.

The adapter trains, alongside explicitly enabled video/SAM LoRA parameters. All randomness is a
function of (seed, step, micro-batch) or (seed, epoch, item), so resuming reproduces the
uninterrupted run exactly.
"""

from __future__ import annotations
import typing

import copy
import chz
import math
import signal
import sys
from contextlib import nullcontext
import json
import logging
import time

import torch
from tqdm.auto import tqdm
from video3d.loss import flow_loss
from video3d.config import ExperimentConfig, config_dict, OptimConfig, LossConfig
from torch.utils.data import DataLoader, Sampler

from video3d.config import save_config
from video3d.data.samples import VAL
from video3d.data.targets import load_latent
from video3d.metrics import voxel_iou
from video3d.models.adapter import build_adapter, count_parameters, pad_video_batch
from video3d.models.encoders import build_kv_encoder, kv_tokens
from video3d.models.sam3d import apply_adapter, build_sam3d
from video3d.utils import (append_jsonl, copy_to_output, find_resume_checkpoint, pick_device,
                            read_jsonl, rng_state, run_dirs, sample_seed, save_checkpoint,
                            seed_everything, set_rng_state, setup_logging)

log = logging.getLogger("video3d")
GATE_NAMES = ("g_attn", "g_ff")


# ---- data stream ----

class StreamBatchSampler(Sampler):
    """Endless seeded stream of (index, epoch) batches, starting at micro-batch `start`."""

    def __init__(self, n: int, batch_size: int, seed: int, start: int = 0) -> None:
        self.n, self.batch_size, self.seed, self.start = n, batch_size, seed, start
        self._perms: dict[int, list[int]] = {}

    def item(self, position: int) -> tuple[int, int]:
        """(dataset index, epoch) at a position of the stream."""
        epoch, pos = divmod(position, self.n)
        if epoch not in self._perms:
            g = torch.Generator().manual_seed(sample_seed(self.seed, "epoch", epoch))
            self._perms[epoch] = torch.randperm(self.n, generator=g).tolist()
        return self._perms[epoch][pos], epoch

    def __iter__(self) -> typing.Any:
        b = self.start
        while True:
            yield [self.item(b * self.batch_size + j) for j in range(self.batch_size)]
            b += 1


def collate_list(batch: list[dict]) -> list[dict]:
    """Keep samples as a list (clip lengths differ; padding happens on tokens)."""
    return batch


# ---- optimisation ----

def param_groups(adapter: torch.nn.Module, ocfg: OptimConfig) -> list[dict]:
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


def sample_tau(n: int, lcfg: LossConfig, gen: torch.Generator, device: typing.Any) -> torch.Tensor:
    """Flow time τ in (0, 1): logit-normal(mean, std) or uniform."""
    z = torch.randn(n, generator=gen, device=device)
    if lcfg.tau_dist == "logit_normal":
        return torch.sigmoid(lcfg.tau_mean + lcfg.tau_std * z)
    if lcfg.tau_dist == "uniform":
        return torch.rand(n, generator=gen, device=device)
    raise ValueError(f"unknown loss.tau_dist {lcfg.tau_dist!r}")


# ---- forward ----

def to_device(inputs: dict, device: typing.Any) -> dict:
    """Move the tensors of a model-input dict to the device."""
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}


def reference_images(batch: list[dict], device: typing.Any) -> tuple[torch.Tensor, torch.Tensor]:
    """Stacked full-resolution reference frames (B, H, W, 3) and visible masks (B, H, W)."""
    return (torch.stack([s["inputs"]["ref_rgb"] for s in batch]).to(device),
            torch.stack([s["inputs"]["ref_mask"] for s in batch]).to(device))


def adapted_condition(sam3d: typing.Any, encoder: typing.Any, adapter: typing.Any, batch: list[dict], cfg: ExperimentConfig, device: typing.Any) -> dict:
    """SAM 3D conditioning tokens of the reference frames with the adapter applied."""
    rgb, mask = reference_images(batch, device)
    with torch.no_grad():
        cond = sam3d.condition(rgb, mask)
    feature_context = nullcontext() if cfg.encoder.vjepa.video_lora else torch.no_grad()
    with feature_context:
        kv = [kv_tokens(encoder, to_device(s["inputs"], device), cfg.encoder.kv_source)
              for s in batch]
    v = pad_video_batch(*map(list, zip(*kv)))
    return apply_adapter(cond, adapter, v["tokens"], v["coverage"], v["dt"],
                         list(cfg.adapter.query_sets), v["kv_padding_mask"])


def layout_input(sam3d: typing.Any, mode: str, rgb: typing.Any, mask: typing.Any, tau: typing.Any, gen: typing.Any) -> dict:
    """Phase A layout latents: pure noise, or SAM 3D's zero-shot layout noised with the same τ.

    With the real model the choice cannot change v_shape (shape tokens do not attend to layout
    tokens, docs/sam3d_internals.md item 7), so `noise` is the default.
    """
    noise = {k: torch.randn((len(rgb), *shape), generator=gen, device=rgb.device)
             for k, shape in sam3d.layout_shape.items()}
    if mode == "noise":
        return noise
    if mode == "zero_shot":
        l1 = sam3d.zero_shot_layout(rgb, mask)
        t = lambda x: tau.view(-1, *[1] * (x.dim() - 1))
        return {k: t(n) * l1[k] + (1 - t(n)) * n for k, n in noise.items()}
    raise ValueError(f"unknown sam3d.phase_a_layout {mode!r}")


def batch_loss(sam3d: typing.Any, encoder: typing.Any, adapter: typing.Any, batch: typing.Any, cfg: typing.Any, device: typing.Any, gen: typing.Any) -> torch.Tensor:
    """Training loss of one micro-batch; τ, noise and layout noise come from `gen`."""
    cond = adapted_condition(sam3d, encoder, adapter, batch, cfg, device)
    x1 = torch.stack([s["target"] if "target" in s else load_latent(cfg, s["info"]["asset_id"]) for s in batch])
    x1 = x1.to(device)
    tau = sample_tau(len(batch), cfg.loss, gen, device)
    x0 = torch.randn(x1.shape, generator=gen, device=device)
    rgb, mask = reference_images(batch, device)
    layout = layout_input(sam3d, cfg.sam3d.phase_a_layout, rgb, mask, tau, gen)
    return flow_loss(sam3d, cond, x1, tau, x0, layout)


# ---- validation ----

@torch.no_grad()
def val_flow_loss(sam3d: typing.Any, encoder: typing.Any, adapter: typing.Any, val_ds: typing.Any, cfg: typing.Any, device: typing.Any) -> float:
    """Mean flow loss on the first val samples at a fixed τ grid with fixed noise."""
    taus = torch.tensor(list(cfg.val.tau_grid), device=device)
    losses = []
    for i in tqdm(range(min(len(val_ds), cfg.val.loss_num_samples)), desc="Validation loss", unit="video"):
        s = val_ds[i]
        cond = (adapted_condition(sam3d, encoder, adapter, [s], cfg, device) if adapter is not None
                else sam3d.condition(*reference_images([s], device)))
        cond = {k: v.expand(len(taus), *v.shape[1:]) for k, v in cond.items()}
        x1 = (s["target"] if "target" in s else load_latent(cfg, s["info"]["asset_id"])).to(device)
        x1 = x1.expand(len(taus), *x1.shape)
        info = s["info"]
        gen = torch.Generator(device).manual_seed(sample_seed("val", info["clip_id"], info["k"], info["r"]))
        x0 = torch.randn(x1.shape, generator=gen, device=device)
        rgb, mask = (x.expand(len(taus), *x.shape[1:]) for x in reference_images([s], device))
        layout = layout_input(sam3d, cfg.sam3d.phase_a_layout, rgb, mask, taus, gen)
        losses.append(flow_loss(sam3d, cond, x1, taus, x0, layout).item())
    return sum(losses) / len(losses)


@torch.no_grad()
def val_viou(sam3d: typing.Any, encoder: typing.Any, adapter: typing.Any, val_ds: typing.Any, cfg: typing.Any, device: typing.Any) -> float:
    """Mean canonical vIoU of sampled shapes for the first val objects (middle reference)."""
    items = [i for i, x in enumerate(val_ds.index) if "middle" in x["rules"] or "reference" in x["rules"]]
    ious = []
    for i in tqdm(items[:cfg.val.sample_num_objects], desc="Validation shapes", unit="video"):
        s = val_ds[i]
        info = s["info"]
        cond = adapted_condition(sam3d, encoder, adapter, [s], cfg, device)
        z, _ = sam3d.sample(cond, [sample_seed(info["clip_id"], info["k"], info["r"])], cfg.sampler)
        from video3d.data import ground_truth
        gt = ground_truth(val_ds, info["asset_id"], cfg)
        ious.append(voxel_iou(sam3d.decode_latent(z)[0].cpu(), torch.from_numpy(gt)))
    return sum(ious) / len(ious)


# ---- loop ----

def restore_metrics(scratch_dir: typing.Any, output_dir: typing.Any, step: int) -> typing.Any:
    """metrics.jsonl on scratch = the persistent copy truncated to `step` (fresh on step 0)."""
    path = scratch_dir / "metrics.jsonl"
    records = [r for r in read_jsonl(output_dir / "metrics.jsonl") if r["step"] <= step]
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return path


def train(cfg: ExperimentConfig, benchmark: bool = False, stop_at: int | None = None) -> dict:
    """Train (or benchmark) an adapter; `stop_at` simulates a pre-empted job in tests."""
    setup_logging()
    if cfg.loss.rotation_weight > 0:                        # converter: targets.convert_pose_to_sam3d
        raise NotImplementedError("Phase B (rotation loss) is not implemented")
    if sys.platform == "darwin":
        raise RuntimeError("Training and benchmarks must run remotely, never on the laptop")
    if benchmark:
        cfg = chz.replace(cfg, run_name=f"{cfg.run_name}_benchmark")
    from video3d.provenance import source_identity, data_identity
    source_hash, data_hash = source_identity(), data_identity(cfg)
    device = pick_device(cfg.device)
    seed_everything(cfg.seed)
    tcfg, ocfg = cfg.train, cfg.optim
    scratch_dir, output_dir = run_dirs(cfg)
    config_path = output_dir / "config.json"
    if config_path.exists() and not benchmark:
        previous = json.loads(config_path.read_text())
        if previous != config_dict(cfg):
            raise ValueError("Resume config/provenance mismatch; use a new run name")
    save_config(cfg, output_dir)

    sam3d = build_sam3d(cfg, device)
    encoder = build_kv_encoder(cfg, sam3d, device)
    adapter = build_adapter(cfg.adapter).to(device)
    ema = copy.deepcopy(adapter).requires_grad_(False)
    extra = {f"encoder.{name}": p for name, p in encoder.named_parameters() if p.requires_grad}
    extra.update({f"sam.{name}": p for name, p in sam3d.named_parameters() if p.requires_grad})
    extra_ema = {name: p.detach().clone() for name, p in extra.items()}
    from video3d.logging import RunLogger
    from video3d.checkpoint import use_ema
    logger = RunLogger(cfg)
    log.info("adapter: %.2fM trainable parameters", count_parameters(adapter) / 1e6)

    from video3d.data import build_dataset
    from video3d.epochs import EpochBatchSampler, update_sizes, epoch_previews
    train_ds = build_dataset(cfg, cfg.data.train_split, train=True)
    epoch_sizes = update_sizes(len(train_ds), tcfg.effective_batch_size, tcfg.batch_size)
    epoch_steps = len(epoch_sizes)
    use_epochs = tcfg.epochs > 0 and not benchmark
    max_steps = int(tcfg.benchmark_steps if benchmark else tcfg.epochs * epoch_steps if use_epochs else tcfg.max_steps)
    accum = tcfg.effective_batch_size // tcfg.batch_size
    assert accum * tcfg.batch_size == tcfg.effective_batch_size, "effective batch % batch size != 0"
    groups = param_groups(adapter, ocfg)
    if extra:
        groups.append({"params": list(extra.values()), "weight_decay": ocfg.weight_decay})
    parameters = [p for group in groups for p in group["params"]]
    parameter_counts = {"adapter":count_parameters(adapter),
                        "sam_lora":sum(p.numel() for p in sam3d.parameters() if p.requires_grad),
                        "video_lora":sum(p.numel() for p in encoder.parameters() if p.requires_grad),
                        "trainable_total":sum(p.numel() for p in parameters),
                        "sam_total":sum(p.numel() for p in sam3d.parameters()),
                        "video_total":sum(p.numel() for p in encoder.parameters())}
    from video3d.storage import atomic_json
    atomic_json(output_dir / "parameter_counts.json", parameter_counts)
    opt = torch.optim.AdamW(groups, lr=ocfg.lr, betas=tuple(ocfg.betas))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(
        s, ocfg.warmup_steps, max_steps, ocfg.min_lr / ocfg.lr))

    step, best = 0, -1.0
    epoch_loss_sum, epoch_examples = 0.0, 0
    epoch_history: list[dict[str, typing.Any]] = []
    ckpt_path = None if benchmark else find_resume_checkpoint(cfg)
    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        adapter.load_state_dict(ckpt["adapter"])
        ema.load_state_dict(ckpt["ema"])
        opt.load_state_dict(ckpt["optimizer"])
        sched.load_state_dict(ckpt["scheduler"])
        set_rng_state(ckpt["rng"])
        if ckpt["source_identity"] != source_hash or ckpt["data_identity"] != data_hash:
            raise ValueError("Checkpoint source/data identity mismatch")
        if ckpt["config"] != chz.asdict(cfg):
            raise ValueError("Checkpoint config/provenance mismatch")
        with torch.no_grad():
            for name, p in extra.items():
                p.copy_(ckpt["extra"][name])
                extra_ema[name].copy_(ckpt["extra_ema"][name])
        step, best = ckpt["step"], ckpt["best_val_viou"]
        epoch_loss_sum = ckpt.get("epoch_loss_sum", 0.0)
        epoch_examples = ckpt.get("epoch_examples", 0)
        epoch_history = ckpt.get("epoch_history", [])
        log.info("resumed from %s at step %d", ckpt_path, step)
    metrics_path = restore_metrics(scratch_dir, output_dir, step)

    sampler = EpochBatchSampler(len(train_ds), tcfg.effective_batch_size, tcfg.batch_size,
                                tcfg.epochs, cfg.seed, start_step=step) if use_epochs else StreamBatchSampler(
                                    len(train_ds), tcfg.batch_size, cfg.seed, start=step * accum)
    loader = DataLoader(train_ds, collate_fn=collate_list, num_workers=cfg.data.num_workers,
                        batch_sampler=sampler)
    val_ds = None if benchmark or not cfg.data.val_asset_frac else build_dataset(cfg, VAL, train=False)
    autocast = torch.autocast(device.type, dtype=torch.bfloat16, enabled=bool(tcfg.bf16))
    if use_epochs and step and step % epoch_steps == 0 and cfg.val.epoch_media_samples:
        completed_epoch = step // epoch_steps
        completed_media = output_dir / "epochs" / f"epoch-{completed_epoch:02d}" / "reconstructions.json"
        if not completed_media.exists():
            with autocast:
                recovered_media = epoch_previews(cfg, sam3d, encoder, adapter, device,
                                                 output_dir / "epochs", completed_epoch, logger.run)
            logger.log(recovered_media, step)

    def checkpoint_state() -> dict:
        return {"adapter": adapter.state_dict(), "ema": ema.state_dict(),
                "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                "rng": rng_state(), "step": step, "best_val_viou": best,
                "schema_version": 3, "config": chz.asdict(cfg),
                "source_identity":source_hash,"data_identity":data_hash,
                "extra": {name: p.detach().cpu().clone() for name, p in extra.items()},
                "extra_ema": extra_ema, "epoch_loss_sum":epoch_loss_sum,
                "epoch_examples":epoch_examples, "epoch_history":epoch_history,
                "epoch":step // epoch_steps if use_epochs else None,
                "run_id":cfg.run_name, "wandb_run_id":logger.run.id if logger.run is not None else None,
                "train_log_tail":metrics_path.read_text().splitlines()[-500:] if metrics_path.exists() else []}

    records = read_jsonl(metrics_path)
    batches, step_times, record = iter(loader), [], records[-1] if records else {"step":step}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    stopping_signal = False
    def request_stop(signum: int, frame: object) -> None:
        nonlocal stopping_signal
        stopping_signal = True
    previous_handlers = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    started = time.monotonic()
    update_in_progress = False
    total_videos = len(train_ds) * tcfg.epochs if use_epochs else max_steps * tcfg.effective_batch_size
    completed_videos = step // epoch_steps * len(train_ds) + epoch_examples if use_epochs else step * tcfg.effective_batch_size
    progress = tqdm(total=total_videos, initial=completed_videos, desc="Training", unit="video")
    try:
        while step < max_steps and not stopping_signal and time.monotonic() - started < tcfg.max_seconds:
            update_in_progress = True
            t0 = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            loss_sum = 0.0
            sizes = epoch_sizes[step % epoch_steps] if use_epochs else [tcfg.batch_size] * accum
            update_examples = sum(sizes)
            for micro, size in enumerate(sizes):
                gen = torch.Generator(device).manual_seed(sample_seed(cfg.seed, "train", step, micro))
                batch = next(batches)
                assert len(batch) == size
                with autocast:
                    loss = batch_loss(sam3d, encoder, adapter, batch, cfg, device, gen) * size / update_examples
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite training loss")
                loss.backward()
                loss_sum += loss.item()
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, ocfg.grad_clip).item()
            component_gradients = {
                f"gradient/{prefix}":math.sqrt(sum(float(p.grad.detach().float().square().sum())
                    for name, p in extra.items() if name.startswith(prefix + ".") and p.grad is not None))
                for prefix in ("sam", "encoder")}
            if not math.isfinite(grad_norm):
                raise FloatingPointError("Non-finite gradient norm")
            opt.step()
            sched.step()
            ema_update(ema, adapter, ocfg.ema_decay)
            with torch.no_grad():
                for name, p in extra.items():
                    extra_ema[name].lerp_(p, 1 - ocfg.ema_decay)
            step += 1
            update_in_progress = False
            step_times.append(time.perf_counter() - t0)
            progress.update(update_examples)
            progress.set_postfix(loss=f"{loss_sum:.4f}", step=step)

            record = {"step": step, "loss": loss_sum, "lr": sched.get_last_lr()[0],
                      "grad_norm": grad_norm, "s_per_step": step_times[-1],
                      **component_gradients, **adapter.gates()}
            media: dict[str, typing.Any] = {}
            epoch_finished = use_epochs and step % epoch_steps == 0
            if use_epochs:
                epoch_loss_sum += loss_sum * update_examples
                epoch_examples += update_examples
            if epoch_finished:
                assert epoch_examples == len(train_ds)
                epoch_record = {"epoch":step // epoch_steps, "epoch/loss":epoch_loss_sum / epoch_examples,
                                "epoch/examples":epoch_examples, "epoch/steps":epoch_steps}
                record.update(epoch_record)
                epoch_history.append(epoch_record)
                from video3d.storage import atomic_json
                atomic_json(output_dir / "epochs.json", epoch_history)
                epoch_loss_sum, epoch_examples = 0.0, 0
                save_checkpoint(checkpoint_state(), cfg, f"epoch_{step // epoch_steps:02d}.pt")
                if cfg.val.epoch_media_samples:
                    with autocast:
                        media = epoch_previews(cfg, sam3d, encoder, adapter, device,
                                               output_dir / "epochs", step // epoch_steps, logger.run)
            if device.type == "cuda":
                record["gpu_mem_gb"] = torch.cuda.max_memory_allocated(device) / 1024**3
            if val_ds is not None and step % cfg.val.loss_every == 0:
                with use_ema(extra, extra_ema), autocast:
                    record["val_loss"] = val_flow_loss(sam3d, encoder, ema, val_ds, cfg, device)
            if val_ds is not None and step % cfg.val.sample_every == 0:
                with use_ema(extra, extra_ema), autocast:
                    record["val_viou"] = val_viou(sam3d, encoder, ema, val_ds, cfg, device)
                if record["val_viou"] > best:
                    best = record["val_viou"]
                    save_checkpoint({**checkpoint_state(), "ema": ema.state_dict(), "step": step, "val_viou": best},
                                    cfg, "best.pt", pointer=None)
                    (output_dir / "best.json").write_text(json.dumps({"step": step, "val_viou": best}))
            append_jsonl(metrics_path, record)
            if step % tcfg.log_every == 0 or step == max_steps or "val_loss" in record:
                log.info(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                                  for k, v in record.items()))
            logger.log({**record, **media}, step)
            stopping = (stop_at is not None and step >= stop_at) or stopping_signal or time.monotonic() - started >= tcfg.max_seconds
            if not benchmark and (step % tcfg.ckpt_every == 0 or step == max_steps or stopping):
                save_checkpoint(checkpoint_state(), cfg, f"ckpt_{step}.pt", keep=tcfg.keep_ckpts)
                copy_to_output(metrics_path, output_dir)
            if stopping:
                break

    finally:
        progress.close()
        if not benchmark and not update_in_progress:
            save_checkpoint(checkpoint_state(), cfg, f"ckpt_{step}.pt", keep=tcfg.keep_ckpts)
        if update_in_progress:
            from video3d.storage import atomic_json
            atomic_json(output_dir / "interrupted_update.json", {"last_complete_step":step,
                        "resume":"Use the previous complete checkpoint; partial accumulation was discarded."})
        copy_to_output(metrics_path, output_dir)
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        logger.finish()
    if benchmark:
        return report_benchmark(cfg, step_times, device, output_dir)
    copy_to_output(metrics_path, output_dir)
    return record


def report_benchmark(cfg: ExperimentConfig, step_times: list[float], device: typing.Any, output_dir: typing.Any) -> dict:
    """Print and save s/step (after 3 warm-up steps), peak memory and projected run time."""
    times = step_times[3:] or step_times
    s_per_step = sum(times) / len(times)
    planned = int(cfg.train.max_steps)
    report = {"s_per_step": s_per_step, "batch_size": cfg.train.batch_size,
              "effective_batch_size": cfg.train.effective_batch_size,
              "peak_mem_gb": (torch.cuda.max_memory_allocated(device) / 1024**3
                              if device.type == "cuda" else None),
              "hours_per_10k_steps": s_per_step * 10_000 / 3600,
              "projected_hours": s_per_step * planned / 3600 if planned else None}
    (output_dir / "benchmark.json").write_text(json.dumps(report, indent=1))
    log.info("benchmark %s", report)
    return report
