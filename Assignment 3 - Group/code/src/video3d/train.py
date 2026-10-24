from __future__ import annotations

import json
import math
import random
import signal
import time
from pathlib import Path
from typing import Any
import chz
import numpy as np
import torch
from torch.utils.data import default_collate
from contextlib import ExitStack
import wandb
from .adapters import restore_trainable, trainable_state
from .config import ExperimentConfig
from .data import Clips, read_manifest
from .evaluate import evaluate, log_training_assets
from .model import NativeSAM
from .storage import atomic_json, atomic_torch, append_log, rng_state, restore_rng


class StopRequest:
    requested: bool = False

    def receive(self, signum: int, frame: Any) -> None:
        self.requested = True


@torch.no_grad()
def fixed_flow_loss(model: NativeSAM, dataset: Clips, seed: int, device: torch.device) -> float:
    state = rng_state()
    losses: list[float] = []
    # Keep the upstream conditional training loss path (CFG sampling is inference
    # only). The frozen video encoder remains in eval mode throughout training.
    for index in range(len(dataset)):
        torch.manual_seed(seed + index)
        batch = default_collate([dataset[index]])
        with torch.autocast(device.type, dtype=torch.bfloat16):
            loss, _ = model.loss(batch["video"].to(device), batch["reference"].to(device), {key: value.to(device) for key, value in batch["target"].items()}, batch["foreground"].to(device))
        losses.append(float(loss))
    restore_rng(state)
    return float(np.mean(losses))


def train(config: ExperimentConfig) -> dict[str, Any]:
    if config.training.objective != "canonical_coarse_shape":
        raise ValueError("Only the reviewed coarse-shape objective is implemented")
    if not torch.cuda.is_available():
        raise RuntimeError("Training must run on the remote GPU, never locally")
    if config.training.microbatch <= 0 or config.training.accumulation <= 0:
        raise ValueError("Batch and accumulation must be positive")
    random.seed(config.training.seed)
    np.random.seed(config.training.seed)
    torch.manual_seed(config.training.seed)
    rows = read_manifest(Path(config.data.prepared) / "manifest.jsonl")
    train_rows, val_rows = [row for row in rows if row.split == "train"], [row for row in rows if row.split == "val"]
    if not train_rows or not val_rows:
        raise ValueError("An object-disjoint training and validation set is required")
    directory = Path(config.training.output) / config.run_id
    directory.mkdir(parents=True, exist_ok=True)
    atomic_json(directory / "config.json", {"schema_version": 2, "config": chz.asdict(config)})
    start = time.monotonic()
    model = NativeSAM(config.model, config.data.frames, torch.device("cuda"))
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config.training.learning_rate, weight_decay=config.training.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min((step + 1) / max(config.training.warmup, 1), 1.0) * 0.5 * (1 + math.cos(math.pi * min(step, config.training.steps) / max(config.training.steps, 1))))
    random.seed(config.training.seed)
    np.random.seed(config.training.seed)
    torch.manual_seed(config.training.seed)
    counts = {"total": sum(p.numel() for p in model.parameters()), "trainable": sum(p.numel() for p in parameters), "lora_targets": model.lora_targets}
    atomic_json(directory / "parameters.json", counts)
    run = wandb.init(project=config.training.project, entity=config.training.entity, group=config.training.group, name=config.run_id, id=config.run_id, resume="allow", config=chz.asdict(config), job_type="train", settings=wandb.Settings(init_timeout=90))
    run.config.update({"parameter_counts": counts, "active_train_objects": sorted({row.object_id for row in train_rows}), "active_validation_objects": sorted({row.object_id for row in val_rows}), "training_clips": len(train_rows)}, allow_val_change=True)
    data_artifact = wandb.Artifact("navi-v1_5-pilot-manifest", type="dataset", metadata={"seed": config.data.seed, "active_train_objects": sorted({row.object_id for row in train_rows}), "active_validation_objects": sorted({row.object_id for row in val_rows})})
    data_artifact.add_file(str(Path(config.data.prepared) / "manifest.jsonl"))
    data_artifact.add_file(str(Path(config.data.prepared) / "object_splits.json"))
    run.log_artifact(data_artifact)
    dataset = Clips(train_rows, config.data)
    step, batches_seen, epoch = 0, 0, 0
    history: list[dict[str, Any]] = []
    latest = directory / "latest.pt"
    if latest.exists():
        checkpoint = torch.load(latest, map_location="cpu", weights_only=False)
        if checkpoint["config"] != chz.asdict(config):
            raise ValueError("Resume config/provenance mismatch")
        restore_trainable(model, checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        step, batches_seen, epoch = checkpoint["step"], checkpoint["batches_seen"], checkpoint["epoch"]
        history = checkpoint["val_results"]
        restore_rng(checkpoint["rng_state"])
    stop = StopRequest()
    signal.signal(signal.SIGINT, stop.receive)
    signal.signal(signal.SIGTERM, stop.receive)
    log_path = directory / "train.jsonl"
    update_in_progress = False

    def save() -> None:
        if update_in_progress:
            atomic_json(directory / "interrupted_update.json", {"step": step, "note": "Latest complete checkpoint preserved; discard interrupted accumulation on resume."})
            return
        payload = {"schema_version": 2, "config": chz.asdict(config), "model_state_dict": trainable_state(model), "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(), "step": step, "batches_seen": batches_seen, "epoch": epoch, "tokens_seen": batches_seen * config.training.microbatch * 4096, "video_tokens_seen": batches_seen * config.training.microbatch * (config.data.frames // 2) * (config.data.image_size // 16) ** 2, "last_val_tokens": history[-1]["step"] * config.training.microbatch * config.training.accumulation * 4096 if history else 0, "best_val_loss": min([row["validation/flow_loss"] for row in history if "validation/flow_loss" in row], default=None), "best_val_chamfer": min([row["eval/video/chamfer"] for row in history if "eval/video/chamfer" in row], default=None), "val_results": history, "run_id": config.run_id, "wandb_run_id": run.id, "git_head": config.provenance.git_head, "git_worktree_sha256": config.provenance.git_worktree_sha256, "rng_state": rng_state(), "train_log_tail": log_path.read_text().splitlines()[-500:] if log_path.exists() else []}
        atomic_torch(latest, payload)
        atomic_json(directory / "latest.pt.meta.json", {"schema_version": 2, "step": step, "run_id": config.run_id, "git_head": config.provenance.git_head})
        atomic_json(directory / "latest_checkpoint_meta.json", {"path": str(latest), "step": step})

    model.train()
    model.conditioning.encoder.eval()
    with ExitStack() as cleanup:
        cleanup.enter_context(run)
        cleanup.callback(save)
        log_training_assets(run, train_rows, step)
        if step == 0:
            initial = {"validation/flow_loss": fixed_flow_loss(model, Clips(val_rows, config.data), config.training.seed + 1000, torch.device("cuda")), "train/fixed_probe_flow_loss": fixed_flow_loss(model, Clips(train_rows[:1], config.data), config.training.seed + 2000, torch.device("cuda"))}
            history.append({"step": 0, **initial})
            run.log(initial, step=0)
            run.summary.update({"initial_validation_flow_loss": initial["validation/flow_loss"], "validation_clip_count": len(val_rows)})
        save()
        metrics: dict[str, float] = {}
        while step < config.training.steps and not stop.requested and time.monotonic() - start < config.training.max_seconds:
            update_in_progress = True
            optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.0
            before = time.monotonic()
            for _ in range(config.training.accumulation):
                indices: list[int] = []
                for position in range(batches_seen * config.training.microbatch, (batches_seen + 1) * config.training.microbatch):
                    epoch, offset = divmod(position, len(dataset))
                    order = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(config.data.seed + epoch))
                    indices.append(int(order[offset]))
                batch = default_collate([dataset[index] for index in indices])
                video, reference = batch["video"].cuda(), batch["reference"].cuda()
                targets = {key: value.cuda() for key, value in batch["target"].items()}
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss, breakdown = model.loss(video, reference, targets, batch["foreground"].cuda())
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite upstream training loss")
                (loss / config.training.accumulation).backward()
                loss_sum += float(loss.detach()) / config.training.accumulation
                batches_seen += 1
            gradient_norm = float(torch.nn.utils.clip_grad_norm_(parameters, config.training.clip_grad))
            if not math.isfinite(gradient_norm):
                raise FloatingPointError("Non-finite trainable gradient norm")
            if step == 0:
                groups = {"connector": model.conditioning.projector}
                if model.conditioning.pooler is not None:
                    groups["pooler"] = model.conditioning.pooler
                if config.model.video_lora:
                    groups["video_lora"] = model.conditioning.encoder
                if config.model.sam_lora:
                    groups["sam_lora"] = model.generator
                gradient_checks = {name: sum(float(parameter.grad.detach().float().abs().sum()) for parameter in component.parameters() if parameter.requires_grad and parameter.grad is not None) for name, component in groups.items()}
                if any(not math.isfinite(value) or value <= 0 for value in gradient_checks.values()):
                    raise RuntimeError(f"Missing/non-finite gradient through selected training path: {gradient_checks}")
                atomic_json(directory / "first_update_gradients.json", gradient_checks)
                run.summary.update({"first_update_gradients": gradient_checks})
            optimizer.step()
            scheduler.step()
            step += 1
            update_in_progress = False
            metrics = {"train/loss": loss_sum, "train/gradient_norm": gradient_norm, "train/lr": scheduler.get_last_lr()[0], "system/step_seconds": time.monotonic() - before, "system/input_frames_per_second": config.data.frames * config.training.microbatch * config.training.accumulation / (time.monotonic() - before), "system/peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9, "system/peak_reserved_gb": torch.cuda.max_memory_reserved() / 1e9}
            if step % config.training.validate_every == 0:
                metrics["validation/flow_loss"] = fixed_flow_loss(model, Clips(val_rows, config.data), config.training.seed + 1000, torch.device("cuda"))
                metrics["train/fixed_probe_flow_loss"] = fixed_flow_loss(model, Clips(train_rows[:1], config.data), config.training.seed + 2000, torch.device("cuda"))
                history.append({"step": step, "validation/flow_loss": metrics["validation/flow_loss"], "train/fixed_probe_flow_loss": metrics["train/fixed_probe_flow_loss"]})
            append_log(log_path, {"step": step, **metrics})
            run.log(metrics, step=step)
            if step % config.training.eval_every == 0:
                result = evaluate(model, val_rows, config, step, run)
                history.append({"step": step, **result})
            if step % config.training.checkpoint_every == 0:
                save()
        save()
        if not stop.requested and not any(row["step"] == step and "eval/video/chamfer" in row for row in history):
            result = evaluate(model, val_rows, config, step, run)
            history.append({"step": step, **result})
            save()
        artifact = wandb.Artifact(config.run_id + "-checkpoint", type="model")
        artifact.add_file(str(latest))
        run.log_artifact(artifact)
        summary = {"schema_version": 2, "config": chz.asdict(config), "step": step, "preempted": stop.requested, "completed_steps": step == config.training.steps, "elapsed_seconds": time.monotonic() - start, "evaluation": history, "wandb_url": run.url, "parameters": counts}
        atomic_json(directory / "result.json", summary)
        return summary
