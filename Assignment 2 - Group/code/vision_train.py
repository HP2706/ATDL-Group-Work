"""Train either released image model on CIFAR-10 or CIFAR-100.

Run ``python vision_train.py -- --help`` for Google Fire help.
"""

import json
import math
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal

import fire
import torch
import wandb
from pydantic import Field, model_validator
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from wandb.sdk.wandb_run import Run

from src.training.common import (
    append_metric,
    code_fingerprints,
    create_run_directory,
    load_checkpoint,
    resolve_device,
    save_checkpoint,
    save_config,
    seed_everything,
    StopRequested,
    TrainingConfig,
)
from src.training.results import export_vision
from src.vision_experiment.data import CIFARExperimentDataset
from src.vision_experiment.model import make_cnn, make_resnet18k

if TYPE_CHECKING:
    from sweep import SweepConfig


class VisionConfig(TrainingConfig):
    architecture: Literal["resnet", "cnn"]
    dataset: Literal["cifar10", "cifar100"]
    data_dir: str = "data"
    output_dir: str = "runs/vision"
    width: int = Field(default=64, gt=0)
    label_noise: float = Field(default=0.0, ge=0.0, le=1.0)
    augmentation: bool = True
    optimizer: Literal["adam", "sgd"] | None = None
    schedule: Literal["constant", "inverse_sqrt", "dynamic_drop"] | None = None
    learning_rate: float | None = Field(default=None, gt=0)
    momentum: float = Field(default=0.0, ge=0.0, lt=1.0)
    weight_decay: float = Field(default=0.0, ge=0.0)
    max_steps: int | None = Field(default=None, gt=0)
    epochs: int | None = Field(default=None, gt=0)
    batch_size: int = Field(default=128, gt=0)
    eval_every: int = Field(default=512, gt=0)
    checkpoint_every: int = Field(default=512, gt=0)
    drop_factor: float = Field(default=0.1, gt=0.0, lt=1.0)
    drop_patience: int = Field(default=2000, gt=0)
    download: bool = False
    wandb_project: str | None = None
    wandb_entity: str | None = None

    @classmethod
    def from_sweep(cls, sweep: "SweepConfig", optimizer: str, width: int, noise: float,
                   sample_count: int | None, train_size: int) -> "VisionConfig":
        import chz

        variant = sweep.variants[optimizer]
        selected_count = sample_count if sample_count is not None else train_size
        if variant.measure_every_steps is not None:
            interval = variant.measure_every_steps
        else:
            assert variant.measure_every_epochs is not None
            interval = math.ceil(selected_count / sweep.batch_size) * variant.measure_every_epochs
        shared = {name: value for name, value in chz.asdict(sweep).items() if name in cls.model_fields}
        protocol = {name: value for name, value in chz.asdict(variant).items() if name in cls.model_fields}
        return cls.model_validate({
            **shared,
            **protocol,
            "optimizer": optimizer,
            "width": width,
            "label_noise": noise,
            "sample_count": sample_count,
            "eval_every": interval,
            "checkpoint_every": interval,
            "data_dir": str(sweep.data_path),
            "output_dir": str(sweep.condition_directory(optimizer, width, noise, sample_count)),
            "results_dir": str(sweep.results_path),
            "download": False,
        })

    @model_validator(mode="after")
    def resolve_protocol(self) -> "VisionConfig":
        if self.optimizer is None:
            self.optimizer = "adam" if self.architecture == "resnet" else "sgd"
        if self.schedule is None:
            self.schedule = "constant" if self.optimizer == "adam" else "inverse_sqrt"
        if self.learning_rate is None:
            self.learning_rate = 1e-4 if self.optimizer == "adam" else 0.1
        if self.epochs is None and self.max_steps is None:
            if self.optimizer == "adam":
                self.epochs = 4000
            else:
                self.max_steps = 1_000_000 if self.architecture == "cnn" and self.dataset == "cifar100" else 500_000
        if self.epochs is not None and self.max_steps is not None:
            raise ValueError("choose either epochs or max_steps")
        if self.optimizer == "adam" and self.momentum:
            raise ValueError("momentum is only valid for SGD")
        if self.wandb_entity is not None and self.wandb_project is None:
            raise ValueError("wandb_project is required when wandb_entity is set")
        return self


def record_evaluation(run_dir: Path, metric: dict[str, Any], wandb_run: Run | None) -> None:
    append_metric(run_dir, metric)
    if wandb_run is not None:
        wandb_run.log({
            "epoch": metric["epoch"],
            "global_step": metric["step"],
            "train_error": metric["train"]["error"],
            "test_error": metric["test"]["error"],
            "train_loss": metric["train"]["loss"],
            "test_loss": metric["test"]["loss"],
        }, step=metric["step"])


def publish_result(result_path: Path, wandb_run: Run | None) -> None:
    print(f"Parquet saved locally: {result_path}", flush=True)
    if wandb_run is not None:
        wandb_run.finish()


def make_optimizer(model: nn.Module, config: VisionConfig) -> torch.optim.Optimizer:
    if config.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=float(config.learning_rate), weight_decay=config.weight_decay)
    return torch.optim.SGD(
        model.parameters(), lr=float(config.learning_rate), momentum=config.momentum, weight_decay=config.weight_decay
    )


def learning_rate_at_step(config: VisionConfig, step: int, drop_count: int) -> float:
    base = float(config.learning_rate)
    if config.schedule == "inverse_sqrt":
        return base / math.sqrt(1 + step // 512)
    if config.schedule == "dynamic_drop":
        return base * config.drop_factor**drop_count
    return base


def evaluate(model: nn.Module, dataset: CIFARExperimentDataset, batch_size: int, device: torch.device) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    with torch.no_grad():
        for images, labels, _ in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            total_loss += float(F.cross_entropy(logits, labels, reduction="sum"))
            total_correct += int((logits.argmax(dim=1) == labels).sum())
            total += labels.numel()
    return {"loss": total_loss / total, "error": 1 - total_correct / total}


def save_predictions(
    model: nn.Module, dataset: CIFARExperimentDataset, batch_size: int, device: torch.device, path: Path
) -> None:
    model.eval()
    predictions: list[Tensor] = []
    indices: list[Tensor] = []
    with torch.no_grad():
        for images, _, batch_indices in DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0):
            predictions.append(model(images.to(device)).argmax(dim=1).cpu())
            indices.append(batch_indices)
    torch.save({"indices": torch.cat(indices), "predictions": torch.cat(predictions)}, path)


def train(
    config: VisionConfig,
    resume: str | None = None,
    progress_callback: Callable[[int], None] | None = None,
) -> Path:
    seed_everything(config.seed)
    device = resolve_device(config.device)
    run_dir = Path(resume) if resume else create_run_directory(
        config.output_dir, f"{config.dataset}-{config.architecture}-k{config.width}", config.seed
    )
    if resume:
        saved = VisionConfig.model_validate_json((run_dir / "config.json").read_text())
        if saved != config:
            raise ValueError("resume config does not match the saved run")
    else:
        save_config(run_dir, config)
    wandb_run = wandb.init(
        project=config.wandb_project,
        entity=config.wandb_entity,
        name=run_dir.name,
        id=run_dir.name,
        resume="allow",
        dir=str(run_dir),
        config=config.model_dump(),
    ) if config.wandb_project is not None else None
    train_set = CIFARExperimentDataset(
        config.dataset, config.data_dir, True, config.download, config.sample_count,
        config.label_noise, config.augmentation, config.seed,
    )
    train_eval_set = CIFARExperimentDataset(
        config.dataset, config.data_dir, True, False, config.sample_count,
        config.label_noise, False, config.seed,
    )
    test_set = CIFARExperimentDataset(config.dataset, config.data_dir, False, config.download, None, 0.0, False, config.seed)
    classes = train_set.num_classes
    model = (make_resnet18k(config.width, classes) if config.architecture == "resnet" else make_cnn(config.width, classes)).to(device)
    metadata_path = run_dir / "run_metadata.json"
    if not resume:
        metadata_path.write_text(json.dumps({
            "train_size": len(train_set),
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "train_split": "train", "test_split": "test",
            "error_unit": "fraction", "loss_unit": "mean cross-entropy in nats",
            "measurement_index": "zero-based position in this run's evaluation history",
            "global_step": "optimizer updates; not inferred from the authors' arrays",
            "trial_index": "null because original Mlist position is unknown",
            "code_fingerprints": code_fingerprints(Path(__file__).resolve().parent),
        }, indent=2) + "\n")
    optimizer = make_optimizer(model, config)
    step, epoch, batch_offset = load_checkpoint(run_dir, model, optimizer) if resume else (0, 0, 0)
    if progress_callback is not None:
        progress_callback(step)
    best_train_loss = math.inf
    stale_steps = 0
    drop_count = 0
    state_path = run_dir / "schedule.json"
    if resume and state_path.exists():
        state = json.loads(state_path.read_text())
        best_train_loss = float(state["best_train_loss"])
        stale_steps = int(state["stale_steps"])
        drop_count = int(state["drop_count"])
    checkpoint_epoch = epoch
    checkpoint_offset = batch_offset
    stop = StopRequested()
    stop.install()
    batches_per_epoch = math.ceil(len(train_set) / config.batch_size)
    total_updates = config.max_steps if config.max_steps is not None else int(config.epochs) * batches_per_epoch
    progress = tqdm(
        total=total_updates,
        initial=step,
        desc=f"{config.dataset} {config.architecture} width={config.width} noise={config.label_noise:g}",
        unit="batch",
        mininterval=5.0,
        dynamic_ncols=True,
    )
    while (config.max_steps is None or step < config.max_steps) and (config.epochs is None or epoch < config.epochs):
        epoch_started = time.perf_counter()
        epoch_start_step = step
        resumed_offset = batch_offset
        train_set.epoch = epoch
        order = torch.randperm(len(train_set), generator=torch.Generator().manual_seed(config.seed + epoch))
        batches = DataLoader(Subset(train_set, order.tolist()), batch_size=config.batch_size, num_workers=0)
        model.train()
        for offset, (images, labels, _) in enumerate(batches):
            if offset < batch_offset:
                continue
            if config.max_steps is not None and step >= config.max_steps:
                break
            for group in optimizer.param_groups:
                group["lr"] = learning_rate_at_step(config, step, drop_count)
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            step += 1
            progress.update(1)
            if progress_callback is not None:
                progress_callback(step)
            if config.schedule == "dynamic_drop":
                current_loss = float(loss.detach())
                if current_loss < best_train_loss:
                    best_train_loss, stale_steps = current_loss, 0
                else:
                    stale_steps += 1
                    if stale_steps >= config.drop_patience:
                        drop_count += 1
                        stale_steps = 0
            next_epoch = epoch + int(offset + 1 == len(batches))
            next_offset = 0 if next_epoch > epoch else offset + 1
            checkpoint_epoch, checkpoint_offset = next_epoch, next_offset
            if step % config.eval_every == 0 or (config.max_steps is not None and step == config.max_steps):
                train_eval = evaluate(model, train_eval_set, config.batch_size, device)
                test_eval = evaluate(model, test_set, config.batch_size, device)
                metric = {"step": step, "epoch": next_epoch, "train": train_eval, "test": test_eval}
                record_evaluation(run_dir, metric, wandb_run)
                progress.set_postfix(train_error=f"{train_eval['error']:.3f}", test_error=f"{test_eval['error']:.3f}")
                tqdm.write(json.dumps(metric))
                model.train()
            if step % config.checkpoint_every == 0 or (config.max_steps is not None and step == config.max_steps):
                save_checkpoint(run_dir, model, optimizer, step, next_epoch, next_offset)
                state_path.write_text(json.dumps({"best_train_loss": best_train_loss, "stale_steps": stale_steps, "drop_count": drop_count}))
            if stop.requested:
                if step % config.eval_every != 0:
                    record_evaluation(run_dir, {"step": step, "epoch": next_epoch, "train": evaluate(model, train_eval_set, config.batch_size, device), "test": evaluate(model, test_set, config.batch_size, device)}, wandb_run)
                save_checkpoint(run_dir, model, optimizer, step, next_epoch, next_offset)
                state_path.write_text(json.dumps({"best_train_loss": best_train_loss, "stale_steps": stale_steps, "drop_count": drop_count}))
                result_path = export_vision(run_dir, Path(config.results_dir))
                publish_result(result_path, wandb_run)
                progress.close()
                return run_dir
        completed_updates = step - epoch_start_step
        if completed_updates == len(batches) - resumed_offset:
            timing = {
                "epoch": epoch + 1,
                "seconds": time.perf_counter() - epoch_started,
                "updates": completed_updates,
                "full_epoch": resumed_offset == 0,
            }
            with (run_dir / "epoch_timings.jsonl").open("a") as handle:
                handle.write(json.dumps(timing) + "\n")
        epoch += 1
        batch_offset = 0
    save_checkpoint(run_dir, model, optimizer, step, checkpoint_epoch, checkpoint_offset)
    state_path.write_text(json.dumps({"best_train_loss": best_train_loss, "stale_steps": stale_steps, "drop_count": drop_count}))
    if step % config.eval_every != 0:
        record_evaluation(run_dir, {"step": step, "epoch": checkpoint_epoch, "train": evaluate(model, train_eval_set, config.batch_size, device), "test": evaluate(model, test_set, config.batch_size, device)}, wandb_run)
    save_predictions(model, test_set, config.batch_size, device, run_dir / "test_predictions.pt")
    result_path = export_vision(run_dir, Path(config.results_dir))
    publish_result(result_path, wandb_run)
    (run_dir / "completed.json").write_text(json.dumps({"step": step, "epoch": epoch}) + "\n")
    progress.close()
    return run_dir


def run(**kwargs: object) -> str:
    resume = kwargs.pop("resume", None)
    progress_fd = kwargs.pop("progress_fd", None)
    config = VisionConfig.model_validate(kwargs)
    if progress_fd is None:
        return str(train(config, str(resume) if resume is not None else None))

    descriptor = int(progress_fd)

    def report_progress(step: int) -> None:
        os.write(descriptor, f"{step}\n".encode())

    return str(train(config, str(resume) if resume is not None else None, progress_callback=report_progress))


if __name__ == "__main__":
    fire.Fire(run)
