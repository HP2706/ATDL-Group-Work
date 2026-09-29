"""Run directories, metrics, and resumable checkpoints."""

import json
import hashlib
import os
import random
import signal
from datetime import datetime
from pathlib import Path
from types import FrameType
from typing import Any

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import nn


class TrainingConfig(BaseModel):
    """Options shared by image and translation training runs."""

    model_config = ConfigDict(extra="forbid")
    data_dir: str
    output_dir: str
    results_dir: str = str(Path(__file__).resolve().parents[3] / "our-results-folder")
    sample_count: int | None = Field(default=None, gt=0)
    eval_every: int = Field(default=1000, gt=0)
    checkpoint_every: int = Field(default=1000, gt=0)
    seed: int = 0
    device: str = "auto"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def code_fingerprints(code_root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted((code_root / "src").rglob("*.py")):
        result[str(path.relative_to(code_root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name in ("vision_train.py", "translation_train.py"):
        path = code_root / name
        result[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    return device


def create_run_directory(root: str, name: str, seed: int) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    path = Path(root) / f"{name}-seed{seed}-{timestamp}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def save_config(path: Path, config: BaseModel) -> None:
    (path / "config.json").write_text(config.model_dump_json(indent=2) + "\n")


def append_metric(path: Path, metric: dict[str, Any]) -> None:
    with (path / "metrics.jsonl").open("a") as handle:
        handle.write(json.dumps(metric, sort_keys=True) + "\n")


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    epoch: int,
    batch_offset: int,
) -> None:
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "epoch": epoch,
        "batch_offset": batch_offset,
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        checkpoint["cuda_rng"] = torch.cuda.get_rng_state_all()
    temporary = path / "checkpoint.tmp"
    torch.save(checkpoint, temporary)
    os.replace(temporary, path / "checkpoint.pt")


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> tuple[int, int, int]:
    checkpoint = torch.load(path / "checkpoint.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    device = next(model.parameters()).device
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)
    random.setstate(checkpoint["python_rng"])
    np.random.set_state(checkpoint["numpy_rng"])
    torch.set_rng_state(checkpoint["torch_rng"])
    if torch.cuda.is_available() and "cuda_rng" in checkpoint:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
    return int(checkpoint["step"]), int(checkpoint["epoch"]), int(checkpoint["batch_offset"])


class StopRequested:
    """Finish the current update and checkpoint on termination."""

    def __init__(self) -> None:
        self.requested = False

    def handle(self, signum: int, frame: FrameType | None) -> None:
        self.requested = True

    def install(self) -> None:
        signal.signal(signal.SIGTERM, self.handle)
        signal.signal(signal.SIGINT, self.handle)
