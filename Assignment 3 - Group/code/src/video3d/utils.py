"""Small helpers: seeding, logging, run directories, checkpoints, JSONL."""

from __future__ import annotations
import typing

import json
import logging
import random
import shutil
import os
import sys
import zlib
from pathlib import Path

import numpy as np
import torch
from video3d.config import ExperimentConfig
from video3d.storage import atomic_json, atomic_torch
from video3d.storage import rng_state, restore_rng as set_rng_state, append_log as append_jsonl

__all__ = ["rng_state", "set_rng_state", "append_jsonl", "seed_everything", "sample_seed",
           "setup_logging", "pick_device", "run_dirs", "read_jsonl", "copy_to_output",
           "save_checkpoint", "find_resume_checkpoint"]

log = logging.getLogger("video3d")


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sample_seed(*parts: typing.Any) -> int:
    """Stable 31-bit seed from ids, e.g. sample_seed(clip_id, k, r); same in every process."""
    return zlib.crc32("|".join(map(str, parts)).encode()) & 0x7FFFFFFF


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure the root logger to print `time level message` to stdout."""
    logging.basicConfig(stream=sys.stdout, level=level, force=True,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    return log


def pick_device(name: str = "auto") -> torch.device:
    """cuda if available (or requested), else cpu."""
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def run_dirs(cfg: ExperimentConfig) -> tuple[Path, Path]:
    """(scratch run dir, persistent output run dir), both created."""
    dirs = Path(cfg.paths.scratch_root) / cfg.run_name, Path(cfg.paths.output_root) / cfg.run_name
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    return dirs




def read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file ([] if it does not exist)."""
    if not Path(path).is_file():
        return []
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def copy_to_output(src: Path, output_dir: Path) -> Path:
    """Copy a file from scratch to the persistent output dir (no rename: GCS FUSE safe)."""
    dst = output_dir / src.name
    if src.resolve() != dst.resolve():
        temporary = dst.with_suffix(dst.suffix + ".tmp")
        shutil.copyfile(src, temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        temporary.replace(dst)
    return dst






def save_checkpoint(state: dict, cfg: ExperimentConfig, name: str, pointer: str | None = "latest.json",
                    keep: int | None = None) -> Path:
    """Save to scratch, copy to output atomically, then write the pointer JSON last. Preserve old checkpoints."""
    scratch_dir, output_dir = run_dirs(cfg)
    local = scratch_dir / name
    atomic_torch(local, state)
    dst = copy_to_output(local, output_dir)
    if pointer:
        info = {"checkpoint": name, "step": state["step"]}
        atomic_json(output_dir / pointer, info)
    return dst


def find_resume_checkpoint(cfg: ExperimentConfig) -> Path | None:
    """`train.resume`: auto -> output/<run>/latest.json if present; none -> None; else a path."""
    resume = str(cfg.train.resume)
    if resume == "none":
        return None
    if resume != "auto":
        return Path(resume)
    pointer = Path(cfg.paths.output_root) / cfg.run_name / "latest.json"
    if not pointer.is_file():
        return None
    return pointer.parent / json.loads(pointer.read_text())["checkpoint"]
