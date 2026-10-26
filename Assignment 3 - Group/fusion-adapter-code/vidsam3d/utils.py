"""Small helpers: seeding, logging, run directories, checkpoints, JSONL."""

from __future__ import annotations

import json
import logging
import random
import shutil
import sys
import zlib
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig

log = logging.getLogger("vidsam3d")


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sample_seed(*parts) -> int:
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


def run_dirs(cfg: DictConfig) -> tuple[Path, Path]:
    """(scratch run dir, persistent output run dir), both created."""
    dirs = Path(cfg.paths.scratch_root) / cfg.run_name, Path(cfg.paths.output_root) / cfg.run_name
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def append_jsonl(path: Path, record: dict) -> None:
    """Append one JSON line."""
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file ([] if it does not exist)."""
    if not Path(path).is_file():
        return []
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def copy_to_output(src: Path, output_dir: Path) -> Path:
    """Copy a file from scratch to the persistent output dir (no rename: GCS FUSE safe)."""
    dst = output_dir / src.name
    if src.resolve() != dst.resolve():
        shutil.copyfile(src, dst)
    return dst


def rng_state() -> dict:
    """RNG states of Python, NumPy, torch (and CUDA)."""
    state = {"python": random.getstate(), "numpy": np.random.get_state(),
             "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict) -> None:
    """Restore the states saved by rng_state()."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(state: dict, cfg: DictConfig, name: str, pointer: str | None = "latest.json",
                    keep: int | None = None) -> Path:
    """Save to scratch, copy to output, then write the pointer JSON last; prune old ckpt_*.pt."""
    scratch_dir, output_dir = run_dirs(cfg)
    local = scratch_dir / name
    torch.save(state, local)
    dst = copy_to_output(local, output_dir)
    if pointer:
        info = {"checkpoint": name, "step": state["step"]}
        (output_dir / pointer).write_text(json.dumps(info) + "\n")
    if keep:
        ckpts = sorted(output_dir.glob("ckpt_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
        for old in ckpts[:-keep]:
            old.unlink()
            (scratch_dir / old.name).unlink(missing_ok=True)
    return dst


def find_resume_checkpoint(cfg: DictConfig) -> Path | None:
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
