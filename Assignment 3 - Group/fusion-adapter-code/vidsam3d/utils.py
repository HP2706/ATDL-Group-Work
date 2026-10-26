"""Small helpers: seeding and logging. Checkpointing and timers are added in P4."""

from __future__ import annotations

import logging
import random
import sys

import numpy as np


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and (if installed) torch."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure the root logger to print `time level message` to stdout."""
    logging.basicConfig(stream=sys.stdout, level=level, force=True,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    return logging.getLogger("vidsam3d")
