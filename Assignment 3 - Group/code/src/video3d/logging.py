"""Optional W&B output; JSONL remains the local source of training records."""
from __future__ import annotations
from typing import Any
import chz
from video3d.config import ExperimentConfig


class RunLogger:
    def __init__(self, cfg: ExperimentConfig) -> None:
        self.run: Any = None
        if cfg.logging.wandb:
            import wandb
            self.run = wandb.init(project=cfg.logging.project, entity=cfg.logging.entity,
                                  name=cfg.run_name, id=cfg.run_name, resume="allow",
                                  config=chz.asdict(cfg))

    def log(self, values: dict[str, Any], step: int) -> None:
        if self.run is not None:
            self.run.log(values, step=step)

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()
