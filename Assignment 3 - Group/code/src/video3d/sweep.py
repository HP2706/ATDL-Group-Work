"""Bounded orchestration of the shared trainer and evaluator."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import subprocess
import sys
import time
import chz
from video3d.config import ExperimentConfig, config_dict
from video3d.storage import atomic_json

VARIANTS = (("frozen", False, False), ("video-lora", True, False),
            ("sam-lora", False, True), ("both-lora", True, True))


def sweep(cfg: ExperimentConfig, max_seconds: int = 4800, evaluate: bool = True) -> Path:
    """Four LoRA configurations, sequentially, each using the same trainer."""
    if sys.platform == "darwin":
        raise RuntimeError("Run experiments on remote compute")
    if max_seconds < 1:
        raise ValueError("max_seconds must be positive")
    group = cfg.run_name or "sweep-" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    directory = Path(cfg.paths.output_root) / (group + "-sweep")
    directory.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + max_seconds
    results: list[dict[str, object]] = []
    for name, video_lora, sam_lora in VARIANTS:
        if deadline - time.monotonic() < cfg.train.max_seconds + 90:
            results.append({"variant": name, "status": "budget_exhausted"})
            continue
        run_cfg = chz.replace(cfg, run_name=f"{group}-{name}",
            encoder=chz.replace(cfg.encoder, vjepa=chz.replace(cfg.encoder.vjepa, video_lora=video_lora)),
            sam3d=chz.replace(cfg.sam3d, lora=sam_lora))
        resolved = directory / f"{name}.json"
        atomic_json(resolved, config_dict(run_cfg))
        # timeout owns the foreground process group; interruption triggers atomic checkpointing.
        command = ["timeout", "--signal=INT", "--kill-after=30", str(cfg.train.max_seconds + 60),
                   sys.executable, "-m", "video3d", "train-json", str(resolved)]
        with (directory / f"{name}.log").open("a") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f"Sweep variant {name} failed; inspect {directory}")
            if evaluate:
                remaining = int(deadline - time.monotonic())
                if remaining < 1:
                    raise RuntimeError("Sweep evaluation budget exhausted")
                result = subprocess.run(["timeout", "--signal=INT", "--kill-after=30", str(remaining),
                    sys.executable, "-m", "video3d", "eval-json", str(resolved),
                    str(Path(cfg.paths.output_root) / run_cfg.run_name)], stdout=log, stderr=subprocess.STDOUT)
                if result.returncode:
                    raise RuntimeError(f"Sweep evaluation failed; inspect {directory}")
        results.append({"variant": name, "status": "completed"})
        atomic_json(directory / "sweep.json", {"results": results})
    atomic_json(directory / "sweep.json", {"results": results})
    return directory
