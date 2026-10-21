from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
import chz
from .config import ExperimentConfig, capture_provenance, sweep_group
from .storage import atomic_json


VARIANTS: tuple[tuple[str, bool, bool], ...] = (("A", False, False), ("B", True, False), ("C", False, True), ("D", True, True))


def sweep(config: ExperimentConfig) -> None:
    root = Path(__file__).resolve().parents[4]
    provenance = capture_provenance(root)
    group = config.training.group or sweep_group(provenance.git_head)
    directory = Path(config.training.output) / group
    directory.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []
    profile_path = Path(config.training.output) / "profile.json"
    if not profile_path.is_file():
        raise RuntimeError("GPU profiling must pass before the four-configuration sweep")
    profile = json.loads(profile_path.read_text())
    if profile["group"] != group or profile["git_head"] != provenance.git_head:
        raise ValueError("Profiling belongs to a different sweep/source revision")
    deadline = time.monotonic() + 4800
    for variant, video_lora, sam_lora in VARIANTS:
        remaining = deadline - time.monotonic()
        if remaining < 120:
            results.append({"variant": variant, "status": "not_started_budget_exhausted"})
            continue
        per_variant = config.training.max_seconds
        if remaining < per_variant + 90:
            results.append({"variant": variant, "status": "not_started_budget_exhausted"})
            continue
        # Serialize the full resolved parent config, not a partial set of overrides.
        launch = chz.asdict(config)
        launch["model"].update(video_lora=video_lora, sam_lora=sam_lora)
        launch["training"].update(group=group, max_seconds=per_variant, microbatch=profile["selected_microbatch"], accumulation=profile["selected_accumulation"])
        launch["provenance"] = chz.asdict(provenance)
        resolved = directory / f"{variant}.json"
        atomic_json(resolved, launch)
        previous = Path(config.training.output) / f"{group}-{variant}" / "result.json"
        if previous.is_file():
            record = json.loads(previous.read_text())
            if record["completed_steps"] and record["config"] == launch:
                results.append({"variant": variant, "returncode": 0, "status": "already_completed"})
                continue
        command = [sys.executable, "-m", "video3d.cli", "train-json", str(resolved)]
        with (directory / f"{variant}.log").open("a") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            try:
                returncode = process.wait(timeout=per_variant + 90)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    returncode = process.wait(timeout=45)
                except subprocess.TimeoutExpired:
                    process.kill()
                    returncode = process.wait()
            completed = previous.is_file() and json.loads(previous.read_text())["completed_steps"]
            results.append({"variant": variant, "returncode": returncode if completed else 1, "status": "completed" if completed else "incomplete", "config": str(resolved), "log": str(directory / f"{variant}.log")})
        atomic_json(directory / "sweep.json", {"schema_version": 2, "group": group, "results": results})
    atomic_json(directory / "sweep.json", {"schema_version": 2, "group": group, "results": results})
    if any(row.get("returncode", 1) != 0 for row in results):
        raise RuntimeError("One or more configurations failed; inspect persistent sweep logs")
