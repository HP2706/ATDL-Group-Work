"""Rebuild a run's comparison Parquet file from its metrics.jsonl."""

import json
from pathlib import Path

import fire

from src.training.results import export_translation, export_vision


def export(run_dir: str, results_dir: str | None = None, name: str | None = None) -> str:
    """``name`` sets a vision run's file name and run ID when another run already uses its directory name."""
    path = Path(run_dir)
    config = json.loads((path / "config.json").read_text())
    target = Path(results_dir or config["results_dir"])
    if "architecture" in config:
        return str(export_vision(path, target, name))
    if name is not None:
        raise ValueError("name is only supported for vision runs")
    return str(export_translation(path, target))


if __name__ == "__main__":
    fire.Fire(export)
