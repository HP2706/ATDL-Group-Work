"""Rebuild a run's comparison Parquet file from its metrics.jsonl."""

import json
from pathlib import Path

import fire

from src.training.results import export_translation, export_vision


def export(run_dir: str, results_dir: str | None = None) -> str:
    path = Path(run_dir)
    config = json.loads((path / "config.json").read_text())
    target = Path(results_dir or config["results_dir"])
    result = export_vision(path, target) if "architecture" in config else export_translation(path, target)
    return str(result)


if __name__ == "__main__":
    fire.Fire(export)
