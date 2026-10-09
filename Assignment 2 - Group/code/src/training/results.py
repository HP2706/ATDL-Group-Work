"""Validate result records with Pydantic and export comparison Parquet files."""

import json
import os
from pathlib import Path
from typing import Any, ClassVar

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


class ResultRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    parquet_dtypes: ClassVar[dict[str, str]] = {}


class VisionResult(ResultRecord):
    """One evaluation of one image run, with the authors' columns first."""

    source_experiment: str
    trial_index: int | None = None
    model_width: int = Field(gt=0)
    sample_size: int | None = None
    parameter_count: int | None = None
    measurement_index: int = Field(ge=0)
    train_error: float = Field(ge=0, le=1)
    test_error: float = Field(ge=0, le=1)
    train_loss: float = Field(ge=0)
    test_loss: float = Field(ge=0)
    robust_test_error: float | None = Field(default=None, ge=0, le=1)
    run_id: str
    seed: int
    dataset: str
    architecture: str
    label_noise: float = Field(ge=0, le=1)
    augmentation: bool
    optimizer: str
    schedule: str
    weight_decay: float = Field(ge=0)
    global_step: int = Field(ge=0)
    epoch: int = Field(ge=0)

    parquet_dtypes: ClassVar[dict[str, str]] = {
        "source_experiment": "object", "trial_index": "Int16", "model_width": "Int32",
        "sample_size": "Int32", "parameter_count": "Int64", "measurement_index": "Int32",
        "train_error": "float64", "test_error": "float64", "train_loss": "float64",
        "test_loss": "float64", "robust_test_error": "float64", "run_id": "object",
        "seed": "Int64", "dataset": "object", "architecture": "object",
        "label_noise": "float64", "augmentation": "boolean", "optimizer": "object",
        "schedule": "object", "weight_decay": "float64", "global_step": "Int64",
        "epoch": "Int64",
    }


class TranslationResult(ResultRecord):
    """One IWSLT evaluation, with the authors' endpoint columns first."""

    source_file: str
    task: str
    sweep: str
    source_row: int | None = None
    model_width: int = Field(gt=0)
    sample_size: int | None = None
    train_error: float = Field(ge=0, le=100)
    test_error: float = Field(ge=0, le=100)
    train_loss: float = Field(ge=0)
    test_loss: float = Field(ge=0)
    run_id: str
    seed: int
    global_step: int = Field(ge=0)
    epoch: int = Field(ge=0)
    train_perplexity: float = Field(ge=1)
    test_perplexity: float = Field(ge=1)
    validation_loss: float = Field(ge=0)
    validation_error: float = Field(ge=0, le=100)

    parquet_dtypes: ClassVar[dict[str, str]] = {
        "source_file": "object", "task": "object", "sweep": "object",
        "source_row": "Int32", "model_width": "Int32", "sample_size": "Int32",
        "train_error": "float64", "test_error": "float64", "train_loss": "float64",
        "test_loss": "float64", "run_id": "object", "seed": "Int64",
        "global_step": "Int64", "epoch": "Int64", "train_perplexity": "float64",
        "test_perplexity": "float64", "validation_loss": "float64",
        "validation_error": "float64",
    }


def read_metrics(run_dir: Path) -> list[dict[str, Any]]:
    with (run_dir / "metrics.jsonl").open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_parquet(rows: list[ResultRecord], path: Path) -> None:
    if not rows:
        raise ValueError("no result records to export")
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [row.model_dump() for row in rows]
    frame = pd.DataFrame.from_records(records, columns=list(type(rows[0]).model_fields))
    frame = frame.astype(rows[0].parquet_dtypes)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_parquet(temporary, engine="pyarrow", compression="zstd", index=False)
    os.replace(temporary, path)


# Columns that identify one image run; a same-named file that differs here came from another run.
VISION_RUN_COLUMNS = ["model_width", "sample_size", "seed", "dataset", "architecture",
                      "label_noise", "augmentation", "optimizer", "schedule", "weight_decay"]


def export_vision(run_dir: Path, results_root: Path, name: str | None = None) -> Path:
    """Write the run's history; ``name`` replaces the run ID when two runs share a directory name."""
    run_id = name or run_dir.name
    config = json.loads((run_dir / "config.json").read_text())
    run_metadata = json.loads((run_dir / "run_metadata.json").read_text())
    history = {int(metric["step"]): metric for metric in read_metrics(run_dir) if "train" in metric and "test" in metric}
    rows: list[VisionResult] = []
    for measurement_index, step in enumerate(sorted(history)):
        metric = history[step]
        rows.append(VisionResult(
            source_experiment=run_id,
            trial_index=None,
            model_width=config["width"],
            sample_size=run_metadata["train_size"],
            parameter_count=run_metadata["parameter_count"],
            measurement_index=measurement_index,
            train_error=metric["train"]["error"],
            test_error=metric["test"]["error"],
            train_loss=metric["train"]["loss"],
            test_loss=metric["test"]["loss"],
            robust_test_error=None,
            run_id=run_id,
            seed=config["seed"],
            dataset=config["dataset"],
            architecture=config["architecture"],
            label_noise=config["label_noise"],
            augmentation=config["augmentation"],
            optimizer=config["optimizer"],
            schedule=config["schedule"],
            weight_decay=config["weight_decay"],
            global_step=metric["step"],
            epoch=metric["epoch"],
        ))
    path = results_root / "data" / "vision" / f"{run_id}.parquet"
    if path.exists():
        previous = pd.read_parquet(path, columns=VISION_RUN_COLUMNS).iloc[0].to_dict()
        changed = [column for column in VISION_RUN_COLUMNS if previous[column] != getattr(rows[0], column)]
        if changed:
            raise FileExistsError(f"{path} belongs to another run ({', '.join(changed)} differ); "
                                  "rerun export_results.py with --name=<unique run ID>")
    write_parquet(rows, path)
    return path


def translation_row(
    metric: dict[str, Any], config: dict[str, Any], run_metadata: dict[str, Any], run_id: str
) -> TranslationResult:
    return TranslationResult(
        source_file=run_id,
        task="IWSLT14_de_en",
        sweep=config["sweep"],
        source_row=None,
        model_width=config["embedding_dim"],
        sample_size=run_metadata["train_size"],
        train_error=metric["train"]["token_error_percent"],
        test_error=metric["test"]["token_error_percent"],
        train_loss=metric["train"]["token_nll"],
        test_loss=metric["test"]["token_nll"],
        run_id=run_id,
        seed=config["seed"],
        global_step=metric["step"],
        epoch=metric["epoch"],
        train_perplexity=metric["train"]["perplexity"],
        test_perplexity=metric["test"]["perplexity"],
        validation_loss=metric["valid"]["token_nll"],
        validation_error=metric["valid"]["token_error_percent"],
    )


def export_translation(run_dir: Path, results_root: Path) -> Path:
    config = json.loads((run_dir / "config.json").read_text())
    run_metadata = json.loads((run_dir / "run_metadata.json").read_text())
    history = {int(metric["step"]): metric for metric in read_metrics(run_dir) if all(key in metric for key in ("train", "valid", "test"))}
    rows = [translation_row(history[step], config, run_metadata, run_dir.name) for step in sorted(history)]
    history_path = results_root / "data" / "translation_history" / f"{run_dir.name}.parquet"
    endpoint_path = results_root / "data" / "translation" / f"{run_dir.name}.parquet"
    write_parquet(rows, history_path)
    write_parquet([rows[-1]], endpoint_path)
    return endpoint_path
