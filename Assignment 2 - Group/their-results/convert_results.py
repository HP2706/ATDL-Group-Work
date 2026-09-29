"""Convert the authors' public metric pickles and CSVs to a HF-ready Parquet dataset.

Run on a remote CPU host. The pickle loader permits only the NumPy constructors
used by the published files; no model code or GPU is needed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import fire
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from inspect_sources import load_metrics


VISION_SCHEMA = pa.schema(
    [
        ("source_experiment", pa.string()),
        ("trial_index", pa.int16()),
        ("model_width", pa.int32()),
        ("sample_size", pa.int32()),
        ("parameter_count", pa.int64()),
        ("measurement_index", pa.int32()),
        ("train_error", pa.float64()),
        ("test_error", pa.float64()),
        ("train_loss", pa.float64()),
        ("test_loss", pa.float64()),
        ("robust_test_error", pa.float64()),
    ]
)

TRANSLATION_SCHEMA = pa.schema(
    [
        ("source_file", pa.string()),
        ("task", pa.string()),
        ("sweep", pa.string()),
        ("source_row", pa.int32()),
        ("model_width", pa.int32()),
        ("sample_size", pa.int32()),
        ("train_error", pa.float64()),
        ("test_error", pa.float64()),
        ("train_loss", pa.float64()),
        ("test_loss", pa.float64()),
    ]
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metric_column(metrics: dict[str, np.ndarray[Any, Any]], key: str, count: int) -> pa.Array:
    if key not in metrics:
        return pa.nulls(count, type=pa.float64())
    return pa.array(np.asarray(metrics[key], dtype=np.float64).reshape(-1))


def vision_table(
    experiment: str,
    trial_index: int,
    widths: np.ndarray[Any, Any],
    metrics: dict[str, np.ndarray[Any, Any]],
    sample_size: int | None = None,
    parameter_counts: np.ndarray[Any, Any] | None = None,
) -> pa.Table:
    shape = metrics["Test Error"].shape
    if len(shape) != 2 or shape[0] != len(widths):
        raise ValueError(f"Unexpected metric shape for {experiment}: {shape}")
    if any(metric.shape != shape for metric in metrics.values()):
        raise ValueError(f"Inconsistent metric shapes for {experiment}, trial {trial_index}")
    count = shape[0] * shape[1]
    sizes = pa.nulls(count, type=pa.int32()) if sample_size is None else pa.array(np.full(count, sample_size, dtype=np.int32))
    parameters = (
        pa.nulls(count, type=pa.int64())
        if parameter_counts is None
        else pa.array(np.repeat(np.asarray(parameter_counts, dtype=np.int64), shape[1]))
    )
    arrays = [
        pa.array([experiment] * count, type=pa.string()),
        pa.array(np.full(count, trial_index, dtype=np.int16)),
        pa.array(np.repeat(np.asarray(widths, dtype=np.int32), shape[1])),
        sizes,
        parameters,
        pa.array(np.tile(np.arange(shape[1], dtype=np.int32), shape[0])),
        metric_column(metrics, "Train Error", count),
        metric_column(metrics, "Test Error", count),
        metric_column(metrics, "Train Loss", count),
        metric_column(metrics, "Test Loss", count),
        metric_column(metrics, "Robust Test Error", count),
    ]
    return pa.Table.from_arrays(arrays, schema=VISION_SCHEMA)


def convert_vision(raw_dir: Path, output_dir: Path) -> list[dict[str, Any]]:
    target_dir = output_dir / "data" / "vision"
    target_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    for source in sorted(raw_dir.iterdir()):
        if not source.is_dir():
            continue
        metric_file = source / "Mlist"
        if not metric_file.exists():
            continue
        width_file = source / "ks"
        widths = np.asarray(load_metrics(width_file), dtype=np.int32)
        trials = load_metrics(metric_file)
        if not isinstance(trials, list):
            raise ValueError(f"Expected trial list in {metric_file}")
        target = target_dir / f"{source.name}.parquet"
        rows = 0
        with pq.ParquetWriter(target, VISION_SCHEMA, compression="zstd", compression_level=7) as writer:
            for trial_index, metrics in enumerate(trials):
                table = vision_table(source.name, trial_index, widths, metrics)
                writer.write_table(table, row_group_size=65536)
                rows += table.num_rows
        summaries.append({"source": source.name, "rows": rows, "trials": len(trials), "sha256": sha256_file(target), "bytes": target.stat().st_size})
        print(f"vision {source.name}: {rows} rows, {target.stat().st_size / 1024**2:.2f} MiB", flush=True)

    grid = raw_dir / "dd_grid_p20"
    if grid.exists():
        metrics = load_metrics(grid / "Ms")
        widths = np.asarray(load_metrics(grid / "ks"), dtype=np.int32)
        sample_sizes = np.asarray(load_metrics(grid / "ns"), dtype=np.int32)
        parameter_counts = np.asarray(load_metrics(grid / "nparams"), dtype=np.int64)
        shape = metrics["Test Error"].shape
        if shape[:2] != (len(sample_sizes), len(widths)):
            raise ValueError(f"Unexpected sample grid shape: {shape}")
        target = target_dir / "dd_grid_p20.parquet"
        rows = 0
        with pq.ParquetWriter(target, VISION_SCHEMA, compression="zstd", compression_level=7) as writer:
            for sample_index, sample_size in enumerate(sample_sizes):
                two_dimensional = {key: values[sample_index] for key, values in metrics.items()}
                table = vision_table("dd_grid_p20", 0, widths, two_dimensional, int(sample_size), parameter_counts)
                writer.write_table(table, row_group_size=65536)
                rows += table.num_rows
        summaries.append({"source": "dd_grid_p20", "rows": rows, "trials": 1, "sha256": sha256_file(target), "bytes": target.stat().st_size})
        print(f"vision dd_grid_p20: {rows} rows, {target.stat().st_size / 1024**2:.2f} MiB", flush=True)
    return summaries


def translation_metadata(filename: str) -> tuple[str, str, int | None]:
    mapping: dict[str, tuple[str, str, int | None]] = {
        "nlp-model-dd.csv": ("IWSLT14_de_en", "model_width", 160000),
        "dd-translation-model-french.csv": ("WMT14_en_fr", "model_width", 200000),
        "df4k.csv": ("IWSLT14_de_en", "model_width", 4000),
        "df18k.csv": ("IWSLT14_de_en", "model_width", 18000),
        "dd-translation-samples-aug26-small.csv": ("IWSLT14_de_en", "sample_size", None),
        "dd-translation-samples-aug27-size80.csv": ("IWSLT14_de_en", "sample_size", None),
    }
    return mapping[filename]


def convert_translation(raw_dir: Path, output_dir: Path) -> dict[str, Any]:
    arrays: list[pa.Table] = []
    for source in sorted(raw_dir.rglob("*.csv")):
        frame = pd.read_csv(source)
        task, sweep, fixed_sample_size = translation_metadata(source.name)
        if sweep == "model_width":
            model_width = frame["n"].to_numpy(dtype=np.int32)
            sample_size = np.full(len(frame), fixed_sample_size, dtype=np.int32)
        else:
            model_width = frame["size"].to_numpy(dtype=np.int32)
            sample_size = frame["n"].to_numpy(dtype=np.int32)
        columns = [
            pa.array([source.name] * len(frame)),
            pa.array([task] * len(frame)),
            pa.array([sweep] * len(frame)),
            pa.array(np.arange(len(frame), dtype=np.int32)),
            pa.array(model_width),
            pa.array(sample_size),
            pa.array(frame["train_error"].to_numpy(dtype=np.float64)),
            pa.array(frame["test_error"].to_numpy(dtype=np.float64)),
            pa.array(frame["train_loss"].to_numpy(dtype=np.float64)),
            pa.array(frame["test_loss"].to_numpy(dtype=np.float64)),
        ]
        arrays.append(pa.Table.from_arrays(columns, schema=TRANSLATION_SCHEMA))
    combined = pa.concat_tables(arrays)
    target_dir = output_dir / "data" / "translation"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "translation.parquet"
    pq.write_table(combined, target, compression="zstd", compression_level=7)
    result = {"source": "translation", "rows": combined.num_rows, "sha256": sha256_file(target), "bytes": target.stat().st_size}
    print(f"translation: {combined.num_rows} rows, {target.stat().st_size / 1024**2:.2f} MiB", flush=True)
    return result


def convert(raw: str, output: str, manifest: str) -> None:
    raw_dir = Path(raw)
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    vision = convert_vision(raw_dir, output_dir)
    translation = convert_translation(raw_dir, output_dir)
    source_manifest = json.loads(Path(manifest).read_text())
    metadata = {
        "source_bucket": source_manifest["bucket"],
        "source_commit": source_manifest["source_commit"],
        "source_object_count": len(source_manifest["objects"]),
        "vision": vision,
        "translation": translation,
        "schema_version": 1,
    }
    (output_dir / "conversion-manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Converted {len(vision)} vision sources and 6 translation CSVs", flush=True)


if __name__ == "__main__":
    fire.Fire(convert)
