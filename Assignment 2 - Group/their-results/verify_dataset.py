"""Verify every converted metric value against the downloaded source snapshot."""

from __future__ import annotations

from pathlib import Path

import fire
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from inspect_sources import load_metrics


METRIC_COLUMNS = {
    "Train Error": "train_error",
    "Test Error": "test_error",
    "Train Loss": "train_loss",
    "Test Loss": "test_loss",
    "Robust Test Error": "robust_test_error",
}


def verify_vision(raw_dir: Path, output_dir: Path) -> tuple[int, int]:
    source_count = 0
    row_count = 0
    for parquet in sorted((output_dir / "data" / "vision").glob("*.parquet")):
        source = raw_dir / parquet.stem
        table = pq.read_table(parquet)
        if parquet.stem == "dd_grid_p20":
            original = load_metrics(source / "Ms")
            sample_sizes = load_metrics(source / "ns")
            widths = load_metrics(source / "ks")
            expected_shape = original["Test Error"].shape
            if table.num_rows != int(np.prod(expected_shape)):
                raise ValueError(f"Row count mismatch: {parquet.name}")
            if not np.array_equal(table["sample_size"].to_numpy().reshape(expected_shape)[:, :, 0], np.broadcast_to(np.asarray(sample_sizes)[:, None], expected_shape[:2])):
                raise ValueError("Sample count coordinates changed")
            if not np.array_equal(table["model_width"].to_numpy().reshape(expected_shape)[0, :, 0], widths):
                raise ValueError("Width coordinates changed")
            for key, column in METRIC_COLUMNS.items():
                if key in original:
                    actual = table[column].to_numpy().reshape(expected_shape)
                    if not np.array_equal(actual, original[key], equal_nan=True):
                        raise ValueError(f"Metric mismatch: {parquet.name}, {key}")
        else:
            trials = load_metrics(source / "Mlist")
            widths = load_metrics(source / "ks")
            shape = trials[0]["Test Error"].shape
            expected_shape = (len(trials), *shape)
            if table.num_rows != int(np.prod(expected_shape)):
                raise ValueError(f"Row count mismatch: {parquet.name}")
            actual_widths = table["model_width"].to_numpy().reshape(expected_shape)[0, :, 0]
            if not np.array_equal(actual_widths, widths):
                raise ValueError(f"Width coordinates changed: {parquet.name}")
            for key, column in METRIC_COLUMNS.items():
                if key in trials[0]:
                    actual = table[column].to_numpy().reshape(expected_shape)
                    expected = np.stack([trial[key] for trial in trials])
                    if not np.array_equal(actual, expected, equal_nan=True):
                        raise ValueError(f"Metric mismatch: {parquet.name}, {key}")
        source_count += 1
        row_count += table.num_rows
        print(f"Verified {parquet.stem}", flush=True)
    return source_count, row_count


def verify_translation(raw_dir: Path, output_dir: Path) -> int:
    table = pq.read_table(output_dir / "data" / "translation" / "translation.parquet")
    frame = table.to_pandas()
    for source in sorted(raw_dir.rglob("*.csv")):
        expected = pd.read_csv(source)
        actual = frame.loc[frame["source_file"] == source.name].sort_values("source_row")
        if len(actual) != len(expected):
            raise ValueError(f"Translation row count mismatch: {source.name}")
        for original, converted in METRIC_COLUMNS.items():
            if original == "Robust Test Error":
                continue
            if not np.array_equal(actual[converted].to_numpy(), expected[converted].to_numpy(), equal_nan=True):
                raise ValueError(f"Translation metric mismatch: {source.name}, {converted}")
    return table.num_rows


def verify(raw: str, output: str) -> None:
    source_count, vision_rows = verify_vision(Path(raw), Path(output))
    translation_rows = verify_translation(Path(raw), Path(output))
    print(f"All values verified: {source_count} vision sources, {vision_rows} vision rows, {translation_rows} translation rows")


if __name__ == "__main__":
    fire.Fire(verify)
