from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

spec = importlib.util.spec_from_file_location(
    "backfill_epoch_tables", Path(__file__).parents[1] / "scripts/backfill_epoch_tables.py")
assert spec is not None and spec.loader is not None
backfill = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backfill)


def table(epoch: int) -> dict[str, Any]:
    return {"columns": ["epoch", "clip", "object", "prediction"],
            "data": [[epoch, "clip0", 1, {"path": f"media/{epoch}.pts.json"}]],
            "nrows": 1, "ncols": 4}


def test_merge_sorts_epochs_and_deduplicates_retries() -> None:
    merged = backfill.combine_tables([table(2), table(1), table(2)])
    assert [row[0] for row in merged["data"]] == [1, 2]
    assert merged["nrows"] == 2
    assert backfill.media_paths(merged) == {"media/1.pts.json", "media/2.pts.json"}


def test_conflicting_reconstruction_fails_before_writing() -> None:
    other = table(1)
    other["data"][0][-1] = {"path": "different.pts.json"}
    with pytest.raises(ValueError, match="Conflicting reconstruction"):
        backfill.combine_tables([table(1), other])
