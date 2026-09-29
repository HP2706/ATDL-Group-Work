"""Print structural summaries of the authors' metric pickles without printing arrays."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import fire
import numpy as np


ALLOWED_GLOBALS: dict[tuple[str, str], Any] = {
    ("numpy", "dtype"): np.dtype,
    ("numpy", "ndarray"): np.ndarray,
    ("numpy.core.multiarray", "_reconstruct"): np._core.multiarray._reconstruct,
    ("numpy.core.multiarray", "scalar"): np._core.multiarray.scalar,
    ("numpy._core.multiarray", "_reconstruct"): np._core.multiarray._reconstruct,
    ("numpy._core.multiarray", "scalar"): np._core.multiarray.scalar,
}


class MetricsUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        key = (module, name)
        if key not in ALLOWED_GLOBALS:
            raise ValueError(f"Unexpected pickle global: {module}.{name}")
        return ALLOWED_GLOBALS[key]


def load_metrics(path: Path) -> Any:
    with path.open("rb") as source:
        return MetricsUnpickler(source).load()


def structure(value: Any) -> str:
    if isinstance(value, np.ndarray):
        return f"ndarray{value.shape}:{value.dtype}"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{key}: {structure(item)}" for key, item in value.items()) + "}"
    if isinstance(value, (list, tuple)):
        first = structure(value[0]) if value else "empty"
        return f"{type(value).__name__}[{len(value)}] of {first}"
    return type(value).__name__


def inspect(root: str) -> None:
    directory = Path(root)
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.name in {"Mlist", "Ms", "ks", "ns", "nparams"}:
            value = load_metrics(path)
            print(f"{path.relative_to(directory)}: {structure(value)}")


if __name__ == "__main__":
    fire.Fire(inspect)
