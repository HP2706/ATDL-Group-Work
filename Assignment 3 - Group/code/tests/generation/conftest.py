import typing
import json
import os
from pathlib import Path

import pytest

from video3d.data.generation.clipio import Clip
from video3d.data.generation.config import Config

DATASET = Path(os.environ.get("SAM3DVID_DATASET", "dataset"))
# Fraction of clips to run the expensive per-clip tests on (spec M2: 2% sample).
SAMPLE_FRAC = float(os.environ.get("SAM3DVID_SAMPLE_FRAC", "1.0"))


def manifest_entries() -> list[dict]:
    path = DATASET / "manifest.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def clip_params(sample: bool) -> list:
    entries = manifest_entries()
    if sample and SAMPLE_FRAC < 1.0:
        import random

        k = max(1, round(SAMPLE_FRAC * len(entries)))
        entries = random.Random(0).sample(entries, k)
    return [pytest.param(DATASET / e["path"], id=e["path"]) for e in entries]


@pytest.fixture(scope="session")
def cfg() -> Config:
    if not (DATASET / "config.json").is_file():
        pytest.skip("no generated dataset configuration")
    return Config.load(DATASET / "config.json")


def pytest_generate_tests(metafunc: typing.Any) -> typing.Any:
    if "clip" in metafunc.fixturenames:
        sample = metafunc.definition.get_closest_marker("sampled") is not None
        metafunc.parametrize("clip_path", clip_params(sample))


@pytest.fixture
def clip(clip_path: typing.Any) -> Clip:
    return Clip(clip_path)
