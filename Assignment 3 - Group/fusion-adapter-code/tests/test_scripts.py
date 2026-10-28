"""Shell scripts: shellcheck, and a fake-SAM 3D dry run through scripts/run.sh with data staging."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests.fixtures.fake_config import FAKE_OVERRIDES

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = sorted(str(p) for p in (REPO / "scripts").rglob("*.sh"))


@pytest.mark.skipif(shutil.which("shellcheck") is None, reason="shellcheck not installed")
def test_shellcheck_clean():
    subprocess.run(["shellcheck", "-x", *SCRIPTS], check=True, cwd=REPO)


def test_dry_run_through_run_sh(fake_root, tmp_path):
    env = {**os.environ, "ACTIVATE": "0", "VIDSAM3D_ENV": "local", "STAGE_TO_SCRATCH": "1",
           "PATH": f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}",
           "DATA_ROOT": str(fake_root), "SCRATCH_ROOT": str(tmp_path / "scratch"),
           "OUTPUT_ROOT": str(tmp_path / "out"), "CACHE_ROOT": str(tmp_path / "cache")}
    overrides = [f"{k}={v}" for k, v in {**FAKE_OVERRIDES, "train.max_steps": 2}.items()]
    run = lambda *args: subprocess.run(["bash", str(REPO / "scripts/run.sh"), *args, *overrides],
                                       env=env, cwd=tmp_path, check=True, capture_output=True, text=True)
    run("cache-latents", "--split", "train")
    out = run("train", "--condition", "c3")
    staged = tmp_path / "scratch" / "dataset"
    assert (staged / "manifest.jsonl").is_file() and (staged / "clips" / "train").is_dir()
    assert not (staged / "clips" / "eval").exists()                # only the splits train reads
    assert "staging" in out.stdout
    assert (tmp_path / "out" / "c3_seed0" / "latest.json").is_file()
