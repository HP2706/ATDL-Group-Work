"""Dataset housekeeping: shard seeds, gen-data with a stand-in generator, merging, validation."""

import json
import shlex
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from vidsam3d.config import load_config
from vidsam3d.data import reader
from vidsam3d.datagen import gen_data, merge_manifests, shard_seeds, validate_data

FIXTURES = Path(__file__).parent / "fixtures"  # make_fake_dataset.py doubles as the generator


def cfg_for(root, **overrides):
    base = {"paths.data_root": root, "generator.repo": FIXTURES,
            "generator.command": f'"{shlex.quote(sys.executable)} {{generator_repo}}/make_fake_dataset.py '
                                 f'{{out_dir}} --split {{split}} --seed-start {{seed_start}} --num-clips {{num_clips}}"'}
    base.update(overrides)
    return load_config(env="local", overrides=[f"{k}={v}" for k, v in base.items()])


def test_shards_cover_every_seed_once():
    cfg = load_config(env="local", overrides=["generator.num_clips.train=10"])
    seeds = []
    for i in range(3):
        start, n = shard_seeds(cfg, "train", i, 3)
        seeds += range(start, start + n)
    assert seeds == list(range(10))
    assert shard_seeds(cfg, "eval", 0, 1)[0] == 2_000_000         # splits never share seeds


def test_gen_data_merge_and_validate(tmp_path):
    cfg = cfg_for(tmp_path, **{"generator.num_clips.train": 3})
    for i in range(2):
        gen_data(cfg, "train", i, 2)
    assert len((tmp_path / "manifest.train.shard000.jsonl").read_text().splitlines()) == 2
    merge_manifests(cfg, "train")
    entries = reader.read_manifest(tmp_path, "train")
    assert sorted(e["seed"] for e in entries) == [0, 1, 2]
    report = validate_data(cfg, "train")
    assert report["clips"] == 3 and report["clips_with_problems"] == 0
    assert report["projection_min"] > 0.9


def test_merge_refuses_duplicate_clip_ids(tmp_path):
    cfg = cfg_for(tmp_path, **{"generator.num_clips.train": 2})
    gen_data(cfg, "train", 0, 1)
    line = (tmp_path / "manifest.train.shard000.jsonl").read_text().splitlines()[0]
    (tmp_path / "manifest.train.shard001.jsonl").write_text(line + "\n")
    with pytest.raises(AssertionError, match="duplicate"):
        merge_manifests(cfg, "train")


def test_gen_data_dry_run_runs_nothing(tmp_path):
    gen_data(cfg_for(tmp_path), "train", 0, 4, dry_run=True)
    assert not (tmp_path / "clips").exists()


def test_validate_detects_broken_files(fake_root, tmp_path):
    root = tmp_path / "ds"
    shutil.copytree(fake_root, root)
    clip_dir = root / "clips" / "eval" / "00000101"
    (clip_dir / "depth" / "031.npz").unlink()                                  # missing file
    meta = json.loads((root / "clips/eval/00000100/meta.json").read_text())
    meta["num_objects"] = 3                                                    # wrong N
    (root / "clips/eval/00000100/meta.json").write_text(json.dumps(meta))
    report = validate_data(cfg_for(root), "eval")
    assert set(report["problems"]) == {"00000100", "00000101"}
    assert "missing" in report["problems"]["00000101"][0]


def test_validate_detects_a_bad_amodal_mask(fake_root, tmp_path):
    root = tmp_path / "ds"
    shutil.copytree(fake_root, root)
    path = root / "clips/train/00000000/mask_amodal/obj1/016.png"
    Image.fromarray(np.full((64, 64), 7, np.uint8)).save(path)
    report = validate_data(cfg_for(root), "train")
    assert "00000000" in report["problems"]
