"""validate-data on the fake export: clean data passes, broken shards are reported."""

import io
import json
import shutil
import tarfile

import pytest

from vidsam3d.__main__ import main
from vidsam3d.data.validate import validate_data


def rewrite_json(split_dir, clip_id, change):
    """Apply `change` to one clip's JSON inside its shard and fix index.json's byte size."""
    index = json.loads((split_dir / "index.json").read_text())
    shard = next(s for s in index["shards"] if clip_id in s["clip_ids"])
    path = split_dir / shard["name"]
    with tarfile.open(path) as tar:
        members = [(m, tar.extractfile(m).read()) for m in tar if m.isfile()]
    with tarfile.open(path, "w", format=tarfile.USTAR_FORMAT) as tar:
        for m, data in members:
            if m.name == f"{clip_id}.json":
                info = json.loads(data)
                change(info)
                data = json.dumps(info).encode()
                m.size = len(data)
            tar.addfile(m, io.BytesIO(data))
    shard["bytes"] = path.stat().st_size
    (split_dir / "index.json").write_text(json.dumps(index))


def test_clean_export_passes(fake_cfg):
    report = validate_data(fake_cfg, "train")
    assert report["clips"] == 4 and report["assets"] == 2 and report["clips_with_problems"] == 0
    assert report["coverage_min"] > 0.95
    assert report["val"] == {"clips": 0, "objects": 0, "train_clips": 4}   # default frac: no fake asset is val
    fake_cfg.data.val_asset_frac = 0.5
    assert validate_data(fake_cfg, "val")["val"] == {"clips": 2, "objects": 2, "train_clips": 2}


def test_broken_clips_are_reported(fake_root, fake_cfg, tmp_path):
    root = tmp_path / "data"
    shutil.copytree(fake_root / "eval", root / "eval")
    fake_cfg.paths.data_root = str(root)

    def bad_pose(info):
        info["objects"][0]["R_co"][5][0][0] += 0.1

    def bad_counts(info):
        info["visibility"]["visible_px"][0][0] += 1

    rewrite_json(root / "eval", "00000100", bad_pose)
    rewrite_json(root / "eval", "00000101", bad_counts)
    report = validate_data(fake_cfg, "eval")
    assert set(report["problems"]) == {"00000100", "00000101"}
    assert "exported pose differs" in report["problems"]["00000100"][0]
    assert "pixel counts" in " ".join(report["problems"]["00000101"])
    out = tmp_path / "report.json"
    with pytest.raises(SystemExit) as exit_info:
        main(["validate-data", "--env", "local", "--split", "eval", "--report", str(out),
              f"paths.data_root={root}", f"paths.cache_root={tmp_path}/cache"])
    assert exit_info.value.code == 1
    assert json.loads(out.read_text())["clips_with_problems"] == 2
