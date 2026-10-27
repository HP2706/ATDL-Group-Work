import json

import numpy as np
import pytest

from sam3dvid.clipio import Clip
from sam3dvid.conventions import object_pose_in_camera
from sam3dvid.export import export
from sam3dvid.export_reader import iter_samples, load_voxels

from .conftest import DATASET

N_CLIPS = 4


def _export_subset(tmp_path_factory, clip_ids):
    out = tmp_path_factory.mktemp("export")
    # Export only the chosen clips by pointing at a manifest-free dataset view.
    view = tmp_path_factory.mktemp("view")
    (view / "clips").mkdir()
    (view / "clips" / "eval").mkdir()
    for cid in clip_ids:
        (view / "clips" / "eval" / cid).symlink_to((DATASET / "clips" / "eval" / cid).resolve())
    (view / "assets").symlink_to((DATASET / "assets").resolve())
    index = export(view, "eval", out, clips_per_shard=2)
    return out, index, view


@pytest.fixture(scope="module")
def roundtrip(tmp_path_factory):
    eval_dir = DATASET / "clips" / "eval"
    if not eval_dir.exists():
        pytest.skip("no eval clips")
    ids = sorted(p.name for p in eval_dir.iterdir())[:N_CLIPS]
    out, index, view = _export_subset(tmp_path_factory, ids)
    samples = {}
    for shard in index["shards"]:
        for cid, rgb, visible, info in iter_samples(out / shard["name"]):
            samples[cid] = (rgb, visible, info)
    return out, index, ids, samples, view


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 10 * np.log10(255.0**2 / max(mse, 1e-12))


def test_index_and_shards(roundtrip):
    out, index, ids, samples, _ = roundtrip
    assert index["format_version"] == 1
    assert [c for s in index["shards"] for c in s["clip_ids"]] == ids
    assert sorted(samples) == ids
    for s in index["shards"]:
        assert (out / s["name"]).stat().st_size == s["bytes"]
    assert json.loads((out / "index.json").read_text())["num_clips"] == len(ids)


def test_masks_bit_exact(roundtrip):
    _, _, ids, samples, _ = roundtrip
    for cid in ids:
        clip = Clip(DATASET / "clips" / "eval" / cid)
        visible = samples[cid][1]
        assert visible.dtype == np.uint8 and visible.shape == (32, 512, 512)
        for t in range(32):
            assert np.array_equal(visible[t], clip.visible(t))


def test_rgb_psnr(roundtrip):
    _, _, ids, samples, _ = roundtrip
    for cid in ids:
        clip = Clip(DATASET / "clips" / "eval" / cid)
        rgb = samples[cid][0]
        assert rgb.dtype == np.uint8 and rgb.shape == (32, 512, 512, 3)
        for t in range(32):
            assert psnr(rgb[t], clip.rgb(t)) > 40


def test_poses_equal(roundtrip):
    _, _, ids, samples, _ = roundtrip
    for cid in ids:
        clip = Clip(DATASET / "clips" / "eval" / cid)
        info = samples[cid][2]
        assert np.array_equal(np.array(info["K"]), clip.K)
        assert np.array_equal(np.array(info["T_cw"]), clip.T_cw)
        assert info["num_objects"] == clip.num_objects == len(info["objects"])
        assert info["visibility"] == clip.meta["visibility"]
        for o in info["objects"]:
            k = o["k"]
            assert o["asset_id"] == clip.objects[k - 1]["asset_id"]
            assert np.array_equal(np.array(o["R_wo"]), clip.objects[k - 1]["R_wo"])
            for t in range(32):
                R, tr, s = object_pose_in_camera(clip, k, t)
                assert np.array_equal(np.array(o["R_co"][t]), R)
                assert np.array_equal(np.array(o["t_co"][t]), tr)
                assert np.array_equal(np.array(o["s_co"][t]), s)


def test_voxels_exact(roundtrip):
    out, index, ids, _, _ = roundtrip
    with np.load(out / index["voxels_file"]) as f:
        used = set()
        for cid in ids:
            clip = Clip(DATASET / "clips" / "eval" / cid)
            for k in range(1, clip.num_objects + 1):
                asset_id = clip.objects[k - 1]["asset_id"]
                used.add(asset_id)
                grid = load_voxels(f, asset_id)
                assert grid.dtype == bool
                assert np.array_equal(grid, np.load(clip.asset_path(k, "voxels")))
        assert set(f.files) == used


def test_resume_skips_finished_shards(roundtrip):
    out, index, _, _, view = roundtrip
    shard = out / index["shards"][0]["name"]
    before = (shard.stat().st_mtime_ns, shard.read_bytes())
    export(view, "eval", out, clips_per_shard=2)
    assert (shard.stat().st_mtime_ns, shard.read_bytes()) == before


def test_deterministic(roundtrip, tmp_path):
    out, index, _, _, view = roundtrip
    export(view, "eval", tmp_path, clips_per_shard=2)
    for s in index["shards"]:
        assert (tmp_path / s["name"]).read_bytes() == (out / s["name"]).read_bytes()
