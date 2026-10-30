import typing
from video3d.config import replace_config
"""Sample builder: subsets, windows, crops, determinism, leakage guard."""

import numpy as np
import pytest
import torch

from video3d.data import reader, samples
from video3d.data.samples import ClipDataset

STRAT_FIELDS = ("visible_frac", "amodal_px", "arc_span_deg", "num_objects", "asset_id")


def test_frame_subset() -> typing.Any:
    assert samples.frame_subset(0) == list(range(0, 32, 2))
    assert samples.frame_subset(1) == list(range(1, 32, 2))


@pytest.mark.parametrize("clip_len", [1, 4, 8, 16])
def test_window_contains_reference_and_is_centred(clip_len: typing.Any) -> typing.Any:
    for ref_pos in range(16):
        pos = samples.window_positions(16, ref_pos, clip_len)
        assert len(pos) == clip_len and ref_pos in pos and 0 <= pos.start and pos.stop <= 16
    assert samples.window_positions(16, 8, 4) == range(6, 10)


@pytest.mark.parametrize("clip_len", [1, 4, 8, 16])
def test_tubelet_offsets(clip_len: typing.Any) -> typing.Any:
    for ref in range(clip_len):
        dt = samples.tubelet_offsets(clip_len, ref)
        assert len(dt) == max(clip_len // 2, 1)
        assert dt.min() >= -7 and dt.max() <= 7 and dt[ref // 2] == 0


def test_crop_window_is_square_and_contains_bbox() -> typing.Any:
    bbox = (100, 200, 140, 260)
    x0, y0, side = samples.crop_window(bbox, (512, 512), scale=1.25, min_side=128)
    assert side == 128 and x0 <= 100 and y0 <= 200 and x0 + side >= 140 and y0 + side >= 260
    _, _, side = samples.crop_window((0, 0, 400, 500), (512, 512), 1.25, 128)
    assert side == 512                                    # clamped to the image size


def test_crop_pad_zero_pads_outside_the_image() -> typing.Any:
    img = np.ones((8, 8, 3), np.uint8)
    out = samples.crop_pad(img, (-2, 4, 6))
    assert out.shape == (6, 6, 3)
    assert out[:, :2].sum() == 0 and out[4:].sum() == 0   # left of / below the image
    assert out[:4, 2:].min() == 1


def test_crop_mask_area_is_preserved(fake_cfg: typing.Any) -> typing.Any:
    ds = ClipDataset(fake_cfg, "eval", train=False)
    sample = ds[0]
    clip = ds.clip(sample["info"]["clip_id"])
    x0, y0, side = sample["info"]["crop_window"]
    size = fake_cfg.data.crop.size
    for i, t in enumerate(sample["info"]["frame_ids"]):
        vis = samples.crop_pad(reader.load_visible_mask(clip, sample["info"]["k"], t), (x0, y0, side))
        area = sample["inputs"]["frame_masks"][i].sum().item() * (side / size) ** 2
        assert area == pytest.approx(vis.sum(), rel=0.1)


def test_eval_samples(fake_cfg: typing.Any) -> typing.Any:
    ds = ClipDataset(fake_cfg, "eval", train=False)
    rules = {(x["clip_id"], x["k"], x["r"]): x["rules"] for x in ds.index}
    assert rules[("00000101", 1, 16)] == ["middle", "most_occluded"]   # coinciding rules -> one sample
    assert all(any(c == cid and kk == k and r == 16 for c, kk, r in rules)
               for cid, k in {(c, kk) for c, kk, _ in rules})          # every object has a middle sample
    size = fake_cfg.data.crop.size
    for i in range(len(ds)):
        s = ds[i]
        x, info = s["inputs"], s["info"]
        assert info["r"] in info["frame_ids"] and info["clip_len"] == 16 and info["offset"] == 0
        assert x["frames"].shape == (16, 3, size, size) and x["frame_masks"].shape == (16, size, size)
        assert x["ref_rgb"].shape == (64, 64, 3) and x["ref_mask"].dtype == torch.bool
        assert info["frame_ids"][x["ref_pos"]] == info["r"] and x["ref_mask"].any()
        assert all(f in info for f in STRAT_FIELDS)


def test_eval_samples_are_deterministic(fake_cfg: typing.Any) -> typing.Any:
    a, b = ClipDataset(fake_cfg, "eval", train=False), ClipDataset(fake_cfg, "eval", train=False)
    assert a.index == b.index
    for i in range(len(a)):
        sa, sb = a[i], b[i]
        assert sa["info"] == sb["info"]
        for key, val in sa["inputs"].items():
            assert torch.equal(torch.as_tensor(val), torch.as_tensor(sb["inputs"][key]))


def test_eval_clip_length_option(fake_cfg: typing.Any) -> typing.Any:
    s = ClipDataset(fake_cfg, "eval", train=False, clip_len=4)[0]
    assert s["inputs"]["frames"].shape[0] == 4 and len(s["inputs"]["dt"]) == 2


def test_train_samples(fake_cfg: typing.Any) -> typing.Any:
    ds = ClipDataset(fake_cfg, "train", train=True)
    assert len(ds) == 6                                   # (clip, k) pairs: 1 + 2 + 1 + 2 objects
    seen_lengths = set()
    for epoch in range(10):
        ds.set_epoch(epoch)
        for i in range(len(ds)):
            info, x = ds[i]["info"], ds[i]["inputs"]
            assert info["r"] in info["frame_ids"] and info["r"] % 2 == info["offset"]
            assert x["frames"].shape[0] == info["clip_len"] == len(info["frame_ids"])
            assert x["frames"].shape[-1] == x["frames"].shape[-2]         # square crop
            seen_lengths.add(info["clip_len"])
    assert seen_lengths <= {1, 4, 8, 16} and len(seen_lengths) > 1


def test_train_samples_are_seeded(fake_cfg: typing.Any) -> typing.Any:
    a, b = ClipDataset(fake_cfg, "train", train=True), ClipDataset(fake_cfg, "train", train=True)
    assert a[1]["info"] == b[1]["info"]
    infos = set()
    for epoch in range(5):
        a.set_epoch(epoch)
        infos.add(str(a[1]["info"]))
    assert len(infos) > 1                                 # epochs give different draws


def test_occlusion_bias_prefers_occluded_frames() -> typing.Any:
    vis = np.ones(32)
    vis[4] = 0.2
    cands = samples.frame_subset(0)
    draw = lambda p: [samples.sample_reference(cands, vis, np.random.default_rng(i), p)
                      for i in range(2000)]
    assert np.mean(np.array(draw(1.0)) == 4) == 1.0      # only frame 4 has weight > 0
    assert np.mean(np.array(draw(0.0)) == 4) < 0.15      # uniform: ~1/16


def test_leakage_guard(fake_cfg: typing.Any) -> typing.Any:
    sample = ClipDataset(fake_cfg, "train", train=True)[0]
    samples.check_no_leakage(sample["inputs"])
    for bad in ("amodal_mask", "depth", "T_cw", "pose", "latent"):
        with pytest.raises(AssertionError):
            samples.check_no_leakage({**sample["inputs"], bad: 0})
    assert not set(sample["inputs"]) & {"amodal", "depth", "R_co", "t_co", "K", "cameras"}


def test_val_is_asset_disjoint_from_train(fake_cfg: typing.Any) -> typing.Any:
    fake_cfg = replace_config(fake_cfg, **{'data.val_asset_frac': 0.5})
    train = ClipDataset(fake_cfg, "train", train=True)
    val = ClipDataset(fake_cfg, samples.VAL, train=False)
    assert {x["clip_id"] for x in train.index} == {"00000000", "00000002"}     # box-only clips
    assert {x["clip_id"] for x in val.index} == {"00000001", "00000003"}       # clips with the sphere
    assert {train[i]["info"]["asset_id"] for i in range(len(train))} == {"box"}
    assert {val[i]["info"]["asset_id"] for i in range(len(val))} == {"sphere"}  # only val assets as targets
    fake_cfg = replace_config(fake_cfg, **{'data.val_asset_frac': 0.0})
    assert len(ClipDataset(fake_cfg, "train", train=True).index) == 6             # 1 + 2 + 1 + 2 objects
    assert samples.is_val_asset("sphere", 0.5) and not samples.is_val_asset("box", 0.5)
