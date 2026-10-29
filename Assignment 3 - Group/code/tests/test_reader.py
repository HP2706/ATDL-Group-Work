"""Reader of the training export, tested on clips packed by the generator's own exporter."""
import typing

import json

import numpy as np
import pytest

from tests.fixtures.make_fake_dataset import generator_export
from video3d.data import reader
from video3d.data.targets import convert_pose_to_sam3d


@pytest.fixture(scope="module")
def train(fake_root: typing.Any) -> typing.Any:
    return reader.open_split(fake_root / "train")


def test_split_index(train: typing.Any) -> typing.Any:
    index = json.loads((train.dir / "index.json").read_text())
    assert list(train.clips) == ["00000000", "00000001", "00000002", "00000003"]
    assert len(index["shards"]) == 2 and train.voxels_file.name == "voxels_train.npz"
    assert train.clips["00000001"].asset_ids == ("box", "sphere")


def test_shard_scan_is_cached(fake_root: typing.Any, tmp_path: typing.Any) -> typing.Any:
    split = reader.open_split.__wrapped__(fake_root / "eval", tmp_path)
    cache = list((tmp_path / "index").glob("eval-*.json"))
    assert len(cache) == 1
    again = reader.open_split.__wrapped__(fake_root / "eval", tmp_path)
    assert again.clips == split.clips


def test_frames_masks_and_image_size(train: typing.Any) -> typing.Any:
    clip = train.load_clip("00000001")
    assert clip.num_frames == 32 and clip.num_objects == 2 and clip.image_size == (64, 64)
    rgb, inst = reader.load_rgb(clip, 0), reader.load_instance_mask(clip, 0)
    assert rgb.shape == (64, 64, 3) and rgb.dtype == np.uint8
    assert inst.dtype == np.uint8 and set(np.unique(inst)) <= {0, 1, 2}
    for t in (0, 16, 31):                                   # visibility stats match the masks
        inst = reader.load_instance_mask(clip, t)
        assert [(inst == k).sum() for k in (1, 2)] == reader.visibility(clip, "visible_px")[t].tolist()


def test_exported_pose_equals_our_composition(train: typing.Any) -> typing.Any:
    for clip_id in train.clips:
        clip = train.load_clip(clip_id)
        for k in range(1, clip.num_objects + 1):
            o = reader.object_entry(clip, k)
            for t in range(clip.num_frames):
                ours = reader.camera_frame_pose(clip, k, t)
                for a, b in zip(ours, (o["R_co"][t], o["t_co"][t], o["s_co"][t])):
                    np.testing.assert_allclose(a, b, atol=1e-9)


def test_voxels_cover_the_visible_masks(train: typing.Any) -> typing.Any:
    clip = train.load_clip("00000001")
    for k in (1, 2):
        vox = reader.load_voxels(train.voxels_file, reader.object_entry(clip, k)["asset_id"])
        assert vox.dtype == bool and vox.shape == (64, 64, 64)
        assert reader.voxel_coverage(clip, k, 16, vox) > 0.95
    box = reader.load_voxels(train.voxels_file, "box")
    assert reader.voxel_coverage(clip, 1, 16, box.transpose(2, 1, 0)) < 0.7   # wrong axis order


def test_arc_span_from_cameras(train: typing.Any) -> typing.Any:
    assert reader.arc_span_deg(train.load_clip("00000000")) == pytest.approx(30.0)
    assert reader.arc_span_deg(train.load_clip("00000001")) == pytest.approx(60.0)


def test_voxel_centres() -> typing.Any:
    vox = np.zeros((64, 64, 64), bool)
    vox[0, 0, 0] = vox[63, 63, 63] = True
    np.testing.assert_allclose(reader.voxel_centers(vox), [[-0.5 + 1 / 128] * 3, [0.5 - 1 / 128] * 3])


def test_pose_converter_matches_the_generator() -> typing.Any:
    generator_export()                                       # puts synthetic-data/src on sys.path
    from video3d.data.generation.sam3d import ours_to_sam3d
    from scipy.spatial.transform import Rotation
    rng = np.random.default_rng(0)
    for R in Rotation.random(20, random_state=0).as_matrix():
        t, s = rng.normal(size=3), np.full(3, rng.uniform(0.4, 1.0))
        for a, b in zip(convert_pose_to_sam3d(R, t, s), ours_to_sam3d(R, t, s)):
            np.testing.assert_allclose(a, b, atol=1e-9)
