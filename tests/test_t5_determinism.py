"""T5: regenerating a clip from its seed reproduces JSON, masks and poses exactly."""

import dataclasses
import json

import numpy as np
import pytest

from sam3dvid.clipio import Clip
from sam3dvid.generate import generate_clip

from .conftest import DATASET, manifest_entries

NONDETERMINISTIC_META = {"timing", "git_commit"}


@pytest.mark.slow
def test_t5_determinism(cfg, tmp_path):
    entries = manifest_entries()
    if not entries:
        pytest.skip("no clips in manifest")
    entry = entries[0]
    original = Clip(DATASET / entry["path"])

    root = tmp_path / "dataset"
    root.mkdir()
    (root / "assets").symlink_to((DATASET / "assets").resolve())
    regen_cfg = dataclasses.replace(cfg, dataset_root=str(root))
    generate_clip(regen_cfg, entry["split"], entry["seed"], root)
    regen = Clip(root / entry["path"])

    assert regen.cameras == original.cameras
    assert json.loads((regen.path / "objects.json").read_text()) == json.loads((original.path / "objects.json").read_text())
    strip = lambda m: {k: v for k, v in m.items() if k not in NONDETERMINISTIC_META}
    assert strip(regen.meta) == strip(original.meta)
    for t in range(original.num_frames):
        np.testing.assert_array_equal(regen.visible(t), original.visible(t))
        for k in range(1, original.num_objects + 1):
            np.testing.assert_array_equal(regen.amodal(k, t), original.amodal(k, t))
        np.testing.assert_array_equal(np.isnan(regen.depth(t)), np.isnan(original.depth(t)))
        np.testing.assert_allclose(regen.depth(t), original.depth(t), rtol=1e-5, equal_nan=True)
        diff = np.abs(regen.rgb(t).astype(int) - original.rgb(t).astype(int))
        assert diff.mean() < 1.0, f"t={t}: mean RGB diff {diff.mean():.3f}"
