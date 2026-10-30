"""Canonical-frame check on preprocessed assets: mesh.glb surface samples must land in voxels.npy."""
import typing

import numpy as np
import pytest
import trimesh

from video3d.data.generation.conventions import CANONICAL_FROM_GLTF

from .conftest import DATASET

ASSET_DIRS = sorted(p.parent for p in DATASET.glob("assets/*/*/meta.json"))
# A spread over both splits rather than every asset.
SAMPLE = ASSET_DIRS[:: max(1, len(ASSET_DIRS) // 40)]


@pytest.mark.parametrize("asset_dir", SAMPLE, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_glb_surface_in_occupied_voxels(asset_dir: typing.Any) -> typing.Any:
    mesh = trimesh.load(asset_dir / "mesh.glb", force="mesh")
    points, _ = trimesh.sample.sample_surface(mesh, 20000, seed=0)
    canonical = points @ CANONICAL_FROM_GLTF.T
    assert np.abs(canonical).max() <= 0.5 + 1e-4
    voxels = np.load(asset_dir / "voxels.npy")
    idx = np.clip(np.floor((canonical + 0.5) * 64).astype(int), 0, 63)
    assert voxels[tuple(idx.T)].mean() >= 0.999


def test_geometric_scores_separate_solid_sheet_and_fragment() -> typing.Any:
    from video3d.data.generation.assets import geometric_scores

    box = trimesh.creation.box((1, 1, 1))
    assert geometric_scores(box)["normal_eig_min"] > 0.3
    sheet = trimesh.creation.box((1, 1, 0.001))
    scores = geometric_scores(sheet)
    assert scores["extent_min"] < 0.01
    # A second part far from the largest one is an outlier fragment: areas 6 and 1.5 give 0.2.
    far = trimesh.creation.box((0.5, 0.5, 0.5))
    far.apply_translation((2, 0, 0))
    scores = geometric_scores(trimesh.util.concatenate([box, far]))
    assert scores["num_components"] == 2
    assert scores["outlier_area_frac"] == pytest.approx(0.2)


def test_rejection_reasons_and_pool(tmp_path: typing.Any) -> typing.Any:
    import json

    from video3d.data.generation.assetfilter import accepted_assets, rejection_reasons
    from video3d.data.generation.config import Config

    cfg = Config(dataset_root=str(tmp_path))
    good = {"extent_min": 0.5, "normal_eig_min": 0.2, "outlier_area_frac": 0.1, "num_voxels": 5000}
    assert rejection_reasons(good, cfg) == []
    assert rejection_reasons({**good, "extent_min": 0.01, "num_voxels": 10}, cfg) == ["thin", "small"]
    assert rejection_reasons({**good, "captions": ["A modern House with a garden."]}, cfg) == ["scene"]
    assert rejection_reasons({**good, "captions": ["A mushroom", "a warehouse-style crate"]}, cfg) == []
    for split in ("train", "eval"):
        for name, scores in {"a": good, "b": {**good, "normal_eig_min": 0.0}}.items():
            d = tmp_path / "assets" / split / name
            d.mkdir(parents=True)
            fs = {k: v for k, v in scores.items() if k != "num_voxels"}
            (d / "meta.json").write_text(json.dumps({"asset_id": name, "filter_scores": fs, "num_voxels": scores["num_voxels"]}))
    assert accepted_assets(cfg, "train") == ["a"]
    assert accepted_assets(cfg, "eval") == ["a", "b"]
