from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch
from typing import Any

from video3d.baselines.api import (BaselineConfig, BaselineInput, chunk_indices,
    load_views, occupancy_from_coords, reconstruct)


def make_sample(root: Path) -> BaselineInput:
    frames, masks = [], []
    for i in range(3):
        frame, mask = root / f"frame-{i}.png", root / f"mask-{i}.png"
        Image.new("RGB", (12, 12), (80 + i, 0, 0)).save(frame)
        Image.new("L", (12, 12), 255).save(mask)
        frames.append(frame)
        masks.append(mask)
    return BaselineInput(sample_id="test", frames=frames, masks=masks)


def test_short_tail_preserves_every_frame() -> None:
    assert chunk_indices(17, 8) == [list(range(8)), list(range(8, 16)), [16]]
    with pytest.raises(ValueError):
        chunk_indices(0, 8)


def test_input_pairing(tmp_path: Path) -> None:
    sample = make_sample(tmp_path)
    with pytest.raises(ValueError, match="Every frame"):
        BaselineInput(sample_id="bad", frames=sample.frames, masks=sample.masks[:1])
    Image.new("L", (12, 12), 0).save(sample.masks[0])
    with pytest.raises(ValueError, match="nonempty"):
        load_views(sample)


def test_coordinate_contract() -> None:
    grid = occupancy_from_coords(torch.tensor([[0, 0, 0, 0], [0, 63, 2, 3]]))
    assert grid.shape == (64, 64, 64) and grid.sum() == 2 and grid[63, 2, 3]
    with pytest.raises(ValueError):
        occupancy_from_coords(torch.tensor([[0, 64, 2, 3]]))


def test_huggingface_snapshot_symlinks_keep_sibling_checkpoint_paths(tmp_path: Path,
                                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    import json
    import hydra.utils
    from video3d.baselines import api
    blob = tmp_path / "blobs" / "hash"
    blob.parent.mkdir()
    blob.write_text("device: cpu\nworkspace_dir: .\n")
    snapshot = tmp_path / "snapshots" / "revision"
    snapshot.mkdir(parents=True)
    (snapshot / "pipeline.yaml").symlink_to(blob)
    metadata = SimpleNamespace(read_text=lambda name: json.dumps(
        {"vcs_info": {"commit_id": api.PINS["mvsam3d"][1]}}))
    monkeypatch.setattr(api, "distribution", lambda name: metadata)
    monkeypatch.setattr(api, "distributions", lambda: [])
    captured: list[str] = []

    def instantiate(config: Any) -> Any:
        captured.append(config.workspace_dir)
        return SimpleNamespace(models={})

    monkeypatch.setattr(hydra.utils, "instantiate", instantiate)
    api.load_pipeline(BaselineConfig(checkpoint=str(snapshot / "pipeline.yaml")))
    assert captured == [str(snapshot)]


def test_mvsam3d_calls_weighted_coarse_upstream(tmp_path: Path) -> None:
    calls: list[dict] = []

    def run(**kwargs: object) -> dict[str, torch.Tensor]:
        calls.append(kwargs)
        return {"coords": torch.tensor([[0, 4, 5, 6]])}

    result = reconstruct(make_sample(tmp_path), BaselineConfig(), tmp_path / "output",
                         pipeline=SimpleNamespace(run_multi_view=run))
    assert calls[0]["stage1_only"] is True and calls[0]["ss_weighting"] is True
    assert calls[0]["ss_entropy_alpha"] == 30
    assert result.backend == "sam3d" and result.occupied_voxels == 1
    assert np.load(result.occupancy)[4, 5, 6]
    assert (tmp_path / "output/result.json").is_file()


def test_dataset_evaluation_uses_shared_identity_and_metrics(fake_cfg: Any, tmp_path: Path,
                                                           monkeypatch: pytest.MonkeyPatch) -> None:
    from importlib import import_module
    from video3d.data import build_dataset, ground_truth
    from video3d.data.controls import sample_id

    module = import_module("video3d.baselines.evaluate")
    dataset = build_dataset(fake_cfg, "eval", train=False, clip_len=fake_cfg.eval.k)
    info = dataset[0]["info"]
    truth = torch.as_tensor(ground_truth(dataset, info["asset_id"], fake_cfg)).bool()
    indices = torch.nonzero(truth)
    coords = torch.cat([torch.zeros(len(indices), 1, dtype=torch.long), indices], dim=1)

    def run(**kwargs: object) -> dict[str, torch.Tensor]:
        return {"coords": coords}

    monkeypatch.setattr(module, "read_config", lambda path: fake_cfg)
    monkeypatch.setattr(module, "load_pipeline", lambda cfg: SimpleNamespace(run_multi_view=run))
    config = tmp_path / "config.json"
    config.write_text("{}")
    output = Path(module.evaluate(str(config), "mvsam3d", str(tmp_path / "eval"), split="eval", limit=1))
    import json
    row = json.loads(output.read_text())
    assert row["sample_id"] == sample_id(info) and row["viou"] == 1.0
    assert row["chamfer"] == 0.0 and row["baseline"]["backend"] == "sam3d"
    assert row["frame_ids"] == info["frame_ids"]


def test_soft_mask_threshold_rejects_background_halo(tmp_path: Path) -> None:
    sample = make_sample(tmp_path)
    mask = np.full((12, 12), 8, dtype=np.uint8)
    mask[3:9, 3:9] = 240
    Image.fromarray(mask).save(sample.masks[0])
    _, masks, _ = load_views(sample)
    assert masks[0].sum() == 36


def test_native_sam_uses_reference_frame_only(tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []

    def run(**kwargs: Any) -> dict[str, torch.Tensor]:
        calls.append(kwargs)
        return {"coords": torch.tensor([[0, 1, 2, 3]])}

    result = reconstruct(make_sample(tmp_path), BaselineConfig(method="sam3d"), tmp_path / "native",
                         pipeline=SimpleNamespace(run=run))
    assert len(calls) == 1 and calls[0]["image"][0, 0, 0] == 80
    assert result.selected_frames == [0]
    assert result.occupied_voxels == 1


def test_exact_mesh_export_does_not_use_simplified_preview(tmp_path: Path) -> None:
    from video3d.baselines.api import save_prediction
    import trimesh

    mesh = trimesh.creation.box()
    decoded = SimpleNamespace(vertices=torch.tensor(mesh.vertices, dtype=torch.float32),
        faces=torch.tensor(mesh.faces), vertex_attrs=None)
    preview = trimesh.creation.icosphere(subdivisions=0)
    save_prediction({"mesh": [decoded], "glb": preview, "shape": torch.ones(1, 8)}, tmp_path)
    saved = np.load(tmp_path / "mesh-exact.npz")
    np.testing.assert_array_equal(saved["faces"], mesh.faces)
    np.testing.assert_array_equal(saved["vertices"], mesh.vertices)
    assert len(preview.vertices) != len(saved["vertices"])


def test_bvh_surface_queries_and_visibility_cache(tmp_path: Path) -> None:
    import json
    import trimesh
    from video3d.baselines.analysis import cache_visibility, closest_surface, visibility

    cube = trimesh.creation.box()
    points = np.array([[0., 0., 1.], [2., 0., 0.], [0., 0., 0.]])
    closest, distances, faces = closest_surface(cube, points)
    np.testing.assert_allclose(distances, [.5, 1.5, .5], atol=1e-6)
    np.testing.assert_allclose(closest, trimesh.triangles.closest_point(cube.triangles[faces], points), atol=1e-6)
    pose = np.eye(4)
    pose[2, 3] = 2
    camera = dict(K=[[100, 0, 100], [0, 100, 100], [0, 0, 1]], resolution=200,
                  T_object_camera=[pose.tolist()])
    seen = visibility(cube, np.array([[0, 0, .5], [0, 0, -.5], [20, 0, .5]]), camera, [0])
    assert seen.tolist() == [[True, False, False]]
    cube.export(tmp_path / 'gt.ply')
    (tmp_path / 'camera.json').write_text(json.dumps(camera))
    path = tmp_path / 'visibility.npz'
    first = cache_visibility(str(tmp_path/'gt.ply'), str(tmp_path/'camera.json'), [0], str(path), samples=128)
    second = cache_visibility(str(tmp_path/'gt.ply'), str(tmp_path/'camera.json'), [0], str(path), samples=128)
    assert first == second
    saved = np.load(path)
    partition = sum(saved[name].astype(np.int64) for name in ['reference_visible', 'newly_revealed', 'never_observed'])
    np.testing.assert_array_equal(partition, np.ones(128))
    with pytest.raises(ValueError, match='protocol changed'):
        cache_visibility(str(tmp_path/'gt.ply'), str(tmp_path/'camera.json'), [0], str(path), samples=256)
