from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import chz
import numpy as np
import pytest
import torch
from torch import nn
from video3d.adapters import LoRALinear, inject_lora, restore_trainable, trainable_state
from video3d.cli import resolved_config
from video3d.config import ExperimentConfig
from video3d.data import read_manifest
from video3d.evaluate import surface_metrics
from video3d.model import NativeSAM, VideoConditioning
from video3d.storage import atomic_torch, rng_state, restore_rng


def test_zero_initialized_lora_preserves_pretrained_function_and_gradients() -> None:
    base = nn.Linear(3, 4)
    layer = LoRALinear(base, rank=2, alpha=2)
    x = torch.randn(2, 3, requires_grad=True)
    assert torch.equal(layer(x), base(x))
    layer(x).sum().backward()
    assert layer.b.grad is not None and layer.b.grad.abs().sum() > 0
    assert x.grad is not None
    assert base.weight.grad is None


def test_targets_fail_loudly_and_only_selected_parameters_train() -> None:
    model = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2)).requires_grad_(False)
    assert inject_lora(model, r"0", 2, 2) == ["0"]
    assert set(trainable_state(model)) == {"0.a", "0.b"}
    with pytest.raises(ValueError, match="No linear"):
        inject_lora(model, r"missing", 2, 2)


def test_checkpoint_restores_adapter_outputs_and_rng(tmp_path: Path) -> None:
    layer = LoRALinear(nn.Linear(3, 4), 2, 2)
    x = torch.randn(2, 3)
    layer.b.data.fill_(0.5)
    expected = layer(x).detach().clone()
    path = tmp_path / "checkpoint.pt"
    atomic_torch(path, {"model": trainable_state(layer), "rng": rng_state()})
    random_expected = torch.rand(3)
    layer.b.data.zero_()
    payload = torch.load(path, weights_only=False)
    restore_trainable(layer, payload["model"])
    restore_rng(payload["rng"])
    assert torch.equal(expected, layer(x))
    assert torch.equal(random_expected, torch.rand(3))


def test_nested_config_roundtrip() -> None:
    cfg = ExperimentConfig()
    restored = resolved_config(chz.asdict(cfg))
    assert chz.asdict(cfg) == chz.asdict(restored)


def test_object_split_leakage_is_rejected(tmp_path: Path) -> None:
    base = {"example_id": "x", "object_id": "same", "split": "train", "frames": ["x.png"], "masks": ["m.png"], "video": "v.mp4", "target": "z.pt", "ground_truth_mesh": "g.glb", "reference_tokens": "r.pt"}
    path = tmp_path / "manifest.jsonl"
    path.write_text(json.dumps(base) + "\n" + json.dumps({**base, "example_id": "y", "split": "val"}))
    with pytest.raises(ValueError, match="leaked"):
        read_manifest(path)


def test_surface_error_is_measured_against_truth_and_failures_count() -> None:
    points = np.array([[0., 0., 0.], [0., 0., 0.1]])
    assert surface_metrics(points, points)["chamfer"] == 0
    assert surface_metrics(points, points)["fscore_002"] == 1
    assert surface_metrics(points + 1, points)["chamfer"] > 1
    assert surface_metrics(np.empty((0, 3)), points)["failed"] == 1


class UnitVideoEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(3, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)


class UnitVelocity(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(6, 2)

    def forward(self, x: dict[str, torch.Tensor], t: torch.Tensor, condition: torch.Tensor, d: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        return {"shape": self.projection(condition.mean(dim=1))[:, None, :].expand_as(x["shape"])}


@pytest.mark.parametrize("shortcut", [False, True])
def test_backward_uses_real_upstream_loss_without_inference_wrapper(shortcut: bool) -> None:
    source = os.environ.get("SAM3D_SOURCE")
    if source is None:
        pytest.skip("Set SAM3D_SOURCE to verify the upstream source contract")
    os.environ["LIDRA_SKIP_INIT"] = "1"
    sys.path.insert(0, source)
    from sam3d_objects.model.backbone.generator.flow_matching.model import FlowMatching
    model = NativeSAM.__new__(NativeSAM)
    nn.Module.__init__(model)
    model.conditioning = VideoConditioning(UnitVideoEncoder(), 4, 6, frames=2)
    if shortcut:
        from sam3d_objects.model.backbone.generator.shortcut.model import ShortCut
        model.generator = ShortCut(reverse_fn=UnitVelocity(), self_consistency_prob=0.0, loss_weights={"shape": 1.0})
    else:
        model.generator = FlowMatching(reverse_fn=UnitVelocity(), loss_weights={"shape": 1.0})
    video = torch.randn(1, 2, 3)
    reference = torch.randn(1, 2, 6)
    loss, detail = model.loss(video, reference, {"shape": torch.randn(1, 3, 2)})
    loss.backward()
    assert loss.isfinite()
    assert "flow_matching_loss" in detail
    assert model.conditioning.projector.weight.grad is not None
    assert model.conditioning.encoder.projection.weight.grad is not None
    assert model.generator.reverse_fn.projection.weight.grad is not None


def test_geometry_artifact_is_a_valid_mesh(tmp_path: Path) -> None:
    import trimesh
    from video3d.evaluate import export_prediction
    grid = torch.stack(torch.meshgrid(*[torch.linspace(-0.5, 0.5, 8)] * 3, indexing="ij"))
    logits = 0.3 - torch.linalg.vector_norm(grid, dim=0)
    points, mesh_path = export_prediction(logits, tmp_path)
    assert mesh_path is not None
    mesh = trimesh.load(mesh_path, force="mesh")
    assert len(mesh.faces) > 0 and len(points) == 4096
    assert np.isfinite(points).all()
    assert (tmp_path / "occupancy_logits.npy").is_file()


def test_reference_control_repeats_only_the_reference(tmp_path: Path) -> None:
    from PIL import Image
    from video3d.config import DataConfig
    from video3d.data import Example, clip_tensor
    paths: list[str] = []
    masks: list[str] = []
    for index, value in enumerate([0, 255]):
        path, mask = tmp_path / f"{index}.png", tmp_path / f"mask-{index}.png"
        Image.new("RGB", (8, 8), (value, value, value)).save(path)
        Image.new("L", (8, 8), 255).save(mask)
        paths.append(str(path))
        masks.append(str(mask))
    row = Example(example_id="x", object_id="x", split="val", frames=paths, masks=masks, video="x.mp4", target="z.pt", ground_truth_mesh="x.glb", reference_tokens="r.pt")
    config = DataConfig(frames=4, image_size=8)
    control, clip = clip_tensor(row, config, True), clip_tensor(row, config)
    assert torch.equal(control[:, 0], control[:, -1])
    assert not torch.equal(clip[:, 0], clip[:, -1])


def test_no_local_training_or_profiling() -> None:
    from video3d.train import train
    from video3d.profile import profile
    if torch.cuda.is_available():
        pytest.skip("This test verifies the laptop guard")
    with pytest.raises(RuntimeError, match="remote GPU"):
        train(ExperimentConfig())
    with pytest.raises(RuntimeError, match="remote GPU"):
        profile(ExperimentConfig())


def test_lora_initialization_does_not_change_training_noise_rng() -> None:
    model = nn.Sequential(nn.Linear(3, 4)).requires_grad_(False)
    state = torch.get_rng_state()
    expected = torch.rand(3)
    torch.set_rng_state(state)
    inject_lora(model, r"0", 2, 2)
    assert torch.equal(expected, torch.rand(3))


def test_wandb_accepts_paired_video_and_3d_artifacts_offline(tmp_path: Path) -> None:
    import imageio.v2 as imageio
    import wandb
    video = tmp_path / "input.mp4"
    imageio.mimwrite(video, np.zeros((2, 16, 16, 3), dtype=np.uint8), fps=4)
    points = np.array([[0., 0., 0.], [0.1, 0.2, 0.3]])
    with wandb.init(mode="offline", project="paired-artifact-contract", dir=str(tmp_path)) as run:
        table = wandb.Table(columns=["example_id", "input_video", "prediction"])
        table.add_data("object-clip-A", wandb.Video(str(video), format="mp4"), wandb.Object3D(points))
        run.log({"eval/paired_reconstructions": table})
        artifact = wandb.Artifact("object-clip-A", type="reconstruction", metadata={"example_id": "object-clip-A"})
        artifact.add_file(str(video))
        run.log_artifact(artifact)
        assert len(table.data) == 1


def test_sweep_rejects_incomplete_runs_and_resumes_completed_variants(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib
    from typing import Any
    from video3d.config import ProvenanceConfig
    module = importlib.import_module("video3d.sweep")
    config = chz.replace(ExperimentConfig(), training=chz.replace(ExperimentConfig().training, output=str(tmp_path), group="test-group"))
    (tmp_path / "profile.json").write_text(json.dumps({"group": "test-group", "git_head": "abc", "selected_microbatch": 1, "selected_accumulation": 4}))
    monkeypatch.setattr(module, "capture_provenance", lambda root: ProvenanceConfig(git_head="abc"))
    calls: list[str] = []

    class FinishedProcess:
        def __init__(self, command: list[str], **options: Any) -> None:
            resolved = Path(command[-1])
            payload = json.loads(resolved.read_text())
            variant = resolved.stem
            calls.append(variant)
            output = tmp_path / f"test-group-{variant}"
            output.mkdir(exist_ok=True)
            (output / "result.json").write_text(json.dumps({"completed_steps": variant != "B", "config": payload}))

        def wait(self, timeout: float | None = None) -> int:
            return 0

    monkeypatch.setattr(module.subprocess, "Popen", FinishedProcess)
    with pytest.raises(RuntimeError, match="failed"):
        module.sweep(config)
    assert calls == ["A", "B", "C", "D"]
    with pytest.raises(RuntimeError, match="failed"):
        module.sweep(config)
    assert calls == ["A", "B", "C", "D", "B"]
    report = json.loads((tmp_path / "test-group/sweep.json").read_text())
    assert report["results"][0]["status"] == "already_completed"
    assert report["results"][1]["status"] == "incomplete"


@pytest.mark.parametrize("numpy_transform", [False, True])
def test_visibility_accepts_mixed_navi_matrix_types(numpy_transform: bool, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from types import SimpleNamespace
    from video3d.prepare import visibility
    transform = np.eye(4, dtype=np.float64) if numpy_transform else torch.eye(4, dtype=torch.float64)
    projection = torch.eye(4, dtype=torch.float32)
    projection[3, 2], projection[3, 3] = 1.0, 0.0
    depth = np.zeros((4, 4), dtype=np.float32)
    depth[2, 2], depth[3, 2] = 1.0, 1.0
    navi = SimpleNamespace(camera_matrices_from_annotation=lambda annotation: (transform, projection), read_depth_from_png=lambda path: depth)
    monkeypatch.setitem(sys.modules, "data_util", navi)
    points = np.array([[0., 0., 1.], [0., 0.5, 1.], [0., 0., -1.], [3., 0., 1.]], dtype=np.float64)
    result = visibility(points, {"image_size": [4, 4], "filename": "frame.jpg"}, tmp_path)
    assert result.tolist() == [True, True, False, False]


def test_upstream_multimodal_block_bf16_backward_preserves_float_norms() -> None:
    from video3d.model import bf16_with_float_norms
    source = os.environ.get("SAM3D_SOURCE")
    if source is None:
        pytest.skip("Set SAM3D_SOURCE to verify the upstream source contract")
    os.environ["LIDRA_SKIP_INIT"] = "1"
    sys.path.insert(0, source)
    from sam3d_objects.model.backbone.tdfy_dit.modules.transformer.modulated import MOTModulatedTransformerCrossBlock
    block = MOTModulatedTransformerCrossBlock(channels=64, ctx_channels=64, num_heads=4, latent_names=["shape", "pose"], use_checkpoint=True, qk_rms_norm=True, qk_rms_norm_cross=True).requires_grad_(False)
    bf16_with_float_norms(block)
    inject_lora(block, r"cross_attn\.shape\.(to_q|to_kv|to_out)", 2, 2)
    context = torch.randn(2, 16, 64, dtype=torch.bfloat16, requires_grad=True)
    latent = {"shape": torch.randn(2, 8, 64, dtype=torch.bfloat16), "pose": torch.randn(2, 4, 64, dtype=torch.bfloat16)}
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = block(latent, torch.randn(2, 64, dtype=torch.bfloat16), context)
        loss = output["shape"].float().square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert context.grad is not None and context.grad.abs().sum() > 0
    assert block.norm2["shape"].weight.dtype == torch.float32
    assert block.cross_attn["shape"].to_out.b.grad is not None
