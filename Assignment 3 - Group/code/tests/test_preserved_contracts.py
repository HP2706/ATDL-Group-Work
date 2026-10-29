from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from torch import nn
from video3d.adapters import LoRALinear, inject_lora, restore_trainable, trainable_state
from video3d.data.navi import read_manifest
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

def test_object_split_leakage_is_rejected(tmp_path: Path) -> None:
    base = {"example_id": "x", "object_id": "same", "split": "train", "frames": ["x.png"], "masks": ["m.png"], "video": "v.mp4", "target": "z.pt", "ground_truth_mesh": "g.glb", "reference_tokens": "r.pt"}
    path = tmp_path / "manifest.jsonl"
    path.write_text(json.dumps(base) + "\n" + json.dumps({**base, "example_id": "y", "split": "val"}))
    with pytest.raises(ValueError, match="leaked"):
        read_manifest(path)

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
