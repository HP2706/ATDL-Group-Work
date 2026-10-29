"""Paired reconstruction artifacts shared by both conditioning strategies."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import numpy as np
import torch


def export_mesh(voxels: torch.Tensor, path: Path) -> Path | None:
    import trimesh
    from skimage.measure import marching_cubes
    grid = voxels.detach().cpu().numpy().astype(np.float32)
    if grid.min() == grid.max():
        return None
    vertices, faces, _, _ = marching_cubes(np.pad(grid, 1), level=0.5)
    vertices = (vertices - 1 + 0.5) / np.array(grid.shape) - 0.5
    trimesh.Trimesh(vertices=vertices, faces=faces, process=False).export(path)
    return path


def save_pair(sample: dict[str, Any], pred: torch.Tensor, truth: torch.Tensor,
              record: dict[str, Any], directory: Path, wandb_run: Any = None) -> None:
    import imageio.v3 as iio
    from PIL import Image
    directory.mkdir(parents=True, exist_ok=True)
    inputs = sample["inputs"]
    frames = (inputs["frames"].detach().cpu().permute(0,2,3,1).numpy() * 255).clip(0,255).astype(np.uint8)
    iio.imwrite(directory / "input.mp4", frames, fps=8)
    Image.fromarray(inputs["ref_rgb"].cpu().numpy()).save(directory / "reference.png")
    Image.fromarray(inputs["ref_mask"].cpu().numpy().astype(np.uint8)*255).save(directory / "mask.png")
    export_mesh(pred, directory / "prediction.glb")
    export_mesh(truth, directory / "ground_truth.glb")
    (directory / "pairing.json").write_text(json.dumps(record, indent=2, allow_nan=False))
    if wandb_run is not None:
        import wandb
        artifact = wandb.Artifact(directory.name, type="reconstruction", metadata=record)
        artifact.add_dir(str(directory))
        wandb_run.log_artifact(artifact)
        media: dict[str, Any] = {"eval/input":wandb.Video(str(directory / "input.mp4")),
                                 "eval/reference":wandb.Image(str(directory / "reference.png"))}
        for name in ["prediction","ground_truth"]:
            path=directory / f"{name}.glb"
            if path.exists():
                media[f"eval/{name}"]=wandb.Object3D(str(path))
        wandb_run.log(media)
