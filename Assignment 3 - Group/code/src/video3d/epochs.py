"""Finite epoch scheduling and fixed reconstruction previews for the shared trainer."""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any
import shutil

import numpy as np
import torch
from torch.utils.data import Sampler

from video3d.config import ExperimentConfig
from video3d.utils import sample_seed


def update_sizes(examples: int, effective_batch: int, microbatch: int) -> list[list[int]]:
    """Every example appears once; the final optimizer update keeps its true size."""
    if min(examples, effective_batch, microbatch) <= 0 or effective_batch % microbatch:
        raise ValueError("Positive sizes and an effective batch divisible by microbatch are required")
    return [[min(microbatch, size - start) for start in range(0, size, microbatch)]
            for size in (min(effective_batch, examples - offset)
                         for offset in range(0, examples, effective_batch))]


class EpochBatchSampler(Sampler[list[tuple[int, int]]]):
    """Seeded, finite epochs that never put the next epoch in a partial final update."""

    def __init__(self, examples: int, effective_batch: int, microbatch: int,
                 epochs: int, seed: int, start_step: int = 0) -> None:
        self.examples = examples
        self.effective_batch = effective_batch
        self.microbatch = microbatch
        self.epochs = epochs
        self.seed = seed
        self.start_step = start_step
        self.sizes = update_sizes(examples, effective_batch, microbatch)

    def __iter__(self) -> Iterator[list[tuple[int, int]]]:
        for epoch in range(self.epochs):
            generator = torch.Generator().manual_seed(sample_seed(self.seed, "epoch", epoch))
            order: list[int] = torch.randperm(self.examples, generator=generator).tolist()
            for update, sizes in enumerate(self.sizes):
                if epoch * len(self.sizes) + update < self.start_step:
                    continue
                offset: int = update * self.effective_batch
                for size in sizes:
                    yield [(index, epoch) for index in order[offset:offset + size]]
                    offset += size


@torch.no_grad()
def epoch_previews(cfg: ExperimentConfig, sam: Any, encoder: Any, adapter: torch.nn.Module,
                   device: torch.device, directory: Path, epoch: int,
                   wandb_run: Any = None) -> dict[str, Any]:
    """Export the same ten clip/object pairs, without building a second model or evaluator."""
    from video3d.data import build_dataset, ground_truth, reader
    from video3d.media import save_pair
    from video3d.metrics import voxel_iou
    from video3d.storage import atomic_json
    from video3d.train import adapted_condition
    from video3d.data.visibility import object_exposure, exposure_path

    dataset = build_dataset(cfg, cfg.data.train_split, train=False, clip_len=16)
    selected: list[int] = []
    seen: set[str] = set()
    for index, info in enumerate(dataset.index):
        if info["clip_id"] not in seen and "middle" in info["rules"]:
            selected.append(index)
            seen.add(info["clip_id"])
        if len(selected) == cfg.val.epoch_media_samples:
            break
    rows: list[list[Any]] = []
    records: list[dict[str, Any]] = []
    for index in selected:
        sample: dict[str, Any] = dataset[index]
        info: dict[str, Any] = sample["info"]
        condition = adapted_condition(sam, encoder, adapter, [sample], cfg, device)
        latent, _ = sam.sample(condition, [sample_seed(info["clip_id"], info["k"], info["r"])], cfg.sampler)
        prediction: torch.Tensor = sam.decode_latent(latent)[0].cpu()
        truth = torch.as_tensor(ground_truth(dataset, info["asset_id"], cfg)).cpu()
        clip = dataset.clip(info["clip_id"])
        exposure_file: Path = exposure_path(cfg, info["clip_id"], info["k"])
        exposure = object_exposure(clip, info["k"],
                                   reader.load_voxels(dataset.voxels_file, info["asset_id"]), exposure_file)
        record: dict[str, Any] = {"epoch": epoch, "clip_id":info["clip_id"], "object":info["k"],
                                  "reference_frame":info["r"], "asset_id":info["asset_id"],
                                  "voxel_iou":voxel_iou(prediction, truth),
                                  "empty_prediction":not bool(prediction.any()),
                                  "weights":"current trained weights, without EMA",
                                  "role":"training reconstruction, not held-out evaluation", **exposure}
        output: Path = directory / f"epoch-{epoch:02d}" / f"{info['clip_id']}-object-{info['k']}"
        save_pair(sample, prediction, truth, record, output)
        shutil.copy2(exposure_file, output / "surface_visibility.npz")
        records.append(record)
        if wandb_run is not None:
            import wandb
            predicted_points = ((prediction.nonzero().float() + 0.5) / prediction.shape[0] - 0.5).numpy()
            truth_points = ((truth.nonzero().float() + 0.5) / truth.shape[0] - 0.5).numpy()
            with np.load(exposure_file) as visibility:
                intensity = visibility["frame_count"].astype(np.float32) / clip.num_frames * 255
                exposure_cloud = np.column_stack([visibility["points"], intensity, intensity, intensity])
            rows.append([epoch, info["clip_id"], info["k"], wandb.Video(str(output / "input.mp4")),
                         wandb.Image(str(output / "reference.png")), wandb.Image(str(output / "mask.png")),
                         wandb.Object3D(predicted_points) if len(predicted_points) else None,
                         wandb.Object3D(truth_points) if len(truth_points) else None,
                         record["voxel_iou"], record["empty_prediction"],
                         exposure["surface_fraction_seen"], exposure["mean_frames_seen"],
                         wandb.Object3D(exposure_cloud)])
    atomic_json(directory / f"epoch-{epoch:02d}" / "reconstructions.json", records)
    media: dict[str, Any] = {"epoch/reconstruction_count":len(records),
                             "epoch/reconstruction_mean_voxel_iou":sum(item["voxel_iou"] for item in records) / len(records)}
    if wandb_run is not None:
        import wandb
        table = wandb.Table(columns=["epoch", "clip", "object", "input_video", "reference", "mask",
                                    "prediction", "ground_truth", "voxel_iou", "empty_prediction",
                                    "fraction_surface_seen", "mean_frames_seen", "surface_frame_count"], data=rows)
        media["epoch/reconstructions"] = table
        artifact = wandb.Artifact(f"{cfg.run_name}-epoch-{epoch:02d}", type="reconstructions")
        artifact.add_dir(str(directory / f"epoch-{epoch:02d}"))
        wandb_run.log_artifact(artifact)
    return media
