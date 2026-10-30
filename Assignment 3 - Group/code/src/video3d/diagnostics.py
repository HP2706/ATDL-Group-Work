"""Shared video-use diagnostics for standalone evaluation and epoch previews.

Attention describes routing, while paired controls measure dependence and benefit.
All controls preserve the reference image, reference mask and sampling noise.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict
import torch

from video3d.config import ExperimentConfig
from video3d.metrics import surface, voxel_iou, voxel_points
from video3d.utils import sample_seed


class VideoEvaluation(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    prediction: torch.Tensor
    diagnostics: dict[str, Any]
    controls: dict[str, dict[str, Any]]


def attention_statistics(weights: torch.Tensor, coverage: torch.Tensor, dt: torch.Tensor,
                         padding: torch.Tensor | None = None) -> dict[str, Any]:
    """Head-resolved mass and enrichment over uniform attention; fractional mask coverage.

    Only called for single-sample evaluation, so no sample identities are averaged.
    V-JEPA keys describe frame pairs, not individually identifiable source frames.
    """
    if len(weights) != 1:
        raise ValueError("Attention diagnostics require single-sample evaluation")
    valid = torch.ones_like(coverage[0], dtype=torch.bool) if padding is None else ~padding[0]
    mass = weights[0].mean(dim=1)[:, valid].float()  # heads x keys, averaged over queries
    coverage = coverage[0, valid].float()
    dt = dt[0, valid]
    object_share = coverage.mean()
    object_mass = (mass * coverage).sum(-1)
    groups: list[dict[str, Any]] = []
    for offset in dt.unique(sorted=True):
        selected = dt == offset
        share = float(selected.float().mean().item())
        value = mass[:, selected].sum(-1)
        groups.append({"offset": int(offset.item()), "token_fraction": share,
                       "mass_per_head": value.tolist(), "mass": float(value.mean().item()),
                       "enrichment": float(value.mean().item()) / share})
    return {"object_token_fraction": float(object_share.item()),
            "object_attention_mass": float(object_mass.mean().item()),
            "object_mass_per_head": object_mass.tolist(),
            "object_enrichment": float((object_mass.mean()/object_share).item()) if object_share > 0 else None,
            "background_enrichment": float(((1-object_mass.mean())/(1-object_share)).item()) if object_share < 1 else None,
            "temporal_groups": groups}


def surface_region_metrics(prediction: torch.Tensor, exposure_file: Path, reference_frame: int,
                           frame_ids: list[int]) -> dict[str, float | int | None]:
    """Canonical surface recall/distance on fixed GT regions, without ICP.

    Regions always use the correct video's frames, including for wrong/repeated
    controls. This prevents changing the target region between comparisons.
    """
    from scipy.spatial import cKDTree
    with np.load(exposure_file) as saved:
        points = saved["points"]
        visible = saved["visible"]
    reference = visible[reference_frame].astype(bool)
    observed = visible[frame_ids].any(axis=0)
    masks = {"reference_visible": reference, "reference_unseen": ~reference,
             "newly_revealed": observed & ~reference, "never_observed": ~observed}
    predicted = voxel_points(surface(prediction.bool().cpu())).numpy()
    distances = cKDTree(predicted).query(points)[0] if len(predicted) else None
    metrics: dict[str, float | int | None] = {}
    for name, selected in masks.items():
        count = int(selected.sum())
        prefix = f"surface/{name}"
        metrics[f"{prefix}/points"] = count
        metrics[f"{prefix}/recall_002"] = (float((distances[selected] < 0.02).mean())
                                            if distances is not None else 0.0) if count else None
        metrics[f"{prefix}/completeness_distance"] = (float(distances[selected].mean())
                                                      if distances is not None and count else None)
    return metrics


def evaluation_metrics(prediction: torch.Tensor, truth: torch.Tensor, exposure_file: Path | None,
                       info: dict[str, Any]) -> dict[str, Any]:
    metrics: dict[str, Any] = {"viou": voxel_iou(prediction, truth)}
    if exposure_file is not None:
        metrics.update(surface_region_metrics(prediction, exposure_file, info["r"], info["frame_ids"]))
    elif info.get("visibility"):
        from video3d.metrics import visibility_metrics
        metrics.update(visibility_metrics(prediction, info["visibility"]))
    return metrics


def exposure_for_sample(cfg: ExperimentConfig, dataset: Any, info: dict[str, Any]) -> Path | None:
    if cfg.data.source != "synthetic":
        return None
    from video3d.data import reader
    from video3d.data.visibility import exposure_path, object_exposure
    path = exposure_path(cfg, info["clip_id"], info["k"])
    object_exposure(dataset.clip(info["clip_id"]), info["k"],
                    reader.load_voxels(dataset.voxels_file, info["asset_id"]), path)
    return path


def scalar_diagnostics(diagnostics: dict[str, Any]) -> dict[str, float]:
    """Flat scalars for W&B charts; detailed per-head/frame data stays in records."""
    output: dict[str, float] = {}
    for name, value in diagnostics.items():
        if isinstance(value, (int, float)):
            output[f"video/{name}"] = float(value)
        elif isinstance(value, dict):
            for key, scalar in value.items():
                if isinstance(scalar, (int, float)):
                    output[f"video/{name}/{key}"] = float(scalar)
    return output


@torch.no_grad()
def evaluate_video(sample: dict[str, Any], other: dict[str, Any] | None, truth: torch.Tensor,
                   sam: Any, encoder: Any, adapter: torch.nn.Module, cfg: ExperimentConfig,
                   device: torch.device, exposure_file: Path | None = None) -> VideoEvaluation:
    """One model build, fixed reference/noise, configurable paired video controls."""
    from video3d.data.controls import repeat_reference, with_video_of
    from video3d.train import adapted_condition
    modes = cfg.eval.video_controls
    if "correct" not in modes or len(set(modes)) != len(modes):
        raise ValueError("Video controls must include correct exactly once")
    unknown = set(modes) - {"correct", "repeated", "wrong", "disabled"}
    if unknown:
        raise ValueError(f"Unknown video controls: {sorted(unknown)}")
    info = sample["info"]
    seed = sample_seed(info["clip_id"], info["k"], info["r"])
    predictions: dict[str, torch.Tensor] = {}
    controls: dict[str, dict[str, Any]] = {}
    correct_diagnostics: dict[str, Any] = {}
    for mode in modes:
        changed = sample
        if mode == "repeated":
            changed = repeat_reference(sample)
        elif mode == "wrong":
            if other is None or other["info"]["clip_id"] == info["clip_id"]:
                raise ValueError("Wrong-video control requires a different clip")
            changed = with_video_of(sample, other)
        # Use the same attention implementation in every arm; otherwise SDPA
        # versus explicit-weight kernels can confound small video effects.
        stats: dict[str, Any] = {}
        condition = adapted_condition(sam, encoder, adapter, [changed], cfg, device,
                                      video_disabled=mode == "disabled", diagnostics=stats)
        latent, _ = sam.sample(condition, [seed], cfg.sampler)
        prediction = sam.decode_latent(latent)[0].cpu()
        predictions[mode] = prediction
        controls[mode] = evaluation_metrics(prediction, truth, exposure_file, info)
        controls[mode]["sampling_seed"] = seed
        if mode == "wrong":
            controls[mode]["video_from"] = changed["info"]["video_from"]
        if mode == "correct":
            correct_diagnostics = stats
    correct = controls["correct"]
    for mode, values in controls.items():
        if mode == "correct":
            continue
        values["changed_voxel_fraction"] = float((predictions[mode].bool() != predictions["correct"].bool()).float().mean())
        values["delta_correct_minus_control"] = {
            key: correct[key] - value for key, value in values.items()
            if (key == "viou" or key.endswith("recall_002") or key.endswith("completeness_distance"))
            and value is not None and correct[key] is not None}
    correct_diagnostics["temporal_resolution"] = "frame pairs" if cfg.encoder.kv_source != "dinov2_ref" else "reference frame"
    correct_diagnostics["frame_ids"] = info["frame_ids"]
    frame_groups = ([[info["r"]]] if cfg.encoder.kv_source == "dinov2_ref"
                    else [info["frame_ids"][i:i+2] for i in range(0, len(info["frame_ids"]), 2)])
    for block in correct_diagnostics.values():
        if isinstance(block, dict) and "temporal_groups" in block:
            for group, frames in zip(block["temporal_groups"], frame_groups, strict=True):
                group["frame_ids"] = frames
    correct_diagnostics["region_protocol"] = "canonical voxel-derived surface estimate; fixed correct-video regions; recall tolerance 0.02"
    return VideoEvaluation(prediction=predictions["correct"], diagnostics=correct_diagnostics, controls=controls)


def attention_table(records: list[dict[str, Any]]) -> Any:
    """W&B table preserves frame-pair and head detail without averaging it away."""
    import wandb
    rows: list[list[Any]] = []
    for record in records:
        for name, block in record.get("video_diagnostics", {}).items():
            if not isinstance(block, dict) or "temporal_groups" not in block:
                continue
            for group in block["temporal_groups"]:
                for head, mass in enumerate(group["mass_per_head"]):
                    rows.append([record["clip_id"], name, head, str(group["frame_ids"]),
                                 mass, group["token_fraction"], mass/group["token_fraction"]])
    return wandb.Table(columns=["clip", "block", "head", "frames", "attention_mass",
                                "token_fraction", "enrichment"], data=rows)


def summarize_video_records(records: list[dict[str, Any]]) -> dict[str, float | int]:
    """Paired means, retaining sample counts; missing surface regions stay missing."""
    values: dict[str, list[float]] = {}
    count = 0
    for record in records:
        controls = record.get("video_controls", {})
        if not controls:
            continue
        count += 1
        for key, value in scalar_diagnostics(record["video_diagnostics"]).items():
            values.setdefault(key, []).append(value)
        for mode, metrics in controls.items():
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and key != "sampling_seed":
                    values.setdefault(f"video_controls/{mode}/{key}", []).append(float(value))
            for key, value in metrics.get("delta_correct_minus_control", {}).items():
                values.setdefault(f"video_benefit/{mode}/{key}", []).append(value)
    summary: dict[str, float | int] = {"video_diagnostic_samples": count}
    for key, samples in values.items():
        summary[key] = sum(samples)/len(samples)
        summary[f"{key}/sample_count"] = len(samples)
    return summary
