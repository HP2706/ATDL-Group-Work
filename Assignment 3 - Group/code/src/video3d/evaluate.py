from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any
import numpy as np
from scipy.spatial import cKDTree
import torch
import wandb
from .config import ExperimentConfig
from .data import Example, clip_tensor
from .model import NativeSAM
from .storage import atomic_json, atomic_torch, rng_state, restore_rng


def surface_metrics(prediction: np.ndarray, truth: np.ndarray, threshold: float = 0.02) -> dict[str, float]:
    if len(prediction) == 0 or len(truth) == 0:
        return {"failed": 1.0, "chamfer": float(np.sqrt(3)), "fscore_002": 0.0}
    p_to_g = cKDTree(truth).query(prediction)[0]
    g_to_p = cKDTree(prediction).query(truth)[0]
    precision, recall = float((p_to_g < threshold).mean()), float((g_to_p < threshold).mean())
    return {"failed": 0.0, "chamfer": float((p_to_g.mean() + g_to_p.mean()) / 2), "fscore_002": 2 * precision * recall / max(precision + recall, 1e-12)}


def export_prediction(logits: torch.Tensor, directory: Path) -> tuple[np.ndarray, Path | None]:
    import trimesh
    from skimage.measure import marching_cubes
    grid = logits.detach().float().cpu().numpy().squeeze()
    np.save(directory / "occupancy_logits.npy", grid)
    if float(grid.min()) >= 0 or float(grid.max()) <= 0:
        return np.empty((0, 3)), None
    vertices, faces, _, _ = marching_cubes(grid, level=0)
    vertices = (vertices + 0.5) / np.array(grid.shape) - 0.5
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    path = directory / "prediction.glb"
    mesh.export(path)
    points, _ = trimesh.sample.sample_surface(mesh, 4096, seed=42)
    np.save(directory / "prediction_points.npy", points)
    return points, path


def evaluate(model: NativeSAM, examples: list[Example], config: ExperimentConfig, step: int, run: Any) -> dict[str, float]:
    import trimesh
    device = next(model.parameters()).device
    table = wandb.Table(columns=["example_id", "object_id", "variant", "step", "input_video", "reference", "mode", "prediction", "ground_truth", "chamfer", "fscore_002", "artifact_name"])
    results: list[dict[str, Any]] = []
    previous_rng = rng_state()
    model.eval()
    selected_examples: list[Example] = []
    seen: set[str] = set()
    for example in examples:
        if example.object_id not in seen:
            selected_examples.append(example)
            seen.add(example.object_id)
        if len(selected_examples) >= config.training.eval_examples:
            break
    for example in selected_examples:
        truth_mesh = trimesh.load(example.ground_truth_mesh, force="mesh")
        truth, _ = trimesh.sample.sample_surface(truth_mesh, 4096, seed=42)
        reference = torch.load(example.reference_tokens, map_location=device, weights_only=True)[None]
        baseline_error: float | None = None
        for mode in ["native_image", "reference_repeated", "video"]:
            directory = Path(config.training.output) / config.run_id / "evaluation" / f"step-{step:06d}" / example.example_id / mode
            directory.mkdir(parents=True, exist_ok=True)
            video = clip_tensor(example, config.data, repeated_reference=mode != "video")[None].to(device)
            torch.manual_seed(config.training.seed)
            torch.cuda.manual_seed_all(config.training.seed)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if mode == "native_image":
                    cache_key = hashlib.sha256((config.model.sam_revision + config.provenance.git_head + str(config.training.seed) + str(config.training.inference_steps)).encode() + Path(example.reference_tokens).read_bytes()).hexdigest()
                    cache = Path(config.training.output) / "native_image_cache" / (cache_key + ".pt")
                    if cache.exists():
                        logits = torch.load(cache, map_location=device, weights_only=True)["logits"]
                    else:
                        logits = model.generate_native_image(reference, config.training.inference_steps)
                        atomic_torch(cache, {"logits": logits.cpu()})
                else:
                    logits = model.generate(video, reference, config.training.inference_steps)
            prediction, mesh_path = export_prediction(logits[0], directory)
            metrics = surface_metrics(prediction, truth)
            if example.occupancy:
                truth_occupancy = np.load(example.occupancy)
                prediction_occupancy = logits[0].float().cpu().numpy().squeeze() > 0
                union = np.logical_or(truth_occupancy, prediction_occupancy).sum()
                metrics["surface_occupancy_iou"] = float(np.logical_and(truth_occupancy, prediction_occupancy).sum() / max(int(union), 1))
            if example.visibility:
                regions = np.load(example.visibility)
                distances = cKDTree(prediction).query(regions["points"])[0] if len(prediction) else np.full(len(regions["points"]), np.sqrt(3))
                for region in ["reference_visible", "newly_revealed", "never_observed"]:
                    selected = regions[region]
                    if selected.any():
                        metrics[f"{region}/completeness_distance"] = float(distances[selected].mean())
                        metrics[f"{region}/recall_002"] = float((distances[selected] < 0.02).mean())
            if mode == "reference_repeated":
                baseline_error = metrics["chamfer"]
            record = {"schema_version": 2, "example_id": example.example_id, "object_id": example.object_id, "split": example.split, "variant": config.variant, "step": step, "mode": mode, "selected_frames": example.frames, "selected_masks": example.masks, "distinct_views": len(set(example.frames)) if mode == "video" else 1, "video": example.video, "reference_frame": example.frames[0], "ground_truth_mesh": example.ground_truth_mesh, "prediction_mesh": str(mesh_path) if mesh_path else None, "metrics": metrics}
            atomic_json(directory / "pairing.json", record)
            artifact_name = f"{config.run_id}-{example.example_id}-{mode}-s{step}"
            artifact = wandb.Artifact(artifact_name, type="reconstruction", metadata=record)
            artifact.add_dir(str(directory))
            if example.visibility:
                artifact.add_file(example.visibility, name="visibility.npz")
            artifact.add_file(example.video, name="input.mp4")
            artifact.add_file(example.frames[0], name="reference.png")
            artifact.add_file(example.ground_truth_mesh, name="ground_truth.glb")
            run.log_artifact(artifact)
            visual = wandb.Object3D(prediction) if len(prediction) else None
            table.add_data(example.example_id, example.object_id, config.variant, step, wandb.Video(example.video, format="mp4"), wandb.Image(example.frames[0]), mode, visual, wandb.Object3D(truth), metrics["chamfer"], metrics["fscore_002"], artifact_name)
            results.append(record)
            if mode == "video" and baseline_error is not None:
                results[-1]["improvement_over_reference"] = baseline_error - metrics["chamfer"]
    aggregate: dict[str, float] = {}
    for mode in ["native_image", "reference_repeated", "video"]:
        selected = [row for row in results if row["mode"] == mode]
        for metric in ["chamfer", "fscore_002", "failed"]:
            aggregate[f"eval/{mode}/{metric}"] = float(np.mean([row["metrics"][metric] for row in selected]))
    for region in ["reference_visible", "newly_revealed", "never_observed"]:
        for mode in ["native_image", "reference_repeated", "video"]:
            key = f"{region}/recall_002"
            values = [row["metrics"][key] for row in results if row["mode"] == mode and key in row["metrics"]]
            if values:
                aggregate[f"eval/{mode}/{key}"] = float(np.mean(values))
    aggregate["eval/improvement_over_reference"] = float(np.mean([row["improvement_over_reference"] for row in results if "improvement_over_reference" in row]))
    run.log({**aggregate, "eval/paired_reconstructions": table}, step=step)
    atomic_json(Path(config.training.output) / config.run_id / f"evaluation-{step:06d}.json", results)
    restore_rng(previous_rng)
    model.train()
    model.conditioning.encoder.eval()
    return aggregate
