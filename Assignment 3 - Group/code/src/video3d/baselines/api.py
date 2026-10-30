"""Coarse and refined baselines using installed upstream libraries, never upstream CLIs.

MV-SAM3D: Li et al., arXiv:2603.11633, commit abb04b5e8af5bc33b0265bdf19937e76bbb6bcdd.
Stream3D: Zhou et al., arXiv:2605.21472, commit be551cc756f5710875ec1c6a4c0fa545b56d2c4f.
The two wheels own the same sam3d_objects namespace: use isolated uv environments.
"""
from __future__ import annotations

from importlib.metadata import distribution, distributions
from pathlib import Path
from typing import Any, Literal
import json
import os
import time

import chz
import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, model_validator

PINS = {
    "sam3d": ("devinli123/MV-SAM3D", "abb04b5e8af5bc33b0265bdf19937e76bbb6bcdd"),
    "mvsam3d": ("devinli123/MV-SAM3D", "abb04b5e8af5bc33b0265bdf19937e76bbb6bcdd"),
    "stream3d": ("kaichen-z/STREAM3D", "be551cc756f5710875ec1c6a4c0fa545b56d2c4f"),
}


@chz.chz
class BaselineConfig:
    method: Literal["sam3d", "mvsam3d", "stream3d"] = "mvsam3d"
    checkpoint: str = "checkpoints/hf/pipeline.yaml"
    steps: int = 25
    sampling_mode: Literal["flow", "shortcut"] = "flow"
    seed: int = 42
    chunk_size: int = 8
    topk: int = 8
    memory_depth: int = 5
    entropy_alpha: float = 30.0
    kappa: float = 8.0
    full_mesh: bool = False


class BaselineInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sample_id: str
    frames: list[Path] = Field(min_length=1)
    masks: list[Path] = Field(min_length=1)
    # Optional camera-local XYZ in PyTorch3D axes, shape (3,H,W).
    pointmaps: list[Path] | None = None
    ground_truth: Path | None = None
    mask_threshold: float = 0.5
    reference_index: int = 0

    @model_validator(mode="after")
    def validate_views(self) -> BaselineInput:
        if not 0 <= self.reference_index < len(self.frames):
            raise ValueError('Reference index must select an input frame')
        if len(self.frames) != len(self.masks):
            raise ValueError("Every frame must have one target mask")
        if self.pointmaps is not None and len(self.pointmaps) != len(self.frames):
            raise ValueError("Every frame must have one pointmap")
        for path in self.frames + self.masks + (self.pointmaps or []):
            if not path.is_file():
                raise ValueError(f"Missing input: {path}")
        return self


class BaselineResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sample_id: str
    method: str
    backend: Literal["sam3d"] = "sam3d"
    scope: Literal["coarse_stage1", "refined_mesh"] = "coarse_stage1"
    revision: str
    seed: int
    steps: int
    sampling_mode: str
    pointmap_source: str
    selected_frames: list[int]
    chunks: list[list[int]]
    occupancy: Path
    occupied_voxels: int
    elapsed_seconds: float
    metrics: dict[str, Any] = Field(default_factory=dict)
    artifacts: dict[str, str] = Field(default_factory=dict)
    resources: dict[str, Any] = Field(default_factory=dict)


def load_views(sample: BaselineInput) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray] | None]:
    images = [np.asarray(Image.open(path).convert("RGB")) for path in sample.frames]
    masks = [np.asarray(Image.open(path).convert("L")) >= round(255 * sample.mask_threshold) for path in sample.masks]
    for image, mask in zip(images, masks, strict=True):
        if image.shape[:2] != mask.shape or not mask.any():
            raise ValueError("Masks must be nonempty and have the same dimensions as their images")
    pointmaps = None if sample.pointmaps is None else [np.load(path, allow_pickle=False) for path in sample.pointmaps]
    if pointmaps is not None:
        for image, pointmap in zip(images, pointmaps, strict=True):
            if pointmap.shape != (3, *image.shape[:2]):
                raise ValueError("Pointmaps must have shape (3,H,W), in PyTorch3D camera axes")
    return images, masks, pointmaps


def chunk_indices(length: int, size: int) -> list[list[int]]:
    if length <= 0 or size <= 0:
        raise ValueError("Frame count and chunk size must be positive")
    # Disjoint chunks ensure each observation enters evidence memory exactly once.
    return [list(range(start, min(start + size, length))) for start in range(0, length, size)]


def occupancy_from_coords(coords: Any) -> np.ndarray:
    indices = coords.detach().cpu().numpy()[:, 1:].astype(np.int64)
    if indices.shape[1] != 3 or (indices < 0).any() or (indices >= 64).any():
        raise ValueError("Expected batch-indexed 64-cubed SAM 3D coordinates")
    occupancy = np.zeros((64, 64, 64), dtype=bool)
    occupancy[tuple(indices.T)] = True
    return occupancy


def load_pipeline(cfg: BaselineConfig) -> Any:
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    if cfg.method not in PINS or min(cfg.steps, cfg.chunk_size, cfg.topk, cfg.memory_depth) < 1:
        raise ValueError("Unknown method or nonpositive baseline configuration")
    package = "streaming3d" if cfg.method == "stream3d" else "sam3d-objects"
    installed = {item.metadata["Name"].lower().replace("_", "-") for item in distributions()}
    if {"sam3d-objects", "streaming3d"} <= installed:
        raise RuntimeError("Conflicting sam3d_objects providers: use separate uv environments")
    metadata = distribution(package)
    source = metadata.read_text("direct_url.json")
    if source is None or json.loads(source).get("vcs_info", {}).get("commit_id") != PINS[cfg.method][1]:
        raise RuntimeError(f"Install the pinned {cfg.method} extra in its own uv environment")
    os.environ.setdefault("LIDRA_SKIP_INIT", "1")
    config = OmegaConf.load(cfg.checkpoint)
    # HF snapshots use symlinked files; resolving the YAML itself points into
    # blobs/, where sibling model configs and checkpoints do not exist.
    config.workspace_dir = str(Path(cfg.checkpoint).absolute().parent)
    config.device, config.compile_model, config.rendering_engine = "cuda", False, "pytorch3d"
    pipeline = instantiate(config)
    if not cfg.full_mesh:
        for name in ("slat_generator", "slat_decoder_gs", "slat_decoder_gs_4", "slat_decoder_mesh"):
            if name in pipeline.models:
                del pipeline.models[name]
    torch.cuda.empty_cache()
    return pipeline


def refinement_options(cfg: BaselineConfig) -> dict[str, Any]:
    if not cfg.full_mesh:
        return {}
    return dict(stage2_inference_steps=cfg.steps, use_stage2_distillation=cfg.sampling_mode == "shortcut",
        decode_formats=["mesh", "gaussian"], with_mesh_postprocess=False,
        with_texture_baking=False, use_vertex_color=True)


def save_prediction(prediction: dict[str, Any], output: Path) -> dict[str, str]:
    """Preserve exact decoder geometry before upstream GLB simplification."""
    import torch
    import trimesh

    mesh = prediction["mesh"][0]
    vertices, faces = mesh.vertices.detach().cpu().numpy(), mesh.faces.detach().cpu().numpy()
    if not len(vertices) or not len(faces):
        raise ValueError("Refinement produced an empty mesh")
    arrays = {"vertices": vertices, "faces": faces}
    if mesh.vertex_attrs is not None:
        arrays["vertex_attrs"] = mesh.vertex_attrs.detach().cpu().numpy()
    np.savez_compressed(output / "mesh-exact.npz", **arrays)
    trimesh.Trimesh(vertices=vertices, faces=faces, process=False).export(output / "mesh-exact.ply")
    prediction["glb"].export(str(output / "preview.glb"))
    tensors = {key: value.detach().cpu() for key, value in prediction.items() if isinstance(value, torch.Tensor)}
    torch.save(tensors, output / "coarse-pose-pointmaps.pt")
    for key in ("token_vote_memory", "raw_view_pointmaps", "view_ss_input_dicts"):
        if key in prediction:
            torch.save(prediction[key], output / (key+".pt"))
    manager = prediction.get("weight_manager")
    if manager is not None:
        # Includes weights and diagnostic arrays exposed by the pinned backend.
        torch.save(manager.get_analysis_data(), output / "view-weight-analysis.pt")
    (output / "prediction-keys.json").write_text(json.dumps({k: type(v).__name__ for k, v in prediction.items()}, indent=2))
    return {"exact_mesh_arrays": str((output / "mesh-exact.npz").resolve()),
            "exact_mesh_ply": str((output / "mesh-exact.ply").resolve()),
            "preview_glb": str((output / "preview.glb").resolve()),
            "coarse_pose_pointmaps": str((output / "coarse-pose-pointmaps.pt").resolve())}


def run_mvsam3d(pipeline: Any, cfg: BaselineConfig, images: list[np.ndarray], masks: list[np.ndarray],
                pointmaps: list[np.ndarray] | None) -> dict[str, Any]:
    refinement = refinement_options(cfg)
    if cfg.full_mesh:
        from sam3d_objects.utils.latent_weighting import WeightingConfig
        refinement["weighting_config"] = WeightingConfig(entropy_alpha=cfg.entropy_alpha)
    return pipeline.run_multi_view(**refinement, view_images=images, view_masks=masks, view_pointmaps=pointmaps,
        seed=cfg.seed, stage1_only=not cfg.full_mesh, stage1_inference_steps=cfg.steps,
        use_stage1_distillation=cfg.sampling_mode == "shortcut", mode="multidiffusion", ss_weighting=True,
        ss_entropy_alpha=cfg.entropy_alpha, ss_entropy_layer=9, ss_warmup_steps=1)


def run_stream3d(pipeline: Any, cfg: BaselineConfig, images: list[np.ndarray], masks: list[np.ndarray],
                 pointmaps: list[np.ndarray] | None) -> tuple[dict[str, Any], list[list[int]]]:
    import torch
    from streaming.backend import backend_sam3d
    from streaming.backend.attention_metric import ConditionMetricMode, stage1_score_by_view_from_warmup
    from streaming.backend.selector.selector import TokenVoteMemory
    from streaming.backend.selector.selector_vote import (view_score_batch_from_score_by_view,
        update_token_vote_memory, select_token_vote_views)

    memory = TokenVoteMemory()
    noise = None
    selections: list[list[int]] = []
    memories: list[dict[str, Any]] = []
    metric = ConditionMetricMode.MASS_RELATIVE
    for indices in chunk_indices(len(images), cfg.chunk_size):
        torch.manual_seed(cfg.seed)
        prepared = backend_sam3d.prepare_views(pipeline,
            view_images=[images[i] for i in indices], view_masks=[masks[i] for i in indices],
            view_pointmaps=None if pointmaps is None else [pointmaps[i] for i in indices],
            runtime_frame_keys=[str(i) for i in indices], seed=cfg.seed, mode="multidiffusion",
            profile=backend_sam3d.new_run_profile())
        warmup = backend_sam3d.collect_ss_warmup_attentions(pipeline=pipeline, prepared=prepared,
            inference_steps=cfg.steps, warmup_steps=1, attention_layer=9, fixed_initial_noise=noise)
        noise = warmup["initial_noise"]
        scores = stage1_score_by_view_from_warmup(warmup=warmup, metric=metric,
            patch_start=5, patch_end=1374, kappa=cfg.kappa)
        batch = view_score_batch_from_score_by_view(score_by_view=scores, global_frame_indices=indices)
        memory = update_token_vote_memory(memory, batch, cfg.memory_depth)
        memories.append({"scores": memory.score.detach().cpu(),
                         "global_frame_index": memory.global_frame_index.detach().cpu()})
        selected = select_token_vote_views(memory, cfg.topk, metric=metric)
        selections.append(sorted(int(row["global_frame_index"]) for row in selected))
        del prepared, warmup
    indices = selections[-1]
    refinement = refinement_options(cfg)
    if cfg.full_mesh:
        from streaming.backend.attention_metric import AttentionWeightingConfig
        refinement["weighting_config"] = AttentionWeightingConfig(
            weight_source="mass_relative", jam_alpha=1.0, jam_kappa=cfg.kappa, uniform_blend=0.0,
            use_patch_mass=False, patch_mass_gamma=1.0, final_temperature=1.0, min_weight=0.01,
            attention_layer=6, attention_step=0, patch_start=5, patch_end=1374)
    result = pipeline.run_multi_view(**refinement, view_images=[images[i] for i in indices],
        view_masks=[masks[i] for i in indices],
        view_pointmaps=None if pointmaps is None else [pointmaps[i] for i in indices],
        seed=cfg.seed, stage1_only=not cfg.full_mesh, stage1_inference_steps=cfg.steps,
        use_stage1_distillation=cfg.sampling_mode == "shortcut", mode="multidiffusion", ss_weighting=True,
        ss_attention_layer=9, ss_weight_source="mass_relative", ss_jam_kappa=cfg.kappa)
    result["token_vote_memory"] = memories
    return result, selections


def reconstruct(sample: BaselineInput, cfg: BaselineConfig, output: Path,
                pipeline: Any | None = None) -> BaselineResult:
    """Generate one prediction; pipeline injection reuses model weights across samples."""
    import torch

    started = time.perf_counter()
    images, masks, pointmaps = load_views(sample)
    load_started = time.perf_counter()
    reused_pipeline = pipeline is not None
    pipeline = load_pipeline(cfg) if pipeline is None else pipeline
    load_seconds = 0.0 if reused_pipeline else time.perf_counter() - load_started
    if cfg.full_mesh:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    inference_started = time.perf_counter()
    if cfg.full_mesh:
        decode = pipeline.decode_slat

        def decode_and_capture(slat: Any, formats: list[str]) -> dict[str, Any]:
            outputs = decode(slat, formats)
            outputs["slat_features"] = slat.feats.detach().cpu()
            outputs["slat_coordinates"] = slat.coords.detach().cpu()
            return outputs

        pipeline.decode_slat = decode_and_capture
    with torch.inference_mode():
        if cfg.method == "sam3d":
            reference = sample.reference_index
            prediction = pipeline.run(image=images[reference], mask=masks[reference], seed=cfg.seed,
                stage1_only=not cfg.full_mesh, stage1_inference_steps=cfg.steps,
                use_stage1_distillation=cfg.sampling_mode == "shortcut",
                pointmap=None if pointmaps is None else pointmaps[reference],
                with_layout_postprocess=False, **refinement_options(cfg))
            selections = [[reference]]
        elif cfg.method == "mvsam3d":
            prediction = run_mvsam3d(pipeline, cfg, images, masks, pointmaps)
            selections = [list(range(len(images)))]
        elif cfg.method == "stream3d":
            prediction, selections = run_stream3d(pipeline, cfg, images, masks, pointmaps)
        else:
            raise ValueError(f"Unknown baseline {cfg.method}")
    if cfg.full_mesh:
        pipeline.decode_slat = decode
        torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - inference_started
    resources = {"model_load_seconds": load_seconds, "model_reused": reused_pipeline, "inference_seconds": inference_seconds}
    if cfg.full_mesh:
        import resource
        resources.update(cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            gpu=torch.cuda.get_device_name())
    occupancy = occupancy_from_coords(prediction["coords"])
    output.mkdir(parents=True, exist_ok=True)
    path = output / "occupancy.npy"
    np.save(path, occupancy)
    artifacts = save_prediction(prediction, output) if cfg.full_mesh else {}
    metrics: dict[str, Any] = {}
    if sample.ground_truth is not None:
        from video3d.metrics import shape_metrics
        from video3d.config import MetricsConfig
        metrics = shape_metrics(occupancy, np.load(sample.ground_truth, allow_pickle=False), MetricsConfig())
    result = BaselineResult(sample_id=sample.sample_id, method=cfg.method, revision=PINS[cfg.method][1],
        scope="refined_mesh" if cfg.full_mesh else "coarse_stage1", artifacts=artifacts, resources=resources,
        seed=cfg.seed, steps=cfg.steps, sampling_mode=cfg.sampling_mode,
        pointmap_source="provided_pytorch3d" if pointmaps else "native_moge",
        selected_frames=selections[-1], chunks=selections, occupancy=path.resolve(),
        occupied_voxels=int(occupancy.sum()), elapsed_seconds=time.perf_counter()-started, metrics=metrics)
    (output / "result.json").write_text(result.model_dump_json(indent=2)+"\n")
    return result


def main(manifest: str, output: str, method: str = "mvsam3d", checkpoint: str = "checkpoints/hf/pipeline.yaml",
         steps: int = 25, seed: int = 42, chunk_size: int = 8, topk: int = 8,
         sampling_mode: str = "flow", full_mesh: bool = False) -> None:
    """Run a JSON manifest using the same frames and masks as another evaluation."""
    cfg = chz.Blueprint(BaselineConfig).apply(dict(method=method, checkpoint=checkpoint,
        steps=steps, seed=seed, chunk_size=chunk_size, topk=topk, sampling_mode=sampling_mode, full_mesh=full_mesh)).make()
    sample = BaselineInput.model_validate_json(Path(manifest).read_text())
    print(reconstruct(sample, cfg, Path(output)).model_dump_json(indent=2))


if __name__ == "__main__":
    import fire
    fire.Fire(main)
