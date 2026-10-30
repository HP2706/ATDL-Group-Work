"""Use the existing dataset selection, sample IDs, seeds, metrics and result schema."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import chz
import numpy as np
from PIL import Image, ImageOps

from video3d.baselines.api import BaselineConfig, BaselineInput, load_pipeline, reconstruct
from video3d.config import read_config
from video3d.storage import atomic_json
from video3d.data import build_dataset, ground_truth, reader
from video3d.data.schemas import EvaluationRecord
from video3d.metrics import shape_metrics, visibility_metrics
from video3d.utils import sample_seed
from video3d.evaluate import sample_id


def export_sample(dataset: Any, index: int, source: str, directory: Path) -> tuple[BaselineInput, dict[str, Any]]:
    sample = dataset[index]
    info = sample["info"]
    frame_ids = info["frame_ids"]
    if source in {"gso", "omniobject3d"}:
        from video3d.data.scanned import read_rgb, read_mask
        row = dataset.rows[index]
        images = [read_rgb(dataset.root / row.frames[t]) for t in frame_ids]
        masks = [read_mask(dataset.root / row.masks[t], dataset.cfg.data.mask_threshold) for t in frame_ids]
        info["visibility"] = str(dataset.root / "visibility" / (row.object_id+".npz"))
    elif source == "navi":
        row = dataset.rows[index]
        images = [np.asarray(ImageOps.exif_transpose(Image.open(row.frames[t])).convert("RGB")) for t in frame_ids]
        masks = [np.asarray(Image.open(row.masks[t]).convert("L")) > 0 for t in frame_ids]
    else:
        clip = dataset.clip(info["clip_id"])
        images = [reader.load_rgb(clip, t) for t in frame_ids]
        masks = [reader.load_visible_mask(clip, info["k"], t) for t in frame_ids]
    directory.mkdir(parents=True, exist_ok=True)
    frames, mask_paths = [], []
    for position, (image, mask) in enumerate(zip(images, masks, strict=True)):
        frame, mask_path = directory / f"rgb-{position:04d}.png", directory / f"mask-{position:04d}.png"
        Image.fromarray(image).save(frame)
        Image.fromarray(mask.astype(np.uint8)*255).save(mask_path)
        frames.append(frame.resolve())
        mask_paths.append(mask_path.resolve())
    identity = sample_id(info)
    manifest = BaselineInput(sample_id=identity, frames=frames, masks=mask_paths,
                             reference_index=frame_ids.index(info['r']))
    (directory / "manifest.json").write_text(manifest.model_dump_json(indent=2)+"\n")
    return manifest, info


def evaluate(config: str, method: str, output: str, split: str | None = None,
             limit: int | None = None, k: int | None = None, chunk_size: int = 8, topk: int = 8, full_mesh: bool = False, resume: bool = True) -> str:
    cfg = read_config(Path(config))
    cfg = chz.replace(cfg, run_name=f'{cfg.run_name}-{method}')
    baseline = chz.Blueprint(BaselineConfig).apply(dict(method=method,
        checkpoint=str(Path(cfg.paths.sam3d_checkpoint_dir) / "pipeline.yaml"), steps=cfg.sampler.steps,
        sampling_mode=cfg.sampler.mode, chunk_size=chunk_size, topk=topk, full_mesh=full_mesh)).make()
    dataset = build_dataset(cfg, split or cfg.data.eval_split, train=False, clip_len=k or cfg.eval.k)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    signature = {"config": json.loads(Path(config).read_text()), "baseline": chz.asdict(baseline),
                 "split": split or cfg.data.eval_split, "frames": k or cfg.eval.k}
    metadata = root / "protocol.json"
    if metadata.exists() and json.loads(metadata.read_text()) != signature:
        raise ValueError("Output directory belongs to a different evaluation protocol")
    metadata.write_text(json.dumps(signature, indent=2)+"\n")
    atomic_json(root / 'evaluation.json', signature)
    atomic_json(root / 'evaluation.meta.json', signature)
    from video3d.logging import RunLogger
    logger = RunLogger(cfg)
    load_started = time.perf_counter()
    pipeline = load_pipeline(baseline)
    atomic_json(root / "runtime.json", {"model_load_seconds": time.perf_counter()-load_started})
    records = []
    for index in range(min(len(dataset), limit) if limit is not None else len(dataset)):
        directory = root / f"sample-{index:06d}"
        completed = directory / "evaluation-record.json"
        if resume and completed.is_file():
            records.append(json.loads(completed.read_text()))
            continue
        sample, info = export_sample(dataset, index, cfg.data.source, directory / "inputs")
        atomic_json(root / "progress.json", {"completed": len(records), "sample_id": sample.sample_id,
                    "index": index, "phase": "reconstructing"})
        run_cfg = chz.replace(baseline, seed=sample_seed(info["clip_id"], info["k"], info["r"]))
        result = reconstruct(sample, run_cfg, directory, pipeline=pipeline)
        metrics_started = time.perf_counter()
        prediction = np.load(result.occupancy)
        metrics = shape_metrics(prediction, ground_truth(dataset, info["asset_id"], cfg),
                                cfg.eval.metrics, seed=sample_seed(sample.sample_id))
        if info.get("visibility"):
            import torch
            metrics.update(visibility_metrics(torch.from_numpy(prediction), info["visibility"]))
        if cfg.data.source in {"gso", "omniobject3d"} and full_mesh:
            import shutil
            from video3d.baselines.analysis import analyze
            row = dataset.rows[index]
            shutil.copy2(dataset.root / row.mesh, directory / "ground-truth.glb")
            if row.cameras is not None:
                shutil.copy2(dataset.root / row.cameras, directory / "cameras.json")
            protocol = {"dataset": row.dataset, "frame_ids": info["frame_ids"],
                "camera_registration": "known_render_poses" if row.cameras is not None else "unverified",
                "visibility_cache": info["visibility"], "seed": run_cfg.seed}
            (directory / "protocol.json").write_text(json.dumps(protocol, indent=2))
            metrics["mesh_surface"] = analyze(str(directory))
        result.resources["metrics_seconds"] = time.perf_counter()-metrics_started
        metrics["flow_loss"] = None
        metrics["flow_loss_status"] = "not_computed: this runner scores generated geometry; fixed-grid training flow loss requires a separately specified conditioning objective"
        result.metrics = metrics
        (directory / "result.json").write_text(result.model_dump_json(indent=2)+"\n")
        record = EvaluationRecord.model_validate({
            **info, **metrics, "sample_id": sample.sample_id, "label": method+("_refined" if full_mesh else "_coarse"),
            "run": cfg.run_name, 'evaluation_id':cfg.run_name,
            'sampler':chz.asdict(cfg.sampler), 'output_scope':result.scope,
            "backend": "sam3d", "baseline": result.model_dump(mode="json")})
        logger.log_evaluation(record.model_dump(), len(records)+1)
        atomic_json(completed, record.model_dump(mode="json"))
        records.append(record.model_dump())
        atomic_json(root / "progress.json", {"completed": len(records), "sample_id": sample.sample_id,
                    "index": index, "phase": "saved"})
        print(json.dumps({"completed": len(records), "sample_id": sample.sample_id}), flush=True)
        (root / "evaluation.jsonl").write_text("".join(json.dumps(row)+"\n" for row in records))
    (root / "evaluation.jsonl").write_text("".join(json.dumps(row)+"\n" for row in records))
    logger.finish_evaluation(records, root, 'evaluation', len(dataset))
    logger.finish()
    return str(root / "evaluation.jsonl")
