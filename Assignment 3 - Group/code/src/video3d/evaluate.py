"""`eval`: sample a shape for every eval sample and write per-sample metrics to `<label>.jsonl`.

Runs: a trained run (directory, or run name under output_root; EMA weights of best.pt, else the
latest checkpoint), `zero-shot`
(C0: plain SAM 3D on the reference frame) or `best-frame` (C0b: plain SAM 3D on the frame of S
with the largest visible_frac; an oracle, because it uses GT visibility).
Seeds depend only on (clip_id, k, r), so all conditions see the same noise. Resumable.
"""

from __future__ import annotations
import typing

import json
import hashlib
import logging
import time
from pathlib import Path

import numpy as np
import torch
import chz
from tqdm.auto import tqdm
from video3d.config import read_config, config_dict
from video3d.config import ExperimentConfig

from video3d.data import reader, build_dataset, ground_truth
from video3d.data.schemas import EvaluationRecord
from video3d.data.samples import ClipDataset, build_sample, frame_subset
from video3d.metrics import shape_metrics
from video3d.models.adapter import build_adapter
from video3d.models.encoders import build_kv_encoder
from video3d.models.sam3d import build_sam3d
from video3d.train import adapted_condition, reference_images
from video3d.utils import (append_jsonl, copy_to_output, pick_device, read_jsonl, sample_seed,
                            setup_logging)

log = logging.getLogger("video3d")
BASELINES = {"zero-shot": "c0", "best-frame": "c0b_oracle"}
MODEL_SECTIONS = ("adapter", "encoder", "target")   # taken from the run; the sampler is shared


def sample_id(info: dict) -> str:
    """Stable id of an eval sample."""
    return f"{info['clip_id']}/{info['k']}/{info['r']}"


def load_run(cfg: ExperimentConfig, run: str, device: typing.Any,
             checkpoint: str | None = None) -> tuple[ExperimentConfig, torch.nn.Module | None, str]:
    """(config, EMA adapter or None, name) for a run directory or a baseline name."""
    if run in BASELINES:
        if cfg.eval.evaluation_id:
            cfg = chz.replace(cfg, run_name=cfg.eval.evaluation_id)
        return cfg, None, BASELINES[run]
    run_dir = Path(run) if Path(run).is_dir() else Path(cfg.paths.output_root) / run   # path or name
    assert (run_dir / "config.json").is_file(), f"no run at {run} or {run_dir}"
    run_cfg = read_config(run_dir / "config.json")
    cfg = chz.replace(cfg, adapter=run_cfg.adapter, encoder=run_cfg.encoder,
                      target=run_cfg.target, sam3d=run_cfg.sam3d,
                      data=chz.replace(cfg.data, crop=run_cfg.data.crop,
                                       video_input=run_cfg.data.video_input, frames=run_cfg.data.frames,
                                       frame_stride=run_cfg.data.frame_stride,
                                       num_subset_frames=run_cfg.data.num_subset_frames),
                      provenance=run_cfg.provenance,
                      condition=run_cfg.condition, run_name=run_cfg.run_name)
    ckpt_path = Path(checkpoint) if checkpoint else run_dir / "best.pt"
    if checkpoint is None and not ckpt_path.is_file():
        ckpt_path = run_dir / json.loads((run_dir / "latest.json").read_text())["checkpoint"]
    adapter = build_adapter(cfg.adapter).to(device).eval()
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    adapter.load_state_dict(state["ema"])
    adapter.extra_ema_state = state.get("extra_ema", {})
    if cfg.eval.evaluation_id:
        cfg = chz.replace(cfg, run_name=cfg.eval.evaluation_id)
    log.info("loaded EMA adapter from %s", ckpt_path)
    return cfg, adapter, cfg.run_name


def result_label(name: str, clip_len: int, default_len: int, shuffle: bool, wrong: bool,
                 adapter: bool) -> str:
    """e.g. c3_seed0, c3_seed0_k4, c3_seed0_shuffle; baselines do not depend on K."""
    label = name
    if adapter and clip_len != default_len:
        label += f"_k{clip_len}"
    return label + ("_shuffle" if shuffle else "") + ("_wrongvideo" if wrong else "")


def shuffle_frames(sample: dict) -> dict:
    """H3 control: permute the clip frames (seeded by the sample id); Δt slots stay in place."""
    x, info = sample["inputs"], sample["info"]
    K = x["frames"].shape[0]
    rng = np.random.default_rng(sample_seed(sample_id(info), "shuffle"))
    perm = torch.from_numpy(rng.permutation(K))
    ref_pos = int((perm == x["ref_pos"]).nonzero()[0, 0])
    inputs = {**x, "frames": x["frames"][perm], "frame_masks": x["frame_masks"][perm], "ref_pos": ref_pos}
    return {"inputs": inputs, "info": {**info, "frame_ids": [info["frame_ids"][i] for i in perm]}}


def wrong_video_partner(ds: ClipDataset, i: int) -> int:
    """T7: a deterministic sample from a different clip (about half the dataset away)."""
    clip_id = ds.index[i]["clip_id"]
    for step in range(len(ds)):
        j = (i + len(ds) // 2 + step) % len(ds)
        if ds.index[j]["clip_id"] != clip_id:
            return j
    raise AssertionError("wrong-video control needs at least two clips")


def with_video_of(sample: dict, other: dict) -> dict:
    """Keep the SAM 3D reference inputs; take the clip (frames, masks, Δt) from another sample."""
    keys = ("frames", "frame_masks", "dt", "ref_pos")
    inputs = {**sample["inputs"], **{k: other["inputs"][k] for k in keys}}
    return {"inputs": inputs, "info": {**sample["info"], "video_from": sample_id(other["info"])}}


def best_frame(ds: ClipDataset, info: dict) -> tuple[reader.Clip, int]:
    """C0b oracle: the frame of S (o = 0) where object k has the largest visible_frac."""
    clip = ds.clip(info["clip_id"])
    vis = reader.visibility(clip)[:, info["k"] - 1]
    subset = frame_subset(0, ds.data_cfg.num_subset_frames, ds.data_cfg.frame_stride)
    return clip, max(subset, key=lambda t: vis[t])


@torch.no_grad()
def evaluate(cfg: ExperimentConfig, run: str, k: int | None = None, shuffle: bool = False,
             wrong_video: bool = False, repeated_reference: bool = False, split: str | None = None,
             limit: int | None = None, checkpoint: str | None = None,
             epoch: int | None = None, trigger: str = "ad_hoc") -> Path:
    """Evaluate one run or baseline; returns the path of its results JSONL."""
    setup_logging()
    device = pick_device(cfg.device)
    cfg, adapter, name = load_run(cfg, run, device, checkpoint)
    clip_len = int(k or cfg.eval.k)
    label = result_label(name, clip_len, int(cfg.eval.k), shuffle, wrong_video, adapter is not None)
    label += "_repeated" if repeated_reference else ""
    sam3d = build_sam3d(cfg, device)
    encoder = build_kv_encoder(cfg, sam3d, device) if adapter is not None else None
    if adapter is not None:
        from video3d.checkpoint import extra_parameters, restore_extra
        restore_extra(extra_parameters(encoder, sam3d), adapter.extra_ema_state)
    if cfg.data.source != "synthetic" and run == "best-frame":
        raise ValueError("The visibility-based oracle requires synthetic visibility metadata")
    ds = build_dataset(cfg, split or cfg.data.eval_split, train=False, clip_len=clip_len)

    scratch_dir, output_dir = Path(cfg.paths.scratch_root) / "eval", Path(cfg.paths.output_root) / "eval"
    if cfg.eval.evaluation_id:
        scratch_dir = Path(cfg.paths.scratch_root) / "evaluations" / cfg.eval.evaluation_id
        output_dir = Path(cfg.paths.output_root) / "evaluations" / cfg.eval.evaluation_id
    scratch_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    local, final = scratch_dir / f"{label}.jsonl", output_dir / f"{label}.jsonl"
    from video3d.storage import atomic_json
    signature = config_dict(cfg)
    from video3d.provenance import source_identity, data_identity
    signature.update(source_identity=source_identity(), data_identity=data_identity(cfg))
    signature.update(evaluation_run=run, clip_len=clip_len, shuffle=shuffle,
                     wrong_video=wrong_video, repeated_reference=repeated_reference, split=split)
    if checkpoint:
        with Path(checkpoint).open('rb') as handle:
            checkpoint_hash = hashlib.file_digest(handle, 'sha256').hexdigest()
        signature.update(checkpoint=str(Path(checkpoint).resolve()), checkpoint_sha256=checkpoint_hash)
    signature.update(epoch=epoch, trigger=trigger, weights='EMA',
                     output_scope='refined_mesh_and_coarse' if cfg.eval.refined_mesh else 'coarse_stage1')
    metadata = output_dir / f"{label}.meta.json"
    if metadata.exists() and json.loads(metadata.read_text()) != signature:
        raise ValueError("Evaluation settings changed; choose a new output directory")
    atomic_json(metadata, signature)
    if cfg.eval.evaluation_id:
        atomic_json(output_dir / "evaluation.json", signature)
    records = {r["sample_id"]: r for r in read_jsonl(final) + read_jsonl(local)}
    local.write_text("".join(json.dumps(r) + "\n" for r in records.values()))
    log.info("%s: %d of %d samples already done", label, len(records), len(ds))

    todo = [i for i in range(len(ds)) if sample_id(ds.index[i]) not in records]
    from video3d.logging import RunLogger
    from video3d.media import save_pair
    logger = RunLogger(cfg)
    for n, i in enumerate(tqdm(todo[:limit] if limit else todo, desc=f"Evaluate {label}", unit="video"), start=1):
        inference_started = time.perf_counter()
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        sample = ds[i]
        if repeated_reference:
            from video3d.evaluate import repeat_reference
            sample = repeat_reference(sample)
        if wrong_video:
            sample = with_video_of(sample, ds[wrong_video_partner(ds, i)])
        if shuffle:
            sample = shuffle_frames(sample)
        info = sample["info"]                    # ids and stratification of the eval sample
        if cfg.eval.refined_mesh:
            if cfg.fake_sam3d or cfg.data.source != 'gso':
                raise ValueError('Heavy refined evaluation requires real SAM3D and audited GSO data')
            info['visibility'] = str(ds.root / 'visibility' / f"{info['asset_id']}.npz")
        extra = {}
        if adapter is not None:
            cond = adapted_condition(sam3d, encoder, adapter, [sample], cfg, device)
        else:
            ref = sample
            if run == "best-frame":
                clip, r_best = best_frame(ds, info)
                ref = build_sample(clip, info["k"], r_best, 0, 1, cfg.data)
                extra["oracle_frame"] = r_best
            cond = sam3d.condition(*reference_images([ref], device))
        seed = sample_seed(info["clip_id"], info["k"], info["r"])
        z, layout = sam3d.sample(cond, [seed], cfg.sampler)
        pred = sam3d.decode_latent(z)[0].cpu()
        gt = torch.from_numpy(ground_truth(ds, info["asset_id"], cfg))
        mesh_metrics: dict[str, typing.Any] = {}
        if cfg.eval.refined_mesh and bool(pred.any()):
            import shutil
            from video3d.media import save_refined_prediction
            from video3d.mesh_analysis import analyze
            refined_dir = output_dir / "samples" / sample_id(info).replace("/", "-")
            refined = sam3d.refine(z, layout, *reference_images([sample], device), seed, cfg.eval.refinement_steps)
            save_refined_prediction(refined, refined_dir)
            # Match cached baselines: score the final pruned/downsampled native coordinates.
            coords = refined['coords'].detach().cpu().long()[:, 1:]
            pred = torch.zeros_like(pred)
            pred[coords[:, 0], coords[:, 1], coords[:, 2]] = True
            row = ds.rows[i]
            shutil.copy2(ds.root / row.mesh, refined_dir / "ground-truth.glb")
            if row.cameras is not None:
                shutil.copy2(ds.root / row.cameras, refined_dir / "cameras.json")
            atomic_json(refined_dir / "protocol.json", {
                "dataset": cfg.data.source, "frame_ids": info["frame_ids"],
                "camera_registration": "known_render_poses" if row.cameras else "unverified",
                "visibility_cache": info["visibility"], "seed": seed})
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            inference_seconds = time.perf_counter() - inference_started
            mesh_metrics = analyze(str(refined_dir), samples=cfg.eval.surface_samples)
        else:
            inference_seconds = time.perf_counter() - inference_started
            if cfg.eval.native_coarse and bool(pred.any()):
                sparse = sam3d.sparse_structure(z, layout)
                coords = sparse['coords'].detach().cpu().long()[:, 1:]
                pred = torch.zeros_like(pred)
                pred[coords[:, 0], coords[:, 1], coords[:, 2]] = True
        metrics = shape_metrics(pred, gt, cfg.eval.metrics, seed=sample_seed(sample_id(info)))
        if info.get('visibility'):
            from video3d.metrics import visibility_metrics
            metrics.update(visibility_metrics(pred, info['visibility']))
        if cfg.eval.refined_mesh:
            metrics['mesh_surface'] = mesh_metrics or None
        resources = {'inference_seconds':inference_seconds,
                     'sample_seconds':time.perf_counter()-inference_started}
        if device.type == 'cuda':
            resources.update(cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                             cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                             gpu=torch.cuda.get_device_name(device))
        record = {"sample_id": sample_id(info), "label": label, "condition": cfg.condition if adapter
                  is not None else name, "run": name, "clip_len": clip_len, "shuffle_frames": shuffle,
                  "wrong_video": wrong_video, **{key: info[key] for key in (
                      "clip_id", "k", "r", "reference_rules", "visible_frac", "amodal_px",
                      "arc_span_deg", "num_objects", "asset_id")},
                  **({"video_from": info["video_from"]} if "video_from" in info else {}),
                  **extra, **metrics}
        record.update(evaluation_id=cfg.eval.evaluation_id, epoch=epoch, trigger=trigger,
                      output_scope='refined_mesh_and_coarse' if cfg.eval.refined_mesh else 'coarse_stage1',
                      sampler=chz.asdict(cfg.sampler), refinement_steps=cfg.eval.refinement_steps if cfg.eval.refined_mesh else None,
                      frame_ids=info['frame_ids'], seed=seed)
        record.update(resources=resources)
        if cfg.eval.refined_mesh:
            record['refinement_status'] = 'complete' if mesh_metrics else 'empty_stage1'
        record = EvaluationRecord.model_validate(record).model_dump()
        logger.log_evaluation(record, len(records) + n)
        from video3d.storage import atomic_torch
        atomic_torch(output_dir / "predictions" / label / (sample_id(info).replace("/", "-") + ".pt"),
                     {"sample_id": sample_id(info), "latent": z[0].float().cpu(),
                      'layout':{key:value.cpu() for key,value in layout.items()},
                      "prediction": pred, "ground_truth": gt, "metrics": record})
        if len(records) + n <= cfg.logging.media_samples:
            media_dir = output_dir / "media" / label / sample_id(info).replace("/", "-")
            save_pair(sample, pred, gt, record, media_dir, logger.run)
        append_jsonl(local, record)
        if n % cfg.eval.flush_every == 0:
            copy_to_output(local, output_dir)
            log.info("%s: %d / %d", label, len(records) + n, len(ds))
    copy_to_output(local, output_dir)
    log.info("wrote %s", final)
    logger.finish_evaluation(read_jsonl(final), output_dir, label, len(ds))
    logger.finish()
    return final


def heavy_epoch_config(cfg: ExperimentConfig, run_dir: Path, epoch: int) -> ExperimentConfig:
    """Shared scheduled/ad-hoc heavy protocol, isolated from quick validation outputs."""
    from video3d.config import MetricsConfig, SamplerConfig
    if not cfg.val.heavy_data_root:
        raise ValueError("Set val.heavy_data_root to the audited GSO manifest directory")
    evaluation_id = f"{cfg.run_name}-epoch{epoch:02d}-heavy25"
    return chz.replace(cfg, sampler=SamplerConfig(mode='flow', steps=25),
        paths=chz.replace(cfg.paths, data_root=cfg.val.heavy_data_root,
                          output_root=str(run_dir.parent)),
        data=chz.replace(cfg.data, source='gso', eval_split='eval'),
        eval=chz.replace(cfg.eval, refined_mesh=True, native_coarse=True, refinement_steps=25,
                         metrics=MetricsConfig(), evaluation_id=evaluation_id),
        logging=chz.replace(cfg.logging, media_samples=200))


def heavy_epoch_selected(epochs: list[int], epoch: int, final_epoch: int) -> bool:
    return epoch in epochs or (-1 in epochs and epoch == final_epoch)


def evaluate_training_epoch(cfg: ExperimentConfig, run_dir: Path, epoch: int) -> None:
    """Run the same evaluator in a subprocess so W&B and Stage-2 memory are isolated."""
    import subprocess
    import sys
    from video3d.storage import atomic_json
    heavy = heavy_epoch_config(cfg, run_dir, epoch)
    config_file = run_dir / f'heavy-eval-epoch-{epoch:02d}.json'
    atomic_json(config_file, config_dict(heavy))
    subprocess.run([sys.executable, '-m', 'video3d', 'eval-json', str(config_file), str(run_dir),
                    f'--checkpoint={run_dir / f"epoch_{epoch:02d}.pt"}',
                    f'--epoch={epoch}', '--trigger=scheduled'], check=True)


def evaluate_full_validation(cfg: ExperimentConfig, run_dir: Path, epoch: int) -> None:
    """Full held-out training-distribution split, not the 50-object monitoring panel."""
    import subprocess
    import sys
    from video3d.config import MetricsConfig, SamplerConfig
    from video3d.storage import atomic_json
    evaluation_id = f'{cfg.run_name}-epoch{epoch:02d}-fullval-flow25'
    full = chz.replace(cfg, sampler=SamplerConfig(mode='flow', steps=25),
        data=chz.replace(cfg.data, source='synthetic', eval_split='val'),
        eval=chz.replace(cfg.eval, refined_mesh=False, native_coarse=True,
                         metrics=MetricsConfig(), evaluation_id=evaluation_id),
        logging=chz.replace(cfg.logging, media_samples=1306))
    config_file = run_dir / f'full-validation-epoch-{epoch:02d}.json'
    atomic_json(config_file, config_dict(full))
    subprocess.run([sys.executable, '-m', 'video3d', 'eval-json', str(config_file), str(run_dir),
                    f'--checkpoint={run_dir / f"epoch_{epoch:02d}.pt"}',
                    f'--epoch={epoch}', '--trigger=scheduled'], check=True)


def repeat_reference(sample: dict) -> dict:
    inputs = sample["inputs"]
    count = len(inputs["frames"])
    pos = inputs["ref_pos"]
    changed = {**inputs, "frames": inputs["frames"][pos:pos+1].expand(count, -1, -1, -1),
               "frame_masks": inputs["frame_masks"][pos:pos+1].expand(count, -1, -1),
               "dt": torch.zeros_like(inputs["dt"])}
    return {**sample, "inputs": changed}
