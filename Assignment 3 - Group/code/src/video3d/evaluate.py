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
import logging
from pathlib import Path

import numpy as np
import torch
import chz
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


def load_run(cfg: ExperimentConfig, run: str, device: typing.Any) -> tuple[ExperimentConfig, torch.nn.Module | None, str]:
    """(config, EMA adapter or None, name) for a run directory or a baseline name."""
    if run in BASELINES:
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
    ckpt_path = run_dir / "best.pt"
    if not ckpt_path.is_file():
        ckpt_path = run_dir / json.loads((run_dir / "latest.json").read_text())["checkpoint"]
    adapter = build_adapter(cfg.adapter).to(device).eval()
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    adapter.load_state_dict(state["ema"])
    adapter.extra_ema_state = state.get("extra_ema", {})
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
             wrong_video: bool = False, repeated_reference: bool = False, split: str | None = None, limit: int | None = None) -> Path:
    """Evaluate one run or baseline; returns the path of its results JSONL."""
    setup_logging()
    device = pick_device(cfg.device)
    cfg, adapter, name = load_run(cfg, run, device)
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
    scratch_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    local, final = scratch_dir / f"{label}.jsonl", output_dir / f"{label}.jsonl"
    from video3d.storage import atomic_json
    signature = config_dict(cfg)
    from video3d.provenance import source_identity, data_identity
    signature.update(source_identity=source_identity(), data_identity=data_identity(cfg))
    signature.update(evaluation_run=run, clip_len=clip_len, shuffle=shuffle,
                     wrong_video=wrong_video, repeated_reference=repeated_reference, split=split)
    metadata = output_dir / f"{label}.meta.json"
    if metadata.exists() and json.loads(metadata.read_text()) != signature:
        raise ValueError("Evaluation settings changed; choose a new output directory")
    atomic_json(metadata, signature)
    records = {r["sample_id"]: r for r in read_jsonl(final) + read_jsonl(local)}
    local.write_text("".join(json.dumps(r) + "\n" for r in records.values()))
    log.info("%s: %d of %d samples already done", label, len(records), len(ds))

    todo = [i for i in range(len(ds)) if sample_id(ds.index[i]) not in records]
    from video3d.logging import RunLogger
    from video3d.media import save_pair
    logger = RunLogger(cfg)
    for n, i in enumerate(todo[:limit] if limit else todo, start=1):
        sample = ds[i]
        if repeated_reference:
            from video3d.evaluate import repeat_reference
            sample = repeat_reference(sample)
        if wrong_video:
            sample = with_video_of(sample, ds[wrong_video_partner(ds, i)])
        if shuffle:
            sample = shuffle_frames(sample)
        info = sample["info"]                    # ids and stratification of the eval sample
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
        z, _ = sam3d.sample(cond, [sample_seed(info["clip_id"], info["k"], info["r"])], cfg.sampler)
        pred = sam3d.decode_latent(z)[0].cpu()
        gt = torch.from_numpy(ground_truth(ds, info["asset_id"], cfg))
        metrics = shape_metrics(pred, gt, cfg.eval.metrics, seed=sample_seed(sample_id(info)))
        if info.get("visibility"):
            from video3d.metrics import visibility_metrics
            metrics.update(visibility_metrics(pred, info["visibility"]))
        record = {"sample_id": sample_id(info), "label": label, "condition": cfg.condition if adapter
                  is not None else name, "run": name, "clip_len": clip_len, "shuffle_frames": shuffle,
                  "wrong_video": wrong_video, **{key: info[key] for key in (
                      "clip_id", "k", "r", "reference_rules", "visible_frac", "amodal_px",
                      "arc_span_deg", "num_objects", "asset_id")},
                  **({"video_from": info["video_from"]} if "video_from" in info else {}),
                  **extra, **metrics}
        record = EvaluationRecord.model_validate(record).model_dump()
        if len(records) + n <= cfg.logging.media_samples:
            media_dir = output_dir / "media" / label / sample_id(info).replace("/", "-")
            save_pair(sample, pred, gt, record, media_dir, logger.run)
        append_jsonl(local, record)
        if n % cfg.eval.flush_every == 0:
            copy_to_output(local, output_dir)
            log.info("%s: %d / %d", label, len(records) + n, len(ds))
    copy_to_output(local, output_dir)
    log.info("wrote %s", final)
    logger.finish()
    return final


def repeat_reference(sample: dict) -> dict:
    inputs = sample["inputs"]
    count = len(inputs["frames"])
    pos = inputs["ref_pos"]
    changed = {**inputs, "frames": inputs["frames"][pos:pos+1].expand(count, -1, -1, -1),
               "frame_masks": inputs["frame_masks"][pos:pos+1].expand(count, -1, -1),
               "dt": torch.zeros_like(inputs["dt"])}
    return {**sample, "inputs": changed}
