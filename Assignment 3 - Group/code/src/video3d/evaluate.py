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
from contextlib import nullcontext
import logging
from datetime import datetime
from pathlib import Path

import torch
import chz
from video3d.config import read_config, config_dict
from video3d.config import ExperimentConfig

from video3d.data import reader, build_dataset, ground_truth
from video3d.data.schemas import EvaluationRecord
from video3d.data.samples import ClipDataset, build_sample, frame_subset
from video3d.metrics import shape_metrics
from video3d.data.controls import sample_id, shuffle_frames, wrong_video_partner, with_video_of, repeat_reference as repeat_reference
from video3d.models.adapter import build_adapter
from video3d.models.encoders import build_kv_encoder
from video3d.models.sam3d import build_sam3d
from video3d.train import adapted_condition, reference_images, sample_flow_loss
from video3d.utils import (append_jsonl, copy_to_output, pick_device, read_jsonl, sample_seed,
                            setup_logging)

log = logging.getLogger("video3d")
BASELINES = {"zero-shot": "c0", "best-frame": "c0b_oracle"}
MODEL_SECTIONS = ("adapter", "encoder", "target")   # taken from the run; the sampler is shared


def load_run(cfg: ExperimentConfig, run: str, device: typing.Any) -> tuple[ExperimentConfig, torch.nn.Module | None, str]:
    """(config, EMA adapter or None, name) for a run directory or a baseline name."""
    if run in BASELINES:
        return chz.replace(cfg, sam3d=chz.replace(cfg.sam3d, lora=False)), None, BASELINES[run]
    run_dir = Path(run) if Path(run).is_dir() else Path(cfg.paths.output_root) / run   # path or name
    assert (run_dir / "config.json").is_file(), f"no run at {run} or {run_dir}"
    run_cfg = read_config(run_dir / "config.json")
    cfg = chz.replace(cfg, adapter=run_cfg.adapter, encoder=run_cfg.encoder,
                      target=run_cfg.target, sam3d=run_cfg.sam3d,
                      data=chz.replace(cfg.data, crop=run_cfg.data.crop, source=run_cfg.data.source,
                                       video_input=run_cfg.data.video_input, frames=run_cfg.data.frames,
                                       frame_stride=run_cfg.data.frame_stride,
                                       num_subset_frames=run_cfg.data.num_subset_frames,
                                       prepared=run_cfg.data.prepared),
                      provenance=run_cfg.provenance,
                      condition=run_cfg.condition, run_name=run_cfg.run_name)
    ckpt_path = run_dir / "best.pt"
    if not ckpt_path.is_file():
        ckpt_path = run_dir / json.loads((run_dir / "latest.json").read_text())["checkpoint"]
    adapter = build_adapter(cfg.adapter).to(device).eval()
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    key = "ema" if cfg.eval.checkpoint_weights == "ema" else "adapter"
    extra_key = "extra_ema" if cfg.eval.checkpoint_weights == "ema" else "extra"
    adapter.load_state_dict(state[key])
    adapter.extra_ema_state = state.get(extra_key, {})
    log.info("loaded %s adapter from %s", cfg.eval.checkpoint_weights, ckpt_path)
    return cfg, adapter, cfg.run_name


def result_label(name: str, clip_len: int, default_len: int, shuffle: bool, wrong: bool,
                 adapter: bool) -> str:
    """e.g. c3_seed0, c3_seed0_k4, c3_seed0_shuffle; baselines do not depend on K."""
    label = name
    if adapter and clip_len != default_len:
        label += f"_k{clip_len}"
    return label + ("_shuffle" if shuffle else "") + ("_wrongvideo" if wrong else "")


def best_frame(ds: ClipDataset, info: dict) -> tuple[reader.Clip, int]:
    """C0b oracle: the frame of S (o = 0) where object k has the largest visible_frac."""
    clip = ds.clip(info["clip_id"])
    vis = reader.visibility(clip)[:, info["k"] - 1]
    subset = frame_subset(0, ds.data_cfg.num_subset_frames, ds.data_cfg.frame_stride)
    return clip, max(subset, key=lambda t: vis[t])


@torch.no_grad()
def evaluate(cfg: ExperimentConfig, run: str, k: int | None = None, shuffle: bool = False,
             wrong_video: bool = False, repeated_reference: bool = False, split: str | None = None, limit: int | None = None, loss_only: bool = False) -> Path:
    """Evaluate one run or baseline; returns the path of its results JSONL."""
    setup_logging()
    device = pick_device(cfg.device)
    cfg, adapter, name = load_run(cfg, run, device)
    clip_len = int(k or cfg.eval.k)
    label = result_label(name, clip_len, int(cfg.eval.k), shuffle, wrong_video, adapter is not None)
    label += "_repeated" if repeated_reference else ""
    label += "_raw" if adapter is not None and cfg.eval.checkpoint_weights == "current" else ""
    label += "_loss" if loss_only else ""
    sam3d = build_sam3d(cfg, device)
    encoder = build_kv_encoder(cfg, sam3d, device) if adapter is not None else None
    if adapter is not None:
        from video3d.checkpoint import extra_parameters, restore_extra
        restore_extra(extra_parameters(encoder, sam3d), adapter.extra_ema_state)
    if cfg.data.source == "navi" and run == "best-frame":
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
                     wrong_video=wrong_video, repeated_reference=repeated_reference, split=split,
                     loss_only=loss_only, flow_loss_protocol="shape-flow-fixed-grid-v1")
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
    logger = RunLogger(chz.replace(cfg, run_name=f"{label}-eval-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"))
    for n, i in enumerate(todo[:limit] if limit else todo, start=1):
        sample = ds[i]
        if repeated_reference:
            sample = repeat_reference(sample)
        if wrong_video:
            sample = with_video_of(sample, ds[wrong_video_partner(ds, i)])
        if shuffle:
            sample = shuffle_frames(sample)
        info = sample["info"]                    # ids and stratification of the eval sample
        extra = {}
        video_result = None
        loss_sample = sample
        if run == "best-frame":
            clip, r_best = best_frame(ds, info)
            loss_sample = build_sample(clip, info["k"], r_best, 0, 1, cfg.data)
            extra["oracle_frame"] = r_best
        context = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" and cfg.train.bf16 else nullcontext()
        with context:
            matched_loss = sample_flow_loss(sam3d, encoder, adapter, loss_sample, cfg, device)
        metrics = {"flow_loss": matched_loss}
        if not loss_only:
            if cfg.eval.video_diagnostics and adapter is not None:
                if shuffle or wrong_video or repeated_reference:
                    raise ValueError("Paired video diagnostics start from the unmodified correct sample")
                from video3d.diagnostics import evaluate_video, exposure_for_sample
                truth = torch.from_numpy(ground_truth(ds, info["asset_id"], cfg))
                other = ds[wrong_video_partner(ds, i)] if "wrong" in cfg.eval.video_controls else None
                video_result = evaluate_video(sample, other, truth, sam3d, encoder, adapter, cfg, device,
                                              exposure_for_sample(cfg, ds, info))
            elif adapter is not None:
                cond = adapted_condition(sam3d, encoder, adapter, [sample], cfg, device)
            else:
                ref = sample
                if run == "best-frame":
                    clip, r_best = best_frame(ds, info)
                    ref = build_sample(clip, info["k"], r_best, 0, 1, cfg.data)
                    extra["oracle_frame"] = r_best
                cond = sam3d.condition(*reference_images([ref], device))
            if video_result is None:
                z, _ = sam3d.sample(cond, [sample_seed(info["clip_id"], info["k"], info["r"])], cfg.sampler)
                pred = sam3d.decode_latent(z)[0].cpu()
            else:
                pred = video_result.prediction
                extra.update(video_diagnostics=video_result.diagnostics, video_controls=video_result.controls)
                metrics_log = {"eval/" + key: value for key, value in video_result.controls["correct"].items()
                               if isinstance(value, (int, float))}
                from video3d.diagnostics import scalar_diagnostics
                metrics_log.update(scalar_diagnostics(video_result.diagnostics))
                for mode, values in video_result.controls.items():
                    metrics_log.update({f"video_benefit/{mode}/{key}": value
                                        for key, value in values.get("delta_correct_minus_control", {}).items()})
                if logger.run is not None:
                    logger.run.log({**metrics_log, "eval/sample_index": len(records) + n})
            gt = torch.from_numpy(ground_truth(ds, info["asset_id"], cfg))
            metrics.update(shape_metrics(pred, gt, cfg.eval.metrics, seed=sample_seed(sample_id(info))))
            if info.get("visibility"):
                from video3d.metrics import visibility_metrics
                metrics.update(visibility_metrics(pred, info["visibility"]))
        record = {"sample_id": sample_id(info), "label": label, "condition": cfg.condition if adapter
                  is not None else name, "run": name, "clip_len": clip_len, "shuffle_frames": shuffle,
                  "wrong_video": wrong_video, **{key: info[key] for key in (
                      "clip_id", "k", "r", "reference_rules", "visible_frac", "amodal_px",
                      "arc_span_deg", "num_objects", "asset_id")},
                  **({"video_from": info["video_from"]} if "video_from" in info else {}),
                  "flow_loss_protocol": "shape-flow-fixed-grid-v1",
                  "flow_loss_tau_grid": list(cfg.val.tau_grid),
                  "flow_loss_seed": sample_seed("val", info["clip_id"], info["k"],
                                                extra.get("oracle_frame", info["r"])),
                  **extra, **metrics}
        record = EvaluationRecord.model_validate(record).model_dump()
        if not loss_only and len(records) + n <= cfg.logging.media_samples:
            media_dir = output_dir / "media" / label / sample_id(info).replace("/", "-")
            visibility_file = None
            visible_frames = None
            if cfg.data.source == "synthetic":
                from video3d.data.visibility import exposure_path, object_exposure
                visibility_file = exposure_path(cfg, info["clip_id"], info["k"])
                object_exposure(ds.clip(info["clip_id"]),info["k"],
                                reader.load_voxels(ds.voxels_file,info["asset_id"]),visibility_file)
                visible_frames = info["frame_ids"]
                if adapter is None or wrong_video:
                    visible_frames = [extra.get("oracle_frame",info["r"])]
                elif repeated_reference:
                    visible_frames = [info["r"]] * clip_len
            save_pair(sample, pred, gt, record, media_dir, logger.run,
                      visibility_file=visibility_file,frame_ids=visible_frames)
        if logger.run is not None and video_result is not None:
            from video3d.diagnostics import attention_table
            logger.run.log({"video/attention": attention_table([record])})
        logger.log({"eval/flow_loss": matched_loss})
        append_jsonl(local, record)
        if n % cfg.eval.flush_every == 0:
            copy_to_output(local, output_dir)
            log.info("%s: %d / %d", label, len(records) + n, len(ds))
    copy_to_output(local, output_dir)
    log.info("wrote %s", final)
    complete = read_jsonl(final)
    mean_loss = sum(row["flow_loss"] for row in complete) / len(complete) if complete else None
    summary = {"samples": len(complete), "clips": len({row["clip_id"] for row in complete}),
               "mean_flow_loss": mean_loss, "tau_grid": list(cfg.val.tau_grid),
               "protocol": "shape-flow-fixed-grid-v1", "weights": "pretrained" if adapter is None else cfg.eval.checkpoint_weights,
               "split": split or cfg.data.eval_split, "label": label}
    if cfg.eval.video_diagnostics:
        from video3d.diagnostics import summarize_video_records
        summary.update(summarize_video_records(complete))
    atomic_json(output_dir / f"{label}.summary.json", summary)
    if logger.run is not None:
        import wandb
        logger.run.summary.update(summary)
        if cfg.eval.video_diagnostics:
            logger.run.save(str(final), base_path=str(output_dir))
            logger.run.save(str(output_dir / f"{label}.summary.json"), base_path=str(output_dir))
        logger.run.log({"eval/paired_losses": wandb.Table(
            columns=["sample_id", "clip_id", "object", "reference", "flow_loss"],
            data=[[row["sample_id"], row["clip_id"], row["k"], row["r"], row["flow_loss"]]
                  for row in complete])})
    log.info("fixed-grid flow loss: %s", summary)
    logger.finish()
    return final
