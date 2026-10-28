"""`eval`: sample a shape for every eval sample and write per-sample metrics to `<label>.jsonl`.

Runs: a trained run (directory, or run name under output_root; EMA weights of best.pt, else the
latest checkpoint), `zero-shot`
(C0: plain SAM 3D on the reference frame) or `best-frame` (C0b: plain SAM 3D on the frame of S
with the largest visible_frac; an oracle, because it uses GT visibility).
Seeds depend only on (clip_id, k, r), so all conditions see the same noise. Resumable.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from vidsam3d.data import reader
from vidsam3d.data.samples import ClipDataset, build_sample, frame_subset
from vidsam3d.data.targets import target_voxels
from vidsam3d.metrics import shape_metrics
from vidsam3d.models.adapter import build_adapter
from vidsam3d.models.encoders import build_kv_encoder
from vidsam3d.models.sam3d import build_sam3d
from vidsam3d.train import adapted_condition, reference_images
from vidsam3d.utils import (append_jsonl, copy_to_output, pick_device, read_jsonl, sample_seed,
                            setup_logging)

log = logging.getLogger("vidsam3d")
BASELINES = {"zero-shot": "c0", "best-frame": "c0b_oracle"}
MODEL_SECTIONS = ("adapter", "encoder", "target")   # taken from the run; the sampler is shared


def sample_id(info: dict) -> str:
    """Stable id of an eval sample."""
    return f"{info['clip_id']}/{info['k']}/{info['r']}"


def load_run(cfg: DictConfig, run: str, device) -> tuple[DictConfig, torch.nn.Module | None, str]:
    """(config, EMA adapter or None, name) for a run directory or a baseline name."""
    if run in BASELINES:
        return cfg, None, BASELINES[run]
    run_dir = Path(run) if Path(run).is_dir() else Path(cfg.paths.output_root) / run   # path or name
    assert (run_dir / "config.yaml").is_file(), f"no run at {run} or {run_dir}"
    run_cfg = OmegaConf.load(run_dir / "config.yaml")
    cfg = OmegaConf.merge(cfg, {k: run_cfg[k] for k in MODEL_SECTIONS},
                          {"data": {"crop": run_cfg.data.crop}})
    cfg.condition, cfg.run_name = run_cfg.condition, run_cfg.run_name
    ckpt_path = run_dir / "best.pt"
    if not ckpt_path.is_file():
        ckpt_path = run_dir / json.loads((run_dir / "latest.json").read_text())["checkpoint"]
    adapter = build_adapter(cfg.adapter).to(device).eval()
    adapter.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False)["ema"])
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
def evaluate(cfg: DictConfig, run: str, k: int | None = None, shuffle: bool = False,
             wrong_video: bool = False, split: str | None = None, limit: int | None = None) -> Path:
    """Evaluate one run or baseline; returns the path of its results JSONL."""
    setup_logging()
    device = pick_device(cfg.device)
    cfg, adapter, name = load_run(cfg, run, device)
    clip_len = int(k or cfg.eval.k)
    label = result_label(name, clip_len, int(cfg.eval.k), shuffle, wrong_video, adapter is not None)
    sam3d = build_sam3d(cfg, device)
    encoder = build_kv_encoder(cfg, sam3d, device) if adapter is not None else None
    ds = ClipDataset(cfg, split or cfg.data.eval_split, train=False, clip_len=clip_len)

    scratch_dir, output_dir = Path(cfg.paths.scratch_root) / "eval", Path(cfg.paths.output_root) / "eval"
    scratch_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    local, final = scratch_dir / f"{label}.jsonl", output_dir / f"{label}.jsonl"
    records = {r["sample_id"]: r for r in read_jsonl(final) + read_jsonl(local)}
    local.write_text("".join(json.dumps(r) + "\n" for r in records.values()))
    log.info("%s: %d of %d samples already done", label, len(records), len(ds))

    todo = [i for i in range(len(ds)) if sample_id(ds.index[i]) not in records]
    for n, i in enumerate(todo[:limit] if limit else todo, start=1):
        sample = ds[i]
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
        gt = torch.from_numpy(target_voxels(ds.voxels_file, info["asset_id"], cfg))
        metrics = shape_metrics(pred, gt, cfg.eval.metrics, seed=sample_seed(sample_id(info)))
        record = {"sample_id": sample_id(info), "label": label, "condition": cfg.condition if adapter
                  is not None else name, "run": name, "clip_len": clip_len, "shuffle_frames": shuffle,
                  "wrong_video": wrong_video, **{key: info[key] for key in (
                      "clip_id", "k", "r", "reference_rules", "visible_frac", "amodal_px",
                      "arc_span_deg", "num_objects", "asset_id")},
                  **({"video_from": info["video_from"]} if "video_from" in info else {}),
                  **extra, **metrics}
        append_jsonl(local, record)
        if n % cfg.eval.flush_every == 0:
            copy_to_output(local, output_dir)
            log.info("%s: %d / %d", label, len(records) + n, len(ds))
    copy_to_output(local, output_dir)
    log.info("wrote %s", final)
    return final
