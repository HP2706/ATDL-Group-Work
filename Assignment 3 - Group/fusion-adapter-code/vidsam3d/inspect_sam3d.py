"""GPU diagnostics of the real SAM 3D: `inspect-sam3d` (runtime internals) and `frame-test` (T4)."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from vidsam3d.data import reader
from vidsam3d.data.samples import ClipDataset
from vidsam3d.data.targets import apply_canonical_permutation, cube_rotations, fill_voxels
from vidsam3d.metrics import voxel_iou
from vidsam3d.models.sam3d import Sam3DGeometry
from vidsam3d.utils import pick_device, sample_seed, setup_logging

log = logging.getLogger("vidsam3d")
DOC = Path(__file__).resolve().parent.parent / "docs" / "sam3d_internals.md"
BEGIN, END = "<!-- RUNTIME:BEGIN", "<!-- RUNTIME:END -->"


def first_eval_sample(cfg: DictConfig) -> dict:
    """The middle-frame sample of the first eval object."""
    return ClipDataset(cfg, cfg.data.eval_split, train=False)[0]


def runtime_values(model: Sam3DGeometry, sample: dict) -> dict:
    """Everything the wrapper depends on, read from the loaded pipeline."""
    p, gen, emb, bb = model.pipeline, model.generator, model.embedder, model.backbone
    cond = model.condition(sample["inputs"]["ref_rgb"][None].cuda(), sample["inputs"]["ref_mask"][None].cuda())
    sets = {name: list(c.shape[1:]) for name, c in cond.items()}
    gcfg = OmegaConf.load(Path(p.workspace_dir) / model.pipeline_cfg.ss_generator_config_path)
    values = {
        "1 condition sets (name: [tokens, width])": sets,
        "1 embedders": [(type(e).__name__, getattr(e, "embed_dim", None), [list(x) for x in info])
                        for e, info in emb.embedder_list],
        "1 positional groups (modality embedding)": dict(emb.positional_embed_map),
        "1 ss_condition_input_mapping": list(p.ss_condition_input_mapping),
        "2 condition computed once before the generator": True,
        "3 unconditional_handling": gen.reverse_fn.unconditional_handling,
        "3 force_zeros_cond": bool(getattr(bb, "force_zeros_cond", False)),
        "3 cfg strength / interval / strength_pm": [p.ss_cfg_strength, list(p.ss_cfg_interval), p.ss_cfg_strength_pm],
        "3 reverse_fn": type(gen.reverse_fn).__name__,
        "4 generator": type(gen).__name__,
        "4 is_shortcut_model": bool(bb.is_shortcut_model),
        "4 inference_steps / rescale_t / time_scale": [p.ss_inference_steps, gen.rescale_t, gen.time_scale],
        "5 training_time_sampler_fn": repr(gen.training_time_sampler_fn),
        "5 yaml time sampler": OmegaConf.to_container(gcfg["module"]["generator"]["backbone"]).get(
            "training_time_sampler_fn", "default (logit-normal(0, 1))"),
        "6 shape latent shape": list(model.latent_shape),
        "7 layout latents": {k: list(v) for k, v in model.layout_shape.items()},
        "7 latent_share_transformer": dict(bb.latent_share_transformer),
        "7 protected modalities": list(bb.blocks[0].self_attn.protect_modality_list),
        "8 encoder in_channels": getattr(model.encoder, "in_channels", None),
        "10 pose_target_convention": gcfg["module"].get("pose_target_convention"),
        "dtype / shape_model_dtype": [str(p.dtype), str(model.dtype)],
    }
    values.update(numerical_checks(model, cond))
    rgb_sets = [n for n in cond if re.search(r"(image|rgb)", n) and "mask" not in n and "pointmap" not in n]
    values["suggested adapter.query_sets (check!)"] = rgb_sets
    return values


@torch.no_grad()
def numerical_checks(model: Sam3DGeometry, cond: dict) -> dict:
    """Layout independence of v_shape, zero null branch, determinism of the condition path."""
    device = next(iter(cond.values())).device
    g = torch.Generator(device).manual_seed(0)
    x = torch.randn((1, *model.latent_shape), generator=g, device=device)
    tau = torch.full((1,), 0.5, device=device)
    lay = lambda: {k: torch.randn((1, *s), generator=g, device=device) for k, s in model.layout_shape.items()}
    with model._autocast():
        v1, _ = model.velocity(x, tau, cond, lay())
        v2, _ = model.velocity(x, tau, cond, lay())
        tokens = torch.cat(list(cond.values()), 1)
        kw = {"d": torch.zeros(1, device=device)} if model.backbone.is_shortcut_model else {}
        latents = {"shape": x, **lay()}
        t = tau * model.generator.time_scale
        u1 = model.backbone(latents, t, tokens, cfg=True, **kw)["shape"]
        u2 = model.backbone(latents, t, torch.randn_like(tokens), cfg=True, **kw)["shape"]
    return {"7 max |v_shape(layout a) - v_shape(layout b)|": float((v1 - v2).abs().max()),
            "3 max |null branch(cond) - null branch(random cond)|": float((u1 - u2).abs().max())}


def write_doc(values: dict) -> None:
    """Replace the RUNTIME section of docs/sam3d_internals.md."""
    lines = [f"{BEGIN} (written by `python -m vidsam3d inspect-sam3d`; do not edit by hand) -->",
             "## Runtime values", "", "| Item | Value |", "|---|---|"]
    lines += [f"| {k} | `{json.dumps(v, default=str)}` |" for k, v in values.items()]
    text = DOC.read_text()
    start, end = text.index(BEGIN), text.index(END)
    DOC.write_text(text[:start] + "\n".join(lines) + "\n" + text[end:])


def inspect_sam3d(cfg: DictConfig) -> dict:
    """Load SAM 3D, print its runtime internals and write them to docs and output_root."""
    setup_logging()
    model = Sam3DGeometry(cfg, pick_device(cfg.device))
    values = runtime_values(model, first_eval_sample(cfg))
    for k, v in values.items():
        print(f"{k:55s} {v}")
    write_doc(values)
    out = Path(cfg.paths.output_root) / "inspect_sam3d.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(values, indent=1, default=str))
    log.info("wrote %s and the runtime section of %s", out, DOC)
    return values


@torch.no_grad()
def frame_test(cfg: DictConfig, num_objects: int = 30, min_visible: float = 0.9,
               model: Sam3DGeometry | None = None) -> dict:
    """T4: zero-shot shapes of large unoccluded eval objects vs. GT under the 24 cube rotations."""
    setup_logging()
    model = model or Sam3DGeometry(cfg, pick_device(cfg.device))
    ds = ClipDataset(cfg, cfg.data.eval_split, train=False)
    items = []
    for i, it in enumerate(ds.index):
        if "middle" not in it["rules"]:
            continue
        clip = reader.load_clip(ds.data_root, ds.entries[it["clip_id"]])
        r, k = it["r"], it["k"]
        if reader.visibility(clip)[r, k - 1] >= min_visible:
            items.append((reader.visibility(clip, "amodal_px")[r, k - 1], i))
    items = [i for _, i in sorted(items, reverse=True)[:num_objects]]
    assert items, f"no eval object with visible_frac >= {min_visible} at the middle frame"
    rotations, ious = cube_rotations(), []
    for i in items:
        s = ds[i]
        info = s["info"]
        cond = model.condition(s["inputs"]["ref_rgb"][None].cuda(), s["inputs"]["ref_mask"][None].cuda())
        z, _ = model.sample(cond, [sample_seed(info["clip_id"], info["k"], info["r"])], cfg.sampler)
        pred = model.decode_latent(z)[0].cpu().numpy()
        gt = reader.load_voxels(ds.data_root, info["voxels"])
        gt = fill_voxels(gt) if cfg.target.voxel_fill == "solid" else gt
        ious.append([voxel_iou(pred, apply_canonical_permutation(gt, P)) for P in rotations])
    ious = np.array(ious)
    wins = np.bincount(ious.argmax(1), minlength=len(rotations))
    best = int(wins.argmax())
    report = {"num_objects": len(items), "winner": rotations[best].tolist(),
              "winner_share": float(wins[best] / len(items)), "winner_mean_iou": float(ious[:, best].mean()),
              "identity_mean_iou": float(ious[:, 0].mean()), "wins_per_rotation": wins.tolist(),
              "mean_iou_per_rotation": ious.mean(0).round(4).tolist(),
              "rotations": [P.tolist() for P in rotations]}
    out = Path(cfg.paths.output_root) / "frame_test.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    log.info("T4 winner %s on %.0f%% of %d objects (mean IoU %.3f; identity %.3f) -> set "
             "target.canonical_permutation=%s", report["winner"], 100 * report["winner_share"],
             len(items), report["winner_mean_iou"], report["identity_mean_iou"],
             json.dumps(report["winner"]).replace(" ", ""))
    return report
