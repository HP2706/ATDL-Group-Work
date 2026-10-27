"""Geometric asset filter (spec Section 4.1, paper App. B.1) and the contact sheet for reviewing it.

Usage:
    uv run python -m sam3dvid.assetfilter --split train [--num 50] [--out dataset/review]

Thresholds live in `Config` and are applied to the `filter_scores` stored in each asset's meta.json,
so changing them needs no reprocessing. The command writes two contact sheets (rejected and
accepted assets, thumbnails labelled with their scores) and a json with the thresholds, counts and
the listed assets.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .blenderrun import run_blenderproc
from .config import Config

THUMB = 256
COLS = 10
LABEL_H = 34
BACKGROUND = (214, 214, 214)


def rejection_reasons(scores: dict[str, float], cfg: Config) -> list[str]:
    """Which App. B.1 criteria (scores from `load_scores`) an asset fails; empty means accepted."""
    reasons = []
    if scores["extent_min"] < cfg.filter_extent_min:
        reasons.append("thin")
    if scores["normal_eig_min"] < cfg.filter_normal_eig_min:
        reasons.append("flat")
    if scores["outlier_area_frac"] > cfg.filter_outlier_area_frac:
        reasons.append("outlier")
    if scores["num_voxels"] < cfg.filter_min_voxels:
        reasons.append("small")
    return reasons


def load_scores(cfg: Config, split: str) -> dict[str, dict[str, float]]:
    """filter_scores plus num_voxels per asset id."""
    root = Path(cfg.dataset_root) / "assets" / split
    metas = (json.loads(p.read_text()) for p in sorted(root.glob("*/meta.json")))
    return {m["asset_id"]: {**m["filter_scores"], "num_voxels": m["num_voxels"]} for m in metas}


def accepted_assets(cfg: Config, split: str) -> list[str]:
    """Asset ids to sample scenes from: train applies the geometric filter, eval keeps everything."""
    scores = load_scores(cfg, split)
    if split != "train":
        return sorted(scores)
    return sorted(a for a, s in scores.items() if not rejection_reasons(s, cfg))


def render_thumbnails(cfg: Config, split: str, asset_ids: list[str], thumb_dir: Path) -> None:
    thumb_dir.mkdir(parents=True, exist_ok=True)
    jobs = [
        {"mesh": str((Path(cfg.dataset_root) / "assets" / split / a / "mesh.glb").resolve()), "out": str((thumb_dir / f"{a}.npy").resolve())}
        for a in asset_ids
        if not (thumb_dir / f"{a}.npy").exists()
    ]
    if not jobs:
        return
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "jobs.json").write_text(json.dumps(jobs))
        run_blenderproc(
            "render_thumbs.py", str(Path(tmp) / "jobs.json"),
            install_path=cfg.blender_install_path, custom_path=cfg.blender_custom_path, stall_timeout=600,
        )


def contact_sheet(cfg: Config, entries: list[tuple[str, dict, list[str]]], thumb_dir: Path, title: str) -> Image.Image:
    font = ImageFont.load_default(size=13)
    rows = max(1, -(-len(entries) // COLS))
    cell_h = THUMB + LABEL_H
    sheet = Image.new("RGB", (COLS * THUMB, rows * cell_h + 28), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    draw.text((6, 5), title, fill=(0, 0, 0), font=ImageFont.load_default(size=16))
    for i, (asset_id, scores, reasons) in enumerate(entries):
        x, y = (i % COLS) * THUMB, (i // COLS) * cell_h + 28
        path = thumb_dir / f"{asset_id}.npy"
        if path.exists():
            rgba = np.load(path).astype(np.float32)
            rgb = rgba[..., :3] * rgba[..., 3:] / 255 + np.array(BACKGROUND) * (1 - rgba[..., 3:] / 255)
            sheet.paste(Image.fromarray(rgb.astype(np.uint8)).resize((THUMB, THUMB), Image.LANCZOS), (x, y))
        draw.text((x + 3, y + THUMB), f"{asset_id[:8]} {','.join(reasons)}", fill=(0, 0, 0), font=font)
        label = f"ext {scores['extent_min']:.2f} eig {scores['normal_eig_min']:.2f} out {scores['outlier_area_frac']:.2f} vox {scores['num_voxels']}"
        draw.text((x + 3, y + THUMB + 16), label, fill=(60, 60, 60), font=font)
    return sheet


def thresholds(cfg: Config) -> dict[str, float]:
    return {
        "reject_if_extent_min_below": cfg.filter_extent_min,
        "reject_if_normal_eig_min_below": cfg.filter_normal_eig_min,
        "reject_if_outlier_area_frac_above": cfg.filter_outlier_area_frac,
        "reject_if_num_voxels_below": cfg.filter_min_voxels,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", choices=["train", "eval"], default="train")
    ap.add_argument("--num", type=int, default=50, help="assets per sheet")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None, help="default: <dataset_root>/review")
    args = ap.parse_args()
    cfg = Config.load(args.config)
    out = args.out or Path(cfg.dataset_root) / "review"
    out.mkdir(parents=True, exist_ok=True)

    scores = load_scores(cfg, args.split)
    reasons = {a: rejection_reasons(s, cfg) for a, s in scores.items()}
    rejected = sorted(a for a, r in reasons.items() if r)
    accepted = sorted(a for a, r in reasons.items() if not r)
    rng = random.Random(args.seed)
    shown = {
        "rejected": sorted(rng.sample(rejected, min(args.num, len(rejected)))),
        "accepted": sorted(rng.sample(accepted, min(args.num, len(accepted)))),
    }
    thumb_dir = out / f"thumbs_{args.split}"
    render_thumbnails(cfg, args.split, shown["rejected"] + shown["accepted"], thumb_dir)
    for name, ids in shown.items():
        entries = [(a, scores[a], reasons[a]) for a in ids]
        title = f"{args.split} {name} ({len(ids)} of {len(rejected) if name == 'rejected' else len(accepted)}); thresholds {thresholds(cfg)}"
        contact_sheet(cfg, entries, thumb_dir, title).save(out / f"assets_{args.split}_{name}.png")

    summary = {
        "split": args.split,
        "thresholds": thresholds(cfg),
        "num_assets": len(scores),
        "num_accepted": len(accepted),
        "num_rejected": len(rejected),
        "accept_rate": len(accepted) / max(len(scores), 1),
        "rejected_by_reason": {k: sum(k in r for r in reasons.values()) for k in ("thin", "flat", "outlier", "small")},
        "shown": {name: {a: {"scores": scores[a], "reasons": reasons[a]} for a in ids} for name, ids in shown.items()},
    }
    (out / f"assets_{args.split}_filter.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "shown"}, indent=1))


if __name__ == "__main__":
    main()
