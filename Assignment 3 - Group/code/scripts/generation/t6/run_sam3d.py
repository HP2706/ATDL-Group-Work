"""Run SAM 3D Objects on the T6 inputs and save its raw outputs (no post-processing).

Uses the shared installed environment:
    python scripts/generation/t6/run_sam3d.py --checkpoint_dir=/models/checkpoints --inputs=t6/inputs --out=t6/outputs
    python scripts/generation/t6/run_sam3d.py ... --smoke=True

Each object is run once per variant: "moge" (SAM 3D estimates the pointmap itself, as in the
paper) and "gt_pointmap" (our rendered depth). Results go to <out>/<clip_id>/obj<k>_<variant>.npz;
existing files are skipped, so an interrupted run can simply be restarted. Conventions are
not interpreted here: everything is saved as SAM 3D returns it.
"""

from __future__ import annotations

import fire
from typing import Any
from pydantic import BaseModel
from hydra.utils import instantiate
from omegaconf import OmegaConf
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

VARIANTS = ("moge", "gt_pointmap")
SEED = 42
# Generic tensors larger than this (latents, full-resolution maps) are not saved.
MAX_RAW_NUMEL = 2_000_000


def to_numpy(x: Any) -> np.ndarray:
    import torch

    if isinstance(x, torch.Tensor):
        x = x.detach().float() if x.is_floating_point() else x.detach()
        return x.cpu().numpy()
    return np.asarray(x)


def collect(output: dict, save_pointmap: bool) -> dict[str, np.ndarray]:
    import torch

    arrays = {}
    for key, value in output.items():
        if isinstance(value, torch.Tensor) and value.numel() <= MAX_RAW_NUMEL and key != "pointmap":
            arrays[key] = to_numpy(value)
        elif isinstance(value, (int, float)):
            arrays[key] = np.asarray(value)
    gs = output.get("gaussian")
    if gs:
        arrays["gs_xyz"] = to_numpy(gs[0].get_xyz)
        arrays["gs_opacity"] = to_numpy(gs[0].get_opacity)
    mesh = output.get("mesh")
    if mesh:
        arrays["mesh_vertices"] = to_numpy(mesh[0].vertices)
        arrays["mesh_faces"] = to_numpy(mesh[0].faces)
    if save_pointmap and "pointmap" in output:
        arrays["pointmap_used"] = to_numpy(output["pointmap"]).astype(np.float16)
    return arrays


def describe(output: dict) -> dict:
    import torch

    desc = {}
    for key, value in output.items():
        if isinstance(value, torch.Tensor):
            desc[key] = f"tensor {tuple(value.shape)} {value.dtype}"
        elif isinstance(value, (list, tuple)):
            desc[key] = f"{type(value).__name__}[{len(value)}] of {type(value[0]).__name__ if value else '-'}"
        else:
            desc[key] = type(value).__name__
    return desc


class Options(BaseModel):
    checkpoint_dir: Path
    inputs: Path = Path("t6/inputs")
    out: Path = Path("t6/outputs")
    variants: list[str] = list(VARIANTS)
    smoke: bool = False


def run(checkpoint_dir: str, inputs: str = "t6/inputs", out: str = "t6/outputs",
        variants: list[str] = list(VARIANTS), smoke: bool = False) -> int:
    args = Options(checkpoint_dir=Path(checkpoint_dir), inputs=Path(inputs), out=Path(out),
                   variants=variants, smoke=smoke)
    if not set(variants).issubset(VARIANTS):
        raise ValueError(f"variants must come from {VARIANTS}")
    inputs, out = args.inputs.resolve(), args.out.resolve()
    index = json.loads((inputs / "index.json").read_text())
    import torch
    if sys.platform == "darwin" or not torch.cuda.is_available():
        raise RuntimeError("Run SAM diagnostics on remote CUDA compute")
    os.environ.setdefault("LIDRA_SKIP_INIT", "1")
    os.environ.setdefault("ATTN_BACKEND", "sdpa")
    config = OmegaConf.load(args.checkpoint_dir / "pipeline.yaml")
    config.workspace_dir = str(args.checkpoint_dir.resolve())
    config.rendering_engine, config.compile_model = "pytorch3d", False
    pipeline = instantiate(config)

    def inference(image: np.ndarray, mask: np.ndarray, seed: int,
                  pointmap: torch.Tensor | None) -> dict[str, Any]:
        rgba = np.concatenate([image[..., :3], mask[..., None].astype(np.uint8) * 255], axis=-1)
        return pipeline.run(rgba, None, seed, stage1_only=False,
                            with_mesh_postprocess=False, with_texture_baking=False,
                            with_layout_postprocess=False, use_vertex_color=True,
                            stage1_inference_steps=None, pointmap=pointmap)

    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "run_log.jsonl"
    jobs = [(c, o, v) for c in index for o in c["objects"] for v in args.variants]
    if args.smoke:
        jobs = jobs[:1]
    for n, (clip, obj, variant) in enumerate(jobs, 1):
        clip_id, k = clip["clip_id"], obj["k"]
        target = out / clip_id / f"obj{k}_{variant}.npz"
        if target.exists() and not args.smoke:
            continue
        target.parent.mkdir(exist_ok=True)
        d = inputs / clip_id
        image = np.asarray(Image.open(d / "image.png").convert("RGB"))
        mask = np.asarray(Image.open(d / f"mask_{k}.png")) > 0
        pointmap = None
        if variant == "gt_pointmap":
            with np.load(d / "pointmap.npz") as f:
                pointmap = torch.from_numpy(f["pointmap"])

        record = {"clip_id": clip_id, "k": k, "variant": variant}
        t0 = time.time()
        try:
            output = inference(image, mask, seed=SEED, pointmap=pointmap)
            arrays = collect(output, save_pointmap=variant == "moge")
            np.savez_compressed(target, **arrays)
            record.update(status="ok", seconds=round(time.time() - t0, 1))
            if args.smoke:
                print(json.dumps(describe(output), indent=1))
                print("saved:", {key: a.shape for key, a in arrays.items()})
        except Exception as e:
            record.update(status="error", error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc())
        finally:
            torch.cuda.empty_cache()
        with open(log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        print(f"[{n}/{len(jobs)}] {clip_id} obj{k} {variant}: {record['status']} {record.get('seconds', '')}", flush=True)
    return 0


if __name__ == "__main__":
    fire.Fire(run)
