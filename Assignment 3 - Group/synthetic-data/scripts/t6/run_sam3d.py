"""Run SAM 3D Objects on the T6 inputs and save its raw outputs (no post-processing).

Runs inside the sam3d-objects conda env, from anywhere:
    python scripts/t6/run_sam3d.py --sam3d-dir ~/sam-3d-objects --inputs t6/inputs --out t6/outputs
    python scripts/t6/run_sam3d.py ... --smoke     # one object only, prints the output keys

Each object is run once per variant: "moge" (SAM 3D estimates the pointmap itself, as in the
paper) and "gt_pointmap" (our rendered depth). Results go to <out>/<clip_id>/obj<k>_<variant>.npz;
existing files are skipped, so an interrupted run can simply be restarted. Conventions are
not interpreted here: everything is saved as SAM 3D returns it.
"""

from __future__ import annotations

import argparse
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


def to_numpy(x):
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sam3d-dir", type=Path, required=True, help="clone of facebookresearch/sam-3d-objects")
    ap.add_argument("--tag", default="hf", help="checkpoint folder under <sam3d-dir>/checkpoints")
    ap.add_argument("--inputs", type=Path, default=Path("t6/inputs"))
    ap.add_argument("--out", type=Path, default=Path("t6/outputs"))
    ap.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    ap.add_argument("--smoke", action="store_true", help="run a single object and print the output layout")
    args = ap.parse_args()

    inputs, out = args.inputs.resolve(), args.out.resolve()
    index = json.loads((inputs / "index.json").read_text())

    # demo.py runs from the repo root; do the same in case the configs use relative paths.
    os.chdir(args.sam3d_dir)
    sys.path.insert(0, str(args.sam3d_dir / "notebook"))
    import torch
    from inference import Inference

    inference = Inference(str(args.sam3d_dir / "checkpoints" / args.tag / "pipeline.yaml"), compile=False)

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
    sys.exit(main())
