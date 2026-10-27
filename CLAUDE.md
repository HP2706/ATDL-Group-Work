# sam3dvid: synthetic video data for SAM 3D

Generator for short synthetic videos of 1-4 static objects seen by an orbiting camera, with exact per-frame ground truth (camera poses, object poses, visible and amodal masks, depth, 64³ voxels). It extends SAM 3D (image -> 3D objects) to video. Eval uses Google Scanned Objects (GSO), train uses Objaverse; backgrounds are SA-1B photos.

## Where things are

- `docs/dataset_spec.md`: the spec (what to build, tests T1-T7, milestones M0-M2).
- `DATASET.md`: every implementer decision with its reason, plus milestone results. Read it before changing sampling, rendering or filters.
- `docs/data_format.md`: the on-disk format and conventions for consumers.
- `docs/m2_handover.md`: the M2 task brief. `docs/m2_runbook.md`: the M2 commands.
- `src/sam3dvid/`: the generator (`generate.py` is the entry point; `blender/` runs inside Blender via BlenderProc). `scripts/`: downloads, cluster and VM scripts. `configs/`: one YAML per milestone; defaults live in `src/sam3dvid/config.py`.

## Commands

```bash
uv sync
uv run pytest -q -m "not slow"                       # unit tests + per-clip tests on dataset/
SAM3DVID_DATASET=<root> SAM3DVID_SAMPLE_FRAC=0.02 uv run pytest -q -m "not slow"   # another dataset, 2% sample
uv run python -m sam3dvid.generate --config configs/m1.yaml --split eval --seeds 0 1 2
uv run python -m sam3dvid.review --split eval --num 20   # mp4s + contact sheets for a visual check
```

## Conventions (don't change without updating DATASET.md and data_format.md)

- Cameras are OpenCV (x right, y down, z forward). `T_cw` maps world -> camera; `R_ab` maps frame b into frame a. Principal point at (W-1)/2.
- The canonical object frame is Blender Z-up after TRELLIS normalization (unit cube). `mesh.glb` is Y-up glTF; outside Blender apply `conventions.CANONICAL_FROM_GLTF`.
- A seed fully determines a clip; retries only redraw layout and camera. Eval seeds are 0-499, train seeds start at 1000000 (a seed fixes the layout regardless of split).
- SAM 3D uses PyTorch3D camera axes (x left, y up); converting is the job of a converter, never of the stored data.

## Rules

- `data/` holds raw downloads and is not read directly (permission rule); scripts process it.
- `data/sa1b_links.txt` contains personal signed URLs: never print, log or commit it.
- SA-1B images and anything composited from them (clip RGB) are not to be committed or published.
- Rendering needs an NVIDIA GPU (OptiX denoiser). Blender 4.2.1 is the default; Blackwell GPUs (B200) need Blender 4.5 LTS via `blender_custom_path` (set in `configs/m2.yaml`).
- Code style: match the surrounding code; comment only the surprising parts; docstrings describe usage. Never use em dashes in code or docs.
