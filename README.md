# sam3dvid: synthetic multi-object videos for SAM 3D

[SAM 3D Objects](https://arxiv.org/abs/2511.16624) reconstructs the shape and pose of objects from a single image. This repository is part of a university course project (ATDL) that extends it from single images to short videos: given a clip and object masks, predict each object's 3D shape and its pose in every frame.

Training and evaluating such a model needs videos with exact 3D ground truth, which real footage doesn't provide. This repo is the **dataset generator**: it renders synthetic clips of 1-4 static objects seen by an orbiting camera, with per-frame ground truth in a format that maps onto SAM 3D's outputs. The model itself lives elsewhere.

## What a clip contains

- 32 frames at 512×512, composited over a real photo (SA-1B) that is warped consistently with the camera motion.
- Per frame: camera pose and intrinsics (OpenCV convention), visible instance masks, amodal masks per object, z-depth.
- Per object: the asset id, its pose in the world (rotation, translation, isotropic scale), and the asset's canonical mesh and 64³ voxels, matching TRELLIS's normalization and voxelization.

Eval clips use Google Scanned Objects; train clips will use Objaverse-XL assets from the TRELLIS-500K subset (planned). See [`docs/data_format.md`](docs/data_format.md) for the full format and conventions, and [`docs/dataset_spec.md`](docs/dataset_spec.md) for how clips are sampled and validated.

## How it works

1. **Assets** (`sam3dvid.assets`): import, normalize into the unit cube and voxelize at 64³, replicating TRELLIS's preprocessing.
2. **Sampling** (`sam3dvid.sampling`): object count, random rotations and scales, non-overlapping placement, an eased orbit arc framed tightly around the objects, lighting.
3. **Pre-check** (`sam3dvid.precheck`): voxel splatting rejects layouts where an object would be nearly hidden, before anything is rendered.
4. **Render** (`sam3dvid/blender/render_clip.py`): BlenderProc / Cycles on the GPU, one main pass plus one amodal pass per object.
5. **Validate and write** (`sam3dvid.generate`, `sam3dvid.validate`): visibility thresholds, compositing over the background, atomic clip write.

The tests check that the ground truth is internally consistent: projections, ray-cast silhouettes and depth agree with the stored poses to sub-pixel accuracy, and generation is deterministic per seed.

## Quickstart

Requires Linux, an NVIDIA GPU (the OptiX denoiser is NVIDIA-only) and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
# Blender 4.2 is downloaded into .blender/ by BlenderProc on first use.

uv run python scripts/download_gso.py                      # Google Scanned Objects -> data/raw/gso
uv run python -m sam3dvid.assets --source gso --split eval # -> dataset/assets/eval
uv run python scripts/download_sa1b.py --links data/sa1b_links.txt --archives sa_000000.tar
uv run python scripts/build_bg_pools.py                    # -> data/bg_pools

uv run python -m sam3dvid.generate --config configs/m1.yaml --split eval --seeds $(seq 0 499)
uv run python -m sam3dvid.review --split eval --num 20    # mp4s and contact sheets for a visual check
uv run pytest
```

SA-1B download links are personal and must be requested from [Meta](https://ai.meta.com/datasets/segment-anything-downloads/); put them in `data/sa1b_links.txt`. No data, assets or rendered clips are included in this repository.

## Status

- **M0** (one validated clip end to end): done.
- **M1** (500 eval clips from GSO): rendered, about 14 s per clip on an RTX 3060 Laptop GPU with two workers.
- **T6** (sanity check against released SAM 3D, including the convention converter): scripts in `scripts/t6/`, see [`docs/t6_runbook.md`](docs/t6_runbook.md); in progress.
- **M2** (10k train clips from Objaverse on a cluster): planned.

## Layout

```
src/sam3dvid/      generator package (sampling, rendering, validation, I/O)
src/sam3dvid/blender/  scripts that run inside Blender via BlenderProc
scripts/           downloads, background pools, T6 GPU scripts
configs/           generation configs per milestone
tests/             convention unit tests and per-clip ground-truth checks
docs/              format, spec, runbooks
```

## Acknowledgements and licenses

- Asset normalization reuses code from [TRELLIS](https://github.com/microsoft/TRELLIS) (MIT License, Copyright (c) Microsoft Corporation); see `src/sam3dvid/blender/normalize_assets.py`.
- Rendering uses [BlenderProc](https://github.com/DLR-RM/BlenderProc) and Blender.
- Data sources keep their own licenses: [Google Scanned Objects](https://app.gazebosim.org/GoogleResearch) (CC BY 4.0), [SA-1B](https://ai.meta.com/datasets/segment-anything/) (SA-1B research license), [Objaverse-XL](https://github.com/allenai/objaverse-xl) (per-object licenses). [SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects) is not included and is under its own license.
