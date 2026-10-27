# M2 runbook: 10k train clips on a GPU cluster

M2 renders 10,000 train clips from Objaverse assets on SA-1B backgrounds, validates them, and packs them into training shards. Everything runs from the repo root on a Linux node with NVIDIA GPUs. Steps 1-3 can be done any time; **step 4 (the full render) waits until T6 is signed off**, because T6 can still change render settings.

Expected volume: about 185 GB of full-format clips on scratch (inferred from M1's 9.2 GB per 500 clips), 40-50 GB of training shards.

## 1. Set up the node

```bash
git clone <this repo> a3 && cd a3
bash scripts/m2/setup_node.sh
```

This installs uv and the Python environment and downloads Blender 4.5.15 LTS into `.blender/`. M2 uses Blender 4.5 rather than BlenderProc's default 4.2.1, because 4.2 has no kernels for Blackwell GPUs (B200). Locally, 4.5 gives identical masks and depth and RGB within 0.03/255 of 4.2. The script prints the GPU, driver and glibc version; Blender needs glibc ≥ 2.28.

## 2. Data

Everything goes under `data/` and `dataset/` in the repo, or symlink those two folders to scratch first. Both are git-ignored.

**SA-1B backgrounds.** Accept the SA-1B license yourself at <https://ai.meta.com/datasets/segment-anything-downloads/> and save the links file it gives you as `data/sa1b_links.txt`. The links are personal signed URLs; don't share or print them. Then:

```bash
uv run python scripts/download_sa1b.py --links data/sa1b_links.txt --archives sa_000000.tar
uv run python scripts/build_bg_pools.py
```

The train/eval split of the backgrounds is a deterministic hash of the image id, so this reproduces exactly the pools used for M1 (7,258 train images).

**Objaverse train assets.** *(Commands to be filled in when the Objaverse pipeline is merged.)*

## 3. Smoke test (first thing on a new GPU type)

```bash
bash scripts/m2/smoke_test.sh            # needs dataset/assets/train; use "eval" to test with GSO
```

It checks four things:
- that Blender renders on GPU 0;
- that the clips pass the test suite;
- that `meta.json` records Blender 4.5;
- which worker count per GPU is fastest: it times 1, 2 and 4 workers.

Send the output back. If the first render fails, the script prints the Blender log. The likeliest cause is the GPU not being supported by this Blender build.

## 4. Render (after T6 sign-off)

Train seeds are **1000000-1009999**. They must not overlap the eval seeds 0-499: a seed fixes N, layout and camera path regardless of split.

```bash
# one job per seed range; adjust the range size to the job time limit
bash scripts/m2/render_shard.sh 1000000 1002500 /scratch/m2/dataset 2
bash scripts/m2/render_shard.sh 1002500 1005000 /scratch/m2/dataset 2
...
```

- **Arguments:** start seed, end seed (exclusive), dataset root, and workers per GPU (from the smoke test).
- **GPUs:** all GPUs on the node are used.
- **Assets:** the dataset root needs the assets, e.g. `ln -s $(realpath dataset/assets) /scratch/m2/dataset/assets`.
- **Resuming:** finished clips are skipped, so a killed job is resumed by running the same command again.
- **Failed seeds:** rejected seeds are logged to `<root>/failures.jsonl`. Replacement seeds come from beyond 1009999 if needed.
- **Logs:** `<root>/logs/`.
- **Pilot first:** start with a range of about 100 seeds to measure throughput and the rejection rate on Objaverse assets before launching the rest.

## 5. Validate

```bash
SAM3DVID_DATASET=/scratch/m2/dataset SAM3DVID_SAMPLE_FRAC=0.02 uv run pytest -q -m "not slow"
uv run python -m sam3dvid.review --dataset /scratch/m2/dataset --split train --num 20 --out /scratch/m2/review
```

The first command runs T1/T1b on a 2% sample, and T3, T4 and the format checks on every clip, as the spec requires. The review videos are for a human check before sign-off.

## 6. Export training shards

*(Command to be filled in when the shard exporter is merged.)* The export packs RGB (JPEG q95), visible masks, poses and voxels into WebDataset tar shards. Only the shards need to leave the cluster.
