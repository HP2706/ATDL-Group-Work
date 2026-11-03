# M2 runbook: 10k train clips on a GPU cluster

M2 renders 10,000 train clips from Objaverse assets on SA-1B backgrounds, validates them, and packs them into training shards. Everything runs from the repo root on a Linux node with NVIDIA GPUs. T6 is signed off, so all steps can run in order. The whole chain (assets, render with Blender 4.5, tests, export) was checked locally on 5 train clips.

Expected volume: about 185 GB of full-format clips on scratch (inferred from M1's 9.2 GB per 500 clips), 40-50 GB of training shards.

## 1. Set up the node

```bash
git clone <this repo> a3 && cd a3
bash scripts/m2/setup_node.sh
```

This installs uv and the Python environment and downloads Blender 4.5.15 LTS into `.blender/`. M2 uses Blender 4.5 rather than BlenderProc's default 4.2.1, because 4.2 has no kernels for Blackwell GPUs (B200). Locally, 4.5 gives identical masks and depth and RGB within 0.03/255 of 4.2. The script prints the GPU, driver and glibc version; Blender needs glibc ≥ 2.28.

## 2. Data

Everything goes under `data/` and `dataset/` in the repo, or symlink those two folders to scratch first. Both are git-ignored.

**CC0 backgrounds (Poly Haven).** Run this once on a CPU node, in persistent storage:

```bash
uv run python scripts/prepare_polyhaven.py --limit 200 --views 4
```

This uses the public Poly Haven API to download 200 tone-mapped panoramas and produce
four 2048x1024 perspective JPEGs per panorama (130 degree horizontal FOV). No GPU,
API key or SA-1B links file is needed. Powered by Poly Haven (https://polyhaven.com).
Assets are CC0: https://polyhaven.com/license. Downloads are sequential, cached and
checksummed. A failed download can be retried with the same command.

- Images and cached selection/API responses: `data/raw/polyhaven/`.
- Frozen source selection, image provenance and checksums: `data/bg_pools_polyhaven/manifest.json`.
- Generator pools: `data/bg_pools_polyhaven/{train,eval}.jsonl`; M2 already selects this directory.
- Train/eval assignment is by source panorama, so different views cannot leak across splits.
- Freeze both pools before rendering. Changed preparation arguments require new `--raw` and
  `--out` directories and corresponding config changes. Do not run `build_bg_pools.py` on
  the prepared views: its image-ID split would break panorama grouping.
- Copy `manifest.json` and `stats.json` with the eventual HF release. Per-clip background
  provenance is also retained in exported sample JSON.
- Preserve the original SA-1B eval privately as historical M1/T6 results. Regenerate any
  eval split intended for redistribution with `configs/m1_polyhaven.yaml` and a NEW root:

```bash
mkdir -p dataset_polyhaven
ln -s "$(realpath dataset/assets)" dataset_polyhaven/assets
uv run python -m sam3dvid.generate --config configs/m1_polyhaven.yaml --split eval --seeds $(seq 0 499)
```

The new background source needs smoke/coverage checks and visual review. Historical T6
numbers refer to the old backgrounds. Do not mix old SA-1B clips into the new release.
A private HF repository does not waive source licenses. Objaverse's per-object licenses
and GSO attribution must be checked before upload; CC0 backgrounds do not license the
complete composite or 3D targets as CC0.

**Objaverse train assets.**

```bash
uv run python scripts/download_objaverse.py --limit 10000                       # -> data/raw/objaverse
uv run python -m sam3dvid.assets --source objaverse --split train --workers 8   # -> dataset/assets/train
uv run python -m sam3dvid.assetfilter --split train                             # -> dataset/review
```

- **Download:** only CC0 / CC BY assets pass the release allowlist, before downloading GLBs. Unknown, NC, ND and SA licenses are excluded, and CC BY entries missing title, author or source URL are excluded. `--limit 10000` counts candidates BEFORE this filter, so expect fewer downloaded assets. Rejections are recorded in `index.json`; `ATTRIBUTION.json` records downloaded assets, license links, creators and modifications. Include that file in the HF release, including private sharing. A seeded subset of TRELLIS-500K's Objaverse-XL Sketchfab list (aesthetic score >= 5.5), with license, author and captions per object in `data/raw/objaverse/index.json`. It resumes, and a larger `--limit` extends a smaller one. The download size per GLB hasn't been measured; check `du -sh data/raw/objaverse/glbs` after the first few hundred.
- **Preprocessing:** finished assets are skipped, so a killed run resumes. Assets that crash or hang Blender are logged to `dataset/assets/failed_train.jsonl` and skipped from then on.
- **Filter:** it is applied when clips are generated, not at preprocessing, so changing a threshold needs no reprocessing. On 100 test assets, 67% were accepted: 27 were rejected as scenes (houses, floor plans, landscapes, found through the TRELLIS captions) and 6 on geometry. The earlier 67% acceptance figure predates the license allowlist. Measure the actual downloaded and geometrically accepted pool size before freezing it.
- **Contact sheets:** `dataset/review/assets_train_{accepted,rejected}.png` show 50 random accepted and rejected assets each. Send them back with the counts the command prints.
- **Freeze the pool before step 4.** A train seed draws from the sorted list of accepted assets, so adding assets or changing a filter threshold after rendering has started changes which assets every later seed gets. Download, preprocess and filter everything first; don't touch `dataset/assets/train` or the filter settings while clips are being rendered.

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

## 4. Render

Use a fresh dataset root for the Poly Haven run. Existing clips are skipped on resume, so
reusing a root containing SA-1B clips would silently mix background sources. Link shared
assets into the fresh root, but do not copy historical clips.

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

```bash
uv run python -m sam3dvid.export --dataset /scratch/m2/dataset --split train --out /scratch/m2/export/train \
    --clips-per-shard 100 --workers 8 --verify 20
```

The export packs RGB (JPEG q95, 4:4:4), visible masks, per-frame camera-frame poses and the voxels of the used assets into WebDataset tar shards plus `index.json`; the format is in `docs/data_format.md` Section 9. Only the export directory needs to leave the cluster: about 4 MB per clip, so about 40 GB for 10,000 clips.

- **Run it only after rendering and validation are finished.** Shards are cut from the sorted clip list, and an existing shard is skipped on resume, so exporting while clips are still missing produces shards whose contents don't match a later index.
- **Round trip:** `--verify 20` reads the first 20 clips back and checks them against the source clips: masks bit-exact, poses and voxels equal, RGB PSNR > 40 dB. It stops with an AssertionError naming the clip and frame otherwise. Re-running the command only verifies, because finished shards are skipped.
- **Not yet tried:** reading the shards with the `webdataset` library itself.


## Alpha-material compatibility check (2026-10-28)

The full-pool benchmark found T1b silhouette failures on 3/29 clips. Their failing objects
used BLEND/MASK materials; a geometry-only ray caster cannot reproduce alpha cutouts.
A conservative material audit excludes such assets via `asset_exclusions_file` in M2.
The runtime file `dataset/asset_exclusions_train.json` contains IDs and reasons, and must
travel with the input-pool freeze. No assets are deleted and geometric tolerances stay
unchanged. The reviewed 5,410-object pool becomes 4,550 opaque-compatible objects.
The original 1/2/4-worker benchmark was 17.7/9.7/6.3 seconds per clip but failed geometry
validation, so it is provisional until the revised pool and pilot pass.
