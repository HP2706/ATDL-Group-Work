# Video SAM 3D

A small gated cross-attention **fusion adapter** (~18M parameters) lets the frozen SAM 3D Objects Geometry model use a video clip through frozen V-JEPA 2.1 features. Only the adapter is trained, with SAM 3D's flow-matching loss on the ground-truth shape latent; evaluation is on the coarse 64³ shape.

Docs: [research plan](docs/research_plan.md) · [adapter design](docs/adapter_design.md) · [data format](../synthetic-data/docs/data_format.md) (generator) · [SAM 3D internals](docs/sam3d_internals.md) · [UCloud guide](docs/ucloud.md) · [progress](docs/progress.md)

## Repository

```
configs/base.yaml           all defaults
configs/conditions/         c1, c2, c3, c3_nomask: only what differs from base
configs/env/                local, ucloud, gcp: paths and machine settings only
vidsam3d/__main__.py        CLI: python -m vidsam3d <command>
vidsam3d/config.py          base <- condition <- env <- key=value overrides
vidsam3d/check_env.py       GPU, versions, paths, HF access, model loading
vidsam3d/data/reader.py     reads the training export (WebDataset shards; numpy + PIL)
vidsam3d/data/samples.py    splits (asset-disjoint val), reference frame, window, crop, dataset
vidsam3d/data/targets.py    GT voxels -> cached SAM 3D latents; pose converter for Phase B
vidsam3d/data/validate.py   validate-data: checks an export folder against what the pipeline reads
vidsam3d/models/adapter.py  gated cross-attention fusion adapter
vidsam3d/models/encoders.py V-JEPA 2.1, SAM 3D's DINOv2, token coverage
vidsam3d/models/sam3d.py    frozen SAM 3D Stage-1 wrapper (fake_sam3d.py: CPU stand-in)
vidsam3d/train.py           training loop, validation, checkpoints, benchmark
vidsam3d/evaluate.py        per-sample results.jsonl for runs and baselines (resumable)
vidsam3d/metrics.py         vIoU, ICP-aligned vIoU, Chamfer, F-score
vidsam3d/analyze.py         paired deltas, cluster bootstrap, Wilcoxon, tables, plots
vidsam3d/inspect_sam3d.py   inspect-sam3d and frame-test (GPU diagnostics)
scripts/                    fetch_upstream, setup_env, activate, run (all platforms)
scripts/ucloud/run.sh       UCloud batch script
scripts/gcp/                Dockerfile, build, submit (Vertex AI), vm_run (GPU VM)
tests/                      CPU tests on fake clips packed by the generator's exporter; tests/gpu/ needs CUDA
```

## Data

The generator in [`../synthetic-data`](../synthetic-data) delivers every split as a training export (`python -m sam3dvid.export`, format §9): WebDataset shards with JPEG frames, visible masks, per-frame poses, and the voxels in one file. The pipeline reads only this export:

| Folder | Content |
|---|---|
| `<data_root>/train/` | 10,000 Objaverse clips. Val is carved out: clips that show one of the held-out assets (`data.val_asset_frac`, 3 %), with only those assets as targets |
| `<data_root>/eval/` | the held-out GSO test set (500 clips), exported from the full format with `python -m sam3dvid.export --dataset <dataset> --split eval --out <data_root>/eval` |

## Quickstart (GPU machine)

| Step | Command |
|---|---|
| Upstream code and checkpoints | `bash scripts/fetch_upstream.sh` (prompts for the HF token) |
| Environment (once) | `bash scripts/setup_env.sh`, then `source scripts/activate.sh` in every shell |
| HF token for this shell | `read -rsp "HF token: " HF_TOKEN && export HF_TOKEN` |
| Check | `python -m vidsam3d check-env --load-models` |
| Check the data | `python -m vidsam3d validate-data --split train` (prints the val split), then `--split eval` |
| Latents | `python -m vidsam3d cache-latents --split train` (covers val), then `--split eval` |
| Benchmark, then train | `python -m vidsam3d train --condition c3 --benchmark`, then without `--benchmark` |
| Evaluate | `python -m vidsam3d eval --run c3_seed0` (or `zero-shot`, `best-frame`; `--k 4`, `--shuffle-frames`, `--wrong-video`) |
| Analyse | `python -m vidsam3d analyze --results $OUTPUT/eval/*.jsonl --baseline c1_seed0` (`$OUTPUT` = `paths.output_root`) |

Config: `--condition c3 --env ucloud key=value ...`; the env also comes from `$VIDSAM3D_ENV`. Every run saves its resolved `config.yaml`.
CPU tests: `uv venv .venv --python 3.12 && uv pip install --python .venv/bin/python torch==2.7.1 -r requirements.txt -e . && .venv/bin/python -m pytest`. The fixtures pack fake clips with the generator's exporter from `../synthetic-data/src` (or `$SAM3DVID_SRC`); the metric and eval tests also need pytorch3d (see [progress](docs/progress.md)).

## Running on UCloud

- **Initialization script:** `scripts/setup_env.sh` (builds the env under `/work/vidsam3d/envs` once, then only verifies).
- **Batch script:** `scripts/ucloud/run.sh <command> [args]`, e.g. `scripts/ucloud/run.sh train --condition c3`.
- Outputs land in `/work/vidsam3d/outputs` (only `/work` persists); scratch is job-local disk when it has room. Details: [docs/ucloud.md](docs/ucloud.md).

## Running on Google Cloud

- Copy `scripts/gcp/gcp.env.example` to `scripts/gcp/gcp.env` and fill it in (project, bucket, machine and accelerator).
- `scripts/gcp/build.sh` builds the image with Cloud Build; `scripts/gcp/submit.sh train --condition c3` starts a Vertex AI job.
- The bucket is mounted at `/gcs/<bucket>/`: `dataset/`, `outputs/`, `cache/`, `checkpoints/sam3d/`.
- Option B: on a GPU VM, `scripts/gcp/vm_run.sh <command> [args]` mounts the bucket with gcsfuse and runs like UCloud.

## Adding a condition or ablation

Add `configs/conditions/<name>.yaml` with only the keys that differ from `base.yaml` (a misspelled key is an error), then `train --condition <name>`, `eval --run <name>_seed0` and include its results in `analyze`.
