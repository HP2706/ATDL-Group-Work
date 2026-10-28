# Video SAM 3D

A small gated cross-attention **fusion adapter** (~18M parameters) lets the frozen SAM 3D Objects Geometry model use a video clip through frozen V-JEPA 2.1 features. Only the adapter is trained, with SAM 3D's flow-matching loss on the ground-truth shape latent; evaluation is on the coarse 64³ shape.

Docs: [research plan](docs/research_plan.md) · [adapter plan](docs/fusion_adapter_plan.md) · [data format v0.1](docs/data_format.md) · [SAM 3D internals](docs/sam3d_internals.md) · [UCloud guide](docs/ucloud.md) · [progress](docs/progress.md)

## Repository

```
configs/base.yaml           all defaults
configs/conditions/         c1, c2, c3, c3_nomask: only what differs from base
configs/env/                local, ucloud, gcp: paths and machine settings only
vidsam3d/__main__.py        CLI: python -m vidsam3d <command>
vidsam3d/config.py          base <- condition <- env <- key=value overrides
vidsam3d/check_env.py       GPU, versions, paths, HF access, model loading
vidsam3d/data/reader.py     reads data format v0.1 (numpy + PIL)
vidsam3d/data/samples.py    frame subset, reference frame, window, object-centred crop, dataset
vidsam3d/data/targets.py    GT voxels -> cached SAM 3D latents; pose stub for Phase B
vidsam3d/models/adapter.py  gated cross-attention fusion adapter
vidsam3d/models/encoders.py V-JEPA 2.1, SAM 3D's DINOv2, token coverage
vidsam3d/models/sam3d.py    frozen SAM 3D Stage-1 wrapper (fake_sam3d.py: CPU stand-in)
vidsam3d/train.py           training loop, validation, checkpoints, benchmark
vidsam3d/evaluate.py        per-sample results.jsonl for runs and baselines (resumable)
vidsam3d/metrics.py         vIoU, ICP-aligned vIoU, Chamfer, F-score
vidsam3d/analyze.py         paired deltas, cluster bootstrap, Wilcoxon, tables, plots
vidsam3d/datagen.py         gen-data (external generator), merge-manifests, validate-data
vidsam3d/inspect_sam3d.py   inspect-sam3d and frame-test (GPU diagnostics)
scripts/                    fetch_upstream, setup_env, activate, run (all platforms)
scripts/ucloud/run.sh       UCloud batch script
scripts/gcp/                Dockerfile, build, submit (Vertex AI), vm_run (GPU VM)
tests/                      CPU tests on a fake dataset and FakeSam3D; tests/gpu/ needs CUDA
```

## Quickstart (GPU machine)

| Step | Command |
|---|---|
| Upstream code and checkpoints | `bash scripts/fetch_upstream.sh` (prompts for the HF token) |
| Environment (once) | `bash scripts/setup_env.sh`, then `source scripts/activate.sh` in every shell |
| HF token for this shell | `read -rsp "HF token: " HF_TOKEN && export HF_TOKEN` |
| Check | `python -m vidsam3d check-env --load-models` |
| Check the data | `python -m vidsam3d validate-data --split eval` |
| Latents | `python -m vidsam3d cache-latents --split train` (also `val`, `eval`) |
| Benchmark, then train | `python -m vidsam3d train --condition c3 --benchmark`, then without `--benchmark` |
| Evaluate | `python -m vidsam3d eval --run c3_seed0` (or `zero-shot`, `best-frame`; `--k 4`, `--shuffle-frames`, `--wrong-video`) |
| Analyse | `python -m vidsam3d analyze --results $OUTPUT/eval/*.jsonl --baseline c1_seed0` (`$OUTPUT` = `paths.output_root`) |

Config: `--condition c3 --env ucloud key=value ...`; the env also comes from `$VIDSAM3D_ENV`. Every run saves its resolved `config.yaml`.
CPU tests: `uv venv .venv --python 3.12 && uv pip install --python .venv/bin/python torch==2.7.1 -r requirements.txt -e . && .venv/bin/python -m pytest` (the metric and eval tests also need pytorch3d; see [progress](docs/progress.md)).

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
