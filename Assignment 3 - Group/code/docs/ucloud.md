# Running on UCloud

Everything lives under `/work/vidsam3d` (`PROJECT_DIR`), the only storage that persists between jobs. Use the Terminal Ubuntu app on a GPU machine (currently B200). The environment uses torch 2.7.1+cu128, because SAM 3D's own pinned stack (torch 2.5.1+cu121) has no B200 kernels.

## Layout under `/work/vidsam3d`

| Path | Content |
|---|---|
| `fusion-adapter-code/`, `synthetic-data/` | this repo and the data generator, side by side as in the group repo (the generator is needed only for the CPU tests and to export the eval split) |
| `sam-3d-objects/`, `vjepa2/` | pinned upstream code (`fetch_upstream.sh`) |
| `checkpoints/sam3d/` | gated SAM 3D Stage-1 checkpoints (`fetch_upstream.sh`) |
| `envs/vidsam3d/`, `miniforge3/` | the environment (`setup_env.sh`) |
| `dataset/train/`, `dataset/eval/` | training exports (`index.json`, `*.tar` shards, `voxels_*.npz`); val is carved out of `train/` |
| `outputs/`, `cache/` | runs, eval results, analysis; latents and shard indexes |
| `hf_home/`, `torch_home/`, `pip-cache/` | download caches |

## One-time setup (interactive Terminal job)

```bash
export PROJECT_DIR=/work/vidsam3d CODE_DIR=/work/vidsam3d/fusion-adapter-code
cd "$CODE_DIR"
bash scripts/fetch_upstream.sh          # pinned repos + checkpoints; prompts for the HF token
bash scripts/setup_env.sh               # 30-60 min (pytorch3d/kaolin builds); rerun continues
source scripts/activate.sh
python -m vidsam3d check-env --load-models
```

The HF token needs access to [`facebook/sam-3d-objects`](https://huggingface.co/facebook/sam-3d-objects). Enter it at a hidden prompt (`read -rsp "HF token: " HF_TOKEN && export HF_TOKEN`), never in a file, and `unset HF_TOKEN` afterwards.

## Batch jobs

| App field | Value |
|---|---|
| Initialization script | `scripts/setup_env.sh` (quick once built: verifies only) |
| Batch processing script | `scripts/ucloud/run.sh <command> [args]`, e.g. `scripts/ucloud/run.sh train --condition c3` |
| Folders | attach `/work/vidsam3d` |

`run.sh` activates the environment, puts scratch on the job's local disk when it has 100 GB free (`MIN_LOCAL_SCRATCH_GB`), and with `data.stage_to_scratch=true` (or `STAGE_TO_SCRATCH=1`) copies the export folders the command reads to scratch first. Checkpoints go to scratch and are copied to `outputs/<run>/`; `train.resume: auto` continues from `latest.json` after a time limit.

## When the data arrives

1. Put the train export from the M2 run (`synthetic-data/docs/m2_runbook.md` §6) in `dataset/train/`. Export the GSO test set once: `uv run python -m sam3dvid.export --dataset <eval dataset> --split eval --out /work/vidsam3d/dataset/eval` (from `synthetic-data/`, CPU only). Then `validate-data --split train` (also reports the val split) and `--split eval`.
2. `inspect-sam3d`, then `VIDSAM3D_ENV=ucloud pytest -m gpu tests/gpu -s -rs`, then `frame-test`. Set `adapter.query_sets`, `target.voxel_fill` (T5) and `target.canonical_permutation` (T4) in `configs/base.yaml` from their output.
3. `cache-latents --split train` (covers val) and `--split eval`.
4. `train --condition c3 --benchmark`, set `train.max_steps`, then train c1, c3, c3_nomask.
5. `eval` for every run and `zero-shot`, `best-frame`; test-time controls `--k 1/4/8`, `--shuffle-frames`, `--wrong-video`; then `analyze`.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `no kernel image is available`, `sm_100 is not compatible` | torch without B200 kernels: use the env from `setup_env.sh`, not SAM 3D's `default.yml` |
| `libGL.so.1: cannot open shared object file` | `setup_env.sh` installs `libgl` into the env; `activate.sh` puts it on `LD_LIBRARY_PATH` |
| `Name or service not known` during setup | no DNS / internet in the job: check `getent hosts pypi.org github.com`, relaunch the job |
| GPU tests all `skipped` | missing CUDA, checkpoints or `dataset/eval/index.json`; `-rs` shows which |
| kaolin or pytorch3d build fails | paste the first error; `FORCE=1 bash scripts/setup_env.sh` rebuilds everything |
| `spconv-cu126==2.3.8` not found | `SPCONV_PACKAGE=spconv-cu124 bash scripts/setup_env.sh` |
