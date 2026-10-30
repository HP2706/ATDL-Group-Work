# Video-to-3D

One `video3d` package, trainer, evaluator, configuration system and Python environment.
Julian's generator lives in `src/video3d/data/generation/`; its rendering, sampling and
sharded export algorithms are retained. Mathias' gated fusion and the foreground attentive
pooler are selectable conditioning strategies. Both support optional video/SAM LoRA.

## Environment

Use Python 3.11 and `uv`. On the remote Linux CUDA 12.8 machine:

```bash
bash scripts/setup_env.sh
source .venv/bin/activate
python -m video3d check-env --offline
```

The shared `pyproject.toml` and `uv.lock` own all dependency groups. Linux Torch is pinned
to CUDA 12.8. `scripts/install_models.py` installs pinned upstream libraries into this same
environment. SAM and PyTorch3D use their upstream packaging. V-JEPA and NAVI do not publish
package metadata, so the installer builds ordinary wheels from their pinned, unmodified
modules in a temporary build directory. No checkout is retained or imported through
`sys.path`. All revisions and the upstream-required Hydra patch are recorded in
`pyproject.toml`. Upstream model checkpoint YAML is consumed by SAM internally;
project experiment and generator configuration uses chz and resolved JSON.

System dependencies include a CUDA toolkit, C++ compiler, Git, EGL/OpenGL, libusb,
FFmpeg and BlenderProc's supported Blender installation. `scripts/verify_remote.py`
provides a bounded clean Linux/CUDA verification environment. Heavy work belongs on
remote compute. The Mac is suitable for editing, lint and import/config checks.

## Configuration and training

```bash
python -m video3d inspect --condition=c3
python -m video3d train --condition=c3 paths.data_root=/data/export paths.output_root=/results paths.cache_root=/cache paths.scratch_root=/scratch
python -m video3d train --condition=pooler paths.data_root=/data/export paths.output_root=/results paths.cache_root=/cache paths.scratch_root=/scratch
python -m video3d train --condition=c3 encoder.vjepa.video_lora=True sam3d.lora=True
```

Set `paths.sam3d_checkpoint_dir` to the pinned downloaded SAM checkpoint directory
containing `pipeline.yaml`. Access to the gated weights must already be granted.
Set `adapter.query_sets` only when overriding detected native RGB conditioning streams.
The presets retain explicit encoder/preprocessing choices; selecting fusion or pooler
alone does not make historical experiments comparable.

`train.max_steps`, `train.max_seconds`, `train.batch_size` and
`train.effective_batch_size` bound a run. Run names include a timestamp.
The same run name automatically resumes complete optimizer updates, including adapter,
LoRA, EMA, optimizer, scheduler, RNG and deterministic sample position. Changed source,
data identity or configuration is rejected on resume. Preserve old output directories.
Use `train-json <resolved.json>` to run a fully resolved configuration.
Optional W&B logging uses `logging.wandb=True` and credentials already in the environment.

For NAVI, set `data.source=navi`, `paths.data_root` to the raw NAVI data,
`data.prepared` to a new prepared directory and `data.eval_split=val` for the pilot, then run `python -m video3d prepare` with
those overrides. Existing prepared manifests feed the same trainer and evaluator.
For synthetic exports, run `cache-latents train` before training; caches include
upstream revision, voxel conventions and dataset identity in their namespace.

`train --benchmark=True` profiles the shared training path. `sweep` runs the four
frozen/video-LoRA/SAM-LoRA/both configurations sequentially through that same trainer,
then evaluates each within a foreground wall-time budget. It does not allocate compute.
Use a provider-native batch entrypoint so process exit releases the GPU allocation.

## Evaluation

The `dino_pooler` training preset uses frozen DINOv2 on each input frame and
learns a foreground attention pooler with 64 queries to produce SAM 3D
conditioning tokens. It retains all frame-level patch embeddings; `c2` keeps its
historical pair averaging. The preset uses the shared object crop and trainer,
with SAM 3D frozen by default: `python -m video3d train --condition=dino_pooler`.

```bash
python -m video3d eval /results/RUN paths.data_root=/data/export paths.output_root=/results paths.scratch_root=/scratch
python -m video3d eval zero-shot paths.data_root=/data/export
python -m video3d eval /results/RUN --repeated_reference=True
python -m video3d eval /results/RUN --shuffle_frames=True
python -m video3d eval /results/RUN --wrong_video=True
python -m video3d eval /results/RUN eval.video_diagnostics=True eval.checkpoint_weights=current --limit=10
```

Evaluation uses EMA weights and paired deterministic seeds. Results resume per sample
and reject mismatched source/data/settings. Media pairs include input video, reference
RGB/mask, GT and prediction meshes. Empty predictions are explicit. `best-frame` is a
synthetic GT-visibility oracle. Canonical voxel IoU and canonical NAVI visibility-region
completeness are separate from bbox-normalized, similarity-ICP surface Chamfer/F-scores.
The Chamfer here sums the directional means. Historical pilot metrics used a different
protocol; do not pool those results. Resolved settings and metric protocol accompany
new results. `analyze --results='["/results/eval/RUN.jsonl"]'` produces summaries.

Every evaluation also records `flow_loss`, the conditional shape-velocity MSE
against the cached ground-truth VAE latent. Pretrained `zero-shot` SAM 3D and
trained runs use identical sample-keyed noise and `val.tau_grid`, without
classifier-free guidance. `--loss_only=True` skips reconstruction and mesh
metrics. Select `eval.checkpoint_weights=current` to compare the raw weights
shown in epoch previews; the default is EMA. Each JSONL has a companion
`.summary.json` with the mean loss, sample count, and protocol. These fixed-grid
losses are comparable across models, but differ from stochastic training losses.
Use the same split, reference rules, grid and data settings for every model.
Evaluating the training split measures fit, not held-out generalization.

```bash
python -m video3d eval zero-shot --loss_only=True --split=train
python -m video3d eval /results/RUN --loss_only=True --split=train eval.checkpoint_weights=current
```

### Video-use diagnostics

`eval.video_diagnostics=True` enables paired correct-video, repeated-reference,
wrong-video and disabled-video controls in both the evaluator and epoch previews.
`eval.video_controls` selects a subset, always including `correct`. The reference
RGB/mask and sampling seed stay fixed. For fusion, disabling video removes only
the gated cross-attention updates and preserves learned feed-forward updates and
SAM LoRA; for a replacement pooler it restores native reference conditioning.

Records contain `video_diagnostics` and `video_controls`. W&B charts show per-block
video-update norm divided by the original conditioning norm, object/background
attention enrichment relative to token coverage, and correct-minus-control metric
deltas. `video/attention` retains per-head temporal mass and enrichment. V-JEPA
and pair-averaged DINO clip features have frame-pair resolution; the table lists
both source frame IDs rather than claiming independent per-frame attention.

Synthetic surface metrics use the cached voxel-derived surface and instance-mask,
first-hit visibility estimate. Regions are reference-visible, reference-unseen,
newly revealed by the correct clip, and never observed. Every control uses these
same regions. Recall uses canonical distance <0.02; empty regions are null, and an
empty prediction has zero recall with undefined distance. Positive correct-minus-
control recall/IoU means video helps; negative distance delta means video helps.
These region recalls are completeness measures, not regional volumetric IoU or
proof of generalization. With all 100 clips used in training, their evaluation is
in-sample. The diagnostics are opt-in because four controls require four shape
samples per example. Full per-epoch records are included in reconstruction artifacts.

### Shared model extension points

All experiments use `ExperimentConfig` (chz), `train.py`, `evaluate.py`, the same
loss, checkpoint/resume machinery, data contracts, and media output. Components:

| Component | Owner | Configuration |
| --- | --- | --- |
| Dataset and preprocessing | `data/` | `data.*` |
| Reference conditioning and geometry model | `models/sam3d.py` | `sam3d.*`, `sampler.*` |
| Video/image token encoder | `models/encoders.py` | `encoder.kv_source`, `encoder.vjepa.*`, `encoder.dinov2.*` |
| Fusion or replacement pooler | `models/adapter.py`, `models/pooler.py` | `adapter.strategy`, `adapter.*` |
| Trainable LoRA and EMA state | `adapters.py`, `checkpoint.py` | `sam3d.lora`, `encoder.vjepa.video_lora` |
| Video comparisons and visibility-region scores | `diagnostics.py` | `eval.video_diagnostics`, `eval.video_controls` |
| Epoch reconstructions | `epochs.py` | `val.epoch_media_samples`, `eval.k` |

The `c2` preset already provides frozen DINO clip tokens, averaged in frame pairs,
through the same fusion path. A DINO-plus-pooler baseline is a configuration of
these factories, not another trainer. New encoders must return tokens with aligned
mask coverage and temporal offsets; new conditioning heads must integrate through
`apply_adapter`. Architecture-specific diagnostics explicitly report availability;
poolers do not pretend to have gated fusion blocks. Extend these interfaces and
parameter factories rather than copying an end-to-end model-specific pipeline.

## Generation

```bash
python -m video3d generation assets --source=gso --config=configs/generation/m0.json
python -m video3d generation generate --config=configs/generation/m0.json --split=eval --seeds='[0,1,2]'
python -m video3d generation export --dataset=/data/generated --split=eval --out=/data/export/eval --verify=1
```

The generator's existing module commands and scripts live under the shared package and
`scripts/generation/`. JSON presets retain the original settings. Datasets stay at configured
storage paths. Existing clips are skipped; explicitly replaced clips and interrupted
staging directories are preserved beside the replacement. Original generation documents
are under `docs/generation/`; older command paths in those historical documents refer to
the former standalone generator. The active commands above use the shared environment.

## Verification

```bash
uv sync --locked --group dev --group generation
uv run --no-sync pytest -q
ruff check src scripts/install_models.py scripts/verify_models.py scripts/verify_generation.py
```

Run the test suite on Linux remote compute because it includes tiny training integration
checks. `tests/test_loss_parity.py` compares the single copied production objective against
the installed upstream loss, including conditioning gradients and timestep sampling.
The source path and pinned commit are recorded directly below the copied function.
`tests/test_unified_pipeline.py` covers both strategies and all four LoRA combinations.
`scripts/verify_models.py` adds real checkpoint conditioning, upstream parity, backward,
checkpoint resume, sampling, metrics/media and a real small Blender render/export check.
Run the wrapper with `MODAL_PROFILE=mleagent VIDEO3D_VERIFICATION_ENV=/path/to/existing.env modal run scripts/verify_remote.py`; that environment file must provide approved Hugging Face access.
The Modal wrapper saves logs and reports on its verification volume and releases compute
when each bounded function ends.

Historical pilot results and notebooks are preserved. `docs/historical-pilot.md` describes
the retired pilot interface and is not the active setup guide.
