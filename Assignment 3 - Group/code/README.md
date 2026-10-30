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

```bash
python -m video3d eval /results/RUN paths.data_root=/data/export paths.output_root=/results paths.scratch_root=/scratch
python -m video3d eval zero-shot paths.data_root=/data/export
python -m video3d eval /results/RUN --repeated_reference=True
python -m video3d eval /results/RUN --shuffle_frames=True
python -m video3d eval /results/RUN --wrong_video=True
```

Evaluation uses EMA weights and paired deterministic seeds. Results resume per sample
and reject mismatched source/data/settings. Media pairs include input video, reference
RGB/mask, GT and prediction meshes. Empty predictions are explicit. `best-frame` is a
synthetic GT-visibility oracle. Canonical voxel IoU and canonical NAVI visibility-region
completeness are separate from bbox-normalized, similarity-ICP surface Chamfer/F-scores.
The Chamfer here sums the directional means. Historical pilot metrics used a different
protocol; do not pool those results. Resolved settings and metric protocol accompany
new results. `analyze --results='["/results/eval/RUN.jsonl"]'` produces summaries.

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

For deliberate occlusion, set `occlusion_mode="recoverable"` and generate two-object
clips. Object 1 is the target; object 2 is an occluder with scale controlled by
`occluder_scale_ratio`. Their bounding spheres remain disjoint. The sampler places
them along the reference viewing direction and searches their lateral offset using
the existing voxel precheck. Camera motion produces changing occlusion through parallax.
Defaults require 30–70% target occlusion at the middle reference, at least eight frames
with 20% occlusion, a frame with at least 75% target visibility, and recovery of at least
30% of the reference-hidden surface samples. These are pilot thresholds, not calibrated
training-distribution recommendations. The original `random` mode remains the default.

Accepted controlled clips store `meta.occlusion`, also retained in the training export:
area-uniform source-mesh samples, scene-depth visibility, reference target-alone visibility,
union coverage, additional coverage, best-frame gain and hidden-surface recovery. These
are finite-sample area estimates; depth agreement uses 1% of the target scale. Camera,
amodal and depth annotations are used only to construct and measure data.

`scripts/modal_generation.py` is a bounded A100 pilot launcher: shared CPU asset
preparation, ten `generate.map` inputs with up to ten GPU workers, persistent raw
artifacts and logs, and downloaded MP4/annotated previews. Each GPU input runs the
same portable `scripts/occlusion_pilot.py generate` command. Completed clips are reused
on retries, and providers release batch workers after the run. This small smoke test
uses cached object meshes and one photographic background rather than a diverse dataset.

## Dataset occlusion and complementary views

After exporting or downloading a synthetic split, measure every video with:

```bash
python -m video3d video-metrics --split_dir=/data/export/train --out=/results/video_metrics.json
```

This CPU diagnostic uses existing camera poses, packed surface voxels and dense rendered
instance masks. It does not load pretrained models, decode RGB videos or run inference.
It writes JSON with every frame/object measurement plus `.clips.csv` and `.objects.csv`.
The default reference is frame `T // 2`; `--reference=0` selects the first frame instead.
Per-object JSON also includes gains/recovery for every possible reference and a separate
summary using that object's most occluded frame. These are dataset diagnostics, not inputs
to the adapter.

Pixel occlusion is exact for the rendered in-image silhouette:
`occlusion = 1 - visible_px / amodal_px`. Every frame's visible pixel counts are checked
against its instance mask. The report includes mean/max/reference occlusion and the number
of frames with at least 10% occlusion; 10% is a descriptive threshold, not a rejection rule.

For complementary evidence, each occupied surface voxel has a persistent identity across
frames. Rays from its centre towards the camera traverse the object's 64³ grid to reject
self-occluded samples. Native-resolution instance masks identify external occluders.
Let `S_t` be the resulting visible voxel set and `U` its union across the clip. The proxy
metrics are `|U|/N`, additional coverage `|U minus S_ref|/N`, and best-frame gain
`(|U| - max_t |S_t|)/N`. External-occlusion recovery is the fraction of reference samples
hidden by other objects that become visible elsewhere. Recovery is null if no reference
samples are externally occluded. Exact zero pixel occlusion overrides coarse silhouette
discrepancies, so unoccluded objects cannot acquire false external occlusion.

These geometry metrics are explicitly **voxel-count proxies**, not true mesh-surface area.
The training export omits meshes, depth and amodal mask images, although it retains exact
amodal pixel counts. Quantization and projected silhouette boundaries affect the proxies.
Image truncation is not counted as external occlusion. Clip summaries weight objects
equally; external recovery averages only objects with defined recovery.

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

Verified results: [2026-10-29 verification record](docs/verification/2026-10-29.md).
