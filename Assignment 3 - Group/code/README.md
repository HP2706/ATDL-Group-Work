# Video view coverage pilot

Question: does additional visual coverage improve canonical object geometry over one reference frame, especially on surfaces newly revealed by the video?

This is a short pipeline pilot, not a furniture benchmark or a segmentation training run. NAVI supplies real object videos, masks, camera/depth annotations, and aligned scans. Input masks are provided; the output is a coarse 3D occupancy/mesh reconstruction of the selected object.

The implementation imports the pinned upstream SAM 3D generator, sparse-structure encoder/decoder, sampler, and `ShortCut.loss`. It does not reimplement them. A V-JEPA 2.1 encoder supplies video tokens to its official attentive pooler, restricted to foreground tubelets by aligned input masks. Its 64 summaries pass through a learned 768→1024 connector and replace native RGB conditioning while retaining native mask conditioning. See [architecture.md](../../architecture.md). Optional rank-8 LoRA adapters affect video attention and SAM shape cross-attention.

| Variant | Pooler + connector | Video LoRA | SAM LoRA |
|---|---|---|---|
| A | train | frozen | frozen |
| B | train | train | frozen |
| C | train | frozen | train |
| D | train | train | train |

All variants train on videos. Each is evaluated with the native image-only generator, the reference frame repeated through the video encoder, and the full video. The repeated-frame control isolates added views while holding the adaptation path fixed.

## Entry points

Install using `uv sync --extra gpu` on the remote machine. Configuration uses nested chz overrides through Google Fire:

```bash
python -m video3d.cli inspect training.steps=25
python -m video3d.cli prepare
python -m video3d.cli profile training.group=<group>
python -m video3d.cli sweep training.group=<group>
```

Use `ucloud/batch.sh` as the Terminal Ubuntu application's **Batch processing** file parameter. The foreground script downloads dependencies/data, prepares targets, profiles, runs A/B/C/D sequentially, persists results, and exits. UCloud then ends the allocation. `ucloud/job.json` specifies one small B200 MIG and a two-hour provider limit. Its staging paths belong to this project's Member Files directory.

`runtime.env` is private, ignored, and staged separately. It supplies `HF_READ_TOKEN`, `WANDB_API_KEY`, `HP_SOURCE_COMMIT`, and `HP_TRAINING_APPROVED=1` after approval of [the first-run specification](RUN_SPEC.md). No credentials are committed.

## Reuse and scope

Only canonical coarse shape receives loss. Pose/layout losses have zero weight; their targets are neutral placeholders. Shortcut self-consistency is disabled through upstream configuration, leaving the upstream flow-matching objective. The refinement, texture, scene placement, and mask-prediction stages are outside this pilot.

Two integration adjustments are recorded: the upstream package initializer skips unused sparse refinement imports in geometry-only mode, and the upstream video loader's testing localhost URL is replaced with its official public checkpoint URL. Model and loss implementations remain upstream.

## Outputs and recovery

Persistent UCloud directory: `/work/hp-video`. It contains dataset/cache/dependency directories, `batch.log`, `job-exit.json`, and `results/`. W&B destination: `hprjdk/atdl-video-to-3d`. Each reconstruction artifact pairs the input MP4 and reference image with the predicted GLB, occupancy logits, sampled point cloud, ground-truth mesh, visibility regions, and JSON identifiers/metrics.

Checkpoints retain trainable weights, optimizer/scheduler, all RNG states, progress, resolved configuration, provenance, and validation history. Saves are atomic; an interrupted accumulation preserves the last complete update. A resubmission with the same source commit resumes the incomplete group and reuses profiling; completed variants are skipped. Changed configurations fail loudly. Caches and datasets are preserved.

## Verification status

Lightweight contract tests exercise adapter gradients, upstream loss backpropagation, checkpoint/RNG restoration, split isolation, controls, mesh export, and offline W&B video/3D serialization. The remote GPU pilot completed all four configurations, including pretrained-model loading, preparation, profiling, training, mesh evaluation and W&B artifact uploads. See [validated pilot results](RUN_RESULTS.md); the adapted video models did not outperform the untouched image-only baseline. Preparation records VAE round-trip and visibility diagnostics and rejects degenerate geometry or completely invisible scan points.
