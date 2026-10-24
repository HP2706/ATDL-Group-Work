# Mask-aware pooler pilot results

Completed on 2026-10-22, UCloud job 12429715, source `32f534db266237125ea7d75be8dec51301509c88`. All four variants completed 25 updates, final evaluation, checkpointing, and W&B synchronization. Provider terminal state and the foreground script exit code were verified; the allocation ended automatically.

## Result

The mask-aware attentive-pooler pipeline runs end to end, but its measured geometry is worse than both untouched native image-only SAM 3D and the previous averaging pilot. Fixed training-probe and validation flow losses decrease from initialization, so lower loss did not imply better generated geometry in this pilot.

## Geometry and run links

Symmetric unsquared Chamfer distance on 4096 surface samples, normalized object coordinates; lower is better. Two held-out objects, same generation seed and 25 sampler steps. Native image-only baseline: **0.035421**.

| Variant | Repeated-reference Chamfer | Video Chamfer | Video F-score at 0.02 | New run | Earlier averaging run |
|---|---:|---:|---:|---|---|
| A | 0.090681 | 0.103101 | 0.133471 | [A](https://forge.coreweave.com/wandb/hprjdk/atdl-video-to-3d/runs/mask-pooler-g32f534db2662-2026-10-22-14-07-23-A) | [A](https://forge.coreweave.com/wandb/hprjdk/atdl-video-to-3d/runs/view-coverage-gd1cee366ecd2-2026-10-22-11-33-45-A) |
| B | 0.090678 | 0.102823 | 0.139235 | [B](https://forge.coreweave.com/wandb/hprjdk/atdl-video-to-3d/runs/mask-pooler-g32f534db2662-2026-10-22-14-07-23-B) | [B](https://forge.coreweave.com/wandb/hprjdk/atdl-video-to-3d/runs/view-coverage-gd1cee366ecd2-2026-10-22-11-33-45-B) |
| C | 0.090245 | 0.102898 | 0.138128 | [C](https://forge.coreweave.com/wandb/hprjdk/atdl-video-to-3d/runs/mask-pooler-g32f534db2662-2026-10-22-14-07-23-C) | [C](https://forge.coreweave.com/wandb/hprjdk/atdl-video-to-3d/runs/view-coverage-gd1cee366ecd2-2026-10-22-11-33-45-C) |
| D | 0.090629 | 0.102817 | 0.139259 | [D](https://forge.coreweave.com/wandb/hprjdk/atdl-video-to-3d/runs/mask-pooler-g32f534db2662-2026-10-22-14-07-23-D) | [D](https://forge.coreweave.com/wandb/hprjdk/atdl-video-to-3d/runs/view-coverage-gd1cee366ecd2-2026-10-22-11-33-45-D) |

All modes in all variants produced nondegenerate meshes: failure fraction zero. This scalar is what `eval/video/failed` measures. It is not a video media key. The previous best video Chamfer was 0.061122; none of the new variants improves it. Scan/SAM canonical rotation remains unresolved, so these scores are limited to the existing evaluation convention.

## Fixed losses and measured throughput

Validation averages every held-out clip with a deterministic seed per clip. One training clip forms a separate fixed probe. Measurements at steps 0, 5, 10, 15, 20 and 25 preserve training RNG state. The previous pilot used only one validation clip, so its loss trace is not directly comparable.

| Variant | Fixed training loss, initial to final | Validation loss, initial to final | Training input frames/s | Trainable parameters |
|---|---|---|---:|---:|
| A | 0.547091 to 0.483712 | 0.949930 to 0.921232 | 12.23 | 7,333,888 |
| B | 0.547290 to 0.483582 | 0.949787 to 0.919941 | 10.75 | 7,776,256 |
| C | 0.547091 to 0.483083 | 0.949930 to 0.920554 | 12.08 | 8,710,144 |
| D | 0.547290 to 0.482993 | 0.949787 to 0.919924 | 10.64 | 9,152,512 |

Training throughput includes input decoding/preprocessing and forward/backward/optimizer work, but excludes validation, final sampling, setup and uploads. The short GPU-only profile selected physical batch four, accumulation one, at 16.39 input frames/s and 6.41 GB peak reserved device memory. The first batch-one profile includes cold-start overhead and is not a steady-state comparison.

## Inspecting the objects

Every new run has direct panels under `final/<example>/`: `input_video`, `reference_rgb`, `reference_mask`, `ground_truth_mesh`, `native_sam3d_coarse_mesh`, `our_repeated_reference_mesh`, and `our_video_mesh`. These are actual GLB surfaces, not only sampled point clouds. The separately labeled comparison table still uses point clouds because the SDK rejects GLB Object3D inside artifact-backed tables.

`training_assets/<object>/` shows both training-object meshes, reference images/masks and representative training videos. A separate training-assets artifact contains both ground-truth GLBs and encoded geometry targets. Each of six reconstruction artifacts pairs its prediction GLB with input MP4, reference RGB/mask, ground-truth GLB, occupancy logits, visibility labels and pairing/metric JSON.

Remote checks confirmed 22 direct media panels per run: four videos, ten GLB meshes and eight images. All 24 reconstruction artifacts contain the required files. A downloaded evaluation MP4 decoded as 16 frames over four seconds at 512 square, with different first and last frames. Downloaded native, adapted and ground-truth files passed GLB header and scene-mesh checks. The model itself receives 16 frames letterboxed to 384 square.

## Scope and interpretation

The implementation reuses upstream SAM 3D model, VAE, flow-matching loss and sampler, plus the official V-JEPA attentive-pooler implementation. All enabled trainable groups had finite nonzero gradients on the remote GPU. All 24 lightweight contract tests passed. No training or GPU benchmark ran on the laptop.

The pooler has 64 fresh learned queries and 6,546,432 trainable parameters; its connector adds 787,456. A/B/C/D vary only the optional video and SAM LoRAs. Reference masks remain as native conditioning, per-frame masks select video tubelets, and native RGB streams are replaced. Two training objects and two validation objects constitute an integration pilot. Test objects remain untouched.

Compared with the previous pilot, pooling, RGB replacement and video preprocessing changed together. This run cannot isolate the cause of the regression. The next controlled ablation should retain native RGB/mask conditioning and add the pooler through a zero-initialized gate, so initialization reproduces the native baseline. This is a proposed experiment, not another launched allocation.

## Evidence

Persistent UCloud root: `/12398691/hp-video`. Sweep results, per-variant logs/checkpoints/configuration and evaluation outputs are under `results/view_coverage/`. `current-sweep.json` marks completion and `job-exit.json` records exit code zero. Local downloaded evidence is ignored under `code/results/mask-pooler-g32f534db2662-2026-10-22-14-07-23/`, including `wandb-verification.json`. Source and evaluation limitations are described in [architecture.md](../../architecture.md) and [the run specification](RUN_SPEC.md).
