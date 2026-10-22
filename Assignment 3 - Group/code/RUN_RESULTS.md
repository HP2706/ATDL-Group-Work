# View coverage pilot results

Completed on 2026-10-22. Source commit: `d1cee366ecd246b8639efd63bf813f3408009aa9`. UCloud job `12429369`, group `view-coverage-gd1cee366ecd2-2026-10-22-11-33-45`. All four variants completed 25 optimizer updates, mesh evaluation, and checkpoint/artifact logging. The batch exited with code 0 and UCloud reached terminal SUCCESS at 11:48:38 Europe/Copenhagen, releasing its GPU. The successful allocation lasted 16 minutes 49 seconds; all four allocations, including failed preparation attempts, totaled approximately 43 minutes 17 seconds, below the authorized two-hour budget.

## Configuration and evidence

See [the run specification](RUN_SPEC.md) for pinned models, objective, dataset and batch details. Actual split: 24/4/4 video-capable objects, with two training and two held-out validation objects active. Validation objects: `pumpkin_showpiece_s` and `welcome_sign_mushrooms`. The target is coarse canonical object geometry, with input masks supplied. This is a small pipeline pilot.

One B200 MIG 1g.23gb, measured capacity 22,011,707,392 bytes. Effective batch four, physical microbatch four, no accumulation. Rank-8 LoRA configurations: A connector only; B adds video LoRA; C adds SAM LoRA; D adds both. Loaded parameter totals range from 1,180,552,134 to 1,182,370,758.

Persistent source artifacts are in UCloud Member Files `/12398691/hp-video/results/view_coverage/`. Local ignored copies of result JSON, per-step logs, per-example evaluation and GPU profiling are in `results/view-coverage-gd1cee366ecd2-2026-10-22-11-33-45/`. W&B has six paired reconstruction artifacts and one checkpoint artifact per variant. Each reconstruction includes input MP4, reference image, ground-truth GLB, predicted GLB, occupancy logits, sampled prediction points, visibility regions, and pairing JSON. No reconstruction failed export (24/24).

## Measurements

| Variant | Trainable parameters | Training input frames/s | Median update seconds | Repeated-reference Chamfer | Video Chamfer |
|---|---:|---:|---:|---:|---:|
| [A](https://forge.coreweave.com/wandb/hprjdk/atdl-video-to-3d/runs/view-coverage-gd1cee366ecd2-2026-10-22-11-33-45-A) | 787,456 | 12.73 | 5.02 | 0.07351 | 0.06115 |
| [B](https://forge.coreweave.com/wandb/hprjdk/atdl-video-to-3d/runs/view-coverage-gd1cee366ecd2-2026-10-22-11-33-45-B) | 1,229,824 | 11.06 | 5.78 | 0.06990 | 0.06112 |
| [C](https://forge.coreweave.com/wandb/hprjdk/atdl-video-to-3d/runs/view-coverage-gd1cee366ecd2-2026-10-22-11-33-45-C) | 2,163,712 | 12.09 | 5.27 | 0.07634 | 0.06155 |
| [D](https://forge.coreweave.com/wandb/hprjdk/atdl-video-to-3d/runs/view-coverage-gd1cee366ecd2-2026-10-22-11-33-45-D) | 2,606,080 | 10.53 | 6.05 | 0.07694 | 0.07225 |

Training throughput uses 64 input-frame exposures per optimizer update divided by mean recorded update time for updates 2–25. It includes frame loading and training, excludes the first update, and excludes validation/checkpoint/evaluation time. These are sampled-frame exposures, not original-video playback speed or per-frame 3D outputs. Each generated object consumes one 16-frame clip.

The separate two-step, both-LoRA GPU probe measured 13.72 input frames/s at microbatch four, excluding batch preparation, with peak allocated memory 5.64 GB and peak reserved memory 6.13 GB. Microbatch two measured 13.56 frames/s; the difference is too small for a firm efficiency conclusion from two samples. The physical batch stopped at four because that is the approved effective batch, rather than because of memory capacity. The microbatch-one result includes substantial first-use overhead. No isolated mesh-generation latency was measured.

## Quality interpretation

Chamfer is the mean of bidirectional, unsquared nearest-surface distances in the normalized object frame; lower is better. The untouched native image-only baseline scored **0.03542 Chamfer**, **0.32138 F-score at distance 0.02**, and **34.84% newly revealed-surface recall at 0.02**. It outperformed all four adapted video models on these metrics. Full-video Chamfer improved over the adapted repeated-reference control by 6.1–19.4%, depending on configuration, but this is not evidence of improvement over the pretrained model.

| Variant | Newly revealed recall, repeated reference | Newly revealed recall, video |
|---|---:|---:|
| A | 14.06% | 16.47% |
| B | 15.47% | 17.64% |
| C | 14.70% | 16.95% |
| D | 15.18% | 13.43% |

The additional-view signal is therefore mixed: A/B/C improve newly revealed recall over their repeated-frame controls, whereas D decreases it. Two validation objects, 25 updates, and a single seed do not support ranking adapter strategies or claiming a general video benefit. Canonical scan orientation relative to SAM remains an interpretation limitation; normalization alone does not prove rotational alignment. Inspect paired reconstructions before changing the architecture or scaling training.

## Integration corrections validated during execution

The dataset has 32 video-capable objects, rather than the originally assumed 36. NAVI camera transforms and intrinsics mix NumPy/Torch types and float64/float32, so visibility computation normalizes them to NumPy float64. NAVI's tutorial uses positive camera Z and positive image Y; both projection conventions are now matched. A real scan/depth-frame check found 1,575 visible points out of 4,096 samples. BF16 SAM execution preserves the upstream FP32 normalization parameters, and backward profiling explicitly enables training mode.

Seventeen lightweight contract tests pass, including an actual upstream multimodal transformer block with BF16 adapters and backward propagation. The full GPU runs also passed gradient-path checks for every enabled trainable component, finite-loss checks, dataset preparation, checkpoint writes, reconstruction generation and W&B upload.

## Next decision

Inspect the paired native-image, repeated-reference and full-video reconstructions, then test a connector initialization or gated residual conditioning that preserves native-image behavior at initialization. Reuse this pipeline for a longer single-configuration overfit/calibration test before expanding the dataset or comparing LoRA strategies. No additional compute has been allocated.
