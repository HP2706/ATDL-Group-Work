# Video-conditioned 3D reconstruction research log

Last reconciled: 2026-10-24. This log records completed work and its limits; the linked run reports retain the detailed measurements and artifact locations.

On 2026-10-24, the training branch was consolidated into `main`. Local contract verification passed 18 tests; six tests were skipped because optional upstream source trees were unavailable. This local check does not repeat the earlier remote GPU runs. Local result copies, datasets, notebook assets, and environments were preserved in the main checkout.

## Research question and controls

Does a masked object video improve coarse canonical geometry, particularly newly revealed surfaces, over one reference image? The pipeline reuses pretrained SAM 3D geometry generation and a V-JEPA video encoder. Controls are untouched native image-only SAM 3D, an adapted model receiving a repeated reference frame, and the same adapted model receiving a full video.

## 2026-10-22: Averaged video-conditioning pilot

Source: `d1cee366ecd246b8639efd63bf813f3408009aa9`. UCloud job `12429369`. Four configurations completed 25 updates each: A connector only, B with video LoRA, C with SAM LoRA, D with both. Two training and two held-out validation objects were active; the underlying video-capable object split was 24/4/4. Each clip supplied 16 frames. Test objects were not evaluated.

| Configuration | Repeated-reference Chamfer | Full-video Chamfer | Newly revealed recall, reference/video |
|---|---:|---:|---:|
| A | 0.07351 | 0.06115 | 14.06% / 16.47% |
| B | 0.06990 | 0.06112 | 15.47% / 17.64% |
| C | 0.07634 | 0.06155 | 14.70% / 16.95% |
| D | 0.07694 | 0.07225 | 15.18% / 13.43% |

Untouched native image-only SAM 3D achieved 0.03542 Chamfer, 0.32138 F-score at 0.02, and 34.84% newly revealed recall. Lower Chamfer is better; higher F-score and recall are better. Video improved Chamfer over each adapted repeated-frame control, but every adapted model remained worse than the native image baseline. Newly revealed recall improved for A/B/C and decreased for D. This does not establish a general benefit from video or rank the adapter strategies.

All 24 reconstructions exported successfully. Training throughput was 10.53–12.73 input-frame exposures/s. Integration fixes covered the actual 32 video-capable objects, mixed camera-array types, NAVI positive-Z/positive-image-Y projection, SAM FP32 normalization under BF16, and profiling in training mode. Three failed preparation allocations preceded the successful batch; total allocation time was about 43 minutes 17 seconds. The successful job terminated and released its GPU.

Evidence: [complete results, W&B links, and persistent artifact locations](code/RUN_RESULTS.md), [initial specification](code/RUN_SPEC_INITIAL.md).

## 2026-10-22: Mask-aware attentive-pooler pilot

Source: `32f534db266237125ea7d75be8dec51301509c88`. UCloud job `12429715`. The official attentive pooler used 64 learned queries and foreground tubelet masks; its connector replaced native RGB conditioning while native mask conditioning remained. The same A/B/C/D optional-LoRA configurations completed 25 updates. Pooling, RGB replacement, and video preprocessing changed together, so the comparison cannot isolate their individual effects.

| Configuration | Repeated-reference Chamfer | Full-video Chamfer | Full-video F-score at 0.02 |
|---|---:|---:|---:|
| A | 0.090681 | 0.103101 | 0.133471 |
| B | 0.090678 | 0.102823 | 0.139235 |
| C | 0.090245 | 0.102898 | 0.138128 |
| D | 0.090629 | 0.102817 | 0.139259 |

Fixed training losses fell from about 0.547 to 0.483 and validation losses from about 0.950 to 0.920. Geometry nevertheless regressed against both the native baseline (0.035421 Chamfer) and the earlier averaging pilot (best video Chamfer 0.061122). The earlier loss trace used a different validation procedure and is not directly comparable. Every variant generated nondegenerate meshes. Training throughput was 10.64–12.23 input-frame exposures/s.

All 24 contract tests passed at the recorded run, all enabled trainable groups had finite nonzero remote-GPU gradients, all 24 paired reconstruction artifacts contained the required files, and each run exposed 22 direct media panels. The foreground job exited successfully and its provider allocation ended automatically.

Evidence: [complete pooler results and W&B links](code/RUN_RESULTS_POOLER.md), [run specification](code/RUN_SPEC.md), [architecture](architecture.md), [comparison notebook](code/notebooks/compare_reconstructions.ipynb).

## Interpretation and next experiment

Both runs are single-seed integration pilots with only two validation objects and 25 updates. Canonical scan/SAM rotation remains unresolved; normalization does not establish alignment. The data demonstrate a working training/evaluation pipeline and an observed regression, rather than a successful reconstruction improvement.

The next controlled experiment should preserve native RGB/mask conditioning and add video through a zero-initialized gate, checking equality to the native baseline at initialization. A longer single-configuration overfit/calibration test should precede scaling or ranking LoRA choices. These are proposals; no additional experiment is recorded as launched.
