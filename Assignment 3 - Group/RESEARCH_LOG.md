# Video-conditioned 3D reconstruction research log

Last reconciled: 2026-10-29. This log records completed work and its limits; the linked run reports retain the detailed measurements and artifact locations.

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

## 2026-10-29: Matched epoch 1 and epoch 10 reconstruction diagnostics

The synthetic-data fusion variants and DINO attention pooler were compared with
untouched pretrained SAM 3D on the same ten in-sample reference frames, using
matched generation seeds and four sampling steps. Flow loss below is matched
evaluation loss, not epoch-average training loss. These are overfit/in-sample
diagnostics, not evidence of held-out generalization.

### Epoch 1

| Model | Flow loss ↓ | Canonical voxel IoU ↑ | Empty predictions |
|---|---:|---:|---:|
| Pretrained SAM 3D | 0.8074 | 0.1371 | 0/10 |
| Fusion adapter | 0.8074 | 0.1371 | 0/10 |
| Fusion + SAM LoRA | 0.8072 | 0.1372 | 0/10 |
| Fusion + SAM LoRA + V-JEPA LoRA | 0.8072 | 0.1372 | 0/10 |
| DINO attention pooler | 0.6796 | 0.0324 | 6/10 |

### Epoch 10

| Model | Flow loss ↓ | Canonical voxel IoU ↑ | Empty predictions |
|---|---:|---:|---:|
| Pretrained SAM 3D, unchanged baseline | 0.8074 | 0.1371 | 0/10 |
| Fusion adapter | 0.8006 | 0.1255 | 0/10 |
| Fusion + SAM LoRA | 0.7749 | 0.1233 | 0/10 |
| Fusion + SAM LoRA + V-JEPA LoRA | 0.7749 | 0.1237 | 0/10 |
| DINO attention pooler | 0.3472 | 0.0729 | 0/10 |

The fusion variants were essentially unchanged after one epoch. After ten
epochs all adapted models reduced matched flow loss but remained below the
pretrained baseline in canonical overlap. DINO recovered from six empty
predictions at epoch 1 to none in its final ten previews, but its final IoU
remained lower than both pretrained SAM 3D and the fusion variants.

The canonical scan/SAM orientation convention remains a limitation of these
scores. Do not compare these voxel-IoU diagnostics directly with the earlier
NAVI Chamfer tables. The rounded values were recovered from the earlier results
conversation and checked against the original saved summary JSON on the new
UCloud B200 workstation. Both summaries contain exactly the same ten sample IDs.
No training or checkpoint inference was repeated to recover these results.

Evidence: [matched comparison](https://forge.coreweave.com/wandb/hprjdk/atdl-video-to-3d/runs/sam3d-exact-preview-comparison-2026-10-29-19-43-12),
[DINO training run](https://forge.coreweave.com/wandb/hprjdk/atdl-video-to-3d/runs/dino-frame-attention-pooler-2026-10-29-19-29-45),
and [recovered table and protocol](code/docs/epoch-comparison-2026-10-29.md).

## 2026-10-29: Installed multi-view baselines verified on UCloud

Added importable coarse-stage MV-SAM3D and Stream3D integrations, using pinned
upstream libraries in mutually exclusive uv environments. Both use the SAM 3D
backend and the shared dataset, frame selection, sample identity, seeds and
geometry evaluator. No upstream source repository is nested inside the project.

Both completed real checkpoint inference on the same four-frame training sample
with four shortcut steps on UCloud B200 job
[12439107](https://cloud.sdu.dk/app/jobs/properties/12439107).
MV-SAM3D produced 11,192 occupied cells; Stream3D produced 16,561. Stream3D
processed two chunks and selected frame positions 0 and 3 at the end, retaining
evidence from the first chunk. Both produced nonempty 64³ outputs and completed
the shared metrics. This verifies integration only; one sample with different
view-selection budgets does not rank the methods.

The ten-hour interactive workstation remains available until 30 October,
07:56 CET. The task-created Modal apps are stopped. Evidence, source revisions,
runtime details and limitations are in the
[verification record](code/docs/baseline-verification-2026-10-29.md).

## Interpretation and next experiment

Both runs are single-seed integration pilots with only two validation objects and 25 updates. Canonical scan/SAM rotation remains unresolved; normalization does not establish alignment. The data demonstrate a working training/evaluation pipeline and an observed regression, rather than a successful reconstruction improvement.

The October 29 diagnostics extend the earlier short pilots with ten-epoch
in-sample checks. Lower flow loss still does not establish improved generated
geometry. The next comparison should resolve canonical conventions, measure
aligned geometry alongside canonical overlap, and evaluate held-out objects
against native SAM 3D and the coarse SAM 3D backends of MV-SAM3D and Stream3D.

## 2026-10-29: GSO random-250 full-mesh baseline jobs submitted

GSO is the held-out evaluation source; OmniObject3D remains the training source.
Froze a random, without-replacement 250-object subset of the official 1,030-object
GSO collection with seed 42. Reused the dataset task's completed 32-frame renders
instead of producing a duplicate dataset. The selected data and all visibility
caches passed their audit; ten integration tests passed on UCloud.

Submitted three separate B200 batch jobs: native SAM 3D
[12439137](https://cloud.sdu.dk/app/jobs/properties/12439137), MV-SAM3D
[12439138](https://cloud.sdu.dk/app/jobs/properties/12439138), and Stream3D
[12439139](https://cloud.sdu.dk/app/jobs/properties/12439139). These are running or
queued experiments, with no completed aggregate results claimed here. All use
25-step full refinement plus saved coarse outputs. Selected frames, sample seeds,
GT points and visibility regions are matched. Exact geometry, shared geometry
metrics, per-point surface errors, poses/latents, fusion diagnostics and resource
measurements are persisted per object. Training flow loss is explicitly not
computed by this generated-geometry runner.

The full protocol, artifact locations, caveats and validation evidence are in
[the GSO-250 run record](code/docs/evaluations/gso-250-2026-10-29.md).
