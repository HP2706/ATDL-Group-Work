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

GSO is the held-out evaluation source. OmniObject3D was the training source for this launch plan; the October 30 training plan below replaces it with the full synthetic export.
Froze a random, without-replacement 250-object subset of the official 1,030-object
GSO collection with seed 42. Reused the dataset task's completed 32-frame renders
instead of producing a duplicate dataset. The selected data and all visibility
caches passed their audit; ten integration tests passed on UCloud.

Submitted three separate B200 batch jobs: native SAM 3D
[12439137](https://cloud.sdu.dk/app/jobs/properties/12439137), MV-SAM3D
[12439138](https://cloud.sdu.dk/app/jobs/properties/12439138), and Stream3D
[12439139](https://cloud.sdu.dk/app/jobs/properties/12439139). All three subsequently completed and validated 250 objects each; final aggregates are recorded below. All use
25-step full refinement plus saved coarse outputs. Selected frames, sample seeds,
GT points and visibility regions are matched. Exact geometry, shared geometry
metrics, per-point surface errors, poses/latents, fusion diagnostics and resource
measurements are persisted per object. Training flow loss is explicitly not
computed by this generated-geometry runner.

The full protocol, artifact locations, caveats and validation evidence are in
[the GSO-250 run record](code/docs/evaluations/gso-250-2026-10-29.md).


## 2026-10-30: Completed GSO-250 baselines and launch recovery

All three native batch jobs finished with a validated 250-record result file and
zero empty predictions. Every method was scored on the same randomly selected
GSO objects. SAM 3D receives one reference image; the two video methods receive
the matched four selected frames. These input budgets must remain explicit.

| Method | Canonical voxel IoU ↑ | Aligned voxel IoU ↑ | Aligned refined-mesh F-score at 0.02 ↑ | Aligned refined-mesh Chamfer L1 sum ↓ |
|---|---:|---:|---:|---:|
| SAM 3D | 0.208078 | 0.251989 | 0.629658 | 0.074598 |
| MV-SAM3D | 0.249300 | 0.298237 | 0.729998 | 0.047774 |
| Stream3D | 0.251513 | 0.301418 | 0.740809 | 0.046882 |

Refined-mesh distances use area-uniform surface samples and exact closest-triangle
distances, with the GT normalized longest side equal to one and single-start
similarity ICP. They are different from the coarse voxel-surface Chamfer used
by training validation, whose normalized longest side is two. Canonical
registration and globally optimal alignment are not established. Stream3D and
MV-SAM3D improve these aggregate scores over the single-image baseline; this
single-seed comparison does not establish significance or isolate view budget.

Evidence: persistent root
`/work/MathiasElmegaardNielsen#6181/gso-baselines-250-2026-10-29-22-33-26`,
method-specific `results/<method>/evaluation.jsonl` and `complete.json`.
All 750 existing records were mirrored to W&B, including every numeric metric,
per-object tables, means/counts, completion status, and downloadable result files;
server-side checks confirmed 250 objects per method and matching IoU aggregates.

The separate 100-video Omni smoke job `12439140` completed one epoch/four updates
and a 10-object GSO panel. Fusion scored 0.2864305 canonical IoU versus native
SAM 3D 0.2864894; matched evaluation flow losses were 0.7348302 and 0.7348146.
This is an execution test with essentially unchanged geometry, not a result on
the 250-object panel. Checkpoint saving and strict automatic resume passed.

The delayed coordinator `12439166` started target preparation `12439197` at the
requested time. It cached 4,318 Omni training targets, 490 validation targets and
250 GSO targets, then profiled the fusion variants. Pooler profiling stopped on
`strawberry_001`: maximum foreground tubelet coverage was 0.49106, below the
configured 0.5 threshold. Consequently no full training jobs were submitted.
Recovery `12439297` completed all six original architecture profiles after the
pooler was fixed to retain the most-covered positive-coverage tubelet(s) when
no token reaches the threshold, excluding padding/background and still rejecting
empty foreground. Batch sizes 8/16/32 were measured; all six selected physical
batch 32 at effective batch 32. These measurements use Omni, not the replacement
synthetic training dataset.

Loss-only W&B evaluation runs from October 29 contain 43 fixed-grid losses each,
not training trajectories. Their three curves differ by at most about 0.00027
and visually overlap. One run was marked failed and lacked its final table.
The shared evaluator now logs every computed finite scalar, full per-object
metric tables, aggregate means and valid counts, explicit partial/full completion,
and result artifacts. Evaluation uses a separate sample axis rather than resetting
the optimizer-step axis. No uncomputed geometry metric was synthesized for a
loss-only run. The shared training validation now records canonical/aligned IoU,
Chamfer, F-scores, empty-prediction counts and per-object JSON on the same decoded
outputs; BF16 training contexts are disabled around FP32 ICP/KNN scoring.

## 2026-10-30: Full-q98 training plan and run priority

Training switches to Harald's immutable full-q98 synthetic export, revision
`16f95cb82a2fbb8b50f866ca2c711bc1146ff223`, at
`/work/MathiasElmegaardNielsen#6181/sam3dvid-full-q98/snapshot/full-q98`.
The native reader sees 10,000 clips. Existing CRC32 asset splitting gives 9,358
training clips / 4,378 target assets / 20,716 clip-object training samples and
642 validation clips / 131 target assets / 1,306 reference samples. Train/val
clips and target assets do not overlap. Sample decoding was verified at the
first, middle and final training and validation positions. Source indices,
explicit split maps, and audit evidence are retained without copying shards.

The requested run length is now **five epochs**, with **200 final GSO objects**
shared across methods. The 200-row evaluation manifest is a fixed prefix of the
previous frozen 250-row selection; the original selection and predictions are
preserved. At effective batch 32 there are 648 updates per epoch, 3,240 total,
including the exact final partial batch. Learning rate is 1e-4, warmup 162 updates,
EMA decay .99, seed 42, 16 frames per video. Physical batch is selected by bounded
GPU profiling while effective batch remains 32. Per-epoch validation uses the
same 50 held-out samples and includes the new Chamfer/F-score metrics. Epoch
checkpoints and recovery checkpoints every 50 updates preserve full optimizer,
scheduler, EMA, RNG, progress and provenance. Evaluation must not be reported as
completed training if a wall-time stop leaves fewer than 3,240 updates.

Vision-encoder fine-tuning is removed from the next matrix. The confirmed launch
order contains **six jobs**, with the pooling alternatives submitted last:

| Submission order | Run | Encoder | SAM 3D training |
|---|---|---|---|
| 1 | Fusion adapter | Frozen V-JEPA | Frozen |
| 2 | Fusion adapter + SAM LoRA | Frozen V-JEPA | LoRA |
| 3 | Native SAM 3D baseline | Native single-image conditioning | Frozen |
| 4 | Attention token pooling | Frozen V-JEPA | Frozen |
| 5 | Attention token pooling + SAM LoRA | Frozen V-JEPA | LoRA |
| 6 | Attention token pooling | Frozen DINOv2, all 16 frames | Frozen |

DINOv3 is explicitly excluded. DINOv2 pooling reuses SAM 3D's installed frozen
DINOv2 ViT-L/14 backbone and the existing attentive pooler; no second inference
pipeline is introduced. DINOv2 uses 224-pixel inputs, 14-pixel patches and
1,024-wide per-frame patch tokens, with CLS/register tokens excluded. The
V-JEPA configs retain their native video token dimensions and tubelet handling.
All variants use the same 16 selected video frames and retain native SAM mask
and pointmap conditioning. Pooling replaces selected native RGB tokens.

A bounded real-B200 check of training, checkpoint resume, Chamfer validation,
and W&B evaluation logging must pass before model fan-out. At this entry the
revised six jobs have not yet been submitted. Final job IDs and GPU verification
will be reconciled here after provider acceptance.

The launch is further prioritized: submit jobs 1–5 immediately after the shared
GPU check; DINOv2 pooling is delegated as a separate sixth job and must not delay
them. The native full-q98 GPU check completed five optimizer updates, restored
step 1 to step 5, and scored two held-out objects with finite Chamfer/F-scores
and no empty predictions. Its saved-output assertion initially used an unpadded
filename; the persisted file was correctly named `step-00000005.json`. The check
was corrected and a one-hour native GPU retry reuses the existing targets and
checkpoint to verify evaluation and W&B logging. These five-update metrics are
execution diagnostics, not final trained-model comparisons.

DINOv2 pooling contract tests verify all-frame patch ordering and foreground
coverage, exclusion of CLS/register tokens, frozen backbone gradients and a
trainable pooler/projector while retaining native conditioning. Its independent
frame encoding followed by pooling has no additional temporal position signal;
V-JEPA pooling receives temporally contextualized video embeddings. This
architectural distinction is not evidence of a performance difference.

### First five priority jobs accepted

The corrected real B200 check passed in native one-hour job
[12439361](https://cloud.sdu.dk/app/jobs/properties/12439361), which then finished
and released its allocation. It reloaded the completed five-update checkpoint,
verified two finite GPU shape scores, evaluated one GSO object, and uploaded W&B
scalars, media, tables and result artifacts. The partial one-object evaluation is
explicitly labelled incomplete against the 200-object final evaluation target.
Measured warm fusion throughput was 4.604 videos/s (6.951 seconds per effective
32-video update), projecting 6.26 training-only hours for five epochs. Validation,
profiling, setup and final evaluation are additional. Other architectures require
their own measurements; this is not a measured completion ETA for all methods.

The first five separate B200 jobs were submitted in the confirmed priority
order, using immutable source `de7f8d81362b79d1f4e77a323eb97f71a5fe8c80`:

| Order | Model | Native job | Provider state at acceptance check |
|---|---|---|---|
| 1 | V-JEPA fusion adapter | [12439375](https://cloud.sdu.dk/app/jobs/properties/12439375) | RUNNING |
| 2 | Fusion + SAM LoRA | [12439376](https://cloud.sdu.dk/app/jobs/properties/12439376) | RUNNING |
| 3 | Frozen SAM 3D | [12439377](https://cloud.sdu.dk/app/jobs/properties/12439377) | RUNNING |
| 4 | V-JEPA attention pooling | [12439378](https://cloud.sdu.dk/app/jobs/properties/12439378) | RUNNING |
| 5 | V-JEPA pooling + SAM LoRA | [12439379](https://cloud.sdu.dk/app/jobs/properties/12439379) | IN_QUEUE |

Training jobs profile physical batches 16/32 in their own allocations before
starting the five-epoch trainer, preserving effective batch 32. Variant-specific
profile files prevent concurrent runs from overwriting each other's progress.
Native foreground jobs release their allocation on pipeline exit; training jobs
have a 12-hour provider fallback, an 11-hour training-loop fallback, and an
explicit final-step completion guard before final evaluation. The baseline is a
separate bounded two-hour evaluation allocation. All use the same immutable
200-object GSO subset. Provider RUNNING may include setup/profiling; it does not
by itself certify that an optimizer update has completed.

Persistent experiment root:
`/work/MathiasElmegaardNielsen#6181/conditioning-full-q98-2026-10-30-08-40-00`.
Submission receipts are in `first-five-submitted.json`; resolved configs,
per-variant profiles, logs, checkpoints, validation records and predictions are
under that root. DINOv2 is handled separately after these five accepted jobs.

At the next provider check all four trainable priority allocations were RUNNING;
the native SAM 3D baseline `12439377` had completed with exit code zero. Fresh
full-q98 profiles selected batch 32 and measured 6.868 seconds/update for fusion,
7.271 for fusion + SAM LoRA, and 6.242 for V-JEPA pooling. Their respective
training-only five-epoch projections are 6.18, 6.54 and 5.62 hours; these short
profiles exclude validation and final evaluation and carry sampling uncertainty.
The separate DINOv2 one-hour real-GPU check is
[12439392](https://cloud.sdu.dk/app/jobs/properties/12439392). It does not change,
stop or delay the first five allocations. The sixth long DINOv2 run remains
conditional on that check passing.

### All six launches verified after GPU checks

The independent DINOv2 one-hour check `12439392` passed and exited zero at
2026-10-30 09:13:20 Europe/Copenhagen, releasing its GPU. It performed three real
optimizer updates with checkpoint resume from step 1 to 3, native shape
validation, one GSO evaluation, and W&B scalar/media/table/artifact uploads.
`dinov2-check-ready.json` records the PASS and finite IoU, Chamfer and F-scores;
these tiny checks are execution diagnostics, not scientific results. Two focused
CPU contract tests also passed. Native DINOv2 encodes all 16 frames into 4,096
patch tokens, excluding CLS/register tokens; the backbone remains frozen while
the 12.66M-parameter pooler/projection trains. Native mask/pointmap conditioning
is retained. The selected batch 32 profile used 29.13 GiB and 5.750 seconds per
update. A checkpointed warm measurement was 6.930 seconds/update, so the short
profile's 5.17-hour training-only projection is optimistic relative to the
6.24-hour warm estimate. Both exclude validation and final evaluation.

Only after the first five accepted launches and the DINOv2 PASS, the sixth
separate native foreground job was submitted:
[12439418](https://cloud.sdu.dk/app/jobs/properties/12439418). It uses the same
immutable `de7f8d8` source, five epochs/3,240 updates, effective batch 32, fixed
200-object GSO evaluation, 12-hour provider fallback and automatic release.
Its receipt is `dinov2-pooler-submitted.json` in the persistent experiment root.

All five trainable jobs were verified doing optimizer updates, beyond provider
RUNNING/setup: fusion `12439375` at step 61, fusion + SAM LoRA `12439376` at
step 56, V-JEPA pooler `12439378` at step 67, pooler + SAM LoRA `12439379` at
step 40, and DINOv2 pooler `12439418` at step 9. All five provider states were
RUNNING at this check. Live metrics are in `scratch/<run_name>/metrics.jsonl`;
the persistent `results/<run_name>/metrics.jsonl` copy advances on checkpoint
flushes every 50 updates. Missing pre-checkpoint copies do not imply missing
training logs. W&B receives the training history during the run.

Frozen SAM 3D `12439377` completed SUCCESS with its own batch exit code zero and
`results/eval/c0.summary.json` confirms 200/200 objects, complete=true:

| Metric | Frozen SAM 3D, GSO 200 |
|---|---:|
| Canonical voxel IoU | 0.211125 |
| Aligned voxel IoU | 0.258143 |
| Coarse-surface aligned Chamfer L1 sum | 0.181470 |
| F-score at 0.02 | 0.359412 |
| F-score at 0.05 | 0.787710 |
| Empty predictions | 0/200 |

These scores use the current coarse voxel-surface, longest-side-2 protocol and
200-object subset. They must not be ranked directly against the earlier
250-object refined-mesh, longest-side-1 results. No trained-model comparison is
available yet. The temporary CPU staging allocation `12439285` was released
after its consumers finished and verified terminal SUCCESS. The obsolete
seven-model delayed-launch follow-up was paused after all six confirmed jobs
were verified; it must not submit duplicate jobs. Active training allocations
were retained.

### Additional native SAM-only random-frame LoRA control

The coordinating task reports full training verified in native UCloud job
[12439491](https://cloud.sdu.dk/app/jobs/properties/12439491), using source
`3e3ee609035c3341e59722f6b1583055ab33c3be`. This additional control bypasses the
video encoder and adapter (`adapter.strategy=native`) and trains only 1,376,256
SAM LoRA parameters. The existing frame sampler selects a uniformly eligible
single training frame with `p_occluded_bias=0` and `clip_lengths=[1]`. Dataset
roots, cached targets, full-q98 train/validation split, five epochs/3,240 updates
and fixed 200-object GSO evaluation match the original six runs.

Thirteen CPU tests and a real GPU check passed before full training: three
optimizer updates, exact resume from step 1 to 3, changed SAM weights, two finite
validation Chamfer scores, and one GSO evaluation with W&B artifacts. The
coordinating task inspected full-run scratch metrics at step 4: finite loss
0.89598, SAM gradient 0.00504, encoder gradient zero, and zero adapter/video
trainable parameters. These early values establish execution, not a scientific
comparison. Batch profiling selected physical/effective batch 32, measuring
6.364 seconds/update and 30.64 GiB peak memory, projecting 5.73 training-only
hours; validation and final evaluation are additional.

Persistent root:
`/work/MathiasElmegaardNielsen#6181/sam-native-2026-10-30-09-49-38`.
Run name: `sam3d-random-frame-lora-2026-10-30-09-49-38`.
The source is isolated in `repo-native-3e3ee60`; existing six-job sources and
allocations were not modified. Earlier attempt `12439484` stopped before GPU
training because its archive lacked Git metadata required by a configuration
test. Its script exit was 1 despite provider SUCCESS. The retry uses a dedicated
Git checkout from a bundle with unchanged model source. Earlier artifacts were
preserved, the failed allocation released, and only the retry trains the control.

### Live checkpoint comparison, 2026-10-30 11:13 CET

All six trainable native jobs are RUNNING and have completed their first epoch
validation at update 648 on the fixed 50 held-out synthetic objects. Their full
200-object GSO evaluations have not run yet. These are interim results:

| Model | Current update / 3240 | Epoch-1 val loss | IoU | Aligned IoU | Chamfer ↓ | F-score 0.02 ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Fusion adapter | 1074 | 0.346960 | 0.099955 | 0.120953 | 0.313792 | 0.195035 |
| Fusion + SAM LoRA | 996 | 0.272933 | 0.110419 | 0.136140 | 0.307209 | 0.222722 |
| V-JEPA pooling | 1156 | 0.331256 | 0.056189 | 0.063754 | 0.438451 | 0.138250 |
| V-JEPA pooling + SAM LoRA | 1086 | 0.259314 | 0.087650 | 0.115823 | 0.319946 | 0.216395 |
| DINOv2 pooling | 1196 | 0.336409 | 0.048147 | 0.062648 | 0.456136 | 0.131514 |
| Native SAM random-frame LoRA | 676 | 0.293189 | 0.092602 | 0.110101 | 0.368824 | 0.198439 |

All six validations have 50 valid shape scores and zero empty predictions.
Fusion + SAM LoRA currently leads these epoch-1 geometry metrics. Pooling + SAM
LoRA has the lowest validation flow loss, which does not imply the best geometry.
The completed frozen SAM baseline remains the separate 200-object GSO result
above; no frozen SAM result on this same synthetic panel is available in these
artifacts, so its GSO scores cannot establish a trained-versus-frozen ranking on
the synthetic panel. Provider states and live scratch metrics were read directly;
the old native-control `batch-exit.json` with exit 1 belongs to its retained
packaging-failure attempt, not the running retry.

### Intermediate prediction persistence repair and matched GSO backfill

The epoch shape-validation writer previously persisted per-object metrics but
discarded sampled predictions. It now saves every scored sample's raw latent,
prediction voxels, ground-truth voxels, metrics, frame IDs and sampler settings
beside the validation JSON, with an explicit relative `prediction_file` link.
This reuses the existing validation inference; it adds no sampling pass. Two
focused remote CPU tests passed, including exact training-resume equivalence
and loading saved predictions to reproduce their recorded IoU. This source fix
applies to future launches; currently running immutable sources are untouched.

Current runs are covered by a separate native CPU coordinator
[12439762](https://cloud.sdu.dk/app/jobs/properties/12439762), which launches
bounded native B200 jobs only when all six immutable epoch checkpoints exist.
It does not run inference on CPU. The first GPU wave is
[12439763](https://cloud.sdu.dk/app/jobs/properties/12439763), verified RUNNING
and producing saved outputs: 52 prediction tensors, exported coarse meshes and
input videos were present at the first persistence check. This is partial
progress, not completion of the first 1,200-prediction wave.

The coordinator backfills epochs 1–4 for all six trained models on exactly the
same 200 GSO objects. It invokes the existing standalone evaluator unchanged,
loads EMA weights from each immutable epoch checkpoint, and saves every sample's
latent/prediction/target tensor, metric record, input video/reference/mask, coarse
prediction mesh and target mesh. W&B receives metrics and artifacts. Empty
predictions remain explicit failures rather than silently missing records.
The existing training pipelines handle final epoch-5 evaluation and prediction
saving on the same 200 objects. Intermediate outputs are isolated from those
final results and from live training checkpoints.

Exact data audit: all 200 manifest rows equal the first 200 rows of the earlier
250-object baseline manifest, in order. Both resolve to the same asset and
visibility directories. All 200 sample IDs are present in each cached SAM3D,
MV-SAM3D and Stream3D baseline. Their output directories are indexed in the
backfill manifest; refined baseline outputs remain distinctly labelled from
coarse model outputs and retain their original sampling/view budgets.

Persistent root:
`/work/MathiasElmegaardNielsen#6181/epoch-eval-2026-10-30-11-30`.
One-off scripts `coordinator.py`, `worker.py`, per-epoch foreground scripts and
payloads are saved there. Each wave checks exact sample-ID equality, 200 metric
rows and prediction tensors, paired media, and summary completeness before
writing a completion marker. Only one evaluation GPU wave runs at a time; each
GPU releases on exit. CPU coordination has a six-hour fallback and each GPU
wave a one-hour fallback. Later checkpoints wait on CPU without holding a GPU.

At 11:23:44 CET, median last-100-update timing projected 4.49 training hours
remaining for fusion + SAM LoRA and 4.43 for native SAM LoRA, the slowest runs.
Allowing validation/final evaluation gives an estimated full-run completion of
16:00–16:30 CET, subject to throughput changes and evaluator queue delays.

All three SAM-LoRA conditions use rank 8 and alpha 8 throughout training,
1,376,256 trainable parameters each, on native shape cross-attention `to_q`,
`to_kv` and `to_out` projections. Rank/alpha and insertion sites are matched
across conditions; learned weights are separate per run and shared across all
objects/videos within that run. Vision encoder LoRA is disabled.
