# Does video conditioning improve SAM 3D geometry?

The current experiment uses the maintained trainer and evaluator with five
trainable configurations and a frozen native SAM 3D baseline. Submit these six
separate jobs in this order:

| Order | Model | Vision encoder | SAM 3D |
|---|---|---|---|
| 1 | Fusion adapter | Frozen V-JEPA | Frozen |
| 2 | Fusion adapter + SAM LoRA | Frozen V-JEPA | LoRA |
| 3 | Native SAM 3D baseline | Native reference image | Frozen |
| 4 | Attention token pooling | Frozen V-JEPA | Frozen |
| 5 | Attention token pooling + SAM LoRA | Frozen V-JEPA | LoRA |
| 6 | Attention token pooling | Frozen DINOv2 frames | Frozen |

The last three are backup approaches. There is no DINOv3 or vision-encoder
fine-tuning in this matrix. `variants.json` defines the exact order.

All adapted models receive the same 16 frames. DINOv2 reuses SAM 3D's installed
ViT-L/14 backbone at its native 224-pixel size and 1,024-wide per-frame patch
tokens, excluding CLS/register tokens. The attentive pooler and projection are
shared with V-JEPA pooling. Both replace the selected native RGB token sets;
native mask/pointmap conditioning remains. Thin objects with no token above the
foreground threshold retain the most-covered positive-coverage token(s), while
background and padding remain excluded. Empty foreground still raises an error.

The current training export is the full 10,000-clip full-q98 synthetic dataset.
The existing asset split produces 20,716 training clip-object samples and 1,306
validation reference samples with no shared target assets or clips. Five epochs
at effective batch 32 require 648 updates per epoch, 3,240 in total. Use the
resolver with explicit parameters:

```bash
python experiments/conditioning_ablation/configure.py \
  --base_config=/persistent/base/config.json \
  --output=/persistent/conditioning-full-q98-<timestamp> \
  --synthetic_root=/persistent/snapshot/full-q98 \
  --gso_root=/persistent/frozen-gso-250 \
  --seed=42 --epochs=5 --evaluation_objects=200
```

The resolver supports audited Omni as an alternative source, but that is not the
current training source. Shards/assets are referenced rather than duplicated;
source indices, target split maps, and selected evaluation IDs are persisted.
The 200 GSO objects are an immutable prefix of the earlier frozen 250-object
selection. Both selections and earlier predictions are preserved.

AdamW uses LR 1e-4, 162 warmup updates and EMA decay .99. Effective batch stays
32 while physical batch is profiled remotely. `profile.py ROOT --only=VARIANT`
profiles one model in its own allocation and writes variant-specific progress
and ready markers, allowing the independent model jobs to run in parallel.
`--resume=True --attempt=recovery1` preserves successful previous profiles and
keeps attempt-specific diagnostics.

A shared native GPU batch caches the synthetic export's targets once (including
its validation assets), caches GSO targets, and verifies real training, strict
checkpoint resume, GPU Chamfer validation and evaluation/W&B logging. After it
passes, submit each model as a separate foreground native batch. The training
jobs profile their own batch size and invoke `run.sh ROOT VARIANT PYTHON`.
Every dedicated allocation must release on foreground pipeline exit, with a
provider wall-time limit as fallback. No GPU is reserved waiting for a timer.

The trainer saves resumable recovery checkpoints every 50 updates and all epoch
checkpoints, evaluates a fixed 50-sample held-out panel each epoch, and logs flow
loss, canonical/aligned voxel IoU, Chamfer, F-scores, empty counts and timing.
`run.sh` verifies the final checkpoint reached the expected update count before
running final evaluation, so a wall-time stop cannot silently count as a complete
training run. W&B receives every computed evaluation scalar, per-object tables,
aggregate means/valid counts/completeness, and result files. Predictions and GT
voxel tensors remain persisted for every evaluation object.

Final evaluation follows existing checkpoint selection: `best.pt` by validation
canonical IoU, falling back to `latest.json`. All epoch checkpoints remain
available. The trained-model evaluator produces Stage-1 latent/voxel geometry;
it does not claim refined-mesh metrics. Its coarse voxel-surface Chamfer protocol
is explicitly distinct from the prior 250-object refined-mesh baseline results.

## Native SAM random-frame fine-tuning control

`sam3d-random-frame-lora` uses native SAM image/mask conditioning and trains only
SAM shape cross-attention LoRA, with the same rank and optimizer as the video
LoRA comparisons. No video encoder or conditioning adapter is instantiated.
The existing synthetic sampler selects a uniformly random eligible reference
frame from its randomly offset temporal subset, independently for each epoch.
Occlusion-biased sampling is disabled; only that one frame is decoded. The
existing visibility filter excludes frames where the target is not visible.
Validation and final evaluation retain the fixed reference-frame protocol.
The maintained trainer, checkpoint/EMA handling, and evaluator are reused.

## Separate appended video and mask conditioning

`append-frozen` and `append-sam-lora` extend this same matrix. Both freeze
V-JEPA 2.1 ViT-L/16 and retain every native SAM conditioning set. The new
`append_video_mask` strategy emits 1,370 video summaries and 1,370 mask
summaries, each of width 1,024, producing 10,268 conditioning tokens for the
pinned 7,528-token native configuration. SAM shape cross-attention LoRA is the
only difference between the two runs; its existing rank/targets remain unchanged.

A 4x4 mask descriptor for each of the two tubelet frames preserves local
silhouette structure. A learned MLP encodes its 32 values. Video and mask
features plus (relative tubelet offset, y, x) determine shared pooling keys;
the two value streams use identical attention weights, learned queries and
query identities, then separate output projections, normalizations and modality
embeddings. Padding is excluded. Invisible frames and background remain valid
context rather than being discarded by a foreground threshold. Relative time
uses the existing frame-offset units, not seconds or camera pose.

The encoder boundary is `VideoFeatures`; each strategy implements
`condition(native, video, query_sets, video_disabled, diagnostics)`. A new encoder
implements `TokenEncoder`; a new strategy is selected by the adapter factory.
SAM assembly no longer checks adapter classes. Legacy low-level tensor callers
remain supported through `apply_adapter`.

Run protocol: fresh adapter initialization; same full-q98 asset-disjoint split,
16 frames, five epochs (3,240 updates), effective batch 32, seed 42, existing
AdamW/flow loss/warmup/EMA/clipping/bfloat16 configuration and checkpoint cadence.
Microbatch is selected by bounded remote profiling at effective batch 32.
Checkpoints include optimizer, scheduler, RNG, data position, adapter, EMA and
optional SAM LoRA. Evaluation uses the existing fixed validation and GSO200
selection and native/repeated/wrong/disabled controls where configured. New
runs log to the existing W&B project; no Hugging Face uploads are requested.

Before full launch, a foreground GPU smoke batch runs regression tests, two
real updates per SAM-LoRA setting with strict save/resume, and one held-out
reconstruction each. The full runs use separate foreground UCloud B200 batch
jobs with a 12-hour allocation cap per run, including profiling and final
evaluation; allocation ends with pipeline exit. A 1-hour smoke cap gives a
maximum combined reservation of 25 GPU-hours, not expected consumption.
Expected full-run duration is provisionally 7–10 hours each based on the
existing fusion runs, with extra cross-attention cost from 2,740 appended tokens;
measured profiling replaces this estimate. Persistent data/caches are reused.
Approximate per-GPU peak is provisionally 45–100 GB at microbatch 16–32: frozen
SAM/V-JEPA weights, SAM activations for adapter backpropagation and attention
workspaces dominate; adapter optimizer state is below 1 GB. No KV cache is
retained across training examples. Profiling records actual parameter counts,
allocated memory and throughput before the final microbatch is fixed.

New runs execute a clean committed main checkout, separate from immutable
historical experiment snapshots. Existing jobs and their source snapshots are
not modified. The source commit and resolved configurations are persisted with
every launch.
