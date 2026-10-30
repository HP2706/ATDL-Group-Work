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
