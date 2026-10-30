# Epoch 1 and epoch 10: matched in-sample diagnostics

Recovered from the earlier comparison of the same ten reference frames.
All rows used matched sampling seeds and four sampling steps. These are
in-sample diagnostics, not held-out generalization results. Flow loss is the
matched evaluation loss, not the epoch-average training loss. Values below
retain the precision of the original reported tables.

| Model | Epoch 1 flow loss | Epoch 1 voxel IoU | Epoch 1 empty predictions | Epoch 10 flow loss | Epoch 10 voxel IoU | Epoch 10 empty predictions |
|---|---:|---:|---:|---:|---:|---:|
| Pretrained SAM 3D (unchanged baseline) | 0.8074 | 0.1371 | 0/10 | 0.8074 | 0.1371 | 0/10 |
| Fusion adapter | 0.8074 | 0.1371 | 0/10 | 0.8006 | 0.1255 | 0/10 |
| Fusion + SAM LoRA | 0.8072 | 0.1372 | 0/10 | 0.7749 | 0.1233 | 0/10 |
| Fusion + SAM LoRA + V-JEPA LoRA | 0.8072 | 0.1372 | 0/10 | 0.7749 | 0.1237 | 0/10 |
| DINO attention pooler | 0.6796 | 0.0324 | 6/10 | 0.3472 | 0.0729 | 0/10 |

Lower flow loss is better; higher voxel IoU is better. At epoch 10 every
adapted model had lower loss but worse canonical overlap than the untouched
baseline. Canonical orientation/convention uncertainty remains relevant to
these overlap scores. No epoch-10 prediction was empty.

Original matched comparison:
https://forge.coreweave.com/wandb/hprjdk/atdl-video-to-3d/runs/sam3d-exact-preview-comparison-2026-10-29-19-43-12

DINO training run:
https://forge.coreweave.com/wandb/hprjdk/atdl-video-to-3d/runs/dino-frame-attention-pooler-2026-10-29-19-29-45

The source tables were recovered from the prior conversation and verified
against the saved summary artifacts on the new UCloud B200 workstation.
The epoch-1 and epoch-10 sample ID lists match exactly. Local evidence copies:
[epoch 1](../results/epoch-comparison-2026-10-29/epoch1-summary.json) and
[epoch 10](../results/epoch-comparison-2026-10-29/epoch10-summary.json).
No training or checkpoint inference was repeated to recover these numbers.
