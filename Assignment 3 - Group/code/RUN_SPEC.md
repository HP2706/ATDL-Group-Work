# First-run specification: view coverage pilot

Status: prepared for review; GPU job not submitted. All settings below are proposed for the first two-hour allocation. This is a smoke ablation to validate the pipeline and obtain initial signals, not a conclusive scientific comparison.

## Models and trainable parameters

- SAM 3D source: `f91db411c50efee93d8db7aeb323885650f6f722`; weights: `facebook/sam-3d-objects@2e73555018d2741ccd486e56c24fac41155a1dc6`.
- V-JEPA 2.1 ViT-B/16 at 384 pixels, 12 transformer blocks, width 768; source: `204698b45b3712590f06245fbfba32d3be539812`; official `vjepa2_1_vitb_dist_vitG_384.pt` checkpoint. Approximately 86M encoder parameters; exact loaded count will be recorded by GPU profiling. Predictor discarded.
- Frozen SAM coarse generator: 959,962,133 parameters, 24 transformer blocks. Frozen native image conditioning is cached during preparation. Frozen sparse-structure VAE encoder and decoder are reused. Total loaded parameters, including both VAE components, are recorded before training; they have not been measured yet.
- Connector: one biased linear layer, 768→1024, exactly 787,456 trained parameters in every variant. Temporal averaging retains 576 spatial video tokens.
- Video LoRA: rank 8, alpha 8, zero-initialized output matrices, qkv/output projections in all 12 attention blocks: expected 24 modules and 442,368 parameters.
- SAM LoRA: rank 8, alpha 8, shape cross-attention q/kv/output projections in all 24 blocks: 72 modules and 1,376,256 parameters.

| Variant | Trainable parts | Trainable parameters |
|---|---|---:|
| A | connector | 787,456 |
| B | connector + video LoRA | 1,229,824 |
| C | connector + SAM LoRA | 2,163,712 |
| D | connector + both LoRAs | 2,606,080 |

Counts other than the loaded encoder/combined totals were checked from pinned SAM module shapes and adapter dimensions. Profiling records actual module names and counts and must precede training.

## Data and prediction target

NAVI v1.5, with seeded object-disjoint pools of 24 train / 6 validation / 6 test objects. The pilot activates two training and two validation objects, at most two video clips each; test objects are untouched. This dataset contains real object videos, not exclusively furniture.

Each input has 16 uniformly sampled masked frames, resized to 384×384, with no frame packing or padding. V-JEPA produces 8×24×24 = 4,608 video tokens; temporal pooling yields 576 projected tokens appended to native reference conditioning. The reference is the first selected frame.

The target is canonical normalized object surface occupancy at 64³, encoded with the upstream VAE into 4,096 tokens of width 8. All shape-latent coordinates receive flow-matching loss, weight 1. Pose/layout terms have weight 0 and neutral targets because this pilot tests object geometry rather than scene placement. Upstream shortcut self-consistency is disabled. The same upstream loss and sampler execute the computation.

## Optimization and batch

One GPU, one replica, all components on that GPU. Effective batch is four clips per optimizer update. A maximum five-minute profile tests physical microbatches 1, 2, and 4 with accumulation 4, 2, and 1 respectively. It selects the fastest measured safe setting, preserving at least 2 GiB of device-memory headroom. All A/B/C/D variants use that setting.

AdamW, learning rate 1e-4, weight decay 0, five-update warmup and cosine decay, gradient-norm clipping 1.0. Frozen generator/video weights use BF16; connector/adapter weights, gradients and Adam states use FP32; upstream VAE weights remain FP32 under BF16 autocast. Activation checkpointing is enabled in video and SAM blocks. No sharding or offloading.

Each variant starts from identical pretrained weights and the same connector initialization, seed 42, and data order. Maximum 25 updates per variant: 100 clip exposures and 409,600 scored shape-token exposures, or 3,276,800 scalar latent targets per variant. Data necessarily repeats in this tiny pilot. Resuming restores optimizer, scheduler, RNG, and data progress; it does not restart training from scratch.

## Memory and time estimates

One `gpu-nvidia-b200-1-mig.1g` allocation, approximately 23 GB GPU memory, 6 CPU cores and 36 GB host RAM; actual device capacity is recorded. Estimates are not measurements: generator BF16 weights ≈1.79 GiB, video BF16 weights ≈0.16 GiB; VAE weights and setup-time native conditioning add further memory not yet measured. Maximum trained parameters require ≈10 MiB each for FP32 weights and gradients and ≈20 MiB for Adam moments. There is no autoregressive KV cache. Saved activations, attention workspaces, VAE decoding and runtime reserve dominate the remaining uncertainty. The profiler measures complete forward/backward/optimizer steps and refuses an unsafe fit. Evaluation peaks still require remote validation.

Provider wall-time cap: two hours, one MIG allocation; foreground pipeline cap: 115 minutes. Setup downloads approximately 30 GB of NAVI plus checkpoints and installs dependencies inside this cap. Profiling is capped at five minutes; each variant has an 18-minute training budget plus bounded shutdown overhead, and the sweep stops starting variants when insufficient time remains. Throughput is unmeasured; 25 updates for all variants are a target, not a guaranteed fit. Partial runs are explicitly marked incomplete. No second allocation is automatically rented.

## Validation and outputs

Fixed validation flow loss every five updates; checkpoint every five updates plus initial/final/interruption saves. Mesh evaluation after 25 updates and at an early normal endpoint, using two held-out objects and 25 sampler steps. Equal generation noise seeds across modes/variants.

Controls: immutable native image-only coarse generation, adapted reference-frame repetition, and adapted full video. Primary paired signal: lower symmetric unsquared surface Chamfer distance for video than reference repetition, especially higher recall within distance 0.02 on scan surfaces newly revealed by video. Also report whole-shape F-score at 0.02, surface-occupancy IoU, visible/unseen region completeness/recall, and failures. Region labels use NAVI camera/depth annotations; empty regions are omitted, failed reconstructions are penalized.

W&B: `hprjdk/atdl-video-to-3d`, one profiling run plus four grouped training runs, with paired video/reference/predicted-3D/ground-truth tables and downloadable mesh artifacts. Persistent storage retains full logs, checkpoints, dataset manifest/splits, data-quality diagnostics, provenance, and resolved configurations. Provider-native batch exit automatically releases the allocation on success or failure; the two-hour limit is a fallback.

## Release gate

Before submission: review these settings, commit/push the source, verify the resolved configuration, and stage the private runtime environment. During the remote run: verify gated weights load, upstream gradients propagate, VAE targets/visibility diagnostics are valid, and memory profiling succeeds. Scan orientation relative to SAM canonical axes must be inspected before interpreting absolute Chamfer values; normalization alone does not establish rotational alignment. A failed gate stops the batch and preserves diagnostics. No GPU training or dataset encoding has run locally.
