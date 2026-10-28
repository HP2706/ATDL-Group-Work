# Progress

Status on 2026-10-27, branch `mathias`. All phases are implemented. What is left needs a GPU and the real datasets, neither of which has been used yet.

| Phase | Content | Status |
|---|---|---|
| P1 | Config merging, env files, CLI, `check-env` | Done |
| P2 | Data reader, sample builder, fake dataset | Done |
| P3 | Token coverage, gated fusion adapter | Done |
| P4 | Training loop, checkpoints, exact resume, benchmark, validation (FakeSam3D) | Done |
| P5 | SAM 3D wrapper, V-JEPA / DINOv2 encoders, latent cache, `inspect-sam3d`, `frame-test` | Code done; GPU checks pending |
| P6 | `evaluate.py`, `metrics.py`, `analyze.py` | Done on the fake dataset; real eval pending |
| P7 | `datagen.py`, setup / run scripts for UCloud, Vertex AI and GPU VMs | Done; not yet run on a cluster |
| P8 | README and docs | Done |

**Tests.** 98 CPU tests pass (`pytest -q`, about 2 min) on a rendered fake dataset with FakeSam3D. They cover the data conventions, the leakage guard, the adapter (identity at init, gradients), exact resume, evaluation (including a CPU version of T1), the metrics, the statistics, the dataset tools and a dry run through `scripts/run.sh`. The 11 GPU tests in `tests/gpu/` (T1, T3a, T4, T5 and SAM 3D facts) skip without CUDA, checkpoints and data. The metric and eval tests need pytorch3d; on macOS it builds with `CFLAGS=-Wno-invalid-specialization CXXFLAGS=-Wno-invalid-specialization uv pip install --no-build-isolation "pytorch3d @ git+https://github.com/facebookresearch/pytorch3d.git@75ebeea"`.

## Decisions

| Topic | Decision | Reason |
|---|---|---|
| Weight decay | Not on gates, biases, embeddings, LayerNorm | P4 review |
| Phase A layout tokens | Pure noise | Shape tokens do not attend to layout tokens, so no zero-shot layout cache is needed (internals item 7) |
| CFG null branch | Zeros; the adapter never touches it | `add_flag` + `force_zeros_cond` in SAM 3D, asserted at runtime |
| Sampler | Distilled shortcut, 4 steps, no CFG, for every condition | Compute; flow mode with CFG is one config switch away (`sampler.mode: flow`) |
| GPU environment | torch 2.7.1+cu128, pytorch3d / kaolin rebuilt for the GPU | Only B200 is available, and SAM 3D's cu121 stack has no sm_100 kernels; the same env runs on A100/H100 |
| V-JEPA | `vjepa2_1_vit_large_384` from the pinned official code, 256 px | Only V-JEPA 2.1 ViT-L release; its RoPE grid is natively 16 × 16 |
| Val split | Own split folders (`clips/val`, `assets/val`) | Agreed |
| Eval references | Middle frame and most-occluded frame of S; one sample if they coincide; ties go to the earliest frame | Default |
| Aligned metrics | Normalise to [-1, 1], similarity ICP, vIoU on the GT grid, Chamfer and F-score on surface voxels; F-score thresholds 0.02 / 0.05 of the longest side | Rigid ICP cannot undo the scale change that bbox normalisation gives a rotated shape |
| Result labels | Run name plus `_k<K>`, `_shuffle`, `_wrongvideo`; baselines `c0`, `c0b_oracle` | One results file per label; seeds depend only on (clip, k, r) |
| Bootstrap | Resample whole clips, 2,000 resamples, seeded | Keeps all objects and reference frames of a clip together |
| Shard manifests | `manifest.<split>.shard<ID>.jsonl`, written from each clip's `meta.json` | Independent of the generator's file naming |

## Waiting for

| Item | Needed for | Who |
|---|---|---|
| Datasets (train, val, eval) in format v0.1 | everything after `check-env` | Julian |
| Generator command template, clip counts, seed ranges (`generator.*` in `base.yaml`) | `gen-data` | Julian |
| `ours_to_sam3d` pose converter | Phase B (rotation loss) only | Julian |
| First run of `scripts/setup_env.sh` on a B200 | all GPU steps | us |
| `inspect-sam3d` output | `adapter.query_sets` | us |
| T5 and T4 results | `target.voxel_fill`, `target.canonical_permutation` | us |
| Benchmark | `train.batch_size`, `train.max_steps`; check the cost of the val loss and of the conditioning path | us |
| GCP project, bucket, machine type | `scripts/gcp/gcp.env` (only if we use Google Cloud) | us |

The order of the GPU steps once the data is in place is in [ucloud.md](ucloud.md#when-the-data-arrives).
