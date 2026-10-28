# Progress

Status on 2026-10-28, branch `mathias`. All phases are implemented and committed. What is left needs a GPU and the real datasets, neither of which has been used yet.

| Phase | Content | Status |
|---|---|---|
| P1 | Config merging, env files, CLI, `check-env` | Done |
| P2 | Data reader, sample builder, fake dataset | Done; reader switched to the generator's training export (2026-10-28) |
| P3 | Token coverage, gated fusion adapter | Done |
| P4 | Training loop, checkpoints, exact resume, benchmark, validation (FakeSam3D) | Done |
| P5 | SAM 3D wrapper, V-JEPA / DINOv2 encoders, latent cache, `inspect-sam3d`, `frame-test` | Code done; GPU checks pending |
| P6 | `evaluate.py`, `metrics.py`, `analyze.py` | Done on the fake dataset; real eval pending |
| P7 | `validate-data`, setup / run scripts for UCloud, Vertex AI and GPU VMs | Done; not yet run on a cluster |
| P8 | README and docs | Done |

**Tests.** 93 CPU tests pass (`pytest -q`, about 1.5 min) with FakeSam3D on fake clips that the generator's own exporter (`../synthetic-data/src/sam3dvid/export.py`) packs, so the reader is tested against exactly what the generator ships. They cover the export reader (poses, masks, voxels, shard index), the asset-disjoint val split, the leakage guard, the adapter (identity at init, gradients), exact resume, evaluation (including a CPU version of T1), the metrics, the statistics, `validate-data`, the pose converter against the generator's, and a dry run through `scripts/run.sh`. The 11 GPU tests in `tests/gpu/` (T1, T3a, T4, T5 and SAM 3D facts) skip without CUDA, checkpoints and data. The metric and eval tests need pytorch3d; on macOS it builds with `CFLAGS=-Wno-invalid-specialization CXXFLAGS=-Wno-invalid-specialization uv pip install --no-build-isolation "pytorch3d @ git+https://github.com/facebookresearch/pytorch3d.git@75ebeea"`.

## Decisions

| Topic | Decision | Reason |
|---|---|---|
| Weight decay | Not on gates, biases, embeddings, LayerNorm | P4 review |
| Phase A layout tokens | Pure noise | Shape tokens do not attend to layout tokens, so no zero-shot layout cache is needed (internals item 7) |
| CFG null branch | Zeros; the adapter never touches it | `add_flag` + `force_zeros_cond` in SAM 3D, asserted at runtime |
| Sampler | Distilled shortcut, 4 steps, no CFG, for every condition | Compute; flow mode with CFG is one config switch away (`sampler.mode: flow`) |
| GPU environment | torch 2.7.1+cu128, pytorch3d / kaolin rebuilt for the GPU | Only B200 is available, and SAM 3D's cu121 stack has no sm_100 kernels; the same env runs on A100/H100 |
| V-JEPA | `vjepa2_1_vit_large_384` from the pinned official code, 256 px | Only V-JEPA 2.1 ViT-L release; its RoPE grid is natively 16 × 16 |
| Data format | Only the generator's training export (WebDataset shards); the GSO test set is exported with the same tool | It is what the M2 run delivers; train and test then both use JPEG frames |
| Val split | Asset-disjoint, carved out of the 10k train clips: clips showing one of 3 % hash-chosen assets, with only those assets as targets | No separate val split is generated; unseen shapes, like the GSO test set |
| Eval references | Middle frame and most-occluded frame of S; one sample if they coincide; ties go to the earliest frame | Default |
| Aligned metrics | Normalise to [-1, 1], similarity ICP, vIoU on the GT grid, Chamfer and F-score on surface voxels; F-score thresholds 0.02 / 0.05 of the longest side | Rigid ICP cannot undo the scale change that bbox normalisation gives a rotated shape |
| Result labels | Run name plus `_k<K>`, `_shuffle`, `_wrongvideo`; baselines `c0`, `c0b_oracle` | One results file per label; seeds depend only on (clip, k, r) |
| Bootstrap | Resample whole clips, 2,000 resamples, seeded | Keeps all objects and reference frames of a clip together |
| Data generation | Owned by `synthetic-data` (`scripts/m2/render_shard.sh`, train seeds 1000000-1009999); no `gen-data` here | The generator has its own sharding, manifest and validation |
| Phase B converter | `targets.convert_pose_to_sam3d`, a port of the generator's T6-verified `ours_to_sam3d` (tested against it) | Only the camera axes and the rotation format differ |

## Waiting for

| Item | Needed for | Who |
|---|---|---|
| Train export (10k clips, M2 on UCloud) | everything after `check-env` | group member running M2 |
| GSO eval split as an export (`sam3dvid.export --split eval`) | evaluation | whoever holds the M1 clips |
| First run of `scripts/setup_env.sh` on a B200 | all GPU steps | us |
| `inspect-sam3d` output | `adapter.query_sets` | us |
| T5 and T4 results | `target.voxel_fill`, `target.canonical_permutation` | us |
| Benchmark | `train.batch_size`, `train.max_steps`; check the cost of the val loss and of the conditioning path | us |
| GCP project, bucket, machine type | `scripts/gcp/gcp.env` (only if we use Google Cloud) | us |

The order of the GPU steps once the data is in place is in [ucloud.md](ucloud.md#when-the-data-arrives).
