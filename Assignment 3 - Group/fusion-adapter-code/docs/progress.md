# Progress: Video SAM 3D fusion adapter

Status on 2026-10-26, branch `mathias`. P1 to P5 are committed. The P5 GPU checks have not run yet; P6 to P8 are not started.

## Phases

| Phase | Content | Status | Commit |
|---|---|---|---|
| P1 | Skeleton, config merging, env files, CLI, `check-env` | Done | `12696e8` |
| P2 | Data reader, sample builder, fake dataset | Done | `76af8fc` |
| P3 | `token_coverage`, gated fusion adapter | Done | `34cdad3` |
| P4 | FakeSam3D, training loop, checkpoints, resume, benchmark, validation | Done | `c40da0d` |
| P5 | SAM 3D inspection, real wrapper, encoders, `cache-latents`, `frame-test` | Code done; GPU checks pending | `d430f16` |
| P6 | `evaluate.py`, full `metrics.py`, `analyze.py` | Not started | |
| P7 | UCloud and GCP scripts, `datagen.py` | Not started | |
| P8 | Final README | Not started (stub only) | |

Tests: 76 CPU tests pass (`pytest -q`, about 15 s). The 11 GPU tests in `tests/gpu/` are skipped without CUDA.

## What exists

| Area | Files | What it does |
|---|---|---|
| Config and CLI | `configs/`, `vidsam3d/config.py`, `vidsam3d/__main__.py` | base ← condition ← env ← `key=value`; struct mode, so a misspelled key is an error. One entrypoint: `python -m vidsam3d <command>`. |
| Environment check | `vidsam3d/check_env.py` | GPU, VRAM, versions, paths, HF token and access, SAM 3D checkpoint files and import; `--load-models` loads SAM 3D and V-JEPA. |
| Data | `vidsam3d/data/reader.py`, `samples.py` | Reads format v0.1 with its conventions. Builds samples: frame subset, reference frame, clip window, object-centred crop, Δt, leakage guard. The val split lives in its own `clips/val`, `assets/val` folders. |
| Targets | `vidsam3d/data/targets.py` | Latent cache per asset with round-trip IoU (T5), canonical permutation, `voxel_fill`, the 24 cube rotations, Phase B pose stub. |
| Adapter | `vidsam3d/models/adapter.py` | Gated cross-attention blocks with zero-init tanh gates (17,869,828 parameters), padding for mixed K, attention-mass diagnostic. |
| Encoders | `vidsam3d/models/encoders.py` | `token_coverage`, `kv_tokens` for the three key/value sources, `VJEPAEncoder` (V-JEPA 2.1 ViT-L from the pinned official code), `DinoEncoder` (SAM 3D's own DINOv2), `FakeEncoder`. |
| SAM 3D | `vidsam3d/models/sam3d.py`, `fake_sam3d.py` | `Sam3DGeometry` wraps the official pipeline (Stage 1 only, frozen, no source changes). `FakeSam3D` has the same interface for CPU tests. |
| Training | `vidsam3d/train.py`, `utils.py` | Flow matching on the adapter only. bf16, gradient accumulation, AdamW, warm-up + cosine, clipping, EMA. Validation loss and vIoU, best checkpoint, benchmark mode. Checkpoints go to scratch and are copied to the output dir; `resume: auto` reproduces the uninterrupted run exactly. |
| Diagnostics | `vidsam3d/inspect_sam3d.py` | `inspect-sam3d` writes the runtime section of `docs/sam3d_internals.md`; `frame-test` runs T4. |
| Docs | `docs/sam3d_internals.md`, `docs/p5_ucloud_setup.md` | What the wrapper relies on in the SAM 3D code; the UCloud GPU setup guide. |

## Decisions so far

| Decision | Value | Reason |
|---|---|---|
| Weight decay | Not on gates, biases, embeddings, LayerNorm | Agreed in P4 review |
| Phase A layout tokens | Pure noise (`phase_a_layout: noise`) | Shape tokens do not attend to layout tokens in SAM 3D, so no zero-shot layout cache is needed |
| CFG null branch | Zeros, never touched by the adapter | `add_flag` + `force_zeros_cond` in SAM 3D (asserted at runtime) |
| V-JEPA route | Pinned `facebookresearch/vjepa2` code, `vjepa2_1_vit_large_384`, 256 px | Only V-JEPA 2.1 ViT-L release; RoPE grid is natively 16 × 16 at 256 px; no new dependencies |
| Val split | Own split folders (`val`) | Agreed |
| Eval reference frames | One sample when middle and most-occluded coincide; ties go to the earliest frame | Default, kept for now |
| C1 crop | Same crop as C3 | Default, kept for now |
| Bootstrap | Resample whole clips | Default, kept for now |
| UCloud / GCS names | `/work/vidsam3d`, bucket `vidsam3d` | Placeholders, overridable by `PROJECT_DIR`, `GCS_BUCKET` |

## Waiting for the GPU run

| Check | Command | What it decides |
|---|---|---|
| Environment | `python -m vidsam3d check-env --env ucloud --load-models` | Both models load |
| Runtime internals | `python -m vidsam3d inspect-sam3d --env ucloud` | `adapter.query_sets` (names of the two RGB token sets); confirms the zero null branch and that the shape ignores the layout |
| GPU tests | `VIDSAM3D_ENV=ucloud pytest -m gpu tests/gpu -s` | T1 (identity at init), T3a (V-JEPA token order), T5 (VAE round trip) |
| T5 | part of the GPU tests | `target.voxel_fill`: surface or solid |
| T4 | `python -m vidsam3d frame-test --env ucloud` | `target.canonical_permutation` |

## Open questions

1. **Sampler for all conditions.** Default: distilled shortcut mode, 4 steps, no CFG (about 4 network passes per sample). Alternative: flow mode, 25 steps with CFG 7 (about 50).
2. **Cost of the conditioning path.** A depth model and 4 DINOv2 passes run per sample and step. If the benchmark shows this dominates, cache the depth map per (clip, frame).
3. **Validation cost.** 200 samples × 9 τ every 1,000 steps; check once the benchmark runs.
4. **For Julian.** Final generator command template, val split, `ours_to_sam3d` (needed for Phase B only).

## Still to do

- **P6:** `evaluate.py` (per-sample `results.jsonl`, resumable, seeds from (clip, k, r); C0 zero-shot, C0b best-frame oracle, `--k`, `--shuffle-frames`, `--wrong-video`), aligned metrics in `metrics.py` (ICP, Chamfer, F-score at 0.02 and 0.05 via pytorch3d), `analyze.py` (paired Δ, cluster bootstrap, Wilcoxon, tables, plots). Done when the metric tests pass and zero-shot eval runs on 10 real eval clips. T7 (wrong-video eval) belongs here.
- **P7:** `scripts/ucloud/setup_env.sh` and `run.sh`, GCP Dockerfile, `build.sh`, `submit.sh`, `vm_run.sh`, `datagen.py` (`gen-data`, `merge-manifests`, `validate-data`), data staging to scratch. The setup script should clone both repos at the pinned commits. Done when the scripts are shellcheck-clean and a `--fake-sam3d` dry run works through `run.sh`.
- **P8:** final one-page README.
