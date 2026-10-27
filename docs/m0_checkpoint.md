# M0 checkpoint (updated 2026-10-26)

Resume by reading this file, then `docs/dataset_spec.md` (already updated with the agreed fixes).

## State

- **Environment:** uv venv (Python 3.11), `uv sync`. BlenderProc 2.8.0 with Blender 4.2.1 in `.blender/`; Cycles uses OptiX on the RTX 3060.
- **Assets:** 50 GSO objects downloaded (`scripts/download_gso.py`, catalogue 1033) and preprocessed into `dataset/assets/eval/` (`python -m sam3dvid.assets --source gso --split eval --limit 50`).
- **SA-1B:** `sa_000000` downloaded. 11,186 images scanned, 9,062 landscape, split into a train pool of 7,258 and an eval pool of 1,804.
- **Clip:** the M0 clip `dataset/clips/eval/00000000` (seed 0, SA-1B background) rendered in about 19 s; the folder is 20 MB with compressed depth. T7 signed off by the user on 2026-10-26; M0 is complete apart from the deferred T6.
- **Tests:** `uv run pytest` runs 15 tests; all pass except one expected failure (xfail). T1, T2 (11 unit tests), T3, T4, T5 (regenerates the clip) and the format checks all pass. T7 tool: `python -m sam3dvid.review`.
- **Git:** first commit made at this checkpoint.

## Decisions made (to go into DATASET.md)

- **Canonical frame:** Blender Z-up after import and TRELLIS `normalize_scene` (copied verbatim), baked into one triangulated mesh with identity transform.
  - GSO OBJs are already Z-up, so they are imported with no axis change. TRELLIS's Y-up OBJ default would lay them on their side.
  - `mesh.glb` is a standard Y-up glTF; outside Blender, apply `conventions.CANONICAL_FROM_GLTF`. Checked: 100% of surface samples land in occupied voxels.
- **Voxels:** TRELLIS `voxelize.py` replicated exactly (open3d, Blender PLY export, clip at ±(0.5-1e-6)).
- **Aesthetic threshold:** TRELLIS-500K's Objaverse-XL subset uses ≥ 5.5. Training configs re-filter at ≥ 4.5.
- **K:** OpenCV pixel-center convention, `cx = cy = 255.5`, the same as BlenderProc's.
- **Placement ball radius:** `ρ(N) = N^(1/3)`. Overlaps resample positions only. A unit test checks that the scale mean stays 0.7.
- **Retries:** N, assets, background and light are fixed per seed. Retries redraw only layout and camera, so rejections don't bias N or the asset mix.
- **Pre-check:** z-buffered voxel splats at 128², with a slack of 0.7 on the thresholds.
- **Render settings:** Cycles 64 samples, OptiX denoiser on the GPU, Standard view transform; sun strength U(2, 5), ambient U(0.15, 0.35).
- **T1 depth check:** drops back-facing points before z-buffering. Without that, the flat package in the dev clip fails because pixels only receive far-side points. Spec updated.

## Open issues

1. **T1 weakness:** resolved by T1b (ray-cast silhouette and depth; catches half-pixel errors).
2. **Object size:** resolved. Tight framing plus orbit factor U(1.0, 1.4): median object 1.5% -> 2.5% of the frame, 3% of objects truncated in some frame.
3. **CPU time:** resolved. BlenderProc's "INTEL" denoiser was a CPU compositor node; switched to OptiX, GPU only. 57 s -> 19 s per clip.
4. **DATASET.md:** resolved; it records all implementer decisions. The consumer-facing format doc is `docs/data_format.md`.
5. **Geometric filter:** scores are computed per asset (`meta.json` `filter_scores`); thresholds and the contact sheet come at M2.

## Next steps

1. Done: T1b (ray-cast) added; tight framing with orbit factor U(1.0, 1.4).
2. Done: DATASET.md; M1 (500 eval clips, 9.2 GB, 13.7 s/clip with two workers).
3. T6 on a cloud A100 (`docs/t6_runbook.md`); M2 on the cluster (`docs/m2_runbook.md`).
