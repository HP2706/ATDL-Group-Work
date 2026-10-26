# M0 checkpoint (updated 2026-10-25)

Resume by reading this file, then `docs/dataset_spec.md` (already updated with the agreed fixes).

## State

- **Environment:** uv venv (Python 3.11), `uv sync`. BlenderProc 2.8.0 with Blender 4.2.1 in `.blender/`; Cycles uses OptiX on the RTX 3060.
- **Assets:** 50 GSO objects downloaded (`scripts/download_gso.py`, catalogue 1033) and preprocessed into `dataset/assets/eval/` (`python -m sam3dvid.assets --source gso --split eval --limit 50`).
- **SA-1B:** `sa_000000` downloaded. 11,186 images scanned, 9,062 landscape, split into a train pool of 7,258 and an eval pool of 1,804.
- **Clip:** the M0 clip `dataset/clips/eval/00000000` (seed 0, SA-1B background) rendered in about 57 s; the folder is 20 MB with compressed depth. T7 sheet: `dataset/review/`; it still needs your sign-off.
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
- **Render settings:** Cycles 64 samples, OIDN denoiser, Standard view transform; sun strength U(2, 5), ambient U(0.15, 0.35).
- **T1 depth check:** drops back-facing points before z-buffering. Without that, the flat package in the dev clip fails because pixels only receive far-side points. Spec updated.

## Open issues

1. **T1 is too weak on small objects.** A 5° rotation error passes T1 (`test_t1_detects_pose_errors`, xfail). Objects in the dev clip are only 2-5k px, so a 3-px dilation absorbs a 5° error, and 1% of the orbit radius (0.055) is loose. Options:
   - tighten T1 (dilation 1 px, or a depth limit relative to object size);
   - add an IoU check between the rendered amodal mask and a silhouette rasterized from the GT pose.
2. **Objects are small in frame.** Because the orbit distance scales with the scene radius and `ρ(N)` grows with N, multi-object clips have tiny objects (largest amodal mask in the dev clip is about 2% of the image). Measure the distribution over around 50 seeds and decide whether to keep the spec as is.
3. **CPU time:** resolved. BlenderProc's "INTEL" denoiser was a CPU compositor node; switched to OptiX, GPU only. 57 s -> 19 s per clip.
4. **DATASET.md** not written yet. The consumer-facing format doc is `docs/data_format.md`.
5. **Geometric filter:** scores are computed per asset (`meta.json` `filter_scores`); thresholds and the contact sheet come at M2.

## Next steps

1. Resolve open issues 1-2 (T1 strength, object size).
2. Write DATASET.md.
3. Run T6 with SAM 3D, then M1 (500 eval clips, about 10 GB, roughly 8-9 h at the current speed).
