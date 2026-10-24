# M0 checkpoint (2026-10-24)

Resume by reading this file, then `docs/dataset_spec.md` (already updated with the agreed fixes).

## State

- **Environment:** uv venv (Python 3.11), `uv sync`. BlenderProc 2.8.0 with Blender 4.2.1 in `.blender/`; Cycles uses OptiX on the RTX 3060.
- **Assets:** 50 GSO objects downloaded (`scripts/download_gso.py`, catalogue 1033) and preprocessed into `dataset/assets/eval/` (`python -m sam3dvid.assets --source gso --split eval --limit 50`).
- **SA-1B:** `scripts/download_sa1b.py` and `scripts/build_bg_pools.py` are written. The tar extraction and pool building passed synthetic tests; the HTTP download code has never been run. **Blocked on `data/sa1b_links.txt`.**
- **Clip:** one dev clip, `dataset/clips/eval/00000000`, rendered in about 65 s. It uses a synthetic grid background (pool in the session scratchpad) because SA-1B isn't there yet. It is not the M0 deliverable.
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
3. **CPU time.** About 65 s per clip, with high CPU use (7 min of user time). OIDN probably runs on the CPU; also check samples/denoiser and the cost of the amodal passes.
4. **DATASET.md** not written yet.
5. **Geometric filter:** scores are computed per asset (`meta.json` `filter_scores`); thresholds and the contact sheet come at M2.

## Next steps

1. Once `data/sa1b_links.txt` exists, run `uv run python scripts/download_sa1b.py` and then `uv run python scripts/build_bg_pools.py`.
2. Delete the dev clip: `rm -rf dataset/clips dataset/manifest.jsonl`.
3. Run `uv run python -m sam3dvid.generate --config configs/m0.yaml --split eval --seeds 0`.
4. Run `uv run pytest`, then `uv run python -m sam3dvid.review --num 1`, and inspect `dataset/review/`.
5. Resolve open issues 1-3, then write DATASET.md. After that comes T6 with SAM 3D.
