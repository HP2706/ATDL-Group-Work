# DATASET.md: implementer decisions

This documents every decision the spec (`docs/dataset_spec.md`, Section 9) leaves to the implementer, plus the milestone results and known limitations. The file format and conventions for consumers are in [`docs/data_format.md`](docs/data_format.md) and are not repeated here. Parameter values are the defaults in `src/sam3dvid/config.py`; the exact config of a dataset is stored in `dataset/config.yaml`. Items marked TBD are not decided yet. Anything marked "inferred" was not measured.

## 1. Asset preprocessing and filters

**Canonical frame: Blender Z-up after import and TRELLIS `normalize_scene` (copied verbatim), baked into one triangulated mesh with identity transform.** This is what the spec asks for (replicate TRELLIS exactly), and it makes the canonical frame well defined independent of the source format.

- GSO OBJs are already Z-up, so they are imported with no axis change. TRELLIS's Y-up OBJ default would lay them on their side.
- `mesh.glb` is a standard Y-up glTF. Outside Blender, apply `conventions.CANONICAL_FROM_GLTF`. Checked: 100% of surface samples land in occupied voxels.

**Voxels: TRELLIS `voxelize.py` replicated exactly** (open3d, Blender PLY export, clip at ±(0.5 - 1e-6)), 64³ surface occupancy.

**Aesthetic threshold: ≥ 5.5** for the Objaverse-XL subset of TRELLIS-500K. TRELLIS training configs re-filter at ≥ 4.5; we keep the stricter list value.

**Geometric filter (paper App. B.1): thresholds TBD.** Scores (near-zero volume, normal variation, outlier fragments) are already computed per asset and stored in `meta.json` under `filter_scores`.

| Criterion | Score | Threshold |
|---|---|---|
| Near-zero volume | TBD | TBD |
| Minimal normal variation | TBD | TBD |
| Spatial outlier fragments | TBD | TBD |

The contact sheet of 50 rejected and 50 accepted assets is TBD with the thresholds.

**Eval assets: all 1033 GSO objects in the catalogue**; 707 of them appear in the M1 clips. Objaverse licenses are per object and recorded in `meta.json`; see Section 10.

## 2. Rendering

**Cycles, 64 samples, OptiX denoiser on the GPU, Standard view transform.** The spec leaves samples and denoiser open; Standard matches the sRGB textures.

- The denoiser choice was driven by cost. BlenderProc's "INTEL" (OIDN) denoiser is a CPU compositor node that took 38 of 42 s per clip. Switching to OptiX, which runs inside Cycles, cut the time from 57 s to 19 s per clip.
- Quality cost of the switch: 0.52/255 mean RGB difference, alpha and masks unchanged.
- OptiX is NVIDIA-only; the generator needs an NVIDIA GPU. Cycles devices are GPU only.

**Light: one sun plus ambient, fixed in world for the clip.** Sun direction uniform on the upper hemisphere, sun strength U(2, 5), ambient strength U(0.15, 0.35). The ranges are our choice (the spec asks only that they be logged; they are in `meta.json` `light`).

**Blender version: BlenderProc 2.8.0 defaults to Blender 4.2.1, which is what M1 used.** Blender 4.5.15 LTS is supported through `blender_custom_path` and is needed for GPUs 4.2 has no kernels for (Blackwell, e.g. B200). On the same seed, 4.5 gives identical masks and depth, RGB within 0.03/255 mean (max 2/255), and is 5-15% faster. `meta.json` records the Blender version, so mixed-version datasets stay identifiable.

**Camera intrinsics: OpenCV pixel-center convention, `cx = cy = 255.5`**, the same as BlenderProc's. A half-pixel error here is exactly what T1b detects (Section 8).

## 3. Pre-render visibility check and retries

**Visibility rule: lower bounds only. We keep visible ≥ 0.2% of the image (524 px) and visible/amodal ≥ 0.10, and deliberately do not apply the paper's ≤ 0.9 upper cap.** The cap belongs to SAM 3D's Flying Occlusions data, whose purpose is to guarantee occlusion. Applied per object per frame it would reject all N = 1 clips and most moving-camera clips, and it would remove the fully visible end of the occlusion changes a video model should learn from. Consumers who want an occlusion-only subset can filter on the stored per-frame `visibility` stats in `meta.json`.

**Pre-render check: z-buffered voxel splats at 128², thresholds scaled by a slack of 0.7.** It is conservative (inferred from design, not measured as a bound), so it rejects only clearly hopeless layouts and the post-render check remains the arbiter.

**Retries: max 20 per clip, with a derived seed per attempt. N, assets, background and light are fixed per seed.** Retries redraw only layout and camera, so rejections don't bias N or the asset mix. The consequence is that rotations, scales, positions and the camera lean slightly towards less-occluded configurations in released clips (documented in `docs/data_format.md` Section 7).

**Background arc resampling: max 50 arc retries**, resampling only `Δφ` and `Δθ` when the analytic coverage pre-check fails (margin 2 px). Retries and rejection reasons are logged in `meta.json`.

M1 numbers are in Section 8.

## 4. Sampling decisions

- **Placement ball radius `ρ(N) = N^(1/3)`** (`ρ(1) = 1.0` as the spec requires). It keeps ball volume per object constant, so N = 4 at s = 1 still fits.
- **Overlap handling:** resample positions only, rotations and scales are kept (spec). A unit test checks that the scale mean stays 0.7. At most 20000 placement attempts.
- **Scene radius from actual geometry:** the largest distance from the centroid to the posed objects' voxel centers plus half a voxel diagonal, not to bounding-cube spheres. The spheres overestimate the extent and made objects small in frame.
- **Orbit factor U(1.0, 1.4)** applied to `r / tan(FOV/2)`. Effect of the tight radius plus this range (measured over 400 eval scenes): median object area 1.5% to 2.5% of the frame. The price is truncation: 3% of objects are cut off in at least one frame (1.7% of object-frames, 7% of clips), with a median 1% of the surface outside. Factors below about 1.1 are what allow this; we accepted it for larger objects, and truncation is recoverable from the stored amodal masks, which are clipped to the image.
- **Everything else follows the spec** (N distribution, SO(3) rotations, `s ~ U(0.4, 1.0)`, arc and elevation ranges, easing, 130° background FOV, virtual reference camera at the arc midpoint). The full distribution table is in `docs/data_format.md` Section 7.

## 5. Backgrounds

**SA-1B, landscape only (W ≥ H), short side ≥ 1024 px, treated as a plane at infinity** (homography per frame, no parallax), with a virtual horizontal FOV of 130°.

- Train and eval pools are assigned by a fixed hash of the image ID. From the downloaded archive `sa_000000`: 11,186 images scanned, 9,062 landscape, split into a train pool of 7,258 and an eval pool of 1,804.
- Only IDs and paths are stored, images are not copied.
- Full coverage is asserted for every frame (T4); there are no undefined pixels.
- The plane-at-infinity assumption is why the arc span is capped at 70°.

## 6. Parallelization

**Local: concurrent worker processes on one GPU; chunked asset preprocessing; batch runs skip existing clips and log failed seeds** (commit `91a00ac`). Two concurrent workers on one RTX 3060 6 GB give 13.7 s/clip effective, against 19.3 s median in a single process.

**M2 cluster: one job per seed range, all GPUs of a node, several workers per GPU** (`scripts/m2/render_shard.sh`; procedure in `docs/m2_runbook.md`). The spec suggests one clip per job; a seed range per job amortizes environment and Blender start-up and works with or without a scheduler. Jobs are resumable because finished clips are skipped. The workers-per-GPU value is measured per GPU type by `scripts/m2/smoke_test.sh`. Train seeds are 1000000-1009999, because a seed fixes N, layout and camera path regardless of split and must not repeat an eval seed. Size estimate, inferred by scaling M1 (9.2 GB per 500 clips): about 185 GB for 10,000 clips.

## 7. SAM 3D converter

**`ours_to_sam3d` and its inverse: TBD, pending T6.** Known so far:

- SAM 3D's pointmaps are in PyTorch3D camera axes (x left, y up, z forward), i.e. `diag(-1, -1, 1)` of OpenCV camera coordinates.
- Whether a fixed object-frame rotation or scale factor is also needed is undetermined. The canonical voxel check in T6 (no ICP, identity should win among the 24 axis-aligned rotations) is what settles it.
- T6 scripts are in `scripts/t6/`, procedure in `docs/t6_runbook.md`.

## 8. Milestone results

### M0 (complete, T7 signed off 2026-10-26)

One clip on 50 GSO objects, seed 0, rendered in about 19 s. Tests T1, T1b, T2 (11 unit tests), T3, T4, T5 and the format checks pass; T6 deferred.

**T1b calibration (ray-cast check).** On 576 correct object-frames: max `boundary_mismatch` 0.007, median relative depth error 1e-5. Thresholds are 0.05 and 5e-4 · s, a wide margin over the correct data. Negative controls all fail: 5° rotation, 0.1 · s shift and a half-pixel `cx` shift (minimum mismatch 0.129).

**T1 depth check drops back-facing points before z-buffering.** Without that, a flat package in the dev clip fails because pixels receive only far-side points. T1 alone was too weak to see half-pixel errors, which is why T1b exists.

### M1 (500 eval clips, GSO, Blender 4.2.1)

- N distribution for N = 1/2/3/4: 128/143/142/87.
- 0 failed seeds.
- Time per clip in a single process: median 19.3 s, mean 22.4 s. Two workers on one RTX 3060 6 GB: 13.7 s/clip effective.
- 54 of 500 clips needed a retry. 87 of 587 attempts were rejected: 30 at the pre-render check, 57 after rendering.
- Mean background arc resamples per clip: 0.004.
- 707 of 1033 GSO assets used, 440 distinct backgrounds.
- 9.2 GB total.

## 9. Known limitations

- **No texture ground truth.** `rgb` is lit; there is no de-lit target.
- **Truncation:** about 3% of objects touch the frame edge in some frame (Section 4). Visibility rules use the image-clipped amodal mask, so they do not flag this.
- **No parallax in the background,** hence the 70° arc cap and a flat photo behind floating objects.
- **Selection bias from retries:** released clips lean towards less-occluded layouts (Section 3). No upper visibility cap is applied, so fully visible objects are common by design.
- **Mixed Blender versions** change RGB by at most 2/255 and nothing else; check `meta.json` if exact RGB reproducibility matters. RGB is reproducible only up to renderer noise (T5).
- **Hardware:** NVIDIA GPU required (OptiX).
- **SAM 3D conventions unverified** until T6 (Section 7).
- **Train split not generated yet:** the Objaverse filter thresholds are TBD and M2 waits for T6.

## 10. Licenses

- **GSO:** CC BY 4.0. Attribution to Google Research is required.
- **SA-1B:** research license. Redistribution of the images is restricted, so we ship only image IDs and paths, never the photos. Composited `rgb` frames contain SA-1B pixels, so a released eval or train set inherits this restriction (our reading, not legal advice; to be confirmed before any distribution beyond the project group and examiner).
- **TRELLIS:** MIT, Copyright (c) Microsoft Corporation, for the copied code in `src/sam3dvid/blender/normalize_assets.py`.
- **Objaverse-XL:** per-object licenses, recorded in each asset's `meta.json`. Which licenses are admitted into the train set is TBD.
- **BlenderProc and Blender** are used as tools. **SAM 3D Objects** is not included and is under its own license.
