# DATASET.md: implementer decisions

This documents every decision the spec (`docs/dataset_spec.md`, Section 9) leaves to the implementer, plus the milestone results and known limitations. The file format and conventions for consumers are in [`docs/data_format.md`](docs/data_format.md) and are not repeated here. Parameter values are the defaults in `src/sam3dvid/config.py`; the exact config of a dataset is stored in `dataset/config.yaml`. Items marked TBD are not decided yet. Anything marked "inferred" was not measured.

## 1. Asset preprocessing and filters

**Canonical frame: Blender Z-up after import and TRELLIS `normalize_scene` (copied verbatim), baked into one triangulated mesh with identity transform.** This is what the spec asks for (replicate TRELLIS exactly), and it makes the canonical frame well defined independent of the source format.

- GSO OBJs are already Z-up, so they are imported with no axis change. TRELLIS's Y-up OBJ default would lay them on their side.
- `mesh.glb` is a standard Y-up glTF. Outside Blender, apply `conventions.CANONICAL_FROM_GLTF`. Checked: 100% of surface samples land in occupied voxels.

**Voxels: TRELLIS `voxelize.py` replicated exactly** (open3d, Blender PLY export, clip at ±(0.5 - 1e-6)), 64³ surface occupancy.

**Aesthetic threshold: ≥ 5.5** for the Objaverse-XL subset of TRELLIS-500K. TRELLIS training configs re-filter at ≥ 4.5; we keep the stricter list value.

**Geometric filter (paper App. B.1), train only.** Scores are computed at preprocessing and stored in `meta.json` under `filter_scores`. The thresholds live in `Config` and are applied when clips are generated (`assetfilter.accepted_assets`), so they can change without reprocessing. They were chosen on contact sheets of 100 Objaverse assets.

| Criterion | Score | Reject if |
|---|---|---|
| Near-zero volume | smallest bounding-box extent `extent_min`; occupied voxels `num_voxels` | `extent_min` < 0.1 or `num_voxels` < 500 |
| Minimal normal variation | smallest eigenvalue of the area-weighted normal moment `normal_eig_min` | < 0.03 |
| Spatial outlier fragments | area share of components far (> 0.25) from the largest one `outlier_area_frac` | > 0.9 |

- **Loose on purpose:** thin rods score low on normal variation (0.05 for a carrot), and characters with detached limbs reach an outlier share of 0.7. Stray far-away fragments show up as `num_voxels` instead, because normalization shrinks the real object: rejected specks had 12-144 voxels, and the next smallest asset 2169.
- **Result:** 6 of 100 rejected (specks, a coin, a disc, a fractal).

**Scene filter: reject Objaverse assets whose TRELLIS-500K captions contain a scene word (house, building, rooms, floor plan, landscape, terrain, diorama, ...; full list in `Config.filter_scene_words`).** About a quarter of the aesthetic-filtered Objaverse assets are whole scenes (floor plans, houses on ground plates, city blocks) that pass every geometric criterion but don't belong at object scale in a 1-4-object scene. Captions are already in the TRELLIS list, which makes this cheaper and more predictable than a geometric heuristic. On the 100 test assets it rejected 27, and the contact sheet showed all 27 to be scenes or buildings. "interior" is left out because it matched a car, and a toy house counts as an acceptable false positive. Overall, 67 of 100 assets are accepted.

**Eval keeps every GSO asset:** neither filter is applied to it, so the eval pool and seeds 0-499 are unchanged.

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

**`sam3d.ours_to_sam3d` and `sam3d_to_ours`: only the camera axes and the rotation format change; the object frame and scale are the same as ours.** SAM 3D's layout maps canonical points as `x_cam = (s * x_local) @ R(q) + t` (row vectors, wxyz quaternion), in PyTorch3D camera axes (x left, y up, z forward) = `diag(-1, -1, 1)` of OpenCV. So `q = quat((F R_co)^T)`, `t = F t_co`, `s = s_co` with `F = diag(-1, -1, 1)`. Round trips and agreement with SAM 3D's posing formula are tested in `tests/test_sam3d_converter.py`.

This was determined from SAM 3D's outputs, not its source (its pose code was not readable here). The evidence:
- **Pose fit:** given the ground-truth pointmap, the converted prediction lands on our ground truth with a median translation error of 2% and a scale ratio of 1.11. A wrong axis or frame convention would show as errors of order 100%.
- **Object frame:** our canonical +Z maps to SAM 3D's +Z for 27 of 50 objects, and the predicted yaws cluster at 0. The rest are flips and side axes of symmetric or ambiguous objects.
- **The canonical voxel check is weak:** identity ranks first among the 24 axis rotations, but only narrowly (mean voxel IoU 0.266 against 0.260), because SAM 3D chooses its own yaw for each object.

**T6 setup:** SAM 3D Objects (commit f91db411) on an A100 40 GB, frame 0 of eval clips 0-19, 50 objects. Each object runs twice: once with SAM 3D's own MoGe pointmap (`moge`), and once with our ground-truth depth as the pointmap (`gt_pointmap`), as in the paper's layout evaluation. Metrics follow App. D.3 (`python -m sam3dvid.t6 analyze`, report in `t6/report.json`); scripts are in `scripts/t6/`, the procedure in `docs/t6_runbook.md`.

| | ADD-S | ADD-S@0.1 | 3D IoU | ICP-Rot | F1@0.01 | Chamfer |
|---|---|---|---|---|---|---|
| Ours, `gt_pointmap` | 0.064 | 0.88 | 0.548 | 14.5° | 0.187 | 0.058 |
| Ours, `moge`, depth-scale aligned | 0.474 | 0.08 | 0.083 | 14.6° | 0.170 | 0.065 |
| Paper (layout: ADT, Table 3; shape: SA-3DAO, Table 2) | 0.077 | 0.767 | 0.497 | 15.25° | 0.234 | 0.040 |

- **Layout with the GT pointmap matches or beats the paper on every metric.** This confirms the converter and shows our renders are within SAM 3D's input distribution.
- **Shape F1 is below the paper.** We sample 100k points where the paper uses 1M, and at that density the point spacing is close to the 0.01 threshold, which biases F1 low (inferred, unchecked). It is also a different object set.
- **MoGe layouts are far off in our units.** MoGe predicts metric depth, our scenes have arbitrary units, and the median scale ratio is 0.29. Even after one global depth-scale factor, ADD-S stays at 0.47. **Consequence for training without a pointmap input:** translation and scale targets need a scale-invariant formulation, e.g. normalized by the depth of the scene or object.

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
- **Full test suite on all 500 clips** (T1, T1b, T3, T4, format; about 45 min): all pass. Before T1 was relaxed (below), 16 clips failed it while T1b passed on every clip.

**T1 fails only on points outside the amodal mask (< 95%); its coverage and depth checks are warnings.** On the full M1 run they flagged object 1 of 16 clips, and T1b passed on all of them, so the data itself is fine. The flags came in two kinds:
- amodal coverage of 0.84-0.89 (bound 0.90) with a depth error near 3e-4: 10k points are too sparse to cover a large silhouette;
- depth error of 0.1-0.2 on hollow objects (bowls, pots, a caddy): the point z-buffer likely sees through to the far wall (inferred).

T1b measures silhouette and depth exactly on the full mesh, so T1 keeps only the check T1b lacks: points projecting outside the mask, which a wrong pose would cause.
- 9.2 GB total.

## 9. Known limitations

- **No texture ground truth.** `rgb` is lit; there is no de-lit target.
- **Truncation:** about 3% of objects touch the frame edge in some frame (Section 4). Visibility rules use the image-clipped amodal mask, so they do not flag this.
- **No parallax in the background,** hence the 70° arc cap and a flat photo behind floating objects.
- **Selection bias from retries:** released clips lean towards less-occluded layouts (Section 3). No upper visibility cap is applied, so fully visible objects are common by design.
- **Mixed Blender versions** change RGB by at most 2/255 and nothing else; check `meta.json` if exact RGB reproducibility matters. RGB is reproducible only up to renderer noise (T5).
- **Hardware:** NVIDIA GPU required (OptiX).
- **Train split not generated yet** (M2).

## 10. Licenses

- **GSO:** CC BY 4.0. Attribution to Google Research is required.
- **SA-1B:** research license. Redistribution of the images is restricted, so we ship only image IDs and paths, never the photos. Composited `rgb` frames contain SA-1B pixels, so a released eval or train set inherits this restriction (our reading, not legal advice; to be confirmed before any distribution beyond the project group and examiner).
- **TRELLIS:** MIT, Copyright (c) Microsoft Corporation, for the copied code in `src/sam3dvid/blender/normalize_assets.py`.
- **Objaverse-XL:** per-object licenses, recorded in each asset's `meta.json` with author and source URL. **All licenses are admitted, because the dataset is for research only.** Of the 100 test assets, 78 are CC BY, 20 are non-commercial (BY-NC, BY-NC-SA) and 2 are CC0. Non-commercial and share-alike terms would matter for any wider release.
- **BlenderProc and Blender** are used as tools. **SAM 3D Objects** is not included and is under its own license.
