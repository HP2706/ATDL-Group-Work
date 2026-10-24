# Synthetic Video Dataset Spec — SAM 3D → Video (ATDL group project)

## 0. Context

We are extending SAM 3D ("SAM 3D: 3Dfy Anything in Images", Meta, arXiv:2511.16624) from single images to short videos. SAM 3D takes an image `I` and an object mask `M` and predicts shape `S`, texture `T`, and layout `(R, t, s)` in camera coordinates. Code and weights: https://github.com/facebookresearch/sam-3d-objects (inference only, no training code).

This repo builds the **synthetic training and evaluation data**. Each sample is a short video clip of a static scene containing the same N rigid objects in every frame, seen from a moving camera, with exact 3D ground truth.

**Guiding principle:** wherever a design choice exists, make the same choice as the SAM 3D paper. The closest analogue in the paper is the *Flying Occlusions (FO)* render-paste data (paper App. B.2.1): synthetic objects alpha-composited onto natural SA-1B images. This dataset is the video version of FO.

Section 2 lists fixed decisions. Do not change them without asking the user. Section 9 lists what the implementer decides and documents.

---

## 1. Deliverables

1. A deterministic, config-driven generation pipeline (`seed → clip`).
2. An **asset preprocessing** step producing normalized meshes and voxels.
3. Milestones (Section 8):
   - **M0:** one validated clip.
   - **M1:** 500 eval clips (GSO).
   - **M2:** 10,000 train clips (Objaverse).
4. A test suite implementing the acceptance tests in Section 7.
5. A short `DATASET.md` describing the final format, conventions, and any thresholds chosen.

---

## 2. Fixed decisions

| Topic | Decision | Paper alignment |
|---|---|---|
| Renderer | BlenderProc (Blender, Cycles on GPU) | TRELLIS (which SAM 3D builds on) renders with Blender |
| Scene style | Objects float, no ground plane, random orientation; RGBA render alpha-composited onto an SA-1B image | FO, App. B.2.1 |
| Train assets | Objaverse-XL subset of the TRELLIS-500K list, aesthetic-filtered + geometric filter | Iso-3DO from Objaverse-XL; aesthetic filter as in Xiang et al. 2025; geometric filter App. B.1 |
| Eval assets | Google Scanned Objects (GSO), all of it | Paper evaluates its VAE on GSO (Table 11); unlikely to be in SAM 3D's shape training data |
| Backgrounds | SA-1B images, landscape orientation only; train/eval image sets disjoint | SA-1B is the paper's main natural-image source |
| Objects per clip | N ∈ {1,2,3,4} with p = {0.30, 0.30, 0.25, 0.15} | SA-3DAO object-count distribution (Fig. 16) |
| Scene dynamics | All objects static and rigid. Only the camera moves. | — |
| Camera motion | Orbit around scene centroid, always looking at it; arc span ~ U(10°, 70°), logged | — (Section 4.4 explains the 70° cap) |
| Intrinsics | Fixed: 512×512, horizontal FOV 50°, principal point at image center, no distortion | — |
| Clip length | 32 frames (model side subsamples to 16) | V-JEPA 2.1 pretrains on 16 frames, tubelet 2 |
| Lighting | One sun light + low ambient; direction and intensity randomized per clip, fixed in world for the whole clip | Random lighting on inputs, App. B.3 |
| Visibility rules | Every object, every frame: visible pixels ≥ 0.2% of image, and visible/amodal ≥ 0.10 | FO thresholds, App. B.2.1 |
| Texture GT | Out of scope (no ambient-lit "de-lighted" target pass) | — |
| Storage | Plain per-clip folders: PNG + `.npy` + JSON | — |

---

## 3. Coordinate conventions (critical — get these right first)

- **World:** right-handed, Z-up.
- **Camera:** OpenCV convention (x right, y down, z forward). Blender cameras look down −Z with +Y up, so convert explicitly. Store `T_cw` (world→camera, 4×4) per frame. Subscripts follow `R_ab`: maps frame-b coordinates into frame a, so `x_c = R_cw · x_w + t_cw`.
- **Object canonical frame:** the TRELLIS-normalized frame of the asset (Section 4.1). Mesh vertices `x_o` live in roughly `[-0.5, 0.5]³`.
- **Object pose:** `x_w = R_wo · (s_wo ⊙ x_o) + t_wo`.
  - `R_wo` is a 3×3 rotation.
  - `s_wo` is a per-axis 3-vector, since SAM 3D's `s ∈ R³`. It is sampled isotropically but stored as a 3-vector.
  - `t_wo` is a 3-vector.
- **Per-frame camera-frame target** (what SAM 3D predicts): `R_co = R_cw · R_wo`, `t_co = R_cw · t_wo + t_cw`, `s_co = s_wo`. Provide a helper `object_pose_in_camera(clip, k, t)`.
- **Rotations are stored as 3×3 matrices.** Conversion to the 6D representation (Zhou et al. 2019) happens on the model side.
- **SAM 3D's output convention must be determined from its repo.** It depends on PyTorch3D (`requirements.p3d.txt`), so its camera frame may follow PyTorch3D's axes (x left, y up) rather than OpenCV. Inspect the inference pipeline's layout output, then write and unit-test a converter `ours_to_sam3d(R, t, s)` and its inverse. Acceptance test T6 verifies this.
- **The converter may also need a fixed object-frame rotation** (and scale factor), not only a camera-axis change: if SAM 3D's canonical shape frame differs from ours (e.g. glTF Y-up vs Blender Z-up after normalization, or a `[-1,1]` vs `[-0.5,0.5]` cube), predicted rotations are off by a constant. T6's ADD-S compares posed points and its shape metrics run after ICP, so neither detects this; the canonical voxel check in T6 does.

---

## 4. Pipeline

### 4.1 Asset preprocessing (one-time)

- **Train:** download the Objaverse-XL entries listed in TRELLIS-500K (TRELLIS's `dataset_toolkits` may be reused for download). Apply TRELLIS's aesthetic-score threshold. Find the value TRELLIS actually uses and record it.
- **Geometric filter (paper App. B.1).** Remove meshes with any of:
  - near-zero volume (point-like)
  - minimal normal variation (flat or sheet-like)
  - spatial outlier fragments far from the main structure

  Thresholds are the implementer's choice. Log them and save a contact sheet of 50 rejected and 50 accepted assets for manual review.
- **Eval:** all GSO objects.
- **Normalization and voxels:** replicate TRELLIS's normalization (center and scale into the unit cube) and its 64³ voxelization **exactly, from TRELLIS source code**, not from this spec. Handle the GLB Y-up to Blender Z-up import consistently. The canonical frame is the asset after import and normalization.
- **Output:** `dataset/assets/{train|eval}/{asset_id}/mesh.glb` (normalized, textured), `voxels.npy` (bool, 64³), `meta.json` (source, license, original scale, filter scores).

### 4.2 Background preprocessing

- Sample SA-1B images. Keep landscape images (W ≥ H) with short side ≥ 1024 px.
- Split into disjoint train and eval pools. Store only image IDs plus paths; do not copy.

### 4.3 Scene sampling

- Sample N from the Section 2 distribution, then sample assets uniformly from the split's pool without replacement within a clip.
- **Rotation:** uniform on SO(3).
- **Scale:** isotropic `s ~ U(0.4, 1.0)`, applied to the canonical unit-cube mesh.
- **Position:** uniform in a ball of radius `ρ(N)` around the world origin, with `ρ` growing with N so that even N = 4 objects at s = 1.0 fit (implementer picks `ρ(N)` and documents it; `ρ(1) = 1.0`). Reject placements whose bounding spheres (radius `s·√3/2`) overlap, so objects never interpenetrate. On overlap, resample **positions only**; rotations and scales are kept, so the scale distribution stays unbiased.
- **Scene centroid:** `c = mean(t_wo)`. Scene radius `r = max_k(‖t_wo,k − c‖ + bounding radius_k)`.

### 4.4 Camera trajectory

- **Orbit radius:** `d = r / tan(FOV/2) · U(1.1, 1.6)`, constant over the clip.
- **Start:** azimuth `φ₀ ~ U(0°, 360°)`, elevation `θ₀ ~ U(10°, 45°)` above the horizontal plane through `c`.
- **Arc:** span `Δφ ~ U(10°, 70°)` with random sign; elevation drift `Δθ ~ U(−10°, 10°)`, linear in the eased parameter.
- **Easing:** `u(t) = (1 − cos(π·t/31)) / 2` for t = 0…31.
- **Orientation:** look at `c`, world up = +Z, no roll.

The arc span is capped at 70° because the background is treated as a plane at infinity (Section 4.6). The background is a homography of a single photo, which can only cover a limited rotation.

### 4.5 Rendering (per frame)

- **Cycles on GPU with a transparent film.** Sample count and denoiser are the implementer's choice. Use the *Standard* view transform (not Filmic or AgX) so colors match the sRGB textures.
- **Outputs:**
  - RGBA of the objects.
  - Visible instance segmentation.
  - Z-depth along the camera axis, in OpenCV z, objects only.
  - **Amodal mask per object:** each object's silhouette with the other objects hidden. A cheap silhouette pass is fine.
- **Lighting:** one sun (direction uniform on the upper hemisphere, intensity randomized within a range the implementer picks and logs) plus low ambient world light. Fixed for the whole clip.

### 4.6 Compositing (background at infinity)

- **Virtual background camera.** Give the SA-1B image a virtual horizontal FOV of 130°, with principal point at its center. This defines `K_bg`. (At 120°, a 35° yaw already puts the frame corner exactly on the image edge.)
- **Reference camera.** Align the background with a *virtual* camera at the arc midpoint `u = 0.5` (azimuth `φ₀ + Δφ/2`, elevation `θ₀ + Δθ/2`), not with a rendered frame. With 32 frames and the easing, t = 16 sits at u = 0.525, which would put frame 0 up to 36.8° from the reference. With the virtual reference, the maximum rotation away from it is ≤ 35°.
- **Warp per frame:** `H_t = K · R_rel(t) · K_bg⁻¹` with `R_rel(t) = R_cw(t) · R_cw,refᵀ`. Warp with `cv2.warpPerspective` onto 512×512.
- **Analytic coverage pre-check.** Before rendering, map the output image corners of every frame into the background image (rays must be in front of the background camera, with an interpolation margin). The valid region is convex, so corners suffice. On failure, resample **only the arc** (`Δφ`, `Δθ`) and log the number of arc retries.
- **Assert full coverage.** No undefined pixels are allowed; if any appear, reject the clip.
- **Composite:** alpha-composite the RGBA foreground over the warped background.

### 4.7 Validation and rejection

- **Every object in every frame must have:**
  - visible pixel count ≥ 0.002 · 512² (= 524 px), and
  - visible / amodal pixel ratio ≥ 0.10, with the amodal mask counted within image bounds.
- On failure, resample the scene with a derived seed (max retries configurable). Log the rejection reason.
- A cheap pre-render check (projected bounding spheres, or a low-resolution silhouette pass) is encouraged to avoid wasted full renders.

---

## 5. Output format

```
dataset/
  manifest.jsonl                 # one line per clip: clip_id, split, N, arc_span_deg, seed, path
  config.yaml                    # exact generator config used
  assets/{train|eval}/{asset_id}/ # Section 4.1 output
  clips/{split}/{clip_id}/
    rgb/000.png … 031.png        # uint8 RGB 512×512, composited
    rgba_fg/000.png … 031.png    # uint8 RGBA, objects only, before compositing
    mask_visible/000.png …       # uint8 instance map: 0 = background, k = 1..N
    mask_amodal/obj{k}/000.png … # uint8 {0,255}
    depth/000.npy …              # float32 H×W, OpenCV z-depth, NaN = background
    cameras.json
    objects.json
    meta.json
```

**`cameras.json`**
```json
{"convention": "opencv", "width": 512, "height": 512,
 "K": [[fx,0,cx],[0,fy,cy],[0,0,1]],
 "frames": [{"t": 0, "T_cw": [[...4x4...]]}, ...]}
```

**`objects.json`** (meshes and voxels are referenced, not copied; paths are relative to `dataset/`)
```json
[{"k": 1, "asset_id": "...", "source": "objaverse|gso",
  "mesh": "assets/train/<id>/mesh.glb", "voxels": "assets/train/<id>/voxels.npy",
  "R_wo": [[...3x3...]], "t_wo": [x,y,z], "s_wo": [sx,sy,sz]}]
```

**`meta.json`** contains:
- seed, git commit, generator version
- `arc_span_deg`, start azimuth and elevation, elevation drift, orbit radius, FOV
- light parameters, SA-1B image ID, `K_bg` virtual FOV
- per frame and per object: `visible_px`, `amodal_px`, `visible_frac`
- retries and render time

---

## 6. Splits

- **Train:** Objaverse assets + SA-1B train pool.
- **Eval:** GSO assets + SA-1B eval pool.
- Assets and background images never cross splits.

---

## 7. Acceptance tests (stage gates)

- **T1 — Projection consistency.** For every object and frame, project 10k surface points sampled from the posed mesh with `K, T_cw`.
  - ≥ 95% must fall inside the 3-px-dilated amodal mask.
  - Amodal mask pixels covered by the **3-px-dilated** projected-point image ≥ 90%. (Undilated, 10k points cannot cover a mask of 40k+ pixels.)
  - Depth: **z-buffer** the projected points (keep the nearest point per pixel), after dropping back-facing points (normal facing away from the camera; otherwise thin objects fail because many pixels only receive far-side points), then at pixels where this object is visible and has a point, the median |z-buffered point depth − rendered depth| must be < 1% of the orbit radius. (Without the z-buffer, the ~50% of points on the far side of the object dominate the median.)
- **T2 — Convention unit tests.** Hand-constructed poses with known projections. Round-trip Blender ↔ OpenCV camera conversion. Round-trip `ours_to_sam3d` ↔ inverse.
- **T3 — Visibility rules.** Assert the Section 4.7 rules on every saved clip.
- **T4 — Background coverage.** Assert there are no undefined background pixels in any frame.
- **T5 — Determinism.** The same seed reproduces identical JSON, masks, and poses; RGB may differ only by renderer noise.
- **T6 — SAM 3D smoke test.** On 20 eval clips, run released SAM 3D on frame 0 with the GT visible mask (and a GT pointmap from depth if the pipeline accepts one). Convert our GT to frame-0 camera coordinates via `ours_to_sam3d`. Report shape F1@0.01 and Chamfer, and layout ADD-S and ADD-S@0.1, using the paper's protocol (App. D.3: normalize to [−1,1], ICP, 1M points). Numbers should look clearly better than chance and broadly in line with paper Tables 2–3. Badly off numbers almost always mean a convention bug.
  - **Canonical voxel check (no ICP).** Compare SAM 3D's predicted canonical voxels/shape directly against our `voxels.npy` (after `ours_to_sam3d`'s object-frame part) without any alignment step. Shape IoU should be clearly highest at the identity among the 24 axis-aligned rotations; otherwise the object-frame convention is wrong.
- **T7 — Visual review.** For 20 random clips, write an mp4 with mask overlays plus a contact sheet. A human checks these before M1 and M2 are signed off.

---

## 8. Milestones

- **M0:** asset preprocessing for ~50 GSO objects; one clip end to end; T1–T5 and T7 pass. T6 and the canonical voxel check are deferred until those pass (SAM 3D is installed after that).
- **M1:** 500 eval clips (GSO). All tests pass. Report sec/clip and rejection rate.
- **M2:** 10,000 train clips (Objaverse), generated in parallel (one clip per job). All tests pass on a random 2% sample, plus T3 and T4 on everything.

---

## 9. Implementer decides (and documents in `DATASET.md`)

- Geometric filter thresholds; the aesthetic threshold value found in TRELLIS.
- Cycles samples, denoiser, light intensity ranges.
- Pre-render visibility check method; retry limit.
- Parallelization and cluster launch scripts.
- How SAM 3D's layout output convention is determined, and the resulting converter.

## 10. Out of scope

Texture ground truth, dynamic objects, ground planes or physics, MOVi-E or other external datasets, model code.
