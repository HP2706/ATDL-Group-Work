# Synthetic video dataset: data format (v0.1)

This describes the data we generate for extending SAM 3D (image -> 3D objects) to short videos. It is meant to be everything a consumer needs. If anything is unclear, ask Julian.

## 1. What a sample is

One sample is a **clip**: 32 RGB frames at 512x512 of a static scene with N in {1,2,3,4} rigid objects floating in space (no ground, random orientations), seen by a camera that orbits the scene. The rendered objects are alpha-composited onto a real photo (SA-1B). Every clip comes with exact ground truth:

- per object: its shape (a 64^3 voxel grid and a textured mesh, in a canonical object frame) and its pose in the world;
- per frame: camera pose and intrinsics, visible instance masks, amodal masks, depth.

The scene never changes during a clip; only the camera moves. So each object's pose **in camera coordinates** changes from frame to frame, and that per-frame camera-frame pose (rotation, translation, scale) is the layout target SAM 3D predicts.

| Split | Objects | Backgrounds | Clips | Where generated |
|---|---|---|---|---|
| `eval` | Google Scanned Objects (GSO) | SA-1B eval pool | 500 | locally, shipped as a tarball |
| `train` | Objaverse-XL (TRELLIS-500K subset, filtered) | SA-1B train pool | 10,000 | on a cluster, later |
| `val` | held-out Objaverse-XL objects (requested) | SA-1B train pool | 300–500 | with `train` |

All splits use the identical format. Objects and backgrounds never cross splits: object sets come from different sources, and SA-1B images are assigned to a pool by a fixed hash of their image ID.

## 2. Directory layout

```
dataset/
  manifest.jsonl                  one line per clip: {clip_id, split, N, arc_span_deg, seed, path}
  config.yaml                     generator config used
  assets/{train|val|eval}/{asset_id}/
    mesh.glb                      normalized textured mesh (see 3.3)
    voxels.npy                    bool (64,64,64) surface occupancy (see 3.3)
    meta.json                     source, source_name (original GSO/Objaverse name), license,
                                  normalization (scale, offset, raw bbox), num_voxels, filter_scores
  clips/{split}/{clip_id}/
    rgb/000.png ... 031.png       uint8 RGB, composited final frames  <- model input
    rgba_fg/000.png ...           uint8 RGBA, objects only, straight (non-premultiplied) alpha
    mask_visible/000.png ...      uint8 instance map: 0 = background, k = object 1..N
    mask_amodal/obj{k}/000.png    uint8 {0,255}: full silhouette of object k
    depth/000.npz ...             key "depth": float32 (512,512) z-depth, NaN = background
    cameras.json
    objects.json
    meta.json
```

All paths stored inside JSON files are relative to `dataset/`. Clips reference shared assets and don't copy them.

## 3. Conventions (read this carefully)

### 3.1 Frames and naming

- `R_ab`, `t_ab`, `T_ab` map coordinates **from frame b into frame a**: `x_a = R_ab x_b + t_ab`. `w` = world, `c` = camera, `o` = object.
- **World:** right-handed, Z up. Units are arbitrary "scene units" (no metric scale; see 3.4).
- **Camera:** OpenCV (x right, y down, z forward, looking along +z).

### 3.2 Camera (`cameras.json`)

```json
{"convention": "opencv", "width": 512, "height": 512,
 "K": [[548.993, 0, 255.5], [0, 548.993, 255.5], [0, 0, 1]],
 "frames": [{"t": 0, "T_cw": [[4x4]]}, ..., {"t": 31, "T_cw": [[4x4]]}]}
```

- `T_cw` is **world -> camera** (extrinsics), 4x4 with last row `[0,0,0,1]`.
- `K` is identical for all frames and clips: horizontal FOV 50 degrees, `f = 256 / tan(25 deg)`, square pixels, no distortion.
- **Pixel centers are at integer coordinates** (OpenCV convention). The principal point `255.5` is the exact image center, and pixel `(u, v)` covers `[u-0.5, u+0.5]`. A point `x_c` projects to `u = fx * x/z + cx`, `v = fy * y/z + cy`.

### 3.3 Objects: canonical frame, shape, pose

**Canonical object frame.** Each asset is normalized exactly as TRELLIS does it:
- Z up, as Blender sees the asset after import.
- The longest bounding-box side is scaled to 1 and the box is centered.
- Vertices therefore lie in `[-0.5, 0.5]^3`. For GSO they touch both faces along the longest axis. For Objaverse assets made of several rotated parts, TRELLIS's box can be slightly larger than the true vertex extent.

**`voxels.npy`**
- `bool[64,64,64]` holding **surface** occupancy (TRELLIS-style, not solid), in the canonical frame.
- Index `[i,j,k]` is along `(x,y,z)`; the voxel center is `(index + 0.5) / 64 - 0.5`.

**`mesh.glb`**
- A standard glTF file, which is Y-up by glTF convention. To get canonical coordinates outside Blender, multiply the vertices:
  ```
  CANONICAL_FROM_GLTF = [[1,0,0],[0,0,-1],[0,1,0]]     # (x, y, z)_gltf -> (x, -z, y)
  ```
  Blender's glTF importer applies this automatically; trimesh, Open3D and PyTorch3D do not.
- One mesh object with an identity transform, already triangulated. Materials and textures are kept as in the source (GSO: one texture).

**Pose in the world (`objects.json`)**, one entry per object, `k` = 1..N, matching the mask ids:

```json
[{"k": 1, "asset_id": "...", "source": "gso|objaverse",
  "mesh": "assets/eval/<id>/mesh.glb", "voxels": "assets/eval/<id>/voxels.npy",
  "R_wo": [[3x3]], "t_wo": [x, y, z], "s_wo": [s, s, s]}]
```

- `x_w = R_wo (s_wo * x_o) + t_wo`, with `*` elementwise.
- `s_wo` is always isotropic (all three entries equal) but stored as a 3-vector to match SAM 3D's `s in R^3`.
- `t_wo` is where the canonical origin (the bbox center) ends up, not the center of mass.
- All rotations are stored as 3x3 matrices.

**Per-frame camera-frame target** (what to train on):

```
R_co = R_cw @ R_wo
t_co = R_cw @ t_wo + t_cw
s_co = s_wo
x_c  = R_co @ (s_co * x_o) + t_co
```

**This is our convention, not necessarily SAM 3D's.** SAM 3D's layout output may use PyTorch3D camera axes and a different canonical frame or scale. A tested converter `ours_to_sam3d` is still to come. Don't assume the two conventions are equal.

### 3.4 Scale

There is no metric scale. The canonical object fits a unit cube, `s ~ U(0.4, 1.0)`, and the camera distance is chosen relative to the scene size. `t_co` and `s_co` are in this generator's scene units, not meters.

## 4. Per-frame images and masks

| File | Content | Notes |
|---|---|---|
| `rgb/t.png` | final frame, sRGB uint8 | lit by the scene light, Blender "Standard" view transform |
| `rgba_fg/t.png` | objects only | anti-aliased alpha; `rgb = fg*a + bg*(1-a)` |
| `mask_visible/t.png` | which object owns each pixel | not anti-aliased; values 0..N |
| `mask_amodal/obj{k}/t.png` | object k's silhouette as if no other object existed | clipped to the image; visible pixels of k are always a subset |
| `depth/t.npz` | z-depth along the optical axis (the OpenCV `z` of the surface point), **not** distance along the ray | finite exactly where `mask_visible > 0`, NaN elsewhere; the background has no depth |

Because the alpha in `rgba_fg` is anti-aliased, it can extend about 1 px beyond `mask_visible`. Those edge pixels have no depth (NaN).

## 5. `meta.json` (per clip)

| Key | Meaning |
|---|---|
| `clip_id`, `split`, `seed`, `attempt` | identity; the clip is fully reproducible from `seed` |
| `num_objects` | N |
| `arc_span_deg`, `arc_direction` | total orbit angle (10-70 deg) and its sign |
| `start_azimuth_deg`, `start_elevation_deg`, `elevation_drift_deg` | trajectory parameters |
| `orbit_radius`, `scene_center` | camera distance to and look-at point (world) |
| `hfov_deg` | 50 |
| `light` | `sun_direction` (unit vector *towards* the sun, world frame), `sun_strength`, `ambient_strength` |
| `background` | `image_id` (SA-1B), `width`, `height`, `virtual_hfov_deg` (130), `K_bg`. Frame t's background is the photo warped by the homography `K @ R_cw(t) @ R_cw,ref^T @ inv(K_bg)` |
| `reference_T_cw` | virtual camera at the arc midpoint that the background is aligned to |
| `visibility` | `visible_px`, `amodal_px`, `visible_frac`, each a `[32][N]` array |
| `render`, `timing`, `git_commit`, `generator_version`, `arc_retries`, `rejections` | provenance; ignore for training |

## 6. Guarantees

Enforced by the generator (failing clips are resampled) and re-checked by the test suite on every released eval clip:

- **Visibility:** every object in every frame has at least 524 visible pixels (0.2% of the image) and at least 10% of its amodal area visible. Objects can be partly occluded but never vanish.
- **Background:** it covers every pixel of every frame.

Enforced by construction:

- **No interpenetration:** objects' bounding spheres (radius `s * sqrt(3) / 2`) are disjoint.

Checked by the test suite:

- **Geometry**, on every eval clip, with 10k surface points per object and frame:
  - at least 95% of projected points fall inside the amodal mask dilated by 3 px;
  - the 3 px-dilated points cover at least 90% of the mask;
  - the median depth error at visible pixels is below 1% of the orbit radius.
  These tolerances are loose for small objects: they don't catch a 5 degree pose error. Tighter checks are planned.
- **Determinism**, on one clip regenerated from its seed:
  - poses, masks and JSON are identical (except `meta.json` `timing` and `git_commit`);
  - depth matches to 1e-5 relative;
  - RGB matches up to renderer noise.

## 7. How the data is distributed

| Quantity | Distribution |
|---|---|
| N | 1, 2, 3, 4 with p = 0.30, 0.30, 0.25, 0.15 |
| object rotation `R_wo` | uniform on SO(3) |
| scale `s` | U(0.4, 1.0) |
| object positions | uniform in a ball of radius N^(1/3), resampled until no overlap |
| camera | looks at the centroid of the object centers, Z up, no roll; distance `r / tan(25 deg) * U(1.1, 1.6)`, where r is the scene's bounding radius |
| start azimuth / elevation | U(0, 360) / U(10, 45) deg (with drift, camera elevation spans 0-55 deg) |
| arc span / elevation drift | U(10, 70) deg with random sign / U(-10, 10) deg |
| camera timing | eased: `u(t) = (1 - cos(pi t / 31)) / 2`, so motion is slow at both ends |
| light | one sun (uniform on the upper hemisphere) plus low ambient, fixed in world for the clip |
| background | an SA-1B photo treated as a plane at infinity, i.e. warped by the camera's rotation only (no parallax) |

These are the distributions *before* rejection. N, the assets, the background and the light are drawn once per seed and never resampled, so their distributions are exact. Rotations, scales, positions and the camera are redrawn when a clip fails the visibility or background checks, so released clips lean slightly towards less-occluded configurations and arcs that keep the background covered.

The model side is expected to subsample 16 of the 32 frames.

## 8. Minimal loader (numpy only)

```python
import json, numpy as np
from PIL import Image

root, clip = "dataset", "dataset/clips/eval/00000000"
cams = json.load(open(f"{clip}/cameras.json"))
objs = json.load(open(f"{clip}/objects.json"))
K = np.array(cams["K"])

t, k = 0, 1
T_cw = np.array(cams["frames"][t]["T_cw"])
o = objs[k - 1]
R_wo, t_wo, s_wo = np.array(o["R_wo"]), np.array(o["t_wo"]), np.array(o["s_wo"])
R_co = T_cw[:3, :3] @ R_wo
t_co = T_cw[:3, :3] @ t_wo + T_cw[:3, 3]

rgb = np.asarray(Image.open(f"{clip}/rgb/{t:03d}.png"))
visible_k = np.asarray(Image.open(f"{clip}/mask_visible/{t:03d}.png")) == k
amodal_k = np.asarray(Image.open(f"{clip}/mask_amodal/obj{k}/{t:03d}.png")) > 0
depth = np.load(f"{clip}/depth/{t:03d}.npz")["depth"]

# Sanity check: project the voxel centers; typically >90% land on the amodal mask (voxel
# centers sit up to half a voxel off the surface, which matters for small objects).
vox = np.load(f"{root}/{o['voxels']}")
x_o = (np.argwhere(vox) + 0.5) / 64 - 0.5
x_c = (x_o * s_wo) @ R_co.T + t_co
uv = np.round(x_c[:, :2] / x_c[:, 2:] @ K[:2, :2].T + K[:2, 2]).astype(int)
inside = (uv >= 0).all(1) & (uv < 512).all(1)
print("voxel centers on amodal mask:", amodal_k[uv[inside, 1], uv[inside, 0]].mean())
```

## 9. Known limitations and things that may change

- **Version 0.1.** The file format is stable. The sampling distributions in section 7 may still be tuned before M1: objects are currently small in multi-object clips, and the largest object in a test clip covers about 2% of the image.
- **No de-lit texture target.** Texture ground truth is out of scope; `rgb` is lit.
- **No SAM 3D converter yet.** The SAM 3D convention converter (`ours_to_sam3d`) and a SAM 3D baseline sanity check are pending.
- **Train split not generated yet.** Its Objaverse asset filtering and pool size are still being decided; the format will be identical.
