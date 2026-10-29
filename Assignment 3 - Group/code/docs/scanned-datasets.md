# OmniObject3D training and GSO evaluation

The existing dataset factory, trainer, checkpoint format and evaluator support
`data.source=omniobject3d` and `data.source=gso`. Each prepared dataset has a
`manifest.jsonl` containing ordered RGB frames, matching masks, normalized scan
geometry and 64³ surface occupancy. Meshes and voxels stay outside model inputs.

OmniObject3D uses the official processed real captures and supplied matting masks.
Keep all supplied frames, normally 200 per object, and sample 16 ordered frames
per training example. Training samples seeded temporal strides and a reference
frame; evaluation samples uniformly. Object identities determine reproducible
90/10 train/validation membership. Captures are symlinked to the retained raw data.

GSO uses a fixed list of 1,030 unique objects from the official Google Research
collection, checked 2026-10-29. GoogleResearch's owner catalogue currently has
1,033 models; the three extra uploads are excluded from evaluation. The collection
API also repeats two entries, which the downloader deduplicates. Membership is
versioned in `src/video3d/data/gso-evaluation-objects.json`. Render 32 views per
object at 384², save exact depth-derived foreground masks, camera intrinsics and
camera-to-normalized-object transforms, and sample 16 views for evaluation. This
is our full-GSO protocol, not a reproduction of a paper's GSO-30 protocol.

Both sources reuse the existing TRELLIS/Open3D voxelizer. Their manifests have
separate provenance and target-latent cache identities. Real SAM 3D target latents
must subsequently be encoded on a GPU with `cache-latents` for each required split.
CPU smoke checks use explicitly fake models and do not create real training targets.

## Persistent UCloud workspace

Branch: `omni-gso-data`, shared by the local worktree and UCloud checkout.

Workspace: `/work/MathiasElmegaardNielsen#6181/omni-gso-data`.

- `repo/Assignment 3 - Group/code`: source checkout.
- `data/raw/{omniobject3d,gso}`: retained archives and extracted originals.
- `data/{omniobject3d,gso}`: manifests, normalized assets, masks and geometry.
- `logs/`: batch stages, tests, source commit and final exit status.
- `data/completion.json`: written only after both full preparations and checks pass.

The foreground Terminal Ubuntu batch script `ucloud/prepare-scanned.sh` installs
the CPU environment, tests the shared path, downloads and prepares GSO, then
downloads and prepares OmniObject3D. It uses four GSO download workers, three
preparation workers, and six Omni archive workers with up to twelve connections
per archive. Omni archives total approximately 1.085 TB compressed. Published
SHA256 checksums are verified; GSO ZIP CRCs are checked. Completed assets are
reused, partial transfers are resumed, and extraction writes files atomically.
The job releases its CPU allocation when its foreground pipeline exits. A failed
stage is retained in its log and prevents a misleading completion report.

Dataset archives, prepared binaries and environments live on persistent storage;
they are not committed to Git. Source code, configuration, fixed evaluation
membership, tests and this operating guide are committed.

## Model configurations

| Preset | Configuration |
|---|---|
| `c3` | Fusion adapter with frozen SAM 3D and V-JEPA |
| `sam_lora` | Fusion adapter plus SAM 3D LoRA |
| `sam_video_lora` | Fusion adapter plus SAM 3D and V-JEPA LoRA |
| `dino_pooler` | Existing per-frame DINO patch tokens, mask-filtered attentive pooler, 64 output tokens |
| `dino_pooler_single` | Same DINO pooling path with one output embedding |

The DINO path retains per-frame patch tokens without pair averaging. The pooler
selects foreground patches using masks, replaces RGB conditioning with its pooled
representation, and retains native non-RGB conditioning, including mask tokens.
The `video_lora` preset also exposes the existing video-only LoRA ablation.
There is no DoRA addition or linear adapter experiment in this change.

## Checks and remaining calibration

`scripts/check_scanned.py --root ROOT --source SOURCE` checks all manifest paths,
all occupancy grids and four decoded RGB/mask pairs per object. It then loads
real samples through each configuration, checks tensor shapes, and performs a
finite CPU smoke forward/backward pass. Tests also cover deterministic sampling,
safe manifest paths, common training, checkpoint resume and switching an
Omni-trained checkpoint to GSO inputs.

The antique source test matched nine meshes to nine captures with 1,800 frames.
Three real GSO meshes rendered successfully for the initial loader test. These
subset checks do not establish full-dataset completion; consult the batch logs
and `data/completion.json` for the full run.

OmniObject3D's scan-to-COLMAP registration and SAM canonical-axis calibration
remain unverified. Do not report pose-dependent or visibility-region metrics as
validated. GSO render cameras are exact. Real pretrained GPU forward/backward
and SAM VAE round-trip checks remain a subsequent GPU validation stage.
