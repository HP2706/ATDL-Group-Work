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
the CPU environment, tests the shared path, downloads and prepares OmniObject3D,
then completes GSO preparation. It uses four GSO download workers, three
preparation workers, and six Omni archive workers with up to twelve connections
per archive. Omni archives total approximately 1.085 TB compressed. Published
SHA256 checksums are verified; GSO ZIP CRCs are checked. Completed assets are
reused, partial transfers are resumed, and extraction writes files atomically.
The job releases its CPU allocation when its foreground pipeline exits. A failed
stage is retained in its log and prevents a misleading completion report.

For parallel downloads, `scripts/download_omniobject.py --create_shards=8`
creates an immutable plan at the supplied `--plan` path, balancing remaining bytes
across eight disjoint archive lists. `ucloud/download-scanned-shard.sh` runs one
provider-native CPU job per list with four archive workers each. Completed
archives and partial aria2 transfers are reused. Archive locks guard against
accidental duplicate writers. Every shard has separate progress, retry and exit
records. Shard zero waits for all eight verified/extracted completion markers,
then invokes the same preparation and loader-check pipeline above. Other jobs
release their allocation when their downloads finish.

Progress counts completed pieces from aria2 control files. File length and
allocated-block counts overestimate sparse transfers on the shared filesystem.
Use `--measure_seconds=30` to save a measured transfer rate and ETA in
`data/raw/omniobject3d/measured-progress.json`; the value includes a UTC sample
time and does not imply that normalization or loader checks are complete.

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

The complete official GSO set has 1,030 prepared objects and 32,960 rendered
frames. Full OmniObject3D preparation is still in progress. Consult the batch
logs and `data/completion.json` for full-dataset completion, independently of the
subset validation below.

The extracted release contains approximately 5,911 scan meshes but 4,810 candidate
matched single-object processed captures. Four wallets legitimately name their
mesh `wallet.obj`. The supplied `duo.tar.gz` contains multi-object scenes and has
no single matching mesh per capture. `book_023` and `watermelon_015` lack matching
source meshes. These raw captures are preserved and reported respectively in
`excluded_multiobject_captures` and `missing_scan_objects`; neither list means a
retained manifest object lacks ground truth. Completion requires
`all_retained_objects_have_mesh=true`, the scope
`available_matched_single_object_processed_captures`, and matching audited
manifest counts. The advertised 6,000 objects must not be presented as 6,000
available real training videos.

Full preparation uses eight CPU batch jobs with three processes per job. The
shared command supports `scanned prepare-omni --shard N --shards 8 --workers 3`.
Each shard only populates locked geometry/frame caches and a
`preparation-shard-NN.json` progress report. A final unsharded invocation reuses
those caches and publishes the complete manifest. The final training-ready count
comes from the completed audit, not the candidate count above.

On 2026-10-29, B200 batch `12439140` completed a real pretrained smoke run with
100 distinct Omni captures, 16 sampled frames per video, and 10 fixed GSO objects
drawn from the separate 250-object evaluation panel. One epoch produced four
optimizer updates with effective batch 32 and physical batch 8. The 17,869,828
fusion-adapter parameters trained; SAM 3D and V-JEPA stayed frozen, with no LoRA.
Training, full-state checkpoint resume, native baseline caching, paired flow
losses and both geometry evaluations passed. All 20 predictions were non-empty.
The dedicated B200 batch exited with code 0 and released its allocation.

The complete capture audit found 48 empty-mask frames across seven of the 100
videos. Preparation now records and excludes unusable RGB/mask pairs while
preserving original files. Cached `capture-validation.json` reports invalidate
when source metadata changes. The audited subset retains 19,924 usable frames;
all of its RGB/mask pairs were decoded before training.

Mean training flow loss was 0.810445. The 10-object evaluation flow loss was
0.734815 for frozen SAM 3D and 0.734830 for the trained EMA fusion adapter.
Canonical coarse voxel IoU was 0.286489 and 0.286430 respectively. Four updates
provide an execution check, not evidence of improved model quality. These
predictions use four-step Stage-1 sampling without mesh refinement. See the
[validated summary](validation/scanned-smoke-2026-10-29.json).

Full artifacts, configurations, source hashes, exclusions, checkpoints and
per-object baseline caches remain on UCloud at
`/work/MathiasElmegaardNielsen#6181/omni-gso-data/smoke-100-10-2026-10-29-23-07-55`.
The runtime source was commit `074542f`; 58 source files matched and 34 CPU tests
passed with one GPU-only test skipped. Measured training peak allocation was
17.35 GiB. The first profiling batch includes startup overhead, so its throughput
is not directly comparable with the subsequent batch. Larger physical batches
and full-run throughput still need tuning within effective batch 32.

OmniObject3D's scan-to-COLMAP registration and SAM canonical-axis calibration
remain unverified. Do not report pose-dependent or visibility-region metrics as
validated. GSO render cameras are exact. Real pretrained GPU forward/backward
and SAM VAE round-trip checks passed on the smoke subset; the Omni round-trip
mean IoU was 0.999999. Full-corpus pretrained validation remains pending.
