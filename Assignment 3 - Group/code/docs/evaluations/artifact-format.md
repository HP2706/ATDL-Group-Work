# Evaluation artifacts

Each evaluation has its own ID, separate from training and other protocols.
Scheduled, recovery and ad hoc jobs call the same evaluator; `trigger` records
why it ran. Each ID contains the model, epoch, dataset, protocol and timestamp.

```text
results/evaluations/<evaluation_id>/
  evaluation.json                 # resolved configuration and protocol
  <label>.meta.json               # checkpoint path/SHA256, data/source identity,
                                  # epoch, EMA weights and trigger
  <label>.jsonl                   # one row per clip/object/reference
  <label>.summary.json            # metrics, valid/expected counts, completeness
  predictions/<label>/<sample>.pt # latent/layout, coarse prediction, GT, metrics
  stages/<sample>/stage1.pt       # pre-refinement latent/layout/RNG checkpoint
  media/<label>/<sample>/         # input MP4, reference/mask PNG, coarse GLBs
  samples/<sample>/
    mesh-exact.npz                # exact decoder vertices/faces/attributes
    mesh-exact.ply
    preview.glb                  # simplified preview, never the scoring mesh
    coarse-pose-pointmaps.pt      # native coordinates/pose/Stage-2 features
    ground-truth.glb
    cameras.json
    protocol.json
    surface-metrics.json
    surface-errors.npz           # sampled errors, nearest triangles, alignment
    *-missing-error.ply
    *-spurious-error.ply
```

Heavy evaluation uses 25 flow steps for Stage 1 and 25 for native Stage 2.
The heavy reference image/mask stays at the original dataset resolution, as in
the older baseline runner. Native pointmap preprocessing runs in its own
precision context; only sampling uses autocast, not mesh extraction.
Coarse metrics score native pruned/downsampled 64-cubed coordinates with 20,000
points and 100 ICP iterations. Mesh metrics reuse the baseline surface analyzer:
8,192 samples per direction, exact closest-triangle distances, 100-iteration
similarity ICP and the shared visibility cache. Coarse and mesh distances use
different normalization and remain separate metric namespaces.

Empty Stage-1 predictions remain explicit failures without an invented mesh.
Partial runs are marked incomplete. Resume requires identical configuration,
checkpoint SHA256 and data identity. Cached baselines join by exact sample ID
and protocol, never by aggregate means alone. Input-view budgets are recorded.

## Scheduling and ad hoc evaluation

```text
val.heavy_epochs=[1,-1]            # first and final epoch; [] disables
val.heavy_data_root=/path/to/gso200
val.final_full_validation=True    # full synthetic held-out split at the end
```

For later evaluation, save a resolved config with a new `eval.evaluation_id`
and run `python -m video3d eval-json eval.json /path/to/run` with
`--checkpoint=/path/to/run/epoch_01.pt --epoch=1 --trigger=ad_hoc`.
The same files and schema are produced. Current immutable training sources are
covered by separate native checkpoint-evaluation jobs, with payloads, receipts,
logs, exit statuses and completion audits preserved.

## Final training-distribution validation

Score all 1,306 held-out reference cases across 642 synthetic clips, including
middle-frame and most-occluded cases, deduplicated when they coincide. Only
held-out target assets are scored. All six trained models and frozen SAM3D,
MV-SAM3D and Stream3D use the identical sample-ID manifest. Multi-view methods
use the same 16-frame window; native SAM consumes the specified reference.

This export provides voxel ground truth, not original target meshes. Its common
25-step protocol therefore reports coarse geometry and visibility metrics.
GSO separately supplies exact refined-mesh comparison. A mesh extracted from
GT voxels must not be presented as the original source mesh.
