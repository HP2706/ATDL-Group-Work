# SAM 3D multi-view baselines

Both methods are installed Python libraries, pinned in pyproject.toml and uv.lock.
There are no nested source repositories, upstream subprocess CLIs, or sys.path patches.
Both forks provide sam3d_objects; the mutually exclusive extras must use separate
environments. Never install the native SAM wheel into either baseline environment.

On the remote CUDA host:

```bash
UV_PROJECT_ENVIRONMENT=.venv-mvsam3d uv sync --locked --extra mvsam3d --group gpu_runtime
UV_PROJECT_ENVIRONMENT=.venv-stream3d uv sync --locked --extra stream3d --group gpu_runtime
```

The upstream native extensions (PyTorch3D, Kaolin and MoGe) are still required.
Use the existing installer with sam=False so it preserves the selected fork:

```bash
.venv-mvsam3d/bin/python scripts/install_models.py --sam=False --metrics=True --runtime=True
.venv-stream3d/bin/python scripts/install_models.py --sam=False --metrics=True --runtime=True
```

Use the CUDA environment and native-extension wheels on the remote GPU workstation.

## Python interface

```python
from pathlib import Path
from video3d.baselines import BaselineConfig, BaselineInput, reconstruct

sample = BaselineInput.model_validate_json(Path("sample.json").read_text())
result = reconstruct(sample, BaselineConfig(method="stream3d",
    checkpoint="/models/hf/pipeline.yaml"), Path("results/stream3d"))
```

The manifest contains sample_id, equally sized frames/masks lists, and optional
pointmaps (one .npy file per view, float camera-local XYZ with shape 3,H,W in
PyTorch3D camera axes). Paths are resolved by the calling process. Binary masks
are supplied; neither method predicts target masks. Without pointmaps, both use
the native monocular MoGe pointmap estimator. This differs from the papers'
DA3 geometry preprocessing and must be reported explicitly.

Use our normal saved experiment configuration for matched dataset evaluation:

```bash
.venv-mvsam3d/bin/python -m video3d.baselines evaluate --config=config.json --method=mvsam3d --output=results/mvsam3d --limit=10
.venv-stream3d/bin/python -m video3d.baselines evaluate --config=config.json --method=stream3d --output=results/stream3d --limit=10
```

This reuses build_dataset, frame selection, ground_truth, per-sample generation
seeds, shape_metrics, visibility_metrics and EvaluationRecord. Occupancy arrays,
input manifests, selected views, sampling settings and upstream revisions are
saved with every result. The output protocol is checked before a directory is reused.

## Scope and upstream reuse

The default is coarse_stage1, matching our adapter's 64³ geometry target.
Set `full_mesh=True` in BaselineConfig or `--full_mesh=True` on the run/evaluate
CLI to keep Stage 2 and export both coarse occupancy and refined geometry.
The `sam3d` method uses the fork's native single-view `run` on reference position 0;
it does not invoke multi-view fusion. It uses the same checkpoint as MV-SAM3D.

Full refinement saves exact unsimplified decoder vertices/faces in mesh-exact.npz
and mesh-exact.ply, plus a vertex-colored preview.glb. Upstream GLB conversion
can simplify geometry; use the NPZ/PLY for numerical analysis. These are decoder
coordinates, not a claim of recovered physical scale. Layout optimization and
texture baking are disabled. Coarse shape/pose, Stage-2 sparse features and
coordinates, available pointmaps, fusion diagnostics and Stream3D token memory
are retained as PyTorch artifacts. Treat PT files as trusted local artifacts.

Peak CUDA allocated/reserved bytes, process peak RSS, and reconstruction wall
time are recorded. Injected/reused pipelines have zero load time in result.json;
the pilot protocol records actual model loading separately. First-use kernels
make cold and warm reconstruction timings incomparable without warmup.

MV-SAM3D uses its upstream run_multi_view with Stage-1 attention-entropy
weighting (alpha=30) and its original sampler. Stream3D imports its upstream
attention probe, mass-relative evidence score, token-vote memory update and
view selection, then generates using its upstream weighted sampler. It uses
disjoint chronological chunks (including a short final chunk), fixed probe noise,
and the token-vote selection variant. Only the final reconstruction is generated; intermediate selections and evidence
memory are recorded. Full refinement adds upstream entropy weighting (MV-SAM3D)
or mass-relative weighting (Stream3D) at Stage 2. Stream3D retains the Stage-1
selected bundle for refinement: it omits a separate Stage-2 selector and newer
distance-diversity re-ranking. Label it the token-vote variant, not a full paper
reproduction. Neither integration currently uses DA3 or visibility-mixed fusion.

Sources:

- Li et al., *MV-SAM3D*, arXiv:2603.11633:
  https://github.com/devinli123/MV-SAM3D (561 stars, checked 29 October 2026),
  commit abb04b5e8af5bc33b0265bdf19937e76bbb6bcdd.
- Zhou et al., *Stream3D*, arXiv:2605.21472:
  https://github.com/kaichen-z/STREAM3D (228 stars, checked 29 October 2026),
  commit be551cc756f5710875ec1c6a4c0fa545b56d2c4f.

No upstream implementation was copied into this repository. Upstream license
terms continue to apply to the installed distributions and model checkpoints.

## UCloud verification

```bash
.venv-mvsam3d/bin/python -m video3d.baselines evaluate --config=config.json --method=mvsam3d --output=results/mvsam3d-smoke --limit=1 --k=4
.venv-stream3d/bin/python -m video3d.baselines evaluate --config=config.json --method=stream3d --output=results/stream3d-smoke --limit=1 --k=4 --chunk_size=2 --topk=2
```

Use the same saved experiment configuration and actual SAM 3D checkpoints for
both methods. The second command exercises two chronological chunks. Inspect
nonempty 64-cubed geometry, per-sample metrics and recorded frame selections.
Sampler step count and flow/shortcut mode come from the saved configuration.
The requested interactive B200 workstation is retained for ten hours; ending
one verification process does not release that workstation.

See [completed real-checkpoint verification](baseline-verification-2026-10-29.md)
for the measured smoke-test outputs and their limitations.

## Surface error analysis

For a run directory containing ground-truth.glb and protocol.json:

```bash
python -m video3d.baselines analyze --directory=results/object
```

The analysis shares the existing ICP implementation. It saves the fitted 4×4
transform, canonical and similarity-aligned bidirectional closest-triangle
errors at 8,192 area-uniform surface samples per mesh, nearest triangle IDs,
normal consistency, precision/recall/F-score at 1%, 2% and 5% of the normalized
GT longest side, and colored PLY error point clouds. Exact refers to geometry
export and nearest-triangle queries; surface coverage is sampled. Single-start
ICP can fail on rotations and symmetries. Scans can also be incomplete.

Registered GSO cameras permit first-hit ray visibility partitions defined over
the common evaluation clip. These partitions stay the same across methods,
including the single-view baseline. OmniObject3D mesh-camera registration is
unverified, so its visibility-specific metrics are explicitly unavailable.

## GSO random-250 evaluation, 29 October 2026

The frozen selection uses Python `random.Random(42).sample` over sorted downloaded
names in the official 1,030-object GSO collection. There are 250 unique objects,
without replacement. OmniObject3D is the training source, not this evaluation set.
The dataset preparation task owns GSO rendering; these evaluations reuse its
32-frame, 384px per-object records without rendering a second copy.

Each method receives the same selected clip, positions `[0, 10, 21, 31]`; native
SAM 3D uses only reference position 0. MV-SAM3D uses all four views. Stream3D
processes two chunks, retaining up to four views in chronological order; this
short protocol does not test compression below the observed view count.
Both stages use 25 flow steps and sample-keyed, matched generation seeds.

Each separate provider-native GPU batch job saves the existing voxel metrics
(canonical/aligned IoU, surface-voxel Chamfer, F-score, empty-prediction flag),
canonical visibility-region completeness/recall, refined-mesh metrics and error
maps, exact geometry, coarse outputs, poses/latents, input frames/masks,
fusion/selection diagnostics, inference/metric timing, and peak memory.
Training flow loss is explicitly null with a reason: this runner evaluates
reconstructed geometry and does not define a matched multi-view training-loss
objective. It must not be confused with the epoch-1/10 adapter flow-loss tables.

GT visibility partitions are cached once per selected object using 8,192
area-uniform surface samples, fixed seed 43, registered render cameras and
first-hit ray occlusion. Cache metadata includes mesh/camera hashes, frame IDs,
sample count and tolerance. All models use identical points and partitions.
These renders measure self-occlusion and additional-view coverage; there are no
external occluders. Dense surface queries use Open3D's native BVH to bound memory.

The completed object record is written atomically only after all enabled metrics
and artifacts are saved. Restarts skip completed objects under an unchanged
protocol. Foreground exit releases each job's dedicated GPU, including failures;
logs and exit diagnostics remain on Member Files. The ten-hour allocation is a
fallback limit. Source hashes and the complete configuration are stored with the
frozen selection in the run root.
