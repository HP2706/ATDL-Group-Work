# Unified video-to-3D pipeline

Status: implemented and verified. Updated: 2026-10-29.

Branch: `unify-video-pipeline`, based on `main` at `7228646`.
Worktree: `ATDL-Group-Work/.worktrees/unify-video-pipeline/`.

## Agreed decisions

- One maintained package, `video3d`, one trainer, one evaluator, and one production loss implementation.
- Use Mathias' SAM wrapper as the starting point. Retain our attentive pooler and LoRA as selectable components alongside his gated fusion adapter.
- Replace Mathias' manually constructed flow loss with a direct copy of the pinned upstream SAM 3D loss function, adapting only where the interface requires it. Inspect the exact function before finalizing that adaptation.
- Immediately beneath the copied function, record the upstream source path, commit, and intentional changes. Preserve any required upstream license notice.
- Add a dedicated parity test against upstream. The reference is test-only, not a second production loss or trainer.
- No `sys.path` manipulation or dependency on a sibling source tree. SAM model imports should use a properly installed dependency; investigate upstream packaging before choosing the installation command.
- Use typed `chz` configuration, a Fire CLI, and resolved JSON. Use Pydantic for persisted metadata and validated data records. Remove project-owned YAML configuration from the active pipeline; upstream model configuration files are a separate dependency concern.
- Move Julian's generator into `code/src/video3d/data/generation/`. Preserve existing generated datasets, caches, checkpoints, results, and notebooks.
- One Python project, one shared `pyproject.toml`, and one `uv.lock` for generation, training, and evaluation. No separately installable generator project. Rendering-specific dependencies may use dependency groups in the shared project.
- Keep Julian's generator code as close to its current implementation as possible. Preserve rendering, sampling, filtering, export behavior, defaults, and data conventions; limit changes to relocation, import/path repairs, shared configuration/CLI integration, and necessary dependency compatibility.
- Implementation approved. Verification uses bounded remote smoke runs; existing datasets and experiment artifacts are preserved.

## Comparison and merge choices

| Area | Our implementation (`code/`) | Mathias (`fusion-adapter-code/`) | Unified implementation |
|---|---|---|---|
| Config and CLI | Typed chz, Fire, resolved JSON | OmegaConf YAML layers and command dispatcher | Extend chz and Fire; named Python presets and CLI overrides |
| Dataset | NAVI manifest, cached reference conditioning and targets | Julian's sharded export; variable windows and object-aware sampling | Shared sample contract; NAVI and synthetic readers |
| Preprocessing | Full scene or per-frame masked crops, fixed frame count | Consistent clip crop, jitter, variable clip lengths | Explicit shared preprocessing policy, preserving historical presets |
| Video encoder | V-JEPA Base, 384 px, optional LoRA | V-JEPA Large, 256 px, frozen; DINOv2 controls | Encoder choice independent of conditioning strategy |
| Conditioning | Foreground attentive pooling; 64 summaries replace RGB tokens | Gated cross-attention updates existing RGB conditioning | Separate pooler and fusion modules behind one interface |
| SAM wrapper | Direct geometry loading; positional conditioning layout; upstream loss call | Inference-pipeline integration; named conditioning sets; velocity interface | Start from Mathias' wrapper, verify named sets and avoid loading unnecessary components where possible |
| Loss | Upstream generator `.loss()` | Explicit flow-matching formula in trainer | One copied upstream loss function; separate numerical parity test |
| Trainer | W&B, finite-gradient checks, signals, resumable checkpoints | EMA, parameter groups, deterministic sampler, gate diagnostics | One trainer combining those capabilities |
| Validation | Seeded upstream loss and reconstruction checks | Fixed noise/time-grid loss and sampled voxel IoU | One validation implementation with explicit objective and sampler |
| Evaluation | Native image/repeated-reference controls; paired video/mesh artifacts | Resumable per-sample results, oracle/shuffle/wrong-video controls | One evaluator with all controls and artifact logging |
| Metrics | Canonical mesh-surface distances | Canonical voxel IoU and normalized, ICP-aligned surface metrics | Explicitly named protocols; historical results retain their definitions |
| Analysis | Result summaries and comparison notebook | Paired deltas, clip bootstrap, Wilcoxon, stratified plots | Reuse Mathias' analysis and our media outputs |
| Helpers | Atomic writes, RNG capture, provenance, bounded sweeps | Scratch/persistent storage, sample seeds, checkpoint rotation | Consolidate each responsibility into one implementation |

## Proposed layout

Relative to `Assignment 3 - Group/`:

```text
code/
  pyproject.toml
  uv.lock
  src/video3d/
    config.py
    cli.py
    data/
      generation/          Julian's existing generator modules, minimally adapted
      schemas.py
      reader.py
      navi.py
      samples.py
      targets.py
    models/
      sam3d.py
      encoders.py
      pooler.py
      adapter.py
    adapters.py           optional shared LoRA layers
    loss.py
    train.py
    evaluate.py
    metrics.py
    analyze.py
    checkpoint.py
    storage.py
    logging.py
    sweep.py
  tests/
    test_loss_parity.py
    generation/            migrated generator tests
  scripts/
    generation/            existing generation scripts, with corrected paths
  docs/
    generation/            existing generator documentation
  notebooks/
```

Generation source lives inside `video3d.data.generation`; actual datasets remain at configurable storage paths. Move the existing generator rather than reimplementing it. Generation, training, and evaluation share the same project and dependency resolution. Preserve the generator's current module boundaries wherever possible.

## Implementation sequence

### 1. Contracts, configuration, and dependency boundary

- Define typed model inputs, supervision, sample identity, conditioning sets, and result records.
- Standardize frame layout/range, visible masks, reference position, temporal offsets, and padding. Normalize pixels exactly once at the encoder boundary.
- Extend chz config for paths, data, encoder, conditioning, LoRA, optimizer/EMA, loss, sampler, validation, evaluation, and provenance.
- Save resolved JSON and reject unknown configuration. Give new runs timestamped names; resume retains the original run identity.
- Check upstream packaging and establish installed imports. Do not add a sibling checkout import path.
- Merge generator and training dependencies into the shared project and lockfile. Resolve the generator's Python 3.11 requirement and overlapping dependency constraints explicitly; use compatible versions rather than creating another Python project or lockfile. Do not change generator behavior to simplify packaging.

### 2. Julian's generator and common data layer

- Move generator modules from `synthetic-data/src/sam3dvid/` into `code/src/video3d/data/generation/` with history preserved; relocate scripts/tests/docs into the same shared project and repair references.
- Adapt its configuration to the shared chz interface and expose generation through Fire. Preserve existing parameter names, defaults, validation, random draws, and serialized dataset semantics; do not rewrite rendering or generation algorithms.
- Keep its current training export: indexed tar shards of RGB/masks/metadata plus packed voxel targets. Do not regenerate existing exports just to change configuration.
- Reuse Mathias' indexed reader and deterministic sampling. Consolidate shared decoding with Julian's export reader rather than maintain duplicate decoders.
- Preserve NAVI through a reader producing the common runtime contract.
- Enforce split isolation and coordinate/voxel conventions. Ground truth must stay outside model inputs; any visibility-based reference-selection oracle must be explicitly documented.
- Include checkpoint revision, target conventions, and preprocessing in cache signatures. Preserve old cache entries rather than deleting them.

### 3. Shared model and copied upstream loss

- Port Mathias' SAM wrapper and named conditioning interface into `video3d`.
- Add pooler and gated-fusion strategies; retain native conditioning as a baseline. Do not introduce a new hybrid by default.
- Keep encoder size, resolution, crop policy, and trainability independent of the strategy.
- Retain optional video/SAM LoRA. Frozen feature extraction may use no-grad; a LoRA-enabled path must retain gradients.
- Inspect the pinned upstream loss and replace Mathias' production loss with the copied function. Keep one active objective implementation.
- Start with shape-only, ordinary flow-matching training. Pose/layout training is outside this merge's initial scope.

### 4. Loss parity check

Reference code is pinned to `f91db411c50efee93d8db7aeb323885650f6f722`. Relevant upstream files:

```text
sam3d_objects/model/backbone/generator/flow_matching/model.py
sam3d_objects/model/backbone/generator/shortcut/model.py
```

- `tests/test_loss_parity.py` imports the real upstream implementation as the reference; do not copy a second reference formula into the test.
- Compare the copied loss with upstream using identical model weights, conditioning, target latents, noise, timesteps, layout inputs, precision, and reductions.
- Match time scaling, shape weights, timestep distribution, conditioning dropout, and shortcut settings explicitly.
- Compare scalar loss and gradients with respect to conditioning, using appropriate numerical tolerances.
- Start with a tiny deterministic backbone for a quick contract check; add a real-checkpoint remote GPU check.
- Ordinary flow matching uses shortcut step-size input `d=0`. Finite-step shortcut inference is a separate sampler setting, not a filesystem shortcut or another trainer.
- Existing tests exercise upstream backward and adapter identity, but do not establish this numerical parity.

### 5. One trainer and checkpoint format

- Consolidate training into `video3d/train.py`; there must be no strategy-specific training loops.
- Combine deterministic stream sampling, gradient accumulation, optimizer groups, optional EMA, W&B logging, provenance, and finite-value checks.
- Handle expected zero gradients in fusion weights at zero-gate initialization correctly.
- Checkpoints include trainable weights, EMA, optimizer, scheduler, RNG, sampler position, config, data identity, and source/checkpoint revisions.
- Save complete optimizer updates atomically; resume must reproduce uninterrupted updates. Preserve previous complete checkpoints on interruption.
- Keep signal handling, persistent logs, and bounded wall time. Evaluation and final artifact writes finish inside the same provider batch lifecycle.

### 6. One evaluator, metrics, and analysis

- Unify resumable per-sample evaluation and paired video/reference/mask/GT/prediction artifacts.
- Include native SAM, repeated reference, pooler, fusion, shuffled-video and wrong-video controls. Label best-frame selection as an oracle.
- Fix the inference sampler, seeds, data, and preprocessing for fair conditioning comparisons.
- Record canonical voxel IoU separately from normalized/ICP-aligned metrics.
- Reconcile Chamfer definitions: ours averages directional distances, Mathias sums them and uses a different normalization/alignment protocol. Do not combine historical values under one label.
- Version result schemas and metric protocols, report failed predictions, and reuse Mathias' paired/statistical analysis.

### 7. Verification and retirement of duplicate paths

- Focused checks: config rejection/serialization, exporter-reader compatibility, split isolation, input leakage, mask alignment, voxel/pose conventions, and cache invalidation.
- Model checks: fusion identity at initialization, pooler behavior, correct gradient paths, and loss parity.
- Trainer checks: uninterrupted versus resumed updates, EMA/optimizer restoration, interruption safety, and output persistence.
- Evaluator checks: deterministic paired seeds, resumability, metric conventions, controls, failure reporting, and media pairing.
- Then run a bounded remote GPU smoke job with real checkpoints and a small sample: conditioning, backward, save/reload, sampling, metrics, and artifact export. Verify automatic compute release.
- Only after parity and integration checks pass, retire the independent `fusion-adapter-code/` package and duplicate trainer/evaluator/helpers. Update references and preserve historical experiment artifacts.

## Measured scope and estimates

Current sizes were measured before editing. Counts below exclude tests, scripts and docs unless named; generator moves should mostly preserve file contents.

| Existing files/modules | Current lines | Work |
|---|---:|---|
| Our `config.py` + `cli.py`; Mathias' equivalents | 147 + 191 | One typed config and CLI |
| Our `data.py` + `prepare.py`; Mathias' reader/samples/targets/validate | 271 + 756 | Common records, reader, sampling, targets and preparation |
| Our `model.py` + `adapters.py`; Mathias' SAM/encoders/adapter | 204 + 490 | Shared geometry wrapper, encoders and conditioning modules |
| Both `train.py` files | 185 + 331 | One trainer; extract one copied loss |
| Both evaluators; Mathias' metrics and analysis | 164 + 161 + 107 + 181 | Common results, controls, metrics and reporting |
| Our storage/sweep/profile; Mathias' utils | 44 + 69 + 69 + 121 | Consolidate helpers and orchestration |
| Generator `config.py` | 102 | Typed config and caller updates |
| Generator source/tests/scripts/docs | Move largely unchanged | Relocate into the shared project and repair paths; preserve generator behavior |
| Shared `pyproject.toml` and `uv.lock` | Existing project files | Merge dependency constraints; one project and lockfile, optional rendering dependency groups |
| New `loss.py`, schema/interface modules, checkpoint/logging modules, parity and integration tests | New | Extract/reuse existing behavior; avoid parallel implementations |

Initial estimate: 1,500–2,500 existing lines modified, 700–1,200 new lines including tests, plus removal of superseded duplicates after verification. Reassess after the contract and dependency inspection.

## Implementation record

- The maintained package is now `code/src/video3d/`. Standalone generator and fusion
  projects have been retired; historical documentation and Julian's T6 report are retained
  under `code/docs/`. Existing pilot results, notebooks, datasets and checkpoints are preserved.
- One `code/pyproject.toml` and `code/uv.lock` cover Python 3.11, Torch 2.7.1 and the shared
  generation/runtime groups. Pinned upstream model libraries install into this same environment.
  V-JEPA/NAVI are built into normal wheels from temporary upstream archives because those
  revisions have no package metadata. No upstream source directory or sibling imports remain.
- The production loss is the copied upstream `FlowMatching.loss` body, with explicit
  noise/time supplied by its small context. Its source comment and upstream license are retained.
- Fusion and foreground pooling share training, validation, EMA, checkpoints, resume and
  evaluation. Optional video/SAM LoRA uses the same optimizer/checkpoint path.
- Synthetic indexed exports and prepared NAVI manifests use one sample contract. Native
  reference RGB/masks remain separate from video preprocessing and ground-truth metadata.
- The object-crop policy retains Mathias' shared clip crop. The pooler preset uses the original
  full-scene letterboxing policy; `masked_crop` remains explicit. Pixel normalization occurs
  once at the encoder boundary. Comparing strategies requires matching those settings explicitly.
- Synthetic cache namespaces include the export identity and target/upstream conventions.
  Existing caches and replaced generator outputs are preserved.
- Evaluation distinguishes canonical voxel IoU/NAVI visibility completeness from normalized
  similarity-ICP metrics; historical numbers keep their original interpretation.
- Verification passed: 139 CPU-suite tests, 13 real-GPU tests, 31 checks on a rendered
  dataset, and seed regeneration. All four fusion/pooler × frozen/LoRA real-model cases
  completed training, checkpoint resume, evaluation and artifact export. Copied and
  upstream losses matched, including conditioning gradients. See
  [the verification record](code/docs/verification/2026-10-29.md) for scope and artifacts.

See `code/README.md` for the active setup and commands. This work is isolated on
`unify-video-pipeline`; it has not been committed or merged into `main`.
