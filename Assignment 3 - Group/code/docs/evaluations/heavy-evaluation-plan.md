# Quick and heavy checkpoint evaluation

- Keep the audited GSO 200 manifest and shared image/mesh/visibility files.
- Quick: existing 4-step shortcut and coarse scores. Heavy: 25-step flow,
  native Stage-2 mesh generation, coarse metrics with 20,000 points/100 ICP
  iterations, and the existing baseline mesh-surface analyzer (8,192 points).
- Evaluate immutable EMA checkpoints for epoch 1 and final epoch 5. Future
  training selects heavy epochs through configuration; `-1` means final epoch.
- Preserve exact mesh arrays, a viewable simplified preview, coarse tensors,
  inputs, targets, sampled surface errors, transforms, per-object JSONL and
  summaries. Scheduled/ad hoc/recovery evaluations use one evaluator and schema.
- Run a real B200 check before submitting six separate epoch-1 jobs and a CPU
  checkpoint watcher for six final-epoch jobs. Reuse cached matching baselines;
  report differing method input budgets explicitly. Dedicated jobs auto-release.

## Files and expected change size

| File | Existing lines | Change |
|---|---:|---|
| `src/video3d/config.py` | 346 | Heavy mode, refinement steps and selected epochs/data root |
| `src/video3d/models/sam3d.py` | 208 | Retain Stage 2 only for heavy eval; refine sampled coarse latent |
| `src/video3d/evaluate.py` | 218 | Explicit checkpoint identity; refinement and structured evaluation output |
| `src/video3d/train.py` | 510 | Invoke shared heavy evaluator at configured checkpoint epochs |
| `src/video3d/cli.py` | 140 | Pass explicit checkpoint and trigger to the shared evaluator |
| `src/video3d/mesh_analysis.py` | new | Reuse existing baseline surface analysis, unchanged scoring |
| `src/video3d/media.py` | 46 | Reuse exact native-mesh export and preserve preview separately |
| `tests/test_evaluate.py` / focused tests | 93 / new | Check protocol isolation, scheduling, persistence and native refinement |
| Evaluation format documentation | new | Layout, identity fields, completion and ad hoc examples |

Estimate: 300–400 added lines, 30–60 changed lines, plus approximately 185 lines
reused from the existing baseline mesh analyzer. Actual launch scripts stay
one-off orchestration outside the reusable model code.
