# Checkpoint (updated 2026-10-27)

Resume by reading this file, then `CLAUDE.md`. Decisions and their reasons are in `DATASET.md`; the M2 handover is `docs/m2_handover.md`.

## Done

- **M0, M1:** 500 eval clips (GSO, seeds 0-499, Blender 4.2.1) in `dataset/clips/eval`, 0 failed seeds, T7 signed off. Stats are in `DATASET.md` Section 8. The full test suite over the 500 clips was running at the checkpoint: check `logs/m1_tests.log`, or rerun `uv run pytest -q -m "not slow" tests/test_clips.py`.
- **Blender 4.5.15 LTS** support (`blender_custom_path`, needed for B200): identical masks and depth to 4.2.1, RGB within 0.03/255; `meta.json` records the version.
- **M2 cluster scripts** (`scripts/m2/`), `configs/m2.yaml` (train seeds from 1000000), `docs/m2_runbook.md`, `docs/m2_handover.md`, project `CLAUDE.md`, `DATASET.md`.
- **T6 inference:** SAM 3D ran on an A100 spot VM (`sam3d-t6`, europe-west4-a, project `sam3dvid`) on frame 0 of 20 eval clips: 50 objects x 2 variants (`moge`, `gt_pointmap`), 100/100 ok, about 14 s each. Outputs are in `t6/outputs/` (1.2 GB, git-ignored); the inputs bundle is `t6_inputs.tar.gz`, not in git because of SA-1B. The VM is **stopped** (disk only); delete it after T6 sign-off with `gcloud compute instances delete sam3d-t6 --zone=europe-west4-a`.
- **T6 conventions, determined empirically** from the outputs; `src/sam3dvid/sam3d.py` and `tests/test_sam3d_converter.py`:
  - SAM 3D layout: `x_cam = (s * x_local) @ R(q) + t`, wxyz quaternion, row vectors, PyTorch3D camera (x left, y up) = `diag(-1,-1,1)` of OpenCV.
  - Object frame = ours (Z-up unit cube), no extra rotation or scale. Evidence: our +Z maps to SAM's +Z for 27/50 objects (the rest are flips and side axes of ambiguous objects); the yaw clusters at 0.
  - With the GT pointmap, translation matches GT to about 1% and the scale ratio is about 1.0-1.1.

## In progress at the checkpoint

- **T6 analysis:** `uv run python -m sam3dvid.t6 analyze` writes `t6/report.json` (about 20 min, CPU). It was 32/100 when stopped; rerun it if `t6/report.json` doesn't exist.
  - Partial results on 18 objects: `gt_pointmap` ADD-S 0.058, ADD-S@0.1 0.89, F1@0.01 0.152. Paper Table 3 (ADT, which uses a GT pointmap) has ADD-S 0.077 and ADD-S@0.1 0.77; Table 2 shape F1 is 0.234 on SA-3DAO.
  - F1 at 100k points is likely biased low against the paper's 1M (point spacing is near the 0.01 threshold; inferred). Check this before reporting the shape numbers.
  - `moge` needs depth-scale alignment; the report includes `layout_scale_aligned`.
- **Two subagents were working at the checkpoint, uncommitted:**
  - **Objaverse train asset pipeline:** `scripts/download_objaverse.py`, `src/sam3dvid/assetfilter.py`, `blender/render_thumbs.py`, changes to `assets.py`, `normalize_assets.py`, `blenderrun.py`, `config.py`, `generate.py` (`asset_pool` uses the filter), `tests/test_assets.py`.
  - **Training export:** `src/sam3dvid/export.py`, `export_reader.py`, `tests/test_export.py`, `docs/data_format.md` section.
  - Their sessions end with this one. On resume: review the diffs, run the tests, finish anything incomplete, commit.

## Next

1. Finish the T6 report. Write the results and the converter into `DATASET.md` Section 7 and `docs/data_format.md`. Get T6 sign-off, then delete the VM.
2. **Training-target implication (tell the model team):** training uses no pointmap input, so SAM 3D's pointmap comes from MoGe at metric scale, while our units are arbitrary (MoGe scale ratio 0.2-0.45). Translation and scale targets need a scale-invariant formulation. SAM 3D's own (`pose_target.py`, `translation_scale`) could not be read: `sam3d_objects/data/**` is still blocked by a `**/data/**` deny rule after the project settings change and a restart, so the rule may come from managed settings. Check `/permissions`.
3. Finish the M2 handover: fill the Objaverse and export commands into `docs/m2_runbook.md`, then run the whole chain locally on a few train clips (assets -> smoke test -> render -> tests -> export).
4. M2 itself runs on UCloud (colleague); the full render is gated on T6 sign-off.
