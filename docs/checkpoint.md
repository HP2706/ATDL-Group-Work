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

## T6 results (`t6/report.json`, 50 objects, `uv run python -m sam3dvid.t6 analyze`)

| | ADD-S | ADD-S@0.1 | 3D IoU | ICP-Rot | F1@0.01 | Chamfer |
|---|---|---|---|---|---|---|
| gt_pointmap | 0.064 | 0.88 | 0.548 | 14.5 deg | 0.187 | 0.058 |
| moge, depth-scale aligned | 0.474 | 0.08 | 0.083 | 14.6 deg | 0.170 | 0.065 |
| Paper (layout: ADT Table 3; shape: SA-3DAO Table 2) | 0.077 | 0.767 | 0.497 | 15.25 deg | 0.234 | 0.040 |

- gt_pointmap layout matches or beats the paper on every metric: the converter is right. Translation error 2%, scale ratio 1.11 (medians).
- Shape F1 is below the paper; 100k points against their 1M may explain part of it (inferred, unchecked).
- MoGe scale ratio to our units is 0.29 (median); the layout stays poor even after global depth-scale alignment. Unaligned moge ICP-Rot (0.8 deg) is meaningless: ICP fails when the clouds are about 2.5 units apart.
- Canonical voxel check is weak: identity ranks first but narrowly (IoU 0.266 vs 0.260) because of SAM 3D's yaw freedom; the +Z evidence above carries the frame conclusion.

## Done since

- Both agents' work reviewed and committed (`5db9f92`, by the user). Objaverse scene filter on TRELLIS captions added (`cc72cba`): 67 of 100 test assets accepted (27 scenes, 6 geometric). All Objaverse licenses admitted (research use, user's decision).
- T1 relaxed (`f57e5a1`): coverage and depth error warn, T1b checks both exactly; the M1 suite now passes.
- M2 runbook has the Objaverse and export commands, plus two ordering rules: freeze the asset pool before rendering, export only after rendering.
- T6 results and the converter are written into `DATASET.md` Section 7 and `docs/data_format.md`.

- Local M2 chain checked on 5 train clips (seeds 1000000-1000004, Blender 4.5, `render_shard.sh`): all render, clip tests pass, export `--verify` exact, review sheets look right. Fixed `render_shard.sh` exiting 1 after a fully successful shard (`782a226`); added `export --verify N` (`92d950f`).
- T6 signed off by the user; the VM is kept stopped for now (pd-standard disk, about $0.30 a day).

## Open

- Full suite on the committed tree: `logs/full_suite.log`.
- Push to origin so the colleague can clone (ask the user first).
- Tell the model team about scale-invariant targets (`DATASET.md` Section 7).
- UCloud notes in `docs/m2_handover.md` and `docs/cluster_checklist.md` are for the colleague to fill in.
- Delete the VM when the user says so: `gcloud compute instances delete sam3d-t6 --zone=europe-west4-a`.
