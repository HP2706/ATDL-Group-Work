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

## Open at the checkpoint

- **M1 full suite** (`logs/m1_tests.log`): 2994 passed, 16 failed, all `test_t1_projection_consistency` on object 1 of clips 11, 94, 112, 149, 185, 246, 253, 273, 311, 355, 392, 402, 411, 437, 475, 483. T1b (exact ray-cast) passes on all 500, `validate.py` is unchanged and the failing meshes predate their clips, so this is a weakness of the point-sampled T1, not bad data (inferred, the clips weren't viewed). Two modes: amodal coverage 0.84-0.89 vs the 0.90 threshold with depth error about 3e-4 (10k points too sparse), and depth error 0.1-0.2 on hollow objects (bowls, pots, caddy; the point z-buffer likely sees the far wall). Proposed: relax T1 (drop its coverage check in favour of T1b, or warn on concave objects). Needs the user's decision.
- **Training export agent: finished, not reviewed, uncommitted.** `src/sam3dvid/export.py`, `export_reader.py`, `tests/test_export.py` (7 tests pass), `docs/data_format.md` Section 9 (old 9 renumbered to 10). 4.0 MB and about 0.5 s per clip on one worker, so about 40 GB per 10k clips. JPEG q95 with 4:4:4 chroma (`JPEG_SUBSAMPLING = 0`), because 4:2:0 gave 37.8-41.5 dB PSNR against the 40 dB test; 4:2:0 would be about 33 GB per 10k. Not tried: `--workers > 1`, reading with the `webdataset` library.
- **Objaverse agent: stopped during its final full pytest run, not reviewed, uncommitted.** `scripts/download_objaverse.py`, `src/sam3dvid/assetfilter.py`, `blender/render_thumbs.py`, changes to `assets.py`, `normalize_assets.py`, `blenderrun.py`, `config.py`, `generate.py`, `tests/test_assets.py`; also `README.md`. Its own status is unknown: run `uv run pytest -q tests/test_assets.py tests/test_export.py` first (the full `-m "not slow"` suite takes about 45 min because of the 500 clips), then review the diff against the spec in `DATASET.md`/`docs/m2_handover.md`.

## Next

1. Review and commit the two agents' work (by explicit paths).
2. Decide on T1 (above).
3. Write the T6 results and the converter into `DATASET.md` Section 7 and `docs/data_format.md`. Get T6 sign-off, then delete the VM.
4. **Training-target implication (tell the model team):** training uses no pointmap input, so SAM 3D's pointmap comes from MoGe at metric scale, while our units are arbitrary (scale ratio 0.29). Translation and scale targets need a scale-invariant formulation. SAM 3D's own (`pose_target.py`, `translation_scale`) could not be read: `sam3d_objects/data/**` is still blocked by a `**/data/**` deny rule after the project settings change and a restart, so the rule may come from managed settings. Check `/permissions`.
5. Finish the M2 handover: fill the Objaverse and export commands into `docs/m2_runbook.md`, then run the whole chain locally on a few train clips (assets -> smoke test -> render -> tests -> export).
6. M2 itself runs on UCloud (colleague); the full render is gated on T6 sign-off.
