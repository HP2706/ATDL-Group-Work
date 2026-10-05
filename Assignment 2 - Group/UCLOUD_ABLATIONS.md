# Running the ablations on UCloud (web UI only)

This runs all [ablations](ABLATIONS.md) inside a UCloud GPU job started from the web UI, on **your own, possibly empty, drive**. It needs no UCloud CLI, SSH key, launch config, W&B key, or access to the team's earlier jobs. [`code/scripts/ablations_direct.sh`](code/scripts/ablations_direct.sh) uses the launcher's `direct` backend on the GPU machine and switches W&B off unless `WANDB_API_KEY` is set. All metrics are still saved to the drive.

**Empty drive vs. team drive.** The team's baseline runs live only on the team drive. On a drive without them, the script adapts automatically:

| | Team drive (baseline runs present) | Your empty drive |
| --- | --- | --- |
| X1 | Continues the 400-epoch / 50k-step baseline runs | Trains the same runs from scratch to 1,200 epochs / 200k steps; same seed, so the first 400 epochs / 50k steps should match the baseline closely |
| X2 | Seeds 1–2; seed 0 predictions come from the baseline | Seeds 0–2 (seed 0 re-run at 400 epochs for the ensemble) |
| Python, CIFAR-10 | Reuses the drive's `.venv` and `data/` | Created once by the `setup` step (needs internet) |

Expect about 10 B200-hours on an empty drive (≈ 8 with the baseline runs).

## 1. Upload the code (on your Mac)

Upload only `ablations-ucloud-code.zip` from the repository root, not the whole repository. It holds `Assignment 2 - Group/code/`, `pyproject.toml`, `uv.lock`, and `.python-version`. Rebuild it after any code change, from the folder that contains `ATDL-Group-Work`:

```bash
zip -qr ATDL-Group-Work/ablations-ucloud-code.zip ATDL-Group-Work/pyproject.toml ATDL-Group-Work/uv.lock ATDL-Group-Work/.python-version "ATDL-Group-Work/Assignment 2 - Group/code" -x '*/__pycache__/*' '*.DS_Store'
```

In UCloud, select your project, open **Files**, create a folder (for example `ablations`) on your drive, and **Upload** two files into it: the zip and [`code/scripts/ucloud_batch.sh`](code/scripts/ucloud_batch.sh). Below, `<drive>` is that folder's name; it is mounted as `/work/<drive>`.

## 2. Submit a batch job

Open an app that runs on GPU machines and offers **Batch processing**, for example **PyTorch** or **Terminal**. On the submission page:

- **Machine type:** one full B200 GPU (`gpu-nvidia-b200`, 1 GPU), not a MIG slice.
- **Hours:** 14 (extendable while running).
- **Folders:** add `<drive>`.
- **Batch processing:** select `<drive>/ucloud_batch.sh`. Leave **Initialization** empty; it is only a short startup hook.

Submit. If all GPUs are busy, the job waits in the queue and starts on its own. `ucloud_batch.sh` unpacks the zip (again whenever you upload a newer one), then runs `setup` (Python 3.12 and the locked packages via [uv](https://docs.astral.sh/uv/), GPU check, CIFAR-10 download), X1, X2, X3, X4, and finally packs the results. The job ends, and billing stops, when the script finishes. The Python version inside the app does not matter.

**Interactive alternative.** Leave Batch processing empty, open the job's terminal, and run `nohup bash /work/<drive>/ucloud_batch.sh > /dev/null 2>&1 &`. Stop the job yourself afterwards: an interactive job is billed until it is stopped or its time runs out.

## 3. Monitor

Open `<drive>/ablations.log` in **Files** (the job's `stdout.txt` under **Files → Jobs** shows the same). Lines starting with `===` mark progress. If you have a terminal on the job, `grep '===' /work/<drive>/ablations.log` lists them, and `tail -f` follows live progress. `setup` takes a few minutes and ends with `=== … setup done`; if it fails, the log shows why (most often no internet or no GPU). Each ablation then logs `start`, `done`, or `FAILED`; a failed ablation does not stop the others.

## 4. Download the results

When the log ends with `All requested targets finished.`, download `<drive>/ablation-results.zip` from **Files** and unzip it inside `Assignment 2 - Group/`. It contains every ablation Parquet file under `our-results-folder/ablations/` and the X2 `test_predictions.pt` files for seeds 0–2. A batch job has already stopped; stop an interactive one yourself.

## Resume, subsets, and two jobs

- Each ablation writes to a fixed run root under `/work/<drive>/runs/sweeps/ablations/`. If a job ends early (time limit, failure), submit the same batch job again: completed runs are skipped and interrupted ones resume from their last checkpoint (every 10 epochs or 1,250 steps).
- In an interactive job you can run only some targets, for example `bash "/work/<drive>/ablation-code/ATDL-Group-Work/Assignment 2 - Group/code/scripts/ablations_direct.sh" x3 x4 package`. `package` rebuilds the results zip from everything finished so far.
- **Two interactive jobs (≈ 6 hours instead of ≈ 11):** in job A run `ablations_direct.sh setup x1`. Once its log shows `setup done`, start job B on the same drive with `ablations_direct.sh x2 x3 x4`. When both have finished, run `ablations_direct.sh package` in either job. Never run the same ablation in two jobs at once.
