# Cluster checklist for rendering the training data

Run these on a **compute node** (not the login node), e.g. in an interactive job.

| # | Check | Command | Need | Why |
|---|---|---|---|---|
| 1 | glibc version | `ldd --version` | ≥ 2.28 | Blender 4.2 won't start on older glibc (we'd need a container) |
| 2 | Internet access | `curl -sI https://download.blender.org \| head -1` | `200` | Downloads Blender, Objaverse, SA-1B (else: download on the login node) |
| 3 | GPU vendor and model | `nvidia-smi --query-gpu=name,memory.total --format=csv` | NVIDIA, ≥ 6 GB for rendering; at least one type with ≥ 32 GB | Our denoiser (OptiX) is NVIDIA-only. SAM 3D inference (test T6, and fine-tuning later) needs ≥ 32 GB VRAM, e.g. A100, H100, A6000/L40 or V100 32 GB |
| 3b | GPU types per partition | `sinfo -o "%P %G"` (Slurm) | which partitions have the ≥ 32 GB GPUs | Rendering can use small GPUs; SAM 3D needs a big one |
| 4 | Scheduler | `which sbatch squeue` | which one | How rendering jobs get launched |
| 5 | Storage quota | `quota -s` or ask the admins | ~250 GB scratch | Full render output before packing (~190 GB) |
| 6 | File count limit | ask the admins (inode quota) | ≥ 3M files | The full format is ~250 files per clip |
| 7 | Containers | `which apptainer singularity` | either, if #1 fails | Fallback way to run Blender |
| 8 | Max job length and GPU limit | ask the admins / `sinfo` | ≥ 1 h jobs | Decides how many clips per job |

Send the output back as-is.
