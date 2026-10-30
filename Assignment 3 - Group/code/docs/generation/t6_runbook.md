# T6 runbook: SAM 3D on a Google Cloud GPU

This runs SAM 3D Objects on 20 of our eval clips (50 objects, frame 0) and brings back its raw outputs. Everything else (convention converter, metrics) runs locally afterwards. Expect about 1 h of setup and 1-2 h of inference (estimate; not measured yet).

## Before you start

- **HuggingFace access** to [`facebook/sam-3d-objects`](https://huggingface.co/facebook/sam-3d-objects) on *your own* account (gated, approval by Meta), and a read token from <https://huggingface.co/settings/tokens>. Don't share tokens.
- **GPU quota** in the project: `GPUS_ALL_REGIONS` ≥ 1 and `NVIDIA_A100_GPUS` ≥ 1 in the region you use (IAM & Admin -> Quotas). New projects start at 0.
- SAM 3D needs ≥ 32 GB VRAM: an A100 40 GB (`a2-highgpu-1g`) is enough.

## 1. Create the VM

```bash
gcloud compute instances create sam3d-t6 \
  --zone=us-central1-a \
  --machine-type=a2-highgpu-1g \
  --maintenance-policy=TERMINATE \
  --image-project=deeplearning-platform-release \
  --image-family=common-cu124-ubuntu-2204-py310 \
  --boot-disk-size=200GB \
  --metadata=install-nvidia-driver=True
```

- `a2-highgpu-1g` includes one A100 40 GB; no `--accelerator` flag is needed.
- The image family name is a guess at the current Deep Learning VM naming; list the available ones with
  `gcloud compute images list --project deeplearning-platform-release --filter="family~common-cu12" --format="value(family)" | sort -u`
  and pick any `common-cu12x-ubuntu-2204` family. Any NVIDIA driver that supports CUDA 12.1 works.
- Optional: `--provisioning-model=SPOT` is much cheaper, but the VM can be stopped at any time; the run resumes where it stopped.
- If the zone has no A100 capacity, try another (`us-central1-b/c/f`, `europe-west4-a/b`).

## 2. Set up and run

```bash
# on your machine: t6_inputs.tar.gz comes from Julian (not in git: it contains SA-1B imagery)
gcloud compute scp t6_inputs.tar.gz sam3d-t6:~ --zone=us-central1-a
gcloud compute ssh sam3d-t6 --zone=us-central1-a
git clone <this repo> a3 && cd a3
tar xzf ~/t6_inputs.tar.gz       # -> t6/inputs/
bash scripts/t6/setup_vm.sh      # ~1 h; asks for the HuggingFace token once
bash scripts/t6/run_vm.sh        # smoke test on one object, then all 100 runs
```

`run_vm.sh` first runs a single object and prints what SAM 3D returned (`t6/outputs/t6_smoke.log`). If that fails, send the log back before running the rest. The full run is restartable: finished objects are skipped.

Use `tmux` (or `screen`) for both scripts so a dropped SSH connection doesn't kill them.

## 3. Bring the results back and stop paying

```bash
# on your machine
gcloud compute scp sam3d-t6:~/a3/t6_outputs.tar.gz . --zone=us-central1-a
gcloud compute instances delete sam3d-t6 --zone=us-central1-a
```

Send `t6_outputs.tar.gz` back (a few hundred MB at most; too big for git). It contains one `.npz` per object and variant, `run_log.jsonl` with timings and errors, and the logs.

## What gets run

For each object, twice, with the GT visible mask:

- **`moge`**: SAM 3D estimates the pointmap itself (the paper's setting).
- **`gt_pointmap`**: our rendered depth as the pointmap, in PyTorch3D camera axes.

Outputs are saved as SAM 3D returns them (pose quaternion, translation, scale, sparse-structure voxels, Gaussian centers, mesh), with seed 42 and no layout post-optimization.
