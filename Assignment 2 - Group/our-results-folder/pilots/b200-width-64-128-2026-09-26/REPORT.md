# Concurrent width-64 and width-128 end-to-end check

Two augmented ResNet18 models trained concurrently on the existing UCloud B200 job `12403590`. Both used the full CIFAR-10 training split, clean test split, 0% label noise, Adam at constant LR `1e-4`, batch size 128, seed 0, and **one epoch only** (391 updates). These are pipeline checks, not converged reproductions.

| Width | Training epoch time | Train error | Test error | W&B run |
| ---: | ---: | ---: | ---: | --- |
| 64 | 7.67 s | 0.49684 | 0.5090 | [width 64](https://wandb.ai/hprjdk/atdl-double-descent/runs/cifar10-resnet-k64-seed0-2026-09-26-21-42-46) |
| 128 | 9.20 s | 0.45832 | 0.4654 | [width 128](https://wandb.ai/hprjdk/atdl-double-descent/runs/cifar10-resnet-k128-seed0-2026-09-26-21-42-46) |

The remote dataset lives on the persistent `/work` drive at `/work/atdl-b200-benchmark-2026-09-26/data`. Both run directories contain a completed marker, checkpoint, predictions, config, metadata, and evaluation history. The Parquet results remain on that drive under `results/width-64-128-cifar10-2026-09-26-19-42/data/vision/` and are copied into this folder's `data/vision/` directory.

Each Parquet file has one measurement row. Its first 11 Arrow fields match the converted CIFAR-10 ResNet source file exactly in name, order, and type. Errors are fractions. Each W&B run has a history point at global step 391 and a `results` artifact containing the Parquet file, `config.json`, `run_metadata.json`, and `metrics.jsonl`. The W&B API confirmed both runs are finished and both artifacts are present.

The full 30-run, 400-epoch CIFAR-10 sweep was **not launched** by this check. `code/scripts/vision.sh --plan` lists all 30 intended conditions and the remote B200 copy of the script passes Bash syntax validation.
