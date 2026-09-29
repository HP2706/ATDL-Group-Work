# Full 400-epoch ResNet18 runs at widths 3 and 64

These are two completed cells of the reduced CIFAR-10 sweep in `EXPERIMENTS.md`. They trained **concurrently** on one UCloud B200 (job `12403590`) with the full 50,000-image CIFAR-10 training split, 20% fixed incorrect-label noise, augmentation, Adam at constant learning rate `1e-4`, batch size 128, and seed 0. Each completed **400 epochs and 156,400 optimizer updates**. Evaluations and checkpoints were scheduled every 10 epochs.

| Width | Train error, epoch 400 | Test error, epoch 400 | Train loss | Test loss | W&B run |
| ---: | ---: | ---: | ---: | ---: | --- |
| 3 | 40.152% | 28.93% | 1.3796 | 0.9504 | [width 3](https://wandb.ai/hprjdk/atdl-double-descent/runs/cifar10-resnet-k3-seed0-2026-09-26-21-52-41) |
| 64 | 0.556% | 20.88% | 0.0181 | 1.6567 | [width 64](https://wandb.ai/hprjdk/atdl-double-descent/runs/cifar10-resnet-k64-seed0-2026-09-26-21-52-41) |

The pair ran from 19:52:36 to 20:14:57 UTC on 26 September 2026: **22 minutes 21 seconds of wall time**, or about **0.373 B200 GPU-hours** for the pair. Width 64's completed marker was written at 20:14:31 UTC and width 3's at 20:14:55 UTC. Per-epoch training-loop time, including scheduled evaluations and checkpoints, summed to 1,301.7 seconds for width 64 and 1,325.3 seconds for width 3. Those per-run sums overlap because the models ran together; do not add them to estimate rented GPU-hours.

For width 3, the median ordinary epoch was 3.084 seconds and the median epoch with both evaluation and checkpointing was 5.373 seconds. For width 64, those medians were 3.097 and 4.758 seconds. The extra time occurs once every 10 epochs; these logs do not separate evaluation from checkpoint writing. Checkpoint files are about 0.38 MiB and 127.96 MiB respectively.

Both Parquet files in `data/vision/` contain 40 rows at epochs 10, 20, …, 400, with matching actual global steps and zero-based measurement indices. Their first 11 Arrow fields match the converted authors' CIFAR-10 ResNet file in name, order, and type. W&B reports both runs `finished` at step 156,400, with 40 history points each, and lists a `results` artifact for each containing the Parquet result, `config.json`, `run_metadata.json`, and `metrics.jsonl`.

The remote run root is `/work/atdl-b200-benchmark-2026-09-26/runs/sweeps/resnet-cifar10-adam-400-2026-09-26-21-52-00`. The remote results are in `/work/atdl-b200-benchmark-2026-09-26/our-results-folder/data/vision/`. The remaining 28 sweep conditions were **not** launched. With the current width list in `code/scripts/cifar10_resnet.sh`, using this exact `RUN_ROOT` will skip the two completed conditions.

## Same-epoch comparison with the authors

The authors' [CIFAR-10 ResNet results](https://storage.googleapis.com/hml-public/dd/cifar10-resnet18k-50k-adam/Mlist) are converted locally at `../../their-results/hf_dataset/data/vision/cifar10-resnet18k-50k-adam.parquet`. Their [plotting notebook](https://gitlab.com/harvard-machine-learning/double-descent/-/blob/master/intro_ocean_plot.ipynb) identifies `Mlist[2]` as 20% label noise and labels measurement position `i` as epoch `i + 1`. We therefore compare our explicit epoch 400 with their `trial_index = 2`, `measurement_index = 399` at the same widths. This is the authors' **plotted epoch convention**; the public Parquet conversion itself has no explicit epoch field.

| Width | Metric | Paper at epoch 400 | Ours at epoch 400 | Ours minus paper |
| ---: | --- | ---: | ---: | ---: |
| 3 | Train error | 42.340% | 40.152% | −2.188 points |
| 3 | Test error | 29.570% | 28.930% | −0.640 points |
| 3 | Train loss | 1.41995 | 1.37959 | −0.04036 |
| 3 | Test loss | 0.96737 | 0.95037 | −0.01700 |
| 64 | Train error | 0.620% | 0.556% | −0.064 points |
| 64 | Test error | 21.890% | 20.880% | −1.010 points |
| 64 | Train loss | 0.01965 | 0.01806 | −0.00159 |
| 64 | Test loss | 1.82865 | 1.65671 | −0.17194 |

The [comparison chart](paper-comparison.png) overlays train and test error at every tenth epoch through 400. The corresponding [comparison CSV](paper-comparison.csv) contains all 80 matched width/epoch rows and the metric differences. Mean absolute test-error difference across those 40 matched epochs is 0.505 percentage points at width 3 and 1.617 points at width 64. The trajectories have similar shapes, but they are independent single runs: the authors' seed, corruption mask, and augmentation random choices are not available, so equality of individual metric values is not expected. The loss columns are shown as published; their exact evaluation reduction is not independently documented.

UCloud job `12403590` was stopped on request after these results were secured; its final status was `SUCCESS`.
