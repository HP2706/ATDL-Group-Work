# Targeted ablations X1–X4

The replication track asks us to support the report with targeted ablations. Our main sweeps ([EXPERIMENTS.md](EXPERIMENTS.md)) are single-seed, use 10× shorter horizons than the paper, and leave some claims only partly reproduced. Each ablation below reuses a completed sweep and changes **one factor**, so its effect can be read directly against that baseline. All use 20% label noise and augmented CIFAR-10.

| ID | Question | Changed factor (all else as baseline) | Baseline | Why it matters | Runs |
| --- | --- | --- | --- | --- | ---: |
| **X1** horizon | Are the missing second descent (epoch-wise) and the missing "more data hurts" region (sample-wise) caused by our short training? | ResNet 400 → 1,200 epochs, k ∈ {3,8,12,16,64}; CNN 50k → 200k steps, k ∈ {16,32,48,64,128} | `cifar10_resnet.sh`, `cifar10_cnn_sgd.sh` | Separates "not reproduced" from "not reached at our horizon" (paper Figs 2, 9, 11a) | 5 + 5 |
| **X2** seeds + ensemble | Is the model-wise peak larger than seed-to-seed variation, and does ensembling help most near the interpolation threshold? | Seeds 1 and 2 (new initialization, data order and noise mask), k ∈ {4,8,12,16,24,64} | `cifar10_resnet.sh` (seed 0) | Every other result is single-seed; tests the paper's variance explanation (Fig 28) | 2 × 6 |
| **X3** weight decay | Does ℓ2 regularization shrink the peak and the sample-wise shift? | Weight decay 0 → 5·10⁻⁴ (paper Fig 21 value), n ∈ {12.5k, 50k}, 11 widths | `figure_11a.sh` (12.5k), `cifar10_cnn_sgd.sh` (50k) | Tests whether "bigger models and more data hurt" holds only for unregularized training | 22 |
| **X4** optimizer | Does epoch-wise double descent survive other optimizers and LR schedules? | (a) Adam LR 10⁻⁴ → 10⁻³; (b) SGD + momentum 0.9 with dynamic drop from LR 0.01 (paper Fig 18c); k ∈ {12, 64} | `cifar10_resnet.sh` (Adam 10⁻⁴) | Checks that the epoch-wise curve is not an artefact of constant-LR Adam (paper App. E.1) | 4 |

Estimated cost: X1 ≈ 3.5, X2 ≈ 1.8, X3 ≈ 0.7, X4 ≈ 1.0 B200-hours.

## Run

Without the UCloud CLI, follow [UCLOUD_ABLATIONS.md](UCLOUD_ABLATIONS.md): one GPU job started from the web UI runs everything. Otherwise, from `Assignment 2 - Group/`, with the launch setup described in [code/README.md](code/README.md#vision-sweep-launcher). Each command submits one GPU job; append `plan=true` to list its conditions without training.

```bash
bash code/scripts/x1_horizon_resnet.sh
bash code/scripts/x1_horizon_cnn.sh
bash code/scripts/x2_seeds.sh            # seed 1
bash code/scripts/x2_seeds.sh seed=2
bash code/scripts/x3_weight_decay.sh
bash code/scripts/x4_optimizer.sh
```

Resume an interrupted job with `RUN_ROOT=<printed run root>` before the same command. Without access to the `hprjdk` W&B team, `export WANDB_MODE=disabled` first: nothing is logged to W&B, the run configs stay identical (so X1 can still continue the baseline runs), and all metrics are still saved in each run directory and Parquet file.

**X1** sets `extend_from` to the completed sweep (relative to `/work/<drive>`). Before training, `sweep.py` copies each matching run into the new run root, changes only its output paths and horizon, and resumes from its last checkpoint. Any other config difference stops the job before training. Data order, augmentation and the inverse-square-root LR depend only on the epoch or step, so the continued run follows the same trajectory as an uninterrupted longer run. The original runs stay untouched; X2 needs seed 0's 400-epoch `test_predictions.pt`.

**X3** uses `sample_counts=12500,50000`. A sample count equal to the full split keeps the full split in its original order, so the 50k arm differs from the baseline only in weight decay.

**X4** uses the default dynamic-drop settings (LR × 0.1 after 2,000 updates without a new lowest batch loss); the paper does not specify them.

## Results

Each job writes Parquet files to `/work/<drive>/our-results-folder/ablations/<id>/data/vision/`. Download them to the same path under [our-results-folder](our-results-folder/README.md), **not** into `data/vision/`: the figure scripts expect exactly the 229 baseline cells, and X1 files keep their baseline run IDs. X1 files hold the full history, so their first 400 epochs or 50k steps repeat the baseline. The Parquet schema has no learning-rate column; within X4, the `optimizer` column separates arm (a) from (b).

X2 also needs each seed's `test_predictions.pt` under `ablations/x2-seeds/predictions/seed<N>/width-<k>/` ([UCLOUD_ABLATIONS.md](UCLOUD_ABLATIONS.md) packs them) and the CIFAR-10 test labels in `code/data/`.

## Plots

[our-plots/ablations.ipynb](our-plots/ablations.ipynb) draws each ablation in the layout of the paper's matching figure (X1: Fig 9; X2: Fig 28; X3: Figs 21 and 11(a); X4: Figs 16 and 18), next to the main-sweep runs with the same settings, and shows the key numbers as tables. The figure code is [our-plots/ablation_figures.py](our-plots/ablation_figures.py); the PNGs are in [plots/ablations/](plots/README.md).

Two caveats from the first runs:
- **X4 Adam LR 10⁻³ is missing.** Both X4 arms started in the same second, so their run directories (`adam/…` and `sgd/…`) and Parquet files got the same name, and the SGD file replaced the Adam one. New runs cannot collide this way: a run name now includes the optimizer when it is not the architecture's default (`cifar10-resnet-sgd-k12-…`), and the export refuses to replace a file written by a run with other settings. The Adam metrics should still be under `/work/<drive>/runs/sweeps/ablations/x4-optimizer/adam/noise-20/width-{12,64}/`. Export them under new names with the current code, either on UCloud or on the Mac after copying each run's `metrics.jsonl`, `config.json` and `run_metadata.json`:

  ```bash
  cd "Assignment 2 - Group/code"
  for run in <x4 run root>/adam/noise-20/width-*/cifar10-resnet-k*/; do
    name="$(basename "$run")"
    ../../.venv/bin/python export_results.py "$run" --results_dir=../our-results-folder/ablations/x4-optimizer --name="${name/resnet-/resnet-adam-}"
  done
  ```

  This writes `cifar10-resnet-adam-k{12,64}-…parquet` next to the SGD files. The `optimizer` column still tells the two arms apart.
- **X4 SGD + dynamic drop stalled.** After about 15k iterations its train error stays flat at about the noise level. The drop rule compares every batch loss with the lowest single-batch loss seen so far, so it cuts the learning rate every 2,000 updates. This arm therefore cannot show epoch-wise double descent; the notebook's X4 check makes this visible.
