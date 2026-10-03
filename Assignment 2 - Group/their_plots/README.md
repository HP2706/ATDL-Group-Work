# Plots from the authors' released results

This directory contains code and notebooks that generate plots from the authors' public results, converted to Parquet in [`../their-results/hf_dataset`](../their-results/hf_dataset). The curated images are in [`../plots/published/`](../plots/README.md). The renderer can generate 32 exploratory source-sweep files under `../plots/exploratory/source_sweeps/`, one for every released vision result source. Each line is the final recorded test error from one `Mlist` entry, with dashed training-error lines. `Mlist` position is shown literally because it can mean a noise setting, a replicate, or another condition depending on the archive.

The numbered PNGs in [`../plots/published/`](../plots/README.md) are **reconstructions**, not copies of the paper images. Where the authors supplied plotting notebooks, the ResNet heatmaps, dynamics curves, final curve, and translation curves reuse their layouts, including their custom inferno colormap for the ResNet heatmaps. Other comparisons use the released numeric arrays with a simple, labeled layout because the paper's plotting code for those panels was not released. They should be cited as plots of the authors' data, not as our experimental results.

The [short-horizon comparison notebook](short_horizon_comparison.ipynb) has one runnable cell per numbered vision figure and one per released source sweep. Each cell renders the original full-width chart beside a freshly generated width-ablated chart at the 1%, 5%, 10%, and final saved-measurement horizons. Fifteen numbered figure cells cover 23 reconstructed panels, including the full five-level CIFAR-10 Figure 4 noise sweep; another 32 cells cover the source sweeps. Set `ABLATION_WIDTHS` once in the helper cell; the default is `{2,4,8,12,16,24,32,40,48,56,64}`. The percentages count saved measurement positions; the public files do not establish an exact epoch or optimizer-step mapping.

## Figure coverage

| Paper figure | Reconstructed output | Limit |
| --- | --- | --- |
| 1 | [`final`](../plots/published/figure_01_resnet_15pct_final.png), [`dynamics`](../plots/published/figure_01_resnet_15pct_dynamics.png) | Both views of the released five-run, 15% noise ResNet grid; the dynamics view shows one run. |
| 2 | [`test heatmap`](../plots/published/figure_02_resnet_15pct_test_heatmap.png), [`train heatmap`](../plots/published/figure_02_resnet_15pct_train_heatmap.png) | One released run. |
| 3 | [`translation, 4k and 18k`](../plots/published/figure_03_translation_small.png) | Endpoint CSVs only. |
| 4 | [`CIFAR-10`](../plots/published/figure_04_resnet_cifar10_noise.png), [`CIFAR-100`](../plots/published/figure_04_resnet_cifar100_noise.png) | Reconstructed final-error comparisons; no original composite code. |
| 5 | [`CNN augmentation`](../plots/published/figure_05_cifar10_cnn_augmentation.png) | Illustrative comparison of two released entries, with separate solid test-error and dashed train-error subfigures. Archive condition mapping does not establish the entire paper panel. |
| 6 | [`CNN optimizers`](../plots/published/figure_06_cifar10_cnn_optimizers.png) | Released no-augmentation SGD and Adam entries, with separate solid test-error and dashed train-error subfigures. Other panel details are unavailable. |
| 7 | [`CIFAR-100 CNN`](../plots/published/figure_07_cifar100_cnn.png) | All five released repetitions, with separate solid test-error and dashed train-error subfigures; training setting in archive is no augmentation. |
| 8 | [`full translation`](../plots/published/figure_08_translation_full.png) | Endpoint CSVs with separate solid test-loss and dashed train-loss subfigures; archive loss units remain unresolved. |
| 9 | [`two-panel reconstruction`](../plots/published/figure_09_resnet_20pct_panels.png), [`dynamics`](../plots/published/figure_09_resnet_20pct_dynamics.png), [`heatmap`](../plots/published/figure_09_resnet_20pct_heatmap.png) | The left panel follows widths 3, 12, and 64; the right panel shows the corresponding width-by-time heatmap from the released 20% noise ResNet run. |
| 10(c) | [`Width-128 CNN trajectories`](../plots/published/figure_10_cifar10_cnn_dynamics.png) | Clean full-data reference and explicit 50,000-example/20% sample-grid reference; logging cadences unverified. The paper's two width-128 ResNet arms are not reproduced. |
| 11(a) | [`CNN sample-size curves`](../plots/published/figure_11a_cifar10_cnn_sample_sizes.png) | One released sample-size grid, with no trial error bars. |
| 11(b) | [`translation sample-size curves`](../plots/published/figure_11b_translation_samples.png) | Endpoint CSVs only. |
| 12 | [`CNN sample-size heatmap`](../plots/published/figure_12_cifar10_cnn_sample_heatmap.png) | Final recorded error from the released grid; not the original paper layout. |
| 13 | — | Exact parameter-count configuration for all three models is unavailable. |
| 14–15 | — | Fashion-MNIST random-feature result arrays were not released in the downloaded bucket. |
| 16–18 | — | Learning-rate schedule comparison arrays were not released. |
| 19 | [`heatmap`](../plots/published/figure_19_cifar100_resnet_heatmap.png), [`dynamics`](../plots/published/figure_19_cifar100_resnet_dynamics.png) | Clean CIFAR-100 ResNet; the full paper composite is not reconstructed. |
| 20 | [`heatmap`](../plots/published/figure_20_cifar100_cnn_heatmap.png), [`dynamics`](../plots/published/figure_20_cifar100_cnn_dynamics.png) | Clean CIFAR-100 CNN; the full paper composite is not reconstructed. |
| 21 | [`weight-decay variants`](../plots/published/figure_21_cifar10_cnn_weight_decay.png) | Two released variant archives; exact decay coefficients cannot be established from filenames alone. |
| 22 | — | A ResNet decay archive exists, but its configuration does not establish the paper's stated SGD comparison. The renderer can generate its `cifar10-resnet18k-p20-decay.png` source sweep. |
| 23–24 | — | Translation epoch histories were not released, only final CSV values. |
| 25 | [`CNN heatmap`](../plots/published/figure_25_cifar10_cnn_10pct_heatmap.png) | One released augmented CNN entry. |
| 26 | — | The available adversarial archive is named `adv-mcnn-25k-p0`, conflicting with the paper's ResNet caption. The renderer can generate its source sweep. |
| 27 | [`wide CNN`](../plots/published/figure_27_cifar10_wide_cnn.png) | Two released wide-CNN conditions; not the complete paper panel. |
| 28–29 | — | Ensemble plots require member predictions or votes, which are absent from the released data. |

## Regenerate

From `Assignment 2 - Group`, with the project virtual environment available:

```bash
MPLCONFIGDIR=/tmp/atdl-mpl ../.venv/bin/python their_plots/render.py
```

This runs the plotting pass only; it does not train models. The script prints each output path. It reads the local Parquet dataset and does not download anything.

Figure 10(c) now uses `paper_figures.figure_10_cnn(..., authors_only=True)` to render actual width-128 training trajectories, not the width-sweep `dynamics` renderer. Authors' 0%/20% full-data histories use saved measurement indices without an inferred optimizer-step axis; the 10% source is excluded pending entry/cadence provenance. No smoothing is applied.

Figure 10 reference selection: clean `pct-cifar10-mcnn-50000-p0-sgd-big` plus `dd_grid_p20` filtered to width 128 and sample size 50,000. The latter ends at 21.67% test error; the previously selected `pct-cifar10-mcnn-50000-p20-sgd-big` ends at 37.05% and is not substituted for the sample-grid history. Saved-position axes remain unverified physical time, with different series lengths.

The matched Figure 1 asset, `figure_01_resnet_10pct_matched.png`, uses the released 10% ResNet run on the same ten widths and 400 displayed epochs as ours. The 15% full-history reconstructions remain available as references.
