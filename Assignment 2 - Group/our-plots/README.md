# Vision comparison plots

The [comparison notebook](resnet_ours_vs_published.ipynb) and [generated figures](../plots/README.md) use all **229 audited vision run files** under [`../our-results-folder/data/vision/`](../our-results-folder/data/vision/). Each cell has 40 measurements and reaches its planned endpoint: 400 epochs for our Adam runs or 50,000 steps for our SGD CNNs. The plots are reduced reproductions: the paper used denser width grids and 4,000 Adam epochs, 500,000 CIFAR-10 SGD CNN steps, or 1,000,000 clean CIFAR-100 CNN steps.

- Figures **1, 2, and 9** retain the CIFAR-10 ResNet adaptations. Figure 1 compares the authors' released 10% run with our 10% run on the same ten widths and 400 displayed epochs; this is now the main-report opening comparison. The original 15% opening figure is retained separately in the appendix. Figure 2 is also adapted to 10% noise.
- [Figure 4](../plots/comparison/figure_04.png) now includes both our CIFAR-10 and CIFAR-100 ResNet panels.
- [Figure 5](../plots/comparison/figure_05.png) now includes both augmented and non-augmented CIFAR-10 CNN SGD panels through width 64.
- [Figure 6](../plots/comparison/figure_06.png) now compares the authors' and our clean CIFAR-10 CNN **SGD and Adam** arms. Our SGD endpoint is at 50,000 steps and Adam at 400 epochs; the authors' curves are at their full recorded horizons. [Figure 7](../plots/comparison/figure_07.png) compares the authors’ five-trial endpoints with the clean CIFAR-100 CNN run that UCloud job `12403784` actually produced, despite its `figure-6` job name.
- [Figure 10(c)](../plots/comparison/figure_10.png) shows width-128 trajectories on separate native coordinates: authors' full 0%/20% histories versus our recorded 0%/20% histories (10% is retained in the standalone chart) through 50,000 steps. No guessed time mapping or smoothing is applied; the ambiguous 10% author source is excluded.
- [Figure 11(a)](../plots/comparison/figure_11a.png) now compares our 20%-noise 12,500/25,000/50,000-example curves with the authors’ released 20%-noise subset data at the same widths; our 10% curves are shown separately because the released archive has no 10%-noise subset source. [Figure 12](../plots/comparison/figure_12.png) places the authors’ and our 20% cells on the same reduced width × sample-size heatmap and slice axes.

The authors' ResNet archive stores measurement indices, displayed as `index + 1` in their notebook. Its CNN archive lacks a confirmed logging cadence. For the approximate-horizon Figure 5 panels only, we estimate 256 steps per author measurement from the 1,952-point complete source series and the stated 500,000-step protocol. Our step and epoch fields were recorded during training. No error bars are inferred from our one-seed cells.

Regenerate the paired and ours-only PNG figures from `Assignment 2 - Group` with:

```bash
MPLCONFIGDIR=/tmp/atdl-mpl ../.venv/bin/python -c "from pathlib import Path; import sys; sys.path.insert(0, 'our-plots'); from paper_figures import save_figures; save_figures(Path.cwd(), Path('plots'))"
```

These Python modules port the layouts in the authors' five plotting notebooks at [commit `0d653296`](https://gitlab.com/harvard-machine-learning/double-descent): the 14×7 blue ResNet curve, 15×8 heatmap with the exact 256-color `colormap_inferno_strong_1.txt`, 15×6 epoch-colored dynamics, and 20×8 translation line plots. The CLI does not require TensorFlow or W&B.

From the `ATDL-Group-Work` project root, after `uv sync`:

```bash
.venv/bin/python 'Assignment 2 - Group/our-plots/cli.py' resnet
.venv/bin/python 'Assignment 2 - Group/our-plots/cli.py' ocean
.venv/bin/python 'Assignment 2 - Group/our-plots/cli.py' dynamics
.venv/bin/python 'Assignment 2 - Group/our-plots/cli.py' translation_model --sweep=small
.venv/bin/python 'Assignment 2 - Group/our-plots/cli.py' translation_model --sweep=full
.venv/bin/python 'Assignment 2 - Group/our-plots/cli.py' translation_samples
```

Each command accepts `--output=<path.png|path.pdf|path.svg>` and `--dataset_dir=<folder>`; the image commands also accept `--source=<original-experiment-folder>`. By default, inputs are read from `../their-results/hf_dataset/` and exploratory output goes to `../plots/exploratory/published_cli/`. Curated numbered figures are in [`../plots/`](../plots/README.md). The same Parquet schema can be used for our runs under `../our-results-folder/`; pass an explicit `--output` when plotting our data.

The visual geometry and colormap follow the authors' code. We corrected the noisy-test-error sign, width tick labels, and translation language/sample labels. Their notebooks called recorded array positions "epochs" even for experiments with unknown logging intervals; this CLI labels them as recorded time points. The red early-stopping curve is an oracle minimum over test measurements, not a validation-selected stopping rule.

Figure 10 reference selection: clean `pct-cifar10-mcnn-50000-p0-sgd-big` plus `dd_grid_p20` filtered to width 128 and sample size 50,000. The latter ends at 21.67% test error; the previously selected `pct-cifar10-mcnn-50000-p20-sgd-big` ends at 37.05% and is not substituted for the sample-grid history. Saved-position axes remain unverified physical time, with different series lengths.
