# Early views of the published vision histories

The [comparison notebook](short_horizon_comparison.ipynb) has one runnable cell per numbered vision figure and one per released source sweep. Each cell renders the original full-width chart and the width-ablated chart side by side at 1%, 5%, 10%, and final saved-measurement horizons. The generated PNGs under `../plots/exploratory/short_horizon/` are ignored by Git and can be recreated: 23 numbered vision panels, including Figure 9's three-trajectory left panel and heatmap plus the extra five-noise Figure 4 view, and all 32 released vision source sweeps. The extra Figure 4 five-noise comparison also has a final panel under `100pct/`.

These are **truncations of the authors' released runs**, not new short-run experiments. The source files record `measurement_index`, an array position. They do not provide a verified epoch or optimizer-step mapping. For the main ResNet sources, 1%, 5%, and 10% are 40, 200, and 400 of 3,999 recorded positions. The SGD source histories vary in length, so 1% cannot be equated with 5,000 SGD steps.

To regenerate the PNGs from the local Parquet files, run from `Assignment 2 - Group`:

```bash
MPLCONFIGDIR=/tmp/atdl-mpl ../.venv/bin/python their_plots/short_horizon.py
```

Run the notebook's setup cells once, then run a figure cell whenever you want to regenerate its paired plots. Edit `ABLATION_WIDTHS` or `SELECTED_HORIZONS` in the helper cell to change the comparison. Translation plots have only final released values, so they remain outside this vision-history notebook.
