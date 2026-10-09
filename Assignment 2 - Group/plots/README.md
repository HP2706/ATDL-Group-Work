# Assignment 2 plots

All image output lives here. The plotting programs remain in
[`../our-plots/`](../our-plots/) and [`../their_plots/`](../their_plots/).

| Directory | Meaning | Git policy |
| --- | --- | --- |
| [`comparison/`](comparison/) | Paired panels showing our measurements alongside the authors' released measurements. These are partial reproductions, with limits described in [`../EXPERIMENTS.md`](../EXPERIMENTS.md). | Track ten numbered PNGs. |
| [`ours/`](ours/) | The same ten figure layouts rendered from our measurements alone. | Track ten numbered PNGs. |
| [`ablations/`](ablations/) | The targeted ablations X1–X4 in the paper's appendix layouts, from [`../our-plots/ablations.ipynb`](../our-plots/ablations.ipynb). | Track five PNGs. |
| [`published/`](published/) | Numbered reconstructions from the authors' released measurements alone. | Track curated numbered PNGs. |
| `exploratory/` | CLI output, source sweeps, and early-horizon views. | Ignore generated images. |
| [`assets/`](assets/) | Input files needed by the plotting code, including the published color map. | Track. |
| `exports/` | Optional duplicate PDF exports. | Ignore. |

The comparison and ours-only PNGs come from the same plotting functions and
data selections. From `Assignment 2 - Group`, generate them with:

```bash
MPLCONFIGDIR=/tmp/atdl-mpl ../.venv/bin/python -c "from pathlib import Path; import sys; sys.path.insert(0, 'our-plots'); from paper_figures import save_figures; save_figures(Path.cwd(), Path('plots'))"
```

Generate the ablation figures (results under `../our-results-folder/ablations/`, CIFAR-10 test labels in `../code/data/`) with:

```bash
MPLCONFIGDIR=/tmp/atdl-mpl ../.venv/bin/python -c "from pathlib import Path; import sys; sys.path.insert(0, 'our-plots'); from ablation_figures import save_ablation_figures; save_ablation_figures(Path.cwd(), Path('plots/ablations'))"
```

Generate the published-only numbered reconstructions with:

```bash
MPLCONFIGDIR=/tmp/atdl-mpl ../.venv/bin/python their_plots/render.py
```

Both commands use local Parquet files and do not start training.
