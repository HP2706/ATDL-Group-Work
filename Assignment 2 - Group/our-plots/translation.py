"""Script ports of the authors' 20×8 translation line-plot layouts."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pyarrow.parquet as pq
from matplotlib.figure import Figure


def load_translation(dataset_dir: Path) -> pd.DataFrame:
    path = dataset_dir / "data" / "translation" / "translation.parquet"
    if not path.is_file():
        raise FileNotFoundError(path)
    return pq.read_table(path).to_pandas()


def translation_plot(dataset_dir: Path, sweep: str = "small") -> Figure:
    """Model-wise width curves from nlp-model-double-descent/SimpleNLP layouts."""
    frame = load_translation(dataset_dir)
    sources = {
        "small": [("df4k.csv", "IWSLT'14 De→En, 4k pairs"), ("df18k.csv", "IWSLT'14 De→En, 18k pairs")],
        "full": [("nlp-model-dd.csv", "IWSLT'14 De→En"), ("dd-translation-model-french.csv", "WMT'14 En→Fr")],
    }
    if sweep not in sources:
        raise ValueError("sweep must be 'small' or 'full'")
    if sweep == "full":
        fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=True)
    else:
        fig, single_ax = plt.subplots(1, 1, figsize=(20, 8))
        axes = [single_ax]
    for filename, label in sources[sweep]:
        selected = frame.loc[frame["source_file"] == filename].sort_values("model_width")
        test_line, = axes[0].plot(selected["model_width"], selected["test_loss"],
                                  "-" if sweep == "full" else "o-", label=label)
        if sweep == "full":
            axes[1].plot(selected["model_width"], selected["train_loss"], "--",
                         color=test_line.get_color(), alpha=0.75)
    if sweep == "full":
        axes[1].set(title="Train Loss", ylabel="Published train loss (archive units)")
        axes[0].set(title="Test Loss", ylabel="Published test loss (archive units)")
        fig.suptitle("Double Descent of Transformers")
    else:
        axes[0].set(title="Double Descent of Transformers", ylabel="Published cross-entropy test loss")
    for ax in axes:
        ax.set_xlabel("Embedding dimension")
    axes[0].legend(fontsize=12 if sweep == "full" else 20)
    fig.tight_layout()
    return fig


def sample_plot(dataset_dir: Path) -> Figure:
    """Sample-count curves using the released d=64 and d=80 CSVs."""
    frame = load_translation(dataset_dir)
    sources = [
        ("dd-translation-samples-aug26-small.csv", "d=64"),
        ("dd-translation-samples-aug27-size80.csv", "d=80"),
    ]
    fig, ax = plt.subplots(1, 1, figsize=(20, 8))
    for filename, label in sources:
        selected = frame.loc[frame["source_file"] == filename].sort_values("sample_size")
        ax.plot(selected["sample_size"], selected["test_loss"], "o-", label=label)
    ax.set_ylabel("Published cross-entropy test loss")
    ax.set_xlabel("Training sentence pairs")
    ax.legend(fontsize=20, ncol=3)
    ax.set_title("Sample-Size Dependence of Transformers", pad=12)
    fig.tight_layout()
    return fig
