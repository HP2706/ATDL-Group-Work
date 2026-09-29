"""Render plots from the authors' released result arrays and endpoint CSVs."""

from __future__ import annotations

import sys
from pathlib import Path

import fire
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PLOTS_CODE = ROOT.parent / "our-plots"
sys.path.insert(0, str(PLOTS_CODE))

from translation import sample_plot, translation_plot  # noqa: E402
from vision import dynamics, figure9_panels, final_curve, load_vision, ocean  # noqa: E402


def save(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def final_rows(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["trial_index", "model_width"]
    if frame["sample_size"].notna().any():
        keys.append("sample_size")
    indices = frame.groupby(keys, dropna=False)["measurement_index"].idxmax()
    return frame.loc[indices].sort_values(keys)


def source_curve(
    source: str, dataset_dir: Path, max_measurements: int | None = None,
    model_widths: tuple[int, ...] | None = None,
    split_metrics: bool = False,
) -> plt.Figure:
    frame = final_rows(load_vision(source, dataset_dir, max_measurements, model_widths))
    if split_metrics:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
        for index, group in frame.groupby("trial_index", sort=True):
            for ax, metric, style in zip(axes, ("test_error", "train_error"), ("-", "--"), strict=True):
                ax.plot(group["model_width"], group[metric], style, label=f"Run {index + 1}")
        for ax, title in zip(axes, ("Test Error", "Train Error"), strict=True):
            ax.set(xlabel="CNN width", ylabel=title, title=title)
            ax.set_ylim(bottom=0)
        axes[0].legend(frameon=False, fontsize=8)
        fig.tight_layout()
        return fig
    fig, ax = plt.subplots(figsize=(12, 6))
    if frame["sample_size"].notna().any():
        for size, group in frame.groupby("sample_size", sort=True):
            ax.plot(group["model_width"], group["test_error"], label=f"n={int(size):,}")
    else:
        for index, group in frame.groupby("trial_index", sort=True):
            ax.plot(group["model_width"], group["test_error"], label=f"Mlist[{index}] test")
            ax.plot(group["model_width"], group["train_error"], linestyle="--", alpha=0.5)
    ax.set(xlabel="Model width", ylabel="Final recorded error", title=source)
    ax.set_ylim(bottom=0)
    widths = sorted(frame["model_width"].unique())
    if len(widths) <= 15:
        ax.set_xticks(widths)
    ax.legend(fontsize=9, ncol=2)
    fig.tight_layout()
    return fig


def comparison(
    sources: list[tuple[str, str, int]], dataset_dir: Path, title: str,
    max_measurements: int | None = None,
    measurement_caps: dict[str, int] | None = None,
    model_widths: tuple[int, ...] | None = None,
    include_train: bool = False,
) -> plt.Figure:
    fig, plot_axes = plt.subplots(1, 2 if include_train else 1,
                                  figsize=(14, 5) if include_train else (13, 6), sharex=include_train)
    axes = list(plot_axes) if include_train else [plot_axes]
    for source, label, index in sources:
        cap = measurement_caps[source] if measurement_caps is not None else max_measurements
        final = final_rows(load_vision(source, dataset_dir, cap, model_widths))
        selected = final.loc[final["trial_index"] == index]
        test_line, = axes[0].plot(selected["model_width"], selected["test_error"], label=label)
        if include_train:
            axes[1].plot(selected["model_width"], selected["train_error"], linestyle="--",
                         color=test_line.get_color(), alpha=0.75)
    for ax, metric_title in zip(axes, ("Test Error", "Train Error")[:len(axes)], strict=True):
        ax.set(xlabel="Model width", ylabel="Final recorded error", title=metric_title if include_train else title)
        ax.set_ylim(bottom=0)
        widths = sorted({int(width) for line in ax.lines for width in line.get_xdata()})
        if len(widths) <= 15:
            ax.set_xticks(widths)
    axes[0].legend(fontsize=10)
    if include_train:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def augmentation_panels(dataset_dir: Path) -> plt.Figure:
    """The available augmented and non-augmented CNN entries, each split by metric."""
    sources = (("cifar10-mcnn-noaug-sgd", 1, "Without augmentation"),
               ("cifar10-mcnn-p10-sgd", 0, "With augmentation"))
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey="col")
    for row, (source, index, label) in enumerate(sources):
        frame = final_rows(load_vision(source, dataset_dir))
        selected = frame.loc[frame["trial_index"].eq(index)]
        for column, (metric, style) in enumerate((("test_error", "-"), ("train_error", "--"))):
            ax = axes[row, column]
            ax.plot(selected["model_width"], selected[metric], style, color="#4f708c")
            ax.set(xlabel="CNN width", ylabel=metric.replace("_", " ").title(),
                   title=f"{label} · {'Test' if column == 0 else 'Train'} Error")
            ax.set_ylim(bottom=0)
    fig.tight_layout()
    return fig


def sample_heatmap(
    dataset_dir: Path, max_measurements: int | None = None,
    model_widths: tuple[int, ...] | None = None,
) -> plt.Figure:
    frame = final_rows(load_vision("dd_grid_p20", dataset_dir, max_measurements, model_widths))
    grid = frame.pivot(index="sample_size", columns="model_width", values="test_error")
    fig, ax = plt.subplots(figsize=(14, 7))
    image = ax.imshow(grid.to_numpy(), aspect="auto", origin="lower", cmap="viridis")
    positions = np.arange(len(grid.columns)) if len(grid.columns) <= 15 else np.arange(0, len(grid.columns), 5)
    ax.set_xticks(positions, [str(grid.columns[index]) for index in positions])
    ax.set_yticks(np.arange(len(grid.index)), [f"{int(size):,}" for size in grid.index])
    ax.set(xlabel="CNN width", ylabel="Training examples", title="CIFAR-10 CNN, 20% noise")
    fig.colorbar(image, ax=ax, label="Final recorded test error")
    fig.tight_layout()
    return fig


def render(dataset_dir: str = str(ROOT.parent / "their-results" / "hf_dataset"), output_dir: str = str(ROOT.parent / "plots" / "published")) -> list[str]:
    data = Path(dataset_dir)
    output = Path(output_dir)
    written: list[str] = []

    def emit(name: str, figure: plt.Figure) -> None:
        path = output.parent / "exploratory" / name if name.startswith("source_sweeps/") else output / name
        save(figure, path)
        written.append(str(path))
        print(path, flush=True)

    # These use the layouts and colormap ported from the authors' notebooks.
    p15 = "cifar10-resnet18k-p15-adam-reps"
    emit("figure_01_resnet_15pct_final.png", final_curve(p15, data, 0.15, 10))
    emit("figure_01_resnet_15pct_dynamics.png", dynamics(p15, data, 0, noise_level=0.15))
    emit("figure_02_resnet_15pct_test_heatmap.png", ocean(p15, data, 0, "test_error"))
    emit("figure_02_resnet_15pct_train_heatmap.png", ocean(p15, data, 0, "train_error", contours=False))
    emit("figure_03_translation_small.png", translation_plot(data, "small"))
    emit("figure_08_translation_full.png", translation_plot(data, "full"))
    emit("figure_09_resnet_20pct_dynamics.png", dynamics("cifar10-resnet18k-50k-adam", data, 2, noise_level=0.20))
    emit("figure_09_resnet_20pct_heatmap.png", ocean("cifar10-resnet18k-50k-adam", data, 2))
    emit("figure_09_resnet_20pct_panels.png", figure9_panels("cifar10-resnet18k-50k-adam", data))
    emit("figure_11b_translation_samples.png", sample_plot(data))
    emit("figure_12_cifar10_cnn_sample_heatmap.png", sample_heatmap(data))
    emit("figure_19_cifar100_resnet_heatmap.png", ocean("cifar100-resnet18k-50k-adam", data, 0))
    emit("figure_19_cifar100_resnet_dynamics.png", dynamics("cifar100-resnet18k-50k-adam", data, 0))
    emit("figure_20_cifar100_cnn_dynamics.png", dynamics("pct-cifar100-mcnn-p0-sgd-noaug-reps", data, 0))
    emit("figure_20_cifar100_cnn_heatmap.png", ocean("pct-cifar100-mcnn-p0-sgd-noaug-reps", data, 0))
    emit("figure_25_cifar10_cnn_10pct_heatmap.png", ocean("cifar10-mcnn-p10-sgd", data, 0))

    # Curves for comparisons whose original composite plotting code was not released.
    emit("figure_04_resnet_cifar10_noise.png", comparison([
        ("cifar10-resnet18k-50k-adam", "Mlist[0], clean", 0),
        ("cifar10-resnet18k-50k-adam", "Mlist[1], 10% noise", 1),
        ("cifar10-resnet18k-50k-adam", "Mlist[2], 20% noise", 2),
    ], data, "CIFAR-10 ResNet18 noise sweep"))
    emit("figure_04_resnet_cifar100_noise.png", comparison([
        ("cifar100-resnet18k-50k-adam", "Mlist[0], clean", 0),
        ("cifar100-resnet18k-50k-adam", "Mlist[1], 10% noise", 1),
        ("cifar100-resnet18k-50k-adam", "Mlist[2], 20% noise", 2),
    ], data, "CIFAR-100 ResNet18 noise sweep"))
    emit("figure_06_cifar10_cnn_optimizers.png", comparison([
        ("cifar10-mcnn-noaug-sgd", "SGD Mlist[0]", 0),
        ("cifar10-mcnn-noaug-adam", "Adam Mlist[0]", 0),
    ], data, "CIFAR-10 CNN optimizer comparison, no augmentation", include_train=True))
    emit("figure_05_cifar10_cnn_augmentation.png", augmentation_panels(data))
    emit("figure_07_cifar100_cnn.png", source_curve("pct-cifar100-mcnn-p0-sgd-noaug-reps", data, split_metrics=True))
    emit("figure_11a_cifar10_cnn_sample_sizes.png", source_curve("dd_grid_p20", data))
    emit("figure_10_cifar10_cnn_dynamics.png", dynamics("cifar10-mcnn-p20-sgd", data, 0))
    emit("figure_21_cifar10_cnn_weight_decay.png", comparison([
        ("pct-cifar10-mcnn-p10-sgd-aug-decay-big", "Decay archive 1", 0),
        ("pct-cifar10-mcnn-p10-sgd-aug-decay5-big", "Decay archive 2", 0),
    ], data, "CIFAR-10 CNN published weight-decay variants"))
    emit("figure_27_cifar10_wide_cnn.png", comparison([
        ("pct-cifar10-mcnn-50000-p0-sgd-big", "Clean", 0),
        ("pct-cifar10-mcnn-50000-p20-sgd-big", "20% noise", 0),
    ], data, "Wide CIFAR-10 CNN published runs"))

    for parquet in sorted((data / "data" / "vision").glob("*.parquet")):
        emit(f"source_sweeps/{parquet.stem}.png", source_curve(parquet.stem, data))
    return written


if __name__ == "__main__":
    fire.Fire(render)
