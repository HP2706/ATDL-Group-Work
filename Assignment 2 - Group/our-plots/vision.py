"""Script ports of the authors' ResNet curve, heatmap, and dynamics layouts."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from matplotlib.figure import Figure


def load_vision(
    source: str,
    dataset_dir: Path,
    max_measurements: int | None = None,
    model_widths: tuple[int, ...] | None = None,
) -> pd.DataFrame:
    path = dataset_dir / "data" / "vision" / f"{source}.parquet"
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pq.read_table(path).to_pandas()
    if max_measurements is not None:
        if max_measurements < 1:
            raise ValueError("max_measurements must be positive")
        frame = frame.loc[frame["measurement_index"] < max_measurements]
    if model_widths is not None:
        if not model_widths:
            raise ValueError("model_widths must contain at least one width")
        frame = frame.loc[frame["model_width"].isin(model_widths)]
        if frame.empty:
            raise ValueError(f"None of the selected widths are in {source}")
    return frame


def noisy_test_error(clean_error: np.ndarray, noise_level: float, class_count: int) -> np.ndarray:
    if not 0.0 <= noise_level <= 1.0 or class_count < 2:
        raise ValueError("noise_level must be in [0, 1] and class_count must be at least 2")
    return noise_level + (1.0 - noise_level - noise_level / (class_count - 1)) * clean_error


def final_curve(
    source: str,
    dataset_dir: Path,
    noise_level: float = 0.0,
    class_count: int = 10,
    max_measurements: int | None = None,
    model_widths: tuple[int, ...] | None = None,
) -> Figure:
    """Port of intro_resnet_plot.ipynb's 14×7 blue train/test curve layout."""
    frame = load_vision(source, dataset_dir, max_measurements, model_widths)
    final_indices = frame.groupby(["trial_index", "model_width"])["measurement_index"].transform("max")
    final = frame.loc[frame["measurement_index"] == final_indices].copy()
    final["test_error"] = noisy_test_error(final["test_error"].to_numpy(), noise_level, class_count)
    by_width = final.groupby("model_width", sort=True)
    summary = by_width.agg(
        test_mean=("test_error", "mean"),
        test_std=("test_error", lambda values: values.std(ddof=0)),
        train_mean=("train_error", "mean"),
        train_std=("train_error", lambda values: values.std(ddof=0)),
    )
    widths = summary.index.to_numpy(dtype=np.int32)
    fig, ax = plt.subplots(1, 1, figsize=(14, 7), dpi=300)
    ax.plot(widths, summary["test_mean"], "-", color="b", lw=3, label="Test")
    ax.fill_between(widths, summary["test_mean"] - summary["test_std"], summary["test_mean"] + summary["test_std"], alpha=0.3, facecolor="b")
    ax.plot(widths, summary["train_mean"], "--", color="b", lw=3, alpha=0.4, label="Train")
    ax.fill_between(widths, summary["train_mean"] - summary["train_std"], summary["train_mean"] + summary["train_std"], alpha=0.3, facecolor="b")
    tick_indices = list(range(len(widths))) if len(widths) <= 15 else [0] + list(range(9, len(widths), 10))
    ax.set_xticks(widths[tick_indices])
    ax.set_xlabel("ResNet18 width parameter", labelpad=10)
    ax.set_ylabel("Test / Train Error")
    ax.set_ylim(bottom=0.0)
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def metric_matrix(frame: pd.DataFrame, trial_index: int, metric: str) -> tuple[np.ndarray, np.ndarray]:
    selected = frame.loc[frame["trial_index"] == trial_index]
    if selected.empty:
        raise ValueError(f"Trial {trial_index} is absent")
    matrix = selected.pivot(index="model_width", columns="measurement_index", values=metric)
    if matrix.isna().any().any():
        raise ValueError(f"Incomplete {metric} grid for trial {trial_index}")
    return matrix.index.to_numpy(dtype=np.int32), matrix.to_numpy(dtype=np.float64)


def log_spaced_indices(length: int, base: float) -> np.ndarray:
    if length < 2 or base <= 1:
        raise ValueError("At least two time points and base > 1 are required")
    indices = np.asarray(base ** np.arange(0, math.log(length) / math.log(base)), dtype=np.int32) - 1
    return np.unique(np.clip(indices, 0, length - 1))


def load_colormap() -> colors.ListedColormap:
    path = Path(__file__).resolve().parent.parent / "plots" / "assets" / "colormap_inferno_strong_1.txt"
    rgb = np.loadtxt(path, dtype=np.float64) / 255.0
    if rgb.shape != (256, 3):
        raise ValueError(f"Unexpected upstream colormap shape: {rgb.shape}")
    return colors.ListedColormap(rgb)


def ocean(
    source: str,
    dataset_dir: Path,
    trial_index: int = 0,
    metric: str = "test_error",
    base: float = 1.1,
    contours: bool = True,
    max_measurements: int | None = None,
    model_widths: tuple[int, ...] | None = None,
) -> Figure:
    """Port of intro_ocean_plot.ipynb's 15×8 heatmap and source colormap."""
    frame = load_vision(source, dataset_dir, max_measurements, model_widths)
    widths, full = metric_matrix(frame, trial_index, metric)
    indices = log_spaced_indices(full.shape[1], base)
    values = full[:, indices].T
    if metric == "test_error":
        positive = values[np.isfinite(values) & (values > 0)]
        if positive.size == 0:
            raise ValueError("Log-scaled test-error heatmap needs positive values")
        norm: colors.Normalize = colors.LogNorm(vmin=float(positive.min()), vmax=float(positive.max()))
    else:
        norm = colors.Normalize(vmin=float(np.nanmin(values)), vmax=float(np.nanmax(values)))
    fig, ax = plt.subplots(1, 1, figsize=(15, 8))
    image = ax.imshow(values, cmap=load_colormap(), norm=norm, aspect="auto", interpolation="none")
    if contours and metric == "test_error":
        _, train = metric_matrix(frame, trial_index, "train_error")
        ax.contour(train[:, indices].T, levels=[0.15, 0.5], colors="white", linestyles="dashed", alpha=0.7)
    tick_values = np.unique(np.clip(np.rint(np.geomspace(1, full.shape[1], num=min(5, full.shape[1]))).astype(int) - 1, 0, full.shape[1] - 1))
    tick_positions = [int(np.argmin(abs(indices - value))) for value in tick_values]
    ax.set_yticks(tick_positions, [str(value + 1) for value in tick_values], fontsize=17)
    ax.invert_yaxis()
    ax.set_ylabel("Recorded time point", fontsize=22)
    x_positions = np.arange(len(widths)) if len(widths) <= 15 else np.arange(0, len(widths), 15)
    ax.set_xticks(x_positions, [str(widths[position]) for position in x_positions], fontsize=17)
    ax.set_xlabel("ResNet18 width parameter", fontsize=22)
    ax.set_title(metric.replace("_", " ").title(), fontsize=23)
    bar = fig.colorbar(image, fraction=0.025, pad=0.04)
    bar.ax.tick_params(labelsize=15)
    fig.tight_layout()
    return fig


def rescale_viridis() -> colors.ListedColormap:
    original = plt.colormaps["viridis"]
    palette = []
    for index in range(256):
        x = index / 256
        adjusted = 0.5 * (1 - np.cos(np.pi * x**0.7)) / 2 + 0.5 * x
        palette.append(original(adjusted))
    return colors.ListedColormap(palette)


def dynamics(
    source: str,
    dataset_dir: Path,
    trial_index: int = 0,
    metric: str = "test_error",
    base: float = 1.1,
    noise_level: float = 0.0,
    class_count: int = 10,
    max_measurements: int | None = None,
    model_widths: tuple[int, ...] | None = None,
) -> Figure:
    """Port of intro_ocean_dynamics.ipynb's epoch-colored 15×6 curves."""
    frame = load_vision(source, dataset_dir, max_measurements, model_widths)
    widths, matrix = metric_matrix(frame, trial_index, metric)
    if metric == "test_error":
        matrix = noisy_test_error(matrix, noise_level, class_count)
    indices = log_spaced_indices(matrix.shape[1], base)
    palette = rescale_viridis()
    plt.rcParams.update({"font.size": 16, "legend.frameon": False})
    fig, ax = plt.subplots(1, 1, figsize=(15, 6))
    for index in indices:
        fraction = math.log(index + 1) / math.log(matrix.shape[1])
        alpha = min(0.2 * math.sqrt(fraction) + 0.2, 1.0)
        ax.plot(widths, matrix[:, index], linestyle="-", color=palette(1.0 - fraction), alpha=alpha)
    ax.plot(widths, np.nanmin(matrix, axis=1), linestyle="dashed", color="red", lw=2.5, label="Oracle minimum over recorded points")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_xlabel("ResNet18 width parameter", labelpad=10)
    if len(widths) <= 15:
        ax.set_xticks(widths)
    ax.legend()
    mapper = plt.cm.ScalarMappable(cmap=palette, norm=colors.Normalize(vmin=0.0, vmax=1.0))
    bar = fig.colorbar(mapper, ax=ax)
    recorded_points = [value for value in (1, 10, 100, 1000) if value <= matrix.shape[1]]
    color_ticks = [1.0 - math.log(value) / math.log(matrix.shape[1]) for value in recorded_points]
    bar.set_ticks(color_ticks)
    bar.set_ticklabels([str(value) for value in recorded_points])
    bar.set_label("Recorded time point")
    fig.tight_layout()
    return fig


def figure9_panels(
    source: str,
    dataset_dir: Path,
    trial_index: int = 2,
    model_widths: tuple[int, int, int] = (3, 12, 64),
    max_measurements: int | None = None,
    selected_widths: tuple[int, ...] | None = None,
) -> Figure:
    """Figure 9's three width trajectories beside its width-by-time heatmap."""
    frame = load_vision(source, dataset_dir, max_measurements, selected_widths)
    widths, test = metric_matrix(frame, trial_index, "test_error")
    _, train = metric_matrix(frame, trial_index, "train_error")
    times = np.arange(1, test.shape[1] + 1)
    fig, (curves_ax, heatmap_ax) = plt.subplots(
        1, 2, figsize=(17, 6), gridspec_kw={"width_ratios": [1, 1.45]}
    )
    trajectory_widths = model_widths
    if selected_widths is not None:
        if len(widths) < 3:
            raise ValueError("Figure 9 needs at least three selected widths")
        trajectory_widths = tuple(int(widths[np.argmin(abs(widths - target))]) for target in model_widths)
        if len(set(trajectory_widths)) < 3:
            raise ValueError("Selected widths cannot provide three distinct Figure 9 trajectories")
    for width, color in zip(trajectory_widths, ("#b8a12f", "#327d79", "#813c58"), strict=True):
        matches = np.flatnonzero(widths == width)
        if len(matches) != 1:
            raise ValueError(f"Width {width} is absent or duplicated in {source}")
        curves_ax.plot(times, test[matches[0]], color=color, lw=2, label=f"Width {width}")
    curves_ax.set_xscale("log")
    curves_ax.set(xlabel="Recorded time point (log scale)", ylabel="Test error",
                  title="Three model widths")
    curves_ax.set_xlim(1, len(times))
    curves_ax.set_ylim(bottom=0)
    curves_ax.legend(frameon=False)

    indices = log_spaced_indices(test.shape[1], 1.1)
    values = test[:, indices].T
    positive = values[np.isfinite(values) & (values > 0)]
    if positive.size == 0:
        raise ValueError("Log-scaled test-error heatmap needs positive values")
    image = heatmap_ax.imshow(
        values, cmap=load_colormap(),
        norm=colors.LogNorm(vmin=float(positive.min()), vmax=float(positive.max())),
        aspect="auto", interpolation="none",
    )
    heatmap_ax.contour(train[:, indices].T, levels=[0.15, 0.5], colors="white",
                       linestyles="dashed", alpha=0.7)
    tick_values = np.unique(np.clip(np.rint(np.geomspace(1, test.shape[1], num=min(5, test.shape[1]))).astype(int) - 1, 0, test.shape[1] - 1))
    tick_positions = [int(np.argmin(abs(indices - value))) for value in tick_values]
    heatmap_ax.set_yticks(tick_positions, [str(value + 1) for value in tick_values])
    heatmap_ax.invert_yaxis()
    x_positions = np.arange(len(widths)) if len(widths) <= 15 else np.arange(0, len(widths), 15)
    heatmap_ax.set_xticks(x_positions, [str(widths[position]) for position in x_positions])
    heatmap_ax.set(xlabel="ResNet18 width parameter", ylabel="Recorded time point",
                   title="Test error by width and time")
    fig.colorbar(image, ax=heatmap_ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig
