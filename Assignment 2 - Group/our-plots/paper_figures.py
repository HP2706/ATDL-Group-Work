"""Partial comparisons for the main-paper vision figures."""

from __future__ import annotations

from itertools import product
from functools import lru_cache
from pathlib import Path

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FixedLocator, NullLocator, StrMethodFormatter
from vision import load_colormap, noisy_test_error, rescale_viridis


WIDTHS = (2, 3, 4, 6, 8, 12, 16, 24, 32, 64)
NOISE = (0.0, 0.1, 0.2)
EPOCHS = tuple(range(10, 401, 10))
TRIAL_NOISE = {0: 0.0, 1: 0.1, 2: 0.2}
CNN_WIDTHS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 36, 40, 48, 64, 96, 128)
CNN_STEPS = tuple(range(1250, 50001, 1250))
CNN_PAPER_ESTIMATED_STEP_INTERVAL = 256
CNN_COMPARISON_STEPS = 50000
CNN_SOURCES = {
    0.0: "pct-cifar10-mcnn-50000-p0-sgd-big",
    0.1: "cifar10-mcnn-p10-sgd",
    0.2: "pct-cifar10-mcnn-50000-p20-sgd-big",
}
CNN_COLORS = {0.0: "#355d8a", 0.1: "#3f9b8f", 0.2: "#cf7653"}
SUBSET_WIDTHS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 64)


@lru_cache(maxsize=1)
def load_all_our_runs(project_dir: Path) -> pd.DataFrame:
    files = sorted((project_dir / "our-results-folder" / "data" / "vision").glob("*.parquet"))
    if not files:
        raise ValueError("No local vision results; download the UCloud Parquet files first")
    return pd.concat([pq.read_table(path).to_pandas() for path in files], ignore_index=True)


def our_runs(project_dir: Path, dataset: str, architecture: str, sample_size: int,
             augmentation: bool, optimizer: str, widths: tuple[int, ...],
             noise_rates: tuple[float, ...], final_time: int, time_column: str) -> pd.DataFrame:
    frame = load_all_our_runs(project_dir)
    selected = frame.loc[
        frame["dataset"].eq(dataset) & frame["architecture"].eq(architecture)
        & frame["sample_size"].eq(sample_size) & frame["augmentation"].eq(augmentation)
        & frame["optimizer"].eq(optimizer) & frame["model_width"].isin(widths)
        & frame["label_noise"].isin(noise_rates)
    ].copy()
    expected = set(product(noise_rates, widths))
    actual = set(selected[["label_noise", "model_width"]].itertuples(index=False, name=None))
    endpoints = selected.loc[selected[time_column].eq(final_time)]
    endpoint_keys = list(endpoints[["label_noise", "model_width"]].itertuples(index=False, name=None))
    if actual != expected or len(endpoint_keys) != len(expected) or set(endpoint_keys) != expected:
        raise ValueError(f"Incomplete {dataset}/{architecture}/{sample_size}/{optimizer} grid")
    if selected[["train_error", "test_error", "train_loss", "test_loss"]].isna().any().any():
        raise ValueError(f"Missing metrics in {dataset}/{architecture}/{sample_size}/{optimizer}")
    return selected


def load_comparison(project_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return the authors' first 400 displayed positions and our 400-epoch sweep."""
    our_dir = project_dir / "our-results-folder" / "data" / "vision"
    files = sorted(our_dir.glob("cifar10-resnet-k*-seed0-*.parquet"))
    if len(files) != 30:
        raise ValueError(f"Expected 30 sweep files in {our_dir}; found {len(files)}")
    ours = pd.concat([pq.read_table(path).to_pandas() for path in files], ignore_index=True)

    author_dir = project_dir / "their-results" / "hf_dataset" / "data" / "vision"
    paper = pq.read_table(
        author_dir / "cifar10-resnet18k-50k-adam.parquet",
        columns=["trial_index", "model_width", "measurement_index", "train_error", "test_error"],
    ).to_pandas()
    paper = paper.loc[
        paper["trial_index"].isin(TRIAL_NOISE)
        & paper["model_width"].isin(WIDTHS)
        & paper["measurement_index"].isin(tuple(epoch - 1 for epoch in EPOCHS))
    ].copy()
    paper["label_noise"] = paper["trial_index"].map(TRIAL_NOISE)
    paper["epoch"] = paper["measurement_index"] + 1

    expected = set(product(NOISE, WIDTHS, EPOCHS))
    for name, frame in (("Authors", paper), ("Ours", ours)):
        keys = list(frame[["label_noise", "model_width", "epoch"]].itertuples(index=False, name=None))
        if len(keys) != len(expected) or set(keys) != expected:
            raise ValueError(f"{name} does not cover the expected 30 × 40 grid")
        if frame[["train_error", "test_error"]].isna().any().any():
            raise ValueError(f"{name} contains missing error values")
    extra_frames: list[pd.DataFrame] = []
    for noise, source in ((0.05, "pct-cf10-res18-50k-p05-adam"),
                          (0.15, "pct-cifar10-resnet18-50k-p15-adam")):
        extra = pq.read_table(
            author_dir / f"{source}.parquet",
            columns=["trial_index", "model_width", "measurement_index", "train_error", "test_error"],
        ).to_pandas()
        extra = extra.loc[extra["trial_index"].eq(0) & extra["model_width"].isin(WIDTHS)
                          & extra["measurement_index"].eq(399)].copy()
        if set(extra["model_width"]) != set(WIDTHS) or len(extra) != len(WIDTHS):
            raise ValueError(f"Authors' {noise:.0%} endpoint does not cover all selected widths")
        extra["label_noise"] = noise
        extra["epoch"] = 400
        extra_frames.append(extra)
    paper = pd.concat([paper, *extra_frames], ignore_index=True)

    paper_cifar100 = pq.read_table(
        author_dir / "cifar100-resnet18k-50k-adam.parquet",
        columns=["trial_index", "model_width", "measurement_index", "train_error", "test_error"],
    ).to_pandas()
    paper_cifar100 = paper_cifar100.loc[
        paper_cifar100["trial_index"].isin(TRIAL_NOISE)
        & paper_cifar100["model_width"].isin(WIDTHS)
        & paper_cifar100["measurement_index"].eq(399)
    ].copy()
    paper_cifar100["label_noise"] = paper_cifar100["trial_index"].map(TRIAL_NOISE)
    paper_cifar100["epoch"] = 400
    expected_cifar100 = set(product(NOISE, WIDTHS))
    keys_cifar100 = list(paper_cifar100[["label_noise", "model_width"]].itertuples(index=False, name=None))
    if len(keys_cifar100) != len(expected_cifar100) or set(keys_cifar100) != expected_cifar100:
        raise ValueError("Authors' CIFAR-100 endpoint does not cover the 3 × 10 grid")
    return paper, ours, paper_cifar100


def load_cnn_comparison(project_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the full author trajectories and our measured 50,000-step subset."""
    ours = our_runs(project_dir, "cifar10", "cnn", 50000, True, "sgd",
                    CNN_WIDTHS, NOISE, 50000, "global_step")
    expected = set(product(NOISE, CNN_WIDTHS, CNN_STEPS))
    keys = list(ours[["label_noise", "model_width", "global_step"]].itertuples(index=False, name=None))
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("CNN results do not cover the expected 48 × 40 grid")
    if not (ours["augmentation"].all() and ours["optimizer"].eq("sgd").all()
            and ours["schedule"].eq("inverse_sqrt").all() and ours["sample_size"].eq(50000).all()):
        raise ValueError("CNN result protocol differs from the planned augmented SGD sweep")

    return load_cnn_paper(project_dir), ours


def load_cnn_paper(project_dir: Path) -> pd.DataFrame:
    """Read the released CNN histories independently of our result files."""
    author_dir = project_dir / "their-results" / "hf_dataset" / "data" / "vision"
    paper_frames: list[pd.DataFrame] = []
    for noise, source in CNN_SOURCES.items():
        frame = pq.read_table(author_dir / f"{source}.parquet",
                              columns=["trial_index", "model_width", "measurement_index",
                                       "train_error", "test_error"]).to_pandas()
        frame = frame.loc[frame["trial_index"].eq(0)].copy()
        if not set(CNN_WIDTHS).issubset(frame["model_width"].unique()):
            raise ValueError(f"Authors' {noise:.0%} CNN source is missing a selected width")
        frame["label_noise"] = noise
        paper_frames.append(frame)
    paper = pd.concat(paper_frames, ignore_index=True)
    if paper[["train_error", "test_error"]].isna().any().any():
        raise ValueError("Published CNN histories contain missing error values")
    return paper


def load_figure10_paper(project_dir: Path) -> pd.DataFrame:
    """Use the clean full-data source and the explicitly labeled 50k/20% sample grid."""
    author_dir = project_dir / "their-results" / "hf_dataset" / "data" / "vision"
    columns = ["trial_index", "sample_size", "model_width", "measurement_index", "test_error", "train_error"]
    clean = pq.read_table(author_dir / "pct-cifar10-mcnn-50000-p0-sgd-big.parquet",
                          columns=columns, filters=[("model_width", "=", 128), ("trial_index", "=", 0)]).to_pandas()
    noisy = pq.read_table(author_dir / "dd_grid_p20.parquet", columns=columns,
                          filters=[("model_width", "=", 128), ("sample_size", "=", 50000),
                                   ("trial_index", "=", 0)]).to_pandas()
    clean["label_noise"] = 0.0
    noisy["label_noise"] = 0.2
    return pd.concat([clean, noisy], ignore_index=True)


def matrix(frame: pd.DataFrame, noise: float, metric: str) -> np.ndarray:
    selected = frame.loc[frame["label_noise"].eq(noise), ["epoch", "model_width", metric]]
    return selected.pivot(index="epoch", columns="model_width", values=metric).loc[list(EPOCHS), list(WIDTHS)].to_numpy(dtype=float)


def width_axis(ax: Axes) -> None:
    ax.set_xlim(1, 64)
    ax.set_xticks((1, 10, 20, 30, 40, 50, 64))
    ax.set_xlabel("ResNet18 Width Parameter")


def caption(fig: Figure, title: str, detail: str) -> Figure:
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.supxlabel(detail, fontsize=9, color="#444444")
    return fig


def figure_1(paper: pd.DataFrame, ours: pd.DataFrame, ours_only: bool = False,
             authors_only: bool = False) -> Figure:
    """Paper layout: blue endpoint curves and dense epoch-colored width curves."""
    if ours_only and authors_only:
        raise ValueError("Choose one standalone source")
    rows = (("Ours", ours),) if ours_only else (("Authors", paper),) if authors_only else (("Authors", paper), ("Ours", ours))
    fig, axes = plt.subplots(len(rows), 2, figsize=(14, 4 * len(rows)), sharex="col", sharey="col", layout="constrained", squeeze=False)
    palette = rescale_viridis().reversed()
    time_norm = colors.LogNorm(vmin=10, vmax=400)
    for row, (name, frame) in enumerate(rows):
        final = frame.loc[frame["label_noise"].eq(0.1) & frame["epoch"].eq(400)]
        final = final.set_index("model_width").loc[list(WIDTHS)]
        left, right = axes[row]
        endpoint_test = noisy_test_error(final["test_error"].to_numpy(dtype=float), 0.1, 10)
        left.plot(WIDTHS, endpoint_test, "-", color="blue", lw=2.3, label="Test")
        left.plot(WIDTHS, final["train_error"], "--", color="blue", alpha=0.45, lw=2.0, label="Train")
        left.set(title=f"{name} · final recorded point", ylabel="Test / Train Error", ylim=(0, 0.55))
        left.legend(frameon=False, loc="upper right")
        width_axis(left)

        values = matrix(frame, 0.1, "test_error")
        noisy_values = noisy_test_error(values, 0.1, 10)
        for epoch, errors in zip(EPOCHS, noisy_values, strict=True):
            right.plot(WIDTHS, errors, color=palette(time_norm(epoch)), alpha=0.35, lw=1.0)
        right.plot(WIDTHS, noisy_values.min(axis=0), "r--", lw=1.8, label="Minimum observed\ntest error")
        right.set(title=f"{name} · varying training time", ylabel="Test Error", ylim=(0.2, 0.72))
        right.legend(frameon=False, loc="upper right")
        width_axis(right)
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=time_norm, cmap=palette), ax=axes[:, 1], shrink=0.8)
    bar.set_ticks((10, 100, 400), labels=("10", "100", "400"))
    bar.ax.invert_yaxis()
    bar.set_label("Our epoch" if ours_only else "Authors' displayed epoch" if authors_only else "Authors' displayed epoch / our epoch")
    return caption(fig, "Figure 1 · CIFAR-10 ResNet18 · 10% label noise",
                   "Our ten widths, one seed, through epoch 400." if ours_only else
                   "Released 10% run: same ten widths and displayed epochs 10–400; test error transformed to 10% noisy labels." if authors_only else
                   "Both: 10% label noise, same ten widths and displayed epochs 10–400. One run per width; test error transformed to 10% noisy labels.")


def figure_2(paper: pd.DataFrame, ours: pd.DataFrame, ours_only: bool = False) -> Figure:
    """Paper layout: test and train heatmaps, log time, custom inferno palette."""
    rows = (("Ours", ours),) if ours_only else (("Authors", paper), ("Ours", ours))
    fig, axes = plt.subplots(len(rows), 2, figsize=(14, 4 * len(rows)), sharex=True, sharey=True, layout="constrained", squeeze=False)
    y_edges = np.r_[EPOCHS[0] / np.sqrt(EPOCHS[1] / EPOCHS[0]),
                    np.sqrt(np.array(EPOCHS[:-1]) * np.array(EPOCHS[1:])),
                    EPOCHS[-1] * np.sqrt(EPOCHS[-1] / EPOCHS[-2])]
    for column, metric in enumerate(("test_error", "train_error")):
        norm: colors.Normalize = colors.LogNorm(vmin=0.1, vmax=0.8) if column == 0 else colors.Normalize(vmin=0, vmax=0.8)
        for row, (name, frame) in enumerate(rows):
            ax = axes[row, column]
            values = matrix(frame, 0.1, metric)
            image = ax.pcolormesh(np.arange(len(WIDTHS) + 1), y_edges, values, cmap=load_colormap(), norm=norm, shading="flat")
            if column == 1:
                ax.contour(np.arange(len(WIDTHS)) + 0.5, EPOCHS, matrix(frame, 0.1, "train_error"),
                           levels=(0.15,), colors="white", linestyles="dashed", linewidths=1.0)
            ax.set_yscale("log")
            ax.set_ylim(10, 400)
            ax.set_yticks((10, 100, 400), labels=("10", "100", "400"))
            ax.set_xticks(np.arange(len(WIDTHS)) + 0.5, labels=WIDTHS)
            ax.set(title=f"{name} · {metric.replace('_', ' ').title()}", xlabel="ResNet18 Width Parameter")
            if column == 0:
                ax.set_ylabel("Our epoch" if ours_only else "Authors' recorded point / our epoch")
        bar = fig.colorbar(image, ax=axes[:, column], shrink=0.8)
        ticks = (0.1, 0.2, 0.3, 0.4, 0.6, 0.8) if column == 0 else (0, 0.2, 0.4, 0.6, 0.8)
        bar.set_ticks(ticks, labels=[f"{tick:.1f}" for tick in ticks])
        bar.set_label("Error fraction")
    return caption(fig, "Figure 2 · CIFAR-10 ResNet18 · 10% label noise",
                   "Our ten widths through epoch 400, using the published colormap and log-time layout." if ours_only else
                   "Paper colormap and log-time layout, adapted from 15% to 10% noise; sparse widths and our 400-epoch horizon.")


def plot_noise_curves(ax: Axes, frame: pd.DataFrame, metric: str) -> None:
    final = frame.loc[frame["epoch"].eq(400)]
    palette = {0.0: "#363d52", 0.05: "#66758f", 0.1: "#668e8b", 0.15: "#b79c84", 0.2: "#b4be8c"}
    for noise in sorted(final["label_noise"].unique(), reverse=True):
        selected = final.loc[final["label_noise"].eq(noise)].set_index("model_width").loc[list(WIDTHS)]
        ax.plot(WIDTHS, selected[metric], "--" if metric == "train_error" else "-",
                color=palette[noise], lw=1.5, label=f"{noise:.0%} label noise")
    ax.set_ylabel("Train Error" if metric == "train_error" else "Test Error")
    width_axis(ax)
    if metric == "test_error":
        ax.legend(frameon=False, fontsize=8, loc="upper right")


def figure_4(paper: pd.DataFrame, ours: pd.DataFrame, paper_cifar100: pd.DataFrame,
             ours_cifar100: pd.DataFrame, ours_only: bool = False) -> Figure:
    """Ours above authors, two dataset columns with separate test/train panels."""
    row_names = ("Ours",) if ours_only else ("Ours", "Authors")
    fig = plt.figure(figsize=(18, 4 * len(row_names)), layout="constrained")
    grid = fig.add_gridspec(len(row_names), 2)
    for row, name in enumerate(row_names):
        for column, dataset in enumerate(("CIFAR-100", "CIFAR-10")):
            pair = grid[row, column].subgridspec(1, 2, wspace=0.18)
            test_ax = fig.add_subplot(pair[0, 0])
            train_ax = fig.add_subplot(pair[0, 1], sharex=test_ax, sharey=test_ax)
            frame = (ours_cifar100 if column == 0 else ours) if row == 0 else (paper_cifar100 if column == 0 else paper)
            plot_noise_curves(test_ax, frame, "test_error")
            plot_noise_curves(train_ax, frame, "train_error")
            test_ax.set_title(f"{name} · {dataset} · Test")
            train_ax.set_title(f"{name} · {dataset} · Train")
            test_ax.set_ylim(0, 0.85 if column == 0 else 0.55)
    return caption(fig, "Figure 4 · ResNet18 model-wise double descent",
                   "Our CIFAR-10 and CIFAR-100 runs at 0/10/20% noise, ten widths, epoch 400." if ours_only else
                   "Separate test and train panels. Authors' CIFAR-10 has 0/5/10/15/20% noise, ours has 0/10/20%, at point/epoch 400.")


def figure_9(paper: pd.DataFrame, ours: pd.DataFrame, ours_only: bool = False) -> Figure:
    """Paper's log-time trajectories and custom-inferno width-by-time heatmap."""
    rows = (("Ours", ours),) if ours_only else (("Authors", paper), ("Ours", ours))
    fig, axes = plt.subplots(len(rows), 2, figsize=(13.5, 4 * len(rows)), sharey="col", layout="constrained", squeeze=False)
    all_values = [matrix(frame, 0.2, "test_error") for _, frame in rows]
    norm = colors.LogNorm(vmin=0.15, vmax=0.8)
    y_edges = np.r_[EPOCHS[0] / np.sqrt(EPOCHS[1] / EPOCHS[0]),
                    np.sqrt(np.array(EPOCHS[:-1]) * np.array(EPOCHS[1:])),
                    EPOCHS[-1] * np.sqrt(EPOCHS[-1] / EPOCHS[-2])]
    for row, ((name, _), values) in enumerate(zip(rows, all_values, strict=True)):
        left, right = axes[row]
        for width, color in ((3, "#ba9a35"), (12, "#2e807b"), (64, "#874e76")):
            left.plot(EPOCHS, values[:, WIDTHS.index(width)], color=color, lw=2, label=f"Width {width}")
        left.set(title=f"{name} · three width regimes", xlabel="Our epoch" if ours_only else "Authors' recorded point / our epoch",
                 ylabel="Test Error", xlim=(10, 400), ylim=(0.1, 0.7))
        left.set_xscale("log")
        left.legend(frameon=False)
        image = right.pcolormesh(np.arange(len(WIDTHS) + 1), y_edges, values,
                                 cmap=load_colormap(), norm=norm, shading="flat")
        right.set_yscale("log")
        right.set(title=f"{name} · width × time", xlabel="ResNet18 Width Parameter",
                  ylabel="Our epoch" if ours_only else "Authors' recorded point / our epoch", ylim=(10, 400))
        right.set_xticks(np.arange(len(WIDTHS)) + 0.5, labels=WIDTHS)
        right.set_yticks((10, 100, 400), labels=("10", "100", "400"))
    bar = fig.colorbar(image, ax=axes[:, 1], shrink=0.8)
    bar.set_ticks((0.2, 0.3, 0.4, 0.6, 0.8), labels=("0.2", "0.3", "0.4", "0.6", "0.8"))
    bar.set_label("Test Error")
    return caption(fig, "Figure 9 · CIFAR-10 ResNet18 · 20% label noise",
                   "Our widths 3/12/64 and ten-width heatmap through epoch 400." if ours_only else
                   "Paper's log-time and color layout; widths 3/12/64 and ten-width heatmap, through our epoch 400.")


def figure_5_cnn(paper: pd.DataFrame, ours: pd.DataFrame,
                 paper_noaug: pd.DataFrame, ours_noaug: pd.DataFrame,
                 ours_only: bool = False) -> Figure:
    """Augmented and non-augmented CNN width curves near 50,000 steps."""
    rows = (("Ours · augmentation · 50k steps", ours, "global_step"),
            ("Ours · no augmentation · 50k steps", ours_noaug, "global_step")) if ours_only else (
            ("Authors · augmentation · estimated ≈50k steps", paper, "measurement_index"),
            ("Ours · augmentation · 50k steps", ours, "global_step"),
            ("Authors · no augmentation · estimated ≈50k steps", paper_noaug, "measurement_index"),
            ("Ours · no augmentation · 50k steps", ours_noaug, "global_step"))
    fig, axes = plt.subplots(len(rows), 2, figsize=(14, 3.5 * len(rows)), sharex=True, sharey="col", layout="constrained", squeeze=False)
    for row, (name, full_frame, time_column) in enumerate(rows):
        frame = (full_frame.loc[((full_frame["measurement_index"] + 1)
                                 * CNN_PAPER_ESTIMATED_STEP_INTERVAL).le(CNN_COMPARISON_STEPS)]
                 if time_column == "measurement_index" else full_frame)
        for column, metric in enumerate(("test_error", "train_error")):
            ax = axes[row, column]
            for noise in NOISE:
                selected = frame.loc[frame["label_noise"].eq(noise) & frame["model_width"].le(64)]
                endpoints = selected.loc[selected[time_column].eq(
                    selected.groupby("model_width")[time_column].transform("max"))]
                endpoints = endpoints.sort_values("model_width")
                ax.plot(endpoints["model_width"], endpoints[metric],
                        "--" if metric == "train_error" else "-", lw=2,
                        color=CNN_COLORS[noise], label=f"{noise:.0%} noise")
            ax.set(title=f"{name} · {'Test' if column == 0 else 'Train'} error",
                   xlabel="CNN width", ylabel="Error fraction", xlim=(1, 64), ylim=(0, 0.85))
            ax.set_xticks((1, 16, 32, 48, 64))
            if column == 0:
                ax.legend(frameon=False)
    return caption(fig, "Figure 5 subset · CIFAR-10 CNN augmentation",
                   "Our augmented and non-augmented one-seed runs through 50,000 steps." if ours_only else
                   "Paper index 194 ≈ 49,920 steps (256 steps/point inferred, unverified); ours exactly 50,000 steps. Our non-augmented grid has 11 widths.")


def figure_10_cnn(
    paper: pd.DataFrame, ours: pd.DataFrame | None = None,
    ours_only: bool = False, authors_only: bool = False,
) -> Figure:
    """Plot width-128 histories on their recorded coordinates without time inference."""
    if ours_only and authors_only:
        raise ValueError("Choose one standalone source")
    author_noise = (0.0, 0.2)
    authors = paper.loc[paper["label_noise"].isin(author_noise)
                        & paper["model_width"].eq(128)].copy()
    if not authors_only:
        if ours is None:
            raise ValueError("Our result histories are required for an ours/comparison chart")
        measured = ours.loc[ours["model_width"].eq(128)].copy()
        expected_ours = set(product(NOISE, CNN_STEPS))
        actual_ours = list(measured[["label_noise", "global_step"]].itertuples(index=False, name=None))
        if len(actual_ours) != len(expected_ours) or set(actual_ours) != expected_ours:
            raise ValueError("Figure 10 needs one width-128 measured history at each noise level")
    expected_authors = {(0.0, index) for index in range(1952)} | {(0.2, index) for index in range(976)}
    actual_authors = list(authors[["label_noise", "measurement_index"]].itertuples(index=False, name=None))
    if len(actual_authors) != len(expected_authors) or set(actual_authors) != expected_authors:
        raise ValueError("Figure 10 needs the complete two full-data author histories")
    if authors_only:
        sources = (("Authors", authors, "measurement_index", author_noise),)
    elif ours_only:
        sources = (("Ours", measured, "global_step", NOISE),)
    else:
        sources = (("Authors", authors, "measurement_index", author_noise),
                   ("Ours", measured, "global_step", author_noise))
    fig, axes = plt.subplots(2, len(sources), figsize=(7 * len(sources), 8),
                             sharex="col", sharey="row", layout="constrained", squeeze=False)
    for column, (name, frame, time_column, levels) in enumerate(sources):
        author_column = time_column == "measurement_index"
        for row, metric in enumerate(("test_error", "train_error")):
            ax = axes[row, column]
            for noise in levels:
                selected = frame.loc[frame["label_noise"].eq(noise)].sort_values(time_column)
                recorded = selected[time_column].to_numpy(dtype=float)
                x = recorded + 1 if author_column else recorded / 1000
                ax.plot(x, selected[metric], "--" if row else "-",
                        lw=1.7, color=CNN_COLORS[noise], label=f"{noise:.0%} label noise",
                        marker=None if author_column else "o", markersize=3)
            ax.set(title=f"{name}: {'test' if row == 0 else 'train'} error",
                   ylabel="Error fraction", ylim=(0, 0.85),
                   xlim=(1, 1952) if author_column else (1.25, 50))
            ax.set_xscale("log")
            ticks = (1, 10, 100, 1000) if author_column else (2, 5, 10, 20, 50)
            ax.xaxis.set_major_locator(FixedLocator(ticks))
            ax.xaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
            ax.xaxis.set_minor_locator(NullLocator())
            ax.tick_params(labelbottom=True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if row == 0:
                ax.legend(frameon=False, loc="upper right")
            if row == 1:
                ax.set_xlabel("Saved measurement number (log scale)" if author_column
                              else "Optimizer steps (thousands, log scale)")
    detail = (
        "Full released histories; logging intervals unavailable."
        if authors_only else
        "One seed; 40 recorded evaluations per noise setting."
        if ours_only else
        "Common noise settings; independent time axes and horizons. Raw measurements, no smoothing."
    )
    return caption(fig, "CIFAR-10 CNN, width 128 · Figure 10(c) reference", detail)


def load_paper_noaug(project_dir: Path) -> pd.DataFrame:
    path = (project_dir / "their-results" / "hf_dataset" / "data" / "vision"
            / "cifar10-mcnn-noaug-sgd.parquet")
    frame = pq.read_table(path, columns=["trial_index", "model_width", "measurement_index",
                                         "train_error", "test_error"]).to_pandas()
    frame["label_noise"] = frame["trial_index"].map(TRIAL_NOISE)
    return frame


def figure_6_cnn(project_dir: Path, ours_sgd: pd.DataFrame,
                 ours_adam: pd.DataFrame, ours_only: bool = False) -> Figure:
    """Compare clean non-augmented SGD and Adam at both recorded horizons."""
    author_dir = project_dir / "their-results" / "hf_dataset" / "data" / "vision"
    rows = (("Ours · reduced horizons",),) if ours_only else (("Authors · full published horizons",), ("Ours · reduced horizons",))
    fig, axes = plt.subplots(len(rows), 2, figsize=(12, 4 * len(rows)), sharex=True, sharey="col", layout="constrained", squeeze=False)
    for column, metric in enumerate(("test_error", "train_error")):
        for name, color, ours, time_column, final_time in (
            ("sgd", "#355d8a", ours_sgd, "global_step", 50000),
            ("adam", "#cf7653", ours_adam, "epoch", 400),
        ):
            if not ours_only:
                paper = pq.read_table(author_dir / f"cifar10-mcnn-noaug-{name}.parquet",
                                      columns=["trial_index", "model_width", "measurement_index", metric]).to_pandas()
                paper = paper.loc[paper["trial_index"].eq(0)]
                paper = paper.loc[paper["measurement_index"].eq(paper.groupby("model_width")["measurement_index"].transform("max"))]
                axes[0, column].plot(paper["model_width"], paper[metric], color=color, label=name.upper())
            ours_final = ours.loc[ours[time_column].eq(final_time)
                                  & ours["label_noise"].eq(0)].sort_values("model_width")
            axes[-1, column].plot(ours_final["model_width"], ours_final[metric], color=color, label=name.upper())
        for row, (title,) in enumerate(rows):
            axes[row, column].set(title=f"{title} · {'Test' if column == 0 else 'Train'}",
                                  xlabel="CNN width", ylabel="Error fraction", xlim=(1, 64), ylim=(0, 0.85))
            axes[row, column].legend(frameon=False)
    return caption(fig, "Figure 6 subset · clean CIFAR-10 CNN without augmentation",
                   "Our SGD endpoint at 50,000 steps and Adam endpoint at 400 epochs; 11 widths, one seed." if ours_only else
                   "SGD: authors' final point versus our 50,000 steps. Adam: authors' final point versus our 400 epochs. Our grid has 11 widths and one seed.")


def figure_7_cnn(project_dir: Path, ours: pd.DataFrame, ours_only: bool = False) -> Figure:
    author_dir = project_dir / "their-results" / "hf_dataset" / "data" / "vision"
    paper = pq.read_table(author_dir / "pct-cifar100-mcnn-p0-sgd-noaug-reps.parquet",
                          columns=["trial_index", "model_width", "measurement_index",
                                   "test_error", "train_error"]).to_pandas()
    paper = paper.loc[paper["model_width"].isin(SUBSET_WIDTHS)]
    paper = paper.loc[paper["measurement_index"].eq(paper.groupby(["trial_index", "model_width"])
                       ["measurement_index"].transform("max"))]
    final = ours.loc[ours["global_step"].eq(50000)].sort_values("model_width")
    rows = ("Ours · 50,000 SGD steps",) if ours_only else ("Authors · final recorded point", "Ours · 50,000 SGD steps")
    fig, axes = plt.subplots(len(rows), 2, figsize=(12, 4 * len(rows)), sharex=True, sharey="col", layout="constrained", squeeze=False)
    for column, metric in enumerate(("test_error", "train_error")):
        if not ours_only:
            summary = paper.groupby("model_width")[metric].agg(["mean", "std"]).loc[list(SUBSET_WIDTHS)]
            axes[0, column].plot(summary.index, summary["mean"], color="#cf7653", lw=2, label="Five-trial mean")
            axes[0, column].fill_between(summary.index.to_numpy(dtype=float),
                                         (summary["mean"] - summary["std"]).to_numpy(dtype=float),
                                         (summary["mean"] + summary["std"]).to_numpy(dtype=float),
                                         color="#cf7653", alpha=0.2, label="±1 SD")
            axes[0, column].legend(frameon=False)
        axes[-1, column].plot(final["model_width"], final[metric], color="#355d8a", lw=2, label="One trial")
        for row, title in enumerate(rows):
            axes[row, column].set(title=f"{title} · {'Test' if column == 0 else 'Train'}",
                                  xlabel="CNN width", ylabel="Error fraction", xlim=(1, 64), ylim=(0, 1))
    return caption(fig, "Figure 7 subset · clean CIFAR-100 CNN",
                   "Our one-seed, 11-width grid at 50,000 SGD steps." if ours_only else
                   "Matched 11 widths; authors show five-trial mean ± SD at their final recorded point, ours one trial at 50,000 of 1,000,000 steps.")


def paper_sample_grid(project_dir: Path) -> pd.DataFrame:
    path = project_dir / "their-results" / "hf_dataset" / "data" / "vision" / "dd_grid_p20.parquet"
    frame = pq.read_table(path, columns=["sample_size", "model_width", "measurement_index",
                                         "test_error", "train_error"]).to_pandas()
    frame = frame.loc[frame["sample_size"].isin((12500, 25000, 50000))
                      & frame["model_width"].isin(SUBSET_WIDTHS)]
    final = frame.loc[frame["measurement_index"].eq(frame.groupby(["sample_size", "model_width"])
                     ["measurement_index"].transform("max"))]
    if len(final) != 3 * len(SUBSET_WIDTHS):
        raise ValueError("Published sample grid lacks matching 3 × 11 endpoints")
    return final


def figure_11a_cnn(project_dir: Path, subsets: pd.DataFrame, ours_only: bool = False) -> Figure:
    final = subsets.loc[subsets["global_step"].eq(50000)]
    palette = {12500: "#355d8a", 25000: "#3f9b8f", 50000: "#cf7653"}
    rows = (("Ours · 20% noise · 50,000 steps", final, 0.2),
            ("Ours · 10% noise · 50,000 steps", final, 0.1)) if ours_only else (
            ("Authors · 20% noise · final recorded point", paper_sample_grid(project_dir), None),
            ("Ours · 20% noise · 50,000 steps", final, 0.2),
            ("Ours · 10% noise · 50,000 steps", final, 0.1))
    fig, axes = plt.subplots(len(rows), 2, figsize=(12, 3.7 * len(rows)), sharex=True, sharey="col", layout="constrained", squeeze=False)
    for row, (title, frame, noise) in enumerate(rows):
        for column, metric in enumerate(("test_error", "train_error")):
            ax = axes[row, column]
            for size, color in palette.items():
                selected = frame.loc[frame["sample_size"].eq(size)]
                if noise is not None:
                    selected = selected.loc[selected["label_noise"].eq(noise)]
                selected = selected.sort_values("model_width")
                ax.plot(selected["model_width"], selected[metric], color=color, lw=2,
                        label=f"{size:,} examples")
            ax.set(title=f"{title} · {'Test' if column == 0 else 'Train'}",
                   xlabel="CNN width", ylabel="Error fraction", xlim=(1, 64), ylim=(0, 0.85))
            if column == 0:
                ax.legend(frameon=False)
    return caption(fig, "Figure 11(a) subset · CIFAR-10 CNN sample sizes",
                   "Our 10% and 20% noise grids: three sample sizes, 11 widths, one seed, 50,000 steps." if ours_only else
                   "20% rows compare the same 3 sample sizes × 11 widths at different horizons. Published 10% subset data were not released; our 10% row is shown separately.")


def figure_12_cnn(project_dir: Path, subsets: pd.DataFrame, ours_only: bool = False) -> Figure:
    ours = subsets.loc[subsets["global_step"].eq(50000) & subsets["label_noise"].eq(0.2)]
    frames = (("Ours · 50,000 steps", ours),) if ours_only else (
        ("Authors · final recorded point", paper_sample_grid(project_dir)),
        ("Ours · 50,000 steps", ours))
    reference_frames = (("Authors", paper_sample_grid(project_dir)), ("Ours", ours))
    reference_grids = [frame.pivot(index="sample_size", columns="model_width", values="test_error")
                       .loc[[12500, 25000, 50000], list(SUBSET_WIDTHS)] for _, frame in reference_frames]
    grids = reference_grids[1:] if ours_only else reference_grids
    if any(grid.isna().any().any() for grid in grids):
        raise ValueError("Figure 12 comparison needs two complete 3 × 11 endpoint grids")
    low = min(float(grid.min().min()) for grid in reference_grids)
    high = max(float(grid.max().max()) for grid in reference_grids)
    fig, axes = plt.subplots(len(frames), 2, figsize=(13, 4 * len(frames)), sharex="col", sharey="col", layout="constrained", squeeze=False)
    for row, ((title, _), grid) in enumerate(zip(frames, grids, strict=True)):
        heatmap, curves = axes[row]
        image = heatmap.imshow(grid.to_numpy(), aspect="auto", origin="lower", cmap="viridis", vmin=low, vmax=high)
        heatmap.set_xticks(range(len(grid.columns)), labels=grid.columns)
        heatmap.set_yticks(range(len(grid.index)), labels=[f"{size:,}" for size in grid.index])
        heatmap.set(xlabel="CNN width", ylabel="Training examples", title=f"{title} · heatmap")
        for size, values in grid.iterrows():
            curves.plot(grid.columns, values, lw=2, label=f"{size:,} examples")
        curves.set(xlabel="CNN width", ylabel="Test error fraction", title=f"{title} · slices",
                   xlim=(1, 64), ylim=(0, 1))
        curves.legend(frameon=False)
    fig.colorbar(image, ax=axes[:, 0], label="Test error fraction", shrink=0.75)
    return caption(fig, "Figure 12 subset · CIFAR-10 CNN, 20% label noise",
                   "Our three sample sizes × 11 widths at 50,000 steps; color scale matches the paired comparison." if ours_only else
                   "Same 3 sample sizes × 11 widths and color scale; authors' final recorded point versus our one-seed 50,000-step endpoint.")


def save_figures(project_dir: Path, output_dir: Path) -> dict[int, Path]:
    """Save matched comparison and ours-only PNGs under one plot root."""
    paper, ours, paper_cifar100 = load_comparison(project_dir)
    ours_cifar100 = our_runs(project_dir, "cifar100", "resnet", 50000, True, "adam",
                             WIDTHS, NOISE, 400, "epoch")
    comparison_dir = output_dir / "comparison"
    ours_dir = output_dir / "ours"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    ours_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[int, Path] = {}
    for number, render in ((1, figure_1), (2, figure_2), (4, figure_4), (9, figure_9)):
        for ours_only, directory in ((False, comparison_dir), (True, ours_dir)):
            fig = (figure_4(paper, ours, paper_cifar100, ours_cifar100, ours_only)
                   if number == 4 else render(paper, ours, ours_only))
            path = directory / f"figure_{number:02d}.png"
            fig.savefig(path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            if not ours_only:
                paths[number] = path
    cnn_paper, cnn_ours = load_cnn_comparison(project_dir)
    cnn_noaug = our_runs(project_dir, "cifar10", "cnn", 50000, False, "sgd",
                         SUBSET_WIDTHS, NOISE, 50000, "global_step")
    cnn_adam_noaug = our_runs(project_dir, "cifar10", "cnn", 50000, False, "adam",
                              SUBSET_WIDTHS, NOISE, 400, "epoch")
    paper_noaug = load_paper_noaug(project_dir)
    cifar100_cnn = our_runs(project_dir, "cifar100", "cnn", 50000, False, "sgd",
                            SUBSET_WIDTHS, (0.0,), 50000, "global_step")
    subsets = pd.concat([
        our_runs(project_dir, "cifar10", "cnn", size, True, "sgd",
                 SUBSET_WIDTHS, (0.1, 0.2), 50000, "global_step")
        for size in (12500, 25000, 50000)
    ], ignore_index=True)
    figures = ((5, figure_5_cnn, (cnn_paper, cnn_ours, paper_noaug, cnn_noaug)),
               (6, figure_6_cnn, (project_dir, cnn_noaug, cnn_adam_noaug)),
               (7, figure_7_cnn, (project_dir, cifar100_cnn)),
               (10, figure_10_cnn, (load_figure10_paper(project_dir), cnn_ours)),
               (11, figure_11a_cnn, (project_dir, subsets)),
               (12, figure_12_cnn, (project_dir, subsets)))
    for number, render, arguments in figures:
        name = "figure_11a.png" if number == 11 else f"figure_{number:02d}.png"
        for ours_only, directory in ((False, comparison_dir), (True, ours_dir)):
            fig = render(*arguments, ours_only=ours_only)
            path = directory / name
            fig.savefig(path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            if not ours_only:
                paths[number] = path
    return paths
    return paths
