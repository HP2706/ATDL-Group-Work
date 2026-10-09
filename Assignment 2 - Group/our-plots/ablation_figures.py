"""Ablation X1–X4 figures in the layouts of the paper's appendix figures (9, 16–18, 21, 28)."""

from __future__ import annotations

from itertools import product
from pathlib import Path

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from torch.nn import functional as F
from torchvision.datasets import CIFAR10

from paper_figures import CNN_STEPS, EPOCHS, SUBSET_WIDTHS, caption, our_runs, style_figure
from vision import rescale_viridis


NOISE = 0.2
X1_RESNET_WIDTHS = (3, 8, 12, 16, 64)
X1_CNN_WIDTHS = (16, 32, 48, 64, 128)
X1_EPOCHS = tuple(range(10, 1201, 10))
X1_STEPS = tuple(range(1250, 200001, 1250))
X2_WIDTHS = (4, 8, 12, 16, 24, 64)
X2_SEEDS = (0, 1, 2)
X3_SIZES = (12500, 50000)
X4_WIDTHS = (12, 64)
X4_BASELINE = "Adam, LR 10⁻⁴, constant (main sweep)"
X4_ARMS = {("adam", "constant"): "Adam, LR 10⁻³, constant",
           ("sgd", "dynamic_drop"): "SGD + momentum 0.9, LR 0.01, dynamic drop"}
X4_COLORS = {X4_BASELINE: "#e3a92b", "Adam, LR 10⁻³, constant": "#2a7f80",
             "SGD + momentum 0.9, LR 0.01, dynamic drop": "#a3324f"}
WIDTH_COLORS = ("#ba9a35", "#6f9f3e", "#2e807b", "#3f5f9e", "#874e76")
SIZE_COLORS = {12500: "#355d8a", 50000: "#cf7653"}
DECAY_COLORS = {0.0: "#5b3a8c", 5e-4: "#3aa655"}
METRICS = ["train_error", "test_error", "train_loss", "test_loss"]


def load_ablation(project_dir: Path, name: str) -> pd.DataFrame:
    """Read one ablation's Parquet files from our-results-folder/ablations/<name>/."""
    folder = project_dir / "our-results-folder" / "ablations" / name / "data" / "vision"
    files = sorted(folder.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No {name} results in {folder}; see UCLOUD_ABLATIONS.md")
    frame = pd.concat([pq.read_table(path).to_pandas() for path in files], ignore_index=True)
    if frame[METRICS].isna().any().any():
        raise ValueError(f"Missing metrics in {name}")
    return frame


def check_protocol(frame: pd.DataFrame, label: str, **expected: object) -> None:
    for column, value in expected.items():
        if not frame[column].eq(value).all():
            raise ValueError(f"{label}: {column} is not {value} for every row")


def check_grid(frame: pd.DataFrame, keys: list[str], expected: set[tuple], label: str) -> None:
    actual = list(frame[keys].itertuples(index=False, name=None))
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError(f"{label} does not cover the expected {len(expected)}-point grid")


def load_x1(project_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return the X1 ResNet and CNN runs and the matching main-sweep runs."""
    runs = load_ablation(project_dir, "x1-horizon")
    check_protocol(runs, "X1", dataset="cifar10", label_noise=NOISE, augmentation=True, weight_decay=0.0, seed=0)
    resnet = runs.loc[runs["architecture"].eq("resnet")]
    cnn = runs.loc[runs["architecture"].eq("cnn")]
    check_protocol(resnet, "X1 ResNet", optimizer="adam", schedule="constant")
    check_protocol(cnn, "X1 CNN", optimizer="sgd", schedule="inverse_sqrt")
    check_grid(resnet, ["model_width", "epoch"], set(product(X1_RESNET_WIDTHS, X1_EPOCHS)), "X1 ResNet")
    check_grid(cnn, ["model_width", "global_step"], set(product(X1_CNN_WIDTHS, X1_STEPS)), "X1 CNN")
    base_resnet = our_runs(project_dir, "cifar10", "resnet", 50000, True, "adam",
                           X1_RESNET_WIDTHS, (NOISE,), 400, "epoch")
    base_cnn = our_runs(project_dir, "cifar10", "cnn", 50000, True, "sgd",
                        X1_CNN_WIDTHS, (NOISE,), 50000, "global_step")
    return resnet, cnn, base_resnet, base_cnn


def x1_endpoints(resnet: pd.DataFrame, cnn: pd.DataFrame) -> pd.DataFrame:
    """Test error at the main-sweep horizon, the extended horizon, and the best observed point."""
    rows = []
    for name, frame, time, horizons in (("ResNet18", resnet, "epoch", (400, 1200)),
                                        ("CNN", cnn, "global_step", (50000, 200000))):
        for width, run in frame.groupby("model_width"):
            final = run.set_index(time)["test_error"]
            rows.append({"model": name, "model_width": width, "time_unit": time,
                         "test_at_main_horizon": final[horizons[0]], "test_at_extended_horizon": final[horizons[1]],
                         "minimum_test": final.min(), "time_of_minimum": final.idxmin()})
    return pd.DataFrame(rows)


def cifar10_test_labels(project_dir: Path) -> torch.Tensor:
    root = project_dir / "code" / "data"
    try:
        return torch.tensor(CIFAR10(root=root, train=False, download=False).targets)
    except RuntimeError as error:
        raise FileNotFoundError(
            f"CIFAR-10 is not in {root}. From 'Assignment 2 - Group/code' run: ../../.venv/bin/python -c "
            "\"from pathlib import Path; from sweep import prepare_cifar; prepare_cifar('cifar10', Path('data'))\""
        ) from error


def load_x2(project_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return per-width single-model and plurality-vote ensemble errors, and the main sweep's seed 0."""
    runs = load_ablation(project_dir, "x2-seeds")
    check_protocol(runs, "X2", dataset="cifar10", architecture="resnet", optimizer="adam", schedule="constant",
                   label_noise=NOISE, augmentation=True, weight_decay=0.0)
    check_grid(runs, ["seed", "model_width", "epoch"], set(product(X2_SEEDS, X2_WIDTHS, EPOCHS)), "X2")
    final = runs.loc[runs["epoch"].eq(400)].set_index(["seed", "model_width"])
    labels = cifar10_test_labels(project_dir)
    folder = project_dir / "our-results-folder" / "ablations" / "x2-seeds" / "predictions"
    rows = []
    for width in X2_WIDTHS:
        predictions = []
        for seed in X2_SEEDS:
            saved = torch.load(folder / f"seed{seed}" / f"width-{width}" / "test_predictions.pt", weights_only=True)
            if not torch.equal(saved["indices"], torch.arange(len(labels))):
                raise ValueError(f"Seed {seed}, width {width}: predictions are not in test-set order")
            error = int((saved["predictions"] != labels).sum()) / len(labels)
            if abs(error - final.loc[(seed, width), "test_error"]) > 1e-6:
                raise ValueError(f"Seed {seed}, width {width}: predictions do not reproduce the recorded test error")
            predictions.append(saved["predictions"])
        votes = F.one_hot(torch.stack(predictions), num_classes=10).sum(dim=0)
        ensemble = votes.argmax(dim=1)  # ties (all models disagree) go to the lowest class index
        single = final.xs(width, level="model_width")
        rows.append({
            "model_width": width,
            "single_mean": single["test_error"].mean(), "single_std": single["test_error"].std(ddof=0),
            **{f"seed_{seed}": single.loc[seed, "test_error"] for seed in X2_SEEDS},
            "ensemble": int((ensemble != labels).sum()) / len(labels),
            "train_mean": single["train_error"].mean(),
            "three_way_ties": float((votes.max(dim=1).values == 1).float().mean()),
        })
    summary = pd.DataFrame(rows)
    summary["ensemble_gain"] = summary["single_mean"] - summary["ensemble"]
    baseline = our_runs(project_dir, "cifar10", "resnet", 50000, True, "adam", X2_WIDTHS, (NOISE,), 400, "epoch")
    return summary, baseline.loc[baseline["epoch"].eq(400)].sort_values("model_width")


def load_x3(project_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the weight-decay runs and the matching weight-decay-free main-sweep runs."""
    runs = load_ablation(project_dir, "x3-weight-decay")
    check_protocol(runs, "X3", dataset="cifar10", architecture="cnn", optimizer="sgd", schedule="inverse_sqrt",
                   label_noise=NOISE, augmentation=True, weight_decay=5e-4, seed=0)
    check_grid(runs, ["sample_size", "model_width", "global_step"],
               set(product(X3_SIZES, SUBSET_WIDTHS, CNN_STEPS)), "X3")
    baseline = pd.concat([our_runs(project_dir, "cifar10", "cnn", size, True, "sgd", SUBSET_WIDTHS,
                                   (NOISE,), 50000, "global_step") for size in X3_SIZES], ignore_index=True)
    check_protocol(baseline, "X3 baseline", weight_decay=0.0, seed=0)
    return runs, baseline


def x3_endpoints(runs: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    """Endpoint errors per weight decay, sample size and width, with the 50k-minus-12.5k difference."""
    frame = pd.concat([runs, baseline], ignore_index=True)
    final = frame.loc[frame["global_step"].eq(50000)]
    table = final.pivot_table(index=["weight_decay", "model_width"], columns="sample_size",
                              values=["test_error", "train_error"])
    table.columns = [f"{metric}_{size}" for metric, size in table.columns]
    table["more_data_effect"] = table["test_error_50000"] - table["test_error_12500"]
    return table.reset_index()


def load_x4(project_dir: Path) -> dict[str, pd.DataFrame]:
    """Return the main-sweep Adam baseline and every X4 arm that has results."""
    runs = load_ablation(project_dir, "x4-optimizer")
    check_protocol(runs, "X4", dataset="cifar10", architecture="resnet", label_noise=NOISE,
                   augmentation=True, weight_decay=0.0, seed=0)
    arms: dict[str, pd.DataFrame] = {}
    for (optimizer, schedule), label in X4_ARMS.items():
        arm = runs.loc[runs["optimizer"].eq(optimizer) & runs["schedule"].eq(schedule)]
        if not arm.empty:
            check_grid(arm, ["model_width", "epoch"], set(product(X4_WIDTHS, EPOCHS)), f"X4 {label}")
            arms[label] = arm
    if sum(len(arm) for arm in arms.values()) != len(runs):
        raise ValueError("X4 contains an unexpected optimizer or schedule")
    baseline = our_runs(project_dir, "cifar10", "resnet", 50000, True, "adam", X4_WIDTHS, (NOISE,), 400, "epoch")
    return {X4_BASELINE: baseline, **arms}


def trajectories(ax: plt.Axes, frame: pd.DataFrame, baseline: pd.DataFrame, widths: tuple[int, ...],
                 time: str, scale: float, horizon: int) -> None:
    for width, color in zip(widths, WIDTH_COLORS, strict=True):
        for source, style, lw in ((frame, "-", 2.0), (baseline, "--", 1.3)):
            run = source.loc[source["model_width"].eq(width)].sort_values(time)
            ax.plot(run[time] / scale, run["test_error"], style, color=color, lw=lw,
                    label=f"Width {width}" if style == "-" else None)
    ax.axvline(horizon / scale, color="0.45", ls=":", lw=1.4)
    ax.set_xscale("log")
    ax.set_ylabel("Test Error")
    handles = ax.get_legend_handles_labels()[0] + [
        Line2D([], [], color="0.3", lw=2.0, label="Ablation run"),
        Line2D([], [], color="0.3", lw=1.3, ls="--", label="Main sweep"),
        Line2D([], [], color="0.45", lw=1.4, ls=":", label="Main-sweep horizon")]
    ax.legend(handles=handles, frameon=False, fontsize=8, ncol=2, loc="upper right")


def endpoints(ax: plt.Axes, frame: pd.DataFrame, widths: tuple[int, ...], time: str,
              horizons: tuple[tuple[int, str, str], ...]) -> None:
    for value, label, color in horizons:
        final = frame.loc[frame[time].eq(value)].set_index("model_width").loc[list(widths)]
        ax.plot(widths, final["test_error"], "-o", color=color, lw=2.2, ms=4, label=f"Test · {label}")
        ax.plot(widths, final["train_error"], "--", color=color, alpha=0.5, lw=1.8, label=f"Train · {label}")
    best = frame.groupby("model_width")["test_error"].min().loc[list(widths)]
    ax.plot(widths, best, "r--", lw=1.6, label="Minimum observed test error")
    ax.set(ylabel="Test / Train Error", ylim=(0, 0.6))
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper right")


def figure_x1(resnet: pd.DataFrame, cnn: pd.DataFrame, base_resnet: pd.DataFrame, base_cnn: pd.DataFrame) -> Figure:
    """Paper Figure 9 trajectories plus endpoint curves at the main and extended horizons."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
    trajectories(axes[0, 0], resnet, base_resnet, X1_RESNET_WIDTHS, "epoch", 1, 400)
    axes[0, 0].set(title="ResNet18 · test error over training", xlabel="Epoch (log scale)", xlim=(10, 1200), ylim=(0.1, 0.7))
    endpoints(axes[0, 1], resnet, X1_RESNET_WIDTHS, "epoch",
              ((400, "epoch 400", "#9db4d6"), (1200, "epoch 1,200", "blue")))
    axes[0, 1].set(title="ResNet18 · model-wise curve", xlabel="ResNet18 Width Parameter")
    trajectories(axes[1, 0], cnn, base_cnn, X1_CNN_WIDTHS, "global_step", 1000, 50000)
    axes[1, 0].set(title="CNN · test error over training", xlabel="Optimizer steps (thousands, log scale)",
                   xlim=(1.25, 200), ylim=(0.1, 0.7))
    endpoints(axes[1, 1], cnn, X1_CNN_WIDTHS, "global_step",
              ((50000, "50k steps", "#9db4d6"), (200000, "200k steps", "blue")))
    axes[1, 1].set(title="CNN · model-wise curve", xlabel="CNN Width Parameter")
    return caption(fig, "Ablation X1 · training horizon · CIFAR-10, 20% label noise",
                   "Augmented ResNet18 (Adam 10⁻⁴) to 1,200 epochs and CNN (SGD, inverse-square-root LR) to 200,000 steps; "
                   "one seed. Dashed: main sweep with the same settings. Red: oracle early stopping.")


def figure_x2(summary: pd.DataFrame, baseline: pd.DataFrame) -> Figure:
    """Paper Figure 28: single models versus their plurality-vote ensemble."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(14, 5), layout="constrained", gridspec_kw={"width_ratios": (2, 1)})
    widths = summary["model_width"].to_numpy(dtype=float)
    mean, std = summary["single_mean"].to_numpy(dtype=float), summary["single_std"].to_numpy(dtype=float)
    left.plot(widths, mean, "-o", color="blue", lw=2.3, ms=4, label="Standard (mean of 3 seeds)")
    left.fill_between(widths, mean - std, mean + std, color="blue", alpha=0.2, label="±1 SD across seeds")
    left.plot(widths, summary["ensemble"], "-o", color="red", lw=2.3, ms=4, label="Ensemble (plurality vote)")
    left.plot(widths, summary["train_mean"], "--", color="blue", alpha=0.45, lw=2, label="Train (mean)")
    left.plot(baseline["model_width"], baseline["test_error"], "x", color="0.3", ms=8, mew=2,
              label="Main sweep, seed 0")
    left.set(title="Test error", xlabel="ResNet18 Width Parameter", ylabel="Test / Train Error", ylim=(0, 0.45), xlim=(1, 66))
    left.legend(frameon=False, fontsize=9, loc="upper right")
    right.plot(widths, summary["ensemble_gain"], "-o", color="#874e76", lw=2.2, ms=5)
    right.set(title="Ensemble gain", xlabel="ResNet18 Width Parameter",
              ylabel="Mean single − ensemble test error", ylim=(0, None), xlim=(1, 66))
    return caption(fig, "Ablation X2 · seeds and ensembling · CIFAR-10 ResNet18, 20% label noise",
                   "Three seeds (new initialization, data order and noise mask each), Adam 10⁻⁴, 400 epochs, clean test set. "
                   "Ties go to the lowest class index. Paper: 5 models, 15% noise, 4,000 epochs.")


def figure_x3_dynamics(runs: pd.DataFrame, baseline: pd.DataFrame) -> Figure:
    """Paper Figure 21 at 50,000 examples: time-colored width curves and endpoint metrics."""
    frames = {0.0: baseline.loc[baseline["sample_size"].eq(50000)], 5e-4: runs.loc[runs["sample_size"].eq(50000)]}
    fig = plt.figure(figsize=(15, 8.5), layout="constrained")
    axes = fig.subplot_mosaic([["none", "test"], ["none", "test"], ["none", "train"],
                               ["decay", "train"], ["decay", "loss"], ["decay", "loss"]],
                              width_ratios=(1.6, 1))
    palette = rescale_viridis().reversed()
    time_norm = colors.LogNorm(vmin=CNN_STEPS[0], vmax=CNN_STEPS[-1])
    for key, (decay, frame) in zip(("none", "decay"), frames.items(), strict=True):
        ax = axes[key]
        grid = frame.pivot(index="global_step", columns="model_width", values="test_error").loc[list(CNN_STEPS), list(SUBSET_WIDTHS)]
        for step, errors in grid.iterrows():
            ax.plot(SUBSET_WIDTHS, errors, color=palette(time_norm(step)), alpha=0.5, lw=1.0)
        ax.plot(SUBSET_WIDTHS, grid.min(axis=0), "r--", lw=1.8, label="Optimal Early Stopping")
        ax.set(title="No Regularization" if decay == 0 else "5e-4 Weight Decay", ylabel="Test Error",
               xlabel="CNN Width Parameter", xlim=(1, 64), ylim=(0.1, 0.75))
        ax.legend(frameon=False, loc="upper right")
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=time_norm, cmap=palette), ax=[axes["none"], axes["decay"]], shrink=0.8)
    bar.ax.invert_yaxis()
    bar.set_label("Optimizer steps")
    for key, metric, label in (("test", "test_error", "Test Error"), ("train", "train_error", "Train Error"),
                               ("loss", "test_loss", "Test Loss")):
        for decay, frame in frames.items():
            final = frame.loc[frame["global_step"].eq(50000)].sort_values("model_width")
            axes[key].plot(final["model_width"], final[metric], color=DECAY_COLORS[decay], lw=2,
                           label="No Regularization" if decay == 0 else "Weight Decay=5e-4")
        axes[key].set(ylabel=label, xlim=(1, 64), ylim=(0, None))
    axes["test"].legend(frameon=False, fontsize=9)
    axes["loss"].set_xlabel("CNN Width Parameter")
    return caption(fig, "Ablation X3 · weight decay · CIFAR-10 CNN, 50,000 examples, 20% label noise",
                   "Paper Figure 21 layout (10% noise, 500K steps). Ours: augmentation, SGD with inverse-square-root LR, "
                   "50,000 steps, one seed; right column at step 50,000. Red: oracle early stopping.")


def figure_x3_samples(runs: pd.DataFrame, baseline: pd.DataFrame) -> Figure:
    """Paper Figure 11(a) with and without weight decay, plus where more data hurts."""
    table = x3_endpoints(runs, baseline)
    fig = plt.figure(figsize=(16, 8), layout="constrained")
    axes = fig.subplot_mosaic([["none_test", "none_train", "effect"], ["decay_test", "decay_train", "effect"]])
    for prefix, decay in (("none", 0.0), ("decay", 5e-4)):
        rows = table.loc[np.isclose(table["weight_decay"], decay)].sort_values("model_width")
        title = "No weight decay (main sweep)" if decay == 0 else "Weight decay 5e-4"
        for metric in ("test", "train"):
            ax = axes[f"{prefix}_{metric}"]
            for size, color in SIZE_COLORS.items():
                ax.plot(rows["model_width"], rows[f"{metric}_error_{size}"], color=color, lw=2, label=f"{size:,} examples")
            ax.set(title=f"{title} · {metric.title()}", xlabel="CNN width", ylabel="Error fraction", xlim=(1, 64), ylim=(0, 0.85))
            if metric == "test":
                ax.legend(frameon=False)
        axes["effect"].plot(rows["model_width"], rows["more_data_effect"], "-o", color=DECAY_COLORS[decay], lw=2, ms=4,
                            label="No weight decay" if decay == 0 else "Weight decay 5e-4")
    effect = axes["effect"]
    effect.axhline(0, color="0.3", lw=1)
    effect.axhspan(0, 1, color="#cf7653", alpha=0.08, label="More data hurts")
    low, high = float(table["more_data_effect"].min()), float(table["more_data_effect"].max())
    effect.set(title="Test error: 50,000 minus 12,500 examples", xlabel="CNN width", ylabel="Test-error difference",
               xlim=(1, 64), ylim=(min(1.1 * low, -0.02), max(1.3 * high, 0.03)))
    effect.legend(frameon=False, loc="lower right")
    return caption(fig, "Ablation X3 · weight decay and sample size · CIFAR-10 CNN, 20% label noise",
                   "Paper Figure 11(a) layout. Endpoints after 50,000 SGD steps at both sample sizes; one seed. "
                   "Right: values above zero mark widths where 4× more data raises test error.")


def figure_x4(arms: dict[str, pd.DataFrame]) -> Figure:
    """Paper Figures 16 and 18: test and train error over iterations for each optimizer setting."""
    fig, axes = plt.subplots(2, len(X4_WIDTHS), figsize=(13, 8), sharex=True, sharey="row", layout="constrained")
    for column, width in enumerate(X4_WIDTHS):
        for row, metric in enumerate(("test_error", "train_error")):
            ax = axes[row, column]
            for label, frame in arms.items():
                run = frame.loc[frame["model_width"].eq(width)].sort_values("global_step")
                ax.plot(run["global_step"], run[metric], color=X4_COLORS[label], lw=1.8, label=label)
            ax.set_xscale("log")
            ax.set(title=f"Width {width}", ylabel="Test Error" if row == 0 else "Train Error", ylim=(0, 0.9))
            if row == 1:
                ax.set_xlabel("Iterations")
    axes[0, 0].legend(frameon=False, fontsize=9, loc="upper right")
    missing = [label for label in X4_COLORS if label not in arms]
    note = f" Not available: {', '.join(missing)}." if missing else ""
    return caption(fig, "Ablation X4 · optimizer and learning-rate schedule · CIFAR-10 ResNet18, 20% label noise",
                   "Paper Figures 16 and 18 layout; augmentation, 400 epochs, one seed." + note)


def save_ablation_figures(project_dir: Path, output_dir: Path) -> dict[str, Path]:
    """Save every ablation figure as a PNG under output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    x1 = load_x1(project_dir)
    x2 = load_x2(project_dir)
    x3 = load_x3(project_dir)
    x4 = load_x4(project_dir)
    renders = (("x1_horizon", lambda: figure_x1(*x1)),
               ("x2_ensemble", lambda: figure_x2(*x2)),
               ("x3_weight_decay_dynamics", lambda: figure_x3_dynamics(*x3)),
               ("x3_weight_decay_samples", lambda: figure_x3_samples(*x3)),
               ("x4_optimizer", lambda: figure_x4(x4)))
    paths: dict[str, Path] = {}
    for name, render in renders:
        fig = render()
        style_figure(fig)
        paths[name] = output_dir / f"{name}.png"
        fig.savefig(paths[name], dpi=200, bbox_inches="tight")
        plt.close(fig)
    return paths
