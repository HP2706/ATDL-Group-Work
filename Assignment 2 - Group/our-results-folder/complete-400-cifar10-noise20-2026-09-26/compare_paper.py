"""Compare our 400-epoch runs with the authors' 20%-noise ResNet measurements."""

from __future__ import annotations

from pathlib import Path
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pyarrow.dataset as ds


RESULT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = RESULT_DIR.parents[1]
PAPER_PARQUET = Path(
    os.environ.get(
        "PAPER_PARQUET",
        PROJECT_DIR / "their-results/hf_dataset/data/vision/cifar10-resnet18k-50k-adam.parquet",
    )
)
METRICS = ("train_error", "test_error", "train_loss", "test_loss")
OUR_PARQUETS = {
    3: RESULT_DIR / "data/vision/cifar10-resnet-k3-seed0-2026-09-26-21-52-41.parquet",
    12: PROJECT_DIR / "our-results-folder/data/vision/cifar10-resnet-k12-seed0-2026-09-27-03-23-12.parquet",
    64: RESULT_DIR / "data/vision/cifar10-resnet-k64-seed0-2026-09-26-21-52-41.parquet",
}
WIDTHS = tuple(OUR_PARQUETS)


def load_ours() -> dict[int, pd.DataFrame]:
    trajectories = {}
    for width, path in OUR_PARQUETS.items():
        frame = pd.read_parquet(path)
        if set(frame["model_width"].unique()) != {width}:
            raise ValueError(f"Expected width {width} in {path}")
        trajectories[width] = frame.sort_values("epoch").reset_index(drop=True)
    return trajectories


def load_paper(measurement_indexes: list[int] | None = None) -> pd.DataFrame:
    predicate = (ds.field("trial_index") == 2) & ds.field("model_width").isin(WIDTHS)
    if measurement_indexes is not None:
        predicate &= ds.field("measurement_index").isin(measurement_indexes)
    paper = ds.dataset(PAPER_PARQUET, format="parquet").to_table(
        columns=["model_width", "measurement_index", *METRICS],
        filter=predicate,
    ).to_pandas()
    paper["epoch"] = paper["measurement_index"] + 1
    return paper.sort_values(["model_width", "epoch"]).reset_index(drop=True)


def compare() -> pd.DataFrame:
    paper = load_paper([epoch - 1 for epoch in range(10, 401, 10)])
    paper = paper.rename(columns={metric: f"paper_{metric}" for metric in METRICS})

    ours = pd.concat(load_ours().values(), ignore_index=True)
    ours = ours[["model_width", "epoch", *METRICS]].rename(
        columns={metric: f"our_{metric}" for metric in METRICS}
    )
    comparison = paper.merge(ours, on=["model_width", "epoch"], validate="one_to_one")
    comparison = comparison.sort_values(["model_width", "epoch"]).reset_index(drop=True)
    if len(comparison) != len(WIDTHS) * 40:
        raise ValueError(f"Expected {len(WIDTHS) * 40} matched rows, found {len(comparison)}")
    for metric in METRICS:
        comparison[f"delta_{metric}"] = comparison[f"our_{metric}"] - comparison[f"paper_{metric}"]
    return comparison


def plot(comparison: pd.DataFrame) -> None:
    figure, axes = plt.subplots(len(WIDTHS), 1, figsize=(10, 11), sharex=True)
    for axis, width in zip(axes, WIDTHS):
        rows = comparison.loc[comparison["model_width"] == width]
        axis.plot(rows["epoch"], 100 * rows["paper_test_error"], color="#244b71", label="Paper test")
        axis.plot(rows["epoch"], 100 * rows["our_test_error"], color="#da713b", label="Our test")
        axis.plot(rows["epoch"], 100 * rows["paper_train_error"], color="#244b71", linestyle="--", alpha=0.75, label="Paper train")
        axis.plot(rows["epoch"], 100 * rows["our_train_error"], color="#da713b", linestyle="--", alpha=0.75, label="Our train")
        axis.set_title(f"ResNet18 width {width}, CIFAR-10, 20% label noise")
        axis.set_ylabel("Error (%)")
        axis.set_ylim(bottom=0)
        axis.grid(alpha=0.2)
    axes[0].legend(ncol=2)
    axes[-1].set_xlabel("Epoch (authors' plotted index + 1)")
    figure.tight_layout()
    figure.savefig(RESULT_DIR / "paper-comparison.png", dpi=200)
    plt.close(figure)


def summarize_trajectory(frame: pd.DataFrame, prefix: str) -> dict[str, float | int]:
    frame = frame.sort_values("epoch")
    errors = frame["test_error"].to_numpy()
    minimum_index = int(errors.argmin())
    later_errors = errors[minimum_index + 1 :]
    peak_index = minimum_index + 1 + int(later_errors.argmax()) if len(later_errors) else None
    summary: dict[str, float | int] = {
        f"{prefix}_first_min_error_pct": 100 * errors[minimum_index],
        f"{prefix}_first_min_epoch": int(frame.iloc[minimum_index]["epoch"]),
        f"{prefix}_final_error_pct": 100 * errors[-1],
        f"{prefix}_final_epoch": int(frame.iloc[-1]["epoch"]),
    }
    summary[f"{prefix}_post_min_peak_error_pct"] = (
        100 * errors[peak_index] if peak_index is not None else float("nan")
    )
    summary[f"{prefix}_post_min_peak_epoch"] = (
        int(frame.iloc[peak_index]["epoch"]) if peak_index is not None else float("nan")
    )
    return summary


def epochwise_summary(ours: dict[int, pd.DataFrame]) -> pd.DataFrame:
    paper = load_paper()
    rows = []
    for width in WIDTHS:
        ours_width = ours[width]
        paper_width = paper.loc[paper["model_width"] == width]
        paper_at_400 = paper_width.loc[paper_width["epoch"] == 400, "test_error"]
        if paper_at_400.empty:
            raise ValueError(f"Authors' width {width} trajectory has no epoch-400 measurement")
        row: dict[str, float | int] = {
            "model_width": width,
            "paper_error_epoch_400_pct": 100 * paper_at_400.iloc[0],
            "paper_last_error_pct": 100 * paper_width.iloc[-1]["test_error"],
            "paper_last_epoch": int(paper_width.iloc[-1]["epoch"]),
        }
        row.update(summarize_trajectory(ours_width, "our"))
        row.update(summarize_trajectory(paper_width, "paper"))
        rows.append(row)
    return pd.DataFrame(rows)


def plot_epochwise(ours: dict[int, pd.DataFrame]) -> None:
    paper = load_paper()
    figure, axes = plt.subplots(len(WIDTHS), 1, figsize=(10, 11), sharex=True)
    for axis, width in zip(axes, WIDTHS):
        paper_rows = paper.loc[paper["model_width"] == width]
        our_rows = ours[width]
        axis.plot(paper_rows["epoch"], 100 * paper_rows["test_error"], color="#244b71", label="Authors' released run")
        axis.plot(our_rows["epoch"], 100 * our_rows["test_error"], color="#da713b", label="Our run")
        axis.axvline(400, color="#555555", linestyle=":", linewidth=1, label="Our horizon" if width == WIDTHS[0] else None)
        axis.set_title(f"ResNet18 width {width}, CIFAR-10, 20% label noise")
        axis.set_ylabel("Test error (%)")
        axis.set_ylim(bottom=0)
        axis.grid(alpha=0.2)
    axes[0].legend(ncol=3)
    axes[-1].set_xlabel("Epoch (authors' plotted index + 1)")
    figure.tight_layout()
    figure.savefig(RESULT_DIR / "epochwise-comparison.png", dpi=200)
    plt.close(figure)


def main() -> None:
    comparison = compare()
    comparison.to_csv(RESULT_DIR / "paper-comparison.csv", index=False)
    plot(comparison)
    ours = load_ours()
    summary = epochwise_summary(ours)
    summary.to_csv(RESULT_DIR / "epochwise-summary.csv", index=False)
    plot_epochwise(ours)
    final = comparison.loc[comparison["epoch"] == 400]
    print(final[["model_width", "paper_train_error", "our_train_error", "paper_test_error", "our_test_error", "delta_test_error"]].to_string(index=False))
    print("\nEpochwise summary (errors in %):")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
