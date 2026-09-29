"""Compare our 400-epoch runs with the authors' 20%-noise ResNet measurements."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pyarrow.dataset as ds


RESULT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = RESULT_DIR.parents[1]
PAPER_PARQUET = PROJECT_DIR / "their-results/hf_dataset/data/vision/cifar10-resnet18k-50k-adam.parquet"
METRICS = ("train_error", "test_error", "train_loss", "test_loss")


def compare() -> pd.DataFrame:
    paper = ds.dataset(PAPER_PARQUET, format="parquet").to_table(
        columns=["model_width", "measurement_index", *METRICS],
        filter=(ds.field("trial_index") == 2)
        & ds.field("model_width").isin([3, 64])
        & ds.field("measurement_index").isin([epoch - 1 for epoch in range(10, 401, 10)]),
    ).to_pandas()
    paper["epoch"] = paper["measurement_index"] + 1
    paper = paper.rename(columns={metric: f"paper_{metric}" for metric in METRICS})

    ours = pd.concat(
        [pd.read_parquet(path) for path in sorted((RESULT_DIR / "data/vision").glob("*.parquet"))],
        ignore_index=True,
    )
    ours = ours[["model_width", "epoch", *METRICS]].rename(
        columns={metric: f"our_{metric}" for metric in METRICS}
    )
    comparison = paper.merge(ours, on=["model_width", "epoch"], validate="one_to_one")
    comparison = comparison.sort_values(["model_width", "epoch"]).reset_index(drop=True)
    if len(comparison) != 80:
        raise ValueError(f"Expected 80 matched rows, found {len(comparison)}")
    for metric in METRICS:
        comparison[f"delta_{metric}"] = comparison[f"our_{metric}"] - comparison[f"paper_{metric}"]
    return comparison


def plot(comparison: pd.DataFrame) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for axis, width in zip(axes, (3, 64), strict=True):
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


def main() -> None:
    comparison = compare()
    comparison.to_csv(RESULT_DIR / "paper-comparison.csv", index=False)
    plot(comparison)
    final = comparison.loc[comparison["epoch"] == 400]
    print(final[["model_width", "paper_train_error", "our_train_error", "paper_test_error", "our_test_error", "delta_test_error"]].to_string(index=False))


if __name__ == "__main__":
    main()
