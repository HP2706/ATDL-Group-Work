"""Command-line plotting for published and reproduced double-descent metrics."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "atdl-matplotlib"))

import fire
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from translation import sample_plot, translation_plot
from vision import dynamics, final_curve, ocean


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = ROOT.parent / "their-results" / "hf_dataset"
DEFAULT_OUTPUT = ROOT.parent / "plots" / "exploratory" / "published_cli"


def save(fig: Figure, output: str) -> str:
    path = Path(output)
    if path.suffix.lower() not in {".png", ".pdf", ".svg"}:
        raise ValueError("Output must end in .png, .pdf, or .svg")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(path.resolve())


class PlotCLI:
    def resnet(
        self,
        source: str = "cifar10-resnet18k-p15-adam-reps",
        output: str | None = None,
        dataset_dir: str = str(DEFAULT_DATASET),
        noise_level: float = 0.15,
        class_count: int = 10,
    ) -> str:
        target = output or str(DEFAULT_OUTPUT / f"{source}-curve.png")
        return save(final_curve(source, Path(dataset_dir), noise_level, class_count), target)

    def ocean(
        self,
        source: str = "cifar10-resnet18k-p15-adam-reps",
        output: str | None = None,
        dataset_dir: str = str(DEFAULT_DATASET),
        trial_index: int = 0,
        metric: str = "test_error",
        base: float = 1.1,
        contours: bool = True,
    ) -> str:
        target = output or str(DEFAULT_OUTPUT / f"{source}-{metric}-ocean.png")
        return save(ocean(source, Path(dataset_dir), trial_index, metric, base, contours), target)

    def dynamics(
        self,
        source: str = "cifar10-resnet18k-p15-adam-reps",
        output: str | None = None,
        dataset_dir: str = str(DEFAULT_DATASET),
        trial_index: int = 0,
        metric: str = "test_error",
        base: float = 1.1,
        noise_level: float = 0.15,
        class_count: int = 10,
    ) -> str:
        target = output or str(DEFAULT_OUTPUT / f"{source}-{metric}-dynamics.png")
        return save(dynamics(source, Path(dataset_dir), trial_index, metric, base, noise_level, class_count), target)

    def translation_model(
        self,
        sweep: str = "small",
        output: str | None = None,
        dataset_dir: str = str(DEFAULT_DATASET),
    ) -> str:
        target = output or str(DEFAULT_OUTPUT / f"translation-model-{sweep}.png")
        return save(translation_plot(Path(dataset_dir), sweep), target)

    def translation_samples(
        self,
        output: str | None = None,
        dataset_dir: str = str(DEFAULT_DATASET),
    ) -> str:
        target = output or str(DEFAULT_OUTPUT / "translation-samples.png")
        return save(sample_plot(Path(dataset_dir)), target)


if __name__ == "__main__":
    fire.Fire(PlotCLI)
