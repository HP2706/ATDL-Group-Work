"""Render early recorded views of the published vision histories.

Fractions refer to saved measurement positions, not verified epochs or steps.
"""

from __future__ import annotations

import math
import sys
from functools import lru_cache
from pathlib import Path
from typing import Literal

import fire
import matplotlib.pyplot as plt
import pyarrow.parquet as pq
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent
PLOTS_CODE = ROOT.parent / "our-plots"
sys.path.insert(0, str(PLOTS_CODE))
SPARSE_WIDTHS: tuple[int, ...] = (2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64)

from vision import dynamics, figure9_panels, final_curve, ocean  # noqa: E402
from render import comparison, sample_heatmap, save, source_curve  # noqa: E402


class FigureSpec(BaseModel):
    kind: Literal["final", "dynamics", "ocean", "figure9", "source", "comparison", "sample_heatmap"]
    source: str = ""
    trial_index: int = 0
    metric: str = "test_error"
    noise_level: float = 0.0
    class_count: int = 10
    contours: bool = True
    title: str = ""
    series: tuple[tuple[str, str, int], ...] = ()


FIGURES: dict[str, FigureSpec] = {
    "figure_01_resnet_15pct_final": FigureSpec(kind="final", source="cifar10-resnet18k-p15-adam-reps", noise_level=0.15),
    "figure_01_resnet_15pct_dynamics": FigureSpec(kind="dynamics", source="cifar10-resnet18k-p15-adam-reps", noise_level=0.15),
    "figure_02_resnet_15pct_test_heatmap": FigureSpec(kind="ocean", source="cifar10-resnet18k-p15-adam-reps"),
    "figure_02_resnet_15pct_train_heatmap": FigureSpec(kind="ocean", source="cifar10-resnet18k-p15-adam-reps", metric="train_error", contours=False),
    "figure_04_resnet_cifar10_noise": FigureSpec(kind="comparison", title="CIFAR-10 ResNet18 noise sweep", series=(
        ("cifar10-resnet18k-50k-adam", "Clean", 0),
        ("cifar10-resnet18k-50k-adam", "10% noise", 1),
        ("cifar10-resnet18k-50k-adam", "20% noise", 2),
    )),
    "figure_04_resnet_cifar10_all_noise": FigureSpec(kind="comparison", title="CIFAR-10 ResNet18 five-level noise sweep", series=(
        ("cifar10-resnet18k-50k-adam", "Clean", 0),
        ("pct-cf10-res18-50k-p05-adam", "5% noise", 0),
        ("cifar10-resnet18k-50k-adam", "10% noise", 1),
        ("pct-cifar10-resnet18-50k-p15-adam", "15% noise", 0),
        ("cifar10-resnet18k-50k-adam", "20% noise", 2),
    )),
    "figure_04_resnet_cifar100_noise": FigureSpec(kind="comparison", title="CIFAR-100 ResNet18 noise sweep", series=(
        ("cifar100-resnet18k-50k-adam", "Clean", 0),
        ("cifar100-resnet18k-50k-adam", "10% noise", 1),
        ("cifar100-resnet18k-50k-adam", "20% noise", 2),
    )),
    "figure_05_cifar10_cnn_augmentation": FigureSpec(kind="comparison", title="CIFAR-10 CNN augmentation", series=(
        ("cifar10-mcnn-p10-sgd", "Augmented", 0),
        ("cifar10-mcnn-noaug-sgd", "No augmentation", 1),
    )),
    "figure_06_cifar10_cnn_optimizers": FigureSpec(kind="comparison", title="CIFAR-10 CNN optimizers", series=(
        ("cifar10-mcnn-noaug-sgd", "SGD", 0),
        ("cifar10-mcnn-noaug-adam", "Adam", 0),
    )),
    "figure_07_cifar100_cnn": FigureSpec(kind="source", source="pct-cifar100-mcnn-p0-sgd-noaug-reps"),
    "figure_09_resnet_20pct_dynamics": FigureSpec(kind="dynamics", source="cifar10-resnet18k-50k-adam", trial_index=2, noise_level=0.20),
    "figure_09_resnet_20pct_heatmap": FigureSpec(kind="ocean", source="cifar10-resnet18k-50k-adam", trial_index=2),
    "figure_09_resnet_20pct_panels": FigureSpec(kind="figure9", source="cifar10-resnet18k-50k-adam", trial_index=2),
    "figure_10_cifar10_cnn_dynamics": FigureSpec(kind="dynamics", source="cifar10-mcnn-p20-sgd"),
    "figure_11a_cifar10_cnn_sample_sizes": FigureSpec(kind="source", source="dd_grid_p20"),
    "figure_12_cifar10_cnn_sample_heatmap": FigureSpec(kind="sample_heatmap", source="dd_grid_p20"),
    "figure_19_cifar100_resnet_heatmap": FigureSpec(kind="ocean", source="cifar100-resnet18k-50k-adam"),
    "figure_19_cifar100_resnet_dynamics": FigureSpec(kind="dynamics", source="cifar100-resnet18k-50k-adam"),
    "figure_20_cifar100_cnn_heatmap": FigureSpec(kind="ocean", source="pct-cifar100-mcnn-p0-sgd-noaug-reps"),
    "figure_20_cifar100_cnn_dynamics": FigureSpec(kind="dynamics", source="pct-cifar100-mcnn-p0-sgd-noaug-reps"),
    "figure_21_cifar10_cnn_weight_decay": FigureSpec(kind="comparison", title="CIFAR-10 CNN weight decay variants", series=(
        ("pct-cifar10-mcnn-p10-sgd-aug-decay-big", "Variant 1", 0),
        ("pct-cifar10-mcnn-p10-sgd-aug-decay5-big", "Variant 2", 0),
    )),
    "figure_25_cifar10_cnn_10pct_heatmap": FigureSpec(kind="ocean", source="cifar10-mcnn-p10-sgd"),
    "figure_27_cifar10_wide_cnn": FigureSpec(kind="comparison", title="Wide CIFAR-10 CNN", series=(
        ("pct-cifar10-mcnn-50000-p0-sgd-big", "Clean", 0),
        ("pct-cifar10-mcnn-50000-p20-sgd-big", "20% noise", 0),
    )),
}


@lru_cache(maxsize=None)
def measurement_count(source: str, dataset_dir: Path) -> int:
    path = dataset_dir / "data" / "vision" / f"{source}.parquet"
    indices = pq.read_table(path, columns=["measurement_index"])["measurement_index"]
    count = int(indices.to_numpy().max()) + 1
    if count < 2:
        raise ValueError(f"No time history in {path}")
    return count


def measurement_cap(source: str, dataset_dir: Path, fraction: float) -> int:
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    return max(2, math.ceil(measurement_count(source, dataset_dir) * fraction))


def parse_widths(value: str) -> tuple[int, ...]:
    widths = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not widths or any(width < 1 for width in widths) or len(set(widths)) != len(widths):
        raise ValueError("Enter distinct positive widths separated by commas")
    return tuple(sorted(widths))


def chart_names(dataset_dir: Path) -> list[str]:
    sources = sorted((dataset_dir / "data" / "vision").glob("*.parquet"))
    return sorted(FIGURES) + [f"source_sweeps/{path.stem}" for path in sources]


def render_figure(
    name: str, fraction: float, dataset_dir: Path,
    model_widths: tuple[int, ...] | None = None,
) -> plt.Figure:
    if name.startswith("source_sweeps/"):
        source = name.split("/", 1)[1]
        return source_curve(source, dataset_dir, measurement_cap(source, dataset_dir, fraction), model_widths)
    spec = FIGURES[name]
    if spec.kind == "comparison":
        caps = {source: measurement_cap(source, dataset_dir, fraction) for source, _, _ in spec.series}
        return comparison(list(spec.series), dataset_dir, spec.title, measurement_caps=caps,
                          model_widths=model_widths)
    cap = measurement_cap(spec.source, dataset_dir, fraction)
    if spec.kind == "final":
        return final_curve(spec.source, dataset_dir, spec.noise_level, spec.class_count, cap, model_widths)
    if spec.kind == "dynamics":
        return dynamics(spec.source, dataset_dir, spec.trial_index, noise_level=spec.noise_level,
                        class_count=spec.class_count, max_measurements=cap,
                        model_widths=model_widths)
    if spec.kind == "ocean":
        return ocean(spec.source, dataset_dir, spec.trial_index, spec.metric,
                     contours=spec.contours, max_measurements=cap, model_widths=model_widths)
    if spec.kind == "figure9":
        return figure9_panels(spec.source, dataset_dir, spec.trial_index, max_measurements=cap,
                             selected_widths=model_widths)
    if spec.kind == "source":
        return source_curve(spec.source, dataset_dir, cap, model_widths)
    return sample_heatmap(dataset_dir, cap, model_widths)


def render(
    fractions: tuple[float, ...] = (0.01, 0.05, 0.1),
    dataset_dir: str = str(ROOT.parent / "their-results" / "hf_dataset"),
    output_dir: str = str(ROOT.parent / "plots" / "exploratory" / "short_horizon"),
    include_source_sweeps: bool = True,
) -> list[str]:
    data = Path(dataset_dir)
    output = Path(output_dir)
    names = chart_names(data) if include_source_sweeps else sorted(FIGURES)
    written: list[str] = []
    for fraction in fractions:
        label = f"{fraction * 100:g}pct"
        for name in names:
            fig = render_figure(name, fraction, data)
            path = output / label / f"{name}.png"
            save(fig, path)
            written.append(str(path))
            print(path, flush=True)
    full_name = "figure_04_resnet_cifar10_all_noise"
    full_path = output / "100pct" / f"{full_name}.png"
    save(render_figure(full_name, 1.0, data), full_path)
    written.append(str(full_path))
    print(full_path, flush=True)
    return written


if __name__ == "__main__":
    fire.Fire(render)
