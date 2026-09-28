"""Render a schematic of the high-resolution Gram anchoring alignment."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch
from PIL import Image


OUTPUT = Path(__file__).with_name("gram-grid-alignment.png")
INK = "#18283A"
MUTED = "#5E7080"
BLUE = "#245F9C"
TEAL = "#168A83"
PAPER = "#F7F9FB"


def feature_map(size: int, noise: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    xx = (xx + 0.5) / size
    yy = (yy + 0.5) / size
    shape_a = np.exp(-(((xx - 0.36) / 0.25) ** 2 + ((yy - 0.52) / 0.28) ** 2))
    shape_b = np.exp(-(((xx - 0.72) / 0.15) ** 2 + ((yy - 0.34) / 0.19) ** 2))
    background = 1.0 - np.maximum(shape_a, shape_b)
    values = np.stack([shape_a, shape_b, background, 0.35 * xx, 0.35 * yy], axis=-1)
    values += rng.normal(0.0, noise, values.shape)
    return values


def downsample_bicubic(values: np.ndarray) -> np.ndarray:
    channels = [
        np.asarray(Image.fromarray(values[:, :, channel].astype(np.float32), mode="F").resize((16, 16), Image.Resampling.BICUBIC))
        for channel in range(values.shape[-1])
    ]
    return np.stack(channels, axis=-1)


def normalize(values: np.ndarray) -> np.ndarray:
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-8)


def colors(values: np.ndarray) -> np.ndarray:
    values = normalize(values)
    red = 0.78 * values[:, :, 0] + 0.18 * values[:, :, 1] + 0.12
    green = 0.30 * values[:, :, 0] + 0.72 * values[:, :, 1] + 0.16
    blue = 0.23 * values[:, :, 0] + 0.49 * values[:, :, 2] + 0.25
    return np.clip(np.stack([red, green, blue], axis=-1), 0.0, 1.0)


def draw_grid(ax: Axes, count: int, color: str, alpha: float) -> None:
    for coordinate in range(1, count):
        position = coordinate / count
        ax.axhline(position, color=color, linewidth=0.38, alpha=alpha)
        ax.axvline(position, color=color, linewidth=0.38, alpha=alpha)


def draw_scene(ax: Axes, count: int, accent: str) -> None:
    ax.set_facecolor("#E6EFF0")
    ax.add_patch(Ellipse((0.36, 0.52), 0.52, 0.58, angle=-18, facecolor="#7CC6B3", edgecolor="none"))
    ax.add_patch(Ellipse((0.72, 0.34), 0.28, 0.38, angle=18, facecolor="#E7B674", edgecolor="none"))
    ax.add_patch(Circle((0.28, 0.66), 0.07, facecolor="#4A9D9C", edgecolor="none"))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    for coordinate in range(1, count):
        position = coordinate / count
        ax.axhline(position, color="white", linewidth=0.33 if count == 32 else 0.55, alpha=0.8)
        ax.axvline(position, color="white", linewidth=0.33 if count == 32 else 0.55, alpha=0.8)
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor=accent, linewidth=2.2))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_features(ax: Axes, values: np.ndarray, count: int, accent: str) -> None:
    ax.imshow(colors(values), interpolation="nearest", origin="lower", extent=(0, 1, 0, 1))
    draw_grid(ax, count, "white", 0.58)
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor=accent, linewidth=2.2))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_gram(ax: Axes, values: np.ndarray, accent: str) -> None:
    flat = normalize(values).reshape(-1, values.shape[-1])
    gram = flat @ flat.T
    ax.imshow(gram, cmap="mako" if "mako" in plt.colormaps() else "viridis", vmin=0, vmax=1, interpolation="nearest")
    ax.add_patch(plt.Rectangle((-0.5, -0.5), 256, 256, fill=False, edgecolor=accent, linewidth=2.2))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def add_arrow(fig: Figure, x_start: float, x_end: float, y: float, color: str) -> None:
    fig.add_artist(
        FancyArrowPatch(
            (x_start, y),
            (x_end, y),
            transform=fig.transFigure,
            arrowstyle="-|>",
            mutation_scale=15,
            linewidth=1.6,
            color=color,
        )
    )


def main() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans"})
    fig = plt.figure(figsize=(14, 8.2), facecolor=PAPER)
    fig.text(0.055, 0.945, "Gram anchoring: align patch relationships", fontsize=22, fontweight="bold", color=INK)
    fig.text(
        0.055,
        0.902,
        "The teacher sees the same scene at twice the pixel resolution; its feature grid is resized before comparison.",
        fontsize=11,
        color=MUTED,
    )

    positions = [0.085, 0.300, 0.515, 0.750]
    width = 0.145
    height = 0.245
    row_y = [(0.565, BLUE), (0.205, TEAL)]
    headings = ["INPUT CROP", "BACKBONE PATCH MAP", "ALIGNED PATCH MAP", "PATCH GRAM MATRIX"]
    for x, heading in zip(positions, headings, strict=True):
        fig.text(x + width / 2, 0.835, heading, ha="center", va="center", color=MUTED, fontsize=9, fontweight="bold")

    student_raw = feature_map(16, noise=0.29, seed=6)
    teacher_raw = feature_map(32, noise=0.025, seed=7)
    teacher_aligned = downsample_bicubic(teacher_raw)

    for row_index, (y, accent) in enumerate(row_y):
        is_student = row_index == 0
        fig.text(0.025, y + 0.155, "STUDENT" if is_student else "GRAM TEACHER", color=accent, fontsize=10, fontweight="bold", rotation=90, ha="center", va="center")
        axes = [fig.add_axes((x, y, width, height)) for x in positions]
        draw_scene(axes[0], 16 if is_student else 32, accent)
        draw_features(axes[1], student_raw if is_student else teacher_raw, 16 if is_student else 32, accent)
        draw_features(axes[2], student_raw if is_student else teacher_aligned, 16, accent)
        draw_gram(axes[3], student_raw if is_student else teacher_aligned, accent)

        for left in range(3):
            add_arrow(fig, positions[left] + width + 0.012, positions[left + 1] - 0.012, y + height / 2, accent)

        labels = (
            ["256 × 256 pixels", "16 × 16 = 256 tokens", "16 × 16 = 256 tokens", "$G_S$: 256 × 256"]
            if is_student
            else ["512 × 512 pixels", "32 × 32 = 1,024 tokens", "16 × 16 = 256 tokens", "$G_G$: 256 × 256"]
        )
        for x, label in zip(positions, labels, strict=True):
            fig.text(x + width / 2, y - 0.025, label, ha="center", va="top", fontsize=9.5, color=INK)

    fig.text(0.480, 0.710, "same grid", fontsize=8.5, color=BLUE, ha="center")
    fig.text(0.480, 0.350, "bicubic ↓ 2", fontsize=8.5, color=TEAL, ha="center")

    box = FancyBboxPatch((0.20, 0.045), 0.60, 0.080, boxstyle="round,pad=0.006,rounding_size=0.012", transform=fig.transFigure, facecolor="#EAF1F5", edgecolor="none")
    fig.add_artist(box)
    fig.text(0.50, 0.086, r"$\mathcal{L}_{\rm Gram}=\|G_S-G_G\|_F^2$", ha="center", va="center", fontsize=19, color=INK)
    fig.text(0.50, 0.025, "Schematic feature maps; dimensions and alignment follow DINOv3 §4.3.", ha="center", fontsize=8.5, color=MUTED)

    fig.savefig(OUTPUT, dpi=190, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
