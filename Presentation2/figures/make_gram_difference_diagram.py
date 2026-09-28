"""Render the elementwise difference behind DINOv3's Gram loss."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from make_gram_grid_diagram import BLUE, INK, MUTED, PAPER, TEAL, downsample_bicubic, feature_map, normalize


OUTPUT = Path(__file__).with_name("gram-difference-frobenius.png")


def gram(values: np.ndarray) -> np.ndarray:
    patches = normalize(values).reshape(-1, values.shape[-1])
    return patches @ patches.T


def arrow(fig: Figure, start: float, end: float, y: float, color: str) -> None:
    fig.add_artist(
        FancyArrowPatch(
            (start, y),
            (end, y),
            transform=fig.transFigure,
            arrowstyle="-|>",
            mutation_scale=18,
            linewidth=2,
            color=color,
        )
    )


def main() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans"})
    student = gram(feature_map(16, noise=0.29, seed=6))
    teacher = gram(downsample_bicubic(feature_map(32, noise=0.025, seed=7)))
    difference = student - teacher
    squared = difference**2

    fig = plt.figure(figsize=(16, 9), facecolor=PAPER)
    fig.text(0.055, 0.94, "What the Gram loss actually measures", fontsize=24, fontweight="bold", color=INK)
    fig.text(
        0.055,
        0.895,
        "Each matrix cell (i, j) is the cosine similarity between two patch features at positions i and j.",
        fontsize=12,
        color=MUTED,
    )

    x_positions = [0.065, 0.305, 0.545, 0.785]
    width = 0.165
    y = 0.39
    titles = ["STUDENT", "GRAM TEACHER", "DIFFERENCE", "SQUARED DIFFERENCE"]
    symbols = [r"$G_S$", r"$G_G$", r"$\Delta=G_S-G_G$", r"$\Delta_{ij}^{,2}$"]
    descriptions = [
        "Current patch similarities",
        "Target patch similarities",
        "Blue: lower · red: higher",
        "Brighter = larger error",
    ]
    matrices = [student, teacher, difference, squared]
    display_difference = float(np.quantile(np.abs(difference), 0.98))
    display_squared = float(np.quantile(squared, 0.98))

    for index, (x, title, symbol, description, matrix) in enumerate(
        zip(x_positions, titles, symbols, descriptions, matrices, strict=True)
    ):
        ax = fig.add_axes((x, y, width, width * 16 / 9))
        if index < 2:
            ax.imshow(matrix, cmap="viridis", vmin=-1, vmax=1, interpolation="nearest")
        elif index == 2:
            ax.imshow(matrix, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-display_difference, vcenter=0, vmax=display_difference), interpolation="nearest")
        else:
            ax.imshow(matrix, cmap="magma", vmin=0, vmax=display_squared, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        fig.text(x + width / 2, 0.785, title, ha="center", va="center", fontsize=11, fontweight="bold", color=BLUE if index == 0 else TEAL if index == 1 else INK)
        fig.text(x + width / 2, 0.737, symbol, ha="center", va="center", fontsize=19, color=INK)
        fig.text(x + width / 2, 0.355, description, ha="center", va="center", fontsize=10, color=MUTED)

    mid_y = y + width * 16 / 9 / 2
    fig.text(0.285, mid_y, "−", ha="center", va="center", fontsize=34, color=MUTED)
    fig.text(0.525, mid_y, "=", ha="center", va="center", fontsize=32, color=MUTED)
    arrow(fig, x_positions[2] + width + 0.012, x_positions[3] - 0.012, mid_y, MUTED)

    box = FancyBboxPatch(
        (0.115, 0.115),
        0.77,
        0.155,
        boxstyle="round,pad=0.010,rounding_size=0.020",
        transform=fig.transFigure,
        facecolor="#E8F0F4",
        edgecolor="none",
    )
    fig.add_artist(box)
    fig.text(0.50, 0.220, "SUM EVERY SQUARED CELL", ha="center", va="center", fontsize=10, fontweight="bold", color=MUTED)
    fig.text(
        0.50,
        0.157,
        r"$\mathcal{L}_{\rm Gram}=\|G_S-G_G\|_F^2=\sum_{i=1}^{256}\sum_{j=1}^{256}(G_{S,ij}-G_{G,ij})^2$",
        ha="center",
        va="center",
        fontsize=21,
        color=INK,
    )
    fig.text(
        0.50,
        0.046,
        "Illustrative patch features; the operation and 256 × 256 matrix dimensions follow DINOv3 §4.2–4.3.",
        ha="center",
        va="center",
        fontsize=9,
        color=MUTED,
    )

    fig.savefig(OUTPUT, dpi=170, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
