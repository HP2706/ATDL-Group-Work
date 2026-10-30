"""Reproducible redraw of SAM 3D Figure 2 and our mask-aware pooling pilot."""
from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

OUT = Path(__file__).resolve().parent
INK = "#26343d"
LINE = "#8c979e"
COLORS = {"encoder": "#d4eeea", "flow": "#fff1ad", "shape": "#ddd4ee", "layout": "#d3e8f7", "mesh": "#dfeede", "splat": "#f7e4dc", "neutral": "#f3f4f5"}
plt.rcParams.update({"font.size": 8, "axes.labelsize": 9, "axes.titlesize": 10, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "figure.titlesize": 11, "lines.linewidth": 1.2, "savefig.facecolor": "white", "savefig.transparent": False, "figure.facecolor": "white", "font.family": "DejaVu Sans", "svg.fonttype": "none"})


def canvas(height: float) -> tuple[Figure, Axes]:
    fig, ax = plt.subplots(figsize=(12, height))
    fig.subplots_adjust(left=0.015, right=0.985, bottom=0.025, top=0.985)
    ax.set(xlim=(0, 120), ylim=(0, height * 10))
    ax.axis("off")
    return fig, ax


def label(ax: Axes, x: float, y: float, text: str, size: float = 9, weight: str = "normal", align: str = "center", color: str = INK) -> None:
    ax.text(x, y, text, ha=align, va="center", fontsize=size, weight=weight, color=color, linespacing=1.5)


def box(ax: Axes, x: float, y: float, w: float, h: float, text: str, kind: str = "neutral", size: float = 9, dashed: bool = False) -> None:
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15,rounding_size=0.7", facecolor=COLORS[kind], edgecolor=LINE, linewidth=0.9, linestyle="--" if dashed else "-", zorder=2))
    label(ax, x + w / 2, y + h / 2, text, size)


def group(ax: Axes, x: float, y: float, w: float, h: float, title: str) -> None:
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=1", facecolor="white", edgecolor="#bcc4c9", linewidth=0.8, linestyle=(0, (2, 3)), zorder=0))
    label(ax, x + 1.5, y + h - 2.2, title, 9, "bold", "left")


def arrow(ax: Axes, points: list[tuple[float, float]], dashed: bool = False) -> None:
    for a, b in zip(points[:-2], points[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=LINE, lw=1.25, linestyle="--" if dashed else "-", zorder=1)
    ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="->", mutation_scale=11, linewidth=1.25, color=LINE, linestyle="--" if dashed else "-", zorder=1))


def tokens(ax: Axes, x: float, y: float, kind: str) -> None:
    for i in range(3):
        ax.add_patch(FancyBboxPatch((x + i * 1.8, y - 0.8), 1.3, 1.6, boxstyle="round,pad=0.1,rounding_size=0.2", facecolor=COLORS[kind], edgecolor=LINE, linewidth=0.7))


def save(fig: Figure, name: str) -> None:
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(OUT / f"{name}.{suffix}", dpi=300, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def sam3d() -> None:
    fig, ax = canvas(6.5)
    label(ax, 1, 63, "SAM 3D architecture", 14, "bold", "left")
    label(ax, 119, 63, "Redrawn from Figure 2 · SAM 3D Team, arXiv:2511.16624", 8, align="right")
    group(ax, 1, 32, 89, 27, "GEOMETRY MODEL")
    group(ax, 1, 5, 89, 24, "TEXTURE & REFINEMENT MODEL")
    label(ax, 8, 51, "Point map", 8)
    label(ax, 8, 43, "Image +\nobject mask", 8)
    box(ax, 18, 48, 16, 5, "Point encoder", "encoder")
    box(ax, 18, 40, 16, 6, "Image encoder", "encoder")
    label(ax, 36, 54.5, "optional", 7)
    arrow(ax, [(13, 51), (18, 51)])
    arrow(ax, [(13, 43), (18, 43)])
    arrow(ax, [(34, 50.5), (40, 50.5), (40, 43), (45, 43)])
    arrow(ax, [(34, 43), (45, 43)])
    label(ax, 8, 37, "Layout tokens", 7.5)
    tokens(ax, 16, 37, "neutral")
    arrow(ax, [(22, 37), (39, 37), (39, 40), (45, 40)])
    label(ax, 8, 34, "Shape tokens", 7.5)
    tokens(ax, 16, 34, "neutral")
    arrow(ax, [(22, 34), (42, 34), (42, 37), (45, 37)])
    box(ax, 45, 35, 17, 12, "Mixture of\nTransformers", "flow", 10)
    tokens(ax, 64, 44, "layout")
    tokens(ax, 64, 37, "shape")
    arrow(ax, [(62, 44), (64, 44)])
    arrow(ax, [(62, 37), (64, 37)])
    box(ax, 73, 48, 14, 6, "Layout decoder", "layout", 8)
    box(ax, 73, 35, 14, 6, "Shape decoder", "shape", 8)
    arrow(ax, [(69, 44), (71, 44), (71, 51), (73, 51)])
    arrow(ax, [(69, 37), (73, 38)])
    label(ax, 80, 56, "Rotation · Translation · Scale", 7)
    label(ax, 82, 32.8, "Coarse voxels", 7, align="left")
    label(ax, 8, 22, "Image +\nobject mask", 8)
    box(ax, 18, 19, 16, 6, "Image encoder", "encoder")
    arrow(ax, [(13, 22), (18, 22)])
    box(ax, 45, 12, 17, 11, "Flow\nTransformer", "flow", 10)
    arrow(ax, [(34, 22), (40, 22), (40, 20), (45, 20)])
    arrow(ax, [(80, 35), (80, 30.5), (55, 30.5), (55, 23)])
    label(ax, 8, 15, "Shape tokens", 7.5)
    tokens(ax, 16, 15, "neutral")
    arrow(ax, [(22, 15), (45, 15)])
    tokens(ax, 64, 17, "neutral")
    arrow(ax, [(62, 17), (64, 17)])
    box(ax, 74, 20, 13, 5, "Mesh decoder", "mesh", 8)
    box(ax, 74, 9, 13, 6, "Gaussian splat\ndecoder", "splat", 8)
    arrow(ax, [(69, 17), (71, 17), (71, 22.5), (74, 22.5)])
    arrow(ax, [(69, 17), (71, 17), (71, 12), (74, 12)])
    group(ax, 93, 32, 26, 27, "MIXTURE OF TRANSFORMERS")
    for x, kind, name in [(95, "shape", "Shape"), (107, "layout", "Layout")]:
        box(ax, x, 50, 10, 3.4, name + " tokens", kind, 7)
        box(ax, x, 45, 10, 3.4, "QKV", kind, 8)
        box(ax, x, 36, 10, 3.4, "FFN", kind, 8)
        box(ax, x, 32.8, 10, 2.5, "Cross-attention", kind, 6.6)
    box(ax, 95, 40.7, 22, 3.1, "Multimodal self-attention", "neutral", 7.5)
    group(ax, 93, 5, 26, 24, "SELF-ATTENTION MASK")
    for row in range(6):
        for col in range(6):
            ax.add_patch(Rectangle((99 + col * 2.35, 10 + (5-row) * 2.2), 2.15, 2, facecolor="#a7b3bd" if row >= 3 or col < 3 else "#eef1f3", edgecolor="white", linewidth=0.5))
    label(ax, 106, 8, "Shape    |    Layout  (keys)", 7)
    ax.text(96, 19.6, "Shape", rotation=90, ha="center", va="center", fontsize=7, color=INK)
    ax.text(96, 13, "Layout", rotation=90, ha="center", va="center", fontsize=7, color=INK)
    label(ax, 102.4, 13.2, "stop grad", 6.5)
    label(ax, 1, 1.7, "Structural redraw: model stages, token streams, decoders, and attention mask. Example object renders omitted.", 8, align="left")
    save(fig, "sam3d_architecture_redrawn")


def pooling() -> None:
    fig, ax = canvas(7.8)
    label(ax, 1, 75, "Mask-aware video pooling adapter", 14, "bold", "left")
    label(ax, 119, 75, "Implemented geometry pilot · same visual grammar as SAM 3D Figure 2", 8, align="right")
    group(ax, 1, 43, 118, 27, "VIDEO CONDITIONING")
    box(ax, 3, 56, 12, 7, "Video clip\n16 frames", "neutral")
    box(ax, 21, 56, 16, 7, "V-JEPA 2.1\nvideo encoder", "encoder")
    box(ax, 43, 56, 16, 7, "Foreground\ntoken selection", "encoder")
    box(ax, 66, 54, 20, 11, "Attentive pooler\n64 learned queries", "flow", 10)
    box(ax, 93, 56, 12, 7, "Linear\n768 → 1024", "encoder")
    tokens(ax, 110, 59.5, "encoder")
    for a,b in [(15,21),(37,43),(59,66),(86,93),(105,110)]:
        arrow(ax, [(a,59.5),(b,59.5)])
    label(ax, 29, 53.1, "Dense spatiotemporal tokens", 7.5)
    box(ax, 3, 45, 34, 5.5, "Aligned masks → 2 × 16 × 16 tubelets", "neutral", 8)
    arrow(ax, [(37,47.75),(51,47.75),(51,56)])
    label(ax, 52, 45.4, "Keep coverage ≥ 0.5", 7.5)
    label(ax, 76, 50.2, "Cross-attention over selected object tokens", 7.5)
    label(ax, 112.5, 54.5, "64 video tokens", 7.5)
    group(ax, 1, 11, 118, 28, "SAM 3D GEOMETRY MODEL")
    box(ax, 3, 24, 20, 7, "Reference mask\nDINOv2 encodings", "encoder")
    box(ax, 30, 23, 30, 10, "Conditioning tokens\n64 video + 2 × 1370 mask\n+ 2048 zero point-map slots", "encoder", 9)
    arrow(ax, [(23,27.5),(30,27.5)])
    arrow(ax, [(115,59.5),(117.5,59.5),(117.5,41),(45,41),(45,33)])
    label(ax, 68, 42, "Video tokens replace both original RGB streams", 8)
    box(ax, 68, 22, 20, 12, "Mixture of\nTransformers", "flow", 10)
    arrow(ax, [(60,28),(68,28)])
    label(ax, 17, 16, "Noisy shape + layout tokens", 8)
    tokens(ax, 39, 16, "shape")
    tokens(ax, 46, 16, "layout")
    arrow(ax, [(52,16),(78,16),(78,22)])
    box(ax, 96, 29, 19, 6, "Shape decoder", "shape", 9)
    box(ax, 96, 19, 19, 6, "Layout decoder", "layout", 9)
    arrow(ax, [(88,31),(96,32)])
    arrow(ax, [(88,25),(96,22)])
    label(ax, 105.5, 36.8, "Coarse geometry → mesh", 8)
    label(ax, 105.5, 16, "Layout head retained;\nno layout training target", 7.5)
    label(ax, 3, 8, "Training: pooler + projection; optional LoRA on V-JEPA and/or SAM 3D. Objective: canonical coarse-shape flow matching.", 8, align="left")
    label(ax, 3, 4.7, "Texture & refinement is outside this pilot. The baseline freezes pretrained backbones; LoRA variants adapt selected attention layers.", 8, align="left")
    label(ax, 3, 1.5, "Source: code/src/video3d/model.py (VideoConditioning), config.py. Architecture styling adapted from SAM 3D Figure 2.", 7.5, align="left")
    save(fig, "pooling_adapter_architecture")


if __name__ == "__main__":
    sam3d()
    pooling()
