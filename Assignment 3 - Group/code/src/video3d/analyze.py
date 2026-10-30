"""`analyze`: tables and plots from eval results.jsonl files.

Paired differences against a baseline label on shared samples, cluster-bootstrap 95 % CIs
(resampling clips, so all objects and reference frames of a clip stay together), Wilcoxon
signed-rank tests, and metric vs. visible_frac / K / arc-span plots.
"""

from __future__ import annotations

import csv
import logging
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from video3d.config import ExperimentConfig, AnalysisConfig
from scipy.stats import wilcoxon

from video3d.utils import read_jsonl, setup_logging

log = logging.getLogger("video3d")
METRICS = ["viou", "viou_aligned", "chamfer", "fscore@0.02", "fscore@0.05"]
LOWER_IS_BETTER = {"chamfer"}


def load_results(paths: list[str | Path]) -> dict[str, list[dict]]:
    """{label: records}; one file may hold one label."""
    results: dict[str, list[dict]] = defaultdict(list)
    for path in paths:
        for record in read_jsonl(path):
            results[record["label"]].append(record)
    assert results, f"no records in {paths}"
    return dict(results)


def paired(base: list[dict], other: list[dict], metric: str) -> tuple[np.ndarray, np.ndarray]:
    """(other - base) per shared sample where both have the metric, and the sample's clip ids."""
    b = {r["sample_id"]: r for r in base if r.get(metric) is not None}
    pairs = [(r[metric] - b[r["sample_id"]][metric], r["clip_id"]) for r in other
             if r.get(metric) is not None and r["sample_id"] in b]
    if not pairs:
        return np.zeros(0), np.zeros(0, dtype=object)
    deltas, clips = zip(*pairs)
    return np.asarray(deltas, dtype=float), np.asarray(clips, dtype=object)


def cluster_bootstrap_ci(values: np.ndarray, clusters: np.ndarray, n_resamples: int,
                         seed: int) -> tuple[float, float]:
    """95 % CI of the mean, resampling whole clusters (clips) with replacement."""
    ids, inverse = np.unique(clusters, return_inverse=True)
    sums = np.bincount(inverse, weights=values, minlength=len(ids))
    counts = np.bincount(inverse, minlength=len(ids))
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(ids), size=(n_resamples, len(ids)))
    means = sums[pick].sum(1) / counts[pick].sum(1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def wilcoxon_p(deltas: np.ndarray) -> float:
    """Two-sided Wilcoxon signed-rank p-value (1.0 if every difference is zero)."""
    if len(deltas) == 0 or not np.any(deltas):
        return 1.0
    return float(wilcoxon(deltas).pvalue)


def summary_rows(results: dict[str, list[dict]]) -> list[dict]:
    """Mean of every metric per label."""
    rows = []
    for label, recs in results.items():
        row = {"label": label, "condition": recs[0]["condition"], "n": len(recs)}
        for m in METRICS:
            vals = [r[m] for r in recs if r.get(m) is not None]
            row[m] = float(np.mean(vals)) if vals else None
        rows.append(row)
    return rows


def delta_rows(results: dict[str, list[dict]], baseline: str, acfg: AnalysisConfig, seed: int) -> list[dict]:
    """Paired Δ vs. the baseline with cluster-bootstrap CI and Wilcoxon p, per label and metric."""
    assert baseline in results, f"baseline {baseline!r} not in {sorted(results)}"
    rows = []
    for label, recs in results.items():
        if label == baseline:
            continue
        for m in METRICS:
            d, clips = paired(results[baseline], recs, m)
            if len(d) == 0:
                continue
            lo, hi = cluster_bootstrap_ci(d, clips, acfg.bootstrap_resamples, seed)
            rows.append({"label": label, "vs": baseline, "metric": m, "n": len(d),
                         "mean_delta": float(d.mean()), "ci_lo": lo, "ci_hi": hi,
                         "wilcoxon_p": wilcoxon_p(d)})
    return rows


def binned_means(recs: list[dict], field: str, edges: list[float], metric: str) -> list[float | None]:
    """Mean metric in [edges[i], edges[i+1]) (last bin closed)."""
    out = []
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        last = i == len(edges) - 2
        vals = [r[metric] for r in recs if r.get(metric) is not None
                and r[field] is not None and lo <= r[field] and (r[field] <= hi if last else r[field] < hi)]
        out.append(float(np.mean(vals)) if vals else None)
    return out


def write_table(rows: list[dict], path: Path) -> None:
    """Write rows as CSV and as a markdown table next to it."""
    if not rows:
        return
    keys = list(rows[0])
    with open(path.with_suffix(".csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, keys)
        writer.writeheader()
        writer.writerows(rows)
    fmt = lambda v: f"{v:.4g}" if isinstance(v, float) else ("" if v is None else str(v))
    lines = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    lines += ["| " + " | ".join(fmt(r[k]) for k in keys) + " |" for r in rows]
    path.with_suffix(".md").write_text("\n".join(lines) + "\n")


def plot_bins(results: dict[str, list[dict]], field: str, edges: list[float], metric: str,
              path: Path) -> None:
    """One line per label: metric per bin of `field`."""
    names = [f"{lo:g}-{hi:g}" for lo, hi in zip(edges[:-1], edges[1:])]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for label, recs in sorted(results.items()):
        ys = binned_means(recs, field, edges, metric)
        ax.plot(names, [np.nan if y is None else y for y in ys], marker="o", label=label)
    ax.set_xlabel(field)
    ax.set_ylabel(metric)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_vs_k(results: dict[str, list[dict]], metric: str, path: Path) -> None:
    """One line per condition (adapter runs only): metric vs. clip length K."""
    by_run: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for recs in results.values():
        for r in recs:
            if r["run"] not in ("c0", "c0b_oracle") and not r["shuffle_frames"] and not r["wrong_video"]\
                    and r.get(metric) is not None:
                by_run[r["run"]][r["clip_len"]].append(r[metric])
    if not by_run:
        return
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for run, per_k in sorted(by_run.items()):
        ks = sorted(per_k)
        ax.plot(ks, [np.mean(per_k[k]) for k in ks], marker="o", label=run)
    ax.set_xlabel("K (frames)")
    ax.set_ylabel(metric)
    ax.set_xscale("log", base=2)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def analyze(cfg: ExperimentConfig, paths: list[str], baseline: str | None = None,
            out_dir: str | Path | None = None) -> Path:
    """Write summary / delta tables and plots for the given results files."""
    setup_logging()
    results = load_results(paths)
    baseline = baseline or next(iter(results))
    out = Path(out_dir or Path(cfg.paths.output_root) / "analysis")
    out.mkdir(parents=True, exist_ok=True)
    acfg = cfg.analysis
    write_table(summary_rows(results), out / "summary")
    write_table(delta_rows(results, baseline, acfg, int(cfg.seed)), out / f"delta_vs_{baseline}")
    for metric in ("viou", "viou_aligned", "fscore@0.05"):
        plot_bins(results, "visible_frac", list(acfg.visible_frac_bins), metric,
                  out / f"{metric}_vs_visible_frac.png")
        plot_bins(results, "arc_span_deg", list(acfg.arc_span_bins), metric,
                  out / f"{metric}_vs_arc_span.png")
        plot_vs_k(results, metric, out / f"{metric}_vs_k.png")
    log.info("wrote tables and plots to %s", out)
    return out
