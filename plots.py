"""results/summary.json -> figs/. Run locally after the matrix:
    python plots.py
"""
from __future__ import annotations

import json
from pathlib import Path


def plot(results_dir: str = "results", figs_dir: str = "figs") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summ = json.loads((Path(results_dir) / "summary.json").read_text())
    figs = Path(figs_dir)
    figs.mkdir(parents=True, exist_ok=True)

    # headline: mean_correct_drafts vs lag, one line per (dataset, temp)
    series: "dict[tuple, list]" = {}
    for s in summ.values():
        series.setdefault((s["dataset"], s["temp"]), []).append((s["lag"], s["mean_correct_drafts"]))

    plt.figure()
    for (dataset, temp), pts in sorted(series.items()):
        pts.sort()
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        plt.plot(xs, ys, marker="o", label=f"{dataset} t={temp}")
    plt.xlabel("staleness lag (decode steps)")
    plt.ylabel("mean correct drafts / step (bonus-excluded)")
    plt.title("Accept-length decay vs staleness (value-stale)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    out = figs / "accept_vs_lag.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    plot()
