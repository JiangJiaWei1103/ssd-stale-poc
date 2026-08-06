"""Run the whole thing on a GPU box that already has the sglang branch installed
(e.g. your pod). Same harness as the Modal launcher.

    python pod_run.py
"""
from __future__ import annotations

from experiment import run_all

if __name__ == "__main__":
    out = run_all(results_dir="results")
    print("gates:", [(g["gate"], g["ok"]) for g in out["gates"]])
    print(f"{len(out['summary'])} cells -> results/summary.json")
