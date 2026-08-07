"""Single source of truth for the stale-hidden ablation matrix.

Pure stdlib -- safe to import anywhere (no torch/sglang).
"""
from __future__ import annotations

from dataclasses import dataclass

# --- models (A10G-friendly official DSpark pair) ---
TARGET_MODEL = "Qwen/Qwen3-4B"
DRAFT_MODEL = "deepseek-ai/dspark_qwen3_4b_block7"
# Draft block size gamma; verify window = gamma+1. None -> auto-infer from the
# draft checkpoint (recommended; block7 => gamma=7).
DSPARK_BLOCK_SIZE = None

# --- the staleness knob (env vars on the sglang dspark-stale-hidden-ablation branch) ---
LAG_ENV = "SGLANG_DSPARK_TARGET_HIDDEN_LAG_STEPS"   # lag in rounds; 0 = vanilla
# Variant-C frontier fill when lag>=1: "repeat" (last real hidden), "gap", "self_kv".
STALE_FILL_ENV = "SGLANG_DSPARK_STALE_FILL_MODE"
FILL_MODE = "repeat"

# --- primary sweep (the headline: num_correct_drafts vs lag) ---
LAGS = [0, 1, 2, 3]                      # lag=0 == vanilla; extend {4,6,8} if not flat
DATASETS = ["gsm8k", "mt_bench", "humaneval"]   # easy / hard / code (mirror the peer's 3)
TEMPS = [0.0, 1.0]                       # greedy (smoothest) + sampling (context reversal)

# per dataset. NOTE: accept length is a PER-STEP metric -- each 256-token gen is
# ~256/accept ~= 60 decode steps, so N prompts ~= N*60 step-samples per cell (64
# -> ~4000). That's why this is far below the ~200 used for per-PROMPT accuracy
# parity (different unit). Bootstrap CIs (metrics.bootstrap_ci, resampled over
# prompts) make precision explicit; bump if a CI is too wide (cost is trivial).
N_PROMPTS = 64
MAX_NEW_TOKENS = 256                     # mirror the peer harness

# fixed seeds => fresh and stale runs see identical prompts (fair comparison)
SUBSET_SEED = 20260805                    # mirror peer's subset seed (prompt parity)
DECODE_SEED = 980406                      # mirror peer's decode seed

# batch size = max concurrent running requests
BS_PRIMARY = 1                           # clean per-request measurement, no reorder
BS_KEYING = 8                            # keying gate: forces batch reorder

# --- engine construction (A10G is pre-Hopper: triton, NOT fa3) ---
ATTENTION_BACKEND = "triton"
PAGE_SIZE = 1
MEM_FRACTION_STATIC = 0.8

# --- gates ---
BASELINE_ACCEPT_LENGTH = 4.07            # RunPod Step-0 fresh anchor (incl bonus); re-anchored on A10G
PARITY_MIN_ACCEPT = 2.5                  # lag=0 must be clearly speculating (harness didn't break spec)
KEYING_TOL = 0.15                        # |bs>1 mean_correct - bs1| tolerance
LOSSLESS_N_PROMPTS = 8                   # a handful suffices for the prefix-agreement gate
# Losslessness gate compares fresh(lag0) vs stale(lagmax) greedy output_ids and
# measures the shared-prefix length. It does NOT demand bitwise equality: at
# temp=0 a single FP near-tie argmax flip permanently diverges the tail (benign,
# see the graph-eager-diff / standalone-dp finding). A REAL injection bug corrupts
# verify and diverges EARLY+systematically -> tiny median shared prefix. This
# floor only trips on gross breakage; subtle corruption is the accuracy-parity
# SHOULD's job. PROVISIONAL -- recalibrate against the first real distribution.
LOSSLESS_MEDIAN_PREFIX_MIN = 8           # fail only if median shared prefix < this (gross-breakage floor)


@dataclass(frozen=True)
class Cell:
    lag: int
    dataset: str
    temp: float
    bs: int = BS_PRIMARY


def matrix() -> "list[Cell]":
    """The MUST deliverable: lag x dataset x temp at bs=1."""
    return [
        Cell(lag=lag, dataset=ds, temp=t, bs=BS_PRIMARY)
        for lag in LAGS
        for ds in DATASETS
        for t in TEMPS
    ]
