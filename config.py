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

# --- the staleness knob (env var on the sglang decoupled-spec-e2e branch) ---
LAG_ENV = "SGLANG_DSPARK_TARGET_HIDDEN_LAG_STEPS"

# --- primary sweep (the headline: num_correct_drafts vs lag) ---
LAGS = [0, 1, 2, 3]                      # lag=0 == vanilla; extend {4,6,8} if not flat
DATASETS = ["gsm8k", "mt_bench", "humaneval"]   # easy / hard / code (mirror the peer's 3)
TEMPS = [0.0, 1.0]                       # greedy (smoothest) + sampling (context reversal)

N_PROMPTS = 32                           # per dataset (mirror the peer's 32 x 3 = 96)
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
LOSSLESS_N_PROMPTS = 8                   # a handful suffices for the greedy-match gate


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
