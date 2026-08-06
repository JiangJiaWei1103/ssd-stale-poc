"""Prompt loaders. Fixed seed => fresh and stale runs see identical prompts.

Module is named `prompts` (NOT `datasets`) on purpose: `datasets` would shadow
the HuggingFace `datasets` library we import below.

NOTE: exact prompt parity with Future-Outlier's harness would require matching
his manifest.py selection. Here we mirror the datasets + N + a deterministic
seeded pick, which guarantees OUR fresh-vs-stale fairness. Cross-harness parity
is an open item (docs/).
"""
from __future__ import annotations

import random

import config


def _pick(items: "list[str]", n: int, seed: int) -> "list[str]":
    rng = random.Random(seed)
    idx = list(range(len(items)))
    rng.shuffle(idx)
    return [items[i] for i in idx[:n]]


def load_prompts(
    dataset: str, n: int = config.N_PROMPTS, seed: int = config.SUBSET_SEED
) -> "list[str]":
    from datasets import load_dataset  # HF library (installed in the image)

    if dataset == "gsm8k":
        ds = load_dataset("gsm8k", "main", split="test")
        prompts = [ex["question"] for ex in ds]
    elif dataset == "humaneval":
        ds = load_dataset("openai_humaneval", split="test")
        prompts = [ex["prompt"] for ex in ds]
    elif dataset == "mt_bench":
        # multi-turn; use the first turn as a single-shot prompt
        ds = load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
        prompts = [ex["prompt"][0] for ex in ds]
    else:
        raise ValueError(f"unknown dataset {dataset!r}; expected one of {config.DATASETS}")
    return _pick(prompts, n, seed)
