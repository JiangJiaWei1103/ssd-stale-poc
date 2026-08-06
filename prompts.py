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
        # bare "gsm8k" is rejected by recent datasets/hf_hub -> must be namespaced.
        # "main" config is required (a "socratic" config also exists).
        ds = load_dataset("openai/gsm8k", "main", split="test")
        prompts = [ex["question"] for ex in ds]
    elif dataset == "humaneval":
        ds = load_dataset("openai/openai_humaneval", split="test")
        prompts = [ex["prompt"] for ex in ds]
    elif dataset == "mt_bench":
        # multi-turn; use the first turn as a single-shot prompt
        ds = load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
        prompts = [ex["prompt"][0] for ex in ds]
    else:
        raise ValueError(f"unknown dataset {dataset!r}; expected one of {config.DATASETS}")
    return _pick(prompts, n, seed)


def validate_all(n: int = 2) -> None:
    """CPU-only, no GPU: confirm every dataset loads and the field access yields
    non-empty strings. Run locally (FREE) before ever touching Modal, and in the
    CPU download step so a dataset bug fails BEFORE any GPU is allocated:

        cd ssd-stale-poc && python -c "import prompts; prompts.validate_all()"
    """
    for d in config.DATASETS:
        got = load_prompts(d, n=n)
        assert got and all(isinstance(p, str) and p for p in got), f"{d}: bad prompts {got!r}"
        print(f"  {d}: OK ({len(got)} prompts, first={got[0][:60]!r})")
    print("all datasets OK")
