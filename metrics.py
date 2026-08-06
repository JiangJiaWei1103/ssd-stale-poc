"""Honest accept-length metrics from per-request meta_info.

Per-request meta_info (populated whenever speculative_algorithm is set; no
--enable-metrics needed) gives us, per request:
  - spec_correct_drafts_histogram : List[int], index k = #decode steps that
    accepted exactly k CORRECT drafts (bonus EXCLUDED)   <- primary signal
  - spec_accept_length            : completion_tokens / spec_verify_ct
                                    (mean committed tokens/step, INCL bonus)
  - spec_num_correct_drafts, spec_verify_ct

Primary metric = mean CORRECT drafts per decode step (bonus-excluded), pooled
across requests via the ready-made histogram. accept_length (incl bonus) is
kept only as a cross-check against the ~4.07 baseline.

Cold-start note: the first `lag` decode rounds of each request use the fresh
fallback (no lag-old hidden yet). meta_info aggregates per REQUEST, not per
round, so we do NOT cheaply drop those rounds; the bias is bounded by roughly
lag / (tokens / accept_len) -- a few percent for lag<=3 at 256 tokens -- and is
reported, not excluded (see docs/, and the user's "fallback fresh 可以丟掉").
"""
from __future__ import annotations


def pool_histograms(metas: "list[dict]") -> "list[int]":
    """Element-wise sum of per-request spec_correct_drafts_histogram."""
    pooled: "list[int]" = []
    for m in metas:
        h = m.get("spec_correct_drafts_histogram") or []
        if len(h) > len(pooled):
            pooled += [0] * (len(h) - len(pooled))
        for k, c in enumerate(h):
            pooled[k] += int(c)
    return pooled


def mean_correct_drafts(pooled: "list[int]") -> float:
    """Bonus-excluded: mean number of correct drafts per decode step."""
    total_steps = sum(pooled)
    if total_steps == 0:
        return 0.0
    return sum(k * c for k, c in enumerate(pooled)) / total_steps


def mean_accept_length_incl_bonus(metas: "list[dict]") -> float:
    """Cross-check vs the ~4.07 baseline (includes the +1 bonus)."""
    vals = [m["spec_accept_length"] for m in metas if m.get("spec_verify_ct", 0) > 0]
    return sum(vals) / len(vals) if vals else 0.0


def summarize_cell(metas: "list[dict]") -> dict:
    pooled = pool_histograms(metas)
    return {
        "n_requests": len(metas),
        "total_steps": sum(pooled),
        "mean_correct_drafts": mean_correct_drafts(pooled),          # PRIMARY (no bonus)
        "accept_length_incl_bonus": mean_accept_length_incl_bonus(metas),  # cross-check
        "correct_drafts_histogram": pooled,                          # index k = #steps w/ k correct
    }


def ratio(stale_mean: float, fresh_mean: float) -> float:
    return stale_mean / fresh_mean if fresh_mean else 0.0
