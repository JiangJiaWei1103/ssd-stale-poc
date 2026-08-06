"""Run cells + the three gates + orchestration.

Env-agnostic: works wherever sgl.Engine constructs (Modal container OR a GPU
pod). Launchers just call run_all(); nothing here knows about Modal.

Cost discipline: run_all loads EVERY dataset up front (CPU) before any engine,
so a dataset bug fails before a single GPU-second is spent. gates run before the
matrix; a gate failure aborts the matrix.
"""
from __future__ import annotations

import itertools
import json
import statistics
from pathlib import Path

import config
import engine as eng
import metrics
import prompts as pr


def run_cell(engine, prompt_list: "list[str]", temp: float) -> "list[dict]":
    sp = {"temperature": temp, "max_new_tokens": config.MAX_NEW_TOKENS}
    outs = engine.generate(prompt_list, sp)
    if isinstance(outs, dict):
        outs = [outs]
    return outs


# ---------------- gates (take preloaded prompts; build engines AFTER) ----------------

def gate_depth0_parity(prompts_by_ds: "dict[str, list[str]]") -> dict:
    """lag=0 (guarded off == vanilla) must be clearly speculating -> confirms the
    harness/wiring didn't break spec. Records the A10G fresh anchor."""
    pl = prompts_by_ds["gsm8k"][: config.LOSSLESS_N_PROMPTS]
    engine = eng.build_dspark_engine(lag_steps=0, max_running_requests=config.BS_PRIMARY)
    try:
        metas = [o["meta_info"] for o in run_cell(engine, pl, temp=0.0)]
        acc = metrics.mean_accept_length_incl_bonus(metas)
    finally:
        engine.shutdown()
    return {
        "gate": "depth0_parity",
        "ok": bool(acc >= config.PARITY_MIN_ACCEPT),
        "a10g_fresh_anchor": acc,
        "vs_runpod_baseline": config.BASELINE_ACCEPT_LENGTH,
        "delta": acc - config.BASELINE_ACCEPT_LENGTH,
    }


def _shared_prefix_len(a: "list[int]", b: "list[int]") -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def gate_losslessness(prompts_by_ds: "dict[str, list[str]]") -> dict:
    """Losslessness-by-construction check, FP-honest.

    Staleness changes only WHICH tokens the draft proposes -> accept length ->
    per-round chunking, NEVER the committed token VALUES (verify corrects every
    position to the target argmax). So fresh and stale must emit the SAME greedy
    stream.

    We do NOT demand bitwise-equal output_ids: at temp=0 a single near-tie argmax
    flip (block-verify numerics vs a later round's round differently) permanently
    diverges the autoregressive tail -- benign FP, NOT a bug (see the graph-eager
    / standalone-dp finding; exact diff at temp=0 was retired there for this exact
    reason). Instead we report per-prompt shared-prefix length: a REAL injection
    bug corrupts verify -> diverges EARLY+systematically -> tiny median prefix;
    benign FP -> LATE+sporadic -> large median prefix (many prompts even exact).

    Reference is fresh(lag0), NOT target-only: at lag0 the knob is OFF (enabled =
    lag>0), so fresh runs the ORIGINAL vanilla path and never touches our new
    cache -- no common-mode failure -- and inherits SGLang's own spec-correctness.
    stale vs fresh isolates exactly ONE variable: the injected hidden's age. (This
    also drops the spec-vs-nonspec FP confound the old target-only baseline stacked
    on top of the injection; same 2-engine cost, strictly cleaner attribution.)"""
    pl = prompts_by_ds["gsm8k"][: config.LOSSLESS_N_PROMPTS]
    fresh = eng.build_dspark_engine(lag_steps=0)
    try:
        ref_ids = [o["output_ids"] for o in run_cell(fresh, pl, temp=0.0)]
    finally:
        fresh.shutdown()
    stale = eng.build_dspark_engine(lag_steps=max(1, config.LAGS[-1]))
    try:
        test_ids = [o["output_ids"] for o in run_cell(stale, pl, temp=0.0)]
    finally:
        stale.shutdown()
    shared = [_shared_prefix_len(a, b) for a, b in zip(ref_ids, test_ids)]
    gen_len = [len(a) for a in ref_ids]
    exact = sum(1 for s, L in zip(shared, gen_len) if s == L)
    med = statistics.median(shared) if shared else 0
    return {
        "gate": "losslessness",
        "ok": bool(med >= config.LOSSLESS_MEDIAN_PREFIX_MIN),  # fail only on gross early breakage
        "n": len(pl),
        "median_shared_prefix": med,
        "floor": config.LOSSLESS_MEDIAN_PREFIX_MIN,
        "exact_match": exact,            # how many prompts were bitwise-identical (FYI)
        "shared_prefix": shared,         # per-prompt identical-prefix length
        "gen_len": gen_len,
        "note": "fresh(lag0) vs stale(lagmax) @temp0; late/sporadic divergence = benign FP, early+systematic = bug",
    }


def gate_keying(prompts_by_ds: "dict[str, list[str]]") -> dict:
    """bs>1 forces batch reorder; the rid-keyed cache must give the same accept
    length as bs=1. A gap => keying is cross-wiring requests (the peer's bs=1
    harness never tested this)."""
    pl = prompts_by_ds["gsm8k"]
    lag = max(1, config.LAGS[-1])
    out = {}
    for tag, bs in (("bs1", config.BS_PRIMARY), ("bsN", config.BS_KEYING)):
        engine = eng.build_dspark_engine(lag_steps=lag, max_running_requests=bs)
        try:
            metas = [o["meta_info"] for o in run_cell(engine, pl, temp=0.0)]
            out[tag] = metrics.summarize_cell(metas)["mean_correct_drafts"]
        finally:
            engine.shutdown()
    diff = abs(out["bs1"] - out["bsN"])
    return {"gate": "keying", "ok": bool(diff <= config.KEYING_TOL),
            "bs1": out["bs1"], "bsN": out["bsN"], "diff": diff, "tol": config.KEYING_TOL}


def run_gates(prompts_by_ds: "dict[str, list[str]]") -> "list[dict]":
    return [gate_depth0_parity(prompts_by_ds),
            gate_losslessness(prompts_by_ds),
            gate_keying(prompts_by_ds)]


# ---------------- deliverable matrix ----------------

def run_matrix(results_dir: Path, prompts_by_ds: "dict[str, list[str]]") -> dict:
    all_summ = {}
    for lag in config.LAGS:                              # outer: fresh engine per lag
        engine = eng.build_dspark_engine(lag_steps=lag, max_running_requests=config.BS_PRIMARY)
        try:
            for dataset, temp in itertools.product(config.DATASETS, config.TEMPS):
                metas = [o["meta_info"] for o in run_cell(engine, prompts_by_ds[dataset], temp)]
                key = f"lag{lag}__{dataset}__t{temp}"
                all_summ[key] = {"lag": lag, "dataset": dataset, "temp": temp,
                                 **metrics.summarize_cell(metas)}
                (results_dir / f"{key}.json").write_text(json.dumps(all_summ[key], indent=2))
        finally:
            engine.shutdown()
    for s in all_summ.values():                          # ratio vs fresh (lag=0)
        fresh = all_summ.get(f"lag0__{s['dataset']}__t{s['temp']}")
        s["ratio_vs_fresh"] = (
            metrics.ratio(s["mean_correct_drafts"], fresh["mean_correct_drafts"])
            if fresh else None
        )
    (results_dir / "summary.json").write_text(json.dumps(all_summ, indent=2))
    return all_summ


def run_all(results_dir: str = "results", gates_only: bool = False) -> dict:
    rd = Path(results_dir)
    rd.mkdir(parents=True, exist_ok=True)
    # CPU-cheap: load EVERY dataset up front, so a dataset bug fails before any GPU.
    prompts_by_ds = {d: pr.load_prompts(d) for d in config.DATASETS}
    gates = run_gates(prompts_by_ds)
    (rd / "gates.json").write_text(json.dumps(gates, indent=2))
    failed = [g["gate"] for g in gates if not g["ok"]]
    if failed:
        raise RuntimeError(
            f"gates failed: {failed} -- fix before trusting the matrix (results/gates.json)"
        )
    if gates_only:
        return {"gates": gates, "summary": None}
    summary = run_matrix(rd, prompts_by_ds)
    return {"gates": gates, "summary": summary}
