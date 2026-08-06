"""Run cells + the three gates + orchestration.

Env-agnostic: works wherever sgl.Engine constructs (Modal container OR a GPU
pod). Launchers just call run_all(); nothing here knows about Modal.

Flow (1 launch): gates first -> abort if any fail -> then the deliverable
matrix (outer=lag needs a fresh engine; inner=dataset x temp reuses it).
"""
from __future__ import annotations

import itertools
import json
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


# ---------------- gates ----------------

def gate_depth0_parity() -> dict:
    """lag=0 (guarded off == vanilla) must be clearly speculating -> confirms the
    harness/wiring didn't break spec. Records the A10G fresh anchor."""
    engine = eng.build_dspark_engine(lag_steps=0, max_running_requests=config.BS_PRIMARY)
    try:
        pl = pr.load_prompts("gsm8k", n=config.LOSSLESS_N_PROMPTS)
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


def gate_losslessness() -> dict:
    """temp=0: stale-DSpark output_ids must equal target-only greedy, token for
    token. Lossless by construction (verify untouched) -- a mismatch is a BUG in
    the injection, not a staleness effect. Engines run sequentially (24GB A10G
    can't hold target-only + target+draft at once)."""
    pl = pr.load_prompts("gsm8k", n=config.LOSSLESS_N_PROMPTS)
    ref = eng.build_target_only_engine()
    try:
        ref_ids = [o["output_ids"] for o in run_cell(ref, pl, temp=0.0)]
    finally:
        ref.shutdown()
    stale = eng.build_dspark_engine(lag_steps=max(1, config.LAGS[-1]))
    try:
        test_ids = [o["output_ids"] for o in run_cell(stale, pl, temp=0.0)]
    finally:
        stale.shutdown()
    mism = [i for i, (a, b) in enumerate(zip(ref_ids, test_ids)) if a != b]
    return {"gate": "losslessness", "ok": len(mism) == 0, "n": len(pl), "mismatch_idx": mism}


def gate_keying() -> dict:
    """bs>1 forces batch reorder; the rid-keyed cache must give the same accept
    length as bs=1. A gap => keying is cross-wiring requests (the peer's bs=1
    harness never tested this)."""
    pl = pr.load_prompts("gsm8k", n=config.N_PROMPTS)
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


def run_gates() -> "list[dict]":
    return [gate_depth0_parity(), gate_losslessness(), gate_keying()]


# ---------------- deliverable matrix ----------------

def run_matrix(results_dir: Path) -> dict:
    all_summ = {}
    for lag in config.LAGS:                              # outer: fresh engine per lag
        engine = eng.build_dspark_engine(lag_steps=lag, max_running_requests=config.BS_PRIMARY)
        try:
            for dataset, temp in itertools.product(config.DATASETS, config.TEMPS):
                pl = pr.load_prompts(dataset)
                metas = [o["meta_info"] for o in run_cell(engine, pl, temp)]
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


def run_all(results_dir: str = "results") -> dict:
    rd = Path(results_dir)
    rd.mkdir(parents=True, exist_ok=True)
    gates = run_gates()
    (rd / "gates.json").write_text(json.dumps(gates, indent=2))
    failed = [g["gate"] for g in gates if not g["ok"]]
    if failed:
        raise RuntimeError(
            f"gates failed: {failed} -- fix before trusting the matrix (results/gates.json)"
        )
    summary = run_matrix(rd)
    return {"gates": gates, "summary": summary}
