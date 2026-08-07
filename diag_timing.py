"""Measure t_draft / t_verify (per-forward GPU time) to settle the throughput
break-even for parallel DSpark. NO GPU accept-length here -- just two timings.

Uses DSpark's OWN CUDA-event instrumentation (no source edit): the env
SGLANG_DSPARK_DEBUG_DUMP with the draft_gpu_time / target_verify_gpu_time
components turns on segment timers that wrap EXACTLY

    * the draft block forward   -> dspark_worker_v2.py:560 segment(DRAFT)      -> draft_gpu_ms
    * the target verify forward -> dspark_worker_v2.py:631 segment(TARGET_VERIFY) -> target_verify_gpu_ms

per decode step. We run a short lag=0 generation (forward compute time does NOT
depend on lag -- lag only changes which hidden is injected in commit_hidden,
which is OUTSIDE the timed verify segment), then read them out of
engine.get_server_info() -> ... -> dspark_info_record.records[*].

Throughput break-even (derived analytically, no GPU): parallel DSpark beats the
existing sync DSpark iff
    A_stale/A_fresh  >  max(t_draft, t_verify) / (t_draft + t_verify)
Our C-repeat lag1 throughput ratio A_stale/A_fresh = 0.598 (results/C_repeat),
so the GO window on k = t_verify/t_draft is ((1-R)/R, R/(1-R)) = (0.67, 1.49).
This script measures k and drops it into that window.

    modal run modal/app.py::timing            # ~8 prompts, super light
    modal run modal/app.py::timing --n 4      # even lighter
"""
from __future__ import annotations

import json
import os
import statistics
from pathlib import Path

import config
import engine as eng
import prompts as pr

DATASET = "gsm8k"
TEMP = 0.0
N_PROMPTS = 8              # ~8 * (128/accept) ~ a few hundred decode-step samples
MAX_NEW_TOKENS = 128
WARMUP_DROP = 5           # drop first few decode records (cuda-graph / first-step outliers)

DEBUG_DUMP_ENV = "SGLANG_DSPARK_DEBUG_DUMP"
DEBUG_DUMP_VALUE = "core,draft_gpu_time,target_verify_gpu_time"


def _find_info_records(obj) -> "list[dict]":
    """Recursively pull every dspark_info_record.records[*] out of get_server_info()
    (internal_states nesting varies: list per rank / dict -> walk it generically)."""
    found: "list[dict]" = []

    def walk(o):
        if isinstance(o, dict):
            rec = o.get("dspark_info_record")
            if isinstance(rec, dict):
                found.extend(rec.get("records", []) or [])
            for v in o.values():
                walk(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                walk(v)

    walk(obj)
    return found


def _stale_ratio(results_dir: Path, default: float = 0.598) -> float:
    """A_stale/A_fresh (incl bonus) for lag1 from the C-repeat run, if present."""
    fp = results_dir / "C_repeat" / "compare.json"
    if fp.exists():
        try:
            return float(json.loads(fp.read_text())["lag1"]["C_repeat_ratio_incl"])
        except Exception:
            pass
    return default


def _go_window(ratio: float) -> "tuple[float, float]":
    return (1.0 - ratio) / ratio, ratio / (1.0 - ratio)


def run_timing(results_dir: str = "results", n: "int | None" = None) -> dict:
    # Must be set BEFORE engine construction: the spawned scheduler inherits
    # os.environ, and the dumper reads components at worker init.
    os.environ[DEBUG_DUMP_ENV] = DEBUG_DUMP_VALUE

    n = n or N_PROMPTS
    pl = pr.load_prompts(DATASET)[:n]

    engine = eng.build_dspark_engine(lag_steps=0, max_running_requests=config.BS_PRIMARY)
    try:
        engine.generate(pl, {"temperature": TEMP, "max_new_tokens": MAX_NEW_TOKENS})
        info = engine.get_server_info()          # pulls dspark_info_record.records
    finally:
        engine.shutdown()

    records = _find_info_records(info)
    draft = [r["draft_gpu_ms"] for r in records if r.get("draft_gpu_ms") is not None]
    verify = [
        r["target_verify_gpu_ms"] for r in records
        if r.get("target_verify_gpu_ms") is not None
    ]
    draft, verify = draft[WARMUP_DROP:], verify[WARMUP_DROP:]
    if not draft or not verify:
        keys = list(info.keys()) if isinstance(info, dict) else type(info).__name__
        raise RuntimeError(
            f"no timing records (draft={len(draft)}, verify={len(verify)}). "
            f"Is {DEBUG_DUMP_ENV} set + DSpark active? server_info top-level: {keys}"
        )

    t_draft = statistics.median(draft)
    t_verify = statistics.median(verify)
    k = t_verify / t_draft
    max_over_sum = max(t_draft, t_verify) / (t_draft + t_verify)
    ratio = _stale_ratio(Path(results_dir))
    lo, hi = _go_window(ratio)

    out = {
        "n_prompts": n,
        "n_decode_steps_used": len(draft),
        "t_draft_ms_median": round(t_draft, 4),
        "t_verify_ms_median": round(t_verify, 4),
        "t_draft_ms_mean": round(statistics.fmean(draft), 4),
        "t_verify_ms_mean": round(statistics.fmean(verify), 4),
        "k_verify_over_draft": round(k, 4),
        "max_over_sum": round(max_over_sum, 4),        # 0.598 must EXCEED this to be GO
        "A_stale_over_fresh": round(ratio, 4),
        "go_window_on_k": [round(lo, 3), round(hi, 3)],
        "k_in_go_window": bool(lo < k < hi),
        "verdict_lag1_parallel": "GO" if ratio > max_over_sum else "NO-GO",
        "note": "bs=1 regime (matches the accept-length run); bs>1 can shift the "
                "compute balance -> re-measure if the target deployment batches.",
    }
    rd = Path(results_dir) / "timing"
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "timing.json").write_text(json.dumps(out, indent=2))
    return out
