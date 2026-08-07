"""Plumbing-only diagnostic: prove the lag-cache path doesn't corrupt hidden.

Set lag_steps HUGE (> max decode rounds for 256 tokens). Then get_lagged's ring
never fills, so it ALWAYS hits the cold-start branch and returns the *fresh* row --
but it does so through the full snapshot -> stack -> reshape -> inject path. So an
"identity" run exercises every line of the plumbing with fresh values.

  identity(lagBIG) == baseline(lag0)  -> plumbing is byte-clean; the accept-length
                                         drop at lag>=1 is a REAL staleness effect.
  identity(lagBIG) <  baseline(lag0)  -> the snapshot/stack/reshape/inject wiring
                                         corrupts hidden -> a bug, fix before trusting
                                         any matrix number.

Same prompts, same temp=0, both inject FRESH -> they should be ~identical (only a
tiny FP delta from the view/stack/reshape round-trip). A big gap is a smoking gun.

    modal run modal/app.py::diag        # one launch, one A10G function
"""
from __future__ import annotations

import json
from pathlib import Path

import config
import engine as eng
import metrics
import prompts as pr

IDENTITY_LAG = 500   # >> max decode rounds for 256 new tokens -> ring never fills -> always fresh
N = 16               # same prompts for both runs; we expect them ~equal, so N=16 is plenty
TOL = 0.15           # keying-gate spirit: |identity - baseline| under this == clean


def _gen(engine, prompt_list):
    sp = {"temperature": 0.0, "max_new_tokens": config.MAX_NEW_TOKENS}
    outs = engine.generate(prompt_list, sp)
    return [outs] if isinstance(outs, dict) else outs


def run_plumbing_check(results_dir: str = "results") -> dict:
    rd = Path(results_dir)
    rd.mkdir(parents=True, exist_ok=True)
    pl = pr.load_prompts("gsm8k")[:N]

    out = {}
    for tag, lag in (("baseline_lag0", 0), ("identity_lagBIG", IDENTITY_LAG)):
        engine = eng.build_dspark_engine(lag_steps=lag, max_running_requests=config.BS_PRIMARY)
        try:
            metas = [o["meta_info"] for o in _gen(engine, pl)]
            out[tag] = {"lag": lag, **metrics.summarize_cell(metas)}
        finally:
            engine.shutdown()

    b = out["baseline_lag0"]["mean_correct_drafts"]
    i = out["identity_lagBIG"]["mean_correct_drafts"]
    out["verdict"] = {
        "n_prompts": N,
        "identity_lag": IDENTITY_LAG,
        "baseline": b,
        "identity": i,
        "abs_diff": abs(b - i),
        "tol": TOL,
        "plumbing_clean": bool(abs(b - i) <= TOL),
        "note": "clean -> lag>=1 drop is REAL staleness; not clean -> wiring bug in snapshot/stack/reshape/inject",
    }
    (rd / "plumbing.json").write_text(json.dumps(out, indent=2))
    return out
