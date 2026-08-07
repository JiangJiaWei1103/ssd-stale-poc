"""Variant C (fresh body + backfill + transient frontier fill) vs vanilla.

The decisive first experiment: does adding BACKFILL + a frontier fill recover the
accept length that the flawed variant-A (permanent uniform overwrite, no backfill)
destroyed? If C-repeat at lag>=1 lands far above variant-A's ~0.33 ratio (ideally
near vanilla), it confirms the reframe -- no-backfill was the structural killer --
and that the target-hidden tail-fill is a live lane.

Compares against results/variantA_blockbased/ (same lags, old mechanism), if present.

    modal run modal/app.py::crepeat            # full (N = config.N_PROMPTS)
    modal run modal/app.py::crepeat --n 2      # cheap smoke first (catch runtime errors)
"""
from __future__ import annotations

import json
from pathlib import Path

import config
import engine as eng
import metrics
import prompts as pr

LAGS = [0, 1, 2, 3]
DATASET = "gsm8k"
TEMP = 0.0


def _gen(engine, prompt_list):
    sp = {"temperature": TEMP, "max_new_tokens": config.MAX_NEW_TOKENS}
    outs = engine.generate(prompt_list, sp)
    return [outs] if isinstance(outs, dict) else outs


def _variantA_ratio(lag: int, results_dir: Path, field: str) -> "float | None":
    """(lag / lag0) ratio of `field` from the old variant-A run, for contrast."""
    a = results_dir / "variantA_blockbased" / f"lag{lag}__{DATASET}__t{TEMP}.json"
    a0 = results_dir / "variantA_blockbased" / f"lag0__{DATASET}__t{TEMP}.json"
    if not (a.exists() and a0.exists()):
        return None
    v = json.loads(a.read_text())[field]
    v0 = json.loads(a0.read_text())[field]
    return v / v0 if v0 else None


def run_c_repeat(results_dir: str = "results", n: "int | None" = None) -> dict:
    rd = Path(results_dir)
    out = rd / "C_repeat"
    out.mkdir(parents=True, exist_ok=True)
    n = n or config.N_PROMPTS
    pl = pr.load_prompts(DATASET)[:n]

    summ = {}
    for lag in LAGS:
        fp = out / f"lag{lag}.json"
        if fp.exists() and json.loads(fp.read_text()).get("n_requests") == n:
            summ[lag] = json.loads(fp.read_text())   # resume: this lag already done at N=n
            continue
        engine = eng.build_dspark_engine(
            lag_steps=lag, max_running_requests=config.BS_PRIMARY, fill_mode="repeat"
        )
        try:
            metas = [o["meta_info"] for o in _gen(engine, pl)]
            summ[lag] = {"lag": lag, "fill": "repeat", **metrics.summarize_cell(metas)}
        finally:
            engine.shutdown()
        fp.write_text(json.dumps(summ[lag], indent=2))

    base = summ[0]["mean_correct_drafts"]              # bonus-EXCLUDED (honest draft quality)
    base_incl = summ[0]["accept_length_incl_bonus"]    # bonus-INCLUDED (throughput-relevant)
    table = {}
    for lag in LAGS:
        c = summ[lag]["mean_correct_drafts"]
        c_incl = summ[lag]["accept_length_incl_bonus"]
        table[f"lag{lag}"] = {
            "n_prompts": n,
            "C_repeat_correct_excl": c,
            "C_repeat_ratio_excl": (c / base if base else 0.0),
            "C_repeat_accept_incl": c_incl,
            # throughput ratio -- compare to peer gap ~0.59 and the 0.5 break-even
            "C_repeat_ratio_incl": (c_incl / base_incl if base_incl else 0.0),
            "variantA_ratio_excl": _variantA_ratio(lag, rd, "mean_correct_drafts"),
            "variantA_ratio_incl": _variantA_ratio(lag, rd, "accept_length_incl_bonus"),
        }
    (out / "compare.json").write_text(json.dumps(table, indent=2))
    return {"summary": summ, "compare": table}
