"""Build offline sgl.Engine handles.

The lag env var is set BEFORE construction (it is read at worker init, and the
spawned scheduler subprocess inherits os.environ), so each lag value needs a
fresh engine. Verify mode is pinned to `static` (Step-3 confound control).
"""
from __future__ import annotations

import os

import config


def build_dspark_engine(lag_steps: int, max_running_requests: int = config.BS_PRIMARY):
    """DSpark spec decoding with staleness = lag_steps (0 == vanilla)."""
    import sglang as sgl

    os.environ[config.LAG_ENV] = str(lag_steps)          # spawned scheduler inherits this
    os.environ["SGLANG_RAGGED_VERIFY_MODE"] = "static"   # pin verify-all (no budget confound)
    kwargs = dict(
        model_path=config.TARGET_MODEL,
        speculative_algorithm="DSPARK",
        speculative_draft_model_path=config.DRAFT_MODEL,
        attention_backend=config.ATTENTION_BACKEND,
        speculative_draft_attention_backend=config.ATTENTION_BACKEND,
        page_size=config.PAGE_SIZE,
        mem_fraction_static=config.MEM_FRACTION_STATIC,
        max_running_requests=max_running_requests,
        random_seed=config.DECODE_SEED,
    )
    if config.DSPARK_BLOCK_SIZE is not None:
        kwargs["speculative_dspark_block_size"] = config.DSPARK_BLOCK_SIZE
    return sgl.Engine(**kwargs)


def build_target_only_engine(max_running_requests: int = config.BS_PRIMARY):
    """Same target, speculative decoding DISABLED -- the losslessness reference."""
    import sglang as sgl

    os.environ.pop(config.LAG_ENV, None)
    return sgl.Engine(
        model_path=config.TARGET_MODEL,
        attention_backend=config.ATTENTION_BACKEND,
        page_size=config.PAGE_SIZE,
        mem_fraction_static=config.MEM_FRACTION_STATIC,
        max_running_requests=max_running_requests,
        random_seed=config.DECODE_SEED,
    )
