# ssd-stale-poc

Pre-experiment for **parallel (SSD-style) DSpark**: measure how much speculative-decoding
**accept length** degrades when the drafter consumes a **stale target hidden state** (lagged
by K decode steps) instead of the fresh one. This is the **go/no-go gate** for whether
parallelizing DSpark's draft+verify is worth pursuing — if accept length holds up under
staleness, parallel is worth building; if it collapses, it isn't.

## Mechanism under test — `value-stale`

Inject the target hidden computed **K rounds ago** at the **current** positions (RoPE current,
value old), **per-position**, keyed by request id. Contrast:

| mechanism | recent tokens' target signal | who |
|---|---|---|
| **value-stale** (this repo) | old but **real** target hidden | us |
| gap | **absent** for one round | Future-Outlier's DeepSpec harness |
| self_kv | the draft's **own** KV | Future-Outlier's DeepSpec harness |

ByteDance's internal parallel-DSpark is **value-stale family**. Sub-mechanism still to pin with
zhendonghua: position (current vs old), depth, aggressive-forward fill.

## The code change (lives in the sglang repo, not here)

On the sglang `decoupled-spec-e2e` branch:
- `SGLANG_DSPARK_TARGET_HIDDEN_LAG_STEPS` env var (0 = vanilla, K = K decode-steps stale)
- `TargetHiddenLagCache` — per-request rid-keyed ring, `.detach().clone()` snapshots,
  fresh cold-start fallback for the first K rounds of each request
- swap at `TargetVerifyExecutor.commit_hidden` (non-compact / static path only)

## What lives here

- `modal/` — one Modal entrypoint that launches every experiment cell
- `results/` — raw per-cell output
- `figs/` — plots
- `docs/` — write-up + open questions for the group

## Honest metric

Primary = `num_correct_drafts = accept_length - 1` (bonus-excluded); report the **ratio**
stale/fresh, the **distribution** (not just mean), and **exclude each request's cold-start
rounds** (which use the fresh fallback, not a stale hidden).

## Go/no-go rule of thumb

Parallel wins roughly iff:

    stale/fresh accept ratio  >  max(t_draft, t_verify) / (t_draft + t_verify)

(≈ 0.5 when draft and verify cost about the same.)

## Losslessness

Staleness is lossless **by construction**: verify (rejection sampling against the true target
logits) is untouched, so the output distribution is the target's at any staleness. The
`greedy_match` gate (temp=0 stale outputs == target-only greedy) only confirms the verify path
wasn't accidentally broken.
