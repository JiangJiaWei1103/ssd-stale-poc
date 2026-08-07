# ssd-stale-poc

**Go/no-go pre-experiment for parallel (SSD-style) DSpark.** In parallel DSpark the drafter
runs *ahead* of the verifier, so it can only consume a **stale target hidden state** (the fresh
one hasn't been computed yet). Question: *does accept length survive that staleness well enough
that overlapping draft+verify is a net throughput win?*

---

## ⛔ VERDICT: NO-GO (Qwen3-4B + dspark_qwen3_4b_block7, static, bs=1)

**One number decides it — the throughput ratio of the two designs:**

```
throughput_parallel   A_stale / max(t_draft, t_verify)   2.856 / 19.763     0.145 tok/ms
───────────────────  = ─────────────────────────────── = ─────────────── = ───────────── = 0.76  < 1
throughput_sync        A_fresh / (t_draft + t_verify)     4.776 / 25.195     0.190 tok/ms
```

`< 1` → **parallel DSpark is ~24% SLOWER than the existing sync DSpark.** Wrapping up here.

**Why (root cause):** verify (19.8 ms, a full 4B target forward) is **3.6× the draft** (5.4 ms, a
lightweight block head). Overlap can only hide the *smaller* stage, so it saves just **+27%**
wall-clock — but even the best stale fill loses **40%** of accept length at lag-1. The loss
swamps the gain, and no fill policy can close a gap that large (see break-even below).

---

## The mechanism under test — variant **C** (fresh body + transient frontier hole)

Real parallel = the verifier lags, so the **last `lag` committed positions have no target hidden
yet** (a moving *frontier hole*), while the body keeps its **fresh, correct** hiddens that arrive
one round late and **backfill**. That is variant **C**, and it is *transient* — every position
eventually gets its real hidden.

> An earlier "variant A" (overwrite *every* position with a K-steps-old hidden, never backfill) was
> **retired**: it's a permanent uniform corruption that matches no real parallel regime. Full design
> space (A/B/C, RoPE phase, tail-fill, the peer's gap/self_kv) → [`docs/design-options.md`](docs/design-options.md).

The harness is **synchronous vanilla DSpark** that *simulates* the lag by separating when a hidden
is **computed** from when it's **used** (a per-request delay queue). `lag=0` is byte-identical vanilla.

```mermaid
sequenceDiagram
    autonumber
    participant D as Draft
    participant V as Verify
    participant Q as LagQueue (maxlen 2)
    participant KV as Draft-KV pool
    Note over D,KV: lag=1, steady state — round 6
    KV->>D: draft reads — body real up to B4, B5 = HOLE (1 block)
    V->>Q: COMPUTE h6 (on time) → queue; NOT used this round
    Q->>KV: USE h5 (made last round) → BACKFILL @ B5 (real, own slot)
    V->>KV: FILL repeat(last real) → B6  (placeholder = new 1-block hole)
    Note over D,KV: hole width = lag; each hidden lands lag rounds after its own round
```

**Fill policies for the hole** (what the draft attends where the real hidden isn't ready yet):
`repeat` = last real target hidden (implemented) · `gap` = empty (peer's) · `self_kv` = draft's own
K/V (peer's) · `extrapolate` = predict the hidden forward = ByteDance "aggressive forward" (our
differentiated arm, **not yet built**).

---

## Results

### 1 · Accept length vs staleness — C-repeat, N=64, gsm8k, temp=0

| lag | num_correct_drafts (excl bonus) | accept_len (incl bonus) | **incl ratio (throughput)** |
|:---:|:---:|:---:|:---:|
| 0 | 3.66 | 4.78 | **1.00** |
| 1 | 1.77 | 2.86 | **0.598** |
| 2 | 0.97 | 2.05 | 0.430 |
| 3 | 0.68 | 1.76 | 0.369 |

lag-1 incl **0.598 ≈ the peer's `gap` ≈ 0.59** → the C frame is **cross-validated across two
independent harnesses**. First-reject rate climbs 16 → 49 → 65 → 71 %; full-block-accept collapses
29 → 11 → 4 → 2 %. (3 gates pass: depth-0 parity 4.53, losslessness median-prefix 105.5/256, keying
Δ0.007. Plumbing lag=500 == vanilla **bitwise** → the drop is real, not wiring.)

### 2 · Per-forward GPU time — bs=1 (DSpark's own CUDA-event instrumentation, no source edit)

| t_draft | t_verify | k = t_verify/t_draft |
|:---:|:---:|:---:|
| 5.4 ms | 19.8 ms | **3.64** |

### 3 · Throughput break-even

Parallel beats sync **iff** `A_stale/A_fresh > max(t_d,t_v)/(t_d+t_v)`. The RHS as a function of the
balance `k` gives a **GO window** on `k` (endpoints are reciprocals — the condition is symmetric under
swapping draft↔verify):

```
GO  ⟺  ratio > max(k,1)/(1+k)   ⟺   k ∈ ( (1-R)/R , R/(1-R) ) = (0.67, 1.49)   with R = 0.598
measured k = 3.64  →  FAR outside the window  →  NO-GO   (needed ratio > 0.784, have 0.598)
```

Draft/verify must be **balanced within ~1.5×** for lag-1 parallel to pay off; they're 3.6× apart.

---

## What could flip it (open levers, in priority order)

1. **`extrapolate` fill** — the only fill with a theoretical path to `ratio > 0.784` (1st-order
   forward-predict vs `repeat`'s 0th-order hold; at lag-1 the trajectory is only 1 step). It's our
   differentiated contribution. But 0.598 → 0.784 is a big jump; **long shot**, needs implementation.
2. **A vehicle where draft ≈ verify** — parallel only pays when the two stages are balanced. DSpark's
   full-target-verify vs light-block-draft is structurally imbalanced. (Timing ratio `k` ≈ the
   draft/target **capacity ratio** and is roughly bs-invariant — the draft is a single block forward,
   not γ autoregressive steps — so bs>1 is very unlikely to flip it; a single `bs=16` timing run would
   confirm.)

**Losslessness is not a lever** — it's free: verify (rejection sampling vs the true target logits) is
untouched, so the output distribution is the target's at *any* staleness. Staleness only moves the
accept *rate* (throughput), never correctness.

---

## Honest metrics (measurement-integrity, aka Q4)

- **Draft quality** → `num_correct_drafts = accept_len − 1` (**excl** bonus). The +1 bonus is the
  target's own guaranteed token; including it floors accept at 1.0 and flatters stale.
- **Throughput** → `accept_len` (**incl** bonus) — the bonus is a real committed token. This is the
  0.598 that feeds the break-even, *not* the excl ratio 0.484.
- Report the **histogram**, not just the mean (lag-1 is bimodal). **Exclude each request's first `lag`
  cold-start rounds** (they inject fresh, not stale).

---

## How to run (Modal, A10G, ~1 min each; models pre-cached on the volume)

```bash
modal run modal/app.py::crepeat            # accept-length curve (C-repeat, lag 0-3)   → results/C_repeat/
modal run modal/app.py::timing             # t_draft / t_verify + throughput verdict   → results/timing/
modal run modal/app.py::diag               # plumbing identity check (lag=500 == vanilla)
modal run modal/app.py                      # full gate + matrix
# download:  modal volume get ssd-stale-results /<remote> ./results/<local>
```

## Repo layout

| path | what |
|---|---|
| `config.py` / `engine.py` / `prompts.py` / `metrics.py` / `experiment.py` | harness core |
| `diag_c.py` | C-repeat accept curve vs vanilla |
| `diag_timing.py` | t_draft / t_verify via `SGLANG_DSPARK_DEBUG_DUMP` → `get_server_info()` |
| `diag_plumbing.py` | lag=500 identity (proves the drop isn't a wiring bug) |
| `modal/app.py` | one launcher, all cells; results land on the `ssd-stale-results` volume |
| `docs/design-options.md` | full mechanism design space + open questions for the group |
| `results/C_repeat/` | the headline accept curve · `results/timing/` timing · `results/variantA_blockbased/` retired |

## The sglang-side change (lives in the sglang repo, branch `dspark-stale-hidden-ablation`)

- env `SGLANG_DSPARK_TARGET_HIDDEN_LAG_STEPS` (0 = vanilla, K = K-round stale) +
  `SGLANG_DSPARK_STALE_FILL_MODE` (`repeat` implemented; `gap`/`self_kv` stubbed).
- `TargetHiddenLagCache` (`dspark_hidden_lag_cache.py`) — per-rid delay queue: emits, per round,
  the delayed **BACKFILL** (real hidden at its own old slots) + the frontier **FILL**.
- seam: `TargetVerifyExecutor.commit_hidden` (`dspark_verify.py`) — `lag=0` early-returns byte-vanilla.

---

## What this isolated experiment **cannot** tell you (→ real parallel system only)

- **Communication cost** — `t_parallel = max(t_d,t_v)` ignores shipping the hidden verifier→drafter.
  The idealized model is an *upper bound*; comm cost only makes it worse (here it's already NO-GO).
- **The exact `lag` ↔ real pipeline depth** mapping (±1 round, depends on the real system's delivery
  latency — a design detail to pin with zhendonghua).
- **Production batching** — `k` measured at bs=1; the go/no-go should ultimately reflect the deployment
  batch regime.
