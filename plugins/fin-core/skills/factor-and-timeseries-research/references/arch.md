# arch — volatility models and the multiple-comparison procedures

`arch` **8.0.0 (2025-10-21)** · Kevin Sheppard · the reference implementation for GARCH-family
models in Python. ✅ 8.0.0 is a **compatibility release, not an API break**.

Two distinct things live in this package. The volatility half is well known; **the
`arch.bootstrap` half is the most under-used correct tool in quantitative finance.**

## 1. Volatility models

```python
from arch import arch_model
am = arch_model(returns * 100, vol='GARCH', p=1, q=1, dist='skewt')  # note the *100
res = am.fit(disp='off')
print(res.summary())
fc = res.forecast(horizon=5, reindex=False)
print(fc.variance.iloc[-1])
```

Supports GARCH, EGARCH, GJR-GARCH, APARCH, FIGARCH, HARCH/HAR, and Normal / Student-t / skew-t /
GED innovations, plus `arch.univariate.ARX` for a mean model.

**Gotchas:**
- 🚨 **Scale matters.** Fit on returns in **percent** (`× 100`). Raw decimal returns produce a
  badly conditioned optimization and convergence warnings — this is the single most common
  complaint about the package and it is a usage issue, not a bug.
- `forecast()` returns **variance**, not volatility. Take the square root, and remember to undo the
  ×100 scaling.
- `reindex=False` is what you almost always want; the default reindexes to the full sample.
- **HAR** on realized volatility remains a very strong baseline — the realized-volatility study in
  `_evidence-papers.md` found only one time-series foundation model beat Log-HAR, narrowly.

## 2. ⭐ `arch.bootstrap` — the multiple-comparison procedures

This answers *"is the best of my N strategies better than the benchmark, given that I searched over
N?"* — the question a bare Sharpe cannot answer.

```python
from arch.bootstrap import (
    SPA, RealityCheck, StepM, MCS,              # multiple-comparison procedures
    IIDBootstrap, IndependentSamplesBootstrap,
    StationaryBootstrap, CircularBlockBootstrap, MovingBlockBootstrap,
    optimal_block_length,                        # Politis-White
)
```

| Question | Class |
|---|---|
| Is the best of N better than the benchmark, accounting for the search? | **`RealityCheck`** (White 2000) — a zero-code subclass of `SPA` |
| Same, studentized and recentered so poor models don't distort it | **`SPA`** (Hansen 2005) |
| ***Which*** strategies beat the benchmark, with FWER control | **`StepM`** (Romano-Wolf) |
| The set of models indistinguishable from the best at 1−α | **`MCS`** (Hansen-Lunde-Nason 2011) |

## 🚨 They all take LOSSES, not returns

✅ Verified from `arch/bootstrap/multiple_comparison.py`: `SPA`/`StepM` document `benchmark` as
*"T element array of benchmark model **losses**"* and `models` as *"T by k element array of
alternative model **losses**"*; `MCS`'s parameter is literally named `losses`.

**Passing returns silently inverts every one of these tests** — it asks which strategy is *worst*.

```python
bm_loss  = -benchmark_returns          # (T,)    NEGATE
mdl_loss = -model_returns              # (T, k)  NEGATE

spa = SPA(bm_loss, mdl_loss, reps=1000, block_size=10, seed=7); spa.compute()
spa.pvalues            # {'lower', 'consistent', 'upper'}  -> report 'consistent'

stepm = StepM(bm_loss, mdl_loss, size=0.05, reps=1000, block_size=10); stepm.compute()
stepm.superior_models  # survive FWER control

mcs = MCS(mdl_loss, size=0.10, reps=1000, block_size=10); mcs.compute()
mcs.included, mcs.excluded

optimal_block_length(bm_loss)   # choose block_size from this, not by eye
```

**Other gotchas:**
- SPA's three p-values are a range — `lower` is liberal, `upper` conservative. **Report
  `consistent`.**
- `block_size` must preserve serial dependence; that is the whole point of a block bootstrap. Use
  `optimal_block_length`.
- Cost is O(reps × T × N) — 1000 reps over hundreds of strategies is minutes, not seconds.
- These procedures control the **family-wise error rate** over the models you pass. They do **not**
  know about the variants you tried and discarded — that is what the trial ledger and the Deflated
  Sharpe Ratio are for. Use both.

## When to reach for which

| Situation | Tool |
|---|---|
| "I have one strategy and want a p-value" | Not this — use PSR/DSR with an honest trial count |
| "I have 50 variants and want to know if any beat SPY" | `SPA` |
| "…and which ones" | `StepM` |
| "I want the surviving set of models" | `MCS` |
| "I want to forecast volatility" | `arch_model` |
