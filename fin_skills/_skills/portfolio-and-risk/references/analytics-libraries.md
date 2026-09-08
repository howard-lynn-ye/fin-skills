# Performance analytics libraries — status and measured bugs

Verified 2026-09-03. The metric numbers below were produced by **executing** the libraries on the
same seeded 1,260-day return series, not read from docs.

## Status

| Library | pip | Version (date) | Licence | ★ / issues | Default-branch commit | Verdict |
|---|---|---|---|---|---|---|
| **ffn** | `ffn` | **1.1.5** (2026-03-24) | MIT | 2,638 / **7** | **2026-09-01** | ✅ **best-tended in the group** |
| **quantstats** | `QuantStats` | 0.0.81 (2026-01-13) | Apache-2.0 | 7,612 / 32 | 2026-01-13 | ⚠️ **burst-maintained** |
| quantstats-lumi | `quantstats-lumi` | 1.1.5 (2026-06-01) | Apache-2.0 | 152 / 41 | 2026-06-01 | active fork, heavy backlog |
| quantstats-reloaded | `quantstats-reloaded` | 0.1.0 (2025-06-16) | Apache-2.0 | — | — | third fork, low adoption |
| pyfolio-reloaded | `pyfolio-reloaded` | 0.9.9 (2025-06-02) | Apache-2.0 | 612 / 17 | 2025-06-02 | ⚠️ maintenance mode |
| empyrical-reloaded | `empyrical-reloaded` | 0.5.12 (2025-06-01) | Apache-2.0 | 121 / 6 | 2025-07-29 | ⚠️ maintenance mode |
| alphalens-reloaded | `alphalens-reloaded` | 0.4.6 (2025-06-02) | Apache-2.0 | 642 / 14 | 2025-06-02 | ⚠️ maintenance mode |
| **arch** | `arch` | 8.0.0 (2025-10-21) | **NCSA** | 1,558 / 51 | 2026-08-10 | ✅ active (Kevin Sheppard) |
| 🔴 pyfolio | `pyfolio` | 0.9.2 (**2019**) | Apache-2.0 | — | — | **dead — use `-reloaded`** |
| 🔴 empyrical | `empyrical` | 0.5.5 (**2020**) | Apache-2.0 | — | — | **dead — use `-reloaded`** |
| 🔴 alphalens | `alphalens` | 0.4.0 (**2020**) | Apache-2.0 | — | — | **dead — use `-reloaded`** |

⚠️ **The "-reloaded" family is frozen, not maintained.** GitHub's `pushed_at` is misleading (it counts
branch/tag pushes). Actual **default-branch** commits: pyfolio-reloaded 2025-06-02, alphalens-reloaded
2025-06-02, empyrical-reloaded 2025-07-29 — **~14 months of no substantive work.** Stefan Jansen keeps
them compiling against new NumPy/pandas but is not developing them. **Safe but frozen: they will work,
they will not improve.**

⚠️ **quantstats' maintenance pattern matters** because of the bugs below. Releases 0.0.74–0.0.77 in
Aug–Sep 2025, then silence, then a burst on **2026-01-13** where 0.0.78 ("2026 Modernization Update")
shipped with circular-import errors hot-fixed through 0.0.79, 0.0.80 and **0.0.81 the same day**. No
default-branch commits since. **The metric bugs below have survived multiple such sprints.**

## ⭐ Same input, different answers

Identical 5-year daily return series, all defaults, `rf=0`:

| Metric | quantstats | empyrical | ffn | Hand-computed |
|---|---|---|---|---|
| Sharpe | 1.279985 | 1.279985 | 1.273744 ¹ | **1.279985** (ddof=1) |
| Sortino | 1.955876 | 1.955876 | — | **1.955876** |
| Max drawdown | −0.226178 | −0.226178 | −0.226178 | **−0.226178** |
| Annual volatility | 0.188495 | 0.188495 | 0.188554 ¹ | **0.188495** (ddof=1) |
| **CAGR** | **0.250378** | **0.250378** | **0.258761** | 0.250378 (`len/252`) / 0.260442 (calendar) |
| **VaR 95%** | **−0.018574** | **−0.018111** | — | −0.018111 (historical) / −0.018574 (**Gaussian**) |
| **CVaR 95%** | **−0.023346** | **−0.022786** | — | −0.022786 (historical tail mean) |

¹ ffn derives returns from prices via `pct_change()`, losing the first observation (n=1259 vs 1260).

**CAGR and VaR/CVaR are real, systematic disagreements, not rounding.** quantstats' VaR is
**Gaussian**; empyrical's is **historical**. With fat-tailed returns these are different risk numbers.
**Know which one you are quoting.**

## 🔴 The risk-free-rate unit trap

| Library | Parameter | Units expected |
|---|---|---|
| quantstats | `rf=` | **ANNUAL** (converted geometrically) |
| **empyrical / pyfolio-reloaded** | `risk_free=` | 🔴 **PER-PERIOD (daily)** — subtracted raw |
| ffn | `rf=` | **ANNUAL** |
| PyPortfolioOpt | `risk_free_rate=` | **ANNUAL** |

empyrical's docstring is explicit — *"Constant **daily** risk-free return"* — but the parameter is
named `risk_free`, defaults to `0.0`, and sits next to `period='daily'`, so almost everyone passes an
annual rate.

```python
qs.stats.sharpe(r, rf=0.05)                       #   1.021119   correct: 5% annual
ep.sharpe_ratio(r, risk_free=0.05)                # -65.565185   ← 5% PER DAY
ep.sharpe_ratio(r, risk_free=(1.05**(1/252)-1))   #   1.021119   correct usage
```

🔴 **A wildly negative Sharpe is the diagnostic signature.** Check units before debugging anything else.

Secondary subtlety: geometric vs simple de-annualization. `(1.05)^(1/252)−1` → 1.021119;
`0.05/252` → 1.014726. ~0.6% apart — immaterial for research, but it means quantstats and a
hand-rolled `rf/252` will never match exactly.

## 🔴 CONFIRMED BUG: `quantstats.stats.cagr()` discards `rf`

Signature `cagr(returns, rf=0.0, compounded=True, periods=252)`; the docstring says it computes the
CAGR *"of excess returns"*. **The parameter has no effect.** Root cause in
`quantstats/utils.py::_prepare_returns` — it dispatches on the **caller's function name**:

```python
function = inspect.stack()[1][3]
unnecessary_function_calls = ["_prepare_benchmark", "cagr", "gain_to_pain_ratio", "rolling_volatility"]
if function not in unnecessary_function_calls:
    if rf > 0:
        return to_excess_returns(data, rf, nperiods)
```

`"cagr"` is on the exclusion list, so `rf` is accepted and silently discarded. **The same list
disables `rf` for `gain_to_pain_ratio` and `rolling_volatility`.** Compute excess-return CAGR yourself.

## Practical rules

1. **For any number you will publish or trade on, use `ffn`** (it infers annualization from the index)
   **plus a hand-rolled check** — not quantstats defaults.
2. **Always state the risk-free convention and the annualization factor** with the metric. 252 is not
   universal: crypto is 365, and a weekly-rebalanced strategy is neither.
3. **Never regress raw returns on raw returns** — the intercept absorbs the risk-free rate and your
   "alpha" is the cash yield.
4. **Recompute one metric by hand for every new pipeline.** Disagreement is the point of the exercise.
5. A 3-year Sharpe of 1.0 has a standard error near **0.58**. Report the interval.

## What none of them do

**Statistical significance.** None of these answers "is this real after I tried 200 variants".
`quantstats` has `sharpe`/`sortino` but **no PSR, DSR or MinTRL**. For that go to
`../../backtest-validation/SKILL.md` — Deflated Sharpe, PBO, and `arch.bootstrap`'s SPA / Reality
Check / StepM / MCS.
