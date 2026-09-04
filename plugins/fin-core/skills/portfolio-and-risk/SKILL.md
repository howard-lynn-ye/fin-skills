---
name: portfolio-and-risk
description: >-
  Construct portfolio weights and compute performance and risk metrics correctly. Covers
  PyPortfolioOpt, Riskfolio-Lib, skfolio, cvxportfolio, deepdow and the cvxpy solver layer for
  optimization; quantstats, pyfolio-reloaded, empyrical-reloaded, ffn and alphalens-reloaded for
  analytics; plus VaR/CVaR, drawdown, GARCH volatility via arch, and attribution. TRIGGER — use
  when the task involves portfolio weights, allocation, rebalancing, mean-variance, Black-Litterman,
  risk parity, HRP/HERC/NCO, covariance shrinkage or denoising, the efficient frontier, position
  sizing across assets; OR when computing or reporting Sharpe, Sortino, Calmar, CAGR, volatility,
  max drawdown, VaR, CVaR, beta, alpha, a tearsheet, or performance attribution. Load this before
  quoting any performance number — several popular libraries disagree on the same input.
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-03"
---

# Portfolio construction and risk analytics

Two independent problems live here. §1 picks an optimizer. §2 is the one you cannot skip: **the
same return series produces different Sharpe ratios in different libraries, and one popular
function silently ignores an argument you passed it.**

## 1. Pick an optimizer

| Task | Use | Not |
|---|---|---|
| Textbook mean-variance / Black-Litterman, small universe, prototype | **PyPortfolioOpt** | — |
| Broadest risk-measure menu (26 convex: CVaR, CDaR, EVaR, RLVaR, OWA), robust worst-case, integer constraints | **Riskfolio-Lib** | PyPortfolioOpt — too few risk measures |
| sklearn `Pipeline` / `GridSearchCV` / cross-validation over portfolio models; entropy pooling; vine copulas | **skfolio** | Riskfolio-Lib — no sklearn API |
| Multi-period, **transaction-cost-aware** simulation of the trading *policy* | **cvxportfolio** — 🚨 **GPL-3.0** | PyPortfolioOpt — single-period only |
| **HRP** | Riskfolio-Lib / skfolio / PyPortfolioOpt `HRPOpt` | 🚨 **mlfinlab — proprietary, delisted from PyPI** |
| **HERC, NCO, Schur complementary** | Riskfolio-Lib or skfolio **only** | PyPortfolioOpt does not implement these |
| **Marcenko–Pastur denoising / detoning** | Riskfolio-Lib `denoiseCov`, skfolio `Denoise`/`Detone` | 🚨 **PyPortfolioOpt has neither** — its `CovarianceShrinkage` is Ledoit-Wolf, a *different* estimator |
| Deep-learning end-to-end allocation | deepdow — ⚠️ dormant since Jan 2024 | — |

**Gotchas that bite immediately:**
- 🚨 **`HRPOpt(returns=...)` takes a RETURNS matrix.** Passing prices runs silently and produces
  garbage.
- **Linkage defaults differ:** skfolio uses **Ward**, PyPortfolioOpt uses **single** (the AFML
  original). Results differ by design, not by bug.
- **skfolio's `CombinatorialPurgedCV.split()` yields `(train, [test_0, test_1, ...])`** — multiple
  test folds per split, *not* sklearn's 2-tuple contract. And `purged_size`/`embargo_size` are counted
  in **observations, not time** — with dollar/volume bars a fixed count is a wildly varying span.
- The solver layer (cvxpy → ECOS/OSQP/SCS/Clarabel/MOSEK/Gurobi) constrains what problems are
  expressible and which licences apply. See `references/_solver-layer.md` first if you hit a
  `SolverError`.

## 2. 🚨 Metric correctness audit

> Verified by **executing** the libraries (quantstats 0.0.81, empyrical-reloaded 0.5.12, ffn 1.1.5,
> pandas 3.0.5) on the same seeded 1,260-day return series — not inferred from docs.

### 2.1 Identical input, different answers (all defaults, `rf=0`)

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

**The CAGR and VaR/CVaR rows are real, systematic disagreements, not rounding.** quantstats' VaR is
**Gaussian**; empyrical's is **historical**. If your returns have fat tails — and they do — these
are different risk numbers. Know which one you are quoting.

### 2.2 🔴 The risk-free-rate unit trap — the #1 real-world Sharpe bug

**Libraries interpret the same `rf` number in incompatible units:**

| Library | Parameter | Units it expects |
|---|---|---|
| **quantstats** | `rf=` | **ANNUAL** (converts geometrically) |
| **empyrical / pyfolio-reloaded** | `risk_free=` | 🔴 **PER-PERIOD (daily!)** — subtracted raw, no conversion |
| **ffn** | `rf=` | **ANNUAL** |
| **PyPortfolioOpt** | `risk_free_rate=` | **ANNUAL** |

empyrical's docstring is explicit — *"risk_free : int, float — Constant **daily** risk-free return"* —
but the parameter is named `risk_free`, defaults to `0.0`, and sits next to `period='daily'`, so
almost everyone passes an annual rate. Measured:

```python
qs.stats.sharpe(r, rf=0.05)                       #   1.021119   correct: 5% annual
ep.sharpe_ratio(r, risk_free=0.05)                # -65.565185   ← subtracts 5% PER DAY
ep.sharpe_ratio(r, risk_free=(1.05**(1/252)-1))   #   1.021119   correct usage
```

🔴 **A wildly negative Sharpe is the diagnostic signature of this bug.** If you ever see one out of
empyrical or pyfolio, check the units before debugging anything else.

### 2.3 🔴 CONFIRMED BUG: `quantstats.stats.cagr()` silently ignores `rf`

Signature is `cagr(returns, rf=0.0, compounded=True, periods=252)` and the docstring says it
computes the CAGR *"of excess returns"*. **The parameter has no effect.** Root cause in
`quantstats/utils.py::_prepare_returns` — it dispatches on the **caller's function name**:

```python
function = inspect.stack()[1][3]
unnecessary_function_calls = ["_prepare_benchmark", "cagr", "gain_to_pain_ratio", "rolling_volatility"]
if function not in unnecessary_function_calls:
    if rf > 0:
        return to_excess_returns(data, rf, nperiods)
```

`"cagr"` is on the exclusion list, so `rf` is accepted and discarded. **Compute excess-return CAGR
yourself.** The same exclusion list silently disables `rf` for `gain_to_pain_ratio` and
`rolling_volatility`.

### 2.4 Practical rules

1. **For any number you will publish or trade on, use `ffn`** (it infers annualization from the
   index) **plus a hand-rolled check**, not quantstats defaults.
2. **Always state the risk-free convention and the annualization factor** alongside the metric.
   `252` is not universal — crypto is `365`, and a strategy that trades weekly is not `252`.
3. **Never subtract a risk-free rate from raw returns and regress raw on raw** — the intercept
   absorbs the rate and your "alpha" is the cash yield.
4. Recompute one metric by hand for every new pipeline. Disagreement is the point of the exercise.

## 3. Risk measures

- **Drawdown** is path-dependent — it is not recoverable from mean and variance, and it is the metric
  most sensitive to the exact rebalance timing you assumed.
- **VaR/CVaR**: know whether you are getting historical, Gaussian, or Cornish-Fisher. They diverge
  exactly in the tail you care about.
- **Volatility modelling:** `arch` (Kevin Sheppard) for GARCH/EGARCH/HAR. It is the reference
  implementation and is actively maintained.
- **Confidence intervals matter.** A 3-year Sharpe of 1.0 has a standard error near 0.58. Reporting
  it without an interval implies precision that does not exist.

## 4. Statistical significance → a separate skill

For "is this result real after I tried many things" — Deflated Sharpe, PBO, White's Reality Check,
Hansen's SPA, StepM, Model Confidence Set (**all in `arch.bootstrap`**) — go to
**`backtest-validation`**. Do not settle that question with a tearsheet.

## 5. Attribution — a genuine ecosystem gap

⚠️ **No mature Brinson-attribution library exists in Python.** Single-period Brinson-Fachler is
~20 lines and is included in `references/_attribution.md`; multi-period linking (Carino, Menchero)
you must implement. Factor attribution is better served by regressing against explicit factor
returns — see `factor-and-timeseries-research`.

## 6. Reference files

`references/<library>.md` — version, licence, what it implements that others don't, exact
signatures, and its measured quirks. `references/_solver-layer.md` covers cvxpy and solver choice.

```bash
grep -ril "risk_free\|annualiz" plugins/fin-core/skills/portfolio-and-risk/references/
```
