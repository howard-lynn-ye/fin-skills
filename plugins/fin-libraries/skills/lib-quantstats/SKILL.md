---
name: lib-quantstats
description: >-
  The tearsheet library whose cagr(rf=...) accepts your risk-free rate and silently discards it -
  "cagr" sits on an exclusion list inside _prepare_returns, which dispatches on the caller's
  function name. TRIGGER - quantstats, "import quantstats as qs", qs.reports.html,
  qs.stats.sharpe, qs.stats.cagr, qs.stats.value_at_risk, expected_shortfall, gain_to_pain_ratio,
  rolling_volatility, qs.extend_pandas, tearsheet, quantstats-lumi; or a wildly negative Sharpe.
  Memory is stale on status and correctness - 0.0.81 shipped in a single-day hotfix burst on
  2026-01-13 with no default-branch commits since, and the cagr bug survived it. SKIP for
  optimizing against these measures (lib-riskfolio, lib-skfolio) and for PSR/DSR, which it does
  not have (lib-purgedcv). SKIP when the question is WHICH library to choose rather than how to
  use this one - that belongs to the domain skill.
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-04"
---

# quantstats

One-call HTML tearsheets and performance analytics — enormously popular, and the source of more quietly wrong
published numbers than anything else in this domain.

| | |
|---|---|
| pip / import | `QuantStats` / `quantstats` (conventionally `import quantstats as qs`) |
| Version | **0.0.81** (2026-01-13) · Python `>=3.10` |
| Licence | Apache-2.0 |
| Status | ⚠️ **burst-maintained** — 7,612★ / 32 issues; no default-branch commits since 2026-01-13 |

The maintenance pattern matters because of the bugs below: 0.0.78 ("2026 Modernization Update") shipped with
circular-import errors hot-fixed through 0.0.79, 0.0.80 and 0.0.81 **the same day**, and the metric bugs survived it.
Forks: `quantstats-lumi` 1.1.5, `quantstats-reloaded` 0.1.0.

## The trap that costs you money

🚨 **CONFIRMED BUG: `quantstats.stats.cagr()` discards `rf`.** The signature is `cagr(returns, rf=0.0, compounded=True,
periods=252)` and the docstring says it computes the CAGR
*"of excess returns"*. **The parameter has no effect.** Root cause in
`quantstats/utils.py::_prepare_returns`, which dispatches on the **caller's function name**:

```python
function = inspect.stack()[1][3]
unnecessary_function_calls = ["_prepare_benchmark", "cagr", "gain_to_pain_ratio", "rolling_volatility"]
if function not in unnecessary_function_calls:
    if rf > 0:
        return to_excess_returns(data, rf, nperiods)
```

`"cagr"` is on the exclusion list, so `rf` is accepted and silently dropped. The **same list disables `rf` for
`gain_to_pain_ratio` and `rolling_volatility`.** Compute excess-return CAGR yourself.

## Its VaR is Gaussian; empyrical's is historical

🚨 `quantstats.stats.value_at_risk` is **parametric Gaussian** — verified from source, it computes `norm.ppf(1
- confidence, mu, sigma)`. The docstring says "variance-covariance method"; the function name does not, and
the tear sheet labels it simply "Daily Value-at-Risk". `expected_shortfall` is an alias of
`conditional_value_at_risk`. Measured, one seeded 1,260-day series, `rf=0`, all defaults:

| Metric | quantstats | empyrical | ffn | Hand-computed |
|---|---|---|---|---|
| **VaR 95%** | **−0.018574** | **−0.018111** | — | −0.018111 historical / −0.018574 **Gaussian** |
| **CVaR 95%** | **−0.023346** | **−0.022786** | — | −0.022786 (historical tail mean) |
| **CAGR** | 0.250378 | 0.250378 | 0.258761 | 0.250378 (`len/252`) / 0.260442 (calendar) |

Systematic disagreements, not rounding. **On fat-tailed strategies quantstats understates tail risk.** Say "historical
95% 1-day VaR" or "Gaussian VaR" — never just "VaR".

## Its `rf=` is ANNUAL — a wildly negative Sharpe is the diagnostic signature

quantstats' `rf=` is **ANNUAL** (converted geometrically), as are ffn's `rf=` and PyPortfolioOpt's `risk_free_rate=`.
🔴 **empyrical / pyfolio-reloaded's `risk_free=` is PER-PERIOD (daily), subtracted raw** — and it sits beside
`period='daily'`, so almost everyone passes an annual rate.

```python
qs.stats.sharpe(r, rf=0.05)                     #   -0.285948  correct: 5% annual
ep.sharpe_ratio(r, risk_free=0.05)              #  -81.323237  <- 5% PER DAY, the signature
ep.sharpe_ratio(r, risk_free=0.05 / 252)        #   -0.293729  <- de-annualised by division
```

✅ Reproduce with `scripts/rf_convention.py` (seed 0, 4 years of daily returns). The residual
between −0.293729 and −0.285948 is quantstats de-annualising **geometrically**, not by dividing
by 252 — so `/252` gets you close to its answer but never equals it. The script's own reference
implementations match both installed libraries to every printed digit, so it demonstrates the
trap on a machine where neither package is installed.

⚠️ Geometric vs simple: `(1.05)**(1/252)-1` → 1.021119, `0.05/252` → 1.014726 — never identical.

## Silent input mutations, and what it does not have

🚨 **It guesses whether your input is prices:** `if data.min() >= 0 and data.max() > 1: data = data.pct_change()`. An
all-non-negative return series containing one +100% period is silently differenced into nonsense — realistic for
crypto, small caps and options. 🚨 **`_prepare_returns` does `fillna(0)`**, turning missing days into zero-return days:
inflated n, deflated volatility, distorted drawdown duration. **`dropna()` first.** 🚨 **Annualization is hard-coded
(`periods=252`), not inferred.** Measured: the same data resampled to monthly gives Sharpe **2.2731** at the default
vs **0.4960** with `periods=12` — a **4.58× (√21) overstatement** from one unchanged default. `ffn` is the exception;
it infers from the DatetimeIndex. ⚠️ **No PSR, no DSR, no MinTRL** — `sharpe` and `sortino`, and it stops there.
Drawdown is the one family everybody agrees on: quantstats, empyrical, ffn and `(P/P.cummax()-1).min()` agree to 6 dp.

## Minimal correct call

```python
import quantstats as qs

r = returns.dropna()                       # 🚨 first: fillna(0) inflates n and deflates vol
assert not ((r.min() >= 0) and (r.max() > 1)), "quantstats would silently pct_change() this"
sharpe = qs.stats.sharpe(r, rf=0.05, periods=252)   # rf ANNUAL; state periods explicitly

excess = (1 + r) / (1 + 0.05) ** (1 / 252) - 1      # 🚨 cagr(rf=) is ignored — do it yourself
cagr_excess = (1 + excess).prod() ** (252 / len(excess)) - 1

var_gaussian = qs.stats.value_at_risk(r)            # 🚨 GAUSSIAN — label it as such
var_hist = r.quantile(0.05)                         # historical, if that is what you meant
qs.reports.html(r, benchmark="SPY", output="tearsheet.html", rf=0.05)
```

**For a number you will publish or trade on, use `ffn` plus a hand-rolled cross-check**, and state
the risk-free convention and annualization factor with the metric. A 3-year Sharpe of 1.0 has a standard error near
**0.58** — report the interval.

## See also

- `../../../fin-core/skills/portfolio-and-risk/SKILL.md` — the domain skill for performance analytics
- `../../../fin-core/skills/portfolio-and-risk/references/analytics-libraries.md` — the source card and measured audit
- `../../../fin-core/skills/portfolio-and-risk/references/risk-measures.md` — VaR/CVaR estimator families
- `../../../fin-core/skills/backtest-validation/SKILL.md` — PSR, DSR and PBO, which quantstats does not have

## Where this sits

This file is the deep dive on **one** library and assumes the choice is already made.
For which library to pick, how it compares with the alternatives, and the traps that span
several of them, the entry point is the domain skill **`portfolio-and-risk`** (`../../../fin-core/skills/portfolio-and-risk/SKILL.md`).
