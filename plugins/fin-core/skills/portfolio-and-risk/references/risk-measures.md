# Risk measures — VaR, CVaR, drawdown, and conditional volatility

Which risk number you are actually quoting depends on the library, not the name: two functions called
`value_at_risk` return different quantities, and neither says so.

Verified 2026-09-04 (package metadata re-checked live); the measured metric divergences were produced by
**executing** the libraries on one seeded 1,260-day return series.

## 🚨 The trap: same name, different estimator

🚨 **`quantstats.stats.value_at_risk` is PARAMETRIC GAUSSIAN. `empyrical.value_at_risk` is
HISTORICAL.** ✅ Verified from source — quantstats computes `norm.ppf(1 - confidence, mu, sigma)`.
The docstring says "variance-covariance method"; the function name does not, and the tear sheet labels
it simply "Daily Value-at-Risk".

Measured on the same series ✅: quantstats **−0.018574** (Gaussian) vs empyrical **−0.018111**
(5th percentile). On Gaussian synthetic data that gap is small. **On fat-tailed real strategies
quantstats systematically understates tail risk**, and the gap widens exactly where it matters.
`quantstats.expected_shortfall` is an alias of `conditional_value_at_risk` ✅.

🚨 **`arch` expects returns in PERCENT (×100).** Feeding decimals gives tiny variances, poor optimizer
conditioning and convergence warnings — the most common `arch` support question ⚠️. Rescale the
forecast back before using it.

🚨 **Annualization is hard-coded, not inferred**, in quantstats and empyrical (`periods=252`). Measured
✅: the same data resampled to monthly gives Sharpe **2.2731** at the default vs **0.4960** with
`periods=12` — a **4.58× (√21) overstatement** from one unchanged default. ffn is the exception: it
calls `infer_nperiods()` from the DatetimeIndex ✅.

⚠️ **VaR is not subadditive; CVaR is.** VaR of a combined book can exceed the sum of its parts, which
is why VaR is a poor *optimization* objective and CVaR (Rockafellar-Uryasev) is the standard convex
substitute. Optimize CVaR; report VaR only if a regulator asks for it.

## VaR / CVaR estimator families → where each lives

| Estimator | Formula sketch | Where ✅ |
|---|---|---|
| **Historical VaR** | `np.percentile(r, 100·α)` | `empyrical.value_at_risk`; numpy directly |
| **Historical CVaR / ES** | `r[r <= VaR].mean()` | `empyrical.conditional_value_at_risk` |
| **Parametric (Gaussian) VaR** | `μ + σ·Φ⁻¹(α)` | **quantstats** `value_at_risk`; `scipy.stats.norm.ppf` |
| **Student-t parametric VaR** | `μ + σ·t⁻¹(α; ν)` | ⚠️ hand-roll with `scipy.stats.t` |
| **Cornish-Fisher (modified) VaR** | Gaussian quantile expanded in skew and excess kurtosis | ⚠️ **no canonical maintained library** — hand-roll, below |
| **GARCH-filtered / conditional VaR** | fit GARCH → forecast σ → quantile | **`arch`** |
| **CVaR / CDaR / EVaR / RLVaR as optimization objectives** | Rockafellar-Uryasev LP / conic | **Riskfolio-Lib** (26 measures), **skfolio** (19), PyPortfolioOpt (`EfficientCVaR`, `EfficientCDaR` only) |

Cornish-Fisher, since nothing packages it well:

```python
from scipy.stats import norm, skew, kurtosis
def cornish_fisher_var(r, alpha=0.05):
    z, s, k = norm.ppf(alpha), skew(r), kurtosis(r)      # k = EXCESS kurtosis
    z_cf = (z + (z**2 - 1) * s / 6
              + (z**3 - 3*z) * k / 24
              - (2*z**3 - 5*z) * s**2 / 36)
    return r.mean() + r.std(ddof=1) * z_cf
```

## Drawdown

✅ **This is the one family everybody agrees on.** quantstats, empyrical, ffn and a hand-rolled
`(P/P.cummax() - 1).min()` match to six decimal places on the same series. quantstats additionally
prepends a phantom baseline point so a first-period loss is counted — historically a real bug, **now
fixed** ✅.

```python
def max_drawdown(r):
    eq = (1 + r).cumprod()
    return float((eq / eq.cummax() - 1).min())
```

Drawdown-family risk measures used as *optimization objectives* — average drawdown, **Ulcer index**,
**CDaR**, **EDaR**, **RLDaR**, max drawdown — exist only in **Riskfolio-Lib** (all of them) and
**skfolio** (CDaR, EDaR, max drawdown). ⚠️ EDaR/RLDaR need exponential- or power-cone solvers
(Clarabel, SCS, MOSEK), not OSQP — see `_solver-layer.md`.

⚠️ **Drawdown depth is not a distribution-free statistic.** Max drawdown is a maximum over the sample,
so it grows with sample length even for an unchanged process. Comparing a 3-year and a 10-year max
drawdown as if they were the same quantity is a category error; compare Calmar or Ulcer instead, and
report the sample span.

## `arch` — conditional volatility and the GARCH family

| Field | Value |
|---|---|
| pip / import | `arch` / `arch` |
| version | **8.0.0** (2025-10-21) ✅ |
| repo | `github.com/bashtage/arch` (Kevin Sheppard, Oxford) — **1,558 ★ / 51 issues**, pushed 2026-08-10 ✅ |
| licence | **NCSA** ✅ — permissive, BSD-like, but *not* one of the usual three; GitHub reports `NOASSERTION`. Flag it in a licence audit. |
| Python | `>=3.10` ✅ |
| verdict | ✅ **actively maintained; the reference GARCH implementation in Python** — statsmodels has no equivalent |

✅ 8.0.0 is a compatibility release (Python 3.14 wheels, NumPy 2.4 / pandas 3 support, doc fixes) —
**not an API break** from 7.x. Safe upgrade.

Covers: **GARCH, EGARCH, GJR-GARCH/TARCH, APARCH, HARCH, HAR-RV, FIGARCH, RiskMetrics, ARCH**;
distributions Normal / Student-t / skew-t / GED; mean models including AR and HAR; plus
`arch.unitroot` (ADF, DF-GLS, Phillips-Perron, KPSS, Zivot-Andrews, Variance Ratio), cointegration,
long-run covariance estimators, and the bootstrap suite.

```python
from arch import arch_model

am  = arch_model(returns * 100,                 # 🚨 PERCENT, not decimals
                 vol="GARCH", p=1, o=1, q=1,    # o=1 -> GJR asymmetry (leverage effect)
                 dist="skewt", mean="Constant")
res = am.fit(disp="off")
f   = res.forecast(horizon=10, reindex=False)   # reindex=False: explicit, avoids the deprecation path
sigma_next = (f.variance.iloc[-1] ** 0.5) / 100  # 🚨 rescale back out of percent

# GARCH-filtered VaR: a CONDITIONAL quantile. Filtered-historical-simulation form —
# quantile the standardized residuals rather than assuming a parametric shape.
z = (res.resid / res.conditional_volatility).dropna()
var_1d = (res.params["mu"] + sigma_next.iloc[0] * 100 * z.quantile(0.05)) / 100
```

⚠️ **`o=1` is what gives you the leverage effect.** Plain `GARCH(1,1)` treats a −5% day and a +5% day
identically, which is empirically false for equities. GJR (`o=1`) or EGARCH is the honest default.

🔑 `arch.bootstrap` in the same package is where SPA / Reality Check / StepM / MCS live — see
`../../backtest-validation/references/significance-tests.md`. 🚨 **Those all take LOSSES, not returns.**

## Which library computes which

| Measure | quantstats | empyrical | ffn | Riskfolio | skfolio | arch |
|---|---|---|---|---|---|---|
| Historical VaR / CVaR | ❌ (Gaussian only) | ✅ | ❌ | as objective | as objective | — |
| Gaussian VaR | ✅ | ❌ | ❌ | — | — | — |
| Max drawdown / Calmar | ✅ | ✅ | ✅ | ✅ objective | ✅ objective | — |
| Ulcer, CDaR, EDaR, RLDaR | ❌ | ❌ | ❌ | ✅ **all** | ✅ CDaR/EDaR | — |
| EVaR / RLVaR | ❌ | ❌ | ❌ | ✅ | ✅ EVaR | — |
| Conditional (GARCH) vol | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| **PSR / DSR / MinTRL** | 🚨 **none of them** ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |

🚨 **`quantstats` has `sharpe` and `sortino` but no PSR, DSR or MinTRL** ✅ — no analytics library in
this group answers "is this real after I tried 200 variants". That question lives in
`../../backtest-validation/SKILL.md`.

## Rules to apply every time

1. **State the estimator with the number.** "VaR 95% = −1.86%" is not a fact; "historical 95% 1-day VaR"
   is. Gaussian and historical VaR are different risk statistics.
2. **Always pass `periods=` / `frequency=` explicitly.** 252 is not universal — crypto is 365, and a
   weekly-rebalanced book is neither.
3. **`dropna()` before handing returns to quantstats** — `_prepare_returns` does `fillna(0)` ✅, turning
   missing days into zero-return days: inflated n, deflated volatility, distorted drawdown duration.
4. 🚨 **Guarantee your input is unambiguously returns.** quantstats guesses: `if data.min() >= 0 and
   data.max() > 1: data = data.pct_change()` ✅ — so an all-non-negative return series containing one
   +100% period is silently differenced into nonsense. Realistic for crypto, small caps, and options.
5. **For a number you will publish or trade on, prefer `ffn`** (it infers annualization from the index,
   and is the best-maintained of the group) plus one hand-rolled cross-check.

See `analytics-libraries.md` for the full measured audit of quantstats/empyrical/ffn disagreements,
`riskfolio-lib.md` and `skfolio.md` for optimizing against these measures rather than reporting them,
and `_solver-layer.md` for which solver each conic risk measure requires.
