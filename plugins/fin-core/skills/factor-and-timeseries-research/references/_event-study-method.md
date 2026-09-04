# Event studies — you have to implement this yourself

🚨 **There is no usable Python event-study library.** The only PyPI package, `eventstudy`, is
**0.1a12 — an alpha from 2021, GPL-3.0, 69★**; alternatives top out at 12★. This file is the
methodology so you can write ~100 lines that are correct.

## 1. Define the event time precisely

This is where most event studies fail, before any statistics happen.

- The event time is `available_at`, not the period the event describes. For an SEC filing that is
  **`acceptanceDateTime`** converted to exchange local time — see `fundamental-and-macro-data` §2.
- **If the event lands at or after 16:00 local, day 0 is the NEXT session.** Apple's earnings 8-Ks
  cluster at 16:30–17:30 ET; treating them as same-day events trades on information that did not
  exist.
- Deduplicate. One economic event often produces several filings (8-K + press release + 10-Q). Build
  event *clusters* and pick one timestamp per cluster, or your sample double-counts.
- Amendments (`10-K/A`, `8-K/A`) are usually not new events. Decide and document.

## 2. Windows

```
   estimation window            gap      event window
[ t-250 ................ t-31 ] [ .. ] [ t-1, t0, t+1 ... t+k ]
```

- **Estimation window**: typically 250 sessions ending ~30 sessions before the event, so the event's
  own volatility does not contaminate the model parameters.
- **Gap**: leave ≥ 10–30 sessions. Without it, anticipation leaks into your betas.
- **Event window**: state it in advance. `[-1, +1]`, `[0, +5]`, `[0, +20]` are conventional.
  🚨 **Choosing the window after seeing results is p-hacking, and it counts as a trial.**

## 3. Normal-return model

| Model | `E[R_it]` | When |
|---|---|---|
| Mean-adjusted | `mean(R_i)` over the estimation window | Weakest; only if you cannot get a market series |
| **Market model** | `alpha_i + beta_i * R_mt` (OLS over the estimation window) | **The default** |
| Market-adjusted | `R_mt` (beta forced to 1) | Short windows, thin data |
| Fama-French 3/5 | multi-factor OLS | When the sample tilts to size/value/momentum |

**Abnormal return:** `AR_it = R_it - E[R_it]`
**Cumulative:** `CAR_i[a,b] = sum(AR_it)` over the window
**Buy-and-hold:** `BHAR_i[a,b] = prod(1+R_it) - prod(1+E[R_it])`

**CAR vs BHAR:** CAR for short windows (≤ ~20 days); BHAR for long horizons, because CAR's
arithmetic summation drifts from what an investor actually earns. BHAR's distribution is badly
skewed — use bootstrapped or skewness-adjusted t-statistics, not the plain one.

## 4. 🚨 Test statistics — the part that is usually wrong

The naive cross-sectional t-test `mean(CAR) / (sd(CAR)/sqrt(N))` assumes independent, homoskedastic
abnormal returns. Events violate both.

**Event-induced variance — and a correction to what this file used to say.** Volatility rises
*because* of the event, so the **estimation-window** variance understates the **event-window**
variance. This file previously claimed that makes the plain cross-sectional t-test over-reject.
✅ **Measured on 300 null panels with event-induced variance, that is wrong:**

| Test | Standardises by | Rejects at nominal 5% |
|---|---|---|
| **Patell** (standardised residual) | **estimation-window** SE | 🚨 **47.0%** — over-rejects catastrophically |
| Corrado rank, classic SE | time-series SE | 🚨 **16.3%** |
| Plain cross-sectional t on raw CARs | cross-sectional SD of CARs | ✅ **3–5%** — correctly sized |
| **BMP** (standardised, cross-sectional) | forecast-error-adjusted SD, then cross-sectional | ✅ **4.3%** |

**Why the plain test survives:** its denominator is the *cross-sectional* dispersion of CARs, which
inflates along with the event-induced variance. Numerator and denominator move together, so a
symmetric scale mixture cancels. **Patell's denominator comes from the estimation window, which
knows nothing about the event** — so its denominator is too small and it rejects a true null nearly
half the time.

**So the reason to use BMP is POWER, not size.** On identical data with a planted +4.0% effect,
BMP's t is **7.57 against the plain test's 3.86 — 2× the statistic**, and it recovers +3.908%
against the planted +4.000%. Use BMP because it detects a real effect the raw test may miss, and
use it *instead of Patell* because Patell is unusable under exactly the conditions event studies
create.

⚠️ Skewness was tested as an alternative mechanism and **hurts BMP as much as the raw test**, so it
is not the separator either. `../scripts/event_study.py` reproduces all of this.

**Cross-sectional correlation.** If events cluster in calendar time (an industry shock, a regulatory
date, an earnings season), abnormal returns are correlated across firms and the effective N is far
below the number of events. Options: calendar-time portfolio regressions, or a
Kolari-Pynnönen-style correction.

**Also report a nonparametric test** — a rank test (Corrado) or sign test. If the parametric and
nonparametric results disagree, the parametric one is being driven by outliers.

## 5. Sample construction

- **Survivorship:** the event sample must include firms that later delisted, especially for
  distress-related events. See `research-integrity-guards` §1.
- **Thin trading** breaks OLS beta estimation; either require a minimum number of non-zero-return
  days in the estimation window, or use Scholes-Williams / Dimson betas.
- **Confounding events:** exclude firms with another material event inside the window, or state
  that you did not.
- Report the **attrition table**: how many events you started with and how many survived each filter.

## 6. Reporting

Report `AAR_t` (average abnormal return per event day) *and* `CAAR[a,b]`, with the number of events,
the test statistic used, and both parametric and nonparametric p-values. Plot CAAR with a confidence
band, and include the **pre-event window** in the plot — a drift that starts before t0 is either
leakage in your timestamps or genuine anticipation, and you need to see which.

## 7. Minimal skeleton

```python
import numpy as np, pandas as pd, statsmodels.api as sm

def event_study(rets, mkt, events, est=250, gap=30, pre=5, post=20):
    """rets: DataFrame (date x ticker); mkt: Series; events: [(ticker, day0_ts)]"""
    rows = []
    for tic, t0 in events:
        r = rets[tic].dropna()
        if t0 not in r.index:
            continue
        i = r.index.get_loc(t0)
        est_slice = slice(i - gap - est, i - gap)
        if i - gap - est < 0 or i + post >= len(r):
            continue
        y = r.iloc[est_slice]
        X = sm.add_constant(mkt.reindex(y.index))
        fit = sm.OLS(y, X, missing="drop").fit()
        win = r.iloc[i - pre: i + post + 1]
        exp = fit.params.iloc[0] + fit.params.iloc[1] * mkt.reindex(win.index)
        ar = win - exp
        # BMP: standardize by the forecast-error-adjusted sd, NOT the raw estimation sd
        sd = np.sqrt(fit.mse_resid)
        rows.append(pd.Series((ar / sd).values,
                              index=range(-pre, post + 1), name=(tic, t0)))
    sar = pd.DataFrame(rows)                       # standardized ARs
    caar = sar.mean().cumsum()
    t = sar.mean() / (sar.std(ddof=1) / np.sqrt(len(sar)))   # cross-sectional on standardized
    return caar, t, len(sar)
```

This is the BMP-standardized version. It still assumes cross-sectional independence — if your events
cluster in time, say so and do not quote the t-statistic as if it were exact.
