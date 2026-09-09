---
name: gdp-nowcasting-dynamic-factor
description: >-
  Nowcast the quarter you are in from monthly data with a ragged edge, using
  statsmodels' DynamicFactorMQ - and score it against the benchmarks it has to beat.
  TRIGGER - nowcast, nowcasting, GDPNow, Atlanta Fed GDP tracker, New York Fed Staff
  Nowcast, DynamicFactorMQ, endog_quarterly, k_endog_monthly, fit_em, dynamic factor model,
  mixed frequency, monthly and quarterly in one model, ragged edge, jagged edge, unbalanced
  panel, Mariano-Murasawa, Banbura Modugno, bridge equation, MIDAS, "how do I combine
  monthly indicators into a GDP forecast", news decomposition of a data release. SKIP for
  the Kalman filter and smoother themselves (state-space-and-kalman), for univariate
  forecasting and its baselines (time-series-forecasting-models), for release timestamps
  (macro-release-calendar-and-embargo), and for vintages of the inputs
  (real-time-macro-backtesting).
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-09"
---

# GDP nowcasting with a dynamic factor model

**A nowcast is not a forecast of the future — it is an estimate of a number that already
happened and has not been published yet.** Everything hard about it is in the shape of the
data: monthly indicators arriving on different days with different lags, a quarterly target
observed once every three months, and a bottom edge of the panel that is jagged rather than
flat. The model that handles that is a state-space one, and it is already installed.

The Kalman machinery itself lives in
`../../../fin-models/skills/state-space-and-kalman/SKILL.md`; univariate baselines and how to
score a forecast live in `../../../fin-models/skills/time-series-forecasting-models/SKILL.md`.
This skill is the mixed-frequency, ragged-edge part and the benchmark discipline.

Everything marked ✅ Measured is printed by `scripts/nowcast.py` (numpy + pandas + scipy, seed
20260909, **6–25 s** depending on machine load, no network). Everything marked ✅
source-verified was read at the URL or in the installed package on 2026-09-09.

## 1. ✅ `DynamicFactorMQ` is the NY-Fed-shaped model, already in a BSD dependency

✅ source-verified in `statsmodels/tsa/statespace/dynamic_factor_mq.py`, statsmodels 0.15.0,
read in the installed package. The class docstring opens:

> *"Implementation of the dynamic factor model of Bańbura and Modugno (2014) and Bańbura,
> Giannone, and Reichlin (2011). Uses the **EM algorithm** for parameter fitting… Can
> incorporate **monthly/quarterly mixed frequency** data along the lines of Mariano and
> Murasawa (2011). **A special case of this model is the Nowcasting model of Bok et al.
> (2017)**."*

✅ Measured — the script greps the installed docstring for each claim and all six come back
`yes`: mixed monthly/quarterly, arbitrary missing data, EM algorithm, factors unidentified,
standardises by default, news decomposition. Also verified in the Notes:

- *"The observed data may contain **arbitrary patterns of missing entries**."* — this is what
  makes the ragged edge a non-problem rather than a preprocessing step.
- *"The estimated factors and the factor loadings in this model are **only identified up to an
  invertible transformation**… This model does not impose any normalization."*
- Mixed frequency is either `endog_quarterly=` (a separate quarterly frame) or
  `k_endog_monthly=` (one frame, monthly columns first).
- `res.news(...)` decomposes a forecast revision into the contribution of each new release.

⚠️ **A dated inconsistency inside one docstring.** The prose says *Mariano and Murasawa
(2011)* and *Bok et al. (2017)*; the References section prints **Mariano & Murasawa, Oxford
Bulletin of Economics and Statistics 72(1) (2010): 27–46** and **Bok, Caratelli, Giannone,
Sbordone & Tambalotti, 2018, "Macroeconomic Nowcasting and Forecasting with Big Data", Annual
Review of Economics 10(1): 615–43**. Cite the References years.

⚠️ The docstring also names what it does *not* do: *"See also Banbura et al. (2013).
'Now-Casting and the Real-Time Data Flow.' for discussion about other types of mixed frequency
data that are not supported by this framework."*

## 2. The ragged edge, and what `dropna()` costs

Quarterly growth is the Mariano-Murasawa weighted average of five unobserved monthly growth
rates — weights `[1, 2, 3, 2, 1] / 3` — observed on the third month of each quarter and
nowhere else. Put that in the design matrix and no monthly GDP has to be imputed.

✅ Measured — the bottom of the synthetic panel at a vintage, with publication lags
`(0, 0, 1, 1, 2, 2)` months across six indicators:

| month | indicators available | GDP |
|---|---|---|
| 288 | 6 of 6 | – |
| 289 | 6 of 6 | – |
| 290 | 6 of 6 | – |
| 291 | **4 of 6** | – |
| 292 (vintage) | **2 of 6** | – |

🚨 **`dropna()` over this panel keeps 95 of 293 months and deletes every month you actually
want a nowcast for.** The whole edge is the object of interest. A state-space filter drops the
missing *rows of the observation equation* for that period instead — five lines — and keeps
the months.

## 3. ✅ Measured — factor recovery, and the revision a nowcast owes itself

At the **true** parameters, so this measures the filter and not an estimator:

| | filtered | smoothed |
|---|---|---|
| correlation with the true factor | **0.9692** | 0.9775 |
| RMSE against the true factor | 0.3584 | **0.3071** |

⚠️ **The smoothed estimate is better because it uses the future, and it is the one you must
not trade.** Same distinction, same direction, as
`../../../fin-core/skills/regime-detection/SKILL.md` measures for regime labels.

✅ Measured at the edge — the filtered value now against the value the same month settles at
once the panel fills in:

| age | series available | filtered now | settled later | revision |
|---|---|---|---|---|
| 5 | 7 | −0.7731 | −0.7889 | −0.0157 |
| 4 | 6 | −0.0335 | −0.0959 | −0.0624 |
| 3 | 6 | 0.3447 | 0.2952 | −0.0494 |
| 2 | 6 | −0.2051 | −0.0632 | **+0.1419** |
| 1 | **4** | 0.8613 | 0.8214 | −0.0399 |
| 0 | **2** | −0.4393 | −0.4176 | +0.0217 |

Mean |revision| **0.0552**; mean |error| against the truth **0.5904** now versus **0.5558**
settled. 🔑 **A nowcast revises itself, and the revision is not a bug — it is the ragged edge
closing.** Report the revision alongside the level, and never score a nowcast against a
back-filled panel that had already closed.

## 4. ✅ Measured — the benchmarks it has to beat

RMSE of the quarterly growth nowcast by position in the data flow. `offset` is months from the
quarter's last month to the vintage: −4 is before the quarter starts, 0 is its final month, +1
is a month after it ends and still before any official estimate. 79 quarters:

| offset | **DFM** | mean | AR(1) | bridge | DFM / AR(1) | DFM / bridge |
|---|---|---|---|---|---|---|
| −4 | 1.6658 | 2.0210 | 2.0928 | – | 0.796 | – |
| −2 | 0.7367 | 1.9983 | 1.7566 | 1.2341 | 0.419 | 0.597 |
| **0** | **0.3732** | 1.9983 | 1.7566 | 0.8607 | **0.212** | **0.434** |
| +1 | 0.3697 | 1.9983 | 1.7566 | 0.8607 | 0.210 | 0.430 |

🔑 **The value of a nowcast is the slope of the first column, not its level.** It is the only
method here whose error falls as the quarter fills in — the mean and the AR(1) never move
because they do not read the monthly flow, and the bridge reads one series of it. The bridge is
blank at −4 because there is no monthly data for a quarter that has not started; that is the
honest answer, not a defect.

🚨 **Report the ratio, not the RMSE.** An RMSE of 0.37 means nothing without the 1.76 it is
being compared with, and a nowcast that beats the sample mean has cleared no bar at all.

## 5. ✅ The published record, for calibration

✅ source-verified, read from the Atlanta Fed's GDPNow FAQ
(`atlantafed.org/research-and-data/data/gdpnow`):

> *"Since we started tracking GDP growth with versions of this model in 2011, the average
> absolute error of final GDPNow forecasts is **0.77 percentage points**. The root-mean-squared
> error of the forecasts is **1.17 percentage points**. These accuracy measures cover initial
> estimates for **2011:Q3–2025:Q2**."*

and, on the same page:

> *"When back-testing with revised data, the root mean-squared error of the model's out-of
> sample forecast with the same data coverage that an analyst would have just before the
> 'advance' estimate is **1.15 percentage points** for the **2000:Q1–2013:Q4** period."*
>
> 🚨 *"**Overall, these accuracy metrics do not give compelling evidence that the model is more
> accurate than professional forecasters.** The model does appear to fare well compared to
> other conventional statistical models."*

🔑 **The best-known nowcast in the world says, on its own page, that it does not beat
forecasters.** Size your expectations accordingly, and note that "final GDPNow" means the
forecast made just before the BEA advance estimate — the easiest one it makes.

🚨 **The New York Fed Staff Nowcast has a two-year hole.** ✅ source-verified at
`newyorkfed.org/research/policy/nowcast`: *"Updates to the New York Fed Staff Nowcast were
**suspended between September 2021 and September 2023**."* The cause, in the Fed's words: *"The
COVID-19 pandemic generated considerable uncertainty and volatility with respect to
macroeconomic data, which posed important challenges to the … model."* The relaunched model is
different — *"still based on a dynamic factor model and still employs Kalman-filtering
techniques. The current model, however, is estimated with Bayesian techniques"*, adding
*"time-varying volatility as well as variance outliers"*. It is published *"each Friday (except
on federal holidays) at or shortly after **12:45 p.m.**, using data available up to 10 a.m."*

**Downloading "the NY Fed Staff Nowcast" as one long feature splices two models across a
two-year gap.** ❓ No accuracy figure appears on that page — see the unverified list.

## 6. ✅ Measured — what `DynamicFactorMQ` recovers

Fitted on 180 months, 6 monthly indicators plus quarterly GDP, one factor, EM,
`maxiter=60`, 2–5 s, 16 parameters:

| | value |
|---|---|
| factor vs the **true** factor | corr **0.9735** |
| factor vs this script's numpy filter | corr **0.9989** |
| factor AR(1): true 0.720 | estimated **0.686** |
| all six monthly loadings negative | **True** |
| loadings vs `λ_i / sd(x_i)`, one common scale | corr **0.9879**, worst relative error **9.0 %** |

🚨 **Every loading came back negative.** That is the identification caveat in the docstring
arriving in your results: the factor and the loadings are determined only up to an invertible
transformation, so a sign flip and a common rescaling are free. ✅ Measured: divide the
estimated loadings by a single common scale (0.7593) and they line up with the truth the model
can actually see — `λ_i / sd(x_i)`, because it standardises by default. **Compare correlations,
ratios and fitted values. Never read a raw loading, and never compare loadings across two
fits.**

## 7. Traps

- 🚨 **`dropna()` before fitting.** §2. It deletes the edge, which is the whole point.
- 🚨 **Imputing the missing edge instead of leaving it missing.** Forward-filling a monthly
  indicator into the vintage month invents an observation with zero uncertainty; the filter's
  own conditional mean already does this, correctly, and carries the variance.
- 🚨 **Reading raw loadings, or comparing them across fits.** §6.
- 🚨 **Scoring against a back-filled panel.** The evaluation must reconstruct the panel *as of*
  each nowcast date. The inputs are revised too — `../real-time-macro-backtesting/SKILL.md` —
  and the monthly indicators are seasonally adjusted, which is a second vintage
  (`../seasonal-adjustment-and-x13/SKILL.md`).
- 🚨 **Comparing to the sample mean and stopping.** §4. Beat an AR(1) and a bridge, and report
  the ratio.
- 🚨 **Splicing a published nowcast series.** §5: the NY Fed's has a two-year hole and a model
  change across it; GDPNow's own page links a "Modifications to GDPNow Model" document. A
  published nowcast is a model's output at a date, not a measurement.
- 🚨 **Using the smoothed factor as a signal.** §3. Filtered for decisions, smoothed for
  description.
- ⚠️ **The vintage of the target matters as much as the inputs.** GDPNow is scored against the
  BEA *advance* estimate — the first print — not against today's revised GDP. Pick one, say
  which, and never mix them inside one error series.
- 🚨 **Every specification is a trial.** Number of factors, factor order, `idiosyncratic_ar1`,
  which indicators, `maxiter`. Log them in
  `../../../fin-core/skills/backtest-validation/scripts/trial_ledger.py`.
- ⚠️ **MIDAS is not `pip install midaspy`.** ⚠️ Secondhand from a prior research pass, not
  re-verified here: PyPI `midaspy` is a multiple-imputation package and PyPI `midas` is a
  gas-detector driver; neither is mixed-data-sampling regression. If you want MIDAS rather than
  a factor model, expect to write the Almon/beta weighting yourself.

## ❓ Not verified

The New York Fed Staff Nowcast's **accuracy** — no figure appears on its page, and none is
quoted here · the exact relaunch date in September 2023 (the page gives only "September 2023")
· whether the pre-2021 and post-2023 NY Fed series are published as one downloadable file ·
GDPNow's performance against Blue Chip quarter by quarter (only the Fed's own summary sentence
is quoted) · the MIDAS package names above.

## 8. Scripts and where this sits

`scripts/nowcast.py` — the seeded mixed-frequency panel with per-series publication lags, the
`ragged()` vintage view, a Kalman filter/smoother with a time-varying observation dimension
carrying the Mariano-Murasawa aggregation in the design matrix, factor recovery and the
edge-revision table, the four-way benchmark evaluation by data-flow position, and the
`DynamicFactorMQ` docstring audit and fit. numpy + pandas + scipy; `statsmodels` imported
inside functions and §6 skipped cleanly without it. Seed 20260909, 6–25 s.

- The Kalman filter and smoother themselves, and the filtered/smoothed distinction —
  `../../../fin-models/skills/state-space-and-kalman/SKILL.md`.
- Forecast baselines, scoring rules and why a naive benchmark is mandatory —
  `../../../fin-models/skills/time-series-forecasting-models/SKILL.md`.
- The vintages of every input series — `../real-time-macro-backtesting/SKILL.md`.
- The seasonal adjustment inside those inputs — `../seasonal-adjustment-and-x13/SKILL.md`.
- When each indicator actually lands, to the minute —
  `../macro-release-calendar-and-embargo/SKILL.md`.
- Recession probabilities as a different question about the same data —
  `../macro-regime-and-recession-indicators/SKILL.md`.
- Where GDP, the monthly indicators and `GDPNOW` come from —
  `../../../fin-core/skills/fundamental-and-macro-data/SKILL.md`.
