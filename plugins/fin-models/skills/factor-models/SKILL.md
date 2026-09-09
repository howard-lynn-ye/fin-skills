---
name: factor-models
description: >-
  Build long-short factor portfolios from a characteristic panel and test the alpha with standard
  errors that survive serial correlation. TRIGGER - factor model, Fama-French, Fama-MacBeth,
  cross-sectional regression, decile or quintile long-short sort, 2x3 sort, SMB and HML, value-
  weight vs equal-weight portfolio, characteristic panel, alpha t-stat, Newey-West, HAC standard
  errors, cov_type="HAC" maxlags, Ken French Data Library, F-F_Research_Data_Factors, book-to-
  market, 11-1 momentum; "my factor has a t-stat of 15", "should I lag the signal", "my HML does
  not match Ken French", "joining monthly factors to daily returns". SKIP for scoring one alpha
  signal with alphalens, IC decay or GARCH (factor-and-timeseries-research), for the covariance
  matrix a factor model implies (covariance-and-risk-models), for turning expected returns into
  weights (portfolio-optimizers), and for counting the specifications you tried
  (backtest-validation).
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-09"
---

# Factor models

**A factor portfolio is a subscript problem before it is a statistics problem.** The formula for
HML is public and takes one line; what decides whether your HML is a factor or a look-ahead is
*when* the characteristic was measured relative to the return it earns. Get that wrong and the
t-statistic is not weak evidence, it is manufactured evidence, and every standard error correction
downstream is beside the point.

> **characteristic known at t -> portfolio formed at t -> return measured over t -> t+1**

Every number below is printed by `scripts/factor_regression.py` (numpy / pandas, seed 0, about
3 s, statsmodels optional). The panel is 500 stocks x 240 months in which **the truth is alpha = 0
and there is no value premium**: `r_it = beta_i * m_t + e_it`, book equity is a slow random walk,
so book-to-market moves almost entirely with the stock's own price. Anything a sort finds in it is
an artefact.

## 1. 🚨 Forming on the contemporaneous characteristic manufactures the alpha

✅ Measured, seed 0, top 30 % minus bottom 30 %, re-sorted monthly, Newey-West lags
`floor(4(T/100)^(2/9))` = 4. `mean` and `CAPM a` are percent per month; `CAPM a` is the intercept
of the long-short return on the value-weighted market:

| formation | mean %/mo | NW t | CAPM alpha % | HAC t | beta |
|---|---|---|---|---|---|
| **lag 1, value-weight** (B/M at t-1) | -0.071 | -0.67 | **+0.002** | **0.02** | -0.13 |
| lag 1, equal-weight | -0.038 | -0.47 | -0.003 | -0.03 | -0.07 |
| lag 6, value-weight (Fama-French-style delay) | -0.104 | -0.93 | -0.033 | -0.31 | -0.12 |
| 🚨 **LEAK lag 0, value-weight** (B/M at t) | -1.378 | -14.78 | **-1.318** | **-14.63** | -0.11 |
| 🚨 LEAK lag 0, equal-weight | -1.870 | -19.50 | -1.846 | -19.46 | -0.04 |
| **11-1 momentum, lag 1** (months t-11..t-1) | +0.065 | 0.46 | **+0.022** | **0.16** | 0.07 |
| 🚨 **LEAK momentum, lag 0** (window ends in month t) | +5.198 | 50.30 | **+5.180** | **52.36** | 0.03 |

**Read the two bolded pairs.** The same panel, the same code, the same true alpha of zero: formed
honestly the alpha is +0.002 %/month with t = 0.02; formed on the contemporaneous characteristic
it is **-1.318 %/month with t = -14.63**, or **+5.180 %/month with t = 52.36** if the leaked
signal is momentum instead of value.

🚨 **The sign is a distraction.** The leaked book-to-market sort is negative because a stock that
fell during month t has a *higher* B/M at the end of month t: "value" becomes a list of this
month's losers, and the long-short is short the winners. Sort the identical panel on a signal
whose leak points the other way — a momentum window whose last month is the month being
predicted — and the same bug reads as a 5 %/month premium with a t-statistic of 52. **A leak
does not announce itself with an implausible sign; it announces itself with an implausible
t-statistic.** A t of 15 on 240 months of a single characteristic is not a discovery.

**The guard.** `sort_portfolio()` and `fama_french_2x3()` refuse `lag=0` unless you pass
`allow_lookahead=True`, so a pipeline can call them and fail loudly rather than print a great
number:

```
ValueError: lag=0: the sorting characteristic would be measured at the end of the return
period it is supposed to predict (a one-period look-ahead). ...
```

**Value-weight unless you mean to bet on microcaps.** ✅ Measured: the equal-weight leak is
40 % larger than the value-weight leak (-1.846 vs -1.318 %/month alpha), because equal weighting
puts the same money in the panel's 12 %-a-month idiosyncratic minnows as in its 5 % giants. The
honest rows differ by almost nothing (+0.002 vs -0.003). Equal weighting does not fix a leak; it
amplifies one, and it changes what the portfolio is a bet on.

## 2. The 2x3 sort, and what Ken French actually publishes

✅ Source-verified 2026-09-09 at Ken French's Data Library
(`mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/`, factor and six-portfolio
description pages):

- The factors come from **"6 value-weight portfolios formed on size and book-to-market"** — they
  are already value-weighted; do not re-weight them.
- **SMB = 1/3 (SV + SN + SG) - 1/3 (BV + BN + BG)** and
  **HML = 1/2 (SV + BV) - 1/2 (SG + BG)**, verbatim from the factor page. `fama_french_2x3()`
  implements exactly these two averages.
- **Rm-Rf** is the value-weighted return of all CRSP firms incorporated in the US and listed on
  NYSE, AMEX or NASDAQ, minus the one-month Treasury bill rate. It is an *excess* return; SMB and
  HML are long-short and already excess.
- Breakpoints: **"The size breakpoint for year t is the median NYSE market equity at the end of
  June of year t"** and **"The BE/ME breakpoints are the 30th and 70th NYSE percentiles"** —
  NYSE-only breakpoints applied to a NYSE + AMEX + NASDAQ universe. A full-universe breakpoint
  puts far more names in the small bucket and is a different factor.
- Portfolios are **formed at the end of each June and held from July of year t through June of
  t+1** — an annual re-form, not a monthly re-sort.
- 🔑 **BE/ME for June of year t is book equity for the last fiscal year end in t-1 divided by ME
  for December of t-1.** So the accounting number is 6 to 18 months stale by the time it earns a
  return, and market equity in the ratio is from the *previous December*, not from June. That gap
  is deliberate: it is the delay that guarantees the data was public.
- Files exist at **daily, weekly, monthly and annual** frequency (the site listed July 1926 -
  July 2026 for the first three, 1927 - 2025 annual, on 2026-09-09).

✅ Measured on the synthetic panel (same seed, same NW lags), where truth is again zero:

| factor | mean %/mo | NW t | CAPM alpha % | HAC t |
|---|---|---|---|---|
| SMB, lag 1, re-sorted every month | -0.021 | -0.30 | +0.010 | 0.14 |
| SMB, lag 1, re-sorted every 12 months | +0.038 | 0.56 | +0.075 | 1.06 |
| 🚨 SMB, LEAK lag 0 | -0.123 | -1.73 | -0.089 | -1.20 |
| HML, lag 1, re-sorted every month | -0.016 | -0.18 | +0.024 | 0.27 |
| HML, lag 1, re-sorted every 12 months | -0.074 | -0.76 | -0.042 | -0.45 |
| 🚨 HML, LEAK lag 0 | -1.616 | -18.84 | -1.585 | -18.97 |

The leak reaches HML through the B/M sort (t = -18.97) and SMB through the size sort — a stock
that fell is "small" at the end of the month — but SMB's leak is an order of magnitude weaker
(-0.089, t = -1.20) because market cap is dominated by its own cross-sectional spread rather than
by one month of return. **A leak in a two-way sort shows up in one leg and hides in the other.**

⚠️ Reproducing the published factors is a different job from implementing the formulas. Monthly
re-sorting, full-universe breakpoints and a contemporaneous ME in the ratio each give a series
that satisfies the formulas above and correlates imperfectly with the file you downloaded. If your
HML "does not match Ken French", check the re-form calendar and the breakpoint universe before
you check your code: `references/ken_french_data_library.md` has the full timing diagram, the two
different market equities in the ratio and the sort, and a seven-step reproduction checklist.

## 3. Newey-West in statsmodels: three call sites, three different answers

✅ Source-verified in the installed statsmodels **0.15.0**
(`statsmodels/stats/sandwich_covariance.py`, `statsmodels/base/covtype.py`):

| what you call | `nlags` / `maxlags` | small-sample correction |
|---|---|---|
| `OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": L})` | **required** — `kwds["maxlags"]`, `KeyError` without it | **off**: `kwds.get("use_correction", False)` |
| `sw.cov_hac_simple(res, nlags=L)` | optional | **on**: `use_correction=True` is the signature default |
| `sw.cov_hac_simple(res, nlags=None)` | `int(np.floor(4 * (T/100)**(2/9)))` inside `S_hac_simple` | on |

- ✅ `weights_bartlett(nlags)` returns `1 - np.arange(nlags + 1) / (nlags + 1.0)`, so the weight
  on lag j is `1 - j/(L+1)` and **L is the highest lag included, not counting lag 0**. An
  implementation that divides by L instead of L+1, or that treats L as a count of terms, gives
  different standard errors from statsmodels on the same data.
- 🚨 **The `floor(4(T/100)^(2/9))` rule is not applied by `fit()`.** It lives in `S_hac_simple`
  and only fires when `nlags is None`. ✅ Measured: `fit(cov_type="HAC")` with no `maxlags` raises
  `KeyError: 'maxlags'`. There is no default lag length in the `fit()` path — you must state one,
  and you should state it in the paper too.
- ✅ Measured: the module's own numpy implementation matches
  `fit(cov_type="HAC", cov_kwds={"maxlags": 4})` to **1.04e-17** on standard errors and
  **5.18e-19** on coefficients. statsmodels labels the result "Standard Errors are
  heteroscedasticity and autocorrelation robust (HAC) using 4 lags and without small sample
  correction".
- ✅ Measured: `cov_hac_simple(res, nlags=4)` standard errors are **1.00421054x** the `fit()`
  ones, exactly `sqrt(T/(T-k))` = 1.00421054. Same estimator, two defaults; quoting the ratio is
  how you tell which one produced a t-statistic.
- ✅ Measured on the honest long-short: plain OLS SE 0.001071, HAC(0) = White HC0 SE 0.001049,
  HAC(4) SE 0.000990. **HAC standard errors are not always larger.** They are larger when the
  residuals are positively autocorrelated and smaller when they are not; "I used Newey-West so I
  am being conservative" is not a valid sentence.
- ⚠️ `cov_hac_simple`'s own docstring says it is "verified only for nlags=0" and that the
  correction factor is a guess needing a reference. The lag-0 case matches White exactly; the
  general case matched this module's independent implementation to 1e-17, which is the check
  worth running rather than trusting either.

## 4. Fama-MacBeth: the leak survives the second pass

`fama_macbeth()` runs a cross-sectional regression of `ret[t]` on the characteristics at `t - lag`
every period, then tests the mean slope. ✅ Measured, 239 monthly cross-sections, slopes x 100:

| characteristic | slope (t-1) | t plain | t NW(4) | 🚨 slope (LEAK, t) | t plain | t NW(4) |
|---|---|---|---|---|---|---|
| log(B/M) | +0.003 | 0.08 | 0.07 | **-0.901** | -19.62 | **-11.61** |
| log(cap) | -0.003 | -0.19 | -0.17 | +0.009 | 0.52 | 0.40 |

Two things to take from it. First, **the two-pass procedure does not repair a subscript error** —
the leaked slope is 300x the honest one and significant at any threshold. Second, the leak's
plain t of -19.62 falls to -11.61 once the slope series is given Newey-West errors, a 41 %
haircut, because the leaked slopes are strongly autocorrelated. On the honest rows the two
t-statistics are indistinguishable (0.08 vs 0.07). 🚨 **The Fama-MacBeth standard error is the
standard deviation of the slope *series*; it fixes cross-sectional correlation and nothing else.**
Serial correlation in the slopes needs a HAC correction on top, and it is the leaked or persistent
specifications where that matters most.

## 5. Joining to the Data Library: the month is not a timestamp

✅ Measured: `pd.to_datetime("202607", format="%Y%m")` returns **2026-07-01**, a first-of-month
timestamp. Inner-joining six monthly factor rows parsed that way to a month-end strategy series
keeps **0 of 6 rows**; `french_period_index()` + `align_monthly()`, which convert both sides to a
`PeriodIndex(freq="M")`, keep **6 of 6**. A silent empty join is the most common way factor
regressions "have no data", and an inner join that silently keeps 3 of 6 is worse.

🚨 **Frequency must match on both sides.** The monthly file's row for 202607 is the return over
the whole month of July 2026; regress monthly strategy returns on it. The daily file stamps
`YYYYMMDD` and joins on the day. Regressing daily returns on monthly factors (or the reverse,
after a forward-fill) produces betas that are arithmetic, not estimates.

🚨 **Check the units.** ✅ Measured: the same decimal strategy regressed on the same factor column
gives beta **-0.004243** if the column is treated as percent and **-0.4243** if it is treated as a
decimal — a factor of exactly 100. ⚠️ This skill did not verify the units of the archive files
themselves; read the first rows of your own download and confirm before you divide by 100.

## 6. Traps

- 🚨 **`lag=0` on any characteristic derived from price.** Book-to-market, market cap, dividend
  yield, earnings yield, 1/P, and any momentum window whose last period is the return period.
  Section 1 is what it costs.
- 🚨 **A t-statistic that is too good.** On zero-alpha data, honest |t| was 0.02-0.67 and leaked
  |t| was 11.6-52.4. If your single-characteristic factor prints |t| > 10, suspect the subscript
  before the economics.
- 🚨 **`fit(cov_type="HAC")` with no `maxlags`.** It raises; there is no default. And the two
  call paths disagree by `sqrt(T/(T-k))` (section 3).
- 🚨 **Equal weighting by default.** It is a different bet (microcaps), and it amplified the leak
  by 40 % here.
- 🚨 **Re-sorting monthly and calling it Fama-French.** French re-forms once a year at the end of
  June with NYSE breakpoints and a Dec-t-1 market equity in the ratio.
- 🚨 **Survivorship in the panel.** Nothing in section 1 detects a universe that only contains
  today's listed names; the sort is honest and the panel is not. That is
  `../../../fin-core/skills/research-integrity-guards/SKILL.md`.
- 🚨 **Every breakpoint, lag and weighting scheme is a trial.** Deciles vs terciles, value vs
  equal, lag 1 vs lag 6, monthly vs annual re-form: this skill's own demo contains eight
  specifications of one idea. Record them in
  `../../../fin-core/skills/backtest-validation/scripts/trial_ledger.py` before reporting the
  best one.
- ⚠️ `linearmodels` (which ships a panel `FamaMacBeth`) is **not installed in this environment**
  and nothing about it was verified here.

## 7. Scripts

- `scripts/factor_regression.py` - all five sections. `simulate_panel`, `sort_portfolio`,
  `momentum`, `fama_french_2x3`, `newey_west`, `ts_alpha`, `fama_macbeth`,
  `french_period_index`, `align_monthly`, and `statsmodels_hac_check`, which prints the
  comparison against statsmodels rather than asserting it. Without statsmodels the numpy
  Newey-West is still the reference and section 3's cross-check says it was skipped. About 3 s.

## 8. Where this sits

`../../../fin-core/skills/factor-and-timeseries-research/SKILL.md` owns scoring a single alpha
signal (alphalens, IC, decay, turnover) and volatility forecasting;
`../../../fin-core/skills/backtest-validation/SKILL.md` counts the specifications;
`../../../fin-core/skills/research-integrity-guards/SKILL.md` owns point-in-time universes and the
result gate; `../covariance-and-risk-models/SKILL.md` turns these factors into a covariance
matrix; `../portfolio-optimizers/SKILL.md` turns expected returns into weights; and
`../../../fin-libraries/skills/lib-alphalens/SKILL.md` is the per-library deep dive for the
signal-scoring tool.
