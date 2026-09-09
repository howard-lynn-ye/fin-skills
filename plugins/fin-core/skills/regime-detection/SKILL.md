---
name: regime-detection
description: >-
  Detect and label market regimes without letting the labels see the future, and state regime
  coverage in the form the result gate demands. TRIGGER - detect market regimes, regime
  detection, bull bear regime labels, volatility regime, high-vol low-vol state, risk-on
  risk-off; hidden markov model on returns, HMM, hmmlearn, markov switching, MarkovRegression,
  smoothed vs filtered probabilities; change point detection, ruptures, structural break;
  turbulence index, Mahalanobis distance; "my strategy only works in one regime", "does it
  survive 2008 or 2020"; "result_manifest says no regime coverage", regimes_covered. SKIP for
  forecasting volatility itself with GARCH or arch (volatility-models), for whether a
  regime-conditional result survives the trials behind it (backtest-validation), for the full
  pre-report audit (research-integrity-guards), and for RL or deep-learning state models
  (rl-and-ml-trading).
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-08"
---

# Regime detection

**The regime chart you are looking at was drawn with the answer key.** A Markov-switching or HMM
fit returns *smoothed* probabilities, and smoothed means conditioned on the whole sample: the label
on the first day of a crash knows what happened in the weeks that followed. A strategy switched on
those labels has been told the crash was coming. This skill measures what that is worth, what the
honest alternatives lag by, and how to state regime coverage so that
`../research-integrity-guards/scripts/result_manifest.py` stops refusing your result card.

Every number below is printed by `scripts/regime_lookahead.py`, `scripts/regime_methods.py` or
`scripts/regime_coverage.py` (numpy / pandas / statsmodels, fixed seeds). ✅ `statsmodels` 0.15.0
(2026-08-30), BSD-3-Clause, is the only estimation dependency. The Hamilton filter and Kim
smoother are re-implemented in numpy inside the headline script and agree with statsmodels to
6.1e-16 (predicted), 6.7e-16 (filtered) and 1.4e-14 (smoothed), so the timing claims are checked
against the library rather than assumed from it.

## 1. 🚨 Three probability series; only one is tradeable

| Series | statsmodels attribute | Conditions on | Usable at the close of t-1? |
|---|---|---|---|
| smoothed | `smoothed_marginal_probabilities` | r_1..r_N, the whole sample | no: it has seen the future |
| filtered | `filtered_marginal_probabilities` | r_1..r_t | no: it needs today's return, the one you are about to trade |
| predicted | `predicted_marginal_probabilities` | r_1..r_{t-1} | yes, and only with parameters estimated before t |

✅ Verified: `predicted[t] == P @ filtered[t-1]` to 2.2e-16, where
`P = res.regime_transition[:, :, 0]` is **column-stochastic** (`P[to, from]`, column sums 1.0).
The textbook row-stochastic form is its transpose; use the wrong one and the predicted series is
silently wrong.

**What the leak is worth.** `scripts/regime_lookahead.py` simulates 2,520 days of two-regime
returns (calm: mean +0.06 %/day, vol 0.8 %; turbulent: -0.10 %/day, vol 1.6 %; expected durations
100 and 25 days), fits `MarkovRegression(k_regimes=2, trend='c', switching_variance=True)`, and
runs "long when P(calm) > 0.5, else flat" over the last 1,764 days with each series. Sharpe is
mean/sd x sqrt(252) at rf = 0.

| Signal (seed 0, vol ratio 2.0) | Sharpe | max DD | label acc. | median delay | flagged early | false alarms |
|---|---|---|---|---|---|---|
| buy & hold | 1.187 | -31.1 % | - | - | - | - |
| oracle: true regime at t-1 | 1.542 | -9.6 % | - | - | - | - |
| **smoothed** (full sample) | 1.528 | -11.1 % | 0.947 | 0 d | **47 %** | 0 |
| filtered (same day) | 1.616 | -11.3 % | 0.918 | 3 d | 19 % | 12 |
| predicted (in-sample params) | 1.753 | -9.4 % | 0.907 | 3 d | 13 % | 14 |
| **walk-forward predicted** (params refit yearly, past only) | 1.789 | -9.4 % | 0.905 | 3 d | 19 % | 16 |

🚨 **The honest reading is not "look-ahead inflates Sharpe".** On this path the smoothed
strategy is *worse* than the walk-forward one by 0.261 Sharpe, and at the true parameters (no
estimation at all) smoothed still trails predicted by 0.170. What the look-ahead reliably buys is
the *shape* of the labels: accuracy 0.947 against 0.905, zero false alarms against 16, zero median
delay against three days, and the new regime flagged **before it started** in 47 % of switches,
because the smoother reads the first days of a crash off the days that follow. Any rule tuned to
those labels is tuned to information a live system never has, and the clean chart is what
convinces people the regime was obvious.

**How the gap behaves.** Medians over three seeds, same window, in-sample parameters; `gap` is
smoothed minus predicted Sharpe, `gap@true` the same at the true parameters, and `gap fp` is
filtered minus predicted, the same-day leak on its own:

| Configuration | B&H | oracle | smoothed | predicted | gap [min, max] | gap@true | gap fp | acc sm / pred | delay filt / sm | false alarms | missed |
|---|---|---|---|---|---|---|---|---|---|---|---|
| vol ratio 1.25, 25-day episodes | 0.93 | 1.50 | -0.06 | 0.67 | 0.02 [-0.10, 0.15] | 0.39 | 0.13 | 0.84 / 0.82 | -12 / -12 | 0 | 13 |
| vol ratio 1.5 | 0.85 | 1.51 | 1.58 | 1.41 | 0.27 [0.16, 0.76] | 0.21 | 0.20 | 0.89 / 0.85 | 3.5 / -1 | 44 | 4 |
| vol ratio 2.0 | 0.71 | 1.54 | 1.53 | 1.65 | 0.25 [-0.23, 0.46] | 0.30 | 0.28 | 0.94 / 0.90 | 3 / 0 | 28 | 3 |
| vol ratio 3.0 (easy) | 0.52 | 1.59 | 1.64 | 1.50 | 0.06 [-0.19, 0.47] | 0.09 | 0.25 | 0.97 / 0.94 | 2 / 0 | 12 | 2 |
| vol ratio 2.0, **7-day crashes** at -0.4 %/day | 1.21 | 1.59 | 1.53 | 1.33 | **0.32 [0.20, 0.62]** | 0.33 | 0.28 | 0.97 / 0.94 | 0 / -1 | 0 | 9 |

- On easy data the causal signal is only modestly worse: at ratio 3.0 the median gap is 0.06 and
  the seed range straddles zero, as it does at ratio 2.0. Single-path differences of about 0.2
  Sharpe are noise here; the two oracles, which differ on 34 days, differ by 0.12.
- The same-day leak in `filtered` is worth a median 0.20 to 0.28 Sharpe wherever the fit finds the
  regimes, which is as much as the whole smoothed-vs-predicted gap. "I used the filtered
  probabilities, so no look-ahead" is the most common wrong sentence in this area.
- The gap is positive in every seed only when episodes are **short relative to the detection
  delay**: 7-day crashes, where a 3-day delay misses half the episode and the filtered series
  misses 9 switches outright. That is the regime a look-ahead label flatters most and a live
  detector handles worst.
- At ratio 1.25 the fit does not find the regimes at all (accuracy 0.84 against a never-flag base
  rate of 0.81); every series is noise and the gap is 0.02. Below some separation there is nothing
  to detect, and a chart that says otherwise was drawn from the smoother.
- ⚠️ The sweep uses in-sample parameters. On the main path the parameter leak was worth -0.036
  Sharpe (predicted minus walk-forward), i.e. nothing measurable. The labels, not the parameters,
  are where the future gets in.

## 2. Methods: what each needs, leaks and lags

`scripts/regime_methods.py` runs the rule-based detectors on the same kind of data (three assets,
correlation 0.3 calm / 0.8 turbulent), every flag computed from data through t-1 unless marked
LEAKS, position "long unless flagged". True turbulent share in the window 0.193; buy & hold Sharpe
0.772; the one-day-lagged oracle 1.125.

| Detector | Needs | Leaks | acc | delay in / out (d) | false alarms | Sharpe |
|---|---|---|---|---|---|---|
| Markov switching, predicted (statsmodels) | returns, k, several starts | parameters if fit in-sample; **labels if smoothed** | 0.886 | 3 / 3 | 38 | 1.205 |
| 21-d realized vol > fixed threshold | a threshold fixed in advance | the threshold, if chosen by looking | 0.837 | 6 / 11 | 0 | 0.779 |
| 21-d vol > expanding 75th pct | a year of history | nothing | 0.846 | 6 / 8 | 2 | 0.775 |
| 21-d vol > full-sample 75th pct | - | **the threshold**: +0.118 Sharpe vs the expanding version, accuracy -0.009 | 0.837 | 8 / 9 | 0 | 0.892 |
| turbulence index, expanding mean/cov (⚠️ Kritzman-Li) | a multi-asset panel and history for its covariance | nothing | 0.855 | 7 / 0 | 20 | 1.011 |
| turbulence index, full-sample cov | - | **the covariance**: -0.056 Sharpe, accuracy -0.008 | 0.848 | 8 / 0 | 16 | 0.955 |
| price < 200-day SMA | prices | nothing | 0.722 | 6 / 0 | 39 | 0.442 |
| downtrend AND high vol (quadrant) | both | nothing | 0.801 | 10 / 0 | 24 | 0.552 |
| drawdown > 10 % from peak | prices | nothing; late by construction | 0.735 | 6 / 0 | 17 | 0.695 |

Reading it: the model is the fastest detector (3 days each way) and the noisiest (38 false
alarms). Trailing-window rules lag by about half the window on the way out (7-11 days for a 21-day
window). The trend rules are worse than doing nothing on this data: the 200-day SMA label agrees
with the true regime 72 % of the time against 81 % for never flagging, and both trend rules cut the
Sharpe (0.44-0.55 against 0.77) while deepening the drawdown. The full-sample leaks are small **on
this stationary series**, and their sign even flips between the two detectors, which is exactly why
they go unnoticed: they matter when the vol level trends, and then the threshold was set by years
the strategy had not lived through.

**When to use which.** Markov switching or an HMM when you need a probability and can live with
false alarms, and only ever through the predicted series under walk-forward parameters. A
realized-vol threshold when the rule has to be pre-registered and explainable; fix the number
before the test window and record it as a trial. The turbulence index when you hold a panel and
the thing you fear is a correlation break rather than a vol spike; it needs a covariance history
and it exits fast. Trend/vol quadrants, change-point segmentation and macro dates for
*describing* which regimes a period contained (section 4), not for switching positions: the first
lags, the second is ex-post by design, the third arrives months late. Whatever you pick, the
oracle row is the ceiling: a perfect detector that is one day late scored 1.125 here against
1.205 for the model with its 38 false alarms, so the noise, not the delay, is where the model's
detections are spent.

**Libraries, checked 2026-09-08 against the PyPI JSON API and the GitHub REST API:**

- ✅ `statsmodels` 0.15.0 (2026-08-30), BSD-3-Clause, repository pushed 2026-09-06.
  `MarkovRegression` and `MarkovAutoregression` in `statsmodels.tsa.regime_switching`;
  `switching_variance=True` for vol regimes; `exog_tvtp=` for time-varying transition
  probabilities; `fit(search_reps=, rng=)` for random starts; `res.expected_durations`,
  `res.mle_retvals['converged']`.
- ✅ `hmmlearn` 0.3.3 (2024-10-31), BSD-3-Clause, not archived, **no push to the repository since
  2024-10-31**, 3,420 stars, 80 open issues plus PRs. ✅ Read from `src/hmmlearn/base.py` on
  `main`: `predict()` decodes with Viterbi and `predict_proba()` returns forward-backward
  posteriors; the docstring says both are computed "given all emissions". The public API exposes
  **no filtered (forward-only) probability**, so every hmmlearn label is a smoothed label. Not
  installed here.
- ✅ `ruptures` 1.1.10 (2025-09-10), BSD-2-Clause, `requires_python <3.14,>=3.9`, a 1.1.11rc1 on
  2025-11-27, repository pushed 2026-07-06. Its README's first sentence is that it does
  **off-line** change point detection: it segments the whole signal at once, so every change point
  it returns is an ex-post label. Fine for describing history; not a signal. Not installed here.
- ✅ `pymc` 6.3.2 (2026-09-08), Apache-2.0, `requires_python >=3.12`. Bayesian switching and
  change-point models are hand-built on it; nothing about them was verified here.
- GARCH-family volatility models belong to `../factor-and-timeseries-research/SKILL.md`. A GARCH
  forecast is a continuous vol state, not a regime label, until you add a threshold, and the
  threshold is a trial (section 3).
- ✅ Macro-defined regimes (NBER recessions, policy cycles) are ex-post by construction: the NBER
  announced the February 2020 peak on 2020-06-08 and the April 2020 trough on 2021-07-19
  (nber.org, Business Cycle Dating Committee announcements, read 2026-09-08). A backtest that
  switches on the recession flag on the recession's first day uses a label that arrived four to
  fifteen months later.

## 3. Every regime definition is a trial

The regime label is not data; it is a model with free choices, and each choice is a trial in the
sense of `../backtest-validation/SKILL.md`: the number of regimes, the feature (returns, vol,
correlation, a macro series), the window, the threshold or percentile, the smoothing, the
probability cut, the refit cadence, and which regime gets called risk-off. `regime_methods.py`
lists nine such choices behind its own table. "The strategy works in the low-vol regime" after
trying four regime definitions is a best-of-four result, and
`../backtest-validation/scripts/trial_ledger.py` is where the four are recorded, before the
out-of-sample number is looked at.

🚨 The worst version: define the regime on the test period, by whichever split makes the
strategy's per-regime Sharpe look best, then report the good regime. That is the label leak and
the trial leak at once.

## 4. Reporting regime coverage for `result_manifest.py`

The gate fails on `regimes_covered == []` and accepts any list of strings, so the strings have to
carry the evidence. `scripts/regime_coverage.py` prints them in this form:

```
"<label> [<ex-ante rule | ex-post label>]: <n obs> obs / <n episodes> episodes, <first>..<last>,
 asset <ann. return>, strategy Sharpe <x> (in market <p>) maxDD <y>"
```

For each label: the **definition** (threshold and window, and whether it was fixed before the test
period), **n_obs** and share, **n_episodes** and the longest episode, the date range, the asset's
own return in the regime, and the strategy's Sharpe, drawdown and **time in market** there. Time
in market is not optional: in the demo the "drawdown > 10 %" regime shows a strategy Sharpe of 1.40
with the strategy in the market 31 % of the time, over 11 episodes with a per-episode t-stat of
1.00. That number reads as strength and is a strategy sitting out.

Ex-post labels (a smoothed Markov fit, NBER dates, the true regime of a simulation) are acceptable
for *describing* coverage as long as they are marked ex-post; they are not acceptable as inputs to
the strategy being described. ✅ The demo card with six such strings clears the regime check
(`problems()` no longer contains "no regime coverage stated"); with `[]` it fails. The card's
other problems belong to other skills and are left alone.

## 5. Traps

- 🚨 **`converged=True` is not "found the regimes".** Started from equal regimes, the fit reports
  converged at log-likelihood 7901.0 against 8104.6 for the real optimum (gap 203.6), with both
  regimes identical: one regime, twice. And `fit(search_reps=5)`, the random-start search, landed
  on a *worse* optimum than the default start in 10 of 15 sweep fits (largest gap 187.8; it won
  once). Its losing solution on the main path has expected durations of 90 and **1.0** days; a
  one-day regime is an outlier detector wearing a regime's name. Fit from several starts, keep the
  highest likelihood, and reject any regime whose expected duration is a day.
- 🚨 **Label switching.** Two fits with the identical likelihood 8104.621 return the calm regime
  as index 0 in one and index 1 in the other. Never hard-code "regime 0 is calm": identify regimes
  by a parameter (`argmin(sigma2)`), re-identify after every refit, and expect the walk-forward
  loop to flip.
- 🚨 **Regimes fitted on the test period.** A smoothed label on the test period is the test
  period's future, read backwards. Use predicted probabilities under walk-forward parameters, or
  accept that the regime chart is descriptive only.
- 🚨 **Prices instead of returns.** A two-regime fit on the log price reports `converged=True`
  with expected durations of 2,476 and 2,554 days; its label agrees with the true regime 52.8 % of
  the time (chance is 50 %) and with "price above its full-sample median" 97.5 % of the time. It
  found the level of the series, not its regimes.
- 🚨 **"The 2008 regime" is one observation.** A per-regime Sharpe is computed from n_obs days,
  but "it survives crises" is a claim about n_episodes things. In the demo window the turbulent
  regime has 341 days and 17 episodes; the per-episode return has a t-stat of 0.94 over those 17,
  and the worst episode is 44 % of all episode losses. Report episodes next to observations. A
  regime seen once is a case study.
- 🚨 **`filtered` is not "past only".** It uses today's return to label today. Across the sweep
  the same-day leak is worth a median 0.13 to 0.28 Sharpe against predicted (it happened to be
  -0.137 on the single main path, which is what single paths do); in the 7-day-crash
  configuration the filtered series "detects" the crash with zero delay because the crash day is
  in its conditioning set. Lag it, or use `predicted_marginal_probabilities`.
- 🚨 **Transition matrix orientation.** `regime_transition[:, :, 0]` is `P[to, from]` with column
  sums of 1. `predicted[t] = P @ filtered[t-1]`, not `filtered[t-1] @ P`.
- ✅ statsmodels arms its own `"always"` warning filters at import time
  (`statsmodels/tools/sm_exceptions.py`), so a `catch_warnings` block that also triggers the
  first import does not silence the EM `EstimationWarning`. Import at module level first; the
  script does, and its stderr is empty.

## 6. Scripts

- `scripts/regime_lookahead.py` - sections 1 and 5: the ladder, the sweep, the pathologies, and
  the numpy Hamilton filter / Kim smoother checked against statsmodels. About 80 s. Without
  statsmodels it runs the true-parameter timing comparison only and says so.
- `scripts/regime_methods.py` - section 2: rule-based detectors, their delays, false alarms and
  leak sizes. Imports the simulator from `regime_lookahead.py`; about 3 s.
- `scripts/regime_coverage.py` - section 4: the per-regime table, the `regimes_covered` strings,
  per-episode outcomes, and the `result_manifest.py` check. Under 1 s.

## 7. Where this sits

`../research-integrity-guards/SKILL.md` decides whether a result is real and owns the gate;
`../backtest-validation/SKILL.md` counts the trials, regime definitions included;
`../factor-and-timeseries-research/SKILL.md` owns GARCH and volatility forecasting;
`../portfolio-and-risk/SKILL.md` owns the Sharpe conventions used above;
`../../../fin-llm/skills/rl-and-ml-trading/SKILL.md` owns learned state representations; and
`../../../fin-llm/skills/llm-finance-agents/SKILL.md` section 3 is the other place the
regime-coverage demand is made.
