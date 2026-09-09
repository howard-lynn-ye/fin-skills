---
name: alpha-combination-and-neutralization
description: >-
  Score several alphas, combine them, and strip the exposures you did not mean to take. TRIGGER -
  information coefficient, IC, rank IC, ICIR, IC decay, "is my IC good", IC t-stat, Newey-West on
  IC, overlapping forward returns; combining alphas, blending signals, alpha weighting, z-score
  or rank combination, multi-factor signal; sector neutral, beta neutral, market neutral signal,
  industry neutralization, residualize the alpha, cross-sectional regression residuals,
  orthogonalize signals; winsorize, clip outliers, cross-sectional standardization;
  turnover-aware combination, signal smoothing, "my alpha dies after costs". SKIP for one
  time-series trend signal (trend-following-models), for the factor library and the alphalens
  forward-return convention (factor-and-timeseries-research, lib-alphalens), for weights under
  constraints (portfolio-and-risk), and for whether the survivor is real (backtest-validation).
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-09"
---

# Alpha combination and neutralization

**Three numbers in a standard alpha research note are routinely wrong, and none of them raise.**
The IC t-stat, because the observations are not independent. The word "neutral", because the
exposure came from the wrong window. And the winsorization step, because it was placed after a
rank and clips nothing.

Everything marked ✅ Measured comes from `scripts/alpha_combine.py` — numpy + pandas + scipy,
seed 20260909, 200 names x 1,200 days x 8 sectors, **7 s**. The panel's alphas have their skill
planted in *known* components of the return, so neutralization can be checked rather than
asserted.

## 1. Conventions this skill uses, so the numbers can be checked

| Quantity | Definition here |
|---|---|
| IC | cross-sectional correlation of the alpha known at `t` with the forward return from `t` |
| rank IC | the Spearman version — Pearson on within-row ranks |
| ICIR | `mean(IC)/std(IC)`, annualized by `sqrt(252)` |
| one-way turnover | `0.5 * Σ_i |w_t,i − w_{t−1,i}|` on a unit-gross book; `1.0` = the book flipped |
| net | gross minus `bps × one-way turnover`, the convention `../../../fin-core/skills/backtest-validation/scripts/cost_curve.py` consumes |

✅ Measured — the script's vectorized rank IC agrees with `scipy.stats.spearmanr` row by row to
**1.39e−17**. Do not take a hand-rolled IC on trust; ten rows against scipy costs nothing.

## 2. IC is not the number that decides anything

✅ Measured, five alphas, daily rebalance, 10 bps one-way:

| Alpha | rank IC | ICIR | turnover | gross SR | net SR |
|---|---|---|---|---|---|
| `idio_fast` | 0.0374 | 8.51 | **70.6%** | 8.48 | **1.77** |
| `idio_slow` | **0.0562** | 12.65 | 12.3% | 12.89 | **11.74** |
| `sector_bet` | 0.0224 | 5.03 | 23.0% | 5.52 | 3.29 |
| `low_beta` | 0.0187 | 2.10 | 8.3% | 2.04 | 1.67 |
| `pure_noise` | 0.0027 | 0.61 | 70.5% | 0.52 | **−6.27** |

🔑 **`idio_fast` and `sector_bet` are ranked opposite ways by IC and by net Sharpe.** IC is a
per-period correlation; it has no idea how often the book has to be rebuilt to harvest it. **An IC
table with no turnover column beside it cannot be acted on.**

⚠️ Note the control: `pure_noise` has an IC of 0.0027 (indistinguishable from zero) and a **net
Sharpe of −6.27**. Costs alone turn a zero-edge signal into a large negative. Always run a noise
alpha through the same pipeline — it calibrates what "no edge" looks like on your own plots.

## 3. 🚨 The IC t-stat, and why "overlapping returns" is the wrong diagnosis

The usual rule — *overlapping forward returns inflate the t-stat, so use Newey-West with `h−1`
lags* — is half right, and the wrong half is load-bearing.

✅ Measured, same panel, the same alpha at four horizons:

**`idio_slow` — a persistent alpha**

| h | mean IC | **AC(1) of IC** | t (overlapping) | t (every h-th day) | t (Newey-West, h−1) |
|---|---|---|---|---|---|
| 1 | 0.0562 | 0.41 | 27.6 | 27.6 | 27.6 |
| 5 | 0.0704 | **0.92** | **35.1** | 15.4 | 16.9 |
| 21 | 0.0616 | 0.96 | **30.4** | 5.9 | 7.9 |
| 63 | 0.0367 | 0.97 | **16.0** | 2.5 | 3.6 |

**`idio_fast` — a near-i.i.d. alpha**

| h | mean IC | **AC(1) of IC** | t (overlapping) | t (every h-th day) | t (Newey-West, h−1) |
|---|---|---|---|---|---|
| 1 | 0.0374 | −0.00 | 18.6 | 18.6 | 18.6 |
| 5 | 0.0107 | 0.04 | 5.3 | 1.5 | 5.0 |
| 21 | 0.0051 | 0.03 | 2.5 | 1.0 | 2.6 |
| 63 | 0.0057 | 0.02 | 2.7 | **0.4** | 2.4 |

🔑 **Read the AC(1) column first. Overlap by itself inflates nothing.** What breaks the i.i.d.
standard error is autocorrelation in the *IC series*, and that needs a persistent **alpha** as well
as an overlapping return. For the fast alpha, daily sampling against a 63-day return really does
carry ~63× the observations: its naive `t = 2.7` is roughly honest and **sampling every 63rd day
throws the evidence away** (`t = 0.4`). For the slow alpha at the same horizon, `16.0` should be
about `3.6`.

🚨 **And `lags = h − 1` is a rule about the overlap, not about the alpha.** At `h = 1` it
prescribes **zero** lags, so `idio_slow`'s daily IC keeps `t = 27.6` — while its IC autocorrelation
is **0.41**. ✅ Measured: with 21 lags the same series gives **`t = 10.8`**. **A daily,
non-overlapping IC can still need a HAC correction**, and the standard recipe will not give it one.

**What to do:** choose the lag from the *IC series' own* autocorrelation, not from the return
horizon. `max(h − 1, first lag where the IC autocorrelation is insignificant)` is a defensible
floor. Report `AC(1)` next to every IC t-stat so a reader can see which regime you are in.

⚠️ `../../../fin-core/skills/factor-and-timeseries-research/SKILL.md` §1.5 states the HAC rule for
Fama-MacBeth; this section is the measurement of when it bites and when it does not.

## 4. Neutralization: what it removes, and how to check it

`neutralize()` is one cross-sectional OLS per period — regress the alpha on sector dummies and/or
beta, keep the residual. The residual has exactly zero cross-sectional correlation with every
column of the design, by construction.

✅ Measured — rank IC, raw and residual:

| Alpha | raw | sector-neutral | beta-neutral | both |
|---|---|---|---|---|
| `idio_fast` | 0.0374 | 0.0376 | 0.0360 | 0.0364 |
| `idio_slow` | 0.0562 | 0.0567 | 0.0575 | 0.0574 |
| **`sector_bet`** | 0.0224 | **−0.0018** | 0.0205 | **−0.0036** |
| `low_beta` | 0.0187 | 0.0187 | 0.0146 | 0.0147 |

🔑 **This table is the acceptance test, and it is the one people skip.** Neutralization must remove
the exposure you named **and nothing else**. `sector_bet` goes to zero under eight dummies, exactly
as designed; the idiosyncratic alphas do not move. 🚨 **If your "idiosyncratic" alpha also dies,
it was never idiosyncratic** — you had a sector bet with a stock-level story attached.

⚠️ `low_beta` is the instructive row: it keeps most of its IC either way. **Neutralization is not
primarily an IC intervention** — which is §5.

## 5. 🚨 Neutralizing with a beta the market has already told you

Beta-neutral is a promise about the **book's realized exposure**. ✅ Measured on `low_beta`, a
persistent tilt against high-beta names, with betas that genuinely drift (AR(1), ~138-day
half-life):

| Beta used to neutralize | rank IC | **realized book beta** | gross SR | net SR |
|---|---|---|---|---|
| (none) | 0.0187 | −0.2870 | 2.04 | 1.67 |
| trailing 126d, shifted 1 day (causal) | 0.0146 | −0.1476 | 2.62 | 1.96 |
| ONE full-sample beta per name | 0.0177 | −0.2179 | 2.56 | 2.07 |
| 126d window ending **63 days ahead** | 0.0155 | **−0.0367** | 3.49 | 2.63 |

🔑 **The IC column barely moves and tells you nothing. The realized-beta column is the whole
story.** And because a single panel proves nothing, the script re-runs it on three:

✅ Measured — |realized book beta| across panel sizes:

| Panel | none | causal | full-sample | **63d ahead** |
|---|---|---|---|---|
| 60 x 500 | 0.2558 | 0.1546 | **0.1362** | **0.0415** |
| 80 x 700 | 0.3095 | **0.1498** | 0.1690 | **0.0412** |
| 200 x 1200 | 0.2870 | **0.1476** | 0.2179 | **0.0367** |

🚨 **Invariant — the leak.** A hedge fitted on a window reaching 63 days past `t` removes about
**85%** of the book's beta on every panel, where every causal hedge removes about **half**. That
gap is not a seed artefact, and it is bought entirely with a hedge ratio you could not have had.
Same shape as the volatility look-ahead in `../trend-following-models/SKILL.md` §3: **the leak is
in the hedge, not the signal, so every causality check aimed at the signal passes.**

⚠️ **Not invariant — and this is the useful part.** Whether ONE full-sample beta beats a noisy
trailing one **flips with the panel** (better at 60 names, worse at 80 and 200). A constant cannot
track drift; a short window is noisy; which loses is an empirical question about *your* data. 🔑
**So you cannot reason your way to the right estimator.** And the gross-Sharpe column is not
reliably signed either — on an 80x700 panel the peeking hedge *lowers* Sharpe — so **"the Sharpe
went up" is not how you detect this.**

**Check it directly:** regress the book's realized P&L on the factor and report the coefficient.
`realized_beta()` is four lines. **A neutralization you did not measure is a comment, not a
constraint.**

## 6. 🚨 Winsorizing after ranking clips exactly nothing

✅ Measured — `idio_fast` with one 60x outlier injected every 37 days (excess kurtosis **2824.5**):

| Pipeline | cells the clip moves | max \|w\| | Pearson IC | gross SR |
|---|---|---|---|---|
| z-score only, **no winsorize** | — | **0.4402** | 0.0374 | 7.44 |
| winsorize → z-score | 0.24% | 0.1849 | 0.0378 | 8.34 |
| z-score → winsorize | 0.24% | 0.1849 | 0.0378 | 8.34 |
| **rank → winsorize** | **0.00%** | 0.0100 | 0.0379 | 8.54 |

🚨 **0.00%.** A rank maps every row onto `[−0.5, 0.5]`, whose cross-sectional standard deviation is
`0.289`, so a 3-sd clip sits at `±0.87` — wider than the data can ever be. The line runs, raises
nothing, and a reviewer reads it as outlier protection.

⚠️ **Two things this does not say.**
1. **Ranking is fine** — it scores best in the table. The objection is to the dead `winsorize` line
   after it, not to the rank.
2. **`winsorize → z-score` and `z-score → winsorize` print identical rows** — and that is a fact
   about the *metrics*, not the frames. ✅ Measured: the two frames differ by up to **10.175**
   (z-scoring a clipped row rescales by the *clipped* sd), but each is an affine image of the
   other, so after `to_weights` they agree to **2.78e−17** and every scale-free statistic
   downstream is identical. 🔑 **Order starts to matter the moment a step is not affine — which is
   exactly what a rank is.** "Winsorize first" is folklore in the affine case and load-bearing in
   the rank case.

🔑 The number to be alarmed by is the first row: **one name takes 44% of a 200-name book** because
one input cell was 60× too large. That is what winsorization is for, and it is what a rank gives
you for free at the price of discarding magnitude.

## 7. Combining, and paying for it

✅ Measured — four alphas (`idio_fast`, `idio_slow`, `low_beta`, `pure_noise`), 10 bps:

| Combination | rank IC | turnover | gross SR | net SR |
|---|---|---|---|---|
| equal-weight z-score | 0.0569 | 50.6% | 9.24 | 5.86 |
| IC-weighted z-score | 0.0698 | 39.2% | 13.97 | 10.75 |
| **IC-weighted rank** | 0.0695 | 39.0% | **14.15** | 10.78 |
| **cost-aware z-score** | 0.0640 | **15.7%** | 12.63 | **11.36** |

🔑 **The best gross book is not the best net book.** IC-weighted rank wins gross; cost-aware wins
net, at 40% of the turnover. `cost_aware_weights()` scales each alpha's IC by its own
`net Sharpe / gross Sharpe` traded alone — it charges an alpha for the turnover it brings, and
zeroes one that cannot pay for itself.

⚠️ Equal weighting is worse than either here, but it is not always: IC weights are estimated on the
same data they are scored on, and that is a fitted parameter like any other. Log it in
`../../../fin-core/skills/backtest-validation/scripts/trial_ledger.py`.

**Smoothing the combined signal** (EWMA halflife, days) is the cheapest turnover control there is:

| halflife | rank IC | turnover | gross SR | net SR |
|---|---|---|---|---|
| none | 0.0698 | 39.2% | 13.97 | 10.75 |
| **0.5** | 0.0681 | 28.4% | 13.34 | **11.04** |
| 1 | 0.0642 | 18.7% | 12.43 | 10.93 |
| 2 | 0.0595 | 11.2% | 11.38 | 10.49 |
| 5 | 0.0522 | 5.6% | 9.86 | 9.42 |
| 60 | 0.0253 | 1.2% | 4.59 | 4.49 |

⚠️ On this panel at 10 bps the net optimum sits at the low end of the grid, not in the middle —
the IC-weighted combination is already fairly slow. **Where the optimum sits is a function of the
cost, not a property of the strategy**, so it moves when the cost assumption moves. Sweep it as a
curve: `../../../fin-core/skills/backtest-validation/scripts/cost_curve.py`.

## 8. Use alphalens for the tear sheet, not for the lag

Do not re-implement quantile tear sheets, IC decay plots or turnover-by-quantile — `alphalens-
reloaded` has them. 🚨 But its forward-return convention is a trap this library has already
verified in source: see `../../../fin-libraries/skills/lib-alphalens/SKILL.md` and
`../../../fin-core/skills/factor-and-timeseries-research/SKILL.md` §1.1. **Lag your factor before
you pass it in.** The functions here are for the parts alphalens does not do — neutralization,
combination weights, and the HAC-corrected t-stat.

## 9. Scripts and where this sits

`scripts/alpha_combine.py` — the panel, IC/ICIR/HAC t-stats, `neutralize`, `realized_beta`,
`cs_zscore`/`cs_rank`/`winsorize`, `to_weights`/`turnover`/`net_of_cost`, the combination
weightings, and every table above. numpy + pandas + scipy, seed 20260909, 7 s.

- Computing the raw signal causally in the first place —
  `../../../fin-core/skills/signal-construction/SKILL.md` and its
  `../../../fin-core/skills/signal-construction/scripts/assert_causal.py`.
- One time-series signal per instrument, rather than a cross-section —
  `../trend-following-models/SKILL.md`.
- Factor libraries, Fama-MacBeth, alpha mining and the trial count that implies —
  `../../../fin-core/skills/factor-and-timeseries-research/SKILL.md`.
- Turning the combined signal into weights under real constraints, and risk decomposition —
  `../../../fin-core/skills/portfolio-and-risk/SKILL.md`.
- How much of the book to hold — `../position-sizing-kelly/SKILL.md`.
- Getting the turnover done at the assumed cost — `../execution-algorithms/SKILL.md` and
  `../../../fin-core/skills/execution-cost-analysis/SKILL.md`.
- Whether the surviving combination is a discovery: trials, SPA and the cost curve —
  `../../../fin-core/skills/backtest-validation/SKILL.md` and
  `../../../fin-core/skills/research-integrity-guards/SKILL.md`.
- The tear sheet — `../../../fin-libraries/skills/lib-alphalens/SKILL.md`.
