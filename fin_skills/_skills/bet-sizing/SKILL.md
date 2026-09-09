---
name: bet-sizing
description: >-
  Turn a predicted probability into a position - the 2*Phi(z)-1 size curve, averaging concurrent
  bets instead of adding them, discretising to buy turnover, and the concurrency budget whose
  divisor is usually a look-ahead. TRIGGER - bet sizing, getSignal, get_signal, getBetSize,
  bet size from probability, avgActiveSignals, average active bets, discreteSignal, discrete
  signal, step size, position from predict_proba, "how big should this trade be", "my positions
  flip every bar", turnover from a probability, concurrent bets leverage, budgeting bets,
  Lopez de Prado chapter 10, AFML bet sizing. SKIP for the Kelly fraction and how large the book
  should be overall (position-sizing-kelly - it owns Kelly, do not repeat it), for the secondary
  model that produces the probability (meta-labeling), for volatility targeting and risk budgets
  (portfolio-and-risk), and for the execution schedule once the size is chosen
  (execution-algorithms).
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-09"
---

# Bet sizing

**A probability is not a position, and the four steps between them each cost something
measurable.** Advances in Financial Machine Learning (Lopez de Prado 2018), chapter 10. The size
curve, the averaging of concurrent bets, the discretisation, and the leverage budget — all four
are usually written in one line each, and three of them are usually wrong.

Every number below is printed by `scripts/bet_sizing.py` (numpy + scipy, seed 0, **2.8 s**).

⚠️ **This chapter is the one place in this plugin with no runnable reference implementation.**
mlfinpy 0.1.2 ships no bet-sizing module (its packages are `labeling`, `sample_weights`,
`sampling`, `filters`, `structural_breaks`, `cross_validation`, `ensemble`, `data_structure`,
`util`, `dataset`), and 🔴 `mlfinlab` is not installable — ✅ checked 2026-09-09: the PyPI JSON API
returns 404 and the simple index lists zero distribution files. The formulas below are transcribed
from the published snippets; what is **measured** is every property they should have.

## 1. The size curve is not `2p - 1`

```
z = (p - 1/n_classes) / sqrt(p (1 - p))        m = 2 * Phi(z) - 1        signal = side * m
```

⚠️ AFML Snippet 10.1 (page 142), `getSignal` / `getBetSize`. ✅ Measured against the obvious
alternative:

| p | z | **m = 2Φ(z)-1** | 2p - 1 | m / (2p-1) |
|---|---|---|---|---|
| 0.50 | 0.000 | **0.0000** | 0.00 | — |
| 0.55 | 0.101 | 0.0801 | 0.10 | **0.801** |
| 0.60 | 0.204 | 0.1617 | 0.20 | 0.809 |
| 0.70 | 0.436 | 0.3375 | 0.40 | 0.844 |
| 0.80 | 0.750 | 0.5467 | 0.60 | 0.911 |
| 0.90 | 1.333 | 0.8176 | 0.80 | **1.022** |
| 0.95 | 2.065 | 0.9611 | 0.90 | 1.068 |
| 0.99 | 4.925 | **1.0000** | 0.98 | 1.020 |

✅ `m(0.5) = 0.0` exactly, `m(1) = 1.0000`, `m(0) = -1.0000`. The curve sits **below** the linear
size up to about p = 0.90 and above it after: it is **sceptical near the coin flip and saturating
at the extremes**, which is the opposite shape to what "just use 2p-1" gives you. At p = 0.55 it
asks for 80 % of the linear size; at p = 0.95, 107 %.

🚨 **`n_classes` is not decoration.** The pivot is `1/n_classes`, not 0.5. ✅ Measured:
`m(0.40, n=3) = +0.1082` — a *long* — while `m(0.40, n=2) = -0.1617`, a short. Pass a
three-class model's probability through the binary formula and every marginal call comes out with
the wrong sign.

⚠️ The z-to-size map assumes `z` is approximately standard normal, which is a modelling
convention rather than a derived fact. What it guarantees is monotonicity, the right sign, and
saturation in [-1, 1]; it does not make `m` an optimal fraction of capital. That question is
Kelly's, and `../../../fin-strategies/skills/position-sizing-kelly/SKILL.md` owns it.

## 2. Averaging concurrent bets, not adding them

⚠️ AFML Snippet 10.2 (page 144), `avgActiveSignals`: at each bar, average the signals of the bets
*active* at that bar. ✅ Measured on 297 overlapping bets over 5,000 bars (holding 5-60 bars,
mean concurrency 1.95, max 8, 14.4 % of bars flat; calibration check: mean true `q` 0.662 against
a realised hit rate of 0.653):

| | mean \|position\| | max \|position\| | \|pos\| > 1 | turnover | Sharpe after 5 bp |
|---|---|---|---|---|---|
| **summed** | 0.448 | **3.196** | **11.0 %** of bars | 2,140.3 | 0.21 |
| **averaged** | 0.225 | **0.998** | **0.0 %** | 1,103.3 | **0.32** |

- 🚨 **Adding the sizes gives you 3.2x leverage at the peak** and puts the book above 1x on 11 %
  of bars — a leverage decision made by the arrival rate of signals, not by anyone.
- ✅ Averaging divides the peak by **3.2x** and the mean by **2.0x**, and halves the turnover.
- 🚨 **It is not a rescaling.** The two position series correlate **0.8662**, because the divisor
  is the concurrency, which moves with time. A bet's contribution to the book shrinks when other
  bets arrive — that is a real, and deliberate, change to the return series, not a change of
  units.

## 3. 🚨 Discretisation: rounding alone barely reduces turnover

⚠️ AFML Snippet 10.3 (page 145): `signal1 = (signal0 / stepSize).round() * stepSize`, clipped to
[-1, 1]. The stated purpose is to stop trading on trivial changes in the probability. ✅ Measured
on the averaged signal, against a variant that only moves when the raw signal has drifted a **full
step** from the level currently held:

| step | grid sizes | round: turnover | vs continuous | corr | round: Sharpe | full-step-move: turnover | vs cont | corr | Sharpe |
|---|---|---|---|---|---|---|---|---|---|
| 1.00 | 3 | 854.0 | 77 % | 0.73 | 0.26 | **0.0** | 0 % | — | — |
| **0.50** | 5 | 1,093.0 | **99 %** | 0.92 | 0.32 | **351.5** | **32 %** | 0.69 | **0.52** |
| 0.25 | 9 | 1,099.0 | 100 % | 0.98 | 0.31 | 775.5 | 70 % | 0.93 | 0.33 |
| 0.10 | 21 | 1,102.2 | 100 % | 1.00 | 0.31 | 1,041.0 | 94 % | 0.99 | 0.36 |
| **0.05** | 41 | **1,106.8** | **100.3 %** | 1.00 | 0.33 | 1,091.2 | 99 % | 1.00 | 0.35 |
| none | 4,283 | 1,103.3 | 100 % | 1.00 | 0.32 | 1,103.3 | 100 % | 1.00 | 0.32 |

🚨 **At step 0.05 the rounded position trades MORE than the continuous one (100.3 %).** A signal
hovering at a grid boundary flips across it every bar, and each flip is a full step of trading.
Rounding changes the *set of values* the position takes; it does not by itself change how often
the position moves. Only the coarsest grid (step 1.00, i.e. {-1, 0, 1}) cuts turnover, and it
costs a correlation of 0.73 with the continuous signal.

✅ Requiring a full step of drift before moving cuts turnover to **32 %** at step 0.50 and costs
tracking (correlation 0.69) rather than resolution. (At step 1.00 that rule never trades at all,
because the averaged signal never reaches 1 — hence the blank row.)

**And whether any of it is worth doing is a cost question, not an accuracy question.** ✅ The same
three position series scored at four cost levels:

| cost per unit of turnover | continuous | round 0.50 | full-step-move 0.50 |
|---|---|---|---|
| 0 bp | **0.90** | 0.84 | 0.68 |
| 5 bp | 0.32 | 0.32 | **0.52** |
| 20 bp | -1.40 | -1.22 | **0.02** |
| 50 bp | -4.62 | -4.01 | **-0.94** |

🚨 **At zero cost discretisation strictly loses** (0.90 → 0.84 → 0.68); at 20 bp it is the only
version still above water. Choose the step from your cost curve
(`../../../fin-core/skills/execution-cost-analysis/SKILL.md` owns that), not from a default, and
count the choice as a trial.

## 4. 🚨 The concurrency budget divides by a number you have not seen yet

The natural way to bound leverage is to divide the summed signal by the maximum number of
simultaneous bets. ✅ Measured:

| divisor | max \|position\| | mean \|position\| | Sharpe after 5 bp |
|---|---|---|---|
| **full-sample max concurrency (8)** | 0.400 | 0.056 | 0.21 |
| **expanding (causal) max** | **0.849** | 0.065 | 0.19 |

- 🚨 The full-sample maximum is not known until bar **1,236 of 5,000** — a quarter of the way
  through. Everything before that is sized with information from the future.
- 🚨 The causal version exceeds the full-sample version's own peak on **0.7 % of bars**: that is
  leverage the look-ahead hid, and it is exactly the tail where it matters.
- 🚨 Over the first quarter of the sample the causal budget runs at **1.58x** the full-sample one,
  because its divisor has not yet met the crowded stretch.

A backtest that divides by a full-sample maximum reports a smoother book than you could have run,
and it will not warn you. Use an expanding maximum, a fixed budget declared in advance, or a
quantile of trailing concurrency — and check the whole position series with
`../../../fin-core/skills/signal-construction/scripts/assert_causal.py`.

## 5. Traps

- 🚨 **`2p - 1`.** Section 1. Wrong shape at both ends, and it ignores `n_classes`.
- 🚨 **A binary size formula on a three-class model.** Section 1: `m(0.40, n=3)` and
  `m(0.40, n=2)` have opposite signs.
- 🚨 **Adding concurrent positions.** Section 2: 3.2x leverage nobody chose.
- 🚨 **Assuming discretisation cuts turnover.** Section 3: at step 0.05 it increases it.
- 🚨 **A full-sample concurrency divisor.** Section 4.
- 🚨 **An uncalibrated probability.** All of section 1 assumes `p` means what it says. A model
  whose reported 0.8 is right 60 % of the time will be sized as if it were right 80 % of the time.
  Check calibration (predicted probability against realised frequency, by bucket) before sizing,
  and note that a probability from an unpurged fold is not calibrated for the future
  (`../../../fin-libraries/skills/lib-purgedcv/SKILL.md`).
- 🚨 **The size at `t` must use `p` known at `t`.** A probability re-reported mid-bet is fine; a
  probability computed with the bet's own outcome is not.
- ⚠️ **`m` is a relative size, not a capital fraction.** Multiplying it by the full account is a
  Kelly-scale decision — `../../../fin-strategies/skills/position-sizing-kelly/SKILL.md` — and
  volatility targeting sits on top of that
  (`../../../fin-core/skills/portfolio-and-risk/SKILL.md`).
- ⚠️ **Both `discretise` variants use half-to-even rounding** (`np.round`, like pandas'
  `.round()`): a size of exactly 0.25 at step 0.5 becomes 0.0, not 0.5. It only matters on exact
  ties, and exact ties happen more often than you expect once a signal is itself discretised.
- 🚨 **The step size is a trial.** Six steps times two rules times four cost levels is 48 numbers,
  and the best of them is not an estimate of anything.
  `../../../fin-core/skills/backtest-validation/scripts/trial_ledger.py`.

## 6. Scripts

- `scripts/bet_sizing.py` — the size curve and its linear alternative, a calibrated overlapping-bet
  simulator whose probability is re-reported every bar, averaging versus summing active signals,
  both discretisation rules, turnover and Sharpe at four cost levels, and the full-sample versus
  expanding concurrency budget. numpy + scipy only. Seed 0, **2.8 s**.

## 7. Where this sits

`../meta-labeling/SKILL.md` produces the probability this skill consumes, and stops at the
threshold. `../../../fin-strategies/skills/position-sizing-kelly/SKILL.md` owns the Kelly
fraction and how much capital the whole book gets — this skill deliberately does not repeat it.
`../../../fin-core/skills/portfolio-and-risk/SKILL.md` owns volatility targeting and the risk
budget. `../../../fin-core/skills/execution-cost-analysis/SKILL.md` owns the cost curve that
decides the step size in section 3, and
`../../../fin-strategies/skills/execution-algorithms/SKILL.md` owns getting the resulting size
into the market. `../sample-weights-and-uniqueness/SKILL.md` owns the concurrency series that
section 4 divides by. `../../../fin-core/skills/signal-construction/SKILL.md` owns the causality
check on the position series.
