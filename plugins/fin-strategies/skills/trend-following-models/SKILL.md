---
name: trend-following-models
description: >-
  Build a trend-following or time-series-momentum strategy the way the paper defines it, and
  measure the two look-aheads that flatter its backtest. TRIGGER - time series momentum, TSMOM,
  Moskowitz Ooi Pedersen, 12-month momentum, trend following, managed futures, CTA replication;
  Donchian channel, turtle rules, breakout system, 20-day high, moving average crossover, golden
  cross, 50/200 MA; volatility targeting, vol scaling, ex-ante volatility, 40% vol target, risk
  parity across futures, inverse-vol sizing, ATR sizing; "my trend backtest has a Sharpe of 3",
  "should I skip the most recent month", "do I trade the close or the next open". SKIP for
  computing the indicator itself and whether it repaints (signal-construction), for
  cross-sectional ranking of many names (factor-and-timeseries-research), for combining several
  alphas into one (alpha-combination-and-neutralization), for how much to bet given an edge
  (position-sizing-kelly), and for the engine that runs the loop (backtesting-engines).
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-09"
---

# Trend-following models

**A trend backtest fails in two places, and both produce a smooth equity curve and no error.**
The signal is easy; the alignment is not. This skill is the paper's definitions checked against
the paper, plus the size of each misalignment measured on one seeded panel.

Everything marked ✅ Measured comes from `scripts/trend_models.py` (numpy + pandas, seed
20260908, 10 synthetic futures x 5,220 daily bars, runs in **under 2 s**). Everything marked
✅ source-verified is quoted in `references/mop-2012.md` from the paper itself.

## 1. What Moskowitz, Ooi & Pedersen (2012) actually specify

✅ source-verified — JFE 104 (2012) 228–250, full quotations in `references/mop-2012.md`.

| Question | The paper's answer | Where |
|---|---|---|
| Lookback | **12 months**, `k=12`, holding period `h=1` month | §4.1, eq. (5) |
| **Skip the most recent month?** | **No.** `sign(r_{t−12,t})` runs up to the rebalance date | eq. (5) |
| Volatility estimator | EWMA of squared daily returns, **centre of mass 60 days** | eq. (1) |
| Annualization | **261**, not 252 | eq. (1) |
| Which sigma sizes which return | **`σ_{t−1}` applied to time-`t` returns** | §2.4 |
| Vol target | **40% annualized, per position** | §4.1 |

```
sigma_t^2 = 261 * sum_{i>=0} (1 - d) d^i (r_{t-1-i} - rbar_t)^2 ,   d/(1-d) = 60 days   (1)

r^TSMOM_{t,t+1} = sign(r_{t-12,t}) * (40% / sigma_t) * r_{t,t+1}                        (5)
```

🚨 **The skipped month is the single most common mis-implementation, and it is imported from
the wrong paper.** The only place MOP mention skipping is footnote 10, p. 240, and it is about
the **cross-sectional** benchmark they compare against: *"Asness, Moskowitz, and Pedersen (2010)
exclude the most recent month when computing 12-month cross-sectional momentum. For consistency,
we follow that convention here."* Cross-sectional momentum skips a month to dodge short-term
reversal in individual stocks. **Time-series momentum on futures does not.**

✅ Measured — the same panel, monthly rebalance, 228 months:

| Variant | Ann. return | Ann. vol | Sharpe | Max DD |
|---|---|---|---|---|
| **12 months, nothing skipped (eq. 5)** | **7.8%** | 14.1% | **0.56** | −24.1% |
| 12−1, most recent month skipped (fn. 10) | 4.3% | 15.2% | 0.29 | **−46.5%** |

⚠️ On this seeded panel the skip roughly halves the Sharpe and doubles the drawdown. MOP say
*"our results do not depend on whether the most recent month is excluded or not"* on **their**
data — this panel is synthetic and trends persistently by construction, so treat the direction,
not the magnitude, as the finding: **the skip is a choice, it costs something, and it is not in
eq. (5).** If you skip, say you skipped.

### 1b. Two readings of eq. (5) that disagree

⚠️ The prose one paragraph above eq. (5) writes the position size as `40%/σ_{t−1}`; eq. (5)
itself prints `40%/σ_t`. **They agree** — because eq. (1) already sums over `r_{t−1−i}`, so
`σ_t` excludes `r_t`. 🚨 Re-implement from eq. (5) alone with a sigma that includes the
contemporaneous return and you have built §3's look-ahead while believing you followed the
paper.

## 2. The estimator is one line of pandas, and it is exactly eq. (1)

```python
# d = com/(1+com) = 60/61; pandas' com IS the paper's centre of mass d/(1-d)
sigma = np.sqrt(returns.ewm(com=60, adjust=True).var(bias=True) * 261).shift(1)
```

✅ Measured — `d = 0.983607`, `Σ_i (1−d)d^i·i = 60.0000` days, weights sum to `1.000000`.
Against eq. (1) written out longhand with explicit weights and an explicitly weighted mean, the
pandas value differs by a **relative 3.188e−16** — machine precision. `eq1_variance()` in the
script is that longhand reference.

🔑 `bias=True` is required. The default `var()` is the unbiased (reliability-weighted) form and
is **not** eq. (1). `adjust=True` (the default) is what renormalises the truncated tail.

## 3. 🚨 Trap 1 — sizing `r_t` with a sigma that contains `r_t`

The whole bug is a missing `.shift(1)`. Nothing raises, the position series looks sane, and the
equity curve is smoother than the honest one because every large return has been divided by a
denominator that already knows about it.

✅ Measured — daily-rebalanced TSMOM, identical signal, only the volatility lag changes:

| Volatility estimator | Sharpe with `σ_{t−1}` | Sharpe with `σ_t` | Gain from the leak |
|---|---|---|---|
| **EWMA com=60 (MOP eq. 1)** | **0.64** | 0.68 | **+0.05** |
| EWMA com=20 | 0.60 | 0.73 | +0.13 |
| EWMA com=5 | 0.60 | 1.01 | **+0.41** |
| rolling 60-day | 0.60 | 0.64 | +0.05 |
| rolling 20-day | 0.60 | 0.74 | +0.15 |
| **rolling 5-day** | 0.60 | **1.28** | **+0.68** |

🔑 **The size of the leak is set by the estimator's window, not by the strategy.** At MOP's own
60-day centre of mass one day is ~1.6% of the weight and the theft is worth +0.05 Sharpe — small
enough to survive review. At a 5-day window one day is ~20% of the denominator and the same
one-character bug doubles the Sharpe. **A short vol window is not a modelling choice here; it is
an amplifier on a bug you cannot see.**

⚠️ Note the honest column barely moves (0.60–0.64). Volatility targeting is not what makes this
strategy work — §5 shows what it actually buys.

## 4. 🚨 Trap 2 — which return the position is actually paid

Three alignments, one signal. Only the third is tradeable.

| | What it assumes |
|---|---|
| **no shift** | the position computed from `close_t` earns `close_{t−1} → close_t` |
| **same close** | signal from `close_{t−1}`, filled **at** `close_{t−1}` |
| **next open** | signal from `close_{t−1}`, filled at `open_t`, earns `open_t → open_{t+1}` |

✅ Measured, Sharpe:

| Model | no shift | same close | next open |
|---|---|---|---|
| TSMOM 12m | **2.51** | 0.64 | 0.60 |
| MA 50/200 | 0.74 | 0.65 | 0.63 |
| **Donchian 20/10** | **7.04** | 0.61 | 0.68 |

🚨 **Donchian goes from 0.61 to 7.04 on a one-bar misalignment.** Breakout rules are the most
sensitive family there is, because the rule fires *precisely on the bar that moved* — crediting
that bar's own return to the position it triggered is close to reading the answer.

⚠️ **same close → next open is the small one here** (0.64 → 0.60 for TSMOM, 0.61 → 0.68 for
Donchian; it can go either way on a given path). It is still not free: a close-to-close backtest
assumes a fill at a price nobody could transact at, and the gap it hides is exactly the overnight
move that trend signals are most exposed to. Report both, and cost the difference with
`../../../fin-core/skills/backtest-validation/scripts/cost_curve.py`.

## 5. 🚨 Trap 3 — the look-ahead test that does not fire

`../../../fin-core/skills/signal-construction/scripts/assert_causal.py` is the right tool and
you should use it. **Its default perturbation cannot see trap 2.**

✅ Measured, perturbing rows `≥ k` and checking that positions at rows `≤ k` did not move:

| Detector | Finds trap 1 (sizing) | Finds trap 2 (no shift) |
|---|---|---|
| `assert_causal`'s `×2.0` scaling | **yes**, all 6 estimators | 🚨 **no**, on all 3 models |
| overwrite the tail: −90% **and** +900% days | yes | **yes**, all 3 models |

🔑 **`np.sign()` is locally flat.** Doubling one daily return moves a 12-month log-sum by a
fraction of a percent, the sign does not flip, the position does not change, and the test reports
clean. Trap 1 is caught because `40%/σ` is *continuous* in the perturbed data; trap 2 is not,
because the signal is a step function of it.

**So: perturb hard enough to cross the threshold.** `position_is_causal(fn, frame, k,
shock=SHOCKS)` overwrites the tail with each extreme return in turn. The correctly shifted
position stays causal under both perturbations — the script asserts that, so the detector is
shown to *distinguish* rather than to always fire.

⚠️ **Two-sided, and worth being precise about.** A single −90% day only flips a signal that was
*positive*; an instrument already trending down is unmoved, so a one-sided shock silently passes
on exactly the names it should catch. Hence `SHOCKS = (-0.90, 9.0)`. And the `×2` row is
data-dependent, not a law: on a smaller 4-asset panel the same `×2` probe *does* catch the
Donchian leak, by luck, because a price-level breakout threshold happened to sit within 1% of
`close_k`. **It never catches the two sign-based models on either panel.** The honest statement is
not "×2 always misses" — it is **"×2 is not a reliable probe for a thresholded signal, and you
cannot tell from a green result which case you are in."**

🔑 And the second silent failure in the same file: **the function under test must rebuild
everything from the frame.** A closure over a signal computed outside makes the check pass for
any input. This skill's own first draft did that, and printed "causal: True" for three
demonstrably leaking models.

## 6. What volatility targeting actually buys

✅ Measured — daily rebalance, same signal, sizing on and off:

| | Ann. return | Ann. vol | Sharpe | Max DD |
|---|---|---|---|---|
| 40% vol target per position | 8.1% | 12.7% | 0.64 | −20.1% |
| sign only, no target | 5.7% | 8.7% | **0.66** | −12.6% |

**Sharpe is unchanged.** What moves is the *concentration*: the variance share of the highest-vol
asset (40% annualized) falls from **22% to 10%** — and `1/10 = 10%` is an even split across ten
instruments. Per-asset Sharpes on this panel run **−0.20 to 0.58**.

🔑 **Vol targeting is a risk-allocation decision, not an alpha.** It equalises how much each
instrument contributes, which is why it matters on a 58-contract futures panel and does almost
nothing on one instrument. ✅ source-verified — MOP agree: *"The choice of 40% is
inconsequential, but it makes it easier to intuitively compare our portfolios to others."*

⚠️ The vol target also **is** a trade. Rebalancing daily to `40%/σ_t` turns a monthly strategy
into a daily one; at MOP's monthly rebalance most of that turnover disappears. Cost it before
you claim it.

## 7. The other two families, and what is not verified

**Donchian / turtle breakout** — long on a close above the prior `entry`-day channel high, flat
on a close below the prior `exit`-day channel low. 🔑 The channel must be `.shift(1)`ed: the bar
that breaks out must not be inside its own channel. The classic 20/10 parameters are folklore
here, not source-verified — this library has not checked the original Turtle rules against a
primary document, so treat `entry=20, exit=10` as a default, not a finding.

**Moving-average crossover** — `sign(SMA_fast − SMA_slow)`. 50/200 is convention, not a result.
⚠️ It is the *least* sensitive of the three to trap 2 (0.74 vs 0.65) precisely because a slow
crossover rarely flips on the bar you are misaligning; that makes it the worst family to use as
a canary for alignment bugs.

🚨 **Nothing here is a claim about real markets.** The panel is synthetic, with AR(1) drift
(`phi=0.997`) and GARCH(1,1) volatility, built so trends exist. It exists to size *errors*
against a fixed strategy, not to establish that trend following works. Any Sharpe above is a
property of the generator.

## 8. Before you believe a trend backtest

1. `assert_causal` on the **position**, not the signal — with a shock, not a scaling (§5).
2. Confirm the sigma that sizes `r_t` is `.shift(1)`ed (§3).
3. Confirm the fill is the next open, and report close-to-close alongside it (§4).
4. Count the parameters you tried. Lookback, vol window, vol target, entry/exit, fast/slow —
   that is a six-dimensional grid, and every cell is a trial. Log them in
   `../../../fin-core/skills/backtest-validation/scripts/trial_ledger.py` and test the survivor
   with `../../../fin-core/skills/backtest-validation/scripts/spa_test.py`.
5. Turn the result into a cost curve, not a number —
   `../../../fin-core/skills/backtest-validation/scripts/cost_curve.py`. A daily-rebalanced vol
   target has real turnover.
6. Measure the fills you actually get:
   `../../../fin-core/skills/execution-cost-analysis/SKILL.md`.

## 9. Scripts and where this sits

`scripts/trend_models.py` — the panel, MOP eq. (1) and eq. (5), Donchian, MA crossover, vol
targeting, and every table above. numpy + pandas, seed 20260908, under 2 s.

`references/mop-2012.md` — the paper's own words for every ✅ source-verified claim above.

- Computing the indicator, warm-up and repainting —
  `../../../fin-core/skills/signal-construction/SKILL.md`, and its
  `../../../fin-core/skills/signal-construction/scripts/assert_causal.py` and
  `../../../fin-core/skills/signal-construction/scripts/warmup_probe.py`, which §5 extends
  rather than repeats.
- Cross-sectional ranking, IC, and factor tests —
  `../../../fin-core/skills/factor-and-timeseries-research/SKILL.md`.
- Combining this signal with others, and neutralizing it —
  `../alpha-combination-and-neutralization/SKILL.md`.
- How large the position should be given the edge — `../position-sizing-kelly/SKILL.md`.
- Getting the order done — `../execution-algorithms/SKILL.md` and
  `../../../fin-core/skills/execution-cost-analysis/SKILL.md`.
- Portfolio-level vol targeting and risk decomposition —
  `../../../fin-core/skills/portfolio-and-risk/SKILL.md`.
- Trials, deflated Sharpe, and whether the survivor survives —
  `../../../fin-core/skills/backtest-validation/SKILL.md` and
  `../../../fin-core/skills/research-integrity-guards/SKILL.md`.
- Running this at scale in a vectorized engine —
  `../../../fin-libraries/skills/lib-vectorbt/SKILL.md`.
- Continuous futures series, roll gaps and back-adjustment (a trend signal on a badly rolled
  series measures the roll) —
  `../../../fin-futures-fx/skills/futures-continuous-contracts/SKILL.md`.
