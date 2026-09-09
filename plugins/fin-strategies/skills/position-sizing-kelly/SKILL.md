---
name: position-sizing-kelly
description: >-
  Decide how much to bet given an edge - Kelly, fractional Kelly, and volatility targeting -
  and the drawdown each implies. TRIGGER - Kelly criterion, Kelly fraction, f star, optimal bet
  size, "how much should I bet", "how much capital per trade", position sizing, bet sizing,
  fractional Kelly, half Kelly, quarter Kelly, over-betting, geometric growth rate, expected
  log wealth, log utility, growth-optimal portfolio; volatility targeting as a sizing rule,
  size to 10% vol, inverse-vol sizing, leverage from a Sharpe ratio, risk of ruin, probability
  of a 50% drawdown, drawdown under leverage, "how much leverage can I take". SKIP for the
  signal that produces the edge (trend-following-models, alpha-combination-and-neutralization),
  for optimizers and constrained portfolio weights (portfolio-and-risk), for whether the edge
  is real at all (backtest-validation), and for working the resulting order
  (execution-algorithms).
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-09"
---

# Position sizing and Kelly

**Kelly answers "how much, given the edge". Almost every misuse of it comes from forgetting
that the edge in the formula is the TRUE one, and the one you have is an estimate.** The second
most common is dividing by `σ` once instead of twice.

Everything marked ✅ Measured comes from `scripts/position_sizing.py` — numpy + scipy, seed
20260909, **21 s**. Everything marked ✅ source-verified was read by this library in the primary
text; equation numbers are Thorp (2006)'s and the full transcription is in the script's
docstring.

🚨 **Nothing here is advice about your money.** These are the properties of a formula on a
stated model. Sizing a real book involves constraints, taxes, borrowing limits and a
liabilities side that none of this contains.

## 1. What is actually Kelly's, and what is not

| Result | Whose |
|---|---|
| `G = p·ln(1+f) + q·ln(1−f)` for an even-money bet | ✅ **Kelly (1956)**, p. 919, **unnumbered** |
| `G_max = 1 + p·log₂p + q·log₂q` — in **bits**, equal to Shannon's rate `R` | ✅ Kelly (1956), p. 920, unnumbered |
| `f* = p − q` for even money | ✅ Kelly, but **implicit** — he writes `(1+l) = 2q` and never displays `l = q − p` |
| **`f* = (b·p − q)/b`** for b-to-1 odds | 🚨 **not Kelly.** ✅ source-verified — Thorp (2006) p. 391 attributes it to his own **Thorp (1984)** |
| `f* = (m − r)/s²` | ✅ **Thorp (2006) eq. (7.3)**; ✅ **Merton (1969)** eq. (25)/(29) with log utility (his `δ = 1`) |

⚠️ **Kelly's `p` is the probability of ERROR and his `q` of correct transmission** — the opposite
of every modern restatement. Reading his paper with the modern convention inverts his results.
His own general-odds result (pp. 924–925) is `a(s) = p(s) − b/α_s`, where **his `b` is the
fraction NOT bet** and `α_s` is the **gross** return per dollar (`B+1` for modern B-to-1 odds).

✅ Measured — `kelly_binary` against a numerical `argmax E[ln(1+fX)]`, agreeing to **≤1.6e−6**
across `(p,b)` = (0.60,1), (0.55,1), (0.52,1), (0.40,2), (0.25,5). 🔑 **Use the numeric version
for anything that is not a two-outcome bet** — a three-outcome payoff has no closed form. A bet
with a non-positive mean returns exactly `0.0`.

## 2. The growth curve, and why half Kelly

At `μ = 8%`, `σ = 20%` (excess, `r = 0`): `f* = 2.00`, `SR = 0.40`, `g(f*) = μ²/(2σ²) = 0.0800
= SR²/2`. ✅ Measured:

| c | f = c·f* | g(c·f*) | g ratio | **c(2−c)** | Sdev(G) | Sdev ratio | **P(ever ≤ ½)** | P(2× before ½) |
|---|---|---|---|---|---|---|---|---|
| 0.25 | 0.50 | 0.0350 | 0.4375 | 0.4375 | 0.10 | 0.25 | 0.0078 | 0.9922 |
| **0.50** | 1.00 | 0.0600 | **0.7500** | 0.7500 | 0.20 | 0.50 | **0.1250** | **0.8889** |
| 1.00 | 2.00 | **0.0800** | 1.0000 | 1.0000 | 0.40 | 1.00 | **0.5000** | **0.6667** |
| 1.50 | 3.00 | 0.0600 | 0.7500 | 0.7500 | 0.60 | 1.50 | 0.7937 | 0.5575 |
| 1.99 | 3.98 | 0.0016 | 0.0199 | 0.0199 | 0.80 | 1.99 | 0.9965 | 0.5009 |

✅ source-verified — Thorp (2006) p. 409: `g(c f*)/g(f*) = c(2−c)` and `Sdev ratio = c`. Both
reproduce exactly. `g(2f*) = 0` to floating point, **for every `μ` and `σ`**.

🔑 **Growth is concave in `c`; dispersion is linear.** That is the entire half-Kelly argument in
one line, and it is why fractional Kelly is nearly free at the top. ✅ source-verified, Thorp
p. 415 verbatim: *"'half Kelly' has 3/4 the growth rate but much less chance of a big loss. The
chance of ever losing half the starting capital is 1/2 for `f = f*` but only 1/8 for
`f = f*/2`."* The `P(2× before ½)` column reproduces his **2/3** and **8/9** exactly.

## 3. The drawdown formula, checked — and the two ways it is misquoted

✅ source-verified — Thorp (2006) **eq. (7.13)**, p. 415:

```
Prob( V(t, c f*) / V0 <= x  for some t )  =  x ^ (2/c - 1)      r = 0,  0 < c < 2
```

🚨 **Use the corrected text.** ✅ source-verified — §7.4 opens: *"In earlier versions of this
chapter the exponent in Equations (3.2), (7.8) and (7.9) were off by a factor of 2, which I had
inadvertently dropped."* The title page says *"Corrections added April 20, 2005."* A pre-2005
printing gives you the wrong exponent.

✅ Measured — 200 years, monthly steps, 20,000 paths, with a Brownian-bridge correction:

| c | x | **formula** | simulated | naive (no bridge) |
|---|---|---|---|---|
| 1.00 | 0.50 | 0.5000 | **0.5002** | 0.4681 |
| 0.50 | 0.50 | 0.1250 | **0.1271** | 0.1147 |
| 0.25 | 0.50 | 0.0078 | **0.0075** | 0.0069 |
| 1.00 | 0.25 | 0.2500 | **0.2518** | 0.2355 |
| 0.50 | 0.25 | 0.0156 | **0.0149** | 0.0137 |

🚨 **Misquote 1 — it is about the INITIAL wealth, not a drawdown from the peak.** ✅ Measured: a
50% fall **from the running maximum** happens on **100.0%** of full-Kelly paths and **99.7%** of
half-Kelly paths over 200 years, against the formula's 50.0% and 12.5%. Over an unbounded
horizon that probability tends to **1 for any `c`**, because log wealth minus its running
maximum is positive-recurrent. **Never quote `x^(2/c−1)` as "the probability of an x drawdown".**
The honest peak-relative numbers here are median maximum drawdowns of **95.7%** (full Kelly) and
**72.1%** (half) over 200 years.

🚨 **Misquote 2 — a discretely sampled minimum misses crossings.** The naive column is 4–7%
below the truth *relative*, because a monthly sample cannot see a dip that happened inside the
month. **Every drawdown study on sampled data has this bias, and it always understates.** The
bridge correction is three lines: given log wealth `a`, `b` at the ends of a step, the
probability the path stayed above `m` is `1 − exp(−2(a−m)(b−m)/(s²dt))`.

⚠️ eq. (7.13) needs `r = 0`. ✅ source-verified — for `0 < r < m` at full Kelly the exponent
becomes `1 + 2rs²/(m−r)²`, which is `> 1`, so ruin gets *less* likely as the risk-free rate
rises.

## 4. 🚨 Trap 1 — full Kelly on an estimated mean

Take `μ̂` from `N` years of data (`se = σ/√N`), set `f = c·μ̂/σ²`, hold it. Then

```
E[g] = g(f*) * c*(2 - c)  -  c^2 / (2N)
```

The first term is Thorp's exact `c(2−c)`. **The second is the price of not knowing `μ`, and it
depends on nothing but `N`** — not on `μ`, not on `σ`.

✅ Measured — true `μ = 8%`, `σ = 20%`, estimate from `N` years, then trade 30 years:

**N = 10 years (t-stat of the estimate = 1.26)**

| c | mean f | **growth (sim)** | theory | median | **P(g<0)** | median maxDD | 95th maxDD |
|---|---|---|---|---|---|---|---|
| **1.00** | 2.08 | **0.0393** | 0.0300 | 0.0397 | **20.4%** | **80.6%** | 99.9% |
| **0.50** | 1.04 | **0.0518** | 0.0475 | 0.0427 | 6.3% | 51.1% | 91.8% |
| 0.25 | 0.52 | 0.0339 | 0.0319 | 0.0279 | 2.9% | 28.5% | 65.3% |

**N = 30 years (t = 2.19)**

| c | mean f | growth (sim) | theory | P(g<0) | median maxDD |
|---|---|---|---|---|---|
| 1.00 | 2.01 | **0.0645** | 0.0633 | 16.4% | 80.5% |
| 0.50 | 1.00 | 0.0565 | 0.0558 | 5.4% | 51.0% |

🚨 **At N = 10, half Kelly beats full Kelly on mean growth (0.0518 vs 0.0393) with a third of
the probability of losing money and a much smaller drawdown.** Full Kelly is not the
growth-maximizing bet once the mean is estimated — it is an over-bet, and the size of the
over-bet is a function of your sample.

🔑 **And there is an exact answer for how much to shrink.** Maximizing `E[g]` over `c` gives

```
c* = t^2 / (1 + t^2)        where  t = SR * sqrt(N)  is the t-stat of your mean estimate
```

**`t = 1` gives exactly half Kelly. `t = 2` gives 0.8. `t → ∞` gives full Kelly.** ✅ Measured —
at N=10 (`t=1.26`) `c* = 0.615` and `E[g] = 0.0492` against `0.0300` at full Kelly; at N=30
(`t=2.19`) `c* = 0.828`, `E[g] = 0.0662` against `0.0633`.

⚠️ **This derivation is this library's own**, under a stated model — normal estimation error,
`σ` known, one estimate held fixed, no re-estimation — and it is confirmed by the simulation
column, not quoted from a source. ✅ source-verified, Thorp §7.3 p. 411 reaches the same
practical place from his Figure 5: *"A disaster occurs when `m_t = .5 m_e` but we choose
`f = 1.5 f*_e` … Then `g = −.75` and we will be ruined. It is still bad to choose `f = f*_e`
when `m_t = .5 m_e` for then `g = 0`."*

⚠️ The simulated growth sits **above** the closed form because the simulation floors `f` at zero
— it refuses to short on a negative estimate, and the formula does not. The floor helps most
exactly where the estimate is worst. **It does not rescue the drawdowns.**

## 5. 🚨 Trap 2 — a Sharpe ratio is not a Kelly fraction

`f* = μ/σ² = SR/σ`. **Setting `f = SR` is wrong by a factor of `1/σ`,** and the direction of the
error flips at `σ = 1`.

✅ Measured, at `SR = 0.5`:

| σ | μ | **f* = SR/σ** | f = SR | f/f* | g(f*) | g(SR) | **growth kept** |
|---|---|---|---|---|---|---|---|
| 10% | 5.0% | **5.00** | 0.50 | 0.10 | 0.1250 | 0.0238 | **19%** |
| 20% | 10.0% | 2.50 | 0.50 | 0.20 | 0.1250 | 0.0450 | 36% |
| 50% | 25.0% | 1.00 | 0.50 | 0.50 | 0.1250 | 0.0938 | 75% |
| **100%** | 50.0% | 0.50 | 0.50 | **1.00** | 0.1250 | 0.1250 | **100%** |
| **200%** | 100.0% | 0.25 | 0.50 | **2.00** | 0.1250 | **0.0000** | **0%** |

🔑 **At equity volatility it is a large under-bet** — at 10% vol you keep 19% of the available
growth. It is right *only* at `σ = 1`, by coincidence. 🚨 **And at `σ = 2` it is exactly double
Kelly, whose growth rate is exactly zero** — the same wealth forever, with unbounded oscillation.
That is the case to remember, because 200% annual volatility is an ordinary crypto number.

## 6. Volatility targeting, and when it is Kelly

Sizing to a volatility target is `f = v/σ`. Kelly is `f* = SR/σ`. So:

🔑 **Volatility targeting is a Kelly bet if and only if the volatility target equals the Sharpe
ratio.** ✅ Measured at `σ = 20%`, `SR = 0.5`: targets of 10%, 30%, 70% give `f` = 0.50, 1.50,
3.50 against `f* = 2.50` — and only the **50%** target matches.

⚠️ That is a demanding statement of what a vol target assumes. "Size everything to 10% vol"
is a Kelly bet on an asset with a Sharpe of 0.10, and an under-bet on anything better. It is
still a perfectly good rule — it makes strategies comparable and caps the tail — but it is a
*risk-budgeting* choice, not a growth-optimal one, and the two coincide at exactly one point.

## 7. Before you size anything with this

1. **The `μ` is an estimate. Shrink for it** (§4), and use `t²/(1+t²)` as the starting point
   rather than a taste. Count the trials that produced the estimate too —
   `../../../fin-core/skills/backtest-validation/scripts/trial_ledger.py`.
2. **Net of costs, not gross.** Kelly on a gross edge sizes a strategy that does not exist —
   `../../../fin-core/skills/backtest-validation/scripts/cost_curve.py` gives the net `μ`.
3. **`σ` is estimated too**, and it appears squared. The script's derivation treats `σ` as
   known; it is not, and a `σ` error hits `f*` twice as hard as a `μ` error of the same
   relative size.
4. **The model is a single GBM.** Real returns are fat-tailed and serially dependent, both of
   which make the same `f` riskier than the formula says. A jump kills a levered book that the
   diffusion approximation says is safe.
5. **Report the drawdown you are actually buying** — the peak-relative number of §3, not
   `x^(2/c−1)`.

## 8. Scripts and where this sits

`scripts/position_sizing.py` — `kelly_binary` / `kelly_discrete` / `kelly_max_growth_binary`,
`kelly_fraction` / `growth_rate` / `growth_ratio`, `ruin_prob` / `double_before_halve` and the
bridge-corrected `simulate_ruin`, `expected_growth_estimated` / `optimal_c_estimated` /
`simulate_estimated_kelly`, and `vol_target_fraction`. numpy + scipy, seed 20260909, 21 s.

- The signal whose edge you are sizing — `../trend-following-models/SKILL.md` and
  `../alpha-combination-and-neutralization/SKILL.md`.
- Whether that edge survives its own trial count and its costs —
  `../../../fin-core/skills/backtest-validation/SKILL.md` and
  `../../../fin-core/skills/research-integrity-guards/SKILL.md`.
- Turning a single `f` into weights across many assets under constraints, and the risk
  decomposition — `../../../fin-core/skills/portfolio-and-risk/SKILL.md`. 🔑 Kelly for a
  portfolio is `f* = Σ⁻¹μ`, which is a *covariance* problem and belongs there.
- Getting the resulting order done — `../execution-algorithms/SKILL.md`.
- Not pointing a levered size at a funded account —
  `../../../fin-core/skills/broker-execution-apis/scripts/paper_account_guard.py`.
