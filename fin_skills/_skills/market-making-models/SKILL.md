---
name: market-making-models
description: >-
  Quote a two-sided market and survive the inventory - Avellaneda-Stoikov reservation price and
  optimal spread, and the adverse selection the model does not price. TRIGGER - Avellaneda
  Stoikov, market making model, optimal market making, reservation price, indifference price,
  inventory skew, optimal bid ask spread, quoting strategy, "how wide should I quote", skew my
  quotes, inventory risk, gamma risk aversion market maker; order arrival intensity, A exp(-k
  delta), Poisson fill model, fill probability vs distance from mid; adverse selection, informed
  flow, toxic flow, getting picked off, Glosten-Milgrom, order flow toxicity, VPIN. SKIP for
  measuring realized and effective spreads from your own tape (intraday-microstructure), for
  working a parent order by taking liquidity (execution-algorithms), for what a fill cost you
  after the fact (execution-cost-analysis), and for exchange connectivity and order types
  (broker-execution-apis).
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-09"
---

# Market-making models

**Avellaneda-Stoikov answers one question — how do I quote so that inventory does not kill me —
and it is routinely quoted as if it answered a second one it never touches: how do I quote so
that informed traders do not.** The formulas are right. The risk they price is not the only
risk there is.

Everything marked ✅ Measured comes from `scripts/market_making.py` — numpy only, seed 20260909,
4,000 paths, **3 s**. Everything marked ✅ source-verified was read by this library in the
published article (*Quantitative Finance* 8(3), 217–224); equation numbers are as printed there
and the full transcription is in the script's docstring.

## 1. The two formulas, verified

✅ source-verified:

```
(8)  r(s, q, t)          = s - q*gamma*sigma^2*(T - t)                 reservation price
(30) delta^a + delta^b   = gamma*sigma^2*(T - t) + (2/gamma)*ln(1 + gamma/k)   TOTAL spread
(12) lambda(delta)       = A * exp(-k * delta)                         arrival intensity
```

eq. (8) is the *average* of the reservation ask (6) and bid (7); eq. (29) shows the trading
agent gets the same price under an expansion in `q` (22) and a **linear** approximation of the
arrival term (26).

🔑 **eq. (30) is the total spread, not a half-spread**, and it is quoted *around the reservation
price*: `p^a = r + spread/2`, `p^b = r − spread/2`. Halving it once more is the most common
implementation error, and it doubles your fill rate silently.

✅ Measured — the sharpest available check on eq. (30) is the paper's own "Average spread"
column, because the time-average of eq. (30) over `[0, T]` is closed-form
`γσ²T/2 + (2/γ)ln(1 + γ/k)`:

| γ | this formula | the paper prints | in |
|---|---|---|---|
| 0.1 | **1.4908** | 1.49 | Table 1 |
| 0.01 | **1.3489** | 1.35 | Table 2 |
| 1 | **3.0217** | 3.02 | Table 3 |

Three independent numbers, three of the paper's own tables, one closed form. If your
`ln(1 + γ/k)` is wrong, this catches it.

✅ Measured — eq. (8) at `s=100, γ=0.1, σ=2, T−t=1`: `q=0 → 100.00`, `q=+5 → 98.00`, and at
`t=T → 100.00`. **Long inventory pushes the quote pair down**, so the ask is likelier to be hit.

🚨 **The spread does not depend on `q`.** ✅ source-verified — AS: *"the bid-ask spread in (25)
is independent of the inventory. This follows from our assumption of exponential arrival
rates."* **All the inventory control is in where the spread is centred.** A market maker who
*widens* when long is doing something the model does not say — possibly sensible, but it is not
Avellaneda-Stoikov and it does not inherit the optimality.

## 2. The paper's own experiment, reproduced

✅ source-verified, §3.3 verbatim: `s=100, T=1, σ=2, dt=0.005, q=0, γ=0.1, k=1.5, A=140`, 1,000
simulations; the "symmetric" benchmark *"uses the average bid/ask spread of the inventory
strategy over the time period, but centres it around the mid-price."*

✅ Measured — 4,000 paths, an independent implementation, a different RNG:

| γ | strategy | spread | PnL | std | q | q std | **the paper** |
|---|---|---|---|---|---|---|---|
| 0.1 | inventory | 1.49 | **64.8** | 6.7 | 0.04 | 2.8 | 1.49 / **65.0** / 6.6 / 2.9 |
| 0.1 | symmetric | 1.49 | **67.9** | 13.7 | −0.15 | 8.4 | 1.49 / **68.4** / 12.7 / 8.4 |
| 0.01 | inventory | 1.35 | **68.1** | 9.1 | −0.06 | 5.1 | 1.35 / **68.6** / 8.7 / 5.1 |
| 0.01 | symmetric | 1.35 | **68.4** | 13.8 | −0.11 | 8.7 | 1.35 / **68.8** / 12.8 / 8.7 |
| 1 | inventory | 3.03 | **31.5** | 4.9 | 0.02 | 1.6 | 3.02 / **31.4** / 5.0 / 1.7 |
| 1 | symmetric | 3.02 | **43.6** | 10.7 | 0.00 | 5.1 | 3.02 / **44.0** / 11.0 / 5.1 |

**All six rows of Tables 1, 2 and 3, within about 1%.**

🔑 And the paper's own reading reproduces: **the symmetric strategy earns MORE** (it sits on the
mid and takes more volume) with **about twice the PnL dispersion** and several times the
inventory dispersion. ✅ source-verified — AS: *"in the limit as γ → 0 the two strategies are
identical"*, which the γ=0.01 rows show. **The inventory strategy is not a way to make more
money. It is a way to make slightly less of it with a much thinner tail.**

## 3. 🚨 "With probability `λ·dt`" is not a probability here

Everyone reimplementing AS reaches this fork, and most do not notice they took it.

✅ Measured — at the paper's own `A = 140`, `dt = 0.005`:

| δ | λ(δ) | **λ·dt** | 1 − e^(−λ·dt) |
|---|---|---|---|
| **−0.5** | 296.4 | **1.000** (clipped) | 0.773 |
| 0.0 | 140.0 | **0.700** | 0.503 |
| 0.5 | 66.1 | 0.331 | 0.282 |
| 1.0 | 31.2 | 0.156 | 0.145 |

`A·dt = 0.70`. **This is not a small-probability regime**, and `λ·dt` **exceeds 1** whenever a
large inventory pushes a quote through the mid — which eq. (8) does not forbid.

✅ Measured — the same six rows, both readings:

| γ | strategy | PnL (`λ·dt`) | PnL (Poisson) | gap |
|---|---|---|---|---|
| 0.1 | inventory | 64.8 | 57.1 | **−12%** |
| 0.1 | symmetric | 67.9 | 60.8 | −11% |
| 0.01 | inventory | 68.1 | 60.1 | −12% |
| 0.01 | symmetric | 68.4 | 60.4 | −12% |
| 1 | inventory | 31.5 | 28.2 | −10% |
| 1 | symmetric | 43.6 | 42.1 | −4% |

🔑 **The reproduction in §2 only works with `λ·dt` read literally.** An implementer who
"corrects" it to the exact Poisson probability `1 − e^{−λ dt}` reports **10–12% less profit on
the same strategy** and stops matching the paper. Neither reading is wrong. **Publishing a
number without saying which one you used is.**

## 4. 🚨 The risk the model has no term for

Every counterparty in AS is uninformed — arrivals depend only on distance from the mid. The
entire risk in the model is inventory risk.

✅ Measured — a fraction `φ` of arrivals know the sign of the next mid move and take only the
side that pays (`φ = 0` is the paper's model exactly):

| φ | strategy | PnL | PnL std | fills | PnL/fill | picked off |
|---|---|---|---|---|---|---|
| 0% | inventory | **64.8** | 6.7 | 86 | 0.750 | 0.0 |
| 0% | symmetric | 67.9 | 13.7 | 81 | 0.839 | 0.0 |
| 10% | inventory | 60.9 | 6.3 | 83 | 0.738 | 4.9 |
| 25% | inventory | 54.9 | 6.1 | 77 | 0.713 | 12.1 |
| **50%** | inventory | **45.0** | 5.7 | 68 | **0.666** | 24.3 |
| 50% | symmetric | 48.0 | 12.2 | 63 | 0.757 | 22.8 |

🚨 **−31% of mean PnL, on quotes that are optimal by eq. (30) at every single step.** Nothing in
the model is wrong; the model simply has no informed trader, so the spread it prescribes
compensates for inventory risk **and for nothing else**.

🔑 **Both strategies degrade almost identically** — the inventory strategy is no defence,
because the reservation-price skew is an *inventory* control, not a *toxicity* one. It knows
how much you hold. It does not know why you were filled.

## 5. So how much wider? Less than the story says

The Glosten-Milgrom argument — the spread must cover the expected loss to informed flow —
arrived at by simulation rather than by their equilibrium.

✅ Measured — `φ = 25%`, sweeping a multiplier on eq. (30)'s spread. "jump" is the extra mid
displacement an informed fill causes; the ratio is `(jump + σ√dt) / half-spread`:

| jump | **adverse / half-spread** | ×1 | ×1.25 | ×1.5 | ×2 | ×3 | **best** |
|---|---|---|---|---|---|---|---|
| 0.0 | 0.19 | **54.9** | 53.5 | 49.6 | 38.9 | 19.5 | ×1 |
| 0.5 | 0.86 | **52.1** | 51.2 | 47.8 | 37.7 | 19.1 | ×1 |
| 1.0 | 1.53 | **49.2** | 48.9 | 46.0 | 36.6 | 18.6 | ×1 |
| 2.0 | 2.87 | 43.6 | **44.3** | 42.4 | 34.3 | 17.7 | **×1.25** |
| 4.0 | 5.56 | 32.2 | 35.1 | **35.2** | 29.8 | 15.8 | **×1.5** |

🔑 **Column 2 is the whole answer.** Widening pays only once the adverse move per informed fill
is well above the half-spread being earned. ⚠️ **And at AS's own parameters it is nowhere near**
— the prescribed half-spread is **0.745 against a mid step of `σ√dt` = 0.141, i.e. 5.3× one
step** — so an AS quoter is genuinely hard to pick off there, and widening mostly starves the
flow.

🔑 **The governing quantity is the ratio, not your risk aversion. eq. (30) prices `γ`. It does
not price `φ`.** Measure your own ratio — the post-fill price move at a horizon, against your
half-spread — before deciding whether to widen. The measurement tools are in
`../../../fin-core/skills/intraday-microstructure/SKILL.md` (effective vs realized spread and
the price-impact decomposition are exactly this quantity).

⚠️ `A`, `k` and the size of an informed move are the three numbers this depends on, and **AS
calibrates none of them.** Read the shape of this table, not its levels.

## 6. What else the model leaves to you

| Not in AS | Why it matters |
|---|---|
| **an inventory bound** | ✅ source-verified: the paper states no `q_max`. eq. (8) skews the quotes but nothing stops `q` growing; every production implementation adds a cap, and that cap is not optimal in the paper's sense |
| **a tick grid** | quotes are continuous reals; rounding to a tick changes the fill rate non-linearly |
| **queue position** | a limit order at a price is assumed to fill at intensity `λ(δ)`; in a real book you are behind a queue |
| **calibration of `A`, `k`** | eq. (12) derives the *form* from a power-law order-size distribution and **logarithmic** impact; the constants are chosen, not fitted |
| **the terminal time `T`** | the whole inventory skew scales with `T − t`. There is no natural `T` for a maker who quotes every day; picking it is a modelling decision the paper does not make for you |

⚠️ **A note on the arrival law that is often garbled:** eq. (12) is `λ(δ) = A·e^{−kδ}`, derived
from a power-law order-size density `f^Q(x) ∝ x^{−1−α}` combined with **logarithmic** impact.
✅ source-verified — the paper's *power-law* intensity, from combining that density with
`Δp ∝ Q^β` instead, is `λ(δ) = B·δ^{−α/β}` (unnumbered) — **not** `A·δ^{−α}`, which is a common
misquote.

## 7. Scripts and where this sits

`scripts/market_making.py` — `reservation_price` (eq. 8), `optimal_spread` (eq. 30),
`mean_optimal_spread` (its time-average, the check against the paper's tables),
`arrival_intensity` (eq. 12), `fill_probability` (both readings), the vectorized `simulate`
with `informed_frac` / `informed_jump`, and `best_spread_multiplier`. numpy only, seed
20260909, 3 s.

- Effective and realized spreads, price impact, trade classification, order-flow imbalance —
  `../../../fin-core/skills/intraday-microstructure/SKILL.md`. That is where the `φ` of §5 is
  measured on real data.
- Working a parent order by *taking* liquidity, and Almgren-Chriss —
  `../execution-algorithms/SKILL.md`.
- What a set of fills actually cost, after the fact —
  `../../../fin-core/skills/execution-cost-analysis/SKILL.md`.
- Order types, venue connectivity, and not pointing this at a funded account —
  `../../../fin-core/skills/broker-execution-apis/SKILL.md` and its
  `../../../fin-core/skills/broker-execution-apis/scripts/paper_account_guard.py`. 🚨 A quoting
  strategy is two-sided and always in the market; it is the worst possible thing to run
  untested against a live account.
- How much capital to put behind the quotes — `../position-sizing-kelly/SKILL.md`.
- Whether a backtested quoting P&L is a discovery —
  `../../../fin-core/skills/backtest-validation/SKILL.md`.
