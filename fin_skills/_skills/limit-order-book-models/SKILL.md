---
name: limit-order-book-models
description: >-
  Model the order book as a queueing system - Cont-Stoikov-Talreja birth-death queues, the
  probability the mid moves up before down given the two queue sizes, and the fill probability
  and adverse selection of a passive order at a given queue position. TRIGGER - Cont Stoikov
  Talreja, stochastic model for order book dynamics, birth-death queue model, limit order book
  model, probability of an up move given queue sizes, queue imbalance, order book imbalance;
  queue position, "will my limit order get filled", passive fill model, queue-position
  backtest, hftbacktest queue model, "the level traded 3x my size so I was filled",
  order arrival rate lambda mu theta. SKIP for measuring spreads, Kyle
  lambda, Amihud or OFI on your own tape and for reconstructing a book from MBO data
  (intraday-microstructure), for how wide to quote and inventory risk (market-making-models),
  for working a parent order by taking liquidity (execution-algorithms), and for what fills
  cost after the fact (execution-cost-analysis).
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-09"
---

# Limit-order-book models

**The order book is a queueing system, and the two quantities a queueing model gives you that a
tape does not are the direction of the next mid move and the fill probability of your own
passive order.** Both depend on the queue *sizes* and your *position*; neither depends on
volume, and volume is what most fill models are built from.

`../../../fin-core/skills/intraday-microstructure/SKILL.md` measures the book you have (spreads,
Kyle's lambda, Amihud, order-flow imbalance, MBO reconstruction). This skill is the *model* —
what the arrival rates imply about events that have not happened yet.

Every number marked ✅ Measured is printed by `scripts/lob_models.py` (numpy 2.2.6 + scipy
1.13.0, seed 20260909, **about 10 s**). ✅ source-verified means it was read in Cont, Stoikov &
Talreja (2010), *A stochastic model for order book dynamics*, Operations Research 58(3),
549–563, in the authors' preprint at `www.columbia.edu/~ww2040/orderbook.pdf`; equation, table
and proposition numbers are as printed there.

## 1. The model, and the one thing that makes it tractable

✅ source-verified — CST §2.2, for a level `i` ticks from the **opposite** best quote:

```
limit orders    arrive at rate  lambda(i)
market orders   arrive at rate  mu
cancellations   arrive at rate  theta(i) * x      when x orders rest at that level
```

🔑 **The cancellation rate is proportional to the depth.** That single choice is what makes the
queue ergodic for any `lambda` (CST Proposition 1) and gives a closed-form stationary law,
`pi_x ∝ lambda^x / prod_{k=1..x}(mu + k*theta)`. A constant cancellation rate does not, and a
model with one will blow up when `lambda > mu + theta`.

✅ source-verified — Table 2, estimated from Tokyo Stock Exchange data for Sky Perfect
Communications, in units of the **average limit-order size** and per minute:

| i | 1 | 2 | 3 | 4 | 5 | | |
|---|---|---|---|---|---|---|---|
| `lambda(i)` | **1.85** | 1.51 | 1.09 | 0.88 | 0.77 | `mu` | **0.94** |
| `theta(i)` | **0.71** | 0.81 | 0.68 | 0.56 | 0.47 | `k`, `alpha` (power-law fit `k/i^alpha`) | 1.92, 0.52 |

✅ Measured — the stationary mean depth those numbers imply at the touch is **1.62 units**. Any
statement below about a 20-deep queue is a statement about the *mechanics*, not about that stock.

## 2. ✅ The direction of the next move, reproduced

CST Proposition 3 (eqs. 9–11) gives `P[mid up before down]` as an inverse Laplace transform of a
continued fraction. **You do not need to implement that.** ✅ source-verified — the proposition's
own proof says the price "changes for the first time exactly when one of the two independent
birth-death processes `X~_A` and `X~_B` reaches the state 0 for the first time … the quantity
(8) is given by `P[sigma_A < sigma_B]`."

That is a first-passage race between two independent birth-death chains, and it is an
**absorption probability of the two-dimensional chain** — one sparse linear solve, no Laplace
inversion, no continued fractions, exact.

✅ Measured — `prob_mid_up(a, b)` against the paper's own Table 3 (bottom panel, its
Laplace-transform values), rows `b` = bid depth, columns `a` = ask depth:

| | a=1 | a=2 | a=3 | a=4 | a=5 |
|---|---|---|---|---|---|
| **b=1** | 0.5000 / .500 | 0.3352 / .336 | 0.2590 / .259 | 0.2158 / .216 | 0.1878 / .188 |
| **b=2** | 0.6648 / .664 | 0.5000 / .500 | 0.4069 / .407 | 0.3479 / .348 | 0.3072 / .307 |
| **b=3** | 0.7410 / .741 | 0.5931 / .593 | 0.5000 / .500 | 0.4366 / .437 | 0.3906 / .391 |
| **b=4** | 0.7842 / .784 | 0.6521 / .652 | 0.5634 / .563 | 0.5000 / .500 | 0.4523 / .452 |
| **b=5** | 0.8122 / .812 | 0.6928 / .693 | 0.6094 / .609 | 0.5477 / .548 | 0.5000 / .500 |

**Worst |difference| over all 25 cells: 8.27e-04** — every cell agrees to the three decimals
Table 3 is printed in. ✅ Two independent Monte Carlo checks: `a=1, b=5` → 0.8117 ± 0.0011
against the exact 0.8122; `a=5, b=1` → 0.1880 ± 0.0011 against 0.1878.

🔑 **What the table says**: the direction of the next mid move is a function of two integers you
can read off the top of the book. `b=5` against `a=1` is a **0.812** chance of an up move. The
diagonal is exactly 0.500 — no drift, no price history, no signal beyond the queues. This is the
model behind every "queue imbalance predicts the next tick" result; the measurement side of it
(order-flow imbalance, and the Cont-Kukanov-Stoikov OFI regression) is
`../../../fin-core/skills/intraday-microstructure/SKILL.md` §4.

## 3. ⚠️ Proposition 5 did not reproduce — reported, not hidden

CST Proposition 5 (eqs. 14–16) is the probability that a limit order at the bid executes before
the mid moves. ✅ source-verified — eq. (14) makes the order's waiting time `epsilon_B` a sum of
independent exponentials with rates `mu + theta*(i-1)`, `i = 1..b` (so `b` counts *you*: `b=1`
is the front of the queue, and the last transition is at rate `mu` because your own order never
cancels), and eq. (16) makes the event `P[epsilon_B < sigma_A]`.

⚠️ Implementing exactly that gives, against the paper's Table 4 (bottom panel):

| | a=1 | a=3 | a=5 |
|---|---|---|---|
| **b=1** | 0.5025 / .497 | 0.7939 / .709 | 0.8824 / .776 |
| **b=3** | 0.2915 / .206 | 0.5784 / .422 | 0.7128 / .528 |
| **b=5** | 0.2243 / .118 | 0.4716 / .287 | 0.6071 / .393 |

**Worst |difference| 0.214.** ✅ An independent Monte Carlo agrees with the left column of each
pair to three decimals, and ✅ `best_lambda_for_table4()` sweeps the limit-order rate over
`(0, 6]` and finds no value that reproduces Table 4 — **the best is `lambda = 1`, still 0.074
off, and Table 3 needs 1.85**. So the arithmetic is not the problem; the reading of the event
is. **Use Proposition 3, which does reproduce, and measure fill probability directly** — §4.

⚠️ If you are checking your own implementation against this paper, check it against Table 3.

## 4. 🚨 The trap: fill probability from volume, ignoring queue position

The rule almost every backtest uses is *"I was filled once the tape printed more size at my
price than was ahead of me."* It has one input, volume, and it cannot see where in the queue you
were.

✅ Measured — a symmetric book, **both queues 20 units deep**, one passive buy order per row,
sweeping only its position (0 = front). Same level, same tape, same printed volume:

| position | fill prob | ± | volume-only rule | vol / true | trades seen | of the places you gained, share from cancels |
|---|---|---|---|---|---|---|
| **0** | **0.977** | 0.001 | 0.977 | 1.00 | 4.11 | — |
| 1 | 0.949 | 0.002 | 0.899 | 0.95 | 4.11 | 0.43 |
| 2 | 0.920 | 0.002 | 0.760 | 0.83 | 4.11 | 0.52 |
| 5 | 0.837 | 0.003 | 0.241 | 0.29 | 4.11 | 0.66 |
| **10** | **0.714** | 0.003 | **0.006** | **0.01** | 4.11 | 0.76 |
| 15 | 0.609 | 0.003 | 0.000 | 0.00 | 4.11 | 0.81 |
| **19** | **0.533** | 0.004 | 0.000 | 0.00 | 4.11 | 0.84 |

- 🚨 **At position 10 the volume-only rule says 0.006 and the truth is 0.714 — 110× too low**,
  because **76% of the places you gain are gained by cancellations ahead of you**, and a
  cancellation never prints on the tape. The level only ever trades 4.11 times per episode; it
  clears mostly by people leaving.
- 🚨 **The same level supports fill probabilities from 0.533 to 0.977.** Volume sees one number.
- ✅ The same shape inside CST's own calibrated depth range (both queues 5 deep):
  pos 0 → **0.862**, 1 → 0.740, 2 → 0.637, 3 → 0.547, 4 → **0.467**.
- 🔑 **The back-of-queue number is structural, not a parameter.** From the back of a symmetric
  queue, being filled is very nearly the same event as *the bid queue emptying* — which is the
  mid falling. 0.533 against `P[mid down]` = 0.501. **No depth and no volume changes that**;
  it is what "the back of the queue" means.

⚠️ The direction of the volume-only error flips with the cancel-to-trade ratio: at the front it
is exact, and in a book where the queue clears by trading rather than cancelling it would
*overstate*. Measure your own ratio before trusting either sign. `hftbacktest` ships an
L3/L2 queue-position model for exactly this reason —
`../../../fin-core/skills/intraday-microstructure/SKILL.md` §6 has the data requirements (you
need order IDs; with L2 you assume the queue, you do not replay it).

## 5. 🚨 The fills you get are the ones you did not want — with no informed traders anywhere

You buy at the bid = mid − 0.5 ticks. Mark to the mid at the end of the episode: **+1.0 tick if
the mid then goes up, 0.0 if it goes down.** "Earning the half spread" means **0.5**.

✅ Measured, same runs. The book is symmetric, so `P[mid up]` over *all* episodes must be 0.500
exactly — the run reports **0.499**, which is the measurement checking itself:

| position | fill prob | P[up \| filled] | P[up] all | mark-to-mid (ticks) | adverse selection |
|---|---|---|---|---|---|
| 0 | 0.977 | 0.491 | 0.499 | 0.491 | **0.009** |
| 2 | 0.920 | 0.469 | 0.499 | 0.469 | 0.031 |
| 5 | 0.837 | 0.439 | 0.499 | 0.439 | 0.061 |
| 10 | 0.714 | 0.394 | 0.499 | 0.394 | 0.106 |
| **19** | 0.533 | **0.345** | 0.499 | **0.345** | **0.155** |

🚨 **Every counterparty in this model is uninformed and anonymous. There is no news, no signal,
no toxic flow — and a back-of-queue fill still gives back 31% of the half-spread.** Conditioning
on *having been filled* is itself the information: a fill is evidence the bid queue is being
consumed, and a consumed bid queue is a mid about to fall.

🔑 **Queue position buys fill probability and fill quality at the same time.** From the front, a
fill is one market sell among many and says almost nothing (0.491 ≈ 0.5). From the back it is
almost the down-move itself (0.345). A fill model that gets the probability right and prices
every fill at half the spread is wrong in the second decimal at the front and by a third at the
back.

🔑 **This is on top of, not instead of, informed-flow adverse selection.**
`../../../fin-strategies/skills/market-making-models/SKILL.md` §4 measures −31% of mean PnL from
a 50% informed arrival rate in Avellaneda-Stoikov, whose §6 table lists "queue position" as
something AS does not model at all — quotes there fill at intensity `lambda(delta)` with no
queue. **The two costs stack.** Avellaneda-Stoikov tells you how wide; this tells you what a
fill at that width is actually worth.

## 6. What the script gives you

`scripts/lob_models.py` — numpy + scipy only, importable, no file writes, no network:

| Function | Does |
|---|---|
| `CST_PARAMS`, `CST_TABLE3`, `CST_TABLE4` | Table 2's fitted rates and both printed panels |
| `stationary_queue_dist(lam, mu, theta)` | closed-form stationary depth law of one queue |
| `prob_mid_up(a, b, ...)` | Proposition 3, exact, by 2-D absorption — §2 |
| `prob_mid_up_grid(n)` | the 5×5 panel |
| `prob_mid_up_sim(a, b, ...)` | seeded Monte Carlo cross-check, returns `(p, std err)` |
| `fill_prob_before_move(b, a, ...)` | Proposition 5 as eqs. (14)/(16) state it — §3 |
| `cst_table_checks()` | both panels against the paper, with the two gaps |
| `best_lambda_for_table4()` | the λ sweep of §3 — no rate reconciles the two panels |
| `book_experiment(position, depth, ...)` | §4 and §5: fill probability, the volume-only rule, the cancellation share, `P[up \| filled]` and the mark-to-mid |
| `position_sweep(depth, positions)` | the sweep |

⚠️ **What this model does not have**: order sizes (everything is one unit — the paper's unit is
the *average* limit-order size), hidden and iceberg liquidity, a spread wider than one tick in
§§4–5, latency, self-impact, and any dependence of arrival rates on recent price moves. CST's
own conclusion names order-size heterogeneity and order-flow/price correlation as the two
extensions it leaves out.

## Where this sits

- `../../../fin-core/skills/intraday-microstructure/SKILL.md` — measuring the book you have:
  effective vs realized spread, Kyle's lambda, Amihud, OFI, trade classification, and what MBO
  data you need before queue position is even computable. **Read that first if you have data.**
- `../../../fin-strategies/skills/market-making-models/SKILL.md` — Avellaneda-Stoikov: where to
  centre the quotes and how wide, and the informed-flow adverse selection the model has no term
  for. This skill is the queue underneath its `lambda(delta)`.
- `../../../fin-strategies/skills/execution-algorithms/SKILL.md` — the other side: working a
  parent order by *taking* liquidity, and Almgren-Chriss.
- `../../../fin-core/skills/execution-cost-analysis/SKILL.md` — what a set of fills actually
  cost after the fact. §5's mark-to-mid is the ex-ante version of that measurement.
- `../hawkes-processes/SKILL.md` — CST's arrivals are Poisson. Real order flow clusters, and
  Poisson standard errors on a clustered arrival rate are too narrow by a measured factor.
- `../monte-carlo-methods/SKILL.md` — the standard errors above are binomial; §4's `± 0.004`
  is an error bar on the estimator, not on the model.
