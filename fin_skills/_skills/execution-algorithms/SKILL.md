---
name: execution-algorithms
description: >-
  Build the schedule that works an order - VWAP, TWAP, POV, Almgren-Chriss - and know what each
  one is optimizing. TRIGGER - VWAP algo, TWAP algo, POV, percentage of volume, participation
  rate, child order slicing, execution schedule, order scheduling, "how should I work this
  order", intraday volume profile, U-shaped volume curve; Almgren-Chriss, optimal execution,
  optimal liquidation, trading trajectory, efficient frontier of execution, risk aversion
  lambda, trade half-life, kappa, market impact model, temporary vs permanent impact, square
  root law; implementation shortfall, Perold, arrival price, decision price, delay cost,
  opportunity cost, unfilled shares. SKIP for measuring fills you already have and choosing a
  benchmark after the fact (execution-cost-analysis), for spreads, trade classification and
  order-flow measures (intraday-microstructure), for quoting rather than taking
  (market-making-models), and for broker order types and routing (broker-execution-apis).
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-09"
---

# Execution algorithms

**Every schedule is optimal for something. The failure is not choosing badly — it is not
knowing what your schedule was chosen to minimize, and then reporting it against a benchmark
that measures something else.**

Everything marked ✅ Measured comes from `scripts/execution_algos.py` — numpy + pandas, seed
20260909, one 390-bin session plus closed forms, **2 s**. Everything marked ✅ source-verified
was read by this library in the December 2000 Almgren-Chriss preprint; the equation numbers are
as printed there and the full transcription is in the script's docstring.

This skill is the **schedule**. `../../../fin-core/skills/execution-cost-analysis/SKILL.md` is
the **measurement** of fills you already have — read it for benchmark choice, the reversion
test, and what to log per parent order. §3 below is the one place they touch, and it is a
different mechanism, stated as such.

## 1. The four schedules, and what each one ignores

| Schedule | Minimizes | Blind to |
|---|---|---|
| **TWAP** | nothing; it spreads evenly | volume, entirely |
| **VWAP** | tracking error to the session VWAP | whether trading at the market's pace was wise |
| **POV `r`** | footprint per bin | **completion — it can simply not finish** |
| **Almgren-Chriss** | `E[cost] + λ·Var[cost]` | the volume profile; it is a clock, not a participation rule |

✅ Measured — 250,000 shares against a 5M-share session drifting +59 bps against the buyer,
with Almgren-Chriss linear impact applied:

| Schedule | avg fill | vs arrival | done by | **unfilled** |
|---|---|---|---|---|
| TWAP | 50.8381 | +166.6 | 390m | 0 |
| VWAP (perfect forecast) | 51.0139 | +201.8 | 390m | 0 |
| POV 10% | 51.7272 | +344.4 | **192m** | 0 |
| **POV 3%** | **50.6792** | **+134.8** | 390m | **100,000** |

🚨 **POV 3% has the best cost per share and finished 40% of the order.** A participation cap
does not reduce cost — it **converts** cost into unfilled shares, which is the one component a
fill-based TCA cannot see. §5 prices that conversion.

⚠️ **VWAP here uses a perfect forecast of the realized profile.** That is not available in
advance, and it is the practical difference between a VWAP *schedule* and a VWAP *benchmark*.
A real VWAP algo tracks a forecast and misses.

## 2. Almgren & Chriss (2000), reproduced

✅ source-verified — every formula below was read in the paper. `η̃ = η − ½γτ` is an
**unnumbered** display immediately after eq. (8), and the `κ`/`κ̃` relation is unnumbered too;
both are commonly mis-transcribed.

```
(6)  g(v) = gamma*v                    (7)  h(n_k/tau) = eps*sgn(n_k) + (eta/tau)*n_k
(8)  E(x) = 0.5*gamma*X^2 + eps*sum|n_k| + (eta~/tau)*sum n_k^2 ,  eta~ = eta - 0.5*gamma*tau
(5)  V(x) = sigma^2 * sum_k tau * x_k^2
(16) (1/tau^2)(x_{j-1} - 2 x_j + x_{j+1}) = kappa~^2 * x_j ,  kappa~^2 = lambda*sigma^2/eta~
     and   (2/tau^2)(cosh(kappa*tau) - 1) = kappa~^2
(17) x_j = sinh(kappa*(T - t_j)) / sinh(kappa*T) * X
(18) n_j = 2*sinh(kappa*tau/2)/sinh(kappa*T) * cosh(kappa*(T - t_{j-1/2})) * X
(19) kappa ~ sqrt(lambda*sigma^2/eta)         section 2.3:  half-life theta = 1/kappa
```

**The paper's own Table 1 test case** (p. 25, verbatim): `S0=50`, `X=10^6`, `T=5` days, `N=5`,
`σ=0.95 ($/share)/day^½`, `ε=0.0625`, `γ=2.5e−7`, `η=2.5e−6`, `λ=1e−6`.

✅ Measured — the script reproduces the three numbers the paper prints for it:

| The paper says | This script computes |
|---|---|
| "κ ≈ 0.6/day, so κT ≈ 3" (p. 24, from eq. 19) | eq. (19): **0.6008**/day; exact: **0.6071**/day; **κT = 3.04** |
| holding untraded: `σ√T` = 2.12 $/share, `√V` = $2.12M | **2.1243** $/share, **$2.12M** |
| half-life `θ = 1/κ` | **1.65 days** |

And four internal checks that the transcription is right:

| Check | Residual |
|---|---|
| eq. (17) substituted back into eq. (16) | 1.16e−10 |
| eq. (18) written out vs `−diff(`eq. 17`)` | 2.91e−11 |
| eq. (20) closed form vs eq. (8)/(5) summed on the trajectory | **1.3e−16** (E), **1.7e−16** (V) |
| eq. (10)/(11) vs the same sums on the linear trajectory | exact |

🔑 **If you implement AC, run the third check.** eq. (20) is long enough that a transposed `T`
and `τ` produces a plausible number, and summing eq. (8) over eq. (17)'s own trajectory is a
four-line independent oracle.

🚨 **`η̃`, not `η`, in the cost.** ✅ Measured: writing `η` where eq. (20) writes `η̃` gives
$949,318 instead of $911,227 — **+4.18%** at the paper's own parameters, and the gap grows with
`γτ/η`. And ⚠️ eq. (19) uses **bare `η`** under the square root while `κ̃²` uses **`η̃`** — that
is not a typo in the paper.

## 3. 🚨 Trap 1 — the VWAP your own impact moved

✅ Measured — the same volume-proportional schedule at four sizes, with permanent impact
(eq. 6) displacing the mid for **every print in the session** and temporary impact (eq. 7) on
the trader's own fills:

| Participation | avg fill | realized VWAP | **clean VWAP** | measured cost | **TRUE cost** | hidden | **% of true hidden** |
|---|---|---|---|---|---|---|---|
| 1% | 50.3462 | 50.1851 | 50.1772 | +32.1 | +33.7 | +1.6 | **5%** |
| 5% | 51.0580 | 50.2526 | 50.1772 | +160.3 | +175.5 | +15.3 | **9%** |
| 10% | 52.0366 | 50.4258 | 50.1772 | +319.4 | +370.6 | +51.1 | **14%** |
| 25% | 55.7555 | 51.7285 | 50.1772 | +778.5 | **+1111.7** | +333.2 | **30%** |

"Clean VWAP" is the session VWAP the market would have had with the order absent — the
counterfactual, and not a number any desk can compute.

🔑 **Both costs rise, so the benchmark is not flat. What grows is the share of the true cost the
benchmark absorbs** — 5% at 1% participation, **30% at 25%**. Permanent impact moved the mid for
everyone else's prints too, so the whole benchmark has been pushed toward you.

⚠️ **This is a different mechanism from
`../../../fin-core/skills/execution-cost-analysis/SKILL.md` §2**, which holds the *fills fixed*
and shows the benchmark diluted by your own prints. Here the **prices move**. Both are
one-signed the same way and **they compound**.

⚠️ **Do not extrapolate the magnitudes.** Linear temporary impact is the paper's own weakest
assumption — ✅ source-verified, AC write that in that term *"we would expect nonlinear effects
to be most important, and the approximation (7) to be most doubtful."* Read the direction of the
table, and stop at ~25%.

## 4. 🚨 Trap 2 — "Almgren-Chriss beats TWAP"

✅ Measured on the paper's own Table 1 parameters, at the paper's own `λ = 1e−6`:

| Trajectory | E[cost] | E in bps | √V | **U = E + λV** |
|---|---|---|---|---|
| TWAP / linear, eq. (9)–(11) | **$662,500** | 132.5 | $1,040,673 | 1,745,500 |
| Almgren-Chriss, eq. (17)/(20) | $911,227 | 182.2 | **$603,431** | **1,275,356** |

🚨 **Almgren-Chriss costs $248,727 MORE in expectation** and $437,242 less in standard
deviation. It wins only on `U = E + λV`, by $470,144 — **the objective it was derived to
minimize.** ✅ source-verified: in this model the constant-rate trajectory **is** the
minimum-expected-cost one; the paper titles eq. (9) *"Minimum impact"*.

🔑 **So "we switched to Almgren-Chriss and cut costs" inverts the result.** What you bought is
variance reduction, and the price is on the invoice. If you report only mean shortfall, an AC
schedule will look *worse* than the TWAP it replaced — correctly.

The frontier, ✅ Measured (`λ = 2e−6`, `0`, `−2e−7` are the paper's trajectories A, B, C):

| λ | κ /day | half-life (d) | E[cost] | √V | % done by T/2 |
|---|---|---|---|---|---|
| 2.0e−6 | 0.8463 | 1.18 | 1,140,715 | 449,368 | 92% |
| 1.0e−6 | 0.6071 | 1.65 | 911,227 | 603,431 | 85% |
| 2.0e−7 | 0.2748 | 3.64 | 688,154 | 895,776 | 69% |
| **0** | 0 | ∞ | **662,500** | 1,040,673 | 60% |
| −2.0e−7 | 0.2766 | 3.62 | 717,958 | 1,258,654 | **47%** |

⚠️ `λ = 0` is exactly the linear trajectory. `λ < 0` is risk-*seeking*: `κ̃²` goes negative, `κ`
is imaginary, and the solution turns **trigonometric** (`sin`/`cos` rather than `sinh`/`cosh`) —
it postpones selling. Most implementations silently return `nan` there.

🔑 ✅ source-verified — **`θ = 1/κ` does not depend on `T`.** The paper: it is *"determined only
by the security price dynamics and the market impact factors."* **Doubling the deadline does not
slow an AC trade down**; it leaves more of the window unused. If your execution policy is "give
it more time when the order is large", AC does not implement it — `λ` does.

## 5. Implementation shortfall, and the column your TCA does not have

```
delay        = filled   * (arrival_px  - decision_px)      the research-to-desk handoff
execution    = filled   * (avg_fill_px - arrival_px)       the algorithm
opportunity  = unfilled * (final_px    - decision_px)      the shares never bought
fees         = explicit
total        = delay + execution + opportunity + fees
```

🚨 **Attribution — this four-way split is not Perold's, and calling it "Perold's decomposition"
is wrong.** ⚠️ Read in the verbatim reprint of Perold (1988), *"The Implementation Shortfall:
Paper Versus Reality"*, J. Portfolio Management 14(3), 4–9, reprinted in *Streetwise* (Princeton
UP, 1998) pp. 106–109. What is actually his:

| Perold (1988) | |
|---|---|
| `IS = paper portfolio performance − real portfolio performance` | ⚠️ his, verbatim |
| the paper portfolio transacts at **the bid-ask midpoint at the moment of the decision** — not at arrival, not at the previous close | ⚠️ his |
| **two** components: **execution cost** (transactions you did execute) and **opportunity cost** (transactions you failed to execute) | ⚠️ his |
| commissions and transfer taxes are folded **into the net transaction price**, not a separate bucket | ⚠️ his |
| a separate `delay = filled × (arrival − decision)` term | 🔴 **not in his math.** There is no arrival price in it; delay is inside his single execution-cost term |

⚠️ **The four-component taxonomy — commission, price impact, timing (the delay term),
opportunity — is Wagner & Edwards (1993), "Best Execution", *Financial Analysts Journal* 49(1),
65–71, p. 67**, which cites Perold. The modern `decision → arrival → fill` algebra is Kissell &
Glantz (2003). Cite the right one.

🔑 Arithmetically it does not matter — `delay + execution = filled × (avg_fill − decision)`, so
the four-way split is a refinement of Perold's execution term, not a contradiction of it. ✅
Measured: `implementation_shortfall()` returns `identity_error`, and it is **0.000e+00** — the
components reconstruct `paper − real` with no residual.

⚠️ **Perold's own sentence for why the opportunity term is not optional**, and the reason this
section exists: *"You could not begin to measure opportunity costs without the paper
portfolio."*

✅ Measured — the POV 3% order from §1, which filled 60%:

| Component | $ | bps of intended notional |
|---|---|---|
| delay | 3,000 | 2.4 |
| **execution** | 101,137 | **80.9** |
| **opportunity** | 31,502 | **25.2** |
| fees | 75 | 0.1 |
| **total** | 135,714 | **108.6** |

🚨 **A fill-only TCA reports 80.9 bps.** The order cost **108.6**, and **23% of it is the
100,000 shares the 3% cap never bought.** Delay and opportunity have no fill record to be
computed from, so they are not "hard to measure" — they are *absent from the input*.

🔑 **Lowering the participation rate does not lower the cost; it moves the cost into the column
your TCA cannot see.** Report fill rate beside every cost number, and log `decision_ts` /
`decision_px` — the field list is in
`../../../fin-core/skills/execution-cost-analysis/SKILL.md` §6.

## 6. Calibration is the part that is not in the paper

🚨 **This library carries no calibrated `η`, `γ` or `ε`.** The script uses the paper's Table 1
values, which are a **1999 US large-cap illustration** — `ε` = half of a **1/8 spread**, `η`
from "1% of daily volume costs one spread", `γ` from "10% of daily volume costs one spread".
Those are the paper's stated rules of thumb, not measurements, and the tick regime they assume
no longer exists.

**Fit them on your own fills.** `η` and `γ` are separable by the reversion test in
`../../../fin-core/skills/execution-cost-analysis/SKILL.md` §5: the part that reverts after your
last fill is temporary, the part that stays is permanent. Until you have that, an AC trajectory
is a *shape* — and the shape is what §4 is about — not a cost estimate.

⚠️ The square-root law `impact ≈ Y·σ·sqrt(Q/V)` is the standard empirical form and is **not**
what AC assume; AC's temporary impact is linear in the rate. Do not mix a square-root `Y` into
eq. (20).

## 7. Scripts and where this sits

`scripts/execution_algos.py` — the session and the three schedules, the full AC closed forms
with their internal checks, the efficient frontier, the VWAP-contamination table and the
shortfall decomposition. numpy + pandas, seed 20260909, 2 s.

- Measuring fills you already have, benchmark choice, the reversion test, what to log —
  `../../../fin-core/skills/execution-cost-analysis/SKILL.md`.
- Spreads, trade classification, order-flow imbalance and the bar-timestamp leak —
  `../../../fin-core/skills/intraday-microstructure/SKILL.md`.
- Quoting instead of taking, and the inventory problem —
  `../market-making-models/SKILL.md`.
- Order types, routing and not sending this to a live account —
  `../../../fin-core/skills/broker-execution-apis/SKILL.md` and its
  `../../../fin-core/skills/broker-execution-apis/scripts/paper_account_guard.py`.
- Turning a cost assumption into a breakeven and a capacity limit —
  `../../../fin-core/skills/backtest-validation/scripts/cost_curve.py`.
- The strategies that generate the orders — `../trend-following-models/SKILL.md` and
  `../alpha-combination-and-neutralization/SKILL.md` §7, whose turnover this prices.
- How large the parent order should have been — `../position-sizing-kelly/SKILL.md`.
