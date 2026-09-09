---
name: duration-convexity-and-dv01
description: >-
  Get the right duration number and the right DV01, for a bond, a floater or a hedge ratio.
  TRIGGER - Macaulay vs modified duration, effective duration, spread duration, key rate
  duration, DV01, PV01, BPV, dollar duration, basis point value, convexity, "my hedge ratio is
  off by a few percent", "duration says the price should be X but it is Y", duration times
  spread, floating rate note duration, FRN duration, "why is my floater duration almost zero",
  BondFunctions.duration, Duration.Macaulay vs Duration.Modified vs Duration.Simple,
  basisPointValue sign. SKIP for turning a price into a yield in the first place
  (yield-measures-and-bill-quotes), for accrued interest and day counts
  (bond-conventions-and-accrued), for the swap annuity and PV01 under OIS discounting
  (ois-discounting-and-multi-curve), and for portfolio VaR and risk aggregation
  (../../../fin-core/skills/portfolio-and-risk).
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-09"
---

# Duration, convexity and DV01

**"Duration" names at least four different numbers and "DV01" at least two.** Substituting one
for another gives a risk report that is a few percent wrong — too small to notice, too large to
hedge with.

Every figure below is printed by `scripts/duration.py` (runs in **0.29 s**; QuantLib optional).
✅ Measured means this file produced it on 2026-09-09 with QuantLib 1.43, numpy 2.2.6,
Python 3.11.3.

> **The rule: Macaulay = (1 + y/m) × Modified, and only Modified belongs in a DV01. DV01 is on
> the DIRTY price. Past 100 bp you need convexity. A floater's rate duration is the time to its
> next reset, not its maturity.**

## 1. Macaulay and Modified differ by exactly (1 + y/m)

✅ Measured — the ratio is the compounding factor to the last digit:

| bond | Macaulay | Modified | ratio | 1 + y/m | DV01 overstated by |
|---|---|---|---|---|---|
| 5% 10y semiannual @ 5% | 7.98944567 | 7.79458114 | 1.02500000 | 1.02500000 | **+2.50%** |
| 5% 10y quarterly @ 5% | 7.92962996 | 7.83173329 | 1.01250000 | 1.01250000 | +1.25% |
| 4% 30y semiannual @ 4% | 17.72805221 | 17.38044334 | 1.02000000 | 1.02000000 | +2.00% |
| **12% 10y semiannual @ 12%** | 6.07905825 | 5.73496061 | 1.06000000 | 1.06000000 | 🚨 **+6.00%** |
| 0% 10y semiannual @ 5% | 10.00000000 | 9.75609756 | 1.02500000 | 1.02500000 | +2.50% |

🚨 **The error is one-directional and grows with the level of rates.** A hedge sized off a
Macaulay duration is always too big, by exactly `y/m`. On a zero-coupon bond Macaulay is the
maturity — which is why the two are so easy to confuse, and why the confusion survives a sanity
check on a zero.

**Definitions, so the names stop moving:**

| | is | units |
|---|---|---|
| **Macaulay** | PV-weighted average time to the cash flows | years |
| **Modified** | `−(1/P)·dP/dy` | years |
| **Money / dollar duration** | `−dP/dy` = Modified × dirty price | currency per unit of yield |
| **DV01 / BPV** | money duration × 1e-4 | currency per basis point |
| **PV01** | the exact reprice for 1 bp | currency per basis point |
| **Effective** | `−(P(+h) − P(−h)) / (2h·P)` | years — the only one that survives optionality |
| **Spread** | the same, bumping the SPREAD not the curve | years |

## 2. 🚨 Duration is a first derivative — a 30y 4% bond at par

✅ Measured: `P0 = 100.000000, ModDur = 17.380443, Convexity = 420.8130`.

| shift | exact | duration only | error | error % | duration + convexity | error |
|---|---|---|---|---|---|---|
| +0.25% | 95.78346 | 95.65489 | −0.12857 | −0.134% | 95.78639 | +0.00293 |
| +1.00% | 84.54567 | 82.61956 | −1.92612 | −2.278% | 84.72362 | +0.17795 |
| +2.00% | 72.32444 | 65.23911 | 🚨 **−7.08532** | 🚨 **−9.797%** | 73.65537 | +1.33094 |
| −1.00% | 119.69013 | 117.38044 | **−2.30969** | **−1.930%** | 119.48451 | −0.20563 |
| −2.00% | 144.95504 | 134.76089 | 🚨 **−10.19415** | 🚨 **−7.033%** | 143.17715 | −1.77789 |

🚨 **At −200 bp duration alone misses 10.19 points — 7.03% of the price.** Convexity cuts that to
1.78 points.

🔑 **The error is one-sided: duration alone always understates the price**, up moves and down
moves alike, because the price/yield curve is convex. So a P&L attribution built on duration
alone has a persistent negative residual that looks like an unexplained loss and is arithmetic.

## 3. DV01 is on the DIRTY price

Modified duration is defined against the **full** price, so a DV01 built on the clean price is
short by `ModDur × accrued × 1e-4`. ✅ Measured, 10y 5% semiannual at y = 4.50%, 1,000,000
notional:

| | clean | accrued | DV01 (dirty) | DV01 (clean) | error |
|---|---|---|---|---|---|
| on a coupon date | 103.990928 | 0.000000 | 817.24 | 817.24 | 0.00 |
| half a period in | 103.990928 | 1.250000 | **827.06** | 817.24 | 🚨 **−9.82 (−1.2%)** |

🚨 **The error breathes with the coupon cycle** — zero on payment dates, largest just before one.
On a portfolio it never nets out, because every bond's cycle is in the same direction.

✅ **DV01 vs PV01: 817.2379 against 816.8504 on 1,000,000, a gap of 0.39.** DV01 is the linear
approximation; PV01 reprices. The gap is the convexity term *inside one basis point* — small on a
10-year, not small on a 30-year, and not zero anywhere.

## 4. ✅ Effective duration reproduces the closed form where the closed form applies

| bond | effective | analytic | \|diff\| | effective convexity | analytic |
|---|---|---|---|---|---|
| 4% 30y @ 4.0% | 17.38046243 | 17.38044334 | 1.9e-05 | 420.8132 | 420.8130 |
| 5% 10y @ 4.5% | 7.85874322 | 7.85874194 | 1.3e-06 | 74.5506 | 74.5506 |
| 0% 30y @ 5.0% | 29.26833658 | 29.26829268 | 4.4e-05 | 870.9108 | 870.9102 |

That agreement is the licence to use the bump everywhere else. **Use effective duration whenever
the cash flows themselves move with the rate** — callables, floaters, MBS, anything with an
embedded option — because the analytic formula assumes they do not.

## 5. 🚨 A floater has two durations and only one of them is five years

✅ Measured on a **5y quarterly FRN, index 5.00% flat, quoted margin = discount margin = 40 bp**.
The naive comparator, a 5y quarterly fixed bond at 5.40%, has modified duration **4.356304**:

| settled | dirty | **rate duration** | **spread duration** | time to next reset | phantom rate risk |
|---|---|---|---|---|---|
| just after a reset | 100.000000 | **0.246670** | **4.356304** | 0.250000 | 🚨 **41,096** |
| half a quarter in | 100.670474 | 0.124162 | 4.233796 | 0.125000 | 🚨 **42,321** |
| one day before reset | 101.336320 | 0.002500 | 4.112134 | 0.002500 | 🚨 **43,538** |

- ✅ **Rate duration tracks the time to the next reset**, not the maturity: 0.246670 against
  0.250000 years just after a reset, 0.002500 the day before the next one. The coupon reprices
  with the curve, so there is almost nothing left to hedge.
- ✅ **Spread duration runs to maturity — 4.356304 years — and on a reset date at par it equals
  the modified duration of the equivalent fixed bond to 1.9e-07** — the residual is the
  central-difference bump, not the model. That identity is the check that the spread bump is
  doing what you think.
- 🚨 **Booking the FRN as a 5-year fixed bond invents about 41,000 per 1,000,000 per 100 bp of
  rate risk the instrument does not have — and hides the spread risk it does.** The two errors
  do not offset; they are risks to different factors.

## 6. 🚨 QuantLib: three `Duration` enum values, two numbers, and one silent 28% error

✅ Measured, 10y 5% semiannual at y = 5.00%:

| `InterestRate` compounding | `Duration.Macaulay` | `Duration.Modified` | `Duration.Simple` | convexity | `basisPointValue` |
|---|---|---|---|---|---|
| **Compounded** | 7.98944567 | **7.79458114** | 7.98944567 | 73.628731 | −0.07794544 |
| Continuous | 🔴 **raises** | 7.98357996 | 7.98357996 | 73.289812 | −0.07944441 |
| **Simple** | 🔴 **raises** | 🚨 **5.56346580** | 🚨 **8.09004438** | 🚨 **68.602574** | 🚨 **−0.05563432** |

- ✅ Under `Compounded` my closed forms match QuantLib: **|Macaulay diff| 4.4e-15, |Modified diff|
  4.4e-15, |convexity diff| 4.3e-14.**
- ✅ **`Duration.Simple` == `Duration.Macaulay`** (|diff| 8.9e-16). Three enum values, two
  numbers. The one you want is `Modified`.
- ✅ `BondFunctions.duration` **raises `compounded rate required`** — but **only for
  `Duration.Macaulay`.**
- 🚨 **`Duration.Modified` on a `Simple` `InterestRate` returns 5.56346580 instead of 7.79458114
  — 28.6% low, silently.** So does `basisPointValue` (−0.05563 vs −0.07795) and `convexity`
  (68.60 vs 73.63). **The guard exists and does not cover the case you will actually hit**, since
  `Modified` is what a DV01 calls. Build the `InterestRate` with `ql.Compounded` explicitly.
- ⚠️ **`basisPointValue` is a signed price CHANGE (−0.07794544 per 100 = −779.45 per 1,000,000)**;
  a DV01 is conventionally quoted positive. My DV01 is 779.46 — the 0.01 is the convexity term
  QuantLib includes.

## 7. What the script gives you

`scripts/duration.py` — numpy only at import; QuantLib inside one function.

| Function | Does |
|---|---|
| `macaulay_duration` / `modified_duration` / `duration(kind=...)` | §1 |
| `convexity(coupon, y, n, freq)` | `(1/P)·d²P/dy²`, years² |
| `price_change(coupon, y, n, dy)` | §2 — exact, duration-only, duration+convexity, and both errors |
| `dv01(..., accrued=)` / `pv01(...)` | §3; pass `accrued` to see the clean-price error |
| `effective_duration(price_fn, y, bump)` / `effective_convexity` | §4 — anything that reprices |
| `frn_price(index, quoted_margin, discount_margin, n, w, current_coupon)` | the flat-forward floater |
| `frn_durations(...)` | §5 — rate duration, spread duration, time to reset |
| `quantlib_cross_checks(coupon, y, years)` | §6, including which calls raise, or `None` |

## Where this sits

- `../yield-measures-and-bill-quotes/SKILL.md` — the yield every number here is a derivative
  with respect to, and 🚨 why a bill discount rate is not one.
- `../bond-conventions-and-accrued/SKILL.md` — the accrued that makes §3's dirty price.
- `../ois-discounting-and-multi-curve/SKILL.md` — 🚨 the swap analogue, where discounting at the
  wrong curve leaves the par rate unchanged and moves the annuity 1.461%, so every PV01 is wrong
  by that factor.
- `../sofr-and-rfr-compounding/SKILL.md` — what a floating coupon actually is, before you take
  its derivative.
- `../../../fin-core/skills/portfolio-and-risk/SKILL.md` — aggregating these into a portfolio
  risk number.
- `../../../fin-models/skills/term-structure-models/SKILL.md` — key-rate and bucketed risk need
  a curve, and a curve is a different object from a yield.
- `../../../fin-libraries/skills/lib-quantlib/SKILL.md` — `BondFunctions` needs `evaluationDate`
  set; a stale one gives NPV exactly 0.0.
