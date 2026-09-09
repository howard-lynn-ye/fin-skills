---
name: sofr-and-rfr-compounding
description: >-
  Compute a compounded-in-arrears overnight rate correctly - SOFR, SONIA, ESTR, TONA, SARON -
  including the lookback, lockout and observation-shift conventions. TRIGGER - SOFR compounded
  in arrears, SOFR Index, SOFR Averages, 30-day 90-day 180-day SOFR average, compounded RFR,
  daily compounding of an overnight rate, lookback, rate shift, observation shift, lockout,
  payment delay, "my SOFR coupon is a few basis points off", "compounded vs simple average
  SOFR", ACT/360 vs ACT/365 on SONIA, SONIA Compounded Index, ESTR, TONA, SARON,
  OvernightIndexedCoupon, RateAveraging.Compound, applyObservationShift, SOFRINDEX,
  SOFR30DAYAVG. SKIP for what a LIBOR contract falls back TO and the statutory spreads
  (libor-transition-and-fallbacks), for building an OIS discount curve
  (ois-discounting-and-multi-curve), for a floater's duration (duration-convexity-and-dv01),
  and for bond accrued interest and day counts (bond-conventions-and-accrued).
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-09"
---

# SOFR and RFR compounding

**A compounded overnight rate is not an average of the fixings.** It is a product of daily
accrual factors, each weighted by the number of **calendar** days that fixing applies for, on a
360-day year for SOFR and a 365-day year for SONIA — and then a lookback, lockout or observation
shift changes which fixings are in the window at all.

Every figure below is printed by `scripts/sofr.py` (runs in **0.3 s**; QuantLib optional).
✅ Measured means this file produced it on 2026-09-09 with QuantLib 1.43, numpy 2.2.6,
Python 3.11.3. The worked coupon is **2026-04-01 → 2026-07-01 (62 business-day fixings, 91
calendar days), 100,000,000 notional**, on a seeded synthetic SOFR path around 4.30% with
month-end spikes and a 25 bp policy step on 2026-06-18.

> **The rule: a compounded RFR is `prod(1 + r_i × d_i / basis)`, weighted by CALENDAR days —
> ACT/360 for SOFR and ESTR, ACT/365 for SONIA and TONA — and "lookback" names two different
> windows until you say whether it shifts.**

## 1. ✅ The NY Fed's own SOFR Index example, reproduced exactly

✅ Source-verified at the **NY Fed, "Additional Information about Reference Rates Administered by
the New York Fed"** (read 2026-09-09): the SOFR Index starts at **1.00000000 on 2018-04-02** and
compounds by `(1 + SOFR × d/360)` where `d` is the number of **calendar days applicable**.

| value date | SOFR | calendar days | NY Fed index | this file | \|diff\| |
|---|---|---|---|---|---|
| 2018-04-02 | 1.80% | 1 | 1.00005000 | 1.00005000 | **0.0e+00** |
| 2018-04-03 | 1.83% | 1 | 1.00010084 | 1.00010084 | **0.0e+00** |
| 2018-04-04 | 1.74% | 1 | 1.00014917 | 1.00014917 | **0.0e+00** |
| 2018-04-05 | 1.75% | 1 | 1.00019779 | 1.00019779 | **0.0e+00** |
| **2018-04-06 (Fri)** | 1.75% | **3** | 1.00034365 | 1.00034365 | **0.0e+00** |

🔑 **The last row is the whole point: a Friday fixing carries three calendar days.** ✅ The NY Fed
also states the weighting around a holiday: if the start date is a Wednesday and Thursday is a
holiday, that Wednesday's rate applies for `2/360`, and the following Friday's for `3/360`.

## 2. 🚨 Four ways to be a few basis points wrong

✅ Measured on the worked coupon:

| method | rate | vs correct | per 100mm/quarter |
|---|---|---|---|
| **compounded, day-weighted, ACT/360 (correct)** | **4.352685%** | — | — |
| simple day-weighted average instead of compounding | 4.329451% | 🚨 **−2.32 bp** | **−5,873** |
| compounded but with **unweighted** fixings | 4.341373% | 🚨 **−1.13 bp** | −2,859 |
| plain mean of the fixings (no weights, no compounding) | 4.325484% | 🚨 **−2.72 bp** | −6,876 |
| ACT/365 used **consistently** | 4.352366% | **−0.03 bp** | −81 |
| 🚨 **ACT/360 accrual annualized on 365** | **4.413139%** | 🚨 **+6.05 bp** | 🚨 **+15,281** |

🔑 **The basis appears TWICE** — once inside every daily factor `(1 + r·d/basis)` and again in the
annualization `× basis/D`. ✅ **Changed in both places it nearly cancels (−0.03 bp); changed in one
it is the full 365/360 (+6.05 bp).** That is why "we use ACT/365, it is only 1.4%" is either
harmless or a 6 bp error depending on a line you did not look at.

🚨 **Compounding vs simple averaging is not a rounding choice.** ARRC and ISDA fallback
conventions compound; the gap is roughly `r²·T/2` and grows with the square of the rate and the
length of the period. At 4.35% over a quarter it is 2.32 bp; it was under 0.01 bp in 2021.

## 3. 🚨 Lookback, lockout and observation shift are different windows

✅ Measured, 5-day conventions on the same coupon:

| convention | fixings | calendar days | rate | vs plain | per 100mm |
|---|---|---|---|---|---|
| plain in-arrears | 62 | 91 | 4.352685% | — | — |
| **5-day lookback (rate shift)** | 62 | 91 | 4.330251% | 🚨 **−2.24 bp** | **−5,671** |
| **5-day lookback + observation shift** | 62 | 91 | 4.333248% | 🚨 **−1.94 bp** | −4,913 |
| 5-day lockout | 62 | 91 | 4.348797% | **−0.39 bp** | −983 |

- 🚨 **The two "5-day lookbacks" differ by 0.30 bp (758 per 100mm) and both are called a 5-day
  lookback.** They use the *same rates*; what differs is the **day weights** — a fixing that is a
  Friday in the observation window can map to a Wednesday in the accrual window, and 3/360
  becomes 1/360.
- 🚨 **The lookback itself is the bigger number: −2.24 bp.** It pushes the 25 bp policy step five
  business days later inside the accrual period, so fewer days of the quarter see the higher rate.
- **Lockout** freezes the last `k` fixings at the last observable one. It is the smallest of the
  three here because it only affects the tail, but it turns the coupon into a step function of
  one fixing.
- **Payment delay** does not change the rate at all — only when the cash moves — which is why it
  is the one convention people get right.

⚠️ The magnitudes depend on where the rate moves sit relative to the window. The invariant is the
*sign structure*, not the size: a lookback pulls the window backwards in time, a lockout freezes
the tail, and an observation shift additionally re-weights.

## 4. ✅ The index is the only thing you need to store

✅ Measured on the same coupon:

| | rate |
|---|---|
| compounded from the 62 fixings | 4.3526852313% |
| from an **unrounded** index | 4.3526852313% (\|diff\| **0.0e+00**) |
| from the **published 8-dp** index | 4.3526848352% (\|diff\| 4.0e-09 = **0.0000 bp**) |

`rate = (I_end / I_start − 1) × basis / days`. **Two index values and a day count reproduce any
tenor**, which is why the NY Fed publishes an index at all. ⚠️ The NY Fed notes that averages
built from the *rounded* index may differ from the published averages in the fifth decimal place
— true, and 4.0e-09 here, i.e. below a hundredth of a basis point.

## 5. Five currencies, two bases

✅ Day counts read out of QuantLib 1.43's own index definitions (`ql.Sofr(...).dayCounter()`):

| rate | ccy | administrator | basis | QuantLib index day count | rate on this path |
|---|---|---|---|---|---|
| SOFR | USD | NY Fed | ACT/360 | Actual/360 | 4.352685% |
| **SONIA** | GBP | Bank of England | **ACT/365** | Actual/365 (Fixed) | 4.352366% |
| ESTR | EUR | ECB | ACT/360 | Actual/360 | 4.352685% |
| **TONA** | JPY | Bank of Japan | **ACT/365** | Actual/365 (Fixed) | 4.352366% |
| SARON | CHF | SIX | ACT/360 | Actual/360 | 4.352685% |

🚨 **The two index families do not even start at the same number.** ✅ The SOFR Index is
**1.00000000 from 2018-04-02** (NY Fed); ✅ the **SONIA Compounded Index is 100.00000000 from
2018-04-23**, is rounded to eight decimal places, carries the previous day's value internally at
18 decimals, and — unlike SOFR's — **is published at 9am on the same London business day** (Bank
of England, "SONIA key features and policies", read 2026-09-09). ✅ SONIA itself is a **trimmed
mean rounded to four decimal places** over the central 50% of the volume-weighted distribution;
✅ SOFR is a **volume-weighted median rounded to the nearest basis point** (NY Fed).

⚠️ **QuantLib's `UnitedStates::SOFR` fixing calendar is not `UnitedStates::GovernmentBond`.**
✅ Measured: they differ in 2026 by **Good Friday, 2026-04-03** — SIFMA recommends a full close and
no SOFR is published. Adding a fixing on that date to `ql.Sofr` raises
`At least one invalid fixing provided`, which is the only place in this file where a convention
mismatch actually errors.

## 6. ✅ Checked against QuantLib on all four window conventions

| | QuantLib `OvernightIndexedCoupon` | this file | \|diff\| |
|---|---|---|---|
| compounded in arrears | 4.35268523% | 4.35268523% | **0.0000 bp** |
| simple average | 4.32945055% | 4.32945055% | **0.0000 bp** |
| 5-day lookback | 4.33025075% | 4.33025075% | **0.0000 bp** |
| 5-day lookback + observation shift | 4.33324779% | 4.33324779% | **0.0000 bp** |

The reference implementation reproduces QuantLib's `lookbackDays` and `applyObservationShift`
exactly, and QuantLib's `RateAveraging.Compound` vs `RateAveraging.Simple` reproduces §2's
+2.32 bp. Two independent implementations, one arithmetic.

## 7. What the script gives you

`scripts/sofr.py` — numpy only at import; QuantLib inside two functions.

| Function | Does |
|---|---|
| `compound_factor(rates, weights, basis)` / `compounded_rate(...)` | §1, §2 — the index arithmetic |
| `calendar_day_weights(dates, period_end)` | the "calendar days applicable" per fixing |
| `simple_average_rate` / `unweighted_average_rate` | §2, the two wrong averages |
| `mixed_basis_rate(rates, weights, accrual_basis, annualization_basis)` | §2 — the basis used twice |
| `sofr_index_path(...)` / `rate_from_index(I0, I1, days, basis)` | §4 |
| `observation_window(start, end, holidays, lookback, lockout, observation_shift)` | §3 — fixing dates and weights |
| `rate_for_window(fixings, window, basis)` | apply a fixing history to a window |
| `synthetic_sofr(..., policy_date, policy_step)` | the seeded path, with spikes and a policy step |
| `quantlib_cross_checks(...)` / `quantlib_index_day_counts()` | §5, §6, or `None` |
| `NYFED_INDEX_EXAMPLE`, `SOFR_CALENDAR_HOLIDAYS_2026`, `RFR_CONVENTIONS` | the dated constants |

## Where this sits

- `../libor-transition-and-fallbacks/SKILL.md` — 🚨 which SOFR a fallback contract actually gets:
  ISDA's compounded Fallback Rate for derivatives, CME Term SOFR for non-consumer cash, and the
  five statutory spreads that are the same for both.
- `../ois-discounting-and-multi-curve/SKILL.md` — turning these fixings and forwards into a
  curve, and 🚨 why discounting at the projection curve leaves the par rate unchanged while
  moving the annuity 1.461%.
- `../duration-convexity-and-dv01/SKILL.md` — a floater's rate duration is the time to its next
  reset; this skill defines what "reset" means.
- `../bond-conventions-and-accrued/SKILL.md` — the ACT/360 vs ACT/365 argument on the bond side,
  and 🚨 ACT/ACT ICMA without its schedule.
- `../../../fin-models/skills/term-structure-models/SKILL.md` — compounding conventions on a zero
  curve, where the same discount factor is six different rates.
- `../../../fin-core/skills/fundamental-and-macro-data/SKILL.md` — FRED `SOFR`, `SOFR30DAYAVG`
  and `SOFRINDEX`, and their revision behaviour.
- `../../../fin-libraries/skills/lib-quantlib/SKILL.md` — `addFixing`, `clearFixings` and the
  global fixing history that survives between objects.
