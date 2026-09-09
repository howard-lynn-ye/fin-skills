---
name: bond-conventions-and-accrued
description: >-
  Compute accrued interest, clean and dirty prices and day-count year fractions on a bond
  without silently picking the wrong convention. TRIGGER - accrued interest, day count,
  daycount, ACT/ACT ICMA vs ISMA vs ISDA, ACT/365F, ACT/360, 30/360, 30E/360, 30E/360 ISDA,
  Thirty360 BondBasis vs USA vs European vs NASD, year fraction, yearFraction,
  ActualActual(ISMA), "my accrued interest is off by a few hundred dollars", "which 30/360 is
  this", clean price vs dirty price vs invoice price, settlement amount, T+1 settlement,
  quasi-coupon date, first and last stub period. SKIP for negative accrued inside a gilt
  ex-dividend window (ex-dividend-and-rebate-interest), for turning a price into a yield
  (yield-measures-and-bill-quotes), for duration and DV01 (duration-convexity-and-dv01), for
  discount-curve conventions and compounding (../../../fin-models/skills/term-structure-models),
  and for QuantLib's evaluationDate global (../../../fin-libraries/skills/lib-quantlib).
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-09"
---

# Bond conventions and accrued

**Accrued interest is not a number. It is a number plus three things: a day-count convention, a
coupon schedule, and a settlement date.** Drop any one and you still get a plausible answer.

Every figure below is printed by `scripts/conventions.py` (runs in **0.6 s**; QuantLib optional,
imported inside `quantlib_cross_checks`). ✅ Measured means this file produced it on 2026-09-09
with QuantLib 1.43, Python 3.11.3. The worked bond is a **5% semiannual, coupon period
2026-01-15 → 2026-07-15 (181 days), settled 2026-04-30 (105 days in), 1,000,000 face**.

> **The rule: carry (day count, coupon schedule, settlement date) together.** ACT/ACT ICMA
> evaluated from two bare dates is a guess, "30/360" alone does not name a convention, and the
> screen quotes clean while the wire settles dirty.

## 1. 🚨 ACT/ACT ICMA without its schedule returns 0.25 and no error

ACT/ACT ICMA (= ISMA = "Bond") is `days / (coupon-period days × coupons per year)`. **The
denominator is the coupon period, not the year**, so two dates are not enough to evaluate it.

✅ Source-verified in QuantLib's own `ql/time/daycounters/actualactual.cpp` (read 2026-09-09):
`implementation()` dispatches `case ISMA: case Bond: if (!schedule.empty())` → `ISMA_Impl`,
**`else` → `Old_ISMA_Impl`** — whose comment says the reference period is taken "equal to
(d1,d2)" when unspecified, and which then estimates
`months = lround(12 * (refPeriodEnd - refPeriodStart) / 365)`.

✅ Measured on the worked bond:

| | year fraction | accrued /100 | cash on 1,000,000 |
|---|---|---|---|
| `ActualActual(ISMA, schedule)` | **0.29005525** | 1.450276 | **14,502.76** |
| `ActualActual(ISMA)` — no schedule | 🚨 **0.25000000** | 1.250000 | **12,500.00** |
| | | | 🚨 **−2,002.76** |

**Why exactly 0.25:** the 105-day stub gives `lround(12 × 105/365) = 3` months, so it decides
this is a **quarterly** bond and returns one whole quarterly period. ✅ My reference
implementation reproduces both numbers to **0.0e+00**.

### 🚨 Where the fallback bites, and where it does not — ✅ measured, and this matters

| call | schedule-free ISMA | correct | verdict |
|---|---|---|---|
| `FixedRateBond.accruedAmount()` | 1.45027624 | 1.45027624 | ✅ **safe** — the bond passes its own coupon period in |
| `BondFunctions.duration(Modified)` | 7.50983567 | 7.50983567 | ✅ **safe** |
| `dayCounter.yearFraction(d1, d2)` | 0.25000000 | 0.29005525 | 🚨 **+2,002.76 of accrued** |
| `InterestRate.compoundFactor(d1, d2)` | 1.0124228366 | 1.0143081035 | 🚨 **+18.6 bp of price** |
| `FlatForward(..., ActualActual(ISMA))` | t = 0.25000000, DF 0.9877295966 | t = 0.28767123, DF 0.9858937305 | 🚨 **+18.6 bp** |

**So the danger is not QuantLib's bond objects — it is your own accrual arithmetic and any curve
or rate object you hand the counter to.** If you write `dc.yearFraction(prev, settle)` yourself,
pass the schedule: `ql.ActualActual(ql.ActualActual.ISMA, schedule)`. **Never use an ACT/ACT ICMA
counter as a curve day count** — a curve's time axis has no coupon period, so the guess is
unavoidable there.

## 2. 🚨 "30/360" does not name a convention

✅ Source-verified in `ql/time/daycounters/thirty360.cpp` (read 2026-09-09): **six
implementations behind nine enum constants** — `US_Impl` (USA), `ISMA_Impl` (ISMA **and
BondBasis**), `EU_Impl` (European, EurobondBasis), `IT_Impl` (Italian), `ISDA_Impl` (ISDA,
German), `NASD_Impl` (NASD). The rules differ **only** on the 31st and on the last day of
February, which is why they agree on most date pairs and then disagree on yours.

✅ Measured, day counts and the accrued spread on 1,000,000 face at a 5% coupon:

| date pair | US | BondBasis | 30E/360 | 30E/360 ISDA | Italian | distinct | spread |
|---|---|---|---|---|---|---|---|
| 2026-01-15 → 2026-07-31 | 196 | 196 | 195 | 195 | 195 | 2 | **138.89** |
| **2026-02-28 → 2026-08-31** | **180** | **183** | **182** | **180** | **180** | 🚨 **3** | 🚨 **416.67** |
| 2028-02-29 → 2028-08-31 | 180 | 182 | 181 | 180 | 180 | 3 | **277.78** |

🚨 **`Thirty360.USA` is not `Thirty360.BondBasis`.** ✅ QuantLib 1.43 on the middle row returns
`USA=180, BondBasis=183, European=182, EurobondBasis=182, Italian=180, German=180, ISMA=183,
ISDA=180, NASD=183` — three answers from nine names. My implementation reproduces all five rules
exactly.

**The rules, from the source:**

- **US** — if *both* dates are the last of February, `dd2 = 30`; if `d1` is the last of February,
  `dd1 = 30`; then `dd2 = 30` if `dd2 == 31 and dd1 >= 30`; then `dd1 = 30` if `dd1 == 31`.
- **BondBasis / ISMA** — the same *without* the two February lines. This is the usual US
  corporate/agency "30/360".
- **30E/360** — `min(dd, 30)` on both legs. February is untouched, which is why 2026-02-28 stays
  28 and the count is 182.
- **30E/360 ISDA** — last day of the month → 30 on both legs, **except** an end date that is the
  *termination* date and falls in February. It needs the maturity to be evaluated at all.
- **Italian** — European, plus a February date after the 27th → 30.
- ✅ **NASD is algebraically identical to BondBasis**: it maps `dd2 = 1; mm2++` where BondBasis
  leaves 31, and `30(m+1) + 1 = 30m + 31`.

## 3. The same bond, every convention — ✅ measured

| convention | year fraction | accrued /100 | cash | vs ICMA |
|---|---|---|---|---|
| **ACT/ACT ICMA** | 0.29005525 | 1.450276 | **14,502.76** | base |
| ACT/ACT ISDA | 0.28767123 | 1.438356 | 14,383.56 | **−119.20** |
| ACT/365F | 0.28767123 | 1.438356 | 14,383.56 | **−119.20** |
| ACT/360 | 0.29166667 | 1.458333 | 14,583.33 | **+80.57** |
| every 30/360 flavour | 0.29166667 | 1.458333 | 14,583.33 | **+80.57** |
| 🚨 ICMA, no schedule | 0.25000000 | 1.250000 | 12,500.00 | 🚨 **−2,002.76** |

✅ QuantLib's `FixedRateBond.accruedAmount` agrees with the reference implementation on all five
counters to **|diff| < 1e-14**. Note **ACT/ACT ISDA and ACT/365F coincide on this period** — they
are different conventions that happen to agree inside a single non-leap year, which is exactly how
a mismatch survives testing.

**ACT/ACT ISDA is not ACT/ACT ICMA.** ISDA splits the interval at each year end and divides each
piece by that year's own length (365 or 366); ICMA divides by the coupon period. They are used for
different instruments — ICMA for bonds, ISDA for swap legs — and swapping them is a silent
119.20 per 1,000,000 here and larger across a leap year.

## 4. Clean, dirty, and what actually settles

✅ At a clean price of 98.500: **dirty = 98.500 + 1.450276 = 99.950276**, and the cash on
1,000,000 face is **999,502.76**, of which **14,502.76 is accrued**.

- **Clean** (flat) is what screens, indices and most APIs quote.
- **Dirty** (full, invoice) is what settles. `dirty = clean + accrued`.
- 🚨 **A P&L or return series built from clean prices alone drops the coupon accrual.** Over a
  year on a 5% bond that is 5 points of return that never appears.
- **Total return = clean change + accrual + coupons received + reinvestment.** Index vendors do
  this for you; a `yfinance`-style price series does not.

## 5. Settlement lag moves the cash at an unchanged price

✅ Measured on the same bond, same clean price, business-day settlement:

| | settle | accrued /100 | cash | vs T+0 |
|---|---|---|---|---|
| T+0 | 2026-04-30 | 1.450276 | 14,502.76 | — |
| T+1 | 2026-05-01 | 1.464088 | 14,640.88 | **+138.12** |
| T+2 | 2026-05-04 | 1.505525 | 15,055.25 | **+552.49** |
| T+3 | 2026-05-05 | 1.519337 | 15,193.37 | **+690.61** |

**One day of settlement is 138.12 per 1,000,000 — the same order as the entire 30/360 argument in
§2.** T+2 lands on Monday 2026-05-04 and picks up the weekend, which is why the step from T+1 to
T+2 is three days of accrual, not one.

✅ Source-verified at **eCFR 17 CFR 240.15c6-1(a)** (fetched 2026-09-09): a broker-dealer may not
contract for settlement "later than the first business day after the date of the contract" — T+1 —
and the rule **excepts** "an exempted security, a government security, a municipal security,
commercial paper, bankers' acceptances, or commercial bills". 🚨 **So Treasuries are outside the
rule entirely**; their T+1 is market convention, not 15c6-1, and new issues priced after 16:30 ET
get T+2 under 15c6-1(c). Calendars, holidays and business-day conventions live in
`../../../fin-core/skills/us-market-rules/SKILL.md`.

## 6. What the script gives you

`scripts/conventions.py` — standard library only at import; QuantLib inside one function.

| Function | Does |
|---|---|
| `year_fraction(d0, d1, convention, ref_start=, ref_end=, freq=, termination=)` | all nine conventions in `CONVENTIONS` |
| `act_act_icma(d0, d1, ref_start, ref_end, freq)` | the real definition; omit the refs to reproduce the fallback |
| `icma_frequency_guess(d0, d1)` | §1 — the `lround(12·days/365)` months QuantLib infers |
| `act_act_isda(d0, d1)` | the year-by-year ISDA split |
| `thirty_360_days(d0, d1, flavour, termination)` | §2, the five 30/360 rules |
| `accrued_interest(...)` / `dirty_price` / `settlement_amount` | §3 and §4; `use_schedule=False` drops the ICMA reference period |
| `convention_table` / `thirty_360_table` / `settlement_lag_table` | the three tables above |
| `add_business_days(d, n, holidays)` | §5 |
| `quantlib_cross_checks(...)` | every QuantLib figure above, or `None` |

## Where this sits

- `../ex-dividend-and-rebate-interest/SKILL.md` — when accrued goes **negative**. Gilts trade
  ex-dividend seven business days before the coupon, and a QuantLib sign convention makes that
  window silently disappear.
- `../yield-measures-and-bill-quotes/SKILL.md` — price → yield, once the accrued is right.
- `../duration-convexity-and-dv01/SKILL.md` — the risk numbers this arithmetic feeds.
- `../../../fin-models/skills/term-structure-models/SKILL.md` — the **curve** side: one discount
  factor is a different zero rate under every day count and compounding (measured there, −17.68
  bp). Bootstrapping, Nelson-Siegel and the short-rate closed forms live there, not here.
- `../../../fin-libraries/skills/lib-quantlib/SKILL.md` — 🚨 `Settings.instance().evaluationDate`
  is a global and a stale one gives **NPV exactly 0.0**; `ql.Date` is day-first.
- `../../../fin-core/skills/us-market-rules/SKILL.md` — trading calendars, holidays and the
  settlement rules §5 only does the arithmetic for.
