---
name: ex-dividend-and-rebate-interest
description: >-
  Handle bonds that trade ex-dividend, where accrued interest goes negative and the buyer is
  paid rebate interest instead of paying it. TRIGGER - gilt, UK gilt, ex-dividend, ex-div,
  ex-coupon, exCouponPeriod, rebate interest, negative accrued interest, "my accrued interest
  is negative", "accrued should be negative but isn't", seven business days before the coupon,
  quasi-coupon date, DMO formulae, "Formulae for Calculating Gilt Prices from Yields",
  ql.FixedRateBond exCouponPeriod, ql.Period(-7, ql.Days), record date vs ex-date on a bond,
  3.5% War Loan, JGB and gilt settlement. SKIP for ordinary positive accrued and day-count
  choice (bond-conventions-and-accrued), for price-to-yield solving in general
  (yield-measures-and-bill-quotes), for index-linked gilt indexation lags
  (../../../fin-models/skills/term-structure-models), and for QuantLib's evaluationDate global
  (../../../fin-libraries/skills/lib-quantlib).
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-09"
---

# Ex-dividend and rebate interest

**A gilt bought inside its ex-dividend window does not receive the coupon that is about to be
paid, so accrued interest is NEGATIVE.** Most fixed-income code has no branch for this. The code
that does can have the sign backwards and never fail.

Every figure below is printed by `scripts/ex_dividend.py` (runs in **0.15 s**; QuantLib optional).
✅ Measured means this file produced it on 2026-09-09 with QuantLib 1.43, Python 3.11.3. The
worked bond is a **4% gilt, coupons 7 Jun / 7 Dec, maturity 2035-06-07, y = 4.20%**, and the
coupon in question is **2026-12-07 (a Monday)**, whose ex-dividend date is **2026-11-26**.

> **The rule: inside the window `AI = (t/s − 1)·d1`, which is negative — and QuantLib's
> `exCouponPeriod` counts days BACK from the coupon, so it takes `Period(7, Days)` and never
> `Period(-7, Days)`.**

## 1. ✅ The rule, from the source

✅ **UK DMO, "Formulae for Calculating Gilt Prices from Yields", 4th edition, 18 December 2024,
Note 5** (read 2026-09-09): the ex-dividend date for all gilts is currently the date **seven
business days** before the dividend date.

✅ Section Three (1)(i), the accrued-interest formula, verbatim in structure:

```
AI = (t/s)     * d1     if the settlement date occurs on or before the ex-dividend date
AI = (t/s - 1) * d1     if the settlement date occurs after the ex-dividend date
```

with `t` = calendar days from the previous quasi-coupon date to settlement, `s` = calendar days
in the full quasi-coupon period, `d1` = the next dividend per £100 nominal. **The `− 1` is the
whole thing**: it subtracts one full coupon.

🔴 **The 3½% War Loan ten-business-day exception is dead.** It was in the 3rd edition
(16 March 2005). ✅ The stock was redeemed in full at par on **2015-03-09** (announced 2014-12-03),
and the 4th edition says seven business days **for all gilts**, with no exception. A model that
recites the War Loan carve-out is quoting a 2005 document about a security that no longer exists.

⚠️ **US Treasuries and USD corporates have no ex-dividend period** — accrued runs straight
through the coupon date — which is exactly why a US-shaped bond library has no code path for
this and a US-trained prior does not expect one.

## 2. ✅ What the window does to accrued, dirty and clean

✅ Measured, per £100 nominal, on 1,000,000 nominal:

| settle | | d1 | accrued | dirty | clean | cash accrued |
|---|---|---|---|---|---|---|
| 2026-11-25 | cum | 2.00 | +1.868852 | 100.445707 | 98.576855 | +18,688.52 |
| 2026-11-26 | cum | 2.00 | **+1.879781** | 100.457115 | 98.577334 | **+18,797.81** |
| **2026-11-27** | **EX** | **0.00** | 🚨 **−0.109290** | **98.470794** | 98.580084 | 🚨 **−1,092.90** |
| 2026-11-30 | EX | 0.00 | −0.076503 | 98.504349 | 98.580851 | −765.03 |
| 2026-12-04 | EX | 0.00 | −0.032787 | 98.549106 | 98.581893 | −327.87 |
| 2026-12-08 | cum | 2.00 | +0.010989 | 98.593945 | 98.582956 | +109.89 |

- ✅ **Accrued falls +1.879781 → −0.109290, a drop of 1.989071 — exactly one coupon (2.00) minus
  one day of accrual.**
- ✅ **The DIRTY price drops −1.986321; the CLEAN price moves +0.002750.** That is the whole
  point of quoting clean: the clean price is continuous across the ex-dividend date and the
  dirty price is not. A model that watches dirty prices sees a 2-point crash that never happened.
- ✅ In the DMO price formula the same fact appears as **`d1 = 0`** — the next quasi-coupon cash
  flow simply is not receivable by the buyer.

## 3. 🚨 `ql.Period(-7, ql.Days)` is accepted and does the opposite

✅ Source-verified in QuantLib's `ql/cashflows/fixedratecoupon.cpp` (`FixedRateLeg::operator
Leg()`, read 2026-09-09):

```cpp
if (exCouponPeriod_ != Period())
{
    exCouponDate = exCouponCalendar_.advance(paymentDate,
                                             -exCouponPeriod_,
                                             exCouponAdjustment_,
                                             exCouponEndOfMonth_);
}
```

**`exCouponPeriod` is already a look-BACK, so QuantLib negates it.** Passing a negative period
double-negates into a forward advance. ✅ Measured on the 2026-12-07 coupon:

| you write | ex-coupon date | |
|---|---|---|
| `ql.Period(-7, ql.Days)` | 🚨 **2026-12-16** | **+9 calendar days — AFTER the coupon** |
| `ql.Period(7, ql.Days)` | 2026-11-26 | −11 calendar days = 7 business days |
| `ql.Period(6, ql.Days)` | 2026-11-27 | reproduces the DMO's cum/ex boundary (§4) |

✅ Accrued per 100 across the window, and the error on 1,000,000 nominal:

| settle | no ex-coupon | 🚨 `Period(-7)` | `Period(7)` | DMO formula | error |
|---|---|---|---|---|---|
| 2026-11-25 | +1.868852 | +1.868852 | +1.868852 | +1.868852 | 0.00 |
| 2026-11-26 | +1.879781 | +1.879781 | −0.120219 | +1.879781 | 🚨 **20,000.00** |
| **2026-11-27** | +1.890710 | 🚨 **+1.890710** | **−0.109290** | −0.109290 | 🚨 **20,000.00** |
| 2026-11-30 | +1.923497 | +1.923497 | −0.076503 | −0.076503 | 🚨 **20,000.00** |
| 2026-12-04 | +1.967213 | +1.967213 | −0.032787 | −0.032787 | 🚨 **20,000.00** |

🚨 **The `Period(-7)` column is character-for-character the no-ex-coupon column, on every date.**
You wrote the ex-dividend handling, it compiled, it ran, and it did nothing. The error is
**exactly one coupon — 2.000000 per 100, 20,000 on 1,000,000 nominal** — on every settlement date
in the window, and it does not shrink with the size of the window.

**How to catch it in one line:** after building the bond, assert the ex-coupon date is *before*
the coupon date.

```python
c = ql.as_coupon(bond.cashflows()[i])
assert c.exCouponDate() < c.date(), "exCouponPeriod sign is wrong"
```

## 4. ⚠️ The boundary is one day wide

✅ Measured: on the ex-dividend date itself (2026-11-26) QuantLib with `Period(7, Days)` is
already **ex** (−0.120219) while the DMO formula is still **cum** (+1.879781). QuantLib's rule is
`exCouponDate <= settlement → ex`; the DMO's is "settlement **after** the ex-dividend date".

⚠️ **They reconcile through the settlement lag, not through the arithmetic.** The DMO formula is
written in *settlement* dates; "trades ex-dividend" is a *trade*-date idea. With gilts settling
T+1, a trade on the ex-dividend date settles the next business day and is therefore ex under both.
**If you feed QuantLib settlement dates, `Period(7, Days)` goes ex one settlement day earlier than
the DMO formula; `Period(6, Days)` reproduces the DMO boundary exactly** (measured: it matches the
DMO column on all seven dates above). Decide which date type your pipeline carries and write the
assertion for it.

## 5. ✅ The DMO price formula, checked against QuantLib

`gilt_price_from_yield` implements DMO Section One for conventional gilts (n ≥ 1):

```
P = v^(r/s) [ d1 + d2 v + (c/f) v^2 (1 - v^(n-1)) / (1 - v) + 100 v^n ],   v = 1/(1 + y/f)
```

✅ Against `ql.BondFunctions.cleanPrice` on the same bond at y = 4.20%, over all seven settlement
dates: **worst |diff| = 4.3e-14** on both clean and dirty. Two independent implementations of the
same pricer, so neither is confirming itself.

🚨 **But only with `paymentConvention = Unadjusted`.** ✅ The DMO states cash flows falling on
non-business days are not adjusted ("not 'bumped'"). QuantLib's usual `ql.Following` moves the
2030-12-07 and 2031-06-07 coupons off their weekends and shifts the clean price by **−0.000947 per
100 = −9.47 per 1,000,000**. Small, but it is a systematic sign-consistent error that never
appears as a failure — and it is the reason a "QuantLib does not match the DMO" investigation
usually ends in the wrong place.

## 6. What the script gives you

`scripts/ex_dividend.py` — standard library only at import; QuantLib inside one function.

| Function | Does |
|---|---|
| `ex_dividend_date(coupon_date, business_days=7, holidays)` | §1; raises if you pass a negative count |
| `dmo_accrued(coupon, settle, maturity, boundary=...)` | §2, the DMO Section Three formula |
| `is_ex_dividend(settle, ex_div, boundary)` | §4 — `"dmo"` or `"quantlib"` |
| `gilt_price_from_yield(coupon, y, settle, maturity, ...)` | §5, DMO Section One; returns dirty, clean, d1, n, r, s |
| `quasi_coupon_dates` / `surrounding_quasi_period` | the quasi-coupon cycle off the maturity date |
| `ex_dividend_window(...)` | the §2 table |
| `quantlib_ex_coupon_signs(...)` | §3 and §5, all four QuantLib bonds, or `None` |
| `GILT_EX_DIVIDEND_BUSINESS_DAYS`, `WAR_LOAN_REDEEMED` | the dated constants |

## Where this sits

- `../bond-conventions-and-accrued/SKILL.md` — ordinary (positive) accrued, the day-count
  choice, clean vs dirty, and 🚨 `ActualActual(ISMA)` without a schedule returning 0.25.
- `../yield-measures-and-bill-quotes/SKILL.md` — solving the DMO price formula the other way,
  and the quotes that are not yields at all.
- `../duration-convexity-and-dv01/SKILL.md` — the risk of a bond whose next cash flow is zero.
- `../../../fin-libraries/skills/lib-quantlib/SKILL.md` — 🚨 `Settings.instance().evaluationDate`
  is a global; every `accruedAmount` call above sets it first. `ql.Date` is day-first, which is
  its own way to get 2026-12-07 wrong.
- `../../../fin-core/skills/us-market-rules/SKILL.md` — the settlement-lag and calendar side of
  §4.
- `../../../fin-models/skills/term-structure-models/SKILL.md` — curve construction; the
  index-linked gilt formulae in the same DMO document (8-month and 3-month indexation lags)
  belong with inflation modelling, not here.
