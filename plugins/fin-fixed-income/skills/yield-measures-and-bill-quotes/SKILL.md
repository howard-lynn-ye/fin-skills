---
name: yield-measures-and-bill-quotes
description: >-
  Turn a bond or bill price into the right yield, and stop treating a discount rate as one.
  TRIGGER - yield to maturity, YTM, current yield, yield to call, yield to worst, YTW, running
  yield, redemption yield; Treasury bill discount rate vs bond-equivalent yield, BEY,
  coupon-equivalent yield, investment rate, "why is the 4-week bill rate different from the
  yield", DTB3 vs DGS3MO, 360 vs 365 on a bill, money-market yield, add-on rate, CD equivalent;
  "my yield does not match Bloomberg", BondFunctions.bondYield, brentq on a bond price,
  callable bond yield. SKIP for accrued interest and day-count choice
  (bond-conventions-and-accrued), for negative accrued in a gilt ex-dividend window
  (ex-dividend-and-rebate-interest), for duration DV01 and convexity
  (duration-convexity-and-dv01), for zero rates and bootstrapping
  (../../../fin-models/skills/term-structure-models), and for compounded RFR averages
  (sofr-and-rfr-compounding).
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-09"
---

# Yield measures and bill quotes

**"5.00%" on a Treasury bill screen is a discount rate. It is not a yield, and it is not
comparable to anything else on the page.** Four numbers describe the same bill and they are all
different.

Every figure below is printed by `scripts/yield_measures.py` (runs in **0.94 s**; QuantLib
optional). ✅ Measured means this file produced it on 2026-09-09 with QuantLib 1.43, scipy 1.13.0,
Python 3.11.3.

> **The rule: a discount rate divides the gain by FACE on a 360-day year; a yield divides it by
> PRICE on a 365-day year. Convert with 31 CFR 356 Appendix B — the simple formula up to one
> half-year, the QUADRATIC beyond it. On a callable, quote yield to worst, not yield to
> maturity.**

## 1. 🚨 The same 5.00% bill, four ways

✅ Measured:

| days | price | discount rate | money-market (add-on) | **bond-equivalent** | **Treasury investment rate** | d → BEY | BEY → IR |
|---|---|---|---|---|---|---|---|
| 28 | 99.611111 | 5.0000% | 5.0195% | 5.0892% | 5.0892% | **+8.9 bp** | +0.0 |
| 91 | 98.736111 | 5.0000% | 5.0640% | **5.1343%** | 5.1343% | **+13.4 bp** | +0.0 |
| 182 | 97.472222 | 5.0000% | 5.1297% | **5.2009%** | 5.2009% | **+20.1 bp** | +0.0 |
| **364** | 94.944444 | 5.0000% | 5.2662% | 5.3394% | 🚨 **5.2701%** | **+33.9 bp** | 🚨 **−6.9 bp** |

Three different denominators are in play:

- **discount rate** = `(100 − P)/100 × 360/days` — the gain over **face**, 360-day year.
- **money-market / add-on / CD-equivalent** = `(100 − P)/P × 360/days` — over **price**, still 360.
- **bond-equivalent yield** = `(100 − P)/P × 365/days` — over **price**, 365-day year.

🚨 **The discount rate is the only one of the three that is not a return on anything you can
invest.** You pay P and receive 100; the return is measured on P.

### The gap grows with the level, not just the tenor — ✅ measured at a fixed 182 days

| discount rate | 1.00% | 3.00% | 5.00% | 8.00% | 12.00% |
|---|---|---|---|---|---|
| bond-equivalent yield | 1.0190% | 3.0885% | 5.2009% | 8.4530% | 12.9524% |
| **gap** | +1.9 bp | +8.9 bp | **+20.1 bp** | **+45.3 bp** | **+95.2 bp** |

**At 12% the two quotes are 95 bp apart on the same instrument.** A conversion tested on a
2020 bill and reused in 2023 is wrong by four times as much.

## 2. ✅ Treasury's own conversion is TWO formulas, and the regulation prints both examples

✅ Source-verified at **eCFR 31 CFR 356 Appendix B, section VI** (fetched 2026-09-09), *Computation
of Purchase Price, Discount Rate, and Investment Rate (Coupon-Equivalent Yield) for Treasury
Bills*:

**B.1 — not more than one half-year to maturity**

```
i = (100 - P)/P * y/r
```

**B.2 — more than one half-year to maturity**

```
P [1 + (r - y/2)(i/y)] (1 + i/2) = 100

solved as a quadratic:  b = r/y,  a = (r/2y) - 0.25,  c = (P - 100)/P
                        i = (-b + sqrt(b^2 - 4ac)) / (2a)
```

with `r` = days to maturity and `y` = 365, **or 366 if the year following the ISSUE date
contains 29 February** — not the year the bill matures in.

🔑 **The quadratic exists because a bill longer than six months has to be made comparable to a
security that pays a coupon halfway through and reinvests it.** That is the `(1 + i/2)` factor.

✅ **The regulation's own two worked examples, reproduced by this file:**

| | P | r | regulation | `treasury_investment_rate` |
|---|---|---|---|---|
| cash-management bill 1990-06-01 → 1990-06-21 | 99.559444 | 20 | **8.076%** | **8.075725% → 8.076%** |
| 52-week bill 1990-06-07 → 1991-06-06 | 92.265000 | 364 | **8.237%** | **8.237324% → 8.237%** |

🚨 **Applying the short formula to that 52-week bill gives 8.406492% — 16.9 bp too high.** The
formulas do not blend at the boundary; there is a branch at `r > y/2` and you have to take it.

⚠️ FRED's `DTB4WK`/`DTB3`/`DTB6`/`DTB1YR` are **discount rates**; `DTB3` is not `DGS3MO`. The
`*_CBOND`-style constant-maturity series are yields. Mixing them inside one time series is the
same 20 bp error moving in and out of the data.

## 3. Current yield ignores the pull to par

✅ Measured — annual coupon over clean price, against the true yield to maturity:

| coupon | clean | years | current yield | YTM | gap |
|---|---|---|---|---|---|
| 3.00% | 85.00 | 5 | 3.5294% | **6.5681%** | 🚨 **+303.9 bp** |
| 3.00% | 85.00 | 20 | 3.5294% | 4.1070% | +57.8 bp |
| 8.00% | 118.00 | 5 | 6.7797% | **3.9930%** | 🚨 **−278.7 bp** |
| 8.00% | 118.00 | 20 | 6.7797% | 6.3927% | −38.7 bp |
| 5.00% | 100.00 | 10 | 5.0000% | 5.0000% | **+0.0** |

**Current yield is exactly right at par and nowhere else**, and it is worst on SHORT bonds away
from par, because that is where the pull to par is spread over the fewest years. It is a carry
number, not a return.

## 4. 🚨 Yield to maturity on a callable above par is a fiction

✅ Measured: a 5% bond, 10 years to maturity, clean 108.00, callable at 102 in 2 years, 101 in
3 years, 100 in 5 years:

| redemption | at | yield |
|---|---|---|
| **102.00** | **2.0y** | 🚨 **1.8909%** ← the worst |
| 101.00 | 3.0y | 2.5366% |
| 100.00 | 5.0y | 3.2534% |
| 100.00 | 10.0y (maturity) | **4.0205%** |

🚨 **Quoting the yield to maturity overstates by 213.0 bp on a bond the issuer will call.** Yield
to worst is the minimum over every redemption date **at that date's own call price** — a call at
102 is not a call at 100, and repricing to par understates the yield.

**Yield to worst is still not the answer**, only the honest floor: it assumes a deterministic
exercise. Once you want the option's value, you need an OAS on a lattice, which lives with
option-adjusted spread work, not here.

## 5. ✅ Checked against QuantLib

| check | mine | QuantLib 1.43 | \|diff\| |
|---|---|---|---|
| 91d bill BEY as a simple ACT/365 rate → compound factor | 1.0128006752 | 1.0128006752 | **0.0e+00** |
| 364d bill, same | 1.0532475132 | 1.0532475132 | **0.0e+00** |
| 10y 5% semiannual clean price at y = 4.50% | 103.9909280925 | 103.9909280925 | **1.6e-13** |
| and back to a yield | 0.0450000000 | 0.0450000000 | **2.1e-17** |

🚨 **The price check only agrees with `paymentConvention = ql.Unadjusted`.** ✅ QuantLib's default
`Following` bumps coupons off weekends and prices the same bond at **103.9870971915, −0.003831
lower** — which reads as a broken yield formula and is not one. Set the convention deliberately.

## 6. What the script gives you

`scripts/yield_measures.py` — scipy (`brentq`) only at import; QuantLib inside one function.

| Function | Does |
|---|---|
| `bill_price_from_discount(d, days)` / `bill_discount_from_price` | §1, the 360-on-face quote |
| `money_market_yield(P, days)` / `bond_equivalent_yield(P, days, year)` | §1, the two add-on rates |
| `treasury_investment_rate(P, r, y=365)` | §2, **both** branches of 31 CFR 356 App. B VI |
| `bill_table(discount, day_counts)` | the §1 table with the bp gaps |
| `bond_price_from_yield(coupon, y, n, w, freq)` | street-convention dirty/clean/accrued, `w` = fraction of the period left |
| `ytm(clean, coupon, n, w, freq)` | §3 and §5, by `brentq` |
| `current_yield(coupon, clean)` | §3 |
| `yield_to_worst(clean, coupon, calls, n_maturity)` | §4; each call priced at its own redemption |
| `quantlib_cross_checks(...)` | §5, or `None` |

## Where this sits

- `../bond-conventions-and-accrued/SKILL.md` — the accrued and the `w` in `bond_price_from_yield`
  come from there; ACT/ACT ICMA without its schedule is 🚨 $2,002.76 per $1mm.
- `../ex-dividend-and-rebate-interest/SKILL.md` — the UK DMO price/yield formula, and what a
  yield means when the next cash flow is zero.
- `../duration-convexity-and-dv01/SKILL.md` — what to do with the yield once you have it, and
  which duration a library actually returned.
- `../sofr-and-rfr-compounding/SKILL.md` — the other family of quotes that are not yields:
  compounded-in-arrears averages and the SOFR Index.
- `../../../fin-models/skills/term-structure-models/SKILL.md` — a YIELD is one number for one
  bond; a zero curve is a different object, and the same discount factor is six different zero
  rates there.
- `../../../fin-core/skills/fundamental-and-macro-data/SKILL.md` — where `DTB3`, `DGS3MO` and the
  Treasury par curve come from, and their revision behaviour.
- `../../../fin-libraries/skills/lib-quantlib/SKILL.md` — `BondFunctions.bondYield` needs a
  `ql.BondPrice`, and `evaluationDate` is a global.
