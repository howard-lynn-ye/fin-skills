---
name: cds-mechanics-and-upfront
description: >-
  Turn a CDS quote into the cash that actually changes hands - standard coupons, points upfront,
  the risky annuity, the IMM roll and the accrual rebate. TRIGGER - points upfront, upfront
  payment on a CDS, convert spread to upfront, conventional spread, quoted spread vs par spread,
  ISDA CDS Standard Model, cdsmodel.com, CDS converter, flat hazard quoting convention, RPV01,
  risky PV01, risky annuity, "upfront = spread difference times duration", 100bp or 500bp
  coupon, SNAC, IMM dates, CDS roll, 20 Mar/Jun/Sep/Dec, "why is my 5y CDS maturing in June",
  accrual on default, accrual rebate, accrued coupon on a CDS,
  cdsMaturity, IsdaCdsEngine, MidPointCdsEngine, ISDA standard rate curve, rfr.spglobal.com.
  SKIP for hazard-rate and default-probability modelling, Merton and recovery sensitivity
  (credit-risk-models), for bond Z-spreads, G-spreads and OAS (credit-spread-measures), for the
  bond tape and TRACE (corporate-bond-data-and-trace), and for building the discount curve
  itself (term-structure-models).
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-09"
---

# CDS mechanics and upfront

**A single-name CDS trades on a fixed coupon with an upfront payment, so the quoted "spread" is
a quoting device, not a cash flow.** Two steps convert one to the other and both have a trap in
them: `upfront = (quoted spread − coupon) × RPV01`, where **RPV01 is the risky annuity in years,
not the tenor** — and the **tenor is not the tenor either**, because "5Y" is a 20 June or
20 December date somewhere between 4.75 and 5.25 years away.

Measured figures below are printed by `scripts/cds.py` (runs in **1.0 s**; scipy plus pure
python dates, QuantLib optional). ✅ Measured means this file produced it on 2026-09-09 with
QuantLib 1.43, scipy 1.13.0, Python 3.11.3. Conventions are ✅ source-verified against ISDA's own
documents at cdsmodel.com, read 2026-09-09.

> **The rule:** `upfront = (quoted spread − coupon) × RPV01`, and **RPV01 is the risky annuity**.
> Never multiply by the tenor, and never trust the label — read the maturity **date**.

## 1. ✅ ISDA's own worked example, reproduced exactly

✅ **Source:** ISDA, *Standard CDS Examples* (April 2009) and *Standard North American Corporate
CDS Contract Specification* (version 4 March 2009), both published at cdsmodel.com. A 1-year
$36mm 100 bp standard CDS traded in Feb-09 maturing 20 Mar 10. ✅ `isda_worked_example()`
rebuilds the schedule from the conventions and matches every published figure:

| # | accrual start | accrual end | days (ISDA) | payment (ISDA) | payment date |
|---|---|---|---|---|---|
| 1 | 2008-12-22 | 2009-03-20 | **88** (88) | **$88,000** ($88,000) | 2009-03-20 |
| 2 | 2009-03-20 | 2009-06-22 | **94** (94) | **$94,000** ($94,000) | 2009-06-22 |
| 3 | 2009-06-22 | 2009-09-21 | **91** (91) | **$91,000** ($91,000) | 2009-09-21 |
| 4 | 2009-09-21 | 2009-12-21 | **91** (91) | **$91,000** ($91,000) | 2009-12-21 |
| 5 | 2009-12-21 | 2010-03-20 | **90** (90) | **$90,000** ($90,000) | 2010-03-22 |

✅ And the cash, quoted at **2 points upfront**: buyer pays clean **$720,000**, seller pays
**$61,000** of accrued over **61** riskless days, **net the buyer pays $659,000** — ISDA's
published $720k / $61k / $659k to the dollar.

**Three conventions do all the work in that table**, and each is a way to be a day out:

- ✅ *"Maturity Date: A CDS Date, unadjusted"* — row 5 accrues to Saturday 20 Mar 10 and **pays**
  on Monday 22 Mar 10. Accrual dates are adjusted **Following**; the last one is not.
- ✅ The last accrual period **includes** the maturity date, so its 90 days is one more than the
  difference of its endpoints. Every other period is end-exclusive.
- ✅ *"Accrual Begin Date: latest Adjusted CDS Date on or before T+1 calendar"* — the **adjusted**
  date. On 9 Sep 2026 that is **22 Jun 2026**, not 20 Jun, and the two differ by 2 days of
  accrued.

## 2. 🚨 The trap: RPV01 is the risky annuity, not the tenor

✅ Measured on a standard contract traded **2026-09-09**, quoted **200 bp** against a **100 bp**
coupon, R = 40%, flat 3% curve:

| | ✅ value |
|---|---|
| standard 5Y maturity | **2031-06-20** (accrual begins 2026-06-22) |
| protection actually bought | **4.7808 years**, not 5.0000 |
| implied flat hazard (the *quoting* convention) | 3.546655% |
| **risky annuity RPV01** | **4.367676 years** |
| clean upfront `(S − C) × RPV01` | **4.367676 points** |
| less accrued coupon (80 riskless days) | 0.222222 points |
| **= cash settlement amount, paid by the buyer** | **4.145454 points** |
| 🚨 `(S − C) × tenor` | **5.000000 points** — **+14.48%** |

🚨 **$144,800 too much on $10mm, from an arithmetic shortcut that looks like it cannot be
wrong.**

### ✅ Where the years go

The label and the multiplier are separated by four independent steps, ✅ each measured:

| step | years |
|---|---|
| the label says | 5.0000 |
| calendar life of the standard contract | **4.7808** |
| ACT/360 accrual on the real IMM schedule | 5.0694 |
| after discounting at 3% | 4.7205 |
| **after survival at the implied hazard** | **4.3493** |
| plus accrual on default | 0.0184 |
| **= RPV01** | **4.3677** |

**Discounting and survival are the two that matter**, and survival is the bigger one. The ACT/360
accrual is *longer* than five calendar years (5.0694), which is why "it is about five, so multiply
by five" survives a sanity check.

### 🚨 The error grows with the tenor and with the spread

Both directions are the same mechanism — more years, or a lower survival probability, means a
larger gap between the annuity and the tenor. ✅ Measured:

| tenor | maturity | RPV01 | vs tenor | 🚨 error of `(S−C) × tenor` |
|---|---|---|---|---|
| 1Y | 2027-06-20 | 0.9857 | −1.43% | **+1.45%** |
| 3Y | 2029-06-20 | 2.7857 | −7.14% | +7.69% |
| **5Y** | 2031-06-20 | **4.3677** | **−12.65%** | **+14.48%** |
| 7Y | 2033-06-20 | 5.7618 | −17.69% | +21.49% |
| 10Y | 2036-06-20 | **7.5447** | **−24.55%** | **+32.54%** |

| quoted spread | implied flat hazard | RPV01 | 🚨 error |
|---|---|---|---|
| 50 bp | 0.8841% | 4.6288 | +8.02% |
| **100 bp** (= the coupon) | 1.7698% | 4.5395 | **n/a — upfront is zero** |
| 200 bp | 3.5467% | 4.3677 | +14.48% |
| 500 bp | 8.9220% | 3.9016 | +28.15% |
| **1000 bp** | 18.0504% | **3.2642** | **+53.17%** |

🚨 **On a distressed name the shortcut is 53% wrong** — exactly where the upfront is largest and
the trade is most likely to be done in a hurry. ✅ The single point where it is right is the one
place there is nothing to compute: when the quote equals the coupon and the upfront is zero.

## 3. 🚨 "Five year" is not five years

✅ **Source-verified:** ISDA specifies *"CDS Dates: 20th of Mar/Jun/Sep/Dec"* and *"Maturity
Date: A CDS Date, unadjusted"*. ✅ Under the post-2015 convention — verified against QuantLib
1.43's own `cdsMaturity(..., DateGeneration.CDS2015)` on **4,000 (trade date, tenor) pairs with
0 mismatches** — contracts mature on **20 June or 20 December** and roll on **20 March and
20 September**.

✅ Measured across the September 2026 roll:

| trade date | standard 5Y maturity | protection years |
|---|---|---|
| 2026-09-14 | 2031-06-20 | 4.7671 |
| **2026-09-18** | **2031-06-20** | **4.7562** |
| **2026-09-21** | **2031-12-20** | **5.2493** |
| 2026-09-28 | 2031-12-20 | 5.2301 |

🚨 **The same "5Y" quote steps from 4.7562 to 5.2493 years over one weekend**, and over a full
cycle it ranges **4.7534 to 5.2548 — a spread of 0.5014 years**. A "5Y CDS" position marked
against a "5Y CDS" quote from before the roll is marked against a different contract. **Store the
maturity date, not the tenor label**, and compare quotes only within a trading period.

⚠️ The 2009 ISDA specification above predates the 2015 change; it describes quarterly rolls with
maturities on any CDS date. The 20 Jun / 20 Dec + semiannual-roll rule is verified here against
QuantLib's implementation, not against an ISDA document.

## 4. ✅ The flat hazard is a quoting convention, not a model

✅ **Source, verbatim, ISDA *Standard CDS Examples*:** the converter *"assumes a single flat
hazard rate rather than a term structure of flat spreads"*, *"credit risk begins at the end of
the trade date (T)"*, and *"Points upfront (and clean price) include only the value of risky
days"* — the days from the accrual begin date through T are *riskless* and contribute only to the
coupon.

**That is the whole reason the accrual rebate exists.** The buyer will pay a full quarterly
coupon covering days they did not own protection for, so at settlement the seller hands back the
accrued: ✅ **80 riskless days = 0.222222 points** on the 2026 contract in §2, taking the payment
from 4.367676 clean to **4.145454** cash. 🚨 **Comparing a clean upfront with a cash settlement
amount is a real 22 bp of notional**, and both are called "the upfront".

🚨 **Do not read the flat hazard as a credit view.** It is one number chosen to reproduce one
quote; it has no term structure and it is not the hazard you would estimate. Everything about
hazard-rate models, recovery sensitivity, and risk-neutral versus physical default probability
belongs to `../../../fin-models/skills/credit-risk-models/SKILL.md` and is deliberately not
repeated here.

⚠️ Two more conventions worth knowing and **not** verified beyond the 2009 documents: the
**40% standard recovery** used in the conversion is a market convention, not a measurement; and
cash settles **T+3** (ISDA's example settles a 20 Feb 09 trade on 25 Feb 09).

✅ And a genuinely surprising one, quoted from the contract specification: *"Legal Protection
Effective Date: today −60 days for credit events and today −90 days for succession events"*.
**Protection is retroactive by 60 days.** The T+1 start used in pricing is a *valuation*
convention for events not yet known, not the contract's coverage.

## 5. ✅ The coupon is packaging; the upfront absorbs it

✅ Measured, same 200 bp name on each standard coupon:

| coupon | RPV01 | clean upfront | PV of protection |
|---|---|---|---|
| 100 bp | 4.3677 | **+4.3677 points** | 0.08735352 |
| 500 bp | 4.3677 | **−13.1030 points** | 0.08735352 |

✅ **Identical PV to eight decimals.** The upfront is exactly the PV of the difference between
the quoted spread and whichever coupon was chosen, which is what makes 100 bp and 500 bp quotes
comparable at all — and why a sign flip on the upfront is a 17.5-point error, not a typo.

## 6. 🔴 The ISDA Standard Rate Curve moved

✅ Read at cdsmodel.com on 2026-09-09: the ISDA Standard Rate Curves are now published at
**https://rfr.spglobal.com/**. 🔴 The old **https://rfr.ihsmarkit.com/** was decommissioned on
**2026-08-15**. Any pipeline or notebook that hard-codes the IHS Markit host is dead, and the
failure is a network error at curve-build time, not a wrong number. The model itself is open
source and maintained by S&P Global as administrator.

## 7. ✅ Cross-check against QuantLib 1.43

Same contract, three engines:

| | fair spread | RPV01 | upfront |
|---|---|---|---|
| `MidPointCdsEngine` | 200.0924 bp | 4.365713 | 4.3697 pts |
| `IntegralCdsEngine` | 200.0702 bp | 4.365965 | 4.3690 pts |
| `IsdaCdsEngine` | 200.0598 bp | 4.366370 | 4.3690 pts |
| **this file** | **200.0000 bp** | **4.367676** | **4.3677 pts** |

✅ **Worst disagreement 0.0924 bp of spread and 0.001962 years of RPV01.** ⚠️ Note the three
engines differ from **each other** by **0.0326 bp** on identical inputs — they discretise the
default time differently. **If you are reconciling to a counterparty, the engine is part of the
trade description.**

🚨 `ql.Settings.instance().evaluationDate` is a global; a stale one gives **NPV exactly 0.0**.
Every QuantLib call above sets it before building a curve.

## 8. What the script gives you

`scripts/cds.py` — scipy (`brentq`) and `datetime` only; QuantLib imported inside two functions.

| Function | Does |
|---|---|
| `isda_contract_spec()` / `isda_converter_assumptions()` | §1, §4, the quoted conventions with their source |
| `isda_worked_example()` | §1, ISDA's published table and cash, reproduced |
| `following(d)` / `previous_imm` / `next_imm` / `previous_roll` | the CDS calendar |
| `standard_cds_maturity(trade_date, tenor)` | §3, the CDS2015 rule |
| `accrual_begin(trade_date)` / `accrued_days` | §1, the adjusted-date subtlety |
| `cds_periods(trade_date, maturity)` | the schedule as `Period` records |
| `premium_leg` / `protection_leg` / `par_spread` | closed-form legs, accrual on default included |
| `implied_flat_hazard(quoted_spread, ...)` | §4, the converter convention |
| `upfront(quoted_spread, coupon, ...)` | §2, clean, accrued and cash, plus the naive error |
| `rpv01_decomposition()` | §2, where the years go |
| `annuity_trap(tenors)` / `spread_sensitivity(spreads_bp)` | §2, both directions of the trap |
| `roll_jump()` / `roll_cycle_range()` | §3 |
| `coupon_invariance()` | §5 |
| `quantlib_roll_check()` / `quantlib_cross_check()` | §3, §7, or `None` |

## Where this sits

- `../../../fin-models/skills/credit-risk-models/SKILL.md` — **the hazard rate as a model**:
  Merton, `N(−d2)`, constant hazard, `lambda(1−R)`, recovery sensitivity, and the risk-neutral
  versus physical gap. This skill only uses a flat hazard as a quoting device and defers all of
  that.
- `../credit-spread-measures/SKILL.md` — the other credit spread. A CDS par spread and a bond
  Z-spread are different instruments; the basis between them is a trade, not an error.
- `../corporate-bond-data-and-trace/SKILL.md` — the cash-bond leg of that basis, and why a
  15-minute-old bond print shows up as basis that is not there.
- `../ratings-transitions-and-migration/SKILL.md` — the 100 bp / 500 bp coupon split follows the
  investment-grade / high-yield line, so it moves when the rating does.
- `../../../fin-models/skills/term-structure-models/SKILL.md` — the discount curve every leg
  above is priced on. 🚨 A zero rate is a triple (day count, compounding, instrument); the ISDA
  Standard Rate Curve is a specific one, published daily at the site in §6.
- `../../../fin-libraries/skills/lib-quantlib/SKILL.md` — `cdsMaturity`, `DateGeneration.CDS2015`,
  `FlatHazardRate`, the three CDS engines, and 🚨 the `evaluationDate` global.
- `../../../fin-core/skills/market-data-sourcing/SKILL.md` — ⚠️ single-name CDS quotes and index
  composition are licensed (S&P Global / Markit); there is no free constituent feed. The ISDA
  Standard **Rate** Curve in §6 is free; the credit quotes are not.
