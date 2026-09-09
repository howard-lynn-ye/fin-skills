---
name: credit-spread-measures
description: >-
  Work out which spread a corporate bond quote actually is and what it was measured against, so
  two "spreads" on the same bond stop disagreeing. TRIGGER - Z-spread, I-spread, G-spread,
  benchmark spread, spread to Treasuries, asset swap spread, ASW, par/par asset swap, discount
  margin, DM on a floater, quoted margin, OAS, option-adjusted spread, option cost, static
  spread, zero-volatility spread, "my Z-spread and my G-spread disagree", "is this spread over
  Treasuries or over swaps", "YTM minus the 5-year Treasury", BondFunctions.zSpread, "why is my
  OAS lower than my Z-spread", spread on a callable bond. SKIP for CDS spreads, points upfront
  and the ISDA model (cds-mechanics-and-upfront), for hazard rates, Merton and default
  probability (credit-risk-models), for building, bootstrapping or interpolating the underlying
  curve (term-structure-models), and for where the price and the trade came from
  (corporate-bond-data-and-trace).
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-09"
---

# Credit spread measures

**Six numbers on one bond are all called "the spread", and a quote almost never says which.**
They differ by a *reference curve*, not by a credit view: G against one point of the government
par curve, I against one point of the swap curve, Z against the whole zero curve, ASW against the
swap annuity, DM against a projected index, OAS against a lattice you had to pick a volatility
for. Nothing raises when you compare two of them.

Every figure below is printed by `scripts/spreads.py` (runs in **0.9 s**; QuantLib optional,
imported inside `quantlib_cross_check`). ✅ Measured means this file produced it on 2026-09-09
with QuantLib 1.43, numpy 2.2.6, scipy 1.13.0, Python 3.11.3.

> **The rule:** a spread is a **pair** — the number and the curve it was measured against.
> "YTM minus the government **zero** rate" is not the G-spread, and a **Z-spread on a callable
> bond is not an OAS**.

## 1. ✅ One bond, six spreads

A 5-year 3% annual-coupon corporate on a steep government zero curve
(**3.00 / 3.50 / 4.00 / 4.50 / 5.00%** at 1–5y, annual compounding), priced at a **150 bp
Z-spread**. Swap zeros are the same curve plus a widening 10 → 26 bp swap spread.
✅ Measured: **price 85.700750, YTM 6.434845%**, government 5y **zero 5.000000%** but 5y
**par 4.902625%**, swap 5y par 5.149527%.

| measure | ✅ bp | measured against |
|---|---|---|
| **G-spread** (YTM − govt **par**) | **153.22** | one point of the government par curve |
| 🚨 **"YTM − govt ZERO"** | **143.48** | one point of the government **zero** curve |
| **I-spread** (YTM − par swap) | **128.53** | one point of the par swap curve |
| **Z-spread** over govt zeros | **150.00** | the **whole** government zero curve |
| Z-spread over swap zeros | **124.53** | the whole swap zero curve |
| **ASW** par/par over swap | **110.71** | the swap annuity, amortising 100 − price |

🚨 **The same bond is 110.71 bp and 153.22 bp on the same afternoon — a 42.5 bp range with no
change in credit.** Two of those numbers are wrong only if you mislabel them; one of them
(143.48) is wrong however you label it.

## 2. 🚨 "YTM minus the government zero rate" is not the G-spread

The G-spread is quoted against the **benchmark yield** — a *par* yield, the coupon that prices
the on-the-run bond at 100. A zero rate is a different animal: on an upward-sloping curve the par
yield sits **below** the zero rate of the same maturity, because the par bond's early coupons are
discounted at lower short rates. ✅ Measured, the 5y zero is 5.000000% and the 5y par is
**4.902625%** — 9.74 bp apart on this curve, and the whole error lands in the spread.

✅ **The error is the slope, not the credit.** The 5-year zero is pinned at 5.00% in every row;
only the shape changes:

| slope (bp/yr) | govt 5y par | **G-spread** | 🚨 YTM − zero | error |
|---|---|---|---|---|
| 0 (flat) | 5.0000% | 150.00 bp | 150.00 bp | **+0.00** |
| 25 | 4.9513% | 151.64 bp | 146.77 bp | −4.87 bp |
| **50** | **4.9026%** | **153.22 bp** | **143.48 bp** | **−9.74 bp** |
| 75 | 4.8540% | 154.74 bp | 140.14 bp | −14.60 bp |
| 100 | 4.8055% | 156.20 bp | 136.75 bp | −19.45 bp |

**On a flat curve the two agree exactly**, which is why the mistake survives testing: it is
invisible in the one case people check. The sign is systematic — an upward-sloping curve always
makes the naive number **too small**, so credit looks tighter than it is.

**Also note the G-spread and the Z-spread are not the same number either** — 153.22 vs 150.00,
**−3.22 bp**. Both are correct; G collapses the curve to one point and Z uses all of it. On a
steep curve, or a bond far from par, the two separate. Quote which one you meant.

## 3. ✅ ASW is not the Z-spread over the swap curve

The par/par asset-swap spread is the margin over the floating index that turns a bond into a par
floater. The investor pays **100** for the package while the bond costs **85.70**, and the swap
absorbs the 14.30-point difference — so the ASW **amortises (100 − price) over the swap annuity**
linearly, where the Z-spread shifts a discount curve. ✅ Measured on the same bond, on the same
swap curve: **ASW 110.71 bp vs Z-over-swap 124.53 bp = −13.82 bp.**

**The gap is a function of distance from par, not of credit.** A bond at 100 has ASW ≈ Z; this
one is 14.30 points below par, and the gap is 13.82 bp. 🚨 Comparing a deep-discount bond's ASW
with a near-par bond's Z-spread ranks them wrongly, and both numbers are "the spread over swaps".

`asset_swap_spread(price_dirty, coupon, maturity, swap_zeros)` implements
`(PV_swap(bond flows) − price) / swap annuity`. ⚠️ This is the **par/par** convention; a
market-value asset swap and a proceeds asset swap divide by different notionals and give
different numbers again.

## 4. ✅ Discount margin: the quoted margin is the DM only at par

For a floater the coupon resets, so a Z-spread is replaced by the **discount margin** — the
spread over the projected index that reprices the bond. The **quoted margin** printed on the
security is a coupon formula, not a valuation. ✅ Measured, 5y quarterly FRN, forward index flat
at 4.00%, quoted margin **120 bp**:

| price | quoted | **DM** | DM − quoted | straight-line approx | its error |
|---|---|---|---|---|---|
| 102.00 | 120 bp | **74.83 bp** | −45.17 bp | 80.00 bp | +5.17 bp |
| **100.00** | 120 bp | **120.00 bp** | **+0.00** | 120.00 bp | −0.00 bp |
| 98.50 | 120 bp | **154.56 bp** | +34.56 bp | 150.00 bp | −4.56 bp |
| 95.00 | 120 bp | **237.57 bp** | +117.57 bp | 220.00 bp | −17.57 bp |

✅ **DM equals the quoted margin exactly at 100.00 and nowhere else.** The straight-line
shortcut `margin + (100 − price)/maturity` is 4.56 bp light at 98.50 and **17.57 bp** light at
95.00 — it ignores discounting, so it always understates the DM on a discount bond.

## 5. 🚨 A Z-spread on a callable bond is not an OAS — the gap IS the option cost

A Z-spread discounts the bond's **promised** cash flows. A callable bond will not pay them, and
its price is depressed by the call, so the Z-spread absorbs the option value and reports it as
credit. The OAS puts the cash flows on a calibrated short-rate lattice, exercises the call
optimally at every node, and solves for the spread that is left over.

`scripts/spreads.py` builds a 10-step lognormal recombining lattice by forward induction.
✅ **Calibration check: the lattice reprices every zero-coupon bond on the curve to
3.33e-16** — that is the only test a calibration needs.

✅ Measured, **10y 6.5% corporate callable at 100 from year 5**, sigma = 20%, true OAS 80 bp:

| | ✅ value |
|---|---|
| bullet price | 102.219687 |
| **callable price** | **99.345962** |
| call option worth | **2.873725 points** |
| **Z-spread** | **120.04 bp** |
| **OAS** | **80.00 bp** |
| **option cost = Z − OAS** | **40.04 bp** |

🚨 **40.04 bp of "credit spread" that is not credit.** Rank this bond against a bullet on
Z-spread and it looks 40 bp cheap; it is not.

### ✅ The identity that proves the lattice, and the volatility you had to choose

✅ **Remove the call and the two measures must agree.** On the same bond with no call option:
Z-spread **80.1455 bp** vs OAS **80.0000 bp**, gap **−0.1455 bp** — the residual is the
lattice's own convexity (discounting `1/(1+r+s)` node by node is not the same operation as
shifting an annually compounded zero curve), not a modelling error. **If your OAS engine misses
the bullet identity by more than a fraction of a basis point, the lattice is the problem.**

🚨 **The OAS is the only one of the six measures that depends on an assumption you made.**
✅ Measured, same bond, same true 80 bp OAS, vol varied:

| sigma | price | Z-spread | OAS | **option cost** |
|---|---|---|---|---|
| **0%** | 102.230006 | 80.00 bp | 80.00 bp | **0.00 bp** |
| 10% | 100.959415 | 97.46 bp | 80.00 bp | 17.46 bp |
| **20%** | 99.345962 | 120.04 bp | 80.00 bp | **40.04 bp** |
| 30% | 97.786750 | 142.32 bp | 80.00 bp | 62.32 bp |

At **sigma = 0 the option cost is 0.0049 bp** — this bond's forward price never reaches the call
price without volatility, so there is nothing to strip out and Z and OAS coincide. **Every basis
point of option cost above that is your vol assumption**, and two desks quoting "the OAS" on the
same bond with 10% and 30% vol are 45 bp apart. An OAS without its vol is not a number.

## 6. ✅ Cross-check against QuantLib 1.43

| | QuantLib | this file | diff |
|---|---|---|---|
| `BondFunctions.zSpread` (annual compounded) | 150.000000 bp | 150.000000 bp | **−3.8e-09 bp** |
| `Bond.bondYield` | 6.434845% | 6.434845% | **+6.2e-15 pp** |

🚨 **In QuantLib 1.43 both take a `BondPrice`, not a float.** `BondFunctions.zSpread(bond,
85.7007, curve, ...)` raises `TypeError: Wrong number or type of arguments for overloaded
function`, because the clean/dirty flag is part of the type: pass
`ql.BondPrice(price, ql.BondPrice.Clean)`. It fails loudly, which is the good case — unlike
`Settings.instance().evaluationDate`, which fails silently.

## 7. What the script gives you

`scripts/spreads.py` — numpy and scipy (`brentq`) only; QuantLib imported inside one function.

| Function | Does |
|---|---|
| `spread_table(coupon, maturity, z_bp)` | §1, all six measures on one bond as a frozen `SpreadSet` |
| `g_spread_trap()` | §2, the three numbers a desk would all call "the spread" |
| `slope_sensitivity(slopes_bp)` | §2, the error as a function of curve slope |
| `par_yields(zeros)` / `discount_factors(zeros, spread)` | the par-vs-zero distinction itself |
| `z_spread(price, coupon, maturity, zeros)` | §1, brentq over a wide bracket |
| `asset_swap_spread(price_dirty, ...)` | §3, par/par ASW |
| `discount_margin(price, index, margin, ...)` / `dm_table` | §4 |
| `calibrate_tree(zeros, sigma)` | §5, forward induction; reprices the curve to 3.3e-16 |
| `lattice_bond_price(..., call_price, first_call)` | §5, bullet when `first_call=None` |
| `lattice_oas` / `oas_vs_zspread` / `option_cost_by_vol` | §5, and the bullet identity |
| `quantlib_cross_check()` | §6, or `None` |

## Where this sits

- `../../../fin-models/skills/credit-risk-models/SKILL.md` — where a spread turns into a
  **default probability**: Merton, `N(−d2)`, the constant hazard, `lambda(1−R)`, and the
  risk-neutral vs physical gap. This skill stops at the spread; that one starts there.
- `../cds-mechanics-and-upfront/SKILL.md` — the *other* credit spread. A CDS par spread and a
  bond Z-spread are different instruments with different discounting, and the basis between them
  is a trade, not an error.
- `../../../fin-models/skills/term-structure-models/SKILL.md` — every curve above is an input.
  🚨 A zero rate is a triple (day count, compounding, instrument); §2 here is what happens when
  you also forget whether it is a **par** rate or a **zero** rate.
- `../corporate-bond-data-and-trace/SKILL.md` — where the 85.70 came from, and why a TRACE print
  is not a mid. A spread computed off a one-sided last print inherits the bid-offer.
- `../ratings-transitions-and-migration/SKILL.md` — the rating that put the bond in the
  index whose OAS you are comparing against.
- `../../../fin-libraries/skills/lib-quantlib/SKILL.md` — 🚨
  `Settings.instance().evaluationDate` is a global and a stale one gives **NPV exactly 0.0**;
  §6 sets it before building any curve. Also `BondPrice`, `DiscountCurve` and `BondFunctions`.
- `../../../fin-core/skills/market-data-sourcing/SKILL.md` — the FRED ICE BofA OAS series
  (`BAMLC0A0CM` and friends) are **index** OAS: option-adjusted, market-value weighted, and not
  comparable to a single bond's G-spread.
