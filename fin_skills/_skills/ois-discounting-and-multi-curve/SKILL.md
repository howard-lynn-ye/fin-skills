---
name: ois-discounting-and-multi-curve
description: >-
  Price a swap with separate projection and discount curves, and catch the single-curve bug that
  the standard par-reprice check cannot see. TRIGGER - OIS discounting, CSA discounting,
  collateral discounting, multi-curve, dual curve, projection curve vs discount curve, tenor
  basis, "my swap reprices at par but the PV01 is wrong", swap annuity, fixedLegBPS,
  DiscountingSwapEngine, RelinkableYieldTermStructureHandle, linkTo, "QuantLib NPV is exactly
  0.0", exogenous discounting rate helpers, bootstrapping a SOFR curve against an OIS discount
  curve, swaption numeraire, forward premium. SKIP for computing the compounded SOFR fixing
  itself (sofr-and-rfr-compounding), for what a legacy LIBOR trade falls back to
  (libor-transition-and-fallbacks), for bond duration and DV01
  (duration-convexity-and-dv01), and for curve bootstrapping and interpolation in general
  (../../../fin-models/skills/term-structure-models).
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-09"
---

# OIS discounting and multi-curve

**A collateralised swap needs two curves: one to project the coupons and one to discount them.**
Build one curve, use it for both, and the check everyone runs — *"does the swap reprice at
par?"* — still passes. The par rate is the one quantity in the whole calculation that is almost
blind to the discount curve.

Every figure below is printed by `scripts/multi_curve.py` (runs in **0.3 s**; QuantLib
optional). ✅ Measured means this file produced it on 2026-09-09 with QuantLib 1.43, numpy 2.2.6,
Python 3.11.3. The worked trade is a **10-year swap, 100,000,000 notional, semiannual fixed
against quarterly floating, K = 4.50%**, on an OIS zero curve rising 2.90% → 3.60% with a
**projection curve of OIS + 26.161 bp**.

> **The rule: the par swap rate is a discount-WEIGHTED AVERAGE of the projected forwards and
> barely moves when you change the discount curve. The annuity is a LEVEL and moves by the whole
> basis. Check the annuity, not the par rate.**

## 1. 🚨 The check everyone runs passes under both curves

✅ Measured:

| | par rate | swap struck at its own par rate prices at |
|---|---|---|
| OIS discounting | 3.75361656% | **+0.00e+00** |
| projection discounting (the bug) | 3.75316548% | **+0.00e+00** |

🚨 **Both reprice at par to machine precision.** The test has no power to distinguish them,
because a swap struck at its own par rate is zero by construction under *whatever* discount curve
computed that par rate. It is a self-consistency check, not a validation.

## 2. 🚨 What actually moved

✅ Measured on the same swap:

| | OIS discounting | projection discounting | difference |
|---|---|---|---|
| par swap rate | 3.75361656% | 3.75316548% | **−0.045 bp** |
| **annuity (years)** | **8.42485066** | **8.31654893** | 🚨 **−1.286 %** |
| **PV01** | **84,248.51** | **83,165.49** | 🚨 **−1,083.02** |
| NPV at K = 4.50% | −6,288,169.05 | −6,211,085.81 | 🚨 **+77,083.24** |

🚨 **The par rate moved 0.045 bp. The annuity moved 1.286%.** That 1.286% is a multiplicative
factor on:

- every **PV01** and therefore every hedge ratio and every DV01-based limit;
- every **off-market NPV** — 77,083 on this one trade;
- every **swaption premium**, because the annuity is the numeraire in the Black formula for a
  swaption, so a 1.286% error in A is a 1.286% error in every premium and every vega;
- every **cashflow-weighted** risk report, since the weights are the discount factors.

**None of it shows up in a par rate, a fair rate, or a repricing test.**

## 3. ✅ Why the par rate cannot see it

```
S = sum(tau_i f_i D_i) / sum(tau_j D_j)
```

**S is a ratio of two sums over the same discount factors.** Change `D` and you change only the
weights of an average. `A = sum(tau_j D_j)` is not an average of anything — it is a level.

✅ Measured on a **flat forward curve with fixed and floating on the same quarterly schedule**,
where the cancellation is exact:

| | OIS discounting | projection discounting | difference |
|---|---|---|---|
| par rate | 3.5775134989% | 3.5775134989% | **−0.00000000 bp** |
| annuity | 8.48237640 | 8.37565384 | **−1.2582 %** |

✅ The same curve with a **semiannual fixed leg against a quarterly float leg** leaves a residual
of **+0.118 bp** — the sums are over different dates, so the cancellation is no longer perfect.
**0.118 bp against an annuity that moved 1.2582%: the par-rate check is not imprecise, it is
looking at the wrong quantity.**

## 4. ✅ It scales with the basis, and the par rate still does not notice

| basis (bp) | par rate S | Δ par rate | Δ annuity | PV01 error | NPV error |
|---|---|---|---|---|---|
| 5.000 | 3.5392% | −0.010 bp | −0.247 % | −208 | +19,195 |
| 10.000 | 3.5898% | −0.019 bp | −0.494 % | −416 | +36,271 |
| **26.161** | 3.7536% | **−0.045 bp** | **−1.286 %** | **−1,083** | **+77,083** |
| 50.000 | 3.9954% | −0.072 bp | −2.438 % | −2,054 | +97,750 |
| **100.000** | **4.5028%** | −0.082 bp | 🚨 **−4.796 %** | **−4,041** | **−7,743** |

- **A 100 bp basis moves the par rate by 0.082 bp and the annuity by 4.8%.** They are not in the
  same units of wrongness.
- 🚨 **The NPV error is `(S − K) × ΔA`, so it vanishes at the money** — at a 100 bp basis
  S = 4.5028% ≈ K = 4.50% and the NPV error collapses to −7,743 while the annuity error is at its
  worst. **A portfolio checked only on at-the-money trades will look clean.** The PV01 error does
  not vanish anywhere.

## 5. ✅ QuantLib reproduces it — and adds two traps of its own

✅ `ql.VanillaSwap` + `ql.DiscountingSwapEngine`, same curves:

| | OIS discounting | projection discounting | difference |
|---|---|---|---|
| `fairRate()` | 3.75561726% | 3.75507846% | **−0.054 bp** |
| annuity from `fixedLegBPS` | 8.42686799 | 8.31816553 | **−1.290 %** |
| `NPV()` | −6,272,815.07 | −6,196,380.69 | **+76,434.38** |

⚠️ QuantLib's absolute numbers differ from the reference implementation in the third decimal of
a percent because it uses real business-day schedules and 30/360 vs ACT/360 accruals where the
reference uses exact `1/freq` periods. **The shape is what matters and it matches: 0.054 bp on
the par rate, 1.290% on the annuity.**

### 🚨 A `RelinkableYieldTermStructureHandle` is a live reference

✅ Measured: relinking the discount handle to an OIS curve **+10 bp** moves an **already
constructed** swap's NPV from **−6,272,815.07 to −6,243,442.24 (+29,372.83)** with no
recalculation call and no notification you can see. That is the design — handles are observables
— and it means:

- **Two swaps sharing a handle cannot be priced on different curves**, however you order the
  code. Give each its own handle.
- **Rate helpers built with `discountingCurve=ois_handle` re-bootstrap when that handle is
  relinked**, so changing the discount curve silently changes the *projection* curve too. Build,
  freeze, and re-read the projection curve rather than assuming it is inert.
- A handle left **unlinked** raises, which is the safe failure. A handle linked to the *wrong*
  curve does not.

### 🚨 A stale `evaluationDate` gives NPV exactly 0.0

✅ Measured: pushing `Settings.instance().evaluationDate` one day past maturity makes the same
swap return **NPV = 0.0** — not an error, not a warning, not `nan`. Every cash flow has occurred,
so the sum is empty. See `../../../fin-libraries/skills/lib-quantlib/SKILL.md`; it is the single
most common way a QuantLib pricing script produces a plausible zero.

## 6. What to check instead

1. **Compare annuities, not par rates.** `sum(tau_j D_j)` under your discount curve against the
   OIS curve directly. It is one line and it is the quantity that moves.
2. **Price an OFF-market swap.** A swap struck 100 bp away from par has an NPV that depends on
   the annuity; one struck at par does not.
3. **Assert the engine's discount curve is the CSA curve**, not the index's forecast curve. In
   QuantLib that is the handle you pass to `DiscountingSwapEngine`, and it is a different
   argument from the one inside the `IborIndex`.
4. **Check `fixedLegBPS`**, not `NPV`, when you change discounting.

## 7. What the script gives you

`scripts/multi_curve.py` — numpy only at import; QuantLib inside one function.

| Function | Does |
|---|---|
| `Curve(times, zeros, spread)` | log-linear-on-discount zero curve; `.discount`, `.forward`, `.zero`, `.shifted` |
| `flat_curve(rate)` | §3's flat forward curve |
| `annuity(discount, tenor, fixed_freq)` | `sum(tau_j D_j)` — the level |
| `floating_leg_pv(projection, discount, ...)` | forwards off one curve, discounting off the other |
| `par_swap_rate(projection, discount, ...)` | §1, §3 — the weighted average |
| `swap_npv(K, N, projection, discount, ...)` / `pv01(N, discount, ...)` | §2 |
| `discounting_comparison(ois, projection, ...)` | the §2 table |
| `reprices_at_par(...)` | §1 — the check with no power |
| `basis_sensitivity(ois, bases_bp, ...)` | §4 |
| `quantlib_cross_checks(...)` | §5, including the relink and the stale-date effects, or `None` |

## Where this sits

- `../sofr-and-rfr-compounding/SKILL.md` — the fixings and averages the projection curve is
  built from, and 🚨 the four few-bp errors in compounding one.
- `../libor-transition-and-fallbacks/SKILL.md` — 🚨 the basis used here (26.161 bp) is the
  statutory three-month tenor spread adjustment in 12 CFR 253.4(c); that skill is where the
  legacy trades get their new floating leg.
- `../duration-convexity-and-dv01/SKILL.md` — the bond side of the same arithmetic: 🚨 Macaulay
  vs modified is exactly (1 + y/m), and QuantLib's `Duration.Modified` is silently 28.6% low on
  a non-compounded `InterestRate`.
- `../../../fin-models/skills/term-structure-models/SKILL.md` — bootstrapping, interpolation and
  🚨 the six different "5-year zero rates" one discount factor can be. Curve *construction*
  lives there; this skill is about which curve goes where.
- `../../../fin-libraries/skills/lib-quantlib/SKILL.md` — 🚨 `evaluationDate` as a global,
  handle lifetimes, and `ql.Date` being day-first.
- `../../../fin-core/skills/derivatives-pricing/SKILL.md` — 🚨 `rateslib` is not open source;
  QuantLib term structures are the permissive route for everything above.
