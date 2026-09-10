---
name: term-structure-models
description: >-
  Build and fit a yield curve, and price a zero-coupon bond in a short-rate model, without the
  convention and identification traps. TRIGGER - bootstrap a zero curve, par rates to zero rates,
  discount factors to zero rates, par bond reprice, curve stripping; day count ACT/365 ACT/360
  30/360, annual vs semiannual vs continuous compounding, "my zero rate is off by a few basis
  points", "which day count did this curve use"; Nelson-Siegel, Svensson, Diebold-Li, lambda
  0.0609, beta0 beta1 beta2 level slope curvature, "my Nelson-Siegel lambda jumps around";
  Vasicek, CIR, Cox-Ingersoll-Ross, Hull-White one factor, A(t,T) B(t,T), affine bond price,
  Feller condition, "sqrt of a negative rate", NaN in my CIR simulation. SKIP for option pricing
  and implied vol (option-pricing-models, implied-vol-surface), for QuantLib's evaluationDate
  global (lib-quantlib), and for macro rate data sourcing such as FRED and Treasury series
  (fundamental-and-macro-data).
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-09"
---

# Term structure models

Four things go wrong here, and none of them raises: **a zero rate quoted without its convention**,
**a Nelson-Siegel lambda that is not identified**, **an Euler CIR step that takes the square root
of a negative rate**, and **two QuantLib constructors that take the same two parameters in
opposite orders.**

Every number below is printed by `scripts/term_structure.py` (runs in **13.8 s**; QuantLib
optional, imported inside `quantlib_cross_checks`). ✅ Measured means this file produced it on
2026-09-09 with QuantLib 1.43, numpy 2.2.6, scipy 1.13.0, Python 3.11.3.

> **The rule:** a zero rate is a triple — **(day count, compounding, instrument)**. Carry all
> three. Fix the Nelson-Siegel lambda unless you can show it is identified. Never take
> `sqrt(r)` of an Euler CIR step without truncation.

## 1. ✅ Bootstrap, and the check that proves it

Par rates 3.00 / 3.50 / 4.00 / 4.40 / 4.70% (annual coupons), solved forward one maturity at a
time:

| year | par | DF | zero (continuous) | zero (annual) |
|---|---|---|---|---|
| 1 | 3.000% | 0.97087379 | 2.95588% | 3.00000% |
| 2 | 3.500% | 0.93335209 | 3.44864% | 3.50879% |
| 3 | 4.000% | 0.88829900 | 3.94823% | 4.02721% |
| 4 | 4.400% | 0.84016179 | 4.35402% | 4.45020% |
| 5 | 4.700% | 0.79203794 | 4.66292% | 4.77334% |

✅ **Every par bond reprices to exactly 100 on the curve it produced — worst |price − 100| =
0.0e+00.** That is the only test a bootstrap needs, and a bootstrap that fails it is broken no
matter how smooth the curve looks. ✅ Round-tripping DFs → zero rates → DFs: max |diff| =
**2.2e-16**. ✅ And QuantLib's own `DiscountingBondEngine` on the same DFs prices all five par
bonds at **100.00000000**.

**The two directions are not the same problem.** From **discount factors** the zero curve is one
`log` per point and cannot fail. From **par rates** each maturity depends on every shorter one, so
an error at year 2 contaminates years 3, 4 and 5 — check the repricing, not the shape.

## 2. 🚨 The same discount factor is six different "5-year zero rates"

✅ Measured on **DF = 0.79203794** for 2026-09-08 → 2031-09-08, against the 30/360-annual rate the
bootstrap actually produced (**4.77334%**):

| day count | compounding | T (years) | zero rate | **bp vs base** |
|---|---|---|---|---|
| ACT/365 | annual | 5.002740 | 4.77067% | **−0.27** |
| ACT/365 | semiannual | 5.002740 | 4.71509% | **−5.83** |
| ACT/365 | continuous | 5.002740 | 4.66037% | **−11.30** |
| ACT/365 | simple | 5.002740 | 5.24844% | **+47.51** |
| ACT/360 | annual | 5.072222 | 4.70380% | **−6.95** |
| ACT/360 | continuous | 5.072222 | 4.59653% | **−17.68** |
| **30/360** | **annual** | **5.000000** | **4.77334%** | **base** |
| 30/360 | continuous | 5.000000 | 4.66292% | **−11.04** |
| 30/360 | simple | 5.000000 | 5.25132% | **+47.80** |

**Decomposed:**
- **compounding alone** (annual → continuous, same day count): **−11.04 bp**
- **day count alone** (30/360 → ACT/365, same compounding): **−0.27 bp**
- **day count alone** (30/360 → ACT/360): **−6.95 bp**
- **both** (ACT/360 continuous vs 30/360 annual): **−17.68 bp**
- 🚨 **simple vs annual: +47.80 bp** — the largest single error, and the easiest to make by
  writing `(1/DF − 1)/T`.

🚨 **17.68 bp is a real trading error and there is nothing in the number to reveal it.** A zero
rate is not a number; it is a number plus a convention. ✅ QuantLib agrees on all three year
fractions to 6 dp (`Actual365Fixed` 5.002740, `Actual360` 5.072222, `Thirty360(BondBasis)`
5.000000) and its `InterestRate(0.05, ...).equivalentRate(Continuous, Annual, 5)` returns
**0.0487901642 = ln(1.05)**. `../../../fin-core/skills/us-market-rules/SKILL.md` owns the
settlement and calendar side; this is the arithmetic side.

## 3. Nelson-Siegel: the Diebold-Li lambda, in the right unit

The curvature loading `(1 − e^{−x})/x − e^{−x}`, with `x = lambda·tau`, ✅ **peaks at
x\* = 1.793282**.

- ⚠️ Diebold & Li (2006) fix **lambda = 0.0609 with tau in MONTHS**, saying it maximises curvature
  at 30 months.
- ✅ **Measured: 0.0609 actually puts the peak at 29.45 months.** Exactly 30 months needs
  **lambda = 0.0598**. The published value is rounded; nothing depends on the difference, but do
  not re-derive 0.0609 and conclude your algebra is wrong.
- 🚨 **In YEARS the same decay is 0.0609 × 12 = 0.7308.** ✅ Feeding **0.0609** to a curve whose
  maturities are in years puts the hump at **29 YEARS instead of 29 months** — a curve that is
  essentially a straight line over the traded range. This is the single most common Nelson-Siegel
  bug and it produces a plausible fit.
- ✅ QuantLib's `NelsonSiegelFitting` uses the same `(beta0, beta1, beta2, lambda)` convention with
  lambda as a decay rate in the curve's own time unit: reproduced at the 5-year point to
  **0.0e+00**. Same for `SvenssonFitting`, **0.0e+00**.

### 3b. 🚨 A free lambda is not identified — measured on 200 resamples

✅ Synthetic curve `NS(beta = 6.0, −2.0, −1.5; lambda = 0.0609)` in percent at the 17 Diebold-Li
maturities (3–120 months) plus 5 bp of noise, seed 0, then 200 residual-bootstrap refits:

| | **lambda free** | **lambda fixed at 0.0609** | ratio |
|---|---|---|---|
| lambda, 5th–95th pct | **[0.0100, 0.0625]** | — | — |
| lambda, full range | [0.0100, 0.0663] | — | — |
| lambda, std | **0.0182** | 0 by construction | — |
| **beta1 (slope) std** | **0.803** | **0.032** | 🚨 **25.1×** |
| **beta2 (curvature) std** | **2.436** | **0.116** | 🚨 **21.1×** |
| **RMSE** | **3.34 bp** | **3.55 bp** | **1.06×** |

🚨 **Read the last two rows together.** Freeing lambda improves the fit by **0.21 bp** and makes
the curvature factor **21× less stable**. The 5th percentile of the free lambda sits **on the
lower bound of the search** — the optimizer runs out of the box on ordinary noise.

**Beta2 is the curvature factor people trade and regress on.** A free lambda turns it into noise
with a plausible RMSE attached. **Fix lambda** (0.0609/month = 0.7308/year, or your own value
estimated once on a long sample and then frozen), and the three betas become a linear least
squares — closed form, no optimizer, no local minima.

### 3c. ⚠️ Svensson adds a fourth factor and a collinearity you must watch

✅ On the same curve: Svensson RMSE **3.28 bp** vs NS-free **3.34 bp** and NS-fixed **3.55 bp** —
two extra parameters buy **0.06 bp**. Its fitted `(beta3, lambda1, lambda2)` = (−2.875, 0.204,
0.06) invents a second hump that the generating curve does not have.

🚨 **The failure mode is lambda2 → lambda1.** ✅ Measured condition number of the loading matrix as
the two decays converge:

| lambda2 / lambda1 | 2.00 | 1.20 | 1.05 | 1.01 |
|---|---|---|---|---|
| condition number | 6.6e+01 | 2.5e+02 | 9.6e+02 | **4.8e+03** |

When the two decays are close the third and fourth loadings are nearly the same column, `beta2`
and `beta3` are only identified by their sum, and they run to large opposite values. **Constrain
`lambda2/lambda1` away from 1**, or use Nelson-Siegel — the fourth factor here bought 0.06 bp.

## 4. ✅ Vasicek, CIR and Hull-White — the closed forms, checked three ways

All three are affine: `P(t,T) = A(t,T) e^{−B(t,T) r_t}`. ✅ P(1, 5) with r(1) = 4%:

| model | this file | QuantLib 1.43 | \|diff\| |
|---|---|---|---|
| Vasicek (a=0.1, b=5%, sigma=1%) | 0.846848705626 | `Vasicek.discountBond` 0.846848705626 | **0.0e+00** |
| CIR (kappa=0.3, theta=5%, sigma=10%) | 0.839702798215 | `CoxIngersollRoss.discountBond` 0.839702798215 | **0.0e+00** |
| Hull-White (flat 5%, a=0.1, sigma=1%) | 0.845755850892 | `HullWhite.discountBond` 0.845755850892 | **2.2e-13** |

✅ **A and B are checked separately, not only through their product** — because a compensating
error in both is exactly what a single price comparison cannot see:

| | B (formula) | −d ln P / dr | \|diff\| | A (formula) | P·e^{Br} | \|diff\| |
|---|---|---|---|---|---|---|
| Vasicek | 3.296799539644 | 3.296799539684 | 4.1e-11 | 0.966222400973 | 0.966222400973 | **0.0e+00** |
| CIR | 2.295503449572 | 2.295503449629 | 5.6e-11 | 0.920455039177 | 0.920455039177 | 1.1e-16 |

✅ **And the sigma → 0 limit**, which both must collapse to `exp(−(level·tau + (r − level)·B))`,
the deterministic ODE integral — an expression that uses neither A nor the model's own code:

| sigma | 1e-2 | 1e-4 | 1e-6 | 1e-9 |
|---|---|---|---|---|
| Vasicek, P − deterministic | +6.8e-04 | +6.8e-08 | +6.8e-12 | **+0.0e+00** |
| CIR, P − deterministic | +1.7e-05 | +1.5e-09 | 🚨 **+2.1e-06** | 🚨 **+7.1e+02** |

🚨 **CIR's `A` is `(·)^(2·kappa·theta/sigma²)`.** At sigma = 1e-9 that exponent is **3.0e+16** on a
base one ulp below 1, and `cir_bond` returns **712.06 instead of 0.838** — a bond price above 700.
The error stops falling below about **sigma = 1e-4**. Vasicek's `A` is a plain exponential and
stays exact to sigma = 1e-12. **This is a floating-point trap, not a modelling one**, but it means
a CIR calibration that wanders toward a tiny sigma will start returning nonsense before it
converges. Bound sigma away from zero.

### 4b. 🚨 QuantLib takes the same two parameters in opposite orders

✅ Verified live on QuantLib 1.43:

| constructor | correct | middle two swapped |
|---|---|---|
| `ql.Vasicek(r0, **speed**=0.1, level=0.05, sigma)` | 0.846848705626 | 0.833971282693 (**−0.012877**, no error) |
| `ql.CoxIngersollRoss(r0, **level**=0.05, speed=0.3, sigma)` | 0.839702798215 | 0.776731850164 (**−0.062971**, no error) |

**`Vasicek` is (r0, speed, level, sigma); `CoxIngersollRoss` is (r0, level, speed, sigma).** Both
accept the swap and return a bond price that is perfectly plausible — 0.78 instead of 0.84 is
**6.3 price points**, and nothing anywhere says so. Use keyword-free calls at your peril; write a
one-line assertion against a known value instead.

## 5. 🚨 Euler CIR takes the square root of a negative rate

`dr = kappa(theta − r) dt + sigma·sqrt(r) dW`. The **Feller condition** `2·kappa·theta > sigma²`
keeps the *continuous* process strictly positive. **A discretised one goes negative anyway** —
Feller says nothing about a finite time step.

✅ Measured, r0 = 3%, T = 5y, monthly steps, 20,000 seeded paths:

| | negative steps | **NaN paths** | MC price | bias vs exact |
|---|---|---|---|---|
| **Feller holds** (kappa=0.5, theta=3%, sigma=15%; 0.030 > 0.022), exact P = 0.863315 | | | | |
| plain Euler | 4,212 | 🚨 **4,060 (20.3%)** | 0.849591 *(survivors only)* | 🚨 **−0.013724** |
| full truncation | 8,332 | **0** | 0.862916 | **−0.000399** |
| **Feller violated** (sigma=20%; 0.030 < 0.040), exact P = 0.865224 | | | | |
| plain Euler | 11,630 | 🚨 **11,401 (57.0%)** | 0.807012 *(survivors only)* | 🚨 **−0.058212** |
| full truncation | 48,791 | **0** | 0.864559 | **−0.000666** |

🚨 **Even with Feller satisfied, one path in five dies.** Once `r < 0`, `sqrt(r)` is `NaN` and that
path is dead for the rest of its life.

🚨 **And then the reflex — drop the NaN paths — is the actual disaster.** The paths that went
negative are the *low-rate* paths, which carry the *high* discount factors. Dropping them is not
a numerical inconvenience, it is **survivorship bias in the estimator**: the surviving average is
**0.849591 against a true 0.863315**, biased low by **−0.0137, which is 27 standard errors**
(se 0.000508). The error bar says the answer is precise.

✅ **Full truncation** (Lord, Koekkoek & van Dijk 2010 — use `max(r, 0)` in the drift, the
diffusion *and* the discounting, and let `r` itself go negative) loses no paths and cuts the bias
to **−0.000399**, inside one standard error. It lets *more* steps go negative (8,332 vs 4,212)
and is far more accurate, which is the point: the negativity is not the problem, the `NaN` is.

## 6. What the script gives you

`scripts/term_structure.py` — numpy + scipy only at import.

| Function | Does |
|---|---|
| `bootstrap_from_par(par_rates)` / `par_bond_price` | §1, and the repricing check |
| `zero_rate` / `discount_factor` / `zero_curve_from_discounts` | DF ↔ zero under any convention |
| `year_fraction(d0, d1, convention)` / `convention_table` | §2, ACT/365, ACT/360, 30/360 |
| `nelson_siegel` / `svensson` / `ns_loadings` | the parameterizations |
| `fit_nelson_siegel(t, y, lam=None)` | `lam=None` frees it, a value fixes it |
| `curvature_loading_peak` / `diebold_li_lambda_facts` | §3, x\* = 1.793282 and the unit conversion |
| `lambda_stability(n_boot)` | §3b, the resampling experiment |
| `svensson_collinearity` | §3c, the condition numbers |
| `vasicek_bond` / `cir_bond` / `hull_white_bond` | §4 |
| `affine_ab(model, tau, ...)` / `deterministic_bond` | A and B separately, and the sigma → 0 limit |
| `simulate_cir(..., scheme)` | `"euler"` or `"full_truncation"` — §5 |
| `quantlib_cross_checks(...)` | everything in §4b and §6, or `None` |

## Where this sits

- `../../../fin-libraries/skills/lib-quantlib/SKILL.md` — 🚨 `Settings.instance().evaluationDate`
  is a global and a stale one gives **NPV exactly 0.0**. Every QuantLib call above sets it before
  constructing a curve. That skill also covers `Date` being day-first and handle lifetimes.
- `../../../fin-core/skills/derivatives-pricing/SKILL.md` — 🚨 **`rateslib` is not open source**;
  QuantLib term structures are the permissive route for anything in this skill.
- `../../../fin-core/skills/us-market-rules/SKILL.md` — settlement, business-day and holiday rules;
  §2 here is only the arithmetic once the dates are settled.
- `../../../fin-market-data/skills/fundamental-and-macro-data/SKILL.md` — where the par rates and Treasury
  series come from, and their revision behaviour.
- `../option-pricing-models/SKILL.md` — the same affine-model machinery for equity options
  (Heston), including a characteristic function that breaks the same way CIR's `A` does: silently,
  and only outside the range you tested.
