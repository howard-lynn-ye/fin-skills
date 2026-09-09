---
name: implied-vol-surface
description: >-
  Build a volatility surface that is not silently arbitrageable - invert prices to implied vols,
  fit a smile, check butterfly and calendar arbitrage, and interpolate between maturities.
  TRIGGER - implied volatility solver, Newton diverges, bisection bracket, "implied vol returns
  0.001", "no implied volatility for this option", price below intrinsic, BelowIntrinsicException;
  SVI, raw SVI, Gatheral, svi calibration, a b rho m sigma, SviSmileSection; volatility smile,
  skew, surface fitting, total variance, log-moneyness; butterfly arbitrage, negative implied
  density, Durrleman g(k), calendar spread arbitrage, static arbitrage check, Gatheral and
  Jacquier 2014; interpolating the vol surface, "my interpolated surface has arbitrage", "vol
  interpolation between expiries". SKIP for pricing one option and the models themselves - Heston,
  CRR, SABR, Monte Carlo (option-pricing-models), for library choice, Greek units and licences
  (derivatives-pricing), and for option chain data and historical chains (options-backtesting).
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-09"
---

# Implied vol surface

A surface is three separate problems — **invert**, **fit**, **interpolate** — and each has a way
of being wrong that produces a number instead of an error.

Every figure below is printed by `scripts/vol_surface.py` (runs in **5.1 s**; vollib and QuantLib
are optional, imported inside functions). ✅ Measured means this file produced it on 2026-09-09
with QuantLib 1.43, vollib 1.0.11, numpy 2.2.6, scipy 1.13.0, Python 3.11.3. The synthetic truth
is a Heston surface, so the "right answer" at any strike and maturity is known.

> **The rule:** invert with a bracketed solver on a **relative** price tolerance that refuses
> sub-intrinsic prices; fit in **total variance**; run **g(k) ≥ 0** and **dw/dt ≥ 0** on every
> fit; and interpolate maturities in **total variance at fixed log-moneyness** — never in vol.

## 1. Inverting a price — the solver is the easy half

✅ **Round trip over 27 cases** (sigma ∈ {0.10, 0.30, 0.80} × K/S ∈ {0.7, 1.0, 1.3} × T ∈ {0.05,
1, 3}, OTM side): worst |recovered − true| = **3.1e-14**. ✅ **vollib** (Jaeckel's *Let's Be
Rational*) on the identical cases: worst |mine − vollib| = **3.2e-14**. Two independent methods,
agreement at the last bit — the inversion itself is a solved problem.

⚠️ **`py_vollib` is not installed here; `vollib` 1.0.11 is.** `py_vollib` is a deprecated shim —
see `../../../fin-libraries/skills/lib-vollib/SKILL.md`. The import used is
`vollib.black_scholes_merton.implied_volatility`, which takes `q`.

### 1a. 🚨 An absolute price tolerance silently returns the initial guess

This one bit this very script. ✅ Measured: a 0.05-year OTM put at K=70, S=100, sigma=10% is worth
**7.935e-59**. Its distance from the model price at the solver's first guess sigma=0.001 is
**7.94e-59 — comfortably below an absolute tolerance of 1e-12** — so a solver whose stopping rule
is `abs(model - market) < 1e-12` **returns 0.001**. The true vol is 0.100000.
🚨 **9.9 vol points wrong, converged on the first iteration, no warning.**

**Stop on a tolerance relative to the price, and keep the bracket as the fallback.** With
`abs(diff) <= tol * price` plus a bracket-width stop, plain bisection recovers 0.10 to **8e-17**
even from a price of 1e-59. The conditioning is fine; the *stopping rule* was the bug.

### 1b. 🚨 Below intrinsic, at intrinsic, and the vol of a penny

✅ Measured, S=K=100, T=1, r=5%, discounted intrinsic **4.877058**:

| price | this solver | vollib |
|---|---|---|
| below intrinsic | `ValueError` | `py_lets_be_rational.exceptions.BelowIntrinsicException` |
| **exactly** intrinsic | `ValueError` | `BelowIntrinsicException` |
| intrinsic + 1e-6 | **0.011251** | **0.011251** |
| intrinsic + 1e-3 | **0.017390** | **0.017390** |

✅ Both refuse a sub-intrinsic price rather than returning a number, and **both treat the exact
intrinsic as below it** (correct: at intrinsic the only consistent vol is 0). Note how violently
the vol collapses just above the boundary: intrinsic + one cent is **2.2480%** implied vol on an
option whose real vol is 20%. **A stale ITM quote sitting near intrinsic does not produce a
slightly-low vol — it produces a near-zero one.**

✅ **What a penny of quote noise is worth in vol** (S=100, T in years, r=3%, q=1%, sigma=20%,
resolution = half a tick / vega):

| K | T | exact price | vega | **vol per half-cent** | vol from the price rounded to $0.01 |
|---|---|---|---|---|---|
| 100 | 0.05 | 1.832588 | 8.9072 | **0.06 pts** | 0.1997 (−0.03 pts) |
| 110 | 0.05 | 0.029571 | 1.0112 | **0.49 pts** | 0.2004 (+0.04 pts) |
| 115 | 0.05 | 0.001267 | 0.0776 | **6.45 pts** | 🚨 rounds to 0.00 — **no implied vol** |
| 120 | 0.05 | 0.000027 | 0.0026 | **190.18 pts** | 🚨 rounds to 0.00 — **no implied vol** |
| 120 | 1.00 | 2.521584 | 30.6625 | **0.02 pts** | 0.1999 (−0.01 pts) |

🚨 **The wing of a short-dated smile is not measurable from a penny-quoted chain.** Drop strikes
whose vol resolution exceeds your tolerance; do not fit a smile through them. This is the pricing
side of the data traps in `../../../fin-core/skills/derivatives-pricing/SKILL.md` §6 (mid, never
`lastPrice`).

## 2. Fitting: Gatheral's raw SVI, in total variance

`w(k) = a + b (rho (k − m) + sqrt((k − m)² + sigma²))`, fitted by `scipy.optimize.least_squares`
in **total variance** with the QuantLib parameter box as bounds and six deterministic rho starts.

✅ **Fitted to a Heston surface** (`v0=0.04, kappa=1.5, theta=0.04, sigma=0.3, rho=−0.7`, r=3%,
q=1%, 13 strikes 70–130):

| T | ATM vol | **RMSE (vol pts)** | max abs err (vol pts) | a | b | rho | m | sigma | min g(k) |
|---|---|---|---|---|---|---|---|---|---|
| 0.25 | 0.1967 | **0.0367** | 0.0547 | 0.0055 | 0.0118 | **−0.9990** | 0.1646 | 0.0679 | +0.3638 ✅ |
| 0.50 | 0.1945 | **0.0249** | 0.0359 | 0.0105 | 0.0212 | **−0.9990** | 0.1790 | 0.0908 | +0.3679 ✅ |
| 1.00 | 0.1926 | **0.0078** | 0.0115 | 0.0190 | 0.0350 | **−0.9990** | 0.2153 | 0.1573 | +0.3804 ✅ |
| 2.00 | 0.1925 | **0.0005** | 0.0007 | 0.0312 | 0.0507 | **−0.9990** | 0.3161 | 0.3379 | +0.4139 ✅ |

**Sub-0.04-vol-point RMSE everywhere** — five parameters reproduce a Heston smile far inside any
bid-ask. ✅ The calendar check over k ∈ [−0.4, 0.3] passes: smallest increment of total variance
between adjacent maturities **+0.005397**.

### 2b. 🚨 The fit is excellent and the parameters are meaningless

Every slice above pins **rho at the −0.999 bound**. That is not a bad optimizer — ✅ measured, by
refitting the other four parameters with rho *held*:

| rho held at | −0.999 | −0.900 | −0.700 | −0.500 | −0.300 |
|---|---|---|---|---|---|
| RMSE, T=0.25 (vol pts) | 0.0380 | 0.0406 | 0.0465 | 0.0526 | 0.0584 |
| RMSE, T=1.00 (vol pts) | 0.0080 | 0.0085 | 0.0095 | 0.0104 | 0.0113 |

**Worst/best RMSE ratio across the whole range of rho: 1.54× at T=0.25 and 1.41× at T=1.0**, and
every one of those fits is under 0.06 vol points. The objective is flat: **rho is not identified
by a 13-strike smile.** The optimizer runs to the bound because nothing stops it.

🚨 **Never report, hedge on, or time-series a fitted raw-SVI parameter.** Report the surface it
implies. A "rho went from −0.99 to −0.4 this morning" story is a solver artefact. If you need
stable parameters, fix one (b or rho) across the surface, or use the arbitrage-free SSVI
parameterization, which shares parameters across maturities by construction.

## 3. ✅ The two static no-arbitrage checks, and a published slice that fails one

From **Gatheral & Jacquier (2014), *Arbitrage-free SVI volatility surfaces***:

- **Butterfly (Lemma 2.2):** a slice is free of butterfly arbitrage iff **Durrleman's**
  `g(k) = (1 − k w'/(2w))² − (w'²/4)(1/w + 1/4) + w''/2 ≥ 0` for all k (plus the wing condition
  `d+(k) -> -inf`). g is proportional to the implied risk-neutral density.
- **Calendar (Lemma 2.1 / Definition 2.2):** a surface is free of calendar-spread arbitrage iff
  **total variance w(k, t) is non-decreasing in t at every fixed k.** Not vol — total variance.

⚠️ Axel Vogt's raw-SVI parameters, quoted in that paper as an example of a plausible-looking
arbitrageable slice: **(a, b, rho, m, sigma) = (−0.0410, 0.1331, 0.3060, 0.3586, 0.4153)** at
t = 1. The values are secondhand from the paper; everything computed from them is measured here.

✅ **min g(k) = −0.0329 at k = +0.879** — butterfly arbitrage, as the paper reports.

✅ **And g is verified, not asserted.** Taking `d²C/dK²` of the *Black* price on the same slice by
central difference — an implementation that contains no `g` anywhere — the density is negative on
**k ∈ [+0.643, +1.256], 614 of 2801 grid points**, minimum **−5.773e-05 at k = +0.788**. `g(k) < 0`
on **exactly the same 614 points, the same interval endpoints**. Breeden-Litzenberger and
Durrleman agree point for point.

### 🚨 No library runs the butterfly check for you

✅ Verified against QuantLib 1.43: **`ql.SviSmileSection(1.0, 1.0, Vogt)` constructs without
complaint.** QuantLib validates SVI parameters in that constructor (`checkSviParameters`, which is
**not exposed to Python separately**), and its five conditions are a **box**, not an arbitrage
test: `b >= 0`, `|rho| < 1`, `sigma > 0`, `a + b·sigma·sqrt(1−rho²) >= 0` (minimum variance), and
`b(1 + |rho|) <= 4`. Every one of them passes on a slice with a negative density over a third of
the strikes tested.

**Run g(k) yourself on every fit.** It costs one vectorised expression.

## 4. 🚨 Interpolating maturities: the trap is not accuracy, it is arbitrage

### The honest accuracy comparison

✅ Interpolating to T=0.75 from the fitted T=0.5 and T=1.0 slices at fixed k, against the true
Heston smile — on a **calm, smooth** term structure:

| method | max abs err vs Heston (vol pts) | ATM err (vol pts) |
|---|---|---|
| total variance | 0.1519 | −0.0113 |
| **vol** | **0.0788** | +0.0520 |
| price | 0.6361 | −0.2592 |

⚠️ **Linear-in-vol is slightly the most accurate here.** Accuracy is not the argument. Note also
that interpolating in *price* is 8× worse than either — a normalised Black price is very nonlinear
in maturity, and the round trip through the inversion adds nothing.

### ✅ The manufactured violation

Now a **steep event term structure** — Heston with `v0=0.36, kappa=8.0, theta=0.04, sigma=0.5,
rho=−0.5`, a 60% spot vol decaying fast. ATM pillars: **T=0.0192 vol 57.96% (w=0.00646)** and
**T=0.5 vol 34.02% (w=0.05787)**. ✅ The true Heston total variance rises monotonically
(**min dw = +3.65e-04**). Interpolate between the two pillars 60 ways:

| method | min dw per step | max vol err vs Heston | verdict |
|---|---|---|---|
| **linear in vol** | **−4.23e-04** | 4.70 pts | 🚨 **CALENDAR ARBITRAGE** — w peaks at 0.06085 at T=0.394, then **falls** to 0.05787 at T=0.5 |
| linear in total variance | +8.71e-04 | 12.11 pts | ✅ monotone |
| linear in price | +4.43e-04 | 16.92 pts | ✅ monotone |

🚨 **The consequence, priced:** a 0.5-year ATM option quoted off the vol-interpolated surface costs
**less** than the 0.39-year one — normalised prices **0.09574 vs 0.09816**. That is a **free
calendar spread worth 24.2 bp of forward**, created entirely by the interpolation. Any optimizer
pointed at that surface will find it and load up on it.

**Linear-in-total-variance is monotone whenever the pillars are** — it is a convex combination of
two ordered numbers. That guarantee is why it wins, and it holds at every k, not just ATM. The
0.08-vol-point accuracy edge that vol interpolation showed in the calm case is not worth a surface
that can invent arbitrage on the steep one.

## 5. ✅ Cross-checks

- **Heston surface, 52 calls:** worst |this implementation − `ql.AnalyticHestonEngine(Gatheral)`|
  = **2.1e-13**. (The `heston1993` formulation that this one avoids is measured breaking in
  `../option-pricing-models/SKILL.md` §1.)
- 🚨 **`ql.SviSmileSection(T, F, params)` takes `[a, b, sigma, rho, m]`, not Gatheral's
  `(a, b, rho, m, sigma)`.** ✅ With the QuantLib order, worst |vol diff| over the four fitted
  slices and 13 strikes = **0.0e+00**. ✅ What the swap costs, measured on
  `a=0.02, b=0.10, |rho|=0.30, m=0.05, sigma=0.20`, forward-ATM:

  | | correct order | swapped |
  |---|---|---|
  | **rho = −0.30** | 0.208062 | ✅ `RuntimeError: sigma (-0.3) must be positive` — caught |
  | **rho = +0.30** | 0.197711 | 🚨 **0.236859 — 3.91 vol points away, NO ERROR** |

  **The swap is only caught when rho happens to be negative.** With a positive rho every value is
  in range, QuantLib constructs, prices, and returns a different smile in silence.

## 6. What the script gives you

`scripts/vol_surface.py` — numpy + scipy only at import; vollib and QuantLib imported inside their
cross-check functions.

| Function | Does |
|---|---|
| `implied_vol(price, S, K, T, r, q, flag)` | safeguarded Newton/bisection, relative tolerance, refuses sub-intrinsic |
| `vol_from_rounded_price(..., tick)` | §1b — the vol you actually get from a penny-quoted price, or `None` |
| `svi_total_variance` / `svi_derivatives` | raw SVI and its analytic w', w'' |
| `fit_svi(k, w)` | least-squares fit in total variance, six starts, QuantLib's box as bounds |
| `svi_rho_profile(k, w)` | §2b — RMSE with rho held, the identifiability probe |
| `durrleman_g` / `butterfly_check` | Gatheral & Jacquier eq. (2.1), Lemma 2.2 |
| `breeden_litzenberger_density` | the density straight off Black's formula — the independent check |
| `calendar_check(k_grid, slices)` | Lemma 2.1 across maturities |
| `interpolate_maturity(..., method)` | `"total_variance"`, `"vol"`, `"price"` |
| `heston_call` / `heston_smile` | the synthetic truth |
| `vollib_cross_check` / `quantlib_cross_check` | or `None` |
| `VOGT` | the published arbitrageable parameter set |

## Where this sits

- `../option-pricing-models/SKILL.md` — the models under the surface: Heston (and the branch-cut
  trap), CRR, SABR, Monte Carlo. That skill prices one option; this one builds the surface.
- `../../../fin-core/skills/derivatives-pricing/SKILL.md` — library choice, licences, Greek units,
  QuantLib's SVI/SABR/no-arbitrage symbol inventory, and §6 the chain-data traps that decide
  whether the prices you are inverting mean anything.
- `../../../fin-libraries/skills/lib-vollib/SKILL.md` — `vollib` vs the dead `py_vollib` shim, and
  the `black_scholes` (no q) vs `black_scholes_merton` (has q) trap.
- `../../../fin-libraries/skills/lib-quantlib/SKILL.md` — 🚨 `Settings.instance().evaluationDate`
  is a global; a stale one gives **NPV exactly 0.0**. Every QuantLib call in §5 sets it first.
- `../../../fin-core/skills/options-backtesting/SKILL.md` — historical chains, and the lifecycle
  that a surface does not model.
