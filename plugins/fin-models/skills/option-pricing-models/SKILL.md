---
name: option-pricing-models
description: >-
  Implement an option pricing model correctly - closed form, tree, characteristic function, Monte
  Carlo - and the four places each silently returns a plausible wrong number. TRIGGER -
  Black-Scholes-Merton with dividend yield, binomial tree, CRR, Cox-Ross-Rubinstein, American
  early exercise, Richardson extrapolation; Heston, "the little Heston trap", branch cut, complex
  log, AnalyticHestonEngine, Gatheral vs BranchCorrection; SABR, Hagan 2002, sabrVolatility, ATM
  0/0, z/x(z); antithetic variates, standard error, Euler discretisation bias; "my Heston price is
  wrong at long maturity", "my Heston price is NaN", "my binomial tree will not converge", "my
  Monte Carlo error bar is tiny but the price is wrong", "my tree does not match QuantLib". SKIP
  for choosing a pricing library, Greek units and licences (derivatives-pricing), for fitting a
  whole surface and its no-arbitrage checks (implied-vol-surface), and for assignment, expiry and
  option lifecycle (options-backtesting).
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-09"
---

# Option pricing models

`../../../fin-core/skills/derivatives-pricing/SKILL.md` tells you **which library** and what the
Greeks mean. This skill is about **the models themselves** — what happens inside, and the four
failure modes that produce a number rather than an exception.

Every number below is printed by `scripts/option_models.py` (runs in **6.1 s**, QuantLib
optional). ✅ Measured means this file produced it on 2026-09-09 with QuantLib 1.43, numpy 2.2.6,
scipy 1.13.0, Python 3.11.3.

> **The four rules, in one line each:** price Heston with the Albrecher/Gatheral form; average CRR
> odd and even step counts; report Monte Carlo *bias* and *standard error* separately; and check
> which CRR probability your reference library uses before you call your tree wrong.

## 1. 🚨 The little Heston trap — the 1993 formula is discontinuous, and it does not raise

Heston (1993) eq. (17)–(18) writes the log-characteristic function with

    g = (b - rho*sigma*i*phi + d) / (b - rho*sigma*i*phi - d)     and     exp(+d*tau)

Albrecher, Mayer, Schoutens & Tistaert (2007), *The Little Heston Trap*, showed the algebraically
identical form with `1/g` and `exp(-d*tau)` is **continuous under the principal branch of the
complex log**, while the original is not. Under the original, the inversion integral integrates
**across a jump**, and you get a finite, plausible, wrong price.

✅ **Measured** (`v0=0.04, kappa=1.5, theta=0.04, sigma=0.3, rho=-0.7`, r=3%, q=0, ATM call
S=K=100, fixed 8-point composite Gauss-Legendre quadrature so a discontinuity shows up as a wrong
number rather than an adaptive-quadrature warning):

| T | Albrecher | Heston 1993 | error | branch jumps |
|---|---|---|---|---|
| 0.5 | 6.24851241 | 6.24851241 | 0.00000000 | 0 |
| 1.0 | 9.19331831 | 9.19331831 | 0.00000000 | 0 |
| **2.0** | 13.79713514 | 13.73227054 | **−0.06486460** | 2 |
| 3.0 | 17.64536655 | 19.03506302 | **+1.38969648** | 2 |
| 5.0 | 24.16114908 | 21.51520946 | **−2.64593962** | 2 |
| 10.0 | 36.85527581 | 24.34697211 | **−12.50830370** | 5 |
| 15.0 | 46.68044534 | 28.34376934 | **−18.33667600** (**39.3% of the price**) | 8 |
| 20.0 | 54.67902254 | **nan** | — | 10 |
| 30.0 | 66.92393348 | **nan** | — | 15 |

- **First maturity that breaks: T = 2.0 years.** Not an exotic long-dated case — a two-year option.
- The Albrecher form has **0 branch jumps at T = 1, 10 and 30**.
- ✅ **The NaN, located:** at T=20 the 1993 `C_1` stops being finite at **phi = 165.3**, where
  `exp(+d*tau)` overflows to `inf`; `inf - inf` in the log term is the NaN. The Albrecher form's
  `exp(-d*tau)` decays instead and is finite everywhere on the grid.
- **The error is not monotone in T and it flips sign.** At T=3 the 1993 form is *too high* by 1.39,
  at T=5 *too low* by 2.65. You cannot calibrate your way past it or spot it from a smooth
  residual plot.

### 1b. ✅ Maturity is not the only axis — kappa*theta breaks it too

The offending log is multiplied by `a/sigma^2 = kappa*theta/sigma^2`. ✅ Measured at **T held at
2 years**, varying kappa only:

| kappa | kappa*theta | a/sigma² | Albrecher | Heston 1993 | error | jumps |
|---|---|---|---|---|---|---|
| 0.25 | 0.010 | 0.111 | 13.09409339 | 13.09409339 | 0.00000000 | 0 |
| 1.00 | 0.040 | 0.444 | 13.63792801 | 13.63792801 | 0.00000000 | 0 |
| **1.50** | 0.060 | 0.667 | 13.79713514 | 13.73227054 | −0.06486460 | 2 |
| **3.00** | 0.120 | 1.333 | 13.98683001 | 14.60576794 | **+0.61893793 (4.4%)** | 2 |
| 6.00 | 0.240 | 2.667 | 14.06427732 | 13.72909202 | −0.33518530 | 4 |
| 12.00 | 0.480 | 5.333 | 14.08154496 | 14.06108713 | −0.02045783 | 6 |

**Exact up to kappa = 1.00, wrong from kappa = 1.50 on, at an ordinary 2-year maturity.** A fast
mean-reverting calibration breaks the original formulation on short-dated options.

### 1c. ✅ QuantLib names both formulations, and agrees with the safe one to 1e-13

`ql.AnalyticHestonEngine` takes a `ComplexLogFormula`. ✅ Verified against QuantLib 1.43:

| Formulation | What it is | Agreement |
|---|---|---|
| `AnalyticHestonEngine.Gatheral` | the Albrecher/Gatheral continuous form | ✅ **worst \|diff\| 1.3e-13 out to T=30** vs this skill's implementation |
| `AnalyticHestonEngine.BranchCorrection` | the **1993 form plus a rotation counter** that repairs the log | ✅ agrees with Gatheral to the same tolerance |
| default constructor `AnalyticHestonEngine(model)` | ✅ **matches Gatheral** at every maturity tested — the safe default |

🚨 **`BranchCorrection` is not the naive 1993 form.** QuantLib repairs it by tracking how many times
the log wraps. The unrepaired textbook transcription — the one in most blog posts and most
first implementations — is what produces the table in §1. **If you write your own Heston, write the
Albrecher form.** If you must use the 1993 form, you owe it a rotation counter.

## 2. 🚨 A CRR tree does not converge — it oscillates, and "more steps" buys almost nothing

✅ Measured, European put, S=K=100, T=1, r=5%, q=2%, sigma=20%, Black-Scholes limit **6.330081**:

| steps | CRR | error | | steps | CRR | error |
|---|---|---|---|---|---|---|
| 50 | 6.291300 | −0.038781 | | 51 | 6.366360 | +0.036280 |
| 100 | 6.310665 | −0.019416 | | 101 | 6.348380 | +0.018299 |
| 200 | 6.320367 | −0.009714 | | 201 | 6.339271 | +0.009190 |
| 800 | 6.327651 | −0.002430 | | 801 | 6.332386 | +0.002305 |

**Even step counts sit below the limit, odd counts above it** — the strike falls between two
terminal nodes on one parity and near one on the other. Going 100 → 800 steps is **64× the work
for an 8.0× smaller error**, and the sign of your error is decided by the parity of a number you
picked arbitrarily.

✅ **The two fixes, measured against the same limit:**

| Method | Price | Error | vs n=100 alone |
|---|---|---|---|
| n = 100 | 6.310665 | −0.019416 | — |
| **average, ½(C(100)+C(101))** | 6.329523 | −0.000558 | ✅ **35× better** |
| **Richardson, 2·C(200) − C(100)** | 6.330068 | **−0.000012** | ✅ **1569× better** |
| 🚨 Richardson, 2·C(202) − C(101) | 6.292546 | −0.037535 | 🚨 **0.49× — i.e. WORSE than n=101** |

🚨 **Richardson extrapolation only works if both legs share the parity.** `2*C(2n) - C(n)` doubles
an even n to an even 2n, so the two errors are on the same side and the O(1/n) term cancels. Start
from an odd n and `2n` is even: the legs straddle the limit, and doubling the leg on the wrong side
*amplifies* the oscillation. This is a one-character bug that makes a "converged" tree worse than
the raw one.

✅ **American early exercise, same tree, 800 steps:** put **6.659528** vs European **6.327651** —
an early-exercise premium of **0.331877**. ✅ QuantLib's `QdFpAmericanEngine` prices the same
American put at **6.660684**, a difference of **−0.001156** (the tree's own discretisation error,
consistent with §2's magnitudes). `derivatives-pricing` §3 ranks the QuantLib American engines;
this is only the tree.

## 3. 🚨 Monte Carlo: paths shrink the error bar, steps shrink the bias, and the error bar cannot see the bias

The single most expensive Monte Carlo mistake is reading a small standard error as "converged".
✅ Measured, European call, exact price **9.227006**, seeded (`numpy.random.default_rng`):

| block | paths | steps | scheme | antithetic | price | std err | price − exact |
|---|---|---|---|---|---|---|---|
| paths | 10,000 | 1 | log | no | 9.0921 | 0.1371 | −0.1350 |
| paths | 40,000 | 1 | log | no | 9.0662 | 0.0688 | −0.1608 |
| paths | 160,000 | 1 | log | no | 9.1775 | 0.0345 | −0.0495 |
| steps | 40,000 | 12 | log | no | 9.1066 | 0.0686 | −0.1204 |
| steps | 40,000 | 52 | log | no | 9.2296 | 0.0686 | +0.0026 |
| anti | 40,000 | 1 | log | **yes** | 9.1793 | **0.0511** | −0.0477 |
| euler | 400,000 | 1 | **euler** | yes | 9.1032 | 0.0126 | **−0.1238** |
| euler | 400,000 | 4 | euler | yes | 9.2130 | 0.0153 | −0.0140 |
| euler | 400,000 | 16 | euler | yes | 9.2229 | 0.0160 | −0.0041 |
| euler | 400,000 | 64 | euler | yes | 9.2140 | 0.0162 | −0.0130 |

- ✅ **4× the paths halves the error bar:** 0.1371 → 0.0688 → 0.0345, ratios **1.99, 1.99** against
  the 2.00 that 1/sqrt(N) predicts.
- ✅ **52× the time steps does nothing to it:** 0.0688 → 0.0686, ratio **1.00**.
- ✅ **Antithetic variates at the same 40,000 paths:** 0.0688 → 0.0511, **1.35× smaller**. Modest —
  antithetics help a monotone payoff, and a call's kink caps the gain.
- 🚨 **The trap, in one row:** with an Euler scheme and **1 step**, 400,000 antithetic paths report a
  standard error of **0.0126** around a price that is **0.1238 too low — 10 standard errors of
  bias.** Nothing in the output says so. Four steps cut the bias by 9×; the error bar barely moved.

**Report both, always.** The standard error measures how much your *estimator* wobbles; it says
nothing about whether the thing it converges to is the price. For plain Black-Scholes use exact
lognormal increments (`scheme="log"`) and the bias is **zero at any step count** — Euler is only
needed when the SDE has no exact solution.

⚠️ **Compute the standard error over antithetic PAIRS, not over 2N correlated draws.** The pair
average is the i.i.d. unit; treating both legs as independent understates the error.

## 4. ✅ SABR (Hagan et al. 2002) — reproduced exactly against QuantLib

Hagan, Kumar, Lesniewski & Woodward (2002) eq. (2.17a)–(2.17c), lognormal implied vol,
F=100, T=1, alpha=0.25, beta=0.6, nu=0.4, rho=−0.25:

| K | 70 | 80 | 90 | 100 | 110 | 120 | 130 |
|---|---|---|---|---|---|---|---|
| vol | 0.081491 | 0.065137 | 0.050534 | **0.040078** | 0.040492 | 0.047187 | 0.054470 |

✅ **Worst \|mine − `ql.sabrVolatility`\| over all seven strikes: 0.0e+00** — bit-identical.
That is the documented reproduction; there is no ambiguity left about the formula.

- 🚨 **The formula is 0/0 at the money.** `z/x(z)` with `z -> 0` returns NaN if you transcribe it
  literally. Below `|z| ~ 1e-7` use the series `1 - rho*z/2 + (2 - 3*rho^2)*z^2/12`; that is the
  branch QuantLib takes, which is why the ATM strike also matches to 0.0e+00. The threshold is a
  choice and it differs between libraries — a smile that is smooth in one library and has a spike
  at K=F in another is this.
- ✅ **The general formula's ATM value and eq. (2.18) agree to 0.0e+00**
  (0.040077965390 both ways). If your ATM special case disagrees with your general formula, one of
  the two is mistyped.
- 🚨 **alpha is not the ATM vol.** alpha=0.25 here gives an ATM vol of **0.0401**, because the
  scaling is `alpha / F^(1-beta)` = 0.25/100^0.4. Seeding a calibration with "alpha ≈ ATM vol"
  diverges for any beta < 1. `derivatives-pricing` §5 measures the same trap from the calibration
  side; this is the formula side.

## 5. ✅ QuantLib's `"crr"` is not the textbook CRR probability

Found while benchmarking §2 against `ql.BinomialVanillaEngine(process, "crr", n)`. Same `u` and
`d`, different `p`:

| n | QuantLib | textbook `p=(e^((r−q)dt)−d)/(u−d)` | diff | `p=1/2 + (r−q−σ²/2)dt / (2σ√dt)` | diff |
|---|---|---|---|---|---|
| 100 | 6.3107897027 | 6.3106650879 | −1.2e-04 | 6.3107897027 | **+3.4e-13** |
| 101 | 6.3485033634 | 6.3483799531 | −1.2e-04 | 6.3485033634 | +3.4e-13 |
| 800 | 6.3276665617 | 6.3276509908 | −1.6e-05 | 6.3276665617 | +3.6e-13 |
| 801 | 6.3324012657 | 6.3323857138 | −1.6e-05 | 6.3324012657 | +5.8e-13 |

✅ **Reproduced to 5.8e-13** by using `pu = 1/2 + (r−q−sigma²/2)·dt / (2·sigma·sqrt(dt))`. Both
probabilities converge to Black-Scholes, so **neither is wrong** — but the textbook one misses
QuantLib by **1.2e-04 at n=100**, six times the tolerance a careful person would pick for a tree
test. ⚠️ This is verified by reproduction against the installed 1.43 wheel, not by reading the C++
(QuantLib ships no sdist — `derivatives-pricing` §3). `crr_price(..., prob="quantlib")` switches.

**Rule: when a tree disagrees with a library at O(1/n), suspect the probability convention before
the tree.**

## 6. ✅ Also measured while cross-checking

- 🚨 **`MCEuropeanEngine(..., antitheticVariate=True, requiredSamples=100_000)` counts PAIRS.**
  ✅ Inverting QuantLib's own `errorEstimate()` against the measured payoff standard deviation
  (13.826 plain, 7.287 per antithetic pair) implies **100,241 payoffs** in the plain run and
  **100,171 pairs** in the antithetic one — about **200,342 paths, not 100,000**. A timing
  comparison at equal `requiredSamples` is a 2× unfair comparison.
- ✅ QuantLib's `MCEuropeanEngine` reproduces §3's two facts in its own error estimate: **0.0437 at
  1 step vs 0.0436 at 52 steps** (steps do not move the error bar), **0.0230 antithetic**.
- ✅ Black-Scholes-Merton put-call parity residual: **+0.00e+00** (exactly zero in double
  precision) at S=K=100, T=1, r=5%, q=2%, sigma=20%; call **9.2270055082**, put **6.3300806275**.

## 7. What the script gives you

`scripts/option_models.py` is importable and library-free at import time (numpy + scipy only;
QuantLib is imported inside `quantlib_cross_checks`, so the module works on a bare install and the
demo degrades to a one-line notice):

| Function | Does |
|---|---|
| `bsm_price(S,K,T,r,q,sigma,flag)` | Black-Scholes-**Merton**, continuous dividend yield |
| `crr_price(..., steps, american, prob)` | CRR tree, `american=True` for early exercise, `prob="quantlib"` for §5 |
| `crr_smoothed(..., method)` | `"average"` or `"richardson"` — §2 |
| `crr_convergence(...)` | the oscillation table |
| `heston_price(..., formulation)` | `"albrecher"` (default) or `"heston1993"` |
| `heston_trap_scan` / `heston_kappa_scan` | §1 and §1b |
| `heston_cf_discontinuities` / `heston_first_nonfinite_phi` | branch-jump count, overflow location |
| `sabr_implied_vol` / `sabr_atm_vol` | Hagan (2.17) and (2.18) |
| `mc_european(..., scheme, antithetic, seed)` | price **and** standard error |
| `quantlib_cross_checks(...)` | every comparison in §1c, §4, §5, §6, or `None` |

## Where this sits

- `../../../fin-core/skills/derivatives-pricing/SKILL.md` — which library, Greek units (vega 100×,
  theta 365×), licences, and the ranked QuantLib American engines. **Read that first.**
- `../implied-vol-surface/SKILL.md` — fitting a whole smile, SVI, and the
  static no-arbitrage checks. This skill prices one option; that one builds the surface.
- `../../../fin-libraries/skills/lib-quantlib/SKILL.md` — 🚨 `Settings.instance().evaluationDate` is a
  global and a stale one returns **NPV exactly 0.0** with no warning. Set it before you construct
  anything, including everything in §1c.
- `../../../fin-libraries/skills/lib-vollib/SKILL.md` — `vollib` is the live package; `py_vollib` is a
  deprecated shim.
- `../../../fin-core/skills/options-backtesting/SKILL.md` — assignment, expiry, pin risk, and
  historical chains. A model price is not a fill.
