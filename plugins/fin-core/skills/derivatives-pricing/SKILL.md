---
name: derivatives-pricing
description: >-
  Price options and fixed income, and get the Greeks and conventions right. TRIGGER - option
  pricing, implied volatility, IV, Greeks, delta, gamma, vega, theta, rho; Black-Scholes,
  binomial, Monte Carlo, American exercise, early exercise, exotics; volatility surface, smile,
  skew, SVI, SABR, ZABR; QuantLib, vollib, py_vollib, financepy, rateslib, optionlab; option
  chains; or yield curve, discount factor, curve bootstrapping, day count, swap and bond pricing.
  Greek scaling differs by 100x and 365x between the three common libraries, and one popular
  fixed-income package is not open source. SKIP for choosing an option DATA vendor
  (market-data-sourcing).
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-03"
---

# Derivatives pricing

Two things will silently corrupt your numbers here: **Greek scaling conventions that differ by 100×
and 365× between libraries**, and **a licence trap in the most-recommended fixed-income package.**

## 1. Pick a library

| Task | Use | Licence |
|---|---|---|
| **Anything serious, broad coverage** | **QuantLib** 1.43 | ✅ **BSD-3 — the only broadly-permissive mature full-coverage option** |
| Vanilla IV + Greeks, fast and simple | **`vollib`** 1.0.11 | MIT |
| Jaeckel's "Let's Be Rational" IV | `lets_be_rational` 1.1.2 | MIT |
| Option strategy P&L / payoff analysis | `optionlab` 1.8.5 | 🚨 **GPL-3.0** — its PyPI licence field is **blank**; the repo says GPL-3.0 |
| Broad instrument coverage, readable source | `financepy` 1.1.2 | 🚨 **GPL-3.0-or-later — copyleft** |
| Fixed income / swaps / curves | `rateslib` | 🚨🚨 **NOT open source — see §2** |
| SABR only | `pysabr` | ⚠️ stale since 2022-04-21 |
| Asian/exotic closed forms | `pyfeng` | 🚨 **GPL-2.0** |

⚠️ **A blank PyPI licence field does not mean permissive.** `optionlab` declares nothing on PyPI and
is GPL-3.0 in its repository. Check the repo's LICENSE file, not the package metadata.

🚨 **`py_vollib` is now a dead shim.** As of 1.0.12 (2026-06-01) it contains **zero library code** —
4 files, summary *"Deprecated transition package for vollib"*, depending on `vollib>=1.0.11`.
Importing it emits a deprecation warning at runtime. **The canonical name is `vollib`.** Every
tutorial written before mid-2026 points at the wrong package.

⚠️ **Star counts mislead here:** `vollib/vollib` has 1,020★ but has not been pushed since 2023-06-05;
`vollib/py_vollib` (433★, pushed 2026-05-29) is the **active** repo. PyPI `vollib` is the maintained
artifact.

🚨 **`py_vollib_vectorized` is doubly broken** — no PyPI release in 5.5 years, and it monkey-patches
`py_vollib` internals that have since moved to `vollib`. Vectorize `vollib` yourself or run the LBR
loop under numba.

**Also stale/dead:** `mibian` (2016, GPL), `tf-quant-finance` (2022, effectively abandoned),
`pyesg` (2021), `quantlib-python` (the old PyPI name, frozen at 1.18 / 2020 — an empty forwarder).

## 2. 🚨 rateslib is not open source

✅ Verified by reading the repo's own licence files: **dual-licensed "Source-Available
Non-Commercial" + a paid "Commercial Subscription Licence"** (Siffrorna Technology Limited). It has
**never** been OSS — v1.1.1 (2024-03-21) shipped under CC BY-NC-ND 4.0.

**Any commercial use requires a paid subscription.** This is the single biggest licensing trap in
this domain, and it is easy to miss because the package installs cleanly from PyPI with abi3 wheels.

For permissive fixed income, use **QuantLib** term structures.

## 3. QuantLib

✅ **v1.43 (2026-07-14), BSD-3, wheels for cp39-abi3 through cp314t.** The historic "QuantLib is a
nightmare to install" advice is **obsolete** — one abi3 wheel covers Python 3.9+.

🚨 **QuantLib ships NO sdist** — `pip install QuantLib` is wheel-or-nothing. Fine on
win/macOS/manylinux/musllinux (x86_64 + arm64); a **hard failure** on any platform without a wheel.

**What it is uniquely good at:** term structures and curve bootstrapping, day-count conventions,
business-day calendars and roll rules, exercise types, and a deep catalogue of pricing engines. If
your problem involves a *convention* — accrual, settlement, roll — QuantLib already encodes it and
your hand-rolled version does not.

**The friction points that produce "why is my NPV zero":**
- 🚨 `ql.Settings.instance().evaluationDate` is a **global**. ✅ Reproduced live: moving it past the
  instrument's expiry returns **NPV exactly `0.0`, silently** — no warning, no exception. Set it
  before constructing anything, and assert it is where you think it is.
- `ql.Date(d, m, y)` is **day-first**, unlike `datetime`.
- Everything is an object: `Calendar`, `DayCounter`, `Quote`, `YieldTermStructureHandle`. Handles
  exist so curves can be re-linked; an **empty handle fails at pricing time**, not construction time.
- 🚨 **The engine names in most tutorials do not exist.** There is no `BaroneAdesiWhaleyEngine` or
  `BjerksundStenslandEngine` — they are `BaroneAdesiWhaleyApproximationEngine` and
  `BjerksundStenslandApproximationEngine`. Check `dir(ql)` rather than trusting a blog.

### American exercise — measured across 9 engines

✅ Benchmarked against a high-accuracy reference:

| Engine | Verdict |
|---|---|
| **`QdFpAmericanEngine`** | ✅ **the default choice** — accurate and fast |
| **`FdBlackScholesVanillaEngine`** | ✅ use when you need Greeks (see below) |
| `BaroneAdesiWhaleyApproximationEngine` | 🚨 **overprices the early-exercise premium by +118%** |
| `BinomialVanillaEngine` (CRR, 100 steps) | 🚨 **understates the premium by 79%** while the *total* price still looks fine — the error hides in the premium |
| `MCAmericanEngine` (Longstaff-Schwartz) | 🚨 **worst on both axes**: 37× slower, +0.71% error, and biased **high in-sample** |

🚨 **`delta()` raises** on `BaroneAdesiWhaleyApproximationEngine`, `QdPlusAmericanEngine`,
`QdFpAmericanEngine` and the MC engines. If you need Greeks from an American option, use
`FdBlackScholesVanillaEngine` — or bump numerically and accept the noise.

## 4. 🚨 Greek scaling — measured, and it will cost you 100×

✅ Cross-checked `vollib` against QuantLib on identical inputs (S=K=100, T=365d ACT/365, r=5%, q=0,
σ=20%, call):

| Greek | vollib | QuantLib | ratio | what it means |
|---|---|---|---|---|
| price | 10.45058357 | 10.45058357 | 1 | agree |
| delta | 0.63683065 | 0.63683065 | 1 | agree |
| gamma | 0.01876202 | 0.01876202 | 1 | agree |
| **vega** | 0.37524035 | 37.52403469 | **100×** | vollib is **per 1 vol point** (÷100) |
| **theta** | −0.01757268 | −6.41402755 | **365×** | vollib is **per calendar day** (÷365) |
| **rho** | 0.53232482 | 53.23248155 | **100×** | vollib is **per 1% rate** (÷100) |

✅ QuantLib's `theta()` is **annual**; its `thetaPerDay()` returns −0.01757268, matching vollib exactly.

🚨 **`financepy` matches QuantLib, NOT vollib.** So the three most common libraries split two-to-one
on convention, and **mixing `financepy` and `vollib` Greeks in one book is a live 100× error.**
There is no way to tell from the number alone which convention produced it.

**The precise errors:**
- **Vega 100× off** — textbook vega is dV/dσ with σ in absolute units (0.20); vollib reports per vol
  point. Mixing a vollib vega into a textbook formula is a 100× P&L error.
- **Theta 365× off, and calendar vs trading days.** vollib divides by 365; desks often quote per
  *trading* day (÷252). Same Greek, 1.45× apart, **indistinguishable from the number alone.**
- **Theta sign** must be negative for a long option. Some libraries report dV/dT (positive) rather
  than dV/dt. Sanity check: long ATM call theta < 0.
- **Put delta** must be negative, in [−1, 0]. A positive `delta('p', ...)` means the flag was ignored.
- 🚨 **`vollib.black_scholes` has NO `q` argument** — it is Black-Scholes without dividends. **Using
  it on SPY biases every delta and every implied vol.** Use `black_scholes_merton` (takes `q`) or
  `black` (forward-based).
- **Analytical vs numerical Greeks:** vollib ships both. The numerical ones bump by a fixed amount
  and **degrade badly near expiry**. Use analytical unless you are validating.

**Always state the convention alongside any Greek you report.** A vega without units is not a number.

## 5. Vol surfaces — QuantLib already has this

The most misunderstood corner of the domain: people search PyPI for an "SVI package", find nothing
maintained, and hand-roll a least-squares fit that produces **butterfly arbitrage**.

✅ Verified present in QuantLib 1.43 via `dir(ql)`:

| Model | Symbols |
|---|---|
| **SVI** | `SviSmileSection`, `SviInterpolatedSmileSection` |
| **SABR** | `SABRInterpolation`, `SabrSmileSection`, `sabrVolatility`, `shiftedSabrVolatility` (negative rates), `sabrFlochKennedyVolatility`, `sabrGuess`, `SabrSwaptionVolatilityCube`, `FdSabrVanillaEngine` |
| **No-arbitrage SABR** | `NoArbSabrSmileSection`, `NoArbSabrInterpolatedSmileSection` |
| **ZABR** | `ZabrFullFdSmileSection`, `ZabrLocalVolatilitySmileSection` |
| **Arbitrage-free repair** | `KahaleSmileSection` |
| **Arbitrage-free surface** | `AndreasenHugeVolatilityInterpl` (+ local-vol adapters) — the under-used gem |
| **Butterfly-arbitrage detector** | `SmileSectionRNDCalculator` (risk-neutral density) |
| FX delta-quoted | `BlackVolatilitySurfaceDelta`, `DeltaVolQuote` |

✅ **SABR calibration verified working** — fitting a synthetic 9-strike smile (F=100, T=1) generated
from α=0.25, β=0.6, ν=0.4, ρ=−0.25 with β fixed recovered all four parameters **exactly to 6 dp**.

**API friction found while doing that:**
- `SABRInterpolation` requires **`ql.Array`**, not `ql.DoubleVector`.
- Python signature is `SABRInterpolation(x, y, t, forward, alpha, beta, nu, rho, alphaIsFixed,
  betaIsFixed, nuIsFixed, rhoIsFixed)` — **no size argument**, unlike the C++ iterator form.
- `.update()`, `.rmsError()`, `.maxError()` are **not exposed** in the Python bindings — calibration
  runs in the constructor; compute residuals yourself.

🚨 **The SABR α trap, measured:** **α is not the ATM vol unless β = 1.** It scales as `α / F^(1−β)` —
so α=0.20 with β=0.5 and F=100 gives an ATM vol of about **2%**, not 20%. Calibrations seeded with
"α ≈ ATM vol" diverge or land in a false minimum for any β < 1.

⚠️ `pysabr` (624★, MIT) is the best-known standalone and is **stale since 2022-04-21**; it also
depends on **`falcon`, a web framework**, as a hard runtime dependency. GitHub's other SVI/SABR repos
are 2–77★ hobby projects. **Use QuantLib.**

## 5b. financepy — trust the maths, not the API

✅ Its numerics **agree with QuantLib to ~1e-6**, so the "poor quality" reputation is only half
earned. The problems are packaging, not mathematics:
- An unfinished module rename left **duplicated modules** (`discount_curve_flat` *and*
  `flat_discount_curve`), so the widely-documented **`DiscountCurveFlat` now raises `ImportError`**.
- A module is literally named `yield_curve_XXX`.
- It prints an **unsolicited banner to stdout on import**.
- 🚨 And it is **GPL-3.0-or-later**.

## 6. 🚨 Options data traps

These corrupt results before any pricing code runs. ✅ **Measured on a live SPY chain:**

| Finding | Number |
|---|---|
| Call `lastPrice` values **outside** the bid-ask | **54.6%** |
| Contracts stale by >1 hour | **100%** (max **8.3 days**) |
| Put-call parity residual noise, `last` vs `mid` | `last` was **53.5× noisier** |

**That is the quantified argument for mid-only.** Never compute IV from `lastPrice`.

- **IV from a stale or one-sided quote is garbage.** Use the **mid**, and reject quotes where the
  spread exceeds a threshold or the bid is 0. Illiquid strikes produce IVs that are pure noise.
- 🚨 **Yahoo's `impliedVolatility` field is untrustworthy.** ✅ Measured: call vs put IV at the *same
  near-ATM strike* differed by a mean of **4.82 vol points**, and at one strike Yahoo's IV sat
  **outside the entire bid-ask IV range**. Compute your own from the mid.
- **yfinance gives no Greeks and no historical chains** — its `date` argument selects the *expiry*,
  not a historical as-of date. It is a live snapshot only.
- **Put-call parity is your data-quality check.** For matched strikes/expiries,
  `C − P ≈ S·e^(−qT) − K·e^(−rT)`. Large violations mean stale quotes, a wrong dividend assumption,
  or a mismatched timestamp — not an arbitrage.
- **Dividends and borrow.** Getting `q` wrong biases every delta and every IV. American calls on
  dividend payers can be optimally exercised early — European pricing understates them.
- **Expiry conventions.** AM vs PM settlement, third-Friday vs weeklies vs month-end. An off-by-one
  on expiry changes T and therefore every Greek.
- **Moneyness vs strike interpolation.** Interpolate in log-moneyness or delta space, not raw strike
  — raw-strike interpolation across a moving spot produces a surface that shifts with the underlying.
- **yfinance option chains are a snapshot with no history** — usable for a live look, useless for
  backtesting. Historical chains are the genuinely hard (and paid) part: CBOE DataShop, ORATS,
  OptionMetrics IvyDB, Polygon, Databento.

## 7. Reference files

`references/<library>.md` for versions, licences, exact signatures and measured comparisons.

## Per-library deep dives

The optional `fin-libraries` plugin carries a dedicated skill for each library below. Load one
only after this skill has told you which library you want:

- **`lib-quantlib`** — quantlib
- **`lib-vollib`** — vollib
