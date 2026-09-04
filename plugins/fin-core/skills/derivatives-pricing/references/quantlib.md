# QuantLib-Python

The only broadly-permissive, mature, full-coverage derivatives library in Python. Everything else in
this domain is either narrow, copyleft, or not open source.

| | |
|---|---|
| pip | **`QuantLib`** · import `QuantLib` (conventionally `import QuantLib as ql`) |
| Version | **1.43 (2026-07-14)** · 31 releases, first `1.16.1` on 2019-08-31 |
| Licence | **BSD-3-Clause** (declared; `LICENSE.TXT` is a BSD-3-style QuantLib licence) |
| Upstream | `lballabio/QuantLib` — 7,573★, 2,312 forks, pushed **2026-09-03**, 27 open issues |
| Bindings | `lballabio/QuantLib-SWIG` — 399★, pushed 2026-09-01 |

⚠️ GitHub reports `NOASSERTION` on both repos — the custom BSD-derived text confuses the detector.
**It is BSD-3-style.**

## Packaging

✅ **The "QuantLib is a nightmare to install" advice is obsolete.** 1.43 ships **26 wheels**:
`cp39-abi3` (one wheel covers Python 3.9+) plus `cp313t`/`cp314t` free-threaded builds, for
win_amd64, macOS x86_64/arm64, manylinux_2_28 x86_64/i686/aarch64, and musllinux_1_2.

🚨 **There is NO sdist at all.** `pip install QuantLib` is **wheel-or-nothing** — fine on the platforms
above, a **hard failure** anywhere else. There is no source fallback.

**Naming history** — a real source of confusion: **`QuantLib-Python` on PyPI is frozen at 1.18
(2020-03-23)**. Its own summary reads *"Backward-compatible meta-package for the QuantLib module"*,
its only dependency is `QuantLib`, and its single file is a `py2.py3-none-any` wheel — **an empty
forwarder.** `QuantLib-Python` was the original name (1.9, 2017-03-01 → 1.18); at **1.16.1 /
2019-08-31** the real bindings moved to `QuantLib`.

## What it is uniquely good at

**Conventions.** Term structures and curve bootstrapping, day counts, business-day calendars and roll
rules, exercise types, and a deep catalogue of pricing engines. **If your problem involves a
convention — accrual, settlement, roll, holiday — QuantLib already encodes it and your hand-rolled
version does not.**

## 🚨 The friction points that produce "why is my NPV zero"

- **`ql.Settings.instance().evaluationDate` is a global.** ✅ Reproduced live: moving it past the
  instrument's expiry returns **NPV exactly `0.0`, silently** — no warning, no exception. **Set it
  before constructing anything, and assert it is where you think it is.**
- **`ql.Date(d, m, y)` is day-first**, unlike `datetime`.
- Everything is an object: `Calendar`, `DayCounter`, `Quote`, `YieldTermStructureHandle`. Handles
  exist so curves can be re-linked; **an empty handle fails at pricing time, not construction time.**
- 🚨 **The engine names in most tutorials do not exist.** There is no `BaroneAdesiWhaleyEngine` or
  `BjerksundStenslandEngine` — they are **`BaroneAdesiWhaleyApproximationEngine`** and
  **`BjerksundStenslandApproximationEngine`**. **Check `dir(ql)` rather than trusting a blog.**
- Some Python bindings do not expose every C++ method. `dir()` is the authority.

## ✅ American exercise — measured across 9 engines

| Engine | Result |
|---|---|
| **`QdFpAmericanEngine`** | ✅ **the default choice** — accurate and fast |
| **`FdBlackScholesVanillaEngine`** | ✅ **use when you need Greeks** |
| `BaroneAdesiWhaleyApproximationEngine` | 🚨 **overprices the early-exercise premium by +118%** |
| `BinomialVanillaEngine` (CRR, 100 steps) | 🚨 **understates the premium by 79%** while the *total* price still looks fine — the error hides in the premium |
| `MCAmericanEngine` (Longstaff-Schwartz) | 🚨 worst on both axes: **37× slower, +0.71% error, biased high in-sample** |

🚨 **`delta()` raises** on `BaroneAdesiWhaleyApproximationEngine`, `QdPlusAmericanEngine`,
`QdFpAmericanEngine` and the MC engines. For Greeks on an American option use
`FdBlackScholesVanillaEngine`, or bump numerically and accept the noise.

## ✅ Volatility surfaces — it already ships them

The most misunderstood corner of the domain: people search PyPI for an "SVI package", find nothing
maintained, and hand-roll a least-squares fit that produces **butterfly arbitrage**.

Verified present in 1.43 via `dir(ql)`:

| Model | Symbols |
|---|---|
| **SVI** | `SviSmileSection`, `SviInterpolatedSmileSection` |
| **SABR** | `SABRInterpolation`, `SabrSmileSection`, `sabrVolatility`, `shiftedSabrVolatility` (negative rates), `sabrFlochKennedyVolatility`, `sabrGuess`, `SabrSwaptionVolatilityCube`, `FdSabrVanillaEngine`, `FdmSabrOp` |
| **No-arbitrage SABR** | `NoArbSabrSmileSection`, `NoArbSabrInterpolatedSmileSection` |
| **ZABR** | `ZabrFullFdSmileSection`, `ZabrLocalVolatilitySmileSection` (+ interpolated variants) |
| **Arbitrage repair** | `KahaleSmileSection` |
| **Arbitrage-free surface** | `AndreasenHugeVolatilityInterpl`, `AndreasenHugeLocalVolAdapter` — **the under-used gem**: fits an arbitrage-free surface directly and yields local vol out the other side |
| **Butterfly detector** | `SmileSectionRNDCalculator` (risk-neutral density) |
| FX delta-quoted | `BlackVolatilitySurfaceDelta`, `DeltaVolQuote` |

✅ **SABR calibration verified working.** Fitting a synthetic 9-strike smile (F=100, T=1) generated
from α=0.25, β=0.6, ν=0.4, ρ=−0.25 with β fixed recovered **all four parameters exactly to 6 dp**.

**API friction found doing it:**
- `SABRInterpolation` requires **`ql.Array`**, not `ql.DoubleVector` —
  `TypeError: in method 'new_SABRInterpolation', argument 2 of type 'Array const &'`.
- Signature is `SABRInterpolation(x, y, t, forward, alpha, beta, nu, rho, alphaIsFixed, betaIsFixed,
  nuIsFixed, rhoIsFixed)` — **no size argument**, unlike the C++ iterator form.
- `.update()`, `.rmsError()`, `.maxError()` are **not exposed** in the Python bindings — calibration
  runs in the constructor; **compute residuals yourself.** ❓ may vary by version.

🚨 **The SABR α trap:** **α is not the ATM vol unless β = 1.** It scales as `α / F^(1−β)`, so α=0.20
with β=0.5 and F=100 gives an ATM vol of about **2%**. Calibrations seeded with "α ≈ ATM vol" diverge
or land in a false minimum for any β < 1.

## Greeks conventions

QuantLib's `theta()` is **annual**; `thetaPerDay()` is per calendar day. Its vega and rho are in
**absolute** units (per 1.0 of vol, per 1.0 of rate).

🚨 **`vollib` differs: vega ÷100, theta ÷365, rho ÷100.** `financepy` matches QuantLib. **Mixing
financepy and vollib Greeks in one book is a live 100× error** — see `_library-matrix.md` for the
measured table.

## Where to learn it

Luigi Ballabio's *Implementing QuantLib* and the `quantlib-python-docs` community documentation are
the practical references; the C++ doxygen is the authority on what exists. The Python bindings are
generated by SWIG from the C++ headers, so **when the Python docs are thin, read the C++ class.**
