---
name: lib-quantlib
description: >-
  The only broadly-permissive, mature, full-coverage derivatives library in Python, whose global
  evaluationDate returns an NPV of exactly 0.0 with no warning once it is past expiry. TRIGGER -
  QuantLib, "import QuantLib as ql", QuantLib-SWIG, QuantLib-Python, Settings.instance(),
  evaluationDate, ql.Date, YieldTermStructureHandle, VanillaOption, AmericanExercise,
  QdFpAmericanEngine, FdBlackScholesVanillaEngine, BinomialVanillaEngine,
  BaroneAdesiWhaleyApproximationEngine, SABRInterpolation, SviSmileSection, thetaPerDay, "NPV is
  zero". Memory is stale on packaging and engine names - it is at 1.43 and ships 26 wheels but no
  sdist. SKIP for vanilla European IV and Greeks in pure Python (lib-vollib). SKIP for choosing
  between libraries - that is the domain skill's job.
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-04"
---

# QuantLib-Python

The only broadly-permissive, mature, full-coverage derivatives library in Python — everything else in this domain is
narrow, copyleft, or not open source.

| | |
|---|---|
| pip / import | **`QuantLib`** / `QuantLib` (conventionally `import QuantLib as ql`) |
| Version | **1.43** (2026-07-14) · 31 releases, first `1.16.1` on 2019-08-31 |
| Licence | **BSD-3-Clause** — GitHub says `NOASSERTION`; the custom BSD-derived text confuses the detector |
| Status | ✅ **actively maintained** — `lballabio/QuantLib` 7,573★, pushed 2026-09-03, 27 open issues; `QuantLib-SWIG` 399★, pushed 2026-09-01 |

## The trap that costs you money

🚨 **`ql.Settings.instance().evaluationDate` is a global, and moving it past an instrument's expiry returns NPV exactly
`0.0` — silently.** No warning, no exception; reproduced live. Set it before constructing anything and
**assert it is where you think it is**: a zero NPV in a book of thousands of positions reads as "worthless
option", not "wrong date".

## Packaging: wheel-or-nothing, and two names

✅ **The "QuantLib is a nightmare to install" advice is obsolete.** 1.43 ships **26 wheels** — `cp39-abi3` (one wheel
covers 3.9+) plus `cp313t`/`cp314t` free-threaded builds, for win_amd64, macOS, manylinux_2_28 and musllinux_1_2.
🚨 **But there is NO sdist at all**: `pip install QuantLib` is wheel-or-nothing, a **hard failure** anywhere else.

🚨 **`QuantLib-Python` on PyPI is frozen at 1.18 (2020-03-23) and is an empty forwarder** — summary *"Backward-
compatible meta-package for the QuantLib module"*, sole dependency `QuantLib`, one `py2.py3-none-any` wheel. It was
the original name (1.9, 2017 → 1.18); at **1.16.1 / 2019-08-31** the real bindings moved to `QuantLib`.

## The engine names in most tutorials do not exist

🚨 There is no `BaroneAdesiWhaleyEngine` and no `BjerksundStenslandEngine` — the real names are
**`BaroneAdesiWhaleyApproximationEngine`** and **`BjerksundStenslandApproximationEngine`**. Some
bindings also omit C++ methods; **`dir(ql)` is the authority, not a blog.** Measured across 9 American-exercise
engines:

| Engine | Result |
|---|---|
| **`QdFpAmericanEngine`** | ✅ the default choice — accurate and fast |
| **`FdBlackScholesVanillaEngine`** | ✅ the one to use when you need Greeks |
| `BaroneAdesiWhaleyApproximationEngine` | 🚨 **overprices the early-exercise premium by +118%** |
| `BinomialVanillaEngine` (CRR, 100 steps) | 🚨 **understates the premium by 79%** while the *total* price still looks fine — the error hides in the premium |
| `MCAmericanEngine` (Longstaff-Schwartz) | 🚨 worst on both axes: **37× slower, +0.71% error, biased high in-sample** |

🚨 **`delta()` raises** on `BaroneAdesiWhaleyApproximationEngine`, `QdPlusAmericanEngine`, `QdFpAmericanEngine` and the
MC engines — for Greeks on an American option use `FdBlackScholesVanillaEngine`. Other friction:
**`ql.Date(d, m, y)` is day-first**, unlike `datetime`; and **an empty handle fails at pricing time, not
construction time.**

## Vol surfaces are already in the box

People search PyPI for an "SVI package", find nothing maintained, and hand-roll a least-squares fit that produces
butterfly arbitrage. Verified present in 1.43 via `dir(ql)`: `SviSmileSection` (+ interpolated); `SABRInterpolation`,
`SabrSmileSection`, `sabrVolatility`, `shiftedSabrVolatility` (negative rates), `sabrGuess`,
`SabrSwaptionVolatilityCube`, `FdSabrVanillaEngine`; `NoArbSabrSmileSection`; `ZabrFullFdSmileSection`;
**`KahaleSmileSection`** (arbitrage repair); **`AndreasenHugeVolatilityInterpl`** + local-vol adapters (the under-used
gem — fits an arbitrage-free surface, yields local vol out the other side); `SmileSectionRNDCalculator` (risk-neutral
density, the butterfly detector); and `BlackVolatilitySurfaceDelta`/`DeltaVolQuote` for FX delta-quoted surfaces.

✅ SABR calibration verified: a synthetic 9-strike smile (F=100, T=1) from α=0.25, β=0.6, ν=0.4, ρ=−0.25 with β fixed
recovered all four parameters **exactly to 6 dp**. Friction found doing it: `SABRInterpolation` requires **`ql.Array`**,
not `ql.DoubleVector`, takes **no size argument**, and does not expose `.update()`, `.rmsError()` or `.maxError()` —
calibration runs in the constructor, so compute residuals yourself. 🚨 **The α trap:** α is **not** the ATM vol unless
β = 1; it scales as `α / F^(1−β)`, so α=0.20 with β=0.5 and F=100 gives an ATM vol of ~**2%**, and seeding
"α ≈ ATM vol" diverges for any β < 1.

## Greeks conventions

`theta()` is **annual**; `thetaPerDay()` is per calendar day. Vega and rho are **absolute** (per 1.0 of vol, per 1.0
of rate). 🚨 **vollib differs: vega ÷100, theta ÷365, rho ÷100.** `financepy` matches QuantLib — the three libraries
split two-to-one, and **mixing financepy and vollib Greeks in one book is a live 100× error** invisible in the number.

## Minimal correct call

```python
import QuantLib as ql

today = ql.Date(4, 9, 2026)                             # 🚨 DAY-first
ql.Settings.instance().evaluationDate = today
assert ql.Settings.instance().evaluationDate == today   # 🚨 assert, or expect a silent 0.0 NPV
dc   = ql.Actual365Fixed()
spot = ql.QuoteHandle(ql.SimpleQuote(100.0))
rTS  = ql.YieldTermStructureHandle(ql.FlatForward(today, 0.05, dc))
qTS  = ql.YieldTermStructureHandle(ql.FlatForward(today, 0.018, dc))
vTS  = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, ql.NullCalendar(), 0.20, dc))
process = ql.BlackScholesMertonProcess(spot, qTS, rTS, vTS)

opt = ql.VanillaOption(ql.PlainVanillaPayoff(ql.Option.Call, 100.0),
                       ql.AmericanExercise(today, today + ql.Period(1, ql.Years)))
opt.setPricingEngine(ql.QdFpAmericanEngine(process))       # ✅ accurate default
assert opt.NPV() > 0.0, "NPV 0.0 usually means evaluationDate is past expiry"

opt.setPricingEngine(ql.FdBlackScholesVanillaEngine(process))  # 🚨 delta() RAISES on QdFp
delta, theta_annual, theta_day = opt.delta(), opt.theta(), opt.thetaPerDay()
```

## See also

- `../../../fin-core/skills/derivatives-pricing/SKILL.md` — library choice and the licence traps
- `../../../fin-core/skills/derivatives-pricing/references/quantlib.md` — the source card
- `../../../fin-core/skills/derivatives-pricing/references/_library-matrix.md` — licences and the measured Greek scaling

## Where this sits

This file is the deep dive on **one** library and assumes the choice is already made.
For which library to pick, how it compares with the alternatives, and the traps that span
several of them, the entry point is the domain skill **`derivatives-pricing`** (`../../../fin-core/skills/derivatives-pricing/SKILL.md`).
