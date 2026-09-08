# Derivatives libraries — verified metadata and licences

Verified 2026-09-03. Much of this was **measured by installing the libraries and running the
comparisons**, not read from docs — those rows are marked ✅ measured.

## Metadata and licences

| Package | Version | Released | Licence | Verdict |
|---|---|---|---|---|
| **`QuantLib`** | **1.43** | 2026-07-14 | **BSD-3-Clause** | ✅ the only broadly-permissive, mature, full-coverage option |
| `QuantLib-Python` | 1.18 | **2020-03-23** | BSD | 🔴 **an empty forwarder** — its only dep is `QuantLib`, its only file a `py2.py3-none-any` wheel |
| **`vollib`** | **1.0.11** | 2026-06-01 | MIT | ✅ the real options library — 47 files, pure Python |
| `py_vollib` | 1.0.12 | 2026-06-01 | MIT | 🔴 **4 files, zero library code.** *"Deprecated transition package for vollib."* Emits a deprecation warning at import |
| `lets_be_rational` | 1.1.2 | 2026-05-29 | MIT | ✅ canonical Jaeckel LBR |
| `py_lets_be_rational` | 1.1.2 | 2026-05-30 | MIT | legacy name, same content |
| `cody-special` | 1.0.0 | 2026-01-13 | MIT | split-out: Cody erf / normal CDF |
| `piecewise-rational` | 1.0.0 | 2026-01-13 | MIT | split-out: Delbourgo-Gregory rational cubic |
| `py_vollib_vectorized` | 0.1.1 | **2021-02-28** | — | 🔴 **doubly broken** — no release in 5.5 yrs, and it monkey-patches `py_vollib` internals that moved to `vollib` |
| `financepy` | 1.1.2 | 2026-08-21 | 🚨 **GPL-3.0-or-later** | numerics sound, API broken (below) |
| `rateslib` | 2.7.1 | 2026-04-04 | 🚨🚨 **NOT open source** | source-available non-commercial + paid commercial |
| `optionlab` | 1.8.5 | 2026-08-10 | 🚨 **GPL-3.0** (PyPI field is **blank**) | strategy payoff analysis |
| `pyfeng` | — | — | 🚨 **GPL-2.0** | Asian/exotic closed forms |
| `pysabr` | 0.4.1 | **2022-04-21** | MIT | ⚠️ stale 4.4 yrs; depends on **`falcon`, a web framework** |
| `mibian` | 0.1.3 | **2016-03-12** | GPL | 🔴 dead 10 yrs |
| `tf-quant-finance` | 0.0.1.dev34 | **2022-08-19** | Apache-2.0 | 🔴 effectively abandoned |
| `pyesg` | 0.1.5 | **2021-01-21** | MIT | 🔴 stale |

⚠️ **A blank PyPI licence field is not permissive.** `optionlab` declares nothing on PyPI and is
GPL-3.0 in its repository. Always read the repo's LICENSE.

⚠️ **Star counts mislead:** `vollib/vollib` has 1,020★ but no push since **2023-06-05**;
`vollib/py_vollib` (433★, pushed 2026-05-29) is the **active** repo. PyPI `vollib` is the maintained
artifact.

## QuantLib packaging

- PyPI dist name **`QuantLib`**, import `QuantLib`. 31 releases, first `1.16.1` on 2019-08-31.
- 🚨 **No sdist at all** — 26 wheels only. `pip install QuantLib` is wheel-or-nothing: fine on
  win_amd64, macOS x86_64/arm64, manylinux_2_28, musllinux_1_2; a **hard failure** anywhere else.
- Wheels: `cp39-abi3` (covers 3.9+) plus `cp313t`/`cp314t` free-threaded builds.
- Upstream `lballabio/QuantLib` 7,573★ (pushed 2026-09-03); bindings `QuantLib-SWIG` 399★.
- GitHub reports `NOASSERTION` on both — the custom BSD-derived text confuses the detector. It is
  BSD-3-style.

**Naming history:** `QuantLib-Python` was the original PyPI name (1.9, 2017-03-01 → 1.18, 2020).
At 1.16.1 / 2019-08-31 the real bindings moved to `QuantLib`; the old name is now a stub.

## ✅ Measured: Greeks scaling

S=K=100, T=365d ACT/365, r=5%, q=0, σ=20%, call:

| Greek | vollib | QuantLib | financepy | ratio (vollib:QL) |
|---|---|---|---|---|
| price | 10.45058357 | 10.45058357 | ≈QL | 1 |
| delta | 0.63683065 | 0.63683065 | ≈QL | 1 |
| gamma | 0.01876202 | 0.01876202 | ≈QL | 1 |
| **vega** | 0.37524035 | 37.52403469 | **≈QL** | **100** |
| **theta** | −0.01757268 | −6.41402755 | **≈QL** | **365** |
| **rho** | 0.53232482 | 53.23248155 | **≈QL** | **100** |

QuantLib's `thetaPerDay()` returns −0.01757268, matching vollib exactly.

🚨 **financepy follows QuantLib, not vollib** — the three common libraries split two-to-one, and
**mixing financepy and vollib Greeks is a live 100× error** that is invisible in the number itself.

- vollib's vega/rho are **per vol point / per 1% rate** (÷100); theta is **per calendar day** (÷365).
- Desks often quote theta per **trading** day (÷252) — same Greek, 1.45× apart.
- 🚨 **`vollib.black_scholes` has no `q` argument.** Use `black_scholes_merton` (takes `q`) or
  `black` (forward-based). Using `black_scholes` on SPY biases every delta and every IV.
- vollib ships analytical **and** numerical Greeks; the numerical ones **degrade badly near expiry**.

## ✅ Measured: American exercise, 9 engines

| Engine | Result |
|---|---|
| **`QdFpAmericanEngine`** | ✅ the default choice |
| **`FdBlackScholesVanillaEngine`** | ✅ use when you need Greeks |
| `BaroneAdesiWhaleyApproximationEngine` | 🚨 **overprices the early-exercise premium by +118%** |
| `BinomialVanillaEngine` (CRR, 100 steps) | 🚨 **understates the premium by 79%** while total price looks fine |
| `MCAmericanEngine` (Longstaff-Schwartz) | 🚨 worst on both axes: **37× slower, +0.71% error, biased high in-sample** |

🚨 **`delta()` raises** on BAW, QdPlus, QdFp and the MC engines.
🚨 **`BaroneAdesiWhaleyEngine` and `BjerksundStenslandEngine` do not exist** — the real names end in
`ApproximationEngine`. Most tutorials name them wrong.

## Vol surfaces — QuantLib already ships them

✅ Verified present in 1.43 via `dir(ql)`: `SviSmileSection`, `SviInterpolatedSmileSection`;
`SABRInterpolation`, `SabrSmileSection`, `sabrVolatility`, `shiftedSabrVolatility` (negative rates),
`sabrFlochKennedyVolatility`, `sabrGuess`, `SabrSwaptionVolatilityCube`, `FdSabrVanillaEngine`;
`NoArbSabrSmileSection`; `ZabrFullFdSmileSection`, `ZabrLocalVolatilitySmileSection`;
`KahaleSmileSection` (arbitrage repair); `AndreasenHugeVolatilityInterpl` (+ local-vol adapters);
`SmileSectionRNDCalculator` (risk-neutral density — the butterfly-arbitrage detector);
`BlackVolatilitySurfaceDelta`, `DeltaVolQuote` (FX delta-quoted).

✅ **SABR calibration verified:** a synthetic 9-strike smile (F=100, T=1) generated from α=0.25,
β=0.6, ν=0.4, ρ=−0.25 with β fixed recovered all four parameters **exactly to 6 dp**.

**API friction found doing it:**
- `SABRInterpolation` requires **`ql.Array`**, not `ql.DoubleVector`.
- Signature is `SABRInterpolation(x, y, t, forward, alpha, beta, nu, rho, alphaIsFixed,
  betaIsFixed, nuIsFixed, rhoIsFixed)` — **no size argument**, unlike the C++ iterator form.
- `.update()`, `.rmsError()`, `.maxError()` are **not exposed** in the Python bindings — calibration
  runs in the constructor; compute residuals yourself.

🚨 **The α trap:** α is **not** the ATM vol unless β=1 — it scales as `α / F^(1−β)`. α=0.20 with
β=0.5 and F=100 gives an ATM vol of ~**2%**. Seeding "α ≈ ATM vol" diverges for any β<1.

## financepy — trust the maths, not the API

✅ Numerics agree with QuantLib to ~1e-6. The problems are packaging:
- An unfinished rename left **duplicated modules** (`discount_curve_flat` *and*
  `flat_discount_curve`), so the widely-documented **`DiscountCurveFlat` raises `ImportError`**.
- A module is named `yield_curve_XXX`.
- It prints an **unsolicited banner to stdout on import**.
- 🚨 GPL-3.0-or-later.

## ✅ Measured: options data quality (live SPY chain)

| Finding | Number |
|---|---|
| Call `lastPrice` outside the bid-ask | **54.6%** |
| Contracts stale >1 hour | **100%** (max **8.3 days**) |
| Put-call parity residual noise, `last` vs `mid` | `last` **53.5× noisier** |
| Yahoo `impliedVolatility`, call vs put at same near-ATM strike | mean **4.82 vol points** apart; at one strike **outside the entire bid-ask IV range** |

**Never compute IV from `lastPrice`.** yfinance gives **no Greeks and no historical chains** — its
`date` argument selects the *expiry*, not an as-of date.

## Options data vendors

| Vendor | Coverage | Greeks/IV | Note |
|---|---|---|---|
| ORATS | back to **2007** | ✅ | $199–$899/mo |
| CBOE DataShop | **Jan 2012+** | ⚠️ greeks are a paid "Calcs" add-on | |
| Databento | raw | 🚨 **none** — no greeks, no IV | $/GB + $125 credits |
| Tradier | live chains | ✅ `bid_iv`/`mid_iv`/`ask_iv` **courtesy of ORATS** | |
| Massive (was Polygon.io) | options aggregates | — | 🚨 **`polygon.io/pricing` 301s to `massive.com/pricing`**; `polygon-api-client` has had no release since **2025-10-30**, the announcement date. ⚠️ price tiers are aggregator-sourced, indicative only |
| OptionMetrics IvyDB | academic standard | ✅ | institutional |
