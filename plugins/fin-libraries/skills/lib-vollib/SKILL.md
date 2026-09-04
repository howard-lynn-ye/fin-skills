---
name: lib-vollib
description: >-
  Machine-precision implied volatility with no bracketing, behind a package name restructured in
  2026 - py_vollib is now a DEAD SHIM with four files and zero library code, and every pre-2026
  tutorial installs it. TRIGGER - vollib, py_vollib, py_vollib_vectorized, lets_be_rational,
  "Let's Be Rational", Jaeckel, black_scholes, black_scholes_merton, implied_volatility,
  greeks.analytical, BelowIntrinsicException, AboveMaximumException, "py_vollib is deprecated", or
  implied volatility returning 0.0. Memory is stale on the package name, on the C++ dependency (it
  is now pure Python) and on the Greek scaling. SKIP for American exercise, exotics, curves or
  conventions (lib-quantlib). SKIP when the question is WHICH library to choose rather than how to
  use this one - that belongs to the domain skill.
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-04"
---

# vollib

Machine-precision implied volatility with no bracketing, in pure Python — the pick for vanilla European pricing, IV
and Greeks. MIT, no compiler, best-in-class solver.

| | |
|---|---|
| pip / import | ✅ **`vollib`** / `vollib` |
| Version | **1.0.11** (2026-06-01) · `py3-none-any` wheel — ✅ **pure Python, no compiler** · `>=3.9,<4` |
| Licence | **MIT** |
| Status | ✅ active — `vollib/py_vollib` 433★, **1 open issue**, pushed 2026-05-29. ⚠️ `vollib/vollib` has **1,020★** but no push since 2023-06-05 — **the stars are on the stale repo** |

## The trap that costs you money

🚨 **`py_vollib` is a DEAD SHIM.** As of 1.0.12 (2026-06-01) its PyPI summary reads verbatim
*"Deprecated transition package for vollib."* — **4 files, zero library code**, sole dependency
`vollib>=1.0.11,<2.0.0`. Importing it emits *"py_vollib is deprecated and will be removed in a future release; please
import from vollib"*. Old code works **for now**; write new code against
**`vollib`**.

The real library is PyPI **`vollib`**: `vollib/black/`, `vollib/black_scholes/`, `vollib/black_scholes_merton/`, each
with `implied_volatility.py` and `greeks/{analytical,numerical}.py`, plus `vollib/ref_python/` and a `py_vollib/`
shim.

🚨 **`py_vollib_vectorized` is a trap on top of the trap.** 0.1.1, released **2021-02-28 — no PyPI release in 5.5
years** — and it **monkey-patches `py_vollib` internals** that have since **moved to `vollib`**, so it patches a
gutted package. ❓ The exact breakage was not reproduced here, but the mechanism is documented and the abandonment
verified. **Vectorize `vollib` yourself, or run the LBR loop under numba.**

⚠️ It is now **pure Python** — historically a SWIG wrapper around Jaeckel's C++, now a port with `numba` as an
**optional** accelerator: peak speed traded for install reliability, the opposite bargain from QuantLib's
wheel-or-nothing.

## `black_scholes` has no `q` — using it on SPY biases everything

🚨 `vollib.black_scholes` is Black-Scholes **without dividends**. There is **no `q` argument to omit or mis-set** — the
dividend yield simply is not in the model. Feed it a dividend-paying underlying (SPY, most single names) and **every
delta and every implied vol is biased**, silently and consistently. Use
**`black_scholes_merton`** (takes `q`) or **`black`** (forward-based).

## Greeks scaling — measured against QuantLib

Identical inputs (S=K=100, T=365d ACT/365, r=5%, q=0, σ=20%, call):

| Greek | vollib | QuantLib | ratio | what vollib reports |
|---|---|---|---|---|
| price / delta / gamma | 10.45058357 / 0.63683065 / 0.01876202 | identical | 1 | agree |
| **vega** | 0.37524035 | 37.52403469 | 🚨 **÷100** | per **1 vol point** |
| **theta** | −0.01757268 | −6.41402755 | 🚨 **÷365** | per **calendar day** |
| **rho** | 0.53232482 | 53.23248155 | 🚨 **÷100** | per **1% rate** |

QuantLib's `theta()` is annual; its `thetaPerDay()` returns −0.01757268 and matches vollib exactly.
**`financepy` follows QuantLib, not vollib** — the three common libraries split two-to-one, and
**mixing financepy and vollib Greeks in one book is a live 100× P&L error** invisible in the number
itself. Theta carries a second trap: vollib divides by **365 (calendar)** while desks often quote per
**trading** day (÷252) — same Greek, **1.45× apart**. Assert that a long ATM call theta is
**negative** and a put delta lies in **[−1, 0]**; a positive put delta means the flag was ignored.

## The silent-zero IV, and why LBR beats a hand-rolled solver

🚨 A deep ITM call (K=20, S=100, T=1) has time value below double resolution, and `implied_volatility` returns
**`0.0` with no exception raised** — same when the price underflows to exactly 0. **Always guard `iv == 0.0`**
and drop deep ITM/OTM strikes; they carry essentially no recoverable vol information. Below discounted intrinsic you
get a proper `BelowIntrinsicException` / `AboveMaximumException`; it is the *silent* case that reaches your dataframe.
⚠️ Prices marginally above intrinsic are hypersensitive: at K=100, T=1, forward-intrinsic 4.87706, a price of 4.87806
—
**one tenth of a cent higher** — gives IV = **1.74%**.

Round-trip tests (price at σ=0.20, solve back) recover σ to **machine precision across the whole surface with no
bracketing**, including deep OTM calls priced at 4.2e-140. Newton-Raphson divides by vega, which → 0 for deep ITM/OTM
and short expiry; Brent/bisection needs a bracket, and `[1e-6, 5.0]` fails silently when true IV exceeds it (crypto,
expiry-day gamma). LBR works in normalised-price space with a rational-cubic guess and a proven-convergent Householder
step: fixed iteration count, no bracket, no division by vega. ⚠️ vollib ships
**analytical and numerical** greeks (`greeks/analytical.py`, `greeks/numerical.py`); the numerical ones bump
by a fixed amount and
**degrade badly near expiry**. Use analytical unless you are validating.

## Minimal correct call

```python
from vollib.black_scholes_merton import black_scholes_merton
from vollib.black_scholes_merton.implied_volatility import implied_volatility
from vollib.black_scholes_merton.greeks.analytical import delta, vega, theta

S, K, t, r, q, sigma, flag = 100.0, 100.0, 1.0, 0.05, 0.018, 0.20, "c"
price = black_scholes_merton(flag, S, K, t, r, sigma, q)   # 🚨 q REQUIRED; black_scholes has none
iv    = implied_volatility(price, S, K, t, r, q, flag)
assert iv > 0.0, "silent-zero: no recoverable vol (deep ITM/OTM or underflowed price)"

v  = vega(flag, S, K, t, r, sigma, q) * 100    # -> per 1.00 of vol, the textbook convention
th = theta(flag, S, K, t, r, sigma, q) * 365   # -> annual, comparable with QuantLib theta()
assert th < 0 and -1.0 <= delta("p", S, K, t, r, sigma, q) <= 0.0
```

⚠️ Never compute IV from a `lastPrice`: on a live SPY chain **54.6%** of call `lastPrice` values sat outside the
bid-ask, and `last`-vs-`mid` put-call-parity residuals were **53.5× noisier**. Deps: `lets-be-rational`,
`cody-special`, `piecewise-rational`, numpy≥1.20, pandas≥2.0, scipy≥1.10.

## See also

- `../../../fin-core/skills/derivatives-pricing/SKILL.md` — library choice and the licence traps
- `../../../fin-core/skills/derivatives-pricing/references/vollib.md` — the source card
- `../../../fin-core/skills/derivatives-pricing/references/_library-matrix.md` — licences and the measured Greeks table
