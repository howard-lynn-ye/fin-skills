# vollib

Machine-precision implied volatility with no bracketing, in pure Python — behind a package name that
was **restructured in 2026**, so every tutorial written before mid-2026 installs the wrong thing.

| | |
|---|---|
| pip | ✅ **`vollib`** · import `vollib` |
| Version | **1.0.11 (2026-06-01)** · MIT · `py3-none-any` wheel — ✅ **pure Python, no compiler** |
| Python | `>=3.9,<4` |
| Deps | `lets-be-rational`, `cody-special`, `piecewise-rational`, numpy≥1.20, pandas≥2.0, scipy≥1.10 |
| Upstream | 🚨 `vollib/py_vollib` — 433★, **1 open issue**, pushed **2026-05-29** ← the *active* repo |
| ⚠️ | `vollib/vollib` — **1,020★** but pushed **2023-06-05**. **The stars are on the stale repo.** |

## 🚨 The 2026 restructuring — get the package name right

| PyPI name | Latest | Date | What is actually inside |
|---|---|---|---|
| ✅ **`vollib`** | **1.0.11** | 2026-06-01 | **The real library.** `vollib/black/`, `vollib/black_scholes/`, `vollib/black_scholes_merton/`, each with `implied_volatility.py` + `greeks/{analytical,numerical}.py`; plus `vollib/ref_python/` and a `py_vollib/` compat shim |
| 🚨 `py_vollib` | 1.0.12 | 2026-06-01 | ✅ Summary reads verbatim *"Deprecated transition package for vollib."* **4 files, zero library code.** Its only dependency is `vollib>=1.0.11,<2.0.0` |
| `lets_be_rational` | 1.1.2 | 2026-05-29 | The canonical **Jaeckel** *Let's Be Rational* implementation. Ships `lets_be_rational/` + legacy `py_lets_be_rational/` |
| `py_lets_be_rational` | 1.1.2 | 2026-05-30 | Legacy name, same content |
| `cody-special` | 1.0.0 | 2026-01-13 | Split-out: Cody's erf / normal CDF. MIT |
| `piecewise-rational` | 1.0.0 | 2026-01-13 | Split-out: Delbourgo–Gregory shape-preserving rational cubic. MIT |

✅ At runtime `import py_vollib.black_scholes` emits *"py_vollib is deprecated and will be removed in
a future release; please import from vollib"*. **Old code keeps working — for now.** Write new code
against `vollib`.

✅ **It is now pure Python.** Historically `vollib` was a SWIG wrapper around Jaeckel's C++; the
current line is a pure-Python port with `numba` as an **optional** accelerator (deliberately not a
hard dependency). Peak speed traded for install reliability — the opposite bargain from
`./quantlib.md`, which is wheel-or-nothing.

## 🚨 `py_vollib_vectorized` is a trap

`py_vollib_vectorized` **0.1.1, released 2021-02-28 — no PyPI release in 5.5 years.** Repo
`marcdemers/py_vollib_vectorized`, 160★, 4 open issues, last pushed 2024-12-02.

Its mechanism is the problem: it **monkey-patches `py_vollib` internals**, pinning
`py-vollib>=1.0.1` and `numba>=0.51`. But `py_vollib` is now a **4-file shim with no library code** —
the internals it patches have **moved to `vollib`**. Patching a gutted package is the failure mode,
not a hypothetical.

❓ Not installed here to confirm the exact breakage, so treat the specific error as unverified — but
the mechanism is documented and the abandonment is verified. **Prefer vectorising `vollib` yourself,
or running the LBR loop under numba.**

## Why LBR beats a hand-rolled solver

Round-trip tests (price at σ=0.20, solve back) recover σ to **machine precision across the whole
surface, with no bracketing** — including deep OTM calls priced at 4.2e-140.

- **Newton–Raphson divides by vega.** Vega → 0 for deep ITM/OTM and short expiry, so the step
  explodes exactly where you most need care.
- **Brent/bisection needs a bracket.** `[1e-6, 5.0]` fails silently when true IV is above it (crypto,
  meme stocks, expiry-day gamma) and burns 40–80 iterations.
- **LBR** works in normalised-price space with a rational-cubic initial guess and a
  proven-convergent Householder step: fixed iteration count, no bracket, no division by vega.

🚨 **The silent-zero failure mode.** A deep ITM call (K=20, S=100, T=1) has time value below double
resolution, and `implied_volatility` returns **`0.0` with no exception raised**. Same when the price
underflows to exactly 0. **Always guard `iv == 0.0`** and drop deep ITM/OTM strikes rather than
trusting the number — they carry essentially no recoverable vol information. Below discounted
intrinsic you get a proper `BelowIntrinsicException` / `AboveMaximumException`; it is the *silent*
case that reaches your dataframe.

⚠️ Prices marginally above intrinsic are hypersensitive: at K=100, T=1, forward-intrinsic 4.87706, a
price of 4.87806 (**one tenth of a cent higher**) gives IV = **1.74%**.

## 🚨 Greeks scaling — measured against QuantLib

Identical inputs (S=K=100, T=365d ACT/365, r=5%, q=0, σ=20%, call):

| Greek | vollib | QuantLib | ratio | what vollib reports |
|---|---|---|---|---|
| price | 10.45058357 | 10.45058357 | 1 | agree |
| delta | 0.63683065 | 0.63683065 | 1 | agree |
| gamma | 0.01876202 | 0.01876202 | 1 | agree |
| **vega** | 0.37524035 | 37.52403469 | 🚨 **÷100** | per **1 vol point** |
| **theta** | −0.01757268 | −6.41402755 | 🚨 **÷365** | per **calendar day** |
| **rho** | 0.53232482 | 53.23248155 | 🚨 **÷100** | per **1% rate** |

✅ QuantLib's `theta()` is annual; its `thetaPerDay()` returns −0.01757268 and matches vollib exactly.

**Mixing a vollib vega into a textbook formula is a 100× P&L error** and nothing will warn you — the
number is plausible either way. Theta has a second trap on top: vollib divides by **365 (calendar)**
while desks often quote per **trading** day (÷252). Same Greek, **1.45× apart**, indistinguishable
from the value alone.

Sanity checks worth asserting: long ATM call theta must be **negative**; put delta must be in
**[−1, 0]** (a positive `delta('p', ...)` means the flag was ignored).

## 🚨 `black_scholes` has no `q` — using it on SPY biases everything

`vollib.black_scholes` is Black–Scholes **without dividends**. There is **no `q` argument to omit or
mis-set** — the dividend yield simply is not in the model.

Feed it a dividend-paying underlying (SPY, most single names) and **every delta and every implied
vol is biased**, silently and consistently. Use **`black_scholes_merton`** (takes `q`) or **`black`**
(forward-based, dividends already in the forward).

```python
from vollib.black_scholes_merton import black_scholes_merton
from vollib.black_scholes_merton.implied_volatility import implied_volatility
from vollib.black_scholes_merton.greeks.analytical import delta, vega, theta

S, K, t, r, q, sigma, flag = 100.0, 100.0, 1.0, 0.05, 0.018, 0.20, "c"

price = black_scholes_merton(flag, S, K, t, r, sigma, q)   # 🚨 q is REQUIRED here — no default
iv    = implied_volatility(price, S, K, t, r, q, flag)
assert iv > 0.0, "silent-zero: no recoverable vol (deep ITM/OTM or underflowed price)"

v = vega(flag, S, K, t, r, sigma, q) * 100     # → per 1.00 of vol, the textbook convention
th = theta(flag, S, K, t, r, sigma, q) * 365   # → annual, to compare with QuantLib theta()
```

⚠️ **Analytical vs numerical greeks:** vollib ships both (`greeks/analytical.py`,
`greeks/numerical.py`). The numerical ones bump by a fixed amount and **degrade badly near expiry**.
Use analytical unless you are validating.

## Where it fits

✅ **The pick for vanilla European pricing, IV and Greeks** — MIT, pure Python, best-in-class solver.
Reach for `./quantlib.md` for anything beyond vanilla: American exercise, exotics, term structures,
or conventions (accrual, settlement, roll, holidays). The full field, including the licence traps in
`optionlab` (GPL-3.0 with a blank PyPI licence field) and `pyfeng` (GPL-2.0), is in
`./_library-matrix.md`.
