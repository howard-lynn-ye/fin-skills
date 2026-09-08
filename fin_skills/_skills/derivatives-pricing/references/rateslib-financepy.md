# rateslib · financepy

Two libraries people install after searching "Python fixed income" or "Python derivatives library",
neither of which you can ship. **rateslib is not open source at all**; **financepy is GPL-3.0**.
Both are on PyPI with ordinary-looking wheels, which is exactly why this card exists.

| | `rateslib` | `financepy` |
|---|---|---|
| version | **2.7.1 (2026-04-04)** · 19 releases ✅ | **1.1.2 (2026-08-21)** · 50 releases ✅ |
| GitHub | `attack68/rateslib` — 355★, 68 forks, 25 open, pushed 2026-05-20 ✅ | `domokane/FinancePy` — **3,127★**, 433 forks, 50 open, pushed 2026-09-02 ✅ |
| Licence | 🚨🚨 **NOT open source** — GitHub reports `NOASSERTION` ✅ | 🚨 **GPL-3.0-or-later** ✅ (PyPI `license_expression`; GitHub `GPL-3.0`) |
| Python | `>=3.10` ✅ · Rust core, `cp310-abi3` wheels | `>=3.10,<3.14` ✅ · pure-Python `py3-none-any` |
| Verdict | ⚠️ real development, **bus-factor 1**, commercially gated | ✅ genuinely active again; numerics sound, API rough |

Verified 2026-09-04 via PyPI JSON and the GitHub REST API; licence terms read from the repository's
own licence files.

## 🚨 rateslib is not open source — and never was

The repo root carries **three** licence files — `LICENCE`, `COMMERCIAL_LICENCE`,
`COMMERCIAL_LICENCE_ADDENDUM1` ✅ (confirmed live via the GitHub contents API). `LICENCE` declares
itself *"Dual Licensing – Source-Available Non-Commercial Licence and Commercial Subscription
Licence"*, © Siffrorna Technology Limited, and states plainly: **"This software is not open source."**

The free branch is *Personal and Educational Use Only*. It grants **view, download and run** for
non-commercial purposes and explicitly withholds the right to distribute, modify, create derivative
works, sublicense, redistribute in source or binary, **incorporate into any other software, library,
service or product**, or **use the software to provide services to third parties**. "Commercial" is
defined broadly — any use directed toward commercial advantage or business operations, direct or
indirect. The paid branch is a **subscription** licensed per single internal user or single
designated system environment, for internal use only, whose **rights terminate when the subscription
lapses**.

🚨 **There is no earlier permissive version to fall back to.** v1.1.1 (2024-03-21) was
**CC BY-NC-ND 4.0** — Attribution-**NonCommercial**-**NoDerivatives**. The 1.6.0 → 2.5.0 releases
left the PyPI licence field *empty*, which reads as permissive to a scanner and is not. The custom
dual licence lands in the metadata at 2.6.0 (2026-02-15).

⚠️ The repo also ships `.ai-opt-out` and `.aiignore`, and a 2026-04-09 commit *"make additional
explicit clauses regarding AI use on code"* — the author has opted out of AI training and use.
Reproducing its code from model memory is not a workaround.

**Practical rule:** rateslib may be an **optional, user-installed, user-licensed backend**. It may
never be a dependency of anything you distribute or run commercially. A permissive-looking abi3
wheel on PyPI is not a licence.

What you would be buying: purpose-built IRS / cross-currency-swap / FX-swap / bond and bond-future
pricing with full curveset construction under market-standard optimisers, and **automatic
differentiation in the Rust core** giving exact delta and cross-gamma instead of bumped finite
differences — which QuantLib's Python bindings have no first-class equivalent for. Excellent docs.
⚠️ Two years of releases, one principal author, last push ~3.5 months before this audit.

## 🚨 financepy — GPL-3.0, correct numerics, broken names

✅ **The numerics are sound.** Benchmarked against QuantLib (S=K=100, T=1y, r=5%, q=0, σ=20%,
European call): NPV differs by **−7.95e-06**, delta by −6e-08, gamma ≈ 0, vega ≈ 0, theta by −9e-08,
rho by +2e-06. The residual is a year-fraction/compounding convention difference, not an error. The
folklore that financepy "needs cross-checking" is overstated for vanilla European equity options.

🚨 **An unfinished rename makes the documented class name an `ImportError`.**
`financepy.market.curves` ships *both* `discount_curve_flat` and `flat_discount_curve`, *both*
`discount_curve_ns` and `ns_discount_curve`, *both* `discount_curve_poly` and `poly_discount_curve`,
*both* `discount_curve_pwf` and `pwf_discount_curve` ✅. The exported class is **`FlatDiscountCurve`**,
while most existing code, tutorials and documentation say **`DiscountCurveFlat` — which now raises
`ImportError`.** A module literally named `yield_curve_XXX` also ships in 1.1.2 ✅.

🚨 **It prints an ASCII banner to stdout on import** ✅ — version plus a "DISTRIBUTED FREE AND WITHOUT
WARRANTY" box. Hostile to any CLI, service, or machine-parsed pipeline. Suppress it or accept the
noise in your logs.

🚨 **Greeks convention: financepy matches QuantLib (raw / annualised), NOT vollib.** financepy vega
= 37.52 vs vollib 0.375 — **100× apart**; financepy theta = −6.414 (annual) vs vollib −0.0176 (per
day) — **365× apart** ✅. Mixing the two in one codebase silently produces greeks off by two orders
of magnitude. See `vollib.md`.

⚠️ **Inconsistent return shapes** — `crr_tree_val_avg(...)` returns a dict
(`{'value','delta','gamma','theta'}`) while `EquityVanillaOption.value(...)` returns a float ✅.

⚠️ **A very tight dependency box**: `numba<0.63,>=0.62.1`, `llvmlite<0.46,>=0.45.0`,
`numpy<2.4,>=2.3.5`, `pandas<=2.4,>=2.3.3` ✅. Expect resolver conflicts with any other numba user.

⚠️ **The licence trap is the academic tone.** GPL-3.0-or-later is declared correctly; people miss it
because the project reads like a textbook. Importing it into a distributed application makes that
application GPL.

## Minimal correct financepy call

```python
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):     # 🚨 suppress the import banner
    from financepy.market.curves import FlatDiscountCurve      # 🚨 NOT DiscountCurveFlat
    from financepy.utils.date import Date

curve = FlatDiscountCurve(Date(1, 1, 2024), 0.05)
# Greeks are RAW/ANNUAL (QuantLib convention): vega is per 1.00 of vol, theta per year.
# 🚨 Do not compare these to vollib without rescaling by 100 (vega) and 365 (theta).
```

## Routing

| Need | Use | Licence |
|---|---|---|
| Anything redistributed, or any commercial deployment | **QuantLib** | BSD-3 ✅ — `quantlib.md` |
| Options IV / greeks, permissive, fast | **vollib** | MIT ✅ — `vollib.md` |
| Rates-desk curve construction with exact AD risk | rateslib | 🚨🚨 paid subscription; never a dependency |
| Reading the theory next to the code, broad asset coverage | financepy | 🚨 GPL-3.0 — learning only |

`_library-matrix.md` has the full metadata and licence grid, including the vollib repackaging and the
other blank-licence traps. **A blank PyPI licence field is not permissive** — read the repo.
