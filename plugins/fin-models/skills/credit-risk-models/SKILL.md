---
name: credit-risk-models
description: >-
  Estimate a default probability and price credit, and keep the two probabilities apart - the
  risk-neutral one that prices and the physical one that forecasts. TRIGGER - Merton model,
  structural credit model, KMV, distance to default, asset value and asset volatility from equity,
  N(-d2), solve the two Merton equations; hazard rate, intensity, reduced form, survival
  probability, constant hazard, credit curve bootstrapping; CDS par spread, premium leg, protection
  leg, risky PV01, RPV01, accrual on default, "spread = lambda times one minus recovery", implied
  hazard from a CDS spread, recovery assumption, 40% recovery; risk-neutral vs physical default
  probability, rating agency default table, "my CDS spread is too low", credit spread from a bond
  price, expected loss, CVA default probability. SKIP for option pricing and Greeks
  (option-pricing-models, derivatives-pricing), for interest-rate curves and short-rate models
  (term-structure-models), and for portfolio risk and VaR (portfolio-and-risk).
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-09"
---

# Credit risk models

**Two model families each produce a number called "the probability of default", and they are not
the same number.** Merton's `N(−d2)` is **risk-neutral**; a rating-agency default table is
**physical**. Using one where the other belongs is the most expensive mistake in this domain, and
neither one carries a label.

Every figure below is printed by `scripts/credit_models.py` (runs in **1.8 s**; QuantLib optional,
imported inside `quantlib_cds_cross_check`). ✅ Measured means this file produced it on 2026-09-09
with QuantLib 1.43, scipy 1.13.0, Python 3.11.3.

> **The rule:** `N(−d2)` is risk-neutral; a rating-table default rate is physical. **Price with
> the first, forecast with the second**, and never quote a hazard without its recovery.

## 1. ✅ Merton (1974), reproducing a textbook example

Equity is a call on the firm's assets struck at the face value of debt. Two equations pin the two
unobservables:

    E      = V N(d1) - D e^{-rT} N(d2)
    sigma_E = N(d1) sigma_V V / E

⚠️ **The reference:** Hull, *Options, Futures, and Other Derivatives*, the worked example in the
credit-risk chapter on using equity prices to estimate default probabilities. **Inputs:
E = 3, sigma_E = 80%, D = 10 payable in 1 year, r = 5%.** The book reports V = 12.40,
sigma_V = 0.2123, N(−d2) = 12.7%, debt worth 9.40 against a promised PV of 9.51, and an expected
loss of about 1.2%. Those published values are secondhand; everything in the "solved" column is
✅ measured here.

| quantity | ⚠️ Hull | ✅ solved here |
|---|---|---|
| asset value V | 12.40 | **12.395387** |
| asset volatility sigma_V | 0.2123 | **0.212305** |
| d2 | 1.1408 | **1.140826** |
| **risk-neutral PD = N(−d2)** | **12.7%** | **12.6971%** |
| market value of debt | 9.40 | **9.395387** |
| PV of the promised payment | 9.51 | **9.512294** |
| expected loss | about 1.2% | **1.2290%** |
| implied credit spread | — | **123.7 bp** |

✅ Solve residual **1.1e-16**, and pushing (V, sigma_V) back through the forward map returns
equity **3.0000000000** and equity vol **0.8000000000**. The reproduction is exact to every digit
the book prints.

### 1b. ✅ The solver fails loudly, which is the property that matters

A two-equation nonlinear solve can converge to a plausible wrong root, and that is the only
failure mode you would not notice. ✅ Measured over a **270-point grid** (E ∈ {0.1 … 50},
sigma_E ∈ {20% … 300%}, D ∈ {1, 10, 100}, T ∈ {0.25, 1, 5}):

- **263 solved and round-tripped** through the forward map to 1e-8
- **7 raised** — all at **sigma_E = 300% with T = 5** (an equity vol that high over five years is a
  firm the two equations cannot separate)
- 🚨 **0 returned a wrong root**

**Keep that property if you rewrite this.** Start from `V = E + D e^{−rT}` and
`sigma_V = sigma_E·E/V`, and **check the residual of both equations before you use the answer.**

⚠️ **The second equation is an instantaneous identity, not a law.** `sigma_E = N(d1)·sigma_V·V/E`
holds at a point in time; equity vol is not constant as V moves, so an equity vol estimated over a
year and an instantaneous relation are being married by assumption. KMV-style implementations
iterate on a time series of V instead. This skill implements the textbook two-equation form, which
is what the reproduction above is of.

## 2. 🚨 The same firm, two default probabilities

`d2` carries the drift. Under the risk-neutral measure the drift is `r`; the physical
distance-to-default uses the asset drift `mu`. ✅ Measured on the Merton firm from §1:

| asset drift mu | distance to default | **PD physical** | **PD risk-neutral** | ratio |
|---|---|---|---|---|
| 5.00% (= r) | 1.1408 | **12.697%** | 12.697% | **1.00** |
| 8.00% | 1.2821 | 9.990% | 12.697% | **1.27** |
| 10.00% | 1.3763 | 8.436% | 12.697% | **1.51** |
| 15.00% | 1.6118 | **5.350%** | 12.697% | **2.37** |

✅ **mu = r reproduces N(−d2) exactly** — that is the check that the two formulas are the same
formula. **Every mu > r gives a smaller physical PD**, and at a 15% asset drift the risk-neutral
probability is **2.37× the physical one**. The gap is the credit risk premium; it is not an error
in either number.

## 3. Constant hazard and the exact CDS — and what `lambda(1 − R)` costs

Survival is `exp(−lambda·t)`. ✅ Measured, **lambda = 2%, R = 40%, r = 3%, 5-year CDS, quarterly
premiums in arrears with accrual on default** — everything in closed form, nothing discretised:

| leg | value per unit notional |
|---|---|
| survival to 5y | **0.904837** |
| protection leg `(1−R)·lambda/(r+lambda)·(1 − e^{−(r+lambda)T})` | **0.05308781** |
| premium annuity | **4.39639204** |
| + accrual on default | **0.01103692** |
| = **risky PV01** | **4.40742896** |
| **exact par spread** | **120.4507 bp** |
| `lambda(1 − R)` | 120.0000 bp — **error −0.4507 bp** |
| exact par spread **without** accrual on default | 120.7531 bp (**+0.3024 bp**) |

✅ **The approximation error is a constant fraction of the spread, and it is bigger than you
would guess:**

| lambda | 0.5% | 1.0% | 2.0% | 5.0% | 10.0% | 20.0% |
|---|---|---|---|---|---|---|
| exact (bp) | 30.11 | 60.23 | 120.45 | 301.13 | 602.25 | 1204.47 |
| `lambda(1−R)` (bp) | 30.00 | 60.00 | 120.00 | 300.00 | 600.00 | 1200.00 |
| error (bp) | −0.11 | −0.23 | −0.45 | −1.13 | −2.25 | **−4.47** |
| **error (%)** | **−0.37** | **−0.37** | **−0.37** | **−0.37** | **−0.37** | **−0.37** |

**The approximation UNDERSTATES the spread, by exactly 0.37% at every hazard.** A constant
relative error is a clue, and it has a clean explanation.

### ✅ `lambda(1 − R)` is not an approximation at all — it is the continuous-premium spread

With a continuously-paid premium the annuity is `(1 − e^{−cT})/c` with `c = r + lambda`, and the
protection leg is `(1−R)·lambda/c·(1 − e^{−cT})`. **The ratio is `(1−R)·lambda` exactly** — `r`,
`T` and the exponentials all cancel. ✅ Reproduced to 12 digits.

🚨 **So the entire gap is the premium FREQUENCY**, not the hazard model. Paying in arrears defers
the annuity, which raises the fair spread. ✅ Measured at lambda = 2%, R = 40%, r = 3%, T = 5:

| premiums per year | 1 | 2 | **4 (market)** | 12 | 52 | 365 |
|---|---|---|---|---|---|---|
| exact (bp) | 121.8120 | 120.9030 | **120.4507** | 120.1501 | 120.0346 | 120.0049 |
| without accrual on default (bp) | 123.0506 | 121.5126 | 120.7531 | 120.2503 | 120.0577 | 120.0082 |
| **vs `lambda(1−R)`** | **+1.8120** | +0.9030 | **+0.4507** | +0.1501 | +0.0346 | **+0.0049** |

It converges to **120.0000 bp** as the premium goes continuous. **Accrual on default gives part of
that annuity back** (−0.30 bp at quarterly, −1.24 bp at annual), which is why the two columns
converge from opposite sides. Use `lambda(1−R)` to sanity-check an order of magnitude; the
0.45 bp it misses at quarterly is real money on size, and 1.81 bp on an annual-pay contract is
not a rounding error.

## 4. 🚨 Recovery is an input, and 40% is a convention

The market quotes a **spread**. The hazard you back out of it depends entirely on the recovery you
assumed. ✅ Measured, **market 5-year spread 100 bp, r = 3%**:

| recovery R | **implied lambda (exact)** | `s/(1−R)` | 5-year PD |
|---|---|---|---|
| 0% | 0.9963% | 1.0000% | 4.859% |
| 20% | **1.2453%** | 1.2500% | 6.037% |
| 40% | **1.6604%** | 1.6667% | 7.967% |
| 60% | **2.4906%** | 2.5000% | 11.709% |
| 80% | 4.9813% | 5.0000% | **22.047%** |

🚨 **The same 100 bp quote implies a hazard 2.00× higher at R = 60% than at R = 20%**, and a
5-year default probability of 22.0% instead of 6.0% at R = 80%. **A hazard rate without its
recovery assumption is not a number.** The standard 40% is a convention inherited from the CDS
market, not a measurement of anything about the issuer.

**The saving grace:** spread and recovery move together in the pricing, so the *price* of a
protection leg is much less sensitive than the hazard is. That is why quoting (lambda, R) as a
pair matters more than getting R right.

## 5. 🚨 The trap: pricing with a historical default probability

Rating-agency and internal default tables are **physical**. CDS spreads, bond prices, and CVA all
need the **risk-neutral** probability. ✅ Measured on the same Merton firm at an asset drift of
mu = 10%:

| | physical | risk-neutral |
|---|---|---|
| 1-year PD | **8.436%** | **12.697%** |
| implied hazard | 8.813% | 13.579% |
| **1-year CDS fair spread (R = 40%)** | **532.1 bp** | **819.8 bp** |

🚨 **−287.7 bp.** You would sell protection at 532 bp on a name whose fair spread is 820 bp, and
your model would show a profit.

🚨 **And then you would mark it.** Revaluing the *fair* risk-neutral contract with the physical
hazard reports a value of **−2.67% of notional that does not exist**. The position looks
mispriced against you, so the natural reaction is to add more of it.

**The direction is systematic, not random.** Physical PD < risk-neutral PD for any firm with a
positive risk premium (§2), so this trap always makes credit look cheap and always makes you
short protection.

## 6. ✅ Cross-check against QuantLib 1.43

Same flat hazard, flat rate, 5-year quarterly schedule:

| | spread | vs this file |
|---|---|---|
| `FlatHazardRate.survivalProbability(5)` | **0.904837** | matches `exp(−0.02·5)` |
| `MidPointCdsEngine.fairSpread()` | **120.4522 bp** | mine 120.4507, **+0.0015 bp** |
| `IsdaCdsEngine.fairSpread()` | **120.4368 bp** | **−0.0139 bp** |
| `MidPointCdsEngine`, `settlesAccrual=False` | **120.7544 bp** | mine 120.7531, **+0.0013 bp** |

✅ **Agreement to about a hundredth of a basis point**, and the residual is explained rather than
waved at: the engines accrue coupons on **ACT/365** dates and place the default time at the
**midpoint** of each quarter, against the exact integral computed here. ⚠️ Note the two QuantLib
engines differ from each other by **0.0154 bp** on identical inputs — the ISDA model has its own
conventions. If you are reconciling to a counterparty, the engine is part of the trade
description.

## 7. What the script gives you

`scripts/credit_models.py` — scipy only at import (`brentq`, `fsolve`, `norm`).

| Function | Does |
|---|---|
| `merton_solve(E, sigma_E, D, r, T)` | §1, returns a frozen `MertonResult` with `residual` |
| `merton_equity(V, sigma_V, D, r, T)` | the forward map — the round-trip check |
| `merton_solver_scan()` | §1b, the 270-point robustness grid |
| `distance_to_default(V, sigma_V, D, mu, T)` | §2, physical DD and N(−DD) |
| `survival_probability` / `hazard_from_default_probability` | the constant-hazard pair |
| `cds_legs(hazard, recovery, r, T, freq, accrual_on_default)` | §3, every leg in closed form |
| `cds_par_spread` / `implied_hazard` / `cds_mark_to_market` | price, invert, revalue |
| `approximation_error_table` | §3, what `lambda(1−R)` costs |
| `recovery_sensitivity(spread, recoveries, r, T)` | §4 |
| `measure_trap(...)` | §5, the same firm priced both ways |
| `quantlib_cds_cross_check(...)` | §6, or `None` |

## Where this sits

- `../term-structure-models/SKILL.md` — the discount curve every leg above is priced on, and the
  day-count and compounding conventions that shift a 5-year rate by 17.68 bp. A CDS spread quoted
  against the wrong curve convention is wrong before the credit model starts.
- `../../../fin-libraries/skills/lib-quantlib/SKILL.md` — 🚨 `Settings.instance().evaluationDate`
  is a global and a stale one gives **NPV exactly 0.0**; §6 sets it before building any curve.
  Also `DefaultProbabilityTermStructureHandle`, `FlatHazardRate`, and the CDS engines.
- `../../../fin-core/skills/derivatives-pricing/SKILL.md` — 🚨 **`rateslib` is not open source**
  and `financepy` is GPL-3.0; QuantLib is the permissive route for CDS.
- `../../../fin-core/skills/portfolio-and-risk/SKILL.md` — portfolio credit, correlation and
  capital. This skill is one name at a time.
- `../../../fin-market-data/skills/fundamental-and-macro-data/SKILL.md` — where the equity value, the
  debt face value and the spread quotes come from, and their point-in-time behaviour. A Merton PD
  computed from a restated balance sheet is a backtest artefact.
- `../option-pricing-models/SKILL.md` — Merton's equity call is a Black-Scholes option, so
  everything there about `N(d1)`, `N(d2)` and dividend yields applies to the firm's assets too.
