---
name: monte-carlo-methods
description: >-
  Make a Monte Carlo converge to the RIGHT number - variance reduction with measured factors,
  Longstaff-Schwartz for American options, scrambled-Sobol QMC, and the discretisation bias a
  standard error cannot see. TRIGGER - variance reduction, antithetic variates, control
  variate, stratified sampling, importance sampling, "how many paths do I need", standard
  error of a Monte Carlo price; Longstaff-Schwartz, LSM, least-squares Monte Carlo, American
  option by simulation, regression on in-the-money paths, continuation value; quasi-Monte
  Carlo, QMC, Sobol, scipy.stats.qmc, scrambling, low discrepancy, "power of 2" warning;
  discretely monitored barrier, continuity correction, "my error bar is tiny but the price is
  wrong", "more paths did not help". SKIP for the model itself - Heston, SABR, trees, Euler
  bias on a GBM (option-pricing-models), for VaR and expected shortfall from simulated
  portfolios (risk-measures-var-cvar), and for the dependence structure you simulate from
  (copulas-and-dependence).
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-09"
---

# Monte Carlo methods

**A standard error measures how much the estimator wobbles. It is blind to whatever the
estimator is converging to.** Variance reduction makes a wrong number precise faster; this
skill is about the order of operations — get the target right, then reduce the variance.

`../../../fin-models/skills/option-pricing-models/SKILL.md` §3 covers the Euler-vs-exact
discretisation of a GBM. This skill is the estimator machinery around it: reduction factors you
can measure, American exercise, QMC, and a bias that survives 600,000 paths.

Every number marked ✅ Measured is printed by `scripts/monte_carlo.py` (numpy 2.2.6 + scipy
1.13.0, seed 20260909, **about 50 s**). ✅ source-verified means read in the cited paper or in
the installed scipy source.

## 1. ✅ The rate, and three reductions with their factors

The test payoff is a discretely monitored **arithmetic-average Asian call**, S=K=100, T=1,
r=5%, σ=25%, 12 monitoring dates.

✅ Measured — `4×` the paths halves the error bar, three times over: se 0.1071 → 0.0538 →
0.0267 → 0.0133, ratios **1.989, 2.014, 2.008** against the 2.000 that `1/sqrt(N)` predicts.
**To cut the error by 10 you pay 100×, and none of it touches a bias.**

✅ Measured — at a fixed **100,000-path** budget:

| method | price | std err | **variance reduction** | independent units |
|---|---|---|---|---|
| plain | 7.34910 | 0.03386 | 1.0× | 100,000 |
| antithetic | 7.35023 | 0.02468 | **1.9×** | **50,000** |
| **control variate** | 7.31241 | **0.00116** | **858.9×** | 100,000 |
| stratified | 7.31942 | 0.01698 | **4.0×** | 100,000 |

- 🔑 **The control variate is not in the same league as the other two.** The geometric-average
  Asian is exactly lognormal, so it has a closed form (⚠️ Kemna & Vorst 1990) — **6.99073**
  here — and it correlates ~0.999 with the arithmetic payoff. One extra exponential per path
  buys **859× the variance**, i.e. the same accuracy as 86 million plain paths.
- 🚨 **Antithetic sampling halves the number of independent units.** The error bar must be
  computed over **pair averages**. Treating 100,000 antithetic draws as 100,000 independent
  samples understates the error and flatters the method — the same trap
  `../../../fin-models/skills/option-pricing-models/SKILL.md` §6 finds in QuantLib's
  `requiredSamples`.
- 🚨 **A stratified estimator with one draw per stratum has no honest standard error.** The
  marginal law is unchanged, so the naive sample sd reports the *plain-MC* number and the
  4.0× disappears. `asian_mc` uses two draws per stratum and pools the within-stratum
  variances, which is the only way to see the reduction you actually got.

## 2. ✅ Longstaff-Schwartz, reproduced from the paper

Longstaff & Schwartz (2001), *Valuing American Options by Simulation: A Simple Least-Squares
Approach*, RFS 14(1), 113–147.

### 2a. The eight-path example of §1, exactly

✅ Measured against the paper's own printed output:

| | this script | the paper |
|---|---|---|
| regression at t=2 | −1.070 + 2.983 X − **1.814** X² | −1.070 + 2.983 X − 1.813 X² |
| regression at t=1 | +2.038 − 3.335 X + 1.356 X² | +2.038 − 3.335 X + 1.356 X² |
| American put | **0.1144** | .1144 |
| European put | **0.0564** | .0564 |

🔑 The paper never prints the price matrix — only the in-the-money `X` and `Y` columns. The
matrix in `LS_PATHS` is the unique one consistent with all of them, and getting both
regressions and both prices back out is the proof that it is right. ⚠️ The regressions are in
the **raw** state, not in moneyness; `lsm_price(..., scale=1.0)` is what reproduces them.

### 2b. Rows of Table 1 — and a contract mismatch that dwarfs the Monte Carlo error

✅ source-verified — K=40, r=6%, *"exercisable 50 times per year"*, *"100,000 (50,000 plus
50,000 antithetic) paths"*, basis *"a constant and the first three Laguerre polynomials"*.

| S | σ | T | LSM here | **Bermudan tree** | paper FD | paper LSM | (s.e.) | LSM − FD | continuous American |
|---|---|---|---|---|---|---|---|---|---|
| 36 | .20 | 1 | 4.4785 | **4.4780** | 4.478 | 4.472 | .010 | +0.0005 | 4.4867 |
| 40 | .20 | 1 | 2.3212 | **2.3140** | 2.314 | 2.313 | .009 | +0.0072 | 2.3194 |
| 44 | .20 | 1 | 1.1172 | **1.1101** | 1.110 | 1.118 | .007 | +0.0072 | 1.1132 |
| 36 | .40 | 2 | 8.4974 | **8.5069** | 8.508 | 8.488 | .024 | −0.0106 | 8.5144 |

**Three unrelated methods in the same place**: the paper's finite difference, an independent
CRR tree, and LSM.

🚨 **The last column is the trap.** It is the *same tree* with exercise allowed at all 2,000
steps — a continuously exercisable American, worth **0.005 to 0.009 more**. That is a different
*contract*, not an error, and it is **several times the paper's own standard error**. A tree
that allows exercise at every node will always look like it beats an LSM Bermudan. Compare a
Bermudan against a Bermudan (`bermudan_tree_put` restricts exercise to the same 50 dates a
year), or the comparison measures the exercise schedule instead of the method.

### 2c. 🚨 Footnote 9: `max(intrinsic, fitted continuation)` is upward biased

✅ source-verified — LS footnote 9: *"if the American option were valued by taking the maximum
of the immediate exercise value and the estimated continuation value, and discounting this
value back, the resulting American option value could be severely upward biased. This bias
arises since the maximum operator is convex; measurement error in the estimated continuation
value results in the maximum operator being upward biased."*

✅ Measured — **same paths, same seed, same regressions**; only the value carried backwards
changes, so the difference is the rule and nothing else:

| S | σ | T | cash flow (correct) | max-value | Bermudan tree | **max − cash** | max − tree |
|---|---|---|---|---|---|---|---|
| 36 | .20 | 1 | 4.4868 | 4.4952 | 4.4780 | **+0.0084** | +0.0172 |
| 40 | .20 | 1 | 2.3188 | 2.3352 | 2.3140 | **+0.0164** | +0.0212 |
| 44 | .20 | 1 | 1.1200 | 1.1279 | 1.1101 | **+0.0079** | +0.0179 |

🚨 **Positive at every strike**, and **no path count fixes it** — it is a property of the
estimator, not of the sample. `max(a, b̂)` with a noisy `b̂` is biased up even when `b̂` is
unbiased. Carry the *realised discounted cash flow* backwards, which is what the paper does.

### 2d. 🚨 Two more one-line ways to lose a cent — or six

✅ Measured, S=36, σ=.20, T=1, against the Bermudan tree's **4.4780**:

| variant | price | error |
|---|---|---|
| in-the-money paths only (the paper) | 4.4785 | **+0.0005** |
| **all paths in the regression** | 4.4191 | **−0.0589** |
| polynomial basis in S/K | 4.4710 | −0.0070 |
| **weighted Laguerre on the RAW S, not S/K** | 4.4146 | **−0.0634** |

- ✅ source-verified — LS §1: *"We use only in-the-money paths since it allows us to better
  estimate the conditional expectation function in the region where exercise is relevant and
  significantly improves the efficiency of the algorithm."* **Regressing on all paths costs 12×
  the error of doing it their way**, because a three-term fit spends its degrees of freedom on
  a region where the exercise decision is never close.
- 🚨 **The weighted Laguerre basis `exp(−X/2)·L_k(X)` underflows on a raw stock price.** At
  S = 40, `exp(−S/2) = 2e-9`, so all three columns are numerically zero beside the constant and
  the fit degenerates to an intercept — **−0.0634, as bad as regressing on everything**, and it
  does not warn. Feed it the moneyness `S/K`. This is why so many published LSM
  implementations quietly use plain polynomials.
- ✅ **In-sample 4.4785 against out-of-sample 4.4818**, a gap of **+0.0034** — the paper's own
  Table 2 diagnostic (fit the stopping rule on one set of paths, value it on another), and the
  paper's conclusion holds: they are *"virtually identical"*.
- 🔑 **LSM's stopping rule is estimated, therefore suboptimal, therefore the out-of-sample
  value is a lower bound on the true price — IN EXPECTATION.** ⚠️ It is not a bound on any
  single run: at 100,000 paths the Monte Carlo noise is several times the bias, which is why
  LS ran the diagnostic five times per row. **An LSM number above the tree is not by itself
  evidence of anything**, and a test that asserts a strict inequality will fail on a seed.

## 3. ✅ Quasi-Monte Carlo: scrambled Sobol

✅ source-verified in the installed scipy 1.13.0 — the constructor is
`Sobol(d, *, scramble=True, bits=None, seed=None, optimization=None)`, *"If True, use LMS+shift
scrambling"*, *"Max dimensionality is 21201"*, and the class carries the warning: *"Sobol'
sequences are a quadrature rule and they lose their balance properties if one uses a sample
size that is not a power of 2, or skips the first point, or thins the sequence"*. ✅ Measured —
`Sobol(d=2).random(5)` raises `UserWarning: The balance properties of Sobol' points require n
to be a power of 2.`

🚨 ✅ Measured — **`Sobol(scramble=False).random(2)[0]` is exactly `[0. 0. 0. 0.]`, and
`norm.ppf(0)` is `-inf`.** Turning scrambling off hands you an infinite first path. The
scrambled first point is `[0.973 0.286 0.229 0.689]`. **`scramble=True` is the default; leave
it on.**

✅ Measured — RMS error over 8 repetitions against a control-variate reference (*not* a QMC
reference — a QMC reference would share its points with the thing being measured):

| N = 2^p | MC rmse | QMC rmse | QMC advantage |
|---|---|---|---|
| 1,024 | 0.46326 | 0.04634 | 10.0× |
| 4,096 | 0.18054 | 0.02006 | 9.0× |
| 16,384 | 0.08907 | 0.00626 | 14.2× |
| 65,536 | 0.04656 | 0.00319 | **14.6×** |

✅ **Fitted convergence rates: MC `N^-0.55`, QMC `N^-0.66`** (theory: −0.50, and up to −1 for a
smooth integrand). ⚠️ The advantage is real and it is not the textbook `N^-1`: this is a
12-dimensional integrand with a kink at the strike, and kinks are what cost QMC its rate.

🚨 **A single QMC run has no error estimate at all** — the points are deterministic, so there is
nothing to take a variance of. The error bar comes from independent *scrambles*: ✅ Measured,
2^14 points × 16 scrambles gives **7.31185 ± 0.00153**. That is the entire point of *randomised*
QMC. Use `random_base2(m)`; slicing or thinning the sequence throws away the balance property
that was the reason to use it.

## 4. 🚨 The trap: a bias that 600,000 paths cannot see

A down-and-out call, S=K=100, H=90, T=1, r=5%, σ=25%. ✅ Continuous monitoring has a closed
form: **9.11122** (against the vanilla call's 12.33600). Simulate it by checking the barrier at
`m` dates — **exact lognormal steps, so there is no Euler bias anywhere** — and every path that
dips under H between two dates and comes back survives.

✅ Measured:

| m | paths | MC price | std err | error | **error / se** | BGK-corrected target | left |
|---|---|---|---|---|---|---|---|
| 12 | 150,000 | 10.69518 | 0.03865 | +1.58395 | **41.0** | 10.67512 | +0.5 |
| **52** | 150,000 | 9.91983 | 0.03920 | +0.80860 | **20.6** | 9.96823 | −1.2 |
| **52** | **600,000** | 9.97331 | **0.01967** | +0.86209 | **43.8** | 9.96823 | +0.3 |
| 252 | 150,000 | 9.50190 | 0.03951 | +0.39068 | **9.9** | 9.52722 | −0.6 |

- 🚨 **Read the two `m = 52` rows against each other.** 4× the paths cuts the error bar
  0.0392 → 0.0197 and the price stays 0.8 too high, so the bias goes from **21 to 44 standard
  errors**. **More paths make the reported error bar a worse description of how wrong the
  number is.**
- ✅ **Monitoring dates are what fix it:** m = 12 → 252 cuts the bias 1.58 → 0.39, a factor of
  **4.1** against the **4.6** that `1/sqrt(m)` predicts.
- ✅ **The last two columns are Broadie-Glasserman-Kou's continuity correction**: evaluate the
  *continuous* formula at the shifted barrier `H·exp(−β·σ·sqrt(T/m))`, with
  `β = −ζ(1/2)/sqrt(2π) = 0.582597`. It turns a 40-sigma bias into **under one sigma at every
  m**, out of one closed-form evaluation and no extra paths.
- 🚨 ✅ **`scipy.special.zeta(0.5, 1)` returns `nan`** — the Hurwitz form carries no analytic
  continuation below 1. **`scipy.special.zetac(0.5) + 1` does**, and gives
  −1.4603545088095866 (its docstring: *"For ``x < 1`` the analytic continuation is computed"*).
  A `nan` there does not raise: it flows through `exp()` into a barrier level and surfaces much
  later as a comparison that is inexplicably `False`. That is how it was found here.

**The rule this section exists for:** report the discretisation and the standard error
*separately*, and pick the discretisation before you spend anything on variance reduction. A
control variate at m = 52 would have given you 9.97 ± 0.001 — three digits of precision around
a number that is wrong in the first one.

## 5. What the script gives you

`scripts/monte_carlo.py` — numpy + scipy only, importable, no file writes, no network:

| Function | Does |
|---|---|
| `mc_stats(payoffs)` | mean and standard error over the *independent unit* |
| `geometric_asian_call(...)` | Kemna-Vorst closed form — the control variate |
| `asian_mc(..., method=)` | `plain` / `antithetic` / `control` / `stratified` — §1 |
| `variance_reduction_table(n)` | the §1 table; `asian_reference()` for a high-accuracy price |
| `design_matrix(x, basis, K)` | `poly`, `laguerre`, `laguerre_raw` — §2d |
| `lsm_price(paths, ..., rule=, scale=)` | LSM; `rule="maxvalue"` is footnote 9's mistake |
| `ls_paper_example()` | §2a, the eight-path reproduction |
| `lsm_american_put(..., out_of_sample=)` | §2b and the in/out-of-sample diagnostic |
| `crr_american_put(..., ex_every=)` / `bermudan_tree_put(...)` | the tree, continuous or Bermudan |
| `sobol_normals` / `sobol_first_point` / `asian_qmc` / `qmc_vs_mc` | §3 |
| `down_and_out_call` / `barrier_mc` / `bgk_corrected` / `BGK_BETA` | §4 |

⚠️ **What is not here**: importance sampling (the right tool when the payoff is a rare event —
§4's barrier at H=50 would need it), multilevel Monte Carlo, Brownian-bridge and PCA path
constructions for QMC (which are what recover the `N^-1` rate in high dimensions), Greeks by
pathwise or likelihood-ratio methods, and the upper-bound (duality) estimators that bracket
LSM's lower bound.

## Where this sits

- `../../../fin-models/skills/option-pricing-models/SKILL.md` — the models themselves: Heston's
  branch cut, CRR's parity oscillation, SABR, and the Euler-vs-exact discretisation bias on a
  GBM. §4 here is the same lesson on a payoff whose discretisation is in the *contract*.
- `../../../fin-models/skills/risk-measures-var-cvar/SKILL.md` — when the simulation output is
  a risk number rather than a price, and the tail quantile has its own error bar.
- `../copulas-and-dependence/SKILL.md` — what you simulate *from*. Its §6 VaR numbers are Monte
  Carlo and inherit everything above.
- `../hawkes-processes/SKILL.md` §5 — the same warning from the other side: a standard error is
  only as good as the independence assumption underneath it, and clustered arrivals break it by
  a measured factor.
- `../../../fin-core/skills/backtest-validation/SKILL.md` — a backtest is a Monte Carlo with one
  path, and §4's moral applies hardest there.
