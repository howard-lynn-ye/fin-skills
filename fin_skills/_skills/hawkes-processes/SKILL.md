---
name: hawkes-processes
description: >-
  Fit and test a self-exciting point process for clustered order arrivals - exponential-kernel
  Hawkes intensity, Ogata thinning, maximum likelihood, the branching ratio, and the random
  time change that tests the fit. TRIGGER - Hawkes process, self-exciting point process, order
  arrival clustering, trade clustering, order flow clustering, mutually exciting, branching
  ratio, alpha over beta, criticality, endogeneity of market activity; Ogata thinning, simulate
  a point process, tick, hawkeslib, conditional intensity; Hawkes MLE, log-likelihood
  recursion, random time change, residual analysis, "are my arrivals Poisson", overdispersed
  counts, Fano factor, "my Poisson confidence interval is too narrow". SKIP for the birth-death
  queue model of a single price level (limit-order-book-models), for measuring realised
  activity from a tape (intraday-microstructure), for GARCH and volatility clustering in
  returns rather than arrivals (volatility-models), and for regime switching
  (regime-detection).
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-09"
---

# Hawkes processes

**Order arrivals cluster, a Poisson model reproduces their average rate perfectly, and its
standard errors are wrong by a factor you can compute in closed form.** That is the whole
skill: the mean is the one statistic that cannot detect the misspecification, and everything
downstream of a count inherits the error.

Every number marked ✅ Measured is printed by `scripts/hawkes.py` (numpy 2.2.6 + scipy 1.13.0,
seed 20260909, **about 30 s**). ✅ source-verified means it was read in **Laub, Taimre & Pollett,
*Hawkes Processes*** (arXiv:1507.02822), whose notation this follows — their background rate
`lambda` is written `mu` here so it is not confused with the intensity.

## 1. The model, in four formulas

✅ source-verified — Laub et al.:

```
lambda*(t) = mu + sum_{t_i < t} alpha * exp(-beta*(t - t_i))       exponential kernel
n          = int_0^inf alpha*exp(-beta*s) ds = alpha/beta          BRANCHING RATIO
mean rate  = mu / (1 - n)                                          stationary intensity
stability  = alpha < beta, i.e. n < 1
```

🔑 `n` is *"the expected number of offspring per immigrant in the immigration-birth
representation"* — the fraction of activity that is triggered by earlier activity rather than
arriving from outside. **`n ≥ 1` is not a large number, it is a different process**: the cascade
never dies out and the count explodes. Every stationary formula below divides by `1 − n`.

✅ source-verified — **Ogata's thinning** (Laub et al. Algorithm 2): from `t`, take
`M = lambda*(t+)` — the intensity *just after* `t`, which bounds the intensity from above until
the next point because the kernel only decays — draw `E ~ Exp(M)`, advance `t` by `E`, accept
with probability `lambda*(t)/M`. Carrying the running sum forward makes a path `O(proposals)`
instead of `O(k²)`.

🚨 **A thinning bug never crashes.** It produces a path with a slightly wrong rate, which then
fits fine. ✅ Measured — the only honest check is a second, unrelated simulator. The
immigration-birth construction (Poisson(`mu`) immigrants; each event has `Poisson(n)` offspring
at `Exp(beta)` delays; recurse) at `mu=0.5, alpha=1.4, beta=2.0`, 16 paths of `T=3000`:

| method | mean rate | ± | ÷ `mu/(1−n)` = 1.6667 | z |
|---|---|---|---|---|
| Ogata thinning | 1.6985 | 0.0202 | 1.0191 | **+1.6** |
| immigration-birth | 1.6434 | 0.0127 | 0.9860 | **−1.8** |

Two mechanisms with nothing in common but the model, both within two standard errors of the
closed form. ⚠️ The `±` is the sample sd across paths, **not** `sqrt(k)/T` — §5 is why.

## 2. ✅ The likelihood, and the recursion that makes it usable

✅ source-verified — Laub et al.:

```
l = sum_i log(mu + alpha*A(i)) - mu*T + (alpha/beta) * sum_i [exp(-beta*(T - t_i)) - 1]
A(1) = 0,   A(i) = exp(-beta*(t_i - t_{i-1})) * (1 + A(i-1))
```

✅ Measured — the recursive form against a direct `O(k²)` transcription of the definition, on a
768-event path:

| parameters | recursive | direct | diff |
|---|---|---|---|
| `mu=0.5, alpha=1.4, beta=2` | 8.99989053 | 8.99989053 | **0.00e+00** |
| `mu=0.3, alpha=0.9, beta=3` | −183.22909663 | −183.22909663 | 2.84e-14 |

🔑 **The recursion is not an optimisation, it is the difference between fitting a tape and not.**
A million-event day is `10¹²` kernel evaluations the naive way.

⚠️ `T` is the observation horizon and defaults to the last event; the compensator term
integrates the kernel out to `T`, so passing the real horizon matters when the window ends
quiet.

## 3. ✅ What a fit actually determines

✅ Measured — one path, `T=4000`, 6,920 events: `mu` 0.4659 (−6.8%), `alpha` 1.4507 (+3.6%),
`beta` 1.9854 (−0.7%), `n` 0.7307 (+4.4%). **One fit says nothing about which parameter is well
determined.** 20 independent seeded paths at `T=2000` do:

| parameter | true | mean fit | bias | sd | **sd / mean** |
|---|---|---|---|---|---|
| `mu` | 0.5000 | 0.4968 | −0.6% | 0.0239 | 4.8% |
| `alpha` | 1.4000 | 1.3955 | −0.3% | 0.0772 | 5.5% |
| `beta` | 2.0000 | 1.9927 | −0.4% | 0.0994 | 5.0% |
| **`n = alpha/beta`** | 0.7000 | **0.7004** | **+0.1%** | 0.0204 | **2.9%** |

🔑 **The ratio is determined roughly twice as sharply as either of its parts**, because `alpha`
and `beta` trade off in the likelihood — a faster kernel with a bigger jump looks much the same.
**Report `n`, not the pair.** It is also the quantity everything in §5 depends on.

⚠️ Nothing in the fit constrains `alpha < beta`. If your data want `n > 1` you should see it,
not have it clipped: it usually means the exponential kernel is wrong (real order flow is closer
to power-law) rather than that the market is supercritical.

## 4. ✅ The random time change is the test — the mean is not

✅ source-verified — with `Lambda(t)` the compensator, the transformed points
`{Lambda(t_1), ..., Lambda(t_k)}` *"form a Poisson process with unit rate"*, so the increments
are i.i.d. `Exp(1)` and a KS test against `Exp(1)` is the goodness-of-fit test.

✅ Measured — the same 6,920-event path, fitted both ways:

| model fitted | log-lik | KS stat | p-value | residual mean | **residual var** |
|---|---|---|---|---|---|
| Hawkes (exponential) | −365.7 | 0.0080 | 0.771 | 0.9995 | **1.0052** |
| homogeneous Poisson | **−3127.0** | **0.2403** | **0.00e+00** | **0.9988** | **4.2237** |

🚨 **`Exp(1)` has mean 1 and variance 1. The Poisson residuals have mean 0.9988 — by
construction.** Under a Poisson fit the residuals are the raw waiting times multiplied by
`k/T`, so their mean is 1 whatever the data do. **The mean cannot see the misspecification.**
The variance (4.22 against 1) and the KS statistic can, and the KS test rejects outright.

🔑 **Never validate an arrival model on its average rate.** A Poisson process fitted to
clustered data always reproduces it.

## 5. 🚨 The trap: a Poisson standard error on a clustered count

⚠️ The classical asymptotic result is `Var[N(T)] / E[N(T)] -> 1/(1-n)^2`. ⚠️ I did not reach a
primary source for it, so it is verified here by measurement rather than cited. ✅ Measured at
`n = 0.7`, where `1/(1−n)² = 11.11`, with a bootstrap error bar (a Fano factor is a variance of
a heavily skewed count, so its own sampling error is large):

| T | runs | mean count | Fano measured | ± | 1/(1−n)² | z |
|---|---|---|---|---|---|---|
| 100 | 1500 | 165.3 | 10.80 | 0.46 | 11.11 | −0.7 |
| 300 | 1500 | 497.9 | 10.15 | 0.36 | 11.11 | −2.7 |
| 1000 | 800 | 1668.0 | 11.70 | 0.58 | 11.11 | +1.0 |
| 3000 | 400 | 5000.2 | 12.63 | 0.91 | 11.11 | +1.7 |

Three of four windows within 1.7 bootstrap standard errors, the fourth within 2.7, no trend in
`T`. **Enough to confirm the closed form; nowhere near enough to eyeball over-dispersion from
one sample.**

✅ Measured — the consequence, at a **held-constant mean arrival rate** (`mu` is adjusted so
every row has the same expected count). "95% CI" is the textbook Poisson interval
`k/T ± 1.96·sqrt(k)/T` for the arrival rate, `T = 300`, 1,200 replications:

| n | mean count | Fano | 1/(1−n)² | se too small by | **95% CI coverage** |
|---|---|---|---|---|---|
| 0.0 | 480.4 | 0.97 | 1.00 | 1.00× | **94.9%** |
| 0.3 | 480.0 | 1.94 | 2.04 | 1.43× | 83.8% |
| 0.5 | 479.4 | 3.86 | 4.00 | 2.00× | 68.4% |
| **0.7** | 479.2 | 10.23 | 11.11 | **3.33×** | **45.7%** |
| 0.9 | 472.0 | 94.38 | 100.00 | 10.00× | **14.9%** |

🚨 **Same rate, same expected count, every row. The only thing that changes is clustering, and
a nominal 95% interval covers the truth 15% of the time at `n = 0.9`.** At the `n = 0.7` that
equity trade arrivals are routinely fitted at, it is 46%.

🔑 **Multiply any Poisson standard error on a count by `1/(1 − n)` before you believe it.** This
propagates into anything built on counts:

- **activity bars** — tick, volume and dollar bars all threshold a count, so bars per day
  fluctuates far more than a Poisson intuition suggests.
  `../../../fin-core/skills/intraday-microstructure/SKILL.md` §1 measures 209–831 bars/day from
  one fixed threshold; this is the mechanism.
- **arrival-rate and trade-count volatility proxies** — the estimator is fine, the error bar is
  not.
- **a bootstrap over events** — an i.i.d. resample of clustered arrivals destroys exactly the
  dependence that widens the interval.
- **queue models with Poisson arrivals** — `../limit-order-book-models/SKILL.md` is built on
  independent Poisson flows, which is what makes it solvable; its *conditional* probabilities
  are still informative, but confidence intervals from it are optimistic for the same reason.

## 6. What the script gives you

`scripts/hawkes.py` — numpy + scipy only, importable, no file writes, no network:

| Function | Does |
|---|---|
| `branching_ratio`, `stationary_intensity` | `alpha/beta` and `mu/(1−n)`; the latter raises rather than returning a negative rate when `n ≥ 1` |
| `intensity(t, times, ...)` | `lambda*(t)`, strictly past points |
| `simulate_thinning(...)` | Ogata's algorithm, `O(proposals)` |
| `simulate_cluster(...)` | the independent immigration-birth check — §1 |
| `log_likelihood(..., recursive=)` | the `O(k)` recursion, or the `O(k²)` definition — §2 |
| `fit_mle(times, T)` | L-BFGS-B in log-parameters; returns `n` and the log-likelihood |
| `fit_poisson(times, T)` | the nested `alpha = 0` fit, directly comparable |
| `compensator_residuals`, `ks_exp1`, `goodness_of_fit` | the random time change — §4 |
| `recovery_study(n_paths, T)` | the sampling spread of each estimate — §3 |
| `count_dispersion(..., T, n_runs)` | Fano factor with a bootstrap se, and Poisson-CI coverage — §5 |

⚠️ **What is not here**: multivariate / mutually exciting kernels (buy flow exciting sell flow
is the interesting financial case), marked processes, power-law and sum-of-exponentials kernels,
and non-parametric kernel estimation. The exponential kernel is the one with an `O(k)`
likelihood; everything else costs a rewrite.

## Where this sits

- `../limit-order-book-models/SKILL.md` — the queueing model whose arrivals are Poisson. This
  skill is the measurement of how wrong that is, and the branching ratio is the size of it.
- `../../../fin-core/skills/intraday-microstructure/SKILL.md` — measuring activity on a real
  tape: bar construction, trade classification, order-flow imbalance, VPIN. §1's bars-per-day
  spread is §5's over-dispersion.
- `../../../fin-models/skills/volatility-models/SKILL.md` — clustering in *returns* (GARCH and
  friends) rather than in *arrival times*. The two are different objects with the same story.
- `../../../fin-core/skills/regime-detection/SKILL.md` — when the clustering is a switch in
  regime rather than self-excitation.
- `../monte-carlo-methods/SKILL.md` — §5 is the same lesson from the other end: a standard error
  is only as good as the independence assumption underneath it.
