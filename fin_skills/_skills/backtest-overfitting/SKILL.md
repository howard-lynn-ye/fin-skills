---
name: backtest-overfitting
description: >-
  Decide whether an edge that passed every mechanical check is still just the best of N tries.
  TRIGGER - PBO, probability of backtest overfitting, CSCV, combinatorially symmetric cross
  validation, logit of the out-of-sample rank; minimum backtest length, MinBTL, "how much history
  do I need", "how many parameter sets can I try on N years of data"; a grid search, Optuna or
  AutoML picked a winner and it decayed; "in-sample Sharpe 2, live Sharpe 0", "the best
  parameter set stopped working", "is this peak on my parameter surface real"; pypbo,
  RiskLabAI CSCV. SKIP for the deflated and probabilistic Sharpe ratios, the trial ledger and
  SPA/StepM/MCS against a benchmark (backtest-validation); for family-wise error and
  false-discovery control over a whole research programme (multiple-testing-ledger); for the
  purged and combinatorial CV splitters themselves (lib-purgedcv); for the mechanical leakage
  gates that come first (research-integrity-guards); and for regime coverage of the test
  window (regime-detection).
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-10"
---

# Backtest overfitting

**Seven guards in this repo ask whether a backtest cheated. This one asks the question that is
still open after they all pass: the backtest is clean, and it is *the maximum of a search*.**

Two numbers answer it. **PBO** asks how often the in-sample winner lands in the bottom half out
of sample — 0.5 means the winner is a coin flip. **MinBTL** asks how many years of data a
claimed Sharpe needs before the honest trial count stops explaining it on its own.

Every number below is printed by `scripts/overfitting.py` (numpy + scipy, seeds fixed and
printed, **24 s**). Nothing here is quoted from a paper without being reproduced.

## 1. CSCV, exactly as defined

✅ Source: Bailey, Borwein, López de Prado & Zhu, *The Probability of Backtest Overfitting*,
Journal of Computational Finance 20(4), 2016 — read from the authors' hosted preprint
(`davidhbailey.com/dhbpapers/backtest-prob.pdf`; the published JCF text is paywalled and was not
reached). Algorithm 2.3:

1. Build a **T x N** performance matrix — one column per configuration you tried, rows aligned
   on the same calendar.
2. Split the **rows** into an even number **S** of disjoint contiguous blocks.
3. Form all **C(S, S/2)** ways of choosing S/2 blocks as in-sample; the complement is
   out-of-sample. Both halves are the same size — that is what "symmetric" means, and it is why
   IS and OS performance are directly comparable.
4. `n*` = the configuration with the best IS performance. Find its **rank** among the N OS
   performances, `1` = worst.
5. ✅ The paper defines the relative rank verbatim as **`w = rank / (N + 1)`**, in (0,1). The
   `N+1` is what keeps the logit finite when the IS winner also wins OS.
6. `lambda = log(w / (1 - w))`. **PBO = the share of splits with `lambda <= 0`.**

✅ **S = 16 is the authors' own recommendation**, and their reason is worth keeping: on four
years of daily data it makes each block a quarter, so the serial correlation inside a block
survives. The script uses T = 1,008 (exactly 4.0 years) and S = 16 for that reason.

🚨 **The preprint prints `C(16,8) = 12,780`, twice.** The correct value is **12,870** — the
script prints it. Their `C(24,12) = 2,704,156` and `C(12,6) = 924` are both right, so it is a
transposition, not a different definition. If a PBO implementation you are reading hard-codes
12,780, it copied the typo.

⚠️ **The paper's recommended threshold is much stricter than "below 0.5":** it says to reject a
model whose estimated PBO exceeds **0.05**. Its two worked examples are the calibration — an
overfit seasonal rule on a random walk scored **55%**, a genuinely planted monthly effect scored
**13%**.

## 2. 🚨 The no-skill line is 0.5 only when N is even

`lambda <= 0` means `w <= 0.5` means `rank <= (N+1)/2`, and a rank is an integer. For odd N that
admits `(N+1)/2` of the N ranks, so the no-skill line is `(N+1)/(2N)` — **0.600 at N = 5**, 0.520
at N = 25. Comparing a five-configuration PBO to 0.5 reports overfitting that is arithmetic.

✅ Measured on pure noise (every configuration's TRUE Sharpe is exactly zero, T = 1,008, S = 16):

| N | panels | no-skill line | mean PBO | std err | **sd across panels** | mean `w` |
|---|---|---|---|---|---|---|
| 5 | 60 | **0.600** | 0.634 | 0.031 | 0.240 | 0.483 |
| 10 | 60 | 0.500 | **0.498** | 0.027 | 0.206 | 0.501 |
| 20 | 60 | 0.500 | 0.481 | 0.024 | 0.190 | 0.514 |
| 50 | 24 | 0.500 | **0.501** | 0.028 | 0.137 | 0.501 |
| 100 | 24 | 0.500 | 0.490 | 0.028 | 0.137 | 0.509 |

Every row sits on its own line inside about two standard errors. **That is the load-bearing
property**: search noise hard enough and the winner is still a coin flip.

🚨 **Read the "sd across panels" column before you report a single PBO.** One panel's PBO
averages 12,870 splits, but all 12,870 reuse the same 1,008 rows, so they are not 12,870
independent draws. The across-panel standard deviation is **0.14 to 0.24**, not the
`sqrt(p(1-p)/12870) = 0.004` a binomial would suggest. A single-run PBO of 0.3 or 0.7 on noise is
ordinary. **PBO is a diagnostic, not a p-value**, and one run does not have three digits of
precision.

## 3. ✅ Planting a real edge pulls PBO down, and the sample never changes

Same 24 noise panels in every row, edge added to column 0 only. The 0.0 row *is* the N = 50 row
above:

| true annualised Sharpe of the planted one | PBO | std err | mean `w` | P(it wins IS) |
|---|---|---|---|---|
| 0.0 | **0.501** | 0.028 | 0.501 | 0.019 |
| 0.5 | 0.477 | 0.028 | 0.517 | 0.084 |
| 1.0 | 0.412 | 0.037 | 0.572 | 0.233 |
| 1.5 | 0.308 | 0.043 | 0.672 | 0.449 |
| 2.0 | **0.194** | 0.041 | 0.787 | 0.670 |

Four years of data, fifty configurations, in every row. **PBO measures the search, not the
sample.** Note the last column: even with a true Sharpe of 2.0, the real strategy wins in-sample
in only **67%** of splits.

## 4. ✅ The in-sample winner's out-of-sample standing decays as the grid grows

One real strategy (true annualised Sharpe 1.0) plus N-1 noise ones. Nested and paired — one
200-column panel per seed, each row keeping the first N columns, so *growing the grid is the only
thing that changes*:

| N | PBO | mean OS rank `w` | P(the real one wins IS) | median OS Sharpe of the IS winner |
|---|---|---|---|---|
| 2 | **0.210** | 0.597 | **0.840** | **0.99** |
| 5 | 0.260 | 0.673 | 0.674 | 0.93 |
| 10 | 0.262 | 0.662 | 0.510 | 0.78 |
| 25 | 0.292 | 0.661 | 0.371 | 0.61 |
| 50 | 0.330 | 0.637 | 0.283 | 0.54 |
| 100 | 0.381 | 0.592 | 0.201 | 0.32 |
| 200 | **0.413** | 0.570 | **0.137** | **0.23** |

Same real strategy, same four years, same true edge. Adding 198 worthless variants takes the
winner's out-of-sample Sharpe from **0.99 to 0.23** and the chance that the winner is the real
strategy from **84% to 14%**. Nothing about the strategy changed; the *search* did.

**This is the answer to "but my grid is cheap, why not try everything".**

## 5. ⚠️ A correlated grid is fewer trials than it looks, and PBO gets noisier

48 moving-average crossovers on ONE random walk — the shape of a real parameter sweep:

- mean pairwise correlation between columns: **0.683** (independent columns would be ~0)
- PBO over 12 random walks: **0.512 +/- 0.070**, ranging **0.198 to 0.956**
- sd across panels **0.244**, against **0.152** for 48 *independent* columns

A random walk has no edge, so the no-skill line is still the honest answer — but a single draw
tells you much less. **Report PBO with N, S and the mean pairwise correlation of the columns
beside it.** A bare "PBO = 0.31" is not interpretable.

## 6. Minimum Backtest Length

✅ Source: the same authors, *Pseudo-Mathematics and Financial Charlatanism*, **Notices of the
AMS 61(5), May 2014, 458–471** — open access, and the formulas below are Proposition 1 and
Theorem 2 as printed there.

```
E[max SR over N trials]  =  (1 - g) * Z^-1(1 - 1/N)  +  g * Z^-1(1 - 1/(N e)) ,  g = 0.5772...
MinBTL (years)           =  ( E[max SR] / SR* )^2        <  2 ln(N) / SR*^2
```

✅ **Three anchors stated in prose in that paper, all reproduced by the script:**

| the paper says | this file |
|---|---|
| N = 10 configurations gives an expected max Sharpe of **1.57** | **1.5746** |
| five years of data supports at most **45** configurations | MinBTL(45) = **4.998 y** |
| **seven** configurations already need a two-year backtest | MinBTL(7) = **1.923 y** |

⚠️ There is no MinBTL *table* in the paper — it is Figure 2, a plot. The values below are the
script's, from the verified formula, at a claimed annualised Sharpe of 1.0:

| N trials | 2 | 5 | 10 | 25 | 50 | 100 | 500 | 1000 |
|---|---|---|---|---|---|---|---|---|
| **MinBTL, years** | 0.27 | 1.42 | 2.48 | 3.99 | **5.18** | 6.40 | 9.32 | 10.60 |

✅ **The approximation is checked, not assumed.** Against 200,000 Monte-Carlo draws of
`max(N standard normals)` it overstates the expected maximum by **+0.036 at N = 10**, **+0.024 at
N = 100**, **+0.013 at N = 1000** — always in the same direction, so MinBTL is mildly
conservative. That is the safe side.

### The contrast that decides the argument

On the script's own 4.0-year sample:

| claimed Sharpe | N trials | needs | has | verdict |
|---|---|---|---|---|
| 1.0 | 10 | 2.48 y | 4.0 y | long enough |
| 1.0 | **50** | **5.18 y** | 4.0 y | **short by 1.2 years** |
| 1.0 | 500 | 9.32 y | 4.0 y | short by 5.3 years |
| 2.0 | 50 | 1.30 y | 4.0 y | long enough |

🚨 **Read the second row as the paper intends it:** 50 honest trials on 4.0 years of daily data
have an expected maximum Sharpe of **1.14** from noise alone. A reported 1.0 is *below* what the
search was expected to produce by luck. No amount of care about leakage, costs or fills changes
that; only more data or fewer trials does.

⚠️ MinBTL assumes **independent** trials, so on a correlated grid it overstates the requirement.
That is why PBO exists — it handles dependence directly by construction, at the cost of needing
the whole per-period panel rather than a count.

## 7. Traps

- 🚨 **Comparing PBO to 0.5 with an odd N** — §2. `null_pbo(N)` prints the right line.
- 🚨 **Quoting one run's PBO to three digits** — §2. Across-panel sd is 0.14 to 0.24.
- 🚨 **Hard-coding 12,780 for C(16,8)** — §1.
- 🚨 **A PBO computed on the configurations you KEPT.** The panel must have a column for every
  configuration you ran, abandoned ones included. Keeping the top 5 of 500 and running CSCV over
  those 5 measures nothing; the search happened before the matrix was built. This is the same
  input failure that neuters DSR, and the ledger that fixes it is
  `../multiple-testing-ledger/SKILL.md`.
- 🚨 **Deflating a Sharpe computed on leaked data.** PBO on a panel whose columns all peek is a
  precise measurement of nothing. Run
  `../research-integrity-guards/SKILL.md` and `../signal-construction/scripts/assert_causal.py`
  first; `fin_skills.api.guards.research_audit.reality_check` enforces that ordering and stops.
- ⚠️ **S too small or too large.** S = 16 on four years of daily data is the paper's
  recommendation because a block is then a quarter. S must be even, and `C(S, S/2)` grows fast —
  S = 24 is 2,704,156 splits.
- ⚠️ **Blocks that are not contiguous in time.** Shuffling rows before blocking destroys the
  serial correlation the block structure exists to preserve.
- ⚠️ **PBO on gross returns.** A configuration that only wins before costs is not a configuration.
  Run `../execution-cost-analysis/SKILL.md` or `../backtest-validation/scripts/cost_curve.py`
  over the panel first.

## 8. The ecosystem, and why this file exists

- ⚠️ **`pypbo`** is the reference CSCV implementation but is **not on PyPI** and is **AGPL-3.0**.
- ⚠️ `RiskLabAI` and `ml4t-diagnostic` both ship a CSCV; neither is a dominant maintained choice.
  `../backtest-validation/SKILL.md` §1 has the current state of that shelf.
- 🚨 `pip install pbo` does not install this. `mlfinlab` is a stub — every function body is `pass`.

Which is why `scripts/overfitting.py` implements CSCV in about 60 lines of numpy from the
published algorithm, and measures the properties instead of trusting them. The whole
`C(S, S/2) x N` grid is two matrix products over per-block moments, so S = 16 with N = 200 runs
in **0.36 s**.

## 9. Scripts

- `scripts/overfitting.py` — `cscv()` (the full CSCV result: logits, ranks, IS winners, the
  OS-on-IS slope), `pbo()`, `null_pbo()`, `expected_max_sharpe()`, `min_backtest_length()`,
  `min_backtest_length_bound()`, `credible()`. `__main__` measures §2-§6 and reproduces §1 and
  §6's published anchors. Seeds 1000+, 2000+, 3000+, 4000+, 9. **24 s.**

## 10. Where this sits

`../backtest-validation/SKILL.md` owns the Deflated and Probabilistic Sharpe Ratios, the trial
ledger those need, and SPA / StepM / MCS against a benchmark — PBO and DSR are complements, and
`expected_max_sharpe` here is the same expression that skill's `trial_ledger.py` uses for its
`sr0`. `../multiple-testing-ledger/SKILL.md` owns family-wise error and false-discovery control
over the whole research programme, and the haircut a claimed Sharpe takes.
`../research-integrity-guards/SKILL.md` owns the five mechanical gates that must pass first.
`../../../fin-libraries/skills/lib-purgedcv/SKILL.md` owns the purged and combinatorial CV
splitters themselves. `../regime-detection/SKILL.md` owns whether the test window contained more
than one regime. `fin_skills.api.guards.research_audit` runs all of them in the sceptic's order
and stops at the first failure.
