---
name: multiple-testing-ledger
description: >-
  Apply family-wise error and false-discovery control to a whole research programme, using the
  trial ledger's own registered count as m. TRIGGER - Bonferroni, Holm, Benjamini-Hochberg,
  Benjamini-Yekutieli, BHY, FDR, FWER, "adjust my p-values", statsmodels multipletests, "which
  of my factors survive the correction"; haircut Sharpe ratio, Harvey-Liu-Zhu, "t-stat of 3",
  factor zoo, "is a t of 2.1 enough"; "I screened 500 signals and 30 came back significant",
  "which correction should I use", "my strategies are all correlated so Bonferroni is too
  strict". SKIP for the deflated and probabilistic Sharpe ratios and the ledger file format
  itself (backtest-validation); for PBO, CSCV and the minimum backtest length
  (backtest-overfitting); for SPA / StepM / MCS, which correct for the search using the
  bootstrap instead of p-values (backtest-validation); and for the leakage and cost gates that
  must pass before a p-value means anything (research-integrity-guards).
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-10"
---

# Multiple-testing ledger

**`../backtest-validation/SKILL.md` deflates ONE Sharpe against a trial count. This skill does
the accounting across the whole programme: given a ledger of trials, which survive family-wise
error or false-discovery control, and how much Sharpe does the correction take off the top.**

It does not invent a second ledger. `scripts/research_history.py` reads the append-only
`trials.jsonl` that `../backtest-validation/scripts/trial_ledger.py` already writes, and takes
`m` from **that ledger's REGISTERED count**, not from the trials that happened to finish.

Every rate below is measured by that script on seeded data (numpy + pandas + scipy, **10 s**).

## 1. 🚨 `m` is the number of hypotheses you tested, and the ledger already knows it

A trial you registered, ran, disliked and abandoned was a test of a hypothesis. `TrialLedger`
records it — `abandon()` exists for exactly this arithmetic — and every correction here uses
`summary()["n_trials"]`, the registered count. Unscored trials are padded with `p = 1.0`, which
is what a step-up or step-down procedure does with a test that did not reject.

✅ Measured. 400 registered trials, 8 of them genuinely predictive (true annualised Sharpe 1.8),
40 scored and 360 abandoned:

| correction | survivors at m = 400 (honest) | at m = 40 (scored only) | inflation |
|---|---|---|---|
| uncorrected | 9 | 9 | +0 |
| Bonferroni | **1** | 2 | **+1** |
| Holm | **1** | 2 | **+1** |
| Benjamini-Hochberg | **2** | 5 | **+3** |
| Benjamini-Yekutieli | **1** | 2 | **+1** |

Dropping the abandoned trials shrinks `m` tenfold and lets extra strategies through every gate.
**`abandon()` is not bookkeeping, it is the denominator.**

## 2. What each correction actually guarantees

| procedure | rule on sorted p-values | controls | valid under |
|---|---|---|---|
| **Bonferroni** | `p_(i) <= alpha/m` | FWER | any dependence |
| **Holm (1979)** | step **down**, `p_(i) <= alpha/(m-i+1)`, stop at the first failure | FWER | any dependence |
| **Benjamini-Hochberg** | step **up**, largest `i` with `p_(i) <= i*q/m` | FDR | independence or **PRDS** |
| **Benjamini-Yekutieli** | BH with `q` divided by `c(m) = sum 1/i` | FDR | **arbitrary** dependence |

✅ Holm (1979), *Scand. J. Statist.* 6(2), 65–70: the comparison sequence is
`alpha/n, alpha/(n-1), ..., alpha/1` against Bonferroni's flat `alpha/n`, and Theorem 1's proof
uses only Boole's inequality — which is why it needs no assumption about dependence. Holm proves
it **uniformly dominates** Bonferroni, so Bonferroni is only ever the easier arithmetic.

✅ Benjamini & Yekutieli (2001), *Annals of Statistics* 29(4), 1165–1188. Theorem 1.2: BH
controls FDR at `(m0/m) q` when the statistics are **PRDS on the true nulls**. Theorem 1.3: with
`q / sum_{i=1..m} (1/i)` in place of `q`, control holds **always**. So `c(m)` divides the
threshold, and it grows like `ln m + 0.5772`:

| m | 10 | 50 | 100 | 500 | 1000 |
|---|---|---|---|---|---|
| **c(m)** | 2.929 | 4.499 | 5.187 | 6.793 | 7.485 |

**FWER and FDR are different promises.** FWER control means "probably not one false discovery in
the whole programme". FDR control means "of what I report, a controlled *fraction* is wrong". A
strategy list you will fund one of wants FWER; a screen you will re-test wants FDR.

## 3. ✅ The corrections do what they claim, measured

200 null strategies, 1,008 days each (4 years), nominal 5% two-sided: **8 of 200 clear the
test** (4.0%, which is what 5% means), the best of them showing an annualised Sharpe of **1.71**
on a true value of zero. Every correction leaves **0** standing.

FWER over 400 replications of m = 100 nulls:

| correction | measured FWER | target |
|---|---|---|
| uncorrected | **0.993** | — |
| Bonferroni | 0.035 | 0.05 |
| Holm | 0.035 | 0.05 |
| Benjamini-Hochberg | 0.043 | 0.05 |
| Benjamini-Yekutieli | 0.007 | 0.05 |

🚨 **Uncorrected, 99.3% of research programmes over 100 dead strategies produce at least one
"significant" result.** That is not a small effect, and it is the base rate against which every
published backtest should be read.

## 4. The power cost, which is the reason people skip the correction

m = 100 with 10 genuinely predictive, 300 replications, alpha = 5%:

| correction | power (SR 1.0) | power (SR 1.5) | FWER (SR 1.5) | realised FDR (SR 1.5) |
|---|---|---|---|---|
| uncorrected | **0.520** | **0.850** | 0.990 | 0.338 |
| Bonferroni | 0.065 | 0.314 | 0.053 | 0.014 |
| Holm | 0.065 | **0.316** | 0.053 | 0.014 |
| Benjamini-Hochberg | **0.097** | **0.509** | **0.233** | 0.043 |
| Benjamini-Yekutieli | 0.030 | 0.274 | 0.040 | 0.011 |

Read across, then down:

- **Holm is never worse than Bonferroni** and here is very slightly better (0.316 vs 0.314) —
  measured, matching Holm's own dominance theorem. There is no reason to report Bonferroni.
- 🚨 **BH's FWER is 0.233, not 0.05.** That is not a defect; BH never promised FWER. But if you
  quote BH survivors as "significant strategies", you are making a family-wise claim that BH does
  not support, and one in four such programmes will contain a false one.
- **BY costs about half of BH's power** (0.274 vs 0.509) and buys freedom from having to argue
  about your dependence structure.
- At a true annualised Sharpe of 1.0 over 4 years — `t = 2.0` — **no correction has usable
  power**. That is not the correction's fault: a t of 2.0 is not evidence at this trial count.
  More data, or fewer trials.

## 5. ⚠️ Dependence: a common factor is the case BH is proved for

100 strategies sharing a factor, 300 replications, 10 real at annualised Sharpe 1.5:

| rho | mean abs correlation | BH FDR | BH power | BY FDR | BY power | Holm power |
|---|---|---|---|---|---|---|
| 0.0 | 0.025 | 0.040 | 0.520 | 0.004 | 0.282 | 0.325 |
| 0.3 | 0.290 | 0.041 | 0.501 | 0.014 | 0.286 | 0.320 |
| 0.7 | 0.688 | 0.035 | 0.477 | 0.010 | 0.291 | 0.318 |

✅ BH's realised FDR stays at or under 5% in every row. A common positive factor is PRDS, the
condition Theorem 1.2 needs. **But "my strategies are correlated" is not on its own a licence to
use BH** — the theorem needs positive regression dependence *on the true nulls*, and a screen
containing long and short versions of the same signal is not that. If you cannot argue it, BY is
the honest choice and the power column above is the bill.

## 6. The haircut: what is left of a Sharpe after the count

Observed Sharpe → `t = SR_annual * sqrt(years)` → p → adjusted p → back to a Sharpe. Bonferroni,
the Sharpe you may still report and the percentage taken off:

| observed SR | years | t | p | N=10 | N=100 | N=1000 |
|---|---|---|---|---|---|---|
| 0.40 | 10 | 1.26 | 2.1e-01 | **0.00 (100%)** | 0.00 (100%) | 0.00 (100%) |
| 0.75 | 20 | 3.35 | 8.0e-04 | 0.59 (21%) | 0.39 (48%) | 0.06 (92%) |
| 1.00 | 4 | 2.00 | 4.6e-02 | 0.37 (63%) | 0.00 (100%) | 0.00 (100%) |
| 1.00 | 10 | 3.16 | 1.6e-03 | 0.76 (24%) | 0.45 (55%) | 0.00 (100%) |
| 1.50 | 10 | 4.74 | 2.1e-06 | 1.35 (10%) | 1.17 (22%) | 0.97 (35%) |
| 2.00 | 10 | 6.32 | 2.5e-10 | **1.88 (6%)** | 1.76 (12%) | 1.63 (18%) |

🚨 **The haircut is wildly non-linear, and the "discount every backtest by 50%" rule of thumb is
wrong in both directions.** At N = 100 trials the same correction takes 100% off a Sharpe of 0.4
and 12% off a Sharpe of 2.0. ✅ Harvey & Liu say this explicitly in *Backtesting* (Journal of
Portfolio Management, 2015; read from the authors' Duke-hosted PDF): a Sharpe under 0.4 is
usually cut by more than half, one above 1.0 by at most about 25%, and 50% is "too lenient for
relatively small Sharpe ratios and too harsh for large ones". Their article opens by naming the
50% discount as the common practice it exists to correct.

Note the third row. **A 4-year Sharpe of 1.0 is `t = 2.0` and survives nothing.** Compare
`../backtest-overfitting/SKILL.md` §6: the same claim needs 5.18 years at 50 trials.

## 7. ✅ Reproducing the published numbers

✅ **First, against a library.** On 500 seeded p-values the four standard procedures here
reproduce `statsmodels.stats.multitest.multipletests` 0.15.0 exactly — max absolute difference in
the adjusted p-values **0.0** (Bonferroni, Holm), **1.1e-16** (BH) and **2.2e-16** (BY), with
identical rejection sets. Nothing here is a private variant. Use statsmodels in production; this
file exists for the ledger integration, the haircut, and the measurements above.

Harvey, Liu & Zhu's own six-test example, `p = [0.005, 0.009, 0.0128, 0.0135, 0.045, 0.06]`,
`c(6) = 2.45`. The script reproduces their adjusted p-values exactly:

| method | adjusted p-values | significant at 5% |
|---|---|---|
| Bonferroni | 0.0300, 0.0540, 0.0768, 0.0810, 0.2700, 0.3600 | 1 |
| Holm | 0.0300, 0.0450, 0.0512, 0.0512, 0.0900, 0.0900 | 2 |
| their BHY | 0.0496 x4, then **0.0600, 0.0600** | 4 |
| standard BY | 0.0496 x4, then **0.1323, 0.1470** | 4 |

🚨 **Their BHY is not the standard Benjamini-Yekutieli adjusted p-value.** Their recursion sets
`p_(M) = p_(M)` — the largest adjusted p-value is the raw one, with no `c(M)` factor — and then
takes running minima downward. This is printed that way in the paper **and coded that way in
their own `Haircut_SR.m`** (the `if kk == (M+1)` branch), so it is the implementation, not a
typo. On this example the rejection sets agree. They do not always: ✅ **six tests all at
p = 0.04 with alpha = 5% — standard BY rejects 0, their BHY rejects all 6.** Use
`benjamini_yekutieli` when you want Theorem 1.3's guarantee; use `harvey_liu_bhy` only to
reproduce their numbers.

✅ Their haircut illustration, reproduced end to end: annualised Sharpe **0.75** over **20
years** is `t = 3.354`, `p = 0.0008`; with **200 independent tests** the adjusted p is
`1 - (1-p)^200 = 0.1473` and the haircut Sharpe is **0.324** — the paper reports 0.32 and
"approximately 60%". ⚠️ Note the adjustment there is `1 - (1-p)^N`, **not** Bonferroni; on this
input Bonferroni gives 0.1592 and a Sharpe of 0.315, a difference of 0.009.

✅ Harvey, Liu & Zhu, *… and the Cross-Section of Expected Returns*, RFS 29(1), 2016: **313
articles, 316 factors**, and the abstract's hurdle is a **t-statistic above 3.0**. ⚠️ The body is
more careful than the headline — it puts the 5%-significance threshold at about **2.8** and
argues 3.0 is if anything too low. Their Figure 3 benchmarks: Bonferroni 3.78 (2012) rising to
4.00 (2032), Holm 3.64 at 316 factors, BHY 3.39 at FDR 1% and 2.78 at FDR 5%. 🔴 The figure
"447 factors" circulates widely and is **not** in that paper; do not quote it.

## 8. Traps

- 🚨 **`m` = the trials that finished** — §1. It is the registered count.
- 🚨 **Reporting BH survivors as "significant"** — §4. BH's FWER was 0.233 here.
- 🚨 **Reaching for BH because "Bonferroni is too strict for correlated tests"** — §5. Correlation
  is not automatically PRDS, and BH is not a dependence correction; BY is.
- 🚨 **`statsmodels.stats.multitest.multipletests(method='fdr_by')` is standard BY**, so it will
  not reproduce Harvey-Liu's BHY column — §7. Neither is wrong; they are different procedures.
- 🚨 **Adjusting a p-value that was never valid.** A t-statistic from a leaked backtest is not a
  test statistic. `../research-integrity-guards/SKILL.md` comes first, and
  `fin_skills.api.guards.research_audit.reality_check` enforces the ordering.
- ⚠️ **Two-sided or one-sided.** Every p-value here is two-sided by default. A directional
  strategy claim is arguably one-sided, which halves p and moves every threshold; pick one, state
  it, and do not switch after seeing the result.
- ⚠️ **The t-statistic assumes IID.** Autocorrelated or fat-tailed returns need a Newey-West or
  bootstrap standard error first — `../backtest-validation/SKILL.md` §3 owns that. Harvey & Liu's
  own program applies an autocorrelation correction before the haircut.
- ⚠️ **A correction cannot rescue an underpowered programme.** §4, first column: at `t = 2.0` no
  method has usable power.

## 9. Scripts

- `scripts/research_history.py` — `bonferroni`, `holm`, `benjamini_hochberg`,
  `benjamini_yekutieli`, `harvey_liu_bhy`, `by_constant`, `compare`, `haircut_sharpe`,
  `independent_pvalue`, and `from_ledger` / `ResearchHistory`, which read the existing
  `TrialLedger` (reached as a package sibling, an installed module, or a relative path — it is
  never re-implemented). `__main__` measures §1 and §3-§6 and reproduces §7. Seeds 11, 20000+,
  30000+, 40000+, 55, 77. **10 s.**

## 10. Where this sits

`../backtest-validation/SKILL.md` owns `trial_ledger.py` (the format this skill reads), the
Deflated and Probabilistic Sharpe Ratios, and the `arch.bootstrap` family — SPA, StepM and MCS —
which correct for the same search by resampling instead of by adjusting p-values, and are the
better answer when you kept every candidate's per-period returns.
`../backtest-overfitting/SKILL.md` owns PBO, CSCV and the Minimum Backtest Length, which handle
dependence by construction rather than by choosing a correction.
`../research-integrity-guards/SKILL.md` owns the leakage, survivorship and cost gates that come
first. `../factor-and-timeseries-research/SKILL.md` owns the factor tests whose p-values land
here. `fin_skills.api.guards.research_audit` runs the whole chain in order and stops at the
first failure.
