---
name: ratings-transitions-and-migration
description: >-
  Estimate and use a credit rating transition matrix without producing negative probabilities or
  a five-year default rate that is five times the wrong number. TRIGGER - rating transition
  matrix, migration matrix, credit migration, cohort estimator, duration estimator,
  Aalen-Johansen, Nelson-Aalen generator, matrix power P^5, matrix root, square root of a
  transition matrix, six-month transition matrix, scipy.linalg.logm, expm, embedding problem,
  generator of a Markov chain, "negative probability in my transition matrix", "logm gave me a
  negative off-diagonal", structural zero, AAA never defaults, withdrawn rating, NR, rating
  withdrawal, notching, cumulative default rate, "5 times the one-year PD", transitionMatrix,
  pyratings. SKIP for pricing a default probability or a hazard rate (credit-risk-models), for
  CDS quotes and upfronts (cds-mechanics-and-upfront), for bond spreads (credit-spread-measures),
  and for regulatory PD floors and Basel calibration (banking-regulatory).
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-09"
---

# Ratings transitions and migration

**A published transition matrix looks like a probability object you can raise to any power, take
any root of, and differentiate.** You can do the first one. The other two fail, quietly, because
of a single property nobody thinks of as a property: the matrix has **structural zeros** —
cells a cohort estimator reported as exactly `0.0000%` because nobody was seen doing it. A
matrix with a structural zero has **no generator**, so `scipy.linalg.logm(P)` returns a matrix
with negative off-diagonals and the implied six-month matrix contains a **negative probability**.
It is a float, it prints, and it still squares back to `P` to machine precision.

Every figure below is printed by `scripts/migration.py` (runs in **0.9 s**; numpy + scipy, seeded
at `SEED = 20260909`, no network). ✅ Measured means this file produced it on 2026-09-09 with
numpy 2.2.6, scipy 1.13.0, Python 3.11.3. The matrix is **synthetic**, shaped like a published
one — the point is the structure, not the agency's numbers.

> **The rule:** a published matrix is a **cohort estimate with structural zeros**, so it has no
> generator. **Take powers of it freely; never take a root of it without checking the sign.**
> A withdrawn rating is **censoring**, not an outcome.

## 1. 🚨 The structural zero, and why it has no generator

The synthetic one-year matrix has `AAA → D = 0.0000%` — the cell every published matrix has as a
zero, because no AAA issuer has defaulted inside a single year. ✅ Measured: rows sum to 1 to
**1.1e-16**, min entry 0, fully stochastic. And **`P²` puts 0.0056% in that same cell**, because
AAA → AA → … → D is two steps.

**No generator can produce both.** ✅ Measured on `logm(P)`:

| | ✅ value |
|---|---|
| negative off-diagonal cells in `logm(P)` | **4** |
| worst | **CCC → AA = −6.507e-05** |
| the `AAA → D` cell of the generator | **−2.6000e-05** |
| negative entries in `expm(logm(P)/2)` (the six-month matrix) | **4** |
| worst — **a negative probability** | **CCC → AA = −1.494e-05** |
| the `AAA → D` cell of the six-month matrix | **−6.6174e-06** |
| `root @ root − P` | **1.1e-15** |

🚨 **The last row is the trap.** The six-month matrix squares back to the one-year matrix to
machine precision, so a repricing check passes, a row-sum check passes, and a "did the root
work" check passes. The only test that catches it is looking at the sign of every entry.

### ✅ The correspondence is exact, which makes it a diagnostic

✅ Measured: the set of negative cells in `logm(P)` is **exactly** the set of structural zeros
of `P` — `AAA→D`, `B→AAA`, `CCC→AAA`, `CCC→AA`, and nothing else. **So you can predict which
cells will break before you call `logm`:** they are the off-diagonal zeros. A matrix with no
structural zeros is usually embeddable; every zero you see is a cell that will come back
negative.

## 2. 🚨 "Not exactly zero" is not enough

The obvious fix is to put a small number in the empty cells. ✅ Measured — the same fill in every
structural zero, rows renormalised:

| fill | min generator off-diagonal | valid generator | min six-month entry |
|---|---|---|---|
| 1e-06 | −6.386e-05 | **no** | −1.439e-05 |
| 1e-05 | −5.288e-05 | **no** | −9.432e-06 |
| 3e-05 | −2.850e-05 | **no** | 0.000e+00 |
| 5e-05 | −4.118e-06 | **no** | 0.000e+00 |
| **1e-04** | **0.000e+00** | **yes** | 0.000e+00 |
| 5e-04 | 0.000e+00 | yes | 0.000e+00 |

🚨 **A token epsilon does not work.** The cell has to be large enough to be consistent with the
two-step paths that reach it; at 1e-06 the generator is barely better than at zero. On this
matrix the threshold is between **5e-05 and 1e-04** — around **1 bp**, which is a hundred times
the epsilon most people would reach for.

🚨 **And the two tests disagree.** At a 3e-05 fill the six-month matrix is already non-negative
while the generator is still invalid. **Checking the root does not check the generator**; if you
need both, test both.

## 3. ✅ The repair, and what it costs

The standard fix (Israel–Rosenthal–Wei) is to clip the negative off-diagonals of `logm(P)` to
zero and put the difference back on the diagonal so the rows still sum to zero.
✅ `regularised_generator()` measured:

| | ✅ result |
|---|---|
| valid generator (non-negative off-diagonals, zero row sums) | **True** |
| min entry of the six-month matrix | **0.000e+00** |
| 🚨 `expm(Q_reg)` vs the original `P`, worst cell | **0.51 bp** |

**You cannot have both properties.** Either the root is a probability matrix, or the generator
reproduces the published matrix. **Decide which one your use needs before you take the root** —
a scenario generator needs non-negativity; a calibration report needs the published numbers back.

## 4. 🚨 `5 × PD₁` is wrong in both directions

✅ Measured, `P⁵` against five times the one-year default rate:

| grade | 1y PD | **true 5y (`P⁵`)** | `5 × PD₁` | error |
|---|---|---|---|---|
| AAA | 0.0000% | **0.0636%** | 0.0000% | −6.4 bp |
| AA | 0.0200% | 0.2581% | 0.1000% | −15.8 bp |
| A | 0.0600% | 0.6255% | 0.3000% | −32.6 bp |
| **BBB** | 0.2400% | **2.1129%** | 1.2000% | **−91.3 bp** |
| BB | 0.9000% | 7.3519% | 4.5000% | −285.2 bp |
| B | 4.2000% | 22.1944% | 21.0000% | −119.4 bp |
| **CCC** | 19.8000% | **55.9554%** | **99.0000%** | **+4,304.5 bp** |

🚨 **AAA has a five-year default probability of 0.0636% and a one-year rate of exactly zero**, so
the naive scaling gives zero for all time. Investment grade defaults arrive through **migration**
— a AAA credit defaults by first becoming a B credit — which linear scaling has no way to
represent. At the other end, **CCC's naive five-year PD is 99.00% against a true 55.96%**: a
one-year rate of 19.8% is simply too large to multiply, and the survival term dominates.

**The sign flips somewhere in single-B**, so there is no "conservative direction" to round in.

## 5. ✅ Cohort vs duration: same data, one usable answer

The two estimators differ in what they are allowed to see:

- **Cohort** (multinomial): compare the rating on 1 January with the rating on 31 December,
  `N_ij / N_i`. It cannot see anything that happened in between, and it reports **exact zeros**.
- **Duration** (Aalen-Johansen / Nelson-Aalen): count every transition and the **time at risk**
  that produced it, `Q_ij = N_ij / Y_i`, then `P = expm(Q)`. It is a generator by construction.

✅ Measured: 4,000 firms simulated over 20 years from a **valid generator whose AAA → D intensity
is exactly zero**. Even so, the truth has a positive one-year `AAA → D` of **0.002472%** — the
two-step paths exist.

| grade | ✅ true 1y PD | cohort | duration |
|---|---|---|---|
| **AAA** | **0.002472%** | **0.000000%** | **0.002175%** |
| AA | 0.020007% | 0.035877% | 0.039312% |
| A | 0.060000% | 0.057127% | 0.070703% |
| BBB | 0.240000% | 0.241501% | 0.245855% |
| BB | 0.899998% | 0.950794% | 0.874775% |
| B | 4.199977% | 4.087229% | 4.033595% |
| CCC | 19.799391% | 18.202899% | 18.602471% |

🚨 **The cohort estimate is exactly zero where the truth is 0.002472%** — not small, *zero*,
which is a different kind of number and the one that destroys embeddability. ✅ **The duration
estimate is 0.002175%, from the same firms**, because `AAA → AA → … → D` was observed even
though the direct jump was not.

✅ And the consequence: the **duration estimate is a valid generator (`True`), and its six-month
root's minimum entry is 0.000e+00**, against **−1.632e-05** for the cohort matrix built from the
identical data. **If you need a matrix at a non-annual horizon, estimate a generator; do not take
the root of a cohort matrix.**

⚠️ The duration estimator is not free: it needs the **dates** of rating actions, not just annual
snapshots, and it assumes time-homogeneity over the estimation window more strongly than the
cohort estimator does.

## 6. 🚨 A withdrawn rating is censoring, not an outcome

Issuers stop being rated — debt is repaid, the issuer is acquired, the mandate ends. The rate is
not uniform: it rises as the grade falls. ✅ Measured with a grade-dependent withdrawal hazard,
**2,433 of 4,000 firms withdrawn over 20 years**, five-year default probabilities:

| grade | ✅ true 5y PD | censored (correct) | error | 🚨 NR as a state | error |
|---|---|---|---|---|---|
| A | 0.6255% | 0.5722% | −5 bp | 0.4732% | −15 bp |
| **BBB** | **2.1128%** | 1.9877% | −13 bp | **1.5742%** | **−54 bp** |
| BB | 7.3518% | 8.3652% | +101 bp | 6.1321% | −122 bp |
| B | 22.1935% | 22.7492% | +56 bp | **16.2643%** | **−593 bp** |
| **CCC** | **55.9491%** | 56.9089% | +96 bp | **38.4711%** | **−1,748 bp** |

🚨 **Treating NR as an absorbing state understates every meaningful default probability, and by
1,748 bp for CCC.** After five years it has parked **22.76% of the BBB cohort in NR** —
probability that should have gone on to migrate and default is sitting in a state that cannot
default. The rows still sum to one and every validity check passes.

✅ **Censoring the firm-year — dropping observations where the rating is missing at either end —
is unbiased**, and its errors above are sampling noise: they are **both signs** and at most 101 bp
on the noisiest row, against a one-directional bias that grows monotonically as the grade falls.

⚠️ Real withdrawals are worse than this simulation, which withdraws at a rating-dependent but
otherwise random hazard. In practice withdrawal is correlated with the *event* — an issuer in
trouble stops paying for a rating — so the residual bias is toward understating default even
after censoring.

## 7. What the script gives you

`scripts/migration.py` — numpy and `scipy.linalg` (`expm`, `logm`) only, seeded.

| Function | Does |
|---|---|
| `published_matrix()` / `check_stochastic(p)` | §1, the synthetic matrix and its validity |
| `structural_zeros(p)` | §1, the cells that will come back negative |
| `generator_log(p)` / `is_valid_generator(q)` | `logm` with a complex-part check, and the test |
| `embedding_report(p)` | §1, the negatives, the root, and the exact correspondence |
| `fill_structural_zeros(p, fill)` / `fill_scan(fills)` | §2, how big the cell has to be |
| `regularised_generator(p)` | §3, the clip-and-rebalance repair and its cost |
| `matrix_power` / `cumulative_pd` / `horizon_trap` | §4 |
| `true_generator()` / `simulate_panel(...)` | §5, the seeded panel of rating histories |
| `cohort_estimator(yearly)` / `duration_estimator(counts, exposure)` | §5, the two estimators |
| `estimator_comparison()` | §5, both against the simulated truth |
| `withdrawal_bias()` | §6, three treatments of NR |

⚠️ Two maintained packages exist and neither is a dependency here: `transitionMatrix` 0.5.1
(2022-02-21, Apache-2.0, open-risk) implements both estimators, and `pyratings` 0.6.1
(2023-02-24, Apache-2.0, HSBC) translates between agency scales and notches. Both are years
past their last release; check them before depending on them.

## Where this sits

- `../../../fin-models/skills/credit-risk-models/SKILL.md` — the PDs above are **physical**
  (they come from counting defaults). Pricing needs the **risk-neutral** one, and that skill
  measures the gap: a 1-year PD of 8.4% physical against 12.7% risk-neutral on the same firm.
  🚨 Putting a column of this matrix into a CDS pricer is exactly the measure trap.
- `../cds-mechanics-and-upfront/SKILL.md` — the 100 bp / 500 bp standard coupon split follows
  the investment-grade / high-yield line, so a downgrade changes which coupon a name trades on.
- `../credit-spread-measures/SKILL.md` — index membership and index OAS series are defined by
  rating, so a migration moves a bond between indices as well as between rating buckets.
- `../corporate-bond-data-and-trace/SKILL.md` — the rating also picks the TRACE dissemination
  cap ($5MM IG vs $1MM HY), which is why a liquidity screen on TRACE volume is partly a screen
  on rating.
- `../../../fin-models/skills/risk-measures-var-cvar/SKILL.md` — a migration matrix is the
  transition kernel of a CreditMetrics-style loss distribution; that skill owns the quantile
  estimation and its backtests.
- **`banking-regulatory`** (federated pack in `.claude-plugin/marketplace.json`) — regulatory PD
  floors, through-the-cycle versus point-in-time calibration, and IRB risk weights. ⚠️ Not
  verified by this repo. This skill deliberately stops at the estimator.
- `../../../fin-market-data/skills/market-data-sourcing/SKILL.md` — where the matrices come from: the
  agencies' annual default studies are free PDFs, issuer-level rating histories are not.
