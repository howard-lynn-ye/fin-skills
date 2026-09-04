# The solver layer

Every convex portfolio optimizer in Python sits on **cvxpy**, which compiles your problem and hands
it to a solver. The solver decides what is expressible, how fast it runs, and — for two of them —
what licence you need. Read this before debugging a `SolverError`.

`cvxpy` 1.9.2 (2026-06-22) is the current line.

## Which solver

| Solver | Handles | Licence | Notes |
|---|---|---|---|
| **Clarabel** | LP, QP, SOCP, SDP, exponential | Apache-2.0 | Rust interior-point; **the modern default** and cvxpy's preferred conic solver |
| **OSQP** | QP only | Apache-2.0 | First-order, very fast on plain mean-variance; loose tolerances by default |
| **ECOS** | LP, QP, SOCP, exponential | GPL-3.0 | Long-time default; **GPL — check before shipping** |
| **SCS** | LP, QP, SOCP, SDP, exponential | MIT | First-order, robust on large/degenerate problems, lower accuracy |
| **MOSEK** | everything, incl. MIP | 🔴 **commercial** (free academic) | Fastest and most reliable on hard problems |
| **Gurobi** | everything, incl. MIP | 🔴 **commercial** (free academic) | The answer for **cardinality / integer** constraints |
| **HiGHS** | LP, MIP | MIT | Good open MIP option |
| **CBC / GLPK_MI** | MIP | EPL / GPL | Slow but free |

```python
import cvxpy as cp
print(cp.installed_solvers())
prob.solve(solver=cp.CLARABEL, verbose=True)   # always name the solver in research code
```

**Pin the solver explicitly.** cvxpy's automatic choice depends on what happens to be installed, so
an unpinned optimization is not reproducible across machines.

## Problem class decides the solver

- **Mean-variance, long-only, budget constraint** → QP → OSQP or Clarabel.
- **Anything with a norm, a turnover constraint, or a CVaR objective** → SOCP → Clarabel/ECOS/MOSEK.
- **Risk parity** → not convex in weights directly; solved via a log-barrier reformulation (SOCP/
  exponential cone) — Clarabel or MOSEK.
- **Cardinality ("at most 30 names"), minimum position size, lot sizes** → **MIP**, which needs
  Gurobi/MOSEK/HiGHS/CBC. 🚨 This is the single most common reason a "simple" constraint makes a
  working optimizer fail — the problem class changed, not the code.
- **Semidefinite** (covariance repair, some robust formulations) → SDP → Clarabel, SCS, MOSEK.

## Failure modes and what they actually mean

| Symptom | Usual cause |
|---|---|
| `SolverError` after adding one constraint | The problem left the solver's cone class — likely became a MIP |
| `infeasible` | Constraints genuinely conflict — most often long-only + a return target above the max achievable return, or sector bounds that cannot sum to 1 |
| `unbounded` | A missing budget constraint (`sum(w) == 1`) or unconstrained shorting |
| Weights that look like noise, tiny objective difference | The covariance matrix is near-singular; the optimizer is picking arbitrarily among equivalent solutions. **Shrink or denoise it** |
| Solves, but weights change wildly with a tiny input change | Estimation error, not a solver problem. See below |
| Slow beyond ~500 assets | Dense covariance in a QP; try OSQP, a factor-model covariance, or reduce the universe |

## 🚨 The estimation-error problem is bigger than the optimizer

Mean-variance is a *maximizer of estimation error*: it loads on whatever asset has the most
overstated expected return and the most understated variance. The optimizer is doing its job
correctly; the inputs are the problem.

| Technique | Where it lives |
|---|---|
| Ledoit-Wolf / OAS shrinkage | `sklearn.covariance`, `pypfopt.risk_models.CovarianceShrinkage` |
| **Marcenko-Pastur denoising / detoning** | `riskfolio.denoiseCov`, `skfolio.moments.DenoiseCovariance`/`DetoneCovariance` — 🚨 **not in PyPortfolioOpt** |
| Factor-model covariance | Build it yourself, or Riskfolio-Lib |
| Hierarchical methods (HRP/HERC/NCO) | Sidestep matrix inversion entirely |
| Black-Litterman | Shrinks expected returns toward equilibrium |

🚨 **`pypfopt.risk_models.CovarianceShrinkage` is Ledoit-Wolf — that is NOT the same as
Marcenko-Pastur denoising.** They are different estimators solving different parts of the problem,
and the two are frequently conflated.

**A practical default:** if you cannot justify your expected-return estimates, do not optimize on
them. Minimum-variance, risk parity or HRP need no return forecast and are far more stable
out-of-sample.

## Validating an optimizer

- Feed it an equal-weight-optimal synthetic covariance and check it returns equal weights.
- Perturb the covariance by 1% and measure the weight change. Large moves mean the result is noise.
- Compare against a naive 1/N benchmark **after costs**. 1/N is a famously hard benchmark to beat,
  and if your optimizer does not, that is the finding.
- Cross-validate the *portfolio construction*, not just the return forecast — this is `skfolio`'s
  distinctive capability.
