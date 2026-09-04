#!/usr/bin/env python3
"""SPA / StepM / RealityCheck / MCS take LOSSES. Feeding them returns inverts the test.

`arch.bootstrap`'s multiple-comparison procedures are documented on losses - lower is
better. `SPA`/`StepM` call their arguments "benchmark losses" and "model losses";
`MCS`'s first positional parameter is literally named `losses`. Returns are the same
shape and the same dtype, so passing them raises nothing and warns nothing.

The internal quantity is `loss_diff = benchmark - models`, and every decision is
`loss_diff.mean(0) > critical_value`. Flip the sign of the input and you flip that
comparison. The procedure still answers, and answers confidently, but it answers about
the models with the LOWEST returns: the worst strategy in the set is named the best,
with a small p-value attached. The p-value is small either way, so it cannot warn you.

Run:  python spa_direction.py
`arch` is optional. Without it the reference SPA/StepM below reproduce the same
algorithm from Hansen (2005) using only numpy/pandas, so the trap still demonstrates.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PERIODS = 252
T = 1500          # ~6 years of daily observations
REPS = 1000
SEED = 7          # bootstrap seed, passed to arch as well
BENCH_SHARPE = 0.3


# --------------------------------------------------------------------------
# Reference implementation: Hansen (2005) SPA, exactly as arch computes it.
# Only numpy/pandas. The stationary-bootstrap draws replicate arch's own RNG
# call order, so the p-values below are comparable digit for digit.
# --------------------------------------------------------------------------
def _stationary_bootstrap_indices(rng: np.random.Generator, t: int, block_size: int):
    """Politis-Romano (1994) stationary bootstrap: geometric blocks, circular wrap.

    Same two RNG draws, in the same order, as arch's StationaryBootstrap.
    """
    idx = rng.integers(t, size=t, dtype=np.int64)
    u = rng.random(t)
    p = 1.0 / block_size
    pos = np.arange(t)
    new_block = u <= p                      # u > p continues the previous block
    new_block[0] = True
    start = np.maximum.accumulate(np.where(new_block, pos, -1))
    return (idx[start] + (pos - start)) % t


def _loss_diff_variance(ld: np.ndarray, block_size: int) -> np.ndarray:
    """Kernel-weighted long-run variance of each loss differential (Hansen 2005 eq. 9)."""
    t = ld.shape[0]
    demeaned = ld - ld.mean(axis=0)
    p = 1.0 / block_size
    var = np.sum(demeaned ** 2, axis=0) / t
    for i in range(1, t):
        kappa = (1.0 - i / t) * ((1 - p) ** i) + (i / t) * ((1 - p) ** (t - i))
        var += 2 * kappa * np.sum(demeaned[: t - i, :] * demeaned[i:, :], axis=0) / t
    return var


class RefSPA:
    """SPA on LOSSES. `loss_diff = benchmark - models`; positive means model is better."""

    def __init__(self, benchmark, models, block_size=None, reps=REPS, seed=None):
        bm = np.asarray(benchmark, dtype=float).reshape(-1, 1)
        md = np.asarray(models, dtype=float)
        self.loss_diff = bm - md                       # the one sign that gets inverted
        self.t, self.k = self.loss_diff.shape
        self.block_size = block_size or int(np.sqrt(self.t))
        self.reps = reps
        self.rng = np.random.default_rng(seed)
        self._sim = None
        self._selector = np.ones(self.k, dtype=bool)
        self.pvalues: dict[str, float] = {}

    def _simulate(self) -> None:
        ld, t = self.loss_diff, self.t
        var = _loss_diff_variance(ld, self.block_size)
        mean = ld.mean(0)
        # Columns far enough below zero are asymptotically irrelevant under the null.
        valid = mean >= -np.sqrt((var / t) * 2 * np.log(np.log(t)))
        upper = mean.copy()                            # always re-centre
        consistent = np.where(valid, mean, 0.0)        # re-centre only relevant columns
        lower = np.where(mean < 0, 0.0, mean)          # never re-centre the losers
        means = [lower, consistent, upper]
        sim = np.zeros((self.k, self.reps, 3))
        for i in range(self.reps):
            star = ld[_stationary_bootstrap_indices(self.rng, t, self.block_size)]
            sm = star.mean(0)
            for j, m in enumerate(means):
                sim[:, i, j] = sm - m
        self._sim = sim

    def compute(self) -> "RefSPA":
        if self._sim is None:
            self._simulate()
        max_sim = np.max(self._sim[self._selector, :, :], axis=0)
        stat = np.max(self.loss_diff[:, self._selector].mean(axis=0))
        p = (max_sim > stat).mean(axis=0)
        self.pvalues = {"lower": p[0], "consistent": p[1], "upper": p[2]}
        return self

    def critical_values(self, pvalue: float = 0.05) -> dict[str, float]:
        max_sim = np.max(self._sim[self._selector, :, :], axis=0)
        cv = np.percentile(max_sim, 100.0 * (1 - pvalue), axis=0)
        return dict(zip(("lower", "consistent", "upper"), cv))

    def better_models(self, pvalue: float = 0.05, pvalue_type: str = "consistent"):
        cv = self.critical_values(pvalue)[pvalue_type]
        better = np.logical_and(self.loss_diff.mean(0) > cv, self._selector)
        return np.argwhere(better).flatten()


def ref_stepm(benchmark, models, size=0.05, reps=REPS, seed=None) -> list[int]:
    """Romano-Wolf StepM: run SPA, remove what it rejected, re-run until nothing new."""
    spa = RefSPA(benchmark, models, reps=reps, seed=seed).compute()
    found = [int(i) for i in spa.better_models(size)]
    all_found = found[:]
    while found and len(all_found) < spa.k:
        spa._selector = np.ones(spa.k, dtype=bool)
        spa._selector[np.array(all_found)] = False
        spa.compute()
        found = [int(i) for i in spa.better_models(size)]
        all_found.extend(found)
    return sorted(all_found)


# --------------------------------------------------------------------------
# Synthetic strategies of exactly known quality
# --------------------------------------------------------------------------
def make_strategies(seed: int = 0):
    """Six strategies rescaled so the REALISED annualised Sharpe equals the target.

    A common market factor gives them realistic cross-correlation - exactly the
    dependence SPA/StepM exist to correct for. Rescaling is affine per column, so it
    fixes each Sharpe without touching the correlation structure.
    """
    rng = np.random.default_rng(seed)
    target = {"S1_terrible": -1.5, "S2_bad": -0.9, "S3_flat": 0.0,
              "S4_ok": 0.5, "S5_good": 1.0, "S6_excellent": 1.6}
    sigma = 0.01                                     # 1% daily vol
    factor = rng.normal(0.0, 0.006, T)               # common market component

    def calibrate(x: np.ndarray, sharpe: float) -> np.ndarray:
        z = (x - x.mean()) / x.std(ddof=1)
        return sharpe * sigma / np.sqrt(PERIODS) + sigma * z

    idx = pd.bdate_range("2018-01-01", periods=T)
    cols = {n: calibrate(factor + rng.normal(0.0, 0.008, T), s) for n, s in target.items()}
    returns = pd.DataFrame(cols, index=idx)
    benchmark = pd.Series(calibrate(factor + rng.normal(0.0, 0.004, T), BENCH_SHARPE),
                          index=idx, name="benchmark")
    return returns, benchmark, target


def ann_sharpe(x) -> float:
    return float(np.mean(x) / np.std(x, ddof=1) * np.sqrt(PERIODS))


# --------------------------------------------------------------------------
def main() -> None:
    returns, benchmark, target = make_strategies()
    names = list(returns.columns)

    print(f"Six synthetic strategies of known quality (seed=0, T={T} daily obs)\n")
    print(f"  {'strategy':<14}{'target Sharpe':>15}{'realised Sharpe':>18}")
    for c in names:
        print(f"  {c:<14}{target[c]:>15.2f}{ann_sharpe(returns[c]):>18.4f}")
    print(f"  {'benchmark':<14}{BENCH_SHARPE:>15.2f}{ann_sharpe(benchmark):>18.4f}")
    print(f"\n  ground truth: best = {names[-1]}, worst = {names[0]}")

    # ---- the correct call: losses = -returns -----------------------------
    losses_bm, losses_md = -benchmark, -returns
    ok = RefSPA(losses_bm, losses_md, reps=REPS, seed=SEED).compute()
    ok_better = [names[i] for i in ok.better_models(0.05)]
    ok_stepm = [names[i] for i in ref_stepm(losses_bm, losses_md, 0.05, REPS, SEED)]

    # ---- the trap: returns passed straight in ----------------------------
    bad = RefSPA(benchmark, returns, reps=REPS, seed=SEED).compute()
    bad_better = [names[i] for i in bad.better_models(0.05)]
    bad_stepm = [names[i] for i in ref_stepm(benchmark, returns, 0.05, REPS, SEED)]

    bar = "=" * 76
    print("\n" + bar + "\n  CORRECT -- losses = -returns\n" + bar)
    print(f"  SPA p-values          lower {ok.pvalues['lower']:.4f}   "
          f"consistent {ok.pvalues['consistent']:.4f}   upper {ok.pvalues['upper']:.4f}")
    print(f"  SPA.better_models     {ok_better}")
    print(f"  StepM.superior_models {ok_stepm}")

    print("\n" + bar + "\n  TRAP -- returns passed as losses (no exception, no warning)\n" + bar)
    print(f"  SPA p-values          lower {bad.pvalues['lower']:.4f}   "
          f"consistent {bad.pvalues['consistent']:.4f}   upper {bad.pvalues['upper']:.4f}")
    print(f"  SPA.better_models     {bad_better}")
    print(f"  StepM.superior_models {bad_stepm}")

    ok_top = names[int(np.argmax(ok.loss_diff.mean(0)))]
    bad_top = names[int(np.argmax(bad.loss_diff.mean(0)))]
    print(f"\n  ranked first by the test:  correct -> {ok_top}   trap -> {bad_top}")
    print(f"  the trap crowns {bad_top} (Sharpe {target[bad_top]:+.1f}) over "
          f"{ok_top} (Sharpe {target[ok_top]:+.1f}) -- a swing of "
          f"{target[ok_top] - target[bad_top]:.1f} Sharpe")
    print(f"  both directions reject at 5% -- consistent p = "
          f"{ok.pvalues['consistent']:.4f} correct, {bad.pvalues['consistent']:.4f} trap -- "
          f"so the\n  p-value cannot tell you which way round the input went")

    # ---- verify the reference implementation against the real library ----
    try:
        from arch.bootstrap import MCS, SPA, StepM
    except ImportError:
        print("\n  arch not installed - reference implementation only "
              "(the inversion above is the point and needs no library)")
        return

    a_ok = SPA(losses_bm, losses_md, reps=REPS, seed=SEED); a_ok.compute()
    a_bad = SPA(benchmark, returns, reps=REPS, seed=SEED); a_bad.compute()
    s_ok = StepM(losses_bm, losses_md, size=0.05, reps=REPS, seed=SEED); s_ok.compute()
    s_bad = StepM(benchmark, returns, size=0.05, reps=REPS, seed=SEED); s_bad.compute()
    m_ok = MCS(losses_md, size=0.10, reps=REPS, seed=SEED); m_ok.compute()
    m_bad = MCS(returns, size=0.10, reps=REPS, seed=SEED); m_bad.compute()

    print("\n" + bar + "\n  VERIFIED against installed arch\n" + bar)
    for tag, ref, live in (("correct", ok, a_ok), ("trap", bad, a_bad)):
        d = max(abs(ref.pvalues[k] - float(live.pvalues[k]))
                for k in ("lower", "consistent", "upper"))
        print(f"  SPA {tag:<8} reference consistent p = {ref.pvalues['consistent']:.4f}   "
              f"arch = {float(live.pvalues['consistent']):.4f}   max|diff| = {d:.6f}")
    same = (ok_stepm == list(s_ok.superior_models)
            and bad_stepm == list(s_bad.superior_models))
    print(f"  StepM.superior_models   correct: {list(s_ok.superior_models)}")
    print(f"  StepM.superior_models   trap:    {list(s_bad.superior_models)}")
    print(f"  reference StepM reproduces arch's sets exactly: {same}")
    print(f"  MCS.included            correct: {list(m_ok.included)}")
    print(f"  MCS.included            trap:    {list(m_bad.included)}")

    print("\n  Neither arch call raised. Neither warned. Only the sign of the input"
          "\n  distinguishes them, and it decided which strategy gets funded.")
    print("\n  Rule: losses = -returns before SPA / StepM / RealityCheck / MCS."
          "\n        Already-loss-shaped inputs (squared error, NLL, drawdown) go in as-is."
          "\n        A Sharpe, or any other ratio, is not a valid input at all.")


if __name__ == "__main__":
    main()
