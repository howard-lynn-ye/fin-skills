"""Guard: Hansen's SPA on the whole search (backtest-validation / spa_test.py)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from fin_skills.api.base import Guard, Outcome, register
from fin_skills.api.guards._common import as_1d, as_2d
from fin_skills.core.spa_test import (ArchMissing, _require_arch, naive_best_of_n_pvalue,
                                      spa_test, spa_test_selfcontained)


@register
class SpaGuard(Guard):
    """Is the best of everything you tried better than the benchmark, after correcting for the search?

    Inputs
        benchmark_returns : (T,) per-period returns of the benchmark / incumbent.
        model_returns     : (T, k) per-period returns of EVERY candidate, including the
                            abandoned ones (omitting them defeats the test).
        reps              : bootstrap replications. Default 1000.
        block_size        : stationary-bootstrap block; from optimal_block_length when None.
        seed              : bootstrap seed.
        alpha             : significance level. Default 0.05.
        use_arch          : True to require arch, False to force the self-contained
                            bootstrap, None (default) to use arch when installed.

    Both take RETURNS (higher is better); the negation to losses happens once, inside
    the wrapped function. Fails when the consistent p-value cannot reject H0: the best
    candidate is consistent with luck. The naive t-test on the winner is in the
    evidence so the size of the correction is visible.
    """

    name = "spa_test"
    skill = "backtest-validation"
    summary = "Superior-predictive-ability test on returns; fails when the best candidate is consistent with luck."
    wraps = ("fin_skills.core.spa_test.spa_test",
             "fin_skills.core.spa_test.spa_test_selfcontained",
             "fin_skills.core.spa_test.naive_best_of_n_pvalue")
    required = ("benchmark_returns", "model_returns")
    optional = ("reps", "block_size", "seed", "alpha", "use_arch")

    def check(self, benchmark_returns: pd.Series | np.ndarray,
              model_returns: pd.DataFrame | np.ndarray, reps: int = 1000,
              block_size: int | None = None, seed: int | None = None, alpha: float = 0.05,
              use_arch: bool | None = None) -> Outcome:
        out = Outcome()
        bench = as_1d(benchmark_returns, "benchmark_returns")
        models = as_2d(model_returns, "model_returns")
        if models.shape[0] != bench.shape[0]:
            raise TypeError(f"benchmark has {bench.shape[0]} periods, models have "
                            f"{models.shape[0]}; align them on one calendar")
        if not (0.0 < alpha < 1.0):
            raise TypeError("alpha must be in (0, 1)")

        have_arch = True
        if use_arch is not False:
            try:
                _require_arch()
            except ArchMissing as exc:
                if use_arch is True:
                    raise TypeError(str(exc).splitlines()[0]) from exc
                have_arch = False
        else:
            have_arch = False
        runner = spa_test if have_arch else spa_test_selfcontained
        res = runner(bench, model_returns if isinstance(model_returns, pd.DataFrame) else models,
                     reps=reps, block_size=block_size, seed=seed)
        winner, p_naive = naive_best_of_n_pvalue(bench, model_returns
                                                 if isinstance(model_returns, pd.DataFrame)
                                                 else models)
        out.note(pvalue=res.pvalue, pvalues=res.pvalues, best_model=res.best_model,
                 better_models=res.better_models, mean_excess=res.mean_excess,
                 block_size=res.block_size, block_source=res.block_source, reps=res.reps,
                 n_obs=res.n_obs, n_candidates=len(res.model_names), source=res.source,
                 naive_winner=winner, naive_pvalue=p_naive, report=res.report(alpha))
        if not have_arch:
            out.warning("arch is not installed: self-contained stationary bootstrap used "
                        "(consistent p-value only); install arch for real work", where="arch")
        if res.pvalue >= alpha:
            out.error(f"CANNOT REJECT H0 at {alpha:.0%}: best of {len(res.model_names)} "
                      f"candidates ({res.best_model}) is consistent with luck, consistent "
                      f"p = {res.pvalue:.4f} (naive t-test on the winner said p = {p_naive:.4f})",
                      where="spa")
        else:
            out.info(f"REJECT H0 at {alpha:.0%}: consistent p = {res.pvalue:.4f}; best "
                     f"{res.best_model}, survivors {res.better_models or '-'}", where="spa")
        return out
