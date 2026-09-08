"""Guard: returns off a stitched futures series (futures-continuous-contracts / continuous_contract.py)."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from fin_skills.api.base import Guard, Outcome, register
from fin_skills.api.guards._common import require_series
from fin_skills.futures_fx.continuous_contract import safe_returns, true_roll_return


@register
class ContinuousContractGuard(Guard):
    """Refuse pct_change() on a series whose stitching method makes returns meaningless.

    Inputs
        stitched   : the Series returned by continuous_contract.stitch (its .attrs carry
                     the method).
        contracts  : optional near->far contract frame used to build it.
        roll_dates : optional roll dates; with `contracts`, the stitched returns are
                     checked against true_roll_return, the ground truth.
        atol       : tolerance for that comparison. Default 1e-9.

    Fails when the series is 'unadjusted' (fake jump at every roll) or 'difference'
    (crosses zero; sign-flipped returns), when it carries no stitching provenance, or
    when its returns disagree with the true rolled-position return.
    """

    name = "continuous_contract"
    skill = "futures-continuous-contracts"
    summary = "Fails returns taken off an unadjusted or difference-adjusted continuous series; verifies ratio vs truth."
    wraps = ("fin_skills.futures_fx.continuous_contract.safe_returns",
             "fin_skills.futures_fx.continuous_contract.true_roll_return")
    required = ("stitched",)
    optional = ("contracts", "roll_dates", "atol")

    def check(self, stitched: pd.Series, contracts: pd.DataFrame | None = None,
              roll_dates: Sequence | None = None, atol: float = 1e-9) -> Outcome:
        out = Outcome()
        stitched = require_series(stitched, "stitched")
        if (contracts is None) != (roll_dates is None):
            raise TypeError("contracts and roll_dates must be given together")
        method = stitched.attrs.get("method")
        out.note(method=method, attrs=dict(stitched.attrs), min_level=float(stitched.min()),
                 goes_negative=bool(stitched.min() < 0))
        try:
            rets = safe_returns(stitched)
        except ValueError as exc:
            out.error(str(exc), where="method")
            return out
        out.note(n_returns=int(len(rets)))
        out.info(f"{method!r}-stitched: pct_change reproduces a rolled position's returns",
                 where="method")

        if contracts is not None:
            if not isinstance(contracts, pd.DataFrame):
                raise TypeError("contracts must be a DataFrame (dates x contracts, near->far)")
            truth = true_roll_return(contracts, roll_dates)
            common = rets.index.intersection(truth.index)
            if len(common) == 0:
                raise TypeError("stitched series and contracts share no dates")
            err = float(np.abs(rets.loc[common] - truth.loc[common]).max())
            out.note(max_abs_error_vs_truth=err, n_compared=int(len(common)))
            if err > atol:
                out.error(f"stitched returns disagree with true_roll_return by up to {err:.3g} "
                          f"over {len(common)} days (atol {atol:g})", where="truth")
            else:
                out.info(f"matches true_roll_return to {err:.1e} over {len(common)} days",
                         where="truth")
        return out
