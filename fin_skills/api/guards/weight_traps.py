"""Guard: prices handed to an optimiser that wants returns (lib-pyportfolioopt / weight_traps.py)."""
from __future__ import annotations

import pandas as pd

from fin_skills.api.base import Guard, Outcome, register
from fin_skills.api.guards._common import require_frame
from fin_skills.libraries.weight_traps import hrp_weights


@register
class WeightTrapsGuard(Guard):
    """HRPOpt(returns=prices) runs, sums to one, and is an inverse-share-price portfolio.

    Inputs
        asset_returns  : the DataFrame you are about to pass as returns (dates x assets).
        linkage_method : 'single' (pypfopt's default) or any scipy linkage. Default 'single'.

    The documented guard: a returns matrix contains negative values, a price matrix
    never does. Fails when the frame has no negative entry at all. Evidence carries
    the HRP weights the script's reference implementation produces on the frame, so
    the damage is visible either way.
    """

    name = "weight_traps"
    skill = "lib-pyportfolioopt"
    summary = "Fails a 'returns' frame with no negative values: it is a price matrix and HRP on it is meaningless."
    wraps = ("fin_skills.libraries.weight_traps.hrp_weights",)
    required = ("asset_returns",)
    optional = ("linkage_method",)

    def check(self, asset_returns: pd.DataFrame, linkage_method: str = "single") -> Outcome:
        out = Outcome()
        df = require_frame(asset_returns, "asset_returns")
        if df.shape[1] < 2 or len(df) < 3:
            raise TypeError("asset_returns needs at least 2 columns and 3 rows")
        if not isinstance(linkage_method, str):
            raise TypeError("linkage_method must be a string")
        num = df.select_dtypes("number")
        if num.shape[1] != df.shape[1]:
            raise TypeError("asset_returns must be all-numeric")
        n_neg = int((num < 0).sum().sum())
        weights = hrp_weights(num.dropna(), linkage_method)
        out.note(n_negative=n_neg, frac_negative=n_neg / float(num.size), min_value=float(num.min().min()),
                 hrp_weights=weights, linkage_method=linkage_method)
        if n_neg == 0:
            out.error("no negative value anywhere: this is a PRICE matrix, not returns. HRP's "
                      "inverse-variance split in dollars-squared buys the cheapest ticker, "
                      "not the least risky one; pass pct_change() instead", where="asset_returns")
        else:
            out.info(f"{n_neg} negative entries ({n_neg / num.size:.0%}): looks like returns",
                     where="asset_returns")
        all_pos = [c for c in num.columns if not (num[c] < 0).any()]
        if all_pos and n_neg:
            out.warning(f"column(s) with no negative value: {all_pos[:8]} - a price series "
                        f"among returns, or a constant", where="asset_returns")
        return out
