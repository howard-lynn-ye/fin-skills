"""Guard: is the assumed cost consistent with the order sizes (execution-cost-analysis)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from fin_skills.api.base import Guard, Outcome, register
from fin_skills.api.guards._common import require_number
from fin_skills.core.cost_plausibility import (DEFAULT_DAILY_VOL, INVERSE_TURNOVER,
                                               MAX_PARTICIPATION, MODEL_MAX_PARTICIPATION,
                                               MODEL_MIN_PARTICIPATION, BETA, check_cost,
                                               render)


def _mean_turnover(turnover: object) -> float:
    """One number for the turnover: a scalar as given, a series as its mean."""
    if isinstance(turnover, (pd.Series, np.ndarray, list, tuple)):
        arr = np.asarray(pd.Series(turnover).astype(float))
        if arr.size == 0:
            raise ValueError("turnover is empty")
        if not np.isfinite(arr).all():
            raise ValueError("turnover contains NaN/inf; decide what a missing period means")
        return float(arr.mean())
    return require_number(turnover, "turnover")


def _adv_and_names(adv: object, n_names: float | None) -> tuple[float, float, float]:
    """(ADV of the median traded name, ADV of the thinnest, names) from scalar/Series/frame.

    A DataFrame is read as a dates x tickers dollar-volume panel and reduced to one ADV per
    column; a Series is one ADV per name; a scalar is 'every traded name looks like this'.
    """
    if isinstance(adv, pd.DataFrame):
        per_name = adv.median(axis=0, skipna=True).dropna().astype(float)
    elif isinstance(adv, (pd.Series, np.ndarray, list, tuple)):
        per_name = pd.Series(adv).astype(float).dropna()
    else:
        v = require_number(adv, "adv")
        if n_names is None:
            raise TypeError("adv is a single number, so n_names must be given: the turnover "
                            "has to be spread over some number of names")
        return v, v, float(n_names)
    if per_name.empty:
        raise ValueError("adv has no usable values")
    if (per_name <= 0).any():
        raise ValueError("adv must be positive dollar volume per name")
    n = float(len(per_name)) if n_names is None else float(n_names)
    return float(per_name.median()), float(per_name.min()), n


@register
class CostPlausibilityGuard(Guard):
    """A cost assumption is a claim about order size. Check it against traded volume.

    `cost_curve` asks whether the edge SURVIVES the cost you stated. It cannot ask whether
    that cost was available: a strategy whose breakeven is 46 bps survives a 2 bps
    assumption and the 2 bps is still impossible if every order is 8% of the day's volume.
    This guard closes that gap by turning turnover and book size into a participation rate
    and pricing it with Almgren et al. (2005).

    Inputs
        turnover         : one-way traded notional per period as a fraction of the book
                          (Series or scalar). The mean is used.
        book             : dollars deployed.
        adv              : dollar volume per name: a scalar, a per-name Series, or a
                          dates x tickers panel (reduced to each column's median).
        cost_bps         : the cost you claim to pay per dollar traded one way - the number
                          cost_curve multiplies by turnover.
        n_names          : names the flow is spread over. Defaults to the width of `adv`;
                          required when `adv` is a single number.
        daily_vol        : daily volatility of a traded name. Default 2% - a stand-in, and
                          impact is LINEAR in it, so pass your own.
        inverse_turnover : Theta/V, days to turn the float. Default 250 (large cap).
        exec_horizon     : days over which one order is worked. Default 1.0 (a full
                          session), which is the cheapest case and so a lower bound.
        beta             : temporary-impact exponent. Default 3/5 as fitted; 1/2 is the
                          folklore square root, which that paper rejects at 95%.
        max_participation: ADV fraction above which no cost is credible. Default 10%.

    Fails when the stated cost is below the modelled impact - which excludes spread, fees
    and borrow, so it is a floor, not an estimate - or when participation exceeds
    `max_participation`. Warns when participation leaves the band the model was fitted on.
    """

    name = "cost_plausibility"
    skill = "execution-cost-analysis"
    summary = ("Turns turnover, book size and ADV into a participation rate and fails when the "
               "stated cost is below the market impact that participation implies.")
    wraps = ("fin_skills.core.cost_plausibility.check_cost",
             "fin_skills.core.cost_plausibility.impact_bps",
             "fin_skills.core.cost_plausibility.plausible_book")
    required = ("turnover", "book", "adv", "cost_bps")
    optional = ("n_names", "daily_vol", "inverse_turnover", "exec_horizon", "beta",
                "max_participation")

    def check(self, turnover: pd.Series | float, book: float, adv: object, cost_bps: float,
              n_names: float | None = None, daily_vol: float | None = None,
              inverse_turnover: float = INVERSE_TURNOVER, exec_horizon: float = 1.0,
              beta: float = BETA,
              max_participation: float = MAX_PARTICIPATION) -> Outcome:
        out = Outcome()
        book = require_number(book, "book")
        cost_bps = require_number(cost_bps, "cost_bps")
        if book <= 0:
            raise TypeError("book must be positive (dollars deployed)")
        if cost_bps < 0:
            raise TypeError("cost_bps must be non-negative")
        turn = _mean_turnover(turnover)
        adv_med, adv_min, n = _adv_and_names(adv, n_names)
        assumed_vol = daily_vol is None
        vol = DEFAULT_DAILY_VOL if assumed_vol else require_number(daily_vol, "daily_vol")

        v = check_cost(turnover=turn, book=book, adv=adv_med, n_names=n, cost_bps=cost_bps,
                       daily_vol=vol, inverse_turnover=inverse_turnover,
                       exec_horizon=exec_horizon, beta=beta,
                       max_participation=max_participation)
        thin = check_cost(turnover=turn, book=book, adv=adv_min, n_names=n, cost_bps=cost_bps,
                          daily_vol=vol, inverse_turnover=inverse_turnover,
                          exec_horizon=exec_horizon, beta=beta,
                          max_participation=max_participation)
        out.note(verdict=v, implied_bps=v.implied_bps, cost_bps=v.cost_bps,
                 participation=v.participation, participation_thinnest=thin.participation,
                 order_usd=v.order_usd, plausible_book=v.plausible_book,
                 avg_turnover=turn, n_names=n, adv_median=adv_med, adv_min=adv_min,
                 daily_vol=vol, table=render(v))

        for reason in v.reasons:
            out.error(reason + f"; {cost_bps:.2f} bps needs a book of "
                               f"{v.plausible_book:,.0f} USD or less, not {book:,.0f}",
                      where="participation")
        if v.ok:
            out.info(f"stated {cost_bps:.2f} bps covers the {v.implied_bps:.2f} bps of "
                     f"modelled impact at {v.participation:.2%} of ADV per name; the "
                     f"assumption holds to a book of {v.plausible_book:,.0f} USD",
                     where="participation")
        if v.participation > MODEL_MAX_PARTICIPATION:
            out.warning(f"participation {v.participation:.2%} is above the few percent of ADV "
                        f"Almgren et al. (2005) fitted; the impact number is an extrapolation",
                        where="model range")
        elif v.participation < MODEL_MIN_PARTICIPATION:
            out.warning(f"participation {v.participation:.2%} is below the 0.25% of ADV floor "
                        f"of that fit; impact is small here but the number is extrapolated",
                        where="model range")
        if thin.participation > max_participation >= v.participation:
            out.warning(f"the thinnest name traded is at {thin.participation:.2%} of its own "
                        f"ADV even though the median name is at {v.participation:.2%}",
                        where="thinnest name")
        if assumed_vol:
            out.info(f"daily_vol was not supplied; assumed {DEFAULT_DAILY_VOL:.1%} a day and "
                     f"impact is linear in it", where="daily_vol")
        return out
