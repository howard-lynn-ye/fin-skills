"""Conventions the skill scripts already compute but do not expose under one name.

Every function here delegates to a generated skill module (nothing is re-derived), is
typed, and names in its docstring the skill that owns it and the trap it exists for.

    from fin_skills.api import conventions as c
    c.annualize_sharpe(daily, "crypto")          # sqrt(365), not sqrt(252)
    c.pip_value("USDJPY", 100_000, 150.25)       # 6.66 USD per pip, not 0.0666
    c.stitch_continuous(contracts, rolls)        # ratio-adjusted: returns are real

The four crypto functions (funding_payment, basis_annualized, inverse_contract_pnl and
liquidation_price) are the exception to "delegate": perp_mechanics.py computes the first
three inline inside its demo sections and only exposes `liq_ratio` as a function. They
are one-line formulas restated here with the module's own constants; the test suite
checks each one against the number the script prints.
"""
from __future__ import annotations

from typing import Literal, Mapping, Sequence

import numpy as np
import pandas as pd

from fin_skills.core import benchmark_choice as _bench
from fin_skills.core import option_lifecycle as _opt
from fin_skills.crypto import perp_mechanics as _perp
from fin_skills.futures_fx import continuous_contract as _cc
from fin_skills.futures_fx import fx_conventions as _fx
from fin_skills.libraries import greeks_scaling as _gs

Calendar = Literal["equity", "crypto", "hourly", "funding_8h"]
Side = Literal["long", "short"]

_PERIODS: dict[str, int] = {
    "equity": _perp.EQUITY_DAYS,          # 252 weekday rows a year
    "crypto": _perp.CRYPTO_DAYS,          # 365 calendar rows a year
    "hourly": _perp.CRYPTO_DAYS * 24,     # 8760 hourly rows
    "funding_8h": _perp.CRYPTO_DAYS * 3,  # 1095 eight-hour funding rows
}


# ------------------------------------------------------------------ annualisation
def annualization_factor(calendar: Calendar) -> int:
    """Rows per year for a calendar: 'equity' 252, 'crypto' 365, 'hourly' 8760, 'funding_8h' 1095.

    Owner: crypto-data-and-execution (perp_mechanics.py, section 1). Trap: sqrt(252) on a
    365-row crypto series understates Sharpe and vol by 17%; the factor is the number of
    ROWS per year in the series you hand the function, nothing else.
    """
    try:
        return _PERIODS[calendar]
    except KeyError:
        raise ValueError(f"calendar must be one of {sorted(_PERIODS)}, got {calendar!r}") from None


def annualize_sharpe(returns: pd.Series | np.ndarray | Sequence[float],
                     periods_per_year: int | Calendar = "equity") -> float:
    """mean / sd(ddof=1) * sqrt(periods_per_year), rf = 0. Accepts a calendar name too.

    Owner: crypto-data-and-execution (perp_mechanics.sharpe). Trap: every library defaults
    to 252, which is wrong for anything that trades at weekends or by the hour.
    """
    periods = periods_per_year if isinstance(periods_per_year, int) \
        else annualization_factor(periods_per_year)
    return _perp.sharpe(pd.Series(np.asarray(returns, dtype=float)), periods)


# --------------------------------------------------------------------------- FX
def pip_size(pair: str) -> float:
    """0.01 for JPY-quoted pairs, 0.0001 for everything else.

    Owner: fx-markets (fx_conventions.pip_size). Trap: sizing USDJPY with the 0.0001
    default makes the position 100x too large.
    """
    return _fx.pip_size(pair)


def pip_value(pair: str, notional: float, price: float) -> _fx.Pip:
    """Value of one pip on `notional` units of base at `price`; .value_usd is what risk needs.

    Owner: fx-markets (fx_conventions.pip_value). Trap: the pip of an inverted pair is in
    the QUOTE currency and must be divided by the price to get back to dollars.
    """
    return _fx.pip_value(pair, notional, price)


def carry_return(spot: float, r_base: float, r_quote: float, days: float,
                 pair: str | None = None) -> _fx.Carry:
    """Interest differential earned by a long-base position over `days`, plus the forward.

    Owner: fx-markets (fx_conventions.carry_return). Trap: a carry pair backtested on spot
    alone reads as a loser; `.ret` is the return to ADD, `.points` has the OPPOSITE sign.
    """
    return _fx.carry_return(spot, r_base, r_quote, days, pair)


# ------------------------------------------------------------------------ options
def crr(S: float, K: float, r: float, q: float, sigma: float, T: float,
        steps: int = 800, call: bool = True, american: bool = True) -> float:
    """Cox-Ross-Rubinstein binomial price; american=True prices early exercise.

    Owner: options-backtesting (option_lifecycle.crr). Trap: marking a short American
    option to intrinsic at expiry ignores assignment before an ex-dividend date.
    """
    return _opt.crr(S, K, r, q, sigma, T, steps=steps, call=call, american=american)


def greeks_screen_units(S: float, K: float, T: float, r: float, q: float, sigma: float,
                        flag: Literal["c", "p"] = "c") -> dict[str, float]:
    """Black-Scholes-Merton greeks in SCREEN units: vega and rho per 1 point, theta per day.

    Owner: lib-vollib (greeks_scaling.raw_greeks + scaled_greeks). Trap: vollib returns
    these pre-scaled numbers while QuantLib returns the raw derivatives, 100x and 365x
    apart, and neither says so.
    """
    return _gs.scaled_greeks(_gs.raw_greeks(S, K, T, r, q, sigma, flag))


# ------------------------------------------------------------------------- crypto
def funding_payment(notional: float, rate: float, periods: int = 1,
                    side: Side = "long") -> float:
    """Funding PAID by the position over `periods` settlements (negative = received).

    Owner: crypto-data-and-execution (perp_mechanics.section_funding, inline). Trap: the
    charge is on NOTIONAL, so at 10x leverage it is 10x the rate against your margin;
    positive rate = longs pay shorts.
    """
    sign = 1.0 if side == "long" else -1.0
    return sign * float(notional) * float(rate) * int(periods)


def liquidation_price(entry: float, lev: float, mmr: float, side: Side,
                      exact: bool = True) -> float:
    """Price at which an isolated linear position is force-closed, fees ignored.

    Owner: crypto-data-and-execution (perp_mechanics.liq_ratio). Trap: liquidation is a
    market order at the maintenance line plus a fee, not a stop; funding moves it closer.
    """
    if side not in ("long", "short"):
        raise ValueError("side must be 'long' or 'short'")
    return float(entry) * _perp.liq_ratio(float(lev), float(mmr), side, exact)


def basis_annualized(mark: float, index: float, days: float, compound: bool = False) -> float:
    """Annualised basis of a dated future: (mark/index - 1) * 365/days, or compounded.

    Owner: crypto-data-and-execution (perp_mechanics.section_basis, inline). Trap: the
    locked basis is only earned if the short leg survives the rally to expiry.
    """
    if days <= 0:
        raise ValueError("days to expiry must be positive")
    b = float(mark) / float(index) - 1.0
    year = _perp.CRYPTO_DAYS
    return (1.0 + b) ** (year / float(days)) - 1.0 if compound else b * year / float(days)


def inverse_contract_pnl(notional: float, entry: float, exit_price: float,
                         side: Side = "long") -> float:
    """P&L in COIN of an inverse (coin-margined) contract: notional * (1/entry - 1/exit).

    Owner: crypto-data-and-execution (perp_mechanics.section_inverse, inline). Trap: gains
    in coin are capped at notional/entry while losses are unbounded as the price falls.
    """
    if entry <= 0 or exit_price <= 0:
        raise ValueError("prices must be positive")
    sign = 1.0 if side == "long" else -1.0
    return sign * float(notional) * (1.0 / float(entry) - 1.0 / float(exit_price))


# ------------------------------------------------------------------------ futures
def stitch_continuous(contracts: pd.DataFrame, roll_dates: Sequence,
                      method: Literal["unadjusted", "difference", "ratio"] = "ratio") -> pd.Series:
    """Weld a near->far contract chain into one series; the method decides what it means.

    Owner: futures-continuous-contracts (continuous_contract.stitch). Trap: 'difference'
    goes negative and its pct_change flips sign; use 'ratio' for returns and 'unadjusted'
    for levels. The result's .attrs record which it is.
    """
    return _cc.stitch(contracts, roll_dates, method)


# -------------------------------------------------------------------- execution
def implementation_shortfall(avg_fill: float, decision_px: float, side: int = 1) -> float:
    """Cost in bps of `avg_fill` against the DECISION price; positive = it cost money.

    Owner: execution-cost-analysis (benchmark_choice.cost_bps). Trap: a perfect VWAP
    score is compatible with any amount of shortfall against the arrival price.
    """
    return _bench.cost_bps(float(avg_fill), float(decision_px), side)


def benchmark_costs(avg_fill: float, px: np.ndarray | Sequence[float],
                    vol: np.ndarray | Sequence[float], decision_px: float,
                    side: int = 1) -> dict[str, float]:
    """The same fill scored against arrival, interval VWAP, TWAP, close and open, in bps.

    Owner: execution-cost-analysis (benchmark_choice.benchmarks + cost_bps). Trap: the
    benchmark must be chosen before the trade; each one tells a different story after.
    """
    px_a = np.asarray(px, dtype=float)
    vol_a = np.asarray(vol, dtype=float)
    bench: Mapping[str, float] = _bench.benchmarks(px_a, vol_a, float(decision_px))
    return {name: _bench.cost_bps(float(avg_fill), float(value), side)
            for name, value in bench.items()}


__all__ = [
    "Calendar", "Side", "annualization_factor", "annualize_sharpe", "basis_annualized",
    "benchmark_costs", "carry_return", "crr", "funding_payment", "greeks_screen_units",
    "implementation_shortfall", "inverse_contract_pnl", "liquidation_price", "pip_size",
    "pip_value", "stitch_continuous",
]
