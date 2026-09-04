"""Report a backtest as a cost-sensitivity CURVE, not a point estimate.

WHY this exists: "Sharpe 1.9" is not a result, it is a result AT ONE ASSUMED COST -- and
the assumed cost is almost always zero or an unstated round number. The whole conclusion
is one unreported parameter away from collapsing. High-turnover strategies (daily
rebalancing, LLM-signal papers, most cross-sectional factors) are profitable at 0 bps,
marginal at 10, and dead by 20-30. The point estimate hides which of those you have.

So report the curve. It costs nothing to compute, it is impossible to fudge, and it turns
"does it work?" into the answerable question: "at what execution cost does it stop
working, and can I actually trade inside that number?"

COST CONVENTION -- state it or you will fool yourself:
    turnover[t] = one-way traded notional at t as a fraction of portfolio value,
                  i.e. sum(|w[t] - w[t-1]|).
    bps         = ROUND-TRIP cost of fully rotating the book, in basis points.
    cost[t]     = turnover[t] * bps / 10_000
If your broker quotes one-way costs, double them before passing them in. This factor of
two is the single most common way a dead strategy passes a cost check.

Usage:
    from cost_curve import cost_curve, breakeven_bps, render
    curve = cost_curve(returns, turnover, bps=(0, 5, 10, 20, 50))
    print(render(curve, breakeven_bps(returns, turnover)))
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _align(returns: pd.Series | np.ndarray,
           turnover: pd.Series | np.ndarray | float) -> tuple[pd.Series, pd.Series]:
    """Coerce to aligned Series. A scalar turnover means 'constant every period'."""
    r = pd.Series(returns).astype(float)
    if np.isscalar(turnover):
        t = pd.Series(float(turnover), index=r.index)
    else:
        t = pd.Series(turnover).astype(float)
        if len(t) != len(r):
            raise ValueError(f"turnover has {len(t)} rows, returns has {len(r)}")
        t.index = r.index
    if (t < 0).any():
        raise ValueError("turnover must be non-negative (it is traded notional, not a delta)")
    if r.isna().any() or t.isna().any():
        raise ValueError("returns/turnover contain NaN; decide what a missing period means "
                         "before averaging over it")
    return r, t


def net_returns(returns: pd.Series | np.ndarray,
                turnover: pd.Series | np.ndarray | float, bps: float) -> pd.Series:
    """Gross returns less turnover-scaled cost at `bps` round-trip."""
    r, t = _align(returns, turnover)
    return r - t * (bps / 1e4)


def max_drawdown(returns: pd.Series) -> float:
    """Worst peak-to-trough on the COMPOUNDED equity curve (negative number)."""
    equity = (1.0 + returns).cumprod()
    return float((equity / equity.cummax() - 1.0).min())


def _stats(net: pd.Series, periods_per_year: int) -> dict[str, float]:
    n = len(net)
    total = float((1.0 + net).prod() - 1.0)
    # Geometric annualisation: arithmetic mean * 252 flatters high-vol strategies.
    ann = float((1.0 + total) ** (periods_per_year / n) - 1.0) if total > -1.0 else -1.0
    sd = float(net.std(ddof=1))
    sharpe = float(net.mean() / sd * np.sqrt(periods_per_year)) if sd > 0 else np.nan
    return {"total_return": total, "ann_return": ann, "sharpe": sharpe,
            "max_drawdown": max_drawdown(net)}


def cost_curve(returns: pd.Series | np.ndarray,
               turnover: pd.Series | np.ndarray | float,
               bps: Sequence[float] = (0, 5, 10, 20, 50),
               periods_per_year: int = TRADING_DAYS) -> pd.DataFrame:
    """Total return, annualised return, Sharpe and max drawdown at each cost level.

    One row per cost level. Read it top to bottom and find where the strategy dies --
    that number, not the 0 bps Sharpe, is what you have to defend.
    """
    r, t = _align(returns, turnover)
    rows = []
    for c in bps:
        s = _stats(r - t * (float(c) / 1e4), periods_per_year)
        rows.append({"bps": float(c), **s})
    out = pd.DataFrame(rows).set_index("bps")
    out.attrs["avg_turnover"] = float(t.mean())
    out.attrs["n_periods"] = len(r)
    return out


def breakeven_bps(returns: pd.Series | np.ndarray,
                  turnover: pd.Series | np.ndarray | float,
                  benchmark: float | pd.Series | np.ndarray | None = None,
                  periods_per_year: int = TRADING_DAYS,
                  hi: float = 10_000.0, tol: float = 1e-4) -> float:
    """Round-trip cost, in bps, at which the strategy stops beating the hurdle.

    `benchmark` may be None (hurdle = zero annualised return), a scalar annualised
    return (e.g. 0.05 for a 5% cash hurdle), or a per-period return series (e.g.
    buy-and-hold), in which case the hurdle is that series' own annualised return.

    Bisection, not a closed form: annualised return is monotonically decreasing in cost
    but not linear in it, because the returns compound. Monotonicity is what makes
    bisection exact here, and it avoids a scipy dependency.
    """
    r, t = _align(returns, turnover)
    if benchmark is None:
        hurdle = 0.0
    elif np.isscalar(benchmark):
        hurdle = float(benchmark)
    else:
        hurdle = _stats(pd.Series(benchmark).astype(float), periods_per_year)["ann_return"]

    def edge(c: float) -> float:
        return _stats(r - t * (c / 1e4), periods_per_year)["ann_return"] - hurdle

    if edge(0.0) <= 0.0:
        return 0.0          # already fails to clear the hurdle before paying anything
    if edge(hi) > 0.0:
        return hi           # survives past the search bound; report the bound, not inf

    lo = 0.0
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if edge(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def render(curve: pd.DataFrame, breakeven: float | None = None,
           title: str = "COST SENSITIVITY") -> str:
    """Fixed-width text table. The one artefact that belongs in every backtest write-up."""
    w = 68
    lines = [title, "=" * w,
             f"{'bps':>6} {'total':>10} {'ann':>9} {'sharpe':>8} {'maxDD':>9}  verdict",
             "-" * w]
    for c, row in curve.iterrows():
        sh = row["sharpe"]
        verdict = ("DEAD" if not np.isfinite(sh) or sh <= 0 else
                   "marginal" if sh < 0.5 else
                   "thin" if sh < 1.0 else "tradeable")
        lines.append(
            f"{c:>6.0f} {row['total_return']:>9.1%} {row['ann_return']:>8.1%} "
            f"{sh:>8.2f} {row['max_drawdown']:>9.1%}  {verdict}"
        )
    lines.append("-" * w)
    if "avg_turnover" in curve.attrs:
        lines.append(f"avg one-way turnover : {curve.attrs['avg_turnover']:.1%} per period "
                     f"over {curve.attrs['n_periods']} periods")
    if breakeven is not None:
        lines.append(f"BREAKEVEN            : {breakeven:.1f} bps round-trip")
        lines.append("                       -> the only number worth quoting. If your "
                     "real")
        lines.append("                          all-in cost is anywhere near it, you have "
                     "nothing.")
    return "\n".join(lines)


if __name__ == "__main__":
    rng = np.random.default_rng(11)
    n = 252 * 4

    # A daily strategy with an entirely plausible gross record: ~17% a year on ~9.5% vol,
    # gross Sharpe near 1.7. This is publishable-looking. It is also fully explained by
    # 35% average daily turnover, which nobody put in the abstract.
    gross = pd.Series(rng.normal(0.00065, 0.006, n),
                      index=pd.bdate_range("2020-01-01", periods=n))
    # Turnover varies day to day (lognormal) the way real rebalancing does.
    turn = pd.Series(np.clip(rng.lognormal(np.log(0.35), 0.30, n), 0.0, 2.0),
                     index=gross.index)

    curve = cost_curve(gross, turn, bps=(0, 2, 5, 10, 15, 20, 30, 50))
    be = breakeven_bps(gross, turn)
    print(render(curve, be, title="SYNTHETIC DAILY CROSS-SECTIONAL STRATEGY"))

    print("\n" + "=" * 68)
    print("The shape to expect: alive at 0 bps, dead by 20. A point-estimate Sharpe of")
    print(f"{curve.loc[0, 'sharpe']:.2f} is true and useless -- at a realistic 10 bps "
          f"round-trip it is {curve.loc[10, 'sharpe']:.2f},")
    print(f"and the whole edge is gone at {be:.0f} bps. Report the curve, not the peak.")
    print("=" * 68)

    # Same test against a benchmark hurdle: the bar is higher than zero in practice,
    # because the money could have sat in something boring instead.
    be_hurdle = breakeven_bps(gross, turn, benchmark=0.05)
    print(f"\nvs a 5% cash hurdle, breakeven falls to {be_hurdle:.1f} bps "
          f"(from {be:.1f} bps vs zero).")
