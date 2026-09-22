"""Qlib's native TopkDropoutStrategy and SimulatorExecutor, with explicit clocks.

Call qlib.init(provider_uri=...) yourself. This adapter never downloads data or
changes the global provider. Prices, calendars and adjustments belong to that provider.
"""
import numpy as np
import pandas as pd

from fin_skills.algorithms.runtime import integer, number


def qlib_backtest(scores, available_at, *, start, end, account=100000., topk=10,
                  n_drop=1, open_cost, close_cost, min_cost, trade_unit=1,
                  limit_threshold=None, benchmark=None):
    """Execute prior-session scores at the next session's open using real Qlib.

Scores and availability are Series with the same (datetime, instrument) index.
The datetime level is a session date, not a fill timestamp. Every score must be
available before the end of its dated session. Delayed data must be redated by
the caller; it is rejected here instead of being silently backdated.
"""
    if not isinstance(scores, pd.Series) or not isinstance(scores.index, pd.MultiIndex):
        raise TypeError("scores must be a Series indexed by datetime, instrument")
    if scores.index.names != ["datetime", "instrument"] or scores.index.has_duplicates:
        raise ValueError("use unique (datetime, instrument) index names")
    if scores.empty or not np.isfinite(scores.to_numpy(dtype=float)).all():
        raise ValueError("finite nonempty scores required")
    if not isinstance(available_at, pd.Series) or not available_at.index.equals(scores.index):
        raise ValueError("available_at must align exactly with scores")
    dates = pd.DatetimeIndex(scores.index.get_level_values("datetime"))
    clocks = pd.DatetimeIndex(pd.to_datetime(available_at))
    if (dates.tz is not None or clocks.tz is not None or dates.hasnans or clocks.hasnans
            or not dates.equals(dates.normalize())):
        raise ValueError("use timezone-naive session dates and nonmissing availability clocks")
    if (clocks >= dates + pd.Timedelta(days=1)).any():
        raise ValueError("score unavailable before session end; do not backdate delayed signals")
    first, last = pd.Timestamp(start), pd.Timestamp(end)
    if (pd.isna(first) or pd.isna(last) or first.tz is not None or last.tz is not None
            or first != first.normalize() or last != last.normalize() or first > last):
        raise ValueError("start/end must be ordered session dates")
    topk = integer(topk, "topk", maximum=10000)
    n_drop = integer(n_drop, "n_drop", minimum=0, maximum=topk)
    account = number(account, "account", minimum=1e-8)
    trade_unit = integer(trade_unit, "trade_unit", maximum=100000)
    costs = {k: number(v, k, minimum=0, maximum=.99)
             for k, v in (("open_cost", open_cost), ("close_cost", close_cost))}
    costs["min_cost"] = number(min_cost, "min_cost", minimum=0)
    if limit_threshold is not None:
        limit_threshold = number(limit_threshold, "limit_threshold", minimum=1e-8, maximum=.99)
    from qlib.backtest import backtest
    from qlib.backtest.executor import SimulatorExecutor
    from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy
    benchmark_mode = "supplied"
    if benchmark is None:
        # Qlib 0.9.7 converts None to {}, then silently defaults to CSI300.
        # An explicit zero-return series preserves the requested absence of a market index.
        from qlib.data import D
        benchmark = pd.Series(0., index=pd.DatetimeIndex(D.calendar(
            start_time=first, end_time=last, freq="day")))
        benchmark_mode = "zero_return_reference"
    strategy = TopkDropoutStrategy(signal=scores.sort_index(), topk=topk, n_drop=n_drop,
                                  risk_degree=1.0, method_sell="bottom", method_buy="top")
    executor = SimulatorExecutor(time_per_step="day", generate_portfolio_metrics=True)
    portfolios, indicators = backtest(
        start_time=first, end_time=last, strategy=strategy, executor=executor,
        benchmark=benchmark, account=account,
        exchange_kwargs=dict(freq="day", deal_price="$open", trade_unit=trade_unit,
                             limit_threshold=limit_threshold, **costs))
    report, positions = portfolios["1day"]
    net_returns = report["return"] - report["cost"]
    nav = (1 + net_returns).cumprod()
    if not np.isfinite(nav).all() or not np.allclose(
            nav.to_numpy(), report["account"].to_numpy() / account, rtol=1e-7, atol=1e-9):
        raise ArithmeticError("Qlib report account and cost-inclusive return do not reconcile")
    return dict(report=report, positions=positions, indicators=indicators,
                net_returns=net_returns, nav=nav,
                provenance=dict(backend="pyqlib", strategy="TopkDropoutStrategy",
                    executor="SimulatorExecutor", timing="previous-session signal; next open",
                    costs=costs, trade_unit=trade_unit, limit_threshold=limit_threshold,
                    benchmark_mode=benchmark_mode))
