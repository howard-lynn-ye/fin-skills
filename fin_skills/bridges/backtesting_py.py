"""Bundle <-> backtesting.Backtest, with the Strategy.I leak tested before the first bar.

🚨 LICENCE FIREWALL. `backtesting.py` is **AGPL-3.0**, which is network copyleft: anyone
serving a derived work over a dashboard, an API or an MCP endpoint owes its users the
complete corresponding source of the combined work. fin-skills is MIT. This module is
therefore an optional RUNTIME import and nothing else - `backtesting` appears in no
`dependencies` list and no extra in `pyproject.toml`, no code is derived from it, and
`_lazy.need()` prints a warning every time it is imported.
`fin_skills.bridges.licence_audit()` carries the row; a test asserts the absence.

THE TRAP (fin_skills.load('lib-backtesting-py'), "The trap that costs you money"):
`Strategy.I` computes the indicator over the FULL series in `init()` and then slices it in
`next()`. For an SMA that is an optimisation. For a centred window, `filtfilt`, a
`.shift(-1)`, a forward-looking `argmax`, or a model fitted on the whole series and then
evaluated pointwise, it is a silent, total leak - and **the framework cannot detect any of
them.** No warning, no assertion; the equity curve simply comes out beautiful.

THE SECOND TRAP: `trade_on_close=True` does not fill at this bar's close. Verified in
`backtesting/backtesting.py`, the fill price is `data.Close[-2]` - the PREVIOUS bar's close
in the broker's frame, which is the signal bar's close in the strategy's. It is legitimate
only if you can name the live order type that produces it (an MOC order, an auction), so
this bridge refuses it unless you say so in the call.

What this bridge does:
  * validates every indicator it can see with the repo's own `assert_causal` guard - the
    bundle's `indicator` slot eagerly (before backtesting.py is even imported), and every
    callable handed to `Strategy.I` at the moment `init()` calls it, which is before the
    first bar is stepped;
  * refuses `trade_on_close=True` unless `acknowledge_close_minus_2=True`;
  * converts the resulting stats back into a Bundle.
"""
from __future__ import annotations

from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from fin_skills.api import Bundle
from fin_skills.bridges import _lazy
from fin_skills.bridges._lazy import NonCausalError

LIBRARY = "backtesting"
LICENCE = _lazy.info(LIBRARY).licence
VERIFIED_ON = _lazy.info(LIBRARY).verified_on


class TradeOnCloseError(ValueError):
    """`trade_on_close=True` was requested without acknowledging the Close[-2] fill."""


def _bars_of(bundle: Bundle, ticker: str | None) -> pd.DataFrame:
    if "bars" in bundle:
        return bundle.bars
    raise TypeError(
        "bundle has no `bars` slot, so no indicator can be tested for causality"
        + (f" (ticker={ticker!r})" if ticker else "")
        + ". backtesting.py is single-asset: fill `bars` with that one name's OHLCV.")


def _named(indicator: Any) -> dict[str, Callable[..., Any]]:
    if isinstance(indicator, Mapping):
        return {str(k): v for k, v in indicator.items()}
    if callable(indicator):
        return {getattr(indicator, "__name__", "indicator"): indicator}
    raise TypeError("the `indicator` slot must be a callable or a {name: callable} mapping")


def _array_fn(func: Callable[..., Any], col: str, args: tuple, kwargs: dict):
    """Wrap an ndarray -> ndarray indicator as the df -> Series assert_causal wants."""
    def fn(d: pd.DataFrame) -> pd.Series:
        out = func(d[col].to_numpy(dtype=float), *args, **kwargs)
        arr = np.asarray(out, dtype=float)
        if arr.ndim > 1:
            arr = arr[0] if arr.shape[0] < arr.shape[-1] else arr[:, 0]
        return pd.Series(arr[:len(d)], index=d.index[:len(arr)])
    fn.__name__ = getattr(func, "__name__", "indicator")
    return fn


def _price_column(bars: pd.DataFrame) -> str:
    for c in ("Close", "close", "CLOSE"):
        if c in bars.columns:
            return c
    raise TypeError(f"bars has no close column; has {list(bars.columns)}")


def _check_indicators(bundle: Bundle, bars: pd.DataFrame, k: int | None) -> list[str]:
    """Test every indicator declared in the bundle. Runs with backtesting.py absent."""
    if "indicator" not in bundle:
        return []
    col = _price_column(bars)
    checked = []
    for name, func in _named(bundle.indicator).items():
        _lazy.prove_causal(_array_fn(func, col, (), {}), bars, k=k, name=name)
        checked.append(name)
    return checked


def _match_column(arr: np.ndarray, bars: pd.DataFrame) -> str | None:
    a = np.asarray(arr, dtype=float)
    for c in bars.columns:
        try:
            b = bars[c].to_numpy(dtype=float)
        except (TypeError, ValueError):
            continue
        if b.shape == a.shape and np.allclose(np.nan_to_num(a), np.nan_to_num(b),
                                              rtol=0, atol=1e-12):
            return str(c)
    return None


def to_backtesting(bundle: Bundle, strategy_cls, *, ticker: str | None = None,
                   acknowledge_close_minus_2: bool = False, k: int | None = None,
                   **bt_kw: Any):
    """Build a `Backtest` whose indicators have been proven causal first.

    strategy_cls                a `backtesting.Strategy` subclass. It is subclassed here so
                                that `Strategy.I` validates each callable as `init()` calls
                                it - before a single bar is stepped.
    acknowledge_close_minus_2   required to pass `trade_on_close=True`. Name the live order
                                type that produces a signal-bar close fill before you set it.
    **bt_kw                     forwarded to `Backtest(...)`: cash, commission, spread,
                                margin, trade_on_close, exclusive_orders, ...

    The refusals fire before `backtesting` is imported, so they are testable on a machine
    where the AGPL library is not installed - which is the point of the firewall.
    """
    bars = _bars_of(bundle, ticker)
    if bt_kw.get("trade_on_close"):
        if not acknowledge_close_minus_2:
            raise TradeOnCloseError(
                "trade_on_close=True does not fill at this bar's close. Verified in "
                "backtesting/backtesting.py: `prev_close = data.Close[-2]` and the fill is "
                "stamped at self._i - 1, so it fills at the close of the bar the strategy "
                "was looking at when it placed the order - a market-on-close fill at the "
                "SIGNAL bar's close. It is legitimate only if you can name the live order "
                "type that produces it (MOC, an auction). Pass "
                "acknowledge_close_minus_2=True to say you can.")
    checked = _check_indicators(bundle, bars, k)

    bt_mod = _lazy.need(LIBRARY, why="to build a Backtest")
    Backtest, Strategy = bt_mod.Backtest, bt_mod.Strategy
    if not (isinstance(strategy_cls, type) and issubclass(strategy_cls, Strategy)):
        raise TypeError("strategy_cls must be a backtesting.Strategy subclass")

    probe_bars, probe_k = bars, k

    class _GuardedStrategy(strategy_cls):                      # type: ignore[misc,valid-type]
        """`strategy_cls` with `Strategy.I` gated by assert_causal."""

        def I(self, func, *args, **kwargs):                    # noqa: E743 - the library's name
            name = kwargs.get("name") or getattr(func, "__name__", "indicator")
            mapped, col = [], None
            for a in args:
                arr = None
                if not isinstance(a, (str, bool, int, float)) and hasattr(a, "__len__"):
                    arr = np.asarray(a)
                if arr is not None and arr.ndim == 1 and len(arr) == len(probe_bars):
                    hit = _match_column(arr, probe_bars)
                    if hit is None:
                        raise NonCausalError(
                            f"{name}: an array argument to Strategy.I matches no column of "
                            f"the bundle's `bars`, so its causality cannot be tested. "
                            f"Compute it from the bar columns, or declare it in the "
                            f"bundle's `indicator` slot. An untested indicator is refused: "
                            f"Strategy.I computes over the FULL series and the framework "
                            f"cannot detect the leak for you.")
                    col = col or hit
                    mapped.append(hit)
                else:
                    mapped.append(a)
            if col is not None:
                rest = tuple(a for a in mapped if not isinstance(a, str) or a not in
                             probe_bars.columns)
                _lazy.prove_causal(_array_fn(func, col, rest, {}), probe_bars, k=probe_k,
                                   name=name)
            return super().I(func, *args, **kwargs)

    _GuardedStrategy.__name__ = f"Guarded{strategy_cls.__name__}"
    bt = Backtest(bars, _GuardedStrategy, **bt_kw)
    bt.fin_skills_provenance = _lazy.provenance(
        LIBRARY, bt_mod, request={"ticker": ticker, "trade_on_close": bool(
            bt_kw.get("trade_on_close")), "indicators_checked": checked})
    return bt


def from_backtesting(stats, *, sessions: Any = None, **extra: Any) -> Bundle:
    """`Backtest.run()` stats -> Bundle.

    `_equity_curve['Equity'].pct_change()` -> `returns`; `_trades` -> `turnover`, one-way
    traded notional per bar as a fraction of equity. backtesting.py folds commission into
    the fill price, so these returns are NET of whatever you configured - `cost_curve` on
    top of them double-counts. Run it with `commission=0` when you want the cost curve to
    be the thing that prices execution.
    """
    curve = getattr(stats, "_equity_curve", None)
    if curve is None:
        raise TypeError("expected the stats Series returned by Backtest.run(), which "
                        "carries _equity_curve and _trades")
    eq = curve["Equity"].astype(float)
    slots: dict[str, Any] = {"returns": eq.pct_change().dropna()}

    trades = getattr(stats, "_trades", None)
    if trades is not None and len(trades):
        notional = pd.Series(0.0, index=eq.index)
        for t in trades.itertuples(index=False):
            for bar, px in ((int(t.EntryBar), float(t.EntryPrice)),
                            (int(t.ExitBar), float(t.ExitPrice))):
                if 0 <= bar < len(notional):
                    notional.iloc[bar] += abs(float(t.Size)) * px
        slots["turnover"] = (notional / eq).fillna(0.0)

    ppy = _lazy.periods_per_year(sessions)
    if ppy is not None:
        slots["periods_per_year"] = ppy
    slots.update(extra)
    return Bundle(**slots)


__all__ = ["LIBRARY", "LICENCE", "TradeOnCloseError", "VERIFIED_ON", "from_backtesting",
           "to_backtesting"]
