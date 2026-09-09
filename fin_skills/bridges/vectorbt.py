"""Bundle <-> vbt.Portfolio, with the same-bar fill made impossible to reach by accident.

THE TRAP (fin_skills.load('lib-vectorbt'), "The trap that costs you money"): vectorbt's
`Order.price` defaults to `np.inf`, and its own `vectorbt/portfolio/enums.py` docstring
resolves `np.inf` to *"the current close"*. `from_orders`/`from_signals` do
`if price is None: price = np.inf`, so

    vbt.Portfolio.from_signals(close, entries, exits)

fills at `close[t]` - the signal's own bar. Combined with the equally default
`fast_ma.ma_crossed_above(slow_ma)`, itself computed on `close[t]`, that is textbook
same-bar execution. The close is not knowable until the bar is over. The library will
never warn you.

🚨 LICENCE: vectorbt is Apache-2.0 + Commons Clause - NOT OSI open source, and PyPI
declares no licence at all. It is an optional RUNTIME import here and appears in no
dependency list; `fin_skills.bridges.licence_audit()` carries the row and `_lazy.need()`
prints the warning. You may not sell a product or service whose value derives
substantially from it; importing it from MIT code is fine.

What this bridge does:
  * shifts entries/exits forward one bar unless `already_lagged=True`;
  * passes an explicit `price=`, never the default;
  * REFUSES `price="close"` with `already_lagged=False` - that combination IS the bug;
  * TESTS an `already_lagged=True` claim with `assert_causal` (via `_lazy.prove_lagged`)
    instead of believing it;
  * records `n_trials = entries.shape[1]` on the result, because a parameter grid IS a
    trial count for Deflated Sharpe (fin_skills.load('backtest-validation')).
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from fin_skills.api import Bundle
from fin_skills.bridges import _lazy

LIBRARY = "vectorbt"
LICENCE = _lazy.info(LIBRARY).licence
VERIFIED_ON = _lazy.info(LIBRARY).verified_on


class SameBarFillError(ValueError):
    """The requested combination reproduces vectorbt's same-bar fill default."""


def _as_frame(x: Any, name: str) -> pd.DataFrame:
    if isinstance(x, pd.Series):
        return x.to_frame()
    if isinstance(x, pd.DataFrame):
        return x
    raise TypeError(f"{name} must be a Series or DataFrame, got {type(x).__name__}")


def _close_of(bundle: Bundle) -> pd.DataFrame:
    """The close panel a Bundle carries, under any of the three slots that hold one."""
    for slot in ("prices", "close"):
        if slot in bundle:
            return _as_frame(bundle.get(slot), slot)
    if "bars" in bundle:
        bars = bundle.bars
        for col in ("close", "Close", "CLOSE"):
            if col in bars.columns:
                return bars[[col]]
    raise TypeError("bundle carries no close: fill the `prices` (wide panel), `close` "
                    "(one instrument) or `bars` (OHLCV) slot")


def _open_of(bundle: Bundle) -> pd.DataFrame:
    if "bars" in bundle:
        for col in ("open", "Open", "OPEN"):
            if col in bundle.bars.columns:
                return bundle.bars[[col]]
    raise TypeError(
        "price='open' needs the open panel and the bundle has none. Fill the `bars` slot "
        "with an OHLCV frame for a single name, or pass the open panel directly as "
        "price=<DataFrame>. Do NOT fall back to the close: that is the trap.")


def to_vectorbt(bundle: Bundle, *, entries: pd.DataFrame | pd.Series,
                exits: pd.DataFrame | pd.Series, already_lagged: bool = False,
                price: str | pd.DataFrame | pd.Series = "open",
                signal_fn: Callable[[pd.DataFrame], Any] | None = None,
                k: int | None = None, **portfolio_kw: Any):
    """Build a `vbt.Portfolio` from a Bundle with the same-bar fill designed out.

    entries / exits  boolean signal frames aligned to the close panel.
    already_lagged   an ASSERTION that entries/exits at bar t were knowable before bar t.
                     It is TESTED, not believed: supply `signal_fn` (or fill the bundle's
                     `signal_fn` + `bars` slots) and `_lazy.prove_lagged` runs the repo's
                     `assert_causal` guard on it. No function, no proof, no claim.
    price            'open' (the default and the honest one), a DataFrame/Series of fill
                     prices, or 'close' - which is REFUSED unless already_lagged=True.
    **portfolio_kw   forwarded to `vbt.Portfolio.from_signals`. `price` is not accepted
                     there: this bridge owns it.

    Every refusal happens BEFORE vectorbt is imported, so the enforcement is testable on a
    machine where vectorbt is not installed.
    """
    ent = _as_frame(entries, "entries").astype(bool)
    exi = _as_frame(exits, "exits").astype(bool)
    if "price" in portfolio_kw:
        raise TypeError("pass the fill price as price=, not inside **portfolio_kw: this "
                        "bridge owns it so that np.inf can never reach Order.price")
    if isinstance(price, str) and price not in ("open", "close"):
        raise TypeError(f"price must be 'open', 'close' or a frame of fill prices, "
                        f"got {price!r}")

    # ---- the refusal, first, before anything is imported or built --------------------
    if price == "close" and not already_lagged:
        raise SameBarFillError(
            "price='close' with already_lagged=False is exactly vectorbt's default bug: "
            "a signal computed from close[t] filled at close[t]. The close is not knowable "
            "until the bar is over. Either pass price='open' (or the open panel) so the "
            "fill happens after the signal bar, or prove the signal is already lagged - "
            "see fin_skills.load('lib-vectorbt'), 'The trap that costs you money'.")

    # ---- the claim is tested, not believed -------------------------------------------
    if already_lagged:
        fn = signal_fn if signal_fn is not None else bundle.get("signal_fn")
        bars = bundle.get("bars")
        if fn is None or bars is None:
            _lazy.refuse_untestable_claim(
                "entries/exits",
                "Pass signal_fn=<the callable that produced them> and fill the bundle's "
                "`bars` slot, so assert_causal can perturb bar t and watch the value at t.")
        _lazy.prove_lagged(fn, bars, k=k, name="entries/exits")
        ent_f, exi_f = ent, exi
    else:
        ent_f, exi_f = ent.shift(1).fillna(False).astype(bool), \
            exi.shift(1).fillna(False).astype(bool)

    close = _close_of(bundle)
    if isinstance(price, str):
        px = close if price == "close" else _open_of(bundle)
    else:
        px = _as_frame(price, "price")
    if len(px) != len(close):
        raise TypeError(f"price has {len(px)} rows and the close panel has {len(close)}")

    # ---- only now does the library get imported --------------------------------------
    vbt = _lazy.need(LIBRARY, why="to build a Portfolio")
    n = max(ent_f.shape[1], exi_f.shape[1], close.shape[1])
    px_b = px if px.shape[1] == n else pd.DataFrame(
        np.repeat(px.to_numpy()[:, :1], n, axis=1), index=px.index, columns=close.columns)
    close_b = close if close.shape[1] == n else pd.DataFrame(
        np.repeat(close.to_numpy()[:, :1], n, axis=1), index=close.index,
        columns=ent_f.columns)
    pf = vbt.Portfolio.from_signals(close_b, ent_f, exi_f, price=px_b, **portfolio_kw)
    prov = _lazy.provenance(LIBRARY, vbt, request={
        "price": price if isinstance(price, str) else "frame",
        "already_lagged": bool(already_lagged), "shifted": not already_lagged,
        "n_trials": int(ent.shape[1])})
    try:
        pf.n_trials = int(ent.shape[1])          # a parameter grid IS a trial count
        pf.fin_skills_provenance = prov
    except (AttributeError, TypeError):          # vbt objects are often slotted/frozen
        pass
    return pf


def _same_bar_fills(pf) -> tuple[int, int]:
    """(fills priced at the signal bar's own close, total fills), or (-1, -1) if unknown.

    The detectable signature of `price=np.inf`: an order record whose price equals the
    close of the bar it is stamped on.
    """
    try:
        rec = pf.order_records
        recs = rec.records if hasattr(rec, "records") else rec
        df = pd.DataFrame(recs)
        if not {"idx", "col", "price"} <= set(df.columns):
            return (-1, -1)
        close = pf.close
        closes = np.asarray(close)
        if closes.ndim == 1:
            closes = closes[:, None]
        hit = 0
        for r in df.itertuples(index=False):
            c = float(closes[int(r.idx), min(int(r.col), closes.shape[1] - 1)])
            hit += int(np.isclose(float(r.price), c, rtol=0, atol=1e-12))
        return (hit, len(df))
    except Exception:                            # noqa: BLE001 - a probe, never a failure
        return (-1, -1)


def from_vectorbt(pf, *, sessions: Any = None, benchmark: pd.Series | None = None,
                  **extra: Any) -> Bundle:
    """`vbt.Portfolio` -> Bundle, with the same-bar signature reported as a finding.

    `returns` is filled from `pf.returns()`, which is GROSS only when the Portfolio was
    built with no fees or slippage - the Bundle slot documents itself as GROSS and
    `cost_curve` applies bps itself, so fees configured inside vectorbt would be counted
    twice. The bridge reads `pf.wrapper` for the fee settings it can see and records what
    it found in `Bundle.turnover`'s companion note rather than silently trusting either.
    """
    rets = pf.returns()
    if isinstance(rets, pd.DataFrame) and rets.shape[1] == 1:
        rets = rets.iloc[:, 0]
    slots: dict[str, Any] = {"returns": rets}

    close = getattr(pf, "close", None)
    if close is not None:
        slots["prices"] = close if isinstance(close, pd.DataFrame) else pd.DataFrame(close)

    try:
        flow = pf.asset_flow()
        px = close if close is not None else 1.0
        traded = (flow.abs() * px)
        val = pf.value()
        val = val.iloc[:, 0] if isinstance(val, pd.DataFrame) and val.shape[1] == 1 else val
        turn = (traded.sum(axis=1) if isinstance(traded, pd.DataFrame) else traded) / val
        slots["turnover"] = turn.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    except Exception:                            # noqa: BLE001 - turnover is best effort
        pass

    ppy = _lazy.periods_per_year(sessions)
    if ppy is not None:
        slots["periods_per_year"] = ppy
    if benchmark is not None:
        slots["benchmark_returns"] = benchmark
    slots.update(extra)
    b = Bundle(**slots)

    hit, total = _same_bar_fills(pf)
    if total > 0 and hit == total:
        raise SameBarFillError(
            f"every one of {total} fills was priced at the close of its own bar: this "
            f"Portfolio was built with vectorbt's default price=np.inf, which its enums.py "
            f"resolves to 'the current close'. The returns are same-bar returns and no "
            f"guard downstream can undo that. Rebuild it through to_vectorbt().")
    return b


__all__ = ["LIBRARY", "LICENCE", "SameBarFillError", "VERIFIED_ON", "from_vectorbt",
           "to_vectorbt"]
