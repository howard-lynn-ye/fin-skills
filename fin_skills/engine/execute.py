"""fin_skills.engine.execute - one bar loop, in the only order that is causal.

Per session: fill what earlier bars ordered, close out what delisted, mark the book, then
decide. The decision is LAST, and its order is queued for bar t+signal_lag, which is why
`Execution(signal_lag=0)` has to be rejected at construction rather than handled here.
"""
from __future__ import annotations

from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from fin_skills.engine import invariants
from fin_skills.engine.costs import SlippageContext, carry_charge, trade_charge
from fin_skills.engine.panel import Panel
from fin_skills.engine.result import BacktestResult
from fin_skills.engine.sizing import Sizer, SizingContext, equal_weight
from fin_skills.engine.spec import Costs, Execution
from fin_skills.engine.universe import Universe

EPS = 1e-12


def _fill_prices(fill_at: str, panel: Panel) -> pd.DataFrame:
    """The fill bar's own price, for every bar. next_vwap is the typical price (H+L+C)/3,
    a PROXY - a bar panel does not carry VWAP, and pretending it does is faked realism."""
    if fill_at == "next_open":
        return panel.open
    if fill_at == "next_close":
        return panel.close
    if fill_at == "next_twap":
        return (panel.open + panel.high + panel.low + panel.close) / 4.0
    return (panel.high + panel.low + panel.close) / 3.0


def run(panel: Panel,
        signal: pd.DataFrame | Callable[[Panel], pd.DataFrame],
        *,
        execution: Execution = Execution(),
        costs: Costs = Costs(),
        sizing: Sizer = equal_weight(),
        universe: Universe | Mapping[pd.Timestamp, Sequence[str]] | None = None,
        capital: float = 1_000_000.0,
        benchmark: pd.Series | None = None,
        strict: bool = True) -> BacktestResult:
    """One bar loop over `panel`, returning a BacktestResult whose to_bundle() every guard reads.

    `signal` as a CALLABLE is the preferred form: the engine then stores it in the Bundle's
    `signal_fn` slot, which is what makes assert_causal runnable on YOUR signal without
    you writing an adapter. Every invariant in fin_skills.engine.invariants is asserted as
    the run proceeds when strict=True; strict=False downgrades them to result.findings.
    """
    signal_fn = signal if callable(signal) else None
    raw = signal(panel) if signal_fn is not None else signal
    findings = invariants.enforce(invariants.before(panel, execution, raw), strict)

    idx, cols = panel.sessions.index, list(panel.close.columns)
    ppy = panel.sessions.periods_per_year
    sig = pd.DataFrame(raw).reindex(index=idx, columns=cols)
    close_ff, alive = panel.close.ffill(), panel.alive()
    uni = (universe.build(panel) if isinstance(universe, Universe)
           else {idx[0]: list(cols)} if universe is None
           else {pd.Timestamp(k): sorted(v) for k, v in dict(universe).items()})
    rebals = sorted(uni)
    adv_lookback = universe.adv_lookback if isinstance(universe, Universe) else 21

    # a name whose last living session is inside the sample must be closed out on it
    closeouts: dict[pd.Timestamp, list[str]] = {}
    for t in cols:
        live = alive.index[alive[t].to_numpy()]
        if len(live) and live[-1] != idx[-1]:
            closeouts.setdefault(live[-1], []).append(t)

    zero = pd.Series(0.0, index=cols)
    fill_px = _fill_prices(execution.fill_at, panel)
    shares, pending = zero.copy(), zero.copy()
    since: dict[str, pd.Timestamp] = {}
    queue: dict[pd.Timestamp, tuple[pd.Series, pd.Timestamp]] = {}
    Wv = np.zeros((len(idx), len(cols)))     # written by position; framed after the loop
    Pv = np.zeros((len(idx), len(cols)))
    eq, csh = pd.Series(0.0, index=idx), pd.Series(0.0, index=idx)
    net, gross = pd.Series(0.0, index=idx), pd.Series(0.0, index=idx)
    fills, unfilled = [], []
    cash = equity = prev_eq = float(capital)
    prev_w = zero

    for i, d in enumerate(idx):
        paid, mark = 0.0, close_ff.iloc[i]
        # ---------------------------------------------------------- 1. fills due this bar
        due, decided = pending.copy(), dict(since)
        order = queue.pop(d, None)
        if order is not None:
            due = due.add(order[0], fill_value=0.0).reindex(cols).fillna(0.0)
            for t in order[0].index[order[0].abs() > EPS]:
                decided.setdefault(t, order[1])
        pending, since = zero.copy(), {}
        want = due[due.abs() > EPS]
        if len(want):
            px = fill_px.iloc[i].reindex(want.index)
            vol = panel.volume.iloc[i].reindex(want.index).fillna(0.0)
            halted = ~(px.notna() & (px > 0) & alive.iloc[i].reindex(want.index)) | (vol <= 0)
            cap = (pd.Series(np.inf, index=want.index) if execution.participation_cap is None
                   else float(execution.participation_cap) * vol)
            qty = want.abs().clip(upper=cap).where(~halted, 0.0)
            filled = np.sign(want) * qty
            rest = want - filled
            traded = qty.index[qty > EPS]
            if len(traded):
                ctx = SlippageContext(d, traded, np.sign(want).reindex(traded),
                                      qty.reindex(traded), px.reindex(traded),
                                      vol.reindex(traded), panel)
                slip = pd.Series(costs.slippage(ctx), index=traded).astype(float)
                notional = qty.reindex(traded) * px.reindex(traded)
                charged = trade_charge(notional, slip, costs)
                cash -= float((filled.reindex(traded) * px.reindex(traded)).sum())
                cash -= float(charged.sum())
                paid += float(charged.sum())
                shares = shares.add(filled.reindex(cols).fillna(0.0), fill_value=0.0)
                for t in traded:
                    fills.append({"date": d, "decided_on": decided.get(t, d), "ticker": t,
                                  "side": float(np.sign(want[t])), "shares": float(qty[t]),
                                  "price": float(px[t]), "slip_bps": float(slip[t]),
                                  "commission": float(charged[t]), "kind": "trade"})
            for t, left in rest[rest.abs() > EPS].items():
                keep = (execution.allow_partial and i + 1 < len(idx)
                        and (not bool(halted[t]) or execution.on_halt == "carry"))
                unfilled.append({"date": d, "ticker": t, "side": float(np.sign(left)),
                                 "shares": float(abs(left)),
                                 "carried_to": idx[i + 1] if keep else pd.NaT})
                if keep:
                    pending[t] += float(left)
                    since[t] = decided.get(t, d)

        # ------------------------------------------------------- 2. delistings close out
        for t in closeouts.get(d, ()):
            pending[t], n = 0.0, float(shares[t])
            if abs(n) <= EPS:
                continue
            if execution.on_delist == "raise":
                raise RuntimeError(f"{t} stops trading on {d.date()} holding {n:g} shares; "
                                   f"Execution(on_delist='raise') refuses to guess a recovery")
            price = 0.0 if execution.on_delist == "zero_recovery" else float(mark[t])
            cash += n * price
            shares[t] = 0.0
            fills.append({"date": d, "decided_on": d, "ticker": t, "side": -float(np.sign(n)),
                          "shares": abs(n), "price": price, "slip_bps": 0.0,
                          "commission": 0.0, "kind": "closeout"})

        # ------------------------------------------------------------- 3. mark the book
        value = shares * mark
        carry = carry_charge(float(-value[value < 0].sum()), float(value.abs().sum()),
                             cash + float(value.sum()), costs, ppy)
        cash += cash * float(costs.cash_rate) / ppy - carry
        paid += carry
        equity = cash + float(value.sum())
        eq.iloc[i], csh.iloc[i], Pv[i] = equity, cash, shares.to_numpy()
        net.iloc[i] = equity / prev_eq - 1.0 if i else 0.0
        gross.iloc[i] = net.iloc[i] + (paid / prev_eq if i else 0.0)

        # ----------------------------------------------------------------- 4. and decide
        j = np.searchsorted(rebals, d, "right") - 1
        names = uni[rebals[j]] if j >= 0 else []
        ctx = SizingContext(d, sig.iloc[i].reindex(names), list(names),
                            panel.head(i + 1), prev_w, equity, panel.sessions)
        w = pd.Series(sizing(ctx), dtype=float).reindex(cols).fillna(0.0)
        g = float(w.abs().sum())
        if g > execution.max_gross:
            w *= execution.max_gross / g
        Wv[i], prev_w = w.to_numpy(), w
        if i + execution.signal_lag < len(idx):
            booked = pending.copy()
            for q, _ in queue.values():
                booked = booked.add(q, fill_value=0.0)
            target = (w * equity / mark.replace(0.0, np.nan)).fillna(0.0)
            when = idx[i + execution.signal_lag]
            prior = queue.get(when)
            new = (target - shares - booked.reindex(cols).fillna(0.0))
            queue[when] = ((new if prior is None else new.add(prior[0], fill_value=0.0)), d)
        prev_eq = equity

    W = pd.DataFrame(Wv, index=idx, columns=cols)
    P = pd.DataFrame(Pv, index=idx, columns=cols)
    turnover = W.diff().abs().sum(axis=1)
    turnover.iloc[0] = float(W.iloc[0].abs().sum())
    result = BacktestResult(
        returns=net.rename("returns"), gross_returns=gross.rename("gross_returns"),
        turnover=turnover.rename("turnover"), weights=W, positions=P, cash=csh.rename("cash"),
        equity=eq.rename("equity"), fills=_frame(fills, _FILL_COLS),
        unfilled=_frame(unfilled, _UNFILLED_COLS), universe=uni, panel=panel,
        benchmark=benchmark, signal_fn=signal_fn, findings=list(findings),
        spec={"execution": execution, "costs": costs, "sizing": getattr(sizing, "__name__", "?"),
              "sessions": repr(panel.sessions), "capital": float(capital),
              "participation_cap": execution.participation_cap, "adv_lookback": adv_lookback,
              "min_adv": universe.min_adv if isinstance(universe, Universe) else 0.0,
              "fingerprint": panel.fingerprint()})
    result.findings.extend(invariants.enforce(invariants.after(result), strict))
    return result


_FILL_COLS = ["date", "decided_on", "ticker", "side", "shares", "price", "slip_bps",
              "commission", "kind"]
_UNFILLED_COLS = ["date", "ticker", "side", "shares", "carried_to"]


def _frame(rows: list[dict], cols: list[str]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


__all__ = ["run"]
