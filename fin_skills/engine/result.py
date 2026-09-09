"""fin_skills.engine.result - the run, as a Bundle.

This is the whole reason for a twelfth engine. benchmarks/leak_bench.py hand-writes twelve
`_adapt_*` functions to marshal a pipeline's artefacts into guard keywords; to_bundle()
does that job once, so `check(result.to_bundle())` runs six guards on a real run with no
adapter and names the slot every skipped guard still needs.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from fin_skills.api import Bundle, RunReport, check
from fin_skills.engine.actions import as_delisting_table
from fin_skills.engine.panel import Panel

COST_LEVELS = (0.0, 5.0, 10.0, 20.0, 50.0)


def _sharpe(r: pd.Series, ppy: int) -> float:
    sd = float(r.std(ddof=1))
    return float(r.mean() / sd * np.sqrt(ppy)) if sd > 0 else float("nan")


@dataclass(frozen=True, eq=False)
class BacktestResult:
    """What one run produced. Every frame is indexed on the declared session calendar."""

    returns: pd.Series          # NET of everything in Costs
    gross_returns: pd.Series    # before commission / spread / slippage / borrow / financing
    turnover: pd.Series         # ONE-WAY target turnover, |w[t] - w[t-1]|.sum()
    weights: pd.DataFrame       # target weights as of the DECISION bar
    positions: pd.DataFrame     # held shares
    cash: pd.Series             # the cash leg - a position, not a residual
    equity: pd.Series
    fills: pd.DataFrame         # date, decided_on, ticker, side, shares, price, slip_bps, ...
    unfilled: pd.DataFrame      # what participation_cap deferred, and to when
    universe: dict
    panel: Panel
    spec: dict
    benchmark: pd.Series | None = None
    signal_fn: object = None
    findings: list = field(default_factory=list)

    # ------------------------------------------------------------------- numbers
    def net(self, round_trip_bps: float | None = None) -> pd.Series:
        """The engine's own net series, or gross less a flat round-trip assumption.

        net(0) IS gross_returns, so a cost curve drawn from this method and one drawn by
        cost_curve() agree by construction rather than by coincidence.
        """
        if round_trip_bps is None:
            return self.returns
        return self.gross_returns - self.turnover * float(round_trip_bps) / 1e4

    def held_weights(self) -> pd.DataFrame:
        """Realised weights from the accounting, not the targets. With cash they sum to 1."""
        return self.positions.mul(self.panel.close.ffill()).div(self.equity, axis=0)

    def cost_curve(self, bps=COST_LEVELS) -> dict[int, float]:
        ppy = self.panel.sessions.periods_per_year
        return {int(b): round(_sharpe(self.net(b), ppy), 4) for b in bps}

    # -------------------------------------------------------------------- bundle
    def to_bundle(self, **extra) -> Bundle:
        """The artefacts of this run under the library's shared vocabulary."""
        p, costs = self.panel, self.spec["costs"]
        slots: dict = {
            "returns": self.gross_returns, "turnover": self.turnover,
            "periods_per_year": p.sessions.periods_per_year, "rf": float(costs.cash_rate),
            "prices": p.close, "universe": self.universe,
            "rebalance_dates": sorted(self.universe),
            "liquidity": p.liquidity(self.spec["adv_lookback"]),
            "min_adv": self.spec["min_adv"] or None,
            "adv_lookback": self.spec["adv_lookback"],
            "underlying_returns": p.close.pct_change(fill_method=None).mean(axis=1).fillna(0.0),
            "position": self.weights.sum(axis=1), "dates": self.returns.index,
            "cost_bps": self.spec["costs"].round_trip_bps() or None,
        }
        if p.listings is not None:
            slots["listings"] = as_delisting_table(p.listings)
            slots["members"] = p.listings
        if self.benchmark is not None:
            slots["benchmark_returns"] = self.benchmark
        if p.actions is not None and len(p.actions):
            t = str(p.actions["ticker"].value_counts().index[0])
            if t in p.close.columns:
                slots["close"] = p.close[t].dropna()
                slots["actions"] = p.actions[p.actions["ticker"] == t][["date", "ratio", "kind"]]
                slots["expected"] = p.DETECTED_AS[p.adjustment]
        if self.signal_fn is not None:
            t = str(p.close.notna().sum().idxmax())
            fn = self.signal_fn
            def signal_fn(df: pd.DataFrame, _t=t, _fn=fn, _p=p):
                return pd.DataFrame(_fn(_p.swap(df, _t)))[_t]
            signal_fn.__name__ = getattr(fn, "__name__", "signal")
            slots["bars"] = p.ohlcv(t).dropna(how="all")
            slots["signal_fn"] = signal_fn
        slots.update(extra)
        return Bundle(**slots)

    def check(self, guards=None, *, strict: bool = False) -> RunReport:
        return check(self.to_bundle(), guards=guards, strict=strict)

    # ---------------------------------------------------------------------- card
    def card(self, strategy_id: str, falsifier: str, **kw):
        """A result_manifest.ResultCard pre-filled from the run.

        Provenance comes from the panel; the cost curve from net() at 0/5/10/20/50 bps.
        The trial count and the falsifier stay the caller's - they are not the engine's to
        know, and a card that invented them would be the thing the card exists to prevent.
        """
        from fin_skills.core import result_manifest as rm
        p, c, ex = self.panel, self.spec["costs"], self.spec["execution"]
        names = sorted({t for v in self.universe.values() for t in v})
        dead = bool(p.listings is not None and p.listings["end_date"].notna().any())
        bench = kw.pop("benchmark", None)
        if bench is None and self.benchmark is not None:
            b = pd.Series(self.benchmark).reindex(self.returns.index).fillna(0.0)
            beta, alpha = np.polyfit(b.to_numpy(), self.returns.to_numpy(), 1)
            bench = {"series": "supplied", "beta": round(float(beta), 4),
                     "capm_alpha": round(float(alpha) * p.sessions.periods_per_year, 4)}
        return rm.ResultCard(
            strategy_id=strategy_id, falsifier=falsifier,
            universe=kw.pop("universe", rm.Universe(
                source=p.source or "declared panel", asof=p.retrieved_at or "unstated",
                includes_delisted=dead, n_names=len(names),
                membership_rule=f"{len(self.universe)} rebalances, trailing "
                                f"{self.spec['adv_lookback']}-bar ADV >= "
                                f"{self.spec['min_adv']:g}")),
            data=kw.pop("data", [rm.DataSource(p.source or "unstated",
                                               p.retrieved_at or "unstated",
                                               f"{p.adjustment} / {p.fingerprint()[:16]}")]),
            split=kw.pop("split", rm.Split("single pass", "0D", "0D", "-",
                                           f"{self.returns.index[0].date()}.."
                                           f"{self.returns.index[-1].date()}")),
            costs=kw.pop("costs", rm.CostModel(
                spread_bps=c.spread_bps, slippage_bps=getattr(c.slippage, "bps_hint", 0.0),
                impact_model=getattr(c.slippage, "__name__", "?"), borrow_bps=c.borrow_bps_annual,
                financing_bps=c.financing_bps_annual,
                cash_rate_series=f"{c.cash_rate:g} annual")),
            trials=kw.pop("trials", rm.TrialCount(0, "")),
            metrics=kw.pop("metrics", {
                "sharpe_net": round(_sharpe(self.returns, p.sessions.periods_per_year), 4),
                "annualization": f"{p.sessions.periods_per_year} bars/yr on "
                                 f"{p.sessions.name or 'a declared calendar'}",
                "rf_convention": f"cash_rate={c.cash_rate:g} ANNUAL, earned on the cash leg",
                "fill": f"{ex.fill_at}, signal_lag={ex.signal_lag}"}),
            cost_curve=kw.pop("cost_curve", self.cost_curve()),
            benchmark=bench or {}, **kw)

    def __repr__(self) -> str:
        return (f"BacktestResult({len(self.returns)} sessions, {len(self.fills)} fills, "
                f"{len(self.universe)} rebalances, {len(self.findings)} finding(s))")


__all__ = ["BacktestResult", "COST_LEVELS"]
