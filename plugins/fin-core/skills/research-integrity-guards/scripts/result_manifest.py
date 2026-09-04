#!/usr/bin/env python3
"""Emit a strategy result card, and refuse to emit one that is missing its provenance.

The point: a bare Sharpe ratio is not a result. This module makes the missing pieces
impossible to omit silently — `render()` raises unless every required field is present,
and it prints the cost curve and trial count next to the headline number.

Usage:
    from result_manifest import ResultCard

    card = ResultCard(
        strategy_id="core-etf-trend-v2",
        universe=Universe(source="EODHD", asof="2026-01-02", includes_delisted=True,
                          n_names=512, membership_rule="ADV>1e6 at each rebalance"),
        data=[DataSource("EODHD EOD", "2026-01-02T09:00Z", "backward-adjusted"),
              DataSource("SEC companyfacts", "2026-01-02T09:05Z", "PIT, filed<=asof")],
        split=Split(scheme="CombinatorialPurgedCV", purge="20D", embargo="5D",
                    train="2010-2019", test="2020-2026"),
        costs=CostModel(spread_bps=2.0, slippage_bps=3.0, impact_model="sqrt(participation)",
                        borrow_bps=50.0, financing_bps=0.0, cash_rate_series="^IRX"),
        trials=TrialCount(n=137, ledger_path="research/trials.jsonl"),
        metrics={"sharpe_net": 0.62, "sharpe_gross": 1.10, "ann_vol": 0.11,
                 "max_drawdown": -0.18, "annualization": 252, "rf_convention": "annual, geometric"},
        cost_curve={0: 1.10, 5: 0.88, 10: 0.62, 20: 0.15, 50: -0.44},
        benchmark={"name": "SPY", "excess_ann": 0.004, "capm_alpha": -0.002, "alpha_t": -0.31},
        falsifier="Fails if net-of-cost excess return over SPY is <=0 across the locked window.",
    )
    print(card.render())
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone


@dataclass
class Universe:
    source: str
    asof: str
    includes_delisted: bool
    n_names: int
    membership_rule: str


@dataclass
class DataSource:
    name: str
    retrieved_at: str
    adjustment: str          # e.g. "backward-adjusted", "raw+factors", "PIT, filed<=asof"


@dataclass
class Split:
    scheme: str              # e.g. "CombinatorialPurgedCV", "walk-forward"
    purge: str               # max label horizon, e.g. "20D"
    embargo: str
    train: str
    test: str


@dataclass
class CostModel:
    spread_bps: float
    slippage_bps: float
    impact_model: str
    borrow_bps: float = 0.0
    financing_bps: float = 0.0
    cash_rate_series: str = ""


@dataclass
class TrialCount:
    n: int
    ledger_path: str


@dataclass
class ResultCard:
    strategy_id: str
    universe: Universe
    data: list[DataSource]
    split: Split
    costs: CostModel
    trials: TrialCount
    metrics: dict
    cost_curve: dict          # {round_trip_bps: sharpe_or_return}
    benchmark: dict
    falsifier: str
    llm_training_cutoff: str = ""     # required if an LLM produced any signal
    regimes_covered: list = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    REQUIRED_METRICS = ("annualization", "rf_convention")

    def problems(self) -> list[str]:
        p: list[str] = []
        if not self.universe.includes_delisted:
            p.append("universe excludes delisted names -> the reported number is an UPPER BOUND, "
                     "not an estimate")
        for m in self.REQUIRED_METRICS:
            if m not in self.metrics:
                p.append(f"metrics is missing {m!r} - a Sharpe without its annualization factor "
                         f"and risk-free convention is not comparable to anything")
        if not self.cost_curve or len(self.cost_curve) < 3:
            p.append("cost_curve needs at least 3 points (e.g. 0/10/20 bps round-trip); "
                     "a point estimate hides where the strategy dies")
        if self.trials.n <= 1:
            p.append("trial count <=1 is almost never true - every parameter you tried counts, "
                     "including abandoned ones and everything an automated search evaluated")
        if not self.trials.ledger_path:
            p.append("no trial ledger path - the trial count is unverifiable")
        if not self.benchmark:
            p.append("no benchmark comparison - an absolute return is not evidence of skill")
        if "capm_alpha" not in self.benchmark:
            p.append("no factor-adjusted alpha - long-equity beta in a bull market is not alpha")
        if not self.regimes_covered:
            p.append("no regime coverage stated - a single bull quarter is not a backtest")
        if not self.falsifier:
            p.append("no falsifier stated - say in advance what result would kill this")
        return p

    def verdict(self) -> str:
        """Where the strategy dies on the cost curve."""
        if not self.cost_curve:
            return "unknown"
        pts = sorted((int(k), v) for k, v in self.cost_curve.items())
        for bps, v in pts:
            if v <= 0:
                return f"unprofitable at {bps} bps round-trip"
        return f"still positive at {pts[-1][0]} bps round-trip"

    def render(self, strict: bool = True) -> str:
        probs = self.problems()
        if strict and probs:
            raise ValueError(
                "REFUSING to render an incomplete result card:\n  - " + "\n  - ".join(probs)
            )
        lines = [
            f"STRATEGY RESULT CARD - {self.strategy_id}",
            f"  generated       : {self.created_at}",
            f"  universe        : {self.universe.n_names} names from {self.universe.source} "
            f"as of {self.universe.asof}; delisted included = {self.universe.includes_delisted}",
            f"  membership rule : {self.universe.membership_rule}",
            f"  data            : " + "; ".join(
                f"{d.name} [{d.adjustment}] @ {d.retrieved_at}" for d in self.data),
            f"  split           : {self.split.scheme}, purge {self.split.purge}, "
            f"embargo {self.split.embargo} | train {self.split.train} | test {self.split.test}",
            f"  costs           : spread {self.costs.spread_bps}bp, slip {self.costs.slippage_bps}bp, "
            f"impact {self.costs.impact_model}, borrow {self.costs.borrow_bps}bp, "
            f"cash {self.costs.cash_rate_series or 'NOT MODELLED'}",
            f"  trials          : N={self.trials.n}  (ledger: {self.trials.ledger_path})",
            f"  regimes         : {', '.join(self.regimes_covered) or 'NONE STATED'}",
        ]
        if self.llm_training_cutoff:
            lines.append(f"  LLM cutoff      : {self.llm_training_cutoff} "
                         f"vs test {self.split.test}  <- check for overlap")
        lines += [
            "  metrics         : " + ", ".join(f"{k}={v}" for k, v in self.metrics.items()),
            "  benchmark       : " + ", ".join(f"{k}={v}" for k, v in self.benchmark.items()),
            "  cost curve      : " + ", ".join(
                f"{k}bp->{v}" for k, v in sorted(self.cost_curve.items(), key=lambda x: int(x[0]))),
            f"  VERDICT         : {self.verdict()}",
            f"  falsifier       : {self.falsifier}",
        ]
        if probs:
            lines.append("  WARNINGS        :")
            lines += [f"    ! {p}" for p in probs]
        return "\n".join(lines)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str, sort_keys=True)


if __name__ == "__main__":
    card = ResultCard(
        strategy_id="demo-trend-v1",
        universe=Universe("yfinance", "2026-09-03", includes_delisted=False,
                          n_names=5, membership_rule="hand-picked core ETFs"),
        data=[DataSource("yfinance", "2026-09-03T12:00Z", "auto_adjust=True")],
        split=Split("walk-forward", "0D", "0D", "2010-2019", "2020-2026"),
        costs=CostModel(1.0, 2.0, "none"),
        trials=TrialCount(1, ""),
        metrics={"sharpe_net": 1.9},
        cost_curve={},
        benchmark={},
        falsifier="",
    )
    print("problems found:")
    for p in card.problems():
        print("  -", p)
    print()
    print(card.render(strict=False))
