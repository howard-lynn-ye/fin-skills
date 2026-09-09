"""fin_skills.engine.universe - membership evaluated only on data available at each rebalance.

The output shape is exactly what `pit_universe`'s `universe` slot wants, so the guard that
flags a current-snapshot screen can be run on the engine's own membership with no adapter.
"""
from __future__ import annotations

import pandas as pd

from fin_skills.engine.panel import Panel


class Universe:
    """The point-in-time membership rule.

    rebalance    : a pandas offset alias ("ME", "QE", "W-FRI") resolved to the LAST session
                   of each period, or an explicit DatetimeIndex snapped back to sessions.
    min_adv      : trailing dollar-ADV floor, measured over a window ENDING at the
                   rebalance - the screen everyone writes with today's ADV by accident.
    members      : ticker / start_date / end_date. Supplied, it is the membership truth;
                   absent, the panel's own `alive()` is, which is weaker and says so.
    """

    def __init__(self, rebalance: str | pd.DatetimeIndex = "ME", min_adv: float = 0.0,
                 adv_lookback: int = 21, max_names: int | None = None,
                 members: pd.DataFrame | None = None) -> None:
        self.rebalance = rebalance
        self.min_adv = float(min_adv)
        self.adv_lookback = int(adv_lookback)
        self.max_names = None if max_names is None else int(max_names)
        self.members = None if members is None else _normalise(members)

    def dates(self, sessions) -> pd.DatetimeIndex:
        idx = sessions.index
        if isinstance(self.rebalance, str):
            last = pd.Series(idx, index=idx).resample(self.rebalance).last().dropna()
            return pd.DatetimeIndex(sorted(set(last)))
        want = pd.DatetimeIndex(self.rebalance)
        pos = idx.searchsorted(want, "right") - 1
        return pd.DatetimeIndex(sorted({idx[p] for p in pos if p >= 0}))

    def build(self, panel: Panel) -> dict[pd.Timestamp, list[str]]:
        """{rebalance date: [tickers]} - the shape pit_universe reads."""
        adv = panel.liquidity(self.adv_lookback)
        alive = None if self.members is not None else panel.alive()
        out: dict[pd.Timestamp, list[str]] = {}
        for d in self.dates(panel.sessions):
            if self.members is not None:
                m = self.members
                live = m[(m["start_date"] <= d) & (m["end_date"].isna() | (m["end_date"] >= d))]
                names = [t for t in sorted(set(live["ticker"])) if t in adv.columns]
            else:
                names = [t for t in alive.columns if bool(alive.loc[d, t])]
            a = adv.loc[d]
            names = [t for t in names if pd.notna(a.get(t)) and float(a[t]) >= self.min_adv]
            if self.max_names is not None:
                names = sorted(names, key=lambda t: -float(a[t]))[:self.max_names]
            out[pd.Timestamp(d)] = sorted(names)
        return out

    def __repr__(self) -> str:
        return (f"Universe(rebalance={self.rebalance!r}, min_adv={self.min_adv:g}, "
                f"adv_lookback={self.adv_lookback}, max_names={self.max_names}, "
                f"members={'-' if self.members is None else len(self.members)})")


def _normalise(members: pd.DataFrame) -> pd.DataFrame:
    from fin_skills.engine.actions import normalize_listings
    return normalize_listings(members)


__all__ = ["Universe"]
