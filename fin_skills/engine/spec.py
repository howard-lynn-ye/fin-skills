"""fin_skills.engine.spec - the three declarations a run refuses to infer.

Sessions, Execution and Costs are frozen and validated at construction, so the errors that
usually surface as a suspiciously good equity curve surface as a ValueError instead.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from fin_skills.engine.costs import SlippageModel, fixed_bps

FILL_AT = ("next_open", "next_close", "next_twap", "next_vwap")
ON_DELIST = ("close_at_last", "zero_recovery", "raise")
ON_HALT = ("carry", "force_flat")


@dataclass(frozen=True, eq=False)
class Sessions:
    """The trading calendar, DECLARED. There is no default and none is computed.

    exchange_calendars derives GLOBAL_DEFAULT_START/_END from pd.Timestamp.now() at IMPORT
    (verified in market-data-sourcing SKILL.md section 5), so a calendar built from its
    defaults returns different sessions tomorrow. This class refuses to have a default.
    """

    index: pd.DatetimeIndex
    tz: str
    periods_per_year: int
    name: str = ""
    half_days: frozenset = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        idx = pd.DatetimeIndex(self.index)
        if len(idx) == 0:
            raise ValueError("Sessions.index is empty")
        if idx.tz is not None:
            raise ValueError("Sessions.index must be tz-naive session dates; the zone is "
                             "declared separately in `tz` so labels cannot drift")
        if not idx.is_monotonic_increasing or idx.has_duplicates:
            raise ValueError("Sessions.index must be sorted and unique")
        if not isinstance(self.tz, str) or not self.tz:
            raise ValueError("Sessions.tz has no default: declare the exchange-local zone")
        if int(self.periods_per_year) <= 0:
            raise ValueError("periods_per_year must be positive (252 equities, 365 crypto, "
                             "260 FX, 12 monthly)")
        object.__setattr__(self, "index", idx)
        object.__setattr__(self, "periods_per_year", int(self.periods_per_year))

    @classmethod
    def from_index(cls, index, tz: str, periods_per_year: int, name: str = "") -> "Sessions":
        return cls(pd.DatetimeIndex(index), tz, periods_per_year, name)

    @classmethod
    def from_panel_index(cls, index, tz: str, periods_per_year: int) -> "Sessions":
        """Adopt the data's own index as the calendar - honest for crypto and for a vendor
        whose sessions you trust, and recorded as name='observed' so the manifest says so."""
        return cls(pd.DatetimeIndex(index), tz, periods_per_year, "observed")

    def bars_between(self, a, b) -> int:
        """BAR count between two timestamps. Never calendar days."""
        return (int(self.index.searchsorted(pd.Timestamp(b), "left"))
                - int(self.index.searchsorted(pd.Timestamp(a), "left")))

    def shift(self, ts, k: int) -> pd.Timestamp:
        i = int(self.index.searchsorted(pd.Timestamp(ts), "left")) + int(k)
        if not 0 <= i < len(self.index):
            raise IndexError(f"{ts} shifted by {k} sessions falls outside the declared calendar")
        return self.index[i]

    def is_session(self, ts) -> bool:
        return pd.Timestamp(ts) in self.index

    def restrict(self, index) -> "Sessions":
        return Sessions(pd.DatetimeIndex(index), self.tz, self.periods_per_year, self.name,
                        self.half_days)

    def __repr__(self) -> str:
        return (f"Sessions({self.name or 'unnamed'}, {len(self.index)} sessions "
                f"{self.index[0].date()}..{self.index[-1].date()}, tz={self.tz}, "
                f"periods_per_year={self.periods_per_year})")


@dataclass(frozen=True)
class Execution:
    """When an order fills, how much of it fills, and what happens when it cannot."""

    fill_at: str = "next_open"
    signal_lag: int = 1
    participation_cap: float | None = 0.05
    allow_partial: bool = True
    on_delist: str = "close_at_last"
    on_halt: str = "carry"
    max_gross: float = 1.0

    def __post_init__(self) -> None:
        if int(self.signal_lag) < 1:
            raise ValueError("signal_lag must be >= 1 session. signal_lag=0 is not an option: "
                             "it is vectorbt's default (Order.price=np.inf resolves to the "
                             "current close) and it is the bug this engine exists to refuse")
        for value, allowed, label in ((self.fill_at, FILL_AT, "fill_at"),
                                      (self.on_delist, ON_DELIST, "on_delist"),
                                      (self.on_halt, ON_HALT, "on_halt")):
            if value not in allowed:
                raise ValueError(f"{label} must be one of {allowed}, got {value!r}")
        if self.participation_cap is not None and float(self.participation_cap) <= 0:
            raise ValueError("participation_cap must be positive, or None for uncapped")
        if float(self.max_gross) <= 0:
            raise ValueError("max_gross must be positive")


@dataclass(frozen=True, eq=False)
class Costs:
    """Every charge, in the units the library's guards expect."""

    commission_bps: float = 0.0
    min_commission: float = 0.0        # per order, in account currency
    spread_bps: float = 0.0            # HALF-spread paid on every fill, both directions
    slippage: SlippageModel = fixed_bps(0.0)
    borrow_bps_annual: float = 0.0     # charged on the short leg's notional
    financing_bps_annual: float = 0.0  # charged on gross leverage above 1.0
    cash_rate: float = 0.0             # ANNUAL decimal (0.05 = 5%). Never a per-period rate.

    def __post_init__(self) -> None:
        if not callable(self.slippage):
            raise ValueError("slippage must be a SlippageModel callable, e.g. fixed_bps(2.0)")
        if abs(float(self.cash_rate)) >= 0.5:
            raise ValueError(f"cash_rate={self.cash_rate!r} looks like a percentage; it is an "
                             f"ANNUAL decimal (0.05 = 5%), and rf_convention exists because "
                             f"empyrical reads the same 5% as 5% PER PERIOD")

    def round_trip_bps(self) -> float:
        """The single number cost_curve is anchored on: 2*(commission + spread + slippage).

        The slippage term is the model's own stated bps where it has one (fixed_bps); an
        impact model has no context-free number, so it contributes 0 here and the honest
        figure is the realised one in `BacktestResult.fills`.
        """
        return 2.0 * (float(self.commission_bps) + float(self.spread_bps)
                      + float(getattr(self.slippage, "bps_hint", 0.0)))


__all__ = ["Costs", "Execution", "FILL_AT", "ON_DELIST", "ON_HALT", "Sessions"]
