"""fin_skills.engine.panel - aligned OHLCV whose adjustment convention is a FIELD.

Every engine in the survey takes a price frame and assumes something about it. This one
makes the assumption a constructor argument with no default, keeps the event tables that
would let you check it, and hashes the lot so a run can cite its data instead of a path.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from fin_skills.engine.actions import cum_factor, normalize_actions, normalize_listings
from fin_skills.engine.spec import Sessions


class Panel:
    """Aligned OHLCV for one universe on ONE declared calendar."""

    FIELDS = ("open", "high", "low", "close", "volume")
    ADJUSTMENTS = ("raw", "back", "forward", "raw+factors")
    # this field -> the convention name fin_skills.market_data.adjustment_check reports
    DETECTED_AS = {"raw": "raw", "raw+factors": "raw",
                   "back": "back-adjusted", "forward": "forward-adjusted"}

    def __init__(self, *, open: pd.DataFrame, high: pd.DataFrame, low: pd.DataFrame,
                 close: pd.DataFrame, volume: pd.DataFrame, sessions: Sessions,
                 adjustment: str, actions: pd.DataFrame | None = None,
                 listings: pd.DataFrame | None = None, currency: str = "USD",
                 source: str = "", retrieved_at: str = "") -> None:
        if adjustment not in self.ADJUSTMENTS:
            raise ValueError(f"adjustment must be one of {self.ADJUSTMENTS}, got {adjustment!r}; "
                             f"there is no default because every vendor has a different one")
        frames = dict(open=open, high=high, low=low, close=close, volume=volume)
        for name, f in frames.items():
            if not isinstance(f, pd.DataFrame):
                raise TypeError(f"{name} must be a DataFrame (dates x tickers)")
            if not f.index.equals(close.index) or list(f.columns) != list(close.columns):
                raise ValueError(f"{name} is not aligned with close (same index and columns)")
        if not close.index.equals(sessions.index):
            raise ValueError("the panel index must BE the declared session index - use "
                             "Panel.from_bars, which raises on a bar dated outside the calendar "
                             "instead of dropping it silently")
        for name, f in frames.items():
            setattr(self, name, f.astype(float))
        self.sessions = sessions
        self.adjustment = adjustment
        self.actions = None if actions is None else normalize_actions(actions)
        self.listings = None if listings is None else normalize_listings(listings)
        self.currency, self.source, self.retrieved_at = currency, source, retrieved_at

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_bars(cls, bars: pd.DataFrame, sessions: Sessions, adjustment: str, **kw) -> "Panel":
        """Long (date, ticker, open..volume) or wide MultiIndex (field, ticker) -> Panel."""
        b = pd.DataFrame(bars)
        if isinstance(b.columns, pd.MultiIndex):
            wide = {f: b[f] for f in cls.FIELDS}
            dates = pd.DatetimeIndex(b.index)
        else:
            miss = [c for c in ("date", "ticker", *cls.FIELDS) if c not in b.columns]
            if miss:
                raise ValueError(f"long bars frame is missing column(s) {miss}")
            b = b.assign(date=pd.to_datetime(b["date"]))
            wide = {f: b.pivot(index="date", columns="ticker", values=f) for f in cls.FIELDS}
            dates = pd.DatetimeIndex(wide["close"].index)
        stray = dates.difference(sessions.index)
        if len(stray):
            raise ValueError(f"{len(stray)} bar date(s) are not sessions on the declared "
                             f"calendar ({sessions.name or 'unnamed'}); first is "
                             f"{stray[0].date()}. A bar outside the calendar is an error, not "
                             f"something to drop silently")
        cols = sorted(wide["close"].columns)
        wide = {f: v.reindex(index=sessions.index, columns=cols) for f, v in wide.items()}
        return cls(sessions=sessions, adjustment=adjustment, **wide, **kw)

    def _new(self, **kw) -> "Panel":
        base = dict(sessions=self.sessions, adjustment=self.adjustment, actions=self.actions,
                    listings=self.listings, currency=self.currency, source=self.source,
                    retrieved_at=self.retrieved_at, **{f: getattr(self, f) for f in self.FIELDS})
        base.update(kw)
        return Panel(**base)

    # ------------------------------------------------------------------ conventions
    def readjust(self, to: str) -> "Panel":
        """Recompute prices onto another convention from `actions`.

        Refuses when `actions` is None, because a convention with no event table is
        unfalsifiable - and an unfalsifiable convention is worse than a raw quote.
        Volume moves by the inverse factor, so dollar volume is invariant across splits.
        """
        if to not in self.ADJUSTMENTS:
            raise ValueError(f"adjustment must be one of {self.ADJUSTMENTS}, got {to!r}")
        if self.actions is None:
            raise ValueError("readjust needs `actions`: a convention with no event table "
                             "cannot be checked, so this engine will not assert one")
        k = (cum_factor(self.actions, self.close.index, self.close.columns, to)
             / cum_factor(self.actions, self.close.index, self.close.columns, self.adjustment))
        px = {f: getattr(self, f) * k for f in ("open", "high", "low", "close")}
        return self._new(adjustment=to, volume=self.volume / k, **px)

    # ------------------------------------------------------------------ views
    def alive(self) -> pd.DataFrame:
        """bool, dates x tickers: the name traded that session on this panel's own data."""
        a = self.close.notna()
        if self.listings is not None:
            for row in self.listings.itertuples(index=False):
                if row.ticker not in a.columns:
                    continue
                if pd.notna(row.start_date):
                    a.loc[a.index < row.start_date, row.ticker] = False
                if pd.notna(row.end_date):
                    a.loc[a.index > row.end_date, row.ticker] = False
        return a

    def liquidity(self, lookback: int = 21) -> pd.DataFrame:
        """Trailing dollar ADV, window ENDING at each session - causal by construction."""
        return (self.close * self.volume).rolling(int(lookback), min_periods=1).mean()

    def ohlcv(self, ticker: str) -> pd.DataFrame:
        """One instrument's OHLCV frame - the shape the `bars` slot wants."""
        return pd.DataFrame({f: getattr(self, f)[ticker] for f in self.FIELDS})

    def swap(self, bars: pd.DataFrame, ticker: str) -> "Panel":
        """This panel restricted to one name, with its OHLCV replaced by `bars`.

        The engine's re-entry point: assert_causal perturbs a bars frame, and this turns
        that frame back into a Panel, so a guard can drive the engine with no adapter.
        """
        cut = {f: bars[[f]].rename(columns={f: ticker}).astype(float) for f in self.FIELDS}
        return self._new(sessions=self.sessions.restrict(pd.DatetimeIndex(bars.index)), **cut)

    def head(self, n: int) -> "Panel":
        """This panel's first n sessions, sharing every table. The bar loop hands one of
        these to the sizer as `history`, so it skips checks the frames already passed."""
        p = object.__new__(Panel)
        p.__dict__.update(self.__dict__)
        for f in self.FIELDS:
            setattr(p, f, getattr(self, f).iloc[:n])
        p.sessions = self.sessions.head(n)
        return p

    def restrict(self, start=None, end=None, tickers=None) -> "Panel":
        idx = self.close.index
        lo = 0 if start is None else int(idx.searchsorted(pd.Timestamp(start), "left"))
        hi = len(idx) if end is None else int(idx.searchsorted(pd.Timestamp(end), "right"))
        cols = (list(self.close.columns) if tickers is None
                else [t for t in tickers if t in self.close.columns])
        cut = {f: getattr(self, f).iloc[lo:hi][cols] for f in self.FIELDS}
        return self._new(sessions=self.sessions.restrict(idx[lo:hi]), **cut)

    # ------------------------------------------------------------------ provenance
    def fingerprint(self) -> str:
        """sha256 over the traded arrays plus every declared field. Store THIS, not a path."""
        h = hashlib.sha256()
        for f in self.FIELDS:
            h.update(f.encode())
            h.update(np.ascontiguousarray(getattr(self, f).to_numpy(dtype=float)).tobytes())
        h.update("|".join(map(str, self.close.columns)).encode())
        h.update(self.close.index.asi8.tobytes())
        for v in (self.adjustment, self.currency, self.source, self.retrieved_at,
                  self.sessions.tz, self.sessions.periods_per_year, self.sessions.name):
            h.update(f"|{v}".encode())
        for table in (self.actions, self.listings):
            h.update(b"|none" if table is None
                     else pd.util.hash_pandas_object(table, index=False).to_numpy().tobytes())
        return h.hexdigest()

    def __repr__(self) -> str:
        return (f"Panel({len(self.close.columns)} names x {len(self.close)} sessions, "
                f"adjustment={self.adjustment!r}, calendar={self.sessions.name or 'unnamed'}, "
                f"actions={0 if self.actions is None else len(self.actions)}, "
                f"listings={'-' if self.listings is None else len(self.listings)})")


__all__ = ["Panel"]
