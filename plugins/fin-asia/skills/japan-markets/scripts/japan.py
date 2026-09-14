"""Tokyo Stock Exchange rules a Western (or a Chinese) backtest engine gets wrong.

WHY this exists - the TSE cash session has a hole in the middle of it and moved its close
in 2024, so every wall-clock assumption about a Japanese trading day is wrong twice:

  1. **The lunch break.** 09:00-11:30 and 12:30-15:30. A wall-clock grid from the open to
     the close is 390 minutes; the session is **330**. `resample('1min')` invents 60 bars
     a day that never traded, and any rolling window measured in bars silently changes the
     amount of real time it covers when it crosses 11:30.
  2. **The close moved on 2024-11-05**, 15:00 -> 15:30. The same session was 300 bars
     before that date and 330 after. A constant bar count is wrong on one side of it.
  3. **Volatility annualisation uses the bar count twice** - once in the sample standard
     deviation and once in the sqrt(bars per year) scaling. Get one of them from the wall
     clock and the other from the session and the error does not cancel.
  4. **The Nikkei 225 is PRICE weighted with a divisor**, not cap weighted. A 40,000-yen
     stock carries ~40x the index weight of a 1,000-yen stock regardless of company size,
     and every split, constituent change and rights issue has to be absorbed by moving the
     divisor. Skip the divisor update and the index gaps by the whole price change.
  5. **Ticks are tiered, and TOPIX500 members trade on a FINER table than everything
     else.** Rounding an order price with a single tick size puts it off the book.

Everything printed by `__main__` is computed here from seeded synthetic data. No network,
no file writes, ASCII-only output (Japanese terms live in the SKILL.md, not in stdout).

Usage:
    from japan import session_minutes, session_bar_count, tick_size, settlement_date

    session_bar_count(1, "2024-11-05")     # 330
    session_bar_count(1, "2024-11-01")     # 300 - the close was 15:00 that week
    tick_size(3000.0, topix500=True)       # 0.5   (ordinary stocks: 1.0)
"""
from __future__ import annotations

import math
from datetime import date, datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

DateLike = str | date | datetime | pd.Timestamp

# ------------------------------------------------------------------ rule effective dates
CLOSE_EXTENDED_FROM = date(2024, 11, 5)   # 15:00 -> 15:30, with a 15:25 pre-closing session
T2_SETTLEMENT_FROM = date(2019, 7, 16)    # T+3 -> T+2 for cash equities
TSE_RESTRUCTURED_ON = date(2022, 4, 4)    # 1st/2nd/Mothers/JASDAQ -> Prime/Standard/Growth
FINE_TICKS_TOPIX500_FROM = date(2023, 6, 5)   # the fine table left TOPIX100 for TOPIX500
NIKKEI_PAF_FROM = date(2021, 10, 1)       # price adjustment factor replaced par-value scaling
TICK_RULE_BECOMES_LIQUIDITY_BASED = date(2027, 3, 1)   # index class -> Spread-to-Tick Ratio
RANDOM_CLOSING_FROM = date(2027, 10, 12)  # the afternoon Itayose time becomes random

MORNING_OPEN = timedelta(hours=9)
MORNING_CLOSE = timedelta(hours=11, minutes=30)
AFTERNOON_OPEN = timedelta(hours=12, minutes=30)

# Short selling: the price restriction (the Japanese trigger rule) arms when a stock has
# fallen this far from the EXCHANGE BASE PRICE, and it then runs to the end of the NEXT
# trading day. Below the trigger a short may not be placed at or under the latest price.
SHORT_SALE_TRIGGER_DROP = 0.10

# Short-position thresholds, as a fraction of shares outstanding.
SHORT_REPORT_THRESHOLD = 0.002    # report via the broker at 0.2%
SHORT_DISCLOSE_THRESHOLD = 0.005  # the exchange publishes the holder at 0.5%

# ------------------------------------------------------------------ tick tables
# (upper price bound INCLUSIVE, tick). The last row's bound is infinite.
# The fine table applies to TOPIX500 constituents (TOPIX100 + TOPIX Mid400) from
# 2023-06-05; before that date it was TOPIX100 only.
TICK_ORDINARY: tuple[tuple[float, float], ...] = (
    (1_000.0, 1.0), (3_000.0, 1.0), (5_000.0, 5.0), (10_000.0, 10.0),
    (30_000.0, 10.0), (50_000.0, 50.0), (100_000.0, 100.0), (300_000.0, 100.0),
    (500_000.0, 500.0), (1_000_000.0, 1_000.0), (3_000_000.0, 1_000.0),
    (5_000_000.0, 5_000.0), (10_000_000.0, 10_000.0), (30_000_000.0, 10_000.0),
    (50_000_000.0, 50_000.0), (math.inf, 100_000.0),
)
TICK_TOPIX500: tuple[tuple[float, float], ...] = (
    (1_000.0, 0.1), (3_000.0, 0.5), (5_000.0, 1.0), (10_000.0, 1.0),
    (30_000.0, 5.0), (50_000.0, 10.0), (100_000.0, 10.0), (300_000.0, 50.0),
    (500_000.0, 100.0), (1_000_000.0, 100.0), (3_000_000.0, 500.0),
    (5_000_000.0, 1_000.0), (10_000_000.0, 1_000.0), (30_000_000.0, 5_000.0),
    (math.inf, 10_000.0),
)


def _to_date(d: DateLike | None) -> date:
    if d is None:
        raise ValueError(
            "date is REQUIRED. The TSE close moved on 2024-11-05 and settlement moved on "
            "2019-07-16; a Japanese session rule without a date is a rule for a year you "
            "did not pick."
        )
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    return pd.Timestamp(d).date()


# ------------------------------------------------------------------ sessions
def afternoon_close(day: DateLike) -> timedelta:
    """15:30 from 2024-11-05, 15:00 before it."""
    return (timedelta(hours=15, minutes=30) if _to_date(day) >= CLOSE_EXTENDED_FROM
            else timedelta(hours=15))


def session_minutes(day: DateLike) -> tuple[int, int]:
    """(morning minutes, afternoon minutes). (150, 180) today; (150, 150) before 2024-11-05."""
    am = int((MORNING_CLOSE - MORNING_OPEN).total_seconds() // 60)
    pm = int((afternoon_close(day) - AFTERNOON_OPEN).total_seconds() // 60)
    return am, pm


def wall_clock_minutes(day: DateLike) -> int:
    """Open to close INCLUDING the lunch break - what a naive resample counts. 390 today."""
    return int((afternoon_close(day) - MORNING_OPEN).total_seconds() // 60)


def lunch_break_minutes() -> int:
    """60. The hole between 11:30 and 12:30 that no bar may span."""
    return int((AFTERNOON_OPEN - MORNING_CLOSE).total_seconds() // 60)


def session_bar_count(freq_minutes: int, day: DateLike) -> int:
    """Intraday bars in one TSE session. 330 at 1 minute from 2024-11-05, 300 before.

    Raises when the frequency does not tile BOTH half-sessions. The morning is 150 minutes
    and the afternoon 180, so 1, 2, 3, 5, 6, 10, 15 and 30 tile and **60 does not**:
    This helper rejects partial hourly bars by design. A generic `resample('1h')` builds an 11:00-12:00
    bar that contains 30 minutes of trading and 30 minutes of nothing, and a 12:00-13:00
    bar that is half lunch. Both print a price and a volume.
    """
    if freq_minutes <= 0:
        raise ValueError("freq_minutes must be positive")
    am, pm = session_minutes(day)
    bad = [n for n, half in ((am, "morning"), (pm, "afternoon")) if n % freq_minutes]
    if bad:
        legal = sorted({m for m in range(1, min(am, pm) + 1)
                        if am % m == 0 and pm % m == 0})
        raise ValueError(
            f"{freq_minutes}-minute bars do not tile the TSE session on {_to_date(day)} "
            f"(morning {am} min, afternoon {pm} min). Such a bar straddles the 11:30-12:30 "
            f"break. Legal choices: {legal}."
        )
    return (am + pm) // freq_minutes


def naive_bar_count(freq_minutes: int, day: DateLike) -> int:
    """What `resample` on a wall-clock grid gives you: 390 at 1 minute. Always too many."""
    if freq_minutes <= 0:
        raise ValueError("freq_minutes must be positive")
    return wall_clock_minutes(day) // freq_minutes


def session_index(day: DateLike, freq_minutes: int = 1) -> pd.DatetimeIndex:
    """Bar-CLOSE timestamps for one session, with the 11:30-12:30 break absent.

    Bar-close labelling (09:01, ..., 11:30, 12:31, ..., 15:30) is the convention
    chosen for this synthetic demonstration, including the closing-auction window. Mixing it with a bar-open convention shifts every signal one bar in
    the look-ahead direction.
    """
    n = session_bar_count(freq_minutes, day)
    d = pd.Timestamp(_to_date(day))
    step = pd.Timedelta(minutes=freq_minutes)
    am = pd.date_range(d + MORNING_OPEN + step, d + MORNING_CLOSE, freq=step)
    pm = pd.date_range(d + AFTERNOON_OPEN + step, d + afternoon_close(day), freq=step)
    idx = am.append(pm)
    if len(idx) != n:
        raise AssertionError((len(idx), n))
    return idx


def wall_clock_index(day: DateLike, freq_minutes: int = 1) -> pd.DatetimeIndex:
    """The grid a naive `resample` produces: continuous from the open to the close."""
    d = pd.Timestamp(_to_date(day))
    step = pd.Timedelta(minutes=freq_minutes)
    return pd.date_range(d + MORNING_OPEN + step, d + afternoon_close(day), freq=step)


def is_break_spanning_bar(ts: pd.Timestamp, freq_minutes: int = 1) -> bool:
    """True for the first bar after lunch - it carries a 61-minute return, not a 1-minute one.

    This single bar is why "330 equal bars" is not quite right either. It closes at 12:31
    but its previous observation is the 11:30 print, so it absorbs the whole break.
    """
    t = pd.Timestamp(ts)
    first_pm = (t.normalize() + AFTERNOON_OPEN + pd.Timedelta(minutes=freq_minutes))
    return t == first_pm


# ------------------------------------------------------------------ settlement
def settlement_date(trade_date: DateLike, holidays: Iterable[DateLike] = ()) -> date:
    """T+2 from 2019-07-16, T+3 before it - in TSE business days, not calendar days.

    Keyed on the TRADE date. A backtest of a 2018 sample that assumes T+2 books cash one
    day early on every trade, which is invisible in a long-only equity P&L and material in
    anything financed.
    """
    d = _to_date(trade_date)
    n = 2 if d >= T2_SETTLEMENT_FROM else 3
    hol = {_to_date(h) for h in holidays}
    cur, moved = d, 0
    while moved < n:
        cur += timedelta(days=1)
        if cur.weekday() < 5 and cur not in hol:
            moved += 1
    return cur


# ------------------------------------------------------------------ ticks
def tick_size(price: float, topix500: bool = False) -> float:
    """TSE tick (yobine) for a price, from the tiered table.

    TOPIX500 constituents trade on a FINER table: 0.1 yen below 1,000 and 0.5 yen to
    3,000, against a flat 1 yen for everything else up to 3,000. Applying the ordinary
    table to a TOPIX500 name rounds a legal sub-yen improvement away; applying the fine
    table to an ordinary name sends a sub-tick price the exchange will reject.

    Two dates to key on: the fine table covered TOPIX100 only until 2023-06-05, when it
    was extended to TOPIX500 (TOPIX100 + TOPIX Mid400); and from 2027-03-01 the rule stops
    being index-membership-based and becomes liquidity-based (a Spread-to-Tick Ratio), so
    an index-membership lookup will silently stop being the right question.
    """
    if not (price > 0) or math.isinf(price):
        raise ValueError(f"price must be positive and finite, got {price}")
    table = TICK_TOPIX500 if topix500 else TICK_ORDINARY
    for bound, tick in table:
        if price <= bound:
            return tick
    raise AssertionError("unreachable: the last tick tier is unbounded")


def round_to_tick(price: float, topix500: bool = False, side: str = "nearest") -> float:
    """Round to a legal price. ROUND_HALF_UP, because a tick of 0.1 is not float-exact."""
    t = Decimal(str(tick_size(price, topix500)))
    p = Decimal(str(price))
    q = p / t
    s = side.strip().lower()
    if s == "nearest":
        q = q.quantize(Decimal(1), ROUND_HALF_UP)
    elif s == "down":
        q = Decimal(math.floor(q))
    elif s == "up":
        q = Decimal(math.ceil(q))
    else:
        raise ValueError("side must be 'nearest', 'down' or 'up'")
    return float(q * t)


# ------------------------------------------------------------------ short selling
def short_sale_price_restricted(base_price: float, last_price: float) -> bool:
    """Trigger test only: last price <= 90% of the exchange's daily BASE price.

    The base can differ from unadjusted previous close. Keep triggered state through the
    next trading day. Actual price restriction also depends on the previous price movement:
    a sale at the last price can be permitted on an uptick, but not on a downtick.
    """
    if not all(math.isfinite(p) and p > 0 for p in (base_price, last_price)):
        raise ValueError("prices must be positive and finite")
    return Decimal(str(last_price)) <= Decimal(str(base_price)) * Decimal("0.90")


# ------------------------------------------------------------------ index construction
def price_weighted_index(prices: Sequence[float] | np.ndarray,
                         divisor: float,
                         adjustments: Sequence[float] | np.ndarray | None = None) -> float:
    """Nikkei-style: sum(price x price-adjustment-factor) / divisor.

    `adjustments` is the per-constituent price adjustment factor (kakaku chosei keisu) the
    index sponsor applies so that a 40,000-yen name does not swamp the average. Pass None
    for the plain price average.
    """
    p = np.asarray(prices, dtype=float)
    if divisor <= 0:
        raise ValueError("divisor must be positive")
    if adjustments is not None:
        a = np.asarray(adjustments, dtype=float)
        if a.shape != p.shape:
            raise ValueError("adjustments must have the same shape as prices")
        p = p * a
    return float(p.sum() / divisor)


def price_weights(prices: Sequence[float] | np.ndarray,
                  adjustments: Sequence[float] | np.ndarray | None = None) -> np.ndarray:
    """Index weight of each constituent in a PRICE-weighted index: price / sum(prices)."""
    p = np.asarray(prices, dtype=float)
    if adjustments is not None:
        p = p * np.asarray(adjustments, dtype=float)
    return p / p.sum()


def cap_weighted_index(prices: Sequence[float] | np.ndarray,
                       shares: Sequence[float] | np.ndarray,
                       free_float: Sequence[float] | np.ndarray | None = None,
                       divisor: float = 1.0) -> float:
    """TOPIX-style: free-float adjusted market capitalisation over a base."""
    p = np.asarray(prices, dtype=float)
    s = np.asarray(shares, dtype=float)
    f = np.ones_like(p) if free_float is None else np.asarray(free_float, dtype=float)
    return float((p * s * f).sum() / divisor)


def cap_weights(prices: Sequence[float] | np.ndarray,
                shares: Sequence[float] | np.ndarray,
                free_float: Sequence[float] | np.ndarray | None = None) -> np.ndarray:
    p = np.asarray(prices, dtype=float)
    s = np.asarray(shares, dtype=float)
    f = np.ones_like(p) if free_float is None else np.asarray(free_float, dtype=float)
    mc = p * s * f
    return mc / mc.sum()


def new_divisor(prices_before: Sequence[float] | np.ndarray,
                prices_after: Sequence[float] | np.ndarray,
                divisor: float) -> float:
    """The divisor that keeps a price-weighted index CONTINUOUS across a corporate action.

    d_new = d_old x sum(after) / sum(before), using adjusted prices. This generic
    continuity example is not the complete Nikkei corporate-action procedure; an actual
    split can be handled through the published price adjustment factor. Miss it and the index falls by the
    entire lost price, which a return series reads as a real -x% day.
    """
    b = np.asarray(prices_before, dtype=float).sum()
    a = np.asarray(prices_after, dtype=float).sum()
    if b <= 0 or divisor <= 0:
        raise ValueError("prices and divisor must be positive")
    return float(divisor * a / b)


# ------------------------------------------------------------------ volatility
def annualised_vol(returns: Sequence[float] | np.ndarray, bars_per_year: float) -> float:
    """sd(returns) x sqrt(bars per year). The bar count appears here and in the sd."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 2:
        raise ValueError("need at least two returns")
    return float(r.std(ddof=1) * math.sqrt(bars_per_year))


def synthetic_tse_minutes(days: int = 60, seed: int = 20260910,
                          per_minute_vol: float = 0.0009,
                          lunch_gap_minutes_equivalent: float = 15.0,
                          start: str = "2025-01-06") -> pd.Series:
    """A seeded minute tape on the REAL 330-bar session grid, bar-close labelled.

    The lunch break is not a vacuum: prices reopen away from the 11:30 print because news
    keeps arriving. `lunch_gap_minutes_equivalent` is the variance the 12:31 bar carries,
    expressed in minutes of trading (default 15) - a stated modelling assumption, not a
    measurement. Everything downstream is measured against THIS tape.
    """
    rng = np.random.default_rng(seed)
    idx_parts, ret_parts = [], []
    d = pd.Timestamp(start)
    made = 0
    while made < days:
        if d.weekday() < 5:
            idx = session_index(d, 1)
            r = rng.normal(0.0, per_minute_vol, size=len(idx))
            # the first afternoon bar absorbs the break
            gap_pos = int(np.flatnonzero(
                [is_break_spanning_bar(t) for t in idx])[0])
            r[gap_pos] *= math.sqrt(lunch_gap_minutes_equivalent)
            idx_parts.append(idx)
            ret_parts.append(r)
            made += 1
        d += pd.Timedelta(days=1)
    index = idx_parts[0].append(idx_parts[1:]) if len(idx_parts) > 1 else idx_parts[0]
    returns = np.concatenate(ret_parts)
    return pd.Series(np.exp(np.cumsum(returns)) * 1000.0, index=index, name="price")


def wall_clock_resample(tape: pd.Series, freq_minutes: int = 1) -> pd.Series:
    """What `tape.resample('1min').last().ffill()` does: the break is filled with stale prints.

    Every one of those filled bars produces a return of exactly zero, so the sample
    standard deviation falls while the bar count rises.
    """
    out = []
    for day, chunk in tape.groupby(tape.index.normalize()):
        grid = wall_clock_index(day, freq_minutes)
        out.append(chunk.reindex(grid).ffill())
    return pd.concat(out)


def vol_comparison(tape: pd.Series, sessions_per_year: int = 245) -> dict[str, float]:
    """Three annualised vols off ONE tape: session-aware, wall-clock, and the mixed pair.

    * session_aware  - 330 real bars, annualised on 330 x sessions
    * wall_clock     - 390 ffilled bars (60 of them stale), annualised on 390 x sessions
    * mixed_high     - real bars, annualised on the WALL-CLOCK count (the common bug)
    * mixed_low      - ffilled bars, annualised on the SESSION count
    * ex_lunch_bar   - session-aware with the break-spanning 12:31 bar dropped
    """
    day0 = tape.index[0].date()
    n_session = session_bar_count(1, day0)
    n_wall = naive_bar_count(1, day0)

    real = np.log(tape).diff()
    # drop the first bar of each day: it is an overnight return, not an intraday one
    first_of_day = tape.groupby(tape.index.normalize()).head(1).index
    real = real.drop(index=first_of_day, errors="ignore").dropna()

    filled = np.log(wall_clock_resample(tape)).diff()
    ffod = wall_clock_resample(tape).groupby(
        lambda t: t.normalize()).head(1).index
    filled = filled.drop(index=ffod, errors="ignore").dropna()

    lunch_mask = np.array([is_break_spanning_bar(t) for t in real.index])
    return {
        "session_aware": annualised_vol(real.values, n_session * sessions_per_year),
        "wall_clock": annualised_vol(filled.values, n_wall * sessions_per_year),
        "mixed_high": annualised_vol(real.values, n_wall * sessions_per_year),
        "mixed_low": annualised_vol(filled.values, n_session * sessions_per_year),
        "ex_lunch_bar": annualised_vol(real.values[~lunch_mask],
                                       (n_session - 1) * sessions_per_year),
        "n_session_bars": float(n_session),
        "n_wall_bars": float(n_wall),
        "zero_return_bars": float(int((filled.values == 0.0).sum())),
    }


# ------------------------------------------------------------------ synthetic index panel
def synthetic_nikkei_panel(n: int = 225, seed: int = 20260910
                           ) -> tuple[np.ndarray, np.ndarray]:
    """(prices, shares outstanding) for a seeded 225-name panel.

    Japanese share prices span three orders of magnitude - a few names above 40,000 yen and
    a long tail near 1,000 - and price is close to UNCORRELATED with company size, because
    it is an artefact of how many times a company has split. That decoupling is exactly
    why a price-weighted index and a cap-weighted index disagree.
    """
    rng = np.random.default_rng(seed)
    prices = np.exp(rng.normal(math.log(2_400.0), 1.05, size=n))
    prices = np.clip(prices, 200.0, 60_000.0)
    # size drawn INDEPENDENTLY of price, then shares = cap / price
    caps = np.exp(rng.normal(math.log(1.2e12), 1.15, size=n))
    shares = caps / prices
    return np.round(prices, 1), shares


def _rule() -> str:
    return ("RULE: the TSE session is 330 minutes with a 60-minute hole, it was 300 before "
            "2024-11-05, and the Nikkei is price-weighted through a divisor.")


def main():
    print("Nominal session grids include the closing-auction window, not 330 executions.")
    for day in ("2024-11-01", "2024-11-05"):
        n, w = session_bar_count(1, day), naive_bar_count(1, day)
        print(f"{day}: {n} bars on nominal session grid; wall clock={w}; excess={w/n-1:+.2%}")
    print("Hourly bars need a partial-bar convention; this helper requires exact tiling.")
    result = vol_comparison(synthetic_tse_minutes())
    for label in ("session_aware", "wall_clock", "mixed_high", "mixed_low", "ex_lunch_bar"):
        value = result[label]
        print(f"{label}: {value:.2%}; relative={value/result['session_aware']-1:+.2%}")
    print(f"Inserted zero-return bars: {result['zero_return_bars']:.0f}")
    prices, shares = synthetic_nikkei_panel()
    adjusted = prices.copy()
    adjusted[prices.argmax()] /= 3
    divisor = new_divisor(prices, adjusted, 25.)
    print(f"Generic divisor-continuity example: 25.0000 -> {divisor:.4f}; "
          "actual Nikkei actions may change PAF instead.")
    print(_rule())


if __name__ == "__main__":
    main()
