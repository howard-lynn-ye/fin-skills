"""Offline Indian price-band, settlement and tax demonstrations.

The proposed synthetic open is a latent unconstrained price, not an observed exchange
print. Reject impossible simulated fills; a legal limit price still does not prove liquidity.
Session changes are date/eligibility keyed, and contract-size examples cover index products.
"""
from __future__ import annotations

import math
from datetime import date, datetime, time, timedelta
from decimal import ROUND_HALF_UP, Decimal
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

DateLike = str | date | datetime | pd.Timestamp

# ------------------------------------------------------------------ rule effective dates
T1_ROLLOUT_START = date(2022, 2, 25)     # first tranche: the smallest 100 scrips
T1_ROLLOUT_COMPLETE = date(2023, 1, 27)  # last tranche: the largest scrips
T0_BETA_FROM = date(2024, 3, 28)         # optional same-day settlement, 25 scrips
T0_TOP500_FROM = date(2025, 1, 31)       # widened to the top 500 by market cap
STT_RAISED_ON = date(2024, 10, 1)        # Finance Act 2024: F&O STT increase
STT_RAISED_AGAIN_ON = date(2026, 4, 1)   # Finance Act 2026: a SECOND F&O STT increase
FNO_CONTRACT_VALUE_RAISED_ON = date(2024, 11, 20)   # min contract value 5 -> 15 lakh
SLIDING_PRICE_BAND_FROM = date(2024, 11, 18)        # the dynamic band slides, not expands
CAS_FROM = date(2026, 8, 3)              # a closing AUCTION replaced the closing session
PRE_OPEN_RESTRUCTURED_ON = date(2026, 9, 7)         # order entry split into two windows

LAKH = 100_000.0

# Securities Transaction Tax, as a fraction of the taxable value. Sell-side unless noted.
# Three regimes, not two: the Finance Act 2024 raised the derivatives rates on 2024-10-01
# and the Finance Act 2026 raised them AGAIN on 2026-04-01. A cost model keyed to the
# 2024 numbers is already stale.
STT_RATES: dict[str, dict[str, float]] = {
    "before_2024_10_01": {
        "delivery_buy": 0.001, "delivery_sell": 0.001, "intraday_sell": 0.00025,
        "futures_sell": 0.000125, "options_sell_premium": 0.000625,
        "options_exercise_buy": 0.00125,
    },
    "from_2024_10_01": {
        "delivery_buy": 0.001, "delivery_sell": 0.001, "intraday_sell": 0.00025,
        "futures_sell": 0.0002, "options_sell_premium": 0.001,
        "options_exercise_buy": 0.00125,
    },
    "from_2026_04_01": {
        "delivery_buy": 0.001, "delivery_sell": 0.001, "intraday_sell": 0.00025,
        "futures_sell": 0.0005, "options_sell_premium": 0.0015,
        "options_exercise_buy": 0.0015,
    },
}

# Index circuit breaker. Per trigger, a tuple of (cutoff time, halt minutes) tried in
# order, then a default. `None` halt minutes means the rest of the day.
# The 10% and 15% rows have DIFFERENT second boundaries - 14:30 and 14:00 - which is the
# detail a simplified three-phase model gets wrong.
CIRCUIT_BREAKER_HALTS: dict[float, tuple[tuple[time, int | None], ...]] = {
    0.10: ((time(13, 0), 45), (time(14, 30), 15)),          # then: no halt
    0.15: ((time(13, 0), 105), (time(14, 0), 45)),          # then: rest of the day
    0.20: (),                                               # always: rest of the day
}
CIRCUIT_BREAKER_DEFAULT: dict[float, int | None] = {0.10: 0, 0.15: None, 0.20: None}
POST_HALT_PRE_OPEN_MINUTES = 15

# NSE cash-session templates; verify each venue's current eligible-security lists/notices.
PRE_OPEN = (time(9, 0), time(9, 15))
PRE_OPEN_SUBPERIODS: tuple[tuple[str, time, time], ...] = (
    ("order entry, limit and market", time(9, 0), time(9, 5)),
    ("order entry, limit only, random close in the last 2 min", time(9, 5), time(9, 10)),
    ("matching and trade confirmation", time(9, 10), time(9, 12)),
    ("buffer", time(9, 12), time(9, 15)),
)
CONTINUOUS = (time(9, 15), time(15, 30))      # no lunch break
CLOSING_AUCTION = (time(15, 15), time(15, 35))   # CAS, from 2026-08-03
CAS_BAND = 0.03                                  # +/-3% of the 15:00-15:15 VWAP
POST_CLOSE = (time(15, 50), time(16, 0))
DERIVATIVES_CLOSE = time(15, 40)
LEGACY_CLOSING_SESSION = (time(15, 40), time(16, 0))   # before 2026-08-03

CASH_BOARD_LOT = 1        # ordinary main-board equity example; SME/special lots differ
TICK = Decimal("0.05")    # chosen scenario tick; current tick comes from security metadata


def _to_date(d: DateLike | None) -> date:
    if d is None:
        raise ValueError(
            "date is REQUIRED. Settlement moved T+2 -> T+1 across 2022-2023, STT on "
            "derivatives rose on 2024-10-01 and the F&O contract size tripled on "
            "2024-11-20."
        )
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    return pd.Timestamp(d).date()


# ------------------------------------------------------------------ price bands
def band_prices(prev_close: float, band_pct: float, tick: float = .05) -> tuple[float, float]:
    """Scenario (lower, upper) rounded HALF UP to the supplied security tick.

    Rounded outward-consistent with the exchange convention rather than with Python's
    banker's rounding: `round(2.475, 2)` is 2.47, and a limit at the band edge that is one
    tick outside is rejected in full, not partially filled.
    """
    if not math.isfinite(prev_close) or prev_close <= 0:
        raise ValueError("prev_close must be positive and finite")
    if not math.isfinite(tick) or tick <= 0:
        raise ValueError("tick must be positive and finite")
    tick_decimal = Decimal(str(tick))
    if math.isinf(band_pct):
        return 0.0, math.inf
    if not (0 < band_pct < 1):
        raise ValueError(f"band_pct must be a fraction in (0,1), got {band_pct}")
    p = Decimal(str(prev_close))
    b = Decimal(str(band_pct))
    lo = float(((p * (Decimal(1) - b)) / tick_decimal).quantize(Decimal(1), ROUND_HALF_UP) * tick_decimal)
    hi = float(((p * (Decimal(1) + b)) / tick_decimal).quantize(Decimal(1), ROUND_HALF_UP) * tick_decimal)
    return lo, hi


def is_within_band(price: float, prev_close: float, band_pct: float) -> bool:
    lo, hi = band_prices(prev_close, band_pct)
    return lo <= price <= hi


def clip_to_band(price: float, prev_close: float, band_pct: float) -> float:
    """Project into a band for an audit; this is NOT permission to fill at the edge."""
    lo, hi = band_prices(prev_close, band_pct)
    return float(min(max(price, lo), hi))


def band_for(segment: str = "cash", band_pct: float | None = None) -> float:
    """Band for a security. F&O-segment scrips have no fixed band - they have a dynamic one.

    `segment='fno'` returns math.inf as the absence of this STATIC band only. This is
    never an unlimited-price fill permission: the current exchange operating range and
    updates are separate required inputs, even for a single instant.
    """
    s = segment.strip().lower()
    if s in ("fno", "f&o", "derivatives"):
        return math.inf
    if band_pct is None:
        raise ValueError(
            "a cash-segment scrip needs its OWN band (2%, 5%, 10% or 20%); there is no "
            "default. The band is published per security and it is changed by the "
            "exchange, so it must come from the daily band file, not from a constant."
        )
    return band_pct


def dynamic_band_after_flexing(base_pct: float, flexes: int, step_pct: float = 0.05,
                               cooling_off_minutes: int = 15) -> dict[str, float]:
    """An F&O scrip's reachable band after `flexes` flexes, and the TIME they take.

    The band starts at 10% and is flexed in 5-point steps once the last trade reaches
    9.90% (then 14.90%, and so on) AND objective participation criteria are met. Each flex
    costs a cooling-off period, so the reachable move is a function of how much of the
    session is left - which is why "F&O scrips have no price band" is the wrong summary.

    Since 2024-11-18 the band SLIDES rather than expands: when it is flexed upward the
    lower edge moves up by the same amount, and pending orders outside the new band are
    CANCELLED by the exchange. A resting bid two flexes below the market is not sitting
    there waiting; it is gone.
    """
    if flexes < 0:
        raise ValueError("flexes must be non-negative")
    shift = flexes * step_pct
    return {
        "upper_edge_pct": base_pct + shift,
        "lower_edge_pct": -base_pct + shift,
        "band_width_pct": 2 * base_pct,      # the WIDTH never changes; the band slides
        "minutes_required": float(flexes * cooling_off_minutes),
    }


# ------------------------------------------------------------------ circuit breakers
def _as_time(when: str | time) -> time:
    return when if isinstance(when, time) else datetime.strptime(str(when), "%H:%M").time()


def index_circuit_breaker(move: float, when: str | time
                          ) -> tuple[str | None, int | None, bool]:
    """(trigger label, halt minutes, pre-open call auction afterwards).

    `move` is the index return from the PREVIOUS CLOSE (signed; the breaker is symmetric).
    A halt of None means the market is closed for the rest of the day. The halt length
    depends on the time of day, and the 10% and 15% rows use DIFFERENT afternoon
    boundaries - 14:30 and 14:00 - so a shared three-phase model is wrong between them.
    The breaker fires on whichever of the Sensex or the Nifty 50 is breached FIRST, so a
    Nifty-only backtest can miss a halt entirely.
    """
    m = abs(float(move))
    t = _as_time(when)
    for trig in (0.20, 0.15, 0.10):
        if m >= trig:
            halt = CIRCUIT_BREAKER_DEFAULT[trig]
            for cutoff, mins in CIRCUIT_BREAKER_HALTS[trig]:
                if t < cutoff:
                    halt = mins
                    break
            if halt == 0:
                return f"{trig:.0%}", 0, False
            return f"{trig:.0%}", halt, halt is not None
    return None, 0, False


def halted_minutes_in_day(move: float, when: str | time) -> int:
    """Total minutes of the 375-minute session lost to a breach, including the pre-open."""
    label, halt, pre = index_circuit_breaker(move, when)
    if label is None:
        return 0
    if halt is None:
        t = _as_time(when)
        return max(0, int((datetime.combine(date(2000, 1, 1), CONTINUOUS[1])
                           - datetime.combine(date(2000, 1, 1), t)).total_seconds() // 60))
    return halt + (POST_HALT_PRE_OPEN_MINUTES if pre else 0)


# ------------------------------------------------------------------ settlement and tax
def settlement_lag(trade_date: DateLike, tranche_date: DateLike | None = None) -> int:
    """T+1 or T+2 in business days, keyed to the security's OWN tranche date.

    The rollout was per-scrip, not market-wide: from 2022-02-25 the smallest 100 scrips
    moved, then a tranche a month by descending market cap until 2023-01-27. Between those
    dates the market ran BOTH cycles at once, so a portfolio-level "T+1 from 2022" is
    wrong for the large caps and a "T+2 until 2023" is wrong for the small ones.
    """
    d = _to_date(trade_date)
    if tranche_date is None:
        if d >= T1_ROLLOUT_COMPLETE:
            return 1
        if d < T1_ROLLOUT_START:
            return 2
        raise ValueError(
            f"{d} is inside the 2022-02-25 to 2023-01-27 T+1 rollout, when both cycles ran "
            f"side by side. Pass the security's own tranche_date; there is no market-wide "
            f"answer for this date."
        )
    return 1 if d >= _to_date(tranche_date) else 2


def settlement_date(trade_date: DateLike, tranche_date: DateLike | None = None,
                    holidays: Iterable[DateLike] = ()) -> date:
    n = settlement_lag(trade_date, tranche_date)
    hol = {_to_date(h) for h in holidays}
    cur, moved = _to_date(trade_date), 0
    while moved < n:
        cur += timedelta(days=1)
        if cur.weekday() < 5 and cur not in hol:
            moved += 1
    return cur


def stt_cost(notional: float, kind: str, when: DateLike) -> float:
    """Securities Transaction Tax on one leg, date-keyed to the 2024-10-01 increase.

    `kind` is one of the keys in STT_RATES. For options the notional is the PREMIUM, not
    the strike times the lot - which is the single most common way an Indian options cost
    model comes out 50x too large.
    """
    d = _to_date(when)
    if d >= STT_RAISED_AGAIN_ON:
        table = STT_RATES["from_2026_04_01"]
    elif d >= STT_RAISED_ON:
        table = STT_RATES["from_2024_10_01"]
    else:
        table = STT_RATES["before_2024_10_01"]
    if kind not in table:
        raise ValueError(f"unknown STT leg {kind!r}; known: {sorted(table)}")
    if notional < 0:
        raise ValueError("notional must be non-negative")
    return float(notional) * table[kind]


def round_trip_stt_bps(notional: float, kind_buy: str, kind_sell: str,
                       when: DateLike) -> float:
    """Round-trip STT in basis points of the buy notional."""
    total = stt_cost(notional, kind_buy, when) + stt_cost(notional, kind_sell, when)
    return 1e4 * total / notional


# ------------------------------------------------------------------ F&O contract size
def min_contract_value(when: DateLike) -> float:
    """Minimum notional for newly introduced INDEX derivatives in the model regime."""
    return (15.0 if _to_date(when) >= FNO_CONTRACT_VALUE_RAISED_ON else 5.0) * LAKH


def fno_lot_size(price: float, when: DateLike, round_to: int = 5) -> int:
    """Theoretical INDEX lot satisfying the lower notional bound; not a contract master.

    The input date is the regime chosen for contract introduction, not the trade date of
    an existing contract. Actual exchange lots depend on review prices, the upper bound,
    expiry and corporate actions. Do not apply the index revision to stock derivatives.
    """
    if not math.isfinite(price) or price <= 0 or not isinstance(round_to, int) or round_to < 1:
        raise ValueError("positive finite price and integer round_to required")
    need = min_contract_value(when) / price
    return max(10, int(math.ceil(need / round_to) * round_to))


def cash_session(when: DateLike, cas_eligible: bool = False) -> dict:
    """Regular-session template; exchange holidays and special sessions are caller inputs."""
    d = _to_date(when)
    cas = d >= CAS_FROM and cas_eligible
    return {"continuous": (time(9, 15), time(15, 15) if cas else time(15, 30)),
            "closing_auction": CLOSING_AUCTION if cas else None}


# ------------------------------------------------------------------ the measurement
def synthetic_daily(n_days: int = 2_500, seed: int = 20260910,
                    gap_vol: float = 0.009, intraday_vol: float = 0.013,
                    jump_prob: float = 0.045, jump_vol: float = 0.075
                    ) -> pd.DataFrame:
    """A seeded daily frame whose OVERNIGHT and INTRADAY moves are INDEPENDENT.

    Independence is deliberate: it means a gap strategy has no edge in this world, so any
    P&L the naive fill model reports was manufactured by the fill, not earned by the
    signal. Jumps are a Bernoulli mixture living in the overnight component, which is what
    makes a band bind at all - a pure lognormal almost never travels 10% overnight.
    """
    rng = np.random.default_rng(seed)
    gap = rng.normal(0.0, gap_vol, n_days)
    gap = gap + rng.normal(0.0, jump_vol, n_days) * (rng.random(n_days) < jump_prob)
    intraday = rng.normal(0.0, intraday_vol, n_days)
    prev_close = np.empty(n_days)
    open_ = np.empty(n_days)
    close = np.empty(n_days)
    px = 500.0
    for i in range(n_days):
        prev_close[i] = px
        open_[i] = px * math.exp(gap[i])
        close[i] = open_[i] * math.exp(intraday[i])
        px = close[i]
    idx = pd.bdate_range("2021-01-04", periods=n_days)
    return pd.DataFrame({"prev_close": prev_close, "open": open_, "close": close},
                        index=idx)


def band_fill_audit(df: pd.DataFrame, band_pct: float) -> dict[str, float]:
    """How often does "fill me at tomorrow's open" ask for a price the band forbids?

    Returns the breach rate, the mean and worst overshoot, and what the overshoot is worth
    per round trip - the P&L a naive engine books on prices that never existed.
    """
    prev = df["prev_close"].to_numpy()
    op = df["open"].to_numpy()
    lo = np.empty_like(prev)
    hi = np.empty_like(prev)
    for i, p in enumerate(prev):
        lo[i], hi[i] = band_prices(p, band_pct)
    outside = (op < lo) | (op > hi)
    reachable = np.clip(op, lo, hi)
    overshoot = np.where(outside, np.abs(op - reachable) / reachable, 0.0)
    return {
        "band_pct": band_pct,
        "n_days": float(len(df)),
        "n_outside": float(int(outside.sum())),
        "breach_rate": float(outside.mean()),
        "mean_overshoot_bps": float(overshoot[outside].mean() * 1e4) if outside.any() else 0.0,
        "max_overshoot_bps": float(overshoot.max() * 1e4),
        "phantom_pnl_bps_per_day": float(overshoot.mean() * 1e4),
    }


def phantom_fill_audit(df: pd.DataFrame, band_pct: float,
                       trigger: float = -0.03) -> dict[str, float]:
    """Buy-the-gap signals filled at tomorrow's open: how much of that price never existed.

    The signal: the stock gapped DOWN by more than `trigger`, so buy at the open. The
    naive engine uses the printed open. If the open is below the band floor, no such price
    could be quoted - the exchange rejects the order before it reaches the book - so the
    engine has bought at a discount to the cheapest price the market was capable of
    producing. That discount is booked as instant profit and is measured here in bps.

    Deliberately a tail strategy, because the band is only breached on a big move: the
    trades the band kills are never a random sample of the trades.
    """
    prev = df["prev_close"].to_numpy()
    op = df["open"].to_numpy()
    sig = (op / prev - 1.0) < trigger
    lo = np.empty_like(prev)
    hi = np.empty_like(prev)
    for i, p in enumerate(prev):
        lo[i], hi[i] = band_prices(p, band_pct)
    outside = sig & (op < lo)
    phantom = np.zeros_like(prev)
    phantom[outside] = (lo[outside] - op[outside]) / lo[outside]
    n_sig = int(sig.sum())
    return {
        "band_pct": band_pct,
        "n_signals": float(n_sig),
        "n_outside": float(int(outside.sum())),
        "outside_share": float(outside.sum() / max(n_sig, 1)),
        "mean_phantom_bps": (float(phantom[outside].mean() * 1e4)
                             if outside.any() else 0.0),
        "max_phantom_bps": float(phantom.max() * 1e4),
        "phantom_bps_per_signal": (float(phantom[sig].mean() * 1e4)
                                   if n_sig else 0.0),
        "total_phantom_pct": float(phantom[sig].sum() * 100),
        # the same audit on the sell side: a gap UP filled above the ceiling
        "n_ceiling_breaches": float(int((((op / prev - 1.0) > -trigger)
                                         & (op > hi)).sum())),
    }


def _rule() -> str:
    return "RULE: an Indian fill outside the security's price band is not a bad fill; reject it."


def main():
    frame = synthetic_daily()
    print("Synthetic unconstrained candidate prices; not an observed NSE tape or trading P&L.")
    for band in (.02, .05, .10, .20):
        audit, selected = band_fill_audit(frame, band), phantom_fill_audit(frame, band)
        print(f"band={band:.0%}: outside={int(audit['n_outside'])}/{len(frame)} "
              f"({audit['breach_rate']:.2%}); selected outside="
              f"{int(selected['n_outside'])}/{int(selected['n_signals'])} "
              f"({selected['outside_share']:.2%})")
    ratio = (phantom_fill_audit(frame,.05)['outside_share'] /
             band_fill_audit(frame,.05)['breach_rate'])
    print(f"Selected breach rate is {ratio:.0f}x the unconditional rate.")
    print("14:15 index triggers:", index_circuit_breaker(-.10,"14:15"),
          index_circuit_breaker(-.155,"14:15"), "(None means closed for the day)")
    print(f"Option sale STT on 12000 premium: "
          f"{stt_cost(12000.,'options_sell_premium','2026-09-14'):.2f}; "
          "using 1500000 strike notional is 125x too big.")
    print("New index-contract minimum:", min_contract_value("2024-11-19"), "->",
          min_contract_value("2024-11-20"))
    print("CAS-eligible current session:", cash_session("2026-09-14",True))
    print(_rule())


if __name__ == "__main__":
    main()
