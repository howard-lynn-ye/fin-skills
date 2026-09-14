"""Hong Kong lot sizing, dated sessions and quota arithmetic, with an offline demo.

Board lots constrain the automatic book; odd-lot holdings and separate trading exist.
Synthetic portfolio weights use total account NAV, including unspent cash. Source-era
lot frequencies are fixed scenario inputs, not a downloaded current security master.
"""
from __future__ import annotations

import math
from datetime import date, datetime, time, timedelta
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

DateLike = str | date | datetime | pd.Timestamp

# ------------------------------------------------------------------ rule effective dates
OPEN_MOVED_TO_0930_ON = date(2011, 3, 7)      # 10:00 -> 09:30
LUNCH_SHORTENED_ON = date(2012, 3, 5)         # the break became 12:00-13:00
SEVERE_WEATHER_TRADING_FROM = date(2024, 9, 23)   # typhoon/black rain no longer close HKEX
AGGREGATE_QUOTA_ABOLISHED_ON = date(2016, 8, 16)
DAILY_QUOTA_QUADRUPLED_ON = date(2018, 5, 1)  # NB 13 -> 52 bn RMB, SB 10.5 -> 42 bn RMB

# Stock Connect daily quotas in RMB, per direction per link. Selling is not blocked by
# the quota gate. Continuous-session exhaustion latches new buys closed for that day.
NORTHBOUND_DAILY_QUOTA_RMB = 52_000_000_000.0
SOUTHBOUND_DAILY_QUOTA_RMB = 42_000_000_000.0

# Recovered 2026-09-11 source-era snapshot, used ONLY as fixed scenario inputs.
# The workbook itself is not bundled, so these counts are not a current empirical claim.
HKEX_BOARD_LOT_VALUES: tuple[int, ...] = (
    10, 15, 20, 30, 40, 50, 60, 80, 100, 150, 200, 250, 300, 400, 500, 600, 660, 800,
    1_000, 1_250, 1_500, 1_800, 2_000, 2_500, 2_800, 3_000, 4_000, 5_000, 6_000, 6_400,
    7_000, 7_500, 8_000, 9_000, 10_000, 12_000, 15_000, 16_000, 18_000, 20_000, 24_000,
    25_000, 30_000, 40_000, 80_000, 100_000,
)
# Fixed scenario frequency weights from the recovered snapshot. These eight cover
# 2,406 of the 2,809 equities; the other 38 values share the remaining 403 and are not
# weighted here.
HKEX_BOARD_LOT_FREQUENCIES: tuple[tuple[int, int], ...] = (
    (2_000, 661), (1_000, 501), (500, 280), (10_000, 259),
    (100, 200), (5_000, 191), (4_000, 172), (200, 142),
)
HKEX_EQUITY_COUNT = 2_809
HKEX_DISTINCT_LOTS = 46

# Sessions (from 2012-03-05). Times are Hong Kong time.
# The 12:00-13:00 window is called the "Extended Morning Session", which sounds like extra
# trading and is not: only designated Extended Trading Securities trade in it. For an
# ordinary stock, treat it as closed.
PRE_OPENING = (time(9, 0), time(9, 30))       # POS
POS_SUBPERIODS: tuple[tuple[str, time, time], ...] = (
    ("order input", time(9, 0), time(9, 15)),
    ("no-cancellation", time(9, 15), time(9, 20)),
    ("random matching", time(9, 20), time(9, 22)),
    ("blocking", time(9, 22), time(9, 30)),
)
CONTINUOUS_AM = (time(9, 30), time(12, 0))
LUNCH_BREAK = (time(12, 0), time(13, 0))      # the "Extended Morning Session"
CONTINUOUS_PM = (time(13, 0), time(16, 0))
CLOSING_AUCTION = (time(16, 0), time(16, 10))  # CAS, random close between 16:08 and 16:10
CAS_SUBPERIODS: tuple[tuple[str, time, time], ...] = (
    ("reference price fixing", time(16, 0), time(16, 1)),
    ("order input, +/-5% of the reference price", time(16, 1), time(16, 6)),
    ("no-cancellation", time(16, 6), time(16, 8)),
    ("random closing", time(16, 8), time(16, 10)),
)

# Volatility Control Mechanism: a 5-minute cooling-off starts when a potential trade price
# deviates from the last automatched price FIVE MINUTES AGO by more than the tier's band.
# It is not a halt - trading continues inside a fixed band and aggressive orders outside it
# are rejected - and it does not run in the first 15 minutes of either continuous session
# or the last 20 minutes of the afternoon.
VCM_BAND_BY_TIER: dict[str, float] = {
    "HSCI LargeCap": 0.10,
    "HSCI MidCap": 0.15,
    "HSCI SmallCap": 0.20,
    "SPAC share": 0.30,          # 15% in the first month of listing
    "SPAC warrant": 0.50,        # 25% in the first month of listing
}
VCM_COOLING_OFF_MINUTES = 5

# Stamp duty per side, ad valorem on the consideration. Buyer AND seller each pay.
STAMP_DUTY_HISTORY: tuple[tuple[date, float], ...] = (
    (date(2001, 9, 1), 0.001),
    (date(2021, 8, 1), 0.0013),
    (date(2023, 11, 17), 0.001),
)


def _to_date(d: DateLike | None) -> date:
    if d is None:
        raise ValueError(
            "date is REQUIRED. The HK open moved in 2011, the lunch break shortened in "
            "2012, the Connect quota quadrupled in 2018 and typhoons stopped closing the "
            "market in 2024."
        )
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    return pd.Timestamp(d).date()


# ------------------------------------------------------------------ board lots
def round_to_lot(shares: float, board_lot: int, side: str = "down") -> int:
    """Round a share count to a whole number of board lots. Anything else is an odd lot.

    Defaults to rounding DOWN, because rounding up spends money you did not allocate and
    a live order manager that rounds up on 200 names quietly levers the book.
    """
    if board_lot <= 0:
        raise ValueError(f"board_lot must be positive, got {board_lot}")
    q = shares / board_lot
    s = side.strip().lower()
    if s == "down":
        n = math.floor(q + 1e-9)
    elif s == "up":
        n = math.ceil(q - 1e-9)
    elif s == "nearest":
        n = int(np.round(q))
    else:
        raise ValueError("side must be 'down', 'up' or 'nearest'")
    return int(max(n, 0) * board_lot)


def is_odd_lot(shares: float, board_lot: int) -> bool:
    """True when the quantity cannot be sent to the main continuous book.

    Odd lots may trade in a separate market. Their execution price/liquidity is not
    inferred here, and this helper does not forbid a strategy from using that market.
    """
    if board_lot <= 0:
        raise ValueError("board_lot must be positive")
    return shares % board_lot != 0


def min_ticket(price: float, board_lot: int) -> float:
    """price x board_lot: one board-lot ticket, excluding separate odd-lot trading."""
    if price <= 0 or board_lot <= 0:
        raise ValueError("price and board_lot must be positive")
    return float(price) * int(board_lot)


def synthetic_hk_universe(n: int = 60, seed: int = 20260910, coupling: float = 0.7
                          ) -> tuple[np.ndarray, np.ndarray]:
    """(prices in HKD, board lots) for a seeded universe.

    Fixed recovered snapshot weights determine the lot distribution; they are not a
    current downloaded security master. Negative price/lot rank coupling is a chosen
    modelling assumption, not a law about issuers. All measured results refer only to
    this synthetic universe, with no claim of representativeness.
    """
    if not 0.0 <= coupling <= 1.0:
        raise ValueError("coupling must be between 0 and 1")
    rng = np.random.default_rng(seed)
    prices = np.round(np.clip(np.exp(rng.normal(math.log(12.0), 1.35, size=n)),
                              0.20, 900.0), 2)
    values = np.array([v for v, _ in HKEX_BOARD_LOT_FREQUENCIES], dtype=int)
    counts = np.array([c for _, c in HKEX_BOARD_LOT_FREQUENCIES], dtype=float)
    lots = np.sort(rng.choice(values, size=n, p=counts / counts.sum()))[::-1]
    # rank the prices, add noise scaled by (1 - coupling), then hand the biggest lots to
    # the cheapest names
    score = np.argsort(np.argsort(prices)).astype(float)
    score = coupling * score + (1.0 - coupling) * rng.permutation(n).astype(float)
    return prices, lots[np.argsort(np.argsort(score))].astype(int)


def equal_weight_with_lots(capital: float, prices: Sequence[float] | np.ndarray,
                           board_lots: Sequence[int] | np.ndarray) -> dict[str, object]:
    """Build an equal-weight book under the lot constraint and measure what it cost you.

    The intent is capital/n in every name. The achievable position is a whole number of
    lots, so the realised weight is `round_down(capital/n / (price x lot)) x price x lot`
    over TOTAL account capital. Returns realised NAV weights, the per-name weight error
    against the 1/n target, the cash that could not be deployed, and the names that are
    UNFILLABLE because one lot already costs more than the target position.
    """
    p = np.asarray(prices, dtype=float)
    raw_lot = np.asarray(board_lots, dtype=float)
    if (p.ndim != 1 or p.size == 0 or not np.all(np.isfinite(p)) or np.any(p <= 0)
            or not np.all(np.isfinite(raw_lot)) or np.any(raw_lot <= 0)
            or np.any(raw_lot != np.floor(raw_lot))):
        raise ValueError("positive finite prices and integer board lots required")
    lot = raw_lot.astype(int)
    if p.shape != lot.shape:
        raise ValueError("prices and board_lots must have the same length")
    if not math.isfinite(capital) or capital <= 0:
        raise ValueError("capital must be positive")
    n = p.size
    target_value = capital / n
    tickets = p * lot
    n_lots = np.floor(target_value / tickets + 1e-9).astype(int)
    unfillable = n_lots == 0
    value = n_lots * tickets
    deployed = float(value.sum())
    target_w = np.full(n, 1.0 / n)
    realised_w = value / capital
    deployed_w = value / deployed if deployed > 0 else np.zeros(n)
    err = realised_w - target_w
    return {
        "target_value": target_value,
        "min_tickets": tickets,
        "n_lots": n_lots,
        "position_value": value,
        "deployed": deployed,
        "cash_drag": float(capital - deployed),
        "cash_drag_pct": float((capital - deployed) / capital),
        "realised_weight": realised_w,
        "deployed_weight": deployed_w,
        "weight_error": err,
        "max_abs_weight_error_bps": float(np.abs(err).max() * 1e4),
        "mean_abs_weight_error_bps": float(np.abs(err).mean() * 1e4),
        "unfillable": unfillable,
        "n_unfillable": int(unfillable.sum()),
    }


def naive_equal_weight(capital: float, prices: Sequence[float] | np.ndarray
                       ) -> dict[str, object]:
    """The same book with fractional shares - what an engine with no lot model produces."""
    p = np.asarray(prices, dtype=float)
    n = p.size
    value = np.full(n, capital / n)
    return {"shares": value / p, "position_value": value,
            "realised_weight": np.full(n, 1.0 / n), "deployed": float(capital)}


def capital_for_weight_error(prices: Sequence[float] | np.ndarray,
                             board_lots: Sequence[int] | np.ndarray,
                             target_bps: float = 100.0,
                             lo: float = 1e4, hi: float = 1e10) -> float:
    """Sufficient capital bound for NAV-weight error, not a claimed minimum.

    Floor rounding leaves strictly less than one ticket uninvested per name. Hence
    max(ticket)/capital bounds every NAV-weight error. Actual errors are sawtoothed;
    bisection on them is invalid because they need not be monotone.
    """
    if not math.isfinite(target_bps) or target_bps <= 0 or not 0 < lo <= hi:
        raise ValueError("positive target and ordered capital bounds required")
    audit = equal_weight_with_lots(lo, prices, board_lots)
    bound = max(lo, float(np.max(audit["min_tickets"])) * 1e4 / target_bps)
    return bound if bound <= hi else math.inf


# ------------------------------------------------------------------ settlement, sessions
def settlement_date(trade_date: DateLike, holidays: Iterable[DateLike] = ()) -> date:
    """T+2 in HKEX business days. Hong Kong holidays are NOT mainland holidays.

    That difference is the entire reason Stock Connect has its own trading calendar: a day
    that settles in Hong Kong may not settle in Shanghai, and Connect closes on any day
    where either leg cannot settle.
    """
    d = _to_date(trade_date)
    hol = {_to_date(h) for h in holidays}
    cur, moved = d, 0
    while moved < 2:
        cur += timedelta(days=1)
        if cur.weekday() < 5 and cur not in hol:
            moved += 1
    return cur


def session_minutes(day: DateLike) -> tuple[int, int]:
    """(morning, afternoon) continuous-trading minutes. (150, 180) from 2012-03-05."""
    d = _to_date(day)
    if d < LUNCH_SHORTENED_ON:
        raise ValueError(
            f"{d} predates the 2012-03-05 session change; the break was 12:00-13:30 then "
            f"and the open moved to 09:30 only on {OPEN_MOVED_TO_0930_ON}. Look the "
            f"session up for your sample rather than defaulting."
        )
    return 150, 180


def continuous_bar_count(freq_minutes: int, day: DateLike) -> int:
    """Continuous-session bars. 330 at 1 minute - the auctions are outside this."""
    am, pm = session_minutes(day)
    if freq_minutes <= 0 or am % freq_minutes or pm % freq_minutes:
        raise ValueError(
            f"{freq_minutes}-minute bars do not tile the HK session (morning {am}, "
            f"afternoon {pm}); such a bar spans the 12:00-13:00 break."
        )
    return (am + pm) // freq_minutes


def stamp_duty_rate(day: DateLike) -> float:
    """Ad valorem stamp duty PER SIDE. 0.1% -> 0.13% (2021-08-01) -> 0.1% (2023-11-17).

    Both the buyer and the seller pay, so the round trip is twice this. A backtest of a
    2020-2024 sample needs all three regimes, and ETFs are exempt - the exchange's own
    List of Securities carries a per-security stamp-duty flag, so it is not a blanket rate.
    """
    d = _to_date(day)
    rate = STAMP_DUTY_HISTORY[0][1]
    for start, r in STAMP_DUTY_HISTORY:
        if d >= start:
            rate = r
    return rate


def weather_arrangement(day: DateLike, signal: str) -> str:
    """What happens to trading under a Typhoon Signal 8+ or a Black Rainstorm warning.

    The rule REVERSED on 2024-09-23. Before it the market closed; after it the market stays
    open. A backtest that spans the change with one rule has invented sessions on one side
    of it and deleted them on the other - and it is not two regimes but three, because the
    afternoon resumption time moved from 13:30 to 13:00 on 2012-03-05.
    """
    d = _to_date(day)
    sig = signal.strip().lower()
    severe = any(k in sig for k in ("8", "9", "10", "black", "extreme"))
    if not severe:
        return "normal trading"
    if d >= SEVERE_WEATHER_TRADING_FROM:
        return ("trading CONTINUES - severe weather trading since 2024-09-23; "
                "clearing and settlement run as usual")
    resume = "13:00" if d >= LUNCH_SHORTENED_ON else "13:30"
    return (f"trading SUSPENDED; it resumes on the first half hour about two hours after "
            f"the signal is lowered, earliest {resume}, and NOT AT ALL if the signal is "
            f"lowered after 12:00")


# ------------------------------------------------------------------ Stock Connect
def daily_quota(direction: str, day: DateLike) -> float:
    """Net-BUY daily quota in RMB, keyed to the date it quadrupled (2018-05-01)."""
    d = _to_date(day)
    dr = direction.strip().lower()
    if dr.startswith("north"):
        return NORTHBOUND_DAILY_QUOTA_RMB if d >= DAILY_QUOTA_QUADRUPLED_ON else 13e9
    if dr.startswith("south"):
        return SOUTHBOUND_DAILY_QUOTA_RMB if d >= DAILY_QUOTA_QUADRUPLED_ON else 10.5e9
    raise ValueError("direction must be 'northbound' or 'southbound'")


def quota_balance(direction: str, day: DateLike, buy_rmb: float, sell_rmb: float,
                  adjustments_rmb: float = 0.) -> float:
    """Quota minus BUY ORDER value plus SELL TRADE value plus released adjustments.

    Restored balance does not reset a continuous-session exhaustion latch.
    """
    if any(not math.isfinite(v) or v < 0 for v in (buy_rmb, sell_rmb, adjustments_rmb)):
        raise ValueError("nonnegative finite order, trade and adjustment values required")
    return daily_quota(direction, day) - buy_rmb + sell_rmb + adjustments_rmb


def connect_buy_allowed(direction: str, day: DateLike, balance_rmb: float, *,
                        exhausted_in_continuous: bool = False) -> bool:
    """Quota gate only. Preserve exhaustion state from the start of continuous trading.

    A positive balance can reopen the opening-auction gate; exhaustion in continuous
    trading closes new buys for the day. Already accepted orders are unaffected.
    Other eligibility, calendar and broker checks remain separate.
    """
    daily_quota(direction, day)  # validate route and date even when blocked
    if not math.isfinite(balance_rmb):
        raise ValueError("finite quota balance required")
    return balance_rmb > 0 and not exhausted_in_continuous


def ah_premium(h_price_hkd: float, a_price_cny: float, hkd_per_cny: float) -> float:
    """A-share price over the H-share price in the SAME currency, minus one.

    Positive means the mainland line is dearer than the Hong Kong line - the usual state.
    The two lines are the same economic claim on the same company and they are not
    fungible: a Connect investor cannot convert one into the other, which is why the gap
    survives. Any pairs trade on it is a bet on the gap, not an arbitrage.
    """
    if min(h_price_hkd, a_price_cny, hkd_per_cny) <= 0:
        raise ValueError("prices and the fx rate must be positive")
    return (a_price_cny * hkd_per_cny) / h_price_hkd - 1.0


def adr_implied_local(adr_price_usd: float, ratio_shares_per_adr: float,
                      hkd_per_usd: float) -> float:
    """The HK-line price an ADR quote implies. `ratio` = ordinary shares per ADR.

    🚨 The ratio is the thing people get wrong, and it is not 1 for most Hong Kong names.
    Getting it wrong scales the whole series by that factor and every return is still
    correct, so the error survives every sanity check that looks at returns.
    """
    if ratio_shares_per_adr <= 0:
        raise ValueError("ratio must be positive")
    return adr_price_usd * hkd_per_usd / ratio_shares_per_adr


def _rule() -> str:
    return ("RULE: Hong Kong has no universal round lot; measure lot rounding against "
            "total account NAV and preserve the Connect exhaustion latch.")


def main():
    prices, lots = synthetic_hk_universe()
    print("Synthetic lot-size scenario; snapshot weights are not a live security master.")
    for capital in (1e6, 5e6, 2e7, 1e8, 1e9):
        a = equal_weight_with_lots(capital, prices, lots)
        print(f"capital={capital:.0f}: zero positions={a['n_unfillable']}/60; "
              f"NAV max error={a['max_abs_weight_error_bps']:.1f} bp; "
              f"cash={a['cash_drag_pct']:.2%}")
    print(f"Sufficient capital for <=100 bp NAV error: "
          f"{capital_for_weight_error(prices, lots):.0f} HKD")
    balance = quota_balance("northbound", "2026-09-14", 40e9, 35e9)
    print(f"Net order/trade quota example: {balance/1e9:.1f} bn RMB left")
    print("Restored balance after continuous exhaustion permits new buy:",
          connect_buy_allowed("northbound", "2026-09-14", balance,
                              exhausted_in_continuous=True))
    print(f"A/H same-currency premium: {ah_premium(40.,42.5,1.08):.2%}")
    print(f"ADR ratio 5 local HKD price: {adr_implied_local(21.4,5.,7.8):.3f}")
    print(_rule())


if __name__ == "__main__":
    main()
