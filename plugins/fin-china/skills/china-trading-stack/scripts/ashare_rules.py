"""A-share market rules a Western backtest engine silently gets wrong, as executable code.

WHY this exists — every one of these is a rule that has no analogue in US equities, so no
US-built engine (backtrader, vectorbt, zipline, bt, most home-grown loops) models it. The
backtest fills anyway, and the resulting equity curve is fiction:

  1. **Daily price limits (涨跌停)**. Main board +/-10%, ST/*ST +/-5%, ChiNext (300/301) and
     STAR (688) +/-20%, BSE (8/43/87/92) +/-30%. A bar locked at the up limit
     (high == low == limit price) has a queue of buyers and no sellers: you CANNOT buy it.
     A naive engine fills you at the limit price and books the next day's gap as profit.
  2. **The ChiNext limit is date-dependent.** ChiNext was +/-10% until the registration
     reform of **2020-08-24** and +/-20% from that day. A backtest spanning 2019-2021 with
     a hardcoded 20% is wrong for the first half of its own sample.
  3. **T+1 settlement.** Shares bought today cannot be sold today. Intraday round trips on
     the same position are impossible for stocks (the market ticks all day; your inventory
     does not). Any engine that lets a signal buy at 10:00 and sell at 14:00 is trading a
     market that does not exist. (Note the asymmetry: cash from a sale IS reusable the same
     day, and ETFs/T+0 instruments differ.)
  4. **Stamp duty is SELL-SIDE ONLY** (印花税), and its rate **halved from 0.10% to 0.05%
     on 2023-08-28**. Symmetric cost models overstate the buy leg, understate the change,
     and misprice every historical turnover study.
  5. **The lunch break.** 09:30-11:30 and 13:00-15:00 = **240 one-minute bars**, not 390
     and not 330. Resampling with a US session assumption invents ~90 empty bars a day and
     every rolling window silently changes length.

Usage:
    from ashare_rules import daily_limit_pct, can_buy, sellable_qty, round_trip_cost

    pct = daily_limit_pct("300750", "宁德时代", "2021-06-01")     # 0.20
    ok  = can_buy({"prev_close": 10.0, "open": 11.0, "high": 11.0,
                   "low": 11.0, "close": 11.0, "volume": 1_000}, pct)   # False, limit-up
"""
from __future__ import annotations

import math
import re
from datetime import date, datetime
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

DateLike = str | date | datetime | pd.Timestamp

# --------------------------------------------------------------------- rule effective dates
CHINEXT_20PCT_FROM = date(2020, 8, 24)    # 创业板注册制改革: 10% -> 20%
STAR_20PCT_FROM = date(2019, 7, 22)       # 科创板开市, 20% from day one
BSE_30PCT_FROM = date(2021, 11, 15)       # 北交所开市 (精选层 already 30%)
STAMP_DUTY_HALVED_FROM = date(2023, 8, 28)   # 0.10% -> 0.05%, sell side only
TRANSFER_FEE_HALVED_FROM = date(2022, 4, 29)  # 过户费 0.002% -> 0.001%, both sides

TICK = Decimal("0.01")
_ST_RE = re.compile(r"^S?\*?ST", re.IGNORECASE)


def _to_date(d: DateLike | None) -> date:
    if d is None:
        raise ValueError(
            "date is REQUIRED. Every limit and every fee rate in this module has changed "
            "at least once; a rule without a date is a rule for an arbitrary year."
        )
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    return pd.Timestamp(d).date()


def normalise_code(code: str) -> str:
    """'sh600000', '600000.SH', ' 600000 ' -> '600000'. Raises on anything else."""
    c = str(code).strip().upper()
    c = re.sub(r"\.(SH|SS|SZ|BJ|XSHG|XSHE|BSE)$", "", c)
    c = re.sub(r"^(SH|SZ|BJ)", "", c)
    if not re.fullmatch(r"\d{6}", c):
        raise ValueError(
            f"unrecognised A-share code {code!r}; expected 6 digits with an optional "
            f"exchange prefix/suffix. A 5-digit code is Hong Kong, not the mainland."
        )
    return c


def board_of(code: str) -> str:
    """Board name — the thing that actually determines the limit, tick and lot rules."""
    c = normalise_code(code)
    if c.startswith(("688", "689")):
        return "STAR"                      # 科创板 (689 = CDR)
    if c.startswith(("300", "301", "302")):
        return "ChiNext"                   # 创业板
    if c.startswith(("4", "8", "92")):
        return "BSE"                       # 北交所 43/83/87/88/92
    if c.startswith(("900", "200")):
        return "B-share"                   # B股, +/-10%
    if c.startswith(("600", "601", "603", "605")):
        return "SSE-Main"
    if c.startswith(("000", "001", "002", "003")):
        return "SZSE-Main"                 # 002/003 = old SME board, merged 2021-04-06
    raise ValueError(f"cannot classify board for code {c!r}")


def is_st(name: str | None) -> bool:
    """ST / *ST / SST / S*ST, matched as a PREFIX.

    Prefix-anchored on purpose: a substring search for 'ST' also matches names that merely
    contain those letters, and mislabelling a normal stock as ST halves its limit and
    quietly rejects legitimate fills.
    """
    if not name:
        return False
    s = str(name).replace(" ", "").replace("　", "").replace("＊", "*")
    return bool(_ST_RE.match(s))


def daily_limit_pct(code: str, name: str | None = None, date: DateLike | None = None,
                    days_since_ipo: int | None = None) -> float:
    """Daily price-limit as a FRACTION of the previous close (0.10 == +/-10%).

    Date-aware because the rules moved:
      * ChiNext 300/301: 10% before 2020-08-24, 20% from that date.
      * STAR 688/689: 20% since launch.
      * BSE 4/8/92: 30%.
      * Main board & B-shares: 10%, or 5% when the name carries an ST marker.

    ST only halves the limit on the MAIN BOARD. ChiNext and STAR keep their 20% band for
    ST-designated names — a detail that costs you every ST fill on those boards if you
    apply the 5% rule everywhere.

    `days_since_ipo` (trading days, 0 = listing day) applies the no-price-limit window for
    the registration-based boards: the first 5 trading days have NO limit and return
    `math.inf`. The main board's own IPO-day band (+44%/-36% historically, and the 2023
    registration reform's own window) is deliberately NOT modelled here — verify it against
    the exchange rulebook for your sample rather than trusting a default.
    """
    d = _to_date(date)
    board = board_of(code)

    if days_since_ipo is not None and days_since_ipo < 5 and board in (
            "STAR", "ChiNext", "BSE"):
        return math.inf                    # 上市前5个交易日不设涨跌幅限制

    if board == "STAR":
        if d < STAR_20PCT_FROM:
            # A 688xxx bar dated before the board opened is bad data, not a 10% stock.
            raise ValueError(f"STAR board did not exist on {d} (opened "
                             f"{STAR_20PCT_FROM}); check the date on this bar")
        return 0.20
    if board == "ChiNext":
        # THE date-dependent one. 2020-08-24 is the registration-reform switch.
        return 0.20 if d >= CHINEXT_20PCT_FROM else 0.10
    if board == "BSE":
        # 30% both before and after BSE_30PCT_FROM: these codes were NEEQ 精选层 first,
        # which already ran a 30% band, so there is no earlier regime to switch on.
        return 0.30
    return 0.05 if is_st(name) else 0.10


def limit_price(prev_close: float, limit_pct: float, side: str = "up") -> float:
    """Limit price rounded to the 0.01 tick with ROUND_HALF_UP (the exchange's rule).

    Python's `round()` is banker's rounding: round(10.555, 2) == 10.55, while the exchange
    publishes 10.56. One tick sounds harmless until your `high == limit_price` test fails
    on exactly the bars you were trying to detect.
    """
    if prev_close <= 0:
        raise ValueError(f"prev_close must be positive, got {prev_close}")
    if math.isinf(limit_pct):
        return math.inf if side == "up" else 0.0
    s = side.strip().lower()
    if s not in ("up", "down"):
        raise ValueError("side must be 'up' or 'down'")
    factor = Decimal(1) + (Decimal(str(limit_pct)) if s == "up"
                           else -Decimal(str(limit_pct)))
    return float((Decimal(str(prev_close)) * factor).quantize(TICK, ROUND_HALF_UP))


def _bar_get(bar: Mapping[str, Any], *names: str) -> Any:
    for n in names:
        if n in bar:
            return bar[n]
    return None


def explain_buy(bar: Mapping[str, Any], limit_pct: float) -> tuple[bool, str]:
    """(can_buy, reason). The reason is what you log when a signal does not become a fill."""
    prev_close = _bar_get(bar, "prev_close", "pre_close", "preclose", "prev_close_price")
    if prev_close is None:
        raise KeyError(
            "bar needs 'prev_close': the limit is a function of the PREVIOUS close, not "
            "of today's open. Deriving it from today's open is the second-most-common "
            "A-share backtest bug."
        )

    paused = _bar_get(bar, "paused", "is_paused", "suspended", "halted")
    if paused:
        return False, "stock is suspended (停牌); there is no market to trade"
    status = _bar_get(bar, "trade_status", "trading_status")
    if status is not None and str(status).strip() in ("停牌", "SUSPEND", "SUSPENDED", "0"):
        return False, f"trade_status={status!r} indicates a suspension"

    vol = _bar_get(bar, "volume", "vol", "turnover_volume")
    if vol is None or (isinstance(vol, float) and math.isnan(vol)) or float(vol) <= 0:
        return False, ("volume is 0 -- a zero-volume bar is a suspension or a placeholder "
                       "row, and a fill against it is invented liquidity")

    high, low = float(bar["high"]), float(bar["low"])
    up = limit_price(float(prev_close), limit_pct, "up")
    if math.isfinite(up) and high == low and abs(high - up) < 0.005:
        return False, (f"bar is LOCKED at the up limit ({up:.2f}): high == low == limit, "
                       f"so the book is all bids and no offers -- you cannot buy")
    return True, "tradable"


def can_buy(bar: Mapping[str, Any], limit_pct: float) -> bool:
    """False when volume is 0, the bar is suspended, or it is locked at the UP limit.

    Asymmetric on purpose: a bar locked at the DOWN limit is buyable (sellers are queued;
    you are the liquidity they want) — it is SELLING that is impossible there. `can_sell`
    is the mirror image.
    """
    return explain_buy(bar, limit_pct)[0]


def can_sell(bar: Mapping[str, Any], limit_pct: float) -> bool:
    """Mirror of `can_buy`: a bar locked at the DOWN limit cannot be sold into."""
    prev_close = _bar_get(bar, "prev_close", "pre_close", "preclose")
    if prev_close is None:
        raise KeyError("bar needs 'prev_close' to derive the limit price")
    if _bar_get(bar, "paused", "is_paused", "suspended", "halted"):
        return False
    vol = _bar_get(bar, "volume", "vol")
    if vol is None or (isinstance(vol, float) and math.isnan(vol)) or float(vol) <= 0:
        return False
    high, low = float(bar["high"]), float(bar["low"])
    down = limit_price(float(prev_close), limit_pct, "down")
    return not (high == low and abs(low - down) < 0.005)


def sellable_qty(lots: Iterable[Mapping[str, Any] | Sequence[Any]],
                 today: DateLike) -> int:
    """T+1: only shares bought on a STRICTLY EARLIER trading day are sellable today.

    `lots` are open buy lots — mappings with 'qty' and 'date', or (date, qty) pairs.

    This is the rule that deletes most intraday A-share "strategies": the signal that buys
    the morning dip and sells the afternoon pop cannot be executed at all, and a backtest
    that executes it is measuring a market that has never existed.
    """
    t = _to_date(today)
    total = 0
    for lot in lots:
        if isinstance(lot, Mapping):
            qty = lot.get("qty", lot.get("quantity", lot.get("shares")))
            when = lot.get("date", lot.get("trade_date", lot.get("buy_date")))
        else:
            when, qty = lot[0], lot[1]
        if qty is None or when is None:
            raise ValueError(f"lot {lot!r} needs a quantity and a buy date")
        if _to_date(when) < t:
            total += int(qty)
    return total


def round_trip_cost(buy_price: float, sell_price: float, qty: int,
                    buy_date: DateLike, sell_date: DateLike | None = None,
                    commission_rate: float = 0.00025, min_commission: float = 5.0
                    ) -> dict[str, float]:
    """Full round-trip cost with SELL-SIDE-ONLY stamp duty and date-keyed rates.

    Line items:
      * 佣金 commission — both sides, broker-set (retail ~0.025%, exchange cap 0.3%), with
        a **RMB 5 minimum per side** that dominates small tickets. The exchange handling
        and regulatory fees (经手费, 证管费) are collected inside this and are not itemised
        separately here to avoid double counting.
      * 印花税 stamp duty — **SELL SIDE ONLY**. 0.10% until 2023-08-27, **0.05% from
        2023-08-28**. Keyed on the SELL date, because that is when it is levied.
      * 过户费 transfer fee — both sides, 0.002% until 2022-04-28, 0.001% from 2022-04-29.

    The asymmetry is the point: a symmetric bps model prices the buy leg too high and the
    sell leg too low, so it cannot rank a high-turnover strategy correctly and it cannot
    see the 2023 stamp-duty cut at all.
    """
    b = _to_date(buy_date)
    s = _to_date(sell_date) if sell_date is not None else b
    if s < b:
        raise ValueError(f"sell_date {s} precedes buy_date {b}")
    if qty <= 0:
        raise ValueError("qty must be positive")
    if qty % 100 != 0:
        # Buys must be in 100-share lots (sells may be odd, e.g. after a rights issue).
        raise ValueError(f"qty={qty} is not a multiple of 100; A-share BUY orders are in "
                         f"round lots (手) of 100 shares")

    buy_notional = buy_price * qty
    sell_notional = sell_price * qty
    stamp_rate = 0.0005 if s >= STAMP_DUTY_HALVED_FROM else 0.0010
    buy_transfer_rate = 0.00001 if b >= TRANSFER_FEE_HALVED_FROM else 0.00002
    sell_transfer_rate = 0.00001 if s >= TRANSFER_FEE_HALVED_FROM else 0.00002

    buy_comm = max(buy_notional * commission_rate, min_commission)
    sell_comm = max(sell_notional * commission_rate, min_commission)
    buy_transfer = buy_notional * buy_transfer_rate
    sell_transfer = sell_notional * sell_transfer_rate
    sell_stamp = sell_notional * stamp_rate          # buy side pays ZERO stamp duty

    buy_total = buy_comm + buy_transfer
    sell_total = sell_comm + sell_transfer + sell_stamp
    total = buy_total + sell_total
    return {
        "buy_commission": buy_comm, "buy_transfer": buy_transfer, "buy_stamp": 0.0,
        "buy_total": buy_total,
        "sell_commission": sell_comm, "sell_transfer": sell_transfer,
        "sell_stamp": sell_stamp, "sell_total": sell_total,
        "round_trip": total,
        "round_trip_bps": 1e4 * total / buy_notional,
        "stamp_rate_used": stamp_rate,
        "gross_pnl": sell_notional - buy_notional,
        "net_pnl": sell_notional - buy_notional - total,
    }


def session_bar_count(freq_minutes: int = 1) -> int:
    """Intraday bars in one A-share session. 240 at 1-minute, lunch break EXCLUDED.

    09:30-11:30 (120 min) + 13:00-15:00 (120 min) = 240 minutes of continuous auction.
    The 09:15-09:25 opening call auction is outside this; the 14:57-15:00 closing call
    auction on Shenzhen falls inside the final bar.

    Raises for a frequency that does not divide 120: a 45-minute bar cannot exist here,
    because it would have to straddle the lunch break. Generic resamplers produce one
    anyway — that bar mixes 11:30 and 13:00 prices and its volume is nonsense.
    """
    if freq_minutes <= 0:
        raise ValueError("freq_minutes must be positive")
    if 120 % freq_minutes != 0:
        raise ValueError(
            f"{freq_minutes}-minute bars do not tile an A-share half-session (120 min). "
            f"Any such bar straddles the 11:30-13:00 lunch break. Legal choices: "
            f"{[m for m in range(1, 121) if 120 % m == 0]}."
        )
    return 240 // freq_minutes


def session_index(day: DateLike, freq_minutes: int = 1) -> pd.DatetimeIndex:
    """Bar-CLOSE timestamps for one session, with the lunch break absent.

    Labelled by bar close (09:31, ..., 11:30, 13:01, ..., 15:00) because that is what
    Chinese data vendors ship — the opposite of the US bar-open convention. Mixing the two
    shifts every signal by one bar in the direction of look-ahead.
    """
    n = session_bar_count(freq_minutes)
    d = pd.Timestamp(_to_date(day))
    step = pd.Timedelta(minutes=freq_minutes)
    morning = pd.date_range(d + pd.Timedelta("9:30:00") + step,
                            d + pd.Timedelta("11:30:00"), freq=step)
    afternoon = pd.date_range(d + pd.Timedelta("13:00:00") + step,
                              d + pd.Timedelta("15:00:00"), freq=step)
    idx = morning.append(afternoon)
    assert len(idx) == n, (len(idx), n)
    return idx


if __name__ == "__main__":
    import sys

    # A-share names are Chinese; a Windows console defaults to cp1252 and would
    # raise UnicodeEncodeError on the first stock name. Fix the stream, not the data.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=" * 86)
    print("1. THE LIMIT-UP BAR: the fill a Western engine invents")
    print("=" * 86)
    # 300750 on a day it locked limit-up: opened at the limit and never traded elsewhere.
    prev_close = 200.00
    pct = daily_limit_pct("300750", "宁德时代", "2021-06-01")
    up = limit_price(prev_close, pct, "up")
    locked = {"prev_close": prev_close, "open": up, "high": up, "low": up,
              "close": up, "volume": 1_200_000}
    normal = {"prev_close": prev_close, "open": 202.0, "high": up, "low": 199.0,
              "close": 239.9, "volume": 8_400_000}
    suspended = {"prev_close": prev_close, "open": prev_close, "high": prev_close,
                 "low": prev_close, "close": prev_close, "volume": 0}

    print(f"code 300750 (ChiNext), prev_close {prev_close:.2f}, limit {pct:.0%} "
          f"-> up limit {up:.2f}")
    for label, bar in [("locked at the up limit", locked),
                       ("touched the limit but traded lower", normal),
                       ("suspended: volume 0", suspended)]:
        ok, why = explain_buy(bar, pct)
        print(f"  can_buy = {str(ok):<5} [{label}]\n            {why}")
    # Next day it limits up again: 240.00 -> 288.00, and the engine books the whole move.
    naive_fill_pnl = (288.00 - up) * 1_000
    print(f"\n  A naive engine fills 1,000 shares at {up:.2f} on the locked bar and books "
          f"{naive_fill_pnl:,.0f} CNY of\n  pure fiction when the stock limits up again. "
          f"It is not slippage -- the trade could not\n  be made at all.")

    print("\n" + "=" * 86)
    print("2. T+1: the intraday round trip that cannot happen")
    print("=" * 86)
    lots = [{"date": "2024-03-11", "qty": 2_000}, {"date": "2024-03-12", "qty": 1_000}]
    for d in ("2024-03-12", "2024-03-13"):
        q = sellable_qty(lots, d)
        note = ("today's 1,000 are locked until tomorrow" if q < 3000 else "all settled")
        print(f"  on {d}: holdings 3,000 shares, sellable {q:>5}  ({note})")
    same_day = sellable_qty([{"date": "2024-03-12", "qty": 1_000}], "2024-03-12")
    assert same_day == 0
    print("  buy 1,000 at 10:00 and sell at 14:00 the same day -> sellable_qty = 0. "
          "REFUSED.")

    print("\n" + "=" * 86)
    print("3. CHINEXT: the same code, the same engine, a different limit")
    print("=" * 86)
    for d in ("2020-08-21", "2020-08-24", "2021-06-01"):
        p = daily_limit_pct("300059", "东方财富", d)
        print(f"  300059 on {d}: limit +/-{p:.0%}   up-limit price from 20.00 = "
              f"{limit_price(20.0, p, 'up'):.2f}")
    assert daily_limit_pct("300059", "东方财富", "2020-08-21") == 0.10
    assert daily_limit_pct("300059", "东方财富", "2020-08-24") == 0.20
    print("  A 2019-2021 backtest with a hardcoded 20% mis-rejects every 2019 limit bar.")
    print("\n  the rest of the board/name matrix:")
    for code, name in [("600519", "贵州茅台"), ("600519", "*ST茅台"), ("688981", "中芯国际"),
                       ("301236", "软通动力"), ("430047", "诺思兰德"), ("873527", "夜光明")]:
        b = board_of(code)
        print(f"    {code} {name:<10} {b:<9} ST={str(is_st(name)):<5} "
              f"limit +/-{daily_limit_pct(code, name, '2024-01-02'):.0%}")

    print("\n" + "=" * 86)
    print("4. ASYMMETRIC COSTS: stamp duty is charged on the SELL LEG ONLY")
    print("=" * 86)
    for label, bd, sd in [("before the cut", "2023-08-01", "2023-08-25"),
                          ("after  the cut", "2023-09-01", "2023-09-05")]:
        c = round_trip_cost(10.00, 10.50, 10_000, bd, sd)
        print(f"  {label} (sell {sd}, stamp {c['stamp_rate_used']:.2%}):")
        print(f"    buy  leg  commission {c['buy_commission']:7.2f}  transfer "
              f"{c['buy_transfer']:5.2f}  stamp {c['buy_stamp']:7.2f}  = "
              f"{c['buy_total']:8.2f}")
        print(f"    sell leg  commission {c['sell_commission']:7.2f}  transfer "
              f"{c['sell_transfer']:5.2f}  stamp {c['sell_stamp']:7.2f}  = "
              f"{c['sell_total']:8.2f}")
        print(f"    round trip {c['round_trip']:8.2f} ({c['round_trip_bps']:.2f} bps)  "
              f"gross P&L {c['gross_pnl']:,.2f} -> net {c['net_pnl']:,.2f}")
    before = round_trip_cost(10.0, 10.5, 10_000, "2023-08-01", "2023-08-25")
    after = round_trip_cost(10.0, 10.5, 10_000, "2023-09-01", "2023-09-05")
    print(f"\n  sell leg costs {before['sell_total'] / before['buy_total']:.1f}x the buy "
          f"leg; the 2023 cut moved the round trip by "
          f"{before['round_trip'] - after['round_trip']:.2f} CNY "
          f"({before['round_trip_bps'] - after['round_trip_bps']:.2f} bps).")
    assert before["buy_stamp"] == 0.0 and after["buy_stamp"] == 0.0

    print("\n" + "=" * 86)
    print("5. THE LUNCH BREAK: 240 bars, not 330 and not 390")
    print("=" * 86)
    print(f"  1-min bars per session: {session_bar_count(1)}   "
          f"(US equities: 390; naive 09:30-15:00: 330)")
    for f in (5, 15, 30, 60):
        print(f"  {f:>3}-min bars per session: {session_bar_count(f)}")
    try:
        session_bar_count(45)
    except ValueError as e:
        print(f"  CAUGHT 45-min: {str(e).splitlines()[0]}")
    idx = session_index("2024-03-12", 30)
    print(f"\n  30-min bar closes: {[t.strftime('%H:%M') for t in idx]}")
    print("  note the 11:30 -> 13:30 jump: no bar spans the break, and the stamps are "
          "bar-CLOSE\n  (vendor convention), not bar-open.")

    print("\n" + "=" * 86)
    print("Limits are date-keyed, settlement is T+1, costs are one-sided, sessions have "
          "a hole.")
    print("=" * 86)
