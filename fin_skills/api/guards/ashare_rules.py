"""Guard: A-share fills a Western engine invents (china-trading-stack / ashare_rules.py)."""
from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

from fin_skills.api.base import Guard, Outcome, register
from fin_skills.china.ashare_rules import (board_of, can_sell, daily_limit_pct, explain_buy,
                                           limit_price, sellable_qty)


@register
class AShareRulesGuard(Guard):
    """Can this bar actually be traded? Price limits are date-keyed and settlement is T+1.

    Inputs
        bar            : mapping with prev_close, open, high, low, close, volume
                         (paused / trade_status honoured if present).
        code           : 6-digit code with optional exchange prefix/suffix.
        date           : the bar's date - REQUIRED because every limit has changed.
        stock_name     : stock name, to detect the ST marker (halves the main-board limit).
        side           : 'buy' (default) or 'sell'.
        days_since_ipo : trading days since listing (first 5 have no limit on the
                         registration-based boards).
        lots           : for a sell, the open buy lots ({'date', 'qty'} mappings or
                         (date, qty) pairs); T+1 makes today's purchases unsellable.

    Fails when the bar is locked at the limit on the side you want to trade, suspended,
    or zero-volume - and, for a sell with `lots`, when nothing is sellable today.
    """

    name = "ashare_rules"
    skill = "china-trading-stack"
    summary = "Fails a fill on a limit-locked, suspended or zero-volume A-share bar, and a same-day sale (T+1)."
    wraps = ("fin_skills.china.ashare_rules.daily_limit_pct",
             "fin_skills.china.ashare_rules.explain_buy",
             "fin_skills.china.ashare_rules.can_sell",
             "fin_skills.china.ashare_rules.sellable_qty")
    required = ("bar", "code", "date")
    optional = ("stock_name", "side", "days_since_ipo", "lots")

    def check(self, bar: Mapping[str, Any], code: str, date: Any, stock_name: str | None = None,
              side: str = "buy", days_since_ipo: int | None = None,
              lots: Iterable[Mapping[str, Any] | Sequence[Any]] | None = None) -> Outcome:
        out = Outcome()
        if not isinstance(bar, Mapping):
            raise TypeError("bar must be a mapping with prev_close/high/low/volume")
        for col in ("high", "low"):
            if col not in bar:
                raise TypeError(f"bar is missing {col!r}")
        if not isinstance(code, str):
            raise TypeError("code must be a string")
        if side not in ("buy", "sell"):
            raise TypeError("side must be 'buy' or 'sell'")
        if date is None:
            raise TypeError("date is required: every limit and fee rate has changed at least once")

        pct = daily_limit_pct(code, stock_name, date, days_since_ipo)   # ValueError -> TypeError
        prev_close = bar.get("prev_close", bar.get("pre_close", bar.get("preclose")))
        if prev_close is None:
            raise TypeError("bar needs prev_close: the limit is a function of the PREVIOUS close")
        up = limit_price(float(prev_close), pct, "up")
        down = limit_price(float(prev_close), pct, "down")
        out.note(board=board_of(code), limit_pct=pct, up_limit=up, down_limit=down, side=side)

        if side == "buy":
            ok, why = explain_buy(bar, pct)
            (out.info if ok else out.error)(why, where=f"{code} buy")
        else:
            ok = can_sell(bar, pct)
            if ok:
                out.info("tradable: not locked at the down limit, not suspended", where=f"{code} sell")
            else:
                out.error(f"cannot sell: bar is locked at the down limit "
                          f"({down:.2f}), suspended, or has zero volume", where=f"{code} sell")
            if lots is not None:
                qty = sellable_qty(lots, date)
                out.note(sellable_qty=qty)
                if qty <= 0:
                    out.error("T+1: nothing bought before this date is sellable today; a "
                              "same-day round trip cannot be executed", where=f"{code} T+1")
                else:
                    out.info(f"{qty} shares settled and sellable", where=f"{code} T+1")
        if math.isinf(pct):
            out.info("no price limit in the first 5 sessions after listing", where=code)
        return out
