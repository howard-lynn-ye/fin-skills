"""Guard: FX quote direction (fx-markets / fx_conventions.py)."""
from __future__ import annotations

from typing import Sequence

from fin_skills.api.base import Guard, Outcome, register
from fin_skills.futures_fx.fx_conventions import check_convention, is_inverted, parse_pair, pip_size


@register
class FxConventionGuard(Guard):
    """A pair written backwards from market convention inverts every return in the book.

    Inputs
        pair : 'EURUSD', 'EUR/USD', ... or a sequence of pairs.

    Fails when a pair is quoted backwards (JPYUSD, USDEUR). Evidence records, per
    pair, the base/quote split, whether USD is the BASE (a rising price = a stronger
    dollar) and the pip size - the two facts a cross-pair signal has to respect.
    """

    name = "fx_conventions"
    skill = "fx-markets"
    summary = "Fails a backwards-quoted FX pair; records which pairs are USD-base and their pip size."
    wraps = ("fin_skills.futures_fx.fx_conventions.check_convention",
             "fin_skills.futures_fx.fx_conventions.is_inverted",
             "fin_skills.futures_fx.fx_conventions.pip_size")
    required = ("pair",)
    optional = ()

    def check(self, pair: str | Sequence[str]) -> Outcome:
        out = Outcome()
        pairs = [pair] if isinstance(pair, str) else list(pair)
        if not pairs or not all(isinstance(p, str) for p in pairs):
            raise TypeError("pair must be a string or a sequence of strings")
        table: dict[str, dict[str, object]] = {}
        for p in pairs:
            base, quote = parse_pair(p)          # ValueError -> TypeError (bad input)
            try:
                check_convention(p)
            except ValueError as exc:
                out.error(str(exc), where=p)
                continue
            try:
                inverted: bool | None = is_inverted(p)
            except ValueError:
                inverted = None                   # a cross: no USD leg
            table[p] = {"base": base, "quote": quote, "usd_is_base": inverted,
                        "pip_size": pip_size(p)}
            side = ("USD is BASE: rising price = stronger dollar" if inverted
                    else "USD is QUOTE: rising price = weaker dollar" if inverted is False
                    else "cross with no USD leg")
            out.info(f"{base}/{quote}, pip {pip_size(p):g}, {side}", where=p)
        out.note(pairs=table)
        return out
