"""Guard: option-greek sign invariants and units (derivatives-pricing / greeks_convention.py)."""
from __future__ import annotations

from typing import Mapping

from fin_skills.api.base import Guard, Outcome, register
from fin_skills.core.greeks_convention import (get_convention, report, sanity_check,
                                               to_absolute)


@register
class GreeksGuard(Guard):
    """What a LONG vanilla option can never do: theta >= 0, gamma <= 0, vega <= 0, a put with delta > 0.

    Inputs
        greeks     : {price, delta, gamma, vega, theta, rho} (any subset).
        flag       : 'c' or 'p'.
        moneyness  : optional S/K; drives soft checks that are reported as warnings.
        convention : optional 'quantlib' | 'vollib' | 'financepy' | 'absolute' |
                     'per_point_per_day' | 'per_point_per_trading_day' - tags the units
                     and adds the absolute-convention conversion to the evidence.

    Fails on any hard sign violation (a sign flip, a short position labelled long, or
    greeks glued together from two pricers). Soft moneyness checks are warnings.
    """

    name = "greeks_convention"
    skill = "derivatives-pricing"
    summary = "Fails greeks whose signs a long vanilla option cannot have; tags and converts their units."
    wraps = ("fin_skills.core.greeks_convention.sanity_check",
             "fin_skills.core.greeks_convention.to_absolute",
             "fin_skills.core.greeks_convention.report")
    required = ("greeks", "flag")
    optional = ("moneyness", "convention")

    def check(self, greeks: Mapping[str, float], flag: str, moneyness: float | None = None,
              convention: str | None = None) -> Outcome:
        out = Outcome()
        if not isinstance(greeks, Mapping) or not greeks:
            raise TypeError("greeks must be a non-empty mapping {greek: value}")
        if not isinstance(flag, str) or flag.strip().lower()[:1] not in ("c", "p"):
            raise TypeError("flag must be 'c' or 'p'")
        try:
            g = {str(k): float(v) for k, v in greeks.items()}
        except (TypeError, ValueError) as exc:
            raise TypeError("every greek must be a number") from exc

        hard: list[str] = []
        try:
            sanity_check(g, flag, raise_on_fail=True)
        except ValueError as exc:
            hard = [line.strip()[2:] for line in str(exc).splitlines()[1:]
                    if line.strip().startswith("- ")]
        soft = [s for s in sanity_check(g, flag, moneyness, raise_on_fail=False)
                if s not in hard]
        for h in hard:
            out.error(h, where="sign")
        for s in soft:
            out.warning(s, where="moneyness")
        if not hard:
            out.info(f"long {'call' if flag.strip().lower().startswith('c') else 'put'} sign "
                     f"invariants hold", where="sign")
        out.note(hard=hard, soft=soft)

        if convention is not None:
            try:
                conv = get_convention(convention)
            except KeyError as exc:
                raise TypeError(str(exc)) from exc
            out.note(convention=conv.name, absolute=to_absolute(g, conv, strict=False),
                     report=report(g, conv))
        return out
