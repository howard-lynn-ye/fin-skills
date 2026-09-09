"""QuantLib, with the global evaluation date made a scope instead of a global.

THE TRAP (fin_skills.load('lib-quantlib'), "The trap that costs you money"):
`ql.Settings.instance().evaluationDate` is a GLOBAL, and moving it past an instrument's
expiry returns an NPV of exactly `0.0` - silently. No warning, no exception. In a book of
thousands of positions a zero NPV reads as "worthless option", not "wrong date".

The second half is the one the guard adds: a term structure built with a FIXED reference
date does not follow the evaluation date, and the engine measures time to expiry from the
CURVE's date. The result is not zero, it is plausible and wrong. `npv_zero` prices both
anchors and reports the gap as a number.

This module gives you a scope:

    with evaluation_date("2026-06-30", curve_ref="2026-06-30", expiry="2026-12-18"):
        npv = option.NPV()
    # the previous evaluationDate is restored here, on the exception path too

`evaluation_date` runs the `npv_zero` guard on ENTRY, so it raises before you can price
rather than after you believe the answer, and it restores whatever the global held before
- including when the body raises, which is the failure mode that leaves a whole process
pricing against yesterday.

Licence BSD-3-Clause (GitHub reports NOASSERTION because the custom BSD-derived text
confuses the detector) - safe as an extra. 🚨 1.43 ships `cp39-abi3` wheels and NO sdist,
so an unsupported platform cannot build it at all.
"""
from __future__ import annotations

import datetime as _dt
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping

from fin_skills.api import Bundle, get
from fin_skills.api.guards._common import as_date
from fin_skills.bridges import _lazy

LIBRARY = "QuantLib"
LICENCE = _lazy.info(LIBRARY).licence
VERIFIED_ON = _lazy.info(LIBRARY).verified_on

#: greeks a VanillaOption exposes; each is attempted and skipped if the engine has none.
GREEKS: tuple[str, ...] = ("delta", "gamma", "vega", "theta", "rho")


class EvaluationDateError(ValueError):
    """The evaluation date, the curve reference and the expiry do not agree."""


class ZeroNpvError(ValueError):
    """NPV() returned exactly 0.0 - the signature of an expired instrument."""


def _ql_date(ql: Any, d: _dt.date):
    return ql.Date(d.day, d.month, d.year)


def _py_date(qd: Any) -> _dt.date:
    return _dt.date(qd.year(), qd.month(), qd.dayOfMonth())


@contextmanager
def evaluation_date(d: Any, *, curve_ref: Any = None,
                    expiry: Any = None) -> Iterator[Any]:
    """Set `Settings.instance().evaluationDate` for the block, and restore it after.

    d          the valuation date.
    curve_ref  the term structure's `referenceDate()`. Defaults to `d`, which is the only
               value that cannot be wrong; pass the curve's actual reference when it was
               built with a fixed date, and the guard will price the gap.
    expiry     the instrument's exercise date. When given, `npv_zero` runs on entry and
               this raises `EvaluationDateError` BEFORE the global is touched. When it is
               not given the check cannot run, and that is said out loud rather than
               skipped quietly.

    Yields the QuantLib module, so `with evaluation_date(...) as ql:` needs no second
    import.
    """
    ev = as_date(d, "eval_date")
    cr = as_date(curve_ref, "curve_ref") if curve_ref is not None else ev
    if expiry is not None:
        ex = as_date(expiry, "expiry")
        res = get("npv_zero").run(eval_date=ev, curve_ref=cr, expiry=ex)
        if not res.passed:
            raise EvaluationDateError(
                "; ".join(str(f) for f in res.errors)
                + f" -- refusing to enter the pricing scope. evidence: "
                  f"{ {k: v for k, v in res.evidence.items() if k.startswith('npv')} }")
    else:
        import warnings                                          # noqa: PLC0415 - lazy
        warnings.warn(
            "evaluation_date() was given no `expiry`, so npv_zero could not run and the "
            "date that returns NPV()==0.0 silently is unchecked. Pass expiry= as soon as "
            "you know the instrument.", stacklevel=3)

    ql = _lazy.need(LIBRARY, why="to set the evaluation date")
    settings = ql.Settings.instance()
    previous = settings.evaluationDate
    settings.evaluationDate = _ql_date(ql, ev)
    try:
        yield ql
    finally:
        settings.evaluationDate = previous


@dataclass(frozen=True)
class PricingResult:
    """One valuation, with the three dates that produced it kept beside the number."""

    npv: float
    greeks: Mapping[str, float] = field(default_factory=dict)
    flag: str = "c"
    eval_date: _dt.date | None = None
    curve_ref: _dt.date | None = None
    expiry: _dt.date | None = None
    provenance: _lazy.Provenance | None = None

    def summary(self) -> str:
        g = ", ".join(f"{k}={v:.6g}" for k, v in self.greeks.items())
        return (f"NPV {self.npv:.8g} on {self.eval_date} (curve {self.curve_ref}, expiry "
                f"{self.expiry}); {g or 'no greeks from this engine'}")


def price(instrument: Any, engine: Any = None, *, eval_date: Any, curve_ref: Any,
          expiry: Any, flag: str = "c", allow_zero: bool = False) -> PricingResult:
    """Price inside a checked evaluation-date scope and refuse a bare 0.0.

    instrument  a QuantLib instrument (`VanillaOption`, a swap, a bond ...).
    engine      a pricing engine to attach, or None if one is already set.
    allow_zero  a genuinely worthless instrument exists; say so explicitly. The default
                refuses, because "expired, so 0.0" and "worth nothing" print identically.
    """
    with evaluation_date(eval_date, curve_ref=curve_ref, expiry=expiry) as ql:
        if engine is not None:
            instrument.setPricingEngine(engine)
        npv = float(instrument.NPV())
        greeks: dict[str, float] = {"price": npv}
        for name in GREEKS:
            fn = getattr(instrument, name, None)
            if fn is None:
                continue
            try:
                greeks[name] = float(fn())
            except (RuntimeError, ValueError):
                continue                       # the engine does not produce this one
        prov = _lazy.provenance(LIBRARY, ql, request={
            "eval_date": str(as_date(eval_date, "eval_date")),
            "curve_ref": str(as_date(curve_ref, "curve_ref")),
            "expiry": str(as_date(expiry, "expiry"))})

    if npv == 0.0 and not allow_zero:
        raise ZeroNpvError(
            "NPV() returned exactly 0.0. QuantLib returns exactly zero for an instrument "
            "whose events are all in the past, with no warning - and an option that is "
            "genuinely worthless returns a tiny positive number, not a hard zero. If this "
            "one really is worth nothing, pass allow_zero=True and say so.")
    return PricingResult(npv=npv, greeks=greeks, flag=str(flag).lower()[:1],
                         eval_date=as_date(eval_date, "eval_date"),
                         curve_ref=as_date(curve_ref, "curve_ref"),
                         expiry=as_date(expiry, "expiry"), provenance=prov)


def to_bundle(res: PricingResult, **extra: Any) -> Bundle:
    """`PricingResult` -> Bundle: `greeks` + `flag`, which unlocks `greeks_convention`.

    That guard is how you find out that vollib's vega is 100x smaller per vol point than
    QuantLib's and its theta 365x smaller per day, and that neither library says so - the
    numbers here are QuantLib's RAW derivatives, tagged `convention='quantlib'` when you
    pass it on.
    """
    if not isinstance(res, PricingResult):
        raise TypeError("res must be a PricingResult")
    slots: dict[str, Any] = {"greeks": dict(res.greeks), "flag": res.flag}
    if res.eval_date is not None:
        slots["eval_date"] = res.eval_date
    if res.curve_ref is not None:
        slots["curve_ref"] = res.curve_ref
    if res.expiry is not None:
        slots["expiry"] = res.expiry
    slots.update(extra)
    return Bundle(**slots)


__all__ = ["EvaluationDateError", "GREEKS", "LIBRARY", "LICENCE", "PricingResult",
           "VERIFIED_ON", "ZeroNpvError", "evaluation_date", "price", "to_bundle"]
