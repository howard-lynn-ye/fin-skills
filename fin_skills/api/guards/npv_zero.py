"""Guard: QuantLib's three dates (lib-quantlib / npv_zero.py)."""
from __future__ import annotations

from datetime import date
from typing import Any

from fin_skills.api.base import Guard, Outcome, register
from fin_skills.api.guards._common import as_date
from fin_skills.libraries.npv_zero import S, act365, reference_npv


@register
class NpvDatesGuard(Guard):
    """evaluationDate, the curve's reference date and the expiry must agree before NPV() means anything.

    Inputs
        eval_date : ql.Settings.instance().evaluationDate.
        curve_ref : the term structure's referenceDate() (frozen at construction for
                    curves built with a fixed date).
        expiry    : the instrument's exercise date.
        spot      : optional spot for the reference magnitude. Default the module's 100.

    Fails when expiry <= eval_date (NPV() returns exactly 0.0, silently) or when the
    curve reference differs from the evaluation date (the engine measures time to
    expiry from the CURVE's date and returns a plausible wrong number). Evidence
    carries the module's reference NPV under both anchors so the error is a number.
    """

    name = "npv_zero"
    skill = "lib-quantlib"
    summary = "Fails a QuantLib valuation whose evaluation date, curve reference and expiry disagree."
    wraps = ("fin_skills.libraries.npv_zero.reference_npv",
             "fin_skills.libraries.npv_zero.act365")
    required = ("eval_date", "curve_ref", "expiry")
    optional = ("spot",)

    def check(self, eval_date: Any, curve_ref: Any, expiry: Any, spot: float = S) -> Outcome:
        out = Outcome()
        ev: date = as_date(eval_date, "eval_date")
        cr: date = as_date(curve_ref, "curve_ref")
        ex: date = as_date(expiry, "expiry")
        got = reference_npv(ev, cr, ex, spot=float(spot))
        right = reference_npv(ev, ev, ex, spot=float(spot))
        out.note(eval_date=str(ev), curve_ref=str(cr), expiry=str(ex),
                 years_from_curve=act365(cr, ex), years_from_eval=act365(ev, ex),
                 npv_as_priced=got, npv_correct_anchor=right)
        if ex <= ev:
            out.error(f"expiry {ex} <= evaluationDate {ev}: the engine drops the expired "
                      f"event and NPV() returns exactly 0.0 with no warning", where="expiry")
        if cr != ev:
            rel = abs(got / right - 1.0) if right else float("nan")
            out.error(f"curve reference {cr} != evaluationDate {ev}: time to expiry is "
                      f"measured from the curve ({act365(cr, ex):.4f}y) not the valuation "
                      f"date ({act365(ev, ex):.4f}y); reference NPV {got:.6f} vs "
                      f"{right:.6f} ({rel:.2%} off). Set evaluationDate BEFORE building "
                      f"curves, or build them with (0, calendar) so they follow it",
                      where="curve_ref")
        if out.passed:
            out.info(f"dates agree; {act365(ev, ex):.4f}y to expiry", where="dates")
        return out
