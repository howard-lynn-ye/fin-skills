"""fin_skills.libraries.npv_zero - QuantLib prices against three dates and never compares them."""
from __future__ import annotations

from datetime import date

import pytest

from _helpers import bs_call
from conftest import requires
from fin_skills.libraries.npv_zero import (EXP, HIST_EXP, HIST_VAL, K, Q, R, S, SIGMA, STALE,
                                           VAL, act365, bs_call as module_bs_call,
                                           live_quantlib, reference_npv)


def test_closed_form_matches_an_independent_implementation():
    assert module_bs_call(S, K, 1.0, R, Q, SIGMA) == pytest.approx(bs_call(S, K, R, Q, SIGMA, 1.0), abs=1e-9)
    assert module_bs_call(120.0, K, 0.0, R, Q, SIGMA) == 20.0          # t <= 0 is intrinsic
    assert act365(date(2020, 1, 1), date(2021, 1, 1)) == pytest.approx(366 / 365)


def test_expired_option_prices_to_exactly_zero():
    assert reference_npv(date.today(), HIST_VAL, HIST_EXP) == 0.0       # forgot evaluationDate
    assert reference_npv(VAL, VAL, VAL, spot=120.0) == 0.0             # expiry ON the eval date
    assert reference_npv(HIST_EXP, HIST_VAL, HIST_EXP) == 0.0


def test_curve_reference_not_evaluation_date_sets_the_time_to_expiry():
    right = reference_npv(VAL, VAL, EXP)
    wrong = reference_npv(VAL, STALE, EXP)
    assert right == pytest.approx(bs_call(S, K, R, Q, SIGMA, act365(VAL, EXP)))
    assert wrong == pytest.approx(bs_call(S, K, R, Q, SIGMA, act365(STALE, EXP)))
    assert abs(wrong / right - 1) > 0.10                                # plausible and wrong


@pytest.mark.skipif(live_quantlib() is not None, reason="QuantLib is installed")
def test_live_is_none_without_quantlib():
    assert live_quantlib() is None


@requires("QuantLib")
def test_installed_quantlib_reproduces_the_reference_model():
    live = live_quantlib()
    assert live is not None
    assert live["forgot"] == 0.0 and live["expiry_today"] == 0.0
    assert live["wrong_order"] == pytest.approx(reference_npv(VAL, STALE, EXP), abs=1e-6)
    assert live["right_order"] == pytest.approx(reference_npv(VAL, VAL, EXP), abs=1e-6)
    assert live["floating"] == pytest.approx(live["right_order"], abs=1e-9)
    assert live["wrong_order_yf"] == pytest.approx(act365(STALE, EXP))
    assert live["expiry_today_included"] == pytest.approx(20.0)
    assert "empty Handle" in live["empty_handle"]


def test_demo_prints_the_rule(run_main):
    out = run_main("fin_skills.libraries.npv_zero")
    assert "An NPV of" in out and "exactly 0.0 is a date bug until proven otherwise" in out
