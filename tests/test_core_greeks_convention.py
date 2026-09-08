"""fin_skills.core.greeks_convention - vega/theta/rho units differ 100x / 365x between libraries."""
from __future__ import annotations

import pytest

from _helpers import QUANTLIB_CALL, VOLLIB_CALL
from fin_skills.core.greeks_convention import (CONVENTIONS, SCALED_GREEKS, UNSCALED_GREEKS,
                                               Convention, black_scholes, convert,
                                               get_convention, pnl, report, sanity_check,
                                               to_absolute)

S = K = 100.0
T, R, Q, SIGMA = 1.0, 0.05, 0.0, 0.20


def test_local_black_scholes_reproduces_the_published_quantlib_numbers():
    mine = black_scholes(S, K, T, R, Q, SIGMA, "c")
    for key, ref in QUANTLIB_CALL.items():
        assert mine[key] == pytest.approx(ref, abs=1e-6), key


def test_conversion_reproduces_vollib_and_round_trips_exactly():
    as_vollib = convert(QUANTLIB_CALL, "quantlib", "vollib")
    for key in QUANTLIB_CALL:
        assert as_vollib[key] == pytest.approx(VOLLIB_CALL[key], abs=5e-9), key
    back = convert(as_vollib, "vollib", "quantlib")
    for key in QUANTLIB_CALL:
        assert back[key] == pytest.approx(QUANTLIB_CALL[key], abs=1e-12), key
    for key in UNSCALED_GREEKS:
        assert as_vollib[key] == QUANTLIB_CALL[key]      # untouched, which is why it is a trap
    assert QUANTLIB_CALL["vega"] / as_vollib["vega"] == pytest.approx(100)
    assert QUANTLIB_CALL["theta"] / as_vollib["theta"] == pytest.approx(365)


def test_trading_day_theta_is_365_over_252_times_the_calendar_day_number():
    td = convert(QUANTLIB_CALL, "quantlib", "per_point_per_trading_day")
    assert td["theta"] / VOLLIB_CALL["theta"] == pytest.approx(365 / 252, rel=1e-6)


def test_unknown_greeks_are_refused_in_strict_mode():
    with pytest.raises(KeyError, match="vanna"):
        convert({**QUANTLIB_CALL, "vanna": 1.0}, "quantlib", "vollib")
    loose = convert({"vanna": 1.0, "vega": 100.0}, "quantlib", "vollib", strict=False)
    assert loose == {"vanna": 1.0, "vega": 1.0}


def test_convention_lookup():
    assert get_convention(" QuantLib ") is CONVENTIONS["quantlib"]
    conv = Convention("mine", 1.0, 1.0, 1.0, "a", "b", "c")
    assert get_convention(conv) is conv
    with pytest.raises(KeyError, match="unknown greek convention"):
        get_convention("blackscholes-4ever")
    # the published vollib numbers carry 8 dp of rounding, which the x100 magnifies to ~3e-7
    assert to_absolute(VOLLIB_CALL, "vollib")["vega"] == pytest.approx(QUANTLIB_CALL["vega"], abs=1e-6)
    assert set(SCALED_GREEKS) == {"vega", "theta", "rho"}
    assert "quantlib" in CONVENTIONS["quantlib"].describe()


def test_pnl_is_convention_independent_and_hand_scaling_is_100x_and_365x():
    kw = dict(d_vol=0.01, d_years=1 / 365, d_rate=1e-4)
    ql = pnl(QUANTLIB_CALL, "quantlib", **kw)
    vl = pnl(VOLLIB_CALL, "vollib", **kw)
    assert ql["total"] == pytest.approx(vl["total"], abs=1e-7)
    assert round(QUANTLIB_CALL["vega"] * 1.0 / ql["vega_pnl"]) == 100
    assert round(QUANTLIB_CALL["theta"] * 1.0 / ql["theta_pnl"]) == 365
    spot = pnl(QUANTLIB_CALL, "quantlib", d_spot=2.0)
    assert spot["delta_pnl"] == pytest.approx(2 * QUANTLIB_CALL["delta"])
    assert spot["gamma_pnl"] == pytest.approx(0.5 * QUANTLIB_CALL["gamma"] * 4)


def test_sanity_check_passes_long_options_and_catches_sign_errors():
    assert sanity_check(VOLLIB_CALL, "c", moneyness=1.0) == []
    put = black_scholes(S, K, T, R, Q, SIGMA, "p")
    assert sanity_check(put, "p", moneyness=1.0) == []
    with pytest.raises(ValueError, match="theta"):
        sanity_check({**VOLLIB_CALL, "theta": +0.0176}, "c")
    with pytest.raises(ValueError, match="put delta"):
        sanity_check({**put, "delta": +0.36}, "p")
    with pytest.raises(ValueError, match="gamma"):
        sanity_check({k: -v for k, v in put.items()}, "p")      # a short pasted into a long
    with pytest.raises(ValueError, match="flag"):
        sanity_check(put, "x")


def test_sanity_check_soft_warnings_are_returned_not_raised():
    hard = sanity_check({**VOLLIB_CALL, "theta": 0.01}, "c", raise_on_fail=False)
    assert len(hard) == 1 and "theta" in hard[0]
    soft = sanity_check({**VOLLIB_CALL, "delta": 0.3}, "c", moneyness=1.2)
    assert len(soft) == 1 and "ITM call" in soft[0]


def test_report_carries_units():
    text = report(VOLLIB_CALL, "vollib")
    assert "convention: vollib" in text and "per CALENDAR day" in text


def test_demo_asserts_its_own_numbers(run_main):
    out = run_main("fin_skills.core.greeks_convention")
    assert "Tag the convention at the boundary" in out
