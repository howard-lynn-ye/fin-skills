"""fin_skills.core.option_lifecycle - early exercise, pin risk and assignment accounting."""
from __future__ import annotations

import pytest

from _helpers import bs_call, bs_put
from fin_skills.core.option_lifecycle import (covered_call_pnl, crr, early_exercise_premium,
                                              pin_exposure)


def test_crr_matches_black_scholes_at_q_zero():
    # the module's own parameters (S=100, K=90, 30 days) with the default 800 steps
    assert crr(100, 90, 0.05, 0.0, 0.2, 30 / 365, american=False) == pytest.approx(
        bs_call(100, 90, 0.05, 0.0, 0.2, 30 / 365), abs=1e-3)
    assert crr(100, 90, 0.05, 0.0, 0.2, 30 / 365, american=False, call=False) == pytest.approx(
        bs_put(100, 90, 0.05, 0.0, 0.2, 30 / 365), abs=1e-3)
    # an at-the-money one-year call needs more steps for the same tolerance
    assert crr(100, 100, 0.05, 0.0, 0.2, 1.0, steps=3000, american=False) == pytest.approx(
        bs_call(100, 100, 0.05, 0.0, 0.2, 1.0), abs=1e-3)


def test_american_call_without_dividends_is_never_exercised_early():
    for K, T in ((90.0, 30 / 365), (100.0, 1.0), (110.0, 0.5)):
        eur = crr(100, K, 0.05, 0.0, 0.2, T, american=False)
        ame = crr(100, K, 0.05, 0.0, 0.2, T, american=True)
        assert ame == pytest.approx(eur, abs=1e-12)


def test_american_put_is_worth_more_than_european_and_at_least_intrinsic():
    eur = crr(100, 110, 0.05, 0.0, 0.2, 0.5, american=False, call=False)
    ame = crr(100, 110, 0.05, 0.0, 0.2, 0.5, american=True, call=False)
    assert ame > eur + 0.1
    assert ame >= 10.0


def test_deep_itm_high_dividend_american_call_equals_intrinsic():
    rows = early_exercise_premium()
    assert [q for q, *_ in rows] == [0.0, 0.02, 0.05, 0.10, 0.20]
    by_q = {q: (eur, ame, prem, intr) for q, eur, ame, prem, intr in rows}
    assert by_q[0.0][2] == pytest.approx(0.0, abs=1e-12)          # no premium without dividends
    assert by_q[0.20][1] == pytest.approx(10.0, abs=1e-6)          # American value IS S - K
    assert by_q[0.20][0] < 10.0                                    # European understates it
    assert crr(100, 90, 0.05, 0.20, 0.2, 30 / 365, american=True) == pytest.approx(10.0, abs=1e-6)
    premiums = [prem for _, _, _, prem, _ in rows]
    assert premiums == sorted(premiums)                            # grows with the yield


def test_pin_exposure_flips_on_one_cent():
    rows = {S: (auto, shares) for S, _, auto, shares, _ in pin_exposure()}
    assert rows[100.0] == ("expires", 0)
    assert rows[99.99] == ("expires", 0)
    assert rows[100.01] == ("auto-exercised", -1000)
    assert rows[100.05] == ("auto-exercised", -1000)


def test_covered_call_assignment_error_changes_sign_with_the_path():
    correct = {S: covered_call_pnl(S_expiry=S)["correct_total"] for S in (110.0, 103.0, 96.0)}
    assert len(set(correct.values())) == 1          # closed at the ex-date whatever follows
    assert covered_call_pnl(S_expiry=110.0)["gap"] > 0
    assert covered_call_pnl(S_expiry=96.0)["gap"] < 0
    r = covered_call_pnl()
    assert r["time_value_at_exdiv"] < r["dividend"]  # the reason assignment happens
    assert r["naive_total"] == pytest.approx(sum(r["naive_parts"]))
    assert r["correct_total"] == pytest.approx(sum(r["correct_parts"]))
    assert r["correct_parts"][1] == 0.0             # the dividend is NOT received


def test_demo_prints_the_rule(run_main):
    out = run_main("fin_skills.core.option_lifecycle")
    assert "Rule: an options backtest must model assignment, not just expiry." in out
