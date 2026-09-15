"""Foreign-room gates and percentage returns, with dated session-routing examples."""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from fin_skills.asia.asean import (
    can_a_foreigner_buy, entry_cost_only, foreign_premium, foreign_price,
    foreign_vs_local_return, market, session_minutes, settlement_date,
    synthetic_room_and_price, vietnam_blocked_days,
)


def test_session_windows_keep_lunch_and_friday_distinct():
    assert session_minutes("SGX") == 420
    assert session_minutes("SET") == 300
    assert session_minutes("IDX") == 320
    assert session_minutes("IDX", friday=True) == 260
    assert session_minutes("BURSA") == 360
    assert session_minutes("PSE") == 255  # Explicit table snapshot, not a live calendar.
    assert session_minutes("HOSE") == 270
    assert market("sgx")["session"][0][1] == "12:00"
    with pytest.raises(ValueError):
        market("unknown")


def test_settlement_transition_is_not_backfilled_into_the_old_regime():
    assert settlement_date("SGX", "2018-12-07") == date(2018,12,12)
    assert settlement_date("SGX", "2018-12-10") == date(2018,12,12)
    assert settlement_date("PSE", "2023-08-23") == date(2023,8,28)
    assert settlement_date("PSE", "2023-08-24") == date(2023,8,28)
    assert settlement_date("SGX", "2026-09-10", ["2026-09-14"]) == date(2026,9,15)


def test_vietnam_does_not_invent_a_t3_t2_switch_from_a_delivery_time_change():
    with pytest.raises(ValueError, match="explicit settlement lag"):
        settlement_date("HOSE", "2022-08-29")
    assert settlement_date("HOSE", "2022-08-26", lag=2) == date(2022,8,30)
    with pytest.raises(ValueError):
        settlement_date("HOSE", "2022-08-26", lag=-1)


def test_ownership_room_does_not_establish_foreign_line_liquidity():
    assert can_a_foreigner_buy(.01, False)[0] is True  # Only the ownership gate.
    assert can_a_foreigner_buy(0., False)[0] is False
    assert can_a_foreigner_buy(0., True)[0] is None
    assert can_a_foreigner_buy(0., False, foreign_transfer_available=True)[0] is True
    with pytest.raises(ValueError):
        can_a_foreigner_buy(float("nan"), True)


def test_premium_shape_is_an_explicit_scenario_and_can_be_turned_off():
    assert foreign_premium(0.) == .2
    assert foreign_premium(.1) < foreign_premium(.01) < foreign_premium(0.)
    assert foreign_price(100., 0.) == 120.
    assert foreign_price(100., 0., cap=0.) == 100.
    for kwargs in ({"scale":0.}, {"cap":-.1}, {"room":float("nan")}):
        params = {"room":0., **kwargs}
        with pytest.raises(ValueError):
            foreign_premium(**params)


def test_constant_premium_cancels_in_percentage_returns():
    frame = pd.DataFrame({"local":[100.,110.,90.], "foreign":[120.,132.,108.],
                          "premium":[.2,.2,.2], "room":[0.,0.,0.]})
    result = foreign_vs_local_return(frame, 1)
    assert result["mean_gap_bps"] == pytest.approx(0.,abs=1e-10)
    assert result["worst_gap_bps"] == pytest.approx(0.,abs=1e-10)
    assert entry_cost_only(frame)["mean_entry_premium_bps"] == pytest.approx(2000.)
    # Price-level premiums are not an extra deduction from those holding returns.


def test_changing_premium_interacts_with_the_local_gross_return():
    frame = pd.DataFrame({"local":[100.,110.], "foreign":[120.,110.],
                          "premium":[.2,0.], "room":[0.,.2]})
    result = foreign_vs_local_return(frame, 1)
    expected = (110./120. - 1) - (110./100. - 1)
    assert result["mean_gap_bps"] == pytest.approx(expected*1e4)
    assert result["mean_foreign_pct"] < 0 < result["mean_local_pct"]


def test_seeded_room_path_and_reported_statistics():
    frame = synthetic_room_and_price()
    pd.testing.assert_frame_equal(frame, synthetic_room_and_price())
    assert not frame.equals(synthetic_room_and_price(seed=1))
    assert frame.foreign_held.between(0,.49).all()
    np.testing.assert_allclose(frame.room, .49-frame.foreign_held)
    assert (frame.foreign >= frame.local).all()  # Chosen nonnegative scenario premium.
    stats = foreign_vs_local_return(frame)
    assert stats["n_windows"] == 1190
    assert stats["mean_gap_bps"] == pytest.approx(81.,abs=.1)
    blocks = vietnam_blocked_days(frame)
    assert blocks["blocked_days"] == 71
    assert blocks["longest_run"] >= 1


def test_short_or_invalid_scenarios_fail_explicitly():
    with pytest.raises(ValueError):
        synthetic_room_and_price(n_days=1)
    with pytest.raises(ValueError):
        synthetic_room_and_price(foreign_limit=.4,start_held=.5)
    with pytest.raises(ValueError):
        foreign_vs_local_return(synthetic_room_and_price(n_days=5),10)


def test_demo_is_ascii_and_does_not_claim_an_empirical_premium(run_main):
    out = run_main("fin_skills.asia.asean")
    assert out.isascii()
    assert "no empirical premium estimate" in out
    assert "Constant 20% premium: return gap=0.0 bp" in out
