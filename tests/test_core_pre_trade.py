"""fin_skills.core.pre_trade - the checks that run BEFORE a person sends an instruction.

Three claims carry this module and each one has a test that can fail:

  * every check fires on the bad input and CLEARS on the good one - a check that only ever
    blocks is indistinguishable from a check that is broken;
  * `allowed` is False when a required fact is ABSENT, not only when a check fails. Every
    required fact is dropped in turn and every drop must block;
  * the session check reads the DECLARED calendar and DECLARED `now`. Proven twice: an AST
    walk that finds no `now()` / `today()` / `utcnow()` call anywhere in the module, and a
    behavioural test that replaces the module's `datetime` import and `pd.Timestamp.now`
    with objects that raise, then asserts the verdict is byte-identical.

The fourth claim - that this module has no path that could transmit an instruction - is
tested next to the execution bridge's, in `tests/test_bridges_execution.py`, because it is
the same grep over the same forbidden names and it must stay one list.
"""
from __future__ import annotations

import ast
import datetime as dt
import inspect
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import fin_skills.api as api
from fin_skills.core import pre_trade as P
from fin_skills.core.cost_plausibility import impact_bps
from fin_skills.core.pre_trade import (CHECKS, EXAMPLE_CAPS, NYSE_HOURS, PLANTED, SSE_HOURS,
                                       Caps, KillSwitch, OrderVerdict, ProposedOrder,
                                       TradingState, VenueHours, broker_positions_from,
                                       check_duplicate, check_exposure, check_fat_finger,
                                       check_kill_switch, check_order, check_participation,
                                       check_session, empty_blotter, seeded_blotter,
                                       threshold_sweep, trip_if)

MODULE_SOURCE = Path(inspect.getsourcefile(P)).read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def book():
    """The seeded blotter: 40 proposals, three of them deliberately wrong, plus the state."""
    proposals, state = seeded_blotter()
    return proposals, state, {p.client_order_id: p for p in proposals}


@pytest.fixture(scope="module")
def clean(book):
    proposals, _, _ = book
    return next(p for p in proposals if p.client_order_id not in PLANTED)


# ====================================================================== the proposal
def test_a_proposal_refuses_a_naive_timestamp():
    with pytest.raises(ValueError, match="tz-aware"):
        ProposedOrder("AA00", "buy", 100, 10.0, "c1", pd.Timestamp("2026-09-10 14:00"))


@pytest.mark.parametrize("kw,match", [
    (dict(side="long"), "side must be"),
    (dict(qty=0), "qty must be"),
    (dict(qty=-5), "qty must be"),
    (dict(limit_price=0.0), "limit_price"),
    (dict(client_order_id=""), "client_order_id"),
    (dict(symbol=" "), "symbol"),
])
def test_a_proposal_refuses_nonsense(kw, match):
    base = dict(symbol="AA00", side="buy", qty=100.0, limit_price=10.0,
                client_order_id="c1", ts=pd.Timestamp("2026-09-10 14:00", tz="UTC"))
    with pytest.raises(ValueError, match=match):
        ProposedOrder(**{**base, **kw})


def test_signed_qty_carries_the_direction():
    ts = pd.Timestamp("2026-09-10 14:00", tz="UTC")
    assert ProposedOrder("A", "buy", 10, 1.0, "c", ts).signed_qty == 10
    assert ProposedOrder("A", "sell", 10, 1.0, "c", ts).signed_qty == -10


def test_venue_hours_have_no_default_zone():
    with pytest.raises(ValueError, match="no default"):
        VenueHours("X", "", dt.time(9), dt.time(16), (dt.time(9), dt.time(9, 1)),
                   (dt.time(15), dt.time(16)))
    with pytest.raises(ValueError, match="before continuous_close"):
        VenueHours("X", "UTC", dt.time(16), dt.time(9), (dt.time(9), dt.time(9, 1)),
                   (dt.time(15), dt.time(16)))


# ====================================================================== the kill switch
def test_a_tripped_switch_must_carry_a_reason():
    with pytest.raises(ValueError, match="reason string"):
        KillSwitch(True, "")
    assert not KillSwitch().tripped
    assert str(KillSwitch()) == "not tripped"


@pytest.mark.parametrize("kw,word", [
    (dict(cum_loss=-150_000.0, max_loss=100_000.0), "cumulative loss"),
    (dict(error_rate=0.30, max_error_rate=0.05), "error rate"),
    (dict(connected=False), "link is down"),
])
def test_trip_if_fires_on_each_stated_condition(kw, word):
    k = trip_if(**kw)
    assert k.tripped and word in k.reason


@pytest.mark.parametrize("kw", [
    dict(),                                              # nothing stated
    dict(cum_loss=-150_000.0),                           # loss but no threshold
    dict(max_loss=100_000.0),                            # threshold but no loss
    dict(cum_loss=-50_000.0, max_loss=100_000.0),        # inside the limit
    dict(error_rate=0.01, max_error_rate=0.05),
    dict(connected=True),
])
def test_trip_if_stays_off_when_no_condition_fired(kw):
    assert not trip_if(**kw).tripped


def test_a_tripped_switch_fails_every_check_regardless_of_input(book, clean):
    _, state, _ = book
    dead = replace(state, kill=trip_if(cum_loss=-1e6, max_loss=1.0))
    for fn in (check_kill_switch, check_session, check_fat_finger, check_participation,
               check_duplicate, check_exposure):
        findings = fn(clean, dead)
        assert len(findings) == 1, fn.__name__
        assert findings[0].severity == "block" and findings[0].code == "kill_switch"
    v = check_order(clean, dead)
    assert not v.allowed
    assert {f.check for f in v.blocks} == set(CHECKS)
    assert len({f.message for f in v.blocks}) == 1, "one reason, repeated - no cleverness"


def test_the_switch_blocks_even_with_no_facts_at_all(clean):
    empty = TradingState(kill=KillSwitch(True, "disconnected"))
    v = check_order(clean, empty)
    assert not v.allowed and len(v.blocks) == len(CHECKS)
    # and the same empty state WITHOUT the switch blocks for a different reason entirely
    v2 = check_order(clean, TradingState())
    assert not v2.allowed and all(f.code == "missing_fact" for f in v2.blocks)


# ========================================================================= the session
def _at(state, when: str) -> TradingState:
    return replace(state, now=pd.Timestamp(when, tz="UTC"))


@pytest.mark.parametrize("when,codes", [
    ("2026-09-10 14:05:00", {"open"}),                       # 10:05 ET, mid-session
    ("2026-09-10 13:15:00", {"outside_hours"}),              # 09:15 ET, before the open
    ("2026-09-10 13:30:30", {"opening_auction", "open"}),    # 09:30 ET
    ("2026-09-10 19:55:00", {"closing_auction", "open"}),    # 15:55 ET
    ("2026-09-10 20:05:00", {"outside_hours"}),              # 16:05 ET
    ("2026-09-12 14:05:00", {"not_a_session"}),              # a Saturday
    ("2026-11-27 17:30:00", {"half_day", "open"}),           # 12:30 ET on a half day
    ("2026-11-27 18:05:00", {"half_day", "outside_hours"}),  # 13:05 ET, after the early close
])
def test_the_session_check_reads_the_declared_calendar(book, clean, when, codes):
    _, state, _ = book
    assert {f.code for f in check_session(clean, _at(state, when))} == codes


def test_the_half_day_close_is_the_only_difference_on_that_date(book, clean):
    _, state, _ = book
    normal = check_session(clean, _at(state, "2026-11-25 18:05:00"))   # 13:05 ET, full day
    half = check_session(clean, _at(state, "2026-11-27 18:05:00"))     # 13:05 ET, half day
    assert {f.code for f in normal} == {"open"}
    assert {f.code for f in half} == {"half_day", "outside_hours"}


def test_a_venue_with_a_lunch_break_blocks_through_it(book, clean):
    _, state, _ = book
    cal = P._declared_calendar()
    sse = replace(state, venue=SSE_HOURS, sessions=cal)
    # 12:00 Shanghai on 2026-09-10 = 04:00 UTC
    codes = {f.code for f in check_session(clean, _at(sse, "2026-09-10 04:00:00"))}
    assert codes == {"lunch_break"}
    # 09:20 Shanghai = 01:20 UTC: the call auction is running, continuous is not
    codes = {f.code for f in check_session(clean, _at(sse, "2026-09-10 01:20:00"))}
    assert codes == {"opening_auction", "auction_only", "open"}


@pytest.mark.parametrize("drop,fact", [("now", "now"), ("sessions", "sessions"),
                                       ("venue", "venue")])
def test_the_session_check_blocks_on_an_absent_fact(book, clean, drop, fact):
    _, state, _ = book
    findings = check_session(clean, replace(state, **{drop: None}))
    assert [f.code for f in findings] == ["missing_fact"]
    assert findings[0].evidence["fact"] == fact


def test_a_naive_now_is_refused_not_localised(book, clean):
    _, state, _ = book
    naive = replace(state, now=pd.Timestamp("2026-09-10 14:05:00"))
    assert [f.code for f in check_session(clean, naive)] == ["naive_now"]


# ------------------------------------------------ and it never consults the wall clock
_CLOCK_CALLS = {"now", "utcnow", "today", "fromtimestamp", "monotonic", "perf_counter"}


def test_no_wall_clock_call_exists_anywhere_in_the_module():
    """An AST walk, not a grep: the docstrings mention `datetime.now()` to say it is not
    used, and prose must not be able to fail - or pass - this test."""
    tree = ast.parse(MODULE_SOURCE)
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Attribute) and fn.attr in _CLOCK_CALLS:
                hits.append(fn.attr)
            elif isinstance(fn, ast.Name) and fn.id in _CLOCK_CALLS:
                hits.append(fn.id)
    assert hits == [], f"pre_trade.py calls the wall clock: {hits}"
    imported = {n.name for node in ast.walk(tree) if isinstance(node, ast.Import)
                for n in node.names}
    assert "time" not in imported, "importing `time` is how the clock gets back in"


def test_the_ast_walk_would_actually_catch_a_clock_call():
    """A negative test is worthless unless it can fail. Poison the source and watch."""
    for poison in ("x = pd.Timestamp.now(tz='UTC')\n", "x = dt.datetime.utcnow()\n",
                   "x = dt.date.today()\n"):
        tree = ast.parse(MODULE_SOURCE + poison)
        hits = [n.func.attr for n in ast.walk(tree)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr in _CLOCK_CALLS]
        assert hits, poison


class _Explodes:
    """Anything touched on this raises. Standing in for `datetime`, it proves absence."""

    def __getattr__(self, name):
        raise AssertionError(f"pre_trade read the wall clock: datetime.{name}")


def test_the_verdict_does_not_change_when_the_clock_is_taken_away(book, clean, monkeypatch):
    _, state, _ = book
    before = check_order(clean, state)
    monkeypatch.setattr(P, "dt", _Explodes())
    monkeypatch.setattr(pd.Timestamp, "now",
                        classmethod(lambda cls, *a, **k: (_ for _ in ()).throw(
                            AssertionError("pre_trade called Timestamp.now"))))
    after = check_order(clean, state)
    assert after == before
    assert after.allowed
    # and a session check at a DECLARED time still answers about that time, not about today
    for when, code in (("2026-09-12 14:05:00", "not_a_session"),
                       ("2026-09-10 20:05:00", "outside_hours")):
        assert code in {f.code for f in check_session(clean, _at(state, when))}


# ======================================================================== the fat finger
def test_the_fat_finger_check_clears_a_clean_ticket(book, clean):
    _, state, _ = book
    findings = check_fat_finger(clean, state)
    assert [f.severity for f in findings] == ["info"]
    assert findings[0].code == "within_caps"


def test_a_ten_times_quantity_slip_trips_the_size_tests_and_not_the_price_ones(book):
    _, state, by_id = book
    codes = {f.code for f in check_fat_finger(by_id["slip-qty"], state)
             if f.severity == "block"}
    assert codes == {"notional_cap", "notional_vs_book", "qty_vs_adv"}
    assert "price_through_last" not in codes


def test_a_transposed_price_trips_the_price_tests_and_not_the_size_ones(book):
    _, state, by_id = book
    findings = check_fat_finger(by_id["slip-px"], state)
    blocks = {f.code for f in findings if f.severity == "block"}
    assert blocks == {"price_through_last"}
    assert "price_outside_day_range" in {f.code for f in findings}
    through = next(f for f in findings if f.code == "price_through_last")
    assert through.evidence["through"] == pytest.approx(0.13, abs=0.005)
    assert through.evidence["aggressive"] is True


def test_a_passive_price_typo_is_named_as_one(book, clean):
    _, state, _ = book
    last = state.last_trade[clean.symbol]
    passive = replace(clean, side="buy", limit_price=round(last * 0.80, 2))
    f = next(x for x in check_fat_finger(passive, state) if x.code == "price_through_last")
    assert f.evidence["aggressive"] is False and "not fill" in f.message


def test_an_unpriced_ticket_warns_that_the_price_tests_could_not_run(book, clean):
    _, state, _ = book
    codes = {f.code for f in check_fat_finger(replace(clean, limit_price=None), state)}
    assert "no_price_on_ticket" in codes
    assert "price_through_last" not in codes and "price_outside_day_range" not in codes


def test_the_day_range_is_a_second_anchor_and_its_absence_blocks(book, clean):
    _, state, _ = book
    findings = check_fat_finger(clean, replace(state, day_high=None))
    assert any(f.code == "missing_fact" for f in findings)
    assert not any(f.severity == "info" for f in findings)


@pytest.mark.parametrize("drop", ["last_trade", "adv_shares", "equity"])
def test_the_fat_finger_check_blocks_on_an_absent_fact(book, clean, drop):
    _, state, _ = book
    findings = check_fat_finger(clean, replace(state, **{drop: None}))
    assert [f.code for f in findings] == ["missing_fact"]


@pytest.mark.parametrize("cap", ["max_notional", "max_book_fraction", "max_adv_multiple",
                                 "max_price_through"])
def test_an_unstated_threshold_is_not_a_pass(book, clean, cap):
    _, state, _ = book
    caps = replace(state.caps, **{cap: None})
    findings = check_fat_finger(clean, replace(state, caps=caps))
    assert [f.code for f in findings] == ["missing_fact"]
    assert findings[0].evidence["fact"] == f"caps.{cap}"


# ====================================================================== participation
def test_participation_clears_a_clean_ticket_and_prices_it(book, clean):
    _, state, _ = book
    findings = check_participation(clean, state)
    assert not [f for f in findings if f.severity == "block"]
    cost = next(f for f in findings if f.code == "impact_cost")
    assert cost.evidence["impact_bps"] > 0


def test_the_impact_number_is_cost_plausibilitys_and_not_a_second_copy(book, clean):
    """If this module ever grows its own Almgren, this test is what notices."""
    _, state, _ = book
    f = next(x for x in check_participation(clean, state) if x.code == "impact_cost")
    part = clean.qty / state.adv_shares[clean.symbol]
    horizon = max(min(part / state.caps.pov_cap, 1.0), 1e-3)
    expected = impact_bps(part, daily_vol=state.daily_vol[clean.symbol],
                          exec_horizon=horizon)["realized_bps"]
    assert f.evidence["impact_bps"] == expected
    assert f.evidence["participation"] == pytest.approx(part)


def test_an_oversized_ticket_blocks_and_says_how_many_sessions_it_needs(book):
    _, state, by_id = book
    f = next(x for x in check_participation(by_id["slip-qty"], state)
             if x.severity == "block")
    assert f.code == "over_adv_cap"
    assert f.evidence["sessions"] == pytest.approx(
        f.evidence["participation"] / state.caps.pov_cap)
    assert "programme" in f.message


def test_a_multi_session_schedule_warns_before_it_blocks(book, clean):
    _, state, _ = book
    caps = replace(state.caps, max_participation=0.90, pov_cap=0.001)
    codes = {f.code for f in check_participation(clean, replace(state, caps=caps))}
    assert "multi_session_schedule" in codes and "over_adv_cap" not in codes


def test_an_absent_volatility_warns_but_does_not_block(book, clean):
    _, state, _ = book
    findings = check_participation(clean, replace(state, daily_vol=None))
    assert not [f for f in findings if f.severity == "block"]
    assert "cost_not_priced" in {f.code for f in findings}


@pytest.mark.parametrize("drop", ["adv_shares", "last_trade"])
def test_participation_blocks_on_an_absent_fact(book, clean, drop):
    _, state, _ = book
    assert [f.code for f in check_participation(clean, replace(state, **{drop: None}))] \
        == ["missing_fact"]


# ================================================================== duplicate and replay
def test_duplicate_clears_a_new_client_order_id(book, clean):
    _, state, _ = book
    findings = check_duplicate(clean, state)
    assert [f.code for f in findings] == ["unique"]


def test_the_same_client_order_id_twice_blocks(book):
    _, state, by_id = book
    codes = {f.code for f in check_duplicate(by_id["replay"], state) if f.severity == "block"}
    assert "duplicate_id" in codes


def test_a_burst_of_near_identical_tickets_blocks_even_with_distinct_ids(book):
    _, state, by_id = book
    p = by_id["replay"]
    # strip the id collision: only the WINDOW can see what is left
    sent = state.already_sent[state.already_sent["client_order_id"] != p.client_order_id]
    f = next(x for x in check_duplicate(p, replace(state, already_sent=sent))
             if x.code == "replay_burst")
    assert f.evidence["in_window"] == 3
    assert "idempotency" in f.message


def test_a_burst_below_the_stated_maximum_only_warns(book):
    _, state, by_id = book
    p = by_id["replay"]
    sent = state.already_sent[state.already_sent["client_order_id"] != p.client_order_id]
    caps = replace(state.caps, burst_max=9)
    findings = check_duplicate(p, replace(state, already_sent=sent, caps=caps))
    assert {f.code for f in findings} == {"near_identical", "unique"}


def test_a_burst_outside_the_window_is_not_a_burst(book):
    _, state, by_id = book
    p = by_id["replay"]
    sent = state.already_sent[state.already_sent["client_order_id"] != p.client_order_id]
    caps = replace(state.caps, burst_window_s=1.0)
    findings = check_duplicate(p, replace(state, already_sent=sent, caps=caps))
    assert [f.code for f in findings] == ["unique"]


def test_an_empty_blotter_is_a_fact_and_none_is_not(book, clean):
    _, state, _ = book
    assert [f.code for f in check_duplicate(clean, replace(state,
                                                           already_sent=empty_blotter()))] \
        == ["unique"]
    findings = check_duplicate(clean, replace(state, already_sent=None))
    assert [f.code for f in findings] == ["missing_fact"]
    assert findings[0].evidence["fact"] == "already_sent"


def test_a_malformed_blotter_raises_rather_than_quietly_passing(book, clean):
    _, state, _ = book
    with pytest.raises(TypeError, match="missing column"):
        check_duplicate(clean, replace(state, already_sent=pd.DataFrame({"ts": []})))
    with pytest.raises(TypeError, match="must be a DataFrame"):
        check_duplicate(clean, replace(state, already_sent=[{"ts": 1}]))


# ================================================================ position and exposure
def test_exposure_clears_a_clean_ticket_against_a_reconciled_book(book, clean):
    _, state, _ = book
    findings = check_exposure(clean, state)
    assert [f.code for f in findings] == ["within_caps"]
    assert "reconciled" in findings[0].message


def test_a_broker_disagreement_blocks_before_any_cap_is_computed(book, clean):
    _, state, _ = book
    drifted = dict(state.broker_positions)
    drifted[clean.symbol] = drifted[clean.symbol] - 300.0
    findings = check_exposure(clean, replace(state, broker_positions=drifted))
    assert [f.code for f in findings] == ["unreconciled"]
    assert findings[0].evidence["mine"] - findings[0].evidence["theirs"] == 300.0


def test_a_position_the_broker_has_never_heard_of_blocks(book, clean):
    _, state, _ = book
    drifted = {k: v for k, v in state.broker_positions.items() if k != clean.symbol}
    findings = check_exposure(clean, replace(state, broker_positions=drifted))
    assert [f.code for f in findings] == ["unreconciled"]
    assert findings[0].evidence["theirs"] is None


def test_a_reconciliation_tolerance_is_honoured_when_stated(book, clean):
    _, state, _ = book
    drifted = dict(state.broker_positions)
    drifted[clean.symbol] = drifted[clean.symbol] - 5.0
    caps = replace(state.caps, reconcile_tol=10.0)
    findings = check_exposure(clean, replace(state, broker_positions=drifted, caps=caps))
    assert [f.code for f in findings] == ["within_caps"]


@pytest.mark.parametrize("cap,value,code", [
    ("name_cap", 0.001, "name_cap"),
    ("gross_cap", 0.05, "gross_cap"),
    ("net_cap", 0.001, "net_cap"),
])
def test_each_exposure_cap_fires_on_its_own(book, clean, cap, value, code):
    _, state, _ = book
    caps = replace(state.caps, **{cap: value})
    assert code in {f.code for f in check_exposure(clean, replace(state, caps=caps))}


def test_a_sector_cap_fires_and_an_unstated_one_is_not_unlimited(book, clean):
    _, state, _ = book
    sector = state.sectors[clean.symbol]
    caps = replace(state.caps, sector_caps={sector: 0.001})
    f = next(x for x in check_exposure(clean, replace(state, caps=caps))
             if x.code == "sector_cap")
    assert sector in f.message and f.evidence["sector_after"] > 0.001
    caps = replace(state.caps, sector_caps={"nothing-like-it": 0.5})
    codes = {x.code for x in check_exposure(clean, replace(state, caps=caps))}
    assert codes == {"no_sector_cap"}


@pytest.mark.parametrize("drop", ["positions", "marks", "equity", "broker_positions",
                                  "sectors"])
def test_exposure_blocks_on_an_absent_fact(book, clean, drop):
    _, state, _ = book
    findings = check_exposure(clean, replace(state, **{drop: None}))
    assert [f.code for f in findings] == ["missing_fact"]
    assert findings[0].evidence["fact"] == drop


def test_broker_positions_come_from_the_read_only_bridges_own_frame():
    frame = pd.DataFrame({"ts": [1, 2, 3], "symbol": ["A", "B", "A"], "qty": [10.0, -5.0, 4.0],
                          "avg_price": [1.0, 2.0, 3.0]})
    assert broker_positions_from(frame) == {"A": 14.0, "B": -5.0}
    with pytest.raises(TypeError, match="missing column"):
        broker_positions_from(pd.DataFrame({"symbol": ["A"]}))
    with pytest.raises(TypeError, match="positions DataFrame"):
        broker_positions_from({"A": 1.0})


def test_the_bridges_position_frame_is_accepted_verbatim():
    """The link is real: ExecutionRecord.positions goes straight in."""
    from fin_skills.bridges import execution as X
    rec = X.ExecutionRecord(fills=pd.DataFrame(), positions=X._positions_frame(
        [{"symbol": "AA00", "qty": 500.0, "avg_price": 10.0}], pd.Timestamp("2026-09-10",
                                                                            tz="UTC")),
        account={}, venue="ib", is_paper=True)
    assert broker_positions_from(rec.positions) == {"AA00": 500.0}


# ========================================================================= check_order
def test_a_clean_ticket_with_every_fact_assembled_is_allowed(book, clean):
    _, state, _ = book
    v = check_order(clean, state)
    assert isinstance(v, OrderVerdict) and v.allowed
    assert v.missing == () and v.blocks == ()
    assert v.checks == CHECKS
    assert v.render().isascii() and "ALLOWED" in v.render()


@pytest.mark.parametrize("cid", sorted(PLANTED))
def test_every_planted_mistake_is_blocked(book, cid):
    _, state, by_id = book
    v = check_order(by_id[cid], state)
    assert not v.allowed and v.blocks
    assert "BLOCKED" in v.render()


def test_the_three_mistakes_are_caught_by_three_different_checks(book):
    _, state, by_id = book
    caught = {cid: {f.check for f in check_order(by_id[cid], state).blocks}
              for cid in PLANTED}
    assert "fat_finger" in caught["slip-qty"] and "participation" in caught["slip-qty"]
    assert caught["slip-px"] == {"fat_finger"}
    assert caught["replay"] == {"duplicate"}


def test_only_the_three_are_blocked_on_the_seeded_blotter(book):
    proposals, state, _ = book
    blocked = {p.client_order_id for p in proposals if not check_order(p, state).allowed}
    assert blocked == set(PLANTED), "a clean ticket blocked is a measured false positive"


@pytest.mark.parametrize("drop", ["now", "sessions", "venue", "last_trade", "day_low",
                                  "day_high", "adv_shares", "marks", "positions",
                                  "broker_positions", "sectors", "equity", "already_sent"])
def test_an_absent_fact_is_never_a_pass(book, clean, drop):
    _, state, _ = book
    v = check_order(clean, replace(state, **{drop: None}))
    assert not v.allowed, f"dropping {drop} still returned allowed=True"
    assert v.missing, f"dropping {drop} did not name the missing fact"
    assert all("missing_fact" == f.code for f in v.blocks)


def test_the_empty_state_names_what_it_is_missing(clean):
    v = check_order(clean, TradingState())
    assert not v.allowed
    assert len(v.missing) >= 6
    # the evaluation time, a price, a volume, the book, a threshold and the broker's side
    assert {"now", "equity", "caps.max_notional", "broker_positions",
            "already_sent"} <= set(v.missing)
    assert any(m.startswith("adv_shares[") for m in v.missing)


def test_check_order_refuses_bad_arguments(book, clean):
    _, state, _ = book
    with pytest.raises(TypeError, match="ProposedOrder"):
        check_order({"symbol": "A"}, state)
    with pytest.raises(TypeError, match="TradingState"):
        check_order(clean, {"now": 1})
    with pytest.raises(ValueError, match="unknown check"):
        check_order(clean, state, ["fat_finger", "vibes"])


def test_a_subset_of_checks_runs_only_that_subset(book):
    _, state, by_id = book
    v = check_order(by_id["replay"], state, ["fat_finger"])
    assert v.allowed and v.checks == ("fat_finger",)


# =================================================================== the seeded blotter
def test_the_blotter_is_reproducible(book):
    proposals, state, _ = book
    again, again_state = seeded_blotter()
    assert [p.client_order_id for p in again] == [p.client_order_id for p in proposals]
    assert [p.qty for p in again] == [p.qty for p in proposals]
    assert again_state.adv_shares == state.adv_shares


def test_the_blotter_carries_exactly_three_planted_mistakes(book):
    proposals, state, _ = book
    assert len(proposals) == 40 and len(PLANTED) == 3
    assert len({p.client_order_id for p in proposals}) == 40
    assert len(state.already_sent) == 29
    assert set(state.already_sent.columns) == set(P.SENT_COLUMNS)


def test_the_threshold_sweep_measures_one_threshold_at_a_time(book):
    proposals, state, _ = book
    px = threshold_sweep(proposals, state, "max_price_through", [0.05, 0.20])
    assert list(px["planted"]) == [1, 0], "0.05 catches the price typo, 0.20 misses it"
    assert list(px["clean"]) == [0, 0]
    adv = threshold_sweep(proposals, state, "max_adv_multiple", [0.25, 0.30])
    assert list(adv["planted"]) == [1, 0]
    tight = threshold_sweep(proposals, state, "max_notional", [2.5e5])
    assert int(tight["clean"].iloc[0]) > 0, "a tight cap must show its false positives"
    with pytest.raises(ValueError, match="cap must be one of"):
        threshold_sweep(proposals, state, "gross_cap", [1.0])


def test_catching_the_replay_with_a_size_cap_would_cost_the_whole_book(book):
    """The point of having a duplicate check at all: the replay ticket is correctly sized
    and correctly priced, so the only size cap that sees it also sees most of the clean
    37. Multiplicity is not a magnitude and no threshold on magnitude can find it."""
    proposals, state, by_id = book
    rep = by_id["replay"]
    findings = check_fat_finger(rep, state) + check_participation(rep, state)
    assert not [f for f in findings if f.severity == "block"]

    tight = replace(state, caps=replace(state.caps, max_notional=rep.qty * rep.limit_price
                                        * 0.999))
    assert any(f.code == "notional_cap" for f in check_fat_finger(rep, tight))
    collateral = sum(1 for p in proposals if p.client_order_id not in PLANTED
                     and any(f.severity == "block" for f in check_fat_finger(p, tight)))
    assert collateral >= 20, "a cap tight enough to see the replay stops the book"


def test_example_caps_state_a_number_for_every_threshold():
    for f in Caps.__dataclass_fields__:
        assert getattr(EXAMPLE_CAPS, f) is not None, f
    assert Caps().max_notional is None, "the type itself must have no default threshold"


def test_the_declared_venue_hours_are_the_ones_that_were_verified():
    assert (NYSE_HOURS.continuous_open, NYSE_HOURS.continuous_close) == (dt.time(9, 30),
                                                                        dt.time(16, 0))
    assert NYSE_HOURS.half_day_close == dt.time(13, 0)
    assert NYSE_HOURS.closing_auction == (dt.time(15, 50), dt.time(16, 0))
    assert SSE_HOURS.lunch == (dt.time(11, 30), dt.time(13, 0))
    assert set(P.VENUE_HOURS) == {"NYSE", "SSE"}


# =============================================================================== the guard
def test_the_guard_is_registered_and_owned_by_this_skill():
    g = api.get("pre_trade")
    assert g.skill == "pre-trade-checks"
    assert set(g.required) == {"proposal", "state"}
    assert "fin_skills.core.pre_trade.check_order" in g.wraps


def test_the_guard_passes_the_clean_ticket_and_fails_each_planted_one(book, clean):
    _, state, by_id = book
    g = api.get("pre_trade")
    r = g.run(proposal=clean, state=state)
    assert r.passed and r.errors == [] and r.evidence["allowed"] is True
    assert r.summary().startswith("PASS") and r.summary().isascii()
    for cid in PLANTED:
        bad = g.run(proposal=by_id[cid], state=state)
        assert not bad.passed and bad.errors
        assert bad.evidence["blocks"] and bad.evidence["allowed"] is False


def test_the_guard_fails_closed_on_an_absent_fact(book, clean):
    _, state, _ = book
    r = api.get("pre_trade").run(proposal=clean, state=replace(state, adv_shares=None))
    assert not r.passed
    assert r.evidence["missing"] and any("adv_shares" in m for m in r.evidence["missing"])


def test_the_guard_takes_a_mapping_for_the_proposal_but_not_for_the_state(book, clean):
    _, state, _ = book
    g = api.get("pre_trade")
    as_map = dict(symbol=clean.symbol, side=clean.side, qty=clean.qty,
                  limit_price=clean.limit_price, client_order_id=clean.client_order_id,
                  ts=clean.ts, sector=clean.sector)
    assert g.run(proposal=as_map, state=state).passed
    with pytest.raises(TypeError, match="assembling it is the point"):
        g.run(proposal=clean, state={"now": state.now})
    with pytest.raises(TypeError, match="does not build a ProposedOrder"):
        g.run(proposal={"symbol": "A"}, state=state)


def test_the_guard_is_deliberately_not_callable_over_json():
    from fin_skills.tools.schema import excluded
    exc = excluded()["pre_trade"]
    assert "proposal" in dict(exc.inputs) and "state" in dict(exc.inputs)
    assert "get('pre_trade')" in exc.reason


def test_the_guard_is_skipped_rather_than_guessed_at_by_run_all():
    report = api.run_all(returns=pd.Series([0.01, -0.01]), turnover=0.05)
    assert "pre_trade" in report.skipped
    assert set(report.skipped["pre_trade"]) == {"proposal", "state"}


# ================================================================================ the demo
def test_the_demo_reproduces_every_measured_number(run_main):
    out = run_main("fin_skills.core.pre_trade")
    assert out.isascii()
    assert "3 of 40 blocked" in out
    assert "clean tickets blocked (false positives): []" in out
    for cid in PLANTED:
        assert cid in out
    assert "allowed=True" in out and "allowed=False" in out
    assert "TRIPPED" in out
    assert P.THE_RULE.split(":")[0] in out
    assert "no wall clock is read anywhere in this module" in out
