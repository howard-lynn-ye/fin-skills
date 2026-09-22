from dataclasses import replace
import math

import pandas as pd
import pytest

from benchmarks.verified_memory.episode_credit import (
    CreditGate, IntervalCredit, NavMark, OptionReceipt, audit_option,
)


def receipt(values=(1.,.998,1.005984),delay=0):
    times = pd.date_range("2025-01-01",periods=len(values)+2,freq="D")
    marks = [NavMark(times[i+1],times[i+1]+pd.Timedelta(days=delay),nav,"independent_ledger")
             for i,nav in enumerate(values)]
    intervals = tuple(IntervalCredit(a,b,math.log(b.nav/a.nav)) for a,b in zip(marks,marks[1:]))
    return OptionReceipt("episode-1",times[0],intervals,marks[-1].available_at)


def test_entry_cost_is_credited_with_later_hold_return():
    r = receipt()
    out = audit_option(r,"2025-02-01")
    assert r.intervals[0].claimed_log_return < 0
    assert out["status"] == "ready" and out["target"] > 0
    assert out["target"] == pytest.approx(math.log(1.005984),abs=1e-12)


def test_valid_loss_is_preserved_without_clipping():
    out = audit_option(receipt((1.,.95,.8)),"2025-02-01")
    assert out["status"] == "ready"
    assert out["target"] == pytest.approx(math.log(.8))


def test_pending_receipt_releases_no_target_then_is_consumed_once():
    r = receipt(delay=3)
    gate = CreditGate()
    for now in ("2025-01-03",r.claimed_available):
        result = gate.consume(r,now)
        assert result["status"] == "pending" and result["target"] is None
    assert gate.consume(r,"2025-02-01")["status"] == "ready"
    assert gate.consume(r,"2025-02-02")["status"] == "duplicate"


def test_backdating_is_rejected_by_actual_library_guard():
    r = receipt(delay=3)
    r = replace(r,claimed_available=r.intervals[-1].after.event_time)
    out = audit_option(r,"2025-02-01")
    assert out["status"] == "invalid" and out["target"] is None
    assert out["guard"] == "synthesis_integrity"


@pytest.mark.parametrize("defect",["gap","duplicate","tamper"])
def test_bad_credit_cannot_be_applied(defect):
    r = receipt()
    a,b = r.intervals
    if defect == "gap":
        shifted=b.before.event_time+pd.Timedelta(hours=1)
        b=replace(b,before=replace(b.before,event_time=shifted,available_at=shifted))
        intervals=(a,b)
    elif defect == "duplicate": intervals=(a,a,b)
    else: intervals=(a,replace(b,claimed_log_return=b.claimed_log_return+.1))
    gate=CreditGate()
    assert gate.consume(replace(r,intervals=intervals),"2025-02-01")["status"] == "invalid"
    assert gate.consume(r,"2025-02-01")["status"] == "ready"


def test_newer_record_changes_cannot_change_an_already_released_target():
    r = receipt()
    gate = CreditGate()
    initial = gate.consume(r,"2025-02-01")
    changed = receipt((1.,.998,2.))
    assert initial["target"] == pytest.approx(math.log(1.005984))
    assert gate.consume(changed,"2025-02-02")["status"] == "duplicate"


def test_same_timestamp_fill_is_not_allowed():
    r = receipt()
    assert audit_option(replace(r,decision_time=r.intervals[0].before.event_time),
                        "2025-02-01")["status"] == "invalid"
