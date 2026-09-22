"""Mature, whole-option account credit; independent of the biological simulator.

This is an engineering component, not a fruit-fly learning rule. The upstream
accounting system must supply independently reconciled, cost-inclusive NAV marks.
The gate verifies clocks, lineage, continuity and the cumulative return identity;
it cannot establish that fabricated source marks reflect real executions.
"""
from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from fin_skills.api import get
from fin_skills.synthesis import Fact, Timeline


@dataclass(frozen=True)
class NavMark:
    event_time: pd.Timestamp
    available_at: pd.Timestamp
    nav: float
    source: str

    def __post_init__(self):
        object.__setattr__(self,"event_time",pd.Timestamp(self.event_time))
        object.__setattr__(self,"available_at",pd.Timestamp(self.available_at))
        if pd.isna(self.event_time) or pd.isna(self.available_at):
            raise ValueError("Missing NAV clock")
        if self.available_at < self.event_time:
            raise ValueError("NAV cannot be available before its event")
        if not np.isfinite(self.nav) or self.nav <= 0 or not self.source:
            raise ValueError("A positive finite NAV and named source are required")

    def fact(self, option_id):
        return Fact("net_nav",f"internal:{option_id}",self.nav,self.event_time,
                    self.available_at,self.available_at,source=self.source)


@dataclass(frozen=True)
class IntervalCredit:
    before: NavMark
    after: NavMark
    claimed_log_return: float


@dataclass(frozen=True)
class OptionReceipt:
    option_id: str
    decision_time: pd.Timestamp
    intervals: tuple[IntervalCredit,...]
    claimed_available: pd.Timestamp


def audit_option(receipt, now):
    """Return a training target only after all source clocks strictly precede now.

    Target covers the complete option, including its entry costs and every later
    HOLD interval. This is a gamma=1 Monte Carlo target, without a bootstrap term.
    Account continuity prevents costs or return intervals being counted twice.
    Invalid receipts can be repaired and resubmitted; losses are valid targets.
    """
    now = pd.Timestamp(now)
    decision = pd.Timestamp(receipt.decision_time)
    claimed = pd.Timestamp(receipt.claimed_available)
    if pd.isna(now) or pd.isna(decision) or pd.isna(claimed):
        raise ValueError("Missing receipt clock")
    if not receipt.option_id or not receipt.intervals:
        return {"status":"invalid","reason":"empty episode","target":None}
    intervals = receipt.intervals
    if decision >= intervals[0].before.event_time:
        return {"status":"invalid","reason":"execution must follow decision","target":None}
    for i,interval in enumerate(intervals):
        if interval.after.event_time <= interval.before.event_time:
            return {"status":"invalid","reason":"nonpositive interval","target":None}
        if i and interval.before != intervals[i-1].after:
            return {"status":"invalid","reason":"gap, overlap or changed account boundary","target":None}
        expected = math.log(interval.after.nav/interval.before.nav)
        if not np.isfinite(interval.claimed_log_return) or not math.isclose(
                interval.claimed_log_return,expected,rel_tol=1e-10,abs_tol=1e-12):
            return {"status":"invalid","reason":"interval accounting mismatch","target":None}
    target = math.fsum(i.claimed_log_return for i in intervals)
    endpoint = math.log(intervals[-1].after.nav/intervals[0].before.nav)
    if not math.isclose(target,endpoint,rel_tol=1e-10,abs_tol=1e-12):
        return {"status":"invalid","reason":"whole-option accounting mismatch","target":None}
    marks = [intervals[0].before]+[i.after for i in intervals]
    parents = tuple(mark.fact(receipt.option_id) for mark in marks)
    if claimed < marks[-1].event_time:
        return {"status":"invalid","reason":"receipt precedes episode end","target":None}
    derived = Fact("option_log_return",f"internal:{receipt.option_id}",target,
                   marks[-1].event_time,claimed,claimed,kind="derived",inputs=parents,
                   source="whole_option_credit")
    guard = get("synthesis_integrity").run(timeline=Timeline([*parents,derived]))
    if not guard.passed:
        return {"status":"invalid","reason":"library lineage/clock check failed",
                "guard":guard.guard,"target":None}
    actual_available = max([claimed]+[m.available_at for m in marks])
    if actual_available >= now:
        return {"status":"pending","reason":"feedback not available before decision",
                "guard":guard.guard,"target":None}
    return {"status":"ready","target":target,"guard":guard.guard,
            "origin":decision.isoformat(),"available_at":actual_available.isoformat(),
            "start_nav":marks[0].nav,"end_nav":marks[-1].nav,
            "intervals":len(intervals)}


class CreditGate:
    """Exactly-once consumption after successful validation and maturation."""
    def __init__(self):
        self.applied = set()

    def consume(self, receipt, now):
        if receipt.option_id in self.applied:
            return {"status":"duplicate","target":None}
        result = audit_option(receipt,now)
        if result["status"] == "ready":
            self.applied.add(receipt.option_id)
        return result
