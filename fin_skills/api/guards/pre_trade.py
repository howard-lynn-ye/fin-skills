"""Guard: check an order BEFORE a person sends it (pre-trade-checks / pre_trade.py).

Deliberately NOT exported as a JSON tool. `proposal` and `state` are Python objects with
no JSON representation, so `fin_skills.tools.schema` puts this guard in `excluded()` and
says why. That is the intended outcome, not an oversight: the facts a pre-trade gate rests
on - a declared calendar, an ADV, the broker's own position - have to be ASSEMBLED by the
caller from things it read, and a model composing them into a JSON blob is precisely the
failure this guard exists to make impossible. Call it from Python:

    from fin_skills.api import get
    from fin_skills.core.pre_trade import ProposedOrder, TradingState
    r = get("pre_trade").run(proposal=ProposedOrder(...), state=TradingState(...))
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from fin_skills.api.base import Guard, Outcome, register
from fin_skills.core.pre_trade import (CHECKS, OrderVerdict, ProposedOrder, TradingState,
                                       check_order)

_SEVERITY = {"block": "error", "warn": "warning", "info": "info"}


@register
class PreTradeGuard(Guard):
    """Every pre-trade check over one PROPOSED order. It checks; it cannot send.

    Inputs
        proposal : a `ProposedOrder` (symbol, side, qty, limit_price, client_order_id,
                   tz-aware ts, sector), or a mapping of those fields.
        state    : a `TradingState` - the facts, assembled: the declared `Sessions`
                   calendar and `now`, the venue's declared hours, last trade, day range,
                   ADV, marks, your positions, the BROKER's positions read back through
                   `fin_skills.bridges.execution`, sectors, equity, the blotter of what
                   has already gone out, the stated `Caps` and the `KillSwitch`.
        checks   : which checks to run; defaults to all six.

    Fails when any check blocks AND when any required fact is ABSENT - the two carry the
    same weight on purpose. An unknown ADV is not a pass, an undeclared calendar is not a
    pass, and a position nobody reconciled is not a pass. A tripped kill switch fails
    every check regardless of what else is in the state.

    This guard runs AFTER `paper_account_guard` (broker-execution-apis), never instead of
    it: proving the session is paper is a different question from whether the order is
    sane, and only one of them is about the connection.
    """

    name = "pre_trade"
    skill = "pre-trade-checks"
    summary = ("Runs every pre-trade check over a proposed order and fails closed when a "
               "check blocks or a required fact is absent. Checks orders; never sends one.")
    wraps = ("fin_skills.core.pre_trade.check_order",
             "fin_skills.core.pre_trade.check_fat_finger",
             "fin_skills.core.pre_trade.check_participation",
             "fin_skills.core.pre_trade.check_session",
             "fin_skills.core.pre_trade.check_duplicate",
             "fin_skills.core.pre_trade.check_exposure",
             "fin_skills.core.pre_trade.trip_if")
    required = ("proposal", "state")
    optional = ("checks",)

    def check(self, proposal: ProposedOrder, state: TradingState,
              checks: Sequence[str] = CHECKS) -> Outcome:
        out = Outcome()
        if isinstance(proposal, Mapping):
            try:
                proposal = ProposedOrder(**proposal)
            except TypeError as exc:
                raise TypeError(f"proposal mapping does not build a ProposedOrder: {exc}") from exc
        if not isinstance(proposal, ProposedOrder):
            raise TypeError("proposal must be a ProposedOrder (or a mapping of its fields), "
                            f"got {type(proposal).__name__}")
        if not isinstance(state, TradingState):
            raise TypeError(
                "state must be a TradingState, not a mapping: assembling it is the point. "
                "Build it from what you READ - a declared Sessions calendar, an ADV, the "
                "broker's own positions - so that a fact you never had is visible as None "
                "and fails the verdict instead of quietly defaulting")
        if isinstance(checks, str):
            raise TypeError("checks must be a sequence of check names, not a single string")

        verdict: OrderVerdict = check_order(proposal, state, tuple(checks))
        for f in verdict.findings:
            out.add(_SEVERITY[f.severity], f.message, where=f"{f.check}/{f.code}")
        out.note(verdict=verdict, allowed=verdict.allowed, missing=list(verdict.missing),
                 codes=list(verdict.codes), blocks=[f.code for f in verdict.blocks],
                 checks=list(verdict.checks), table=verdict.render())
        if verdict.allowed:
            out.info(f"{proposal.side} {proposal.qty:,.0f} {proposal.symbol} clears all "
                     f"{len(verdict.checks)} checks. A person still has to send it; nothing "
                     f"in this library can", where="verdict")
        return out
