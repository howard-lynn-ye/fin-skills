"""fin_skills.engine.invariants - the twelve claims, asserted as the run happens.

Each entry names the property and the test in tests/test_engine.py that proves it. Six of
the twelve are proven by running a guard this repo already ships on the engine's own
output rather than by a bespoke assertion, because anything the engine claims that no
guard can falsify is not worth claiming.
"""
from __future__ import annotations

import pandas as pd

from fin_skills.api.base import Finding
from fin_skills.core.adjustment_check import detect_convention

# id -> (what holds, the test that proves it, the guard that proves it or "-")
INVARIANTS: dict[str, tuple[str, str, str]] = {
    "I1": ("no output cell before bar k moves when panel rows >= k are perturbed",
           "test_engine_is_causal", "assert_causal"),
    "I2": ("a signal from bar t can first fill on bar t+signal_lag, and signal_lag >= 1",
           "test_signal_lag_zero_is_rejected", "-"),
    "I3": ("no same-bar fill: every fill is dated strictly after its decision bar",
           "test_no_same_bar_fill", "-"),
    "I4": ("the adjustment convention is declared and consistent with the action table",
           "test_forward_adjusted_panel_labelled_back_is_rejected", "adjustment_check"),
    "I5": ("a delisting is a booked closeout, not a NaN, and the name never reappears",
           "test_delisting_closes_out", "survivorship_audit"),
    "I6": ("universe[d] depends only on panel rows <= d",
           "test_universe_is_point_in_time", "pit_universe"),
    "I7": ("turnover[t] == |weights[t] - weights[t-1]|.sum()",
           "test_turnover_identity", "-"),
    "I8": ("costs are subtracted once: net(0) is gross, and net(b) is non-increasing in b",
           "test_cost_curve_is_monotone", "cost_curve"),
    "I9": ("every result index is a subset of sessions.index; windows count BARS",
           "test_calendar_is_declared_not_inferred", "-"),
    "I10": ("no fill exceeds participation_cap * volume; the remainder carries",
            "test_participation_cap_carries_the_remainder", "-"),
    "I11": ("cash is a position: held weights plus the cash weight are exactly 1",
            "test_cash_is_a_position_and_earns_the_declared_rate", "rf_convention"),
    "I12": ("run() is deterministic and holds no state across folds",
            "test_engine_is_deterministic", "fold_leak_test"),
}

TOL = 1e-9


def _err(inv: str, message: str) -> Finding:
    return Finding("error", message, where=f"{inv} ({INVARIANTS[inv][1]})")


def before(panel, execution, signal) -> list[Finding]:
    """I4 and I9, checked before a single bar runs."""
    out: list[Finding] = []
    idx = pd.DatetimeIndex(getattr(signal, "index", panel.sessions.index))
    stray = idx.difference(panel.sessions.index)
    if len(stray):
        out.append(_err("I9", f"the signal carries {len(stray)} row(s) that are not sessions on "
                              f"the declared calendar, first {stray[0].date()}"))
    if panel.actions is not None and len(panel.actions):
        expected = panel.DETECTED_AS[panel.adjustment]
        counts = panel.actions["ticker"].value_counts()
        for ticker in [t for t in counts.index if t in panel.close.columns][:1]:
            acts = panel.actions[panel.actions["ticker"] == ticker]
            got = detect_convention(panel.close[ticker].dropna(), acts[["date", "ratio", "kind"]])
            if got["convention"] not in (expected, "unknown"):
                out.append(_err("I4", f"panel declares adjustment={panel.adjustment!r} "
                                      f"({expected}) but {ticker} reads as "
                                      f"{got['convention']}: {got['reason']}"))
    return out


def after(result) -> list[Finding]:
    """I3, I7, I10 and I11, checked against the run's own output."""
    out: list[Finding] = []
    w = result.weights
    traded = result.fills[result.fills["kind"] == "trade"]   # a closeout is an event, not a decision
    if len(traded):
        early = traded[traded["date"] <= traded["decided_on"]]
        if len(early):
            out.append(_err("I3", f"{len(early)} fill(s) are dated on or before their decision "
                                  f"bar - that is the vectorbt default, not an execution model"))
        cap = result.spec.get("participation_cap")
        if cap is not None:
            room = result.panel.volume.stack(future_stack=True)
            got = traded.set_index(["date", "ticker"])["shares"]
            over = got[got > room.reindex(got.index).fillna(0.0) * float(cap) + TOL]
            if len(over):
                out.append(_err("I10", f"{len(over)} fill(s) exceed participation_cap="
                                       f"{cap:g} of the fill bar's volume"))
    identity = w.diff().abs().sum(axis=1)
    identity.iloc[0] = float(w.iloc[0].abs().sum())
    gap = float((identity - result.turnover).abs().max())
    if gap > 1e-12:
        out.append(_err("I7", f"turnover is not |dw|.sum(): max gap {gap:.3g}"))
    book = (result.positions * result.panel.close.ffill()).sum(axis=1) + result.cash
    closes = float((book / result.equity - 1.0).abs().max())
    if closes > 1e-10:
        out.append(_err("I11", f"the book does not close: positions + cash differs from equity "
                               f"by up to {closes:.3g} of equity"))
    return out


def enforce(findings: list[Finding], strict: bool) -> list[Finding]:
    """strict=True raises on any error; strict=False hands them back as findings."""
    errors = [f for f in findings if f.severity == "error"]
    if strict and errors:
        raise ValueError("engine invariant violated (strict=True):\n  - "
                         + "\n  - ".join(str(f) for f in errors))
    return findings


def describe() -> str:
    lines = ["fin_skills.engine invariants - the claim, the test, the guard that falsifies it"]
    for key, (what, test, guard) in INVARIANTS.items():
        lines.append(f"  {key:<4} {what}")
        lines.append(f"       proven by {test}" + (f" via {guard}" if guard != "-" else ""))
    return "\n".join(lines)


__all__ = ["INVARIANTS", "after", "before", "describe", "enforce"]
