#!/usr/bin/env python3
"""What the availability clock costs when you get it wrong, measured on a seeded panel.

The rule, in one line:

    a combined fact is knowable only when its LAST input is
    -> available_at = max(inputs), never min(inputs) and never the date the join ran

Why `min` is the seductive error: an earnings yield is a price DIVIDED BY a fundamental.
The price is known at t. The fundamental describes a quarter that ended weeks ago. Join
them on the fundamental's PERIOD END and every column of the output row is populated on
date t - there is a price in it, there is an EPS in it, nothing is NaN, no warning fires.
The row is a fact about t in every respect except the one that matters: half of it was
published a month and a half later.

This script builds a panel where that difference is the ONLY difference, and measures it.

    * 40 names, 1,512 sessions, quarterly EPS.
    * Each quarter's number becomes knowable 30-75 calendar days after the period ends
      (the SEC's own range for a 10-Q), plus a session when it lands after the close.
    * The market prices each quarter in gradually over the 63 sessions FOLLOWING the
      period end - so most of the drift happens before the release, which is exactly
      what makes the period-end clock look profitable.
    * Three arms, identical in every other respect:
          period_end clock  = min(inputs): trade the number the day the quarter ends
          available_at clock= max(inputs): trade it the day it is published   <- the rule
          join-date clock   = the moment the research ran

Run:  python availability_clock.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SEED = 20260910
N_NAMES = 40
N_SESSIONS = 1512                 # about six years of business days
START = "2018-01-02"
DRIFT_SESSIONS = 63               # one quarter of gradual price discovery
KAPPA = 0.0006                    # daily drift per unit of standardised quarterly score
SIGMA = 0.014                     # daily idiosyncratic vol
MIN_LAG_DAYS, MAX_LAG_DAYS = 30, 75
AFTER_CLOSE_SHARE = 0.4           # filings that land after the close and trade a day later
ANNUAL = 252


# --------------------------------------------------------------------------- the rule
def combined_available_at(available_ats) -> pd.Timestamp:
    """The clock a combined fact is allowed to carry: the LATEST of its inputs."""
    stamps = [pd.Timestamp(t) for t in available_ats]
    if not stamps:
        raise ValueError("a combined fact with no inputs has no clock to inherit")
    return max(stamps)


def check_combined_clock(claimed, available_ats) -> None:
    """Raise when a combined fact claims to be knowable before its last input was."""
    need = combined_available_at(available_ats)
    claimed = pd.Timestamp(claimed)
    if claimed < need:
        raise ValueError(
            f"available_at={claimed.date()} is earlier than its last input ({need.date()}); "
            f"a combined fact is knowable only when its LAST input is, and taking the "
            f"earliest backdates {(need - claimed).days} day(s) of information")


# ------------------------------------------------------------------------ the panel
def quarter_ends(sessions: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Calendar quarter ends spanning the panel, plus one before it starts."""
    lo = sessions.min() - pd.Timedelta(days=200)
    return pd.date_range(lo, sessions.max(), freq="QE")


def build_panel(seed: int = SEED) -> dict:
    """A seeded panel of returns plus the quarterly facts that drive them.

    Returns a dict with `sessions`, `names`, `returns` (sessions x names), `prices`, and
    `facts`: one row per (name, quarter) with period_end, filed_at, available_at and the
    EPS the market eventually learns.
    """
    rng = np.random.default_rng(seed)
    sessions = pd.bdate_range(START, periods=N_SESSIONS)
    names = [f"N{i:02d}" for i in range(N_NAMES)]
    qends = quarter_ends(sessions)

    rows = []
    drift = pd.DataFrame(0.0, index=sessions, columns=names)
    pos_of = np.arange(len(sessions))
    for q in qends:
        after = pos_of[sessions > q]
        window = after[:DRIFT_SESSIONS]
        for name in names:
            score = float(rng.normal())
            lag = int(rng.integers(MIN_LAG_DAYS, MAX_LAG_DAYS + 1))
            filed = q + pd.Timedelta(days=lag)
            after_close = bool(rng.random() < AFTER_CLOSE_SHARE)
            available = filed + pd.Timedelta(days=1 if after_close else 0)
            rows.append({"name": name, "period_end": q, "filed_at": filed,
                         "available_at": available, "score": score,
                         "eps": 5.0 + 1.5 * score, "after_close": after_close})
            if len(window):
                drift.iloc[window, names.index(name)] += KAPPA * score

    noise = pd.DataFrame(rng.normal(0.0, SIGMA, (len(sessions), len(names))),
                         index=sessions, columns=names)
    returns = drift + noise
    prices = 100.0 * np.exp(returns.cumsum())
    facts = pd.DataFrame(rows).sort_values(["name", "period_end"]).reset_index(drop=True)
    return {"sessions": sessions, "names": names, "returns": returns, "prices": prices,
            "facts": facts}


# ------------------------------------------------------------------------- the clocks
def eps_panel(facts: pd.DataFrame, sessions: pd.DatetimeIndex, clock: str,
              join_date=None) -> pd.DataFrame:
    """The EPS knowable on each session under one clock. An as-of filter, never a fill.

    clock='period_end'   the number is used from the day its quarter ENDS (min(inputs))
    clock='available_at' the number is used from the day it is PUBLISHED (max(inputs))
    clock='join_date'    every number is stamped with the moment the research ran
    """
    if clock not in ("period_end", "available_at", "join_date"):
        raise ValueError(f"clock must be period_end, available_at or join_date, "
                         f"got {clock!r}")
    out = pd.DataFrame(np.nan, index=sessions, columns=sorted(facts["name"].unique()))
    # the research ran the day after the sample ends - the usual case, and the one that
    # makes the failure total rather than nearly total
    stamp = (pd.Timestamp(join_date) if join_date is not None
             else sessions.max() + pd.Timedelta(days=1))
    for name, group in facts.groupby("name", sort=False):
        g = group.sort_values("period_end")
        if clock == "period_end":
            when = g["period_end"].to_numpy()
        elif clock == "available_at":
            when = g["available_at"].to_numpy()
        else:
            when = np.full(len(g), np.datetime64(stamp.to_datetime64()))
        pos = np.searchsorted(when.astype("datetime64[ns]"),
                              sessions.to_numpy().astype("datetime64[ns]"),
                              side="right") - 1
        vals = g["eps"].to_numpy(dtype=float)
        out[name] = np.where(pos >= 0, vals[np.clip(pos, 0, len(vals) - 1)], np.nan)
    return out


def signal_from(eps: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """The COMBINED fact: earnings yield = a fundamental over a price, cross-sectionally z."""
    ey = eps / prices
    z = ey.sub(ey.mean(axis=1), axis=0).div(ey.std(axis=1).replace(0.0, np.nan), axis=0)
    return z


def backtest(signal: pd.DataFrame, returns: pd.DataFrame) -> dict:
    """Dollar-neutral, unit-gross, one-session holding. Position at t earns return at t+1."""
    w = signal.fillna(0.0)
    gross = w.abs().sum(axis=1).replace(0.0, np.nan)
    w = w.div(gross, axis=0).fillna(0.0)
    pnl = (w.shift(1) * returns).sum(axis=1)
    pnl = pnl.iloc[1:]
    live = int((w.abs().sum(axis=1) > 0).sum())
    sd = float(pnl.std(ddof=1))
    sharpe = float(pnl.mean() / sd * np.sqrt(ANNUAL)) if sd > 0 else 0.0
    return {"sharpe": sharpe, "ann_return": float(pnl.mean() * ANNUAL),
            "ann_vol": float(sd * np.sqrt(ANNUAL)), "n_live_sessions": live,
            "n_sessions": int(len(pnl))}


# -------------------------------------------------------------------------- measure
def measure(seed: int = SEED) -> dict:
    """Every number this script reports, from one seeded panel."""
    panel = build_panel(seed)
    sessions, facts = panel["sessions"], panel["facts"]
    prices, returns = panel["prices"], panel["returns"]

    arms = {}
    for clock in ("period_end", "available_at", "join_date"):
        eps = eps_panel(facts, sessions, clock)
        arms[clock] = backtest(signal_from(eps, prices), returns)

    lag = (facts["available_at"] - facts["period_end"]).dt.days
    lag_sessions = lag * 5.0 / 7.0
    return {
        "n_names": len(panel["names"]), "n_sessions": len(sessions),
        "n_facts": int(len(facts)),
        "lag_min": int(lag.min()), "lag_median": float(lag.median()),
        "lag_max": int(lag.max()),
        "drift_before_release_pct": float(
            100.0 * min(1.0, lag_sessions.median() / DRIFT_SESSIONS)),
        "sharpe_period_end": arms["period_end"]["sharpe"],
        "sharpe_available_at": arms["available_at"]["sharpe"],
        "sharpe_join_date": arms["join_date"]["sharpe"],
        "sharpe_cost": arms["period_end"]["sharpe"] - arms["available_at"]["sharpe"],
        "sharpe_ratio": (arms["period_end"]["sharpe"] / arms["available_at"]["sharpe"]
                         if arms["available_at"]["sharpe"] else float("nan")),
        "ann_return_period_end": arms["period_end"]["ann_return"],
        "ann_return_available_at": arms["available_at"]["ann_return"],
        "live_period_end": arms["period_end"]["n_live_sessions"],
        "live_available_at": arms["available_at"]["n_live_sessions"],
        "live_join_date": arms["join_date"]["n_live_sessions"],
    }


def _print_report(m: dict) -> None:
    print("=" * 78)
    print("THE AVAILABILITY CLOCK - measured on a seeded panel")
    print("=" * 78)
    print(f"panel        : {m['n_names']} names x {m['n_sessions']} sessions, "
          f"{m['n_facts']} quarterly facts, seed {SEED}")
    print(f"filing lag   : {m['lag_min']}-{m['lag_max']} calendar days, median "
          f"{m['lag_median']:.0f} days from period end to knowable")
    print(f"             : about {m['drift_before_release_pct']:.0f}% of the "
          f"{DRIFT_SESSIONS}-session drift window is OVER by the median release")
    print()
    print("  clock                       what it means                    Sharpe   ann ret")
    print("  " + "-" * 76)
    print(f"  period_end  (= min inputs)  trade it when the quarter ends "
          f"{m['sharpe_period_end']:8.2f}  {m['ann_return_period_end']:7.1%}")
    print(f"  available_at(= max inputs)  trade it when it is published  "
          f"{m['sharpe_available_at']:8.2f}  {m['ann_return_available_at']:7.1%}")
    print(f"  join_date                   stamp it with the research run "
          f"{m['sharpe_join_date']:8.2f}  {0.0:7.1%}")
    print()
    print(f"MEASURED: taking the EARLIEST input clock costs "
          f"{m['sharpe_cost']:+.2f} Sharpe of pure look-ahead "
          f"({m['sharpe_period_end']:.2f} vs {m['sharpe_available_at']:.2f}, "
          f"{m['sharpe_ratio']:.1f}x).")
    print(f"MEASURED: the join-date clock leaves {m['live_join_date']} of "
          f"{m['n_sessions']} sessions with a signal (the correct clock leaves "
          f"{m['live_available_at']}) - it fails SILENTLY, as an empty backtest, "
          f"not as an error.")
    print()

    print("-" * 78)
    print("THE SAME MISTAKE, ONE FACT AT A TIME")
    print("-" * 78)
    price_known = pd.Timestamp("2024-02-02")
    filing_known = pd.Timestamp("2024-03-18")
    print(f"  price     close = 189.40   knowable {price_known.date()}")
    print(f"  filing    eps   =   6.42   period ended 2023-12-30, knowable "
          f"{filing_known.date()}")
    print(f"  combined  earnings_yield = eps / close")
    print(f"    min(inputs) -> {price_known.date()}   the row LOOKS complete: a price, "
          f"an eps, no NaN")
    print(f"    max(inputs) -> {filing_known.date()}   the rule")
    try:
        check_combined_clock(price_known, [price_known, filing_known])
    except ValueError as exc:
        print(f"    REFUSED: {exc}")
    print(f"    accepted: available_at="
          f"{combined_available_at([price_known, filing_known]).date()}")
    print()
    print("=" * 78)
    print("RULE: a combined fact is knowable only when its LAST input is - "
          "available_at = max(inputs), never min and never the join date.")
    print("=" * 78)


if __name__ == "__main__":
    _print_report(measure())
