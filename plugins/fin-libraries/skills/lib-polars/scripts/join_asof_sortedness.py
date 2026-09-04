#!/usr/bin/env python3
"""`join_asof(..., by=)` cannot check sortedness, so unsorted input returns FUTURE quotes.

polars 1.44.1 does NOT raise when `by=` is present. It emits a UserWarning
    "Sortedness of columns cannot be checked when 'by' groups provided"
and returns rows anyway. The warning fires on EVERY `by=` call — including
correctly sorted ones — so it carries no information, gets filtered out early in
every codebase, and is not a guard. Without `by=` polars DOES raise
`InvalidOperationError`, which is why the failure only shows up in the per-symbol
quote join, i.e. the one everybody actually writes.

The mechanism is a per-group cursor that only moves forward. Feed it a signal
timestamp EARLIER than the one before it and the cursor cannot rewind, so the
signal is matched against a quote from the future. In a backtest that is lookahead
bias with a clean-looking join and no error.

Run:  python join_asof_sortedness.py
polars is optional. Without it this reproduces both the correct as-of join and the
cursor bug in numpy/pandas, so the demonstration still runs.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

SYMS = ("AAPL", "MSFT", "NVDA")


# --- reference implementations, numpy/pandas only ---------------------------

def asof_backward_correct(left: pd.DataFrame, right: pd.DataFrame,
                          on: str, by: str, val: str) -> pd.Series:
    """Correct per-group backward as-of. Sorts the right side itself and uses
    searchsorted, so the answer does not depend on the order of either frame."""
    out = np.full(len(left), np.nan)
    for g, idx in right.groupby(by).groups.items():
        r = right.loc[idx].sort_values(on)
        rt, rv = r[on].values, r[val].values
        mask = (left[by] == g).values
        pos = np.searchsorted(rt, left.loc[mask, on].values, side="right") - 1
        out[mask] = np.where(pos >= 0, rv[np.clip(pos, 0, None)], np.nan)
    return pd.Series(out, index=left.index)


def asof_backward_cursor(left: pd.DataFrame, right: pd.DataFrame,
                         on: str, by: str, val: str) -> pd.Series:
    """The algorithm a sorted-merge as-of join actually uses: one cursor per `by`
    group, advanced in the order rows arrive, NEVER rewound. Identical to the
    correct version on sorted input; silently wrong on anything else."""
    out = np.full(len(left), np.nan)
    groups = {g: right.loc[idx] for g, idx in right.groupby(by, sort=False).groups.items()}
    cursor: dict[object, int] = {}
    for i, (_, row) in enumerate(left.iterrows()):
        r = groups.get(row[by])
        if r is None:
            continue
        rt, rv = r[on].values, r[val].values
        c = cursor.get(row[by], -1)
        while c + 1 < len(rt) and rt[c + 1] <= row[on]:
            c += 1
        cursor[row[by]] = c
        out[i] = rv[c] if c >= 0 else np.nan
    return pd.Series(out, index=left.index)


# --- fixtures ---------------------------------------------------------------

def tiny_case():
    """Four signals, six quotes, two symbols. Small enough to check by eye."""
    quotes = pd.DataFrame({
        "time":   [1, 2, 3, 4, 5, 6],
        "symbol": ["A", "A", "A", "B", "B", "B"],
        "px":     [10.0, 11.0, 12.0, 20.0, 21.0, 22.0],
    })
    # B's signal at t=2 predates every B quote, so its only correct answer is null.
    signals = pd.DataFrame({
        "time":   [6, 3, 5, 2],
        "symbol": ["A", "A", "B", "B"],
        "sig":    [1, 2, 3, 4],
    })
    return signals, quotes


def panel(seed: int = 0, n_quotes: int = 400, n_signals: int = 150):
    """A seeded signal/quote panel, sorted within each symbol."""
    rng = np.random.default_rng(seed)
    qs, ss = [], []
    for s in SYMS:
        t = np.sort(rng.choice(np.arange(0, 2000), size=n_quotes, replace=False))
        qs.append(pd.DataFrame({"time": t, "symbol": s,
                                "px": 100 + np.cumsum(rng.normal(0, 0.5, n_quotes))}))
    for s in SYMS:
        t = np.sort(rng.choice(np.arange(0, 2000), size=n_signals, replace=False))
        ss.append(pd.DataFrame({"time": t, "symbol": s,
                                "sig": rng.normal(0, 1, n_signals)}))
    return (pd.concat(ss, ignore_index=True), pd.concat(qs, ignore_index=True))


def _eq(a, b) -> bool:
    an, bn = (a is None or (isinstance(a, float) and np.isnan(a))), \
             (b is None or (isinstance(b, float) and np.isnan(b)))
    if an or bn:
        return an and bn
    return abs(a - b) < 1e-12


def demo(seed: int = 0) -> dict:
    sig, qt = tiny_case()
    out = {
        "tiny_signals": sig,
        "tiny_quotes": qt,
        "tiny_cursor": asof_backward_cursor(sig, qt, "time", "symbol", "px"),
        "tiny_correct": asof_backward_correct(sig, qt, "time", "symbol", "px"),
    }
    out["tiny_diff"] = int(sum(not _eq(a, b) for a, b in
                               zip(out["tiny_cursor"], out["tiny_correct"])))

    # Scale: shuffle the left frame, which is all it takes.
    s2, q2 = panel(seed)
    shuffled = s2.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    cur = asof_backward_cursor(shuffled, q2, "time", "symbol", "px")
    cor = asof_backward_correct(shuffled, q2, "time", "symbol", "px")
    out["n_rows"] = len(shuffled)
    out["panel_diff"] = int(sum(not _eq(a, b) for a, b in zip(cur, cor)))
    out["panel_shuffled"] = shuffled
    out["panel_quotes"] = q2

    # If polars is installed, verify BOTH reference implementations against it and
    # record what 1.44.1 actually does rather than asserting it.
    try:
        import polars as pl

        out["LIVE_version"] = pl.__version__
        psig, pqt = pl.from_pandas(sig), pl.from_pandas(qt)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            live_unsorted = psig.join_asof(pqt, on="time", by="symbol",
                                           strategy="backward")["px"].to_list()
            out["LIVE_warnings"] = [str(x.message) for x in w]
        # The warning above fires on every `by=` call, so silence it from here on
        # rather than letting it bury the output it is supposedly warning about.
        warnings.filterwarnings("ignore", message="Sortedness of columns")
        live_sorted = (psig.sort(["symbol", "time"])
                       .join_asof(pqt.sort(["symbol", "time"]), on="time", by="symbol",
                                  strategy="backward")["px"].to_list())
        out["LIVE_tiny_unsorted"] = live_unsorted
        out["LIVE_tiny_sorted"] = live_sorted
        out["LIVE_cursor_matches"] = all(
            _eq(a, b) for a, b in zip(live_unsorted, out["tiny_cursor"]))
        srt = sig.sort_values(["symbol", "time"]).reset_index(drop=True)
        out["LIVE_correct_matches"] = all(
            _eq(a, b) for a, b in
            zip(live_sorted, asof_backward_correct(srt, qt, "time", "symbol", "px")))

        # Panel: carry the quote timestamp through and count look-ahead matches.
        pq = pl.from_pandas(q2).rename({"time": "quote_time"})
        j_bad = pl.from_pandas(shuffled).join_asof(
            pq, left_on="time", right_on="quote_time", by="symbol", strategy="backward")
        j_ok = pl.from_pandas(shuffled.sort_values(["symbol", "time"])
                              .reset_index(drop=True)).join_asof(
            pq, left_on="time", right_on="quote_time", by="symbol", strategy="backward")
        out["LIVE_lookahead_bad"] = j_bad.filter(
            pl.col("quote_time") > pl.col("time")).height
        out["LIVE_lookahead_ok"] = j_ok.filter(
            pl.col("quote_time") > pl.col("time")).height
        a = j_bad.sort(["symbol", "time"])["quote_time"].to_list()
        b = j_ok.sort(["symbol", "time"])["quote_time"].to_list()
        out["LIVE_panel_diff"] = sum(1 for x, y in zip(a, b) if x != y)
        out["LIVE_height_ok"] = (j_bad.height == len(shuffled))

        # What raises and what does not.
        matrix = []

        def probe(label, fn):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                try:
                    fn()
                    warned = "UserWarning" if w else "silent"
                    matrix.append((label, f"returns rows, {warned}"))
                except Exception as exc:  # noqa: BLE001 - the class is the finding
                    matrix.append((label, f"RAISES {type(exc).__name__}: {exc}"))

        probe("by=, unsorted left, eager",
              lambda: psig.join_asof(pqt, on="time", by="symbol"))
        probe("by=, unsorted left, check_sortedness=False",
              lambda: psig.join_asof(pqt, on="time", by="symbol", check_sortedness=False))
        probe("by=, unsorted left, lazy",
              lambda: psig.lazy().join_asof(pqt.lazy(), on="time", by="symbol").collect())
        probe("by=, unsorted left, streaming",
              lambda: psig.lazy().join_asof(pqt.lazy(), on="time", by="symbol")
              .collect(engine="streaming"))
        # A genuinely unsorted right frame: reverse it so `time` descends.
        pqt_rev = pqt.reverse()
        probe("by=, unsorted RIGHT, eager",
              lambda: psig.sort(["symbol", "time"]).join_asof(pqt_rev, on="time",
                                                              by="symbol"))
        probe("NO by=, unsorted left, eager",
              lambda: psig.drop("symbol").join_asof(pqt.drop("symbol").sort("time"),
                                                    on="time"))
        probe("NO by=, unsorted RIGHT, eager",
              lambda: psig.drop("symbol").sort("time").join_asof(
                  pqt_rev.drop("symbol"), on="time"))
        out["LIVE_matrix"] = matrix
        out["LIVE_heights"] = (j_bad.height, j_ok.height)
        import inspect

        out["LIVE_check_sortedness_default"] = inspect.signature(
            pl.DataFrame.join_asof).parameters["check_sortedness"].default
    except ImportError:
        pass
    return out


def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "null"
    return f"{v:g}"


if __name__ == "__main__":
    res = demo()
    sig, qt = res["tiny_signals"], res["tiny_quotes"]

    print("A signal-to-quote as-of join. Quotes, sorted (symbol, time):")
    print("    " + "  ".join(f"{s}@t{t}={p:g}" for t, s, p
                             in zip(qt["time"], qt["symbol"], qt["px"])))
    print("\nSignals, NOT sorted by time. Every B quote is at t>=4, so the B signal")
    print("at t=2 has no prior quote and its only correct answer is null.\n")

    print(f"  {'signal':<12}{'cursor (unsorted)':>20}{'correct (sorted)':>20}")
    for i in range(len(sig)):
        s, t = sig['symbol'][i], sig['time'][i]
        c, k = res["tiny_cursor"][i], res["tiny_correct"][i]
        flag = ""
        if not _eq(c, k):
            flag = "  <- WRONG"
            if not (c is None or np.isnan(c)):
                qtime = int(qt.loc[(qt["symbol"] == s) & (qt["px"] == c), "time"].iloc[0])
                flag += f": quote from t={qtime}, {qtime - t} steps in the FUTURE"
        print(f"  {s}@t={t:<8}{_fmt(c):>20}{_fmt(k):>20}{flag}")
    print(f"\n  rows differing: {res['tiny_diff']} of {len(sig)}")

    print(f"\nSame failure at scale. One seeded panel, {res['n_rows']} signals,"
          f" left frame shuffled.")
    print(f"  reference cursor model vs correct: {res['panel_diff']} of {res['n_rows']}"
          f" rows differ ({res['panel_diff'] / res['n_rows']:.1%})")

    if "LIVE_version" in res:
        print(f"\n  verified against the installed polars {res['LIVE_version']}:")
        print(f"    polars, unsorted left : {[_fmt(v) for v in res['LIVE_tiny_unsorted']]}")
        print(f"    cursor reference      : {[_fmt(v) for v in res['tiny_cursor']]}"
              f"   identical: {res['LIVE_cursor_matches']}")
        print(f"    polars, sorted        : {[_fmt(v) for v in res['LIVE_tiny_sorted']]}")
        print(f"    correct reference     : matches polars-sorted:"
              f" {res['LIVE_correct_matches']}")
        print(f"\n    warnings raised by the unsorted call: {res['LIVE_warnings']}")
        print(f"\n    panel, {res['n_rows']} rows, quote_time carried through the join:")
        print(f"      unsorted: {res['LIVE_lookahead_bad']} rows matched a FUTURE quote"
              f"  ({res['LIVE_lookahead_bad'] / res['n_rows']:.1%})")
        print(f"      sorted  : {res['LIVE_lookahead_ok']} rows matched a future quote")
        print(f"      rows differing between the two: {res['LIVE_panel_diff']}"
              f" of {res['n_rows']}")
        print(f"      row count: {res['LIVE_heights'][0]} unsorted,"
              f" {res['LIVE_heights'][1]} sorted, input {res['n_rows']}"
              f"   <- `j.height == sig.height` does NOT catch this")
        print(f"    (the reference model reproduces polars row-for-row on the tiny case;"
              f"\n     at scale it reproduces the CLASS of error, not the exact rows,"
              f"\n     because polars chunks the scan internally -- hence"
              f" {res['panel_diff']} vs {res['LIVE_panel_diff']}.)")

        print(f"\n  What polars {res['LIVE_version']} actually does"
              f"  (check_sortedness default = {res['LIVE_check_sortedness_default']}):")
        for label, behaviour in res["LIVE_matrix"]:
            mark = "  <- silent corruption" if "RAISES" not in behaviour else ""
            print(f"    {label:<44} {behaviour}{mark}")
    else:
        print("\n  polars not installed — reference implementations only")

    print("\n  Rule: polars does NOT raise on unsorted input when `by=` is given."
          "\n        The UserWarning fires on every `by=` call, sorted or not, so it"
          "\n        is noise, not a guard. Sort BOTH frames globally by the `on` key"
          "\n        (satisfies polars and pandas), carry the right-hand timestamp"
          "\n        through, and assert quote_time <= time. Row counts stay correct"
          "\n        through the whole failure, so only the timestamp assert catches it.")
