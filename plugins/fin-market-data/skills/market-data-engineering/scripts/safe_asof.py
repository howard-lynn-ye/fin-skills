"""Guarded as-of join. The default `merge_asof` call is a look-ahead bug with good ergonomics.

WHY this exists — four silent failures, all of which ship to production:

1. `allow_exact_matches=True` is pandas' DEFAULT. A signal stamped 09:30:01 joined to a
   quote stamped 09:30:01 takes that quote. You did not have it when you decided. This is
   the single most common leak in intraday research, and nothing warns you. The right side
   of an as-of join is normally information you REACT to, so the safe default is the
   opposite of pandas': strictly prior.
2. `tolerance=None` is the DEFAULT. An unbounded as-of join reaches back across a weekend,
   a trading halt or a delisting and hands you a two-month-old quote as though it were
   live. The backtest gets a fill at a stale price; reality gets nothing at all. There is
   no such thing as a correct as-of join without a stated staleness budget.
3. `polars.join_asof` does not verify sortedness when `by=` is given — its own docstring
   says it "cannot check the sortedness if 'by' groups are provided", so unsorted input
   returns wrong rows with no error. pandas raises. Forcing a global sort here makes the
   same code correct under both engines.
4. A `by=` key that is `category` on one side and `object` on the other matches nothing,
   or raises from deep inside the merge internals long after you stopped looking.

Usage:
    from safe_asof import safe_merge_asof
    out = safe_merge_asof(signals, quotes, on="time", by="symbol", tolerance="5min")
    # out["_asof_right_ts"] carries the right-hand stamp; every matched row is STRICTLY
    # prior to its left-hand timestamp, and len(out) == len(signals), both asserted.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

RIGHT_TS_COL = "_asof_right_ts"
_LEFT_POS = "_asof_left_pos"


def _as_list(keys: str | Sequence[str] | None) -> list[str]:
    if keys is None:
        return []
    if isinstance(keys, str):
        return [keys]
    return list(keys)


def check_key_dtypes(left: pd.DataFrame, right: pd.DataFrame,
                     keys: str | Sequence[str] | None, kind: str = "by") -> None:
    """Raise if a join key has a different dtype on each side.

    Failure prevented: a symbol column stored as `category` on one side and `object` on
    the other. Categorical codes never compare equal to raw strings, so the join returns
    all-unmatched rows (or an opaque MergeError) rather than an obvious type error.
    """
    for k in _as_list(keys):
        if k not in left.columns:
            raise KeyError(f"{kind} key {k!r} is not a column of left")
        if k not in right.columns:
            raise KeyError(f"{kind} key {k!r} is not a column of right")
        lt, rt = left[k].dtype, right[k].dtype
        if lt == rt:
            continue
        hint = ""
        if isinstance(lt, pd.CategoricalDtype) or isinstance(rt, pd.CategoricalDtype):
            hint = (" -- category-vs-object is the classic silent-empty-join. Cast BOTH "
                    "sides with .astype('string'), or to the same CategoricalDtype with "
                    "identical categories.")
        raise TypeError(f"{kind} key {k!r} dtype mismatch: left={lt}, right={rt}{hint}")


def assert_row_count_preserved(left: pd.DataFrame, out: pd.DataFrame,
                               context: str = "") -> None:
    """An as-of join must return exactly one row per left row. Anything else is a bug.

    Failure prevented: reaching for a plain `pd.merge` on the symbol key, which drops
    left rows with no counterparty and fans out rows with several. Both change your
    position count, and neither raises.
    """
    if len(out) != len(left):
        delta = len(out) - len(left)
        verb = "GAINED" if delta > 0 else "LOST"
        raise AssertionError(
            f"row count changed{' in ' + context if context else ''}: left={len(left)}, "
            f"out={len(out)} ({verb} {abs(delta)}). An as-of join is one-row-per-left-row; "
            f"a fan-out or a drop means keys duplicated or an inner join silently occurred."
        )


def assert_strictly_prior(out: pd.DataFrame, on: str,
                          right_ts_col: str = RIGHT_TS_COL) -> None:
    """Assert every MATCHED row used right-hand data stamped strictly before `on`.

    Unmatched rows (NaT stamp) are fine — they are the honest "I had nothing yet".
    """
    matched = out[right_ts_col].notna()
    if not matched.any():
        return
    bad = out.loc[matched, right_ts_col] >= out.loc[matched, on]
    n_bad = int(bad.sum())
    if n_bad:
        row = out.loc[matched][bad.to_numpy()].iloc[0]
        raise AssertionError(
            f"LOOK-AHEAD: {n_bad} matched row(s) used right-hand data stamped at or after "
            f"the left timestamp. First offender: left {on}={row[on]!r} matched right "
            f"{right_ts_col}={row[right_ts_col]!r}."
        )


def _coerce_tolerance(tolerance: Any, on_values: pd.Series) -> Any:
    """Require an explicit, positive tolerance and coerce it to the key's units."""
    if tolerance is None:
        raise ValueError(
            "tolerance is REQUIRED. An unbounded as-of join matches across weekends, "
            "trading halts and delistings, silently pricing a position off a quote that "
            "may be months old. State the staleness you are willing to accept, e.g. "
            "tolerance='5min' for intraday quotes or tolerance='3D' for daily bars."
        )
    if pd.api.types.is_datetime64_any_dtype(on_values):
        tol = pd.Timedelta(tolerance)
        if tol <= pd.Timedelta(0):
            raise ValueError(f"tolerance must be positive, got {tol}")
        return tol
    tol = float(tolerance)
    if tol <= 0:
        raise ValueError(f"tolerance must be positive, got {tol}")
    return tol


def safe_merge_asof(
    left: pd.DataFrame,
    right: pd.DataFrame,
    on: str,
    by: str | Sequence[str] | None = None,
    tolerance: Any = None,
    allow_exact_matches: bool = False,
    require_prior: bool = True,
    direction: str = "backward",
    right_ts_col: str = RIGHT_TS_COL,
    suffixes: tuple[str, str] = ("", "_right"),
) -> pd.DataFrame:
    """As-of join with the defaults inverted and the invariants asserted.

    Differences from `pd.merge_asof`, each one a bug this prevents:
      * `allow_exact_matches` defaults to **False** (pandas: True) -- no same-stamp match.
      * `tolerance` is **mandatory** (pandas: None) -- no unbounded reach-back.
      * both frames are **globally sorted by `on`** before the join -- polars does not
        check sortedness when `by=` is passed and returns wrong rows instead of raising.
      * the right-hand timestamp is carried through as `right_ts_col` so staleness is
        auditable after the fact, and every matched row is asserted strictly prior.
      * output row count is asserted equal to the left row count.
      * `by` and `on` key dtypes are checked on both sides before the join runs.

    Returns a frame sorted by `on`, carrying left's original index labels.
    """
    if direction != "backward" and require_prior:
        raise ValueError(
            f"require_prior=True is only meaningful for direction='backward'; got "
            f"direction={direction!r}. A forward/nearest join looks into the future BY "
            f"DESIGN -- pass require_prior=False only if you know why that is acceptable."
        )
    if allow_exact_matches and require_prior:
        raise ValueError(
            "allow_exact_matches=True contradicts require_prior=True: an exact match is "
            "by definition not strictly prior. Set require_prior=False to opt into "
            "same-timestamp matching, and write down why it is not look-ahead."
        )
    if right_ts_col in left.columns or right_ts_col in right.columns:
        raise ValueError(f"{right_ts_col!r} already exists; pass a different right_ts_col")

    by_keys = _as_list(by)
    check_key_dtypes(left, right, [on], kind="on")
    check_key_dtypes(left, right, by_keys, kind="by")

    if left[on].isna().any():
        raise ValueError(f"left[{on!r}] contains nulls; an as-of key must be fully stamped")
    if right[on].isna().any():
        raise ValueError(f"right[{on!r}] contains nulls; an as-of key must be fully stamped")

    tol = _coerce_tolerance(tolerance, left[on])

    lf = left.copy()
    rf = right.copy()
    # Positional stash: lets us restore left's index labels AND prove no row was
    # invented or dropped, even though merge_asof discards the index.
    lf[_LEFT_POS] = np.arange(len(lf))
    # Carry the right-hand stamp so staleness stays auditable downstream.
    rf[right_ts_col] = rf[on]

    # FORCED GLOBAL SORT (stable). pandas raises on unsorted keys; polars silently
    # returns wrong rows when `by=` is set. Sorting here is correct under both.
    lf = lf.sort_values(on, kind="mergesort")
    rf = rf.sort_values(on, kind="mergesort")

    out = pd.merge_asof(
        lf, rf, on=on,
        by=by_keys or None,
        tolerance=tol,
        allow_exact_matches=allow_exact_matches,
        direction=direction,
        suffixes=suffixes,
    )

    assert_row_count_preserved(left, out, context="safe_merge_asof")
    out.index = left.index[out[_LEFT_POS].to_numpy()]
    out = out.drop(columns=[_LEFT_POS])
    if require_prior:
        assert_strictly_prior(out, on, right_ts_col)
    return out


if __name__ == "__main__":
    pd.set_option("display.width", 120)
    ts = pd.Timestamp

    # ---------------------------------------------------------------- 1. LOOK-AHEAD
    print("=" * 78)
    print("1. THE DOCUMENTED LOOK-AHEAD: signal at 09:30:01, quote stamped 09:30:01")
    print("=" * 78)
    signals = pd.DataFrame({
        "time": [ts("2024-01-02 09:30:01")],
        "symbol": ["AAA"],
        "signal": [1],
    })
    quotes = pd.DataFrame({
        "time": [ts("2024-01-02 09:29:59"), ts("2024-01-02 09:30:01"),
                 ts("2024-01-02 09:30:05")],
        "symbol": ["AAA", "AAA", "AAA"],
        "px": [100.0, 101.0, 102.0],
    })
    print("\nsignals:\n", signals.to_string(index=False))
    print("\nquotes:\n", quotes.to_string(index=False))

    naive = pd.merge_asof(signals, quotes, on="time", by="symbol")
    print(f"\npandas default (allow_exact_matches=True) -> px = {naive.px.iloc[0]}"
          "   <-- the SAME-TIMESTAMP quote. LOOK-AHEAD.")

    safe = safe_merge_asof(signals, quotes, on="time", by="symbol", tolerance="5min")
    print(f"safe_merge_asof                           -> px = {safe.px.iloc[0]}"
          f"   <-- strictly prior ({safe[RIGHT_TS_COL].iloc[0]:%H:%M:%S}). CORRECT.")

    # ------------------------------------------------- 2. UNBOUNDED / STALE MATCHING
    print("\n" + "=" * 78)
    print("2. THE UNBOUNDED MATCH: 'ZZZ' stopped quoting on 2024-01-05 (delisted)")
    print("=" * 78)
    late = pd.DataFrame({
        "time": [ts("2024-03-01 10:00:00")], "symbol": ["ZZZ"], "signal": [1],
    })
    stale = pd.DataFrame({
        "time": [ts("2024-01-05 15:59:00")], "symbol": ["ZZZ"], "px": [42.0],
    })
    unbounded = pd.merge_asof(late, stale, on="time", by="symbol")
    age = late.time.iloc[0] - stale.time.iloc[0]
    print(f"\npandas default (tolerance=None) -> px = {unbounded.px.iloc[0]}, "
          f"quote is {age.days} DAYS old. Silently priced off a dead ticker.")
    try:
        safe_merge_asof(late, stale, on="time", by="symbol")
    except ValueError as e:
        print(f"\nsafe_merge_asof(tolerance=None) CAUGHT:\n  ValueError: {str(e)[:96]}...")
    bounded = safe_merge_asof(late, stale, on="time", by="symbol", tolerance="3D")
    print(f"\nsafe_merge_asof(tolerance='3D') -> px = {bounded.px.iloc[0]} (NaN = "
          "honestly unmatched). The row survives; the fake price does not.")

    # ---------------------------------------------------------------- 3. ROW LOSS
    print("\n" + "=" * 78)
    print("3. ROW LOSS: reaching for a plain merge on the symbol key")
    print("=" * 78)
    multi_sig = pd.DataFrame({
        "time": pd.to_datetime(["2024-01-02 09:31", "2024-01-02 09:31",
                                "2024-01-02 09:32"]),
        "symbol": ["AAA", "BBB", "AAA"], "signal": [1, -1, 1],
    })
    naive_join = multi_sig.merge(quotes, on="symbol", how="inner")  # BBB has no quotes
    try:
        assert_row_count_preserved(multi_sig, naive_join, context="pd.merge(on='symbol')")
    except AssertionError as e:
        print(f"\nCAUGHT: {e}")
    kept = safe_merge_asof(multi_sig, quotes, on="time", by="symbol", tolerance="5min")
    print(f"\nsafe_merge_asof keeps {len(kept)}/{len(multi_sig)} rows; BBB is unmatched "
          f"(px={kept.px.iloc[1]}) rather than deleted.")

    # ---------------------------------------------------------- 4. by-KEY DTYPE DRIFT
    print("\n" + "=" * 78)
    print("4. by-KEY DTYPE DRIFT: category on the left, object on the right")
    print("=" * 78)
    cat_sig = signals.assign(symbol=signals.symbol.astype("category"))
    try:
        safe_merge_asof(cat_sig, quotes, on="time", by="symbol", tolerance="5min")
    except TypeError as e:
        print(f"\nCAUGHT: {e}")

    print("\n" + "=" * 78)
    print("Four defaults, four leaks. The wrapper costs one import.")
    print("=" * 78)
