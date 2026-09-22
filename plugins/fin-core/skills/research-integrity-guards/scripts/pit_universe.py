#!/usr/bin/env python3
"""Build a POINT-IN-TIME universe, so a backtest screens on what was investable that day.

WHY: `survivorship_audit.py` answers "does my price panel still contain the names that
died?". This file answers the question that comes immediately after, and that a clean
panel does NOT settle on its own: **which of those names was I allowed to buy on 2013-06-28?**

Getting the panel right and the universe wrong loses you nothing at import time and
everything in the results. The two failure modes:

  * TODAY'S-MEMBERS SCREEN -- take the current S&P 500 / index constituents / "liquid
    names" list and run it over ten years of history. Membership is a forward-looking
    label: a company is in today's index BECAUSE it grew, and the ones deleted after a
    70% drawdown are exactly the observations your strategy needed to lose money on.
  * FULL-SAMPLE LIQUIDITY FILTER -- `adv.mean() > 5e6` computed over the whole panel and
    applied to every date. A name that was untradeable in 2013 and enormous by 2023 passes
    the 2013 screen on the strength of its 2023 volume.

Both are caught by the same tell, and it needs no external data: **a point-in-time
universe LOSES names.** If a decade of monthly rebalances never removes anybody, the
universe was built from a current snapshot. Additions alone are not evidence of anything --
IPOs and index promotions are real. It is the absence of DELETIONS that gives it away.

Usage:
    from pit_universe import universe_on, rebalance_universe, audit_universe_stability

    members = pd.DataFrame({"ticker": [...], "start_date": [...], "end_date": [...]})
    uni = rebalance_universe(rebalance_dates, members, liquidity=adv_df, min_adv=5e6)

    audit = audit_universe_stability(uni)
    print(audit.report())
    if audit.never_loses_a_name:
        raise SystemExit("refusing to backtest on a current-snapshot universe")

MEMBERSHIP TABLE SHAPE:
    ticker | start_date | end_date
    the date the name BECAME investable, and the date it stopped. `end_date` NaT means
    still a member. Intervals are treated as CLOSED on both ends: a name deleted on
    2019-03-15 was still in the index that morning, and a backtest that rebalanced that
    day held it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------------------
# point-in-time membership
# --------------------------------------------------------------------------------------
def _normalise_members(members: pd.DataFrame) -> pd.DataFrame:
    if "ticker" not in members.columns:
        raise ValueError("membership table needs a 'ticker' column")
    if "start_date" not in members.columns:
        raise ValueError("membership table needs a 'start_date' column -- a table without "
                         "one is a current snapshot, which is the thing this module exists "
                         "to stop you from backtesting")
    out = members.copy()
    out["start_date"] = pd.to_datetime(out["start_date"], errors="coerce")
    out["end_date"] = (pd.to_datetime(out["end_date"], errors="coerce")
                       if "end_date" in out.columns else pd.NaT)
    if out["start_date"].isna().any():
        bad = out.loc[out["start_date"].isna(), "ticker"].tolist()
        raise ValueError(f"unparseable start_date for {bad[:8]}")
    bad_span = out["end_date"].notna() & (out["end_date"] < out["start_date"])
    if bad_span.any():
        raise ValueError(f"end_date precedes start_date for "
                         f"{out.loc[bad_span, 'ticker'].tolist()[:8]}")
    return out


def universe_on(date: str | pd.Timestamp,
                members: pd.DataFrame,
                liquidity: pd.DataFrame | None = None,
                min_adv: float | None = None,
                adv_lookback: int = 21) -> list[str]:
    """Tickers investable ON `date`: index members, optionally screened on trailing ADV.

    date      : the as-of date. Membership intervals are closed, so a name whose
                `end_date` IS `date` still counts -- it traded that session.
    members   : ticker / start_date / end_date table.
    liquidity : DataFrame of average daily traded VALUE, dates x tickers. Optional.
    min_adv   : minimum trailing ADV to pass the screen.
    adv_lookback : sessions of trailing liquidity to average, ending ON `date`.

    🚨 The ADV is computed from `liquidity.loc[:date]` ONLY. Screening on a full-sample
    mean -- `adv.mean() > min_adv` applied to every date -- admits names on the strength
    of volume they had not yet traded, which is look-ahead of exactly the same kind as a
    survivor-only ticker list, and much harder to see in a diff.
    """
    ts = pd.Timestamp(date)
    m = _normalise_members(members)
    live = m[(m["start_date"] <= ts) & (m["end_date"].isna() | (m["end_date"] >= ts))]
    names = sorted(set(live["ticker"]))

    if liquidity is None or min_adv is None:
        return names

    hist = liquidity.loc[liquidity.index <= ts]
    if hist.empty:
        return []
    window = hist.iloc[-int(adv_lookback):]
    adv = window.mean(axis=0, skipna=True)
    return [t for t in names
            if t in adv.index and pd.notna(adv[t]) and float(adv[t]) >= float(min_adv)]


def rebalance_universe(dates: Sequence[str | pd.Timestamp] | pd.DatetimeIndex,
                       members: pd.DataFrame,
                       liquidity: pd.DataFrame | None = None,
                       min_adv: float | None = None,
                       adv_lookback: int = 21) -> dict[pd.Timestamp, list[str]]:
    """`universe_on` at each rebalance date. Returns {date: [tickers]}.

    Normalises the membership table once rather than per date -- the whole point is that
    this runs inside a backtest loop, so it must not be quadratic in the table size.
    """
    m = _normalise_members(members)
    tickers = m["ticker"].to_numpy()
    starts = m["start_date"].to_numpy()
    ends = m["end_date"].to_numpy()
    open_ended = pd.isna(m["end_date"]).to_numpy()

    out: dict[pd.Timestamp, list[str]] = {}
    for d in dates:
        ts = pd.Timestamp(d)
        live = (starts <= np.datetime64(ts)) & (open_ended | (ends >= np.datetime64(ts)))
        names = sorted(set(tickers[live].tolist()))
        if liquidity is not None and min_adv is not None:
            hist = liquidity.loc[liquidity.index <= ts]
            if hist.empty:
                out[ts] = []
                continue
            adv = hist.iloc[-int(adv_lookback):].mean(axis=0, skipna=True)
            names = [t for t in names if t in adv.index and pd.notna(adv[t])
                     and float(adv[t]) >= float(min_adv)]
        out[ts] = names
    return out


# --------------------------------------------------------------------------------------
# stability audit
# --------------------------------------------------------------------------------------
@dataclass
class UniverseStabilityAudit:
    n_dates: int
    first_date: str
    last_date: str
    sizes: pd.Series
    additions: pd.Series
    removals: pd.Series
    turnover: pd.Series
    mean_turnover: float
    median_turnover: float
    total_additions: int
    total_removals: int
    n_ever: int
    n_at_end: int
    never_loses_a_name: bool
    monotonic_growth: bool
    verdict: str
    notes: list[str] = field(default_factory=list)

    def report(self) -> str:
        w = 74
        lines = [
            "POINT-IN-TIME UNIVERSE STABILITY",
            "=" * w,
            f"  rebalances          : {self.n_dates}  "
            f"({self.first_date} -> {self.last_date})",
            f"  size                : {self.sizes.iloc[0]:.0f} at the start, "
            f"{self.sizes.iloc[-1]:.0f} at the end, "
            f"{self.sizes.mean():.1f} mean",
            f"  names ever a member : {self.n_ever}   still a member at the end: "
            f"{self.n_at_end}",
            f"  additions / removals: {self.total_additions} / {self.total_removals}",
            f"  one-way turnover    : {self.mean_turnover:.2%} mean, "
            f"{self.median_turnover:.2%} median per rebalance",
            f"  VERDICT             : {self.verdict}",
        ]
        lines += [f"    ! {n}" for n in self.notes]
        return "\n".join(lines)


def audit_universe_stability(universe: dict[pd.Timestamp, Sequence[str]] | Iterable,
                             min_expected_turnover: float = 0.005
                             ) -> UniverseStabilityAudit:
    """Turnover profile of a rebalanced universe, and the current-snapshot red flag.

    universe : {date: [tickers]} as returned by `rebalance_universe`.
    min_expected_turnover : one-way turnover below which the universe is suspiciously
        static. 0.5% per rebalance is a deliberately LOW bar -- a large-cap index runs
        well above it -- chosen so the flag means "something is structurally wrong",
        not "your index is unusually stable".

    Turnover here counts NAMES, not weights: (added + removed) / previous size. It is a
    membership diagnostic, not a trading-cost estimate; for cost use the traded notional
    (see ../../backtest-validation/scripts/cost_curve.py).
    """
    items = sorted((pd.Timestamp(d), sorted(set(t))) for d, t in dict(universe).items())
    if len(items) < 2:
        raise ValueError("need at least two rebalance dates to measure turnover")

    dates = [d for d, _ in items]
    sizes = pd.Series([len(t) for _, t in items], index=dates, dtype=float)

    adds, rems, turn = [np.nan], [np.nan], [np.nan]
    ever: set[str] = set(items[0][1])
    for (_, prev), (_, cur) in zip(items, items[1:]):
        ps, cs = set(prev), set(cur)
        a, r = len(cs - ps), len(ps - cs)
        adds.append(float(a))
        rems.append(float(r))
        turn.append((a + r) / len(ps) if ps else np.nan)
        ever |= cs

    additions = pd.Series(adds, index=dates)
    removals = pd.Series(rems, index=dates)
    turnover = pd.Series(turn, index=dates)
    total_rem = int(np.nansum(removals))
    total_add = int(np.nansum(additions))
    mean_t = float(np.nanmean(turnover.to_numpy()))
    med_t = float(np.nanmedian(turnover.to_numpy()))
    never_loses = total_rem == 0
    mono = bool((sizes.diff().dropna() >= 0).all())

    notes: list[str] = []
    if never_loses:
        verdict = (f"CURRENT-SNAPSHOT SCREEN (red flag): {len(items)} rebalances over "
                   f"{(dates[-1] - dates[0]).days / 365.25:.1f} yr and not one name ever "
                   f"left the universe")
        notes.append("a real universe deletes names -- bankruptcies, takeouts, index "
                     "demotions, liquidity failures")
        notes.append("results from this universe are an UPPER BOUND, not an estimate")
    elif mean_t < min_expected_turnover:
        verdict = (f"SUSPICIOUSLY STATIC: {mean_t:.2%} mean one-way turnover, below the "
                   f"{min_expected_turnover:.2%} floor")
        notes.append("check whether deletions are being applied at the right dates")
    else:
        verdict = (f"PLAUSIBLY POINT-IN-TIME: {total_rem} removals and {total_add} "
                   f"additions, {mean_t:.2%} mean turnover")
    if mono and not never_loses:
        notes.append("universe size never falls, though names do rotate -- check that "
                     "deletions are not being back-filled by construction")
    if never_loses and total_add > 0:
        notes.append(f"{total_add} additions with zero removals is the exact signature of "
                     f"a fixed ticker list whose members' price history starts at "
                     f"different dates")

    return UniverseStabilityAudit(
        n_dates=len(items), first_date=str(dates[0].date()), last_date=str(dates[-1].date()),
        sizes=sizes, additions=additions, removals=removals, turnover=turnover,
        mean_turnover=mean_t, median_turnover=med_t,
        total_additions=total_add, total_removals=total_rem,
        n_ever=len(ever), n_at_end=len(items[-1][1]),
        never_loses_a_name=never_loses, monotonic_growth=mono,
        verdict=verdict, notes=notes,
    )


# --------------------------------------------------------------------------------------
# putting a number on it
# --------------------------------------------------------------------------------------
def equal_weight_backtest(universe: dict[pd.Timestamp, Sequence[str]],
                          prices: pd.DataFrame) -> pd.Series:
    """Equal-weight the universe at each rebalance and hold to the next one.

    Deliberately simple, because its only job is to price the difference between two
    universes over identical dates and identical prices. `fill_method=None` matters: a
    forward-filled price manufactures a flat, riskless tail for a company that stopped
    trading, which is the very loss the point-in-time universe is trying to keep.
    """
    rets = prices.pct_change(fill_method=None)
    dates = sorted(pd.Timestamp(d) for d in universe)
    chunks: list[pd.Series] = []
    for i, d in enumerate(dates):
        end = dates[i + 1] if i + 1 < len(dates) else rets.index.max()
        held = [t for t in universe[d] if t in rets.columns]
        seg = rets.loc[(rets.index > d) & (rets.index <= end)]
        chunks.append(seg[held].mean(axis=1, skipna=True) if held
                      else pd.Series(0.0, index=seg.index))
    out = pd.concat(chunks).sort_index() if chunks else pd.Series(dtype=float)
    return out.dropna()


def annualised(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    if len(returns) < 2:
        return float("nan")
    years = len(returns) / periods_per_year
    return float((1.0 + returns).prod() ** (1.0 / years) - 1.0)


# --------------------------------------------------------------------------------------
# offline demo
# --------------------------------------------------------------------------------------
def _synthetic_index(n_names: int = 150, years: int = 8, seed: int = 5
                     ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A price panel plus the membership table of an index that deletes its losers.

    The deletion rule is the realistic one and the reason the bias exists: a name is
    dropped once it has fallen 60% from its peak. Deletions therefore FOLLOW bad returns,
    so a universe built from the survivors is conditioned on the outcome.
    """
    rng = np.random.default_rng(seed)
    n = years * TRADING_DAYS
    idx = pd.bdate_range("2016-01-04", periods=n)
    cols = [f"S{i:03d}" for i in range(n_names)]

    mu = rng.normal(0.06, 0.16, n_names) / TRADING_DAYS
    sig = rng.uniform(0.22, 0.60, n_names) / np.sqrt(TRADING_DAYS)
    px = 100.0 * np.exp(np.cumsum(rng.normal(mu, sig, size=(n, n_names)), axis=0))
    prices = pd.DataFrame(px, index=idx, columns=cols)

    # a fifth of the names join partway through (IPOs / promotions), the rest start day 1
    joins = np.where(rng.random(n_names) < 0.20,
                     rng.integers(TRADING_DAYS // 2, n - TRADING_DAYS, n_names), 0)

    rows = []
    for j, c in enumerate(cols):
        j0 = int(joins[j])
        prices.iloc[:j0, j] = np.nan                     # no history before joining
        series = prices[c].to_numpy()
        peak = np.fmax.accumulate(np.where(np.isnan(series), -np.inf, series))
        drawdown = np.where(peak > 0, series / peak - 1.0, 0.0)
        hit = np.where((np.arange(n) > j0 + 60) & (drawdown < -0.60))[0]
        if len(hit):
            k = int(hit[0])
            prices.iloc[k + 1:, j] = np.nan              # deleted names stop reporting
            rows.append({"ticker": c, "start_date": idx[j0], "end_date": idx[k]})
        else:
            rows.append({"ticker": c, "start_date": idx[j0], "end_date": pd.NaT})
    return prices, pd.DataFrame(rows)


if __name__ == "__main__":
    prices, members = _synthetic_index()
    rebals = pd.bdate_range(prices.index[0], prices.index[-1], freq="BQE")

    # A. the honest universe: whoever was a member on each rebalance date
    pit = rebalance_universe(rebals, members)

    # B. what a screen against a current snapshot produces: TODAY's members, applied to
    #    every historical date. Names only appear as their price history begins.
    today = sorted(members.loc[members["end_date"].isna(), "ticker"])
    snap = {pd.Timestamp(d): [t for t in today
                              if pd.notna(prices[t].asof(pd.Timestamp(d)))]
            for d in rebals}

    print("=== A. point-in-time universe (membership as of each rebalance) ===")
    a_pit = audit_universe_stability(pit)
    print(a_pit.report())

    print("\n=== B. same dates, universe screened from TODAY's member list ===")
    a_snap = audit_universe_stability(snap)
    print(a_snap.report())

    r_pit = equal_weight_backtest(pit, prices)
    r_snap = equal_weight_backtest(snap, prices)
    c_pit, c_snap = annualised(r_pit), annualised(r_snap)

    print("\n" + "=" * 74)
    print("=== C. the same strategy, the same prices, the same dates ===")
    print("=" * 74)
    print(f"  equal-weight, quarterly rebalanced, {len(rebals)} rebalances\n")
    print(f"  {'universe':<34}{'CAGR':>10}{'total':>12}{'names':>9}")
    print("  " + "-" * 63)
    print(f"  {'point-in-time (honest)':<34}{c_pit:>9.2%}"
          f"{(1 + r_pit).prod() - 1:>12.1%}{a_pit.n_ever:>9}")
    print(f"  {'todays members (survivor screen)':<34}{c_snap:>9.2%}"
          f"{(1 + r_snap).prod() - 1:>12.1%}{a_snap.n_ever:>9}")
    print("  " + "-" * 63)
    print(f"  {'INFLATION':<34}{c_snap - c_pit:>9.2%}"
          f"{(1 + r_snap).prod() - (1 + r_pit).prod():>12.1%}")
    print(f"\n  {(c_snap - c_pit) * 1e4:.0f} bps/yr of pure look-ahead, from a screen that "
          f"looks completely\n  reasonable in code and never raises anything.")
    print(f"  The point-in-time universe deleted {a_pit.total_removals} names over the "
          f"sample. The snapshot\n  universe deleted {a_snap.total_removals} -- and that "
          f"single number is the whole tell, available\n  before you run a backtest and "
          f"without any external data.")
    print("\n  Note what is NOT evidence: the snapshot universe still shows "
          f"{a_snap.total_additions} additions, because\n  members' price histories begin "
          f"at different dates. Additions are normal. Zero\n  removals over "
          f"{(rebals[-1] - rebals[0]).days / 365.25:.0f} years is not.")

    # ----------------------------------------------------------------------------------
    # D. the second trap, isolated: correct membership, look-ahead liquidity screen
    # ----------------------------------------------------------------------------------
    adv = prices * 2.0e5              # traded value tracks price, as it does in reality
    thresh = 2.0e7
    liq_pit = rebalance_universe(rebals, members, liquidity=adv, min_adv=thresh)
    full_sample_mean = adv.mean(axis=0, skipna=True)
    passing = set(full_sample_mean.index[full_sample_mean >= thresh])
    liq_trap = {d: [t for t in names if t in passing] for d, names in pit.items()}

    r_liq_pit = equal_weight_backtest(liq_pit, prices)
    r_liq_trap = equal_weight_backtest(liq_trap, prices)
    c_liq_pit, c_liq_trap = annualised(r_liq_pit), annualised(r_liq_trap)
    first = rebals[0]
    extra = sorted(set(liq_trap[first]) - set(liq_pit[first]))

    print("\n" + "=" * 74)
    print("=== D. the same trap in the LIQUIDITY filter, with membership held correct ===")
    print("=" * 74)
    print(f"  both universes use the honest point-in-time membership; only the ADV screen "
          f"differs\n  (min ADV {thresh:,.0f})\n")
    print(f"  {'liquidity screen':<34}{'CAGR':>10}{'total':>12}{'size@t0':>10}")
    print("  " + "-" * 64)
    print(f"  {'trailing 21d ADV (honest)':<34}{c_liq_pit:>9.2%}"
          f"{(1 + r_liq_pit).prod() - 1:>12.1%}{len(liq_pit[first]):>10}")
    print(f"  {'full-sample mean ADV (trap)':<34}{c_liq_trap:>9.2%}"
          f"{(1 + r_liq_trap).prod() - 1:>12.1%}{len(liq_trap[first]):>10}")
    print("  " + "-" * 64)
    print(f"  {'INFLATION':<34}{c_liq_trap - c_liq_pit:>9.2%}"
          f"{(1 + r_liq_trap).prod() - (1 + r_liq_pit).prod():>12.1%}")
    print(f"\n  A further {(c_liq_trap - c_liq_pit) * 1e4:.0f} bps/yr, and this one "
          f"survives a correct membership table.")
    print(f"  {len(extra)} names pass the screen on {first.date()} on the strength of "
          f"volume they had\n  not traded yet. `adv.mean() > min_adv` is one line, reads "
          f"as a liquidity filter,\n  and is a winner filter.")
