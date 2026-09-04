"""Generate futures roll dates by rule -- and expose roll-rule choice as an uncounted trial.

WHY this exists: "roll 5 days before expiry" looks like plumbing. It is a free parameter,
and it is the one nobody logs. A futures backtest is not one backtest; it is one backtest
PER ROLL RULE, and there are easily a dozen defensible rules (calendar N days out for
N in 1..15, open-interest crossover, volume crossover, first-notice-day, last-trade-day,
optimised roll yield). Each one produces a different price series, therefore a different
equity curve, therefore a different Sharpe.

The failure this catches: you run the backtest with the vendor's default roll, see Sharpe
0.3, try the open-interest roll because "it's more realistic", see Sharpe 0.9, and ship the
second one. You have just run a two-trial search and reported the maximum as if it were the
only thing you tried. The Deflated Sharpe Ratio you did not compute would have told you 0.9
out of 4 rule variants on 8 years of data is not significant. Nothing in the code path
warns you, because switching rules feels like fixing a data problem rather than fitting.

WHY roll timing moves the number at all -- it is not noise:
  * Term structure slope is seasonal in most physical commodities (gas, power, grains,
    and anything with a storage cycle). Rolling at a fixed distance from expiry samples
    the SAME PHASE of that cycle every quarter, so the difference compounds instead of
    averaging out over 30 rolls.
  * The front contract is pushed around in the days before expiry by everyone else rolling
    (the "index roll" window). Roll into that and you buy the deferred rich and sell the
    front cheap, every single time.
  * Rolling early means trading the deferred contract while it is still thin, so the
    slippage per roll is higher even though the number of rolls is identical.

So: pick the rule ON EX-ANTE GROUNDS (liquidity at the moment you trade, delivery risk,
your fund's actual operational calendar), write it down BEFORE you look at the equity
curve, and log every rule you evaluated to the trial ledger:
    ../../../fin-core/skills/backtest-validation/scripts/trial_ledger.py

Usage:
    from roll_schedule import roll_dates, compare_rules, render
    rolls = roll_dates(px, "open_interest", expiries, open_interest=oi)
    print(render(compare_rules(px, ("calendar", "open_interest", "volume", "first_notice"),
                               expiries, open_interest=oi, volume=vol,
                               first_notice=fnd)))
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

# Sibling module in the same skill. `_stats` is shared deliberately: if the two files
# annualised differently, the rule comparison below would be measuring the difference
# between two Sharpe conventions rather than between two roll rules.
try:
    from continuous_contract import TRADING_DAYS, _stats, true_roll_return
except ImportError:  # imported from outside this scripts/ directory
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from continuous_contract import TRADING_DAYS, _stats, true_roll_return

RULES = ("calendar", "open_interest", "volume", "first_notice")

# Defaults are DEFENSIBLE, not optimal. That distinction is the whole point of this file.
DEFAULT_OFFSET = {
    "calendar": 5,        # business days before expiry
    "open_interest": 0,   # roll on the crossover day itself
    "volume": 0,
    "first_notice": 2,    # business days before first notice day -- never take delivery
}


def _prev_index_date(idx: pd.DatetimeIndex, target: pd.Timestamp) -> pd.Timestamp | None:
    """Snap to the last trading day at or before `target`. Roll dates must be REAL dates."""
    pos = idx.searchsorted(pd.Timestamp(target), side="right") - 1
    return idx[pos] if pos >= 0 else None


def _tradeable_roll(contracts: pd.DataFrame, front: str, back: str,
                    candidate: pd.Timestamp, floor: pd.Timestamp | None,
                    ceiling: pd.Timestamp) -> pd.Timestamp:
    """Walk a candidate roll date back to the nearest day BOTH legs actually printed.

    Failure prevented: a rule that fires on a day the deferred contract has no quote. The
    gap is then NaN, and back/ratio adjustment propagates that NaN through every price
    before the roll -- silently deleting the first half of your sample.
    """
    idx = contracts.index
    pos = idx.searchsorted(min(candidate, ceiling), side="right") - 1
    while pos >= 0:
        d = idx[pos]
        if floor is not None and d <= floor:
            break
        if np.isfinite(contracts.at[d, front]) and np.isfinite(contracts.at[d, back]):
            return d
        pos -= 1
    raise ValueError(
        f"no tradeable roll date for {front}->{back} at or before "
        f"{min(candidate, ceiling):%Y-%m-%d}"
        + (f" and after the previous roll {floor:%Y-%m-%d}" if floor is not None else "")
        + ". Both legs must quote on the roll date.")


def _crossover(front_stat: pd.Series, back_stat: pd.Series,
               window: pd.DatetimeIndex) -> pd.Timestamp | None:
    """First day inside `window` on which the DEFERRED contract's stat exceeds the front's.

    Requires the crossover to STICK (the deferred stays ahead for the rest of the window),
    because open interest and volume are noisy enough to cross for a single day and cross
    back. Rolling on a one-day head-fake means rolling twice and paying twice.
    """
    f = front_stat.reindex(window).astype(float)
    b = back_stat.reindex(window).astype(float)
    ahead = (b > f).to_numpy()
    if not ahead.any():
        return None
    # last index where the deferred is NOT ahead; the crossover is the day after it
    not_ahead = np.flatnonzero(~ahead)
    first_sticky = 0 if len(not_ahead) == 0 else int(not_ahead[-1]) + 1
    return window[first_sticky] if first_sticky < len(window) else None


def roll_dates(contracts: pd.DataFrame,
               rule: str,
               expiries: Mapping[str, pd.Timestamp] | pd.Series,
               open_interest: pd.DataFrame | None = None,
               volume: pd.DataFrame | None = None,
               first_notice: Mapping[str, pd.Timestamp] | pd.Series | None = None,
               offset: int | None = None,
               search_window: int = 30) -> list[pd.Timestamp]:
    """One roll date per adjacent contract pair, by `rule`.

    'calendar'      : `offset` business days before the front contract's expiry. Simple,
                      auditable, and blind to whether anyone is still trading that contract.
    'open_interest' : first day the DEFERRED contract's open interest sticks above the
                      front's. Closest to where real money is; rolls late, into a front
                      that other people are still leaving.
    'volume'        : same test on volume. Volume migrates BEFORE open interest, so this
                      rolls earlier -- typically into a thinner deferred book.
    'first_notice'  : `offset` business days before first notice day. Mandatory for
                      physically-delivered contracts unless you own a tank farm. Rolls
                      earliest of all, which changes the sampled term structure the most.

    `contracts` columns must be ordered near->far by expiry (same layout as
    `continuous_contract.stitch`). Returns exactly `n_contracts - 1` strictly increasing
    dates, each one a real trading day on which BOTH legs printed.
    """
    if rule not in RULES:
        raise ValueError(f"rule must be one of {RULES}, got {rule!r}")
    off = DEFAULT_OFFSET[rule] if offset is None else int(offset)
    cols = list(contracts.columns)
    idx = contracts.index
    exp = pd.Series(expiries).reindex(cols)
    if exp.isna().any():
        raise ValueError(f"expiries missing for {list(exp[exp.isna()].index)}")
    exp = pd.to_datetime(exp)

    if rule in ("open_interest", "volume"):
        stat = open_interest if rule == "open_interest" else volume
        if stat is None:
            raise ValueError(f"rule={rule!r} needs the {rule} frame; pass it explicitly "
                             f"rather than silently falling back to a calendar roll -- a "
                             f"silent fallback is how you end up reporting a rule you "
                             f"never actually ran")
        if list(stat.columns) != cols or not stat.index.equals(idx):
            raise ValueError(f"{rule} frame must have the same index and columns as "
                             f"contracts")
    if rule == "first_notice":
        if first_notice is None:
            raise ValueError("rule='first_notice' needs first_notice dates per contract")
        fnd = pd.to_datetime(pd.Series(first_notice).reindex(cols))
        if fnd.isna().any():
            raise ValueError(f"first_notice missing for {list(fnd[fnd.isna()].index)}")

    out: list[pd.Timestamp] = []
    prev: pd.Timestamp | None = None
    for k in range(len(cols) - 1):
        front, back = cols[k], cols[k + 1]
        e = exp.iloc[k]
        # Hard ceiling: never roll after the front stops printing, and never on expiry day.
        last_print = contracts[front].last_valid_index()
        ceiling = min(x for x in (e, last_print) if x is not None)
        ceiling = idx[max(0, idx.searchsorted(ceiling, side="right") - 2)]

        if rule == "calendar":
            pos = idx.searchsorted(e, side="right") - 1 - off
            cand = idx[max(pos, 0)]
        elif rule == "first_notice":
            pos = idx.searchsorted(fnd.iloc[k], side="right") - 1 - off
            cand = idx[max(pos, 0)]
        else:
            stat_df = open_interest if rule == "open_interest" else volume
            hi = idx.searchsorted(ceiling, side="right")
            window = idx[max(0, hi - search_window):hi]
            cross = _crossover(stat_df[front], stat_df[back], window)
            # No crossover inside the window means the deferred never took over. Falling
            # back to a calendar roll would quietly change the rule you claim to be
            # testing, so use the LAST day of the window and let it be visible.
            cand = cross if cross is not None else ceiling

        d = _tradeable_roll(contracts, front, back, cand, prev, ceiling)
        out.append(d)
        prev = d
    return out


def roll_costs(contracts: pd.DataFrame, rolls: Sequence[pd.Timestamp],
               open_interest: pd.DataFrame | None = None,
               base_bps: float = 1.5, illiquidity_mult: float = 6.0) -> pd.Series:
    """Round-trip cost of each roll, in bps of notional, indexed by roll date.

    A roll is TWO trades (sell the front, buy the deferred), so the floor is 2 x base_bps.
    On top of that, rolling before the deferred contract has taken over open interest means
    lifting a thin book: the penalty scales with the front's remaining share of total OI.
    This is why "roll earlier to dodge the index-roll squeeze" is not free.
    """
    cols = list(contracts.columns)
    vals = []
    for k, d in enumerate(rolls):
        share = 0.0
        if open_interest is not None:
            f = float(open_interest.at[d, cols[k]])
            b = float(open_interest.at[d, cols[k + 1]])
            if np.isfinite(f) and np.isfinite(b) and (f + b) > 0:
                share = f / (f + b)      # 1.0 = deferred is empty, 0.0 = front is done
        vals.append(2.0 * base_bps * (1.0 + illiquidity_mult * share))
    return pd.Series(vals, index=pd.DatetimeIndex(rolls), name="roll_cost_bps")


def compare_rules(contracts: pd.DataFrame,
                  rules: Sequence[str] = RULES,
                  expiries: Mapping[str, pd.Timestamp] | pd.Series | None = None,
                  open_interest: pd.DataFrame | None = None,
                  volume: pd.DataFrame | None = None,
                  first_notice: Mapping[str, pd.Timestamp] | pd.Series | None = None,
                  base_bps: float = 1.5,
                  periods_per_year: int = TRADING_DAYS) -> pd.DataFrame:
    """One row per roll rule: annualised return, Sharpe and total roll cost.

    Returns come from `continuous_contract.true_roll_return`, so the comparison is of the
    RULES and nothing else -- no stitching method is involved and no adjustment artefact
    can leak in. Read the Sharpe column as a list of trials, because that is what it is.
    """
    if expiries is None:
        raise ValueError("expiries are required")
    rows = []
    for rule in rules:
        rolls = roll_dates(contracts, rule, expiries, open_interest=open_interest,
                           volume=volume, first_notice=first_notice)
        gross = true_roll_return(contracts, rolls)
        cost_bps = roll_costs(contracts, rolls, open_interest, base_bps=base_bps)
        drag = cost_bps.reindex(gross.index).fillna(0.0) / 1e4
        net = gross - drag
        g, n = _stats(gross, periods_per_year), _stats(net, periods_per_year)
        # How far before expiry each roll fired, in trading days -- the rule's fingerprint.
        exp = pd.to_datetime(pd.Series(expiries).reindex(contracts.columns))
        lead = [contracts.index.searchsorted(exp.iloc[k], side="right") - 1
                - contracts.index.searchsorted(d, side="left") for k, d in enumerate(rolls)]
        rows.append({
            "rule": rule, "n_rolls": len(rolls),
            "bd_before_expiry": float(np.mean(lead)),
            "ann_gross": g["ann_return"], "sharpe_gross": g["sharpe"],
            "roll_cost_bps_total": float(cost_bps.sum()),
            "roll_cost_bps_per_roll": float(cost_bps.mean()),
            "ann_net": n["ann_return"], "sharpe_net": n["sharpe"], "vol": n["vol"],
        })
    out = pd.DataFrame(rows).set_index("rule")
    out.attrs["n_days"] = len(true_roll_return(
        contracts, roll_dates(contracts, rules[0], expiries,
                              open_interest=open_interest, volume=volume,
                              first_notice=first_notice)))
    return out


def render(table: pd.DataFrame, title: str = "ROLL RULE COMPARISON") -> str:
    """Fixed-width report. The spread in the Sharpe column IS the researcher degree of
    freedom -- print it next to the backtest, not instead of it."""
    w = 100
    lines = [title, "=" * w,
             f"Same prices, same contracts, same {table.attrs.get('n_days', 0):,} days. "
             f"Only the roll rule changes.",
             "-" * w,
             f"{'rule':<15} {'rolls':>6} {'bd b/f exp':>11} {'ann gross':>10} "
             f"{'Sharpe gr':>10} {'roll cost':>10} {'ann net':>9} {'Sharpe net':>11}",
             "-" * w]
    for rule, r in table.iterrows():
        lines.append(
            f"{rule:<15} {int(r['n_rolls']):>6} {r['bd_before_expiry']:>11.1f} "
            f"{r['ann_gross']:>10.2%} {r['sharpe_gross']:>10.2f} "
            f"{r['roll_cost_bps_total']:>9.0f}b {r['ann_net']:>9.2%} "
            f"{r['sharpe_net']:>11.2f}")
    lines.append("-" * w)
    lo, hi = table["sharpe_net"].min(), table["sharpe_net"].max()
    best, worst = table["sharpe_net"].idxmax(), table["sharpe_net"].idxmin()
    lines.append(f"SHARPE SPREAD: {hi - lo:.2f}  ({best} {hi:+.2f} vs {worst} {lo:+.2f})")
    lines.append(f"RETURN SPREAD: {table['ann_net'].max() - table['ann_net'].min():.2%} "
                 f"annualised, from a parameter most write-ups do not mention.")
    lines.append("=" * w)
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
def _synthetic_market(years: int = 8, seed: int = 3, base_carry: float = 0.06,
                      seasonal: float = 0.30, roll_pressure: float = 0.004,
                      vol: float = 0.20
                      ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
                                 pd.Series, pd.Series]:
    """A quarterly commodity with a SEASONAL term structure. Offline, no network.

    The slope of the curve oscillates on the contract cycle -- true of natural gas, power,
    heating oil and every grain. That is what makes roll timing a real economic choice
    rather than a formatting preference: a rule that always rolls 5 days before expiry
    samples the same phase of the cycle every single quarter, so its edge (or its bleed)
    compounds over 30 rolls instead of averaging away.

    Returns (prices, open_interest, volume, expiries, first_notice_days).
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2016-01-04", periods=TRADING_DAYS * years)
    n = len(idx)
    dt = 1.0 / TRADING_DAYS
    spot = 100.0 * np.exp(np.cumsum(rng.normal(-0.5 * vol ** 2 * dt,
                                               vol * np.sqrt(dt), n)))

    cycle = 63                                     # business days between expiries
    t = np.arange(n)
    # Annualised curve slope: seasonal on the contract cycle + a slow stochastic drift.
    slope = (base_carry + seasonal * np.sin(2 * np.pi * t / cycle)
             + pd.Series(rng.normal(0, 0.010, n)).rolling(40, min_periods=1).mean()
             .to_numpy() * 3.0)

    exp_pos = list(range(60, n, cycle))
    names = [f"F{i:02d}" for i in range(len(exp_pos))]
    px = pd.DataFrame(np.nan, index=idx, columns=names)
    oi = pd.DataFrame(np.nan, index=idx, columns=names)
    vo = pd.DataFrame(np.nan, index=idx, columns=names)

    for i, e in enumerate(exp_pos):
        start = max(0, e - 190)
        rows = np.arange(start, e + 1)
        d2e = (e - rows).astype(float)             # business days to expiry
        tau = d2e / TRADING_DAYS
        basis = rng.normal(0.0, 0.0012, len(rows)).cumsum()
        price = spot[rows] * np.exp(slope[rows] * tau + basis)

        # INDEX-ROLL PRESSURE: in the well-known window everybody rolls in, the expiring
        # contract is sold down and the deferred is bid up. Roll inside it and you pay it.
        squeeze = (d2e >= 4) & (d2e <= 9)
        price = price * np.exp(-roll_pressure * squeeze)
        if i > 0:                                  # this contract is the DEFERRED leg then
            prev_e = exp_pos[i - 1]
            d2prev = (prev_e - rows).astype(float)
            price = price * np.exp(roll_pressure * ((d2prev >= 4) & (d2prev <= 9)))
        px.iloc[start:e + 1, i] = price

        # Open interest peaks ~40bd out; volume peaks earlier, so volume crosses first.
        jitter = rng.integers(-6, 7)
        oi.iloc[start:e + 1, i] = 1e5 * np.exp(-((d2e - (40 + jitter)) ** 2) / (2 * 35 ** 2))
        vo.iloc[start:e + 1, i] = 4e4 * np.exp(-((d2e - (47 + jitter)) ** 2) / (2 * 33 ** 2))

    expiries = pd.Series([idx[e] for e in exp_pos], index=names)
    # First notice day: ~22 business days before expiry, the usual physical-delivery gate.
    first_notice = pd.Series([idx[max(0, e - 22)] for e in exp_pos], index=names)
    return px, oi, vo, expiries, first_notice


if __name__ == "__main__":
    pd.set_option("display.width", 140)
    px, oi, vo, expiries, fnd = _synthetic_market()

    print("=" * 100)
    print(f"SYNTHETIC SEASONAL COMMODITY -- {px.shape[1]} quarterly contracts, "
          f"{len(px):,} business days (~8 years)")
    print("Curve slope oscillates on the contract cycle; an index-roll squeeze sits 4-9 "
          "days before each expiry.")
    print("ONE dataset. ONE strategy (long the front contract). FOUR defensible roll "
          "rules.")
    print("=" * 100)

    table = compare_rules(px, RULES, expiries, open_interest=oi, volume=vo,
                          first_notice=fnd)
    print()
    print(render(table))

    # -------------------------------------------------- where each rule actually fires
    print("\n" + "=" * 100)
    print("WHERE EACH RULE FIRES (first 6 rolls, business days before expiry)")
    print("=" * 100)
    lead = {}
    for rule in RULES:
        rolls = roll_dates(px, rule, expiries, open_interest=oi, volume=vo,
                           first_notice=fnd)
        lead[rule] = [int(px.index.searchsorted(expiries.iloc[k], side="right") - 1
                          - px.index.searchsorted(d, side="left"))
                      for k, d in enumerate(rolls)]
        print(f"{rule:<15} {str(lead[rule][:6]):<24} "
              f"spread {min(lead[rule])}-{max(lead[rule])} bd   "
              f"first roll {rolls[0]:%Y-%m-%d}  last {rolls[-1]:%Y-%m-%d}")
    print("-" * 100)
    print("The open-interest and volume rules DRIFT (the crossover moves contract to "
          "contract);")
    print("the calendar and first-notice rules are pinned. Pinned rules sample the same "
          "phase of the")
    print("seasonal curve every quarter, so their edge compounds instead of averaging out.")

    # ------------------------------------------------------------------ the consequence
    hi_rule = table["sharpe_net"].idxmax()
    lo_rule = table["sharpe_net"].idxmin()
    hi, lo = table.loc[hi_rule, "sharpe_net"], table.loc[lo_rule, "sharpe_net"]
    print("\n" + "=" * 100)
    print("THE CONSEQUENCE")
    print("=" * 100)
    print(f"If you had shipped the vendor default ('calendar', 5 days out) you would "
          f"report Sharpe {table.loc['calendar', 'sharpe_net']:+.2f}.")
    print(f"If you had 'fixed the roll to be more realistic' you would report Sharpe "
          f"{hi:+.2f} ({hi_rule}).")
    print(f"Same prices. Same signal. Same costs. The spread is {hi - lo:.2f} Sharpe and "
          f"{table['ann_net'].max() - table['ann_net'].min():.1%} a year.")
    print()
    print("EACH RULE YOU EVALUATED IS A TRIAL. Picking the best one after seeing the "
          "results is p-hacking,")
    print("and it is invisible in the writeup because a roll rule reads like a data "
          "decision, not a parameter.")
    print()
    print("So: choose the rule ex ante on liquidity and delivery risk, and log every rule "
          "you looked at ->")
    print("  ../../../fin-core/skills/backtest-validation/scripts/trial_ledger.py")
    print("      led = TrialLedger('research/trials.jsonl')")
    for rule in RULES:
        print(f"      led.record(strategy='front_futures', "
              f"params={{'roll_rule': {rule!r}}})")
    print(f"      # n_trials >= {len(RULES)} before you have even touched a signal "
          f"parameter.")
    print(f"      led.deflated_sharpe(best_sharpe={hi:.2f}, n_obs={table.attrs['n_days']})")
    print("=" * 100)
