"""fin_skills.engine - the reference backtester, written so the library's own guards can falsify it.

It is not fast, it is not event-driven and it does not trade. It exists for one property no
other engine in `backtesting-engines` has: **its result is a Bundle**, so every guard in
this library runs on a real run with no adapter.

    from fin_skills.api import check
    from fin_skills.engine import Sessions, Panel, Universe, Execution, Costs, run

    sessions = Sessions.from_index(pd.bdate_range("2020-01-01", "2023-12-29"),
                                   tz="America/New_York", periods_per_year=252, name="XNYS")
    panel = Panel.from_bars(bars, sessions, adjustment="raw+factors",
                            actions=actions, listings=listings).readjust("back")
    res = run(panel, momentum, execution=Execution(fill_at="next_open", signal_lag=1),
              costs=Costs(commission_bps=0.5, spread_bps=1.0, cash_rate=0.05),
              universe=Universe(rebalance="ME", min_adv=8e6), benchmark=spy)
    print(res.to_bundle().coverage().summary())   # what is ready, what each gap unlocks
    print(check(res.to_bundle()).summary())       # six guards, no hand-written adapter

The twelve invariants it holds, and where each is proven (`invariants.describe()` prints
this list with the test names):

    I1  causality        no output cell before bar k moves when rows >= k are perturbed
    I2  signal lag       Execution(signal_lag=0) raises; a bar-t signal fills at t+lag
    I3  no same-bar fill every fill is dated strictly after its decision bar
    I4  adjustment       the convention is declared, and checked against the action table
    I5  delisting        a booked closeout at the last price, never a NaN that vanishes
    I6  point-in-time    universe[d] uses only rows <= d
    I7  turnover         turnover[t] == |weights[t] - weights[t-1]|.sum(), exactly
    I8  costs            net(0) IS gross; net(b) is non-increasing in b
    I9  calendar         every index is a subset of the DECLARED sessions; windows are bars
    I10 participation    no fill exceeds cap * volume; the remainder carries, never vanishes
    I11 cash             cash is a position; held weights plus cash weight are exactly 1
    I12 determinism      two runs are identical; folds share no state

Seven of them - I1, I4, I5, I6, I8, I11, I12 - are proven by running `assert_causal`,
`adjustment_check`, `survivorship_audit`, `pit_universe`, `cost_curve`, `rf_convention`
and `fold_leak_test` on the engine's own output. Six is the number
`check(result.to_bundle())` runs unaided: fold_leak_test needs `run_fold` and `folds`,
which are not artefacts of a single run. What the engine refuses to do is documented in
`backtesting-engines/SKILL.md` section 5; the short version is that it has no intrabar
model, no order types beyond market, no margin or borrow availability, and no live path.
"""
from __future__ import annotations

from fin_skills.engine import invariants
from fin_skills.engine.actions import cum_factor, normalize_actions, normalize_listings
from fin_skills.engine.costs import (SlippageContext, SlippageModel, fixed_bps, spread_share,
                                     square_root_impact, volume_share)
from fin_skills.engine.execute import run
from fin_skills.engine.invariants import INVARIANTS
from fin_skills.engine.panel import Panel
from fin_skills.engine.result import BacktestResult
from fin_skills.engine.sizing import (Sizer, SizingContext, equal_weight, fractional_kelly,
                                      from_weights, rank_long_short, vol_target)
from fin_skills.engine.spec import Costs, Execution, Sessions
from fin_skills.engine.universe import Universe

__all__ = [
    "BacktestResult", "Costs", "Execution", "INVARIANTS", "Panel", "Sessions", "Sizer",
    "SizingContext", "SlippageContext", "SlippageModel", "Universe", "cum_factor",
    "equal_weight", "fixed_bps", "fractional_kelly", "from_weights", "invariants",
    "normalize_actions", "normalize_listings", "rank_long_short", "run", "spread_share",
    "square_root_impact", "vol_target", "volume_share",
]
