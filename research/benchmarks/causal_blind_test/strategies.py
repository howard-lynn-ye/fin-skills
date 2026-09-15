"""The six pre-registered strategy arms.

PRE-REGISTRATION. These six arms were fixed, with these parameters, before any
of them was run on this data. Nothing here was tuned. Every arm reports its
result whether it wins or loses, and the count of six is what feeds the
multiple-testing correction in the trial ledger. That is the whole point: a
strategy chosen after seeing six backtests is not a strategy, it is a lookup.

Parameter provenance -- all from the literature or from convention, none fitted:
  60-day moving average     the textbook trend filter; already falsified once
                            in this project (-13.05%) and re-run here so the
                            failure is on the record rather than in a memory.
  50% breadth threshold     the classic advance/decline reading; "half the
                            market is above its own average" is the definition,
                            not a chosen cut point.
  12% target volatility     Moreira & Muir (2017) scale to a constant target;
                            12% is the conventional long-only equity number.
  20-day realised vol       the standard estimation window for that paper's
                            one-month-ahead scaling.
  Top 6 of 50 by 20d return the plain cross-sectional momentum formation
                            window; 6 names is ~12% each, below the 20% cap.

THE CAUSALITY RULE, enforced by construction: a strategy is handed `history`,
which is the data up to and including day t-1 close. Day t itself is invisible.
Orders produced here fill at day t's open. There is no path by which a decision
can see the bar it trades on.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import (LOOKBACK_LONG, LOOKBACK_MED, MAX_SINGLE_WEIGHT, N_SELECTED,
                    REBALANCE_EVERY, TARGET_ANNUAL_VOL, TRADING_DAYS)

BREADTH_MA = 50
BREADTH_THRESHOLD = 0.50
VOL_WINDOW = 20


class Strategy:
    """Base arm. `history` never contains the day being traded."""

    name = "base"
    description = ""

    def weights(self, step: int, history: pd.DataFrame,
                index_history: pd.Series) -> dict[str, float]:
        raise NotImplementedError

    def trades_today(self, step: int) -> bool:
        return True


class CashOnly(Strategy):
    name = "cash_only"
    description = "Hold nothing. Earns the 2% risk-free rate. The null hypothesis."

    def weights(self, step, history, index_history):
        return {}


class BuyAndHold(Strategy):
    name = "buy_and_hold"
    description = ("Equal weight across the point-in-time universe on day one, "
                   "then never trade again. Pays friction exactly once.")

    def weights(self, step, history, index_history):
        codes = list(history.columns)
        return {c: 1.0 / len(codes) for c in codes}

    def trades_today(self, step):
        return step == 0


class CrossSectionalMomentum(Strategy):
    name = "xs_momentum_weekly"
    description = (f"Every {REBALANCE_EVERY} days hold the top {N_SELECTED} of the "
                   f"universe by trailing {LOOKBACK_MED}-day return, equal weighted.")

    def weights(self, step, history, index_history):
        if len(history) < LOOKBACK_MED + 1:
            return {}
        window = history.iloc[-(LOOKBACK_MED + 1):]
        mom = (window.iloc[-1] / window.iloc[0] - 1.0).dropna()
        if mom.empty:
            return {}
        picks = mom.sort_values(ascending=False).head(N_SELECTED).index
        w = min(1.0 / len(picks), MAX_SINGLE_WEIGHT)
        return {c: w for c in picks}

    def trades_today(self, step):
        return step % REBALANCE_EVERY == 0


class MovingAverageTiming(Strategy):
    name = "ma60_timing"
    description = (f"Equal-weight the universe while the index sits above its "
                   f"{LOOKBACK_LONG}-day moving average; otherwise hold cash.")

    def weights(self, step, history, index_history):
        if len(index_history) < LOOKBACK_LONG:
            return {}
        ma = index_history.iloc[-LOOKBACK_LONG:].mean()
        if index_history.iloc[-1] <= ma:
            return {}
        codes = list(history.columns)
        return {c: 1.0 / len(codes) for c in codes}

    def trades_today(self, step):
        return step % REBALANCE_EVERY == 0


class VolatilityTargeting(Strategy):
    name = "vol_target_timing"
    description = (f"Moreira & Muir (2017): scale equal-weight exposure by "
                   f"{TARGET_ANNUAL_VOL:.0%} / realised {VOL_WINDOW}-day volatility, "
                   f"capped at 1.0. No leverage -- the cap is what makes it a "
                   f"long-only retail strategy rather than the paper's version.")

    def weights(self, step, history, index_history):
        if len(history) < VOL_WINDOW + 1:
            return {}
        rets = history.pct_change().iloc[-VOL_WINDOW:]
        port = rets.mean(axis=1)              # equal-weight portfolio return
        realised = float(port.std()) * np.sqrt(TRADING_DAYS)
        if not np.isfinite(realised) or realised <= 1e-8:
            return {}
        exposure = min(1.0, TARGET_ANNUAL_VOL / realised)
        codes = list(history.columns)
        return {c: exposure / len(codes) for c in codes}

    def trades_today(self, step):
        return step % REBALANCE_EVERY == 0


class BreadthTiming(Strategy):
    name = "breadth_timing"
    description = (f"Fully invested when at least {BREADTH_THRESHOLD:.0%} of the "
                   f"universe trades above its own {BREADTH_MA}-day average; "
                   f"otherwise flat. Breadth is measured on the names held, not "
                   f"on an index, so it cannot be distorted by index weights.")

    def weights(self, step, history, index_history):
        if len(history) < BREADTH_MA:
            return {}
        window = history.iloc[-BREADTH_MA:]
        above = (window.iloc[-1] > window.mean()).mean()
        if above < BREADTH_THRESHOLD:
            return {}
        codes = list(history.columns)
        return {c: 1.0 / len(codes) for c in codes}

    def trades_today(self, step):
        return step % REBALANCE_EVERY == 0


def all_strategies() -> list[Strategy]:
    """The registered set. Its length is `m` for the multiple-testing correction."""
    return [CashOnly(), BuyAndHold(), CrossSectionalMomentum(),
            MovingAverageTiming(), VolatilityTargeting(), BreadthTiming()]
