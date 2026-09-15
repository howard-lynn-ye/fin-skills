"""A cash brokerage account under mainland A-share rules.

Extracted from research/simulations/multi_budget_trading_simulation.py and
cleaned up: fees come from config instead of module globals, stamp duty /
commission / slippage are tracked separately rather than lumped together, and
price-limit checks live here instead of being duplicated by each caller.

What this models, and why each piece matters:

  T+1 settlement   shares bought today cannot be sold today. Modelled by
                   `settled_shares`, which only catches up to `total_shares`
                   at the next `start_of_day_settlement()`. A backtest that
                   ignores this can dodge a drawdown it would have eaten.

  Board lots       orders round down to 100 shares. At 50k of capital a
                   600-yuan stock is simply unbuyable, which silently changes
                   the achievable universe -- this is a real constraint, not a
                   rounding detail.

  Price limits     a name locked at +-10% at the open cannot be traded in the
                   direction you want. Ignoring this lets a backtest sell into
                   a limit-down, which nobody can do.

  Stamp duty       0.05% on sells only. Tracked separately because in every
                   experiment so far it, not alpha, decided the outcome.
"""
from __future__ import annotations

from config import (BOARD_LOT, COMMISSION_MIN, COMMISSION_RATE,
                    LIMIT_TOUCH_EPS, PRICE_LIMIT_PCT, SLIPPAGE,
                    STAMP_DUTY_SELL)


def limit_up_price(preclose: float) -> float:
    return round(preclose * (1.0 + PRICE_LIMIT_PCT), 2)


def limit_down_price(preclose: float) -> float:
    return round(preclose * (1.0 - PRICE_LIMIT_PCT), 2)


def is_locked_up(price: float, preclose: float) -> bool:
    """True when the name is pinned at the upper limit -- buyers cannot fill."""
    return abs(price - limit_up_price(preclose)) < LIMIT_TOUCH_EPS


def is_locked_down(price: float, preclose: float) -> bool:
    """True when the name is pinned at the lower limit -- sellers cannot fill."""
    return abs(price - limit_down_price(preclose)) < LIMIT_TOUCH_EPS


class BrokerAccount:
    """Cash account. No margin, no shorting -- which is the A-share retail reality."""

    def __init__(self, initial_cash: float):
        self.initial_cash = float(initial_cash)
        self.cash = float(initial_cash)
        # code -> {"total_shares", "settled_shares", "avg_cost"}
        self.positions: dict[str, dict] = {}

        self.stamp_duty_paid = 0.0
        self.commission_paid = 0.0
        self.slippage_paid = 0.0
        self.n_buys = 0
        self.n_sells = 0
        self.gross_traded = 0.0   # denominator for turnover accounting
        self.blocked_by_limit = 0

    # ------------------------------------------------------------- accounting
    @property
    def friction_paid(self) -> float:
        return self.stamp_duty_paid + self.commission_paid + self.slippage_paid

    def market_value(self, prices: dict[str, float]) -> float:
        return sum(p["total_shares"] * prices.get(c, p["avg_cost"])
                   for c, p in self.positions.items())

    def equity(self, prices: dict[str, float]) -> float:
        return self.cash + self.market_value(prices)

    def shares_of(self, code: str) -> int:
        pos = self.positions.get(code)
        return pos["total_shares"] if pos else 0

    # ----------------------------------------------------------------- T+1
    def start_of_day_settlement(self) -> None:
        for pos in self.positions.values():
            pos["settled_shares"] = pos["total_shares"]

    def accrue_cash_interest(self, daily_rate: float) -> None:
        """Idle cash is not dead money -- it earns reverse-repo-ish interest.

        Omitting this makes every cash-heavy arm look worse than it is, which
        biases the comparison toward staying invested.
        """
        self.cash *= (1.0 + daily_rate)

    # ------------------------------------------------------------- execution
    def sell(self, code: str, shares: int, price: float,
             preclose: float | None = None) -> int:
        """Sell up to `shares`. Returns shares actually filled."""
        pos = self.positions.get(code)
        if pos is None or price <= 0:
            return 0
        if preclose is not None and is_locked_down(price, preclose):
            self.blocked_by_limit += 1
            return 0

        fill = min(int(shares), pos["settled_shares"])
        fill = (fill // BOARD_LOT) * BOARD_LOT if fill < pos["total_shares"] else fill
        if fill <= 0:
            return 0

        gross = fill * price
        duty = gross * STAMP_DUTY_SELL
        comm = max(COMMISSION_MIN, gross * COMMISSION_RATE)
        slip = gross * SLIPPAGE

        self.cash += gross - duty - comm - slip
        self.stamp_duty_paid += duty
        self.commission_paid += comm
        self.slippage_paid += slip
        self.gross_traded += gross
        self.n_sells += 1

        pos["total_shares"] -= fill
        pos["settled_shares"] -= fill
        if pos["total_shares"] <= 0:
            del self.positions[code]
        return fill

    def buy(self, code: str, shares: int, price: float,
            preclose: float | None = None) -> int:
        """Buy up to `shares`, shrinking the order until cash covers it."""
        if price <= 0:
            return 0
        if preclose is not None and is_locked_up(price, preclose):
            self.blocked_by_limit += 1
            return 0

        fill = (int(shares) // BOARD_LOT) * BOARD_LOT
        while fill >= BOARD_LOT:
            gross = fill * price
            comm = max(COMMISSION_MIN, gross * COMMISSION_RATE)
            slip = gross * SLIPPAGE
            if gross + comm + slip <= self.cash:
                break
            fill -= BOARD_LOT
        if fill < BOARD_LOT:
            return 0

        gross = fill * price
        comm = max(COMMISSION_MIN, gross * COMMISSION_RATE)
        slip = gross * SLIPPAGE

        self.cash -= gross + comm + slip
        self.commission_paid += comm
        self.slippage_paid += slip
        self.gross_traded += gross
        self.n_buys += 1

        pos = self.positions.get(code)
        if pos is None:
            self.positions[code] = {"total_shares": fill,
                                    "settled_shares": 0,  # T+1
                                    "avg_cost": price}
        else:
            new_total = pos["total_shares"] + fill
            pos["avg_cost"] = (pos["total_shares"] * pos["avg_cost"] + gross) / new_total
            pos["total_shares"] = new_total
        return fill

    def rebalance_to(self, targets: dict[str, float], prices: dict[str, float],
                     preclose: dict[str, float], deadband: float) -> None:
        """Move holdings toward `targets` (weights of current equity).

        Sells run first so their proceeds can fund the buys -- the opposite
        order silently caps how much the strategy can actually take on.
        """
        equity = self.equity(prices)
        if equity <= 0:
            return

        held = set(self.positions) | set(targets)
        orders = []
        for code in held:
            px = prices.get(code)
            if px is None or px <= 0:
                continue
            cur_val = self.shares_of(code) * px
            tgt_val = targets.get(code, 0.0) * equity
            # Deadband applies only to trimming/adding existing positions, never blocks entry or exit
            if cur_val > 0 and tgt_val > 0:
                # Relative drift threshold: skip rebalancing if position is within 25% of target
                if abs(tgt_val - cur_val) / tgt_val < 0.25:
                    continue
            elif cur_val == 0 and tgt_val == 0:
                continue
            orders.append((code, tgt_val - cur_val, px))

        for code, delta, px in sorted(orders, key=lambda o: o[1]):
            if delta < 0:
                want = int(round(-delta / px))
                # A full exit should not be left stranded by lot rounding.
                if targets.get(code, 0.0) == 0.0:
                    want = self.shares_of(code)
                self.sell(code, want, px, preclose.get(code))
            else:
                self.buy(code, int(delta / px), px, preclose.get(code))
