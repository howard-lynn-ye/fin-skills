"""Stateful fly controller: continuing cash/share account, true HOLD, mature rewards."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from benchmarks.verified_memory.fly_v2 import Eligibility, MushroomBody, PARAMETERS
from benchmarks.verified_memory.model import WARMUP, HORIZON, expert_weights, features
from fin_skills.api import get
from fin_skills.synthesis import Fact, Timeline, price_fact

ACTIONS = ("hold", "equal_weight", "momentum", "reversal", "inverse_vol",
           "cash", "halve", "increase")
ARMS = {
    "fly_v3": ("fly_trace", "full", True, "account_net"),
    "fly_no_hold": ("fly_trace", "full", False, "account_net"),
    "fly_market_only": ("fly_trace", "market_only", True, "account_net"),
    "fly_no_cost_input": ("fly_trace", "no_cost", True, "account_net"),
    "fly_hold_advantage": ("fly_trace", "full", True, "hold_advantage"),
    "dense_v3": ("dense_trace", "full", True, "account_net"),
    "linear_v3": ("linear_trace", "full", True, "account_net"),
    "fly_frozen_v3": ("fly_frozen", "full", True, "account_net"),
}


@dataclass
class Account:
    cash: float
    units: np.ndarray
    last_trade: int

    @classmethod
    def fresh(cls, assets, t):
        return cls(1., np.zeros(assets), t)

    def nav(self, prices):
        return float(self.cash + self.units @ prices)

    def weights(self, prices):
        return self.units * prices / self.nav(prices)

    def rebalance(self, target, prices, t, cost_bps):
        """Fixed-point self-financing equation, separate from final scorer's bisection."""
        if target is None:
            return {"fee": 0., "turnover": 0.}
        w = np.asarray(target, dtype=float)
        if w.shape != self.units.shape or not np.isfinite(w).all() or (w < 0).any() or w.sum() > 1 + 1e-10:
            raise ValueError("invalid long-only target")
        if not 0 <= cost_bps <= 100:
            raise ValueError("cost out of range")
        before = self.nav(prices)
        holdings = self.units * prices
        after = before
        c = cost_bps / 10000
        for _ in range(30):
            nxt = before - c * np.abs(after * w - holdings).sum()
            if abs(nxt - after) <= 1e-14 * before:
                after = float(nxt)
                break
            after = float(nxt)
        traded = float(np.abs(after * w - holdings).sum())
        if abs(after + c * traded - before) > 1e-11 * before:
            raise ArithmeticError("self-financing equation did not converge")
        self.units = after * w / prices
        self.cash = float(after * (1 - w.sum()))
        if traded > 1e-10 * before:
            self.last_trade = t
        return {"fee": c * traded, "turnover": traded / before}


def targets_for(account, prices, experts):
    current = account.weights(prices)
    exposure = current.sum()
    increase = current * min(1.25, 1 / exposure) if exposure > 1e-12 else np.full(len(current), 0.25 / len(current))
    return [None, *[w.copy() for w in experts], current * 0.5, increase]


def observation(market, account, prices, targets, t, cost_bps, mode):
    current = account.weights(prices)
    cash_weight = account.cash / account.nav(prices)
    age = np.tanh((t - account.last_trade) / 60)
    costs = [0 if w is None else cost_bps / 10000 * np.abs(w - current).sum() for w in targets]
    context = np.r_[current, cash_weight, age, np.tanh(np.asarray(costs) / 0.005)]
    if mode == "market_only":
        context[:] = 0
    elif mode == "no_cost":
        context[-len(ACTIONS):] = 0
    elif mode != "full":
        raise ValueError(mode)
    return np.r_[market, context], costs


def choose(brain, z, t, hold_allowed):
    q = brain.values(z)
    allowed = np.ones(len(ACTIONS), dtype=bool)
    allowed[0] = hold_allowed
    best = np.isclose(q, q[allowed].max(), rtol=0, atol=1e-12) & allowed
    p = PARAMETERS["epsilon"] * allowed / allowed.sum()
    p[best] += (1 - PARAMETERS["epsilon"]) / best.sum()
    action = int(brain.rng.choice(len(ACTIONS), p=p))
    return Eligibility(t, z.copy(), action, float(q[action])), p


def receipt_for(active, account, prices, t, cost_bps, *, terminal=False):
    """Close an action's account interval before the next action's fees are charged."""
    before_terminal = account.nav(prices)
    exit_fee = cost_bps / 10000 * float(account.units @ prices) if terminal else 0.
    end_nav = before_terminal - exit_fee
    # Independently reconstruct the endpoint from the saved post-fill state.
    reconstructed = float(active["cash_after"] + np.asarray(active["units_after"]) @ prices) - exit_fee
    hold_nav = float(active["cash_before"] + np.asarray(active["units_before"]) @ prices)
    if terminal:
        hold_nav -= cost_bps / 10000 * float(np.asarray(active["units_before"]) @ prices)
    base = active["base"]
    net = end_nav / base - 1
    hold_net = hold_nav / base - 1
    return {"origin": active["memory"].origin, "fill": active["fill"], "available": t,
            "base_nav": base, "end_nav": end_nav, "reconstructed_nav": reconstructed,
            "net_reward": net, "hold_reward": hold_net, "hold_advantage": net - hold_net,
            "fee_at_entry": active["fee"], "fee_at_exit": exit_fee,
            "selected_action": active["memory"].action, "terminal": terminal}


def audit_receipt(receipt, index):
    parents = (
        price_fact("base_nav", "internal:account", receipt["base_nav"], index[receipt["fill"]], source="cash_share_account"),
        price_fact("end_nav", "internal:account", receipt["end_nav"], index[receipt["available"]], source="cash_share_account"),
    )
    stamped = receipt.get("claimed_available", receipt["available"])
    stamp = index[stamped]
    fact = Fact("net_reward", "internal:account", receipt["net_reward"], stamp, stamp,
                stamp, kind="derived", inputs=parents, source="account_reward_adapter")
    checked = get("synthesis_integrity").run(timeline=Timeline([*parents, fact]))
    scale = receipt["base_nav"]
    accounting = (np.isfinite(list(v for v in receipt.values() if isinstance(v, float))).all()
                  and scale > 0
                  and abs(receipt["end_nav"] - receipt["reconstructed_nav"]) <= 1e-10 * scale
                  and abs(receipt["net_reward"] - (receipt["end_nav"] / scale - 1)) <= 1e-10
                  and abs(receipt["hold_advantage"] - receipt["net_reward"] + receipt["hold_reward"]) <= 1e-10)
    return {"passed": bool(checked.passed and accounting), "guard": checked.guard,
            "lineage_passed": checked.passed, "accounting_passed": bool(accounting)}


def prepare_market(prices):
    if prices.shape[1] != 4 or not np.isfinite(prices).all().all() or (prices <= 0).any().any():
        raise ValueError("finite positive four-column prices required")
    if not prices.index.is_monotonic_increasing or prices.index.has_duplicates:
        raise ValueError("strict chronology required")
    schedule = list(range(WARMUP, len(prices) - 1, HORIZON))
    return features(prices).to_numpy(), {t: expert_weights(prices, t) for t in schedule}


def simulate(prices, arm, seed, *, start, end, cost_bps=5, prepared=None,
             fixed_action=None, stop_before=None):
    """Account advances one observation at a time. Only matured receipts train memory.

    Evaluation starts from cash; learner weights persist from warm-up. Any pending
    training order and unfinished training reward interval are explicitly cancelled
    at this reset. Full evaluation NAV is reconciled against an independent scorer.
    stop_before is a causal-prefix test hook; it does not liquidate or read later rows.
    """
    if arm not in ARMS or not WARMUP <= start < end < len(prices):
        raise ValueError("unknown arm or invalid interval")
    variant, input_mode, hold_allowed, reward_mode = ARMS[arm]
    market, experts = prepared if prepared is not None else prepare_market(prices)
    values = prices.to_numpy()
    brain = MushroomBody(variant, seed, dimension=34, actions=len(ACTIONS))
    account = Account.fresh(4, WARMUP)
    pending, active, queue = None, None, []
    receipts, updates, actions, navs = [], [], [], []
    turnovers, fees, resets = [], [], []
    until = end + 1 if stop_before is None else min(end + 1, stop_before)

    def close_interval(t, terminal=False):
        receipt = receipt_for(active, account, values[t], t, cost_bps, terminal=terminal)
        receipt["audit"] = audit_receipt(receipt, prices.index)
        receipts.append(receipt)
        queue.append((active["memory"], receipt))

    for t in range(WARMUP, until):
        if t == start:
            resets.append({"at": t, "cancelled_order": pending[0].origin if pending else None,
                           "cancelled_reward": active["memory"].origin if active else None})
            account, pending, active = Account.fresh(4, t), None, None
        px = values[t]
        fee, turnover = 0., 0.
        if pending is not None:
            memory, target = pending
            if memory.origin + 1 != t:
                raise AssertionError("order execution lag must be one observation")
            if active is not None:
                close_interval(t)
            base = account.nav(px)
            before_cash, before_units = account.cash, account.units.copy()
            fill = account.rebalance(target, px, t, cost_bps)
            fee, turnover = fill["fee"], fill["turnover"]
            active = {"memory": memory, "fill": t, "base": base, "fee": fee,
                "cash_before": before_cash, "units_before": before_units.tolist(),
                "cash_after": account.cash, "units_after": account.units.tolist()}
            pending = None
        if t == end:
            if active is not None:
                close_interval(t, terminal=True)
            liquidation = account.rebalance(np.zeros(4), px, t, cost_bps)
            fee += liquidation["fee"]
            turnover += liquidation["turnover"]
        if t > start:
            navs.append(account.nav(px))
            fees.append(float(fee))
            turnovers.append(float(turnover))
        if t not in experts or t >= end:
            continue
        targets = targets_for(account, px, experts[t])
        x, costs = observation(market[t], account, px, targets, t, cost_bps, input_mode)
        z = brain.encode(x)
        waiting = []
        for memory, receipt in queue:
            if receipt["available"] >= t:
                waiting.append((memory, receipt))
                continue
            ok = receipt["audit"]["passed"]
            update = {"origin": memory.origin, "at": t, "available": receipt["available"],
                      "applied": ok, "net_reward": receipt["net_reward"],
                      "hold_advantage": receipt["hold_advantage"], "action": memory.action}
            if ok:
                raw = receipt["net_reward"] if reward_mode == "account_net" else receipt["hold_advantage"]
                update.update(brain.reinforce(memory, raw / PARAMETERS["market_reward_scale"], t, current_z=z))
            updates.append(update)
        queue = waiting
        if fixed_action is None:
            memory, probabilities = choose(brain, z, t, hold_allowed)
        else:
            probabilities = np.eye(len(ACTIONS))[fixed_action]
            memory = Eligibility(t, z.copy(), fixed_action, float(brain.values(z)[fixed_action]))
        target = targets[memory.action]
        pending = memory, target
        if t >= start:
            actions.append({"decision_index": t, "weights": None if target is None else target.tolist(),
                "action": ACTIONS[memory.action], "prediction": memory.prediction,
                "choice_probabilities": probabilities.tolist(),
                "current_weights": account.weights(px).tolist(),
                "observations_since_trade": t - account.last_trade,
                "estimated_fee_fraction": float(costs[memory.action])})
    return {"actions": actions, "updates": updates, "receipts": receipts, "nav": navs,
            "fees": fees, "daily_turnover": turnovers, "reset_events": resets,
            "audit": {"blocked_receipts": sum(not r["audit"]["passed"] for r in receipts),
                "early_updates": sum(u["available"] >= u["at"] for u in updates if u["applied"]),
                "negative_rewards_learned": sum(u["net_reward"] < 0 for u in updates if u["applied"]),
                "hold_fraction": sum(a["action"] == "hold" for a in actions) / max(len(actions), 1),
                "actual_fees": float(sum(fees)), **brain.diagnostics()}}
