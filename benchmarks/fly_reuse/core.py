"""Reuse upstream memory and paper execution; make financial timing explicit.

Upstream dependencies are loaded from separate pinned checkouts, not copied here.
The actor/critic, risk gate and market-time adapter are engineering hypotheses.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import math
import sys

import numpy as np
import pandas as pd


def activate(upstream):
    root = Path(upstream).resolve()
    for suffix in ("fruit-fly-fund", "fruit-fly-fund/vendor", "stonkfly-lab"):
        path = root / suffix
        if not path.is_dir():
            raise ValueError(f"Missing upstream checkout: {path}")
        sys.path.insert(0, str(path))


def settings(learning=True, fee_bps=8):
    from stonkfly.config import Settings
    return Settings(capital="10", order_limit="10", loss_stop="10",
                    paper_fee=str(fee_bps / 10000), learning=learning)


def quote(price, timestamp, half_spread_bps=2.5):
    from stonkfly.config import D
    from stonkfly.market import Quote
    from flyvsly.market import INCREMENTS
    mid, half = D(price), D(half_spread_bps) / D(10000)
    return Quote("BTC-USDC", (mid * (1-half)).quantize(D(".01")),
                 (mid * (1+half)).quantize(D(".01")), float(timestamp),
                 INCREMENTS["base_increment"], INCREMENTS["quote_increment"],
                 INCREMENTS["price_increment"], INCREMENTS["minimum_quote"],
                 INCREMENTS["minimum_base"])


class Account:
    """Use fruit-fly-fund's clock/guard and Stonkfly's Decimal ledger/broker.

    An independent fin-skills cost/cash/quantity calculation checks every fill.
    The spread is in the execution quote, so it is not charged a second time.
    """
    def __init__(self, path, fee_bps=8):
        from stonkfly.ledger import Ledger
        from stonkfly.broker import PaperBroker
        from flyvsly.replay import ReplayGuard, VirtualClock
        self.s = settings(fee_bps=fee_bps)
        self.ledger = Ledger(path, self.s, "paper")
        self.clock = VirtualClock(0)
        self.guard = ReplayGuard(self.s, self.ledger, self.clock, Path(path).with_suffix(".STOP"))
        self.broker = PaperBroker(self.s, self.ledger)
        self.cash, self.units = 10.0, 0.0
        self.mark_price = 0.0
        self.fills = []
        self.vetoes = []
        self.fee_bps = fee_bps

    def nav(self, q):
        self.mark_price = float(q.bid)
        nav = float(self.ledger.equity({q.product: q}))
        independent = self.cash + self.units * float(q.bid)
        if not math.isclose(nav, independent, rel_tol=1e-10, abs_tol=1e-9):
            raise AssertionError("Independent fin-skills accounting disagrees")
        return nav

    def held(self):
        return self.units * self.mark_price >= 1.0

    def execute(self, target, q, decision_time):
        from stonkfly.risk import Veto
        from fin_skills.engine.costs import trade_charge
        from fin_skills.engine.spec import Costs
        if q.timestamp <= decision_time:
            raise ValueError("Fills must occur strictly after the decision")
        side = "BUY" if target else "SELL"
        # LONG means maintain exposure, not buy another lot at every observation.
        if (target and self.held()) or (not target and not self.held()):
            return
        self.clock.set(q.timestamp)
        try:
            plan = self.guard.plan(q.product, side, {q.product: q})
            plan = self.ledger.reserve(plan, q.timestamp)
            fill = self.broker.execute(plan, self.guard.before_submit)
        except Veto as exc:
            self.vetoes.append(str(exc))
            return
        amount, qty, fee = (float(fill[k]) for k in ("quote", "base", "fee"))
        independent_fee = float(trade_charge(pd.Series([amount]), pd.Series([0.0]),
                                             Costs(commission_bps=self.fee_bps)).iloc[0])
        if not math.isclose(fee, independent_fee, abs_tol=1e-10):
            raise AssertionError("Fee mismatch")
        sign = 1 if side == "BUY" else -1
        self.cash -= sign * amount + independent_fee
        self.units += sign * qty
        self.fills.append({"decision_time": decision_time, "fill_time": q.timestamp,
                           "side": side, "amount": amount, "quantity": qty, "fee": fee})
        self.nav(q)

    def close(self):
        self.ledger.close()


class Compact:
    """Unmodified upstream KC/MBON computation, with action-specific TD feedback.

    All trainable arms update on the same completed batches. The shuffled arm
    permutes rewards within each matured batch, not their signs. Validation and
    test never update weights. Opportunity encoding excludes account/partner
    channels; a separate critic receives inventory and the optional gate sees risk.
    """
    def __init__(self, kind, seed, risk_threshold, batch_size=16):
        from flylab.config import Settings
        from flylab.mb import MushroomBody
        from flylab.odors import OdorEncoder
        self.kind, self.seed = kind, seed
        self.s = Settings(seed=seed, eta=0.01)
        self.brain = MushroomBody(self.s, "adapter", seed)
        self.encoder = OdorEncoder(self.s.n_pn)
        self.rng = np.random.default_rng(seed)
        self.shuffle_rng = np.random.default_rng(seed + 10000)
        self.linear = np.zeros(self.s.n_kc)
        self.critic = np.zeros(self.s.n_kc + 2)
        self.risk_threshold = risk_threshold
        self.batch_size = batch_size
        self.batch = []
        self.shuffle_records = []
        self.updates = 0
        self.gates = 0

    def encode(self, closes, timestamp, held):
        from flylab.odors import GROUPS, _one_hot_fill
        ret = float(closes[-1] / closes[-2] - 1)
        vol = float(np.std(np.diff(np.log(closes[-21:]))))
        odor = self.encoder.encode(ret=ret, vol=vol, position_qty=0,
                                   include_partner=False)
        # Replace the upstream wall-clock session channel with market UTC time.
        hour = pd.Timestamp(timestamp, unit="s", tz="UTC").hour
        session = 0 if hour < 7 else 1 if hour < 13 else 2 if hour < 21 else 3
        sl = self.encoder.layout["session"]
        width = sl.stop - sl.start
        center = round(session / (len(GROUPS["session"])-1) * (width-1))
        odor.vector[sl] = _one_hot_fill(width, center)
        # Keep account state out of opportunity memory, including the flat label.
        odor.vector[self.encoder.layout["pos"]] = 0
        state = self.brain.step(odor.vector)
        kc = state.kc.copy()
        x = kc / max(float(kc.sum()), 1.0)
        return {"kc": kc, "x": x, "cx": np.r_[x, 1.0, float(held)], "vol": vol}

    def decide(self, state, training):
        x, kc = state["x"], state["kc"]
        if self.kind.startswith("ordinary"):
            score = float(self.linear @ x)
        else:
            buy = float((kc @ self.brain.w_buy).mean())
            sell = float((kc @ self.brain.w_sell).mean())
            score = (buy - sell) / (abs(buy) + abs(sell) + 1e-12)
        probability = 1 / (1 + math.exp(-float(np.clip(3 * score, -20, 20))))
        target = int(self.rng.random() < probability) if training else int(score > 0)
        gated = self.kind.endswith("gated") and state["vol"] > self.risk_threshold
        self.gates += int(gated)
        return {**state, "action": 0 if gated else target, "probability": probability,
                "gated": bool(gated), "score": score}

    def receive(self, previous, reward, next_state, available_at, training):
        if not training or self.kind in ("frozen", "frozen_gated"):
            return
        self.batch.append({"previous": previous, "reward": float(reward),
                           "next": next_state, "available_at": available_at})
        if len(self.batch) >= self.batch_size:
            self.flush(available_at)

    def flush(self, now):
        if not self.batch:
            return
        if any(row["available_at"] > now for row in self.batch):
            raise AssertionError("Unmatured reward")
        n = len(self.batch)
        order = np.arange(n)
        if self.kind == "shuffled" and n > 1:
            cycle = self.shuffle_rng.permutation(n)
            order[cycle] = np.roll(cycle, 1)  # actual derangement, identical marginal rewards
            self.shuffle_records.append({"size": n, "order": order.tolist(), "at": now})
        rewards = [row["reward"] for row in self.batch]
        for i, row in enumerate(self.batch):
            old, nxt = row["previous"], row["next"]
            delta = 100 * rewards[int(order[i])] + 0.95 * float(self.critic @ nxt["cx"])
            delta -= float(self.critic @ old["cx"])
            delta = float(np.clip(delta, -1, 1))
            self.critic += 0.05 * delta * old["cx"]
            if not old["gated"]:
                signed = (old["action"] - old["probability"]) * delta
                if self.kind.startswith("ordinary"):
                    self.linear += 0.1 * signed * old["x"]
                else:
                    self.brain.reinforce(old["kc"], signed)
            self.updates += 1
        self.batch.clear()

    def weights_hash(self):
        arrays = [self.brain.w_buy, self.brain.w_sell, self.linear, self.critic]
        return hashlib.sha256(b"".join(a.tobytes() for a in arrays)).hexdigest()

    def frozen_copy(self):
        fresh = Compact(self.kind, self.seed, self.risk_threshold, self.batch_size)
        for key in ("w_buy", "w_sell"):
            getattr(fresh.brain, key)[:] = getattr(self.brain, key)
        fresh.linear[:] = self.linear
        fresh.critic[:] = self.critic
        return fresh


class FullGraph:
    def __init__(self, learning, data_root):
        from flyvsly.backends.neural import NeuralBackend
        self.learning, self.data_root = learning, data_root
        self.backend = NeuralBackend(settings(learning), data_root=data_root)
        self.feedback = "none"

    def weights_hash(self):
        return self.backend.controller.brain.memory()["sha256"]

    def receive_reward(self, reward_dollars):
        self.feedback = "reward" if reward_dollars > .01 else "aversive" if reward_dollars < -.01 else "none"

    def decide(self, closes, q, training, held):
        from stonkfly.display import market_frame
        out = self.backend.observe(market_frame(q.product, closes[-120:], q.bid, q.ask), self.feedback)
        self.feedback = "none"
        side = out["side"]
        return {"action": 1 if side == "BUY" else 0 if side == "SELL" else int(held),
                "gated": False, "score": float(out.get("difference_hz", 0))}

    def frozen_copy(self):
        fresh = FullGraph(False, self.data_root)
        old, new = self.backend.controller.brain, fresh.backend.controller.brain
        # Copy ONLY the learned efficacies. All dynamic traces start identically fresh.
        new.weight[new.circuit["edges"]] = old.weight[old.circuit["edges"]]
        new.weights_frozen = True
        return fresh


def run_window(bars, start, end, policy, output, training=False, constant=None, fee_bps=8):
    """Decide at completed close t; fill at open t+1; feedback at close t+1.

    Mark-to-bid returns include inherited exposure across gaps and actual fees.
    Terminal inventory is marked, not fictitiously liquidated; it is reported.
    """
    if not 21 <= start < end <= len(bars):
        raise ValueError("Invalid bounds")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    account = Account(output / "ledger.sqlite", fee_bps)
    closes = bars.close.to_numpy()
    times = bars.t.to_numpy(dtype=np.int64)
    pending = None
    previous_nav = 10.0
    records = []
    initial_hash = policy.weights_hash() if policy else None
    try:
        for i in range(start, end):
            # A close decision occurs 1 ms before the next UTC daily bar starts.
            close_time = int(times[i] + 86400) - .001
            open_q = quote(float(bars.open.iloc[i]), int(times[i]))
            close_q = quote(float(closes[i]), close_time)
            if pending is not None:
                account.execute(pending["action"], open_q, pending["decision_time"])
            nav = account.nav(close_q)
            reward = math.log(nav / previous_nav)
            state = None
            if isinstance(policy, Compact):
                state = policy.encode(closes[max(0, i-120):i+1], close_time, account.held())
                if pending is not None:
                    policy.receive(pending, reward, state, close_time, training)
            elif isinstance(policy, FullGraph):
                policy.receive_reward(nav - previous_nav if pending is not None else 0)
            row = {"index": i, "time": close_time, "nav": nav, "reward": reward,
                   "cash": account.cash, "units": account.units}
            if i < end-1:
                if policy is None:
                    pending = {"action": int(constant), "gated": False, "score": 0.0}
                elif isinstance(policy, Compact):
                    pending = policy.decide(state, training)
                else:
                    pending = policy.decide(closes[:i+1], close_q, training, account.held())
                pending["decision_time"] = close_time
                row.update({k: pending[k] for k in ("action", "score", "gated")})
            else:
                if isinstance(policy, Compact) and training:
                    policy.flush(close_time)
                elif isinstance(policy, FullGraph) and training:
                    policy.decide(closes[:i+1], close_q, training, account.held())
                pending = None
            records.append(row)
            previous_nav = nav
        final_hash = policy.weights_hash() if policy else None
        if policy and not training and final_hash != initial_hash:
            raise AssertionError("Evaluation changed memory")
        navs = np.array([10.0] + [row["nav"] for row in records])
        summary = {"net_return_pct": (navs[-1] / 10 - 1) * 100,
                   "max_drawdown_pct": float(np.min(navs / np.maximum.accumulate(navs) - 1) * 100),
                   "fills": len(account.fills), "fees": sum(f["fee"] for f in account.fills),
                   "ending_units": account.units, "veto_count": len(account.vetoes),
                   "gated_decisions": sum(bool(r.get("gated")) for r in records),
                   "initial_weights": initial_hash, "final_weights": final_hash,
                   "first_bar": int(times[start]), "last_bar": int(times[end-1]),
                   "bars": end-start, "training": training,
                   "accounting_parity": True, "same_bar_fills": False}
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        (output / "trace.json").write_text(json.dumps(records) + "\n")
        (output / "fills.json").write_text(json.dumps(account.fills) + "\n")
        return summary
    finally:
        account.close()
