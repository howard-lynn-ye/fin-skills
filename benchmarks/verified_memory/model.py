"""Causal expert selection with optional verified feedback; no LLM or broker calls.

This is a full-information online value regressor, not a biological brain emulation
or a policy-gradient algorithm. All experts' standalone payoffs become labels.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from fin_skills.api import get
from fin_skills.synthesis import Fact, Timeline, price_fact

EXPERTS = ("equal_weight", "momentum", "reversal", "inverse_vol", "cash")
MODES = ("unchecked", "output_audit", "verified_update")
MODELS = ("linear", "dense", "fly_sparse")
SCENARIOS = ("clean", "early_feedback", "wrong_payoff")
WARMUP = 64
HORIZON = 5


def synthetic(seed, n=1600):
    """Recurring momentum/reversal/null regimes; entirely synthetic development data."""
    rng = np.random.default_rng(seed)
    r = np.zeros((n, 4))
    for t in range(1, n):
        regime = (t // 200) % 4
        phi = (0.3, -0.3, 0.0, 0.3)[regime]
        sigma = (0.006, 0.009, 0.012, 0.006)[regime]
        r[t] = phi * r[t - 1] + rng.normal(0, sigma, 4)
    return pd.DataFrame(100 * np.exp(np.cumsum(r, axis=0)),
                        index=pd.bdate_range("2000-01-03", periods=n, tz="UTC"),
                        columns=[f"asset_{j}" for j in range(4)])


def features(prices):
    """Fixed transforms of past/current observations; no global fit or normalization."""
    r = prices.pct_change(fill_method=None)
    vol = r.rolling(20).std().clip(lower=1e-4)
    pieces = [np.tanh(prices.pct_change(h, fill_method=None) / (vol * np.sqrt(h)))
              for h in (1, 5, 20, 60)]
    pieces.append(np.tanh(vol / 0.01 - 1))
    return pd.concat(pieces, axis=1).set_axis(range(20), axis=1).fillna(0)


def expert_weights(prices, t):
    past = prices.iloc[:t + 1]
    mom = (past.iloc[-1] / past.iloc[-21] - 1).to_numpy()
    vol = past.pct_change(fill_method=None).iloc[-20:].std().to_numpy()
    w = np.zeros((len(EXPERTS), prices.shape[1]))
    w[0] = 1 / prices.shape[1]
    if mom.max() > 0:
        w[1, mom.argmax()] = 1
    if mom.min() < 0:
        w[2, mom.argmin()] = 1
    iv = 1 / np.maximum(vol, 1e-4)
    w[3] = iv / iv.sum()
    return w


def payoff(entry, exit_, weights, cost_bps=5):
    """Standalone buy/hold/liquidate NAV from unit cash, including both fee legs."""
    c = cost_bps / 10000
    after_entry = 1 / (1 + c * weights.sum(axis=1))
    risky_end = (weights * (exit_ / entry)).sum(axis=1)
    return after_entry * (1 - weights.sum(axis=1) + risky_end * (1 - c)) - 1


@dataclass
class Packet:
    origin: int
    due: int
    required: int
    values: np.ndarray
    audit: dict


def audit_packet(prices, origin, due, required, weights, values, cost_bps=5):
    """Library checks lineage clocks; local adapter independently reconciles payoff.

    Never calls the final strategy scorer. Root input timestamps are trusted simulator
    evidence, not independently authenticated vendor data. Known planted defects only.
    """
    entry = prices.iloc[origin + 1].to_numpy()
    exit_ = prices.iloc[required].to_numpy()
    parents = (
        price_fact("entry", "internal:portfolio", entry.tolist(),
                   prices.index[origin + 1], source="frozen_input"),
        price_fact("exit", "internal:portfolio", exit_.tolist(),
                   prices.index[required], source="frozen_input"),
    )
    stamp = prices.index[due]
    fact = Fact("reported_reward", "internal:portfolio", values.tolist(), stamp,
                stamp, stamp, kind="derived", inputs=parents, source="feedback_adapter")
    checked = get("synthesis_integrity").run(timeline=Timeline([*parents, fact]))
    # Separate cash/share accounting implementation from payoff() above.
    expected = []
    fee = cost_bps / 10000
    for target in weights:
        nav = 1 / (1 + fee * sum(target))
        units = nav * target / entry
        cash = nav * (1 - sum(target))
        proceeds = sum(units * exit_)
        expected.append(float(cash + proceeds - fee * proceeds - 1))
    gap = float(np.max(np.abs(values - expected)))
    accounting_ok = gap <= 1e-10
    return {"library_guard": checked.guard, "lineage_passed": checked.passed,
            "findings": [{"severity": f.severity, "message": f.message}
                         for f in checked.findings],
            "accounting_passed": accounting_ok, "max_payoff_gap": gap,
            "passed": checked.passed and accounting_ok}


def prepare(prices, scenario, *, cost_bps=5):
    if scenario not in SCENARIOS:
        raise ValueError(scenario)
    if prices.shape[1] != 4 or not np.isfinite(prices).all().all() or (prices <= 0).any().any():
        raise ValueError("four finite positive price columns required")
    if not prices.index.is_monotonic_increasing or prices.index.has_duplicates:
        raise ValueError("unique ordered timestamps required")
    x = features(prices).to_numpy()
    schedule = list(range(WARMUP, len(prices) - 1, HORIZON))
    weights = {t: expert_weights(prices, t) for t in schedule}
    packets = []
    for j, t in enumerate(schedule):
        required = t + 1 + HORIZON
        if required >= len(prices):
            continue
        values = payoff(prices.iloc[t + 1].to_numpy(), prices.iloc[required].to_numpy(),
                        weights[t], cost_bps)
        due = required
        injected = j % 7 == 0
        if scenario == "early_feedback" and injected:
            due = t  # broken API returns next week's realized label immediately
        if scenario == "wrong_payoff" and injected:
            values = values.copy()
            values[1] += 0.05  # plausible-shaped but incorrect expert return
        receipt = audit_packet(prices, t, due, required, weights[t], values, cost_bps)
        packets.append(Packet(t, due, required, values, receipt))
        if scenario == "early_feedback" and injected:
            # All arms receive the corrected, mature revision. Consume each origin once.
            receipt = audit_packet(prices, t, required, required, weights[t], values, cost_bps)
            packets.append(Packet(t, required, required, values, receipt))
    packets.sort(key=lambda p: (p.due, p.origin))
    return x, schedule, weights, packets


class OnlineValue:
    """Normalized LMS readout; identical projection/readout sizes for dense and sparse."""
    def __init__(self, kind, seed, dimension=20, width=256, active=16, rate=0.1):
        if kind not in MODELS:
            raise ValueError(kind)
        self.kind, self.active, self.rate = kind, active, rate
        rng = np.random.default_rng(seed)
        # A fixed sparse random expansion of positive/negative input channels.
        self.projection = np.zeros((dimension * 2 + 1, width))
        for col in range(width):
            self.projection[rng.choice(dimension * 2 + 1, 6, replace=False), col] = 1
        self.readout = np.zeros((dimension + 1 if kind == "linear" else width, len(EXPERTS)))

    def encode(self, x):
        if self.kind == "linear":
            z = np.r_[x, 1.]
        else:
            inputs = np.r_[np.maximum(x, 0), np.maximum(-x, 0), 1.]
            z = inputs @ self.projection
            if self.kind == "fly_sparse":
                selected = np.argsort(z, kind="stable")[-self.active:]
                sparse = np.zeros_like(z)
                sparse[selected] = z[selected]
                z = sparse
        return z / max(np.linalg.norm(z), 1e-12)

    def update(self, z, target):
        error = np.clip(target, -0.2, 0.2) - z @ self.readout
        self.readout += self.rate * np.outer(z, error)

    def mixture(self, z):
        scores = z @ self.readout
        logits = (scores - scores.max()) / 0.005
        probs = np.exp(logits)
        return probs / probs.sum()


def simulate(prepared, model, mode, seed, *, start, end):
    if mode not in MODES:
        raise ValueError(mode)
    x, schedule, experts, packets = prepared
    learner = OnlineValue(model, seed)
    encoded = {t: learner.encode(x[t]) for t in schedule}
    consumed, cursor, actions, updates = set(), 0, [], []
    for t in schedule:
        if t >= end:
            break
        # Strictly prior availability: even clean same-timestamp rewards wait one tick.
        while cursor < len(packets) and packets[cursor].due < t:
            packet = packets[cursor]
            cursor += 1
            if packet.origin in consumed:
                continue
            permitted = mode != "verified_update" or packet.audit["passed"]
            updates.append({"origin": packet.origin, "at": t, "due": packet.due,
                            "required": packet.required, "applied": permitted,
                            "audit_passed": packet.audit["passed"],
                            "negative_targets": int((packet.values < 0).sum())})
            if permitted:
                learner.update(encoded[packet.origin], packet.values)
                consumed.add(packet.origin)
        if t >= start:
            mixture = learner.mixture(encoded[t])
            target = mixture @ experts[t]
            actions.append({"decision_index": t, "weights": target.tolist(),
                            "expert_probabilities": mixture.tolist()})
    applied = [u for u in updates if u["applied"]]
    # Reporting audit never retroactively changes actions or drops bad runs from scores.
    audit_ok = all(u["audit_passed"] for u in applied)
    return actions, updates, {"updates": len(applied),
        "invalid_updates": sum(not u["audit_passed"] for u in applied),
        "early_updates": sum(u["required"] >= u["at"] for u in applied),
        "blocked_packets": sum(not u["applied"] for u in updates),
        "negative_targets_learned": sum(u["negative_targets"] for u in applied),
        "final_audit_passed": audit_ok if mode != "unchecked" else None,
        "parameter_count": int(learner.readout.size),
        "representation": model}
