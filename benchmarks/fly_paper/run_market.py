"""Chronological ECB development backtest of the published-memory trading actor.

SPDX-License-Identifier: GPL-3.0-or-later
Neural equations derive from Luo/Huang 2024. State bins, financial rewards, critic,
exploration and a maximum holding period are engineering additions.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.fly_paper.model import parameter_matrices, reset_valence
from benchmarks.fly_paper.run_control import Learner
from benchmarks.library_utility.evaluate import ledger
from benchmarks.verified_memory.episode_credit import IntervalCredit, NavMark, OptionReceipt
from benchmarks.verified_memory.fly_v3 import Account
from benchmarks.verified_memory.model import WARMUP, HORIZON, features
from benchmarks.verified_memory.run import read_ecb
from fin_skills.api import get

SEEDS = (11, 23, 37, 53, 71)
ARMS = ("fly_actor_critic", "ordinary_mc_value")
MAX_HOLD = 60


def state_for(market_row, held, age):
    # Fixed bins; no thresholds/normalizers fitted on future or evaluation data.
    cues = [int(np.digitize(np.mean(market_row[i:i+4]), [-.2, .2]))
            for i in (4, 8, 12)]  # five-, twenty-, sixty-observation trends
    return (*cues, bool(held), min(age // 20, 3) if held else 0)


def simulate(prices, learner, *, start, end, cost_bps=5, stop_before=None):
    """Single chronological pass, persistent memory and strict delayed feedback.

    Action 0 waits/holds; action 1 buys an equal basket/exits. An actual entry-to-exit
    trajectory supplies suffix returns to its entry and subsequent HOLD decisions.
    No clipping, bootstrap or counterfactual rewards. Forced time exits are costs
    on the preceding decisions, never represented as learned choices. Waiting in
    cash is a completed zero-return option at the next execution opportunity.
    """
    if not WARMUP < start < end < len(prices):
        raise ValueError("invalid chronological bounds")
    market = features(prices).to_numpy()
    # Stable dot-product reduction order across copied/column-major dataframes.
    px = np.ascontiguousarray(prices.to_numpy())
    schedule = set(range(WARMUP, end, HORIZON))
    account = Account.fresh(4, WARMUP)
    pending = None
    active = None
    queue = []
    actions, updates, navs, fees, receipt_rows = [], [], [], [], []
    entry_fill = None
    sequence = 0
    until = end + 1 if stop_before is None else min(end + 1, stop_before)

    def mark(t, minute):
        event = prices.index[t] + pd.Timedelta(minutes=minute)
        return NavMark(event, event + pd.Timedelta(seconds=1), account.nav(px[t]),
                       "ecb_reference_cash_share_ledger")

    def close(t, minute):
        nonlocal active, sequence
        if active is None:
            return
        active["marks"].append(mark(t, minute))
        marks = active["marks"]
        for state, action, origin, begin in active["decisions"]:
            tail = marks[begin:]
            intervals = tuple(IntervalCredit(a, b, math.log(b.nav/a.nav))
                              for a, b in zip(tail, tail[1:]))
            receipt = OptionReceipt(f"market-{sequence}", prices.index[origin],
                                    intervals, tail[-1].available_at)
            sequence += 1
            queue.append((state, action, receipt))
            receipt_rows.append({"id": receipt.option_id, "origin": origin,
                "available": receipt.claimed_available.isoformat(), "closed_index": t,
                "target": math.log(tail[-1].nav/tail[0].nav), "action": action,
                "start_nav": tail[0].nav, "end_nav": tail[-1].nav})
        active = None

    for t in range(WARMUP, until):
        # The pre-evaluation episode was closed at start-1. Memory persists; NAV
        # starts at one, with no inherited units or unfilled training order.
        if t == start:
            account = Account.fresh(4, t)
            pending, entry_fill = None, None
            assert active is None
        fee = 0.
        if pending is not None:
            state, action, origin, forced, target = pending
            assert origin + 1 == t
            was_held = entry_fill is not None
            if active is not None and not was_held:
                close(t, 1)  # cash wait ends before the new action's fees
            if active is None:
                active = {"marks": [], "decisions": []}
            active["marks"].append(mark(t, 2))
            begin = len(active["marks"]) - 1
            fill = account.rebalance(target, px[t], t, cost_bps)
            fee += fill["fee"]
            active["marks"].append(mark(t, 3))
            if not forced:
                active["decisions"].append((state, action, origin, begin))
            if action == 1:
                if was_held:
                    close(t, 4)
                    entry_fill = None
                else:
                    entry_fill = t
            pending = None
        if t in (start-1, end):
            fill = account.rebalance(np.zeros(4), px[t], t, cost_bps)
            fee += fill["fee"]
            close(t, 5)
            entry_fill = None
        if t > start:
            navs.append(account.nav(px[t]))
            fees.append(float(fee))
        if t not in schedule or t >= end or t == start-1:
            continue
        now = prices.index[t]
        waiting = []
        for state, action, receipt in queue:
            if receipt.claimed_available >= now:
                waiting.append((state, action, receipt))
                continue
            learner.learn(state, action, receipt, now)
            updates.append({"receipt": receipt.option_id, "at": now.isoformat(),
                "available": receipt.claimed_available.isoformat(),
                "target": math.fsum(i.claimed_log_return for i in receipt.intervals),
                "action": action})
        queue = waiting
        held = entry_fill is not None
        age = t-entry_fill+1 if held else 0
        state = state_for(market[t], held, age)
        forced = bool(held and age >= MAX_HOLD)
        action = 1 if forced else learner.choose(state, True, 1.)
        target = (np.zeros(4) if held else np.full(4, .25)) if action else None
        pending = state, action, t, forced, target
        if t >= start:
            actions.append({"decision_index": t,
                "weights": None if target is None else target.tolist(),
                "action": ("exit" if held else "enter") if action else ("hold" if held else "wait"),
                "forced_time_exit": forced, "state": list(state)})
    return {"actions": actions, "updates": updates, "receipts": receipt_rows,
        "nav": navs, "fees": fees,
        "audit": {"updates": len(updates),
            "early_updates": sum(u["available"] >= u["at"] for u in updates),
            "negative_targets": sum(u["target"] < 0 for u in updates),
            "duplicate_updates": len(updates)-len({u["receipt"] for u in updates}),
            "forced_exits": sum(a["forced_time_exit"] for a in actions),
            "actual_fees": float(sum(fees)), "unconsumed_end_receipts": len(queue)}}


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parameters", type=Path, required=True)
    p.add_argument("--bridge", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if not json.loads(args.bridge.read_text())["all_checks_passed"]:
        raise RuntimeError("interface prerequisite failed")
    args.output.mkdir(parents=True, exist_ok=False)
    prices, provenance = read_ecb()
    start = int(np.searchsorted(prices.index.year, 2016))
    end = len(prices)-1
    files = [*Path(__file__).parent.glob("*.py"),
        ROOT/"benchmarks/verified_memory/episode_credit.py",
        ROOT/"benchmarks/verified_memory/fly_v3.py",
        ROOT/"benchmarks/verified_memory/model.py",
        ROOT/"benchmarks/verified_memory/run.py",
        ROOT/"benchmarks/library_utility/evaluate.py"]
    protocol = {"created_utc": datetime.now(timezone.utc).isoformat(),
        "study": "published_fly_memory_real_price_development_v1", "seeds": SEEDS, "arms": ARMS,
        "cost_bps": 5, "cost_replays_bps": [0, 5, 20], "max_hold_observations": MAX_HOLD,
        "decision_every_observations": HORIZON, "warmup_index": WARMUP,
        "start": start, "end": end, "evaluation_start": str(prices.index[start]),
        "evaluation_end": str(prices.index[end]), "provenance": provenance,
        "exploration_probability": .2, "critic_or_value_rate": .1, "reward_scale": .01,
        "training": "one chronological online pass; pre-2016 warmup, then online learning continues",
        "reward": "unclipped complete entry-to-exit log-NAV suffix credit, gamma=1",
        "actions": "wait/hold versus equal-basket entry/full exit; next-observation fill",
        "state": "fixed [-.2,.2] bins of mean normalized 5/20/60 trends, holding, age//20 capped at 3",
        "return_label": "EUR-denominated indicative FX price-only proxy, not executable total return",
        "limitations": ["already inspected development data, not unseen holdout",
            "no FX carry, funding, bid/ask quotes or real fills", "single historical path",
            "biological assay clock does not equal market elapsed time",
            "v3 had eight actions and different features: historical context, not controlled ablation",
            "control task actor transferred with fixed state bins and 60-observation exit rule",
            "cost sweeps replay fixed primary-cost actions without retraining"],
        "parameter_sha256": hashlib.sha256(args.parameters.read_bytes()).hexdigest(),
        "source_hashes": {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
        "runtime": {"python": sys.version, "numpy": np.__version__, "pandas": pd.__version__,
            "slurm_job_id": os.getenv("SLURM_JOB_ID"), "cwd": str(Path.cwd()),
            "output": str(args.output.resolve()), "tmpdir": os.getenv("TMPDIR"),
            "cache": os.getenv("XDG_CACHE_HOME")}}
    write(args.output/"protocol.json", protocol)
    guard = get("assert_causal").run(fn=features, df=prices, k=start)
    if not guard.passed:
        raise RuntimeError(guard.summary())
    write(args.output/"feature_guard.json", {"passed": guard.passed, "summary": guard.summary()})
    d = loadmat(args.parameters)
    params = reset_valence(parameter_matrices(d["para_mu"].T, d["mat_lu_cell"]), 0.)
    first_decision = next(t for t in range(WARMUP, end, HORIZON) if t >= start)
    baselines = {}
    for label, decisions in (("cash", []), ("buy_hold_equal", [
            {"decision_index": first_decision, "weights": [.25]*4}])):
        baselines[label] = {str(c): ledger(prices, decisions, start=start, end=end, cost_bps=c)
                            for c in (0, 5, 20)}
    write(args.output/"baselines.json", baselines)
    rows = []
    for arm in ARMS:
        for seed in SEEDS:
            learner = Learner(arm, params, seed)
            trajectory = simulate(prices, learner, start=start, end=end)
            label = f"{arm}-{seed}"
            write(args.output/f"{label}-trajectory.json", trajectory)
            scores = {str(c): ledger(prices, trajectory["actions"], start=start, end=end, cost_bps=c)
                      for c in (0, 5, 20)}
            gap = float(np.max(np.abs(np.asarray(trajectory["nav"])-scores["5"]["nav"])))
            if gap > 1e-10 or trajectory["audit"]["early_updates"] or trajectory["audit"]["duplicate_updates"]:
                raise AssertionError(f"invalid result: {label} {gap}")
            write(args.output/f"{label}-scores.json", scores)
            row = {"arm": arm, "seed": seed, "metrics_by_cost":
                {c: score["metrics"] for c, score in scores.items()}, "max_nav_gap": gap,
                "audit": trajectory["audit"]}
            rows.append(row)
            print(json.dumps(row), flush=True)
    summary = []
    for arm in ARMS:
        returns = [r["metrics_by_cost"]["5"]["total_return"] for r in rows if r["arm"] == arm]
        summary.append({"arm": arm, "mean_total_return": float(np.mean(returns)),
                        "min_total_return": min(returns), "max_total_return": max(returns)})
    write(args.output/"results.json", {"status": "complete", "rows": rows, "summary": summary,
        "baselines": {k: v["5"]["metrics"] for k, v in baselines.items()}})
    print(json.dumps({"status": "complete", "summary": summary}), flush=True)


if __name__ == "__main__":
    main()
