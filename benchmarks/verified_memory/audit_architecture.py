"""Minimal counterexamples for v2/v3 objectives; not a new return benchmark.

The production circuit is imported unchanged. Stress tests distinguish an exposed
failure mechanism from evidence that it caused the historical-market result.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.verified_memory.fly_v2 import Eligibility, MushroomBody, PARAMETERS


def update(brain, z, reward, t, action=0):
    memory = Eligibility(t, z.copy(), action, float(brain.values(z)[action]))
    return brain.reinforce(memory, reward, t + 10)


def delayed_credit(seed):
    # A two-stage authored control problem. Entry costs 0.2%, and only entering
    # unlocks a deterministic 0.8% later gain. Staying in cash earns zero.
    # At stage 2 HOLD is mandatory, so all exploration is forced/balanced at entry.
    immediate = MushroomBody("fly_trace", seed, dimension=20, actions=2)
    episodic = MushroomBody("fly_trace", seed, dimension=20, actions=2)
    z = immediate.encode(np.linspace(-1, 1, 20))
    entry, later = -0.002, 0.008
    total = (1 + entry) * (1 + later) - 1
    for trial in range(400):
        action = trial % 2  # 0=enter, 1=cash; avoids confounding with exploration.
        update(immediate, z, entry / 0.01 if action == 0 else 0., trial * 20, action)
        # Diagnostic full-episode credit comparator, NOT a proposed fly rule.
        update(episodic, z, total / 0.01 if action == 0 else 0., trial * 20, action)
    q_now, q_episode = immediate.values(z), episodic.values(z)
    assert q_now[0] < q_now[1] and q_episode[0] > q_episode[1]
    return {"seed": seed, "entry_return": entry, "later_return": later,
            "episode_return": total, "immediate_objective_q": q_now.tolist(),
            "episode_objective_q": q_episode.tolist(),
            "interpretation": "Same circuit ranks entry oppositely when credited with the full episode. "
                              "v3 has no continuation-value or episode-credit term. "
                              "This is an objective counterexample, not a market-performance attribution."}


def clipping_counterexample():
    # Enumerated ten-outcome distribution; no sampling uncertainty or market fit.
    returns = np.array([0.04] * 4 + [-0.01] * 6)
    scaled = returns / PARAMETERS["market_reward_scale"]
    clipped = np.clip(scaled, -1, 1)
    assert returns.mean() > 0 and clipped.mean() < 0
    return {"outcomes": returns.tolist(), "mean_raw_return": float(returns.mean()),
            "mean_scaled_target": float(scaled.mean()),
            "mean_clipped_target": float(clipped.mean()), "cash_target": 0.,
            "interpretation": "Clipping changes the optimized utility and reverses this ranking; "
                              "it is not merely numerical rescaling."}


def reversal_stress(seed):
    brain = MushroomBody("fly_trace", seed)
    z = brain.encode(np.linspace(-1, 1, 20))
    blocks = []
    for block in range(24):
        target = 1. if block % 2 == 0 else -1.
        for j in range(200):
            update(brain, z, target, (block * 200 + j) * 20)
        active = z > 0
        blocks.append({"block": block, "target": target,
            "prediction": float(brain.values(z)[0]),
            "active_approach_mean": float(brain.approach[active, 0].mean()),
            "active_avoid_mean": float(brain.avoid[active, 0].mean())})
    return {"seed": seed, "blocks": blocks,
            "first_two_absolute_error": float(np.mean([abs(b["target"] - b["prediction"])
                                                       for b in blocks[:2]])),
            "last_two_absolute_error": float(np.mean([abs(b["target"] - b["prediction"])
                                                      for b in blocks[-2:]])),
            "interpretation": "Repeated same-cue reversals stress both depression pools. "
                              "Does not establish depletion on actual ECB trajectories."}


def action_identity():
    # At a unit-cash account with equal expert weights, equal and inverse-vol
    # propose identical portfolios, while HOLD, cash, and halve all do nothing.
    from benchmarks.verified_memory.fly_v3 import Account, targets_for
    experts = np.array([[.25] * 4, [1., 0., 0., 0.], [0., 1., 0., 0.],
                        [.25] * 4, [0.] * 4])
    targets = targets_for(Account.fresh(4, 64), np.ones(4), experts)
    effective = [np.zeros(4) if w is None else w for w in targets]
    groups = {}
    for i, w in enumerate(effective):
        groups.setdefault(tuple(w), []).append(i)
    return {"identical_effect_groups_at_cash": list(groups.values()),
            "interpretation": "Separate action columns learn separately even for identical effects. "
                              "Not a bookkeeping error, but a sample-efficiency limitation."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    files = [Path(__file__), ROOT / "benchmarks/verified_memory/fly_v2.py",
             ROOT / "benchmarks/verified_memory/fly_v3.py"]
    protocol = {"created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "architecture diagnosis, no economic tuning or model modification",
        "seeds": [11, 23, 37, 53, 71], "parameters": PARAMETERS,
        "source_hashes": {f.relative_to(ROOT).as_posix(): hashlib.sha256(f.read_bytes()).hexdigest()
                          for f in files},
        "job": os.getenv("SLURM_JOB_ID"), "cwd": str(Path.cwd()), "output": str(args.output)}
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    result = {"delayed_credit": [delayed_credit(s) for s in protocol["seeds"]],
              "clipping": clipping_counterexample(),
              "reversal_stress": [reversal_stress(s) for s in protocol["seeds"]],
              "action_identity": action_identity()}
    (args.output / "diagnostics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"status": "complete", "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
