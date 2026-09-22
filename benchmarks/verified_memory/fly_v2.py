"""Rate-based mushroom-body-inspired circuit with delayed chosen-action feedback.

This is an engineered approximation, not a connectome, spiking brain, or numerical
reproduction of a published circuit. Biological motifs and engineering choices are
separated in FLY_V2.md. Only the action actually chosen is reinforced.
"""
from dataclasses import dataclass

import numpy as np

VARIANTS = ("fly_trace", "dense_trace", "fly_no_trace", "fly_frozen",
            "fly_absolute", "linear_trace")
PARAMETERS = {"width": 800, "active": 40, "fan_in": 6, "rate": 0.2,
              "epsilon": 0.1, "trace_tau": 20.0, "background_recovery": 0.0005,
              "market_reward_scale": 0.01}


@dataclass
class Eligibility:
    """The active KCs, selected compartment and prediction at the original decision."""
    origin: int
    z: np.ndarray
    action: int
    prediction: float


class MushroomBody:
    def __init__(self, variant, seed, *, dimension=20, actions=5, rate=None):
        if variant not in VARIANTS:
            raise ValueError(variant)
        self.variant = variant
        self.rate = PARAMETERS["rate"] if rate is None else rate
        self.actions = actions
        self.rng = np.random.default_rng(seed + 1777)
        rng = np.random.default_rng(seed)
        width = PARAMETERS["width"]
        self.inputs = np.stack([rng.choice(dimension * 2, PARAMETERS["fan_in"],
                                          replace=False) for _ in range(width)])
        self.size = dimension + 1 if variant == "linear_trace" else width
        self.approach = np.full((self.size, actions), 0.5)
        self.avoid = np.full((self.size, actions), 0.5)
        self.linear = np.zeros((self.size, actions))
        self.update_count = 0

    def encode(self, x):
        x = np.asarray(x, dtype=float)
        if not np.isfinite(x).all():
            raise ValueError("nonfinite observation")
        if self.variant == "linear_trace":
            z = np.r_[x, 1.]
        else:
            pn = np.r_[np.maximum(x, 0), np.maximum(-x, 0)]
            z = pn[self.inputs].sum(axis=1)
            if self.variant != "dense_trace":
                selected = np.argsort(z, kind="stable")[-PARAMETERS["active"]:]
                code = np.zeros_like(z)
                code[selected] = z[selected]
                z = code
        return z / max(float(np.linalg.norm(z)), 1e-12)

    def values(self, z):
        weights = self.linear if self.variant == "linear_trace" else self.approach - self.avoid
        return z @ weights

    def choose(self, z, t):
        q = self.values(z)
        best = np.isclose(q, q.max(), rtol=0, atol=1e-12)
        probabilities = np.full(self.actions, PARAMETERS["epsilon"] / self.actions)
        probabilities[best] += (1 - PARAMETERS["epsilon"]) / best.sum()
        action = int(self.rng.choice(self.actions, p=probabilities))
        return Eligibility(t, z.copy(), action, float(q[action])), probabilities

    def reinforce(self, memory, reward, now, *, current_z=None):
        if now <= memory.origin:
            raise ValueError("feedback must follow its originating decision")
        if not np.isfinite(reward):
            raise ValueError("nonfinite feedback")
        target = float(np.clip(reward, -1, 1))
        delta = target if self.variant == "fly_absolute" else target - memory.prediction
        if self.variant == "fly_frozen":
            return {"delta": delta, "changed": 0, "trace_origin": memory.origin}
        if self.variant == "fly_no_trace":
            if current_z is None:
                raise ValueError("current observation required by no-trace ablation")
            z = current_z
        else:
            z = memory.z
        trace = z * np.exp(-(now - memory.origin) / PARAMETERS["trace_tau"])
        a = memory.action
        if self.variant == "linear_trace":
            self.linear *= 1 - PARAMETERS["background_recovery"]
            self.linear[:, a] += self.rate * delta * trace
        else:
            # Slow background recovery is explicit and separate from local plasticity.
            recovery = PARAMETERS["background_recovery"]
            self.approach += recovery * (0.5 - self.approach)
            self.avoid += recovery * (0.5 - self.avoid)
            # Positive RPE depresses avoidance; negative RPE depresses approach.
            pool = self.avoid if delta >= 0 else self.approach
            pool[:, a] = np.clip(pool[:, a] - self.rate * abs(delta) * trace, 0, 1)
        self.update_count += 1
        return {"delta": float(delta), "changed": int(np.count_nonzero(trace)),
                "trace_origin": memory.origin}

    def diagnostics(self):
        weights = self.linear if self.variant == "linear_trace" else self.approach - self.avoid
        return {"updates": self.update_count, "weight_l2": float(np.linalg.norm(weights)),
                "readout_parameters": int(self.linear.size if self.variant == "linear_trace"
                                          else self.approach.size + self.avoid.size),
                "saturation_fraction": float(np.mean((self.approach == 0) | (self.avoid == 0)))
                    if self.variant != "linear_trace" else None}


def run_market(prepared, variant, mode, seed, *, start, end):
    if mode not in ("unchecked", "verified_update"):
        raise ValueError(mode)
    x, schedule, experts, packets = prepared
    brain = MushroomBody(variant, seed)
    encoded = {t: brain.encode(x[t]) for t in schedule}
    memories, consumed, cursor = {}, set(), 0
    actions, updates = [], []
    for t in schedule:
        if t >= end:
            break
        while cursor < len(packets) and packets[cursor].due < t:
            packet = packets[cursor]
            cursor += 1
            if packet.origin in consumed:
                continue
            memory = memories[packet.origin]
            permitted = mode == "unchecked" or packet.audit["passed"]
            record = {"origin": packet.origin, "at": t, "required": packet.required,
                "selected_expert": memory.action, "applied": permitted,
                "audit_passed": packet.audit["passed"]}
            if permitted:
                # The learner gets one scalar; unchosen expert payoffs are never passed in.
                reward = float(packet.values[memory.action])
                record.update(reward=reward, **brain.reinforce(memory,
                    reward / PARAMETERS["market_reward_scale"], t, current_z=encoded[t]))
                consumed.add(packet.origin)
            updates.append(record)
        memory, probabilities = brain.choose(encoded[t], t)
        memories[t] = memory
        if t >= start:
            actions.append({"decision_index": t,
                "weights": experts[t][memory.action].tolist(),
                "selected_expert": memory.action, "prediction": memory.prediction,
                "choice_probabilities": probabilities.tolist()})
    applied = [u for u in updates if u["applied"]]
    audit = {"received_labels": len(applied), "invalid_updates": sum(not u["audit_passed"] for u in applied),
        "early_updates": sum(u["required"] >= u["at"] for u in applied),
        "negative_labels": sum(u["reward"] < 0 for u in applied),
        "blocked_packets": sum(not u["applied"] for u in updates), **brain.diagnostics()}
    return actions, updates, audit


def cue_task(variant, seed, task, *, block=800, delay=3):
    """Known-answer mechanism tasks, not a replication of published benchmark scores."""
    if task not in ("acquisition", "retention", "reversal"):
        raise ValueError(task)
    environment = np.random.default_rng(991)
    # Cues are fixed, discriminable but overlapping; every learner sees identical cues.
    prototypes = environment.uniform(-1, 1, (8, 20))
    brain = MushroomBody(variant, seed, actions=4)
    codes = [brain.encode(x) for x in prototypes]
    queue, rows, probes = [], [], []
    for t in range(2 * block):
        stage = t // block
        pool = np.arange(stage * 4, stage * 4 + 4) if task == "retention" else np.arange(8)
        cue = int(environment.choice(pool))
        z = codes[cue]
        pending = []
        for due, memory, reward in queue:
            if due < t:
                brain.reinforce(memory, reward, t, current_z=z)
            else:
                pending.append((due, memory, reward))
        queue = pending
        target = (cue % 4 + int(task == "reversal" and stage == 1)) % 4
        memory, _ = brain.choose(z, t)
        correct = memory.action == target
        queue.append((t + delay, memory, 1.0 if correct else -0.3))
        rows.append({"t": t, "cue": cue, "target": target,
                     "action": memory.action, "correct": correct})
        if (t + 1) % 100 == 0 or t + 1 == block:
            # Evaluation probes neither learn nor advance the policy RNG.
            old, current = [], []
            for i, code in enumerate(codes):
                values = brain.values(code)
                ties = np.isclose(values, values.max(), rtol=0, atol=1e-12)
                initial = i % 4
                target_now = (initial + int(task == "reversal" and stage == 1)) % 4
                old.append(float(ties[initial] / ties.sum()))
                current.append(float(ties[target_now] / ties.sum()))
            probes.append({"after": t + 1, "initial_mapping": old, "current_mapping": current})
    scores = {"last_200_choice_accuracy": float(np.mean([r["correct"] for r in rows[-200:]])),
        "first_100_after_switch_accuracy": float(np.mean([r["correct"] for r in rows[block:block + 100]])),
        "final_probe_accuracy": float(np.mean(probes[-1]["current_mapping"])),
        "first_group_retention": float(np.mean(probes[-1]["initial_mapping"][:4])),
        **brain.diagnostics()}
    return {"variant": variant, "seed": seed, "task": task,
            "scores": scores, "probes": probes, "decisions": rows}
