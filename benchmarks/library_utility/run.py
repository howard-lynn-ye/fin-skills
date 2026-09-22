"""Frozen, three-arm LLM+library proxy-return feasibility study on existing ECB data.

No generated Python execution, no external tools, no new data downloads. The provider
receives only prepared past-data messages and returns a bounded allocation JSON object.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.library_utility.evaluate import allocation, digest, ledger, paired_blocks

ARMS = ("no_library", "skills_text", "full_library")
SKILLS = ("portfolio-and-risk", "backtest-validation", "trend-following-models")
SYSTEM = """You allocate a research portfolio of anonymous price series in a common numeraire.
Use only the supplied historical data. Return exactly one JSON object with keys weights
and reason, no code. Weights are a list in the given asset order, each in [0,1], total <=1;
the balance remains zero-interest cash. No shorting or leverage. A decision executes at
the NEXT observation and is then held until the next scheduled rebalance. Proportional
fees apply to buys and sells. Seek net growth with controlled drawdown and turnover.
The observations are indicative price proxies, not executable quotes or total returns.
Give a concise reason under 200 characters. Do not predict guaranteed returns."""


def dump(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=True, allow_nan=False)
        stream.write("\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    files = sorted((ROOT / "fin_skills").rglob("*.py"))
    files += sorted((ROOT / "benchmarks/library_utility").glob("*.py"))
    files += [ROOT / "benchmarks/agent_study/transformers_chat.py"]
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in files}


def read_prices():
    source = ROOT / "benchmarks/data/ecb_fx.csv"
    provenance = json.loads(source.with_suffix(".provenance.json").read_text())
    if sha(source) != provenance["fixture_sha256"]:
        raise ValueError("ECB fixture hash differs from recorded provenance")
    raw = pd.read_csv(source, index_col="Date", parse_dates=True)
    if raw.index.has_duplicates or not raw.index.is_monotonic_increasing:
        raise ValueError("unordered or duplicate observations")
    if list(raw.columns) != ["USD", "JPY", "GBP", "CHF"] or not np.isfinite(raw).all().all():
        raise ValueError("unexpected ECB schema")
    # ECB quote = foreign units per EUR. Foreign-currency price in EUR is its reciprocal.
    if (raw <= 0).any().any():
        raise ValueError("nonpositive reference rate")
    prices = 1 / raw
    prices.index = prices.index.tz_localize("UTC") + pd.Timedelta(hours=20)
    prices.columns = ["asset_0", "asset_1", "asset_2", "asset_3"]
    return prices, provenance


def skill_excerpt():
    import fin_skills
    parts = []
    for name in SKILLS:
        text = fin_skills.load(name)
        parts.append({"name": name, "sha256": hashlib.sha256(text.encode()).hexdigest(),
                      "text": text[:1800], "truncated": len(text) > 1800})
    return parts


def context(prices, t, *, history=126):
    """Strict past-only feature builder. Future rows must have no effect."""
    past = prices.iloc[t - history + 1:t + 1]
    if len(past) != history:
        raise ValueError("insufficient decision history")
    returns = past.pct_change(fill_method=None).iloc[1:]
    normalized = past / past.iloc[0]
    return {"asset_order": list(prices.columns),
            "normalized_price_history": np.round(normalized.to_numpy(), 6).tolist(),
            "past_returns_20": (past.iloc[-1] / past.iloc[-21] - 1).to_dict(),
            "past_returns_60": (past.iloc[-1] / past.iloc[-61] - 1).to_dict(),
            "annualized_vol_20": (returns.iloc[-20:].std() * np.sqrt(252)).to_dict(),
            "correlation": returns.corr().round(6).to_numpy().tolist()}, past


def library_evidence(past):
    """Run actual public library APIs. Record inputs/results, not model-written citations."""
    from fin_skills.algorithms import run, recommend_strategy
    returns = past.pct_change(fill_method=None).iloc[1:]
    calls = []
    for algorithm, parameters in (("inverse_volatility", {}),
                                  ("cross_sectional_momentum", {"lookback": 60, "top_k": 2})):
        weights = run(algorithm, {"asset_returns": returns}, **parameters)
        calls.append({"function": "fin_skills.algorithms.run", "algorithm": algorithm,
                      "parameters": parameters, "result": weights.to_dict()})
    signals = {}
    for asset in past:
        report = recommend_strategy(past[asset], as_of=past.index[-1], allow_short=False)
        signals[asset] = {key: report[key] for key in
                          ("regime", "indicators", "selected", "target_exposure", "warnings")}
    calls.append({"function": "fin_skills.algorithms.recommend_strategy", "result": signals})
    receipt = {"past_prices_sha256": digest(past.to_numpy().tolist()),
               "calls": calls, "computed_through": past.index[-1].isoformat()}
    receipt["receipt_sha256"] = digest(receipt)
    return receipt


def parse_action(content, n_assets):
    content = content.strip()
    if content.startswith("```") and content.endswith("```"):
        lines = content.splitlines()
        if lines[0] not in ("```", "```json"):
            raise ValueError("unsupported fenced content")
        content = "\n".join(lines[1:-1])
    obj = json.loads(content)
    if not isinstance(obj, dict) or set(obj) != {"weights", "reason"}:
        raise ValueError("exactly weights and reason required")
    allocation(obj["weights"], n_assets)
    if not isinstance(obj["reason"], str) or len(obj["reason"]) > 1000:
        raise ValueError("reason must be a bounded string")
    return obj


def run_arm(chat, prices, decisions, arm, seed, output, excerpts, *, cost_bps):
    trace, actions = [], []
    previous_target = [0.0] * prices.shape[1]
    for i, t in enumerate(decisions):
        basic, past = context(prices, t)
        data = dict(basic, previous_requested_weights=previous_target, cost_bps=cost_bps,
                    decision_number=i, rebalance_observations=21)
        tools = None
        if arm != "no_library":
            data["library_skill_excerpts"] = excerpts
        if arm == "full_library":
            tools = library_evidence(past)
            # Dates/identities withheld from the model in every arm; receipts retain them.
            data["executed_library_tools"] = tools["calls"]
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": json.dumps(data, ensure_ascii=True, allow_nan=False)}]
        chat.seed, chat.calls = seed + i * 1009, 0  # paired RNG seed per decision, across arms
        response = chat(messages)
        record = {"decision_index": int(t), "observed_through": past.index[-1].isoformat(),
                  "messages": messages, "response": response, "tool_receipt": tools}
        try:
            action = parse_action(response["choices"][0]["message"]["content"], prices.shape[1])
            record.update(status="valid", action=action)
            previous_target = action["weights"]
            weights = action["weights"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            record.update(status="invalid_hold", error=str(exc))
            weights = None
        trace.append(record)
        actions.append({"decision_index": int(t), "weights": weights})
        # Preserve raw decisions immediately, before any return scoring.
        dump(output / f"decision-{i:03d}.json", record)
        print(json.dumps({"arm": arm, "seed": seed, "step": i,
                          "status": record["status"]}), flush=True)
    dump(output / "frozen_actions.json", {"actions": actions, "sha256": digest(actions)})
    return actions, {"decisions": len(trace), "valid": sum(r["status"] == "valid" for r in trace),
                     "invalid_hold": sum(r["status"] == "invalid_hold" for r in trace),
                     "prompt_tokens": sum(r["response"]["usage"]["prompt_tokens"] for r in trace),
                     "completion_tokens": sum(r["response"]["usage"]["completion_tokens"] for r in trace)}


def controls(prices, decisions, start, end, output):
    baseline_actions = {"cash": [], "buy_hold_equal_weight": [], "library_rules_only": []}
    baseline_actions["cash"] = [{"decision_index": start, "weights": [0.] * 4}]
    baseline_actions["buy_hold_equal_weight"] = [{"decision_index": start, "weights": [.25] * 4}]
    for t in decisions:
        _, past = context(prices, t)
        report = library_evidence(past)
        signals = report["calls"][-1]["result"]
        baseline_actions["library_rules_only"].append({"decision_index": int(t),
            "weights": [signals[a]["target_exposure"] / 4 for a in past]})
    reports = {}
    for name, actions in baseline_actions.items():
        reports[name] = {str(cost): ledger(prices, actions, start=start, end=end, cost_bps=cost)
                         for cost in (0, 5, 20)}
    dump(output / "controls.json", reports)
    return reports


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--models", type=Path)
    p.add_argument("--model-indices", default="0,1")
    p.add_argument("--seeds", default="11,23")
    p.add_argument("--year", type=int, default=2025)
    p.add_argument("--sessions", type=int, default=84)
    p.add_argument("--controls-only", action="store_true")
    args = p.parse_args()
    if args.sessions < 42:
        p.error("at least 42 observations required")
    if args.output.exists():
        p.error("output must be new; reruns are separate registered attempts")
    args.output.mkdir(parents=True)
    prices, provenance = read_prices()
    candidates = np.flatnonzero(prices.index.year == args.year)
    if len(candidates) < args.sessions:
        p.error("evaluation interval not present")
    start, end = int(candidates[0]) - 1, int(candidates[args.sessions - 1])
    decisions = list(range(start, end - 1, 21))
    excerpts = skill_excerpt()
    seeds = [int(s) for s in args.seeds.split(",")]
    models = [] if args.controls_only else json.loads(args.models.read_text())
    selected_models = [] if args.controls_only else [models[int(i)] for i in args.model_indices.split(",")]
    protocol = {"study": "library_utility_feasibility_v1", "exploratory": True,
        "source_hashes": source_hashes(), "data_provenance": provenance,
        "model_specs": selected_models, "arms": ARMS, "seeds": seeds,
        "start": str(prices.index[start]), "end": str(prices.index[end]),
        "decision_indices": decisions, "generation_max_tokens": 256, "temperature": .1,
        "history_observations": 126, "skills": excerpts, "system_prompt": SYSTEM,
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
        "gpu_job": os.environ.get("SLURM_JOB_ID"),
        "primary_contrast": "full_library minus no_library at 5 bps",
        "execution": "decision t, indicative fill t+1, first price gain t+1 to t+2",
        "invalid_output": "no rebalance, counted and retained; no retry",
        "limitations": ["indicative FX reference prices, not executable quotes",
            "zero interest and carry; not total-return investment results",
            "public retrospective convenience fixture, not private holdout",
            "model pretraining exposure not independently established",
            "no point-in-time historical news archive included",
            "same generation cap, unequal measured prompt/completion token consumption",
            "forced tool augmentation measures pipeline access, not autonomous tool discovery",
            "one short correlated market path; seeds are not independent markets"]}
    dump(args.output / "protocol.json", protocol)
    dump(args.output / "protocol_sha256.json", {"sha256": digest(protocol)})
    control_reports = controls(prices, decisions, start, end, args.output)
    if args.controls_only:
        print(json.dumps({"controls": {k: v["5"]["metrics"] for k, v in control_reports.items()}}))
        return
    from benchmarks.agent_study.transformers_chat import TransformersChat
    completed = []
    for spec in selected_models:
        model_dir = args.output / spec["model"].split("/")[-1]
        model_dir.mkdir()
        chat = TransformersChat(spec["model"], spec["revision"], max_tokens=256)
        trials = [(seed, arm) for seed in seeds for arm in ARMS]
        np.random.default_rng(20260921).shuffle(trials)
        model_results = {}
        for seed, arm in trials:
            path = model_dir / f"{seed}-{arm}"
            path.mkdir()
            started = time.monotonic()
            try:
                actions, usage = run_arm(chat, prices, decisions, arm, seed, path, excerpts, cost_bps=5)
            except Exception as exc:
                dump(path / "failure.json", {"status": "provider_or_harness_error", "error": repr(exc)})
                raise  # poisoned CUDA/provider state must not generate fake independent failures
            scores = {str(cost): ledger(prices, actions, start=start, end=end, cost_bps=cost)
                      for cost in (0, 5, 20)}
            result = {"model": spec, "seed": seed, "arm": arm, "usage": usage,
                      "elapsed_seconds": time.monotonic() - started, "scores": scores,
                      "frozen_actions_sha256": digest(actions)}
            dump(path / "scores.json", result)
            model_results[(seed, arm)] = result
            completed.append({"model": spec["model"], "seed": seed, "arm": arm,
                              "usage": usage, "metrics_5bps": scores["5"]["metrics"]})
        contrasts = []
        for seed in seeds:
            baseline = model_results[(seed, "no_library")]["scores"]["5"]
            for arm in ("skills_text", "full_library"):
                treated = model_results[(seed, arm)]["scores"]["5"]
                contrasts.append({"seed": seed, "arm": arm,
                    "total_return_difference": treated["metrics"]["total_return"] - baseline["metrics"]["total_return"],
                    "paired_dates": paired_blocks(treated["daily_returns"], baseline["daily_returns"])})
        dump(model_dir / "contrasts.json", contrasts)
        del chat
        gc.collect()
        import torch
        torch.cuda.empty_cache()
    dump(args.output / "summary.json", {"status": "complete", "rows": completed,
        "claim_scope": protocol["limitations"], "protocol_sha256": digest(protocol)})
    print(json.dumps({"status": "complete", "trials": len(completed)}), flush=True)


if __name__ == "__main__":
    main()
