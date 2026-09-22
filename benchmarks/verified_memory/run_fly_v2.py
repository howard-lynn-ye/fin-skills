"""Frozen fruit-fly mechanism and market-proxy study, executed on Beacon RADFM."""
import argparse
from datetime import datetime, timezone
import itertools
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks.library_utility.evaluate import digest, ledger, paired_blocks
from benchmarks.verified_memory.fly_v2 import PARAMETERS, VARIANTS, cue_task, run_market
from benchmarks.verified_memory.model import EXPERTS, SCENARIOS, features, prepare, synthetic
from benchmarks.verified_memory.run import dump, manifest, read_ecb, sha
from fin_skills.api import get
from fin_skills.core.trial_ledger import TrialLedger


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", default="11,23,37,53,71")
    parser.add_argument("--market-seeds", default="101,202,303")
    parser.add_argument("--synthetic-length", type=int, default=1600)
    parser.add_argument("--cue-block", type=int, default=800)
    parser.add_argument("--synthetic-only", action="store_true")
    args = parser.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]
    market_seeds = [int(x) for x in args.market_seeds.split(",")]
    if len(seeds) != len(set(seeds)) or len(market_seeds) != len(set(market_seeds)):
        raise ValueError("duplicate seeds")
    args.output.mkdir(parents=True, exist_ok=False)
    datasets = {f"synthetic_{s}": (synthetic(s, args.synthetic_length),
                {"kind": "synthetic", "seed": s}) for s in market_seeds}
    if not args.synthetic_only:
        datasets["ecb_proxy"] = read_ecb()
    cues = [{"phase": "cue", "task": task, "variant": variant, "seed": seed}
            for task, variant, seed in itertools.product(
                ("acquisition", "retention", "reversal"), VARIANTS, seeds)]
    markets = [{"phase": "market", "dataset": name, "variant": variant,
                "seed": seed, "scenario": "clean", "mode": "verified_update"}
               for name, variant, seed in itertools.product(datasets, VARIANTS, seeds)]
    for name, scenario, mode, seed in itertools.product(datasets, SCENARIOS,
                                                       ("unchecked", "verified_update"), seeds):
        if scenario == "clean" and mode == "verified_update":
            continue
        markets.append({"phase": "market", "dataset": name, "variant": "fly_trace",
                        "seed": seed, "scenario": scenario, "mode": mode})
    rng = np.random.default_rng(20260922)
    rng.shuffle(cues)
    rng.shuffle(markets)
    hashes = manifest()
    hashes["tests/test_fly_v2.py"] = sha(ROOT / "tests/test_fly_v2.py")
    protocol = {"created_utc": datetime.now(timezone.utc).isoformat(),
        "study": "fly-circuit-v2", "parameters": PARAMETERS, "source_hashes": hashes,
        "cue_block": args.cue_block, "cue_environment_seed": 991, "cue_delay": 3,
        "initialization_seeds": seeds, "planned_cue_cells": cues,
        "planned_market_cells": markets,
        "datasets": {name: {"provenance": prov, "rows": len(px),
                            "first": str(px.index[0]), "last": str(px.index[-1])}
                     for name, (px, prov) in datasets.items()},
        "runtime": {"python": sys.version, "numpy": np.__version__,
                    "slurm_job_id": os.getenv("SLURM_JOB_ID"), "cwd": str(Path.cwd()),
                    "output": str(args.output.resolve()), "tmpdir": os.getenv("TMPDIR"),
                    "cache": os.getenv("XDG_CACHE_HOME")},
        "limitations": ["exploratory follow-up after v1, no preregistration claim",
            "engineered rate model, not full connectome or exact paper reproduction",
            "cue tasks are authored mechanism checks, not external benchmarks",
            "ECB reference rates are historical development proxies excluding carry and real fills",
            "selected-only standalone expert reward differs from continuing-account reward",
            "initialization repeats are not independent markets; no tuning or selection on these results"]}
    dump(args.output / "protocol.json", protocol)
    ledger_ = TrialLedger(args.output / "trials.jsonl")
    for cell in cues + markets:
        cell["trial_id"] = ledger_.record("fly_v2", cell.copy())
    for name in ("cue", "market", "datasets"):
        (args.output / name).mkdir()
    cue_rows, market_rows, failures, returns = [], [], [], {}
    began = time.monotonic()
    for number, cell in enumerate(cues, 1):
        try:
            result = cue_task(cell["variant"], cell["seed"], cell["task"], block=args.cue_block)
            dump(args.output / "cue" / f"{cell['trial_id']}.json", result)
            row = {**cell, "scores": result["scores"]}
            cue_rows.append(row)
            ledger_.complete(cell["trial_id"], result["scores"])
        except Exception as exc:
            failures.append({**cell, "error": repr(exc)})
            ledger_.abandon(cell["trial_id"], repr(exc))
        if number % 10 == 0 or number == len(cues):
            print(json.dumps({"phase": "cue", "finished": number, "planned": len(cues),
                              "failures": len(failures)}), flush=True)
    prepared, bounds = {}, {}
    for name, (prices, _) in datasets.items():
        start = int(np.searchsorted(prices.index.year, 2016)) if name == "ecb_proxy" else int(len(prices) * 0.6)
        end = len(prices) - 1
        bounds[name] = start, end
        guard = get("assert_causal").run(fn=features, df=prices, k=len(prices) // 2)
        if not guard.passed:
            raise RuntimeError(guard.summary())
        for scenario in SCENARIOS:
            prepared[name, scenario] = prepare(prices, scenario)
        directory = args.output / "datasets" / name
        directory.mkdir()
        prices.to_csv(directory / "prices.csv")
        dump(directory / "checks.json", {"guard": guard.summary(), "start": start, "end": end})
        fixed = {}
        _, schedule, experts, _ = prepared[name, "clean"]
        for i, expert in enumerate(EXPERTS):
            tid = ledger_.record("fixed_expert", {"dataset": name, "expert": expert})
            actions = [{"decision_index": t, "weights": experts[t][i].tolist()}
                       for t in schedule if start <= t < end]
            score = ledger(prices, actions, start=start, end=end, cost_bps=5)
            fixed[expert] = score["metrics"]
            ledger_.complete(tid, score["metrics"])
        dump(directory / "fixed_experts.json", fixed)
    for number, cell in enumerate(markets, 1):
        tid = cell["trial_id"]
        directory = args.output / "market" / tid
        directory.mkdir()
        try:
            name, scenario = cell["dataset"], cell["scenario"]
            start, end = bounds[name]
            actions, updates, audit = run_market(prepared[name, scenario], cell["variant"],
                cell["mode"], cell["seed"], start=start, end=end)
            dump(directory / "actions.json", actions)
            dump(directory / "updates.json", updates)
            scores = {str(cost): ledger(datasets[name][0], actions, start=start, end=end,
                      cost_bps=cost) for cost in (0, 5, 20)}
            row = {**cell, "audit": audit, "actions_sha256": digest(actions),
                   "metrics_by_cost": {cost: value["metrics"] for cost, value in scores.items()}}
            dump(directory / "scores.json", {**row, "scores": scores})
            market_rows.append(row)
            returns[name, scenario, cell["variant"], cell["mode"], cell["seed"]] = scores["5"]["daily_returns"]
            ledger_.complete(tid, row["metrics_by_cost"]["5"])
        except Exception as exc:
            failure = {**cell, "error": repr(exc)}
            dump(directory / "failure.json", failure)
            failures.append(failure)
            ledger_.abandon(tid, repr(exc))
        if number % 20 == 0 or number == len(markets):
            print(json.dumps({"phase": "market", "finished": number, "planned": len(markets),
                              "failures": len(failures)}), flush=True)
    contrasts = []
    for name, other in itertools.product(datasets, VARIANTS[1:]):
        try:
            a = np.mean([returns[name, "clean", "fly_trace", "verified_update", s] for s in seeds], axis=0)
            b = np.mean([returns[name, "clean", other, "verified_update", s] for s in seeds], axis=0)
        except KeyError:
            continue
        contrasts.append({"dataset": name, "a": "fly_trace", "b": other,
                          "paired_dates": paired_blocks(a, b, block=21, draws=1000, seed=20260922)})
    dump(args.output / "contrasts.json", contrasts)
    summary = {"status": "complete" if not failures else "complete_with_failures",
        "protocol_sha256": sha(args.output / "protocol.json"),
        "planned_cue_cells": len(cues), "planned_market_cells": len(markets),
        "cue_rows": cue_rows, "market_rows": market_rows, "failures": failures,
        "trial_ledger": ledger_.summary(), "elapsed_seconds": time.monotonic() - began,
        "limitations": protocol["limitations"]}
    dump(args.output / "summary.json", summary)
    print(json.dumps({"status": summary["status"], "cue_cells": len(cue_rows),
                      "market_cells": len(market_rows), "failures": len(failures),
                      "elapsed_seconds": summary["elapsed_seconds"]}), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
