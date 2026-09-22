"""Frozen daily BTC-USDC development comparison using two pinned upstream projects."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
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
from benchmarks.fly_reuse.core import activate, Compact, FullGraph, run_window

PINS = {"fruit-fly-fund": "56f01f6426a4d6d6293a4e06afcf4a035133a968",
        "stonkfly-lab": "09e4529e2e1a135083838abd00ccacfd95c63a29",
        "vendored_stonkfly": "78ef3e05ab0fa086032098558d893667068944a0"}
ARMS = ("fly", "fly_gated", "frozen", "shuffled", "ordinary", "ordinary_gated")
SEEDS = (11, 23, 37, 53, 71)


def snapshot(path):
    """Use upstream's real public-data adapter once; preserve its raw normalized rows."""
    from flyvsly.market import kraken_candles
    path = Path(path)
    if path.exists():
        payload = json.loads(path.read_text())
    else:
        rows = kraken_candles("BTC-USDC", 86400)
        now = time.time()
        rows = sorted([r for r in rows if int(r[0]) + 86400 <= now])
        payload = {"retrieved_at": datetime.now(timezone.utc).isoformat(),
                   "source": "https://api.kraken.com/0/public/OHLC?pair=XBTUSDC&interval=1440",
                   "product": "BTC-USDC", "seconds_per_bar": 86400,
                   "rows": rows, "status": "public_development_data_not_unseen_holdout"}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=1) + "\n")
    frame = pd.DataFrame(payload["rows"], columns=["t", "low", "high", "open", "close", "volume"])
    if len(frame) < 180 or frame.t.duplicated().any() or not frame.t.is_monotonic_increasing:
        raise ValueError("Insufficient or invalid market snapshot")
    if not np.isfinite(frame.to_numpy()).all() or (frame[["open", "close", "low"]] <= 0).any().any():
        raise ValueError("Invalid prices")
    if (np.diff(frame.t) != 86400).any():
        raise ValueError("Daily gaps require an explicit missing-data policy")
    return frame, hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("compact", "full"), default="compact")
    parser.add_argument("--connectome", type=Path)
    parser.add_argument("--pilot-bars", type=int, default=180)
    args = parser.parse_args()
    activate(args.upstream)
    if args.output.exists():
        raise SystemExit("Refusing to overwrite a run")
    args.output.mkdir(parents=True)
    bars, data_hash = snapshot(args.data)
    if args.mode == "full":
        if args.connectome is None:
            raise SystemExit("Full graph requires --connectome")
        bars = bars.iloc[-(args.pilot_bars+21):].reset_index(drop=True)
    n = len(bars)
    train_end, val_end = 21 + int((n-21)*.6), 21 + int((n-21)*.8)
    windows = {"train": (21, train_end), "validation": (train_end, val_end), "test": (val_end, n)}
    # Fit only on train. This is an engineered risk-budget proxy, not biological hunger.
    vol = np.log(bars.close).diff().rolling(20).std(ddof=0)
    threshold = float(vol.iloc[21:train_end].quantile(.8))
    protocol = {"pins": PINS, "mode": args.mode, "data_sha256": data_hash,
                "rows": n, "windows": {k: {"start": a, "end_exclusive": b,
                    "first_day": int(bars.t.iloc[a]), "last_day": int(bars.t.iloc[b-1])}
                    for k, (a, b) in windows.items()},
                "commission_bps_per_fill": 8, "half_spread_bps": 2.5,
                "costs_status": "fixed modeling assumptions, not a venue fee-tier claim",
                "execution": "completed close decision; next open fill; next completed close feedback",
                "initial_capital": 10, "max_order_notional": 10,
                "terminal_inventory": "marked to bid, not liquidated; reported",
                "risk_gate": {"source": "training-only 80th percentile daily 20-bar realized volatility",
                              "threshold": threshold},
                "batch_size": 16, "gamma": .95, "actor_eta": .01,
                "seeds": list(SEEDS) if args.mode == "compact" else [],
                "selection": "No hyperparameter or model selection on validation/test in this run",
                "status": "development experiment; no claim of unseen confirmatory evidence",
                "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                            "pandas": pd.__version__, "hostname": platform.node()},
                "source_hashes": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in Path(__file__).parent.glob("*.py")}}
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    result = {"protocol": protocol, "runs": [], "benchmarks": {}}
    def checkpoint():
        (args.output / "results.partial.json").write_text(json.dumps(result, indent=2) + "\n")
    for name, action in (("cash", 0), ("buy_hold_same_guard", 1)):
        for window in ("validation", "test"):
            a, b = windows[window]
            result["benchmarks"][name+"_"+window] = run_window(
                bars, a, b, None, args.output/name/window, constant=action)
    jobs = [(arm, seed) for seed in SEEDS for arm in ARMS] if args.mode == "compact" else [
        ("full_plastic", 0), ("full_frozen", 0)]
    for arm, seed in jobs:
        policy = Compact(arm, seed, threshold) if args.mode == "compact" else FullGraph(
            arm == "full_plastic", str(args.connectome))
        stem = args.output / f"{arm}-{seed}"
        a, b = windows["train"]
        row = {"arm": arm, "seed": seed, "train": run_window(
            bars, a, b, policy, stem/"train", training=True)}
        if isinstance(policy, Compact):
            (stem / "shuffle.json").write_text(json.dumps(policy.shuffle_records))
        for window in ("validation", "test"):
            a, b = windows[window]
            frozen = policy.frozen_copy()
            row[window] = run_window(bars, a, b, frozen, stem/window)
            del frozen
        result["runs"].append(row)
        checkpoint()
        print(json.dumps({"arm": arm, "seed": seed, "test": row["test"]["net_return_pct"]}), flush=True)
        del policy
    result["test_mean_return_pct"] = {
        arm: float(np.mean([r["test"]["net_return_pct"] for r in result["runs"] if r["arm"] == arm]))
        for arm in dict.fromkeys(r["arm"] for r in result["runs"])}
    result["complete"] = True
    (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["test_mean_return_pct"]), flush=True)


if __name__ == "__main__":
    main()
