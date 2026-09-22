"""Summarize completed runs without selecting seeds or treating them as markets."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def summarize(path):
    path = Path(path)
    data = json.loads(path.read_text())
    if not data.get("complete"):
        raise ValueError("A partial run is not a completed experiment")
    windows = data["protocol"]["windows"]
    assert windows["train"]["last_day"] < windows["validation"]["first_day"]
    assert windows["validation"]["last_day"] < windows["test"]["first_day"]
    grouped = {}
    for run in data["runs"]:
        for name in ("validation", "test"):
            row = run[name]
            assert row["initial_weights"] == row["final_weights"]
            assert row["accounting_parity"] and not row["same_bar_fills"]
        if run["arm"] in ("frozen", "full_frozen"):
            assert run["train"]["initial_weights"] == run["train"]["final_weights"]
        grouped.setdefault(run["arm"], []).append(run)
    table = {}
    expected = data["protocol"]["seeds"]
    for arm, runs in grouped.items():
        if expected:
            assert sorted(r["seed"] for r in runs) == sorted(expected)
        values = [r["test"]["net_return_pct"] for r in runs]
        table[arm] = {"n_seeds": len(values), "mean_net_return_pct": statistics.mean(values),
                      "min_net_return_pct": min(values), "max_net_return_pct": max(values),
                      "seed_returns_pct": {str(r["seed"]): r["test"]["net_return_pct"] for r in runs},
                      "mean_fills": statistics.mean(r["test"]["fills"] for r in runs),
                      "mean_fees": statistics.mean(r["test"]["fees"] for r in runs),
                      "mean_gated_decisions": statistics.mean(r["test"]["gated_decisions"] for r in runs)}
    contrasts = {}
    for a, b in (("fly_gated", "fly"), ("ordinary_gated", "ordinary"),
                 ("fly_gated", "ordinary_gated"), ("fly", "ordinary")):
        if a in table and b in table:
            left, right = table[a]["seed_returns_pct"], table[b]["seed_returns_pct"]
            contrasts[a+"_minus_"+b] = statistics.mean(left[s]-right[s] for s in left)
    return {"input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "data_sha256": data["protocol"]["data_sha256"],
            "mode": data["protocol"]["mode"], "windows": windows,
            "test_table": table, "paired_mean_difference_percentage_points": contrasts,
            "benchmarks_test_return_pct": {k: v["net_return_pct"] for k, v in data["benchmarks"].items()
                                           if k.endswith("_test")},
            "integrity_checks": "pass",
            "interpretation": "Seed variation on one public development path, not independent markets or a confidence interval."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = [summarize(path) for path in args.inputs]
    if len(result) > 1:
        assert len({x["data_sha256"] for x in result}) == 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
