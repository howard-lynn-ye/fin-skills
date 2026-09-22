"""Offline probes of an unchanged, pinned Stonkfly Lab checkout; no market backtest.

Run from any directory with --lab-root and --output. The upstream code is MIT;
this script imports it from the caller-supplied checkout rather than vendoring it.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys


PIN = "09e4529e2e1a135083838abd00ccacfd95c63a29"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lab-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.lab_root.resolve()
    head = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    ).strip()
    if head != PIN or dirty:
        raise SystemExit("Expected the unchanged pinned Stonkfly Lab checkout")
    sys.path.insert(0, str(root))
    from flylab.config import Settings
    from flylab.mb import MushroomBody
    from flylab.odors import OdorEncoder
    from flylab.reinforcement import grade

    settings = Settings(seed=11)
    brain = MushroomBody(settings, "probe", 11)
    odor = OdorEncoder(settings.n_pn).planted("A").vector
    before = brain.step(odor).score
    brain.reinforce(brain.snapshot_eligibility(), 1.0)
    after = brain.step(odor).score
    outcome = grade(
        equity_now=99.992, equity_then=99.992, action="BUY", filled=True,
        fee_paid=0.008, deadband=0, exclude_fees=True,
        proportional=True, order_notional=10,
    )
    result = {
        "upstream": "https://github.com/paappraiser/stonkfly-lab",
        "commit": head,
        "kind": "offline_function_probe_not_market_return",
        "positive_feedback": {"score_before": before, "score_after": after},
        "flat_price_post_fill_reference": {
            "deadband_override": 0,
            "reward": outcome.valence,
            "raw_delta": outcome.raw_delta,
            "limitation": "Default deadband may suppress this small amount; this isolates arithmetic.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
