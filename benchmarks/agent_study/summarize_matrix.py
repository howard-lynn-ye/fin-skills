"""Summarize all planned cells, retaining missing runs and ungradable accepted outputs."""
import argparse
import json
import math
from pathlib import Path
from collections import defaultdict


def summarize(roots):
    groups = defaultdict(list)
    for root in roots:
        protocol = json.loads((root / "protocol.json").read_text())
        for cell in protocol["cells"]:
            path = root / f"s{cell['market_seed']}-r{cell['repetition']}-{cell['condition']}"
            record = {**cell, "model": protocol["model"], "completed": False,
                      "accepted": False, "correct": None, "status": "MISSING"}
            if (path / "result.json").exists():
                run = json.loads((path / "result.json").read_text())
                record.update(completed=True, accepted=bool(run["accepted"]), status=run["termination"],
                              turns=run["turns"], wall_seconds=run["wall_seconds"],
                              tokens=sum(x.get("total_tokens", 0) for x in run["usage"]))
            if (path / "grade.json").exists():
                grade = json.loads((path / "grade.json").read_text())
                keys = ["sharpe_gap", "leakage_rate", "same_session_rate", "post_delisting_mass"]
                if all(isinstance(grade.get(k), (int, float)) and math.isfinite(grade[k]) for k in keys):
                    record["correct"] = (abs(grade["sharpe_gap"]) <= .05
                        and grade["leakage_rate"] == 0 and grade["same_session_rate"] == 0
                        and grade["post_delisting_mass"] <= 1e-12)
                    record["grade"] = {k: grade[k] for k in keys}
            groups[(protocol["model"], cell["condition"])].append(record)
    output = []
    for (model, condition), rows in sorted(groups.items()):
        n, complete = len(rows), sum(x["completed"] for x in rows)
        accepted = sum(x["accepted"] for x in rows)
        wrong = sum(x["accepted"] and x["correct"] is False for x in rows)
        unknown = sum(x["accepted"] and x["correct"] is None for x in rows)
        output.append({"model": model, "condition": condition, "planned": n, "completed": complete,
            "accepted": accepted, "incorrect_accepted": wrong, "ungradable_accepted": unknown,
            "acceptance_rate": accepted / n if complete == n else None,
            "incorrect_accepted_per_attempt": wrong / n if complete == n and unknown == 0 else None,
            "correct_among_accepted": (accepted - wrong) / accepted if complete == n and not unknown and accepted else None,
            "cells": rows})
    return {"groups": output, "scope": "public development cells; descriptive results only",
            "all_planned_cells_completed": bool(output) and all(g["completed"] == g["planned"] for g in output)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("roots", type=Path, nargs="+")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.write_text(json.dumps(summarize(args.roots), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
