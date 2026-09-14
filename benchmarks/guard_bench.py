"""Repeat the existing defect/false-alarm benchmark across independent seeded worlds."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
SEEDS = (17, 29, 43)


def main():
    reports = []
    with tempfile.TemporaryDirectory(prefix="fin-guard-bench-") as directory:
        for seed in SEEDS:
            path = Path(directory) / f"seed-{seed}.md"
            proc = subprocess.run([sys.executable, str(ROOT / "leak_bench.py"), "--quick",
                "--seed", str(seed), "--output", str(path)], capture_output=True,
                text=True, errors="replace", timeout=300)
            if proc.returncode:
                raise RuntimeError(proc.stderr[-4000:] + proc.stdout[-1000:])
            reports.append(json.loads(path.with_suffix(".json").read_text()))
    result = {"worlds": reports, "scope": "seed robustness of one synthetic strategy/defect design",
              "false_alarms": sum(len(r["false_alarms"]) for r in reports),
              "gaps": sum(len(r["gaps"]) for r in reports),
              "errors": sum(len(r["errors"]) for r in reports)}
    (ROOT / "GUARD_ROBUSTNESS.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))
    return int(bool(result["errors"]))


if __name__ == "__main__":
    raise SystemExit(main())
