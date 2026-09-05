#!/usr/bin/env python3
"""Run every skill script and report which ones actually work.

The scripts are the evidence for the claims in the SKILL.md files, so a script
that does not run turns a measured claim back into an assertion. This runs each
one in a subprocess with the platform's DEFAULT console encoding - not UTF-8 -
because that is the only way to catch the failure mode this file exists for:

    a script that prints a non-ASCII character runs fine under
    PYTHONIOENCODING=utf-8 and dies with UnicodeEncodeError on a stock
    Windows console (cp1252).

A static scan for non-ASCII source characters is the wrong check: several
scripts carry Chinese terms in comments and docstrings and run perfectly. Only
what reaches stdout matters, and the honest way to know that is to run it.

Run:  python scripts/check_scripts.py [--timeout N]
Exit: 0 if every script exits 0, 1 otherwise.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    timeout = 300
    if "--timeout" in sys.argv:
        timeout = int(sys.argv[sys.argv.index("--timeout") + 1])

    scripts = sorted(ROOT.glob("plugins/*/skills/*/scripts/*.py"))
    if not scripts:
        print("no scripts found under plugins/*/skills/*/scripts/")
        return 1

    # Strip any inherited encoding override, so the child sees the real console
    # default. Inheriting PYTHONIOENCODING=utf-8 is exactly what hides the bug.
    env = {k: v for k, v in os.environ.items() if k != "PYTHONIOENCODING"}

    failures: list[tuple[str, str]] = []
    slow: list[tuple[str, float]] = []
    for s in scripts:
        rel = s.relative_to(ROOT).as_posix()
        t0 = time.time()
        try:
            r = subprocess.run([sys.executable, str(s)], cwd=s.parent, env=env,
                               capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            failures.append((rel, f"timed out after {timeout}s"))
            continue
        dt = time.time() - t0
        if dt > 30:
            slow.append((rel, dt))
        if r.returncode != 0:
            tail = r.stderr.decode("utf-8", "replace").strip().split("\n")[-1]
            failures.append((rel, tail))

    print(f"{len(scripts) - len(failures)}/{len(scripts)} scripts run clean "
          f"on the default console encoding")
    if slow:
        print(f"\nslow (over 30s) - fine, but keep them from growing:")
        for rel, dt in sorted(slow, key=lambda x: -x[1]):
            print(f"  {dt:6.1f}s  {rel}")
    if failures:
        print(f"\nFAIL - {len(failures)} script(s):")
        for rel, why in failures:
            print(f"  {rel}\n      {why}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
