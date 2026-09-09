"""examples/ - the three worked examples must run, and must still conclude what they claim.

Each one is executed the way a reader would run it: a fresh subprocess, from a directory that
is not the repo, with any inherited PYTHONIOENCODING stripped so a non-ASCII character would
die here the way it dies on a stock Windows console (scripts/check_scripts.py does the same
for the skill scripts, which it globs from plugins/ and so never sees these).

The assertions are on the CONCLUSIONS, not just the exit code: an example whose numbers have
quietly reversed still exits 0.

Marked `slow` (about 10 s for the three subprocesses) so `pytest -q` stays lean; CI runs it
in the `-m slow` job, and `python -m pytest -q -m slow tests/test_examples.py` runs it here.
"""
from __future__ import annotations

import functools
import os
import subprocess
import sys

import pytest

from _helpers import REPO_ROOT

pytestmark = pytest.mark.slow

EXAMPLES_DIR = REPO_ROOT / "examples"
EXAMPLES = sorted(p.name for p in EXAMPLES_DIR.glob("*.py"))
TIMEOUT = 120


@functools.lru_cache(maxsize=None)
def run_example(name: str) -> str:
    """Run examples/<name> in a subprocess and return its stdout. Fails loudly on non-zero.

    Cached: every example is seeded and deterministic, so several tests can read one run.
    """
    env = {k: v for k, v in os.environ.items() if k != "PYTHONIOENCODING"}
    env["PYTHONPATH"] = str(REPO_ROOT)      # the editable install may not be present
    proc = subprocess.run([sys.executable, str(EXAMPLES_DIR / name)],
                          capture_output=True, timeout=TIMEOUT, env=env)
    out = proc.stdout.decode("utf-8", "replace")
    err = proc.stderr.decode("utf-8", "replace")
    assert proc.returncode == 0, f"{name} exited {proc.returncode}\nstderr:\n{err}"
    assert out.isascii(), f"{name} printed non-ASCII, which dies on a stock Windows console"
    assert "TAKEAWAY" in out, f"{name} printed no TAKEAWAY block"
    return out


def test_there_are_three_examples_and_all_of_them_are_indexed():
    assert EXAMPLES == ["audit_a_backtest.py", "futures_roll.py",
                        "point_in_time_fundamentals.py"]
    index = (EXAMPLES_DIR / "README.md").read_text(encoding="utf-8")
    for name in EXAMPLES:
        assert f"({name})" in index, f"examples/README.md does not link {name}"


@pytest.mark.parametrize("name", EXAMPLES)
def test_every_example_runs_and_prints_its_takeaway(name: str):
    out = run_example(name)
    takeaway = out.split("TAKEAWAY", 1)[1]
    assert len(takeaway.strip().splitlines()) >= 3, "the takeaway is one line of nothing"


def test_audit_a_backtest_finds_two_defects_and_then_finds_none():
    out = run_example("audit_a_backtest.py")
    assert "1. COVERAGE" in out and "one slot away" in out       # coverage comes first
    before, after = out.split("4. FIXED", 1)
    assert "2 failed" in before and "FAIL  assert_causal" in before
    assert "FAIL  survivorship_audit" in before
    assert "0 failed" in after and "FAIL" not in after.split("TAKEAWAY")[0]
    assert "assert_causal, survivorship_audit" in out


def test_point_in_time_join_shows_the_leak_and_the_guard_agrees():
    out = run_example("point_in_time_fundamentals.py")
    rows = {line.split("  ")[1].strip(): line for line in out.splitlines()
            if line.strip().startswith(("latest vintage", "filed-date vintage"))}
    assert len(rows) == 2
    wrong = float(out.split("latest vintage, exact match")[1].split()[0])
    right = float(out.split("filed-date vintage, backward as-of")[1].split()[0])
    assert wrong > right + 0.5, f"the wrong join should flatter the Sharpe: {wrong} vs {right}"
    assert "FAIL  safe_asof" in out and "LOOK-AHEAD" in out


def test_futures_roll_finds_exactly_two_exact_operators_and_negative_prices():
    out = run_example("futures_roll.py")
    section = out.split("2. WHICH OPERATOR")[1].split("3. WHAT")[0]
    exact = [ln for ln in section.splitlines() if ln.rstrip().endswith("EXACT")]
    wrong = [ln for ln in section.splitlines() if ln.rstrip().endswith("WRONG")]
    assert len(exact) == 2 and len(wrong) == 2
    assert any("ratio" in ln and "pct_change()" in ln for ln in exact)
    assert any("difference" in ln and "diff()" in ln for ln in exact)
    assert "days are below zero" in out and "SIGN FLIPPED" in out
    assert "refused" in out and "1,884 returns, valid" in out
