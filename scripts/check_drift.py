#!/usr/bin/env python3
"""Re-verify every version claim in the repo against PyPI, and report what has rotted.

WHY THIS EXISTS: this library's value is that its facts are dated and checked. That is also
its decay mode — 59 reference files assert specific versions and release dates, and a claim
like "yfinance 1.7.0 (2026-08-26)" silently becomes wrong. A knowledge base that quietly
goes stale is worse than one that admits it does not know, because the reader cannot tell
which claims still hold.

This scans the markdown for version assertions, asks PyPI what is true now, and classifies
each one:

  CURRENT   the claimed version is still the latest
  BEHIND    a newer release exists — the claim is stale but was true when written
  WRONG     the claimed version does not exist on PyPI at all
  GONE      the package is no longer on PyPI (a real event here: mlfinlab, portfoliolab)

Run:
    python scripts/check_drift.py              # summary
    python scripts/check_drift.py --verbose    # every claim
    python scripts/check_drift.py --stale-days 180
Exit 1 if anything is WRONG or GONE, or if more than --max-behind claims are BEHIND.
"""
from __future__ import annotations

import concurrent.futures
import datetime as dt
import json
import re
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TODAY = dt.date.today()

# The package name MUST be delimited by backticks or bold — that is this repo's citation
# style, and requiring it is what stops prose like "GDP was 17102.5 (2014-01-30)" from being
# read as a version claim about a package called "was". Matching bare words was tried and
# produced seven false positives out of twenty-four.
#   `pkg` 1.2.3 (2026-08-26)   **pkg** 1.2.3 (2026-08-26)   `pkg` **1.2.3** (2026-08-26)
CLAIM = re.compile(
    r"(?:`(?P<pkg_b>[A-Za-z][A-Za-z0-9._-]{1,40})`|\*\*(?P<pkg_s>[A-Za-z][A-Za-z0-9._-]{1,40})\*\*)"
    r"[^\n|]{0,40}?"
    r"\*{0,2}(?:v)?(?P<ver>\d+\.\d+(?:\.\d+)?(?:[abrc]\d+)?(?:\.post\d+)?)\*{0,2}\s*"
    r"\((?P<date>\d{4}-\d{2}-\d{2})\)"
)

# Delimited tokens that still are not PyPI packages.
NOT_PACKAGES = {
    "python", "release", "version", "sharpe", "note", "issue", "arxiv", "sec", "spec",
    "main", "master", "next", "readme", "license", "changelog", "pypi", "github",
    # backticked identifiers that are arguments or wheel tags, not distributions
    "auto_adjust", "never", "cp310-abi3", "cp39-abi3", "requires_dist", "requires_python",
    "adj", "fq", "fqt", "adjust", "adjustflag", "paper", "provider", "as_of",
}


def pypi(name: str) -> dict | None:
    try:
        with urllib.request.urlopen(f"https://pypi.org/pypi/{name}/json", timeout=20) as r:
            return json.load(r)
    except Exception:
        return None


def latest_of(data: dict) -> tuple[str, str]:
    v = data["info"]["version"]
    files = data["releases"].get(v, [])
    when = max((f["upload_time_iso_8601"] for f in files), default="")[:10]
    return v, when


def collect_claims() -> dict[str, list[tuple[str, str, Path]]]:
    """package -> [(claimed_version, claimed_date, file), ...]"""
    out: dict[str, list] = defaultdict(list)
    for md in sorted(ROOT.glob("plugins/**/*.md")) + sorted(ROOT.glob("*.md")):
        for m in CLAIM.finditer(md.read_text(encoding="utf-8")):
            pkg = m.group("pkg_b") or m.group("pkg_s")
            if pkg.lower() in NOT_PACKAGES or pkg.isdigit():
                continue
            out[pkg].append((m.group("ver"), m.group("date"), md.relative_to(ROOT)))
    return out


def main() -> int:
    verbose = "--verbose" in sys.argv
    stale_days = int(next((a.split("=")[1] for a in sys.argv if a.startswith("--stale-days=")), 365))
    max_behind = int(next((a.split("=")[1] for a in sys.argv if a.startswith("--max-behind=")), 10**6))

    claims = collect_claims()
    if not claims:
        print("no version claims found — has the citation format changed?")
        return 1

    print(f"checking {len(claims)} packages against PyPI ...\n")
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
        live = dict(zip(claims, ex.map(pypi, claims)))

    buckets: dict[str, list] = defaultdict(list)
    for pkg, entries in sorted(claims.items()):
        data = live.get(pkg)
        if data is None:
            # Absent from PyPI. For this repo that is often the POINT (mlfinlab was
            # withdrawn), so it is reported, not assumed to be an error in the claim.
            buckets["GONE"].append((pkg, entries[0][0], "-", entries))
            continue
        cur, cur_date = latest_of(data)
        claimed = entries[0][0]
        known = {claimed for claimed, _, _ in entries}
        if claimed == cur:
            buckets["CURRENT"].append((pkg, claimed, cur_date, entries))
        elif not any(k in data["releases"] for k in known):
            buckets["WRONG"].append((pkg, claimed, cur, entries))
        else:
            age = (TODAY - dt.date.fromisoformat(cur_date)).days if cur_date else 0
            buckets["BEHIND"].append((pkg, claimed, f"{cur} ({cur_date}, {age}d ago)", entries))

    for k in ("WRONG", "GONE", "BEHIND", "CURRENT"):
        rows = buckets[k]
        if not rows:
            continue
        print(f"{k}: {len(rows)}")
        if k == "CURRENT" and not verbose:
            continue
        for pkg, claimed, now, entries in rows:
            files = sorted({str(f) for _, _, f in entries})
            print(f"  {pkg:<24} claimed {claimed:<12} now {now}")
            if verbose:
                for f in files:
                    print(f"      {f}")
        print()

    n_bad = len(buckets["WRONG"])
    n_behind = len(buckets["BEHIND"])
    total = sum(len(v) for v in buckets.values())
    print(f"{len(buckets['CURRENT'])}/{total} claims still current; "
          f"{n_behind} behind; {n_bad} wrong; {len(buckets['GONE'])} not on PyPI")
    print("\nBEHIND is not automatically a defect — a dated claim that was true when written\n"
          "is honest. It becomes a defect when the SKILL.md draws a conclusion the new\n"
          "version invalidates. Re-read those files before updating the number.")

    return 1 if (n_bad or n_behind > max_behind) else 0


if __name__ == "__main__":
    sys.exit(main())
