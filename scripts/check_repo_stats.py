#!/usr/bin/env python3
"""Verify the GitHub statistics asserted in the reference cards — and catch one specific
mistake this repo already made.

THE MISTAKE: `gh api repos/<o>/<r>` returns `open_issues_count`, and that field **counts pull
requests as issues**. Every "N open issues" figure taken from it is inflated — measured here
by 30% (freqtrade) to 262% (ccxt, where 614 of 848 are PRs). The correct source is the search
API:

    gh api 'search/issues?q=repo:<o>/<r>+is:issue+is:open' -q .total_count

That matters because this library used those numbers as EVIDENCE — "2,445 open issues is the
worst ratio in this catalogue" was a judgement resting on a count that was 72% PRs.

Run:
    python scripts/check_repo_stats.py            # report drift
    python scripts/check_repo_stats.py --fix      # rewrite the counts in place
Exit 1 if any asserted count is wrong by more than --tolerance (default 10%).
"""
from __future__ import annotations

import concurrent.futures
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# "`owner/repo` — 1,234★, 56 forks, 78 open issues"  and the many looser variants used here.
# We anchor on the repo slug, then look ahead a short distance for a count labelled as issues.
REPO = re.compile(r"[`*]{0,2}(?P<slug>[A-Za-z0-9][\w.-]{0,38}/[A-Za-z0-9][\w.-]{0,60})[`*]{0,2}")
ISSUES = re.compile(r"(?P<n>[\d,]+)\s*(?:\*\*)?\s*open[\s-]+issues", re.I)


def gh_json(path: str) -> dict | None:
    try:
        out = subprocess.run(["gh", "api", path], capture_output=True, text=True, timeout=45)
        return json.loads(out.stdout) if out.returncode == 0 else None
    except Exception:
        return None


def true_issue_count(slug: str) -> int | None:
    d = gh_json(f"search/issues?q=repo:{slug}+is:issue+is:open&per_page=1")
    return d.get("total_count") if d else None


def collect() -> dict[str, list[tuple[Path, int, str]]]:
    """slug -> [(file, claimed_count, matched_text)]"""
    found: dict[str, list] = defaultdict(list)
    for md in sorted(ROOT.glob("plugins/**/*.md")):
        text = md.read_text(encoding="utf-8")
        for line in text.splitlines():
            im = ISSUES.search(line)
            if not im:
                continue
            # the repo slug governing this line is the last one mentioned on it
            slugs = [m.group("slug") for m in REPO.finditer(line)
                     if "/" in m.group("slug") and not m.group("slug").endswith(".md")]
            if not slugs:
                continue
            n = int(im.group("n").replace(",", ""))
            found[slugs[-1]].append((md.relative_to(ROOT), n, line.strip()[:110]))
    return found


def main() -> int:
    fix = "--fix" in sys.argv
    tol = float(next((a.split("=")[1] for a in sys.argv if a.startswith("--tolerance=")), 0.10))

    claims = collect()
    if not claims:
        print("no 'N open issues' claims found alongside a repo slug")
        return 0

    print(f"verifying {len(claims)} repos ...\n")
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        truth = dict(zip(claims, ex.map(true_issue_count, claims)))

    bad, ok, unknown = [], 0, []
    for slug, entries in sorted(claims.items()):
        real = truth.get(slug)
        if real is None:
            unknown.append(slug)
            continue
        for f, claimed, line in entries:
            if real == 0:
                drift = 1.0 if claimed else 0.0
            else:
                drift = abs(claimed - real) / real
            if drift > tol:
                bad.append((slug, f, claimed, real, line))
            else:
                ok += 1

    for slug, f, claimed, real, line in bad:
        print(f"  {slug:<34} claims {claimed:>6}  actual issues {real:>6}   {f}")
    if unknown:
        print(f"\ncould not resolve: {', '.join(unknown[:8])}")
    print(f"\n{ok} accurate, {len(bad)} inflated, {len(unknown)} unresolved")

    if bad and fix:
        by_file: dict[Path, list] = defaultdict(list)
        for slug, f, claimed, real, line in bad:
            by_file[f].append((claimed, real))
        for f, pairs in by_file.items():
            p = ROOT / f
            t = p.read_text(encoding="utf-8")
            for claimed, real in pairs:
                # only rewrite the number when it is immediately followed by "open issues"
                t = re.sub(rf"(?<![\d,]){claimed:,}\s*(\*\*)?\s*(open[\s-]+issues)",
                           rf"{real:,} \1\2".replace("  ", " "), t)
                t = re.sub(rf"(?<![\d,]){claimed}\s*(\*\*)?\s*(open[\s-]+issues)",
                           rf"{real} \1\2".replace("  ", " "), t)
            p.write_text(t, encoding="utf-8")
        print(f"\nrewrote counts in {len(by_file)} files — re-run to confirm")
        return 0

    if bad:
        print("\nRe-run with --fix to rewrite them. Then re-read any sentence that drew a\n"
              "CONCLUSION from the old number: an issue count used as evidence of neglect\n"
              "has to be re-argued, not just re-numbered.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
