#!/usr/bin/env python3
"""Measure whether a query's words point at exactly one skill description.

WHAT THIS IS: a lexical proxy, not a live model test. It scores every skill
description against a query by weighted term overlap and reports which one wins.
It cannot tell you what Claude will actually select.

WHY IT IS STILL WORTH RUNNING: the failure mode it catches is real and common —
a query whose distinctive words appear in three descriptions equally. When the
margin between the top two is thin, the model is choosing close to arbitrarily,
and that is fixable by editing the descriptions. Treat a thin margin as a defect
even when the top-1 pick happens to be right.

Run:  python scripts/eval_triggers.py [--verbose]
Exit: 0 if top-1 accuracy is 100% and no case is below the margin floor.
"""
from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).parent))
from validate import parse_frontmatter  # noqa: E402

MARGIN_FLOOR = 0.15   # top1 must beat top2 by this fraction of top1
STOP = set("""a an and are as at be but by for from how i if in is it my me of on or that the
this to use used using want with what when which who why you your do does can could should would
get got need help please make made new now here there all any some more most""".split())


def toks(s: str) -> list[str]:
    # keep CJK characters as individual tokens; ASCII words lowercased
    s = s.lower()
    words = re.findall(r"[a-z][a-z0-9_.\-]{1,}", s)
    cjk = re.findall(r"[一-鿿]", s)
    return [w for w in words if w not in STOP and len(w) > 1] + cjk


SKIP_RE = re.compile(r"\bSKIP\b(.*)$", re.S)


def load_skills() -> list[dict]:
    """Split each description into its positive part and its SKIP clause.

    A SKIP clause names the COMPETING skill and its topic ("SKIP for filings and
    macro (fundamental-and-macro-data)"). A bag-of-words scorer would count those
    words as evidence FOR this skill, which is backwards — the model reads them as
    a negative. So they are scored negatively here, which is what makes the SKIP
    mechanism measurable at all.
    """
    out = []
    for md in sorted(ROOT.glob("plugins/*/skills/*/SKILL.md")):
        fm, _ = parse_frontmatter(md.read_text(encoding="utf-8"))
        name = fm.get("name", md.parent.name)
        desc = fm.get("description", "")
        m = SKIP_RE.search(desc)
        pos = desc[: m.start()] if m else desc
        neg = m.group(1) if m else ""
        out.append({
            "name": name, "desc": desc,
            "toks": Counter(toks(name + " " + pos)),
            "neg": Counter(toks(neg)),
        })
    return out


def score(qt: list[str], skill: dict, idf: dict) -> float:
    """idf-weighted overlap with the positive part, minus overlap with the SKIP clause."""
    st, neg = skill["toks"], skill["neg"]
    q = set(qt)
    pos_hits = sum(idf.get(t, 1.0) for t in q if st.get(t))
    neg_hits = sum(idf.get(t, 1.0) for t in q if neg.get(t) and not st.get(t))
    return pos_hits - 0.75 * neg_hits


def main() -> int:
    verbose = "--verbose" in sys.argv
    skills = load_skills()
    if not skills:
        print("no skills found")
        return 1

    # idf over descriptions: a term in every description carries no signal
    n = len(skills)
    df = Counter()
    for s in skills:
        df.update(set(s["toks"]))
    idf = {t: math.log(1 + n / (1 + c)) for t, c in df.items()}

    qs = [json.loads(l) for l in (ROOT / "evals" / "queries.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

    hits = 0
    thin: list[tuple] = []
    misses: list[tuple] = []
    for case in qs:
        qt = toks(case["q"])
        ranked = sorted(((score(qt, s, idf), s["name"]) for s in skills), reverse=True)
        top, second = ranked[0], ranked[1]
        ok = top[1] == case["expect"]
        hits += ok
        margin = (top[0] - second[0]) / top[0] if top[0] > 0 else 0.0
        if not ok:
            misses.append((case["q"], case["expect"], top[1], second[1]))
        elif margin < MARGIN_FLOOR:
            thin.append((case["q"], top[1], second[1], margin))
        if verbose:
            mark = "ok " if ok else "MISS"
            print(f"{mark} {margin:5.2f}  {case['q'][:58]:<58} -> {top[1]}")

    acc = hits / len(qs)
    print(f"\ntop-1 accuracy : {hits}/{len(qs)} = {acc:.0%}")
    print(f"thin margins   : {len(thin)} (top-2 gap < {MARGIN_FLOOR:.0%})")

    if misses:
        print("\nMISSES — the query points at the wrong skill:")
        for q, exp, got, snd in misses:
            print(f"  {q[:60]!r}\n      expected {exp}, got {got} (2nd {snd})")
    if thin:
        print("\nTHIN — right answer, but the model is nearly coin-flipping:")
        for q, got, snd, m in thin:
            print(f"  {m:.2f}  {q[:56]!r}  {got}  vs  {snd}")

    return 0 if (not misses and not thin) else 1


if __name__ == "__main__":
    sys.exit(main())
