#!/usr/bin/env python3
"""Blind skill-selection test — the real one, not the lexical proxy.

`eval_triggers.py` scores word overlap. This script prepares the inputs for a
test where a MODEL does the selecting, seeing exactly what it sees at discovery
time: a list of skill names and descriptions, and one user query. Nothing else.

Usage
-----
  python scripts/eval_blind.py prepare
      writes evals/_listing.md (the discovery-time view) and evals/_batch<N>.txt

  then, for each batch, ask a fresh model instance:
      "Here is the skill listing. For each numbered query, name the ONE skill you
       would load, or 'none'. Honour any SKIP clause. Answer with a JSON array."
      -> save its reply to evals/_blind_batch<N>.json

  python scripts/eval_blind.py score
      compares every saved batch against evals/queries.jsonl and reports accuracy

Why blind matters: an instance that has read the skill bodies, or that knows the
repo's structure, will infer the intended answer rather than select on the
description alone. Only the description is available at discovery time, so only
the description may be used.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVALS = ROOT / "evals"
N_BATCHES = 3


def load_queries() -> list[dict]:
    return [json.loads(l) for l in (EVALS / "queries.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]


def prepare() -> int:
    listing = []
    for md in sorted(ROOT.glob("plugins/*/skills/*/SKILL.md")):
        fm = re.match(r"(?s)^---\n(.*?)\n---\n", md.read_text(encoding="utf-8")).group(1)
        name = re.search(r"(?m)^name:\s*(\S+)", fm).group(1)
        desc = " ".join(re.search(r"(?m)^description: >-\n((?:  .*\n?)+)", fm).group(1).split())
        listing.append(f"- **{name}**: {desc}")
    text = "\n\n".join(listing)
    (EVALS / "_listing.md").write_text(text, encoding="utf-8")

    qs = load_queries()
    for i in range(N_BATCHES):
        batch = qs[i::N_BATCHES]
        (EVALS / f"_batch{i}.txt").write_text(
            "\n".join(f"{j+1}. {c['q']}" for j, c in enumerate(batch)), encoding="utf-8")
    print(f"listing: {len(listing)} skills, {len(text)} chars  "
          f"(~{len(text)//4} tokens at discovery time)")
    print(f"batches: {N_BATCHES} x ~{len(qs)//N_BATCHES} queries -> evals/_batch<N>.txt")
    return 0


def score() -> int:
    qs = load_queries()
    total = hits = 0
    misses: list[tuple] = []
    missing_batches = []
    for i in range(N_BATCHES):
        f = EVALS / f"_blind_batch{i}.json"
        if not f.exists():
            missing_batches.append(i)
            continue
        batch = qs[i::N_BATCHES]
        ans = json.loads(f.read_text(encoding="utf-8"))
        if len(ans) != len(batch):
            print(f"batch{i}: {len(ans)} answers for {len(batch)} queries — skipped")
            continue
        # `ans` is a dict keyed by 1-based query number. Iterating it yields KEYS, so
        # zip(batch, ans) silently compares query numbers to skill names and reports 0%.
        # Order by numeric key, and accept either a bare name or a {"pick": ...} object.
        def pick(v, key):
            if isinstance(v, str):
                return v
            if isinstance(v, dict) and isinstance(v.get("pick"), str):
                return v["pick"]
            raise SystemExit(f"batch{i} answer {key!r} is {v!r} — expected a skill name "
                             f"or an object with a 'pick' string")
        picks = [pick(ans[k], k) for k in sorted(ans, key=lambda s: int(s))]
        for c, a in zip(batch, picks):
            total += 1
            if a == c["expect"]:
                hits += 1
            else:
                misses.append((c["q"], c["expect"], a))

    if missing_batches:
        print(f"missing batches: {missing_batches} (run `prepare`, ask a model, save the JSON)")
    if not total:
        return 1

    print(f"\nblind top-1 accuracy: {hits}/{total} = {hits/total:.0%}")
    if misses:
        print("\nmisses:")
        for q, exp, got in misses:
            print(f"  {q[:62]!r}\n      expected {exp}\n      got      {got}")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "score"
    sys.exit(prepare() if cmd == "prepare" else score())
