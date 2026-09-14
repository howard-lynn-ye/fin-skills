#!/usr/bin/env python3
"""Historical selection replay. This script does NOT enforce answer isolation.

`eval_triggers.py` scores word overlap. This script prepares the inputs for a
test where a MODEL does the selecting, seeing exactly what it sees at discovery
time: a list of skill names and descriptions, and one user query. Keeping other
files out of the prompt does not prevent an agent with tools from reading them.
For enforced API input separation use scripts/eval_sealed.py instead.

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

import argparse
import json
import re
import sys
from pathlib import Path

# Tooling output carries markers and dashes; on a stock Windows console (cp1252) a bare
# print of them raises UnicodeEncodeError. Skill scripts stay ASCII; tooling may not.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
EVALS = ROOT / "evals"
N_BATCHES = 3
QUERIES = EVALS / 'queries.jsonl'


def load_queries() -> list[dict]:
    return [json.loads(l) for l in QUERIES.read_text(encoding="utf-8").splitlines() if l.strip()]


def prepare() -> int:
    EVALS.mkdir(parents=True, exist_ok=True)
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
    print("Historical replay: answer isolation is NOT verified by this scorer.")
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
        def unique_keys(pairs):
            result = {}
            for k, v in pairs:
                if k in result:
                    raise ValueError("duplicate JSON key")
                result[k] = v
            return result
        try:
            ans = json.loads(f.read_text(encoding="utf-8"), object_pairs_hook=unique_keys)
        except ValueError:
            print(f"batch{i}: invalid JSON or duplicate answer IDs")
            missing_batches.append(i)
            continue
        if not isinstance(ans, (list, dict)):
            print(f"batch{i}: expected an answer array or numbered object")
            missing_batches.append(i)
            continue
        if len(ans) != len(batch):
            print(f"batch{i}: {len(ans)} answers for {len(batch)} queries — skipped")
            missing_batches.append(i)
            continue
        # `ans` is a dict keyed by 1-based query number. Iterating it yields KEYS, so
        # zip(batch, ans) silently compares query numbers to skill names and reports 0%.
        # Order by numeric key, and accept either a bare name or a {"pick": ...} object.
        def pick(v, key, batch_number=i):
            if isinstance(v, str):
                return v
            if isinstance(v, dict) and isinstance(v.get("pick"), str):
                return v["pick"]
            raise SystemExit(f"batch{batch_number} answer {key!r} is {v!r} — expected a skill name "
                             f"or an object with a 'pick' string")
        if isinstance(ans, dict) and set(ans) != {str(k + 1) for k in range(len(batch))}:
            print(f"batch{i}: answer IDs must be exactly 1..{len(batch)}")
            missing_batches.append(i)
            continue
        picks = ([pick(value, j + 1) for j, value in enumerate(ans)] if isinstance(ans, list)
                 else [pick(ans[str(k + 1)], k + 1) for k in range(len(batch))])
        for c, a in zip(batch, picks):
            total += 1
            if a == c["expect"]:
                hits += 1
            else:
                misses.append((c["q"], c["expect"], a))

    if missing_batches:
        print(f"missing or invalid batches: {missing_batches}; no complete score")
        return 1
    if not total:
        return 1

    print(f"\nhistorical top-1 accuracy (isolation unverified): {hits}/{total} = {hits/total:.0%}")
    if misses:
        print("\nmisses:")
        for q, exp, got in misses:
            print(f"  {q[:62]!r}\n      expected {exp}\n      got      {got}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', nargs='?', choices=['prepare', 'score'], default='score')
    parser.add_argument('--output-dir', type=Path, default=EVALS,
                        help='keep a new evaluation separate from historical answers')
    parser.add_argument('--queries', type=Path, default=QUERIES)
    args = parser.parse_args()
    EVALS, QUERIES = args.output_dir, args.queries
    sys.exit(prepare() if args.command == 'prepare' else score())
