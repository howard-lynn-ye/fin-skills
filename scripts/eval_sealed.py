#!/usr/bin/env python3
"""Prepare a public input packet, run a tool-free model, then separately score its receipt.

Use --help on each subcommand. Private keys and private holdout sources belong outside
the repository. Existing evals/ questions are public regression fixtures, not hidden tests.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fin_skills.api import sealed_eval as se  # noqa: E402


def skill_listing():
    sys.path.insert(0, str(ROOT / "scripts"))
    from validate import parse_frontmatter
    entries, names = [], []
    for path in sorted(ROOT.glob("plugins/*/skills/*/SKILL.md")):
        fm, errors = parse_frontmatter(path.read_text(encoding="utf-8"))
        if errors:
            raise se.IntegrityError(f"invalid frontmatter in {path.relative_to(ROOT)}")
        names.append(fm["name"])
        entries.append(f"- {fm['name']}: {fm['description']}")
    return "\n\n".join(entries), names + ["none"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "prepare-pit"):
        sub = commands.add_parser(command)
        sub.add_argument("--source", required=True, type=Path,
                         help="routing JSONL with q/expect, or PIT JSON; trusted author input")
        sub.add_argument("--public-dir", required=True, type=Path)
        sub.add_argument("--private-key", required=True, type=Path)
        sub.add_argument("--exposure", required=True,
                         choices=("public-fixture", "private-holdout"))
    sub = commands.add_parser("run")
    sub.add_argument("--public-dir", required=True, type=Path)
    sub.add_argument("--output-dir", required=True, type=Path)
    sub.add_argument("--model", required=True, choices=se.MODELS)
    sub.add_argument("--max-tokens", type=int, default=1024)
    sub = commands.add_parser("score")
    sub.add_argument("--public-dir", required=True, type=Path)
    sub.add_argument("--output-dir", required=True, type=Path)
    sub.add_argument("--private-key", required=True, type=Path)
    sub.add_argument("--receipt-sha256", required=True,
                     help="hash saved by the controller BEFORE scoring; do not recalculate it")
    args = parser.parse_args(argv)
    try:
        if args.command.startswith("prepare"):
            if args.exposure == "private-holdout" and (
                    args.source.resolve().is_relative_to(ROOT) or se.in_git_checkout(args.source)):
                raise se.IntegrityError("a private holdout source must be outside the repository")
            if args.command == "prepare":
                source = [se.parse_json(line) for line in
                          args.source.read_text(encoding="utf-8").splitlines() if line.strip()]
                # The routing surface has only q; do not forward arbitrary context or notes.
                cases = [{"q": c["q"], "expect": c["expect"]} for c in source]
                listing, allowed = skill_listing()
                instructions = "Select the most appropriate skill. Honor SKIP clauses."
            else:
                source = se.read_json(args.source)
                cases = se.point_in_time_cases(
                    source["decisions"], source["observations"],
                    feature_names=source["feature_names"], selection_end=source["selection_end"],
                    selection_labels_end=source["selection_labels_end"],
                    holdout_start=source["holdout_start"])
                listing, allowed = "", source["allowed"]
                instructions = "Use only the observations available at this decision time."
            result = se.prepare(cases, listing=listing, allowed=allowed,
                                public_dir=args.public_dir, private_key=args.private_key,
                                repository=ROOT, exposure=args.exposure, instructions=instructions)
            print(json.dumps(result, indent=2))
        elif args.command == "run":
            receipt_hash = se.run(args.public_dir, args.output_dir, model=args.model,
                                  max_tokens=args.max_tokens)
            print(f"receipt_sha256={receipt_hash}")
            print("Retain this hash outside the output directory before scoring.")
        else:
            result = se.score(args.public_dir, args.private_key, args.output_dir,
                              receipt_sha256=args.receipt_sha256)
            print(json.dumps(result, indent=2))
        return 0
    except (se.IntegrityError, OSError, KeyError, TypeError, ValueError) as exc:
        # Transport errors are already sanitized by the API layer. Never print API key values.
        print(f"Evaluation refused: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
