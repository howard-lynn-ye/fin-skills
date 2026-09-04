#!/usr/bin/env python3
"""Validate every SKILL.md against the Agent Skills spec (agentskills.io/specification).

The spec allows EXACTLY six frontmatter fields. Claude Code accepts more, but any
non-spec key is a HARD ERROR on claude.ai upload / Skills API / package_skill.py.
This repo targets the portable subset, so extra keys fail here too.

Run:  python scripts/validate.py
Exit: 0 clean, 1 on any error.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SPEC_FIELDS = {"name", "description", "license", "compatibility", "metadata", "allowed-tools"}
REQUIRED = {"name", "description"}
NAME_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
RESERVED_WORDS = ("anthropic", "claude")

# Discovery budget: ~100 tokens/skill, default listing budget ~2000 tokens.
DESC_HARD_CAP = 1024          # spec limit
DESC_LISTING_CAP = 1536       # per-entry cap in the runtime listing (chars)
SKILL_BUDGET_WARN = 20        # skills per plugin before the listing starts truncating


def parse_frontmatter(text: str) -> tuple[dict, list[str]]:
    """Minimal YAML frontmatter parse: top-level `key:` pairs, block scalars, one nested map."""
    errs: list[str] = []
    if not text.startswith("---"):
        return {}, ["frontmatter must start at byte 0 with '---'"]
    end = text.find("\n---", 3)
    if end == -1:
        return {}, ["frontmatter is not terminated with '---'"]
    body = text[3:end].strip("\n")

    fm: dict = {}
    key = None
    buf: list[str] = []
    nested: str | None = None

    def flush():
        nonlocal key, buf
        if key is not None:
            fm[key] = " ".join(x.strip() for x in buf if x.strip()) if buf else fm.get(key, "")
        key, buf = None, []

    for line in body.split("\n"):
        if not line.strip():
            continue
        if line.startswith("  ") and nested:
            k, _, v = line.strip().partition(":")
            fm.setdefault(nested, {})[k.strip()] = v.strip().strip('"\'')
            continue
        if line.startswith("  ") and key:
            buf.append(line)
            continue
        flush()
        nested = None
        k, sep, v = line.partition(":")
        if not sep:
            errs.append(f"unparsable frontmatter line: {line!r}")
            continue
        k = k.strip()
        v = v.strip()
        if v in ("", ">-", ">", "|", "|-"):
            if v == "":
                nested = k
                fm.setdefault(k, {})
            else:
                key = k
            continue
        fm[k] = v.strip('"\'')
    flush()
    return fm, errs


def check_skill(skill_md: Path) -> list[str]:
    rel = skill_md.relative_to(ROOT).as_posix()
    text = skill_md.read_text(encoding="utf-8")
    fm, errs = parse_frontmatter(text)
    errs = [f"{rel}: {e}" for e in errs]
    if not fm:
        return errs

    extra = set(fm) - SPEC_FIELDS
    if extra:
        errs.append(f"{rel}: non-spec frontmatter key(s) {sorted(extra)} — "
                    f"these hard-error on claude.ai upload. Allowed: {sorted(SPEC_FIELDS)}")
    for f in REQUIRED - set(fm):
        errs.append(f"{rel}: missing required field '{f}'")

    # A block scalar that is not de-indented correctly swallows the keys that follow it,
    # so `license:`/`metadata:` end up INSIDE the description and disappear as fields.
    # Both required fields are still present, so the checks above pass. Catch it directly.
    for key in ("license", "metadata", "name", "compatibility", "allowed-tools"):
        if re.search(rf"(?<![\w-]){key}\s*:", str(fm.get("description", ""))):
            errs.append(f"{rel}: the description contains '{key}:' — the block scalar has "
                        f"swallowed the frontmatter keys that follow it")
    if "license" not in fm:
        errs.append(f"{rel}: missing 'license' (present on every other skill — likely swallowed)")
    if not (fm.get("metadata") or {}).get("verified_on"):
        errs.append(f"{rel}: metadata.verified_on missing — every claim in this repo carries a date")

    name = fm.get("name", "")
    if name:
        if not NAME_RE.match(name):
            errs.append(f"{rel}: name {name!r} must be lowercase a-z0-9 with single hyphens, "
                        f"no leading/trailing/double hyphen")
        if len(name) > 64:
            errs.append(f"{rel}: name is {len(name)} chars (max 64)")
        if name != skill_md.parent.name:
            errs.append(f"{rel}: name {name!r} must match parent directory "
                        f"{skill_md.parent.name!r}")
        for w in RESERVED_WORDS:
            if w in name:
                errs.append(f"{rel}: name contains reserved word {w!r} "
                            f"(rejected by claude.ai upload)")

    desc = fm.get("description", "")
    if desc:
        if len(desc) > DESC_HARD_CAP:
            errs.append(f"{rel}: description is {len(desc)} chars (spec max {DESC_HARD_CAP})")
        elif len(desc) > DESC_LISTING_CAP:
            errs.append(f"{rel}: description is {len(desc)} chars — the runtime listing "
                        f"truncates at {DESC_LISTING_CAP}")
        if not desc[0].isupper() and not desc.startswith(("A ", "An ", "The ")):
            errs.append(f"{rel}: description should read as a third-person sentence")

    # Level 2 budget: SKILL.md body under ~5k tokens (~20k chars) is the documented guidance.
    body_chars = len(text)
    if body_chars > 24_000:
        errs.append(f"{rel}: SKILL.md is {body_chars} chars (~{body_chars // 4} tokens); "
                    f"guidance is under 5k tokens — move detail into references/")

    # Every referenced file must exist (level-3 files are dead weight if unnamed, and
    # broken if named but absent).
    skill_dir = skill_md.parent
    for m in re.finditer(r"`(references/[\w./-]+\.md)`", text):
        if not (skill_dir / m.group(1)).exists():
            errs.append(f"{rel}: references missing file {m.group(1)}")
    for m in re.finditer(r"`(scripts/[\w./-]+\.py)`", text):
        if not (skill_dir / m.group(1)).exists():
            errs.append(f"{rel}: references missing script {m.group(1)}")

    # An existing-but-empty references/ is the state that makes a skill's own pointers
    # unfulfillable: the model follows the instruction, finds nothing, and is worse off
    # than if the directory had never existed. Either put files in it or delete it.
    ref_dir = skill_dir / "references"
    if ref_dir.is_dir() and not list(ref_dir.glob("*.md")):
        errs.append(f"{rel}: references/ exists but is empty — add files or remove the directory "
                    f"(an empty one invites the model to grep nothing)")

    # Same for scripts/.
    scr_dir = skill_dir / "scripts"
    if scr_dir.is_dir() and not list(scr_dir.glob("*.py")):
        errs.append(f"{rel}: scripts/ exists but is empty — add files or remove the directory")
    return errs


def main() -> int:
    skills = sorted(ROOT.glob("plugins/*/skills/*/SKILL.md"))
    if not skills:
        print("no SKILL.md found under plugins/*/skills/*/")
        return 1

    all_errs: list[str] = []
    for s in skills:
        all_errs.extend(check_skill(s))

    # Per-plugin discovery-budget check
    for plugin_dir in sorted(ROOT.glob("plugins/*")):
        n = len(list(plugin_dir.glob("skills/*/SKILL.md")))
        if n > SKILL_BUDGET_WARN:
            all_errs.append(
                f"{plugin_dir.name}: {n} skills exceeds the ~{SKILL_BUDGET_WARN}-skill discovery "
                f"budget; descriptions will be silently dropped to name-only")

    # marketplace.json must list every skill that exists, and only those
    mp = ROOT / ".claude-plugin" / "marketplace.json"
    if mp.exists():
        data = json.loads(mp.read_text(encoding="utf-8"))
        for p in data.get("plugins", []):
            pdir = ROOT / p["source"].lstrip("./")
            declared = {s.rstrip("/").split("/")[-1] for s in p.get("skills", [])}
            actual = {d.parent.name for d in pdir.glob("skills/*/SKILL.md")}
            for missing in sorted(actual - declared):
                all_errs.append(f"marketplace.json: {p['name']} does not list skill {missing!r}")
            for ghost in sorted(declared - actual):
                all_errs.append(f"marketplace.json: {p['name']} lists {ghost!r} which has no SKILL.md")

    if all_errs:
        print(f"FAIL — {len(all_errs)} problem(s):\n")
        for e in all_errs:
            print("  •", e)
        return 1
    print(f"OK — {len(skills)} skills validated against the 6-field Agent Skills spec")
    return 0


if __name__ == "__main__":
    sys.exit(main())
