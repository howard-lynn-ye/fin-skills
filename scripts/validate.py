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

# Tooling output carries markers and dashes; on a stock Windows console (cp1252) a bare
# print of them raises UnicodeEncodeError. Skill scripts stay ASCII; tooling may not.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent

# Validate THIS checkout, not whichever fin_skills happens to be importable. Running
# `python scripts/validate.py` puts scripts/ on sys.path[0], so `import fin_skills` finds
# the editable install - which, in a git worktree, is a DIFFERENT tree. Three agents hit
# that as a false "README says N guards; there are N+1", a count read from another
# checkout. Prepend the repo root so the live-count check below reads the tree it is
# actually validating.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
        # A description may legitimately open with the lowercase package name — for a
        # per-library skill the name IS the trigger, and package names are lowercase.
        first = desc.split()[0].strip("`*.,").lower()
        bare = name.replace("lib-", "")
        starts_with_own_name = bool(first) and (first in bare or bare in first)
        if (not desc[0].isupper() and not desc.startswith(("A ", "An ", "The "))
                and not starts_with_own_name):
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

    # A per-library skill is only half a skill without a pointer back to the domain skill
    # that owns it: on its own it answers "how do I use X" but never "should I use X".
    # A concurrent writer dropped this section from 8 files once and nothing caught it.
    if skill_dir.parent.parent.name == "fin-libraries" and "## Where this sits" not in text:
        errs.append(f"{rel}: no '## Where this sits' section — a lib skill must link back to "
                    f"the domain skill that owns it, or the two tiers compete instead of chaining")

    # Same for scripts/.
    scr_dir = skill_dir / "scripts"
    if scr_dir.is_dir() and not list(scr_dir.glob("*.py")):
        errs.append(f"{rel}: scripts/ exists but is empty — add files or remove the directory")

    # A skill script must run standalone AND import inside the generated package, where a
    # bare `import sibling` no longer resolves. The accepted form is the dual-mode idiom:
    #     try:    from .sibling import x      # inside fin_skills.<ns>
    #     except ImportError: from sibling import x   # run as a script
    # so a bare sibling import is flagged only when no relative form for it exists.
    if scr_dir.is_dir():
        sibs = {p.stem for p in scr_dir.glob('*.py')}
        hits = set()
        for py in sorted(scr_dir.glob('*.py')):
            src = [l.strip() for l in py.read_text(encoding='utf-8').splitlines()]
            for sib in sibs - {py.stem}:
                bare = any(s == f'import {sib}' or s.startswith(f'import {sib} ')
                           or s.startswith(f'from {sib} import') for s in src)
                relative = any(s.startswith(f'from .{sib} import') or s == f'from . import {sib}'
                               for s in src)
                if bare and not relative:
                    hits.add((py.name, sib))
        for name, sib in sorted(hits):
            errs.append(f'{rel}: {name} imports sibling script {sib} with no relative form - use '
                        f'try: from .{sib} import ... except ImportError: from {sib} import ...')

    return errs


def _live_counts():
    """[(label, regex with one capture group, actual count)] for the README's prose counts.

    Imported lazily and tolerantly: if the package cannot be imported (a half-finished
    regeneration, a missing optional dependency) this returns nothing rather than failing
    the whole validator on a documentation check.
    """
    out = []
    try:
        from fin_skills.api import registry
        out.append(("guards", r"(\d+) guards that return a `GuardResult`", len(registry())))
    except Exception:
        pass
    try:
        from fin_skills.tools.runner import list_tools
        out.append(("tools", r"(\d+) tools an agent can call over JSON", len(list_tools())))
    except Exception:
        pass
    return out


def main() -> int:
    skills = sorted(ROOT.glob("plugins/*/skills/*/SKILL.md"))
    if not skills:
        print("no SKILL.md found under plugins/*/skills/*/")
        return 1

    all_errs: list[str] = []
    warnings: list[str] = []
    mp_path = ROOT / '.claude-plugin' / 'marketplace.json'
    for s in skills:
        all_errs.extend(check_skill(s))

    # Per-plugin discovery-budget check
    for plugin_dir in sorted(ROOT.glob("plugins/*")):
        n = len(list(plugin_dir.glob("skills/*/SKILL.md")))
        if n > SKILL_BUDGET_WARN:
            # The check exists to catch ACCIDENTAL over-budget. A plugin whose marketplace
            # entry states the cost has made an informed choice, so warn rather than fail.
            declared = False
            if mp_path.exists():
                for p in json.loads(mp_path.read_text(encoding="utf-8")).get("plugins", []):
                    # "Declared" means the entry STATES the cost. An opt-in plugin says so;
                    # a default-installed one like fin-core cannot honestly claim to be
                    # opt-in, so quoting its measured listing budget counts too. Either way
                    # an accidental extra skill still fails until someone writes the cost
                    # down and re-runs build_index.py for the figure.
                    desc = p.get("description") or ""
                    if p["name"] == plugin_dir.name and ("OPT-IN" in desc
                                                         or "skill-listing budget" in desc):
                        declared = True
            msg = (f"{plugin_dir.name}: {n} skills exceeds the ~{SKILL_BUDGET_WARN}-skill discovery "
                   f"budget; descriptions will be silently dropped to name-only")
            (warnings if declared else all_errs).append(msg)

    # marketplace.json: local plugins must list every skill that exists and only those.
    # Federated entries (external `source` objects) get the offline shape checks from the
    # documented schema, and must carry provenance metadata - Claude Code ignores `metadata`,
    # so this is the only place the repo's "every claim dated" rule can bite for them.
    mp = ROOT / ".claude-plugin" / "marketplace.json"
    if mp.exists():
        data = json.loads(mp.read_text(encoding="utf-8"))
        EXT_REQUIRED = {"github": ("repo",), "git-subdir": ("url", "path"), "url": ("url",),
                        "npm": ("package",), "archive": ("url",), "command": ("command",)}
        seen = set()
        for p in data.get("plugins", []):
            name, src = p.get("name", "?"), p.get("source")
            if name in seen:
                all_errs.append(f"marketplace.json: duplicate plugin name {name!r}")
            seen.add(name)
            if isinstance(src, str):
                if not src.startswith("./"):
                    all_errs.append(f"marketplace.json: {name}: string source must start with './'")
                    continue
                pdir = ROOT / src[2:]
                declared = {s.rstrip("/").split("/")[-1] for s in p.get("skills", [])}
                actual = {d.parent.name for d in pdir.glob("skills/*/SKILL.md")}
                for missing in sorted(actual - declared):
                    all_errs.append(f"marketplace.json: {name} does not list skill {missing!r}")
                for ghost in sorted(declared - actual):
                    all_errs.append(f"marketplace.json: {name} lists {ghost!r} which has no SKILL.md")
            elif isinstance(src, dict):
                kind = src.get("source")
                if kind not in EXT_REQUIRED:
                    all_errs.append(f"marketplace.json: {name}: unknown source kind {kind!r}")
                    continue
                for k in EXT_REQUIRED[kind]:
                    if k not in src:
                        all_errs.append(f"marketplace.json: {name}: {kind} source needs {k!r}")
                sha = src.get("sha")
                if sha is not None and not re.fullmatch(r"[a-f0-9]{40}", str(sha)):
                    all_errs.append(f"marketplace.json: {name}: sha must be a full 40-hex commit")
                for s in p.get("skills", []):
                    if not str(s).startswith("./"):
                        all_errs.append(f"marketplace.json: {name}: skills paths must start with './'")
                meta = p.get("metadata") or {}
                for k in ("upstream", "license", "verified_on", "upstream_pushed_at", "skill_count"):
                    if k not in meta:
                        all_errs.append(f"marketplace.json: {name}: external entry needs metadata.{k}")
                if p.get("defaultEnabled", True) is not False:
                    all_errs.append(f"marketplace.json: {name}: external entries install disabled "
                                    f"(defaultEnabled: false)")
                if "not verified by this repo" not in (p.get("description") or "").lower():
                    all_errs.append(f"marketplace.json: {name}: description must state that its claims "
                                    f"are not verified by this repo")
            else:
                all_errs.append(f"marketplace.json: {name}: source must be a './' path or an object")

    # The importable package under fin_skills/ is generated from the skills. A stale copy
    # would ship old guards under a current version number - same rule as catalog/index.json.
    import subprocess
    r = subprocess.run([sys.executable, str(ROOT / 'scripts' / 'build_package.py'), '--check'],
                       capture_output=True, text=True)
    if r.returncode != 0:
        first = (r.stdout.strip().splitlines() or ['no output'])[0]
        all_errs.append('fin_skills/ is out of date - run scripts/build_package.py (' + first + ')')

    # README counts that are NOT in the generated table and so cannot be regenerated by
    # build_index.py: it reads SKILL.md frontmatter only, and importing fin_skills.api to
    # count guards would make the index generator depend on the package that the package
    # generator produces. Checking them here is safe because the sync check above has
    # already established that fin_skills/ is current. "28 guards" was stale for a day.
    readme_path = ROOT / "README.md"
    if readme_path.exists():
        readme = readme_path.read_text(encoding="utf-8")
        for label, pattern, actual in _live_counts():
            m = re.search(pattern, readme)
            if m is None:
                warnings.append(f"README: no '{label}' sentence to check - was it reworded?")
            elif int(m.group(1)) != actual:
                all_errs.append(f"README says {m.group(1)} {label}; there are {actual}. "
                                f"Update the sentence at README.md.")

    if all_errs:
        print(f"FAIL — {len(all_errs)} problem(s):\n")
        for e in all_errs:
            print("  •", e)
        return 1
    for w in warnings:
        print("  WARN:", w)
    print(f"OK - {len(skills)} skills validated against the 6-field Agent Skills spec"          + (f" ({len(warnings)} warning)" if warnings else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
