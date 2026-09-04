#!/usr/bin/env python3
"""Generate catalog/index.json and the README skill table FROM the SKILL.md frontmatter.

Nothing reads index.json at runtime — Claude Code does not consume it. Its value is
out-of-band: CI validation, coverage checks, and future semantic retrieval. The rule
that makes it worth having is that it is GENERATED, never hand-maintained, so it
cannot drift from the skills themselves.

Run:  python scripts/build_index.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from validate import parse_frontmatter  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BEGIN = "<!-- BEGIN GENERATED SKILL TABLE -->"
END = "<!-- END GENERATED SKILL TABLE -->"


def first_sentence(desc: str) -> str:
    s = re.split(r"(?<=[.!?])\s+", desc.strip())[0]
    return s[:180]


def collect() -> list[dict]:
    out = []
    for md in sorted(ROOT.glob("plugins/*/skills/*/SKILL.md")):
        fm, _ = parse_frontmatter(md.read_text(encoding="utf-8"))
        skill_dir = md.parent
        meta = fm.get("metadata", {}) or {}
        refs = sorted(p.name for p in (skill_dir / "references").glob("*.md")) \
            if (skill_dir / "references").exists() else []
        scripts = sorted(p.name for p in (skill_dir / "scripts").glob("*.py")) \
            if (skill_dir / "scripts").exists() else []
        body = md.read_text(encoding="utf-8")
        out.append({
            "name": fm.get("name", skill_dir.name),
            "plugin": md.parents[2].name,
            "path": md.relative_to(ROOT).as_posix(),
            "description": fm.get("description", "").strip(),
            "summary": first_sentence(fm.get("description", "")),
            "license": fm.get("license", ""),
            "version": meta.get("version", ""),
            "verified_on": meta.get("verified_on", ""),
            "body_chars": len(body),
            "approx_body_tokens": len(body) // 4,
            "references": refs,
            "scripts": scripts,
        })
    return out


def render_table(skills: list[dict]) -> str:
    lines = ["| Plugin | Skill | Covers | Refs | Scripts |", "|---|---|---|---:|---:|"]
    for s in sorted(skills, key=lambda x: (x["plugin"], x["name"])):
        lines.append(
            f"| `{s['plugin']}` | [`{s['name']}`]({s['path']}) | {s['summary']} "
            f"| {len(s['references'])} | {len(s['scripts'])} |"
        )
    return "\n".join(lines)


def main() -> int:
    skills = collect()
    if not skills:
        print("no skills found")
        return 1

    total_desc = sum(len(s["description"]) for s in skills)
    index = {
        "generated_by": "scripts/build_index.py — do not edit by hand",
        "n_skills": len(skills),
        "n_reference_files": sum(len(s["references"]) for s in skills),
        "n_scripts": sum(len(s["scripts"]) for s in skills),
        "discovery_cost": {
            "total_description_chars": total_desc,
            "approx_tokens": total_desc // 4,
            "note": "Claude Code's default skill-listing budget is ~1% of the context window. "
                    "Past roughly 20 skills, descriptions are silently dropped to name-only and "
                    "those skills stop auto-triggering. Skills are split across plugins so a user "
                    "installs only what they need.",
        },
        "plugins": sorted({s["plugin"] for s in skills}),
        "skills": skills,
    }

    out = ROOT / "catalog" / "index.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    readme = ROOT / "README.md"
    if readme.exists():
        txt = readme.read_text(encoding="utf-8")
        if BEGIN in txt and END in txt:
            new = f"{BEGIN}\n\n{render_table(skills)}\n\n{END}"
            txt = re.sub(re.escape(BEGIN) + r".*?" + re.escape(END), new, txt, flags=re.S)
            readme.write_text(txt, encoding="utf-8")
            print("README skill table regenerated")

    per_plugin = {}
    for s in skills:
        per_plugin.setdefault(s["plugin"], []).append(s)

    print(f"catalog/index.json written — {len(skills)} skills, "
          f"{index['n_reference_files']} reference files, {index['n_scripts']} scripts")
    print(f"discovery cost: ~{total_desc // 4} tokens if ALL plugins are installed")
    for p, items in sorted(per_plugin.items()):
        cost = sum(len(s["description"]) for s in items) // 4
        print(f"  {p:<12} {len(items):>2} skills  ~{cost:>4} tokens")
    return 0


if __name__ == "__main__":
    sys.exit(main())
