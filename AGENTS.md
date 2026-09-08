# AGENTS.md

Instructions for coding agents working in this repository.

## What this repo is

A library of [Agent Skills](https://agentskills.io/specification) about the Python
quantitative-finance ecosystem. The content is not tutorials — it is the set of defaults, licence
changes and silent failure modes that a model's training prior gets wrong, each one dated and
marked with how it was verified.

Two tiers:

- **domain skills** (`fin-core`, `fin-china`, `fin-asia`, `fin-crypto`, `fin-futures-fx`,
  `fin-llm`) — one per task a person has. These answer "which library, and what will bite me".
- **`fin-libraries`** — one skill per library, opt-in because it costs ~5,300 tokens of listing
  budget on its own (`build_index.py` prints the current figure per plugin). These answer "how do I use this one correctly", and each links back to its
  domain skill.

## Before you commit

```bash
python scripts/build_index.py   # 1. regenerate the catalog and README table
python scripts/build_package.py # 2. regenerate fin_skills/ - AFTER build_index, it copies index.json
python scripts/validate.py      # 3. must print OK (it runs build_package.py --check)
```

`catalog/index.json`, the README skill table and everything under `fin_skills/` except `__init__.py` are
**generated**. Never hand-edit them.

## Rules that exist because they were broken

**Do not state a number you did not produce.** Numbers in a SKILL.md should come from a script in
that skill, or from a command whose output you saw. Several claims here were wrong on the first
pass and were caught only by running them.

**Do not reflow prose you are not editing.** Re-wrapping a file to a different column width makes
every line show as modified, which hides deletions. An agent doing this once flattened three
markdown tables into slash-separated prose and deleted a set of cross-references, and the diff
looked like pure whitespace churn. If you must check, compare with whitespace normalised:

```bash
python -c "
import re,subprocess,pathlib,sys
f=sys.argv[1]
old=subprocess.run(['git','show',f'HEAD:{f}'],capture_output=True,text=True).stdout
new=pathlib.Path(f).read_text(encoding='utf-8')
n=lambda s: re.sub(r'\s+',' ',s).strip()
print('content identical' if n(old)==n(new) else 'CONTENT CHANGED, not just wrapping')
" <path>
```

**Do not edit files another agent is writing.** Concurrent writes to the same SKILL.md have
silently reverted committed work here. If you are one of several agents, stay inside the files you
were assigned.

**The frontmatter spec is exactly six fields** — `name`, `description`, `license`, `compatibility`,
`metadata`, `allowed-tools`. Extra keys pass in Claude Code and hard-fail on claude.ai upload, so
`validate.py` rejects them. A mis-indented block scalar silently swallows the keys that follow it;
the validator checks for that specifically.

## Testing

`python -m pytest -q` runs the unit-test suite under `tests/` - one module per generated skill
script plus the `fin_skills.api` layer (over 500 tests; slow demo paths are marked `slow` and
deselected by default, `-m slow` runs them). Tests import the GENERATED package, so after editing
a plugin script run `build_package.py` before the tests can see the change. Library-specific
tests skip when the library is absent. Beyond the unit tests, verification is:

| Script | Checks |
|---|---|
| `scripts/validate.py` | spec compliance, name/directory match, description caps, dead reference links, empty `references/`, domain backlinks, README skill count |
| `scripts/build_index.py` | regenerates the catalog and README table |
| `scripts/eval_triggers.py` | lexical smoke test — reports strict top-1 and routed accuracy. Deliberately reproducible across `PYTHONHASHSEED` values |
| `scripts/eval_blind.py` | the real measurement: a model picks a skill from the listing alone. `prepare`, then have models answer, then `score` |
| `scripts/check_drift.py` | re-checks version claims against PyPI (network) |
| `scripts/check_repo_stats.py` | re-checks GitHub stats. Note `open_issues_count` includes pull requests |
| `scripts/check_scripts.py` | runs every skill script in a subprocess with `PYTHONIOENCODING` stripped. A script that prints a non-ASCII character passes under UTF-8 and dies on a stock Windows console |
| `scripts/build_package.py` | regenerates `fin_skills/` - the skills as an importable package (`--check` is run by `validate.py`). Never edit `fin_skills/` by hand |

`eval_triggers.py` is a bag-of-words proxy and degrades once two skills cover the same package.
`eval_blind.py` is the ground truth. Do not tune descriptions to the proxy.

## Conventions

- Lines wrap at about 98 columns
- Markers: ✅ verified at a primary source · ⚠️ secondhand · 🚨 silently wrong · 🔴 dead or broken
- `metadata.verified_on` is a date, and it should move when you re-check the file
- Scripts use only numpy / pandas / scipy, run with a fixed seed, and work whether or not the
  library they describe is installed
