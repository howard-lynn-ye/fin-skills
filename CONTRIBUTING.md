# Contributing

This repo's only real asset is that its claims are **checked**. A skill that sounds right and is
wrong is worse than no skill, because a model will act on it without hesitating. So the bar here is
not writing quality — it is evidence.

## The one rule

**Every claim carries a marker and a date.**

| Marker | Means |
|---|---|
| ✅ | Verified at a primary source — you ran the code, or read the library's own source, or hit its API |
| ⚠️ | Secondhand — a maintainer's issue comment, a changelog, a paper |
| 🚨 | A trap that silently produces wrong numbers |
| 🔴 | Dead, broken, or relicensed |

`metadata.verified_on` in the frontmatter is the date the skill was last checked as a whole.

**Do not write a number you did not produce.** If you state that a function returns `-81.323237`,
a script in that skill should print `-81.323237`. This is the most common way a contribution gets
rejected.

## Before you open a PR

```bash
python scripts/validate.py      # spec compliance, cross-links, README count
python scripts/build_index.py   # regenerates catalog/index.json and the README table
python scripts/eval_triggers.py # lexical smoke test for skill selection
python scripts/check_drift.py   # re-checks version claims against PyPI (needs network)
```

`validate.py` must print OK. `build_index.py` must leave the tree clean — if `git diff` shows
changes after running it, commit them.

## Adding a skill

The Agent Skills spec allows **exactly six** frontmatter fields: `name`, `description`, `license`,
`compatibility`, `metadata`, `allowed-tools`. Any other key is a hard error on claude.ai upload,
even though Claude Code tolerates it. `name` must equal the parent directory name.

Descriptions use a **TRIGGER / SKIP** shape. The SKIP clause is not decoration — it names the
competing skill, and it is what stops two skills fighting over the same question:

```
TRIGGER - <concrete symptoms, error strings, function names, and the words a user actually types>
SKIP for <neighbouring topic> (<the skill that owns it>).
```

Write triggers as the things a user *says*, including pasted error text and non-English phrasings —
not as a taxonomy of the subject.

### Which plugin

- `fin-core` and the market plugins — one skill per **domain** (a task someone has)
- `fin-libraries` — one skill per **library**, and it must link back to the domain skill that owns
  it under a `## Where this sits` heading. `validate.py` enforces this. Without it a library skill
  can answer "how do I use X" but never "should I use X".

### Discovery budget

The skill listing costs roughly 1% of the context window by default (~2,000 tokens). Past about
20 skills in one plugin, descriptions are silently dropped to name-only and stop auto-triggering.
That is why `fin-libraries` is opt-in and says its own cost in the marketplace entry. If you add
skills, re-run `build_index.py` and check the printed per-plugin token cost.

## Correcting a claim

This is the most valuable contribution and it needs no new skill. Libraries change defaults,
relicense, and break. Open an issue with:

1. The file and line
2. What it says
3. What you observed, with the command or code that shows it
4. Version numbers and the date

A correction that shows a claim is now wrong is worth more than a new skill. Several claims in this
repo were reversed exactly this way — including one that had the safe default backwards.

## Scripts

Scripts must:

- run standalone on **numpy / pandas / scipy** only, with a fixed seed
- demonstrate the trap even when the library in question is **not installed**, via their own
  reference implementation
- when the library **is** importable, verify the reference implementation against it and print the
  comparison rather than asserting agreement

`plugins/fin-libraries/skills/lib-quantstats/scripts/rf_convention.py` is the template.

## Style

- Lines wrap at about 98 columns
- **Never reflow or re-wrap prose, tables or code blocks you are not editing.** A wide re-wrap
  makes every line show as changed and hides real deletions inside the diff. This has already
  destroyed content in this repo once
- No emoji beyond the four markers above
