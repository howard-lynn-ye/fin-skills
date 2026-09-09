## What this changes

<!-- One paragraph. If it corrects a claim, name the file and line and paste the reproduction. -->

## Checklist

The regeneration order matters: `build_package.py` copies `catalog/index.json`, which
`build_index.py` writes, and `validate.py` runs `build_package.py --check`.

- [ ] `python scripts/build_index.py` - regenerates `catalog/index.json` and the README skill
      table, and leaves `git diff` clean afterwards
- [ ] `python scripts/build_package.py` - regenerates `fin_skills/`, run AFTER `build_index.py`
- [ ] `python scripts/validate.py` - prints OK
- [ ] `python -m pytest -q` - passes (tests import the GENERATED package, so regenerate first)
- [ ] `python scripts/check_scripts.py` - passes (every skill script on a stock Windows console
      encoding)
- [ ] **Every number I wrote is measured, not asserted:** it is printed by a script in the same
      skill, or the command and its output are pasted in this PR. Do not state a number you
      did not produce.
- [ ] `metadata.verified_on` moved on every skill I re-checked, and every claim carries a marker
- [ ] I did not reflow or re-wrap prose, tables or code I was not editing
- [ ] Frontmatter is the six spec fields only (`name`, `description`, `license`,
      `compatibility`, `metadata`, `allowed-tools`); a `lib-*` skill has `## Where this sits`
- [ ] Nothing under `fin_skills/` (except `__init__.py`, `py.typed` and `api/`),
      `catalog/index.json` or the README skill table was edited by hand
