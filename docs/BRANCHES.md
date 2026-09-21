# Branches: `master` is the one line of work

**`v2` has been merged into `master`. Please develop on `master`.**

## What happened

`v2` was cut from `master` on 2026-09-14 (`ffb46be`) and the two lines ran side by side for
three days: `master` took the algorithm-research and sealed-evaluation work, `v2` took the
A-share pre-trade guards, the KOL credibility registry, the signal reconciler and the
research/production tooling.

- 2026-09-17, `a941d82` - `v2` merged into `master`. Conflicts were the four documents whose
  counts had drifted apart; the catalog, package and benchmark files were regenerated. The
  combined tree has **129 skills, 36 guards and 53 JSON/MCP tools**, and the full CI matrix
  (Linux/Windows/macOS x Python 3.10-3.13, minimum dependencies, optional backends, packaging)
  passed on the merge.
- 2026-09-17, `36db57d` - a hygiene pass over the merged tree: an internal workstation
  hostname and personal home paths replaced by environment variables
  (`FIN_SKILLS_BENCHMARK_DIR`, `FIN_SKILLS_DASHBOARD_HOST`, `STOCK_PREDICTION_DIR`); the
  embedded KOL profiles pseudonymised and labelled as an illustrative external-audit fixture;
  the unbundled "2,521 KOL database" claim removed; per-run outputs under
  `research/production/` untracked.
- 2026-09-19, `dd9125c` - the paper and benchmark work on `v2`, which had `master` merged into
  it first. `master` has since been fast-forwarded onto it, so the two are identical.

## What this means in practice

There is nothing left on `v2` that is not on `master`. Continuing on `v2` only re-creates the
divergence that cost a day of merging. New work, new branches and pull requests should start
from `master`.

To move over:

```bash
git fetch origin
git checkout master
git pull
```

If you have uncommitted work on `v2`, commit or stash it first: the hygiene pass above
rewrites files you may still have in their earlier form.

`v2` is left in place rather than deleted, so nobody loses a reference they still have
checked out. It should be treated as archived.
