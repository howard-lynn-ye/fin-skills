# Branches: `master` is the one line of work

**`v2` has been merged into `master`. Please develop on `master`.**

## 2026-09-22 consolidation

The GitHub default branch is named `master`; references to merging into the main branch
mean this branch. The current workspace changes and the updated `origin/master` work are
being consolidated on `codex/consolidate-project-2026-09-22`.

After the 2026-09-19 synchronization, `v2` diverged again through `a264437`. Its later
paper and benchmark materials contain assigned model outcomes and synthetic profiles
presented as measured results. Preserve that history and its exact divergent files in
[the v2 archive](../paper/archive/v2_2026-09-22/README.md); do not replace the active
study, manuscript or benchmark outputs with those claims. The consolidation retains
the `v2` ancestry using an explicit history merge while keeping the reviewed working
tree and the source archive. This is archival integration, not empirical validation.

The active research protocol is [STUDY_PLAN.md](../paper/STUDY_PLAN.md). Remaining
experiments are recorded in [EXPERIMENTS_NEXT_ZH.md](../paper/EXPERIMENTS_NEXT_ZH.md).

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

The two branches were synchronized on 2026-09-19; the later divergence and its archival
integration are described above. New work, new branches and pull requests should start
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
