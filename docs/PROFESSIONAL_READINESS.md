# Engineering acceptance record

Date: 2026-09-14. This document tracks implemented work and observed verification, not a
claim of PyOD-equivalent maturity or external adoption. The package remains Alpha.

## Implemented

- Shared preflight for suitability selection and execution, including constant-asset,
  HRP linkage, classification, signal-window and parameter checks.
- Research workflow with task-specific temporal validation, frozen holdout selection,
  input fingerprints, supplied provenance, existing trial ledger and audit integration.
- Fitted forecast/supervised models, repeatable prediction and trusted version-checked persistence.
- Execution adapters for previously catalog-only methods, with bounded configurations
  documented in the catalog and workflow guide. Optional imports remain lazy.
- Frozen ECB reference-rate and seeded synthetic forecast benchmark with naive comparison.
- Cross-platform CI definitions, minimum-dependency and optional-backend jobs, coverage,
  slow tests, wheel/sdist installation checks and release-candidate artifact generation.
- Workflow tutorial, compatibility/migration policy, adapter contribution criteria and
  a feedback template for independent users.

## Verification

The first core workflow pass completed with 68 passed and 4 absent-optimizer skips.
The expanded algorithm/tools pass completed with 162 passed and 6 optimizer skips in that
environment. A separate optimizer/MCP/bridge environment completed with 112 passed and
9 unrelated optional-package skips. All 47 slow tests passed. Exact installed versions are
recorded in [backend environments](../benchmarks/BACKEND_ENVIRONMENTS.json).

The frozen forecast benchmark completed with 21 cases: 8 wins, 12 ties and 1 loss against
naive. Additional task panels completed with 18 cases: 7 wins, 8 ties and 3 losses against
their named baselines. The single-candidate signal panel tests execution, not selection.
See [full case results](../benchmarks/ALGORITHM_RESULTS.json) and
[benchmark methods and limits](../benchmarks/ALGORITHMS.md).

The guard robustness benchmark ran seeds 17, 29 and 43. Each detected all 12 injected defect
types with at least one guard, with no false alarms or guard errors. Individual guard misses
remain in [the results](../benchmarks/GUARD_ROBUSTNESS.json); this is scoped synthetic evidence.

Wheel and sdist both passed strict metadata checks and installed into separate clean
environments outside the checkout; skills, research JSON and model roundtrips passed there.
The retained local build is under `dist/professional-review-20260914/`. It uses the existing
Alpha version and is a review build, not a new public release.

The initial full run found a stale test that demanded an import crash in every environment.
Reverification reproduced the crash on numpy 2.2.6/sklearn 1.4.2/osqp 1.1.3 and successful
imports in both orders on numpy 2.4.6/sklearn 1.9.1/osqp 1.1.3. The skill and test now scope
that claim to the measured environments. Final full regression/script results follow below.

- Final default regression: **2935 passed, 49 skipped, 47 slow tests deselected**, 3 warnings.
  The skips are enumerated by pytest; separate environments cover optimizer and MCP paths.
- Separate slow suite: **47 passed**.
- Default-console script check: **111/111 scripts run clean** with numeric thread counts
  fixed to one. The earlier unrestricted-thread run was stopped and replaced by this run.
- Parent-process line coverage: **82%** (36495 statements, 6512 missed). Isolated backend
  subprocess checks are additional evidence; this percentage does not include their execution.
- The original full-size defect benchmark also passed `--check`: **12/12** defect types
  caught by at least one guard, **0** false alarms and **0** uncovered defects.
- Documentation build: **246 pages** generated successfully.
- Both workflow YAML files parse; package/runtime version agreement passes.
- Ordered index/package generation and validation: **127 skills valid**, with the existing
  library-plugin discovery-budget warning. `git diff --check` reports no whitespace errors.

## External acceptance still required

- Push the reviewed change and observe the hosted OS/Python CI matrix. Local Windows
  verification does not establish Linux/macOS success. `gh auth status` on 2026-09-14
  reported `gist`, `read:org`, `repo` scopes and no `workflow` scope. To grant the missing
  GitHub authorization, run `gh auth refresh -h github.com -s workflow` and complete its
  browser authorization before pushing workflow changes. No credential refresh was initiated.
- Choose a release version and publish tested artifacts using configured PyPI credentials
  or trusted publishing. This work does not claim a public PyPI release.
- Obtain independent user reproductions and maintenance history. No real user feedback
  or adoption is manufactured by the internal acceptance tests.
- Expand beyond the current FX and synthetic panels to independently specified datasets
  and task-specific holdouts. Current panels are not evidence for every adapter or asset class.
