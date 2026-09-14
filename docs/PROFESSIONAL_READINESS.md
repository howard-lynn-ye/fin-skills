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

## Local verification before hosted CI

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

## Hosted verification follow-up (2026-09-14)

The implementation was pushed as `2be0474` after the owner completed GitHub workflow
authorization. The [first hosted matrix](https://github.com/howard-lynn-ye/fin-skills/actions/runs/34892049735)
finished with 17 successful jobs and 4 failures. It exposed unsupported minimum dependencies,
Python 3.10 TOML imports, and macOS-sensitive degenerate regressions. Commit `d431bc7` fixes
these issues without dropping the failing checks, and adds regression cases at multiple scales.

The [complete rerun](https://github.com/howard-lynn-ye/fin-skills/actions/runs/34893229959)
tested `d431bc7e80fb9e0bb6bb7eee2a94999558e6db96` and finished **21/21 jobs successful**:

- Linux and Windows full regression on Python 3.10, 3.11, 3.12 and 3.13; macOS on 3.12.
- Exact minimum base dependencies on Python 3.10; all four optional extras on Linux/Windows.
- Clean wheel/sdist installation on Linux/Windows, slow tests, every standalone script,
  the strict defect benchmark, multi-seed guard benchmark and algorithm-selection benchmark.

For a concrete full-suite result, Windows/Python 3.12 reported **2918 passed, 72 skipped,
47 slow tests deselected**, with 18 warnings. Optional extras have separate jobs; the skip
counts must not be interpreted as backend parity coverage. The local minimum-dependency
environment on Python 3.11 reported **2887 passed, 103 skipped, 47 deselected**. Focused
numerical/bridge checks reported **74 passed** in the broader local environment and
**68 passed, 6 absent-statsmodels skips** at the minimum dependency versions.

These checks validate the implemented scope. They do not establish PyOD-level adoption,
algorithm-selection superiority on arbitrary data, or a public package release.

## External acceptance still required

- Integration into `master`: the reviewed implementation and fixes are on
  `codex/complete-library-capabilities`; no merge is claimed by this record.
- Choose a release version and publish tested artifacts using configured PyPI credentials
  or trusted publishing. This work does not claim a public PyPI release.
- Obtain independent user reproductions and maintenance history. No real user feedback
  or adoption is manufactured by the internal acceptance tests.
- Expand beyond the current FX and synthetic panels to independently specified datasets
  and task-specific holdouts. Current panels are not evidence for every adapter or asset class.
