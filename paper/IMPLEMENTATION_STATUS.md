# Implementation and evidence status

Updated 2026-09-21. This is an execution record, not a claim of publication readiness.

## Implemented

- Empty or rejected audits cannot pass. Explicit required-guard policies distinguish
  detected failures from incomplete coverage. Either/or input requirements are visible
  before running a guard. The research interface retains `partial` for an empty audit.
- The agent runner now executes real checks on submission snapshots, binds receipts to
  artifact hashes, distinguishes stale and unexecuted citations, and retains failures.
- Four experimental conditions have equal turn/token caps and ordinary accounting
  feedback. The independent oracle runs after submission freezing; its grades are not
  available for repairs. Rejected submissions are graded when executable.
- Beacon workers require Linux filesystem/network/process restrictions. The actual
  host probe denied outside reads, outside writes and sockets while allowing scratch
  writes. In-process evaluator tamper resistance remains unproven.
- Row-level prediction evaluation checks declared availability timestamps and recomputes
  pooled/daily Rank IC with paired date-block intervals. The existing KOL summary fixture
  is explicitly excluded from claims of prediction reproduction.
- The working manuscript, related work and evidence ledger now distinguish historical
  pilot observations, synthetic regression evidence and outstanding experiments.
- Beacon's optional-dependency tests exposed an X-13 wrapper compatibility issue:
  statsmodels 0.14.6 uses `_binary_names`. The adapter now supports this and the older
  `BINARY_NAMES` interface, with tests for both. Generated package code was rebuilt.

## Verification record

| Check | Observed result |
|---|---|
| Index, package generation and repository validator | 129 skills; OK; one existing discovery-budget warning |
| Local targeted suite after contract fixes | 143 passed, 1 optional-dependency skip |
| Local full suite in the isolated minimum environment | 3065 passed, 107 skipped, 47 deselected |
| X-13 module after compatibility fix, with statsmodels installed | 12 passed, 1 slow test deselected |
| Beacon targeted suite with confined workers | 118 passed |
| Beacon repair validation after full-suite findings | 28 passed, 1 deselected; confined worker rerun 7 passed |
| Synthetic defect benchmark rerun | 12/12 planted defect families caught; 0 clean-control false alarms |
| Numerical parity rerun | 8/8 within stated tolerances |
| Handwritten causal positive control on public seed 11 | Mandatory audit PASS; independent Sharpe gap, future/same-session dependence and post-delisting mass all zero |
| Manuscript build and visual review | Four-page PDF compiled; revised pages inspected |

The local full-suite count precedes the isolated X-13 follow-up; that module was retested
with and without statsmodels. Beacon's full-suite failures were resolved by the
targeted reruns recorded below; inference outcomes remain pending. The first Windows
full run exposed two audit-contract interactions; both were corrected and the complete
suite then passed. The general Anaconda environment had an unrelated optional-library
native crash, so the isolated minimum-dependency environment is used for local validation.

## Beacon execution

Workspace: `/beacon-projects/radfm/wy891/fin-skills-audit-20260921`.
All working files and caches are under RADFM, not Beacon HOME.

- Preparation: Slurm `1612983`.
- Full validation: Slurm `1613334`; 3038 passed, 131 skipped, 43 deselected, three
  failures (two missing synced fixtures and the X-13 interface compatibility issue).
- Repair validation: Slurm `1613510`, completed successfully; 28 tests passed across
  the three affected modules, and all seven confined worker tests passed again.
  Other passing modules were unchanged.
- First GPU preflight: Slurm `1613543` (indices 0 and 1, one RTX 6000 Ada each) and `1613544`
  (index 2, two RTX 6000 Ada devices). The pending H200 array `1613230` was cancelled
  before inference because the scheduler estimated a next-day start. Model revisions,
  precision and per-condition budgets are unchanged; hardware is logged per job.
  All three jobs reached RUNNING. They used the account's permitted `medium` QoS and
  64 GiB host-memory limit; the default QoS permits only one GPU and 32 GiB per job.
- Prior validation attempts `1613049` and `1613227` remain in the scheduler history:
  one caught an old test expectation, and the other exposed omitted research Python
  modules in the source-sync allowlist. No model inference ran in those attempts.

The source sync also includes the marketplace manifest, CI metadata and the
repository's empty sample holdings fixture, required by the complete test suite.
Its archive SHA-256 is
`12ccd55f182fd7d57d4c1652615465626e50269eaba21bbdf208c59f8cced4ed`.
Do not overwrite the remote source while these inference jobs are active.

The first preflight was subsequently stopped for a transport-format defect: Markdown
JSON fences were rejected before tool dispatch. Fourteen completed cell records and
all partial/planned cells are preserved under `results-transport-v1`; their original
source is preserved under `source-transport-v1` and matches the frozen protocol hashes.
The revised parser passed 12 local execution/summary tests. Beacon format validation
`1613605` also passed all 11 confined execution tests. Replacement inference jobs
`1613621` (7B/14B) and `1613622` (32B) finished in a new result directory. Their
source archive SHA-256 is
`490a58c3148ba077d1598f9f420cc516ec2e37dbfa1bdab42f0270a558ab669a`.
This is a declared protocol amendment, not a hidden retry of financial failures.

## Observed feasibility outcomes

| Model | Recorded cells | Accepted submissions | Independent numerical grades |
|---|---:|---:|---:|
| Qwen2.5-Coder 7B | 12 | 0 | 0 |
| Qwen2.5-Coder 14B | 12 | 7 | 0 |
| Qwen2.5-Coder 32B, initial two-GPU attempt | 12 infrastructure-failure records | 0 | 0 |

7B ended at the turn limit in eleven cells and at the context limit in one. 14B
accepted seven submissions and reached the turn limit in five. Independent grading
could not obtain valid positions/reports, including CSV date-column errors, malformed
position indices and a missing report. These results do not establish an effect on
financial correctness. In particular, accepted-but-ungradable outputs are not successes.

32B failed before its first tool call with nonfinite generation values and a CUDA
device assertion. Later errors from that process are not independent model failures.
A separate GPU-copy probe passed on its sampled devices; an SDPA/eager comparison did
not identify a reliable fix. No root cause is claimed. Single-H200 recovery job
`1613767` is pending; its output directory is `results-single-gpu/model-2`.
The scheduler currently estimates a next-day start, which is not a guaranteed time.

Raw records were downloaded to `runs/beacon-20260921/evidence/` and summarized in
`benchmarks/agent_study/BEACON_FEASIBILITY_RESULTS.json`. Their frozen code hashes match
the corresponding local runtime source. A prepared remote finalization script exists
at `logs/finalize_results.py`; submitting an additional dependent job hit the account's
submission limit, so automatic final aggregation has not been scheduled.

Each model has 12 planned cells, as produced by `run_matrix.py --plan-only`: four
conditions on three public development seeds. The model family is Qwen2.5-Coder
Instruct at 7B, 14B and 32B, with exact revisions recorded by the preparation script.
This is an operational feasibility study. It does not supply independent holdout
authorship, replication across model families or a confirmatory sample-size calculation.

## Outstanding evidence

The original stock_prediction/KOL row-level data have not been located. Source predictions,
outcomes and genuine availability timestamps are needed; summary metrics cannot substitute
for them. Author details, accountable human source review and final declarations also
remain unresolved. No paper has been submitted and no publication outcome is claimed.

See `STUDY_PLAN.md`, `RELATED_WORK.md`, `evidence.json` and `../docs/BEACON_STUDY.md`.
