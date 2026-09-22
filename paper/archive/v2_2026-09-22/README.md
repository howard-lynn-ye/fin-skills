# Archived v2 materials — not current research evidence

Source: `a2644375ad9030cca0518af56dda02c334bc7788`, inspected on 2026-09-22.
These materials are preserved for traceability. Their inclusion in the repository does not
validate their empirical claims or make the archived manuscript submission-ready.

[v2-divergent-materials.zip](v2-divergent-materials.zip) contains the exact 53 surviving files
changed on `v2` since the common base `dd9125c2a99a10055f1668ab8a1747b5a8d80807`.
[manifest.json](manifest.json) records their hashes and sizes, plus the one deleted path.
Each archived file was checked against its source Git blob. The archive SHA-256 is
`9e0fe402b372b9f86d53394d0dec1ca41ca8477db28cfbbad6fb28cca8d4430c`.
The Git history also retains the complete original source tree.

## Subsequent v2 increment

During consolidation, `v2` advanced to `3b64a02eb52e8743213d8e3735ec11cf9bb33da6`.
[v2-increment-3b64a02.zip](v2-increment-3b64a02.zip) preserves all 21 files changed
since `a264437`, with no deletions. Each file was checked against the source Git blob.
[increment-3b64a02.json](increment-3b64a02.json) records the source commit, comparison
base, file hashes and sizes. The archive SHA-256 is
`94d4641ddd2e69a71bd4dfe2e5ae811ee5895de26668c45256b8e27388a64c11`.
These are incremental source files to overlay on the earlier snapshot when inspecting
that history, not replacements for active study files or independently verified results.

The increment's `benchmarks/run_extended_ablations_e9_e12.py` directly assigns the
feedback outcomes (line 47 onward), regime/rolling-prior metrics (line 239 onward),
component-ablation metrics (line 304 onward), and delivery tokens/latencies/outcomes
(line 388 onward). Its `main` at lines 459–486 serializes those tables into result JSONs.
This is not evidence of model inference, measured self-repair, market replay, or a timed
RAG comparison. The E9–E12 tables and their downstream figures do not close those gaps.

## Why these files do not replace the active study

In the archived `benchmarks/agent_study/run_finguard_bench_60.py`, lines 169–264 return
literal model-tier metrics. Lines 294–301 assign condition outcomes by task number, and
lines 325–330 construct reporting gaps from fixed multipliers. Running real guards on
known defective and clean fixtures does not turn those assigned outcomes into observed
LLM performance or self-repair trajectories.

The archived `benchmarks/real_world_kol_audit.py` synthesizes missing US profiles from
specified distributions and contains fixed IC, holding-horizon and cost assumptions.
The resulting “2,521 real bilingual KOL” and predictive-efficacy claims need original,
timestamped observations and independent recomputation.

FinGuardBench-180 does execute guard calls, but some proposed boundary and compound
cases copy existing fixtures. Its saved results are development regression evidence.
The additional parity rows include formula or identity checks rather than independent
comparisons against all corresponding library implementations. Timing records only
support the particular functions and sizes actually measured.

The archive is intentionally excluded from the active paper evidence generator and normal
test discovery. Do not run its scripts in place over current result files or use its
PDFs, figures, tables or significance tests as verified model-study evidence.

Current sources are [STUDY_PLAN.md](../../STUDY_PLAN.md),
[OUTLINE.md](../../OUTLINE.md), [evidence.json](../../evidence.json), and the
[remaining-experiments record](../../EXPERIMENTS_NEXT_ZH.md).
