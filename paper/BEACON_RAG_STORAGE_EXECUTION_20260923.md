# Additional Beacon experiments: execution and evidence

This record distinguishes accepted jobs, actual execution and verified completed results.
All remote roots below are direct children of `/beacon-projects/radfm/wy891`, prefixed
`fin-skills-campaign-`. Source snapshots, temporary files, model caches and Slurm stdout /
stderr remain in RADFM. Existing task outputs and negative results are preserved.

## Report-QA workflow ablation

Protocol: [FINQA_RAG_ABLATION_PROTOCOL_20260923.md](FINQA_RAG_ABLATION_PROTOCOL_20260923.md).
Implementation: `benchmarks/agent_study/finqa_rag_ablation.py`.

The actual BM25S / LlamaIndex / FinSkills CPU qualification **1654259 completed 0:0**.
It exercised the official calculator, all model-facing input conditions and public
RAGPipeline citation checks on scripted software fixtures. This is not model-quality
evidence. Dependencies are isolated in this run's RADFM `deps/` directory and hashed;
production verifies both dependency and study-source agreement with qualification.

- Qwen **1654284**, `finqa-rag-qwen-20260923-v1`: running; actual model episodes written.
- Mistral **1654286**, `finqa-rag-mistral-20260923-v1`: running; actual model episodes written.

Each job plans 128 generated episodes on the same 32 public questions, plus 32 paired gate
outcomes without additional generation. The full-report condition is a context-coverage
comparator; it does not mean full original annual reports. FinQA supplies extracted report
text/tables. Reusing the memory study's questions does not create independent extra tasks.
The experiment tests report-QA workflow components, not the whole library or human usability.

## BGE and local Kev downstream transfer

Protocol: [FINQA_RERANK_TRANSFER_PROTOCOL_20260923.md](FINQA_RERANK_TRANSFER_PROTOCOL_20260923.md).
Implementation: `benchmarks/agent_study/finqa_rerank_transfer.py`.

- Ranking **1654297**, `finqa-rerank-rank-20260923-v1`: accepted, queued behind the account's
  active-job limit. Same frozen BM25 top ten candidates, pinned BGE and local Kev-4B.
- Combined answer execution **1654323**, `finqa-rerank-answers-20260923-v1`: accepted with
  `afterok:1654297`. Runs Qwen then Mistral in separate Python processes in one allocation,
  retaining each model's original source and output root. Each model plans 64 answers.

The first Qwen dependent job **1654299** was accepted but cancelled while still PENDING,
before episodes or completion existed. The Mistral submission was explicitly rejected
with `AssocMaxSubmitJobLimit`, without a job ID. These receipts are retained. Combining
the two unstarted jobs avoids another queued slot; it changes scheduling only. The wrapper
checks each original manifest, records its shared Slurm ID and runs both even if one fails.
No source, model, prompt, task, scorer or budget was changed after examining model outcomes.

Original per-model output roots: `finqa-rerank-qwen-20260923-v1` and
`finqa-rerank-mistral-20260923-v1`. The replacement wrapper writes its own completion too.
Ranking and answer generation are distinct costs. No inference result is yet claimed here.

## Completed storage reliability experiment

Job **1654281 completed 0:0**, root `storage-recovery-20260923-v1`.
Implementation: `benchmarks/library_workflows/storage_recovery.py`.
Evidence: `benchmarks/library_workflows/evidence/20260923-storage/`.

The parent killed real child processes with SIGKILL at four exact barriers, three repeats
each: before save, during the batch, immediately before COMMIT, and after COMMIT returned.
All **12/12** cases preserved the expected atomic state across events, latest pointers,
alert outbox and collector cursor; SQLite integrity checks passed. A separate read-only
SQLite replay verified all 12 saved databases and the source manifest. These are synthetic
engineering fixtures, not independent financial questions or simulated machine power loss.

| Inserted events | Ingest seconds | Reopen seconds | Median first-100 latest query, seconds |
|---|---:|---:|---:|
| 100 | 0.153941 | 0.045794 | 0.003889 |
| 1,000 | 0.404466 | 0.056572 | 0.004535 |
| 10,000 | 3.035598 | 0.058106 | 0.008668 |

Every scale retained the exact event/latest/outbox counts and final cursor. Query timing
uses one warmup and seven repetitions within this allocation. It is a descriptive snapshot,
not evidence of superiority over native SQL or a broad latency guarantee.

Duplicate/out-of-order behavior matters: sequential arrivals A, A, B, B, older A inserted
1, 0, 1, 0, 1 records. Adjacent unchanged content is deduplicated, but an older revision
arriving last becomes `latest`. This is arrival ordering, **not publication-time ordering**.
The result does not establish safe historical-version selection; callers need a separate
chronological/as-of policy. The original records and provenance remain available.

## Audit and publication boundary

Six local boundary tests passed before deployment. Mandatory index/package regeneration
and repository validation are required before committing. Qualification and Slurm receipts
are preserved under `benchmarks/agent_study/evidence/20260923-rag/` with file hashes.

`scripts/audit_finqa_rag.py` is staged under each answer root's `analysis/` directory for
execution only after completion. It verifies all planned outputs, source/receipt hashes,
official test-file identity, real tool results, original grades and paired gate decisions.
Do not report successful task quality from job submission or partial progress. The existing
FinQA memory and FinanceBench/BGE jobs remain separate; they were not redeployed.

Still distinct from these experiments: independent pipeline-building tasks, real historical
publication/revision timestamps with supported answers, semantic citation entailment, and
human development-time measurements. No result here fills those gaps by relabeling fixtures.


## Status update: 2026-09-23 12:55 UTC

The dated running/queued entries above are superseded by this update. Both report-QA
jobs completed: Qwen 1654284 in 00:45:39 and Mistral 1654286 in 01:46:56. Each retained
all 128 generation episodes. The separate read-only auditor verified every original
numerical/program grade, calculator receipt, output hash and paired gate decision.
Evidence is in `benchmarks/agent_study/evidence/20260923-rag-completion/`.

| Reported condition | Qwen correct / 32 | Qwen incorrect accepted / 32 | Mistral correct / 32 |
|---|---:|---:|---:|
| components | 3 | 5 | 0 |
| skills | 2 | 1 | 0 |
| rag_api | 2 | 0 | 0 |
| rag_gate (same frozen answers) | 2 | 0 | 0 |
| full_report | 1 | 5 | 0 |

Mistral accepted one incorrect component answer and none in its other conditions.
Format failures were frequent. These results do not establish an overall library
accuracy advantage; fewer wrong acceptances must be considered alongside rejection and
low correct completion. The gate added no outcome change to these frozen API answers.

Ranking job 1654297 completed in 00:01:14. All 32 BGE and 32 Kev ranking records exist,
with zero recorded per-row ranking errors. This is not a downstream-answer score.
Combined answer job 1654323 failed before either model started: the parent bound hashes
of locally serialized CRLF manifests while the remote dispatcher had written LF bytes.
The JSON objects and all scientific-source hashes are identical. No model episodes or
shared-allocation markers existed in either child root before repair.

A v2 wrapper bound the exact previously archived remote manifest bytes, but its submission
was rejected with `Job dependency problem` for the old completed ranking job. That rejection
is retained without a job ID. The v3 wrapper removes only the obsolete scheduler dependency;
ranking completion and every ranking receipt are still checked by the child workers.
No scientific code, model, input, prompt, scorer or inference budget changed.

Replacement **1660691** was accepted at 12:54:22 UTC and is **PENDING (AssocMaxJobsLimit)**
at this check. Wrapper root: `finqa-rerank-answers-20260923-v3`; the two original per-model
answer roots remain. There are still 64 planned Qwen and 64 planned Mistral answers.
The completed QA timings suggest roughly 1–1.5 hours of serial computation for these
half-sized answer batches; this is an estimate, excludes queue time and assumes no new
execution failure. The experiment is not yet fully complete.

The existing FinQA memory jobs 1653915/1653932 and FinanceBench retrieval/reranking jobs
1653482/1653602 also report COMPLETED 0:0. Memory completion auditors already exist and
were not rerun. Their task units overlap this study and must not be counted again.
