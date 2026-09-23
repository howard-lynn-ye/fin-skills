# Financial decision loop: fixed-seed replication results

Verified on 2026-09-23 at approximately 05:10 UTC. All four dynamically allocated
jobs completed with exit 0 and all worker commands returned 0. The complete audit
passed for **1,408/1,408** planned episodes: 704 acquisition and 704 later episodes.
Together with the separate 704-episode seed-11 pilot, the bounded decision-loop
campaign has 2,112 audited episodes. This count is not a count of independent tasks.

The protocol remains [DECISION_REPLICATION_PROTOCOL_20260923.md](DECISION_REPLICATION_PROTOCOL_20260923.md).
Each model/seed runs eleven conditions in one GPU allocation with fresh per-condition
memory and the original parser, scoring, task inputs and response budgets. These are
four authored synthetic mechanisms and eight contracts, repeated across inference seeds.
There is no independently authored holdout or real-time transfer evaluation here.

Remote root: `/beacon-projects/radfm/wy891/fin-skills-campaign-decision-replication-20260923-v1`.
The immutable `decision-verified-summary.json` has SHA-256
`dc8d507c918a382d0c02830638193e1a0a68b28684e877a3f84eea1865e1f7e6`.
Auditor `analysis/audit_beacon_decision_v3.py` has SHA-256
`8a9294ba40ca641c62ec6e65a1106978997dd19e1a5d84d8e6e426a9d3e55566`.
It checked source hashes, accepted submissions and completions, input identity, every
episode, numerical tool outputs, exact original grade booleans, memory availability and
state continuity, response budgets, context selections and saved row summaries. The
documented recomputation-only numerical tolerance remains unchanged. No inference reran.

## Correct completion by seed

Every cell is acquisition / later correct counts; **each count has denominator 16**.
Success requires method, parameters, numerical answer and receipt use together.
Malformed output, missing parameters and incorrect decisions remain in the denominator.

| Interface / memory / context | Qwen 23 | Qwen 37 | Mistral 23 | Mistral 37 |
|---|---:|---:|---:|---:|
| Generic / none / full | 4 / 4 | 3 / 3 | 11 / 13 | 12 / 12 |
| Domain guidance / none / full | 0 / 1 | 0 / 1 | 13 / 14 | 12 / 15 |
| FinSkills / none / full | 2 / 1 | 1 / 1 | 14 / 13 | 15 / 12 |
| FinSkills / frozen / full | 0 / 1 | 0 / 0 | 15 / 16 | 16 / 16 |
| FinSkills / reflection / full | 0 / 1 | 0 / 2 | 14 / 14 | 15 / 14 |
| FinSkills / retrieval / full | 0 / 0 | 0 / 1 | 15 / 16 | 16 / 16 |
| FinSkills / fly / full | 0 / 2 | 0 / 0 | 15 / 16 | 16 / 16 |
| FinSkills / none / recency | 6 / 8 | 8 / 6 | 14 / 14 | 15 / 12 |
| FinSkills / fly / recency | 2 / 1 | 5 / 1 | 15 / 16 | 16 / 16 |
| FinSkills / none / HiSTrim | 5 / 6 | 4 / 4 | 15 / 10 | 14 / 10 |
| FinSkills / fly / HiSTrim | 2 / 1 | 2 / 2 | 16 / 13 | 15 / 12 |

The replication does not establish a broad FinSkills advantage over the generic or
guidance interfaces. On Mistral, frozen, retrieval and fly memory again all achieve
16/16 later, versus 13/16 and 12/16 without memory. This supports a descriptive benefit
of supplying experience in this task setup, but does not distinguish fly's online update
from simpler memory. Qwen does not show a consistent memory benefit and has substantial
strict-JSON compliance failures. Recorded experience reads and subsequent choices show
that the loop executes; they do not by themselves prove the model's causal reasoning.

Without fly, HiSTrim is below recency on later correctness in both seeds and models.
With fly, Qwen HiSTrim ties recency in seed 23 and exceeds it by one case in seed 37;
Mistral HiSTrim is below recency in both seeds. There is no stable joint quality benefit.
The seed-11 results remain separate in [DECISION_LOOP_RESULTS_20260923.md](DECISION_LOOP_RESULTS_20260923.md).
No best seed was selected, and correlated repetitions were not treated as independent
samples for a confidence interval. Per-mechanism counts, component scores, transitions,
errors and per-condition costs are retained in the verified remote summary.

## Actual cost and failures

| Model / seed | Job | Responses | Tool receipts | Input tokens | Output tokens | Episode wall seconds | Worker seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen / 23 | 1652208 | 1,056 | 99 | 1,042,981 | 97,502 | 4,575.097 | 4,732.411 |
| Qwen / 37 | 1652210 | 1,056 | 93 | 1,040,950 | 95,855 | 4,496.847 | 4,639.085 |
| Mistral / 23 | 1652209 | 1,056 | 324 | 1,145,954 | 97,809 | 3,859.326 | 3,999.802 |
| Mistral / 37 | 1652211 | 1,056 | 321 | 1,138,860 | 94,981 | 3,446.916 | 3,511.178 |

There are zero missing model-call records. Worker times include qualification/loading;
their sum is not parallel calendar duration. Router calibration took 0.563/0.593 seconds
for Qwen 23/37 and 0.514/0.589 seconds for Mistral 23/37. Different models/tokenizers
are not an equal-compute ranking. Equal response caps do not equal actual computation.

Recorded error events include 498 selection/execution first-character JSON errors,
608 final-answer first-character JSON errors, 14 incompatible data-field errors,
59 missing/extra required-parameter errors, one extra-data JSON error and four delimiter
errors. These are events, not mutually exclusive episode counts. Original responses,
strict parser and missing parameter failures are preserved. None justifies an
environment-failure retry, stripping Markdown fences, filling parameters or extra turns.

## Context cost and retained experience

`decision-context-description.json` SHA-256:
`b49c1cd2e18f6ef1412009395ef724b5261487352e04007a90e645c9b9ed189a`.
The descriptor verifies each episode hash against the audit before aggregation. The
following means cover 32 later selection/final calls per condition, excluding reflection
and calibration. Columns within each row are full / recency / HiSTrim. Each comparison
within model/seed shares one dynamically allocated GPU.

| Model / seed / memory | Mean KV MiB | Peak allocated GiB | Mean generation seconds |
|---|---|---|---|
| Qwen / 23 / none | 231.551 / 199.547 / 186.527 | 27.957 / 27.884 / 27.865 | 4.636 / 4.908 / 4.726 |
| Qwen / 23 / fly | 332.063 / 251.520 / 214.297 | 28.180 / 27.945 / 27.914 | 4.291 / 3.841 / 4.580 |
| Qwen / 37 / none | 230.303 / 195.967 / 183.926 | 27.959 / 27.887 / 27.866 | 4.517 / 4.447 / 4.500 |
| Qwen / 37 / fly | 327.451 / 253.158 / 216.281 | 28.113 / 27.990 / 27.917 | 4.294 / 3.723 / 4.225 |
| Mistral / 23 / none | 206.987 / 175.616 / 165.403 | 23.235 / 23.167 / 23.154 | 4.423 / 4.073 / 3.984 |
| Mistral / 23 / fly | 309.165 / 235.130 / 202.873 | 23.456 / 23.243 / 23.208 | 4.706 / 4.574 / 4.102 |
| Mistral / 37 / none | 206.792 / 174.508 / 165.608 | 23.236 / 23.165 / 23.156 | 3.696 / 3.734 / 3.783 |
| Mistral / 37 / fly | 305.171 / 231.871 / 199.054 | 23.459 / 23.246 / 23.203 | 4.380 / 4.143 / 3.606 |

KV bytes and allocated-memory peaks fall, but total generation latency does not improve
uniformly. Output lengths differ, and allocator reserved peaks depend on earlier arms.
The descriptor records prompt/output lengths, prefill latency and reserved peaks for
every row; generation latency includes the implemented router/gather work. This is not
a fixed-output-length throughput benchmark or optimized multi-request VarLen kernel.

Final-depth retained/offered fly experience item occurrences are:

| Model / seed | Full | Recency | HiSTrim |
|---|---:|---:|---:|
| Qwen / 23 | 80/80 | 14/80 | 0/80 |
| Qwen / 37 | 80/80 | 14/80 | 0/80 |
| Mistral / 23 | 80/80 | 2/80 | 2/80 |
| Mistral / 37 | 80/80 | 4/80 | 0/80 |

HiSTrim final-depth historical token means are zero for both Qwen seeds with and without
fly; Mistral has 21 without fly in each seed and 32.625/21 with fly in seeds 23/37.
The current calibrated gates and whole-item selection underfill the shared token caps.
Actual retained tokens are not matched between policies. This limits the interpretation
of memory interaction and quality, and is not corrected by tuning thresholds on these
scores. Deleted final-depth history may already have influenced protected shallow-layer
states, so these counts do not establish that history had no effect.

## Bounded campaign closure and remaining evidence

The prespecified seeds 11, 23 and 37 are complete and audited. No more seeds are added to
seek a favorable result. This closes the current tool/memory/context development matrix,
not the research plan or evidence required for a conference paper. Jev still requires
its real API key. Independently authored tasks, real-period transfer, independent answer
and citation-support labels, and human-efficiency evaluation remain absent. The broader
ready-made component/framework comparison remains a separate uncompleted comparison;
these numerical-interface arms must not be described as completing that benchmark.
No manuscript scores, user edits or synchronized Overleaf files were overwritten.
