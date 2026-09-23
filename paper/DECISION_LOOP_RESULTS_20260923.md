# Financial decision-loop pilot: complete seed-11 results

All six dynamically allocated Beacon jobs completed. The audit verified **704/704**
planned episodes, including 352 acquisition and 352 later-evaluation episodes. Each model
has 352 episodes across eleven conditions, with 16 acquisition and 16 later episodes per
condition. The audit found no missing model-call records. This is a development pilot
with four authored synthetic mechanisms, eight client contracts and one inference seed.
Later synthetic blocks are not independently authored unseen tasks or real-market transfer.

Remote campaign: `/beacon-projects/radfm/wy891/fin-skills-campaign-decision-loop-20260923-v1`.
`decision-verified-summary.json` SHA-256:
`45cc991907197086e1e16814e45915ac90ce47feb876b72b9519549bc2dd5bf8`.
The successful auditor is `analysis/audit_beacon_decision_v2.py`, SHA-256
`fce4da5d6ec5fa0adbfe01c4222798fd2c09e8763929fac16a4badecda9b3852`.

The audit checked source hashes, submission/completion receipts, every planned episode,
paired numerical inputs, actual tool outputs, the original scoring predicates, memory
eligibility/state continuity, response budgets, retained positions and saved summaries.
The first auditor incorrectly required recomputed floating-point references to match
bit for bit. Thirty EWMA reference values differed by at most 1.1102230246251565e-16.
Version 2 allows rtol=1e-12/atol=1e-14 for recomputed numerical values only; all original
grade booleans still match exactly. Original receipt hashes and result files are unchanged.
No parser, score threshold, labels or model output was changed, and no inference reran.

## Correct completion

Each entry below is correct / 16 planned episodes. Correctness requires the right method,
parameters, numerical values and cited tool receipt together. Invalid JSON, omitted
parameters and wrong methods stay in the denominator.

| Interface / memory / context | Qwen acquisition | Qwen later | Mistral acquisition | Mistral later |
|---|---:|---:|---:|---:|
| Generic / none / full | 5/16 | 3/16 | 11/16 | 12/16 |
| Domain guidance / none / full | 1/16 | 1/16 | 12/16 | 14/16 |
| FinSkills / none / full | 5/16 | 3/16 | 14/16 | 12/16 |
| FinSkills / frozen / full | 3/16 | 2/16 | 16/16 | 16/16 |
| FinSkills / reflection / full | 5/16 | 2/16 | 15/16 | 14/16 |
| FinSkills / retrieval / full | 3/16 | 1/16 | 16/16 | 16/16 |
| FinSkills / fly / full | 3/16 | 2/16 | 16/16 | 16/16 |
| FinSkills / none / recency | 10/16 | 6/16 | 14/16 | 13/16 |
| FinSkills / fly / recency | 7/16 | 4/16 | 15/16 | 16/16 |
| FinSkills / none / HiSTrim | 8/16 | 5/16 | 14/16 | 10/16 |
| FinSkills / fly / HiSTrim | 7/16 | 2/16 | 15/16 | 13/16 |

FinSkills' named, organized interface does not consistently beat generic tools or
domain guidance in this pilot. Mistral benefits descriptively from several memory
conditions relative to no memory, but frozen, retrieval and fly all reach the same
16/16 later score. This does not identify a benefit of online fly updates over simpler
memory. Qwen shows no consistent memory benefit. HiSTrim does not beat the corresponding
recency condition for either model in later correctness. No preferred arm was selected
for an additional favorable-only analysis.

Method selection alone differs from complete success. For Mistral full-context generic,
guidance and FinSkills, later method-correct counts are 15/16, 15/16 and 16/16, while
complete-correct counts are 12/16, 14/16 and 12/16. The per-condition, per-phase summary
retains all method, parameter, numerical and receipt-use components and per-family counts.
Response wording and choice transitions are traces, not proof of the model's reasoning.

## Failure and cost accounting

Qwen has 88 correct episodes out of 352 across all conditions; Mistral has 310/352.
These pooled counts are inventory checks, not an equal-compute model leaderboard.
Across both models, logged error events include 225 selection JSON parse failures,
261 final JSON failures at the first character, two other final delimiter failures,
one final extra-data failure, 29 missing-parameter errors and four wrong-method/input
errors. Error events can overlap within an episode and must not be summed as failed
episodes. Inspected Qwen failures include Markdown fences forbidden by the declared
interface. The results partly measure interface compliance, limiting conclusions about
financial competence. The original parser is retained.

| Model | Completed / planned | Returned model calls | Actual numerical tool receipts | Input tokens | Output tokens | Episode wall seconds |
|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-Coder-14B | 352/352 | 1,056 | 123 | 1,051,453 | 97,228 | 4,427.12 |
| Mistral-Nemo-12B | 352/352 | 1,056 | 323 | 1,142,205 | 97,547 | 3,736.14 |

Every condition generated 96 responses for its 32 episodes; output lengths and actual
tool use still differ. No missing-call usage was imputed. Episode wall includes model
and tool work; do not add those components to it. Summed worker wall time is 8,700.22
seconds, including setup and qualification, and is not parallel calendar duration.
Recorded router calibration across the six jobs totals 3.186 seconds. Per-condition
actual costs and job-level evidence are preserved in the verified summary.

## Context cost and retained experience

The next table covers only the two action/answer calls for each later episode: 32 calls
per row. Reflection and calibration are excluded. KV is mean physical prefill KV in MiB;
GPU allocation is peak allocated GiB. Latency is mean complete generation time per call.
Different jobs could receive different hardware and generate different output lengths,
so this pilot table does not establish a controlled end-to-end speedup. Reserved-memory
peaks, prefill time, prompt/output counts and raw selection traces remain in the audit.

| Model | Memory | Context | KV MiB | Peak GPU GiB | Seconds/call | Final-depth experience occurrences retained / offered |
|---|---|---|---:|---:|---:|---:|
| Qwen | None | Full | 231.61 | 27.969 | 4.192 | 0/0 |
| Qwen | None | Recency | 194.13 | 27.881 | 4.431 | 0/0 |
| Qwen | None | HiSTrim | 183.66 | 27.865 | 4.152 | 0/0 |
| Qwen | Fly | Full | 332.07 | 28.162 | 3.851 | 80/80 |
| Qwen | Fly | Recency | 256.86 | 27.976 | 4.141 | 10/80 |
| Qwen | Fly | HiSTrim | 216.97 | 27.911 | 4.376 | 0/80 |
| Mistral | None | Full | 205.90 | 23.237 | 4.145 | 0/0 |
| Mistral | None | Recency | 175.54 | 23.165 | 4.132 | 0/0 |
| Mistral | None | HiSTrim | 165.39 | 23.155 | 4.027 | 0/0 |
| Mistral | Fly | Full | 307.92 | 23.454 | 4.419 | 80/80 |
| Mistral | Fly | Recency | 232.20 | 23.254 | 4.507 | 4/80 |
| Mistral | Fly | HiSTrim | 203.66 | 23.201 | 4.329 | 2/80 |

The shared history cap does not mean equal retained history. Qwen HiSTrim retains zero
explicit history tokens at the last pruning stage in these later calls, with or without
fly. Mistral HiSTrim retains means of 21.00 and 32.31 history tokens respectively. Whole
items and learned thresholds can leave budgets mostly empty. This is a substantive
limitation of this new implementation, not evidence that memory is intrinsically useless.
Tokens removed at later layers may already have influenced protected representations;
zero final-depth items does not mean that history had no computational influence.

The separately hash-checked descriptive artifact is `decision-context-description.json`,
SHA-256 `e5fce55c1a0c773ea3a01c5493f5fa9027004006eef351f95dd8312c2429f378`.
Item occurrences across calls are not independent observations or content-support labels.
Smaller KV and modest allocated-memory changes do not imply better task quality or
uniformly faster generation. Do not retune the router on these results and call that
independent confirmation.

## Bounded follow-up

The previously identified seed and cost-comparison limitations are addressed prospectively
in [DECISION_REPLICATION_PROTOCOL_20260923.md](DECISION_REPLICATION_PROTOCOL_20260923.md):
seeds 23 and 37, unchanged tasks/parser/scorer, all eleven conditions within one dynamic
GPU allocation per model/seed. Four jobs were accepted as 1652208–1652211. This adds
1,408 planned episodes, not new independent tasks. Stop after these fixed seeds and audit
them; do not extend seeds until a desired result appears. Independent tasks/time transfer,
human efficiency and independently labeled evidence support remain unresolved.
