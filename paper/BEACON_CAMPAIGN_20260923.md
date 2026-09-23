# Beacon campaign prepared on 2026-09-23 UTC

Status: **Decision-loop pilot and fixed replication complete: 2,112/2,112 episodes audited**.
The first five jobs were submitted at 2026-09-23 00:20 UTC. The documented
password-file connection route succeeded with the pinned host key. Earlier SSH key
authentication failure did not establish a Beacon outage. The current local and remote
submission environments have no `TYPESAFE_API_KEY`, so Jev remains blocked.

## 05:10 UTC: fixed replication complete and audited

Jobs 1652208/09/10/11 all completed with exit 0; each has 352/352 records and every
worker command returned 0. The existing v3 auditor passed all 1,408 planned episodes.
Verified summary SHA-256:
`dc8d507c918a382d0c02830638193e1a0a68b28684e877a3f84eea1865e1f7e6`.
Context descriptor SHA-256:
`b49c1cd2e18f6ef1412009395ef724b5261487352e04007a90e645c9b9ed189a`.
Both artifacts remain in the replication RADFM root. No deployed source, original result,
parser or scoring rule changed. All placement was dynamic; no current-batch job remains
running or pending. Full seed-separated results and costs are in
[DECISION_REPLICATION_RESULTS_20260923.md](DECISION_REPLICATION_RESULTS_20260923.md).

Mistral frozen/retrieval/fly again tie at 16/16 later in both seeds; this does not identify
an extra benefit of fly online updates. Qwen retains substantial JSON-format failures.
FinSkills has no stable overall interface advantage; HiSTrim lowers KV/allocated-memory
measurements but has mixed latency and no stable quality advantage over recency. Its
final-depth experience retention remains near zero. All failures stay in the denominator.

The bounded seeds 11/23/37 matrix is closed; do not submit more seeds for a preferred
outcome. Broader external-task/time-transfer/human evidence and the separate mature
framework comparison remain incomplete. Jev remains blocked on its real key. Closing
this batch does not mean the whole research plan or conference evidence is complete.

## 03:30 UTC: complete pilot audited; fixed replication accepted

All six seed-11 jobs completed with zero scheduler exit codes. The full 704/704 episode
audit passed using `analysis/audit_beacon_decision_v2.py`. Original grade booleans, sources
and result files are unchanged. The audit-only correction accepts machine-roundoff in
recomputed numerical references (30 differences, maximum 1.11e-16); it does not alter
the experiment's correctness tolerances. Verified summary SHA-256:
`45cc991907197086e1e16814e45915ac90ce47feb876b72b9519549bc2dd5bf8`.
Full outcomes, component scores, failure categories, costs and context-retention limitations
are in [DECISION_LOOP_RESULTS_20260923.md](DECISION_LOOP_RESULTS_20260923.md).

There is no consistent broad advantage: Mistral frozen/retrieval/fly all score 16/16 later;
Qwen suffers exact-JSON compliance failures; HiSTrim later correctness is below recency
in both memory settings for both models. HiSTrim preserves very few explicit experience
items at final pruning depth. These are development findings, not external generalization.

The bounded follow-up uses two prespecified inference seeds, 23 and 37, and puts all eleven
unchanged conditions within one GPU allocation per model/seed for cost comparison.
See [DECISION_REPLICATION_PROTOCOL_20260923.md](DECISION_REPLICATION_PROTOCOL_20260923.md).
It contains 1,408 planned episodes (704 later) on the same four synthetic mechanisms.
Only the `joint` grouping in `decision_loop.py` differs from the pilot's 622-file source
snapshot; parser, algorithms, scores and budgets are unchanged. The old deployed source
remains immutable. The new manifest embeds the prospective replication protocol and hash.

| Model / seed | Accepted job |
|---|---:|
| Qwen / 23 | 1652208 |
| Mistral / 23 | 1652209 |
| Qwen / 37 | 1652210 |
| Mistral / 37 | 1652211 |

Remote: `/beacon-projects/radfm/wy891/fin-skills-campaign-decision-replication-20260923-v1`.
Local receipt: `runs/beacon-campaign-decision-replication-20260923-v1/deployment-receipt.json`.
All four submissions are accepted: never redeploy. Dynamic placement has no node/exclusion
list. At the initial post-submission check they were pending Priority. Each job repeats
qualification; each must produce 352 episodes before the four-job audit. The staged
`analysis/audit_beacon_decision_v3.py` separates seeds and checks 1,408 total episodes;
SHA-256 `8a9294ba40ca641c62ec6e65a1106978997dd19e1a5d84d8e6e426a9d3e55566`.
Beacon syntax and eleven-unique-condition/1,408-denominator checks passed. This is not
completed inference. Stop after these two fixed seeds; repeated seeds do not add task
independence, and no additional seeds should be chosen to improve a preferred result.

At 03:36 UTC, all four replication jobs were RUNNING after dynamic allocation. Each
passed numerical, decoder-equivalence and physical-compaction qualification and started
episode inference. No failed qualification or resubmission is pending.

## 02:30 UTC: qualification verified and the complete decision matrix submitted

Qualification **1651173** completed in 266.20 seconds, with all commands returning zero
and four tests passing. Both models passed all 32 numerical reference cases, official
decoder equivalence, protected-position preservation, nested selection and physical KV
compaction. Max absolute logit errors were 0.03125 (Qwen) and 0.0625 (Mistral), with equal
top tokens. Each two-episode smoke completed correctly and the second prompt contained
the first experience and actual circuit readout. These are engineering checks, not
evidence of general task improvement or cross-task learning.

Single calibration probes recorded full / recency / HiSTrim prefill KV bytes of
98,500,608 / 61,800,448 / 47,054,848 for Qwen and
82,575,360 / 50,659,328 / 44,834,816 for Mistral. This confirms shorter physical caches;
probe timing and allocation peaks are not a randomized efficiency benchmark. In particular,
Qwen's HiSTrim probe prefill took 0.4843 seconds versus 0.1222 for full context. Do not
turn cache reduction into an unmeasured latency or total-memory improvement claim.

All **622 source hashes** in the frozen production matrix match the qualified bundle.
The remote verification is `fin-skills-campaign-decision-loop-20260923-v1/analysis/qualification-verification.json`,
SHA-256 `1f7ce30f3ca768eecd687d9cae908d3eb357b20db28755e0f275366060ee0df9`.
The original frozen archive and sources were not edited.

Capacity was rechecked before each submission. Two association slots became free while
preparing the batch, permitting all six jobs without a rejection or retry. The separate
dispatcher `scripts/submit_decision_pending.py` records append-only batch receipts and
uses the frozen scheduler command and worker. No node names or exclusions were requested.

| Study | Mistral job | Qwen job | Planned episodes across models |
|---|---:|---:|---:|
| Tool choice and use | 1651427 | 1651428 | 192 |
| Long-term experience | 1651429 | 1651430 | 256 |
| Context policy and memory interaction | 1651431 | 1651432 | 256 |

The full denominator is **704 episodes**, including acquisition; 352 are later episodes.
This remains one inference seed and four authored synthetic mechanisms. At the first
post-submission check, 1651427/28/29 were running and 1651430/31/32 were pending Priority.
The local authoritative receipt is
`runs/beacon-campaign-decision-loop-20260923-v1/deployment-receipt.json`; the remote batch
receipt is `submission-batches/20260923T023014097239Z/receipt.json`. All six have accepted
IDs: do not redeploy or resubmit. Final results belong under each job's
`results/decision-loop/study/`; completion still requires per-episode trace and cost audits.

At 02:34 UTC, all four tools/memory jobs were running, had passed their own unit and
engineering checks, and were writing episode records. The two context jobs were waiting
on `AssocMaxJobsLimit`. Production logit errors remained within the frozen 0.15 bound
(0.03125 to 0.125 across those jobs). No completion or aggregate effect is inferred from
these partial logs; the known Mistral tokenizer heuristic warning remains unchanged.

## 03:09 UTC: five study jobs completed; final Qwen context job progressing

Qwen memory **1651430** and Mistral context **1651431** have now completed, with their
`results.json` files containing 128/128 episodes each. Worker elapsed times are 1,733.58
and 1,459.39 seconds; scheduler exit codes are zero. Together with the three earlier
completed jobs, five jobs now have complete result files. These await the single full
matrix audit; no correctness comparison has been selected from the completed subset.

Only Qwen context **1651432** remains running. Its latest inspected log reports 100/128
episodes, in the final fly + HiSTrim condition. Logs show no new execution fault.
Do not resubmit or change the parser. Run the staged auditor once its completion receipt
and 128-row result are present, retaining all 704 planned episodes and original failures.

## 02:54 UTC: three study jobs completed; all remaining jobs running

Scheduler accounting and `completion.json` agree that Mistral tools **1651427**, Qwen
tools **1651428**, and Mistral memory **1651429** completed successfully. All commands
returned zero; their saved results contain the declared 96, 96 and 128 episode records,
respectively. Worker elapsed times are 1,078.88, 1,254.16 and 1,449.13 seconds. Completion
and record counts do not substitute for the pending complete-matrix integrity audit.

Qwen memory **1651430** and both context jobs **1651431/32** are running. Both context
jobs passed engineering qualification and began inference after dynamic allocation; no
study job remains queued. The observed record counts at this check were 102/128 for
Qwen memory, 63/128 for Mistral context and 32/128 for Qwen context. These are progress
counts, not correctness rates. Inspected errors remain JSON-format, missing-parameter
and wrong-method failures; there is no new infrastructure failure to repair or resubmit.
Keep the full 704-episode denominator and run the staged final audit after all six finish.

## 02:43 UTC: execution continues; failure diagnosis and final audit prepared

The four tools/memory jobs continue to emit records. The two context jobs still wait on
the association's concurrent-job limit. No scheduler failure, CUDA exception or new
infrastructure fault appeared in inspected logs. No completed-matrix result exists yet.

Partial traces identify model/interface failures that must remain in the denominator.
Inspected Qwen selection and final responses wrap JSON in Markdown fences, contrary to
the declared exact-JSON interface; the frozen parser rejects them. One Qwen action chose
a volatility method for allocation inputs. Inspected Mistral tool responses omit required
parameters such as confidence or horizon. These observations do not establish a harness
defect. Do not strip fences, fill parameters, increase budgets or reclassify old failures
to improve these runs. Full rates await all conditions. The strict interface is also a
limitation when interpreting this experiment as evidence about financial decision quality.

`scripts/audit_beacon_decision.py` is staged separately under the campaign's `analysis/`.
It checks all six completion receipts before reading performance, then validates frozen
sources, all 704 episode files, paired task inputs, recorded numerical tool outputs,
frozen grade calculations, causal memory eligibility/state continuity, response budgets,
selection traces and saved summaries. It groups acquisition/later outcomes separately,
includes errors, and preserves actual call/token/time/peak-memory/KV and retention records.
The grouping does not treat episodes as independent tasks or infer causal reasoning from
choice changes. Failed calls without returned usage traces remain explicit missing costs.

Beacon syntax validation passed and the script correctly refused the incomplete matrix,
without creating `decision-verified-summary.json`. Full-result validation remains pending.
The staged auditor SHA-256 is
`553d0e61762ec7fb170cf6223b618f6e9dfe3d2b5df1d366e3cf965c27b0eb9b`;
`analysis/audit-readiness.json` records these limited readiness checks. Once all six jobs
complete, execute the staged auditor with the RADFM runtime and campaign root. If any job
fails, first diagnose it and report its missing denominator; do not bypass the completeness
guard or overwrite prior result files. Only a confirmed audit implementation bug warrants
a new version of the auditor; it must not change the frozen experiment's scoring rules.

## 02:26 UTC: Mistral matched repair completed and verified

Jobs **1650301/02/03** completed and the same Beacon audit checked all frozen source
hashes, starting-file equality, result receipts, final artifacts, grades and summaries.
All **15/15** planned cells are present: **0 correct, 15 incorrect, 0 ungradable**, all
ending at `TURN_LIMIT`, with zero accepted outputs in every condition. This is failure
to complete under the declared interface; zero wrong acceptances does not show useful
repair or superior gate behavior.

Across 240 responses, 234 failed the single-tool JSON format and three contained multiple
code blocks. Only three tool calls executed, all self-review records. An inspected trace
contained several concatenated tool-call JSON objects after being asked for one. No
environment error has been established; keep these model/interface failures and do not
relax the frozen parser or rerun them as if they were infrastructure failures.

| Mistral condition | Responses / tool calls | Input tokens | Output tokens | Task wall seconds | Check seconds (included) |
|---|---:|---:|---:|---:|---:|
| Accounting only | 48 / 0 | 288,128 | 21,044 | 901.67 | 39.59 |
| Self-review | 48 / 3 | 326,345 | 24,643 | 1,046.51 | 37.34 |
| Generic assertions | 48 / 0 | 335,682 | 26,130 | 1,136.90 | 63.55 |
| Domain feedback | 48 / 0 | 327,122 | 18,278 | 871.03 | 108.41 |
| Final domain gate | 48 / 0 | 308,840 | 15,449 | 756.24 | 109.59 |

Totals are 1,586,117 input and 105,544 output tokens, 4,712.34 task-wall seconds including
358.48 check seconds, and 7,317.67 summed job seconds including setup and scoring. The
sum is not parallel calendar duration. Model sizes and tokenizers differ from Qwen.
Remote `fin-skills-campaign-model-transfer-20260923-v1/matched-repair-verified-summary.json`
SHA-256: `f6ab3486ca40d97bb5cdf70effa8e97cafc2f6dce23c78461326f2163f5a0156`.
The original scorer's coverage blind spot remains disclosed; no historical scores changed.

## 02:05 UTC: new main-paper loop authorized and submitted for qualification

The user clarified that HiSTrim's implementation is our work, not an external-code
blocker. New code and the full prospective design are in
[DECISION_LOOP_PROTOCOL_20260923.md](DECISION_LOOP_PROTOCOL_20260923.md).
Beacon accepted **1651173**, `decision-qualification`, under
`/beacon-projects/radfm/wy891/fin-skills-campaign-decision-qualification-serial-20260923-v1`.
Its local deployment receipt is
`runs/beacon-campaign-decision-qualification-serial-20260923-v1/deployment-receipt.json`.
It will run numerical tests, full-decoder equivalence, physical KV checks and two
closed-loop smoke episodes for each of Qwen and Mistral. No new local tests were run.
Initial state is PENDING / AssocMaxJobsLimit. Submission requests generic `--gres=gpu:1`
and contains no node or exclusion list.

The complete three-group development matrix is frozen, not yet submitted, in
`runs/beacon-campaign-decision-loop-20260923-v1`. Qualification must pass before its
six model/group jobs expand. The first two-job qualification bundle without `serial`
was prepared but never deployed; leave it unused. The HiSTrim implementation is new
paper-equation code with batch size one, not an original author checkpoint or a verified
reproduction of the reported recommendation speedup. No new experiment result exists yet.

Read-only scheduler check also confirmed prior Qwen matched-repair jobs
1649894/1649895/1649964 completed (26:11 / 18:13 / 26:50); their complete 15-cell
combined audit is still pending. Boundary-recheck 1650725 completed in 03:08 and still
needs result/hash verification. Mistral 1650301/02/03 remained running. Do not resubmit
these completed or running campaigns when following the new main-paper work.

## 02:12 UTC: completed Qwen repair and boundary results verified

`scripts/audit_beacon_repair.py` ran on Beacon against frozen campaign sources. It checked
all source hashes, completed-job receipts (including seed 37's accepted second attempt),
identical initial files within each seed, final artifact digests, result receipts, saved
grades and per-seed summaries. The complete Qwen matrix contains all **15/15** planned
cells: **0 correct, 15 incorrect, 0 ungradable** under the original frozen predicate.

| Qwen condition | Planned / completed | Accepted | Incorrect accepted | Correct completion |
|---|---:|---:|---:|---:|
| Accounting only | 3 / 3 | 3 | 3 | 0 |
| Self-review | 3 / 3 | 3 | 3 | 0 |
| Generic assertions | 3 / 3 | 3 | 3 | 0 |
| Domain feedback | 3 / 3 | 3 | 3 | 0 |
| Final domain gate | 3 / 3 | 0 | 0 | 0 |

The gate prevents these three incorrect acceptances but does not improve correct
completion. These are three seeds of one authored repair task, not independent tasks or
evidence of cross-episode learning. The original predicate's universe-coverage blind spot
is preserved and disclosed; it was not adjusted after results.

| Condition | Responses / tool calls | Input tokens | Output tokens | Task wall seconds | Check seconds (included in wall) |
|---|---:|---:|---:|---:|---:|
| Accounting only | 12 / 12 | 34,326 | 1,469 | 141.92 | 68.46 |
| Self-review | 15 / 15 | 47,123 | 2,082 | 178.25 | 80.17 |
| Generic assertions | 12 / 12 | 37,379 | 1,464 | 174.88 | 104.54 |
| Domain feedback | 12 / 12 | 55,259 | 1,461 | 271.20 | 195.30 |
| Final domain gate | 48 / 48 | 492,444 | 5,790 | 1,228.46 | 835.28 |

Totals: 99 responses and 99 tool calls, 666,531 input and 12,266 output tokens; task wall
time 1,994.71 seconds includes initial public checks. Sum of scheduler job elapsed times
is 4,272.13 seconds and additionally includes loading, tests and independent scoring;
it is not simultaneous calendar duration. Equal maxima did not produce equal actual cost.
Remote summary: `fin-skills-campaign-matched-repair-20260923-v1/matched-repair-verified-summary.json`.
SHA-256: `9bfe8a4c83e6dcad31a30b1f74fe0f854b0cb5df8e913e6b2b2a6ada1dd38646`.

Boundary recheck **1650725** is also fully verified: all six copied artifacts retain their
original digests, the three clean-news cases pass all checks, and all three zero-position
cases return structured `UNASSESSABLE/passed=false` for survivorship. Accounting still
returns `ValueError: insufficient nonconstant evaluation returns`; it is not silently
converted into successful financial-defect detection. No old score or model result changed.
Remote `boundary-verified-summary.json` SHA-256:
`2a619ba128bc7224c6f01a3890668bcb58293ff053b1e0c43495b4903973aeb3`;
the six-row original results SHA-256 is
`85bfc5071de4eb2c923327a89de09142e63efa11a2aab6d9eb1fdb1d9df29308`.

New-main-line qualification **1651173** remains pending (latest reason `Priority` after
`AssocMaxJobsLimit`); no test results exist yet. Mistral **1650301/02/03** remain running,
so their complete matrix is not summarized early. Neither campaign was resubmitted.

| Job | Slurm ID | First observed state |
|---|---|---|
| agent-s11 | 1649141 | RUNNING on an allocated L40S |
| agent-s23 | 1649142 | RUNNING on an allocated RTX 6000 Ada |
| agent-s37 | 1649143 | PENDING, AssocMaxJobsLimit |
| temporal | 1649144 | PENDING, AssocMaxJobsLimit |
| fly-gate | 1649145 | PENDING, AssocMaxJobsLimit |

No nodes were specified. The pending jobs wait for the account association's concurrent
job limit, not a hard-coded target node. Other projects' jobs were left untouched. Both
running Agent jobs passed the Landlock/seccomp preflight. Initial submission receipts
are saved locally in `runs/beacon-campaign-20260923-v1/deployment-receipt.json` and in
each remote job directory. These states are not completed experimental outcomes.

Subsequent inspection confirmed all three Agent jobs RUNNING, with all sandbox preflights
passing and the first two model loads complete. The temporal and fly jobs were still waiting
on `AssocMaxJobsLimit`. Task heartbeat `fin-skills-beacon` checks this campaign every 15
minutes, stays quiet on unchanged state, and records actual completion or execution failures.

## Frozen runnable batch

### Matched repair controls

New immutable campaign: `/beacon-projects/radfm/wy891/fin-skills-campaign-matched-repair-20260923-v1`.
Submission at 2026-09-23 01:11 UTC accepted seed 11 as **1649894** and seed 23 as
**1649895**. Seed 37 was explicitly rejected with `AssocMaxSubmitJobLimit`; it has no
job ID and is still part of the planned denominator. Preserve that rejection receipt.
Check queue/accounting before a new attempt; never resubmit the two accepted jobs.
Local receipt: `runs/beacon-campaign-matched-repair-20260923-v1/deployment-receipt.json`.

At 01:17 UTC, a capacity recheck, empty seed-37 scheduler history, absence of runtime
outputs, and exact frozen-source verification permitted one new attempt. Seed 37 was
accepted as **1649964**. The original rejection and start marker are unchanged; the new
remote receipt is `jobs/matched-repair-s37/submission-attempt-02/submission-receipt.json`,
also recorded locally as `seed37-attempt02-receipt.json` beside the deployment receipt.
All 15 cells now have accepted scheduler submissions; none is yet a completed result.

The five arms are accounting only, required model self-review, generic output assertions,
domain feedback, and domain feedback with final enforcement. All receive identical task,
manifest, flawed source, domain text, execution/accounting assistance and writing tools.
Each has at most 16 model responses of 2,048 output tokens, using the pinned 14B model.
There are 15 planned arm-seed cells. Actual input tokens, model calls, check calls, check
times and wall time can differ and must be reported. This matches maximum budgets, not
realized compute. Early successful submission is permitted in every arm.

The new common interface preloads task/manifest/starting artifacts and supports writing
both artifacts in one tool call. Costs are explicitly 10 bps in every arm to match the
frozen oracle's convention. This is a distinct protocol, not a reclassification or repair
of the previous negative results. Self-review records the model's own critique and must
refer to the current artifact digest; its presence does not prove review quality.
Generic checks test schema, finite values, exposure, determinism, full coverage and
nonzero activity; they intentionally do not intervene on information timing.

Calibration inspected the actual isolated clean-news and clean-price outputs for seed 11
in validity job 1649726: all public checks passed and oracle outputs had zero future and
same-session leakage, zero post-delisting mass, and numerical accounting agreement.
This is evidence that the reference can complete the interface, not a full validity result.
An inspected prior failed historical pipeline output incorrectly included missing-timestamp
documents; it remains a model implementation failure, not a harness failure.

Matched-repair, prior feedback and campaign local checks passed **21 tests in 9.14 seconds**
(71 existing pandas dtype deprecation warnings). Remote tests run again before inference.
The frozen scorer's known scope limits remain; independently authored cases and additional
task/model families are still required for generalization claims.

### Verified validity results and second-family replication

All validity jobs **1649726/27/28** completed. The frozen-source, artifact, per-case
receipt, oracle-file hash and policy checks passed for all **27 planned cases**, with
all three receipt-mutation outputs present. The combined result is
`/beacon-projects/radfm/wy891/fin-skills-campaign-validity-20260923-v1/validity-combined-summary.json`,
SHA-256 `d5f18806ad814892f0efc2100f33efff9d823217d604e7fd630a4957ec8468e0`.
Analysis source is preserved at `analysis/audit_beacon_validity.py` in that campaign.

| Policy or check | Clean artifacts rejected / 6 | Constructed financial defects rejected / 12 |
|---|---:|---:|
| Execution/schema alone | 0 | 0 |
| assert_causal | 0 | 0 |
| survivorship_audit | 0 | 3 |
| accounting | 0 | 3 |
| same_session_probe | 0 | 6 |
| All four checks with execution precondition | 0 | 12 |

There were no check execution errors among these 12 financial-defect constructions.
These are four author-defined mechanisms repeated across three market seeds, not twelve
independently authored tasks. The three invalid-schema cases separately failed execution
and oracle grading; all remain in the 27-case denominator. They are not financial diagnoses.
All 12 file mutations changed the bound digest and marked both guard citations stale;
each revised-source final check ran four fresh checks and rejected the revised artifact.
The initial case check receipts sum to **862.3745 seconds**, excluding oracle time and
the separate mutation rechecks; per-job elapsed times are preserved in the summary.

The independent scorer labels the survivor-only construction correct in all three seeds
under its existing numerical predicate. Oracle universe coverage values are 0.6166667,
0.7 and 0.35; this measured blind spot must accompany agent result summaries.
Cost-5-bps boundary cases all pass public accounting at their stated cost but fail the
oracle's fixed-10-bps convention; these are convention differences, not fabricated reports.
Zero-position boundary cases have oracle predicate=true but public accounting errors for
constant returns. Survivorship checking additionally crashes when serializing NaN evidence
for an empty traded universe. Abstention must not be counted as useful research completion.

The empty-universe transport defect is corrected locally by returning an explicit
`UNASSESSABLE`, `passed=false` result before invoking the survivorship routine. A separate
frozen CPU recheck **1650725**, submitted at 01:48 UTC, uses original unchanged zero-position
and clean-news artifacts for all three seeds (six cases, all four checks). Root:
`/beacon-projects/radfm/wy891/fin-skills-campaign-boundary-recheck-20260923-v1`.
It verifies artifact digests against the original receipts; clean cases must still pass.
No original result, numerical grade, active model job, or acceptance threshold was changed.
The new execution is pending verification, and the original errors remain reported.

The following paragraphs retain the earlier interim observations and deployment history.

By the 01:13--01:17 UTC inspection, clean-news and clean-price cases for all three seeds
had finished public checks and independent grading. All six passed the four public checks,
with zero oracle future/same-session leakage and post-delisting mass. Accounting differences
were numerical roundoff. The full 27-case batch and receipt tests remain incomplete.

The completed seed-37 survivor-only case was rejected by `survivorship_audit`, but the
oracle returned zero leakage, same-session dependence, Sharpe gap and post-delisting mass,
with universe coverage **0.35**. The current `summarize_matrix.py` correctness predicate
uses the former metrics and omits universe coverage, so it would call this constructed
defect correct. Preserve both the mechanism label and raw grade; report this as a scorer
coverage gap. Do not retroactively relabel old results or present the oracle as complete.
For completed future-price cases at seeds 23/37, `assert_causal` passed while the separate
same-session probe failed; the oracle found nonzero future leakage. These are interim
per-case findings, not aggregate sensitivity estimates or grounds to tune test thresholds.

Second-family bundle submitted at **01:28 UTC**:
`runs/beacon-campaign-model-transfer-20260923-v1` targets
`/beacon-projects/radfm/wy891/fin-skills-campaign-model-transfer-20260923-v1`.
It freezes Mistral-Nemo-Instruct-2407 revision
`04d8a90549d23fc6bd7f642064003592df51e9b3`, verified against the official
[model metadata](https://huggingface.co/api/models/mistralai/Mistral-Nemo-Instruct-2407)
and [model card](https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407).
The card lists Apache-2.0 and Transformers support. The CPU staging job runs affected
tests, downloads only Transformers files into this campaign's RADFM cache, hashes them,
and checks that the official chat template preserves a multi-turn tool conversation.
Three GPU jobs depend on successful staging and repeat the same 15-cell matched-repair
protocol. The model identity differs; maximum turns/output tokens, seeds, tools, text,
costs, temperature and context limit remain fixed. Token counts and model capacity differ,
so this is a family replication, not a compute-matched model leaderboard.
The CPU staging job is **1650300**; GPU seeds 11/23/37 are **1650301/1650302/1650303**,
each with an `afterok` dependency on staging. All four submissions succeeded and are in
the local deployment receipt. Staging passed seven tests in 9.62 seconds and completed.
Its recorded Python runtime elapsed time was 158.1346 seconds; the scheduler includes
a slightly longer setup interval. All 13 selected model/configuration files were staged
and hashed and the official multi-turn template check passed. Receipt SHA-256:
`98437df373d0d20c7958a300cdf1b213b524e570f67b68367c93276ba92fc197`.
All three GPU replications are running; their results are not yet complete.

Transformers 4.57.6 emits an incorrect-regex warning when loading this pinned Mistral
tokenizer. Inspection found the snapshot regex already exactly equals the correction
constant in the installed Transformers source. The warning branch uses model/version
heuristics rather than checking that equality. Tokenizer SHA-256 matches the staging
receipt; no proven tokenizer defect or reason to alter the ongoing protocol was found.
The evidence is `jobs/model-staging/results/tokenizer-warning-audit.json`.
No node names were requested. The first family's three repair jobs are now running.

Validity seeds 23 and 37 completed with all nine case records and receipt-mutation outputs.
Each has one oracle error, the intentional unknown-ticker schema case; this is an ungradable
invalid submission, not a numerical grade or a successful financial defect explanation.
All four file mutations per seed marked both citations stale; final changed-source checking
executed four new checks and rejected the submission. Seed 11 is still running, so the
full 27-case combined audit has not yet been calculated.

### Correct native-component comparison

Job **1649796** completed successfully in **1217.8475 seconds** including tests and setup.
Receipt: `runs/beacon-campaign-components-20260923-v1/deployment-receipt.json`.
Remote root: `/beacon-projects/radfm/wy891/fin-skills-campaign-components-20260923-v1`.
Implementation: `benchmarks/library_workflows/component_comparison.py`.

This adds a correct native SQLite window-query baseline to the historical-document
experiment. Both methods query the same Store-populated database and return materialized
identifier/revision mappings, checked against a raw-event reference frozen before querying.
The protocol crosses 80/800/4000 records with seeds 11/23/37 and six historical cutoffs:
54 queries per method, each with one warmup and seven randomized-order timed repetitions.
Report exact answers and latency by size, retaining all failed or missing queries.

This is a synthetic development query-endpoint comparison. Shared ingestion is not
compared, and the library additionally constructs provenance documents internally.
Latency therefore measures endpoint cost, not equal internal work. Neither repeated
timings nor different generator seeds constitute independent tasks or human time savings.
Correct native SQL may match correctness and beat latency; that result must be retained.
Local component/campaign checks passed nine tests in 1.95 seconds before freezing.
Frozen source hashes, completion commands, all 108 method-query records, raw-reference
answer maps, timing repetitions and summary medians were checked on Beacon. Both methods
were exact on **54/54 queries**, with no missing query cells. Each row reports seven
correct timed repetitions; the retained selected-answer map is from the last repetition.

| Base records | Native SQL exact | Library exact | SQL median endpoint time | Library median endpoint time |
|---|---:|---:|---:|---:|
| 80 | 18/18 | 18/18 | 3.576 ms | 12.039 ms |
| 800 | 18/18 | 18/18 | 6.883 ms | 40.461 ms |
| 4000 | 18/18 | 18/18 | 30.077 ms | 177.948 ms |

These medians summarize per-query medians across paired seeds and historical cutoffs.
The correct native SQL baseline is faster at every tested size; this experiment does not
show a library query-speed advantage. Extra provenance-document construction and shared
ingestion limits remain as stated above. No human work or broader retrieval quality is
measured. Full results: `jobs/component-comparison/results/components/results.json`, SHA-256
`c93ad03349d13c42a077d97ba0bf49b1f26b7ad3b7d1867a5ced8df7dd6b2de3`.
Adjacent `verification.json` records the audit. Protocol SHA-256:
`cac5ce74420f233b5d92927445a4f841cf4c5cbbe7687da5c42fe8c7384679ec`.

### Observed results and subsequent batches

The initial three Agent jobs completed all 12 planned cells. Independent grading produced
one numerical grade, which failed correctness; the other 11 cells were ungradable. There
were zero independently verified correct cells. `skills_optional_guards` accepted one
incorrect output (seed 23; same-session dependence 0.6041666666666667 and Sharpe gap
4.137495734184188). `no_library` accepted one ungradable output (seed 37). The other two
conditions accepted none. Rejection is not evidence that an agent completed the task.
Multiple JSON calls, wrong filenames and missing reports remain failures in the denominator.
The combined remote summary is `agent-combined-summary.json`, SHA-256
`74e7c17015c0ef0e393c7210c58527e137b72977ea54ca5a123e58afa4688664`.

Temporal job 1649144 completed and reproduced all values in the development table below.
These are public synthetic correctness checks, not an independent dataset evaluation.

Initial fly job 1649145 failed before inference because the previously deployed upstream
was an extracted archive, not a Git checkout. The correction verifies every archived file
against the pinned `upstream.tar.gz` SHA-256
`b147e0494f1b792b5b83e065d60d219ccb7cdb3703181a815b74dad7ac6de880`, then compares
the prior protocol's source pins and market data hash. This establishes byte-identical
reuse of the prior deployment, not a fresh independent Git verification.

Pipeline batch `fin-skills-campaign-pipeline-20260923-v1` jobs 1649314, 1649315 and
1649320 failed the Linux preflight before inference: OpenBLAS inherited four threads while
the evaluator's seccomp policy denies thread creation. The correction sets numerical
libraries to one thread inside the isolated child process. Initial seed-37 submission
hit `MaxSubmitJobs`; its rejection receipt was retained, and the accepted retry has its
own receipt. No other project's jobs were cancelled.

Recovery batch `/beacon-projects/radfm/wy891/fin-skills-campaign-recovery-20260923-v2`:

| Job | ID | Observed status |
|---|---:|---|
| Source and Linux evaluator validation | 1649351 | COMPLETED, exit 0 |
| Pipeline seed 11 | 1649352 | COMPLETED, exit 0 |
| Pipeline seed 23 | 1649353 | COMPLETED, exit 0 |
| Pipeline seed 37 | 1649354 | COMPLETED, exit 0 |
| Same-gate fly versus frozen | 1649355 | COMPLETED, exit 0 |

At the 00:46 UTC follow-up, all three pipeline seeds completed. All 18 planned arm-stage
cells were included; each source hash was checked against its submission receipt and each
reported correctness flag against its saved grade before aggregation.

| Allowed components | Correct / planned | Generation attempts | Total tokens | Summed stage seconds |
|---|---:|---:|---:|---:|
| Standard components | 2 / 9 | 30 | 111650 | 730.8015789650381 |
| Standard components plus fin_skills | 4 / 9 | 27 | 92823 | 596.1014604163356 |

By stage, standard components scored build 2/3, historical 0/3, restore 0/3; the
fin_skills-available condition scored build 1/3, historical 0/3, restore 3/3. The model
could choose whether to use the library in that condition. These small, correlated
development results do not establish a general efficiency gain; both arms failed every
historical-change cell. Summed stage times exclude final evaluation and model loading,
include generation and public feedback, and must not be described as human working time.
The frozen aggregate is `pipeline-combined-summary.json` in the recovery batch root,
SHA-256 `0b493556fa72420fa844e1978cb2a7bdc99b98d1977de0f89e9d98efb0c1614e`.

Fly job 1649355 subsequently completed. The independent standard-library ledger audit
passed 34 windows, 2,158 fills and 7,550 NAV observations with maximum absolute NAV error
0.0. Frozen-gated training weights were also explicitly checked unchanged. Results:

| Same-gate arm | Mean test net return (%) | Mean fills |
|---|---:|---:|
| Learned fly | 2.326094621303838 | 4 |
| Frozen fly | -0.12339144914143692 | 43.4 |
| Buy and hold comparator | 6.731657857200002 | — |

The paired mean difference is 2.4494860704452748 percentage points, with seed differences
2.9255, -6.9831, 17.7250, -12.7176 and 11.2975. This is descriptive seed variation on one
public market path. Learned memory does not beat the buy-and-hold comparator on the mean;
the contrast also includes different turnover and costs. The independent audit does not
certify real-market execution. Remote evidence is under `jobs/fly-gate/results/`:
`paired-summary.json`, `independent-ledger-audit.json` and `paired-evidence.tar.gz`.
The evidence archive SHA-256 is
`dd73f7ca41a29e930cc3988481bea2b4bf6c2b96dc493538cad32aae3b4c3d20`.

The pipeline protocol pairs components (stdlib/numpy/pandas/scipy) against `fin_skills` for
BM25 construction, adding historical filtering, and persistence/reload in fresh processes.
It uses seeds 11/23/37, four attempts per stage, and 4096 output tokens per attempt. The
stdlib mathematical scorer runs on separately frozen evaluation fixtures only after code
and receipt freeze. Report correctness alongside attempts, tokens and time; this measures
coding-agent effort, not human developer time. All cases are author-generated development
fixtures. The latest local pipeline/snapshot/campaign/submission tests passed 23 tests with
one Linux-only skip; the recovery validation passed that Linux test on Beacon.

The separate feedback repair protocol is implemented in
`benchmarks/agent_study/feedback_ablation.py`: identical executable flawed starters,
identical guidance, four feedback/enforcement arms, seeds 11/23/37, 16 turns and 2048
output tokens per turn. Initial feedback is supplied before those turns in all arms.
Independent grades are never returned during repair. It does not replace or pool with
the original end-to-end study. Targeted feedback/campaign tests passed 14 tests.

The feedback batch `fin-skills-campaign-feedback-20260923-v1` was submitted at
00:38 UTC: seeds 11/23/37 are jobs 1649417/1649418/1649419. Seed 11 passed its Linux
checks and started; the other jobs initially waited on `AssocMaxJobsLimit`.

All feedback jobs subsequently completed. The 12 submission hashes and final workspace
digests matched their receipts. Four cells were independently gradable; none was correct.
Execution-only accepted all three incorrect submissions. Accounting feedback accepted none;
full guard feedback accepted one ungradable submission; full feedback plus final gate
accepted none. Rejection is not correct task completion, so this does not establish agent
utility. The frozen `feedback-combined-summary.json` SHA-256 is
`28a07318e76074c53c5b9e614fb60606627e4f52771c80c191a4a752449c80fc`.

The user then authorized further experiments tied to conference-level claims. The prospective
plan is [NEXT_EXPERIMENTS_20260923.md](NEXT_EXPERIMENTS_20260923.md). A new audit-validity
batch `fin-skills-campaign-validity-20260923-v1` was submitted at 00:59:54 UTC:
jobs 1649726/1649727/1649728 cover seeds 11/23/37. They run clean/defective/boundary artifacts,
all four-check policy subsets, the independent scorer, and post-check artifact mutations.
These are author-created instrument tests, not independent defects or additional model runs.

Retrieval batch `fin-skills-campaign-retrieval-20260923-v1`, job 1649462, runs the existing
six-question English public pilot with static context, BM25 and E5, followed by Qwen answer
choice scoring at seeds 11/23/37. All arms share a 4,000-character context cap, but actual
context lengths differ and are recorded. Static context includes every eligible document
that fits in frozen corpus order. Answer correctness, invalid output and relevant-source
citation counts are separate; citation relevance is not entailment. Jev is absent, not mocked.

Job 1649462 completed in 126.9337292322889 seconds. All 54 responses and their contexts
matched frozen receipts; no future document appeared. Recomputed parsing/scoring matched
the saved scores. Across the same six questions at three seeds:

| Context condition | Correct / attempts | Invalid responses | Relevant source cited | Total tokens |
|---|---:|---:|---:|---:|
| Static context | 16 / 18 | 2 | 16 / 18 | 8300 |
| BM25 | 18 / 18 | 0 | 18 / 18 | 4821 |
| E5 vector | 18 / 18 | 0 | 18 / 18 | 4818 |

These are six distinct public questions, not 54 independent examples. BM25 and E5 tie
on this pilot; it cannot demonstrate superior semantic retrieval or free-text answer
quality. Both invalid static-context responses (q-citation at seeds 11 and 23) contained
the correct choice and relevant citation inside a Markdown JSON fence. The frozen protocol
required bare JSON and forbade fences, so they remain invalid. The score difference is
format compliance, not evidence of a factual-answer advantage. The combined summary is
`jobs/retrieval-pilot/results/combined-summary.json`,
SHA-256 `5df1ca8332e4616d92d1fb25300eea18733a5ecd9e3b4c76fe8c8a03811c4a53`.

The encoder is `intfloat/e5-small-v2`, revision
`ffb93f3bd4047442299a41ebb6fa998a38507c52`, verified through its model API on 2026-09-23 UTC.
Encoding follows the author's [model card](https://huggingface.co/intfloat/e5-small-v2):
query/passage prefixes, masked mean pooling, normalized vectors. Inputs exceeding its
512-token limit cause an explicit failure. The job records downloaded file hashes and
keeps the encoder cache in RADFM.

Routing batch `fin-skills-campaign-routing-20260923-v1`, CPU job 1649498, compares BM25
and that encoder against 129 catalog descriptions and all 106 English questions selected
from the existing 108-row `evals/queries.jsonl`. The two Chinese questions are excluded
because this encoder is English-only. Query packets strip expected labels and notes;
all rankings are frozen before scoring top-1, recall@3/@10 and MRR@10. These are exposed
public regression questions, not independently authored holdout data. This expands the
retrieval measurement but does not expand the six-question downstream answer pilot.

Routing job 1649498 completed in 37.71114270063117 seconds. All 106 questions and 129
document IDs were covered; saved rankings and source hashes matched the receipts, ranked
IDs were valid and unique, and metrics were recomputed from the public labels:

| Retriever | Top-1 correct / 106 | Recall@3 count / 106 | Recall@10 count / 106 | MRR@10 |
|---|---:|---:|---:|---:|
| BM25 | 75 | 98 | 103 | 0.8151317759808326 |
| E5-small-v2 | 57 | 75 | 96 | 0.6441786463012878 |

This pinned E5 configuration underperformed BM25 on the exposed English routing fixture.
Do not retune it on these labels and present the new score as independent confirmation.
The remote `jobs/routing/results/routing/score.json` SHA-256 is
`bd5cdf89d596a1f8a0642fe19ed3502d7e0a25258df0f413b3f4d8966c239d3b`;
the adjacent `audit.json` records the checks.

Final local targeted checks for the updated runners, scorer boundaries, snapshot validation,
temporal cases, campaign and submission handling: **42 passed, 1 Linux-only skip**, in
7.43 seconds. Existing pandas dtype warnings remain in the public shock probe; the pinned
runtime executed it successfully. No local GPU model result is included in those tests.

Local bundle: `runs/beacon-campaign-20260923-v1/`.
Remote destination: `/beacon-projects/radfm/wy891/fin-skills-campaign-20260923-v1/`.
Archive SHA-256: `da7b89ab7acbd52fb2dd5b646bcbde33afc66cf417d7fee0de5c2953d5ac53b3`.

| Job | Experiment | Resource request |
|---|---|---|
| agent-s11 | Four conditions, market seed 11, pinned Qwen2.5-Coder-14B | One GPU, four CPUs, 64G RAM |
| agent-s23 | Same conditions and budget, seed 23 | One GPU, four CPUs, 64G RAM |
| agent-s37 | Same conditions and budget, seed 37 | One GPU, four CPUs, 64G RAM |
| temporal | Historical revisions, duplicate ingestion, database reopen and transaction rollback | One CPU, 8G RAM |
| fly-gate | Learning versus frozen memory under the same gate and frozen prior market snapshot | Four CPUs, 32G RAM |
| rag-jev | Existing paired BM25/Jev public development retrieval pilot | One CPU, 8G RAM; actual Jev API key required |

Slurm chooses nodes; no node list, node exclusion or reservation is requested. GPU jobs
check for at least 40 GiB of device memory before model loading. Whether the generic GPU
request matches the available hardware still needs remote resource inspection.
CPU jobs request no GPU. Each job records its submission attempt before calling `sbatch`;
unknown outcomes are retained rather than retried. A missing prerequisite blocks that job
without silently replacing it with a mock result. All job workspaces, scheduler logs,
temporary files and application caches are inside RADFM. Existing model weights are reused
offline from the prior RADFM runtime.

Each Agent job covers all four paired conditions for one market. Aggregate the three
`jobs/agent-s*/results/matrix` directories with `summarize_matrix.py`; do not compare only
successful jobs. A completed scheduler job does not imply an independently correct answer.

## Ready local commands

From `D:/fin_skill`, use the system Python for the transport (Paramiko 5.0.0 was importable).
The professional virtual environment was used for numerical checks but lacks Paramiko.

```powershell
python -m scripts.beacon_campaign validate runs/beacon-campaign-20260923-v1
python -m scripts.beacon_campaign deploy runs/beacon-campaign-20260923-v1
```

The deploy command requests the Beacon password privately in a local terminal. Alternatively,
an explicitly supplied `--credential-file <local-path>` reads the first line without logging
or uploading that file. The deployment checks the pinned Beacon host key first.
If `TYPESAFE_API_KEY` is configured locally, its value is transferred separately to a
restricted file in the campaign's private directory, excluded from archives and receipts;
the Jev worker deletes that file before its requests. Do not paste either secret into chat.

After any interrupted deployment, inspect status before retrying:

```powershell
python -m scripts.beacon_campaign status runs/beacon-campaign-20260923-v1
```

## Local evidence

The targeted campaign, temporal, Agent interface, Jev evaluation, fly-gate and submission
checks passed: **39 passed in 6.69 seconds**. After subsequent worker environment and
upstream revision-check corrections, the affected local campaign, temporal and submission
checks passed again: **18 passed in 1.39 seconds**. Python compilation and frozen archive
validation passed. Ordered index/package regeneration and `validate.py` passed, retaining
the existing skill-discovery-budget warning. None of these are remote execution results.

The temporal development experiment ran locally under
`runs/campaign-temporal-development-20260923-v1/`. It froze synthetic event histories and
raw-event expected outputs before querying the library. Across 18 historical queries:

| Method | Exact queries | Wrong selected versions | Omitted eligible records |
|---|---:|---:|---:|
| Latest version without cutoff | 0 | 840 | 0 |
| Latest version, then cutoff | 2 | 0 | 337 |
| Latest eligible historical version using the library composition | 18 | 0 | 0 |

The first two methods are intentional negative controls. The correct raw-event reference
is the comparator; these results do not prove superiority over a correctly implemented
alternative database. Reopening and transactional rollback do not establish process-kill
or distributed recovery behavior. No financial return or model quality is measured here.

## Still required for the full requested experimental program

- Retrieval expansion: the 106-question public routing comparison and six-question
  downstream pilot are complete. Broader independent downstream questions and content
  support labels remain necessary; the current pilot cannot substitute for them.
- Pipeline building efficiency: all 18 coding-agent cells are completed and summarized above.
  A human developer study additionally needs participants; it cannot be inferred from
  coding-agent token use or wall time.
- Independent defects and clean cases: obtain cases from authors who did not implement
  the guards. Newly generated local defects alone cannot establish this independence.
- Feedback ablation: all 12 repair cells are complete. Matched-budget self-review and
  generic-check controls still need the additional prospective protocol described above.

These are explicitly pending in the frozen manifest. The runnable batch does not mean the
whole experimental program is ready or complete. Paper result tables and Overleaf have
not been updated with any new Beacon result.
