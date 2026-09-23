# Additional experiments tied to the paper's claims

Decision date: 2026-09-23 UTC. This is a prospective work plan, not a list of completed
results or a promise of conference acceptance. User authorization covers implementation
and Beacon execution. Freeze each protocol before inference; preserve prior negative runs.

## Reuse-first correction after primary-source review (2026-09-23)

Final execution update: the submitted reuse batch is complete and audited. Actual
LlamaIndex/Reflexion memory evaluation covers 352 acquisition/evaluation episodes;
DocFinQA covers 384 planned units including 60 unavailable-input units; FinanceBench
covers 300 retrieval and 300 reranking method-query units. Detailed outcomes are in
[FinQA memory results](FINQA_MEMORY_RESULTS_20260923.md),
[DocFinQA context results](DOCFINQA_CONTEXT_RESULTS_20260923.md) and
[FinanceBench reuse results](FINANCEBENCH_REUSE_RESULTS_20260923.md).
These counts are repeated conditions, not independent task counts. No accepted job in
this batch remains running or pending. Do not rerun for better scores or extend seeds.

Subsequent bounded review identified one executable missing control: actual LlamaIndex
with multiple financial tools under generic/flat-guidance/organized catalogs. The prior
LlamaIndex FinQA adapter exposed only one calculator. See
[FRAMEWORK_SCOPE_REVIEW_20260923.md](FRAMEWORK_SCOPE_REVIEW_20260923.md) and the frozen
[framework tool protocol](FRAMEWORK_TOOLS_PROTOCOL_20260923.md). This is 192 cells on
the same authored contracts, not another independent benchmark or FinRobot reproduction.
CPU qualification 1655240 passed, including its source/fixture audit. Production
Qwen 1655292 and Mistral 1655293 are complete and audited, 96 cells each;
their source hashes match qualification. Generic/flat/organized correctness is Qwen
25/28/26 and Mistral 10/13/3 out of 32. See
[framework results](FRAMEWORK_TOOLS_RESULTS_20260923.md) and the
[bounded campaign closeout](BEACON_CLOSEOUT_20260923.md). No submitted campaign job
remains; pause follow-up without declaring the entire research complete. Do not redeploy
or expand the fixed matrix.

Actual framework/module reuse is now measured, but a matched full-system comparison
against FinRobot/FinMem/InvestorBench is not completed. Their task compatibility and
necessary inputs must be established before calling such a comparison executable;
do not infer that every remaining comparison is blocked solely from Jev's missing key.
Jev still needs a genuine key. New human efficiency/support annotation and prospective
private-time evidence remain separate requirements. This batch does not finish the
entire research plan or establish broad FinSkills/HiSTrim superiority.

The user requested reuse of existing benchmarks and implementations. The statement that
independent answer/evidence labels must all await new external input was too broad:
FinQA and the public FinanceBench subset already supply relevant third-party annotations.
Use the source-checked mapping and execution order in
[REUSE_EXISTING_BENCHMARKS_20260923.md](REUSE_EXISTING_BENCHMARKS_20260923.md).
Prioritize upstream scoring/data, actual LlamaIndex tool execution and Reflexion code;
then evaluate DocFinQA context and suitable FinRobot/FinMem/InvestorBench components.
Public benchmark exposure, overlapping FinQA-derived datasets, private test labels,
external API dependencies and local-embedding adaptations must be reported explicitly.
Jev's official SDK still requires a key; local BGE reranking can be a separately named
baseline. The following candidate plan is historical; use the final execution update
above to distinguish completed comparisons from unimplemented candidates.

## Current status: bounded decision matrix complete (05:10 UTC)

The seed-11 pilot and fixed seeds 23/37 replication are now fully audited: 704 plus
1,408 episodes, with acquisition/later and all seeds kept separate. See
[DECISION_REPLICATION_RESULTS_20260923.md](DECISION_REPLICATION_RESULTS_20260923.md)
for complete correctness, failures, actual costs and context-retention limitations.
The four replication jobs 1652208/09/10/11 are complete; there are no remaining jobs
in this batch. Stop at these prespecified seeds. Repeated seeds are not new tasks.

The data do not establish broad FinSkills or HiSTrim superiority, or an online-fly
advantage over frozen/retrieval memory. The separate mature-framework comparison below
is still uncompleted; it must not be conflated with the numerical-interface comparison.
Independent task/time transfer, independently labeled answer/support data and human
efficiency remain future evidence requirements; Jev still needs its actual API key.
The following dated preparation/submission entries are historical, not live job status.

## Updated main-paper priority: tools, context and experience (03:30 UTC)

The user authorized all three integrated financial-loop experiments and clarified that
HiSTrim must be implemented here; an external checkpoint is not a prerequisite.
The executable prospective design is now in
[DECISION_LOOP_PROTOCOL_20260923.md](DECISION_LOOP_PROTOCOL_20260923.md).
`decision_tasks.py`, `decision_memory.py`, `histrim.py` and `decision_loop.py` under
`benchmarks/agent_study/` implement the first development version. It has independent
numerical references, LLM-facing experience reads, an actual causal fly circuit, and
single-sequence nested physical KV compaction. Both models passed remote engineering
qualification; broad quality or efficiency improvements are not yet established.

Qualification job **1651173** completed in 266.20 seconds. Both models passed 32 numerical
references, official-decoder equivalence, protection/nesting/physical-compaction checks
and two closed-loop smoke episodes. All 622 source hashes match the full frozen matrix.
Beacon accepted all six jobs at 02:30 UTC: Mistral tools/memory/context **1651427/29/31**;
Qwen tools/memory/context **1651428/30/32**. The authoritative receipt is
`runs/beacon-campaign-decision-loop-20260923-v1/deployment-receipt.json`. Do not redeploy.
All placement is dynamic. Audit all 704 planned episodes (352 later episodes), failures,
actual memory reads, choices, parameters, receipt use and compute costs before reporting
effects. Never deploy the unused two-job qualification bundle without `serial`.

The seed-11 matrix is now fully audited, 704/704, with results in
[DECISION_LOOP_RESULTS_20260923.md](DECISION_LOOP_RESULTS_20260923.md). It does not show
consistent FinSkills or HiSTrim superiority; Mistral frozen/retrieval/fly all reach the
same 16/16 later score, and Qwen is limited by JSON-format failures. Final-depth HiSTrim
history retention is often near zero. Preserve these negatives and the original parser.

Bounded replication is submitted under
[DECISION_REPLICATION_PROTOCOL_20260923.md](DECISION_REPLICATION_PROTOCOL_20260923.md):
Qwen/Mistral seeds 23 and 37, jobs 1652208/09/10/11, 1,408 planned episodes. All eleven
conditions share one dynamic GPU allocation per model/seed, addressing pilot latency
confounding across allocations. No algorithms, task inputs, scoring or response budgets
changed. Audit both seeds separately, do not treat them as independent tasks, and stop
after these prespecified seeds rather than expand until a preferred method wins.

The earlier audit-centered priorities below remain supporting work. They must not replace
this shared loop or trigger unrelated scale/error sweeps. The first loop has four authored
task mechanisms and later synthetic blocks; independent real-world task/time transfer and
additional inference seeds remain separate requirements for broad main-paper claims.

## What currently limits the main claim

The outline centers on correct financial research under guidance, optional checks and
final-artifact verification. The completed development end-to-end run produced no verified
correct output. Retrieval has a six-question downstream ceiling and an exposed routing
fixture. The pipeline study contains only three paired seeds and correlated stages.
These experiments are useful diagnostics but do not yet establish broad agent utility.
Repeated seeds of one task generator do not create independently authored tasks.

## Execution order

| Priority | Experiment and claim it tests | Controlled comparison | Main outcomes | Execution status |
|---|---|---|---|---|
| P0 | Audit validity: do the checks detect defects without rejecting correct artifacts? | Identical clean and defective artifacts; execution alone and every subset of four public checks; separately run the frozen independent scorer | False alarms, missed defects, execution errors, scorer blind spots, check time | All 27 cases verified; full policy accepts 6 clean and rejects 12 constructed financial defects; scorer and zero-position boundary gaps documented |
| P0 | Receipt freshness: does evidence still apply after the artifact changes? | Change code, numerical report, market data or manifest after checking; compare old receipts with fresh final checks | Stale receipt recognition and final acceptance of revised defective code | All 12 mutations marked stale; three final rechecks reject altered sources |
| P0 | Agent harness and usable completion | Reference implementation through the actual isolated harness; then paired agent runs after the reference is scoreable | Parse/file/schema failures separated from financial errors; correct accepted completion | Existing reference and new validity results are the prerequisite; do not inflate budgets until failures are classified |
| P1 | Mechanism and matched budgets | Common accounting utility; self-review, generic assertions, domain feedback and final gate | Correct completion and incorrect acceptance against actual tokens, model calls, check time and wall time | Both models 15/15 verified, none correct. Qwen accepts 12 incorrect outputs; gate rejects 3/3. Mistral reaches turn limit in all 15, accepts none; 237/240 responses fail tool-call parsing. Actual cost differs substantially |
| P1 | Unseen defect mechanisms and realistic clean controls | Checks fixed before external authors provide cases; group split by defect mechanism and source project, not just seed | Per-family sensitivity, clean specificity, abstentions and coverage | Requires independently authored inputs; locally generated cases are development only |
| P1 | Model and task transfer | At least two model families with immutable revisions; distinct research tasks under the same four conditions | Paired task-level differences, correct completion, cost and failure categories | Mistral repair replication complete and verified; new four-mechanism decision loop submitted for both models. Independent task/time transfer still needed |
| P1 | Real documents and point-in-time transfer | Source-frozen filings/report revisions; available-at filtering on/off; same questions and context budget | Unsupported claims, numeric answer accuracy, future-source use, citation content support | Need a rights/provenance-checked task packet and independent answer/support labels |
| P2 | Reliability and scale of collection/RAG | Duplicate and out-of-order events, database reopen, process termination around commit, partial writes, missing timestamps; increasing corpus/history sizes | Lost/duplicated records, incorrect versions, recovery time, memory and query latency | Current reopen/rollback results do not cover process-kill recovery; prepare as a separate storage benchmark |

P0/P1 carry the central audit/resource paper. A large additional fruit-fly sweep is not a
substitute for them. If the memory result remains a main claim, add several independent
market paths, regimes and cost assumptions, with learned/frozen and conventional baselines
under exactly the same gate. Otherwise keep the current one-path result exploratory.

## Newly frozen audit-validity batch

Implementation: `benchmarks/agent_study/audit_validity.py`.
Remote: `/beacon-projects/radfm/wy891/fin-skills-campaign-validity-20260923-v1`.
Seeds 11/23/37 are paired development worlds. Each job runs the declared nine cases:

- Two nominally clean constructions: lagged news and lagged price momentum.
- Same-session news, future-price dependence, survivor-only universe and a wrong Sharpe report.
- A stated-cost convention boundary and a zero-position abstention boundary, reported separately.
- Invalid position schema.

The truth label follows the constructed mechanism, not whether FAST or its scorer likes
the case. In particular, a scorer that misses survivor-only selection must be reported as
a scorer gap. Explicit cost-convention differences must not be mislabeled as model lying.
An execution error is not a successful detection with a financial explanation.

Every artifact is frozen before independent grading. The matrix records all four public
checks and all their subsets with a common execution precondition. These are descriptive
policy ablations on a fixed corpus, not interactive agent-policy effects and not a search
for a winning acceptance rule. Receipt tests measure freshness under ordinary file changes;
they do not establish security against a hostile host or in-process evaluator tampering.

The accounting construction has a separate numerical implementation; its agreement with
the public accounting routine is unit-tested. Local validity/campaign checks passed
12 tests in 4.18 seconds. Formal checks and oracle execution take place on Beacon.

## Library advantage: comparisons that separate the source of a benefit

The user requested direct evidence for the library's comparative value. The existing
pipeline result (4/9 correct library-available cells versus 2/9 component-only cells) is
a development pilot, with correlated stages and only three paired seeds. Its comparator
implements BM25 from scratch; it does not establish superiority over mature complete
frameworks. Library availability also does not prove actual API use.

| Question | Required comparator | Outcomes and limits | Status |
|---|---|---|---|
| Are historical answers correct at practical cost? | Correct native SQL on the same stored revisions | Exact version selection, query latency by corpus size; shared ingestion and extra provenance work disclosed | Job 1649796 verified: both 54/54 exact; native SQL faster at all three sizes |
| Does reuse help complete build/change/restore tasks? | Ready-to-use component APIs, with the same domain guidance and interface documentation | Correct completion, repair attempts, model calls/tokens, elapsed time and actual API use | Existing pilot complete; stronger ready-component control still to implement |
| Is domain verification better than extra checking alone? | Same guidance plus generic assertions and matched-budget self-review, alongside domain checks and final gate | Correct accepted completion, wrong acceptance, false rejection and check cost | Validity 27 cases verified; both 15-cell repair matrices verified, none correct. Qwen gate prevents its three wrong acceptances; Mistral accepts none and reaches turn limit in all cells |
| Which part of the library accounts for any difference? | No library, text guidance only, APIs without domain checks, full library, with paired tasks and identical budgets | Task-level effect and resource use; preserve missing and rejected outputs | Initial four-arm study exists but has zero verified correct outputs; calibrated follow-up needed |

The new component protocol uses 80/800/4000 synthetic records, three seeds and six cutoffs
per dataset. Each of its 54 queries per method is repeated seven times after warmup.
These repetitions estimate timing variability; they are not 378 independent task trials.
Its remote root is `/beacon-projects/radfm/wy891/fin-skills-campaign-components-20260923-v1`.

Numerical agreement with NumPy/SciPy references is a correctness prerequisite, not proof
of superiority over those dependencies. BM25 versus E5 measures a retrieval choice, not
the whole library's benefit. Claims of developer time savings require a separate human
study; lower coding-agent token use cannot supply that evidence. All comparisons should
allow a result of parity or disadvantage, without selecting tasks or metrics after results.

## Calibration and scaling rules

1. Audit the reference implementation through the same filenames, public checks, isolation
   and oracle used for model submissions. Retain any false rejection rather than changing
   the clean label until the checker passes.
2. Freeze one interface correction if an actual harness defect is demonstrated. Keep its
   results separate from the old protocol. Formatting normalization, if adopted, must be
   identical across arms and fixed before the new run, not applied only to losing outputs.
3. Define task families and group splits before assigning models. Keep development traces
   separate from final cases. A different seed does not conceal a known defect template.
4. Pair conditions by task and inference seed. Randomize order. Use identical maxima and
   report actual resource use; add self-review and generic-check controls rather than
   attributing every extra call's benefit to domain tools.
5. Set the primary endpoint as correct accepted completion alongside incorrect acceptance
   per attempt. Include rejection, empty artifacts, provider errors and ungradable accepted
   outputs in explicit denominators. Report the trivial reject-all policy to expose a
   supposed improvement that is achieved only by refusing every task.
6. Determine sample size from a prespecified precision or detectable paired difference.
   Compute uncertainty at the independent task/source level; do not bootstrap tokens,
   repeated seeds or pipeline stages as if they were independent tasks. Threshold
   sensitivity is descriptive and must not select a new acceptance threshold on the test set.

## Related-work implications

[CheckList](https://aclanthology.org/2020.acl-main.442/) motivates testing behavior by
capability and perturbation, rather than relying only on aggregate task scores.
[Profit Mirage](https://arxiv.org/abs/2510.07920) already studies financial information
leakage and introduces FinLake-Bench and FactFin; our contribution cannot be simply that
financial agents leak information. The artifact-bound execution mechanism needs its own
ablation and clean-case evidence.
[Finance Agent Benchmark](https://arxiv.org/abs/2508.00828) evaluates research over SEC
filings with expert-authored tasks. Such tasks are relevant to external transfer, but their
answering benchmark is not directly interchangeable with a portfolio-code leakage test.
Check benchmark licenses, task compatibility and exposed-label status before importing it;
do not compare scores across different tasks as if they were a shared leaderboard.

Sources above were opened on 2026-09-23 UTC. These connections are experimental-design
judgments, not claims that a venue requires this exact list or guarantees acceptance.
