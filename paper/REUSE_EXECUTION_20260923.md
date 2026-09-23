# Reuse campaign execution log

User authorized autonomous continuation on 2026-09-23. This continuation reuses third-party
data, scorers and actual frameworks. It does not rerun the closed 2,112-episode synthetic
decision matrix or alter any of its negative results.

## Upstream and environment qualification

CPU bootstrap v1, job **1653290**, acquired and hashed the following upstream snapshots:

- FinQA: `czyssrs/FinQA`, commit `0f16e2867befa6840783e58be38c9efb9229d742`.
- FinanceBench: `patronus-ai/financebench`, commit `cc39aeb4afdf33909ee1412188bf89035950c2eb`.
- Reflexion: `noahshinn/reflexion`, commit `218cf0ef1df84b05ce379dd4a8e47f17766733a0`.
- LlamaIndex core selected and pinned to `0.14.25` before installation.

Root: `/beacon-projects/radfm/wy891/fin-skills-campaign-reuse-bootstrap-20260923-v1`.
The job failed during scorer import because the isolated environment inherited only the
first runtime site-packages directory; the runtime's second RADFM dependency directory
contains SymPy and was omitted. This is an evidenced environment setup defect, not a
model result. Original source, logs and failure completion remain unchanged.

CPU bootstrap **v2**, job **1653317**, reuses the exact acquired source bytes after SHA-256
verification and creates a new isolated environment. It includes both existing RADFM
runtime dependency paths, leaving the shared runtime untouched. Its root is
`/beacon-projects/radfm/wy891/fin-skills-campaign-reuse-bootstrap-20260923-v2`.
No model inference occurs in either bootstrap job. Inspect `completion.json`,
`qualification.json`, `upstream-manifest.json`, install report and resolved requirements.
Never resubmit an accepted job or modify its frozen source.

V2 is now **COMPLETED / exit 0**, elapsed 71.756 seconds. Its dependency/tool check passed;
all **883** development reference programs agree with their official execution answers.
Stripped input inventories are train 6,251, dev 883 and public test 1,147. FinanceBench
contains 150 public labeled questions. These are acquisition/calibration counts, not
model accuracy. The original failed v1 record remains intact.

The bootstrap strips FinQA input down to ID, question, pre-text, post-text and table.
Programs, execution answers, supporting-fact labels and retrieved gold material remain
outside public-input files. It runs the official numerical executor on known arithmetic
and checks all development references for discrepancies without changing labels.
FinanceBench metadata/labels and Reflexion Python files are staged; full FinanceBench PDFs
and Reflexion's non-Python prompt dependencies are not yet claimed to be staged.

## FinQA / actual LlamaIndex agent calibration protocol

Prepared source: `benchmarks/agent_study/finqa_reuse.py`; worker:
`scripts/beacon_finqa_qualification.py`. Bundle:
`runs/beacon-campaign-finqa-qualification-20260923-v1`.
After successful v2 qualification, GPU job **1653327** was accepted at 05:24:14 UTC.
Receipt: `submission-attempt-02/deployment-receipt.json` under the local bundle. The
first transport attempt timed out before authentication/staging; remote root absence and
absence of queue/accounting records were verified before the separate attempt. Both local
attempt histories remain. Do not submit again. Remote root:
`/beacon-projects/radfm/wy891/fin-skills-campaign-finqa-qualification-20260923-v1`.
At 05:26 UTC the actual ReAct/FunctionTool scripted software check had passed and Qwen
had produced its first development episode; the two-model calibration was still running.

V1 subsequently **COMPLETED / exit 0** in 372.688 seconds. Each model retained all four
planned dev attempts; neither produced a scored final program (0/4 each). Qwen exhausted
the six-call budget on all four; Mistral did so on three. This is not a successful agent
quality result. A fourth Mistral case (`PM/2017/page_38.pdf-1`) failed on call 3 with an
actual adapter `TemplateError`: native ReAct parsing recovery supplied roles
`system,user,user,assistant,user,assistant,user`, incompatible with the official strict
alternating-role template. This is distinct from the model's invalid program/format output.

The implementation fix joins only adjacent same-role messages using two newlines,
preserves every original text (including repeated report text), and logs original messages,
template messages and their index mapping. It changes neither framework parser nor prompt,
answers, budget, model revision or numerical executor. An additional scripted native parser
recovery test must pass before inference. The repaired **v2** root is
`/beacon-projects/radfm/wy891/fin-skills-campaign-finqa-qualification-20260923-v2`;
job **1653408** was accepted at 05:32:50 UTC. The same fixed eight dev attempts are engineering
requalification, not extra independent test tasks or a best-of-two score. V1 remains intact.

Both previously pinned model revisions run sequentially in one dynamic GPU allocation.
Four dev examples are chosen before inference by ascending SHA-256 of
`finqa-dev-calibration-v1|<id>`; the same examples go to both models. There are eight
planned model/task episodes. Each permits at most six responses of 512 output tokens,
32,768 total prompt-plus-output tokens and eight framework iterations. Temperature is
0.1; per-case seed is 11 + 100 * case index. There is no silent input truncation.

The installed `ReActAgent` controls the loop and its native ReAct parser/retry messages.
`FunctionTool` calls the unchanged official FinQA program executor on the supplied table;
it cannot read gold answers. A local Transformers chat adapter is the only LLM backend.
The official chat templates and fixed offline model caches are used. Framework-native
format handling differs from the old synthetic strict-JSON loop; scores cannot be merged.
The final answer protocol remains exactly a JSON object with a `program` string.

A scripted two-response software test first verifies real ReAct dispatch, real tool
execution and final-answer handling. It is expressly not LLM performance evidence.
Real inference records all messages, responses, model-call errors and tool receipts.
Only after an immutable inference receipt exists does a separate process load labels
and invoke the original execution/program-equivalence functions. Every failed or
unscorable planned attempt stays in the denominator; scorer errors are reported separately.
Same-user file separation is not an adversarial sandbox or proof against contamination.

This is development calibration of the reused runtime, not a held-out paper result and
not yet a FinSkills treatment. Low model quality alone does not authorize prompt hunting,
extra attempts or edits to the benchmark labels. Actual implementation failures require
evidence and a new frozen directory. After this passes, freeze the formal paired task
selection, framework/FinSkills integration, ordinary-memory controls and cost protocol.

## Next authorized work

1. Verify both bootstrap and real agent calibration, including actual framework/tool usage.
2. Reuse FinQA official grading for an independently authored task evaluation; prevent
   company/report overlap between experience acquisition and evaluation, disclose public
   data exposure and retain all reference disagreements.
3. Stage full FinanceBench documents from the same source revision, distinguish public
   evidence labels from actual searchable corpus, and add mature retrieval plus BGE.
4. Bring in Reflexion's required prompt assets and original update flow as a clearly
   labeled financial adaptation. A copied prompt alone is not an executed framework.
5. Advance the remaining DocFinQA and suitable financial-framework comparisons in the
   reuse plan. Do not replace original model revisions or expand seeds to seek wins.

Jev awaits its genuine key; other public-data work proceeds. All execution/downloads,
analysis, outputs, temporary files and caches stay on Beacon/RADFM. Slurm allocation is
dynamic, with no nodelist/exclude. Automatic follow-up is active again, notifying only
on material progress, failure, completion or required input. User manuscript edits and
previous GitHub/Overleaf synchronization are not overwritten.

## Full FinanceBench corpus preparation

CPU job **1653365** was accepted at 05:28:43 UTC. Root:
`/beacon-projects/radfm/wy891/fin-skills-campaign-financebench-corpus-20260923-v1`;
local receipt: `runs/beacon-campaign-financebench-corpus-20260923-v1/deployment-receipt.json`.
This is preparation, not a completed retrieval experiment. Its frozen worker is
`scripts/beacon_financebench_corpus.py`.

The pinned repository tree lists 368 PDFs totaling 705,172,379 bytes. The prospective
corpus is **every PDF in that tree**, not the annotated pages or a gold-selected subset.
Each download must match upstream Git blob identity and records SHA-256. pypdf extracts
all pages with zero-based PDF numbering; each document gets a bounded subprocess and
explicit failure/timeout status. No OCR, gold-text replacement or silent denominator
reduction is allowed. Public question inputs exclude answers, justification and evidence.

The job resolves and freezes pypdf/BM25S package versions before isolated installation,
records installation hashes, and executes the real BM25S index/retrieve API on a tiny
software fixture. It also stages the pinned Reflexion few-shot asset missing from the
first bootstrap. These checks do not establish model, retrieval or memory effectiveness.
Subsequent comparison must use the same complete extracted corpus/chunks/query inputs,
report page-evidence retrieval separately from answer correctness, and preserve every
missing PDF, page extraction error and unanswered public question.

At 05:31 UTC all 368 download attempts had completed and page extraction was running.
BM25S **0.3.11** and pypdf **6.19.0** were fixed before installation; the actual BM25S
index/retrieve software fixture passed. The final corpus receipt still requires checking.

## FinanceBench paired retrieval: submitted with a corpus dependency

CPU job **1653394** was accepted at 05:31:48 UTC with `afterok:1653365`. It must not be
resubmitted. Root:
`/beacon-projects/radfm/wy891/fin-skills-campaign-financebench-retrieval-20260923-v1`.
The local bundle has its plan, frozen source and deployment receipt. It contains unchanged
copies of the actual `fin_skills.rag` source, not a renamed substitute retriever.

All 150 public questions are evaluated against the same entire extracted corpus. Both
APIs share 1,200-character chunks/200-character overlap, FinSkills tokenization, unique
query terms, k1=1.5 and b=0.75. BM25S uses its Lucene implementation with float64 scores;
an arithmetic fixture must confirm the expected common 2.5 scale difference before the
benchmark. Native top-k tie behavior remains and is disclosed. There is no gold-document
filter. Public input `doc_name` is not used in retrieval.

For each question and API, one warmup precedes three timed runs, interleaved in seeded
random order on the same CPU allocation. API return materialization is included; index
construction and shared preprocessing are reported separately. Save top 50 chunks before
loading evidence labels, then report any-page hit, all-pages hit and page recall at
1/5/10/20/50. All 300 method-question units and missing corpus/failed queries remain in
the denominator. This evaluates evidence-page retrieval and component costs, not answer
correctness, human efficiency or the whole library's superiority.

## Follow-up verification and bounded repairs (05:44 UTC)

FinQA qualification **v2 / 1653408 completed**, elapsed 372.067 seconds. The read-only
auditor `scripts/audit_finqa_qualification.py` verified frozen source/input hashes, all
eight episode receipts, message-normalization mappings, budgets, numerical tool receipts
and exact recomputation of the original scores. Remote `qualification-verified.json`
SHA-256: `1c7838ccdea9934e93af2d3d63c454195c1d146371d5763e689ee07ee18beaf3`.

| Model | Planned | Correct execution/program | Generated responses | Input/output tokens | Returned tool receipts | Episode seconds |
|---|---:|---:|---:|---:|---:|---:|
| Qwen | 4 | 0 / 0 | 22 | 47,083 / 2,247 | 21 | 111.164 |
| Mistral | 4 | 0 / 0 | 24 | 51,621 / 2,228 | 24 | 90.440 |

The template failure is gone. One Qwen final JSON passes the output-format check but scores incorrect;
the other seven attempts end at the six-call cap without a final answer. These remain
development failures. Tool counts above mean executor receipts, not every attempted
framework dispatch. No further prompt/parser/budget changes are justified by this low
score. V1 and v2 are separate engineering attempts, not independent task evidence.

FinanceBench corpus v1 **1653365 remains running**. Its extraction logs expose an
environment defect on AES-encrypted PDFs, including Adobe reports:
`pypdf.errors.DependencyError: cryptography>=3.1 is required for AES algorithm`.
Before any retrieval started, job **1653394** was verified PENDING/Dependency, with no
qualification/completion output, and canceled. Its remote
`dependency-cancellation.json` and `dependency-cancellation-result.json` preserve the
reason, prior state and successful cancellation; original submission is untouched.

Corpus **v2 / 1653479** was accepted with `afterok:1653365`. Root:
`/beacon-projects/radfm/wy891/fin-skills-campaign-financebench-corpus-20260923-v2`.
It creates another isolated environment, freezes a cryptography version before install,
checks an AES primitive, and reuses original PDF bytes and successful page files only
after hashes match. Only failures with the explicit AES dependency exception are retried.
Other missing/failed documents remain failures; all old logs remain. This does not alter
extraction settings, pypdf/BM25S versions or labels.

Retrieval **v2 / 1653482** was accepted with `afterok:1653479`. Root:
`/beacon-projects/radfm/wy891/fin-skills-campaign-financebench-retrieval-20260923-v2`.
It retains the same comparison code/design and exact frozen FinSkills source; only corpus
and environment paths change. Check completion/qualification before reporting results.

## Reused reranker and memory implementation

BGE preparation **1653548 completed** in 53.902 seconds. Root:
`/beacon-projects/radfm/wy891/fin-skills-campaign-bge-prepare-20260923-v1`.
The fixed `BAAI/bge-reranker-v2-m3` revision is
`953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`; all seven downloaded files have hashes in
`model-receipt.json`. Tokenizer/config qualification passed; staging did not load weights
for inference or produce reranking scores.

Actual BGE reranking **1653602** is accepted with `afterok:1653482`. Root:
`/beacon-projects/radfm/wy891/fin-skills-campaign-financebench-rerank-20260923-v1`.
The prospective design follows the author's Transformers sequence-classification recipe:
float32, batch 8, maximum 512 query/passage tokens, raw logits, original-rank tie break.
It takes each API's frozen top 50 candidates for all 150 questions, preserving the same
candidate sets before/after reranking. It records untruncated lengths, actual/padded
tokens, truncation counts, wall time and GPU memory. All 300 candidate sets remain in
the denominator; labels load only after a rerank receipt. Report candidate ceiling and
evidence-page recall separately from answer quality. This is a BGE baseline, not Jev.

Memory preparation **1653591 completed**, without LLM inference. Root:
`/beacon-projects/radfm/wy891/fin-skills-campaign-memory-reuse-prepare-20260923-v1`.
It executes the unchanged author's `update_memory` and `EnvironmentHistory` from pinned
Reflexion, using an explicitly scripted completion dependency for the software fixture.
Checks cover unsuccessful-only updates, last-three-plan reads, success/skip behavior and
fresh history lists. The original ALFWorld few-shot asset is staged and hashed; this
does not claim a completed financial memory baseline or cross-task learning.

Using stripped FinQA inputs only, the same job froze **16 acquisition questions from
8 training companies** and **32 evaluation questions from 16 different test companies**,
one question per company/year report, 48 reports total. The three calibration companies
are excluded. Selection uses prospective SHA ordering, not labels or model outcomes.
Input hashes: acquisition `7b83d7c5a0abf2fb3e82bc3de981645d2d727eb4de1af072f6cbe4e0672d4cd7`;
evaluation `5ccb41f80b933c8f68df158e9df74b84c156b714a58d99e3f5ed5181c9393719`.
Task IDs are now fixed; the financial memory-arm protocol and LLM execution are still
to be implemented/frozen. Public exposure and similar question templates remain limits.

## Five-arm external memory study frozen and submitted (06:02 UTC)

The preceding preparation status is superseded by
[FINQA_MEMORY_PROTOCOL_20260923.md](FINQA_MEMORY_PROTOCOL_20260923.md) and
`benchmarks/agent_study/finqa_memory_study.py`. The fixed arms are none, recent experience
with Reflexion text, lexical retrieval of experience, retrieval plus an unconditioned fly
readout, and the identical retrieval plus an acquisition-conditioned fly readout. Recent
means the last three acquisition records, including successful records with no reflection;
it is not presented as the original ALFWorld same-task evaluation. All records include
only their past question, attempted program, boolean/error feedback and generated plan.

The original Reflexion update function generates at most one real 512-token reflection
for each failed acquisition attempt. A fixed financial-transfer wrapper is logged; the
author's examples and update implementation remain. No test feedback is visible during
evaluation. The circuit cue mapping (scalar versus table program intent) and correctness
reward are explicitly engineering adaptations. Both circuit arms read the same retrieved
text, and all arms use the same model/tool/response caps. No model replacement or prompt
tuning was made in response to the earlier 0/4 development results.

CPU qualification **1653758 completed** in 39.397 seconds. Real ReAct/FunctionTool and
native parsing recovery passed, all five payloads reached the model interface, memory reads
were nonmutating, the frozen circuit stayed unchanged and the conditioned circuit changed.
The native Reflexion update executed. These are software checks, not LLM quality results.
Qualified parameter SHA: `cb7b555026f97b59678adf8902b9f4a02975413a7eed9d6e03f2e2864821915d`;
prior equation-audit SHA: `3f2286c93e1f1fcd12d90cc6c009f5206ed45112e7d9eca1469e3f33783c0d08`.

Initial production Qwen **1653783** and Mistral **1653784** failed before acquisition,
after passing the software gate. Transformers/Hugging Face had initialized its cache
constants before the Python worker changed cache environment variables; offline model
lookup therefore searched the new campaign cache rather than the existing pinned weights.
Both failure completions/logs are retained. Their roots have no acquisition or evaluation
episodes, so all planned units are recorded as not started, not model mistakes.

Verified the pinned snapshot config files exist. New launchers export each correct cache
before Python starts; **all study source hashes are unchanged**, verified remotely against
the v1 manifests. Accepted replacements:

- Qwen **1653915**, root `fin-skills-campaign-finqa-memory-qwen-20260923-v2`.
- Mistral **1653932**, root `fin-skills-campaign-finqa-memory-mistral-20260923-v2`.

Each root is under `/beacon-projects/radfm/wy891`; local `runs/beacon-campaign-...`
deployment receipts preserve the accepted submissions. Do not redeploy. Each job has
16 shared acquisition attempts and 160 evaluation episodes (32 tasks × five arms),
for 352 episodes across both models. Acquisitions are charged once per model. All ranks,
raw/template messages, reflections, feedback receipts, circuit transitions, memory reads,
costs and missing/error outcomes must be retained. A job submission is not task completion.

`scripts/audit_finqa_memory.py` is uploaded to each v2 root's `analysis/` directory and
passed syntax checking on Beacon. Run it only after full completion. It verifies source,
all planned tasks, receipt hashes, acquisition-only feedback, exact model-visible memory,
nonmutating bank state, replayed circuit values and original numerical/symbolic grades.
Only read-only circuit float replay has tolerance (rtol 1e-10, atol 1e-12); original grade
booleans are exact. It writes `memory-verified-summary.json` exclusively. No final audit
or memory-effectiveness result exists yet.

Subsequent v2 logs confirm both models loaded successfully and entered acquisition:
both wrote the first two immutable episode/receipt pairs, with completed acquisition
progress messages. The cache repair is therefore verified past the former failure point.
This is startup/progress evidence, not an evaluation score. FinQA evaluator messages such
as `structure error` on generated programs remain model/program failures unless a separate
execution exception provides evidence of an implementation defect.

## 06:18 UTC: verified corpus repair and DocFinQA preparation

FinanceBench corpus v1 job **1653365 completed** in 2,623.509 seconds: all 368 PDFs
downloaded and matched the pinned upstream Git blobs; 362 extracted, six failed. The
independent `scripts/audit_financebench_reuse.py` checked source and receipt hashes,
every PDF blob/SHA, page-file hashes, page numbering and all 150 public inputs. There
were 53,527 pages, including 53,312 nonempty pages. The original six failures remain.

The previously accepted AES repair job **1653479 completed**. Its independent audit
confirmed byte-identical reuse of the 362 successful extractions and recovery of four
Adobe PDFs using the evidenced dependency repair. The final corpus has **366/368**
extracted PDFs, **53,901** pages and **53,686** nonempty pages. The two Intel 8-K PDFs
remain failures: pypdf reports a missing EOF marker and a prematurely ended stream.
Their bytes match the upstream Git blobs, so this is not evidence of a transport error;
no replacement source, OCR or gold-page substitution was introduced.

Both roots contain exclusive `reuse-verified-summary.json` files. Their corpus receipt
hashes are `3d9dc7e954df2ccb091b4bcfb94fb742c94d3da1ed0d0a3421663636bda54b45` (v1)
and `38a59aac5ca692cec031c3776a786bf777d59c1f261bef1a18b0f16ed7b9ff06` (v2).
Auditor SHA: `6d1691f36fb1cd6f98dd9559ab237d9d70e4ee23570f910cb203b7e017bee3da`.
The same auditor is staged under the retrieval-v2 and BGE-rerank roots' `analysis/`
directories; run it only after each root completes. It verifies all 300 method-query
units, original grades, candidate correspondence and reported costs. It does not rerun
or tune retrieval. Retrieval **1653482 is running**; BGE **1653602** retains its dependency.

DocFinQA preparation **1654193** was accepted at 06:13 UTC and is now running, root
`fin-skills-campaign-docfinqa-prepare-20260923-v1`. The local deployment receipt is in
the corresponding `runs/beacon-campaign-...` directory. The prospective specification
is [DOCFINQA_PREPARATION_20260923.md](DOCFINQA_PREPARATION_20260923.md). The author's
dataset revision is `64ebaff62f692495bcc182f45cf9a9606251b19b`; all three source-file
sizes and LFS SHA-256 values were pinned before submission. This CPU job isolates the
original Program/Answer labels, deduplicates exact report text, retains ambiguous FinQA
question matches, and probes the two fixed tokenizers without inference or truncation.
Dev (780 rows) and test (922 rows) were downloaded, verified and isolated; train was
downloaded and verified at the last check. These are preparation counts, not new scores.

FinQA memory jobs **1653915/1653932** continue without a new execution failure.
At the latest inspected logs Qwen had completed 13/16 acquisition attempts and Mistral
had completed 16/16 plus 4/160 evaluation episodes. No complete memory-effect estimate
is available. Original parser/program errors and response limits remain in the results.

DocFinQA **1654193 subsequently completed** in 251.958 seconds. The full pinned corpus
has 5,735 train, 780 dev and 922 test rows (7,437 total), with 885 distinct exact context
strings. Source and derived-artifact hashes and label-free public-input schemas were
checked again in `preparation-verification.json`. Corpus receipt SHA:
`a5b6b0a5ae4218228712f47df590cea11a212e62cfb77367f7974de89e91ac35`.
Question matching finds candidates for 43 of the existing 48 fixed FinQA IDs; the other
five remain missing. There are 211 exact context hashes shared across original splits
and 24 normalized question hashes shared across splits. Original split membership alone
therefore cannot establish independent report exposure.

All 51 prospectively selected capacity probes exceed the current 32,768-token ceiling
with a 3,072-token reserve, for both models. Qwen input tokens min/median/max are
56,882 / 137,747 / 333,157; Mistral 64,670 / 154,996 / 374,980. These are minimal-template
tokenizer measurements, not inference or performance results. The length warning is
expected from this deliberately untruncated capacity probe. No weights were loaded.
Future HiSTrim work must define a common retrieved-input condition separately from full
reports; the current fixed model budget cannot support the latter. Do not silently
truncate, select only shorter reports, or change the model's configured limit and call
that a valid full-context baseline. Preserve the five missing prior-task mappings and
resolve ambiguous report mapping before any independent-task claim.

## 06:28 UTC: DocFinQA context protocol frozen; submission capacity unavailable

[DOCFINQA_CONTEXT_PROTOCOL_20260923.md](DOCFINQA_CONTEXT_PROTOCOL_20260923.md) now
specifies a bounded context-by-memory comparison. It retains all 32 previously fixed
evaluation tasks, including missing/ambiguous mappings, and shares eight report-local
BM25 chunks across full-pack, recency and HiSTrim conditions. Each is crossed with no
memory or the existing acquisition-conditioned memory package. These are 384 planned
units across two models, not 384 independent questions. One scalar-program response is
scored with the original FinQA executor and an empty table; no gold table is supplied.
This is a separate one-pass retrieved-text experiment, not the six-response ReAct study.

`benchmarks/agent_study/docfinqa_context.py` implements label-free input preparation and
GPU interface qualification. It checks unambiguous same-split question mapping, exact
report overlap against acquisition, shared chunk offsets, acquisition-bank receipts and
read-only circuit replay. Official-template token counts preserve the existing explicit
decoder's **8,192-token** input-plus-512-output ceiling. The router calibration prompts,
threshold fitting, pruning budgets and beta are unchanged from the original implementation.
The production inference/scoring runner is still to be completed after these interfaces
are qualified; no formal context-effect result or GPU qualification result exists yet.

CPU bundle `runs/beacon-campaign-docfinqa-packs-20260923-v1` was transported and its
hashes and Python syntax checked on Beacon. Its **06:27 submission was explicitly
rejected with AssocMaxSubmitJobLimit; no job ID exists**. Both original local deployment
receipt and remote submission-started/submission-receipt files remain. The remote root
is `fin-skills-campaign-docfinqa-packs-20260923-v1`; it has no completion or run outputs.
The study module SHA is `9f7d4121ad6d82e645afeea2a0a02c92c3a05960b756ec0e01861bab521461cb`.
Do not run the deploy helper again on this root. After independently confirming queue
capacity and no acceptance/start, an additional attempt may use the original frozen
job.sh and source with a separate submission-attempt directory and receipt.

The matching GPU bundle `runs/beacon-campaign-docfinqa-context-qualification-20260923-v1`
is prepared locally but **never submitted**. It sequentially qualifies Qwen and Mistral
in one dynamic allocation. Only submit it once the CPU gate has passed and capacity is
available; add an afterok dependency on the actual accepted CPU job if appropriate.
It produces per-family completion/qualification files, not a single root completion.

The account currently has 11 Beacon-association jobs, so no additional submissions are
attempted. Other newly present finance/RAG jobs are outside this follow-up and must not
be cancelled or modified. At the 06:22 read, both memory jobs were in evaluation
(Qwen 17/160, Mistral 24/160); retrieval had 16/150 question records and no new execution
error. Existing accepted work continues. These are progress counts, not final scores.

## 06:41 UTC: DocFinQA runner and audit prepared, no additional submission

`benchmarks/agent_study/docfinqa_run.py` now implements the formal one-response loop,
independent scoring process and full-denominator audit. It loads the qualified routers
without refitting, verifies exact prepared prompt hashes, uses the same task seed across
all six conditions, and records input failures even when no generation is attempted.
CUDA OOM aborts as an execution failure rather than being converted to an incorrect
model answer. Immutable inference receipts precede any test-label read. The audit checks
all 192 units per model, original numerical grades, memory identities, prompt/token hashes,
protected positions, nested selection, actual costs and retained item types.

Runner SHA: `42a2fc2dcbc2c941e787cf0255e60a44359859fbe2e5fc3ff3f25ff6056a010c`.
Beacon AST parsing passed. Software fixtures have **not yet run**; they cover correct
and wrong answers, the original five-decimal rounding, comparisons, strict JSON, rejected
table operators, division by zero and missing inputs. They are included in the still
unsubmitted GPU qualification job before model loading. No deployed preparation source
or earlier protocol was changed.

Two local production bundles are prepared but **never submitted**:
`runs/beacon-campaign-docfinqa-context-qwen-20260923-v1` and the matching
`...-mistral-20260923-v1`. Each launches inference, separate scoring, then auditing.
Their common helper/decoder/runner hashes must match the actual GPU qualification
manifest. Do not submit either before CPU preparation and GPU qualification pass.
The qualified memory banks remain acquisition-only; this adds no evaluation feedback.

The Beacon association remains at 11 submitted jobs, so the rejected CPU preparation
was not retried and no extra GPU job was submitted. At 06:36, memory evaluations had
52/160 Qwen and 59/160 Mistral records, and retrieval had 55/150 question records.
There was no new execution failure and no completed aggregate result to report.

## 06:54 UTC: DocFinQA preparation passed; GPU qualification running

After the account dropped below its submission limit, the original rejection, absence
of an accepted job in queue/accounting, unchanged source hashes and empty execution
outputs were checked. The frozen CPU preparation was accepted as **1654668** at 06:52
UTC. Its new receipt is under remote `submission-attempt-02/` and local
`runs/beacon-campaign-docfinqa-packs-20260923-v1/submission-attempt-02-receipt.json`.
The original rejection and submission-started files remain unchanged. Do not submit again.

CPU preparation completed in **24.043 seconds**. The original scalar executor fixtures,
memory-bank receipt checks, read-only circuit replay and prepared-input checks passed.
Each model has 64 task/memory payloads covering 32 questions. There are 54 valid payloads
and ten unavailable payloads: eight missing/multiple DocFinQA mappings and two ambiguous
or cross-split mappings. Thus 27 questions are runnable; all five unavailable questions
remain in every condition. The formal denominator remains 192/model, including 30
input-failure units/model. Maximum offered input tokens are 5,785 Qwen and 5,759 Mistral;
no valid retrieved input exceeds the unchanged 8,192 total-token ceiling.

Payload SHA values: Qwen `4eafcc8c6109afe5a5bb948cd9bd1bff1cf42c324d21b93dc2d6d7f75b46abe0`,
Mistral `2818aa6c391e7c71958f05171f67b3f1bfddfbec46d34c84c4b3de1d3a58ccfd`.
These preparation checks are not evidence of model correctness.

GPU qualification **1654671** was accepted at 06:53 UTC and is running in one dynamic
allocation. Runner software fixtures have now passed; model/decoder/physical-cache gates
are still pending. `scripts/audit_docfinqa_qualification.py` is uploaded under its
`analysis/` directory. Run it once both family completion files and scheduler completion
are available. Only a passed audited qualification permits the two already prepared
production bundles to be submitted. Neither production job has been submitted yet.

## 07:00 UTC: qualified DocFinQA production accepted

Qualification v1 **1654671 completed** and passed the decoder/physical-cache gates.
Before submitting production, static inspection found that the runner attempted
`DualRouter.from_dict`, which the existing class does not define. No production task had
run. The fix uses the class's actual constructor with the saved fields and adds an
explicit save/restore software fixture. Old snapshots were retained; no parser, prompt,
budget, router threshold or scoring rule changed.

New qualification v2 **1654701 completed** in 56 scheduler seconds. The added restoration
fixture, original scoring fixtures, both models' manual and real-input decoder checks,
protection/nesting and physical KV checks passed. The independent qualification audit
is `fin-skills-campaign-docfinqa-context-qualification-20260923-v2/qualification-verified.json`,
SHA `d77538af97aa4ad95dc321004dbae8aecf43091cd5b92e54d2c4b693e523c9ae`.
Routers are identical to v1: Qwen SHA
`de2bf96987102d57cba4e063622e3148bf0a70f8fbcfca132c6eb14857b0b6cd`, Mistral
`f7033c4cf3854f4ed3999538cd4a5915f91220d47e7c68d1c953487f49be2755`.
This is implementation qualification, not answer-quality evidence.

Both formal jobs were accepted at 06:59 UTC, using dynamic allocation:

- Qwen **1654705**, root `fin-skills-campaign-docfinqa-context-qwen-20260923-v2`.
- Mistral **1654706**, root `fin-skills-campaign-docfinqa-context-mistral-20260923-v2`.

Each local corresponding `runs/beacon-campaign-.../deployment-receipt.json` records
acceptance. Do not deploy again. At 07:00 Qwen was running with weights loaded; Mistral
was pending. Each plans 192 units, including 30 known input-failure units; the full
384-unit denominator is preserved. Check episodes, inference receipt, independent scores,
infer/score completion files, overall completion and `context-verified-summary.json`.
The audit is already the final command in each job. If only auditing fails, diagnose the
audit without rerunning completed inference or changing original scores. The prepared
production v1 bundles were never submitted and have local superseded markers.

Older ongoing work at this check: FinQA memory Qwen 111/160 and Mistral 110/160 evaluation
episodes; FinanceBench retrieval 117/150 queries. These remain partial progress counts,
not aggregate effects. No new execution failure appeared in their logs.

## 07:08 UTC: first DocFinQA model complete and audited

Qwen **1654705 completed**, including its automatic 192-unit audit. Summary SHA
`211dcb46e93fe67adb8218659bb596e32f80465679404fe66fc0d082d0963689`; source, inference
receipt and score hashes were checked again. Full outcomes and costs are recorded in
[DOCFINQA_CONTEXT_RESULTS_20260923.md](DOCFINQA_CONTEXT_RESULTS_20260923.md).
In none/full, none/recency, none/HiSTrim, conditioned/full, conditioned/recency and
conditioned/HiSTrim order, correctness is **10, 1, 0, 4, 3, 0 out of 32 each**. Every
condition includes five input failures and 27 responses; generation/scorer exceptions
are zero. Both HiSTrim conditions retain zero final-depth document items, consistent
with a substantial quality loss under this fixed calibration. Earlier-layer influence
is not excluded. Keep the negative results and do not tune routers after observing them.

Mistral **1654706 is running**, now dynamically allocated after Qwen finished. At the
07:06 check the memory jobs had Qwen 127/160 and Mistral 121/160 evaluation records;
FinanceBench retrieval had 135/150 queries. No other batch had a completion receipt.
Do not merge the single completed DocFinQA model with partial results or rerun its audit
over the existing exclusive summary.

## Final submitted-batch completion and audits

All remaining submitted jobs are COMPLETED/exit 0: memory **1653915/1653932**,
retrieval **1653482**, BGE **1653602**, and both DocFinQA **1654705/1654706**.
The four pending memory/retrieval audits were run on Beacon and passed without changing
the auditor, original scores or inference. Both DocFinQA automatic audits passed;
Mistral source/receipt/score hashes were independently rechecked. No job was resubmitted.

- [FinQA memory](FINQA_MEMORY_RESULTS_20260923.md): all 352 planned episodes audited.
  Per-arm evaluation counts (none/recent/retrieval/frozen/learned, denominator 32) are
  Qwen 3/7/5/6/6 and Mistral 0/0/0/0/0. Learned conditioning does not exceed frozen
  memory on these aggregate counts. Model/protocol failures remain in the denominator.
- [DocFinQA](DOCFINQA_CONTEXT_RESULTS_20260923.md): all 384 planned units audited,
  including 60 input failures. None-full/recency/HiSTrim and learned-full/recency/HiSTrim
  correct counts are Qwen 10/1/0/4/3/0 and Mistral 1/0/0/4/0/0, each out of 32.
  This fixed HiSTrim implementation has no observed correctness gain on this slice.
- [FinanceBench](FINANCEBENCH_REUSE_RESULTS_20260923.md): 300 retrieval and 300 BGE
  units audited. Both APIs have identical aggregate page hits; BGE raises top-1 hits
  from 9 to 14 out of 150, within a top-50 ceiling of 28. Median query latency is
  5.503675 seconds for FinSkills versus 0.004404 for BM25S, a negative speed result.

Immutable audit hashes and actual costs are listed in those result documents. Repeated
conditions, overlapping FinQA/DocFinQA questions, public-label exposure and the two failed
upstream PDFs remain explicit. The 2,112 older decision-loop episodes are unchanged.
No inference workload from this submitted batch remains. Do not expand it for better
scores. Full financial-system baselines and remaining external-input requirements must
be assessed separately; this completion is not completion of the entire research plan.

## 07:43 UTC: bounded mature-framework tool control submitted

The scope review found a concrete remaining P0 control with available inputs: previous
LlamaIndex runs exposed only the FinQA calculator, while multi-method organization was
tested in a hand-written loop. The new fixed comparison uses actual LlamaIndex tools on
the same 32 authored contracts, three catalog conditions and the same two fixed models.
See [FRAMEWORK_SCOPE_REVIEW_20260923.md](FRAMEWORK_SCOPE_REVIEW_20260923.md) and
[FRAMEWORK_TOOLS_PROTOCOL_20260923.md](FRAMEWORK_TOOLS_PROTOCOL_20260923.md).
It is not new external evidence, a whole-library treatment or a FinRobot reproduction.
All arms receive identical rules and algorithms. Additional domain text is repeated,
not hidden new knowledge. No additional seeds or result-dependent changes are planned.

CPU qualification **1655240 completed** in 43.263 worker seconds (49 scheduler seconds):
all 32 numerical references, three actual-framework scripted tool calls, strict-final
and incorrect-receipt rejection fixtures passed. All 29 copied task/algorithm/model
files match the old decision-loop source bytes. Its source manifest and fixture/receipt
audit passed; `qualification-verified.json` SHA is
`dc1937ca4271b9dd6095789de71f08d09f022c07a8ace6209d0296e9663dce59`.
New `framework_tools.py` SHA is
`f2dc56cad6040612bbed214c459d6deb5f2ed5762e85215aa7d3c98e07c54888`.

Two production jobs were accepted and are running with dynamic placement:

- Qwen **1655292**, `fin-skills-campaign-framework-tools-qwen-20260923-v1`.
- Mistral **1655293**, `fin-skills-campaign-framework-tools-mistral-20260923-v1`.

Both roots are direct children of `/beacon-projects/radfm/wy891`; corresponding local
`runs/beacon-campaign-framework-tools-.../deployment-receipt.json` files record acceptance.
Each has 96 planned cells, total 192. Deployed source hashes match qualification.
Do not submit again. Check completion, frozen inference receipt, independent scores and
automatic `tools-verified-summary.json`; if only the audit fails, fix a separate auditor
after diagnosing it, without rerunning inference or changing grades. Framework output
errors remain outcomes. Costs and old handwritten-loop scores cannot be pooled because
the new interaction protocol has different maxima. Previous completed batches are intact.

## 07:51 UTC heartbeat: production progressing

Both framework-tool jobs remain RUNNING with no completion receipt. Log checkpoints
showed Qwen 26/96 and Mistral 16/96; the subsequent read-only episode check found
28 and 19 completed episode files respectively. All deployed source hashes and the
96-cell unique order per model passed checks. No infrastructure failure appeared.

Observed partial failures are model/interface outcomes: Qwen repeated missing required
parameters; Mistral sometimes chose a mismatched tool or omitted its parameter wrapper.
Inspected original responses support these classifications. The fixed call limit was
reached in some episodes. Do not supply missing parameters, loosen parsing or restart
these attempts. No correctness aggregate or final summary is reported before completion.

## 08:06 UTC heartbeat: fixed matrix still running

Jobs 1655292 and 1655293 remain RUNNING, each at 23:05 scheduler elapsed. Logs show
Qwen 90/96 and Mistral 66/96 completed episodes. Neither has a completion receipt or
final audit summary yet; stderr contains only the previously known loading warnings.
No new infrastructure failure, submission, source change or partial-score summary.

## 08:21 UTC: final framework matrix complete; bounded campaign closed

Qwen **1655292** and Mistral **1655293** both completed/exit 0, with 96/96 cells each
and successful automatic audits. Frozen source/qualification equality, receipt, episode
and score hashes were checked again. No inference or audit summary was overwritten.
Worker elapsed times are 1,418.098 and 1,991.299 seconds; scheduler times 23:45 and 33:15.

Generic/flat-guidance/organized totals are Qwen **25/28/26** and Mistral **10/13/3**,
each out of 32. The corresponding later-phase counts are Qwen **11/14/13** and Mistral
**6/7/2**, each out of 16. All failure and cost counts are retained in
[FRAMEWORK_TOOLS_RESULTS_20260923.md](FRAMEWORK_TOOLS_RESULTS_20260923.md). The organized
condition does not show stable superiority. Recorded framework exceptions are fixed-call
limits, not execution defects. Captured callbacks are not all attempted framework actions.

Summary SHA Qwen `fe65d115d40dceb573930f3c9827723b18864b3d4d087a90bd6858e2c783f8d3`;
Mistral `fa19a2750d4f2fe88598641fee1436fdc7dbeca42e5500d2161bd4dd70862425`.
There are no remaining submitted jobs in this bounded campaign. Its claim-level
conclusions and remaining external-input/task-boundary gaps are in
[BEACON_CLOSEOUT_20260923.md](BEACON_CLOSEOUT_20260923.md). Pause automated follow-up
after this closeout; do not call the entire research plan complete or expand the matrix.
