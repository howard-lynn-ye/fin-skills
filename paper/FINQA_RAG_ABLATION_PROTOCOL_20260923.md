# Prospective FinQA report-QA workflow ablation

This protocol is frozen before the new model runs. User authorization covers Beacon
execution. All source snapshots, outputs, temporary files and caches remain under
`/beacon-projects/radfm/wy891`. Existing runs and negative results are preserved.

## Scope and task units

Reuse exactly the 32 FinQA evaluation questions from 16 companies already frozen in
`fin-skills-campaign-memory-reuse-prepare-20260923-v1`. This is the SAME task set as the
memory study, not 32 additional independent tasks. Inference reads only stripped report
inputs. Official test programs and numerical answers are opened by a separate scoring
process after every planned output has a hash-bound inference receipt. Public benchmark
exposure and unknown model-training contamination remain limitations.

The task is report-scoped financial question answering. Each sentence and each table row
(with its header) becomes a source document. Shared FinSkills chunking uses 1,200 characters
and 200 overlap. Retrieval uses question text only, top five positive BM25 hits, whole chunks
and a 12,000-character context cap. BM25S 0.3.11 is the actual mature component baseline,
with Lucene BM25, k1=1.5, b=.75, float64 and a common ID tie rule. Context agreement is
recorded; small implementation ranking differences are not hidden or corrected from labels.

## Conditions

| Condition | Retrieval / input | Guidance | Reporting policy |
|---|---|---|---|
| components | Actual BM25S | Common task and tool instructions | Exact final schema |
| skills | Same BM25S input | Entire existing fundamental-and-macro-data skill | Exact final schema |
| rag_api | Actual FinSkills RAGIndex / RAGPipeline.prepare | Same frozen skill | Exact final schema |
| rag_gate | SAME frozen rag_api generation | Same as rag_api | Actual RAGPipeline.answer citation-label check |
| full_report | Every source chunk, no retrieval cutoff | Common task and tool instructions | Exact final schema |

The gate is a prespecified **paired post-generation policy ablation**, not an independently
generated interactive agent. It receives the frozen original answer through the public
generator callback; it cannot change the answer or retry. It validates label membership,
not whether cited text entails the claim. A numerically wrong answer with valid labels may
pass and must be reported. Full-report inputs exceeding model capacity fail explicitly.

The chosen skill is frozen in full before inference. It is not optimized for the task using
test scores. Results measure this guidance choice, not ideal skill routing. Shared chunking
means the component baseline reuses preprocessing; this is an API/workflow comparison, not
an independent implementation of the entire library.

## Shared agent and resources

Both existing pinned Qwen and Mistral models use actual LlamaIndex ReActAgent and FunctionTool,
the unchanged official FinQA calculator, six model responses, 512 output tokens per response,
eight framework iterations and the existing 32,768 prompt-plus-output limit. The calculator
can operate on the current report table in every condition; it does not expose reference
answers. This shared tool is not credited to FinSkills. The final schema requires program
and citation strings. No parser relaxation or response-budget tuning follows test results.

Four generation conditions yield 128 model episodes per model; the gate adds 32 paired policy
outcomes but no new model inference. Across two models this is 256 generated episodes and
320 reported policy outcomes, on 32 underlying questions. Conditions are interleaved in a
fixed shuffled order. Each question has the same seed across conditions. Actual token use,
tool receipts, failures, time and peak GPU allocation are retained. Equal caps are not equal
compute; the skill text adds prompt tokens and the full report adds context.

## Endpoints and qualification

Primary endpoints are correct accepted answers and incorrect accepted answers per planned
task. Also report ungated official execution/program correctness, rejection, format and
scorer errors, citation-label validity, actual costs and a reject-all reference. Recompute
original grades from immutable outputs. Never count a valid citation label as semantic
support. Any uncertainty estimate must group questions by company and keep models separate;
the paired gate does not increase sample size. This bounded sample is exploratory, not a
powered claim of small gains.

Before production, a CPU Slurm job must execute actual BM25S, RAG APIs, ReAct and official
calculator on synthetic software fixtures, including a valid and an unknown citation label.
No fixture success is counted as model-quality evidence. Source hashes must match production.

This experiment does not establish whole-library superiority, developer time savings,
point-in-time correctness, document semantic support or fruit-fly model effectiveness.
Those remain separate experiments; the running memory study and FinanceBench jobs continue.
