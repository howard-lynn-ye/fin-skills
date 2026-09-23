# DocFinQA common retrieved-input context study

Prospective specification, 2026-09-23. This extends the same fixed 32 FinQA evaluation
questions to their DocFinQA reports. It adds a document-input condition, not 32 new
independent tasks. Full reports exceeded capacity for every selected probe; none of the
conditions below is called a full-report baseline.

The public question mapping must have one DocFinQA candidate, one FinQA candidate equal
to the fixed ID, and the same test split. Otherwise the task remains a missing/ambiguous
input in every condition. Exact report overlap with the fixed acquisition mappings is
also retained as an unavailable evaluation input. No answer field resolves mappings.
The 32-task denominator remains even when no valid report mapping exists.

Actual FinSkills RAGIndex retrieves the eight highest positive-score chunks from the
mapped report, using the existing 1,200-character/200-character overlap defaults. This is
within the report supplied by DocFinQA, not global document retrieval. The same exact
eight chunks, ordered by source offset, are offered to every condition. No evidence label
filters or reranks them. Store source offsets, hashes, scores, original ranks and timing.

Six paired conditions cross none/acquisition-conditioned memory with full-pack, recency
and HiSTrim. Memory uses the existing model-specific bank of 16 acquisition attempts,
the actual RAGIndex top-three experience retrieval and the existing conditioned fly
readout. It is frozen throughout this study. Bank receipt hashes, replayed circuit values,
selected experience IDs and payloads are recorded. This compares the complete learned
memory package to no memory; it does not isolate the fly's contribution from text, which
is addressed separately by the five-arm memory study.

Memories are separate prunable items, followed by report chunks in source order. The
current question and output instructions are protected. The full-pack condition retains
all supplied items; recency and HiSTrim use the existing 50%/25% maximum historical-token
budgets at the two existing layers. Actual retained tokens may differ. Keep the existing
four manual-relevance calibration prompts, threshold fitting, beta .1, original positions
and physical KV compaction. No new task-answer labels calibrate the routers, and no
threshold is tuned from prior negative results. All three use the same explicit decoder.

The model makes one response of at most 512 tokens containing exactly one JSON object
with a scalar FinQA DSL program. The allowed operators are add, subtract, multiply,
divide, exp and greater. The original FinQA executor evaluates the generated program
with an empty table: no gold table or missing retrieved facts are supplied at scoring.
The output is compared directly with the original FinQA execution answer, using the
executor's unchanged rounding. Table operators, malformed JSON, unsupported operators,
missing inputs, over-capacity prompts and generation failures remain failures with
separate categories. This is one-pass program generation from retrieved text, not the
six-response LlamaIndex tool-use study or the published full FinQA evaluation protocol.
The two studies' denominators and scores must not be merged.

The pinned Qwen and Mistral models each run 32 tasks × six conditions = 192 planned
units (384 total), randomized in one GPU allocation per model with a fixed paired task
seed. The unchanged decoder has an 8,192-token prompt-plus-output ceiling; do not change
it to rescue an oversized input. CPU preparation records actual official-template token
counts for every payload without loading weights. No label read or truncation is allowed
during preparation. Before study inference, verify scalar scoring on authored fixtures,
official/full-pack decoder equivalence and protected/nested/physical-KV behavior on
development inputs. Model correctness in smoke checks is not a software pass criterion.

An immutable inference receipt precedes a separate scoring invocation that reads test
labels. Report every condition's 32 planned outcomes, format/execution/input failures,
exact numerical correctness, actual prompt/output tokens, prefill/total wall time,
allocated/reserved GPU memory, KV bytes and experience/document retention. Whole-item
selection and allocator history limit cost comparisons. Acquisition cost is reused and
reported separately, not billed six times. Public historical exposure, question/report
matching limits and prior poor model performance remain explicit limitations.
