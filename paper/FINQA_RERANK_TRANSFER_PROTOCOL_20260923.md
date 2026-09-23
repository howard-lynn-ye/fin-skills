# Frozen downstream BGE / Kev comparison

This extends the same 32 public FinQA questions used in the report-QA and memory studies.
It adds no independent tasks. It uses the report-QA prompt, exact output schema, actual
LlamaIndex agent, official calculator, full frozen fundamental-and-macro-data skill and
per-question seeds unchanged. Test answers cannot select candidates or ranks.

Freeze the actual FinSkills BM25 top ten positive passages before either reranker. Each
model ranks that same candidate set; select five whole chunks within 12,000 characters.
Compare downstream answers with the previously frozen `rag_api` BM25 condition, separately
for pinned Qwen and Mistral. Every ranking failure remains a planned downstream failure.
There is no fallback to BM25 on a reranker error and no retry from answer feedback.

BGE uses `BAAI/bge-reranker-v2-m3` at
`953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`, float32, batches of eight, 512-token
query/passage pairs with explicit longest-first truncation. Record full lengths and every
truncated pair. Rank raw logits; ties retain original BM25 position.

Kev uses the existing library `KevModel.rerank` on the trusted local Kev-4B checkpoint
`485ace8703592fcf405488b262449990824cfed1`, with Qwen3.5-4B-Base at
`1001bb4d826a52d1f399e183466143f4da7b741b`, bfloat16. Preserve upstream inference,
all response receipts and strict input truncation rejection. This is the open Kev model,
not a claimed run of the hosted Jev API. Each passage receives the existing three-level
support score. Score ties retain the original BM25 position. No tuning of the relevance
prompt, thresholds or temperatures is allowed using these test answers.

Ranking runs in an isolated GPU job before answer generation. Two later GPU jobs each
produce 64 answer episodes (32 questions times two rerankers). BM25 answer results are
reused from the separate report-QA jobs, not rerun until they improve. All answer conditions
share the same response caps; ranking cost and generation cost are reported separately.
The primary endpoint is official FinQA execution correctness per planned task, with
program correctness, errors and actual tokens/time as secondary outcomes. Preserve paired
task-level outcomes, group uncertainty by company and do not multiply the sample size by
methods or models.

This tests report-scoped relevance reranking and numerical answers. It does not establish
global-document retrieval, semantic citation entailment, Chinese coverage, option-order
robustness for multi-option routing, human efficiency, or unseen pretraining contamination.
The earlier 108-query routing regression is separate and retains its original results.
