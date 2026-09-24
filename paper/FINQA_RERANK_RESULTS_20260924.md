# FinQA reranking transfer: completed results

Verified against Beacon outputs on 2026-09-24. This is an execution and automated
scoring audit, not human scientific sign-off. Protocol:
[FINQA_RERANK_TRANSFER_PROTOCOL_20260923.md](FINQA_RERANK_TRANSFER_PROTOCOL_20260923.md).

## Completion and evidence

Replacement job **1660691** completed with exit **0:0** on 2026-09-23 at
**10:01:50 America/New_York** (14:01:50 UTC), after **00:57:52**. Its wrapper
recorded successful Qwen and Mistral subprocesses. Each model retained all **64/64**
planned answer episodes: 32 BGE and 32 local Kev conditions. No batch remains pending.

The frozen `analysis/audit_finqa_rag.py` was executed separately against each completed
answer root. Both audits passed: source hashes, task coverage, inference receipt hashes,
official FinQA label identity, calculator receipts and numerical/program grades agree.
The auditor does not establish semantic citation support or absence of contamination.

Evidence: [20260924-rerank-completion](../benchmarks/agent_study/evidence/20260924-rerank-completion/).
The directory preserves scores, audit summaries, inference receipts, protocols, manifests,
wrapper completion/submission records, scheduler accounting and original stderr, with
`SHA256.json` binding the downloaded bytes. Raw episode files remain at the two RADFM
roots recorded in the manifests; their individual hashes are in the inference receipts.
Earlier failed/rejected wrapper attempts remain in the 20260923-rag-completion evidence.

## Observed outcomes

| Answer model | Ranking method | Correct / 32 | Scoring/answer-format errors / 32 |
|---|---|---:|---:|
| Qwen | BGE | 2 | 30 |
| Qwen | Local Kev | 0 | 31 |
| Mistral | BGE | 0 | 32 |
| Mistral | Local Kev | 0 | 32 |

The error column counts rows whose saved scorer result has a non-null `error`:
`JSONDecodeError` or `TypeError`. These are not failed Slurm jobs. One Qwen/Kev row
was incorrect without a recorded scoring error. The 128 generated answers reuse the
same **32 public questions from 16 companies**, also used by the memory and RAG studies;
they are not 128 independent task samples.

These results do not demonstrate a local Kev downstream-answer advantage. Correct
completion is low and answer-format errors dominate. Successful ranking execution and
successful receipt audits must not be interpreted as successful financial reasoning.

## Recorded limitations and remaining work

The original Mistral stderr warns about the tokenizer regex and recommends
`fix_mistral_regex=True`. The frozen run was not changed or repeated after inspecting
outcomes. Its all-zero scores therefore cannot isolate model capability from this
tokenization warning and the observed answer-format failures. Any correction would need
an explicitly documented follow-up with a consistent comparison protocol.

Both models also emitted the `torch_dtype` deprecation warning. The audits replay the
saved grades; they do not certify every runtime configuration choice as valid.

This closes the submitted reranking batch, not the entire research plan. Human workflow
efficiency, independent new-data validation and native system-level framework comparisons
remain separate. This evidence update is for GitHub; no Overleaf synchronization is
claimed for these final results.
