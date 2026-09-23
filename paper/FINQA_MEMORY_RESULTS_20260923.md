# FinQA external-task memory results

Both fixed-model jobs completed and independent audits passed: Qwen **1653915** and
Mistral **1653932**. Each has 16 shared training acquisitions and 160 evaluations
(32 public test tasks across five conditions): 352 planned episodes across models.
All episodes, original grades, receipts, visible memories, source hashes and read-only
circuit replay were checked. See [the frozen protocol](FINQA_MEMORY_PROTOCOL_20260923.md).
The original model-loading failures remain recorded as execution attempts, not extra
independent samples. No inference or scoring was rerun to improve these outcomes.

## Correctness and failures

Each row retains 32 evaluation tasks. Format failures include absent or invalid final
answers; a tool result alone is not a correct final answer. Scorer exceptions are zero.
Other incorrect answers remain in the denominator, including malformed executable programs.

| Model | Memory | Execution correct | Program correct | Format failures |
|---|---|---:|---:|---:|
| Qwen | None | 3 | 3 | 16 |
| Qwen | Reflexion recent | 7 | 6 | 8 |
| Qwen | Reflexion retrieval | 5 | 5 | 10 |
| Qwen | Fly frozen | 6 | 6 | 6 |
| Qwen | Fly learned | 6 | 6 | 9 |
| Mistral | None | 0 | 0 | 28 |
| Mistral | Reflexion recent | 0 | 0 | 30 |
| Mistral | Reflexion retrieval | 0 | 0 | 32 |
| Mistral | Fly frozen | 0 | 0 | 32 |
| Mistral | Fly learned | 0 | 0 | 32 |

The Qwen slice shows higher observed correctness with experience than without it, but
ordinary recent Reflexion has the highest count. Learned and frozen fly have equal counts;
these results do not establish an incremental conditioning benefit. Mistral produces no
correct final answers under this fixed ReAct interface and budget. Its failures cannot
be interpreted as evidence that memory has no effect in every setting.

## Actual evaluation cost

Tools count returned receipts, not every attempted dispatch. Times are sums of episode
wall time, not parallel calendar duration. Input/output tokens and work differ across arms.

| Model | Memory | Responses | Tool receipts | Input tokens | Output tokens | Wall seconds |
|---|---|---:|---:|---:|---:|---:|
| Qwen | None | 135 | 114 | 288,817 | 15,489 | 726.707 |
| Qwen | Recent | 136 | 101 | 551,589 | 19,125 | 964.985 |
| Qwen | Retrieval | 123 | 98 | 441,084 | 15,166 | 756.924 |
| Qwen | Frozen | 118 | 89 | 432,550 | 15,152 | 756.090 |
| Qwen | Learned | 122 | 91 | 450,362 | 14,899 | 747.901 |
| Mistral | None | 177 | 154 | 411,190 | 20,811 | 815.885 |
| Mistral | Recent | 186 | 173 | 653,585 | 19,449 | 813.933 |
| Mistral | Retrieval | 192 | 176 | 648,152 | 21,765 | 897.836 |
| Mistral | Frozen | 192 | 168 | 672,393 | 19,346 | 813.423 |
| Mistral | Learned | 192 | 163 | 681,500 | 20,455 | 856.515 |

Shared acquisition costs, paid once per model: Qwen 73 responses, 64 tool receipts,
162,036 input / 9,809 output tokens and 463.899 seconds; Mistral 91 responses,
79 receipts, 209,103 / 8,284 tokens and 334.810 seconds. Reflection additionally used
15 responses and 54,225 / 5,134 tokens for Qwen, and 16 responses and 58,056 / 2,987
tokens for Mistral, with zero reflection errors. Total job worker times, including
loading, acquisition, reflection, evaluation and scoring, were 4,771.434 and 4,766.471
seconds respectively; these are not the sum of elapsed parallel calendar time.

The implementation executes upstream Reflexion updates and LlamaIndex ReAct; its financial
cross-task wrapper is an adaptation, not an original ALFWorld result. Training and test
companies are disjoint, but these public questions are not proven absent from pretraining.
The bank and circuit stay frozen during evaluation. Correctness rewards are an engineering
mapping, not returns or NAV. The official FinQA calculator is not a new FinSkills capability.
One inference seed and 32 tasks do not establish stable general superiority. DocFinQA
uses these same test tasks and must not be counted as an independent replication.

Immutable summaries under `/beacon-projects/radfm/wy891/`:

- `fin-skills-campaign-finqa-memory-qwen-20260923-v2/memory-verified-summary.json`,
  SHA `0a903f989b2fd08356191ac4bac7ab7e95e0bf8b00fd381b49c2eb23fa29ca64`.
- `fin-skills-campaign-finqa-memory-mistral-20260923-v2/memory-verified-summary.json`,
  SHA `ee9203ca28e8b436d7bcc652b2fcc8940fdde3cfa86ebd03fa727004e327b3e9`.
