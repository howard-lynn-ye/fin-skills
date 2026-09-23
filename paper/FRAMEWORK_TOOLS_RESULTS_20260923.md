# LlamaIndex multi-tool organization: final audited results

Verified on 2026-09-23 at the 08:21 UTC heartbeat. Qwen **1655292** and Mistral
**1655293** both completed with exit 0. Their automatic audits passed all 96 cells each;
source hashes, qualification-source equality, frozen receipts, episode hashes and
scores were independently checked again without rewriting the summaries or rerunning
inference. All **192 planned cells** are present, with no unfinished or missing cells.
See [the frozen protocol](FRAMEWORK_TOOLS_PROTOCOL_20260923.md).

The dataset is the same 32 previously exposed authored contracts, with four mechanisms
and eight client policies. Three conditions reuse the same data, domain rules and
numerical algorithms through actual LlamaIndex ReActAgent and FunctionTool. The two
phase names identify the old numerical blocks; **no memory or learning occurs here**.
These are not 192 independent financial tasks or an external validation set.

## Correct completion and failures

Each acquisition/later column has 16 planned cells; each total has 32. Completion
requires the right method, parameters, numerical answer and exact cited tool receipt.
Format and framework errors can overlap and must not be summed as disjoint outcomes.

| Model | Catalog | Acquisition correct | Later correct | Total correct | Format errors | Framework errors |
|---|---|---:|---:|---:|---:|---:|
| Qwen | Generic | 14 / 16 | 11 / 16 | 25 / 32 | 3 | 1 |
| Qwen | Flat guidance | 14 / 16 | 14 / 16 | 28 / 32 | 0 | 0 |
| Qwen | Organized | 13 / 16 | 13 / 16 | 26 / 32 | 3 | 3 |
| Mistral | Generic | 4 / 16 | 6 / 16 | 10 / 32 | 22 | 14 |
| Mistral | Flat guidance | 6 / 16 | 7 / 16 | 13 / 32 | 19 | 15 |
| Mistral | Organized | 1 / 16 | 2 / 16 | 3 / 32 | 29 | 21 |

All recorded framework exceptions were the prespecified model-call limit: four Qwen
and 50 Mistral episodes. These are protocol/model outcomes, not evidence of a broken
environment. Qwen's captured tool callbacks include 23 rejected parameter dictionaries.
The original model text, parser retries, observations and failures remain available.
No Markdown fences were removed, missing parameters supplied or budgets extended.

## Method choice, parameters and use of output

The following first-call fields refer to the first **captured tool callback**. Requests
rejected by the framework before entering the callback are not counted as callbacks;
their model text is retained in the message trace. Thus this is not a complete measure
of every first attempted tool request. Numeric correctness alone does not imply correct
method use or a valid cited receipt. Final method/parameter correctness is evaluated on
the action corresponding to the cited receipt, not an uncited successful tool invocation.

| Model | Catalog | First captured method correct | First captured parameters correct | Numerical answer correct | Valid receipt used |
|---|---|---:|---:|---:|---:|
| Qwen | Generic | 30 / 32 | 24 / 32 | 27 / 32 | 25 / 32 |
| Qwen | Flat guidance | 32 / 32 | 26 / 32 | 30 / 32 | 28 / 32 |
| Qwen | Organized | 32 / 32 | 26 / 32 | 28 / 32 | 26 / 32 |
| Mistral | Generic | 23 / 32 | 23 / 32 | 10 / 32 | 10 / 32 |
| Mistral | Flat guidance | 22 / 32 | 22 / 32 | 13 / 32 | 13 / 32 |
| Mistral | Organized | 13 / 32 | 13 / 32 | 3 / 32 | 3 / 32 |

Exact phase-specific method/parameter and cost counts remain in the immutable summaries.

## Actual cost

All cells share a six-response, 512-output-token-per-response maximum. All attempted
model calls here generated responses. Tool callbacks include failed calls that reached
the bound function; returned receipts count successful numerical executions, which may
still use the wrong method. They do not count every malformed framework action.

| Model | Catalog | Responses | Captured callbacks | Receipts | Input tokens | Output tokens | Episode wall seconds |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen | Generic | 91 | 33 | 25 | 345,830 | 8,260 | 431.495 |
| Qwen | Flat guidance | 91 | 34 | 28 | 362,322 | 8,295 | 439.921 |
| Qwen | Organized | 94 | 35 | 26 | 370,428 | 7,956 | 422.306 |
| Mistral | Generic | 148 | 41 | 41 | 657,819 | 14,623 | 643.086 |
| Mistral | Flat guidance | 154 | 46 | 46 | 686,264 | 15,254 | 678.029 |
| Mistral | Organized | 154 | 21 | 21 | 656,898 | 12,303 | 554.118 |

Worker elapsed time was 1,418.098 seconds for Qwen and 1,991.299 for Mistral; scheduler
elapsed time was 23:45 and 33:15 respectively. They ran concurrently under dynamic
allocation, so their sum is not calendar time. A lower wall time with fewer correct
completions is not an equal-quality efficiency gain. Hardware/model/tokenizer differences
also preclude an equal-compute model ranking.

## Interpretation and limits

Flat guidance has the highest observed completion count for both models. Organized
descriptions are slightly above generic for Qwen but below it for Mistral, and below
flat guidance for both. This does not support stable superiority of the organized
catalog. The treatment combines naming and placement of repeated rule text; it does
not isolate each component, test the automatic algorithm router, or establish whole-library
advantage. All groups have the same numerical implementations and financial knowledge.

These budgets and prompt formats differ from the earlier hand-written decision loop.
Do not pool the scores or claim a controlled LlamaIndex-versus-handwritten framework
winner. No new seed, task selection or prompt revision is performed after these results.
The fixed bounded comparison is complete. Native FinRobot/FinMem/InvestorBench trading
or report-generation systems were not reproduced by this module-level experiment.

Artifacts under `/beacon-projects/radfm/wy891/`:

- `fin-skills-campaign-framework-tools-qwen-20260923-v1/tools-verified-summary.json`,
  SHA `fe65d115d40dceb573930f3c9827723b18864b3d4d087a90bd6858e2c783f8d3`.
- `fin-skills-campaign-framework-tools-mistral-20260923-v1/tools-verified-summary.json`,
  SHA `fa19a2750d4f2fe88598641fee1436fdc7dbeca42e5500d2161bd4dd70862425`.

Both audits used frozen source SHA
`f2dc56cad6040612bbed214c459d6deb5f2ed5762e85215aa7d3c98e07c54888`.
