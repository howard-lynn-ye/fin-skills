# DocFinQA context results: model-specific audited outcomes

Final status: both models completed and all 384 planned units passed audit. These are the
same 32 FinQA evaluation questions used previously, now supplied with shared retrieved
DocFinQA report excerpts. This is neither full-report evaluation nor a new independent
task set. See [DOCFINQA_CONTEXT_PROTOCOL_20260923.md](DOCFINQA_CONTEXT_PROTOCOL_20260923.md).

## Qwen: all 192 planned units retained

Job **1654705** completed with exit 0 in 6 minutes 35 seconds of scheduler time.
Inference plus separate scoring took 372.679 seconds. The automatic audit checked all
192 entries, frozen source/input/receipt hashes, original numerical scores, memory IDs,
prompt token counts and protected/nested selections. Its immutable artifact is
`fin-skills-campaign-docfinqa-context-qwen-20260923-v2/context-verified-summary.json`,
SHA `211dcb46e93fe67adb8218659bb596e32f80465679404fe66fc0d082d0963689`.

Every condition retains the same 32-task denominator: 27 generated responses and five
unavailable report mappings. There were no generation exceptions or scorer exceptions.

| Memory | Context | Correct / planned | Input failures | Format failures | Invalid execution |
|---|---|---:|---:|---:|---:|
| None | Full retrieved pack | 10 / 32 | 5 | 0 | 13 |
| None | Recency | 1 / 32 | 5 | 7 | 12 |
| None | HiSTrim | 0 / 32 | 5 | 14 | 9 |
| Acquisition-conditioned package | Full retrieved pack | 4 / 32 | 5 | 6 | 11 |
| Acquisition-conditioned package | Recency | 3 / 32 | 5 | 2 | 17 |
| Acquisition-conditioned package | HiSTrim | 0 / 32 | 5 | 7 | 12 |

Wrong but executable numerical answers are not included in the format/invalid columns;
they remain incorrect in the full denominator. Inspected raw failures include invented
`table_lookup` operations and Markdown-fenced JSON. Neither is repaired after inference.
The empty-table scalar scorer and original answer units remain unchanged.

## Cost and retention, Qwen only

Each row sums the 27 actual responses. Prompt tokens are offered tokens, not effective
per-layer compute. All conditions used the same GPU allocation and paired task seeds;
generation lengths and allocator history still affect time and memory comparisons.

| Memory | Context | Input tokens | Output tokens | Episode wall seconds | Prefill seconds | Final document items retained |
|---|---|---:|---:|---:|---:|---:|
| None | Full pack | 95,206 | 672 | 64.770 | 33.715 | 216 |
| None | Recency | 95,206 | 670 | 46.465 | 18.697 | 45 |
| None | HiSTrim | 95,206 | 453 | 31.123 | 13.323 | 0 |
| Conditioned | Full pack | 138,264 | 714 | 90.418 | 53.247 | 216 |
| Conditioned | Recency | 138,264 | 565 | 54.304 | 29.158 | 69 |
| Conditioned | HiSTrim | 138,264 | 626 | 47.914 | 21.425 | 0 |

The memory-package rows retained 108, 14 and zero final-depth memory/circuit items under
full-pack, recency and HiSTrim respectively. The count combines experience records and
the circuit guidance item; it is not a count of distinct experiences. HiSTrim retained
zero document items at its final pruning depth in both memory conditions. Earlier
layers can already have mixed those items into protected states, so this does not mean
the source content had no influence anywhere in the decoder.

Median KV bytes for none/full, none/recency, none/HiSTrim are 681,836,544;
394,067,968; and 287,899,648. The corresponding conditioned-memory medians are
1,007,419,392; 587,464,704; and 401,866,752. Median peak reserved bytes are the same
33,321,648,128 in all six conditions, reflecting allocator history. Physical KV reduction
therefore does not establish a reduction in reserved GPU memory.

This completed model slice shows a quality loss under the current pruning implementation.
With the full retrieved pack, adding conditioned memory reduced correctness from 10 to 4.
This comparison does not isolate the fly
from retrieved reflection text, establish general failure across models/tasks, or justify
retuning the frozen routers. Latency reductions accompany fewer retained items and, in
some conditions, shorter outputs; they are not an unqualified efficiency improvement at
equal quality.

## Mistral: all 192 planned units retained

Job **1654706** completed with exit 0 in 6 minutes 53 seconds; inference plus scoring
took 387.072 seconds. Automatic audit passed. Source hashes and the inference-receipt
and score hashes were independently checked again. Summary
`fin-skills-campaign-docfinqa-context-mistral-20260923-v2/context-verified-summary.json`
has SHA `0510e0d8c0a8c184071f0a76efff34915d50c88279be45df1faa8e960f1d0445`.
Each arm has 32 planned tasks, five input failures and 27 responses. Generation and
scorer exceptions are zero throughout.

| Memory | Context | Correct / planned | Format failures | Invalid execution |
|---|---|---:|---:|---:|
| None | Full retrieved pack | 1 / 32 | 0 | 21 |
| None | Recency | 0 / 32 | 1 | 22 |
| None | HiSTrim | 0 / 32 | 0 | 21 |
| Conditioned | Full retrieved pack | 4 / 32 | 9 | 10 |
| Conditioned | Recency | 0 / 32 | 1 | 25 |
| Conditioned | HiSTrim | 0 / 32 | 16 | 11 |

| Memory | Context | Input tokens | Output tokens | Wall seconds | Prefill seconds | Final document items |
|---|---|---:|---:|---:|---:|---:|
| None | Full pack | 107,084 | 767 | 50.386 | 16.925 | 216 |
| None | Recency | 107,084 | 1,229 | 58.761 | 9.414 | 46 |
| None | HiSTrim | 107,084 | 877 | 41.290 | 7.421 | 14 |
| Conditioned | Full pack | 138,780 | 1,080 | 76.217 | 23.615 | 216 |
| Conditioned | Recency | 138,780 | 1,237 | 64.651 | 12.595 | 54 |
| Conditioned | HiSTrim | 138,780 | 1,135 | 57.443 | 11.431 | 0 |

Conditioned full/recency/HiSTrim retain 108/19/26 final-depth memory/circuit items.
Median KV bytes in the table's order are 641,433,600; 371,298,304; 294,166,528;
835,092,480; 485,240,832; 345,141,248. Median peak allocated bytes are 25,635,294,720;
25,202,327,040; 25,202,327,040; 25,990,891,008; 25,426,213,376; 25,426,213,376.
Median reserved bytes are 28,248,637,440 except none/recency and conditioned/HiSTrim,
which are 27,751,612,416. Allocation history and output length remain confounders.

Across both models, this fixed HiSTrim implementation shows no correctness improvement
on the retrieved-report slice. Full-pack memory changes Qwen 10 to 4 and Mistral 1 to 4;
that mixed result does not establish a general memory benefit or isolate fly conditioning.
All 60 unavailable-input units remain in the 384-unit denominator. These are repeated
conditions on the same 32 public tasks, not 384 independent tasks. No router was retuned.
