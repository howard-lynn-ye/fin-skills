# FinanceBench retrieval and independent BGE reranking

Jobs **1653482** and **1653602** completed; both independent audits passed all 300 planned
method-query units. These are 150 public questions evaluated through two actual APIs,
not 300 independent tasks. FinSkills RAGIndex and BM25S use the same complete available
corpus, chunks, tokenizer and BM25 parameters. Candidate receipts were frozen before
evidence-page labels were loaded. BGE reranks each API's original top 50 without adding
candidates. This measures evidence-page retrieval, not answer correctness or Jev.

## Evidence-page hits

Every count uses all 150 questions. The two APIs have identical aggregate hit counts
both before and after reranking; this does not assert byte-identical rankings.

| Method (either API) | Any page @1 | @5 | @10 | @20 | @50 |
|---|---:|---:|---:|---:|---:|
| BM25 retrieval | 9 | 15 | 18 | 23 | 28 |
| Same candidates + BGE | 14 | 23 | 26 | 28 | 28 |

All-required-pages hit counts at the same cutoffs are 9/15/17/22/25 before and
14/22/24/25/25 after BGE. The top-50 candidate ceiling is only 28/150 for any evidence
page; reranking cannot recover evidence absent from that candidate pool. Query errors
are zero. The corpus still includes two failed upstream Intel PDFs in its 368-document
accounting: 366 extracted, 53,901 pages, 53,686 nonempty. No replacement or OCR was used.

## Cost

Warmup plus three randomized timed repetitions were run on the same CPU. Median query
latency was **5.503675 seconds for FinSkills and 0.004404 seconds for BM25S**. The
matched retrieval implementation therefore provides no measured speed advantage here.
These query latencies exclude one-time construction and are not end-to-end answer time.

BGE median reranking latency was 0.448141 seconds on FinSkills candidates and 0.437268
on BM25S candidates. Each method scored 7,500 pairs; five pairs per method were truncated
at the fixed 512-token cap. Actual tokens were 2,287,148 and 2,287,149; padded tokens
were 2,612,970 each. Peak allocated/reserved GPU bytes were 2,472,680,960 / 3,089,104,896.
The fixed BGE revision, float32 precision and batch size eight were preserved. No key
or output was mocked. Worker elapsed time was 3,465.972 seconds for retrieval and
211.000 seconds for reranking, including their setup and scoring.

The public annotations supply third-party evidence labels, not new human annotations
or proof of pretraining non-exposure. Findings support a reranking improvement within a
weak candidate pool, parity between these BM25 APIs on page hits, and substantially
lower query latency for BM25S. They do not establish broad FinSkills superiority.

Immutable summaries under `/beacon-projects/radfm/wy891/`:

- `fin-skills-campaign-financebench-retrieval-20260923-v2/reuse-verified-summary.json`,
  SHA `b1f5d64e0c1985e64630dd8d5048872feea26e6e1245104a5d464c15c4bac072`.
- `fin-skills-campaign-financebench-rerank-20260923-v1/reuse-verified-summary.json`,
  SHA `e423d49c1c1aa2997ec0f9713d6da5445ce2cb4fb14d08b06f688eccb56bed2e`.
