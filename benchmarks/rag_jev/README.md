# Paired BM25 and Jev retrieval evaluation

This runnable development protocol asks whether Jev improves the order of a
fixed BM25 candidate set. It uses the library's actual `RAGIndex`, `RAGPipeline`
and `JevModel.rerank`, with the same query, time cutoff, candidate passages,
top-k limit and character budget in both conditions.

`development.json` is a small, public, synthetic example written alongside this
runner. `development-qrels.json` contains author-assigned document relevance
grades, not independent human annotations. These fixtures are useful for
integration checks; they are not a financial benchmark or an unseen test set.

## Run the three stages

From the repository root, first freeze candidates and the baseline:

```bash
python -m benchmarks.rag_jev.run prepare \
  --dataset benchmarks/rag_jev/development.json --output runs/jev-example/prepared
```

The next command sends the frozen query and candidate text to TypeSafe. Set
`TYPESAFE_API_KEY` in the execution environment first. No labels are read by
this command. There are no automatic retries; `--max-requests` caps requests
before execution. Use a new output directory for every attempt.

```bash
python -m benchmarks.rag_jev.run infer --prepared runs/jev-example/prepared \
  --output runs/jev-example/inference --allow-network --max-requests 6
```

Then score the saved result using the separately supplied relevance labels:

```bash
python -m benchmarks.rag_jev.run score --prepared runs/jev-example/prepared \
  --run runs/jev-example/inference --qrels benchmarks/rag_jev/development-qrels.json \
  --output runs/jev-example/score.json
```

Preparation and inference have separate hash receipts. Every planned query is
retained, including blocked credentials, provider errors and empty retrievals.
Requests, responses, resolved model IDs, token usage and timings are preserved.
Request records contain JSON bodies, not HTTP authorization headers. A request
that fails validation may still have incurred provider charges; the summary's
validated-response token count is not an invoice or dollar-cost estimate.

The scorer refuses partial relevance labels and explicit fixture transports.
It reports per-query document Recall and nDCG over the passages that fit the
actual context budget. Duplicate chunks of one document do not earn repeated
relevance credit. Unanswerable queries are reported but excluded from relevance
means; failed Jev calls remain missing and are counted. Means over successful
pairs must not be presented as success over all planned queries. Hash receipts
detect accidental changes, not tampering by an actor who can replace receipts.

`jev-latest` is a mutable alias. Start with a connectivity probe, then use the
provider-supported resolved version through `--model` for comparisons. If
response versions differ within a run, the scorer withholds a pooled treatment
effect. The [official API](https://docs.typesafe.ai/api) and
[confidence documentation](https://docs.typesafe.ai/confidence), checked on
2026-09-22, describe structured answers and distribution-derived confidence.
That confidence is not an empirically established probability of correctness
for this dataset.

## Beacon execution

Stage the committed repository source under a new direct RADFM child, such as
`/beacon-projects/radfm/wy891/fin-skills-rag-jev-20260922-v1/source`.
`beacon_run.sh` uses the existing RADFM Python runtime, runs the regression tests,
prepares the snapshot, runs up to six real requests and scores the receipt.
It puts temporary files and application caches under the same new root.
Submit with absolute scheduler paths; create `logs` under the new root first:

```bash
root=/beacon-projects/radfm/wy891/fin-skills-rag-jev-20260922-v1
mkdir -p "$root/logs"
sbatch --account=angliece --partition=beacon --qos=medium --nodes=1 --ntasks=1 \
  --cpus-per-task=1 --mem=8G --time=00:10:00 --chdir="$root" \
  --output="$root/logs/slurm-%j.out" --error="$root/logs/slurm-%j.err" \
  "$root/source/benchmarks/rag_jev/beacon_run.sh" "$root"
```

This is a prepared command, not evidence that the current account can submit
or reach TypeSafe from a compute node. The hosted model does not use a Beacon
GPU. The API key must be available to the scheduled process without putting it
in a command argument, source file or log.

## Before a paper comparison

Freeze a larger representative corpus and questions, arrange independent
document or passage annotations, and keep evaluation labels out of the model
inputs. Record public exposure and any development reuse. This runner currently
reports document relevance only: it cannot establish that a particular chunk
entails an answer, or that downstream financial tasks become more accurate.
Add an unchanged answering model and independent support/correctness assessment
for those claims. Tool-routing accuracy and confidence-based escalation require
their own held-out labels and calibration procedure.
