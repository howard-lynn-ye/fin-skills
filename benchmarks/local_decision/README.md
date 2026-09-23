# Local decision models: qualification and public routing regression

The library adapters use caller-supplied, pinned local checkpoints. They do not
call TypeSafe. Model and source revisions are in
[`OPEN_DECISION_MODELS_LOCK.json`](../../docs/OPEN_DECISION_MODELS_LOCK.json);
the [execution report](../../docs/OPEN_DECISION_EXECUTION_20260923.md) distinguishes
integration evidence from model quality.

`qualification-inputs.json` contains the short authored integration requests.
The Beacon workers `scripts/qualify_laya_adapter.py` and
`scripts/qualify_kev_adapter.py` exercise the actual factory, response contract,
oversized-input rejection, reranking and RAG. Their recorded RADFM prerequisites
must exist; the workers do not install dependencies or download checkpoints.
`scripts/prepare_kev.py` prepares Kev in a separate Slurm job and records its
resolved dependencies and pinned downloads.

## Paired routing protocol

All 108 existing public routing queries are used. BM25 retrieves three skill
descriptions. Each model sees the full same candidates in a seeded shuffled
order and in the reverse of that order. There is no fallback for errors or
overlong inputs. Labels were historically exposed during library development;
these are regression results, not a held-out benchmark.

Prepare locally, before inference:

```bash
python -m benchmarks.local_decision.routing prepare \
  --dataset evals/queries.jsonl --prepared /absolute/new/path/routing-inputs.json
```

The prepared file contains queries, candidate descriptions, order and hashes;
it contains no target labels. A configuration JSON is a list of records with
`name`, `model_id` (`kev` or `laya`) and `parameters` accepted by `create_model`.
Checkpoint paths must refer to complete local weights. Use a dedicated copy for
Laya, whose loader may update tokenizer compatibility metadata.

Run inference inside a GPU Slurm allocation, with all outputs and runtime caches
under the user's RADFM workspace. Kev and Laya use separate compatible environments.
Set `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` and explicit RADFM cache variables;
the submitted job files are retained alongside each remote source snapshot.

```bash
python -m benchmarks.local_decision.routing infer \
  --prepared /absolute/path/routing-inputs.json \
  --configurations /absolute/path/configurations.json \
  --output /absolute/new/path/predictions
```

After inference writes its complete hash receipt, score the saved responses:

```bash
python -m benchmarks.local_decision.routing score \
  --prepared /absolute/path/routing-inputs.json \
  --predictions /absolute/path/predictions \
  --dataset evals/queries.jsonl --output /absolute/new/path/score.json
```

The scorer verifies source and response hashes and rejects partial runs. It
reports the complete query denominator, BM25 correctness, candidate coverage,
model correctness, failures and order changes, separately for English and Chinese.
Choosing a skill is not equivalent to executing it correctly or answering a
financial question. Two orderings of one query are not independent examples.

For the recorded 2026-09-23 run, extract either `*-routing-predictions.tar.gz`
to a fresh directory and use `evidence/20260923/routing-inputs.json` as prepared
input and `evidence/20260923/routing-source.jsonl` as the dataset. The latter is
the exact public source used during preparation; evidence files retain their
original line endings so their byte hashes remain valid across operating systems.
