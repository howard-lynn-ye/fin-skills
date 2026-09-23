"""Static context / BM25 / pinned E5 and downstream answer-choice pilot.

Uses the existing six English public development questions. It does not become a full
retrieval benchmark by adding an encoder or answer choices. No free-text entailment,
independent holdout or general financial-answer quality claims are supported.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import time

from benchmarks.agent_study.prepare_followup import MODEL, REVISION
from benchmarks.rag_jev.run import digest, index_for, load, write
from fin_skills.rag.documents import timestamp

HERE = Path(__file__).parent
ENCODER = "intfloat/e5-small-v2"
ENCODER_REVISION = "ffb93f3bd4047442299a41ebb6fa998a38507c52"
ARMS = ("static_context", "bm25", "e5_vector")


def eligible_documents(dataset, query):
    cutoff = timestamp(query["as_of"], "as_of")
    return [d for d in dataset["documents"] if d["available_at"] is not None
            and timestamp(d["available_at"], "available_at") <= cutoff]


def encoder_snapshot(output):
    """Download only this pinned public model, into the job's RADFM cache."""
    from huggingface_hub import snapshot_download
    path = Path(snapshot_download(ENCODER, revision=ENCODER_REVISION,
        cache_dir=str(output.parent / "encoder-cache"), token=False,
        allow_patterns=["*.json", "*.safetensors", "vocab.txt"]))
    receipt = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
               for p in path.iterdir() if p.is_file()}
    write(output / "encoder-files.json", dict(model=ENCODER, revision=ENCODER_REVISION,
                                               files_sha256=receipt))
    return path


def vector_contexts(dataset, snapshot):
    import torch
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    encoder = AutoModel.from_pretrained(snapshot, local_files_only=True).eval()

    def embed(texts):
        batch = tokenizer(texts, padding=True, truncation=False, return_tensors="pt")
        if batch.input_ids.shape[1] > 512:
            raise ValueError("encoder input exceeds 512 tokens; no silent truncation")
        with torch.inference_mode():
            hidden = encoder(**batch).last_hidden_state
            hidden = hidden.masked_fill(~batch.attention_mask[..., None].bool(), 0.0)
            pooled = hidden.sum(1) / batch.attention_mask.sum(1)[..., None]
            return torch.nn.functional.normalize(pooled, p=2, dim=1)

    contexts = {}
    for query in dataset["queries"]:
        docs = eligible_documents(dataset, query)
        vectors = embed(["query: " + query["query"], *["passage: " + d["text"] for d in docs]])
        scores = (vectors[:1] @ vectors[1:].T)[0].tolist()
        ranked = sorted(zip(scores, docs), key=lambda x: (-x[0], x[1]["id"]))[:3]
        contexts[query["id"]] = [dict(doc, retrieval_score=score) for score, doc in ranked]
    del encoder
    gc.collect()
    return contexts


def bounded_context(documents, limit=4000):
    rows, used = [], 0
    for doc in documents:
        item = dict(id=doc["id"], text=doc["text"])
        size = len(json.dumps(item, ensure_ascii=False))
        if used + size > limit:
            break
        rows.append(item)
        used += size
    return rows


def parse(content, choices, context_ids):
    # Strict output; malformed or unsupported citation IDs remain misses.
    value = json.loads(content)
    if not isinstance(value, dict) or set(value) != {"choice", "citations"}:
        raise ValueError("exactly choice and citations required")
    if type(value["choice"]) is not int or not 0 <= value["choice"] < choices:
        raise ValueError("invalid choice")
    citations = value["citations"]
    if not isinstance(citations, list) or any(not isinstance(c, str) or c not in context_ids for c in citations):
        raise ValueError("invalid citation IDs")
    return value


def score(output, key_path):
    receipt = load(output / "receipt.json")
    records = load(output / "responses.json")
    if digest(records) != receipt["responses_sha256"]:
        raise ValueError("response receipt mismatch")
    key = load(key_path)
    rows = []
    for arm in ARMS:
        selected = [r for r in records if r["arm"] == arm]
        correct = relevant_citation = invalid = retrieved = 0
        for row in selected:
            ids = [d["id"] for d in row["context"]]
            relevant = key["relevant"][row["query_id"]]
            retrieved += relevant in ids
            try:
                response = parse(row["response"]["choices"][0]["message"]["content"],
                                 row["n_options"], ids)
                correct += response["choice"] == key["answers"][row["query_id"]]
                relevant_citation += relevant in response["citations"]
            except (ValueError, TypeError, KeyError, IndexError):
                invalid += 1
        rows.append(dict(arm=arm, planned=6, completed=len(selected), correct=correct,
            invalid=invalid, relevant_source_in_context=retrieved,
            relevant_source_cited=relevant_citation,
            tokens=sum((r.get("response") or {}).get("usage", {}).get("total_tokens", 0) for r in selected)))
    result = dict(rows=rows, scope="six public development answer-choice questions",
        limits="Relevance and choice accuracy are not free-text entailment or independent validation.")
    write(output / "score.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Slurm allocation required")
    args.output.mkdir(parents=True, exist_ok=False)
    dataset, options = load(HERE / "development.json"), load(HERE / "answer-options.json")
    index = index_for(dataset)
    write(args.output / "protocol.json", dict(scope="public_development_pilot", seed=args.seed,
        dataset_sha256=digest(dataset), options_sha256=digest(options),
        answer_key_sha256=hashlib.sha256((HERE / "answer-key.json").read_bytes()).hexdigest(),
        model=MODEL, revision=REVISION, encoder=ENCODER, encoder_revision=ENCODER_REVISION,
        encoder_source="https://huggingface.co/intfloat/e5-small-v2", top_k=3,
        context_character_limit=4000, max_tokens=256, arms=ARMS,
        static_context="all eligible development documents in frozen source order, budget permitting",
        jev="not included: missing API key; no mock replacement",
        limits="Same context cap, not equal actual input tokens; no independent holdout."))
    # The worker enables download only in this process; Qwen still uses its pinned cached files.
    snapshot = encoder_snapshot(args.output)
    vectors = vector_contexts(dataset, snapshot)
    contexts = {}
    for query in dataset["queries"]:
        bm25 = index.search(query["query"], top_k=3, as_of=query["as_of"])
        by_id = {d["id"]: d for d in dataset["documents"]}
        contexts[query["id"]] = {
            "static_context": bounded_context(eligible_documents(dataset, query)),
            "bm25": bounded_context([by_id[h["document_id"]] for h in bm25]),
            "e5_vector": bounded_context(vectors[query["id"]]),
        }
    write(args.output / "contexts.json", contexts)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    from benchmarks.agent_study.transformers_chat import TransformersChat
    model_path = (Path(os.environ["HF_HUB_CACHE"]) / ("models--" + MODEL.replace("/", "--"))
                  / "snapshots" / REVISION)
    if not model_path.is_dir():
        raise RuntimeError("pinned Qwen snapshot is absent; no model download fallback")
    chat = TransformersChat(str(model_path), REVISION, max_tokens=256)
    cells = [(i, arm) for i in range(len(dataset["queries"])) for arm in ARMS]
    random.Random(args.seed).shuffle(cells)
    records = []
    for i, arm in cells:
        query = dataset["queries"][i]
        context = contexts[query["id"]][arm]
        messages = [{"role": "system", "content": "Select the best answer to the question. "
            "Return only JSON with choice (zero-based integer option index) and citations "
            "(list of supporting document IDs from the supplied context). No prose or code fences."},
            {"role": "user", "content": json.dumps(dict(question=query["query"],
                options=options[query["id"]], context=context))}]
        chat.seed, chat.calls = args.seed * 1000 + i, 0
        started = time.monotonic()
        row = dict(query_id=query["id"], arm=arm, context=context,
                   n_options=len(options[query["id"]]), messages=messages)
        try:
            row["response"] = chat(messages)
        except Exception as exc:
            row.update(response=None, error=f"{type(exc).__name__}: {exc}")
        row["seconds"] = time.monotonic() - started
        records.append(row)
        write(args.output / f"response-{len(records):03d}.json", row)
    write(args.output / "responses.json", records)
    write(args.output / "receipt.json", dict(responses_sha256=digest(records),
        contexts_sha256=digest(contexts), planned=len(cells), completed=len(records)))
    print(json.dumps(score(args.output, HERE / "answer-key.json")))


if __name__ == "__main__":
    main()
