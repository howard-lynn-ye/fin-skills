"""BM25 versus pinned E5 over the existing public English skill-routing fixtures.

Retrieval only: this evaluates catalog descriptions, not full skill execution or answer
quality. Public regression labels were historically used in development, not held out.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import time

from benchmarks.rag_jev.downstream import ENCODER, ENCODER_REVISION, encoder_snapshot
from benchmarks.rag_jev.run import digest, write


def prepare_rows(rows):
    return [dict(id=f"q{i:04d}", query=row["q"]) for i, row in enumerate(rows)
            if not re.search(r"[\u3400-\u9fff]", row["q"])]


def metrics(ranks, expected):
    reciprocal = [1 / (r.index(expected[q]) + 1) if expected[q] in r else 0
                  for q, r in ranks.items()]
    n = len(ranks)
    return dict(questions=n, top1=sum(bool(r) and r[0] == expected[q] for q, r in ranks.items()),
        recall_at_3=sum(expected[q] in r[:3] for q, r in ranks.items()),
        recall_at_10=sum(expected[q] in r for q, r in ranks.items()),
        mrr_at_10=sum(reciprocal)/n if n else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Slurm allocation required")
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    source = root / "evals/queries.jsonl"
    raw = [json.loads(x) for x in source.read_text(encoding="utf-8").splitlines() if x.strip()]
    queries = prepare_rows(raw)
    import fin_skills
    from fin_skills.rag import RAGIndex
    documents = [dict(id=s["name"], text=s["name"] + "\n" + s["description"],
                      source="catalog:" + s["name"]) for s in fin_skills.catalog()]
    write(args.output / "protocol.json", dict(scope="public_regression_retrieval", queries=queries,
        documents=documents, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        excluded_non_english=len(raw)-len(queries), encoder=ENCODER,
        encoder_revision=ENCODER_REVISION, top_k=10, corpus="skill names and descriptions only",
        limits="Historically exposed public labels; no private holdout or end-task claim."))
    started = time.monotonic()
    index = RAGIndex(documents, chunk_size=4000, overlap=0)
    lexical = {q["id"]: [h["document_id"] for h in index.search(q["query"], top_k=10)]
               for q in queries}
    lexical_seconds = time.monotonic()-started
    snapshot = encoder_snapshot(args.output)
    import torch
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    encoder = AutoModel.from_pretrained(snapshot, local_files_only=True).eval()

    def embed(texts):
        vectors = []
        for start in range(0, len(texts), 16):
            batch = tokenizer(texts[start:start+16], padding=True, truncation=False, return_tensors="pt")
            if batch.input_ids.shape[1] > 512:
                raise ValueError("encoder context exceeds 512 tokens; no silent truncation")
            with torch.inference_mode():
                hidden = encoder(**batch).last_hidden_state
                hidden = hidden.masked_fill(~batch.attention_mask[..., None].bool(), 0.0)
                pooled = hidden.sum(1)/batch.attention_mask.sum(1)[..., None]
                vectors.append(torch.nn.functional.normalize(pooled, p=2, dim=1))
        return torch.cat(vectors)

    started = time.monotonic()
    passages = embed(["passage: " + d["text"] for d in documents])
    questions = embed(["query: " + q["query"] for q in queries])
    scores = (questions @ passages.T).tolist()
    semantic = {q["id"]: [d["id"] for _, d in sorted(zip(s, documents),
                 key=lambda x: (-x[0], x[1]["id"]))[:10]] for q, s in zip(queries, scores)}
    vector_seconds = time.monotonic()-started
    rankings = dict(bm25=lexical, e5_vector=semantic)
    write(args.output / "rankings.json", rankings)
    write(args.output / "receipt.json", dict(rankings_sha256=digest(rankings),
        planned=len(queries), completed=len(semantic),
        bm25_seconds=lexical_seconds, vector_seconds=vector_seconds,
        timing="CPU indexing plus query ranking; excludes model download and loading"))
    # Expected labels are used only after all rankings and their receipt are frozen.
    expected = {f"q{i:04d}": row["expect"] for i, row in enumerate(raw)}
    result = dict(scope="public_regression_retrieval", documents=len(documents),
                  results={arm: metrics(ranks, expected) for arm, ranks in rankings.items()})
    write(args.output / "score.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
