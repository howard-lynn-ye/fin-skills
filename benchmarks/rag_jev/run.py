"""Freeze BM25 candidates, run Jev once per query, then score a frozen receipt.

This first protocol supports public development data only. Labels are read only
by the separate score command. Document relevance is not citation entailment.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import time

from fin_skills.model_zoo.jev import JevModel, _http_post
from fin_skills.rag import RAGIndex, RAGPipeline
from fin_skills.rag.documents import positive, timestamp

SCOPE = "public_development"
MAX_BYTES = 16 * 1024 * 1024


def encoded(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def load(path):
    with Path(path).open("rb") as stream:
        raw = stream.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError("JSON exceeds the evaluation size limit")
    value = json.loads(raw)
    encoded(value)  # Reject nonfinite JSON numbers.
    return value


def write(path, value):
    with Path(path).open("xb") as stream:
        stream.write(encoded(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def now():
    return datetime.now(timezone.utc).isoformat()


def index_for(dataset):
    if set(dataset) != {"scope", "documents", "queries"} or dataset["scope"] != SCOPE:
        raise ValueError("dataset must declare public_development and contain documents/queries")
    if not isinstance(dataset["documents"], list) or not dataset["documents"]:
        raise ValueError("documents must be a nonempty list")
    for doc in dataset["documents"]:
        if set(doc) != {"id", "text", "source", "available_at"}:
            raise ValueError("document fields must be id/text/source/available_at; no labels")
        timestamp(doc["available_at"], "available_at")
    queries = dataset["queries"]
    if not isinstance(queries, list) or not queries:
        raise ValueError("queries must be a nonempty list")
    seen = set()
    for query in queries:
        if set(query) != {"id", "query", "as_of"}:
            raise ValueError("query fields must be id/query/as_of; no labels")
        key = query["id"]
        if not isinstance(key, str) or not key.strip() or key in seen:
            raise ValueError("query IDs must be unique nonempty strings")
        if not isinstance(query["query"], str) or not query["query"].strip():
            raise ValueError("query text must be nonempty")
        timestamp(query["as_of"], "as_of")
        seen.add(key)
    return RAGIndex(dataset["documents"], chunk_size=1200, overlap=200)


def prepare(dataset_path, output, *, top_k=3, fetch_k=8, max_context_chars=4000):
    dataset = load(dataset_path)
    index = index_for(dataset)
    for name, value in (("top_k", top_k), ("fetch_k", fetch_k),
                        ("max_context_chars", max_context_chars)):
        positive(value, name)
    if fetch_k < top_k:
        raise ValueError("fetch_k must be at least top_k")
    options = dict(top_k=top_k, fetch_k=fetch_k, max_context_chars=max_context_chars)
    cases = []
    for query in dataset["queries"]:
        candidates = index.search(query["query"], top_k=fetch_k, as_of=query["as_of"])
        baseline = RAGPipeline(index).prepare(query["query"], as_of=query["as_of"], **options)
        cases.append(dict(id=query["id"], candidates=candidates,
                          candidates_sha256=digest(candidates), baseline=baseline))
    snapshot = dict(schema_version=1, scope=SCOPE, dataset=dataset,
                    dataset_sha256=digest(dataset), options=options, cases=cases,
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    interpretation="Document retrieval only; no downstream accuracy or entailment claim.")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write(output / "snapshot.json", snapshot)
    write(output / "prepared.json", dict(status="prepared_not_run", created_utc=now(),
                                         snapshot_sha256=digest(snapshot), queries=len(cases)))
    return dict(status="prepared_not_run", queries=len(cases), snapshot_sha256=digest(snapshot))


def read_snapshot(prepared):
    prepared = Path(prepared)
    snapshot, receipt = load(prepared / "snapshot.json"), load(prepared / "prepared.json")
    if digest(snapshot) != receipt["snapshot_sha256"]:
        raise ValueError("snapshot hash mismatch")
    if snapshot["schema_version"] != 1 or digest(snapshot["dataset"]) != snapshot["dataset_sha256"]:
        raise ValueError("unsupported or modified dataset")
    index = index_for(snapshot["dataset"])
    queries = snapshot["dataset"]["queries"]
    if len(queries) != len(snapshot["cases"]):
        raise ValueError("case count mismatch")
    for query, case in zip(queries, snapshot["cases"]):
        hits = index.search(query["query"], top_k=snapshot["options"]["fetch_k"], as_of=query["as_of"])
        if (case["id"] != query["id"] or digest(hits) != case["candidates_sha256"]
                or digest(case["candidates"]) != case["candidates_sha256"]):
            raise ValueError("frozen candidates no longer reproduce")
        baseline = RAGPipeline(index).prepare(query["query"], as_of=query["as_of"], **snapshot["options"])
        if baseline != case["baseline"]:
            raise ValueError("frozen baseline no longer reproduces")
    return snapshot, index


def infer(prepared, output, *, model="jev-latest", allow_network=False,
          timeout=30., max_requests=20, transport=None):
    snapshot, index = read_snapshot(prepared)
    positive(max_requests, "max_requests")
    required_calls = sum(bool(case["candidates"]) for case in snapshot["cases"])
    if required_calls > max_requests:
        raise ValueError("planned calls exceed max_requests; no calls were made")
    if transport is None and not allow_network:
        raise ValueError("real inference requires explicit --allow-network")
    traces, current = [], []

    def record(payload, *, timeout):
        entry = dict(request=json.loads(encoded(payload)), started_utc=now())
        traces.append(entry)
        current.append(entry)
        start = time.perf_counter()
        try:
            response = (transport or _http_post)(payload, timeout=timeout)
            entry["response"] = json.loads(encoded(response))
            return response
        except Exception as exc:
            entry["error_type"] = type(exc).__name__  # Never persist exception text/credentials.
            raise
        finally:
            entry["elapsed_seconds"] = time.perf_counter() - start

    client = JevModel(model=model, allow_network=allow_network, timeout=timeout, transport=record)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write(output / "started.json", dict(started_utc=now(), snapshot_sha256=digest(snapshot),
        requested_model=model, planned_requests=required_calls, max_requests=max_requests,
        python=platform.python_version(), transport="test_fixture" if transport else "typesafe_http",
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    missing_key = transport is None and not bool(os.environ.get("TYPESAFE_API_KEY", "").strip())
    rows = []
    for number, (query, case) in enumerate(zip(snapshot["dataset"]["queries"], snapshot["cases"])):
        current.clear()
        row = dict(id=query["id"], baseline=case["baseline"], status="pending")
        start = time.perf_counter()
        if missing_key and case["candidates"]:
            row["status"] = "blocked_missing_api_key"
        else:
            try:
                def rerank(text, passages):
                    if digest(passages) != case["candidates_sha256"]:
                        raise ValueError("candidate mismatch before inference")
                    return client.rerank(text, passages)
                row["jev"] = RAGPipeline(index, reranker=rerank).prepare(
                    query["query"], as_of=query["as_of"], **snapshot["options"])
                row["status"] = "completed" if case["candidates"] else "no_candidates"
            except Exception as exc:
                row.update(status="failed", error_type=type(exc).__name__)
        row.update(elapsed_seconds=time.perf_counter() - start, traces=list(current))
        write(output / f"case-{number:04d}.json", row)
        rows.append(row)
    result = dict(schema_version=1, snapshot_sha256=digest(snapshot), scope=SCOPE,
                  transport="test_fixture" if transport else "typesafe_http", cases=rows)
    write(output / "results.json", result)
    receipt = dict(status="completed" if all(r["status"] in ("completed", "no_candidates")
                    for r in rows) else "incomplete", finished_utc=now(),
                   snapshot_sha256=digest(snapshot), results_sha256=digest(result),
                   planned_cases=len(rows), actual_requests=len(traces),
                   completed=sum(r["status"] == "completed" for r in rows))
    write(output / "receipt.json", receipt)
    return receipt


def retrieval_metrics(passages, relevance, k):
    positive(k, "k")
    gains, found = [], set()
    for passage in passages[:k]:
        key = passage["document_id"]
        gains.append(0 if key in found else relevance[key])
        found.add(key)
    relevant = {key for key, grade in relevance.items() if grade > 0}
    dcg = sum((2 ** gain - 1) / math.log2(rank + 2) for rank, gain in enumerate(gains))
    ideal = sum((2 ** gain - 1) / math.log2(rank + 2)
                for rank, gain in enumerate(sorted(relevance.values(), reverse=True)[:k]))
    return dict(recall=len(found & relevant) / len(relevant) if relevant else None,
                ndcg=dcg / ideal if ideal else None, answerable=bool(relevant))


def score(prepared, run, qrels_path, output):
    snapshot, _ = read_snapshot(prepared)
    run = Path(run)
    result, receipt = load(run / "results.json"), load(run / "receipt.json")
    if (receipt["results_sha256"] != digest(result) or result["snapshot_sha256"] != digest(snapshot)
            or receipt["snapshot_sha256"] != digest(snapshot)):
        raise ValueError("inference receipt hash mismatch")
    if result["transport"] != "typesafe_http":
        raise ValueError("fixture transport is not admissible as Jev experimental evidence")
    qrels = load(qrels_path)
    queries = snapshot["dataset"]["queries"]
    if set(qrels) != {query["id"] for query in queries}:
        raise ValueError("qrels must cover every planned query")
    if [row["id"] for row in result["cases"]] != [query["id"] for query in queries]:
        raise ValueError("missing, duplicate or reordered result cells")
    k, rows, models = snapshot["options"]["top_k"], [], set()
    tokens = dict(input_tokens=0, output_tokens=0)
    for query, row in zip(queries, result["cases"]):
        relevance = qrels[query["id"]]
        eligible = {doc["id"] for doc in snapshot["dataset"]["documents"]
                    if timestamp(doc["available_at"], "available_at") <= timestamp(query["as_of"], "as_of")}
        if (not isinstance(relevance, dict) or set(relevance) != eligible or any(
                type(grade) is not int or grade not in (0, 1, 2) for grade in relevance.values())):
            raise ValueError("each qrels map must exhaustively grade eligible documents with 0/1/2")
        baseline = retrieval_metrics(row["baseline"]["passages"], relevance, k)
        treatment = (retrieval_metrics(row["jev"]["passages"], relevance, k)
                     if row["status"] in ("completed", "no_candidates") else None)
        rows.append(dict(id=row["id"], status=row["status"], bm25=baseline, jev=treatment))
        metadata = row.get("jev", {}).get("reranking") or {}
        if metadata.get("model"):
            models.add(metadata["model"])
        for key in tokens:
            tokens[key] += metadata.get("usage", {}).get(key, 0)
    paired = [row for row in rows if row["jev"] is not None and row["bm25"]["answerable"]]
    means = {metric: sum(row["jev"][metric] - row["bm25"][metric] for row in paired) / len(paired)
             if paired else None for metric in ("recall", "ndcg")}
    report = dict(scope=SCOPE, planned_queries=len(rows), paired_answerable_queries=len(paired),
        failed_or_blocked=sum(row["jev"] is None for row in rows), models=sorted(models),
        model_version_consistent=len(models) <= 1, validated_response_tokens=tokens,
        paired_mean_delta=means, cases=rows, top_k=k, qrels_sha256=digest(qrels),
        snapshot_sha256=digest(snapshot), results_sha256=digest(result),
        interpretation="Document relevance on public development data; failed calls remain missing. "
                       "Paired means exclude failures and are not an all-planned success rate. "
                       "No citation entailment, dollar cost, held-out or downstream accuracy claim.")
    if len(models) > 1:
        report["paired_mean_delta"] = None
        report["interpretation"] += " Model versions changed; pooled treatment effect withheld."
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--fetch-k", type=int, default=8)
    p.add_argument("--max-context-chars", type=int, default=4000)
    p = commands.add_parser("infer")
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", default="jev-latest")
    p.add_argument("--allow-network", action="store_true")
    p.add_argument("--timeout", type=float, default=30.)
    p.add_argument("--max-requests", type=int, default=20)
    p = commands.add_parser("score")
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--qrels", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = vars(parser.parse_args())
    action = args.pop("action")
    if action == "prepare":
        args["dataset_path"] = args.pop("dataset")
    elif action == "score":
        args["qrels_path"] = args.pop("qrels")
    result = {"prepare": prepare, "infer": infer, "score": score}[action](**args)
    print(json.dumps(result, indent=2))
    if action == "infer" and result["status"] != "completed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
