"""Public regression routing; frozen BM25 candidates, no unseen-test claim."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys
import time


def write(path, data):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(dataset, output):
    import fin_skills
    from fin_skills.rag import RAGIndex
    raw = [json.loads(line) for line in Path(dataset).read_text(encoding="utf-8").splitlines() if line.strip()]
    cards = fin_skills.catalog()
    descriptions = {c["name"]: c["description"] for c in cards}
    index = RAGIndex([dict(id=c["name"], text=c["name"] + "\n" + c["description"])
                      for c in cards], chunk_size=4000, overlap=0)
    rows = []
    for i, row in enumerate(raw):
        candidates = [h["document_id"] for h in index.search(row["q"], top_k=3)]
        assert len(candidates) == len(set(candidates))
        shuffled = candidates[:]
        random.Random(20260923 + i).shuffle(shuffled)
        rows.append(dict(id=f"q{i:04d}", query=row["q"],
            language="zh" if re.search(r"[\u3400-\u9fff]", row["q"]) else "en",
            bm25=candidates, order=shuffled,
            descriptions={name: descriptions[name] for name in candidates}))
    write(output, dict(scope="public_regression_routing", rows=rows,
        dataset_sha256=sha(dataset), descriptions_sha256=hashlib.sha256(
            json.dumps(descriptions, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
        candidate_count=3, truncation="forbidden; errors retained", seed=20260923,
        variants=["shuffled", "reversed"],
        limits="Historically exposed public skill-routing fixtures; not a holdout or end-task evaluation."))


def infer(prepared, configurations, output):
    import torch
    from fin_skills.model_zoo import create_model
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("Beacon GPU allocation required")
    output.mkdir(parents=True, exist_ok=False)
    inputs = json.loads(prepared.read_text(encoding="utf-8"))
    configs = json.loads(configurations.read_text(encoding="utf-8"))
    write(output / "protocol.json", dict(inputs_sha256=sha(prepared),
        configurations=configs, job_id=os.environ["SLURM_JOB_ID"],
        gpu=torch.cuda.get_device_name(), torch=torch.__version__,
        calls_per_case=2, variants=inputs["variants"],
        timing="Each request synchronized; first call includes loading. No speed superiority claim."))
    receipts = []
    for config in configs:
        model = create_model(config["model_id"], **config["parameters"])
        directory = output / config["name"]
        directory.mkdir()
        for row in inputs["rows"]:
            outcomes = {}
            for variant in inputs["variants"]:
                order = row["order"] if variant == "shuffled" else row["order"][::-1]
                result = dict(status="empty_candidates", selected=None)
                if order:
                    question = dict(type="choice", instructions="Choose the skill most directly suited to the user's task.",
                        criteria={name: row["descriptions"][name] for name in order})
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    try:
                        response = model.predict(row["query"], questions={"skill": question})
                        torch.cuda.synchronize()
                        result = dict(status="ok", selected=response["answers"]["skill"]["choice"],
                                      response=response)
                    except Exception as exc:
                        result = dict(status="error", selected=None, error_type=type(exc).__name__, error=str(exc))
                    result["seconds"] = time.perf_counter() - started
                outcomes[variant] = result
            path = directory / (row["id"] + ".json")
            write(path, dict(id=row["id"], outcomes=outcomes))
            receipts.append(dict(model=config["name"], id=row["id"],
                file=path.relative_to(output).as_posix(), sha256=sha(path)))
        del model
        gc.collect()
        torch.cuda.empty_cache()
    write(output / "receipt.json", dict(planned=len(inputs["rows"]) * len(configs),
        completed=len(receipts), rows=receipts, protocol_sha256=sha(output / "protocol.json")))


def score(prepared, predictions, labels):
    inputs = json.loads(prepared.read_text(encoding="utf-8"))
    protocol = json.loads((predictions / "protocol.json").read_text(encoding="utf-8"))
    receipt = json.loads((predictions / "receipt.json").read_text(encoding="utf-8"))
    if protocol["inputs_sha256"] != sha(prepared) or receipt["protocol_sha256"] != sha(predictions / "protocol.json"):
        raise ValueError("protocol or input hash mismatch")
    if inputs["dataset_sha256"] != sha(labels):
        raise ValueError("label dataset differs from prepared source")
    targets = {f"q{i:04d}": r["expect"] for i, r in enumerate(
        json.loads(line) for line in labels.read_text(encoding="utf-8").splitlines() if line.strip())}
    expected = {(c["name"], r["id"]) for c in protocol["configurations"] for r in inputs["rows"]}
    observed = {(r["model"], r["id"]) for r in receipt["rows"]}
    if expected != observed or len(receipt["rows"]) != len(expected):
        raise ValueError("partial or duplicate inference receipt")
    outcomes = {}
    for r in receipt["rows"]:
        path = predictions / r["file"]
        if sha(path) != r["sha256"]:
            raise ValueError("prediction hash mismatch")
        item = json.loads(path.read_text(encoding="utf-8"))
        if item["id"] != r["id"] or set(item["outcomes"]) != set(inputs["variants"]):
            raise ValueError("prediction identity or variants mismatch")
        outcomes[r["model"], r["id"]] = item["outcomes"]
    groups = {}
    for language in ("all", "en", "zh"):
        rows = [r for r in inputs["rows"] if language == "all" or r["language"] == language]
        if not rows:
            continue
        group = dict(planned=len(rows), bm25_correct=sum(bool(r["bm25"]) and r["bm25"][0] == targets[r["id"]] for r in rows),
            candidate_coverage=sum(targets[r["id"]] in r["bm25"] for r in rows), models={})
        for config in protocol["configurations"]:
            values = [outcomes[config["name"], r["id"]] for r in rows]
            group["models"][config["name"]] = dict(
                variants={v: dict(correct=sum(x[v]["status"] == "ok" and x[v]["selected"] == targets[r["id"]] for r, x in zip(rows, values)),
                    successful=sum(x[v]["status"] == "ok" for x in values),
                    errors=sum(x[v]["status"] == "error" for x in values)) for v in inputs["variants"]},
                both_successful=sum(all(x[v]["status"] == "ok" for v in inputs["variants"]) for x in values),
                order_flips=sum(all(x[v]["status"] == "ok" for v in inputs["variants"]) and
                    x["shuffled"]["selected"] != x["reversed"]["selected"] for x in values))
        groups[language] = group
    return dict(scope=inputs["scope"], groups=groups, receipt_sha256=sha(predictions / "receipt.json"),
                limits=inputs["limits"], denominator="All planned queries, including errors and missing candidates.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "infer", "score"])
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--configurations", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.dataset, args.prepared)
    elif args.action == "infer":
        infer(args.prepared, args.configurations, args.output)
    else:
        write(args.output, score(args.prepared, args.predictions, args.dataset))
