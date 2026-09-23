"""Exercise the library factory and RAG callback with real local Laya on Beacon."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys


def write(path, value):
    with path.open("x", encoding="utf-8") as out:
        json.dump(value, out, ensure_ascii=False, indent=2, allow_nan=False)


def run(root):
    if not os.environ.get("SLURM_JOB_ID") or root.resolve() != root or root.parent != Path("/beacon-projects/radfm/wy891"):
        raise ValueError("RADFM Slurm workspace required")
    sys.path[:0] = [str(root / "source/library"), str(root / "source/upstream")]
    from fin_skills.model_zoo import create_model
    from fin_skills.model_zoo.jev import _response
    from fin_skills.rag import Document, RAGIndex, RAGPipeline
    prior = root.parent / "fin-skills-campaign-open-decision-20260923-v2"
    evidence = json.loads((prior / "completion.json").read_text())
    inputs = json.loads((prior / "qualification-inputs.json").read_text())
    rows = []
    for name, repo, revision in (
        ("laya", "convaiinnovations/laya", "1c5edc17a7acd8701df6fc341c0d179f1c62c982"),
        ("laya-multilingual", "convaiinnovations/laya-multilingual", "052592a15d198d9ad47da779604259b10b47b7aa"),
    ):
        original = prior / "cache/hf/hub" / ("models--" + repo.replace("/", "--")) / "snapshots" / revision
        target = root / "checkpoints" / name
        shutil.copytree(original, target)
        model = create_model("laya", checkpoint=target, revision=revision, device="cuda")
        for case in inputs["cases"]:
            result = model.predict(case["state"], questions=inputs["questions"])
            _response(result, inputs["questions"])
            row = dict(model=name, case=case["id"], response=result)
            rows.append(row)
            write(root / f"adapter-{name}-{case['id']}.json", row)
        rejected = False
        try:
            model.predict("long evidence " * 5000, questions=inputs["questions"])
        except ValueError as exc:
            rejected = "truncate" in str(exc)
        if not rejected:
            raise AssertionError("oversized evidence was not explicitly refused")
        ranked = model.rerank("What was revenue?", [
            dict(text="Revenue was 100 units.", source="synthetic:revenue"),
            dict(text="The office walls are blue.", source="synthetic:office")])
        assert len(ranked["receipts"]) == 2
        write(root / f"rerank-{name}.json", ranked)
        index = RAGIndex.from_documents([Document("revenue", "Revenue was 100 units.", "synthetic:revenue"),
                          Document("office", "Revenue report: the office walls are blue.", "synthetic:office")])
        prepared = RAGPipeline(index, reranker=model.rerank).prepare("revenue", top_k=2)
        assert len(prepared["passages"]) == 2 and len(prepared["reranking"]["receipts"]) == 2
        write(root / f"rag-{name}.json", prepared)
        del model
        import torch
        torch.cuda.empty_cache()
    return dict(status="completed", predictions=len(rows), models=2,
        oversized_requests_rejected=2, reranking_responses=8, rag_pipelines=2,
        prior_diagnostic_sha256=hashlib.sha256((prior / "completion.json").read_bytes()).hexdigest(),
        limits="Authored smoke and integration cases only; no measured financial advantage.")


if __name__ == "__main__":
    root = Path(sys.argv[1])
    try:
        result = run(root)
    except Exception as exc:
        write(root / "completion.json", dict(status="failed", error_type=type(exc).__name__, error=str(exc)))
        raise
    write(root / "completion.json", result)
    print(json.dumps(result))
