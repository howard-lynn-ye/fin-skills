"""Run pinned local Kev models through the library factory and RAG on Beacon."""
import gc
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
import time
import traceback


def write(path, value):
    with path.open("x", encoding="utf-8") as out:
        json.dump(value, out, ensure_ascii=False, indent=2, allow_nan=False)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def qualify(root, snapshots, name, inputs):
    import torch
    from fin_skills.model_zoo import create_model
    from fin_skills.model_zoo.jev import _response
    from fin_skills.rag import Document, RAGIndex, RAGPipeline
    adapter = snapshots["jaredpalmer/kev-" + name]
    base = snapshots["Qwen/Qwen3.5-" + name.upper() + "-Base"]
    identities = {}
    for label, snapshot in (("adapter", adapter), ("base", base)):
        directory = Path(snapshot["path"])
        if not directory.resolve().is_relative_to(root.parent):
            raise ValueError("checkpoint outside RADFM")
        identities[label] = dict(snapshot=snapshot, sha256={
            p.relative_to(directory).as_posix(): digest(p)
            for p in directory.rglob("*") if p.is_file()})
    write(root / f"identity-{name}.json", identities)
    model = create_model("kev", checkpoint=adapter["path"], revision=adapter["revision"],
        base_checkpoint=base["path"], base_revision=base["revision"], device="cuda",
        dtype="bfloat16")
    timings = []
    for case in inputs["cases"]:
        torch.cuda.synchronize()
        start = time.perf_counter()
        response = model.predict(case["state"], questions=inputs["questions"])
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        _response(response, inputs["questions"])
        timings.append(seconds)
        write(root / f"adapter-{name}-{case['id']}.json",
              dict(case=case["id"], response=response, wall_seconds=seconds))
    from kev.model import ContextOverflow
    try:
        model.predict("long evidence " * 20000, questions=inputs["questions"])
    except ContextOverflow as exc:
        overflow = dict(rejected=True, reason=str(exc))
    else:
        raise AssertionError("oversized state was not rejected before inference")
    passages = [dict(text="Revenue was 100 units.", source="synthetic:revenue"),
                dict(text="The office walls are blue.", source="synthetic:office")]
    ranked = model.rerank("What was revenue?", passages)
    assert len(ranked["receipts"]) == 2
    write(root / f"rerank-{name}.json", ranked)
    index = RAGIndex.from_documents([
        Document("revenue", "Revenue was 100 units.", "synthetic:revenue"),
        Document("office", "Revenue report: the office walls are blue.", "synthetic:office")])
    prepared = RAGPipeline(index, reranker=model.rerank).prepare("revenue", top_k=2)
    assert len(prepared["passages"]) == 2 and len(prepared["reranking"]["receipts"]) == 2
    write(root / f"rag-{name}.json", prepared)
    return dict(model=name, status="completed", predictions=len(timings),
        prediction_wall_seconds=timings, first_request_includes_loading=True,
        oversized_request=overflow, reranking_responses=4, rag_pipelines=1)


def run(root):
    if (not os.environ.get("SLURM_JOB_ID") or root.resolve() != root
            or root.parent != Path("/beacon-projects/radfm/wy891")):
        raise ValueError("requires canonical RADFM Slurm workspace")
    sys.path.insert(0, str(root / "source/library"))
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required")
    prior = root.parent / "fin-skills-campaign-kev-preparation-20260923-v1"
    if json.loads((prior / "completion.json").read_text())["status"] != "prepared_not_inferred":
        raise ValueError("preparation did not complete")
    snapshot_file = prior / "snapshots.json"
    snapshots = {s["repo"]: s for s in json.loads(snapshot_file.read_text())["models"]}
    input_file = root / "source/qualification-inputs.json"
    inputs = json.loads(input_file.read_text())
    write(root / "environment.json", dict(job_id=os.environ["SLURM_JOB_ID"],
        gpu=torch.cuda.get_device_name(), cuda=torch.version.cuda,
        versions={p: version(p) for p in ("torch", "transformers", "peft", "kev")},
        snapshot_manifest_sha256=digest(snapshot_file), inputs_sha256=digest(input_file)))
    rows = []
    for name in ("0.8b", "4b"):
        try:
            row = qualify(root, snapshots, name, inputs)
        except Exception as exc:
            row = dict(model=name, status="failed", error_type=type(exc).__name__, error=str(exc))
            (root / f"traceback-{name}.txt").write_text(traceback.format_exc(), encoding="utf-8")
        rows.append(row)
        write(root / f"result-{name}.json", row)
        gc.collect()
        torch.cuda.empty_cache()
    return dict(status="completed" if all(r["status"] == "completed" for r in rows) else "failed",
        models=rows, limits="Authored smoke and integration cases only; no measured financial advantage.")


if __name__ == "__main__":
    root = Path(sys.argv[1])
    try:
        result = run(root)
    except Exception as exc:
        result = dict(status="failed", error_type=type(exc).__name__, error=str(exc))
        write(root / "completion.json", result)
        raise
    write(root / "completion.json", result)
    print(json.dumps(result))
    sys.exit(0 if result["status"] == "completed" else 1)
