"""Beacon-only Laya qualification using pinned public weights; no hosted API.

This checks loading and response contracts, not financial quality or calibration.
The source package is staged separately at its audited upstream revision.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time


MODELS = {
    "laya": ("convaiinnovations/laya", "1c5edc17a7acd8701df6fc341c0d179f1c62c982"),
    "laya-multilingual": ("convaiinnovations/laya-multilingual", "052592a15d198d9ad47da779604259b10b47b7aa"),
}
CODE_REVISION = "c7527708f9f5220c669d8aa385077cd28d04708a"


def save(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def file_hashes(snapshots: dict) -> dict:
    result = {}
    for name, directory in snapshots.items():
        for path in Path(directory).rglob("*"):
            if path.is_file():
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(block)
                result[name + "/" + str(path.relative_to(directory))] = digest.hexdigest()
    return result


def run(root: Path) -> dict:
    if (not os.environ.get("SLURM_JOB_ID")
            or root.parent != Path("/beacon-projects/radfm/wy891")
            or root.resolve() != root):
        raise ValueError("requires a Slurm allocation and canonical RADFM root")
    import torch
    from huggingface_hub import snapshot_download
    if not torch.cuda.is_available():
        raise RuntimeError("GPU qualification cannot silently fall back to CPU")
    snapshots = {}
    for name, (repo, revision) in MODELS.items():
        snapshots[name] = snapshot_download(repo, revision=revision, token=False,
            cache_dir=str(root / "cache/hf/hub"),
            allow_patterns=["model.safetensors", "rl_agent_config.json", "encoder/*", "tokenizer/*"])
    save(root / "model-file-hashes-before.json", file_hashes(snapshots))
    # Every model load below uses the downloaded local directory, including encoder config.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    sys.path.insert(0, str(root / "source/upstream"))
    import laya
    from contract import _response  # Byte-identical copy of fin_skills/model_zoo/jev.py.
    questions = {
        "route": {"type": "choice", "instructions": "Which task is requested?",
                  "criteria": {"retrieve": "Find evidence in documents", "calculate": "Calculate a numerical result"}},
        "urgent": {"type": "noul", "instructions": "Is the request explicitly urgent?"},
        "priority": {"type": "score", "instructions": "How urgent is the request?",
                     "criteria": ["No urgency stated", "Soon", "Immediately"]},
    }
    cases = [
        {"id": "english-retrieval", "state": "Please find the annual report passage describing revenue. No rush."},
        {"id": "english-calculation", "state": "Urgent: calculate 100 minus 40 immediately."},
        {"id": "chinese-retrieval", "state": "请查找年报中解释营业收入的原文，不着急。"},
    ]
    save(root / "qualification-inputs.json", {"cases": cases, "questions": questions,
        "purpose": "Public authored smoke cases; no generalization or quality score."})
    records = []
    for name, path in snapshots.items():
        started = time.monotonic()
        model = laya.load(path, device="cuda")
        if model.device.type != "cuda":
            raise RuntimeError("upstream loader fell back from CUDA")
        torch.cuda.synchronize()
        load_seconds = time.monotonic() - started
        for case in cases:
            torch.cuda.synchronize()
            started = time.monotonic()
            result = model.predict(case["state"], questions)
            torch.cuda.synchronize()
            elapsed = time.monotonic() - started
            row = dict(model=name, revision=MODELS[name][1], case=case["id"],
                       response=result, seconds=elapsed, load_seconds=load_seconds)
            # Preserve the upstream response even when the existing strict contract rejects it.
            save(root / f"response-{name}-{case['id']}.json", row)
            try:
                _response(result, questions)
                row["library_contract"] = "passed"
            except Exception as exc:
                row["library_contract"] = "failed"
                row["contract_error"] = str(exc)
            records.append(row)
        del model
        torch.cuda.empty_cache()
    save(root / "model-file-hashes-after.json", file_hashes(snapshots))
    status = ("completed" if all(r["library_contract"] == "passed" for r in records)
              else "completed_with_contract_failures")
    return dict(status=status, code_revision=CODE_REVISION, records=records,
        gpu=torch.cuda.get_device_name(), versions={p: importlib.metadata.version(p)
            for p in ("torch", "transformers", "huggingface_hub", "safetensors")},
        limits=["Six authored smoke requests, not a financial benchmark.",
                "Response validity does not establish correctness or calibrated confidence.",
                "Timing includes cold requests and is not a comparative speed result."],
        hosted_api_calls=0)


if __name__ == "__main__":
    directory = Path(sys.argv[1])
    try:
        outcome = run(directory)
    except Exception as exc:
        save(directory / "completion.json", dict(status="failed", error_type=type(exc).__name__, error=str(exc)))
        raise
    save(directory / "completion.json", outcome)
    print(json.dumps(dict(status=outcome["status"], requests=len(outcome["records"]))))
