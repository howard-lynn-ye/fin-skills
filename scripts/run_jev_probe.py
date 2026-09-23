"""Bounded Jev connectivity/contract probe on fixed, public synthetic text.

    python scripts/run_jev_probe.py --output artifacts/jev-plan
    python scripts/run_jev_probe.py --run --output artifacts/jev-live

Only --run enables hosted inference, using TYPESAFE_API_KEY from the environment.
Every invocation requires a new output directory. No retries or quality/benefit
estimates are made. Hashes use canonical JSON, not HTTP header or wire bytes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from datetime import datetime, timezone
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fin_skills.model_zoo.jev import ENDPOINT, JevError, JevModel

MAX_REQUESTS = 2
INTERPRETATION = (
    "Connectivity and response-contract probe on fixed synthetic text; "
    "not a quality, calibration, or financial benefit estimate."
)


def canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")


def sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_once(path: Path, value: object) -> None:
    """Exclusive creation; never replace an earlier receipt, even on failure."""
    data = canonical_json(value) + b"\n"
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def fixture_operations() -> list[dict]:
    """Fresh copies of public synthetic inputs, with no user documents or labels."""
    return [
        {"operation": "decisions", "inputs": {
            "state": {
                "fixture": "jev-probe-synthetic-v1",
                "request": "Find supporting passages explaining document retrieval.",
                "available_components": {
                    "rag": "Retrieves supporting passages from a document index.",
                    "memory": "Updates internal memory after receiving feedback.",
                },
            },
            "questions": {
                "component": {
                    "type": "choice",
                    "instructions": "Choose the component that addresses the request.",
                    "criteria": {"rag": "Document retrieval", "memory": "Feedback memory"},
                },
                "support": {
                    "type": "score",
                    "instructions": "How directly does the RAG description support the request?",
                    "criteria": ["Unrelated", "Partially related", "Directly supports"],
                },
                "needs_retrieval": {
                    "type": "noul",
                    "instructions": "Does the request ask for document retrieval?",
                },
            },
        }},
        {"operation": "rerank", "inputs": {
            "query": "Which component retrieves supporting document passages?",
            "passages": [
                {"id": "memory", "source": "synthetic:jev-probe-v1/memory",
                 "text": "The memory component updates internal state after feedback."},
                {"id": "retrieval", "source": "synthetic:jev-probe-v1/retrieval",
                 "text": "The RAG retriever selects supporting passages from a document index."},
                {"id": "formatting", "source": "synthetic:jev-probe-v1/formatting",
                 "text": "The formatter controls the font size of a displayed heading."},
            ],
        }},
    ]


def invoke(model: JevModel, operation: dict) -> dict:
    inputs = operation["inputs"]
    if operation["operation"] == "decisions":
        return model.predict(inputs["state"], questions=inputs["questions"])
    if operation["operation"] == "rerank":
        return model.rerank(inputs["query"], inputs["passages"])
    raise ValueError("Unknown probe operation")


class _RequestCaptured(Exception):
    pass


class _PlanModel(JevModel):
    """Use the adapter's rerank builder without a transport or pretend response."""

    def predict(self, state, *, questions):
        self.request = {"model": self.model, "state": state, "questions": questions}
        raise _RequestCaptured


def build_plan(model: str, timeout: float) -> dict:
    capture = _PlanModel(model=model, timeout=timeout)
    requests = fixture_operations()
    if len(requests) != MAX_REQUESTS:
        raise ValueError("Probe must contain exactly two planned requests")
    for entry in requests:
        try:
            invoke(capture, entry)
        except _RequestCaptured:
            entry["request"] = json.loads(canonical_json(capture.request))
        else:
            raise ValueError("Probe operation did not produce one request")
        entry["input_sha256"] = sha256(entry["inputs"])
        entry["request_sha256"] = sha256(entry["request"])
    return {
        "schema": "jev-probe-v1", "created_at": utc_now(),
        "endpoint": ENDPOINT, "requested_model": model, "timeout_seconds": timeout,
        "maximum_provider_calls": MAX_REQUESTS, "automatic_retries": 0,
        "fixture_exposure": "public_synthetic", "interpretation": INTERPRETATION,
        "hash_encoding": "SHA-256 of UTF-8 JSON, sorted keys, compact separators",
        "adapter_sha256": hashlib.sha256(
            (ROOT / "fin_skills/model_zoo/jev.py").read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "requests": requests,
    }


def downstream_example(decisions: dict, reranking: dict) -> dict:
    """An arbitrary fixed threshold policy for document review, not a model claim."""
    answer = decisions["answers"]["component"]
    accepted = (answer["choice"] == "rag" and answer["confidence"] >= 0.8
                and decisions["answers"]["needs_retrieval"]["noul"] >= 0.8)
    selected = [row["id"] for row in reranking["passages"]
                if row["jev_relevance"] >= 1.5] if accepted else []
    return {
        "illustrative_only": True,
        "thresholds": {"component_confidence": 0.8, "needs_retrieval": 0.8,
                       "passage_relevance": 1.5},
        "action": "review_selected_passages" if selected else "manual_review",
        "passage_ids": selected,
        "interpretation": "Arbitrary deterministic workflow example; thresholds are not calibrated.",
    }


def run_probe(output: Path, *, run: bool = False, model: str = "jev-latest",
              timeout: float = 30., contract_transport: Callable | None = None) -> dict:
    """Write a plan or execute it. Injected test transports cannot claim live inference."""
    if type(run) is not bool:
        raise ValueError("run must be a boolean")
    if run and contract_transport is not None:
        raise ValueError("Contract transports cannot be used with live --run")
    if (type(timeout) not in (int, float) or not math.isfinite(timeout)
            or not 0 < timeout <= 60):
        raise ValueError("timeout must be finite and in (0, 60] seconds")
    if contract_transport is not None and not callable(contract_transport):
        raise ValueError("contract_transport must be callable")
    plan = build_plan(model, timeout)
    mode = "live" if run else "offline_contract_test" if contract_transport else "dry_run"
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_once(output / "plan.json", {**plan, "mode": mode})
    summary = {
        "schema": plan["schema"], "mode": mode, "status": "dry_run",
        "interpretation": INTERPRETATION, "live_inference": False,
        "provider_calls_attempted": 0, "contract_calls_attempted": 0,
        "maximum_provider_calls": MAX_REQUESTS, "records": [],
    }

    def finish(status: str) -> dict:
        summary.update(status=status, completed_at=utc_now())
        write_once(output / "summary.json", summary)
        return summary

    if mode == "dry_run":
        return finish("dry_run")
    if run:
        credential = os.environ.get("TYPESAFE_API_KEY", "").strip()
        if not credential:
            return finish("missing_credentials")
        if not credential.isascii() or any(c.isspace() for c in credential):
            return finish("invalid_credentials")
        del credential

    adapter = JevModel(model=model, timeout=timeout, allow_network=run,
                       transport=contract_transport)
    results = {}
    counter = "provider_calls_attempted" if run else "contract_calls_attempted"
    for index, entry in enumerate(plan["requests"], start=1):
        if summary[counter] >= MAX_REQUESTS:
            raise RuntimeError("Probe request budget exhausted")
        stem = f"{index:02d}_{entry['operation']}"
        record = {
            "operation": entry["operation"], "mode": mode, "started_at": utc_now(),
            "input_sha256": entry["input_sha256"],
            "request_sha256": entry["request_sha256"], "live_inference": False,
            "returned_model": None, "usage": None,
        }
        write_once(output / f"{stem}.started.json", record)
        summary[counter] += 1
        started = time.perf_counter()
        try:
            result = invoke(adapter, entry)
        except Exception as error:
            # Never store exception text, provider error bodies, or credentials.
            http = re.fullmatch(
                r"Jev API request failed \(HTTP ([0-9]{3})\); no answer was returned",
                str(error),
            ) if isinstance(error, JevError) else None
            record.update(
                status="provider_failure" if run else "contract_failure",
                error_category="jev_adapter_error" if isinstance(error, JevError)
                else "execution_error", http_status=int(http[1]) if http else None,
                elapsed_ms=round((time.perf_counter() - started) * 1000, 3),
            )
            write_once(output / f"{stem}.result.json", record)
            summary["records"].append(record)
            summary["unattempted_operations"] = [
                item["operation"] for item in plan["requests"][index:]]
            return finish(record["status"])
        record.update(
            status="success", live_inference=run, returned_model=result["model"],
            usage=result["usage"], result=result,
            elapsed_ms=round((time.perf_counter() - started) * 1000, 3),
        )
        write_once(output / f"{stem}.result.json", record)
        summary["records"].append(record)
        summary["live_inference"] = run
        results[entry["operation"]] = result
    summary["downstream_example"] = downstream_example(results["decisions"], results["rerank"])
    return finish("live_success" if run else "offline_contract_success")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="New output directory; existing paths are refused")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true", help="Allow at most two hosted Jev calls")
    mode.add_argument("--dry-run", action="store_true", help="Only save the plan (default)")
    parser.add_argument("--model", default="jev-latest")
    parser.add_argument("--timeout", type=float, default=30., help="Seconds per call, at most 60")
    args = parser.parse_args(argv)
    try:
        result = run_probe(args.output, run=args.run, model=args.model, timeout=args.timeout)
    except FileExistsError:
        print("output_exists: choose a new output directory", file=sys.stderr)
        return 2
    except (ValueError, TypeError):
        print("invalid_configuration: check model and timeout", file=sys.stderr)
        return 2
    except OSError:
        print("output_error: probe receipts could not be written", file=sys.stderr)
        return 2
    print(json.dumps({key: result[key] for key in (
        "status", "mode", "live_inference", "provider_calls_attempted")}, sort_keys=True))
    return {"dry_run": 0, "live_success": 0, "missing_credentials": 2,
            "invalid_credentials": 2, "provider_failure": 1}[result["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
