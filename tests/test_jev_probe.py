"""Offline probe contract tests. No credentials or live inference are required."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from fin_skills.model_zoo import jev

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_jev_probe.py"
spec = importlib.util.spec_from_file_location("run_jev_probe", SCRIPT)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def fixture_response(payload):
    """Known test doubles, never evidence about the hosted model."""
    answers = {}
    for key, question in payload["questions"].items():
        kind = question["type"]
        if kind == "choice":
            answers[key] = {"type": kind, "choice": "rag", "confidence": 0.8,
                            "probabilities": {"rag": 0.9, "memory": 0.1}}
        elif kind == "noul":
            answers[key] = {"type": kind, "noul": 0.8}
        else:
            passage = question["instructions"]
            level = 2 if isinstance(passage, str) or "RAG" in passage["passage"] else 0
            answers[key] = {
                "type": kind, "score": float(level), "confidence": 1.,
                "probabilities": {str(i): float(i == level) for i in range(3)},
                "legend": {str(i): text for i, text in enumerate(question["criteria"])},
            }
    return {"model": "jev-OFFLINE-CONTRACT-FIXTURE-v1", "answers": answers,
            "usage": {"input_tokens": 13, "output_tokens": 7}}


@pytest.fixture(autouse=True)
def refuse_provider(monkeypatch):
    monkeypatch.setattr(jev, "_http_post", lambda *a, **k: pytest.fail("Unexpected HTTP call"))


def test_default_dry_run_freezes_both_inputs_without_credentials_or_inference(tmp_path, monkeypatch):
    def no_environment_read(*args):
        pytest.fail("Offline plan tried to read an environment credential")
    monkeypatch.setattr(probe.os.environ, "get", no_environment_read)
    output = tmp_path / "plan"
    result = probe.run_probe(output)
    assert result["status"] == "dry_run" and not result["live_inference"]
    assert result["records"] == []
    assert result["provider_calls_attempted"] == result["contract_calls_attempted"] == 0
    plan = load(output / "plan.json")
    assert plan["maximum_provider_calls"] == len(plan["requests"]) == 2
    assert plan["automatic_retries"] == 0
    assert plan["fixture_exposure"] == "public_synthetic"
    assert {x["type"] for x in plan["requests"][0]["request"]["questions"].values()} == {
        "choice", "score", "noul"}
    for entry in plan["requests"]:
        for name in ("input", "request"):
            value = entry["inputs" if name == "input" else "request"]
            expected = hashlib.sha256(probe.canonical_json(value)).hexdigest()
            assert entry[f"{name}_sha256"] == expected
    assert {p.name for p in output.iterdir()} == {"plan.json", "summary.json"}


def test_plan_hashes_are_stable_and_change_with_requested_model():
    a = probe.build_plan("jev-pinned-a", 9)
    b = probe.build_plan("jev-pinned-a", 9)
    c = probe.build_plan("jev-pinned-b", 9)
    assert a["requests"] == b["requests"]
    for before, after in zip(a["requests"], c["requests"]):
        assert before["input_sha256"] == after["input_sha256"]
        assert before["request_sha256"] != after["request_sha256"]


@pytest.mark.parametrize("key,status", [(None, "missing_credentials"), (" ", "missing_credentials"),
                                       ("bad token", "invalid_credentials"),
                                       ("nonascii-\u00e9", "invalid_credentials")])
def test_live_preflight_never_turns_missing_or_invalid_key_into_results(
        tmp_path, monkeypatch, key, status):
    if key is None:
        monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    else:
        monkeypatch.setenv("TYPESAFE_API_KEY", key)
    output = tmp_path / "preflight"
    result = probe.run_probe(output, run=True)
    assert result["status"] == status
    assert result["provider_calls_attempted"] == 0
    assert result["records"] == [] and not result["live_inference"]
    assert load(output / "summary.json") == result
    if key and key.strip():
        assert key not in "".join(p.read_text(encoding="utf-8") for p in output.iterdir())


def test_two_offline_calls_match_prefrozen_payloads_and_preserve_provider_fields(tmp_path):
    output = tmp_path / "contract"
    calls = []

    def transport(payload, *, timeout):
        plan = load(output / "plan.json")
        entry = plan["requests"][len(calls)]
        prefix = f"{len(calls) + 1:02d}_{entry['operation']}"
        assert load(output / f"{prefix}.started.json")["request_sha256"] == probe.sha256(payload)
        assert not (output / f"{prefix}.result.json").exists()
        assert payload == entry["request"] and timeout == 9.
        assert "TYPESAFE_API_KEY" not in probe.canonical_json(payload).decode()
        calls.append(payload)
        return fixture_response(payload)

    result = probe.run_probe(output, timeout=9, contract_transport=transport)
    assert len(calls) == result["contract_calls_attempted"] == 2
    assert result["provider_calls_attempted"] == 0
    assert result["mode"] == "offline_contract_test"
    assert result["status"] == "offline_contract_success" and not result["live_inference"]
    for record in result["records"]:
        assert record["status"] == "success" and not record["live_inference"]
        assert record["returned_model"] == "jev-OFFLINE-CONTRACT-FIXTURE-v1"
        assert record["usage"] == {"input_tokens": 13, "output_tokens": 7}
        assert record["elapsed_ms"] >= 0
    ranked = result["records"][1]["result"]["passages"]
    assert ranked[0]["id"] == "retrieval"
    assert ranked[0]["source"] == "synthetic:jev-probe-v1/retrieval"
    assert result["downstream_example"]["illustrative_only"]
    assert result["downstream_example"]["passage_ids"] == ["retrieval"]
    assert load(output / "summary.json") == result


@pytest.mark.parametrize("failure_at", [1, 2])
def test_failures_stop_without_retry_or_invented_answers(tmp_path, failure_at):
    output = tmp_path / "failure"
    calls = []

    def transport(payload, *, timeout):
        calls.append(payload)
        if len(calls) == failure_at:
            raise RuntimeError("secret-provider-body-synthetic-token")
        return fixture_response(payload)

    result = probe.run_probe(output, contract_transport=transport)
    assert result["status"] == "contract_failure" and not result["live_inference"]
    assert len(calls) == result["contract_calls_attempted"] == failure_at
    assert result["provider_calls_attempted"] == 0
    failure = result["records"][-1]
    assert failure["returned_model"] is None and failure["usage"] is None
    assert "result" not in failure and "downstream_example" not in result
    assert result["unattempted_operations"] == (["rerank"] if failure_at == 1 else [])
    assert "secret-provider-body-synthetic-token" not in "".join(
        p.read_text(encoding="utf-8") for p in output.iterdir())


def test_malformed_response_is_a_failure_not_an_inference_result(tmp_path):
    result = probe.run_probe(tmp_path / "malformed", contract_transport=lambda *a, **k: {})
    assert result["status"] == "contract_failure"
    assert result["contract_calls_attempted"] == 1 and not result["live_inference"]
    assert result["records"][0]["returned_model"] is None


def test_existing_output_and_receipts_cannot_be_replaced(tmp_path):
    output = tmp_path / "immutable"
    probe.run_probe(output)
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    with pytest.raises(FileExistsError):
        probe.run_probe(output, contract_transport=lambda *a, **k: pytest.fail("No calls allowed"))
    with pytest.raises(FileExistsError):
        probe.write_once(output / "plan.json", {"replaced": True})
    assert {p.name: p.read_bytes() for p in output.iterdir()} == before


def test_contract_transport_cannot_be_labeled_live(tmp_path):
    with pytest.raises(ValueError, match="Contract transports"):
        probe.run_probe(tmp_path / "invalid", run=True, contract_transport=fixture_response)
    assert not (tmp_path / "invalid").exists()


@pytest.mark.parametrize("timeout", [0, -1, 61, float("nan"), float("inf"), True])
def test_invalid_timeouts_fail_before_creating_receipts(tmp_path, timeout):
    with pytest.raises(ValueError, match="timeout"):
        probe.run_probe(tmp_path / "invalid", timeout=timeout)
    assert not (tmp_path / "invalid").exists()


def test_downstream_thresholds_have_explicit_boundary_and_manual_review():
    decision = {"answers": {"component": {"choice": "rag", "confidence": 0.8},
                            "needs_retrieval": {"noul": 0.8}}}
    ranked = {"passages": [{"id": "passage", "jev_relevance": 1.5}]}
    assert probe.downstream_example(decision, ranked)["action"] == "review_selected_passages"
    decision["answers"]["component"]["confidence"] = 0.799
    assert probe.downstream_example(decision, ranked)["action"] == "manual_review"


def test_cli_default_and_missing_key_exit_statuses(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    assert probe.main(["--output", str(tmp_path / "default")]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "dry_run"
    assert probe.main(["--run", "--output", str(tmp_path / "missing")]) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "missing_credentials"
    assert probe.main(["--output", str(tmp_path / "default")]) == 2
    assert "output_exists" in capsys.readouterr().err
