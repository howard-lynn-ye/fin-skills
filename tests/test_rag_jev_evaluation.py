"""Exercise scoring failures and experiment isolation with explicitly fake transport."""
import copy
import json
from pathlib import Path

import pytest

from benchmarks.rag_jev import run

FIXTURE = Path(__file__).resolve().parents[1] / "benchmarks/rag_jev/development.json"
QRELS = FIXTURE.with_name("development-qrels.json")


@pytest.fixture
def prepared(tmp_path):
    output = tmp_path / "prepared"
    run.prepare(FIXTURE, output, top_k=2, fetch_k=5, max_context_chars=1200)
    return output


def fake_response(payload, *, timeout):
    answers = {key: dict(type="score", score=2., confidence=1.,
                         probabilities={"0":0., "1":0., "2":1.},
                         legend={"0":"Unrelated", "1":"Related", "2":"Direct"})
               for key in payload["questions"]}
    return dict(model="test-only-jev", answers=answers,
                usage=dict(input_tokens=100, output_tokens=30))


def simulate_http(monkeypatch, prepared, output, transport=fake_response):
    # Only pytest temporary directories contain these simulated HTTP results.
    monkeypatch.setenv("TYPESAFE_API_KEY", "unit-test-placeholder")
    monkeypatch.setattr(run, "_http_post", transport)
    return run.infer(prepared, output, allow_network=True)


def test_freeze_excludes_future_and_preserves_context_budget(prepared):
    snapshot, _ = run.read_snapshot(prepared)
    for case in snapshot["cases"]:
        assert all(hit["document_id"] != "future" for hit in case["candidates"])
        assert len(case["baseline"]["context"]) <= 1200
        assert len(case["baseline"]["passages"]) <= 2
    with pytest.raises(FileExistsError):
        run.prepare(FIXTURE, prepared)


def test_changed_snapshot_rejected(prepared):
    path = prepared / "snapshot.json"
    snapshot = run.load(path)
    snapshot["dataset"]["queries"][0]["query"] = "changed"
    path.write_bytes(run.encoded(snapshot))
    with pytest.raises(ValueError, match="hash mismatch"):
        run.read_snapshot(prepared)


def test_budget_and_consent_fail_before_creating_run(prepared, tmp_path):
    output = tmp_path / "run"
    with pytest.raises(ValueError, match="max_requests"):
        run.infer(prepared, output, allow_network=True, max_requests=1)
    assert not output.exists()
    with pytest.raises(ValueError, match="allow-network"):
        run.infer(prepared, output)
    assert not output.exists()


def test_missing_key_keeps_every_planned_cell_without_calls(prepared, tmp_path, monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    output = tmp_path / "run"
    receipt = run.infer(prepared, output, allow_network=True)
    assert receipt["status"] == "incomplete"
    assert receipt["actual_requests"] == 0
    result = run.load(output / "results.json")
    assert len(result["cases"]) == 6
    assert all(row["status"] == "blocked_missing_api_key" for row in result["cases"])
    scored = run.score(prepared, output, QRELS, tmp_path / "score.json")
    assert scored["failed_or_blocked"] == 6
    assert scored["paired_answerable_queries"] == 0
    assert scored["paired_mean_delta"] == {"recall":None, "ndcg":None}


def test_explicit_fixture_results_cannot_be_reported_as_live(prepared, tmp_path):
    output = tmp_path / "run"
    run.infer(prepared, output, transport=fake_response)
    with pytest.raises(ValueError, match="fixture transport"):
        run.score(prepared, output, QRELS, tmp_path / "score.json")


def test_independent_scoring_never_places_qrels_in_requests(prepared, tmp_path, monkeypatch):
    captured = []
    def transport(payload, *, timeout):
        captured.append(copy.deepcopy(payload))
        return fake_response(payload, timeout=timeout)
    output = tmp_path / "run"
    receipt = simulate_http(monkeypatch, prepared, output, transport)
    assert receipt["actual_requests"] == 6
    assert all(set(payload["state"]) == {"query"} for payload in captured)
    assert all(set(payload) == {"state", "questions", "model"} for payload in captured)
    result = run.load(output / "results.json")
    snapshot, _ = run.read_snapshot(prepared)
    for row, case in zip(result["cases"], snapshot["cases"]):
        original = {hit["id"]:hit for hit in case["candidates"]}
        for passage in row["jev"]["passages"]:
            assert all(passage[key] == value for key, value in original[passage["id"]].items())
    report = run.score(prepared, output, QRELS, tmp_path / "score.json")
    assert report["paired_answerable_queries"] == 6
    assert report["models"] == ["test-only-jev"]
    assert report["paired_mean_delta"] == {"recall":0., "ndcg":0.}
    with pytest.raises(FileExistsError):
        run.score(prepared, output, QRELS, tmp_path / "score.json")


def test_failures_recorded_without_retry_or_secret_error_text(prepared, tmp_path, monkeypatch):
    calls = []
    def failure(payload, *, timeout):
        calls.append(payload)
        raise RuntimeError("sensitive exception detail must not appear in receipts")
    output = tmp_path / "run"
    receipt = simulate_http(monkeypatch, prepared, output, failure)
    assert receipt["status"] == "incomplete"
    assert len(calls) == receipt["actual_requests"] == 6
    text = (output / "results.json").read_text()
    assert "sensitive exception detail" not in text
    assert "unit-test-placeholder" not in text
    assert all(row["status"] == "failed" for row in json.loads(text)["cases"])


def test_metric_can_penalize_wrong_ranking_and_duplicate_chunks():
    labels = {"a":2, "b":1, "wrong":0}
    perfect = run.retrieval_metrics([{"document_id":"a"}, {"document_id":"b"}], labels, 2)
    wrong = run.retrieval_metrics([{"document_id":"wrong"}, {"document_id":"a"}], labels, 2)
    duplicate = run.retrieval_metrics([{"document_id":"a"}, {"document_id":"a"}], labels, 2)
    assert perfect["recall"] == perfect["ndcg"] == 1.
    assert wrong["recall"] == duplicate["recall"] == .5
    assert wrong["ndcg"] < duplicate["ndcg"] < perfect["ndcg"]
    assert run.retrieval_metrics([], {"a":0}, 2)["ndcg"] is None


def test_partial_labels_refused(prepared, tmp_path, monkeypatch):
    output = tmp_path / "run"
    simulate_http(monkeypatch, prepared, output)
    qrels = run.load(QRELS)
    del qrels["q-clock"]["format"]
    qrels_path = tmp_path / "partial.json"
    run.write(qrels_path, qrels)
    with pytest.raises(ValueError, match="exhaustively"):
        run.score(prepared, output, qrels_path, tmp_path / "score.json")


def test_modified_results_refused(prepared, tmp_path, monkeypatch):
    output = tmp_path / "run"
    simulate_http(monkeypatch, prepared, output)
    path = output / "results.json"
    results = run.load(path)
    results["cases"].pop()
    path.write_bytes(run.encoded(results))
    with pytest.raises(ValueError, match="hash mismatch"):
        run.score(prepared, output, QRELS, tmp_path / "score.json")


def test_changing_model_versions_withholds_pooled_effect(prepared, tmp_path, monkeypatch):
    calls = []
    def changing(payload, *, timeout):
        calls.append(payload)
        response = fake_response(payload, timeout=timeout)
        response["model"] = "test-only-v1" if len(calls) == 1 else "test-only-v2"
        return response
    output = tmp_path / "run"
    simulate_http(monkeypatch, prepared, output, changing)
    report = run.score(prepared, output, QRELS, tmp_path / "score.json")
    assert not report["model_version_consistent"]
    assert report["paired_mean_delta"] is None


def test_label_fields_refused_in_public_query():
    dataset = run.load(FIXTURE)
    dataset["queries"][0]["answer"] = "should not be in inference data"
    with pytest.raises(ValueError, match="no labels"):
        run.index_for(dataset)
