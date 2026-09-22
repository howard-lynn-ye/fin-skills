"""Hosted Jev contract tests with injected transports; no live inference or credentials."""
import copy
import io
import json
from urllib.error import HTTPError, URLError

import pytest

from fin_skills.model_zoo import create_model, model_catalog
from fin_skills.model_zoo import jev
from fin_skills.rag import Document, RAGIndex, RAGPipeline
from fin_skills.tools import call_tool


def questions():
    return {"route": {"type": "choice", "instructions": "Choose a component.",
                      "criteria": {"rag": "Retrieval", "memory": "Memory"}},
            "quality": {"type": "score", "instructions": "Rate relevance.",
                        "criteria": ["None", "Partial", "Direct"]},
            "relevant": {"type": "noul", "instructions": "Is it relevant?"}}


def response():
    return {"model": "jev-contract-fixture", "usage": {"input_tokens": 10, "output_tokens": 5},
            "answers": {
                "route": {"type": "choice", "choice": "rag", "confidence": .8,
                          "probabilities": {"rag": .9, "memory": .1}},
                "quality": {"type": "score", "score": 1.5, "confidence": .6,
                            "probabilities": {"0": 0., "1": .5, "2": .5},
                            "legend": {"0": "None", "1": "Partial", "2": "Direct"}},
                "relevant": {"type": "noul", "noul": .9}}}


def scoring_transport(payload, *, timeout):
    answers = {}
    for key, question in payload["questions"].items():
        level = 2 if "beta" in question["instructions"]["passage"] else 0
        answers[key] = {"type": "score", "score": float(level), "confidence": 1.,
                        "probabilities": {str(i): float(i == level) for i in range(3)},
                        "legend": dict(enumerate(question["criteria"]))}
        answers[key]["legend"] = {str(k): v for k, v in answers[key]["legend"].items()}
    return {"model": "jev-rerank-fixture", "answers": answers,
            "usage": {"input_tokens": 1, "output_tokens": 0}}


def test_factory_catalog_and_injected_roundtrip_are_explicit():
    from fin_skills.algorithms.knowledge import get_method
    seen = []
    returned = response()
    def transport(payload, *, timeout):
        seen.append((payload, timeout))
        return returned
    model = create_model("jev", transport=transport, model="jev-pinned", timeout=7)
    state = {"content": "source", "__kind__": "ordinary input, not a dataframe"}
    result = model.predict(state, questions=questions())
    assert seen == [({"model": "jev-pinned", "state": state, "questions": questions()}, 7.)]
    assert result == response() and result is not returned
    card = next(c for c in model_catalog() if c["id"] == "jev")
    assert card["deployment"] == "hosted_api" and not card["weights_bundled"]
    assert "fly_memory" in {c["id"] for c in model_catalog()}
    assert get_method("model:jev")["kind"] == "decision"


def test_default_never_calls_network_and_custom_transport_is_credential_free(monkeypatch):
    monkeypatch.setattr(jev, "_http_post", lambda *a, **k: pytest.fail("unexpected HTTP request"))
    with pytest.raises(jev.JevError, match="allow_network"):
        create_model("jev").predict("text", questions=questions())
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    assert create_model("jev", transport=lambda *a, **k: response()).run(
        {"state": "text", "questions": questions()}) == response()


@pytest.mark.parametrize("bad", [None, {}, {"x": {"type": "unknown", "instructions": "q"}},
    {"x": {"type": "score", "instructions": "q", "criteria": ["only"]}},
    {"x": {"type": "choice", "instructions": "q", "criteria": {str(i): None for i in range(256)}}},
    {"x": {"type": "noul", "instructions": "q", "criteria": {"maybe": "no"}}}])
def test_malformed_questions_fail_before_transport(bad):
    model = create_model("jev", transport=lambda *a, **k: pytest.fail("invalid request was sent"))
    with pytest.raises(ValueError):
        model.predict("state", questions=bad)


@pytest.mark.parametrize("field,value", [("timeout", 0), ("timeout", float("inf")),
                                        ("allow_network", 1), ("model", " ")])
def test_bad_configuration(field, value):
    with pytest.raises((ValueError, TypeError)):
        create_model("jev", **{field: value})


@pytest.mark.parametrize("case", ["missing_answer", "wrong_type", "probability_keys", "sum",
                                  "choice", "weighted_score", "legend", "legend_value",
                                  "usage", "nan", "huge_noul", "model"])
def test_malformed_provider_answers_are_rejected(case):
    result = response()
    if case == "missing_answer":
        del result["answers"]["route"]
    elif case == "wrong_type":
        result["answers"]["relevant"]["type"] = "choice"
    elif case == "probability_keys":
        result["answers"]["route"]["probabilities"] = {"other": 1.}
    elif case == "sum":
        result["answers"]["route"]["probabilities"]["rag"] = .2
    elif case == "choice":
        result["answers"]["route"]["choice"] = "memory"
    elif case == "weighted_score":
        result["answers"]["quality"]["score"] = .1
    elif case == "legend":
        result["answers"]["quality"]["legend"] = {"wrong": "Wrong"}
    elif case == "legend_value":
        result["answers"]["quality"]["legend"]["0"] = {"not": "a string"}
    elif case == "usage":
        result["usage"]["input_tokens"] = -1
    elif case == "nan":
        result["answers"]["relevant"]["noul"] = float("nan")
    elif case == "huge_noul":
        result["answers"]["relevant"]["noul"] = 10 ** 400
    elif case == "model":
        result["model"] = ""
    with pytest.raises(jev.JevError):
        create_model("jev", transport=lambda *a, **k: result).predict("state", questions=questions())


def test_finite_json_request_and_bounded_payload():
    model = create_model("jev", transport=lambda *a, **k: pytest.fail("invalid request was sent"))
    for state in ({"value": float("inf")}, {1: "nonstring key"}, {"value": object()}):
        with pytest.raises(ValueError, match="finite JSON"):
            model.predict(state, questions=questions())
    with pytest.raises(ValueError, match="limit"):
        model.predict("x" * (jev.MAX_REQUEST_BYTES + 1), questions=questions())


def test_reranking_preserves_sources_stable_ties_and_never_mutates_input():
    model = create_model("jev", transport=scoring_transport)
    rows = [{"id": "a", "text": "alpha", "source": "source/a"},
            {"id": "b", "text": "beta", "source": "source/b"},
            {"id": "c", "text": "beta", "source": "source/c"}]
    original = copy.deepcopy(rows)
    out = model.rerank("query", rows)
    assert [r["id"] for r in out["passages"]] == ["b", "c", "a"]
    assert rows == original and out["passages"][0]["source"] == "source/b"
    assert len(model.rerank("query", rows, top_k=1)["passages"]) == 1
    assert create_model("jev").rerank("query", [])["model"] is None
    with pytest.raises(ValueError, match="reserved"):
        model.rerank("query", [{"text": "text", "jev_relevance": 2}])


def test_jev_composes_directly_with_rag_generation():
    index = RAGIndex.from_documents([Document("a", "alpha document", "source/a"),
                                    Document("b", "beta document", "source/b")])
    model = create_model("jev", transport=scoring_transport)
    pipe = RAGPipeline(index, reranker=model.rerank,
                       generator=lambda messages: "The beta document is selected. [S1]")
    out = pipe.answer("document", top_k=1, fetch_k=2)
    assert out["citations"][0]["source"] == "source/b"
    assert out["reranking"]["model"] == "jev-rerank-fixture"
    assert out["citation_check"]["valid"]


def test_json_tool_uses_same_request_without_numeric_decoding(monkeypatch):
    seen = []
    def mocked_http(payload, *, timeout):
        seen.append(payload)
        return response()
    monkeypatch.setattr(jev, "_http_post", mocked_http)
    state = {"values": [1, 2], "index": ["a", "b"]}
    out = call_tool("run_model", {"model_id": "jev", "parameters": {"allow_network": True},
                                 "data": {"state": state, "questions": questions()}})
    assert seen[0]["state"] == state and out["result"] == response()
    with pytest.raises(ValueError, match="parameters"):
        call_tool("run_model", {"model_id": "jev", "parameters": {"api_key": "fixture"},
                                "data": {"state": "text", "questions": questions()}})


def test_http_contract_is_mocked_and_does_not_expose_error_bodies(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "synthetic-test-token")
    class Opener:
        def open(self, request, timeout):
            assert request.full_url == "https://api.typesafe.ai/v1/systemone"
            assert request.get_header("Authorization") == "Bearer synthetic-test-token"
            assert request.get_method() == "POST" and timeout == 9
            assert json.loads(request.data)["model"] == "jev-latest"
            return io.BytesIO(json.dumps(response()).encode())
    monkeypatch.setattr(jev, "build_opener", lambda *a: Opener())
    assert create_model("jev", allow_network=True, timeout=9).predict("text", questions=questions()) == response()
    class FailingOpener:
        def open(self, request, timeout):
            raise HTTPError(request.full_url, 429, "provider details", {}, io.BytesIO(b"private body"))
    monkeypatch.setattr(jev, "build_opener", lambda *a: FailingOpener())
    with pytest.raises(jev.JevError, match="HTTP 429") as error:
        create_model("jev", allow_network=True).predict("text", questions=questions())
    assert "private body" not in str(error.value)


def test_missing_key_redirect_and_transport_errors(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(jev.JevError, match="TYPESAFE_API_KEY"):
        create_model("jev", allow_network=True).predict("text", questions=questions())
    with pytest.raises(jev.JevError, match="redirect"):
        jev._NoRedirect().redirect_request(None, None, 302, "redirect", {}, "https://example.com")
    def failed(*args, **kwargs):
        raise URLError("untrusted transport details")
    with pytest.raises(jev.JevError, match="custom transport failed") as error:
        create_model("jev", transport=failed).predict("text", questions=questions())
    assert "untrusted transport details" not in str(error.value)
