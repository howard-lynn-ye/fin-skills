"""Local decision serialization, input loss and explicit lifecycle checks."""
import copy
import json
import sys
from types import SimpleNamespace, ModuleType

import pytest

from fin_skills.model_zoo.laya import LayaModel, canonical_response
from fin_skills.model_zoo.jev import JevError, _response


QUESTIONS = {"quality": {"type": "score", "instructions": "Rate relevance",
                         "criteria": ["None", "Partial", "Direct"]}}


def rounded():
    return dict(model="laya-rl-agent", usage=dict(input_tokens=10, output_tokens=0),
        answers={"quality": dict(type="score", score=1.0, confidence=0.1,
            probabilities={"0": 0.3333, "1": 0.3333, "2": 0.3333},
            legend={"0": "None", "1": "Partial", "2": "Direct"})})


def test_rounding_is_bounded_and_raw_response_retained():
    raw = rounded()
    original = copy.deepcopy(raw)
    with pytest.raises(JevError, match="sum to one"):
        _response(raw, QUESTIONS)
    result = canonical_response(raw, QUESTIONS)
    assert result["raw_response"] == original == raw
    assert sum(result["answers"]["quality"]["probabilities"].values()) == pytest.approx(1)
    assert result["answers"]["quality"]["confidence"] == .1
    assert result["serialization"]["calibration_claim"] is False
    _response(result, QUESTIONS)


@pytest.mark.parametrize("mutation", [
    lambda r: r["answers"]["quality"]["probabilities"].update({"0": .3}),
    lambda r: r["answers"]["quality"].update(score=1.1),
    lambda r: r["answers"]["quality"].update(score=-.00001),
    lambda r: r["answers"]["quality"]["probabilities"].update({"0": True}),
    lambda r: r["answers"]["quality"].update(legend={"0": "None"}),
    lambda r: r.update(answers=[]),
])
def test_malformed_outputs_are_not_repaired(mutation):
    raw = rounded()
    mutation(raw)
    with pytest.raises((JevError, ValueError)):
        canonical_response(raw, QUESTIONS)


def checkpoint(tmp_path):
    for name in ("model.safetensors", "rl_agent_config.json", "encoder/config.json",
                 "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json"):
        path = tmp_path / name
        path.parent.mkdir(exist_ok=True)
        path.write_text("{}")
    return tmp_path


def test_factory_discovers_python_only_adapter_without_downloading(tmp_path, monkeypatch):
    from fin_skills.model_zoo import catalog, create_model
    monkeypatch.setattr(catalog, "_available", lambda module: True)
    model = create_model("laya", checkpoint=checkpoint(tmp_path), revision="a" * 40)
    assert model._backend is None
    card = next(c for c in catalog.model_catalog() if c["id"] == "laya")
    assert card["deployment"] == "local" and card["credentials"] is None
    assert card["json_run"] is False
    from fin_skills.tools.models import run_model
    with pytest.raises(ValueError, match="Python lifecycle"):
        run_model("laya", {})


def test_missing_checkpoint_never_calls_upstream(tmp_path):
    with pytest.raises(FileNotFoundError, match="incomplete"):
        LayaModel(checkpoint=tmp_path, revision="a" * 40)


def test_truncation_and_device_fallback_stop_inference(tmp_path, monkeypatch):
    common = ModuleType("laya.common")
    common.serialize_state = lambda s: s
    common.render_options = lambda q: ["None", "Partial", "Direct"]
    common.build_sequence = lambda *a: ([1, 2], [0, 1, 2])  # upstream truncated input
    monkeypatch.setitem(sys.modules, "laya.common", common)
    class Tokenizer:
        mask_token = "[MASK]"
        cls_token_id, sep_token_id, mask_token_id = 1, 2, 3
        def __call__(self, text, **kw):
            return {"input_ids": [ord(c) for c in text]}
    backend = SimpleNamespace(device="cpu", tok=Tokenizer(), cfg={},
        _to_internal=lambda q: {"t": q["type"], "ins": q["instructions"]},
        predict=lambda *a: pytest.fail("inference must not run on truncated inputs"))
    model = LayaModel(checkpoint=checkpoint(tmp_path), revision="a" * 40)
    model._backend = backend
    with pytest.raises(ValueError, match="would truncate"):
        model.predict("long state", questions=QUESTIONS)
    model.device = "cuda"
    with pytest.raises(RuntimeError, match="fallback"):
        model.predict("state", questions=QUESTIONS)


def test_reranking_preserves_provenance_and_every_response(tmp_path, monkeypatch):
    model = LayaModel(checkpoint=checkpoint(tmp_path), revision="a" * 40)
    seen = []
    def predict(state, *, questions):
        seen.append(state)
        return dict(answers={"relevance": dict(score=2 if "supports" in state["passage"] else 0,
                    confidence=.5)}, usage=dict(input_tokens=5, output_tokens=0), raw_response={})
    monkeypatch.setattr(model, "predict", predict)
    rows = [dict(text="irrelevant", source="a", offset=7),
            dict(text="supports query", source="b", offset=8)]
    result = model.rerank("query", rows, top_k=1)
    assert result["passages"][0]["source"] == "b"
    assert result["passages"][0]["offset"] == 8
    assert len(result["receipts"]) == len(seen) == 2
    assert result["usage"]["input_tokens"] == 10
    assert "decision_relevance" not in rows[0]
