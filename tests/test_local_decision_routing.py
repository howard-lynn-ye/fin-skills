"""Failed decisions stay in the denominator; incomplete evidence cannot be scored."""
import json

import pytest

from benchmarks.local_decision.routing import score, sha, write


def fixture(tmp_path):
    labels = tmp_path / "labels.jsonl"
    labels.write_text(json.dumps({"q": "query", "expect": "right"}) + "\n")
    prepared = tmp_path / "inputs.json"
    write(prepared, dict(dataset_sha256=sha(labels), scope="public_regression_routing", limits="public",
        rows=[dict(id="q0000", language="en", bm25=["wrong", "right"])], variants=["shuffled", "reversed"]))
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    write(predictions / "protocol.json", dict(inputs_sha256=sha(prepared), configurations=[dict(name="m")]))
    # An error carrying a selected field must not become a successful prediction.
    write(predictions / "row.json", dict(id="q0000", outcomes={
        "shuffled": dict(status="ok", selected="right"),
        "reversed": dict(status="error", selected="right")}))
    write(predictions / "receipt.json", dict(protocol_sha256=sha(predictions / "protocol.json"), rows=[
        dict(model="m", id="q0000", file="row.json", sha256=sha(predictions / "row.json"))]))
    return prepared, predictions, labels


def test_errors_retained_and_order_pairs_require_two_successes(tmp_path):
    result = score(*fixture(tmp_path))["groups"]["all"]
    assert result["planned"] == 1 and result["candidate_coverage"] == 1
    assert result["bm25_correct"] == 0
    model = result["models"]["m"]
    assert model["variants"]["shuffled"]["correct"] == 1
    assert model["variants"]["reversed"] == dict(correct=0, successful=0, errors=1)
    assert model["both_successful"] == 0 and model["order_flips"] == 0


def test_changed_or_partial_evidence_is_refused(tmp_path):
    prepared, predictions, labels = fixture(tmp_path)
    receipt = predictions / "receipt.json"
    data = json.loads(receipt.read_text())
    data["rows"] = []
    receipt.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="partial"):
        score(prepared, predictions, labels)
