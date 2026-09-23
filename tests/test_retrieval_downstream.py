import pytest

from benchmarks.rag_jev.downstream import HERE, bounded_context, eligible_documents, parse
from benchmarks.rag_jev.run import load


def test_future_document_cannot_enter_context():
    dataset = load(HERE / "development.json")
    for query in dataset["queries"]:
        docs = eligible_documents(dataset, query)
        assert "future" not in {d["id"] for d in docs}
        assert bounded_context(docs, limit=1) == []


@pytest.mark.parametrize("text", ['{"choice":true,"citations":[]}',
    '{"choice":3,"citations":[]}', '{"choice":0,"citations":["missing"]}',
    '```json\n{"choice":0,"citations":[]}\n```'])
def test_invalid_outputs_are_not_repaired(text):
    with pytest.raises(ValueError):
        parse(text, 3, ["clock"])


def test_option_keys_cover_existing_queries():
    dataset = load(HERE / "development.json")
    options, key = load(HERE / "answer-options.json"), load(HERE / "answer-key.json")
    ids = {q["id"] for q in dataset["queries"]}
    assert ids == set(options) == set(key["answers"]) == set(key["relevant"])
    for q in ids:
        assert 0 <= key["answers"][q] < len(options[q])


def test_routing_strips_labels_and_preserves_missing_rank_denominator():
    from benchmarks.rag_jev.routing import metrics, prepare_rows
    rows = prepare_rows([dict(q="English", expect="a", note="secret"), dict(q="中文", expect="b")])
    assert rows == [dict(id="q0000", query="English")]
    result = metrics({"a": ["x", "correct"], "b": []}, {"a": "correct", "b": "missing"})
    assert result == dict(questions=2, top1=0, recall_at_3=1, recall_at_10=1, mrr_at_10=.25)
