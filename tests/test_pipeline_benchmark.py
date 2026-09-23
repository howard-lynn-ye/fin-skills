"""Check the independent scorer and real Linux execution before spending GPU inference."""
import math
import sys

import pytest

from benchmarks.library_workflows.pipeline import cases, evaluate, extract_code, matches, reference
from fin_skills.rag import RAGIndex


LIBRARY_SOURCE = '''from pathlib import Path
from fin_skills.rag import RAGIndex
def solve(request, state_dir):
    path = Path(state_dir) / 'index.json'
    if request['documents'] is not None:
        index = RAGIndex(request['documents'], chunk_size=1200, overlap=0)
        if path.exists():
            path.unlink()
        index.save(path)
    else:
        index = RAGIndex.load(path)
    return [{'id': h['document_id'], 'score': h['score']} for h in index.search(
        request['query'], top_k=request['top_k'], as_of=request['as_of'])]
'''


def test_hand_computed_score_and_negative_results():
    docs = [dict(id="a", text="risk", available_at=None),
            dict(id="b", text="cash", available_at=None)]
    request = dict(query="risk risk", top_k=3, as_of=None)
    result = reference(request, docs, "build")
    assert result == [{"id": "a", "score": math.log(2)}]
    assert matches(result, result)
    assert not matches([], result)
    assert not matches([{"id": "a", "score": float("nan")}], result)
    assert not matches([{"id": "b", "score": math.log(2)}], result)


@pytest.mark.parametrize("stage", ["build", "historical", "restore"])
def test_library_matches_independent_arithmetic(stage):
    for case in cases(11, stage):
        docs = case["requests"][0]["documents"]
        index = RAGIndex(docs, chunk_size=1200, overlap=0)
        for request, expected in zip(case["requests"], case["expected"]):
            hits = index.search(request["query"], top_k=request["top_k"],
                as_of=None if stage == "build" else request["as_of"])
            result = [{"id": h["document_id"], "score": h["score"]} for h in hits]
            assert matches(result, expected)


def test_code_extraction_rejects_ambiguous_outputs():
    assert extract_code("```python\nx = 1\n```") == "x = 1"
    with pytest.raises(ValueError):
        extract_code("```python\nx = 1\n```\n```python\nx = 2\n```")


@pytest.mark.skipif(sys.platform != "linux", reason="real confinement needs Linux")
def test_confined_worker_persists_across_processes_and_rejects_bad_candidate():
    fixtures = cases(11, "restore", public=True)
    good = evaluate(LIBRARY_SOURCE, fixtures, "fin_skills")
    assert good["passed"], good
    bad = evaluate("def solve(request, state_dir): return []", fixtures, "components")
    assert not bad["passed"]
    denied = evaluate(LIBRARY_SOURCE, fixtures, "components")
    assert not denied["passed"]
