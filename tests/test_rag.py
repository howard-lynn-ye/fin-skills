"""RAG retrieval, provenance, time filtering and callback composition, entirely offline."""
import json

import numpy as np
import pytest

from fin_skills.rag import Document, RAGIndex, RAGPipeline, chunk_documents, documents_from_skills
from fin_skills.tools import call_tool


def corpus():
    return [Document("retrieval", "RAG retrieves source documents for a question.",
                     "manual/retrieval", {"kind": "manual"}, "2026-09-01T10:00:00Z"),
            Document("memory", "Memory updates from feedback.", "manual/memory",
                     available_at="2026-09-02T10:00:00Z")]


def test_chunks_are_stable_exact_substrings_and_isolated():
    doc = Document("unicode", "\u68c0\u7d22 alpha beta gamma delta", metadata={"tags": ["original"]})
    chunks = chunk_documents([doc], chunk_size=10, overlap=3)
    assert chunks == chunk_documents([doc], chunk_size=10, overlap=3)
    assert chunks[0]["start"] == 0 and chunks[-1]["end"] == len(doc.text)
    assert len({c["id"] for c in chunks}) == len(chunks)
    for chunk in chunks:
        assert chunk["text"] == doc.text[chunk["start"]:chunk["end"]]
    for previous, current in zip(chunks, chunks[1:]):
        assert current["start"] == previous["end"] - 3
    chunks[0]["metadata"]["tags"].append("mutated")
    assert doc.metadata["tags"] == ["original"]


@pytest.mark.parametrize("options", [{"chunk_size": 0}, {"chunk_size": True},
                                     {"overlap": -1}, {"chunk_size": 4, "overlap": 4}])
def test_invalid_chunk_configuration(options):
    with pytest.raises(ValueError):
        chunk_documents(corpus(), **options)


def test_bm25_returns_actual_matches_stably_without_leaking_mutation():
    index = RAGIndex.from_documents(corpus())
    hits = index.search("source documents")
    assert [h["document_id"] for h in hits] == ["retrieval"]
    assert hits[0]["score"] > 0 and hits[0]["score_kind"] == "bm25"
    hits[0]["metadata"]["kind"] = "changed"
    assert index.search("source")[0]["metadata"]["kind"] == "manual"
    assert index.search("unrelated_xyz") == []
    assert index.search("!!!") == []
    assert RAGIndex.from_documents([]).search("source") == []
    assert RAGIndex.from_documents([Document("empty", " ")]).chunks == []
    assert RAGIndex.from_documents([Document("cn", "\u68c0\u7d22\u6765\u6e90")]).search("\u68c0\u7d22")


@pytest.mark.parametrize("options", [{"top_k": 0}, {"top_k": True}, {"method": "unknown"},
                                     {"min_score": float("nan")}, {"as_of": "2026-09-01"}])
def test_invalid_search_configuration(options):
    with pytest.raises(ValueError):
        RAGIndex.from_documents(corpus()).search("source", **options)


def test_as_of_excludes_undated_future_and_future_corpus_statistics():
    base = RAGIndex.from_documents(corpus())
    expanded = RAGIndex.from_documents(corpus() + [
        Document("future", "source " * 40, available_at="2027-01-01T00:00:00Z"),
        Document("undated", "source documents")])
    cutoff = "2026-09-01T06:00:00-04:00"
    assert expanded.search("source", as_of=cutoff) == base.search("source", as_of=cutoff)
    assert len(expanded.search("source")) == 3
    assert expanded.search("source", as_of="2025-01-01T00:00:00Z") == []


def test_document_inputs_are_validated():
    with pytest.raises(ValueError, match="duplicate"):
        RAGIndex.from_documents([Document("id", "one"), Document("id", "two")])
    with pytest.raises(ValueError, match="timezone"):
        Document("id", "text", available_at="2026-09-01")
    with pytest.raises(ValueError, match="finite JSON"):
        Document("id", "text", metadata={"value": float("inf")})
    with pytest.raises(TypeError):
        RAGIndex.from_documents([{"id": "id", "text": "text", "unknown": 1}])


def embedding(texts):
    # Controlled vectors test cosine mechanics, not an actual semantic model.
    mapping = {"motorcar": [2., 0.], "orchard": [0., 3.], "automobile": [1., 0.]}
    return [mapping[t] for t in texts]


def test_vector_retrieval_does_not_require_word_overlap_and_survives_json(tmp_path):
    index = RAGIndex.from_documents([Document("car", "motorcar"), Document("fruit", "orchard")],
                                    embedder=embedding, embedding_id="fixture-v1")
    assert index.search("automobile") == []
    hits = index.search("automobile", method="vector")
    assert len(hits) == 1 and hits[0]["document_id"] == "car"
    assert hits[0]["score"] == pytest.approx(1.)
    path = index.save(tmp_path / "index.json")
    restored = RAGIndex.load(path, embedder=embedding, embedding_id="fixture-v1")
    assert restored.search("automobile", method="vector") == hits
    with pytest.raises(ValueError, match="matching"):
        RAGIndex.load(path, embedder=embedding, embedding_id="different")
    with pytest.raises(ValueError, match="embedder"):
        RAGIndex.load(path).search("automobile", method="vector")
    with pytest.raises(FileExistsError):
        index.save(path)


@pytest.mark.parametrize("vectors", [[[0., 0.]], [[float("nan"), 1.]], [[1.], [2.]], [1., 2.]])
def test_bad_embedding_results_fail(vectors):
    with pytest.raises(ValueError):
        RAGIndex.from_documents([Document("id", "text")], embedder=lambda texts: vectors,
                                embedding_id="fixture")


def test_query_dimension_and_large_finite_vectors():
    calls = []
    def inconsistent(texts):
        calls.append(texts)
        return [[1., 2.]] if len(calls) == 1 else [[1., 2., 3.]]
    index = RAGIndex.from_documents([Document("id", "text")], embedder=inconsistent,
                                    embedding_id="fixture")
    with pytest.raises(ValueError, match="dimension"):
        index.search("query", method="vector")
    index = RAGIndex.from_documents([Document("id", "text")],
        embedder=lambda texts: [[1e308, 1e308] for _ in texts], embedding_id="large-fixture")
    assert np.isfinite(index.search("query", method="vector")[0]["score"])


def test_packaged_skills_are_available_without_repository_paths():
    docs = documents_from_skills(["backtest-validation"])
    assert docs and all(d.source.startswith("fin-skills:") for d in docs)
    assert all(d.available_at is None for d in docs)
    assert RAGIndex.from_skills(["backtest-validation"]).chunks
    with pytest.raises(ValueError, match="unknown"):
        documents_from_skills(["../../outside"])


def test_generation_receives_cited_context_and_retains_sources():
    seen = []
    def generator(messages):
        seen.append(messages)
        assert "source documents" in json.loads(messages[1]["content"])["retrieved_context"]
        return "Use the document retrieval component. [S1]"
    result = RAGPipeline(RAGIndex.from_documents(corpus()), generator=generator).answer("source")
    assert result["status"] == "generated" and result["citation_check"]["valid"]
    assert result["citations"][0]["source"] == "manual/retrieval"
    assert result["citations"][0]["metadata"] == {"kind": "manual"}
    assert len(seen) == 1
    json.dumps(result, allow_nan=False)


def test_no_evidence_and_budget_do_not_invoke_callbacks():
    def fail(*args):
        pytest.fail("no-evidence retrieval must not call a model")
    pipe = RAGPipeline(RAGIndex.from_documents(corpus()), generator=fail, reranker=fail)
    assert pipe.answer("unmatched_xyz")["status"] == "no_evidence"
    result = RAGPipeline(pipe.index, generator=fail).answer("source", max_context_chars=1)
    assert result["answer"] is None and result["citations"] == []
    small = RAGPipeline(pipe.index).prepare("source", max_context_chars=200)
    assert len(small["context"]) <= 200
    with pytest.raises(ValueError, match="generator"):
        RAGPipeline(pipe.index).answer("source")


@pytest.mark.parametrize("answer,unknown", [("Unsupported [S99]", ["S99"]), ("No citation", [])])
def test_generated_unknown_or_missing_citations_are_flagged(answer, unknown):
    result = RAGPipeline(RAGIndex.from_documents(corpus()), generator=lambda messages: answer).answer("source")
    assert not result["citation_check"]["valid"]
    assert result["citation_check"]["unknown_ids"] == unknown


@pytest.mark.parametrize("mutation", ["text", "source", "duplicate", "unknown"])
def test_rerankers_cannot_invent_or_modify_evidence(mutation):
    def rerank(query, hits):
        if mutation == "duplicate":
            return [hits[0], hits[0]]
        hits[0]["id" if mutation == "unknown" else mutation] = "invented"
        return hits
    pipe = RAGPipeline(RAGIndex.from_documents(corpus()), reranker=rerank)
    with pytest.raises(ValueError, match="reranker"):
        pipe.prepare("source")


def test_json_tool_returns_same_source_context_and_empty_means_empty():
    result = call_tool("retrieve_context", {"query": "source", "documents": [d.to_dict() for d in corpus()]})
    assert result["citations"][0]["document_id"] == "retrieval"
    assert result["tool"] == "retrieve_context"
    assert call_tool("retrieve_context", {"query": "source", "documents": []})["no_evidence"]
    json.dumps(result, allow_nan=False)


def test_persistence_rejects_bad_schema_and_preserves_lexical_scores(tmp_path):
    index = RAGIndex.from_documents(corpus())
    path = index.save(tmp_path / "lexical.json")
    assert RAGIndex.load(path).search("source") == index.search("source")
    payload = json.loads(path.read_text())
    payload["schema_version"] = 2
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="schema"):
        RAGIndex.load(path)
