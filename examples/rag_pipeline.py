"""Offline RAG + Jev interface composition; callbacks are fixtures, not live inference."""
import json
from pathlib import Path
import tempfile

from fin_skills.model_zoo import create_model, model_catalog
from fin_skills.rag import Document, RAGIndex, RAGPipeline


def demo_transport(payload, *, timeout):
    """Deterministic contract fixture. No request is sent to TypeSafe."""
    answers = {}
    for key, question in payload["questions"].items():
        direct = "retrieves source" in question["instructions"]["passage"]
        level = 2 if direct else 1
        answers[key] = {"type": "score", "score": float(level), "confidence": 1.,
                        "legend": {str(i): text for i, text in enumerate(question["criteria"])},
                        "probabilities": {str(i): float(i == level) for i in range(3)}}
    return {"model": "offline-contract-fixture", "answers": answers,
            "usage": {"input_tokens": 0, "output_tokens": 0}}


def demo_generator(messages):
    """An extractive fixture; replace with your own generator(messages) callback."""
    context = json.loads(messages[-1]["content"])["retrieved_context"]
    return context.splitlines()[1] + " [S1]"


def main():
    documents = [
        Document("rag", "RAG retrieves source documents and preserves citations.", "example/rag"),
        Document("jev", "JEV can rerank RAG passages and return structured decisions.", "example/jev"),
        Document("fly", "Fly memory updates from validated feedback.", "example/fly"),
    ]
    index = RAGIndex.from_documents(documents)
    jev = create_model("jev", transport=demo_transport)
    pipeline = RAGPipeline(index, generator=demo_generator, reranker=jev.rerank)
    result = pipeline.answer("RAG documents", top_k=1, fetch_k=3)
    assert result["citation_check"]["valid"]
    assert result["citations"][0]["source"] == "example/rag"
    assert pipeline.answer("no_matching_term_xyz")["status"] == "no_evidence"
    with tempfile.TemporaryDirectory(prefix="fin-rag-example-") as folder:
        path = index.save(Path(folder) / "index.json")
        assert RAGIndex.load(path).search("RAG") == index.search("RAG")
    assert {"jev", "fly_memory"} <= {card["id"] for card in model_catalog()}
    print("MODE: offline callbacks; no live Jev inference")
    print("ANSWER:", result["answer"])
    print("SOURCE:", result["citations"][0]["source"])
    print("CITATION LABEL CHECK:", result["citation_check"]["valid"])
    print("MODEL COMPONENTS: jev, fly_memory")
    print("TAKEAWAY")
    print("  RAG retrieval, source citations and JSON persistence run without model access.")
    print("  Jev reranking plugs into the pipeline through the same model factory as fly memory.")
    print("  These callbacks test composition; they do not measure provider quality or speed.")


if __name__ == "__main__":
    main()
