"""SQLite revisions -> historical RAG, with optional real Jev reranking.

Default: an in-memory database and real local retrieval; no model calls or mocks.
--live-jev --output <new-directory> permits one hosted call on public example text.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path

from fin_skills.collect import Batch, Event, Store, Watch
from fin_skills.model_zoo import create_model, model_catalog
from fin_skills.rag import Document, RAGIndex, RAGPipeline

AS_OF = "2026-09-22T12:00:00+00:00"
PINNED_MODEL = "jev-1.13.0"  # docs.typesafe.ai/models, checked 2026-09-22.


def observed_documents(store: Store, as_of: str) -> list[Document]:
    """Select the latest *eligible* revision, not today's latest revision.

    Availability is conservative: both observation and known publication must
    precede the cutoff. This uses caller-supplied timestamps, not proof of them.
    """
    cutoff = datetime.fromisoformat(as_of.replace("Z", "+00:00"))
    if cutoff.tzinfo is None:
        raise ValueError("as_of requires an explicit timezone")
    selected = {}
    cursor = 0
    while rows := store.events(after=cursor, limit=1000, latest=False):
        for row in rows:
            observed = datetime.fromisoformat(row["observed_at"])
            published = datetime.fromisoformat(row["published_at"]) if row["published_at"] else observed
            available = max(observed, published)
            key = (row["watch_id"], row["id"])
            order = (observed, row["seq"])
            if available <= cutoff and (key not in selected or order > selected[key][0]):
                selected[key] = (order, row, available)
        cursor = rows[-1]["seq"]
    return [Document(
        id=f"collection:{row['seq']}", text=row["title"] + "\n" + row["data"]["text"],
        source=row["url"], available_at=available.isoformat(),
        metadata={key: row[key] for key in
                  ("watch_id", "id", "seq", "source", "observed_at", "published_at")},
    ) for _, row, available in selected.values()]


def example_store(store: Store) -> None:
    watch = Watch("example-notes", "synthetic", "public-example", enabled=False)
    store.put_watch(watch)
    # Author-written examples, not collected financial observations.
    for observed, text in (
        ("2026-09-22T09:00:00Z", "Document retrieval must respect when a revision was observed."),
        ("2026-09-22T14:00:00Z", "FUTURE_REVISION: a correction observed after the query cutoff."),
    ):
        event = Event("availability", "synthetic", "note", "Document availability",
                      "synthetic:collection/availability", observed,
                      published_at="2026-09-21T08:00:00Z", data={"text": text})
        store.save(watch, Batch(events=[event]), next_due=0)


def run(*, live_jev: bool = False, output: Path | None = None,
        model: str = PINNED_MODEL) -> dict:
    if live_jev and output is None:
        raise ValueError("--live-jev requires --output to preserve the result")
    if live_jev and not os.environ.get("TYPESAFE_API_KEY", "").strip():
        raise ValueError("TYPESAFE_API_KEY is not configured")
    if output is not None:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
    with Store(":memory:" if output is None else output / "example.sqlite3") as store:
        example_store(store)
        documents = observed_documents(store, AS_OF)
    index = RAGIndex.from_documents(documents)
    query = "document retrieval availability observed revision"
    baseline = RAGPipeline(index).prepare(query, as_of=AS_OF, top_k=1)
    result = {"mode": "live_jev" if live_jev else "local_retrieval",
              "fixture_exposure": "public_synthetic", "as_of": AS_OF,
              "live_inference": False, "baseline": baseline, "evidence": baseline}
    # Current methods guidance is kept distinct from historical source evidence.
    result["current_method_guidance"] = RAGPipeline(
        RAGIndex.from_skills(["backtest-validation"])
    ).prepare("point in time information leakage", top_k=1)
    result["model_components"] = [card["id"] for card in model_catalog()
                                  if card["id"] in ("jev", "fly_memory")]
    if output is not None:
        (output / "plan.json").write_text(json.dumps({
            "mode": result["mode"], "requested_model": model,
            "maximum_provider_calls": int(live_jev), "baseline": baseline,
            "interpretation": "Public composition example, not a model-quality experiment.",
        }, indent=2), encoding="utf-8")
    if live_jev:
        jev = create_model("jev", model=model, allow_network=True, timeout=30)
        try:
            result["evidence"] = RAGPipeline(index, reranker=jev.rerank).prepare(
                query, as_of=AS_OF, top_k=1)
            result["live_inference"] = result["evidence"]["reranking"] is not None
        except Exception as exc:
            # No retry, no simulated answer, no provider body/credential in the receipt.
            result.update(status="provider_failure", error_type=type(exc).__name__)
            result["evidence"] = None
            (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            raise RuntimeError("Jev failed; result.json records failure without a replacement answer") from None
    result["status"] = "context_prepared"
    if output is not None:
        (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live-jev", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model", default=PINNED_MODEL)
    args = parser.parse_args()
    result = run(**vars(args))
    assert "FUTURE_REVISION" not in result["evidence"]["context"]
    print("MODE:", result["mode"])
    print("LIVE JEV INFERENCE:", result["live_inference"])
    print("DATABASE: SQLite; public synthetic revisions")
    print("SOURCE:", result["evidence"]["citations"][0]["source"])
    print("FUTURE REVISION EXCLUDED: True")
    print("MODEL COMPONENTS:", ", ".join(result["model_components"]))
    print("TAKEAWAY")
    print("  Filter availability before selecting each record's latest revision.")
    print("  Current skill guidance is separate from historical document evidence.")
    print("  Pass evidence['messages'] to your generator; no generated answer is claimed here.")


if __name__ == "__main__":
    main()
