"""Historical revision and opt-in network behavior of the composed example."""
import pytest

from examples.collected_rag import AS_OF, example_store, observed_documents, run
from fin_skills.collect import Store


def test_eligible_old_revision_survives_future_latest():
    with Store(":memory:") as store:
        example_store(store)
        assert "FUTURE_REVISION" in store.events(latest=True)[0]["data"]["text"]
        documents = observed_documents(store, AS_OF)
    assert len(documents) == 1
    assert "FUTURE_REVISION" not in documents[0].text
    assert documents[0].metadata["seq"] == 1
    assert documents[0].available_at == "2026-09-22T09:00:00+00:00"


def test_empty_history_and_timezone_validation():
    with Store(":memory:") as store:
        example_store(store)
        assert observed_documents(store, "2026-09-21T12:00:00Z") == []
        with pytest.raises(ValueError, match="timezone"):
            observed_documents(store, "2026-09-22T12:00:00")


def test_live_request_needs_credentials_and_new_output(tmp_path, monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    output = tmp_path / "example"
    with pytest.raises(ValueError, match="TYPESAFE_API_KEY"):
        run(live_jev=True, output=output)
    assert not output.exists()
    result = run(output=output)
    assert not result["live_inference"]
    assert result["evidence"]["reranking"] is None
    assert (output / "example.sqlite3").is_file()
    with pytest.raises(FileExistsError):
        run(output=output)
