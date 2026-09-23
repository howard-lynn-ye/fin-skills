"""The temporal measurement must detect wrong ordering, omissions and rollback failures."""
from benchmarks.library_workflows.temporal import run


def test_reference_and_negative_controls_discriminate(tmp_path):
    result = run(tmp_path / "study", seeds=(11,), records=12)
    summary = result["summary"]
    assert summary["versioned_library"]["exact_queries"] == summary["versioned_library"]["queries"]
    assert summary["latest_unfiltered"]["wrong_versions"] > 0
    assert summary["latest_then_filter"]["omitted_records"] > 0
    for row in result["storage"]:
        assert row["reopen_results_identical"]
        for arm in ("uninterrupted", "reopened"):
            assert row[arm]["duplicate_revisions"] == 0
            assert row[arm]["rollback_ok"]


def test_frozen_result_cannot_be_overwritten(tmp_path):
    import pytest
    output = tmp_path / "study"
    run(output, seeds=(23,), records=4)
    with pytest.raises(FileExistsError):
        run(output, seeds=(23,), records=4)
