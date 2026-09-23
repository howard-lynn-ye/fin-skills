from benchmarks.library_workflows.component_comparison import run


def test_correct_native_sql_and_library_use_the_same_expected_outputs(tmp_path):
    result = run(tmp_path / "comparison", sizes=(8,), seeds=(11,), repetitions=2)
    assert result["complete"]
    assert {r["method"] for r in result["summary"]} == {"native_sql", "library_composition"}
    assert all(r["queries"] == r["exact_queries"] == 6 for r in result["summary"])
    for row in result["rows"]:
        assert row["selected"] == row["expected"]
        assert row["repetitions_correct"] == 2
