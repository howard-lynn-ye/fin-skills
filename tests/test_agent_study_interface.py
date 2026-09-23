"""Exercise the exported task interface, including the previous blank CSV header fault."""
import json

import pandas as pd
import pytest

from benchmarks.agent_study.build_task import export
from benchmarks.agent_study.open_agent_runner import AgentWorkspaceSession, CONDITIONS


@pytest.fixture(scope="module")
def exported_task(tmp_path_factory):
    task = tmp_path_factory.mktemp("agent-interface")
    export(11, task)
    return task


@pytest.mark.parametrize("filename", ["close_quoted.csv", "volume.csv", "llm_score.csv"])
def test_exported_matrix_supports_documented_date_reader(exported_task, filename):
    manifest = json.loads((exported_task / "manifest.json").read_text())
    path = exported_task / "data" / filename
    named = pd.read_csv(path, index_col="date", parse_dates=["date"])
    positional = pd.read_csv(path, index_col=0, parse_dates=True)
    pd.testing.assert_frame_equal(named, positional)
    assert isinstance(named.index, pd.DatetimeIndex)
    assert not named.index.has_duplicates
    assert list(named.columns) == manifest["tickers"]
    assert manifest["task_interface_version"] == 2
    assert manifest["matrix_csv_schema"]["index_column"] == "date"


def test_all_conditions_receive_the_same_io_contract(exported_task):
    task = (exported_task / "TASK.md").read_text()
    assert 'index_col="date", parse_dates=["date"]' in task
    assert '"guards_cited": []' in task
    for condition in CONDITIONS:
        prompt = AgentWorkspaceSession(exported_task, condition).system_prompt()
        assert "Write both submission.py and report.json before inspecting" in prompt
        assert "date-indexed weights with ticker columns" in prompt
