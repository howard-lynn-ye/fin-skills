"""Historical scoring must not certify answer isolation or hide incomplete batches."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def legacy(tmp_path):
    spec = importlib.util.spec_from_file_location("legacy_eval", ROOT / "scripts/eval_blind.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.EVALS = tmp_path
    module.QUERIES = tmp_path / "queries.jsonl"
    module.QUERIES.write_text("\n".join(json.dumps({"q": f"q{i}", "expect": "alpha"})
                                         for i in range(6)), encoding="utf-8")
    for i in range(3):
        (tmp_path / f"_blind_batch{i}.json").write_text('["alpha", "alpha"]')
    return module


@pytest.mark.parametrize("answers", ['["alpha", "alpha"]',
                                     '{"2":{"pick":"alpha"},"1":"alpha"}'])
def test_complete_legacy_answers_replay_with_explicit_limitation(legacy, capsys, answers):
    (legacy.EVALS / "_blind_batch0.json").write_text(answers)
    assert legacy.score() == 0
    output = capsys.readouterr().out
    assert "6/6" in output
    assert "isolation unverified" in output


@pytest.mark.parametrize("bad", [None, '["alpha"]', '{"2":"alpha","3":"alpha"}',
                                '{"1":"alpha","1":"beta","2":"alpha"}',
                                'null', '"aa"', '{', '[]'])
def test_missing_or_invalid_batch_refuses_partial_score(legacy, capsys, bad):
    path = legacy.EVALS / "_blind_batch1.json"
    if bad is None:
        path.unlink()
    else:
        path.write_text(bad)
    assert legacy.score() == 1
    output = capsys.readouterr().out
    assert "no complete score" in output
    assert "top-1 accuracy" not in output
