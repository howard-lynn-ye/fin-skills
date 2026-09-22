import json
from benchmarks.agent_study.summarize_matrix import summarize


def test_missing_and_ungradable_cells_do_not_turn_into_success(tmp_path):
    cell = {"market_seed": 11, "repetition": 0, "condition": "no_library"}
    (tmp_path / "protocol.json").write_text(json.dumps({"model": "test", "cells": [cell]}))
    report = summarize([tmp_path])
    assert not report["all_planned_cells_completed"]
    assert report["groups"][0]["incorrect_accepted_per_attempt"] is None
    run = tmp_path / "s11-r0-no_library"
    run.mkdir()
    (run / "result.json").write_text(json.dumps({"accepted": True, "termination": "accepted",
        "turns": 1, "wall_seconds": 1, "usage": [{"total_tokens": 20}]}))
    report = summarize([tmp_path])["groups"][0]
    assert report["ungradable_accepted"] == 1
    assert report["incorrect_accepted_per_attempt"] is None
    (run / "grade.json").write_text(json.dumps({"sharpe_gap": 3, "leakage_rate": 0,
        "same_session_rate": 0, "post_delisting_mass": 0}))
    report = summarize([tmp_path])["groups"][0]
    assert report["incorrect_accepted_per_attempt"] == 1
