"""fin_skills.core.trial_ledger - the append-only trial count the Deflated Sharpe Ratio needs."""
from __future__ import annotations

import json
import tempfile

import numpy as np
import pytest

from fin_skills.core.trial_ledger import TrialLedger


@pytest.fixture
def ledger(tmp_path):
    return TrialLedger(tmp_path / "research" / "trials.jsonl")


def test_ledger_creates_its_directory_and_ids_are_content_hashes(ledger, tmp_path):
    assert (tmp_path / "research").is_dir() and not ledger.path.exists()
    tid = ledger.record("ma_cross", {"fast": 10, "slow": 50})
    assert len(tid) == 16 and int(tid, 16) >= 0
    assert tid == TrialLedger(tmp_path / "other.jsonl").record("ma_cross", {"slow": 50, "fast": 10})
    assert tid != ledger.record("ma_cross", {"fast": 11, "slow": 50})


def test_file_is_append_only_jsonl(ledger):
    tid = ledger.record("s", {"p": 1}, note="first")
    ledger.complete(tid, {"sharpe": 0.4, "n_obs": 100})
    ledger.abandon(tid, "changed my mind")
    lines = ledger.path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["event"] for line in lines] == ["registered", "completed", "abandoned"]
    assert [r["event"] for r in ledger.read()] == ["registered", "completed", "abandoned"]
    assert ledger.read()[0]["note"] == "first"


def test_summary_counts_every_registration_including_abandoned(ledger):
    a = ledger.record("s", {"p": 1})
    b = ledger.record("s", {"p": 2})
    ledger.complete(a, {"sharpe": 1.0})
    ledger.abandon(b, "slow")
    s = ledger.summary()
    assert s["n_trials"] == 2 and s["n_completed"] == 1 and s["n_abandoned"] == 1
    assert s["best_sharpe"] == 1.0 and s["sharpe_variance"] is None
    assert TrialLedger(ledger.path.with_name("empty.jsonl")).summary()["n_trials"] == 0


def test_dsr_needs_two_completed_trials(ledger):
    tid = ledger.record("s", {"p": 1})
    ledger.complete(tid, {"sharpe": 1.0})
    assert "error" in ledger.deflated_sharpe(best_sharpe=1.0, n_obs=1000)


def test_best_of_fifty_noise_strategies_dies_under_dsr(ledger):
    rng = np.random.default_rng(0)
    for fast in range(5, 55, 5):
        for slow in range(60, 160, 20):
            tid = ledger.record("ma_cross", {"fast": fast, "slow": slow})
            ledger.complete(tid, {"sharpe": float(rng.normal(0, 0.45)), "n_obs": 1260})
    s = ledger.summary()
    assert s["n_trials"] == 50
    dsr = ledger.deflated_sharpe(best_sharpe=s["best_sharpe"], n_obs=1260)
    assert dsr["n_trials"] == 50 and dsr["expected_max_sharpe_from_noise"] > 0
    assert dsr["deflated_sharpe_ratio"] < 0.95
    assert dsr["verdict"] == "NOT distinguishable from noise"
    assert ledger.deflated_sharpe(best_sharpe=5.0, n_obs=1260)["verdict"] == "survives"


def test_demo_writes_only_into_the_temp_dir_it_asks_for(run_main, monkeypatch, tmp_path):
    demo_dir = tmp_path / "demo_ledger"
    demo_dir.mkdir()
    monkeypatch.setattr(tempfile, "mkdtemp", lambda *a, **k: str(demo_dir))
    out = run_main("fin_skills.core.trial_ledger")
    assert (demo_dir / "trials.jsonl").is_file()
    assert "dies under DSR" in out
