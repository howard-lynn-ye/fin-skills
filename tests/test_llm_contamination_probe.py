"""fin_skills.llm.contamination_probe - a backtest the model may have memorised."""
from __future__ import annotations

from datetime import date

import pytest

from fin_skills.llm.contamination_probe import (ProbeItem, build_probe_set, score_probe,
                                                window_overlap)


def test_window_overlap_verdicts():
    inside = window_overlap("2024-06-01", "2020-01-01", "2024-01-01")
    assert inside["contaminated_fraction"] == 1.0 and inside["clean_days"] == 0
    assert inside["verdict"].startswith("INVALID")
    straddle = window_overlap("2024-06-01", "2023-01-01", "2026-01-01")
    assert 0 < straddle["contaminated_fraction"] < 1
    assert straddle["contaminated_fraction"] == pytest.approx(517 / 1096, abs=1e-4)
    assert straddle["clean_days"] == (date(2026, 1, 1) - date(2024, 6, 1)).days
    assert straddle["verdict"].startswith("CONTAMINATED: 47%")
    clean = window_overlap(date(2024, 6, 1), date(2025, 1, 1), date(2026, 1, 1))
    assert clean["contaminated_fraction"] == 0.0 and clean["verdict"].startswith("CLEAN")
    assert clean["test_window"] == "2025-01-01..2026-01-01"
    with pytest.raises(ValueError):
        window_overlap("2024-06-01", "2025-01-01", "2025-01-01")


FACTS = [(f"2024-0{m}-15", f"Q{m}?", f"A{m}") for m in range(1, 6)] + \
        [(f"2025-0{m}-15", f"Q{m + 5}?", f"A{m + 5}") for m in range(1, 6)]


def test_probe_set_is_balanced_seeded_and_tagged():
    items = build_probe_set(FACTS, "2024-06-01", seed=1)
    assert len(items) == 10 and sum(i.pre_cutoff for i in items) == 5
    assert all(isinstance(i, ProbeItem) and (i.asked_about < date(2024, 6, 1)) == i.pre_cutoff
               for i in items)
    assert [i.question for i in items] == [i.question for i in build_probe_set(FACTS, "2024-06-01", seed=1)]
    smaller = build_probe_set(FACTS[:2] + FACTS[5:], "2024-06-01", seed=0)
    assert len(smaller) == 4                                      # min(pre, post) each side
    with pytest.raises(ValueError, match="BOTH sides"):
        build_probe_set(FACTS[:5], "2024-06-01")


def test_score_probe_detects_the_cutoff_discontinuity():
    items = build_probe_set(FACTS, "2024-06-01", seed=1)
    truth = {i.question: i.truth for i in items}
    pre = {i.question for i in items if i.pre_cutoff}

    def memoriser(q):
        return truth[q] if q in pre else "I don't know"

    r = score_probe(items, memoriser)
    assert (r["pre_cutoff_accuracy"], r["post_cutoff_accuracy"], r["gap"]) == (1.0, 0.0, 1.0)
    assert r["n_pre"] == r["n_post"] == 5 and r["verdict"].startswith("MEMORIZATION LIKELY")
    r = score_probe(items, lambda q: truth[q])
    assert r["gap"] == 0.0 and r["verdict"].startswith("No cutoff discontinuity")

    def slightly_leaky(q):
        return truth[q] if q in pre or q in ("Q6?", "Q7?", "Q8?", "Q9?") else "no"

    assert score_probe(items, slightly_leaky)["verdict"].startswith("SUSPICIOUS")
    exact = score_probe(items, lambda q: "the answer is " + truth[q].lower(),
                        match=lambda ans, t: ans == t)
    assert exact["pre_cutoff_accuracy"] == 0.0                      # a custom matcher is honoured


def test_demo_prints_both_checks(run_main):
    out = run_main("fin_skills.llm.contamination_probe")
    assert "--- structural check ---" in out and "MEMORIZATION LIKELY" in out
