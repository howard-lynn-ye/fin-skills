#!/usr/bin/env python3
"""Training-cutoff contamination probe for LLM-driven financial backtests.

Why: an LLM has memorized the period you are backtesting. Reported edge in an
LLM trading study is frequently recall, not prediction. The diagnostic signature
(Gao et al., arXiv 2512.23847) is that "lookahead propensity" collapses to ~zero
immediately after the model's training cutoff -- so accuracy that is high before
the cutoff and at-chance after it is memorization, not skill.

This module does two things:

  1. `window_overlap()` -- a zero-API-call structural check. Run this FIRST; it
     invalidates most published results on its own.
  2. `build_probe_set()` / `score_probe()` -- construct a balanced set of
     verifiable questions straddling the cutoff, and score the accuracy split.
     You supply the model-calling function; nothing here calls an API.

Neither check proves absence of contamination. A pass means "not caught",
not "clean".
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import date
from typing import Callable, Sequence


# --- 1. Structural check: does the backtest window overlap the training data? ---

def window_overlap(cutoff: str | date, test_start: str | date, test_end: str | date) -> dict:
    """Compare a model's training cutoff against the backtest window.

    Returns a verdict dict. `contaminated_fraction` is the share of the test
    window that the model may have seen during training.
    """
    def d(x):
        return date.fromisoformat(x) if isinstance(x, str) else x

    cutoff, ts, te = d(cutoff), d(test_start), d(test_end)
    total = (te - ts).days
    if total <= 0:
        raise ValueError("test_end must be after test_start")
    overlap = max(0, (min(cutoff, te) - ts).days)
    frac = overlap / total

    if frac >= 0.999:
        verdict = ("INVALID: the entire backtest window predates the training cutoff. "
                   "The model may be recalling outcomes rather than predicting them.")
    elif frac > 0:
        verdict = (f"CONTAMINATED: {frac:.0%} of the test window predates the cutoff. "
                   f"Report the post-cutoff sub-period separately as the real out-of-sample.")
    else:
        verdict = "CLEAN on this axis: the whole test window is post-cutoff."

    return {
        "training_cutoff": cutoff.isoformat(),
        "test_window": f"{ts.isoformat()}..{te.isoformat()}",
        "contaminated_fraction": round(frac, 4),
        "clean_days": max(0, (te - max(ts, cutoff)).days),
        "verdict": verdict,
    }


# --- 2. Behavioural probe: does accuracy collapse at the cutoff? ---

@dataclass
class ProbeItem:
    """A question with a verifiable answer, tagged by whether it predates the cutoff."""
    asked_about: date
    question: str
    truth: str
    pre_cutoff: bool


def build_probe_set(
    facts: Sequence[tuple[str, str, str]],
    cutoff: str | date,
    seed: int = 0,
) -> list[ProbeItem]:
    """Build a balanced probe set from verifiable facts.

    facts: sequence of (iso_date, question, ground_truth_answer). Use facts the
    model could only know from having seen the period -- e.g. "what was AAPL's
    reported EPS for the quarter announced on <date>", "did <ticker> close up or
    down on <date>". Balance matters: an unbalanced set makes the split
    uninterpretable.
    """
    cut = date.fromisoformat(cutoff) if isinstance(cutoff, str) else cutoff
    items = [ProbeItem(date.fromisoformat(d), q, a, date.fromisoformat(d) < cut)
             for d, q, a in facts]
    pre = [i for i in items if i.pre_cutoff]
    post = [i for i in items if not i.pre_cutoff]
    n = min(len(pre), len(post))
    if n == 0:
        raise ValueError("need facts on BOTH sides of the cutoff to interpret the split")
    rng = random.Random(seed)
    return rng.sample(pre, n) + rng.sample(post, n)


def score_probe(
    items: Sequence[ProbeItem],
    ask: Callable[[str], str],
    match: Callable[[str, str], bool] | None = None,
) -> dict:
    """Ask the model each question and report the pre/post-cutoff accuracy split.

    `ask` is your model call: str -> str. `match` compares (answer, truth);
    defaults to a case-insensitive substring test.
    """
    if match is None:
        def match(ans: str, truth: str) -> bool:
            return truth.strip().lower() in ans.strip().lower()

    pre_ok = pre_n = post_ok = post_n = 0
    for it in items:
        ok = match(ask(it.question), it.truth)
        if it.pre_cutoff:
            pre_n += 1
            pre_ok += ok
        else:
            post_n += 1
            post_ok += ok

    pre_acc = pre_ok / pre_n if pre_n else float("nan")
    post_acc = post_ok / post_n if post_n else float("nan")
    gap = pre_acc - post_acc

    if gap > 0.25:
        verdict = ("MEMORIZATION LIKELY: accuracy drops sharply at the cutoff. "
                   "Pre-cutoff performance is recall; do not report it as predictive skill.")
    elif gap > 0.10:
        verdict = "SUSPICIOUS: a material accuracy gap at the cutoff. Investigate before reporting."
    else:
        verdict = ("No cutoff discontinuity detected. This does NOT prove the result is clean -- "
                   "it means this probe did not catch it.")

    return {
        "pre_cutoff_accuracy": round(pre_acc, 4),
        "post_cutoff_accuracy": round(post_acc, 4),
        "gap": round(gap, 4),
        "n_pre": pre_n,
        "n_post": post_n,
        "verdict": verdict,
    }


if __name__ == "__main__":
    print("--- structural check ---")
    for case in [
        ("2024-06-01", "2020-01-01", "2024-01-01"),   # fully inside training data
        ("2024-06-01", "2023-01-01", "2026-01-01"),   # straddles
        ("2024-06-01", "2025-01-01", "2026-01-01"),   # clean
    ]:
        r = window_overlap(*case)
        print(f"cutoff={r['training_cutoff']}  window={r['test_window']}")
        print(f"  contaminated={r['contaminated_fraction']:.0%}  "
              f"clean_days={r['clean_days']}\n  {r['verdict']}\n")

    print("--- behavioural probe (simulated model that memorized pre-cutoff facts) ---")
    facts = [(f"2024-0{m}-15", f"Q{m}?", f"A{m}") for m in range(1, 6)] + \
            [(f"2025-0{m}-15", f"Q{m + 5}?", f"A{m + 5}") for m in range(1, 6)]
    items = build_probe_set(facts, cutoff="2024-06-01", seed=1)
    truth = {f.question: f.truth for f in items}

    def fake_model(q: str) -> str:
        it = next(i for i in items if i.question == q)
        return truth[q] if it.pre_cutoff else "I don't know"

    res = score_probe(items, fake_model)
    print(f"  pre={res['pre_cutoff_accuracy']:.0%}  post={res['post_cutoff_accuracy']:.0%}  "
          f"gap={res['gap']:.0%}\n  {res['verdict']}")
