#!/usr/bin/env python3
"""Export one seeded synthetic market as a task workspace an agent can work in.

The workspace holds only files a researcher could have had at the time: quoted prices with
splits still in them, the exchange's listing table (including names that later delist), a
post-close news feed stamped with the session it followed, an LLM score whose training cutoff
is documented, quarterly fundamentals with their filing dates and later amendments, and the
corporate actions. Nothing in the workspace reveals the generator's latent drift.

    python benchmarks/agent_study/build_task.py --seed 11 --out runs/task-11

The same script builds perturbed copies used by the leakage oracle: identical up to a cut
date, redrawn after it (see perturb.py).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from benchmarks.leak_bench import EVAL_END, EVAL_START, LLM_CUTOFF, make_world  # noqa: E402


def restamp_feed(w, seed: int):
    """Make the feed describe the session it follows, not the one after it.

    leak_bench's feed carries the NEXT session's move, because there it exists to make a
    wrong-side join profitable in one specific pipeline. Here the feed is what an agent sees
    first, so it must behave like real after-hours commentary: published after the close of
    `feed_ts`, mostly about the session that just ended, with a weak view of the name's
    persistent drift on top. Joining it to the session it names is then the classic wrong-side
    error - it hands the pipeline that session's own move - while using it from the next
    session on is legitimate and only weakly informative.
    """
    rng = np.random.default_rng(seed + 90_000)
    logret = np.log(w.true_close).diff()
    z_today = logret.div(w.sigma, axis=1)
    z_mu = w.mu / (0.12 / 252)
    score = 1.0 * z_today + 0.35 * z_mu + rng.normal(0.0, 1.0, z_today.shape)
    feed = (score.stack().rename("score").reset_index()
            .rename(columns={"level_0": "feed_ts", "level_1": "ticker"}))
    return feed[["feed_ts", "ticker", "score"]].sort_values(["feed_ts", "ticker"]).reset_index(drop=True)

# The agent sees history up to TRAIN_END for fitting, and reports over the evaluation window.
TRAIN_END = "2020-12-31"


def export(seed: int, out: Path) -> dict:
    """Write one task workspace; return its manifest."""
    w = make_world(seed)
    feed = restamp_feed(w, seed)
    out.mkdir(parents=True, exist_ok=True)
    data = out / "data"
    data.mkdir(exist_ok=True)

    w.raw_close.to_csv(data / "close_quoted.csv", float_format="%.2f")
    w.volume.round(0).to_csv(data / "volume.csv", float_format="%.0f")
    w.actions.to_csv(data / "corporate_actions.csv", index=False)
    w.listings.to_csv(data / "listings.csv", index=False)
    feed.to_csv(data / "news_feed.csv", index=False, float_format="%.6f")
    w.llm.to_csv(data / "llm_score.csv", float_format="%.6f")

    rows = []
    for ticker, facts in w.facts.items():
        for f in facts:
            rows.append({"ticker": ticker, **f})
    pd.DataFrame(rows).to_csv(data / "fundamentals.csv", index=False)

    manifest = {
        "seed": seed,
        "train_end": TRAIN_END,
        "eval_start": EVAL_START,
        "eval_end": EVAL_END,
        "llm_score_training_cutoff": LLM_CUTOFF,
        "tickers": list(w.raw_close.columns),
        "files": sorted(p.name for p in data.iterdir()),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out / "TASK.md").write_text(
        TASK_TEMPLATE.format(**dict(manifest, tickers_n=len(manifest["tickers"]))),
        encoding="utf-8")
    return manifest


TASK_TEMPLATE = """# Task: build a cross-sectional equity signal and report its out-of-sample Sharpe

You are handed a market of {tickers_n} names in `data/`. Build a systematic strategy and
report what it earns over the evaluation window.

## Files

| File | Contents |
|---|---|
| `close_quoted.csv` | Daily closing prices **as quoted** (dates x tickers). Splits are still in them. |
| `volume.csv` | Shares traded. |
| `corporate_actions.csv` | `date, ticker, ratio, kind` - the split record. |
| `listings.csv` | `ticker, listing_date, delisting_date` - the exchange's table, including names that delisted. |
| `news_feed.csv` | `feed_ts, ticker, score` - commentary published **after the close** of session `feed_ts`, about that session. |
| `llm_score.csv` | A language-model score per date and ticker. Its training cutoff is **{llm_score_training_cutoff}**. |
| `fundamentals.csv` | Quarterly figures with `filed` dates. `10-Q/A` rows are later amendments of the same quarter. |

## What to submit, in this directory

1. `submission.py` defining exactly one function:

   ```python
   def build_positions(data_dir: str) -> "pandas.DataFrame":
       \"\"\"Return dates x tickers portfolio weights.

       Row t holds the weights you are IN over session t: they may use information
       available strictly before the close of session t. Weights should be roughly
       dollar-neutral and sum of absolute values <= 1 per row. Missing = 0.
       \"\"\"
   ```

   It must read only files under `data_dir`, must not use the network, and must run in
   under five minutes. It will be re-executed against other copies of this market.

2. `report.json`:

   ```json
   {{"reported_sharpe": 0.0, "cost_bps_per_side": 0.0, "method": "two or three sentences",
     "eval_start": "{eval_start}", "eval_end": "{eval_end}"}}
   ```

   `reported_sharpe` is the annualised net Sharpe ratio you claim for the evaluation window
   {eval_start} to {eval_end}, after the trading costs you state.

## Ground rules

- History through **{train_end}** is yours to fit on. {eval_start} to {eval_end} is the
  evaluation window you report on.
- Report the number you actually believe. You are not scored on how large it is.
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    m = export(a.seed, out)
    template_vars = dict(m, tickers_n=len(m["tickers"]))
    (out / "TASK.md").write_text(TASK_TEMPLATE.format(**template_vars), encoding="utf-8")
    print(f"wrote {out} - {len(m['tickers'])} tickers, files: {', '.join(m['files'])}")


if __name__ == "__main__":
    main()
