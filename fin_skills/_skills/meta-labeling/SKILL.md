---
name: meta-labeling
description: >-
  A primary model picks the side, a secondary model trained on "was the primary right" decides
  whether to act - raising precision, lowering recall, and paying for itself in costs. The trap
  is training the secondary on rows the primary was fitted on. TRIGGER - meta-labeling,
  metalabeling, meta labels, secondary model, primary model side, get_events side_prediction,
  bin in {0,1}, "should I take this signal", precision vs recall trade-off in trading, F1 of a
  trading model, filter model, trade filter, "my model has high recall but loses money", "how do
  I improve a strategy without changing its signal", Lopez de Prado chapter 3 section 3.6, AFML
  meta-labeling. SKIP for producing the triple-barrier labels and the `side` argument itself
  (triple-barrier-labeling), for turning the secondary's probability into a position size
  (bet-sizing), for the Kelly fraction (position-sizing-kelly), for purged cross-validation of
  either model (lib-purgedcv), and for which features matter (feature-importance-financial).
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-09"
---

# Meta-labeling

**Two models, two questions.** The primary answers "long or short". The secondary answers "is the
primary right this time", as a binary {0, 1} problem, from features about the *state* of the
market rather than its direction. Advances in Financial Machine Learning (Lopez de Prado 2018),
chapter 3, section 3.6. It is one of the few techniques that improves a strategy without touching
its signal — and it has one failure mode that silently disables it.

**The failure mode: the secondary must be trained on rows where the primary's prediction was out
of sample.** Fit the primary, predict on its own training rows, and those predictions are better
than the primary can actually do. The meta-labels are then too optimistic, the secondary learns
"act almost always", and you have added a component that filters nothing.

Every number below is printed by `scripts/meta_labeling.py` (numpy, seed 0, **2.9 s**,
`scikit-learn` optional).

## 1. The setup, so the numbers mean something

24,000 bars from a hidden two-state chain (`P(stay) = 0.985`, mean regime length **67 bars**). In
state 1 momentum continues; in state 0 it reverses; state 0 is also the noisier one. ✅ Measured
realized volatility: **0.01584 in state 0 against 0.01150 in state 1** — so volatility is a real
but imperfect clue about which state you are in, which is exactly the kind of thing a secondary
model can learn and a directional model cannot use.

**4,759 non-overlapping 5-bar bets**, split into thirds: primary-train / meta-train / test. Bets
are spaced by the holding period on purpose — overlap is a separate problem with a separate fix
(`../sample-weights-and-uniqueness/SKILL.md`).

The primary is a logistic regression on the momentum feature **plus 100 pure-noise columns**. That
is not a straw man: it is what every real primary model has — capacity beyond its signal — and it
is the whole reason section 3 happens.

✅ The script's numpy IRLS logistic matches `sklearn.linear_model.LogisticRegression(solver="lbfgs",
C=1/ridge)` to **2.8e-09** in predicted probability on the same standardised design, so the
dependency-free implementation is checked rather than assumed.

## 2. What meta-labeling buys, measured

The positive class is "this bet was profitable". The primary alone acts on every signal, so its
recall is 1.0 by construction and its precision is its hit rate. ✅ On the held-out third:

| | precision | recall | F1 | bets taken | coverage |
|---|---|---|---|---|---|
| primary alone | 0.6408 | **1.0000** | **0.7811** | 1,587 | 100.0 % |
| **primary + meta** | **0.7200** | 0.7483 | 0.7338 | 1,057 | **66.6 %** |

✅ **Precision +0.0791 (+12.3 % relative), recall -0.2517.** The secondary turns 33 % of the
signals into no-trades. That is the entire mechanism: it converts false positives into missed
opportunities.

🚨 **F1 goes DOWN (0.7811 → 0.7338), and that is fine.** F1 weights precision and recall equally;
a trader does not. A missed trade costs zero plus opportunity; a taken loser costs the loss plus
the round trip. **Do not select a meta model on F1** — select it on the after-cost objective in
section 4. If you report F1 at all, report it next to coverage so the reader can see what was
given up.

## 3. 🚨 The trap: training the secondary on the primary's own rows

✅ Measured on the same data, changing one thing — the meta model is trained on the *first* third,
where the primary's predictions were made in sample:

- the primary is right on **0.699** of its own training rows and on **0.626** of fresh ones — a
  gap of **+0.073** that the meta model cannot see;
- so the meta-label base rate it trains on is **0.699 instead of 0.626**.

| on the test third | precision | recall | F1 | coverage |
|---|---|---|---|---|
| primary alone | 0.6408 | 1.0000 | 0.7811 | 100.0 % |
| primary + **honest** meta | **0.7200** | 0.7483 | 0.7338 | 66.6 % |
| primary + **leaked** meta | 0.6516 | 0.9381 | 0.7690 | **92.2 %** |

🚨 **The leaked secondary acts on 92.2 % of signals instead of 66.6 %, and its test precision is
0.0683 below the honest one** — barely above the unfiltered primary's 0.6408. It was trained to
believe the primary is right 70 % of the time, so it almost never says no. The component is
present, it runs, it produces probabilities, and it does nothing.

🚨 **Its higher F1 (0.7690 vs 0.7338) makes it look better.** The leaked model wins on the metric
people report and loses on the money.

## 4. Sharpe after costs

✅ Annualised over the test third, cost charged on every bet actually taken:

| cost per bet | primary alone | primary + meta | primary + leaked meta |
|---|---|---|---|
| 0 bp | 3.312 | **3.468** | 3.346 |
| 5 bp | 3.252 | **3.432** | 3.291 |
| 20 bp | 3.071 | **3.320** | 3.125 |
| **50 bp** | 2.710 | **3.092** | 2.793 |

- ✅ The honest meta model adds **+0.156 Sharpe at zero cost and +0.382 at 50 bp**: it improves the
  bet selection *and* it stops paying for the bets it drops, so the gap widens with cost. That is
  the argument for meta-labeling in one row.
- 🚨 The leaked model captures **+0.083 of that +0.382 at 50 bp — 22 %.** Three quarters of the
  benefit is destroyed by a train/predict split nobody would see in a code review.

## 5. The probability is a size, not only a switch

✅ The threshold is a free parameter and it trades coverage against precision:

| threshold | coverage | precision | Sharpe @ 20 bp |
|---|---|---|---|
| 0.40 | 94.2 % | 0.6475 | 3.107 |
| 0.50 | 66.6 % | 0.7200 | 3.320 |
| 0.55 | 55.1 % | 0.7543 | 3.360 |
| 0.60 | 47.6 % | 0.8122 | 3.541 |
| **0.65** | **41.8 %** | **0.8522** | **3.630** |

✅ A continuous size `clip(2 * (p - 0.5), 0, 1)` instead of a switch: Sharpe **3.433** at 20 bp
with a mean gross exposure of **0.317** — close to the best threshold's Sharpe on a third of the
capital deployed.

🚨 **The threshold is a researcher degree of freedom.** Five thresholds tried on the test set is
five trials; the table above is reported *after* the split precisely so it can be counted. Record
it in `../../../fin-core/skills/backtest-validation/scripts/trial_ledger.py`. The sizing rule
itself belongs to `../bet-sizing/SKILL.md`, and the Kelly fraction to
`../../../fin-strategies/skills/position-sizing-kelly/SKILL.md`.

## 6. Traps

- 🚨 **Meta-labels built from in-sample primary predictions.** Section 3. Use a walk-forward or
  purged cross-validated prediction of the primary
  (`../../../fin-libraries/skills/lib-purgedcv/SKILL.md`) and label *those*. If the primary is a
  fixed rule with no fitted parameters, this trap does not apply — but check that "no parameters"
  is true, including the ones you chose by looking at the data.
- 🚨 **Selecting the meta model on F1 or accuracy.** Section 2 and section 3: the leaked model wins
  on F1. Select on after-cost Sharpe, or on precision at a fixed coverage.
- 🚨 **Giving the secondary the primary's direction features.** The secondary's job is *state*, not
  direction. Feed it the same momentum feature and it re-learns the primary's own errors instead
  of the conditions under which they happen. Its input here is volatility, an efficiency ratio,
  `|momentum|` and mean absolute return — all sign-free.
- 🚨 **Standardising features on the full sample.** The script standardises inside `fit_logistic`
  using training rows only and carries `mu`/`sd` to prediction time. A single
  `(X - X.mean()) / X.std()` over the whole array before splitting leaks the test set's moments
  into training. Check with
  `../../../fin-core/skills/signal-construction/scripts/assert_causal.py`.
- 🚨 **Meta-labels are unbalanced and the imbalance is informative.** A base rate of 0.63 means
  "always act" scores 63 % accuracy. Report the base rate next to every accuracy figure or the
  number means nothing.
- ⚠️ **Meta-labeling cannot rescue a primary with no edge.** It reallocates the primary's hit rate
  across states; if the hit rate is 0.50 everywhere there is nothing to condition on. Check that
  the primary's out-of-sample hit rate varies by state *before* building the secondary.
- 🚨 **Three splits is three chances to leak.** Primary-train, meta-train and test must be disjoint
  *in time*, and with overlapping labels they must also be purged and embargoed. Here they are
  non-overlapping by construction; in a triple-barrier pipeline they are not
  (`../triple-barrier-labeling/SKILL.md`, `../sample-weights-and-uniqueness/SKILL.md`).
- ⚠️ **The book's `get_events(side_prediction=...)` path changes the label set.** With `side`
  supplied, `get_bins` returns `bin` in {0, 1} and forces a 0 whenever the realised return is
  non-positive, whatever barrier was touched — ✅ source-verified in mlfinpy 0.1.2
  (`mlfinpy/labeling/labeling.py`). A meta-label is "did this bet make money", not "which barrier".

## 7. Scripts

- `scripts/meta_labeling.py` — the two-state generator, causal features, non-overlapping bets, a
  numpy IRLS logistic regression checked against scikit-learn, the honest and leaked meta pipelines,
  precision/recall/F1, Sharpe after costs at four levels, and the threshold/continuous-size sweep.
  numpy only; `sklearn` is imported inside one function and the demo says so when it is missing.
  Seed 0, **2.9 s**.

## 8. Where this sits

`../triple-barrier-labeling/SKILL.md` produces the `side` the primary supplies and the {0, 1}
meta-labels themselves. `../bet-sizing/SKILL.md` turns the secondary's probability into a position
and owns averaging and discretisation.
`../../../fin-strategies/skills/position-sizing-kelly/SKILL.md` owns the Kelly fraction — this
skill does not repeat it. `../../../fin-libraries/skills/lib-purgedcv/SKILL.md` owns the purged
cross-validation that produces honest primary predictions on overlapping labels.
`../sample-weights-and-uniqueness/SKILL.md` owns the weights once the bets do overlap.
`../feature-importance-financial/SKILL.md` owns which of the secondary's state features actually
matter. `../../../fin-core/skills/backtest-validation/SKILL.md` owns the trial ledger for the
threshold.
