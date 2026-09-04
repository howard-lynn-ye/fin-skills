---
name: rl-and-ml-trading
description: >-
  Reinforcement learning and deep learning for trading — which packages actually install, and what
  the evidence says about whether any of it beats a linear model. Covers FinRL, FinRL-Meta,
  ElegantRL, stable-baselines3, gymnasium, gym-anytrading, TensorTrade, Qlib's RL module, and the
  LSTM/Transformer-versus-linear literature. TRIGGER — use before building or recommending an RL
  trading agent, a deep-learning return predictor, or a custom trading gym environment; when asked
  whether RL or deep learning works for trading; when choosing between LSTM, Transformer and linear
  models for returns; or when FinRL, gym or TensorTrade fail to import.
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-04"
---

# RL and deep learning for trading

Two separate questions, both with unwelcome answers: **does the software work**, and **does the
method work**. Section 1 is verified installation state; section 2 is the evidence.

## 1. 🚨 What actually installs

| Package | Version | Released | `requires_dist` | State |
|---|---|---|---|---|
| 🔴 **`finrl`** | 0.3.7 | 2024-04-12 | **NONE** | **BROKEN — see below** |
| 🔴 `elegantrl` | 0.3.6 | 2023-02-07 | **NONE** | same failure mode, frozen |
| 🔴 `finrl-meta` | 0.3.6 | 2023-02-07 | — | frozen |
| 🔴 `gym` | 0.26.2 | 2022-10-04 | NONE | **repo `archived: true`** — dead |
| 🔴 `tradinggym` | — | — | — | **does not exist on PyPI (404)** |
| ⚠️ `tensortrade` | 1.0.4 | 2026-02-06 | 21 deps | needs **Python ≥3.12 + TensorFlow**; its own deps (`ta`, `stochastic`) are stale two levels deep |
| ✅ **`gymnasium`** | 1.3.0 | 2026-04-22 | 56 deps | the live successor to `gym` |
| ✅ **`stable-baselines3`** | 2.9.0 | 2026-06-15 | 27 deps | healthy |
| ✅ `gym-anytrading` | 2.0.0 | 2023-08-27 | — | small, works with gymnasium |

🚨 **FinRL is broken, not merely stale.** ✅ Independently verified twice: the 0.3.7 wheel declares
**`requires_dist: None`**, so **`pip install finrl` installs zero dependencies** and `import finrl`
dies on `ModuleNotFoundError: gymnasium`. Reproduced in a clean virtualenv. PyPI has been frozen
since 2024-04-12; `setup.py` says 0.3.8, which was never published.

⚠️ **AI4Finance's apparent activity is README churn.** FinRL's last 100 commits: **93 in 2026-03,
mostly README edits**; 4 in 2026-04; 3 in 2026-07 — a Colab link fix and a one-line `threading.Thread`
bug. ElegantRL's recent log is six README edits and a commit titled `r`. **Commit counts on these
repos are not a maintenance signal.**

✅ **The healthy stack was installed and run:** gymnasium 1.3.0 + stable-baselines3 2.9.0 +
torch 2.14.0 + gym-anytrading, with the correct **5-tuple `step()`** return
(`obs, reward, terminated, truncated, info`) — note this differs from `gym`'s 4-tuple, which is the
usual reason old tutorials crash.

⚠️ **Qlib's RL module froze functionally in 2023** and is scoped to **order execution only** — which,
per §3, is the one place RL has a real case.

## 2. 🚨 Does RL work for trading? The evidence

**No RL trading result has been independently replicated.** Systematic arXiv searches for critical or
negative work return essentially only papers claiming wins — **that is a publication filter, not
evidence of success.**

**FinRL's own authors say so.** arXiv **2209.05559** (Gort, Xiao-Yang Liu et al.) exists specifically
to *reject overfitted agents*, and FinRL-Meta's abstract (**2211.03107**) names **low signal-to-noise
ratio, survivorship bias and backtest overfitting** as the field's problems.

**Deep RL fails reproducibility even in clean simulators**, where the environment is free and exact:
- **Henderson et al., arXiv 1709.06560** — random seeds alone flip conclusions.
- **Engstrom et al., arXiv 2005.12729** — undocumented **code-level tricks**, not the algorithm,
  explain PPO's reported gain over TRPO.
- **Agarwal et al., arXiv 2108.13264** — point estimates on a handful of runs are statistically
  meaningless.

If RL cannot be reproduced in Atari, where data is unlimited and the dynamics are exact, the prior
for a noisy, non-stationary, 30-year-sample financial environment should be very low.

### 🔑 The two arguments that settle it

**Arithmetic.** Thirty years of daily bars is **~7,560 observations**. Atari benchmarks use **~200
million frames**. RL is a sample-hungry method being applied to one of the smallest datasets in
machine learning.

**The MDP is fake.** In every mainstream trading gym, **the agent's actions do not affect state
transitions at all** — prices evolve from a fixed replay of history regardless of what the agent
does. Without action→state feedback there is no sequential decision problem: **it is supervised
prediction wearing an MDP costume**, and it inherits none of RL's justification while paying all of
its sample-efficiency costs.

🔑 **Where RL genuinely applies: execution and market making.** There, your actions *do* move the
state — your order changes the book, your quotes change your inventory and adverse selection. That is
a real MDP, it is what Qlib's RL module targets, and it is where the credible work lives.

## 3. 🚨 Deep learning vs linear models for returns

| Evidence | Finding |
|---|---|
| **DLinear**, arXiv **2205.13504** | A single linear layer **beat every Transformer** tested on long-horizon forecasting |
| **GBRT**, arXiv **2101.02118** | Gradient-boosted regression trees **matched** deep models |
| **Kang, arXiv 2601.07131** | Decisive: **linear Sharpe 1.30 / +272.6%** vs **ICA-Wavelet-LSTM 0.07 / −5.1%**; raw LSTM collapsed to the unconditional mean at a **47.5% hit rate** |
| **Chen / Hanauer / Kalsbach** | **Design choices alone span Sharpe 0.08–1.82** on the same problem — the pipeline, not the model, determines the result |

That last row is the important one. **When arbitrary design choices move Sharpe by a factor of 20,
a reported Sharpe is a statement about the researcher, not the model.** Every one of those choices is
a trial — see `../../fin-core/skills/backtest-validation/SKILL.md`.

**A defensible protocol if you proceed anyway:**
1. **Beat a linear baseline first**, on the same features, same CV, same costs. If it does not, stop.
2. **Purged/combinatorial CV**, not a random split — `purgedcv` or `skfolio`.
3. **Multiple seeds**, reported as a distribution with intervals, never a single run (Agarwal).
4. **Count every architecture, hyperparameter and feature set as a trial**, and deflate accordingly.
5. **Report the cost curve**, not a point estimate.

## 4. Building an environment, if you must

```python
import gymnasium as gym                 # NOT `gym` — that is archived
from stable_baselines3 import PPO
# gymnasium's step() returns 5 values: obs, reward, terminated, truncated, info
```

Design pitfalls specific to trading environments:
- **Reward shaping is where the leakage hides.** A reward using the next bar's close is look-ahead;
  a Sharpe-based reward computed over the whole episode leaks the episode's own future.
- **The observation must be causally available** at the decision timestamp — the same rules as any
  feature. Run `assert_causal` from
  `../../fin-core/skills/signal-construction/scripts/assert_causal.py`.
- **Include costs in the reward**, not as a post-hoc adjustment, or the agent learns to churn.
- **Episode boundaries leak** if the agent sees state carried across a train/test split.
- **Actions must not be able to consume more liquidity than the bar had** — otherwise the agent
  discovers infinite size.

## 5. What to tell someone who wants to do this

Not "no". The honest framing is: **the software is mostly broken, the method is unreplicated, the
sample is three orders of magnitude too small, and the environment is not a real MDP — but execution
and market making are a genuine application, and a linear baseline is the benchmark you have to beat
before any of it matters.**
