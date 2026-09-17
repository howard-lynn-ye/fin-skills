---
name: kol-credibility-registry
description: >-
  [fin-china] Score and calibrate financial KOL credibility across Xueqiu and StockTwits using Bayesian win-rate updating, Brier scores, and contrarian inversion.
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-17"
---

# KOL Credibility Registry

The standard retail sentiment strategy is fatally flawed: **follower count does not equal predictive accuracy**. In quantitative analysis of social sentiment across Chinese and global equities, raw unweighted post counts act as a retail FOMO trap that buys tops and sells bottoms.

The `kol-credibility-registry` skill solves this by providing empirical credibility scoring, Bayesian track-record shrinkage, Brier calibration metrics, and contrarian signal inversion for financial KOLs (Key Opinion Leaders) on **Xueqiu (雪球)**, **StockTwits**, and **FinTwit**.

---

## 1. The Trap of Raw Sentiment

A naive sentiment indicator counts positive vs. negative posts or computes a simple average of LLM sentiment scores. In practice, this guarantees underperformance for three structural reasons:

1. **The Coin-Flip Reality** ⚠️ secondhand:
   An external audit of Xueqiu authors (2018–2026, in the separate `stock_prediction` project, not bundled or re-run here) reported an unweighted 5-day directional win rate of about **49%** — no better than a coin toss.
2. **Mega-Influencers as FOMO Top Chasers**:
   In that external audit, several accounts with 100,000+ followers had 5-day win rates below 40% with negative realized returns. High-follower retail KOLs grow their audience by validating the crowd's excitement at market peaks. Following them directionally leads to buying at distribution tops (主力派发、散户接盘).
3. **Media News Aggregators Have Zero Directional Alpha**:
   News-wire and media aggregator accounts were near a coin flip directionally in the same external audit. Financial news reports *events that already occurred*. They are invaluable for event detection and volume heat, but disastrous as directional trading signals.

```
+-----------------------------------------------------------------------------------+
|                           THE DIVERGENCE PARADOX                                  |
|                                                                                   |
|  2 Contrarian "Big Vs" (575k fans):   "Bull market breakout! All-in margin long!" |
|  1 News Wire Aggregator (155k fans):  "Market rally breaks resistance level."     |
|  2 Alpha Researchers (58k fans):      "Valuation stretched, distribution risk."   |
|                                                                                   |
|  Naive Sentiment Average:             +0.42 (BULLISH TRAP -> Buy the top!)        |
|  KOL Credibility-Weighted Sentiment:  -0.65 (TRUE ALPHA -> Reduce exposure!)      |
+-----------------------------------------------------------------------------------+
```

---

## 2. Bayesian Track Record Formulation

To separate genuine forecasting skill from small-sample luck or high-variance gambling, the registry models author accuracy using a conjugate **Beta-Binomial Bayesian framework** and evaluates forecast reliability with **Brier scores**.

### 2.1 Conjugate Beta-Binomial Updating

Let an author's true directional win probability be $	heta \in [0, 1]$. We assign an empirical prior:

$$	heta \sim 	ext{Beta}(lpha_0, eta_0)$$

Where $lpha_0 = 5.0$ and $eta_0 = 5.0$, encoding a prior expectation of 50% win rate equivalent to 10 pseudo-observations.

Given $n$ forward-evaluated directional predictions containing $k$ successful directional calls (where realized return $R_{t, t+5d} > 0$ for bullish calls or $R_{t, t+5d} < 0$ for bearish calls), the posterior distribution is:

$$	heta \mid k, n \sim 	ext{Beta}(lpha_0 + k, eta_0 + n - k)$$

The Bayesian shrunk win rate $\hat{p}_{	ext{Bayes}}$ is the posterior expectation:

$$\hat{p}_{	ext{Bayes}} = \mathbb{E}[	heta \mid k, n] = rac{lpha_0 + k}{lpha_0 + eta_0 + n} = rac{5 + k}{10 + n}$$

This formulation guarantees:
- An author with 2 wins out of 2 calls ($100\%$ raw win rate) is shrunk to $rac{7}{12} = 58.3\%$, preventing premature amplification.
- An author with 32 wins out of 35 calls ($91.4\%$ raw win rate) retains a high posterior $rac{37}{45} = 82.2\%$.

### 2.2 Brier Calibration Score

Forecasting quality is measured by the quadratic scoring rule:

$$	ext{Brier} = rac{1}{N} \sum_{t=1}^N (f_t - o_t)^2$$

Where $f_t \in [0, 1]$ is the forecasted probability and $o_t \in \{0, 1\}$ is the binary outcome.
- $	ext{Brier} \le 0.15$: Elite calibration with sharp directional accuracy.
- $	ext{Brier} pprox 0.25$: Uninformative baseline (random 50/50 guessing).
- $	ext{Brier} > 0.35$: Severe anti-calibration (consistent contrarian indicator).

### 2.3 Empirical Tier Taxonomy & Multipliers

Each author is classified into one of 6 operational tiers with calibrated directional weights:

| Tier Code | Criteria | Directional Weight | Role in Strategy |
|---|---|---:|---|
| `TIER_0_ELITE_KOL` | 5D Win Rate $>75\%$, Bayesian $>68\%$, Payoff $>1.5$ | **+3.0x** | Core Alpha Amplifier |
| `TIER_1_CORE_ALPHA` | 5D Win Rate $>65\%$, Bayesian $>62\%$, Payoff $>1.2$ | **+2.5x** | High-Conviction Researcher |
| `TIER_2_SOLID_RESEARCHER` | 5D Win Rate $>58\%$, Bayesian $>55\%$ | **+1.5x** | Fundamental Analyst |
| `TIER_MEDIA_AGGREGATOR` | News wire / fast telegraph aggregator | **0.0x** | Direction stripped; volume heat only |
| `TIER_NEUTRAL_RETAIL` | Unindexed or baseline retail poster | **+0.2x** | Low-weight retail baseline |
| `TIER_CONTRARIAN_INDICATOR` | High followers, Win Rate $<42\%$, Payoff $<0.8$ | **-1.5x** | Inverted signal (FOMO top / panic bottom) |

---

## 3. KOL Database Schema

The registry ships only a **small embedded fixture of 40 pseudonymous profiles** (`cn_elite_01`,
`us_contrarian_01`, ...). Their statistics were copied from an external audit in the separate
`stock_prediction` project and are illustrative: no script here reproduces them, and the handles
deliberately do not identify real people. ⚠️ secondhand.

Full databases can be loaded when you have them, from `$FIN_SKILLS_BENCHMARK_DIR`
(default `../stock_prediction/data/benchmark`) or an explicit `csv_path`:
- `XUEQIU_KOL_ALPHA_PROFILES.csv` (Chinese mainland authors)
- `GLOBAL_KOL_ALPHA_PROFILES.csv` (StockTwits / FinTwit authors)

Neither file is bundled or verified by this repository. Before publishing tiers for named real
accounts, keep the evaluation code and data with the results and check platform terms.

### Schema Fields

| Column Name | Type | Description |
|---|---|---|
| `author` | string | Normalized username handle (stripped `@` prefix). |
| `region` | string | Market scope (`CHINA_A_SHARE`, `GLOBAL_US`, `HONG_KONG`). |
| `tier` | string | Operational tier (`TIER_0_ELITE_KOL`, `TIER_CONTRARIAN_INDICATOR`, etc.). |
| `fans` | integer | Follower count at the time of track record evaluation. |
| `total_predictions` | integer | Sample count of evaluated calls with verified forward returns. |
| `win_rate_5d` | float | Realized 5-day horizon directional win rate ($k / n$). |
| `bayesian_win_rate_5d` | float | Beta-Binomial posterior mean win rate. |
| `mean_realized_return_5d` | float | Average arithmetic return across all directional calls over 5 days. |
| `payoff_ratio` | float | Ratio of average winning return to average losing return. |
| `is_contrarian` | boolean | Set to `True` if author exhibits systematic inverse alpha. |
| `is_media` | boolean | Set to `True` if author is an institutional or news wire account. |

---

## 4. Python Usage Examples

### 4.1 Basic Lookup and Profile Inspection

```python
from fin_skills.china.kol_registry import KOLCredibilityRegistry, KOLProfile

registry = KOLCredibilityRegistry()

# Query an elite alpha author
elite_prof = registry.get_author_profile("cn_elite_01")
print(f"Author: {elite_prof.author} | Tier: {elite_prof.tier}")
print(f"5D Win Rate: {elite_prof.win_rate_5d:.1%} | Weight: {elite_prof.directional_weight}x")

# Query a contrarian-indicator profile from the fixture (反向明灯)
contrarian_prof = registry.get_author_profile("cn_contrarian_01")
print(f"Author: {contrarian_prof.author} | Fans: {contrarian_prof.fans:,}")
print(f"Win Rate: {contrarian_prof.win_rate_5d:.1%} | Inverted Weight: {contrarian_prof.directional_weight}x")
```

### 4.2 Calibrating a Batch of Social Posts

```python
from fin_skills.china.kol_registry import calibrate_sentiment

# Realistic divergence: Retail mega-influencers shouting long, alpha researchers warning short
posts = [
    {"author": "cn_contrarian_01", "polarity": +0.90, "quality_score": 0.85},  # 425k fans, 40% win rate -> Inverted to -1.15
    {"author": "cn_contrarian_08", "polarity": +0.85, "quality_score": 0.85},        # 150k fans, 40% win rate -> Inverted to -1.08
    {"author": "cn_media_04", "polarity": +0.60, "quality_score": 0.90},        # News aggregator -> Direction 0.0x
    {"author": "cn_elite_01", "polarity": -0.75, "quality_score": 0.95},   # Elite researcher (91.7% win rate) -> Amplified to -2.14
    {"author": "cn_core_01", "polarity": -0.65, "quality_score": 0.90},     # Core alpha researcher (70.8% win rate) -> Amplified to -1.46
]

result = calibrate_sentiment(posts)

print(f"Raw Unweighted Sentiment:  {result.raw_unweighted_polarity:+.2f} (Retail Trap)")
print(f"KOL Calibrated Sentiment:  {result.kol_weighted_polarity:+.2f} (True Alpha Signal)")
print(f"Elite Authors Amplified:   {result.elite_alpha_count}")
print(f"Contrarians Inverted:      {result.contrarian_inverted_count}")
print(f"Media Accounts Filtered:   {result.media_neutralized_count}")
print(f"Summary Rationale:         {result.summary_explanation}")
```

### 4.3 Updating Track Records with Bayesian Shrinkage

```python
from fin_skills.china.kol_registry import update_bayesian_track_record, compute_brier_score

# New author with 7 wins out of 8 calls
post_alpha, post_beta, shrunk_wr = update_bayesian_track_record(
    prior_alpha=5.0,
    prior_beta=5.0,
    wins=7,
    total=8,
)
print(f"Posterior Mean Win Rate: {shrunk_wr:.1%} (Raw was {7/8:.1%})")

# Evaluate Brier calibration score
forecast_probs = [0.85, 0.70, 0.90, 0.40]
realized_outcomes = [1, 1, 0, 0]  # Third call was a loss
brier = compute_brier_score(forecast_probs, realized_outcomes)
print(f"Author Brier Calibration Score: {brier:.4f}")
```

---

## 5. Anti-Patterns and Point-in-Time Causality Safeguards

| Anti-Pattern | Consequence | Safe Practice |
|---|---|---|
| **Weighting by follower count** | Maximizes allocation at FOMO tops; causes catastrophic drawdowns. | Weight strictly by Bayesian shrunk win rate and payoff ratio. |
| **Full-sample look-ahead contamination** | Applying an author's 2026 win rate to a 2021 backtest bar introduces future information. | Calculate author track records using strictly point-in-time expanding windows: only calls evaluated *before* $t$ are known at $t$. |
| **Using media feeds directionally** | Trades backward-looking news headlines; pays high turnover with ~47% win rate. | Force media feeds to `TIER_MEDIA_AGGREGATOR` ($0.0x$ directional weight); use only for event flags and volatility triggers. |
| **Treating contrarian signals as noise** | Ignores strong predictive power of systematic retail failure. | Invert contrarian signals with a calibrated negative weight ($-1.5x$). |
| **Unbounded raw weight summation** | Outlier volume can artificially dominate fundamental and macro signals. | Bound final calibrated sentiment strictly within $[-1.0, +1.0]$ with uncertainty attenuation. |
