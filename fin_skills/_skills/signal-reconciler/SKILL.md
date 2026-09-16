---
name: signal-reconciler
description: >-
  [fin-china] Resolve conflicting bullish and bearish signals across macro, fundamental, technical, and social sentiment sources using entropy-weighted evidence combination.
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-16"
---

# Signal Reconciler

In quantitative trading across China A-shares and cross-border portfolios, quantitative models ingest diverse information modalities: **macro policy, northbound smart money, fundamental valuation, news wires, and social sentiment**.

In production, these channels frequently and violently contradict each other:
- Central bank/regulators issue risk warnings while retail social forums scream "Buy the breakout!".
- Northbound institutional capital quietly accumulates blue chips at a 10-year valuation trough while retail investors dump in panic.
- US tech equities rally on StockTwits while domestic Chinese QDII ETFs trade at an unsustainable +3% market premium.

A naive linear weighted average across conflicting signals is **fatal**: averaging institutional selling ($-0.8$) with retail FOMO ($+1.0$) yields $+0.1$ (buy), walking directly into a **Distribution Trap (主力派发、散户接盘)**.

The `signal-reconciler` skill implements a **5-Tier Hierarchical Conflict Matrix**, **Credibility-Weighted Belief Entropy**, and **Uncertainty Attenuation Damping** to resolve contradictions into rigorous, actionable portfolio decisions.

---

## 1. The Multi-Source Conflict Problem

Financial signals originate from information sources with radically different reliability profiles, latency characteristics, and behavioral biases:

```
+------------------------------------------------------------------------------------+
|                             HIERARCHY OF EVIDENCE                                  |
|                                                                                    |
|  [Tier 1] REGULATORY & POLICY VETO   -> Hard circuit breaker (Level 1 Policy/QDII) |
|  [Tier 2] SMART MONEY FLOWS          -> Northbound Capital, Margin, Block Orders   |
|  [Tier 3] FUNDAMENTAL VALUATION      -> PE/PB percentiles, Dividend Yields, Filings|
|  [Tier 4] MAINSTREAM TELEGRAPH NEWS  -> 7x24 Macro headlines, Flash catalysts      |
|  [Tier 5] RETAIL SOCIAL SENTIMENT    -> Xueqiu, Guba, StockTwits contrarian flow   |
+------------------------------------------------------------------------------------+
```

### Why Naive Averaging Fails

Consider two archetypal market inflection points:

1. **The Distribution Trap**:
   - Smart Money: $-0.75$ (Institutions aggressively dumping $-6.2$B RMB).
   - Retail Social: $+0.85$ (Retail traders euphoric on breakout).
   - *Naive Linear Average*: $+0.05$ (Stay long or buy) $\implies$ **Heavy losses as the rally collapses**.
   - *Hierarchical Reconciliation*: $-0.85$ (Follow institutional skin-in-the-game; penalize retail FOMO) $\implies$ **Defensive exit**.

2. **The Contrarian Bottom (黄金坑)**:
   - Fundamental Valuation: $+0.80$ (Dividend yield $>4.5\%$, 10-year valuation bottom).
   - Smart Money: $+0.60$ (Northbound capital quietly buying $+4.5$B RMB).
   - Retail Social: $-0.85$ (Retail capitulation panic selling).
   - *Naive Linear Average*: $+0.18$ (Weak neutral/hesitant).
   - *Hierarchical Reconciliation*: $+0.90$ (Institutional accumulation confirmed by contrarian exhaustion) $\implies$ **High-conviction buy**.

---

## 2. Credibility-Weighted Belief Entropy Formulation

To measure the severity of contradiction across modalities, the reconciler computes cross-channel dispersion and Shannon belief entropy.

### 2.1 Cross-Channel Disagreement Index

Let $S_{	ext{active}} = \{s_i \mid |s_i| \ge 0.05\}$ be the set of non-trivial channel signals. The disagreement index $\sigma_{	ext{dispersion}}$ is the sample standard deviation of active signals:

$$ar{s} = rac{1}{|S_{	ext{active}}|} \sum_{i=1}^M s_i, \quad \sigma_{	ext{dispersion}} = \sqrt{rac{1}{|S_{	ext{active}}|} \sum_{i=1}^M (s_i - ar{s})^2}$$

Where $\sigma_{	ext{dispersion}} \in [0, 1]$:
- $\sigma \le 0.20$: Unanimous cross-channel consensus.
- $0.20 < \sigma < 0.45$: Moderate informational divergence.
- $\sigma \ge 0.45$: Severe polarity contradiction (e.g., smart money selling vs. retail buying).

### 2.2 Shannon Belief Entropy

For normalized evidence probabilities $p_i = rac{|s_i|}{\sum_j |s_j|}$, the Shannon belief entropy is:

$$	ext{Entropy}(p) = -\sum_{i=1}^M p_i \log_2(p_i)$$

Higher entropy indicates that evidence mass is evenly split across opposing camps rather than concentrated in a clear thesis.

### 2.3 Uncertainty Attenuation Multiplier

When signals disagree without triggering a structural archetype, the system applies **exponential dispersion damping** to shrink position sizing:

$$	ext{Confidence Multiplier} = \max\left(0.25, \, \min\left(1.0, \, \exp(-\lambda \cdot \sigma_{	ext{dispersion}}^2)ight)ight)$$

Where $\lambda = 1.2$ is the dispersion damping parameter. When channels disagree sharply ($\sigma = 0.8$), the confidence multiplier drops to $46\%$, preventing excessive leverage during noisy market regimes.

---

## 3. The 5 Conflict Resolution Rules

The engine evaluates signals through a prioritized cascade of 5 structural archetypes:

```
                  +-----------------------------------+
                  |      Incoming Channel Signals     |
                  +-----------------+-----------------+
                                    |
                 [Rule 1: Regulatory / QDII Veto?]
                       /                                      YES                  NO
                     /                            [Defensive Circuit Breaker]    [Rule 2: Cross-Border Disconnect?]
                                             /                                                            YES                  NO
                                           /                                                  [Penalize Premium Risk]     [Rule 3: Distribution Trap?]
                                                               /                                                                              YES                  NO
                                                             /                                                                   [Follow Smart Money Dump]  [Rule 4: Contrarian Bottom?]
                                                                               /                                                                                              YES                  NO
                                                                             /                                                                                   [High-Conviction Buy]    [Rule 5: Synthesis & Damping]
```

### Rule 1: Macro & Regulatory Hard Veto (一票否决风控)
- **Condition**: Regulatory policy circuit breaker active (`veto_flag=True` or score $\le -0.8$), or domestic ETF premium $\ge 2.5\%$.
- **Action**: Overrides all other channels immediately. Sets final score to $-1.0$ and recommended active tilt to $-	ext{max\_tilt}$ ($-1.5\%$).

### Rule 2: Cross-Border QDII Premium Disconnect (跨境溢价脱节)
- **Condition**: Overseas social sentiment/momentum is bullish ($s_{	ext{social}} > 0.3$), but domestic secondary market ETF trades at a premium $\ge 1.5\%$ over IOPV.
- **Action**: Structural instrument pricing risk strictly overrides underlying asset momentum. Forbids chasing highs; enforces a $-0.6$ score and negative tilt.

### Rule 3: High-Conviction Contrarian / Distribution Trap (诱多派发背离)
- **Condition**: Smart money flow is strongly negative ($s_{	ext{flow}} \le -0.35$) while retail social sentiment is euphoric ($s_{	ext{social}} \ge +0.45$).
- **Action**: Follows institutional capital and applies contrarian penalty to retail FOMO:
  $$	ext{Score} = \max(-1.0, \, s_{	ext{flow}} - 0.35 \cdot s_{	ext{social}})$$
  Enforces maximum defensive tilt ($-	ext{max\_tilt}$).

### Rule 4: Contrarian Bottom / Golden Pit (黄金坑底部背离)
- **Condition**: Institutional anchor (weighted Smart Money + Valuation) $\ge +0.25$, while retail sentiment is in deep capitulation panic ($s_{	ext{social}} \le -0.45$).
- **Action**: Identifies institutional accumulation during retail washouts:
  $$	ext{Score} = \min(1.0, \, s_{	ext{anchor}} - 0.25 \cdot s_{	ext{social}})$$
  Enforces maximum positive tilt ($+	ext{max\_tilt}$).

### Rule 5: Dominant Consensus & Deadband Resolution (共振与死区阻尼)
- **Condition**: No structural divergence active.
- **Action**: Computes composite weighted score with retail contrarian sign inversion for extreme values ($|s_{	ext{social}}| > 0.6$). Applies uncertainty attenuation factor:
  $$	ext{Score}_{	ext{final}} = 	ext{Score}_{	ext{composite}} 	imes 	ext{Confidence Multiplier}$$
- **Deadband Filter**: If $|	ext{Score}_{	ext{final}}| < 0.15$, recommended tilt is $0.0$, eliminating noisy portfolio churn.

---

## 4. Python Usage Examples

### 4.1 Resolving a Distribution Trap

```python
from fin_skills.china.signal_reconciler import ChannelSignal, SignalReconciler, reconcile_views

# Define contradictory signals
signals = [
    ChannelSignal(channel="SMART_MONEY_FLOW", score=-0.75, evidence="Northbound outflow -6.2B RMB"),
    ChannelSignal(channel="FUNDAMENTAL_VALUATION", score=-0.20, evidence="PE at 75th percentile"),
    ChannelSignal(channel="RETAIL_SOCIAL_CN", score=+0.85, evidence="Retail screaming 'To the moon!'"),
]

# Run resolution
result = reconcile_views("510300", signals)

print(f"Conflict Archetype:     {result.conflict_type}")
print(f"Dominant Channel:       {result.dominant_channel}")
print(f"Disagreement Index:     {result.disagreement_index:.2f}")
print(f"Final Resolved Score:   {result.final_score:+.2f}")
print(f"Recommended Tilt:       {result.recommended_tilt * 100:+.2f}%")
print(f"Rationale:              {result.resolution_rationale}")
```

### 4.2 Resolving a Golden Pit Contrarian Bottom

```python
# Smart money and low valuation buying during retail panic
signals = [
    ChannelSignal(channel="SMART_MONEY_FLOW", score=+0.60, evidence="Northbound inflow +4.5B RMB"),
    ChannelSignal(channel="FUNDAMENTAL_VALUATION", score=+0.85, evidence="Dividend yield 4.8%, 10y bottom"),
    ChannelSignal(channel="RETAIL_SOCIAL_CN", score=-0.80, evidence="Retail capitulation panic selling"),
]

result = reconcile_views("510880", signals)
assert result.conflict_type == "CONFLICT_CONTRARIAN_BOTTOM"
assert result.final_score > 0.70
assert result.recommended_tilt == +0.015
print(f"Contrarian Tilt: {result.recommended_tilt * 100:+.2f}% (Accumulate)")
```

### 4.3 Handling Cross-Border QDII Premium Disconnect

```python
# Overseas US tech is roaring, but domestic ETF trades at toxic 2.8% premium
signals = [
    ChannelSignal(channel="RETAIL_SOCIAL_US", score=+0.90, evidence="StockTwits tech extreme bullish"),
    ChannelSignal(channel="SMART_MONEY_FLOW", score=+0.40, evidence="Moderate inflows"),
]

result = reconcile_views("513100", signals, qdii_premium_pct=2.85)
assert result.conflict_type == "CONFLICT_VETO_OVERRIDE"
assert result.final_score == -1.0
print(f"Veto Triggered: {result.resolution_rationale}")
```

---

## 5. Execution Gate Rules for Portfolios

Before any reconciled tilt is executed in production, it must satisfy four mandatory execution gates:

| Gate Name | Parameter / Rule | Enforcement Action |
|---|---|---|
| **T+1 Settlement Gate** | Chinese equities bought on day $t$ cannot be sold until $t+1$. | Reject same-day sell order if share lot was acquired in current session. |
| **Price Limit Gate (涨跌停)** | Main Board $\pm 10\%$, ChiNext/STAR $\pm 20\%$, BSE $\pm 30\%$, ST $\pm 5\%$. | Forbid buy orders if price is locked at limit-up; forbid sell orders if locked at limit-down. |
| **QDII Premium Gate** | ETF secondary market premium $>1.5\%$ over IOPV. | Block buy execution; substitute with OTC primary creation or overseas direct share allocation. |
| **Portfolio Tilt Deadband** | $|	ext{Tilt}| \le 1.5\%$ maximum active deviation from core index. | Hard-clamp recommended active tilt within $[-0.015, +0.015]$. |
