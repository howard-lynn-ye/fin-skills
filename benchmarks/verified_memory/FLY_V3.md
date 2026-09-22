# Fruit-fly controller v3: account state and cost-aware actions

This is exploratory development after v2. The rate-based mushroom-body circuit,
projection width, sparsity, learning rate, exploration and trace constants are unchanged.
The new variables are the state representation, action set and reward accounting.

## Contract

The observation contains the existing 20 past-price features, four current risky
weights, cash weight, time since the last nonzero trade, and eight current-price
estimates of transaction fees. Age measures time since rebalance, not tax-lot holding
period. Future fill prices never enter an observation or its cost estimates.

Actions: hold existing shares/cash, equal weight, momentum, reversal, inverse volatility,
cash, halve risky exposure, or increase current exposure by 25% subject to a 100% cap.
From cash, increase allocates 25% equally. HOLD emits no order and never rebalances
drifted weights. Targets from other actions are fixed at the decision observation and
execute at the next observation. All holdings are long-only, unlevered and fractional.

A persistent cash/share account advances chronologically. When a new action executes,
the previous action's reward ends BEFORE the new action's fee. Each reward begins at
NAV immediately before its own fill, so its entry costs and ensuing market changes
are included exactly once. Terminal liquidation belongs to the final interval. The
reward becomes usable only strictly after the endpoint observation. The learner keeps
the original KC activity, action compartment and prediction for delayed attribution.

The primary reward is net account return over this interval. A separate declared arm
uses net account return minus the same interval's return from leaving the pre-action
shares/cash unchanged. This hold counterfactual uses only prices that have become
available by the reward endpoint. It is computed in the simulator, never supplied as
future information at decision time. Both paths include terminal liquidation when due.

The evaluation account resets to cash at its declared start, while learned weights
remain warm-started. Any outstanding warm-up order and unfinished warm-up reward are
explicitly cancelled and logged at the reset. Mature warm-up feedback may still update
the learner. This reset allows exact reconciliation with the independent evaluator.

## Checks and comparisons

Actual fin-skills feature-causality and reward-lineage guards run. The reward adapter
also reconstructs each interval endpoint from saved cash and shares. After actions are
frozen, every daily NAV and turnover value must agree with the separately implemented
final scorer. Missing or inconsistent records fail the cell. Cost estimates are not
broker quotes, and passing these checks is not a certificate of executable FX trading.

Arms: full fly v3; no HOLD action; market features only; no cost-estimate features;
hold-relative reward; parameter-matched dense circuit; simpler linear circuit; and
frozen fly. Account-feature ablations retain identical input dimensions/projections.
The no-HOLD arm changes the available action set, and the linear arm has fewer weights.
The v2 fly is rerun as a historical reference, not a single-variable causal comparison:
state, action count and reward definition differ. Five seeds are initialization repeats,
not independent market samples. Fixed expert/cash baselines accompany every dataset.

Zero/5/20 bps scores replay frozen target weights and HOLD events. They do not retrain
the policies or regenerate state-dependent actions at the alternative fee rates.
The primary training/evaluation fee is 5 bps per traded notional. Net performance,
turnover, exposure, HOLD rate and performance against cash must be read together;
reducing trading or staying in cash is not itself evidence of predictive alpha.

The prior synthetic market seeds and ECB 2016-2025 proxy are development data. No unseen
holdout claim is made. ECB excludes carry, funding and executable bid/ask fills. Inputs
and model settings are frozen before scoring, failures remain in the denominator, and
all Beacon files remain under a fresh RADFM directory.
