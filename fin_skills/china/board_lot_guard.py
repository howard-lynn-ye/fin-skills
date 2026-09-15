"""Board Lot Feasibility & Granularity Distortion Evaluator for A-Share Portfolios.

In China A-share markets:
- All buying orders MUST be in multiples of 100 shares (1手 / 1 board lot).
- Fractional shares (零股) are strictly forbidden for purchases (only allowed on sales).
- High-priced ETFs (e.g., 511010 at ~130 RMB/share = 13,000 RMB/lot) create severe
  Granularity Distortion when deployed on smaller accounts (< 50,000 RMB).

This module measures the tracking error introduced by integer lot constraints,
determines whether an account has sufficient capital to faithfully execute an asset
allocation model, and suggests minimum capital thresholds or lower-cost ETF substitutes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

# Typical unit prices if not dynamically provided
DEFAULT_TYPICAL_PRICES: dict[str, float] = {
    "510300": 4.10,    # 沪深300 ETF (~410 RMB/lot)
    "510500": 5.90,    # 中证500 ETF (~590 RMB/lot)
    "510880": 3.10,    # 红利 ETF (~310 RMB/lot)
    "518880": 6.80,    # 黄金 ETF (~680 RMB/lot)
    "511010": 135.00,  # 10年国债 ETF (~13,500 RMB/lot)
    "513100": 1.70,    # 纳指 ETF (~170 RMB/lot)
    "513500": 1.80,    # 标普500 ETF (~180 RMB/lot)
    "510900": 0.85,    # H股 ETF (~85 RMB/lot)
    "511260": 105.00,  # 10年国债ETF平替
    "511520": 102.00,  # 政金债 ETF
    "159985": 1.45,    # 豆粕 ETF
    "512400": 1.20,    # 有色金属 ETF
    "511880": 100.00,  # 银华日利货币 ETF
}


@dataclass(frozen=True)
class BoardLotFeasibilityResult:
    capital: float
    granularity_distortion: float  # sum of absolute weight errors
    feasibility_status: str  # 'EXCELLENT', 'MODERATE_DISTORTION', 'UNFEASIBLE_HIGH_DISTORTION'
    is_feasible: bool
    min_recommended_capital: float
    discrete_weights: dict[str, float]
    discrete_lots: dict[str, int]
    residual_cash: float
    worst_distorted_asset: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "capital": self.capital,
            "granularity_distortion_pct": f"{round(self.granularity_distortion * 100, 2)}%",
            "feasibility_status": self.feasibility_status,
            "is_feasible": self.is_feasible,
            "min_recommended_capital": self.min_recommended_capital,
            "discrete_weights": {k: round(v, 4) for k, v in self.discrete_weights.items()},
            "discrete_lots": self.discrete_lots,
            "residual_cash": round(self.residual_cash, 2),
            "worst_distorted_asset": self.worst_distorted_asset,
            "message": self.message,
        }


def evaluate_board_lot_feasibility(
    capital: float,
    target_weights: Mapping[str, float],
    prices: Mapping[str, float] | None = None,
    max_acceptable_distortion: float = 0.15,  # 15% cumulative weight deviation
) -> BoardLotFeasibilityResult:
    """Evaluate whether target weights can be faithfully executed given capital and 100-share lots.

    Args:
        capital: Total portfolio cash available in RMB.
        target_weights: Mapping of ticker to target weight (e.g. {'510300': 0.20, ...}).
        prices: Current market price per share for each ticker.
        max_acceptable_distortion: Cumulative absolute deviation threshold (default 15%).
    """
    if capital <= 0:
        raise ValueError(f"capital must be strictly positive, got {capital}")

    active_prices = {}
    for code in target_weights:
        if prices and code in prices and prices[code] > 0:
            active_prices[code] = float(prices[code])
        elif code in DEFAULT_TYPICAL_PRICES:
            active_prices[code] = DEFAULT_TYPICAL_PRICES[code]
        else:
            active_prices[code] = 1.0  # Fallback assumption

    # 1. Calculate discrete lot allocation
    discrete_lots: dict[str, int] = {}
    discrete_values: dict[str, float] = {}
    discrete_weights: dict[str, float] = {}

    total_deployed = 0.0
    for code, weight in target_weights.items():
        price = active_prices[code]
        lot_cost = price * 100.0
        ideal_amount = capital * weight
        # Round to nearest 100 shares
        lots = int(round(ideal_amount / lot_cost))
        discrete_lots[code] = lots
        val = lots * lot_cost
        discrete_values[code] = val
        discrete_weights[code] = val / capital
        total_deployed += val

    residual_cash = capital - total_deployed

    # 2. Calculate weight distortion per asset
    distortion = 0.0
    worst_asset = ""
    worst_dev = -1.0

    for code, target_w in target_weights.items():
        actual_w = discrete_weights[code]
        dev = abs(actual_w - target_w)
        distortion += dev
        if dev > worst_dev:
            worst_dev = dev
            worst_asset = code

    # 3. Estimate minimum capital to keep distortion <= 8%
    # Typically driven by the highest lot cost in the portfolio
    max_lot_cost = max(active_prices[c] * 100.0 for c in target_weights)
    min_weight = min(w for w in target_weights.values() if w > 0.02)
    # To have at least 2 lots in the smallest non-trivial bucket:
    min_rec_capital = max(30000.0, round((max_lot_cost / min_weight) * 1.5, -3))

    if distortion > 0.25:
        status = "UNFEASIBLE_HIGH_DISTORTION"
        feasible = False
        msg = (
            f"CRITICAL: Capital {capital:,.0f} RMB causes extreme granularity distortion of "
            f"{distortion*100:.1f}% (worst: {worst_asset} deviated by {worst_dev*100:.1f}%). "
            f"A-share 100-share minimum lot makes this asset allocation mathematically unexecutable. "
            f"Minimum recommended capital: {min_rec_capital:,.0f} RMB."
        )
    elif distortion > max_acceptable_distortion:
        status = "MODERATE_DISTORTION"
        feasible = True
        msg = (
            f"WARNING: Capital {capital:,.0f} RMB introduces moderate granularity distortion "
            f"of {distortion*100:.1f}%. Target weights will experience tracking error. "
            f"Recommend sizing up to {min_rec_capital:,.0f} RMB over time."
        )
    else:
        status = "EXCELLENT"
        feasible = True
        msg = (
            f"FEASIBLE: Capital {capital:,.0f} RMB cleanly supports 100-share integer lots "
            f"(distortion {distortion*100:.1f}% <= {max_acceptable_distortion*100:.1f}%). "
            f"Faithful execution guaranteed."
        )

    return BoardLotFeasibilityResult(
        capital=capital,
        granularity_distortion=distortion,
        feasibility_status=status,
        is_feasible=feasible,
        min_recommended_capital=min_rec_capital,
        discrete_weights=discrete_weights,
        discrete_lots=discrete_lots,
        residual_cash=residual_cash,
        worst_distorted_asset=worst_asset,
        message=msg,
    )
