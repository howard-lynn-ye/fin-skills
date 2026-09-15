"""Guard: Board Lot Feasibility & Granularity Distortion for A-Share Portfolios."""
from __future__ import annotations

from typing import Any, Mapping

from fin_skills.api.base import Guard, Outcome, register
from fin_skills.china.board_lot_guard import evaluate_board_lot_feasibility


@register
class BoardLotFeasibilityGuard(Guard):
    """Audits tracking error caused by A-share 100-share minimum order constraints.

    Inputs
        capital                   : total investment capital in RMB
        target_weights            : mapping of ticker to desired allocation weight
        prices                    : optional mapping of current market prices
        max_acceptable_distortion : allowable cumulative weight deviation (default 0.15 = 15%)

    Fails when capital is too small to execute the allocation within tolerance (e.g. 1 lot
    of 10-year treasury ETF alone accounts for >50% of the account).
    """

    name = "board_lot_feasibility"
    skill = "china-trading-stack"
    summary = "Audits tracking error caused by 100-share minimum lot constraints on account capital."
    wraps = ("fin_skills.china.board_lot_guard.evaluate_board_lot_feasibility",)
    required = ("capital", "target_weights")
    optional = ("prices", "max_acceptable_distortion")

    def check(
        self,
        capital: float,
        target_weights: Mapping[str, float],
        prices: Mapping[str, float] | None = None,
        max_acceptable_distortion: float = 0.15,
    ) -> Outcome:
        out = Outcome()
        if not isinstance(capital, (int, float)) or capital <= 0:
            raise TypeError(f"capital must be a positive float, got {capital!r}")
        if not isinstance(target_weights, Mapping) or not target_weights:
            raise TypeError("target_weights must be a non-empty mapping")

        result = evaluate_board_lot_feasibility(
            capital=float(capital),
            target_weights=target_weights,
            prices=prices,
            max_acceptable_distortion=float(max_acceptable_distortion),
        )

        out.note(
            capital=result.capital,
            granularity_distortion=result.granularity_distortion,
            feasibility_status=result.feasibility_status,
            is_feasible=result.is_feasible,
            min_recommended_capital=result.min_recommended_capital,
            residual_cash=result.residual_cash,
            discrete_lots=result.discrete_lots,
        )

        if not result.is_feasible:
            out.error(result.message, where="portfolio capital sizing")
        elif result.feasibility_status == "MODERATE_DISTORTION":
            out.warning(result.message, where="portfolio capital sizing")
        else:
            out.info(result.message, where="portfolio capital sizing")

        return out
