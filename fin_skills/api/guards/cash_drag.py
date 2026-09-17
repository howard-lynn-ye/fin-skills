"""Guard: Cash Drag and Overnight Sweep Optimizer for A-share portfolios."""
from __future__ import annotations

from typing import Any

from fin_skills.api.base import Guard, Outcome, register
from fin_skills.china.cash_yield_optimizer import (DEFAULT_BENCHMARK_CASH_YIELD,
                                                  DEFAULT_DEMAND_RATE,
                                                  plan_cash_placement)


@register
class CashDragGuard(Guard):
    """Detects unmanaged idle cash drag and prescribes zero-risk overnight sweep actions.

    Inputs
        idle_cash                 : uninvested RMB cash balance in the account
        total_capital             : total portfolio net asset value (NAV)
        trade_date                : trade date (to check for Thursday 3-day repo bonus)
        demand_rate               : broker demand deposit rate (default 0.30% = 0.0030)
        expected_repo_rate        : achievable GC001 / Money Market yield (default 2.10% = 0.0210)
        max_acceptable_idle_ratio : tolerance threshold for idle cash ratio (default 0.05 = 5%)

    Warns when idle cash exceeds 5% of portfolio and >= 1,000 RMB without an active sweep,
    informing the operator of exact annual drag in RMB and how to deploy it before 15:30.
    """

    name = "cash_drag"
    skill = "china-trading-stack"
    summary = "Audits unmanaged idle cash in A-share accounts and prescribes GC001 reverse repo actions."
    wraps = (
        "fin_skills.china.cash_yield_optimizer.plan_cash_placement",
        "fin_skills.china.cash_yield_optimizer.calculate_cash_drag",
        "fin_skills.china.cash_yield_optimizer.get_repo_interest_days",
    )
    required = ("idle_cash", "total_capital")
    optional = ("trade_date", "demand_rate", "expected_repo_rate", "max_acceptable_idle_ratio")

    def check(
        self,
        idle_cash: float,
        total_capital: float,
        trade_date: Any = None,
        demand_rate: float = DEFAULT_DEMAND_RATE,
        expected_repo_rate: float = DEFAULT_BENCHMARK_CASH_YIELD,
        max_acceptable_idle_ratio: float = 0.05,
    ) -> Outcome:
        out = Outcome()
        if not isinstance(idle_cash, (int, float)) or idle_cash < 0:
            raise TypeError(f"idle_cash must be non-negative float, got {idle_cash!r}")
        if not isinstance(total_capital, (int, float)) or total_capital <= 0:
            raise TypeError(f"total_capital must be positive float, got {total_capital!r}")

        cash_ratio = idle_cash / total_capital
        plan = plan_cash_placement(
            idle_cash=float(idle_cash),
            current_date=trade_date,
            demand_rate=float(demand_rate),
            expected_repo_rate=float(expected_repo_rate),
        )

        out.note(
            idle_cash=plan.idle_cash,
            total_capital=total_capital,
            cash_ratio=cash_ratio,
            annual_drag_rmb=plan.annual_drag_rmb,
            interest_days=plan.interest_days,
            recommended_vehicle=plan.recommended_vehicle,
            order_action=plan.order_action,
        )

        if cash_ratio > 0.20 and idle_cash >= 1000.0:
            out.error(
                f"Severe unmanaged cash drag: {cash_ratio*100:.1f}% ({idle_cash:,.0f} RMB) sitting in demand deposit. "
                f"Annual drag: -{plan.annual_drag_rmb:.1f} RMB. Deploy via {plan.order_action}.",
                where="portfolio cash management",
            )
        elif cash_ratio > max_acceptable_idle_ratio and idle_cash >= 1000.0:
            out.warning(
                f"Idle cash ratio of {cash_ratio*100:.1f}% ({idle_cash:,.0f} RMB) exceeds tolerance "
                f"{max_acceptable_idle_ratio*100:.1f}%. Annual drag: -{plan.annual_drag_rmb:.1f} RMB. "
                f"Action before {plan.urgency_deadline}: {plan.order_action}. ({plan.notes})",
                where="portfolio cash management",
            )
        else:
            out.info(
                f"Idle cash {idle_cash:,.0f} RMB ({cash_ratio*100:.1f}%) within normal operations. {plan.notes}",
                where="portfolio cash management",
            )

        return out
