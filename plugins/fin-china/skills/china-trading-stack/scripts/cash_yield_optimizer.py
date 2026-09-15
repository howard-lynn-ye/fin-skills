"""Cash Drag Evaluator and Overnight Yield Optimizer for China A-Share Portfolios.

In China A-share accounts:
1. Demand Deposit (活期存款): Default rate paid by securities brokers is only ~0.20% - 0.35% p.a.
2. National Debt Reverse Repo (国债逆回购 GC001 / 204001, R-001 / 131810):
   - Min order: 1,000 RMB (Shanghai / Shenzhen unified).
   - Zero default risk: Backed by sovereign central clearing (CSDC / 中国结算).
   - Settlement: T+0 usable for buying stocks/ETFs next morning at 09:15, T+1 withdrawable.
   - Yield: 1.8% ~ 2.5% normal; surges to 4%~10%+ during quarter-ends and holiday eves.
   - The "Thursday Multiplier": Placing 1-day GC001 on Thursday earns 3 days of interest
     (Fri/Sat/Sun) while capital is usable for trading on Friday morning!
3. On-Exchange Money Market ETFs (场内货币基金):
   - 银华日利 (511880), 华宝添益 (511990)
   - T+0 trading, 0 stamp duty, 0 commission at most brokers.
   - Min unit: 100 shares (~10,000 RMB for 511880 or ~10,000 RMB for 511990).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import pandas as pd

DEFAULT_DEMAND_RATE = 0.0030     # 0.30% broker demand deposit rate
DEFAULT_BENCHMARK_CASH_YIELD = 0.0210 # 2.10% typical 1-day repo / money market rate
GC001_MIN_LOT_RMB = 1000.0       # 1,000 RMB minimum for GC001 reverse repo


@dataclass(frozen=True)
class CashYieldRecommendation:
    idle_cash: float
    demand_rate: float
    optimized_rate: float
    annual_drag_rmb: float
    interest_days: int
    recommended_vehicle: str
    order_action: str
    urgency_deadline: str
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "idle_cash": round(self.idle_cash, 2),
            "demand_rate_pct": f"{round(self.demand_rate * 100, 2)}%",
            "optimized_rate_pct": f"{round(self.optimized_rate * 100, 2)}%",
            "annual_drag_rmb": round(self.annual_drag_rmb, 2),
            "interest_days": self.interest_days,
            "recommended_vehicle": self.recommended_vehicle,
            "order_action": self.order_action,
            "urgency_deadline": self.urgency_deadline,
            "notes": self.notes,
        }


def get_repo_interest_days(d: date | datetime | str) -> int:
    """Calculate the number of interest-accruing days for 1-day GC001 executed on date `d`.

    Standard rules:
    - Monday -> Tuesday: 1 day
    - Tuesday -> Wednesday: 1 day
    - Wednesday -> Thursday: 1 day
    - Thursday -> Friday: 3 DAYS (Friday, Saturday, Sunday interest; cash usable Friday)
    - Friday -> Monday: 1 day (Friday interest; Saturday/Sunday NOT counted because
      settlement pushes to Monday)
    """
    if isinstance(d, str):
        dt = pd.to_datetime(d).date()
    elif isinstance(d, datetime):
        dt = d.date()
    else:
        dt = d

    weekday = dt.weekday()  # Monday is 0, Sunday is 6
    if weekday == 3:  # Thursday
        return 3
    return 1


def calculate_cash_drag(
    cash_amount: float,
    demand_rate: float = DEFAULT_DEMAND_RATE,
    benchmark_yield: float = DEFAULT_BENCHMARK_CASH_YIELD,
    holding_days: int = 365,
) -> float:
    """Calculate the monetary loss incurred by leaving cash unmanaged in demand deposit."""
    if cash_amount <= 0:
        return 0.0
    spread = max(0.0, benchmark_yield - demand_rate)
    return cash_amount * spread * (holding_days / 365.0)


def plan_cash_placement(
    idle_cash: float,
    current_date: date | datetime | str | None = None,
    current_time_str: str = "14:45",
    demand_rate: float = DEFAULT_DEMAND_RATE,
    expected_repo_rate: float = DEFAULT_BENCHMARK_CASH_YIELD,
) -> CashYieldRecommendation:
    """Produce actionable cash placement instructions for daily closing operations.

    Args:
        idle_cash: Uninvested cash balance in RMB.
        current_date: Current trade date (defaults to today).
        current_time_str: Current local time (HH:MM). GC001 closes at 15:30.
        demand_rate: Broker demand deposit interest rate.
        expected_repo_rate: Annualized rate achievable via GC001 or Money Market ETF.
    """
    dt = date.today() if current_date is None else current_date
    interest_days = get_repo_interest_days(dt)
    annual_drag = calculate_cash_drag(idle_cash, demand_rate, expected_repo_rate)

    if idle_cash < 100.0:
        return CashYieldRecommendation(
            idle_cash=idle_cash,
            demand_rate=demand_rate,
            optimized_rate=demand_rate,
            annual_drag_rmb=0.0,
            interest_days=1,
            recommended_vehicle="NONE",
            order_action="KEEP_IN_ACCOUNT",
            urgency_deadline="N/A",
            notes="Residual balance too small to deploy; maintain in account.",
        )

    if idle_cash < GC001_MIN_LOT_RMB:
        # Between 100 and 1000 RMB: cannot buy GC001, but can buy Money Market ETF (511880)
        return CashYieldRecommendation(
            idle_cash=idle_cash,
            demand_rate=demand_rate,
            optimized_rate=expected_repo_rate,
            annual_drag_rmb=annual_drag,
            interest_days=1,
            recommended_vehicle="511880 (Yinhua Rili MM ETF)",
            order_action=f"Buy 511880 with residual cash {idle_cash:.2f} RMB before 15:00",
            urgency_deadline="15:00",
            notes="Cash below 1,000 RMB GC001 threshold; deploy into on-exchange money market ETF.",
        )

    # 1,000 RMB or more: prime candidate for GC001 reverse repo
    lots = int(idle_cash // GC001_MIN_LOT_RMB)
    usable_amount = lots * GC001_MIN_LOT_RMB
    residual = idle_cash - usable_amount

    thursday_bonus = " (Thursday 3-Day Interest Multiplier Active!)" if interest_days == 3 else ""

    return CashYieldRecommendation(
        idle_cash=idle_cash,
        demand_rate=demand_rate,
        optimized_rate=expected_repo_rate,
        annual_drag_rmb=annual_drag,
        interest_days=interest_days,
        recommended_vehicle="204001 (GC001) / 131810 (R-001)",
        order_action=f"Execute 1-day reverse repo for {usable_amount:,.0f} RMB ({lots} lots)",
        urgency_deadline="15:30 (A-share repo trading cutoff)",
        notes=(
            f"Lending {usable_amount:,.0f} RMB overnight captures {interest_days} days of ~{expected_repo_rate*100:.2f}% "
            f"interest{thursday_bonus}. Cash is 100% usable for stock/ETF trading tomorrow morning at 09:15."
        ),
    )
