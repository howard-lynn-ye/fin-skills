"""Guard: QDII Secondary Market Premium & IOPV Circuit Breaker (china-trading-stack)."""
from __future__ import annotations

from typing import Any

from fin_skills.api.base import Guard, Outcome, register
from fin_skills.china.qdii_premium_guard import (DEFAULT_HARD_CIRCUIT_PREMIUM,
                                                DEFAULT_MAX_ALLOWED_PREMIUM,
                                                evaluate_qdii_order)


@register
class QDIIPremiumGuard(Guard):
    """Enforces secondary market premium limits on cross-border / QDII ETFs.

    Inputs
        code                   : ETF ticker (e.g. '513100', '513500')
        price                  : secondary market trading price
        iopv                   : indicative optimized portfolio value (IOPV real-time reference NAV)
        premium_rate           : precomputed premium rate ((price - iopv) / iopv), optional
        max_allowed_premium    : soft tolerance threshold (default 0.015 = 1.5%)
        hard_circuit_premium   : hard circuit breaker threshold (default 0.030 = 3.0%)
        fallback_safety_asset  : ticker to receive redirected capital (default '518880' Gold ETF)

    Fails (error) when secondary market premium exceeds hard_circuit_premium (> 3.0%),
    preventing unhedged bubble entry.
    Warns when premium is in the [1.5%, 3.0%] derating zone.
    Passes (info) when trading normally (<= 1.5%) or at a favorable discount.
    """

    name = "qdii_premium"
    skill = "china-trading-stack"
    summary = "Checks secondary market premium over IOPV for cross-border QDII ETFs and blocks bubble entries."
    wraps = (
        "fin_skills.china.qdii_premium_guard.evaluate_qdii_order",
        "fin_skills.china.qdii_premium_guard.calculate_premium_rate",
    )
    required = ("code", "price")
    optional = ("iopv", "premium_rate", "max_allowed_premium", "hard_circuit_premium", "fallback_safety_asset")

    def check(
        self,
        code: str,
        price: float,
        iopv: float | None = None,
        premium_rate: float | None = None,
        max_allowed_premium: float = DEFAULT_MAX_ALLOWED_PREMIUM,
        hard_circuit_premium: float = DEFAULT_HARD_CIRCUIT_PREMIUM,
        fallback_safety_asset: str = "518880",
    ) -> Outcome:
        out = Outcome()
        if not isinstance(code, str):
            raise TypeError("code must be a string")
        if not isinstance(price, (int, float)) or price <= 0:
            raise TypeError(f"price must be a positive float, got {price!r}")

        result = evaluate_qdii_order(
            code=code,
            price=float(price),
            iopv=float(iopv) if iopv is not None else None,
            premium_rate=float(premium_rate) if premium_rate is not None else None,
            max_allowed_premium=float(max_allowed_premium),
            hard_circuit_premium=float(hard_circuit_premium),
            fallback_safety_asset=fallback_safety_asset,
        )

        out.note(
            code=result.code,
            name=result.name,
            price=result.price,
            iopv=result.iopv,
            premium_rate=result.premium_rate,
            status=result.status,
            allowed_weight_factor=result.allowed_weight_factor,
            redirect_target=result.redirect_target,
        )

        if result.status == "HARD_CIRCUIT":
            out.error(result.message, where=f"{result.code} {result.name}")
        elif result.status == "DERATE_50":
            out.warning(result.message, where=f"{result.code} {result.name}")
        else:
            out.info(result.message, where=f"{result.code} {result.name}")

        return out
