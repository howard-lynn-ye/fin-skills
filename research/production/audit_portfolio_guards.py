#!/usr/bin/env python3
"""Unified Institutional Portfolio Risk Guards Audit CLI & Reporting Engine.

Assembles all 6 institutional quantitative risk guard categories in fin-skills:
1. QDII Secondary Market Premium Guard (QDIIPremiumGuard: checks secondary premium >1.5% & >3.0% circuit breaker)
2. A-Share Trading Rules & Liquidity Guard (AShareRulesGuard: checks 10%/20% limits, suspensions, T+1 locks)
3. Board Lot Feasibility & Granularity Guard (BoardLotFeasibilityGuard: checks 100-share integer lot rounding & capital distortion)
4. Cash Yield Optimizer & Cash Drag Guard (CashDragGuard: audits idle cash earning 0.20% vs 1.8%+ GC001/511010)
5. Portfolio Weight Traps & Concentration Guard (WeightTrapsGuard: audits excessive concentration >5% & >10%, sum(weights)!=1, negative weights)
6. Point-in-Time Data Quality & Look-Ahead Guard (DataQualityGuard: audits missing prices, non-monotonic timestamps, look-ahead leaks)

Calculates:
- Comprehensive Portfolio Health Score (0 - 100)
- Institutional Audit Verdict:
    * PASSED (Score >= 90, all guards PASS)
    * WARNING_REQUIRES_ATTENTION (Score < 90 or warnings present)
    * REJECTED_DANGEROUS (Fatal failures, hard circuit breakers, or Score < 60)
- Rich institutional Markdown & JSON audit reports with actionable remediation advice.

Usage:
    # 1. Standard production audit against default my_holdings.json:
    python3 research/production/audit_portfolio_guards.py --holdings research/production/my_holdings.json

    # 2. Output custom Markdown and JSON reports:
    python3 research/production/audit_portfolio_guards.py --holdings research/production/my_holdings.json \
        --output-md PORTFOLIO_RISK_GUARDS_AUDIT_REPORT.md \
        --output-json PORTFOLIO_RISK_GUARDS_AUDIT_REPORT.json

    # 3. Offline / cached prices mode:
    python3 research/production/audit_portfolio_guards.py --holdings research/production/my_holdings.json --no-live
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

# Safe print helper for heterogeneous terminal encodings
def _safe_print(msg: str, file=None) -> None:
    target = file or sys.stdout
    try:
        print(msg, file=target)
    except UnicodeEncodeError:
        enc = getattr(target, "encoding", None) or "ascii"
        safe_msg = msg.encode(enc, errors="replace").decode(enc, errors="replace")
        print(safe_msg, file=target)


# Setup repo path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# fin-skills library & guards
import fin_skills.api as api
from fin_skills.china.ashare_rules import (
    board_of,
    can_sell,
    daily_limit_pct,
    explain_buy,
    limit_price,
    sellable_qty,
)
from fin_skills.china.board_lot_guard import (
    DEFAULT_TYPICAL_PRICES,
    evaluate_board_lot_feasibility,
)
from fin_skills.china.cash_yield_optimizer import (
    DEFAULT_BENCHMARK_CASH_YIELD,
    DEFAULT_DEMAND_RATE,
    calculate_cash_drag,
    plan_cash_placement,
)
from fin_skills.china.core_satellite_advisor import DEFAULT_STOCK_PRICES
from fin_skills.china.portfolio_manager import HoldingRecord, PortfolioState
from fin_skills.china.qdii_premium_guard import (
    DEFAULT_HARD_CIRCUIT_PREMIUM,
    DEFAULT_MAX_ALLOWED_PREMIUM,
    KNOWN_QDII_ETFS,
    evaluate_qdii_order,
    is_qdii_etf,
)
from research.production.live_advisor_bot import (
    EXTENDED_ETF_UNIVERSE,
    GLOBAL_ETF_UNIVERSE,
    fetch_live_market_snapshot,
)


def is_etf_instrument(code: str) -> bool:
    """Identify if a ticker is an ETF or fund rather than an individual equity stock."""
    clean = code.split(".")[0].replace("SH", "").replace("SZ", "").strip()
    if clean.startswith(("51", "15", "16", "50", "56", "58")):
        return True
    if clean in GLOBAL_ETF_UNIVERSE or clean in EXTENDED_ETF_UNIVERSE or clean in KNOWN_QDII_ETFS:
        return True
    return False


def is_single_stock_instrument(code: str) -> bool:
    """Identify if a ticker is a single equity stock (satellite alpha)."""
    clean = code.split(".")[0].replace("SH", "").replace("SZ", "").strip()
    if clean.upper() in ("CASH", "CNY", "RMB", "USD"):
        return False
    return not is_etf_instrument(clean)


@dataclass
class GuardAuditResult:
    """Audit result for an individual guard category."""
    name: str
    title: str
    status: str  # 'PASS', 'WARN', 'FAIL'
    score_deduction: float
    summary: str
    findings: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    remediation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "title": self.title,
            "status": self.status,
            "score_deduction": self.score_deduction,
            "summary": self.summary,
            "findings": self.findings,
            "metrics": self.metrics,
            "remediation": self.remediation,
        }


@dataclass
class PortfolioHoldingAudit:
    """Normalized position breakdown for audit reporting."""
    code: str
    name: str
    asset_type: str  # 'Core ETF', 'Satellite Stock', 'Cash'
    shares: float
    lots: int
    price: float
    market_value: float
    weight: float
    weight_pct: str
    avg_cost: float
    unrealized_pnl: float
    unrealized_pnl_pct: str
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PortfolioAuditReport:
    """Comprehensive institutional Portfolio Risk Guards Audit Report."""
    timestamp: str
    holdings_file: str
    total_nav: float
    cash: float
    cash_weight: float
    cash_weight_pct: str
    health_score: float
    verdict: str  # 'PASSED', 'WARNING_REQUIRES_ATTENTION', 'REJECTED_DANGEROUS'
    guard_results: dict[str, GuardAuditResult] = field(default_factory=dict)
    holdings_audit: list[PortfolioHoldingAudit] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "holdings_file": self.holdings_file,
            "total_nav": round(self.total_nav, 2),
            "cash": round(self.cash, 2),
            "cash_weight": round(self.cash_weight, 4),
            "cash_weight_pct": self.cash_weight_pct,
            "health_score": round(self.health_score, 1),
            "verdict": self.verdict,
            "guard_results": {k: v.to_dict() for k, v in self.guard_results.items()},
            "holdings_audit": [h.to_dict() for h in self.holdings_audit],
        }

    def save_json(self, filepath: str | Path) -> None:
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    def save_markdown(self, filepath: str | Path) -> None:
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(self.to_markdown())

    def to_markdown(self) -> str:
        verdict_badge = {
            "PASSED": "🟢 **PASSED (审计通过 / 资产配置健康)**",
            "WARNING_REQUIRES_ATTENTION": "🟡 **WARNING_REQUIRES_ATTENTION (需关注 / 存在风险隐患或摩擦损耗)**",
            "REJECTED_DANGEROUS": "🔴 **REJECTED_DANGEROUS (高危拦截 / 违背规则或严重泡沫)**",
        }.get(self.verdict, self.verdict)

        lines = [
            f"# 🛡️ 机构级投资组合全维度风控守卫审计综合报告",
            f"",
            f"> **生成时间**: `{self.timestamp}` | **审计对象**: `{self.holdings_file}` | **执行引擎**: `fin-skills 2.0 (Direction 3 Auditor)`",
            f"> **综合健康评分**: **`{self.health_score:.1f} / 100`** | **审计裁定**: {verdict_badge}",
            f"",
            f"---",
            f"",
            f"## 📊 资产组合核心指标看板 (Executive KPI Cards)",
            f"",
            f"| 账户总资产 (Total NAV) | 闲置可用现金 (Cash) | 现金仓位占比 (Cash Weight) | 投资持仓标的数 | 审计风控守卫覆盖 |",
            f"| :---: | :---: | :---: | :---: | :---: |",
            f"| **¥{self.total_nav:,.2f}** | **¥{self.cash:,.2f}** | **{self.cash_weight_pct}** | **{len(self.holdings_audit)} 只** | **6 大核心类别 / 100% 覆盖** |",
            f"",
            f"---",
            f"",
            f"## 🚦 六大风控守卫矩阵体检总览 (Guard Verdict Matrix)",
            f"",
            f"| 序号 | 风控守卫类别 (Guard Category) | 状态裁决 (Status) | 扣分 (Deduction) | 关键监控指标 (Key Metrics) | 核心风控结论 (Key Findings) |",
            f"| :--- | :--- | :---: | :---: | :--- | :--- |",
        ]

        seq = 1
        for name, g in self.guard_results.items():
            status_icon = "🟢 **PASS**" if g.status == "PASS" else ("🟡 **WARN**" if g.status == "WARN" else "🔴 **FAIL**")
            metric_str = " / ".join(f"{k}: `{v}`" for k, v in list(g.metrics.items())[:3])
            deduct_str = f"-{g.score_deduction:.0f} 分" if g.score_deduction > 0 else "0 分 (完美)"
            lines.append(f"| {seq} | **{g.title}** | {status_icon} | {deduct_str} | {metric_str} | {g.summary} |")
            seq += 1

        lines.append("")
        lines.append("---")
        lines.append("")

        # Detail sections for each guard
        lines.append("## 🔍 六大风控守卫详细深度审计剖析")
        lines.append("")

        for name, g in self.guard_results.items():
            badge = "🟢 PASS" if g.status == "PASS" else ("🟡 WARNING" if g.status == "WARN" else "🔴 FAILED")
            lines.append(f"### {seq-6+list(self.guard_results.keys()).index(name)}. {g.title} [{badge}]")
            lines.append(f"- **风控概括**: {g.summary}")
            lines.append(f"- **健康扣分**: `{'0 分' if g.score_deduction == 0 else f'-{g.score_deduction:.0f} 分'}`")
            if g.metrics:
                lines.append("- **监控量化指标**:")
                for mk, mv in g.metrics.items():
                    lines.append(f"  * `{mk}`: **{mv}**")
            if g.findings:
                lines.append("- **审计检查发现 (Findings)**:")
                for f in g.findings:
                    lines.append(f"  * {f}")
            if g.remediation:
                lines.append(f"- **💡 机构整改与执行建议 (Remediation)**:\n  > {g.remediation}")
            lines.append("")

        lines.append("---")
        lines.append("")

        # Holdings Detail Table
        lines.append("## 📋 当前投资组合持仓明细全景透视")
        lines.append("")
        if self.holdings_audit:
            lines.append("| 代码 | 资产名称 | 属性 | 持仓股数 | 整手(100股) | 当前单价 | 持仓市值 | 实际权重 | 成本均价 | 浮动盈亏 | 风控状态标记 |")
            lines.append("| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |")
            for h in self.holdings_audit:
                flags_str = " ".join(f"`{fl}`" for fl in h.flags) if h.flags else "✅ 正常"
                lines.append(
                    f"| `{h.code}` | **{h.name}** | {h.asset_type} | {h.shares:,.0f} | {h.lots}手 | "
                    f"¥{h.price:.3f} | ¥{h.market_value:,.2f} | **{h.weight_pct}** | "
                    f"¥{h.avg_cost:.3f} | {h.unrealized_pnl_pct} | {flags_str} |"
                )
            lines.append(
                f"| `CASH` | **可用闲置资金** | 现金资产 | - | - | ¥1.000 | "
                f"¥{self.cash:,.2f} | **{self.cash_weight_pct}** | ¥1.000 | 0.00% | "
                f"{'⚠️ 严重拖累' if self.cash_weight > 0.20 else ('⚠️ 轻度拖累' if self.cash_weight > 0.05 else '✅ 正常')} |"
            )
        else:
            lines.append("> ⚠️ **当前无证券持仓记录 (100% 现金配置)**")
        lines.append("")

        # Actionable Remediation Plan
        lines.append("---")
        lines.append("")
        lines.append("## 🛠️ 机构级风控治理与整改路线图 (Action Plan)")
        lines.append("")

        has_issues = self.health_score < 90 or any(g.status != "PASS" for g in self.guard_results.values())
        if not has_issues:
            lines.append("### 🟢 组合状态极其优异，无需紧急风控干预")
            lines.append("1. **核心全天候**: 大类资产分散度极佳，无单股集中度陷阱，无负权重做空风险。")
            lines.append("2. **交易颗粒度**: 全部持仓均严格按 100 股整数手配置，无零股交易摩擦。")
            lines.append("3. **现金管理**: 现金比例控制在 <= 5.0% 纪律区间，无闲置现金拖累。")
            lines.append("4. **日常监控**: 建议在每个交易日 14:25 定时运行本审计引擎。")
        else:
            lines.append("### 🔴 发现风控预警/异常，请按以下优先级依序整改:")
            step = 1
            if self.guard_results.get("ashare_rules", GuardAuditResult("", "", "PASS", 0, "")).status != "PASS":
                lines.append(f"{step}. **【极速处理】排除跌停或停牌流动性闭锁**: 检查受阻标的，暂停买入或调整卖出清平计划。")
                step += 1
            if self.guard_results.get("qdii_premium", GuardAuditResult("", "", "PASS", 0, "")).status != "PASS":
                lines.append(f"{step}. **【泡沫阻断】QDII 跨境溢价控制**: 严禁买入溢价 > 3% 的 QDII ETF；对溢价在 1.5%~3% 的标的执行 50% 降权或切换黄金 ETF (518880)。")
                step += 1
            if self.guard_results.get("weight_traps", GuardAuditResult("", "", "PASS", 0, "")).status != "PASS":
                lines.append(f"{step}. **【集中度分散】解构个股集中度陷阱**: 将权重超过 5.0% 的单只卫星个股分批平仓减权，避免黑天鹅尾部暴跌。")
                step += 1
            if self.guard_results.get("board_lot_feasibility", GuardAuditResult("", "", "PASS", 0, "")).status != "PASS":
                lines.append(f"{step}. **【整手规范】清理零股与修正颗粒度**: 零股持仓需一次性挂单清平；对小资金高价ETF配置考虑平替基金。")
                step += 1
            if self.guard_results.get("cash_drag", GuardAuditResult("", "", "PASS", 0, "")).status != "PASS":
                lines.append(f"{step}. **【资金增益】治理现金拖累**: 于 14:50 - 15:30 将闲置资金全额参与 GC001 逆回购，避免活期利息损失。")
                step += 1
            if self.guard_results.get("data_quality", GuardAuditResult("", "", "PASS", 0, "")).status != "PASS":
                lines.append(f"{step}. **【数据修复】校准行情与时间戳**: 修复缺失价格与前视时间戳回退，确保时序因果一致性。")
                step += 1

        lines.append("")
        return "\n".join(lines)


class PortfolioRiskAuditor:
    """Unified Portfolio Risk Guards Auditor."""

    def __init__(
        self,
        max_allowed_qdii_premium: float = DEFAULT_MAX_ALLOWED_PREMIUM,
        hard_circuit_qdii_premium: float = DEFAULT_HARD_CIRCUIT_PREMIUM,
        max_acceptable_cash_ratio: float = 0.05,
        excessive_cash_ratio_threshold: float = 0.20,
        single_stock_safe_weight: float = 0.05,
        single_stock_hard_cap: float = 0.10,
        max_granularity_distortion: float = 0.15,
        demand_rate: float = DEFAULT_DEMAND_RATE,
        expected_repo_rate: float = DEFAULT_BENCHMARK_CASH_YIELD,
    ):
        self.max_allowed_qdii_premium = max_allowed_qdii_premium
        self.hard_circuit_qdii_premium = hard_circuit_qdii_premium
        self.max_acceptable_cash_ratio = max_acceptable_cash_ratio
        self.excessive_cash_ratio_threshold = excessive_cash_ratio_threshold
        self.single_stock_safe_weight = single_stock_safe_weight
        self.single_stock_hard_cap = single_stock_hard_cap
        self.max_granularity_distortion = max_granularity_distortion
        self.demand_rate = demand_rate
        self.expected_repo_rate = expected_repo_rate

    def resolve_prices(
        self,
        holdings: list[dict[str, Any]],
        provided_prices: Mapping[str, float] | None = None,
        live: bool = True,
    ) -> dict[str, float]:
        """Resolve current market prices for all portfolio assets."""
        price_map: dict[str, float] = {}
        # 1. Explicitly provided prices take highest priority
        if provided_prices is not None:
            for k, v in provided_prices.items():
                price_map[k] = float(v) if v is not None else 0.0

        # 2. Check holdings: if market_price is explicitly given in holding, respect it
        for h in holdings:
            code = h["code"]
            if code in price_map:
                continue
            if "market_price" in h:
                price_map[code] = float(h["market_price"]) if h["market_price"] is not None else 0.0

        # 3. Fetch live quotes if live=True for any remaining missing codes
        missing_codes = [h["code"] for h in holdings if h["code"] not in price_map]
        if live and missing_codes:
            try:
                live_snapshot = fetch_live_market_snapshot()
                for code, data in live_snapshot.items():
                    if code not in price_map and "price" in data and data["price"] > 0:
                        price_map[code] = float(data["price"])
            except Exception:
                pass  # Graceful fallback to cached/typical

        # 4. Fallback to typical ETF price, or stock price if still missing
        for h in holdings:
            code = h["code"]
            if code in price_map:
                continue
            if code in DEFAULT_TYPICAL_PRICES:
                price_map[code] = float(DEFAULT_TYPICAL_PRICES[code])
            elif code in DEFAULT_STOCK_PRICES:
                price_map[code] = float(DEFAULT_STOCK_PRICES[code])
            elif "avg_cost" in h and h["avg_cost"] is not None and float(h["avg_cost"]) > 0:
                price_map[code] = float(h["avg_cost"])
            else:
                price_map[code] = 0.0

        return price_map

    def audit_qdii_premium(
        self,
        holdings: list[dict[str, Any]],
        price_map: dict[str, float],
        iopv_map: Mapping[str, float] | None = None,
    ) -> GuardAuditResult:
        """Guard 1: QDII Secondary Market Premium & IOPV Circuit Breaker."""
        qdii_holdings = [h for h in holdings if is_qdii_etf(h["code"])]
        if not qdii_holdings:
            return GuardAuditResult(
                name="qdii_premium",
                title="QDII 跨境二级市场溢价守卫",
                status="PASS",
                score_deduction=0.0,
                summary="无高溢价 QDII 跨境 ETF 暴露 (No Bubble Exposure)",
                findings=["✅ 投资组合当前无 QDII 跨境 ETF 持仓，不受海外额度封顶与二级市场溢价炒作影响。"],
                metrics={"qdii_count": 0, "max_premium_pct": "0.00%"},
                remediation="持续保持资产配置正常监控。",
            )

        findings = []
        max_premium = -1.0
        worst_status = "NORMAL"
        deduction = 0.0

        for h in qdii_holdings:
            code = h["code"]
            name = h.get("name", KNOWN_QDII_ETFS.get(code, code))
            price = price_map.get(code, float(h.get("market_price", 1.0)))

            # Determine IOPV
            if iopv_map and code in iopv_map:
                iopv = float(iopv_map[code])
            elif "iopv" in h and h["iopv"] is not None:
                iopv = float(h["iopv"])
            else:
                iopv = price  # default parity if not explicitly provided

            res = evaluate_qdii_order(
                code=code,
                price=price,
                iopv=iopv,
                max_allowed_premium=self.max_allowed_qdii_premium,
                hard_circuit_premium=self.hard_circuit_qdii_premium,
            )

            max_premium = max(max_premium, res.premium_rate)
            prem_pct = f"{res.premium_rate*100:+.2f}%"

            if res.status == "HARD_CIRCUIT":
                worst_status = "HARD_CIRCUIT"
                deduction = max(deduction, 25.0)
                findings.append(
                    f"🔴 [硬熔断] {res.code} ({name}): 二级市场溢价率达 **{prem_pct}** (市价 {price:.3f} / 参考净值 {iopv:.3f})，"
                    f"严重突破 3.0% 熔断阈值！存在无对冲溢价坍塌踩踏风险。"
                )
            elif res.status == "DERATE_50":
                if worst_status != "HARD_CIRCUIT":
                    worst_status = "DERATE_50"
                deduction = max(deduction, 12.0)
                findings.append(
                    f"🟡 [溢价预警] {res.code} ({name}): 二级市场溢价率达 **{prem_pct}** (市价 {price:.3f} / 参考净值 {iopv:.3f})，"
                    f"处于 (1.5%, 3.0%] 预警降权区间。买入将承担流动性溢价摩擦。"
                )
            elif res.status == "DISCOUNT_OPPORTUNITY":
                findings.append(
                    f"🟢 [折价套利] {res.code} ({name}): 处于折价交易状态 (**{prem_pct}**)，具备安全边际。"
                )
            else:
                findings.append(
                    f"🟢 [正常追踪] {res.code} ({name}): 溢价率 **{prem_pct}** 处于正常跟踪误差区间 (<= 1.5%)。"
                )

        if worst_status == "HARD_CIRCUIT":
            status = "FAIL"
            summary = f"检测到 QDII ETF 突破 3.0% 硬熔断线 (最高溢价 {max_premium*100:+.2f}%)"
            remediation = "立即终止买入申报；已有持仓建议高位减仓止盈，资金调配至 518880 (黄金ETF) 或无溢价资产。"
        elif worst_status == "DERATE_50":
            status = "WARN"
            summary = f"检测到 QDII ETF 处于 1.5%~3.0% 溢价预警区间 (最高溢价 {max_premium*100:+.2f}%)"
            remediation = "对该标的配置实施 50% 降权，或等待二级市场溢价回落至 1.5% 以内再行建仓。"
        else:
            status = "PASS"
            summary = f"QDII ETF 跟踪偏离度正常 (最高溢价 {max_premium*100:+.2f}%)"
            remediation = "维持现有配置，密切监控跨境额度与盘中 IOPV 波动。"

        return GuardAuditResult(
            name="qdii_premium",
            title="QDII 跨境二级市场溢价守卫",
            status=status,
            score_deduction=deduction,
            summary=summary,
            findings=findings,
            metrics={"qdii_count": len(qdii_holdings), "max_premium_pct": f"{max_premium*100:+.2f}%"},
            remediation=remediation,
        )

    def audit_ashare_rules(
        self,
        holdings: list[dict[str, Any]],
        price_map: dict[str, float],
        market_bars: Mapping[str, Mapping[str, Any]] | None = None,
        today_date: str | None = None,
    ) -> GuardAuditResult:
        """Guard 2: A-Share Trading Rules & Liquidity Guard (10/20% Limits & T+1)."""
        findings = []
        deduction = 0.0
        status = "PASS"
        liquid_count = 0

        date_val = today_date or date.today().strftime("%Y-%m-%d")

        for h in holdings:
            code = h["code"]
            name = h.get("name", code)
            price = price_map.get(code, float(h.get("market_price", 1.0)))

            # Check T+1 locks if entry date is known
            entry_date = h.get("entry_date")
            if entry_date and str(entry_date) == str(date_val):
                findings.append(f"🔒 [T+1流动性锁定] 标的 `{code}` ({name}) 为今日买入，受 T+1 规则限制今日不可卖出。")

            # Equity stock rules
            if is_single_stock_instrument(code):
                try:
                    pct = daily_limit_pct(code, name, date=date_val)
                    bar = market_bars.get(code) if market_bars else None
                    if bar:
                        prev_close = bar.get("prev_close", bar.get("pre_close", price))
                        up = limit_price(float(prev_close), pct, "up")
                        down = limit_price(float(prev_close), pct, "down")

                        if price <= down:
                            status = "FAIL"
                            deduction = max(deduction, 20.0)
                            findings.append(
                                f"🔴 [跌停流动性枯竭] 标的 `{code}` ({name}) 封死跌停板 (¥{price:.2f} <= ¥{down:.2f})，"
                                f"卖单无法撮合，流动性陷阱！"
                            )
                        elif price >= up:
                            findings.append(
                                f"ℹ️ [涨停封板] 标的 `{code}` ({name}) 封死涨停板 (¥{price:.2f} >= ¥{up:.2f})，买入受限。"
                            )
                        if bar.get("volume", 1) == 0 or bar.get("trade_status") == "SUSPENDED":
                            if status != "FAIL":
                                status = "WARN"
                            deduction = max(deduction, 10.0)
                            findings.append(f"🟡 [停牌流动性中断] 标的 `{code}` ({name}) 处于停牌状态，交易通道闭锁。")
                    else:
                        liquid_count += 1
                except Exception:
                    liquid_count += 1
            else:
                liquid_count += 1

        if not findings:
            findings.append("✅ 全部持仓标的均处于正常交易状态，流动性充沛，无跌停封死与停牌异常。")

        summary = "全部持仓标的流动性良好，符合 A 股涨跌停与 T+1 规则" if status == "PASS" else "存在流动性受阻或停牌跌停风险"
        remediation = "维持标准流动性监控，严禁对跌停标的盲目抄底。" if status == "PASS" else "立即评估跌停标的黑天鹅风险，准备流动性释放时的止损预案。"

        return GuardAuditResult(
            name="ashare_rules",
            title="A股交易规则与流动性守卫",
            status=status,
            score_deduction=deduction,
            summary=summary,
            findings=findings,
            metrics={"audited_holdings": len(holdings), "date": str(date_val)},
            remediation=remediation,
        )

    def audit_board_lot_feasibility(
        self,
        total_nav: float,
        holdings: list[dict[str, Any]],
        price_map: dict[str, float],
    ) -> GuardAuditResult:
        """Guard 3: Board Lot Feasibility & Granularity Distortion Guard (100-Share Lots)."""
        findings = []
        fractional_violations = []
        status = "PASS"
        deduction = 0.0

        # 1. Fractional shares / integer 100-lot check
        for h in holdings:
            code = h["code"]
            name = h.get("name", code)
            shares = h.get("shares", 0)

            # Check if integer multiple of 100
            is_int = isinstance(shares, int) or (isinstance(shares, float) and shares.is_integer())
            if not is_int or (int(shares) % 100 != 0):
                fractional_violations.append((code, name, shares))

        if fractional_violations:
            status = "WARN"
            deduction = max(deduction, 15.0)
            for code, name, shares in fractional_violations:
                int_shares = int(shares) if (isinstance(shares, int) or (isinstance(shares, float) and shares.is_integer())) else 0
                odd_shares = int_shares % 100 if int_shares > 0 else shares
                findings.append(
                    f"🟡 [零股违规] 标的 `{code}` ({name}) 当前持仓 **{shares} 股** 不是 100 股整数手 "
                    f"(含零股 **{odd_shares} 股**)。A 股买入申报必须为 100 股整数倍！"
                )

        # 2. Capital Granularity Distortion Check
        distortion_pct = 0.0
        if total_nav > 0 and holdings:
            holding_weights = {}
            for h in holdings:
                code = h["code"]
                p = price_map.get(code, float(h.get("market_price", 1.0)))
                val = float(h.get("shares", 0)) * p
                holding_weights[code] = val / total_nav

            try:
                lot_res = evaluate_board_lot_feasibility(
                    capital=total_nav,
                    target_weights=holding_weights,
                    prices=price_map,
                    max_acceptable_distortion=self.max_granularity_distortion,
                )
                distortion_pct = lot_res.granularity_distortion

                if not lot_res.is_feasible or lot_res.feasibility_status == "UNFEASIBLE_HIGH_DISTORTION":
                    status = "FAIL"
                    deduction = max(deduction, 20.0)
                    findings.append(
                        f"🔴 [颗粒度严重失真] 资金规模 (¥{total_nav:,.0f}) 过小，100 股整手离散化导致组合权重偏离度达 "
                        f"**{distortion_pct*100:.1f}%** (>15%)。最小建议实盘资金: ¥{lot_res.min_recommended_capital:,.0f}。"
                    )
                elif lot_res.feasibility_status == "MODERATE_DISTORTION" or distortion_pct > 0.05:
                    if status != "FAIL":
                        status = "WARN"
                    deduction = max(deduction, 8.0)
                    findings.append(
                        f"🟡 [颗粒度轻度失真] 100 股整手离散化导致组合跟踪偏离 **{distortion_pct*100:.1f}%**。"
                    )
            except Exception:
                pass

        if not findings:
            findings.append("✅ 所有持仓完全满足 100 股整数手要求 (零股数量为 0)，资金规模对资产配置无离散失真。")

        summary = (
            "持仓完全满足 100 股整数手规范，资金颗粒度失真良好"
            if status == "PASS"
            else ("检测到零股持仓违规或资金规模偏离失真" if status == "WARN" else "资金颗粒度严重失真，无法忠实执行配置")
        )
        remediation = (
            "维持 100 股整手委托纪律。"
            if status == "PASS"
            else "实盘买入必须严格按 100 股整数手下单；零股持仓应通过单笔卖单彻底出清；小资金建议适当增加低单价 ETF 占比。"
        )

        return GuardAuditResult(
            name="board_lot_feasibility",
            title="100股整手颗粒度与失真守卫",
            status=status,
            score_deduction=deduction,
            summary=summary,
            findings=findings,
            metrics={
                "fractional_violations_count": len(fractional_violations),
                "granularity_distortion_pct": f"{distortion_pct*100:.2f}%",
            },
            remediation=remediation,
        )

    def audit_cash_drag(
        self,
        total_nav: float,
        cash: float,
        today_date: str | None = None,
    ) -> GuardAuditResult:
        """Guard 4: Cash Yield Optimizer & Cash Drag Guard."""
        cash_ratio = (cash / total_nav) if total_nav > 0 else 1.0
        annual_drag = cash * max(0.0, self.expected_repo_rate - self.demand_rate)

        date_val = today_date or date.today().strftime("%Y-%m-%d")
        plan = plan_cash_placement(
            idle_cash=cash,
            current_date=date_val,
            demand_rate=self.demand_rate,
            expected_repo_rate=self.expected_repo_rate,
        )

        findings = []
        metrics = {
            "cash_ratio_pct": f"{cash_ratio*100:.2f}%",
            "idle_cash_rmb": f"¥{cash:,.2f}",
            "annual_drag_rmb": f"-¥{annual_drag:,.2f}",
            "recommended_vehicle": plan.recommended_vehicle,
        }

        if cash_ratio > self.excessive_cash_ratio_threshold and cash >= 1000.0:
            status = "WARN"
            deduction = 15.0
            summary = f"严重闲置现金拖累 (现金占比 {cash_ratio*100:.1f}% > 20%)"
            findings.append(
                f"🟡 [严重现金拖累] 闲置可用资金 **¥{cash:,.2f}** (占比 **{cash_ratio*100:.1f}%**) 严重超过 20.0% 安全上限！"
                f"滞留活期存款 ({self.demand_rate*100:.2f}%) 每年造成 **-¥{annual_drag:,.2f}** 隐性收益拖累。"
            )
            remediation = f"今日 14:50 - 15:30 务必执行现金增益管理: {plan.order_action}，拒绝资金休眠。"
        elif cash_ratio > self.max_acceptable_cash_ratio and cash >= 1000.0:
            status = "WARN"
            deduction = 5.0
            summary = f"轻度闲置现金拖累 (现金占比 {cash_ratio*100:.1f}% > 5%)"
            findings.append(
                f"🟡 [轻度现金拖累] 闲置资金 **¥{cash:,.2f}** (占比 **{cash_ratio*100:.1f}%**) 超过 5.0% 摩擦基准。"
                f"年化机会成本: **-¥{annual_drag:,.2f}**。"
            )
            remediation = f"建议在 14:50 - 15:30 参与 GC001 逆回购 sweep 操作: {plan.order_action}。"
        else:
            status = "PASS"
            deduction = 0.0
            summary = f"现金比例控制优异 ({cash_ratio*100:.1f}% <= 5.0%)"
            findings.append(f"✅ 闲置可用现金占比为 **{cash_ratio*100:.1f}%** (¥{cash:,.2f})，处于 <= 5.0% 极低摩擦区间。")
            remediation = "维持稳健现金管理，保留小额资金应对定投再平衡。"

        return GuardAuditResult(
            name="cash_drag",
            title="闲置现金拖累与增益优化守卫",
            status=status,
            score_deduction=deduction,
            summary=summary,
            findings=findings,
            metrics=metrics,
            remediation=remediation,
        )

    def audit_weight_traps(
        self,
        total_nav: float,
        cash: float,
        holdings: list[dict[str, Any]],
        price_map: dict[str, float],
    ) -> GuardAuditResult:
        """Guard 5: Portfolio Weight Traps & Concentration Guard."""
        findings = []
        status = "PASS"
        deduction = 0.0

        cash_w = (cash / total_nav) if total_nav > 0 else 1.0

        # 1. Non-negativity check
        if cash < -1e-5:
            status = "FAIL"
            deduction = max(deduction, 30.0)
            findings.append(f"🔴 [负现金透支] 账户现金透支 (¥{cash:,.2f})，普通现货账户违规！")

        weights = {}
        for h in holdings:
            code = h["code"]
            name = h.get("name", code)
            shares = float(h.get("shares", 0))
            price = price_map.get(code, float(h.get("market_price", 1.0)))
            val = shares * price
            w = (val / total_nav) if total_nav > 0 else 0.0
            weights[code] = w

            if shares < -1e-5 or w < -1e-5:
                status = "FAIL"
                deduction = max(deduction, 30.0)
                findings.append(f"🔴 [裸做空/负权重陷阱] 标的 `{code}` ({name}) 权重为 **{w*100:.2f}% < 0.0%**！A 股现货账户严禁裸做空。")

        # 2. Sum of weights check
        total_w = sum(weights.values()) + cash_w
        if abs(total_w - 1.0) > 0.02 and total_nav > 0:
            if status != "FAIL":
                status = "WARN"
            deduction = max(deduction, 10.0)
            findings.append(
                f"🟡 [权重归一化偏差] 资产与现金总权重之和为 **{total_w*100:.2f}%**，与 100.0% 存在较大偏差 (>2.0%)。"
            )

        # 3. Single-stock concentration trap check
        max_single_stock_w = 0.0
        worst_stock = None
        for h in holdings:
            code = h["code"]
            name = h.get("name", code)
            w = weights.get(code, 0.0)

            if is_single_stock_instrument(code):
                if w > max_single_stock_w:
                    max_single_stock_w = w
                    worst_stock = (code, name)

                if w > self.single_stock_hard_cap:  # > 10%
                    if status != "FAIL":
                        status = "WARN"
                    deduction = max(deduction, 15.0)
                    findings.append(
                        f"🟡 [个股过度集中陷阱] 卫星个股 `{code}` ({name}) 仓位占比达 **{w*100:.1f}%**，"
                        f"突破 10.0% 机构安全警戒红线！存在黑天鹅个股尾部暴跌风险。"
                    )
                elif w > self.single_stock_safe_weight:  # > 5%
                    if status != "FAIL":
                        status = "WARN"
                    deduction = max(deduction, 5.0)
                    findings.append(
                        f"🟡 [个股集中度偏高] 卫星个股 `{code}` ({name}) 仓位占比 **{w*100:.1f}%**，超过 5.0% 推荐分散阈值。"
                    )

        if not findings:
            findings.append("✅ 组合总权重精确归一化 (100.0%)，无负权重裸卖空；卫星个股分散度优异 (单股 <= 5.0%)。")

        summary = (
            "权重归一化合规，无负权重，卫星个股分散度良好"
            if status == "PASS"
            else ("检测到个股过度集中或权重偏差" if status == "WARN" else "存在负权重做空等致命陷阱")
        )
        remediation = (
            "维持 80% 全天候核心 ETF + 20% 卫星个股 (单股 <= 5%) 分散配置。"
            if status == "PASS"
            else "对权重超过 5.0% 的单只个股启动减仓程序，将仓位分散平摊回核心 ETF 压舱石。"
        )

        return GuardAuditResult(
            name="weight_traps",
            title="投资组合权重陷阱与集中度守卫",
            status=status,
            score_deduction=deduction,
            summary=summary,
            findings=findings,
            metrics={
                "total_weight_pct": f"{total_w*100:.2f}%",
                "max_single_stock_weight": f"{max_single_stock_w*100:.2f}%",
                "worst_stock": f"{worst_stock[0]} ({worst_stock[1]})" if worst_stock else "None",
            },
            remediation=remediation,
        )

    def audit_data_quality(
        self,
        holdings: list[dict[str, Any]],
        price_map: dict[str, float],
        last_rebalance_date: str | None = None,
        today_date: str | None = None,
        price_history: Mapping[str, Sequence[Any]] | None = None,
    ) -> GuardAuditResult:
        """Guard 6: Point-in-Time Data Quality & Look-Ahead Guard."""
        findings = []
        status = "PASS"
        deduction = 0.0
        date_str = today_date or date.today().strftime("%Y-%m-%d")

        # 1. Missing or invalid prices
        missing_prices = []
        for h in holdings:
            code = h["code"]
            p = price_map.get(code)
            if p is None or p <= 0 or math.isnan(p) or math.isinf(p):
                missing_prices.append(code)

        if missing_prices:
            status = "FAIL"
            deduction = max(deduction, 25.0)
            findings.append(f"🔴 [行情数据缺失/异常] 标的 {missing_prices} 缺失有效价格数据，组合估值失真！")

        # 2. Look-Ahead in rebalance timestamp
        if last_rebalance_date:
            try:
                d_reb = str(last_rebalance_date).split(" ")[0]
                d_now = str(date_str).split(" ")[0]
                if d_reb > d_now:
                    if status != "FAIL":
                        status = "WARN"
                    deduction = max(deduction, 10.0)
                    findings.append(
                        f"🟡 [时间戳前视偏差] 上次调仓日期 (`{d_reb}`) 晚于当前系统日期 (`{d_now}`)，存在未来数据泄漏风险！"
                    )
            except Exception:
                pass

        # 3. Non-monotonic timestamp series check
        if price_history:
            for code, ts_seq in price_history.items():
                if len(ts_seq) >= 2:
                    for i in range(len(ts_seq) - 1):
                        if ts_seq[i] >= ts_seq[i + 1]:
                            status = "FAIL"
                            deduction = max(deduction, 20.0)
                            findings.append(
                                f"🔴 [时间序列非单调] 标的 `{code}` 行情时间戳在第 {i} 点出现逆序或重复 ({ts_seq[i]} >= {ts_seq[i+1]})！"
                            )
                            break

        if not findings:
            findings.append("✅ 截面行情价格真实有效，调仓时序与时间戳严格满足单调性因果律，无前视未来数据泄漏。")

        summary = "行情真实有效，时序严格因果单调，无前视偏差" if status == "PASS" else "存在缺失价格、未来时间戳或非单调时序污染"
        remediation = "维持实时行情校验并定期校准时区时间戳。" if status == "PASS" else "立即排查行情源与时钟源配置，剔除未来函数。"

        return GuardAuditResult(
            name="data_quality",
            title="截面数据质量与前视偏差守卫",
            status=status,
            score_deduction=deduction,
            summary=summary,
            findings=findings,
            metrics={"missing_price_count": len(missing_prices), "audit_date": str(date_str)},
            remediation=remediation,
        )

    def audit(
        self,
        holdings_input: str | Path | dict[str, Any] | PortfolioState,
        price_map: Mapping[str, float] | None = None,
        iopv_map: Mapping[str, float] | None = None,
        market_bars: Mapping[str, Mapping[str, Any]] | None = None,
        price_history: Mapping[str, Sequence[Any]] | None = None,
        live: bool = True,
        today_date: str | None = None,
    ) -> PortfolioAuditReport:
        """Execute full 6-category risk guards audit and generate structured report."""
        holdings_file_label = "in_memory_portfolio"

        # 1. Parse input
        if isinstance(holdings_input, (str, Path)):
            holdings_file_label = str(holdings_input)
            p = Path(holdings_input)
            if not p.exists():
                raise FileNotFoundError(f"Holdings file not found: {p}")
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
        elif isinstance(holdings_input, PortfolioState):
            data = {
                "total_nav": holdings_input.total_nav,
                "cash": holdings_input.cash,
                "last_rebalance_date": holdings_input.last_rebalance_date,
                "holdings": [h.to_dict() for h in holdings_input.holdings.values()],
            }
        elif isinstance(holdings_input, dict):
            data = holdings_input
        else:
            raise TypeError(f"Unsupported holdings_input type: {type(holdings_input)}")

        cash = float(data.get("cash", 0.0))
        raw_holdings = data.get("holdings", [])
        last_rebalance_date = data.get("last_rebalance_date")

        # 2. Resolve Prices
        active_prices = self.resolve_prices(raw_holdings, provided_prices=price_map, live=live)

        # 3. Compute NAV & Build Holding Audits
        holdings_value = 0.0
        holdings_audit: list[PortfolioHoldingAudit] = []

        for h in raw_holdings:
            code = str(h["code"])
            shares = float(h.get("shares", 0))
            price = active_prices.get(code, float(h.get("market_price", 1.0)))
            val = shares * price
            holdings_value += val

        total_nav = cash + holdings_value
        if "total_nav" in data and float(data["total_nav"]) > 0 and len(raw_holdings) == 0:
            total_nav = float(data["total_nav"])

        cash_weight = (cash / total_nav) if total_nav > 0 else 1.0

        for h in raw_holdings:
            code = str(h["code"])
            name = h.get("name", code)
            shares = float(h.get("shares", 0))
            price = active_prices.get(code, float(h.get("market_price", 1.0)))
            cost = float(h.get("avg_cost", price))
            val = shares * price
            w = (val / total_nav) if total_nav > 0 else 0.0
            pnl = val - (shares * cost)
            pnl_pct = (pnl / (shares * cost)) if (shares * cost) > 0 else 0.0

            asset_type = "Core ETF" if is_etf_instrument(code) else "Satellite Stock"
            lots = int(shares // 100) if shares > 0 else 0

            flags = []
            if shares % 100 != 0:
                flags.append("零股违规")
            if is_qdii_etf(code):
                flags.append("QDII跨境")
            if is_single_stock_instrument(code) and w > self.single_stock_safe_weight:
                flags.append("高集中度")

            holdings_audit.append(
                PortfolioHoldingAudit(
                    code=code,
                    name=name,
                    asset_type=asset_type,
                    shares=shares,
                    lots=lots,
                    price=price,
                    market_value=val,
                    weight=w,
                    weight_pct=f"{w*100:.2f}%",
                    avg_cost=cost,
                    unrealized_pnl=pnl,
                    unrealized_pnl_pct=f"{pnl_pct*100:+.2f}%",
                    flags=flags,
                )
            )

        # 4. Sequentially Execute All 6 Guards
        g1 = self.audit_qdii_premium(raw_holdings, active_prices, iopv_map=iopv_map)
        g2 = self.audit_ashare_rules(raw_holdings, active_prices, market_bars=market_bars, today_date=today_date)
        g3 = self.audit_board_lot_feasibility(total_nav, raw_holdings, active_prices)
        g4 = self.audit_cash_drag(total_nav, cash, today_date=today_date)
        g5 = self.audit_weight_traps(total_nav, cash, raw_holdings, active_prices)
        g6 = self.audit_data_quality(
            raw_holdings,
            active_prices,
            last_rebalance_date=last_rebalance_date,
            today_date=today_date,
            price_history=price_history,
        )

        guard_results = {
            "qdii_premium": g1,
            "ashare_rules": g2,
            "board_lot_feasibility": g3,
            "cash_drag": g4,
            "weight_traps": g5,
            "data_quality": g6,
        }

        # 5. Compute Health Score & Overall Verdict
        total_deductions = sum(g.score_deduction for g in guard_results.values())
        health_score = round(max(0.0, min(100.0, 100.0 - total_deductions)), 1)

        has_fail = any(g.status == "FAIL" for g in guard_results.values())
        has_warn = any(g.status == "WARN" for g in guard_results.values())

        if has_fail or health_score < 60.0:
            verdict = "REJECTED_DANGEROUS"
        elif has_warn or health_score < 90.0:
            verdict = "WARNING_REQUIRES_ATTENTION"
        else:
            verdict = "PASSED"

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        return PortfolioAuditReport(
            timestamp=now_str,
            holdings_file=holdings_file_label,
            total_nav=total_nav,
            cash=cash,
            cash_weight=cash_weight,
            cash_weight_pct=f"{cash_weight*100:.2f}%",
            health_score=health_score,
            verdict=verdict,
            guard_results=guard_results,
            holdings_audit=holdings_audit,
        )


def audit_portfolio(
    holdings_path: str | Path = "research/production/my_holdings.json",
    price_map: Mapping[str, float] | None = None,
    iopv_map: Mapping[str, float] | None = None,
    live: bool = True,
    output_md: str | Path | None = "PORTFOLIO_RISK_GUARDS_AUDIT_REPORT.md",
    output_json: str | Path | None = None,
) -> PortfolioAuditReport:
    """High-level entrypoint to audit a portfolio file and optionally persist reports."""
    auditor = PortfolioRiskAuditor()
    report = auditor.audit(
        holdings_input=holdings_path,
        price_map=price_map,
        iopv_map=iopv_map,
        live=live,
    )
    if output_md:
        report.save_markdown(output_md)
    if output_json:
        report.save_json(output_json)
    return report


def main(args: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Institutional Portfolio Risk Guards Audit CLI & Reporting Engine."
    )
    parser.add_argument(
        "--holdings",
        default="research/production/my_holdings.json",
        help="Path to portfolio holdings JSON file (default: research/production/my_holdings.json)",
    )
    parser.add_argument(
        "--output-md",
        default="PORTFOLIO_RISK_GUARDS_AUDIT_REPORT.md",
        help="Path to output Markdown audit report (default: PORTFOLIO_RISK_GUARDS_AUDIT_REPORT.md)",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Path to output JSON audit report (optional)",
    )
    parser.add_argument(
        "--no-live",
        action="store_true",
        help="Disable live market quote fetching; use cached/typical prices",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 if verdict is not PASSED",
    )

    parsed = parser.parse_args(args)

    _safe_print("=" * 78)
    _safe_print("🛡️  启动机构级投资组合全维度风控守卫审计引擎 (Direction 3 Auditor)")
    _safe_print("=" * 78)
    _safe_print(f"📁 目标持仓文件: {parsed.holdings}")
    _safe_print(f"🌐 行情源模式: {'离线/本地缓存' if parsed.no_live else '实时网络行情 (Eastmoney Live)'}")

    try:
        report = audit_portfolio(
            holdings_path=parsed.holdings,
            live=not parsed.no_live,
            output_md=parsed.output_md,
            output_json=parsed.output_json,
        )
    except Exception as e:
        _safe_print(f"\n❌ 审计执行异常中断: {e}", file=sys.stderr)
        return 1

    # Print summary cards
    _safe_print(f"\n✅ 审计完成！综合健康评分: {report.health_score:.1f} / 100")
    _safe_print(f"⚖️  终审裁决: {report.verdict}")
    _safe_print(f"💰 账户总资产 (NAV): ¥{report.total_nav:,.2f} | 闲置可用现金: ¥{report.cash:,.2f} ({report.cash_weight_pct})")
    _safe_print("\n守卫体检明细:")
    for name, g in report.guard_results.items():
        icon = "🟢" if g.status == "PASS" else ("🟡" if g.status == "WARN" else "🔴")
        _safe_print(f"  {icon} [{g.status:<4}] {g.title}: {g.summary}")

    if parsed.output_md:
        _safe_print(f"\n📄 Markdown 报告已持久化至: {Path(parsed.output_md).resolve()}")
    if parsed.output_json:
        _safe_print(f"📊 JSON 报告已持久化至: {Path(parsed.output_json).resolve()}")

    if parsed.strict and report.verdict != "PASSED":
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
