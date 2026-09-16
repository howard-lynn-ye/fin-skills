"""Core-Satellite (80% Core All-Weather ETFs + 20% Tier-S Stock Event Alpha) Portfolio Advisor.

Why this exists:
1. Pure broad-index ETF portfolios (All-Weather / Risk Parity) deliver high Sharpe and low drawdown
   (80% Core sleeve), but miss high-conviction single-stock event alpha triggered by institutional
   researchers and retail capitulation bottoms.
2. Conversely, unconstrained retail stock-picking suffers from high noise and severe drawdowns on
   low-predictability random-walk stocks (Tier C: e.g., 02015 Li Auto, 09868 XPeng, SZ300014 EVE Energy).
3. The Unified Core-Satellite Cockpit bridges both worlds by enforcing a strict 80/20 risk budget:
   - 80% Core All-Weather ETFs: Macro risk parity / all-weather foundation.
   - 20% Satellite Event Alpha Sleeve: Strictly gated by all 4 intelligence modules:
     * DataQualityAuditor: Sanitizes scraped text, rejects spam/clickbait, scores information density.
     * KOLCredibilityRegistry: Weights verified Elite Alpha KOLs (+3.0x/+2.5x) and inverts Contrarian
       Indicators (-1.5x) while stripping directional weight from media aggregators (0.0x).
     * SignalReconciler: Resolves multi-channel conflicts and vetoes distribution traps.
     * StockPredictabilityStratifier: Allows ONLY Tier S (1.2x sizing amplification) and Tier A (1.0x)
       high-predictability stocks into the satellite sleeve (capped at max 5% per single stock, T+5
       holding horizon). Intercepts Tier C noisy stocks and safely reroutes them to broad ETFs
       (510900 / 510500 / 510300).
   - Strict Sum-to-1.0 Invariant: Any unallocated satellite weight budget automatically parks in
     Core Treasury Bond ETF (511010) / GC001 cash yield so total portfolio weights strictly sum to 1.0.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any

try:
    from .data_cleaning_filter import DataQualityAuditor
    from .kol_credibility_registry import KOLCredibilityRegistry
    from .signal_reconciler import ChannelSignal, SignalReconciler
    from .stock_predictability_stratifier import (
        EMBEDDED_PREDICTABILITY_TIERS,
        StockPredictabilityStratifier,
    )
except ImportError:
    from data_cleaning_filter import DataQualityAuditor
    from kol_credibility_registry import KOLCredibilityRegistry
    from signal_reconciler import ChannelSignal, SignalReconciler
    from stock_predictability_stratifier import (
        EMBEDDED_PREDICTABILITY_TIERS,
        StockPredictabilityStratifier,
    )

logger = logging.getLogger("core_satellite_advisor")

# Default reference prices for standalone execution and fallback
DEFAULT_STOCK_PRICES: dict[str, float] = {
    "SZ300760": 265.00,  # 迈瑞医疗 (Tier S)
    "SH600036": 36.50,   # 招商银行 (Tier S)
    "SZ000963": 32.80,   # 华东医药 (Tier S)
    "BEKE": 18.50,       # 贝壳找房 (Tier S)
    "SZ300729": 15.20,   # 乐歌股份 (Tier S)
    "SZ002223": 35.60,   # 鱼跃医疗 (Tier S)
    "SZ002142": 24.80,   # 宁波银行 (Tier S)
    "SZ002352": 38.90,   # 顺丰控股 (Tier S)
    "SZ000001": 11.50,   # 平安银行 (Tier S)
    "SZ002027": 6.40,    # 分众传媒 (Tier S)
    "SH601318": 46.20,   # 中国平安 (Tier S)
    "SH600276": 45.80,   # 恒瑞医药 (Tier S)
    "TME": 11.80,        # 腾讯音乐 (Tier S)
    "FUTU": 68.50,       # 富途控股 (Tier S)
    "PEP": 172.00,       # 百事可乐 (Tier S)
    "NVDA": 120.00,      # 英伟达 (Tier A)
    "COST": 880.00,      # 好市多 (Tier A)
    "KO": 68.00,         # 可口可乐 (Tier A)
    "02015": 85.00,      # 理想汽车-W (Tier C - Noisy)
    "09868": 32.00,      # 小鹏汽车-W (Tier C - Noisy)
    "SZ300014": 42.00,   # 亿纬锂能 (Tier C - Noisy)
    "SH601155": 11.20,   # 新城控股 (Tier C - Noisy)
    "SH603288": 38.50,   # 海天味业 (Tier C - Noisy)
}


@dataclass
class SatelliteAlphaTicket:
    """Actionable T+5 sniper trade ticket for Tier-S / Tier-A satellite alpha sleeve."""

    symbol: str
    name: str
    tier: str
    kol_trigger_summary: str
    kol_weighted_sentiment: float
    action: str
    shares_to_buy: int
    entry_price: float
    target_holding_days: int = 5
    stop_loss_pct: float = -0.05
    take_profit_pct: float = 0.08
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SatellitePositionRecord:
    """Persistent tracking record for an open T+5 satellite equity position."""

    symbol: str
    name: str
    shares: int
    entry_price: float
    entry_date: str
    target_holding_days: int = 5
    stop_loss_pct: float = -0.05
    take_profit_pct: float = 0.08
    kol_trigger_summary: str = ""
    current_price: float = 0.0
    current_holding_days: int = 0
    unrealized_pnl_pct: float = 0.0
    lifecycle_status: str = "HOLD_IN_PROGRESS"  # 'HOLD_IN_PROGRESS', 'EXIT_TARGET_HORIZON_REACHED', 'EXIT_TAKE_PROFIT', 'EXIT_STOP_LOSS'
    action: str = "HOLD"  # 'HOLD', 'SELL_EXIT'
    exit_reason: str = ""

    def update_valuation(self, current_price: float, today_date: str) -> None:
        self.current_price = float(current_price)
        if self.entry_price > 0:
            self.unrealized_pnl_pct = (self.current_price - self.entry_price) / self.entry_price
        else:
            self.unrealized_pnl_pct = 0.0

        try:
            from datetime import datetime
            d0 = datetime.strptime(self.entry_date[:10], "%Y-%m-%d").date()
            d1 = datetime.strptime(today_date[:10], "%Y-%m-%d").date()
            self.current_holding_days = max(0, (d1 - d0).days)
        except Exception:
            self.current_holding_days = 0

        # Check exit triggers
        if self.unrealized_pnl_pct >= self.take_profit_pct:
            self.lifecycle_status = "EXIT_TAKE_PROFIT"
            self.action = "SELL_EXIT"
            self.exit_reason = (
                f"达标止盈 (+{self.unrealized_pnl_pct*100:.1f}% >= +{self.take_profit_pct*100:.1f}%)，"
                f"锁定收益，资金全额归集回国债ETF/GC001"
            )
        elif self.unrealized_pnl_pct <= self.stop_loss_pct:
            self.lifecycle_status = "EXIT_STOP_LOSS"
            self.action = "SELL_EXIT"
            self.exit_reason = (
                f"触发硬止损 ({self.unrealized_pnl_pct*100:.1f}% <= {self.stop_loss_pct*100:.1f}%)，"
                f"截断个股尾部风险，资金回流压舱石"
            )
        elif self.current_holding_days >= self.target_holding_days:
            self.lifecycle_status = "EXIT_TARGET_HORIZON_REACHED"
            self.action = "SELL_EXIT"
            self.exit_reason = (
                f"已满 T+{self.target_holding_days} 事件驱动窗口期，正常获利退出，资金归集回国债ETF"
            )
        else:
            self.lifecycle_status = "HOLD_IN_PROGRESS"
            self.action = "HOLD"
            days_left = self.target_holding_days - self.current_holding_days
            self.exit_reason = (
                f"持有中 (已持有 {self.current_holding_days} 天，浮动盈亏 {self.unrealized_pnl_pct*100:+.1f}%，"
                f"距 T+{self.target_holding_days} 还剩 {days_left} 天)"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SatelliteLifecycleManager:
    """Manages persistence and T+5 holding lifecycle for satellite equity positions."""

    def __init__(self, ledger_path: str | Path | None = None):
        import os
        from pathlib import Path
        self.ledger_path = Path(ledger_path) if ledger_path else Path("research/production/satellite_ledger.json")
        self.positions: dict[str, SatellitePositionRecord] = {}
        self.history: list[dict[str, Any]] = []
        self.load()

    def load(self) -> int:
        import json
        if not self.ledger_path.exists():
            return 0
        try:
            with open(self.ledger_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.positions.clear()
            for p in data.get("active_positions", []):
                rec = SatellitePositionRecord(**p)
                self.positions[rec.symbol] = rec
            self.history = data.get("closed_history", [])
            return len(self.positions)
        except Exception as exc:
            logger.warning(f"Failed to load satellite ledger {self.ledger_path}: {exc}")
            return 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "active_positions": [p.to_dict() for p in self.positions.values()],
            "closed_history": list(self.history),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], ledger_path: str | Path | None = None) -> SatelliteLifecycleManager:
        inst = cls(ledger_path=ledger_path)
        inst.positions.clear()
        for p in data.get("active_positions", []):
            rec = SatellitePositionRecord(**p)
            inst.positions[rec.symbol] = rec
        inst.history = list(data.get("closed_history", []))
        return inst

    def save(self) -> None:
        import json
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        data = self.to_dict()
        data["closed_history"] = self.history[-50:]
        with open(self.ledger_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def add_ticket(self, ticket: SatelliteAlphaTicket, entry_date: str) -> SatellitePositionRecord:
        rec = SatellitePositionRecord(
            symbol=ticket.symbol,
            name=ticket.name,
            shares=ticket.shares_to_buy,
            entry_price=ticket.entry_price,
            entry_date=entry_date,
            target_holding_days=ticket.target_holding_days,
            stop_loss_pct=ticket.stop_loss_pct,
            take_profit_pct=ticket.take_profit_pct,
            kol_trigger_summary=ticket.kol_trigger_summary,
            current_price=ticket.entry_price,
            current_holding_days=0,
            unrealized_pnl_pct=0.0,
            lifecycle_status="HOLD_IN_PROGRESS",
            action="HOLD",
            exit_reason=f"今日开仓买入，目标持有 T+{ticket.target_holding_days} 天",
        )
        self.positions[ticket.symbol] = rec
        return rec

    def close_position(self, symbol: str, exit_date: str, exit_price: float, reason: str) -> None:
        if symbol in self.positions:
            pos = self.positions.pop(symbol)
            pnl_pct = (exit_price - pos.entry_price) / pos.entry_price if pos.entry_price > 0 else 0.0
            self.history.append({
                "symbol": pos.symbol,
                "name": pos.name,
                "shares": pos.shares,
                "entry_price": pos.entry_price,
                "entry_date": pos.entry_date,
                "exit_price": round(exit_price, 3),
                "exit_date": exit_date,
                "holding_days": pos.current_holding_days,
                "realized_pnl_pct": round(pnl_pct, 4),
                "reason": reason,
            })

    def audit_positions(
        self,
        current_prices: dict[str, float],
        today_date: str,
    ) -> tuple[list[SatellitePositionRecord], list[SatellitePositionRecord]]:
        active_list: list[SatellitePositionRecord] = []
        exit_list: list[SatellitePositionRecord] = []
        for sym, pos in self.positions.items():
            price = current_prices.get(sym, pos.current_price or pos.entry_price)
            pos.update_valuation(price, today_date)
            if pos.action == "SELL_EXIT":
                exit_list.append(pos)
            else:
                active_list.append(pos)
        return active_list, exit_list


@dataclass
class CoreSatellitePlan:
    """Unified Core-Satellite (80% Core ETFs + 20% Satellite Event Alpha) allocation plan."""

    core_weight_budget: float = 0.80
    satellite_weight_budget: float = 0.20
    core_etf_weights: dict[str, float] = field(default_factory=dict)
    satellite_stock_weights: dict[str, float] = field(default_factory=dict)
    satellite_tickets: list[SatelliteAlphaTicket] = field(default_factory=list)
    blocked_noisy_stocks: list[dict[str, Any]] = field(default_factory=list)
    active_positions: list[SatellitePositionRecord] = field(default_factory=list)
    exit_tickets: list[SatellitePositionRecord] = field(default_factory=list)

    @property
    def total_weight_sum(self) -> float:
        return round(
            sum(self.core_etf_weights.values()) + sum(self.satellite_stock_weights.values()),
            6,
        )

    @property
    def combined_weights(self) -> dict[str, float]:
        return {**self.core_etf_weights, **self.satellite_stock_weights}

    @property
    def unused_satellite_budget(self) -> float:
        return round(
            max(0.0, self.satellite_weight_budget - sum(self.satellite_stock_weights.values())),
            6,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "core_weight_budget": self.core_weight_budget,
            "satellite_weight_budget": self.satellite_weight_budget,
            "core_etf_weights": dict(self.core_etf_weights),
            "satellite_stock_weights": dict(self.satellite_stock_weights),
            "satellite_tickets": [t.to_dict() for t in self.satellite_tickets],
            "blocked_noisy_stocks": list(self.blocked_noisy_stocks),
            "active_positions": [p.to_dict() for p in self.active_positions],
            "exit_tickets": [p.to_dict() for p in self.exit_tickets],
            "total_weight_sum": self.total_weight_sum,
            "unused_satellite_budget": self.unused_satellite_budget,
        }

    def to_markdown(self) -> str:
        """Render the Core-Satellite cockpit section for terminal and Feishu/WeCom cards."""
        lines: list[str] = []
        lines.append("### 🎯 卫星增强仓 (20% Tier-S 个股大V事件 Alpha 狙击单 - T+5 策略)")
        used_sat_pct = sum(self.satellite_stock_weights.values()) * 100.0
        unused_sat_pct = self.unused_satellite_budget * 100.0
        lines.append(
            f"- **架构预算分配**: 核心全天候 ETF 仓位 **{self.core_weight_budget*100:.0f}%** | "
            f"卫星事件 Alpha 预算 **{self.satellite_weight_budget*100:.0f}%** "
            f"(当前激活个股 **{used_sat_pct:.1f}%**，未用预算 **{unused_sat_pct:.1f}%** 自动停泊于 `511010` 国债ETF/GC001)"
        )
        lines.append("")

        # 1. Exit sell tickets
        if self.exit_tickets:
            lines.append("#### ⏰ T+5 到期平仓 / 止盈止损卖出单 (资金归集回国债 ETF / GC001)")
            lines.append("| 代码 | 标的名称 | 买入日期 | 持仓天数 | 成本价 | 最新价 | 浮动盈亏 | 操作指令 | 退出平仓理由 |")
            lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
            for ex in self.exit_tickets:
                lines.append(
                    f"| `{ex.symbol}` | **{ex.name}** | {ex.entry_date} | {ex.current_holding_days} 天 | "
                    f"{ex.entry_price:.2f} | {ex.current_price:.2f} | **{ex.unrealized_pnl_pct*100:+.2f}%** | "
                    f"🔴 **全部卖出平仓 ({ex.shares:,}股)** | {ex.exit_reason} |"
                )
            lines.append("")

        # 2. Active holding positions
        if self.active_positions:
            lines.append("#### 📋 当前在持卫星仓生命周期跟踪 (T+5 观察中)")
            lines.append("| 代码 | 标的名称 | 买入日期 | 持仓天数 | 成本价 | 最新价 | 浮动盈亏 | 状态 | 目标平仓倒计时 |")
            lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
            for ap in self.active_positions:
                days_left = max(0, ap.target_holding_days - ap.current_holding_days)
                lines.append(
                    f"| `{ap.symbol}` | {ap.name} | {ap.entry_date} | {ap.current_holding_days} 天 | "
                    f"{ap.entry_price:.2f} | {ap.current_price:.2f} | {ap.unrealized_pnl_pct*100:+.2f}% | "
                    f"🟢 持仓正常 | 还剩 **{days_left}** 个交易日 |"
                )
            lines.append("")

        # 3. New entry tickets
        if self.satellite_tickets:
            lines.append("#### 🚀 今日新增激活 Tier-S / Tier-A 事件 Alpha 狙击单 (单票上限 5% | T+5 持有期)")
            lines.append("| 代码 | 标的名称 | 可预测性分层 | 触发大V / 逆向指标共振信号 | 加权情绪 | 建议买入 | 入场价 | 权重 | 止盈 / 止损 | 策略逻辑 |")
            lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
            for t in self.satellite_tickets:
                w_pct = self.satellite_stock_weights.get(t.symbol, 0.0) * 100.0
                tier_badge = "🔥 Tier-S (1.2x放大)" if "TIER_S" in t.tier else "✅ Tier-A (1.0x标准)"
                lines.append(
                    f"| `{t.symbol}` | **{t.name}** | {tier_badge} | {t.kol_trigger_summary} | "
                    f"`{t.kol_weighted_sentiment:+.2f}` | **{t.shares_to_buy:,}股** | {t.entry_price:.2f} | "
                    f"**{w_pct:.2f}%** | +{t.take_profit_pct*100:.0f}% / {t.stop_loss_pct*100:.0f}% | "
                    f"T+{t.target_holding_days}持有: {t.rationale} |"
                )
            lines.append("")
        else:
            lines.append("> ℹ️ **今日无新触发的 Tier-S/A 个股建仓信号，未用卫星预算自动停泊于 `511010` 国债ETF 享受无风险收益。**")
            lines.append("")

        if self.blocked_noisy_stocks:
            lines.append("#### 🛡️ 股票可预测性分层门控：已拦截 Tier-C 高噪音标的 (自动路由至宽基 ETF)")
            lines.append("| 拦截代码 | 标的名称 | 识别分层 | 原始热度信号 | 自动替代宽基 ETF | 门控拦截与替代理由 |")
            lines.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
            for b in self.blocked_noisy_stocks:
                sym = b.get("symbol", "")
                name = b.get("name", sym)
                raw_s = b.get("raw_signal", 0.0)
                etf_sub = b.get("etf_substitute", "510900")
                reason = b.get("explanation") or b.get("reason", "")
                lines.append(
                    f"| `{sym}` | {name} | ⛔ `Tier-C 随机游走噪音` | `{raw_s:+.2f}` | "
                    f"**`{etf_sub}`** | {reason} |"
                )
            lines.append("")

        return "\n".join(lines)


def get_default_candidate_stock_signals() -> list[dict[str, Any]]:
    """Return realistic daily point-in-time social/KOL event signals for live advisory."""
    return [
        {
            "symbol": "SZ300760",
            "name": "迈瑞医疗",
            "is_substantive_event": True,
            "kol_trigger_summary": "头部大V[阿尔法工场]看多 + 反向明灯[钟华守正出奇]恐慌割肉倒置",
            "posts": [
                {
                    "author": "阿尔法工场",
                    "text": "迈瑞医疗当前PE估值仅22倍处于近10年15%分位数，海外高端医疗器械订单超预期增长25%，安全边际极高，建议重仓买入配置。",
                    "polarity": 0.90,
                    "verified": True,
                },
                {
                    "author": "钟华守正出奇",
                    "text": "医疗板块彻底没戏了，迈瑞医疗跌破支撑位，赶紧清仓割肉止损！",
                    "polarity": -0.85,
                    "verified": True,
                },
            ],
        },
        {
            "symbol": "SH600036",
            "name": "招商银行",
            "is_substantive_event": True,
            "kol_trigger_summary": "头部大V[雪球调研团]基本面看多 + 反向明灯[朱酒]看空倒置",
            "posts": [
                {
                    "author": "雪球调研团",
                    "text": "招商银行当前股息率高达5.4%，PB仅0.88倍，资产质量与ROE稳居行业第一，主力资金连续5日净流入18亿元，具备极强防御与分红价值，建议买入。",
                    "polarity": 0.88,
                    "verified": True,
                },
                {
                    "author": "朱酒",
                    "text": "银行股净息差还要跌，招商银行反弹就是最后逃命机会，坚决卖出清仓！",
                    "polarity": -0.80,
                    "verified": True,
                },
            ],
        },
        {
            "symbol": "SZ000963",
            "name": "华东医药",
            "is_substantive_event": True,
            "kol_trigger_summary": "高胜率投研大V[价投傻鱼]看多 + 反向明灯[青侨阳光]看空倒置",
            "posts": [
                {
                    "author": "价投傻鱼",
                    "text": "华东医药医美与创新药管线双轮驱动，三季报业绩超预期增长18.5%，当前估值仅15倍PE，迎来确定性业绩拐点，建议建仓买入。",
                    "polarity": 0.85,
                    "verified": True,
                },
                {
                    "author": "青侨阳光",
                    "text": "医药股还要阴跌，华东医药赶紧止损回避！",
                    "polarity": -0.78,
                    "verified": True,
                },
            ],
        },
        {
            "symbol": "BEKE",
            "name": "贝壳",
            "is_substantive_event": True,
            "kol_trigger_summary": "海外高胜率大V[THE_TRADE]看多 + 反向明灯[JimCramer]看空倒置",
            "posts": [
                {
                    "author": "THE_TRADE",
                    "text": "BEKE 贝壳现金储备充裕，伴随一二线城市二手房成交量回暖超预期增长30%，回购力度持续加大，PE估值处于历史底部，强烈看好买入。",
                    "polarity": 0.86,
                    "verified": True,
                },
                {
                    "author": "JimCramer",
                    "text": "Avoid Chinese real estate stocks like BEKE at all costs, sell now!",
                    "polarity": -0.85,
                    "verified": True,
                },
            ],
        },
        # Tier C Noisy Stocks (Retail FOMO chasing -> should be intercepted and rerouted to 510900 / 510500)
        {
            "symbol": "02015",
            "name": "理想汽车-W",
            "raw_signal": 0.78,
            "kol_trigger_summary": "散户社区FOMO追涨 (Tier-C高噪音随机游走标的)",
            "posts": [
                {
                    "author": "新能源散户先锋",
                    "text": "理想汽车周销量暴涨，即将突破历史新高，满仓买入冲冲冲！",
                    "polarity": 0.78,
                    "verified": False,
                }
            ],
        },
        {
            "symbol": "09868",
            "name": "小鹏汽车-W",
            "raw_signal": 0.74,
            "kol_trigger_summary": "散户社区FOMO追涨 (Tier-C高噪音随机游走标的)",
            "posts": [
                {
                    "author": "电车大队长",
                    "text": "小鹏汽车智驾概念大热，赶紧加仓买入！",
                    "polarity": 0.74,
                    "verified": False,
                }
            ],
        },
        {
            "symbol": "SZ300014",
            "name": "亿纬锂能",
            "raw_signal": 0.68,
            "kol_trigger_summary": "散户概念题材炒作 (Tier-C高噪音随机游走标的)",
            "posts": [
                {
                    "author": "锂电散户",
                    "text": "固态电池概念爆发，亿纬锂能估值低位，马上反弹建仓买入！",
                    "polarity": 0.68,
                    "verified": False,
                }
            ],
        },
    ]


def _normalize_candidate_list(
    candidate_stock_signals: list[dict[str, Any]] | dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Convert flexible candidate signal inputs into a standardized list of dicts."""
    if candidate_stock_signals is None:
        return get_default_candidate_stock_signals()
    if isinstance(candidate_stock_signals, dict):
        normalized: list[dict[str, Any]] = []
        for sym, val in candidate_stock_signals.items():
            if isinstance(val, dict):
                normalized.append({"symbol": str(sym).strip().upper(), **val})
            else:
                normalized.append({"symbol": str(sym).strip().upper(), "raw_signal": float(val)})
        return normalized
    if isinstance(candidate_stock_signals, list):
        out: list[dict[str, Any]] = []
        for item in candidate_stock_signals:
            if isinstance(item, dict):
                sym = str(item.get("symbol") or item.get("code") or "").strip().upper()
                if sym:
                    out.append({**item, "symbol": sym})
        return out
    return []


def generate_core_satellite_plan(
    total_capital: float,
    core_base_weights: dict[str, float],
    candidate_stock_signals: list[dict[str, Any]] | dict[str, Any] | None = None,
    market_prices: dict[str, float] | None = None,
    core_weight_budget: float = 0.80,
    satellite_weight_budget: float = 0.20,
    max_single_stock_weight: float = 0.05,
    min_sentiment_threshold: float = 0.20,
    lifecycle_mgr: SatelliteLifecycleManager | None = None,
    today_date: str | None = None,
) -> CoreSatellitePlan:
    """Evaluate stock signals via all 4 intelligence modules and build an 80/20 Core-Satellite plan.

    Args:
        total_capital: Total portfolio NAV / capital in RMB.
        core_base_weights: Base weights for Core All-Weather ETFs (will be normalized and scaled to 80%).
        candidate_stock_signals: List or dict of candidate stock signals with social posts / KOL triggers.
        market_prices: Current market prices for stocks and ETFs.
        core_weight_budget: Target weight budget for Core ETF sleeve (default 0.80).
        satellite_weight_budget: Target weight budget for Satellite stock alpha sleeve (default 0.20).
        max_single_stock_weight: Concentration cap per single satellite stock (default 0.05 = 5%).
        min_sentiment_threshold: Minimum KOL-weighted sentiment required to trigger a satellite buy.

    Returns:
        CoreSatellitePlan with strict sum-to-1.0 portfolio weights, T+5 tickets, and blocked Tier-C records.
    """
    auditor = DataQualityAuditor()
    kol_registry = KOLCredibilityRegistry()
    reconciler = SignalReconciler()
    stratifier = StockPredictabilityStratifier()

    prices = {**DEFAULT_STOCK_PRICES, **(market_prices or {})}
    candidates = _normalize_candidate_list(candidate_stock_signals)

    blocked_noisy_stocks: list[dict[str, Any]] = []
    qualified_candidates: list[dict[str, Any]] = []

    for cand in candidates:
        symbol = cand["symbol"]
        prof = stratifier.profiles.get(symbol)
        stock_name = cand.get("name") or (prof.name if prof else symbol)

        # ---------------------------------------------------------------------
        # 1. DataQualityAuditor & KOLCredibilityRegistry Evaluation
        # ---------------------------------------------------------------------
        posts = cand.get("posts", [])
        kol_trigger_summary = str(cand.get("kol_trigger_summary", "")).strip()

        if posts and isinstance(posts, list):
            valid_posts_for_kol: list[dict[str, Any]] = []
            elite_authors: list[str] = []
            contrarian_authors: list[str] = []

            for p in posts:
                if isinstance(p, str):
                    p_text, p_author, p_pol, p_ver = p, "anonymous", None, False
                else:
                    p_text = str(p.get("text", ""))
                    p_author = str(p.get("author", "anonymous")).strip()
                    p_pol = p.get("polarity")
                    p_ver = bool(p.get("verified", True))

                if p_text:
                    audit_res = auditor.audit_text(p_text, author_is_verified=p_ver)
                    if not audit_res.is_valid and audit_res.rejection_reason in ("SPAM_SOLICITATION", "CLICKBAIT_RUMOR"):
                        continue
                    effective_pol = float(p_pol) if p_pol is not None else audit_res.sentiment_polarity
                    q_score = float(p.get("quality_score", audit_res.quality_score))
                else:
                    effective_pol = float(p_pol if p_pol is not None else 0.0)
                    q_score = float(p.get("quality_score", 1.0))

                author_prof = kol_registry.get_author_profile(p_author)
                if author_prof.tier in ("TIER_0_ELITE_KOL", "TIER_1_CORE_ALPHA", "TIER_2_SOLID_RESEARCHER") and effective_pol > 0:
                    elite_authors.append(p_author)
                elif author_prof.is_contrarian and effective_pol < 0:
                    contrarian_authors.append(p_author)

                valid_posts_for_kol.append({
                    "author": p_author,
                    "polarity": effective_pol,
                    "quality_score": q_score,
                })

            if valid_posts_for_kol:
                kol_res = kol_registry.evaluate_weighted_sentiment(valid_posts_for_kol)
                kol_weighted_sentiment = kol_res.kol_weighted_polarity
                raw_signal = float(cand.get("raw_signal", kol_res.raw_unweighted_polarity))
                if not kol_trigger_summary:
                    parts = []
                    if elite_authors:
                        parts.append(f"头部大V[{','.join(elite_authors[:2])}]看多")
                    if contrarian_authors:
                        parts.append(f"反向明灯[{','.join(contrarian_authors[:2])}]恐慌割肉倒置")
                    kol_trigger_summary = " + ".join(parts) if parts else kol_res.summary_explanation
            else:
                kol_weighted_sentiment = 0.0
                raw_signal = float(cand.get("raw_signal", 0.0))
        elif "kol_weighted_sentiment" in cand:
            kol_weighted_sentiment = float(cand["kol_weighted_sentiment"])
            raw_signal = float(cand.get("raw_signal", kol_weighted_sentiment))
        elif "author" in cand or "kol_author" in cand:
            author_name = str(cand.get("author") or cand.get("kol_author")).strip()
            raw_signal = float(cand.get("raw_signal") or cand.get("polarity") or cand.get("signal") or 0.8)
            kol_res = kol_registry.evaluate_weighted_sentiment(
                [{"author": author_name, "polarity": raw_signal, "quality_score": 1.0}]
            )
            kol_weighted_sentiment = kol_res.kol_weighted_polarity
            if not kol_trigger_summary:
                kol_trigger_summary = f"大V[{author_name}]信号触发 ({kol_res.summary_explanation})"
        else:
            raw_signal = float(cand.get("raw_signal") or cand.get("signal") or cand.get("score") or 0.0)
            kol_weighted_sentiment = raw_signal

        if not kol_trigger_summary:
            kol_trigger_summary = f"大V信誉加权情绪 ({kol_weighted_sentiment:+.2f}) 共振触发"

        # ---------------------------------------------------------------------
        # 2. StockPredictabilityStratifier Evaluation (Tier C Interception)
        # ---------------------------------------------------------------------
        is_substantive = bool(cand.get("is_substantive_event", True))
        strat_res = stratifier.evaluate_stock_signal(
            symbol=symbol,
            raw_signal=kol_weighted_sentiment,
            is_substantive_event=is_substantive,
        )
        tier = strat_res["tier"]

        if tier == "TIER_C_LOW_NOISY_RANDOM_WALK" or strat_res["action"] == "REJECT_NOISY_STOCK_USE_ETF":
            # Determine ETF substitute (510900 for HK/ADR like 02015/09868, 510500 for ChiNext/Mid-cap like SZ300014)
            etf_sub = strat_res.get("recommended_instrument") or (prof.etf_substitute if prof else "510500")
            if symbol in EMBEDDED_PREDICTABILITY_TIERS:
                etf_sub = EMBEDDED_PREDICTABILITY_TIERS[symbol].get("etf_sub", etf_sub)
            elif symbol in ("02015", "09868"):
                etf_sub = "510900"
            elif symbol.startswith("SZ300"):
                etf_sub = "510500"
            win_r = prof.win_rate_5d if prof else 0.48
            r_ic = prof.rank_ic if prof else 0.002
            explanation = (
                f"[{stock_name}] 属于 Tier C 低可预测性噪音标的 (Rank IC={r_ic:+.3f}, 5D胜率={win_r*100:.1f}%)，"
                f"禁止个股追涨杀跌，已自动替换为宽基 ETF ({etf_sub})。"
            )
            blocked_noisy_stocks.append({
                "symbol": symbol,
                "name": stock_name,
                "tier": tier,
                "raw_signal": round(raw_signal, 4),
                "kol_weighted_sentiment": round(kol_weighted_sentiment, 4),
                "action": "REJECT_NOISY_STOCK_USE_ETF",
                "etf_substitute": etf_sub,
                "recommended_instrument": etf_sub,
                "kol_trigger_summary": kol_trigger_summary,
                "reason": explanation,
                "explanation": explanation,
            })
            continue

        # Only Tier S and Tier A stocks are eligible for the 20% Satellite sleeve
        if tier not in ("TIER_S_HIGH_PREDICTABILITY", "TIER_A_SOLID_PREDICTABILITY"):
            continue

        # ---------------------------------------------------------------------
        # 3. SignalReconciler Multi-Channel Check
        # ---------------------------------------------------------------------
        raw_channels = cand.get("channel_signals")
        if raw_channels and isinstance(raw_channels, list):
            ch_signals = [
                c if isinstance(c, ChannelSignal) else ChannelSignal(**c)
                for c in raw_channels
            ]
        else:
            ch_signals = [
                ChannelSignal("SMART_MONEY_FLOW", score=float(cand.get("smart_money_flow", kol_weighted_sentiment)), confidence=0.9),
                ChannelSignal("FUNDAMENTAL_VALUATION", score=float(cand.get("fundamental_valuation", kol_weighted_sentiment)), confidence=0.9),
            ]
        rec_res = reconciler.reconcile_asset_signals(symbol, ch_signals)
        if rec_res.conflict_type in ("CONFLICT_VETO_OVERRIDE", "CONFLICT_DISTRIBUTION_TRAP") and rec_res.final_score < 0:
            continue

        # Require strong positive KOL-weighted conviction
        if kol_weighted_sentiment < min_sentiment_threshold:
            continue

        # ---------------------------------------------------------------------
        # 4. Sizing with Tier S (+20% Amplification) vs Tier A (1.0x Standard)
        # ---------------------------------------------------------------------
        alloc_mult = prof.allocation_multiplier if prof else (1.20 if tier == "TIER_S_HIGH_PREDICTABILITY" else 1.00)
        if "target_weight" in cand:
            raw_w = float(cand["target_weight"])
            if cand.get("apply_tier_multiplier", False):
                raw_w *= alloc_mult
        else:
            # Base slot weight is max_single_stock_weight / 1.20 (~4.1667%) so:
            # - Tier S (1.20x multiplier) amplifies to 5.00% (hitting the exact 5% single-stock cap)
            # - Tier A (1.00x multiplier) receives 4.17% standard sizing
            base_slot_w = float(cand.get("base_weight", max_single_stock_weight / 1.20))
            raw_w = base_slot_w * alloc_mult

        desired_weight = min(max_single_stock_weight, round(raw_w, 6))
        pred_score = prof.predictability_score if prof else (85.0 if tier == "TIER_S_HIGH_PREDICTABILITY" else 55.0)
        win_rate = prof.win_rate_5d if prof else 0.56
        rank_ic = prof.rank_ic if prof else 0.11

        qualified_candidates.append({
            "symbol": symbol,
            "name": stock_name,
            "tier": tier,
            "alloc_mult": alloc_mult,
            "desired_weight": desired_weight,
            "kol_weighted_sentiment": kol_weighted_sentiment,
            "kol_trigger_summary": kol_trigger_summary,
            "priority_score": pred_score * kol_weighted_sentiment,
            "win_rate": win_rate,
            "rank_ic": rank_ic,
            "cand": cand,
        })

    # Sort qualified candidates by priority score descending
    qualified_candidates.sort(key=lambda x: (x["priority_score"], x["desired_weight"]), reverse=True)

    # Allocate up to satellite_weight_budget (default 0.20)
    remaining_sat = round(satellite_weight_budget, 6)
    satellite_stock_weights: dict[str, float] = {}
    satellite_tickets: list[SatelliteAlphaTicket] = []

    for q in qualified_candidates:
        if remaining_sat <= 1e-6:
            break
        sym = q["symbol"]
        alloc_w = round(min(q["desired_weight"], remaining_sat), 6)
        if alloc_w <= 1e-6:
            continue

        remaining_sat = round(remaining_sat - alloc_w, 6)
        satellite_stock_weights[sym] = alloc_w

        entry_price = float(prices.get(sym, DEFAULT_STOCK_PRICES.get(sym, 50.0)))
        if entry_price <= 0:
            entry_price = 50.0

        target_amount = total_capital * alloc_w
        raw_shares = target_amount / entry_price

        # Round to 100-share board lots for A-shares where applicable, or exact integer shares
        is_ashare = sym.startswith(("SH", "SZ")) or (len(sym) == 6 and sym.isdigit())
        if is_ashare:
            lot_shares = int(round(raw_shares / 100.0)) * 100
            if lot_shares >= 100 and (lot_shares * entry_price) <= total_capital * (max_single_stock_weight + 0.005):
                shares_to_buy = lot_shares
            elif int(raw_shares // 100) * 100 >= 100:
                shares_to_buy = int(raw_shares // 100) * 100
            else:
                shares_to_buy = max(1, int(round(raw_shares)))
        else:
            shares_to_buy = max(1, int(round(raw_shares)))

        cand_dict = q["cand"]
        hold_days = int(cand_dict.get("target_holding_days", 5))
        stop_loss = float(cand_dict.get("stop_loss_pct", -0.05))
        take_profit = float(cand_dict.get("take_profit_pct", 0.08))

        rationale = (
            cand_dict.get("rationale")
            or f"{q['tier']} 高确定性标的 (历史5D胜率 {q['win_rate']*100:.1f}%, Rank IC {q['rank_ic']:+.3f})，"
               f"享受 {q['alloc_mult']:.2f}x 权重系数放大，目标持有 T+{hold_days} 交易日"
        )

        ticket = SatelliteAlphaTicket(
            symbol=sym,
            name=q["name"],
            tier=q["tier"],
            kol_trigger_summary=q["kol_trigger_summary"],
            kol_weighted_sentiment=round(q["kol_weighted_sentiment"], 4),
            action=str(cand_dict.get("action", "BUY")),
            shares_to_buy=int(shares_to_buy),
            entry_price=round(entry_price, 3),
            target_holding_days=hold_days,
            stop_loss_pct=stop_loss,
            take_profit_pct=take_profit,
            rationale=rationale,
        )
        satellite_tickets.append(ticket)

    # -------------------------------------------------------------------------
    # 5. Scale Core ETF Weights to 80% + Fallback Unused Satellite to 511010
    # -------------------------------------------------------------------------
    total_sat_used = round(sum(satellite_stock_weights.values()), 6)
    unused_sat_budget = round(max(0.0, satellite_weight_budget - total_sat_used), 6)

    if not core_base_weights or sum(core_base_weights.values()) <= 0:
        norm_core = {"511010": 1.0}
    else:
        base_sum = sum(core_base_weights.values())
        norm_core = {k: v / base_sum for k, v in core_base_weights.items()}

    core_etf_weights: dict[str, float] = {
        k: round(v * core_weight_budget, 6) for k, v in norm_core.items()
    }

    # Add unused satellite budget to Core Bond ETF 511010
    core_etf_weights["511010"] = round(
        core_etf_weights.get("511010", 0.0) + unused_sat_budget,
        6,
    )

    # Enforce strict sum-to-1.0 invariant across all weights
    current_total = round(
        sum(core_etf_weights.values()) + sum(satellite_stock_weights.values()),
        6,
    )
    residual = round(1.0 - current_total, 6)
    if abs(residual) > 0:
        core_etf_weights["511010"] = round(core_etf_weights.get("511010", 0.0) + residual, 6)

    active_positions: list[SatellitePositionRecord] = []
    exit_tickets: list[SatellitePositionRecord] = []
    if lifecycle_mgr is not None:
        t_date = today_date or "2026-09-16"
        active_positions, exit_tickets = lifecycle_mgr.audit_positions(prices, t_date)

    return CoreSatellitePlan(
        core_weight_budget=core_weight_budget,
        satellite_weight_budget=satellite_weight_budget,
        core_etf_weights=core_etf_weights,
        satellite_stock_weights=satellite_stock_weights,
        satellite_tickets=satellite_tickets,
        blocked_noisy_stocks=blocked_noisy_stocks,
        active_positions=active_positions,
        exit_tickets=exit_tickets,
    )


if __name__ == "__main__":
    sample_core_base = {
        "510300": 0.15,
        "510880": 0.15,
        "511010": 0.40,
        "518880": 0.15,
        "513500": 0.075,
        "513100": 0.075,
    }
    plan = generate_core_satellite_plan(
        total_capital=200000.0,
        core_base_weights=sample_core_base,
    )
    print(plan.to_markdown())
    print(f"Total Portfolio Weight Sum: {plan.total_weight_sum:.6f}")
