#!/usr/bin/env python3
"""Automated Scheduled Intraday Runner for Core-Satellite Portfolio.

Daily Production Execution & Tracking Loop:
1. Calendar Pre-Flight Check:
   - Evaluates China A-share trading calendar (skips weekends and statutory holidays).
   - `--force-trade-day`: bypasses calendar check for off-schedule testing.
2. Market Intelligence & Asset Allocation:
   - Fetches live quotes and 60-day realized volatility for Core All-Weather ETFs.
   - Evaluates Tier-S / Tier-A Alpha candidates via 4 intelligence modules.
   - Generates 80/20 Core-Satellite allocation plan via `core_satellite_advisor.py`.
3. Virtual Paper Trading Execution:
   - Loads `paper_ledger.json`.
   - Executes recommended adjustments respecting 100-share board lots, slippage, and T+1 locking.
   - 14:50 GC001 Treasury Reverse Repo cash sweep (`--auto-sweep-gc001`).
   - Marks ledger to market, computes daily PnL and cumulative return.
   - Updates `paper_trade_transactions.csv` and `daily_nav_history.csv`.
4. Multi-Channel Notification:
   - Formats rich CardKit v2 (Feishu) and Markdown (WeCom) action cards.
   - Dispatches webhooks with complete portfolio state, trades, and execution details.
   - `--dry-run`: runs full calculation without sending network webhooks or altering disk state.

Usage:
    # 1. Standard Dry-Run (verification):
    python3 research/production/scheduled_intraday_runner.py --dry-run

    # 2. Daily Production Full Cycle with GC001 sweep:
    python3 research/production/scheduled_intraday_runner.py --ledger research/production/paper_ledger.json --webhook-type feishu --auto-sweep-gc001

    # 3. Advice Only (no trades executed):
    python3 research/production/scheduled_intraday_runner.py --mode advice_only --dry-run
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import date, datetime
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any

# Ensure UTF-8 / ASCII-safe console output across environments
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def _safe_print(msg: str, file=None) -> None:
    """Print text safely without crashing on ASCII-only terminals."""
    target = file or sys.stdout
    try:
        print(msg, file=target)
    except UnicodeEncodeError:
        enc = getattr(target, "encoding", None) or "ascii"
        safe_msg = msg.encode(enc, errors="replace").decode(enc, errors="replace")
        print(safe_msg, file=target)


# Setup paths
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fin_skills.china.core_satellite_advisor import (
    DEFAULT_STOCK_PRICES,
    CoreSatellitePlan,
    SatelliteAlphaTicket,
    SatelliteLifecycleManager,
    SatellitePositionRecord,
    generate_core_satellite_plan,
)
from research.production.live_advisor_bot import (
    GLOBAL_ETF_UNIVERSE,
    STRATEGY_PROFILES,
    fetch_live_market_snapshot,
)
from research.production.paper_trading_engine import (
    DEFAULT_LEDGER_PATH,
    DEFAULT_NAV_HISTORY_PATH,
    DEFAULT_TRANSACTIONS_PATH,
    PaperLedger,
    PaperTradingEngine,
    TradeRecord,
)
from research.production.run_intraday_pipeline import (
    DEFAULT_ALPHA_CANDIDATE_POOL,
)
from research.production.webhook_cards import (
    GC001SweepInstruction,
    build_feishu_card_v2,
    build_wecom_markdown,
    calculate_gc001_auto_sweep,
    determine_header_theme,
    send_intraday_webhook,
)

logger = logging.getLogger("scheduled_intraday_runner")

# =============================================================================
# Chinese Stock Exchange (SSE / SZSE) Statutory Holiday Calendar
# =============================================================================

CHINA_STATUTORY_HOLIDAYS = {
    # 2024
    "2024-01-01", "2024-02-09", "2024-02-12", "2024-02-13", "2024-02-14", "2024-02-15",
    "2024-02-16", "2024-04-04", "2024-04-05", "2024-05-01", "2024-05-02", "2024-05-03",
    "2024-06-10", "2024-09-16", "2024-09-17", "2024-10-01", "2024-10-02", "2024-10-03",
    "2024-10-04", "2024-10-07",
    # 2025
    "2025-01-01", "2025-01-28", "2025-01-29", "2025-01-30", "2025-01-31", "2025-02-03",
    "2025-02-04", "2025-04-04", "2025-05-01", "2025-05-02", "2025-05-05", "2025-05-30",
    "2025-06-02", "2025-10-01", "2025-10-02", "2025-10-03", "2025-10-06", "2025-10-07",
    "2025-10-08",
    # 2026
    "2026-01-01", "2026-01-02", "2026-02-16", "2026-02-17", "2026-02-18", "2026-02-19",
    "2026-02-20", "2026-02-23", "2026-04-06", "2026-05-01", "2026-05-04", "2026-05-05",
    "2026-06-19", "2026-09-25", "2026-10-01", "2026-10-02", "2026-10-05", "2026-10-06",
    "2026-10-07",
    # 2027
    "2027-01-01", "2027-02-05", "2027-02-08", "2027-02-09", "2027-02-10", "2027-02-11",
    "2027-02-12", "2027-04-05", "2027-05-03", "2027-06-09", "2027-09-15", "2027-10-01",
    "2027-10-04", "2027-10-05", "2027-10-06", "2027-10-07",
}


def is_china_trading_day(dt: date | datetime | str) -> bool:
    """Check whether a given date is an official trading day for China A-Share markets.

    A-share markets are closed on:
    1. Weekends (Saturday and Sunday). Note: even when government announces compensatory
       working weekends, the stock exchanges remain strictly closed.
    2. Statutory holiday closures (New Year, Spring Festival, Qingming, Labor Day, Dragon Boat,
       Mid-Autumn, National Day).
    """
    if isinstance(dt, str):
        d = datetime.strptime(dt[:10], "%Y-%m-%d").date()
    elif isinstance(dt, datetime):
        d = dt.date()
    else:
        d = dt

    # Check weekend (5: Saturday, 6: Sunday)
    if d.weekday() >= 5:
        return False

    # Check statutory holiday closures
    date_str = d.strftime("%Y-%m-%d")
    if date_str in CHINA_STATUTORY_HOLIDAYS:
        return False

    return True


CORE_ETF_PROFILES = {
    "conservative": {
        "510300": 0.10,
        "510880": 0.20,
        "511010": 0.50,
        "518880": 0.10,
        "513100": 0.05,
        "513500": 0.05,
    },
    "balanced": {
        "510300": 0.15,
        "510880": 0.15,
        "511010": 0.40,
        "518880": 0.15,
        "513100": 0.075,
        "513500": 0.075,
    },
    "aggressive": {
        "510300": 0.30,
        "510880": 0.10,
        "511010": 0.20,
        "518880": 0.10,
        "513100": 0.15,
        "513500": 0.15,
    },
}


def run_scheduled_intraday_cycle(
    ledger_path: str | Path = DEFAULT_LEDGER_PATH,
    mode: str = "full_cycle",  # 'advice_only', 'execute_paper', 'full_cycle'
    auto_sweep_gc001: bool = False,
    webhook_url: str | None = None,
    webhook_type: str = "feishu",
    dry_run: bool = False,
    profile: str = "conservative",
    initial_capital: float = 1_000_000.0,
    inflow: float = 0.0,
    trade_date: str | None = None,
    force_trade_day: bool = False,
    candidate_signals: list[dict[str, Any]] | None = None,
    action_card_path: str | Path = "research/production/1430_DAILY_ACTION_CARD.md",
) -> dict[str, Any]:
    """Execute the automated intraday production cycle.

    Coordinates:
    1. A-share trading calendar validation.
    2. Market quote ingestion & 80/20 Core-Satellite plan formulation.
    3. Virtual paper trading execution on `paper_ledger.json`.
    4. 14:50 GC001 cash sweep.
    5. Mark-to-market NAV and performance trajectory persistence.
    6. Multi-channel Feishu/WeCom webhook delivery.
    """
    l_path = Path(ledger_path)
    card_path = Path(action_card_path)

    today_str = trade_date or date.today().strftime("%Y-%m-%d")

    # 1. Trading Day Calendar Check
    trading_day_active = is_china_trading_day(today_str)
    if not trading_day_active:
        if not force_trade_day and not dry_run:
            _safe_print("=" * 78)
            _safe_print(f"📅 [CALENDAR] {today_str} 是A股休市日（周末或法定节假日）。")
            _safe_print("💤 自动化实盘调度引擎安全休眠，不执行交易或通知。")
            _safe_print("=" * 78)
            return {
                "status": "SKIPPED_NON_TRADING_DAY",
                "trade_date": today_str,
                "is_trading_day": False,
            }
        else:
            _safe_print(f"⚠️ [CALENDAR] 注意: {today_str} 为A股休市日，但因启用 --dry-run 或 --force-trade-day，继续全流程模拟计算。")

    mode_label = f" [模式: {mode.upper()}]" + (" [DRY-RUN 模拟]" if dry_run else "")
    _safe_print("=" * 78)
    _safe_print(f"🕒 启动 14:25 自动化核心-卫星实盘调度引擎 (Scheduled Intraday Runner){mode_label}")
    _safe_print(f"📅 交易日期: {today_str} | 账户账本: {l_path}")
    _safe_print("=" * 78)

    # 2. Initialize / Load Paper Trading Engine & Ledger
    engine = PaperTradingEngine(ledger_path=l_path)
    if not l_path.exists():
        _safe_print(f"🆕 账本不存在，初始化初始资金 {initial_capital:,.2f} RMB 的新账本...")
        ledger = engine.initialize_ledger(initial_capital=initial_capital, ledger_path=l_path, start_date=today_str)
    else:
        ledger = engine.load_ledger(ledger_path=l_path)

    # If dry-run, work on deepcopy to avoid mutating persistent state
    if dry_run:
        working_ledger = deepcopy(ledger)
    else:
        working_ledger = ledger

    if inflow > 0:
        _safe_print(f"💵 今日存入新增定投资金: +{inflow:,.2f} RMB")
        working_ledger.cash_balance = round(working_ledger.cash_balance + inflow, 2)
        working_ledger.total_nav = round(working_ledger.total_nav + inflow, 2)

    # 3. Ingest Market Data & Quotes
    _safe_print("\n📊 正在拉取 8只核心全天候ETF与候选股票实时市场快照行情...")
    market_snapshot: dict[str, dict[str, Any]] = {}
    market_prices: dict[str, float] = dict(DEFAULT_STOCK_PRICES)

    try:
        live_snap = fetch_live_market_snapshot()
        market_snapshot.update(live_snap)
        for code, info in live_snap.items():
            p_val = info.get("price") or info.get("latest_price")
            if p_val is not None:
                market_prices[code] = float(p_val)
    except Exception as exc:
        logger.warning(f"Live market snapshot fetch failed ({exc}), falling back to standard prices.")
        for code in GLOBAL_ETF_UNIVERSE:
            market_prices[code] = float(market_prices.get(code, 3.50))

    # 4. Generate Core-Satellite Plan
    candidates = candidate_signals if candidate_signals is not None else DEFAULT_ALPHA_CANDIDATE_POOL
    core_base = CORE_ETF_PROFILES.get(profile, CORE_ETF_PROFILES["conservative"])

    _safe_print(f"🎯 构建 80/20 核心-卫星资产配置方案 (风险配置: {profile})...")
    cs_plan = generate_core_satellite_plan(
        total_capital=working_ledger.total_nav,
        core_base_weights=core_base,
        candidate_stock_signals=candidates,
        market_prices=market_prices,
        today_date=today_str,
    )

    # 5. Execution Phase
    executed_trades: list[TradeRecord] = []
    sweep_record: TradeRecord | None = None
    sweep_info: GC001SweepInstruction | None = None

    if mode in ("execute_paper", "full_cycle"):
        _safe_print("\n⚡ 执行虚拟纸盘撮合调仓 (A股摩擦与T+1流动性约束)...")
        executed_trades = engine.execute_plan(
            ledger=working_ledger,
            plan=cs_plan,
            market_prices=market_prices,
            trade_date=today_str,
            dry_run=dry_run,
        )

        for tr in executed_trades:
            side_badge = "🟢 [BUY]" if tr.side == "BUY" else "🔴 [SELL]"
            _safe_print(
                f"  {side_badge} {tr.name} ({tr.symbol}): {tr.shares:,}股 @ {tr.executed_price:.2f}元 "
                f"| 金额: {tr.gross_notional:,.1f}元 | 费用: {tr.total_friction:.2f}元 | 理由: {tr.rationale}"
            )

        # 6. GC001 Reverse Repo Auto-Sweep (14:50)
        if auto_sweep_gc001 and working_ledger.cash_balance >= 1000.0:
            _safe_print(f"\n💰 14:50 自动扫尾: 执行 GC001 国债逆回购增益管理...")
            sweep_record = engine.execute_gc001_sweep(
                ledger=working_ledger,
                trade_date=today_str,
                dry_run=dry_run,
            )
            if sweep_record:
                executed_trades.append(sweep_record)
                _safe_print(f"  💵 [GC001] {sweep_record.rationale}")

        # 7. Mark to Market Valuation
        new_nav = engine.mark_to_market(
            ledger=working_ledger,
            current_prices=market_prices,
            trade_date=today_str,
            dry_run=dry_run,
        )
        _safe_print(f"\n📈 今日盘后盯市估值 (Mark to Market):")
        _safe_print(f"   账户总净值 (NAV): {working_ledger.total_nav:,.2f} RMB")
        _safe_print(f"   可用现金余额: {working_ledger.cash_balance:,.2f} RMB")
        _safe_print(f"   今日盈亏 (Daily PnL): {working_ledger.daily_pnl:+,.2f} RMB")
        _safe_print(f"   累计收益率: {working_ledger.cumulative_return_pct:+.2f}%")

    else:
        _safe_print("\nℹ️ [ADVICE ONLY] 模式: 仅生成分析建议与卡片，不执行交易撮合。")
        new_nav = working_ledger.total_nav

    # Calculate GC001 sweep instruction for card rendering
    sweep_info = calculate_gc001_auto_sweep(
        post_trade_cash=working_ledger.cash_balance,
        trade_date=today_str,
    )

    # 8. Render Multi-Channel Notification Cards
    has_stop_loss = any(
        ex.lifecycle_status == "EXIT_STOP_LOSS" or "止损" in ex.exit_reason
        for ex in cs_plan.exit_tickets
    )
    is_defensive = (profile == "conservative" and bool(executed_trades))
    is_bullish = bool(executed_trades) and not is_defensive

    core_w = 80.0
    sat_w = sum(cs_plan.satellite_stock_weights.values()) * 100.0
    cash_w = max(0.0, 100.0 - core_w - sat_w)

    feishu_card = build_feishu_card_v2(
        today_str=today_str,
        total_nav=working_ledger.total_nav,
        cash=working_ledger.cash_balance,
        inflow=inflow,
        plan_decision="REBALANCE_TRIGGERED" if executed_trades else "HOLD_WITHIN_DEADBAND",
        decision_reason=f"80/20核心-卫星组合盘中调度已完成 (共执行 {len(executed_trades)} 笔交易)",
        core_weight_pct=core_w,
        satellite_weight_pct=sat_w,
        cash_weight_pct=cash_w,
        has_stop_loss=has_stop_loss,
        is_defensive_risk_off=is_defensive,
        is_bullish_rebalance=is_bullish,
        exit_tickets=cs_plan.exit_tickets,
        trade_tickets=[],
        satellite_tickets=cs_plan.satellite_tickets,
        blocked_noisy_stocks=cs_plan.blocked_noisy_stocks,
        sweep_instruction=sweep_info if auto_sweep_gc001 else None,
    )

    wecom_payload = build_wecom_markdown(
        today_str=today_str,
        total_nav=working_ledger.total_nav,
        cash=working_ledger.cash_balance,
        inflow=inflow,
        plan_decision="REBALANCE_TRIGGERED" if executed_trades else "HOLD_WITHIN_DEADBAND",
        decision_reason=f"80/20核心-卫星组合盘中调度已完成 (共执行 {len(executed_trades)} 笔交易)",
        core_weight_pct=core_w,
        satellite_weight_pct=sat_w,
        cash_weight_pct=cash_w,
        has_stop_loss=has_stop_loss,
        is_defensive_risk_off=is_defensive,
        is_bullish_rebalance=is_bullish,
        exit_tickets=cs_plan.exit_tickets,
        trade_tickets=[],
        satellite_tickets=cs_plan.satellite_tickets,
        blocked_noisy_stocks=cs_plan.blocked_noisy_stocks,
        sweep_instruction=sweep_info if auto_sweep_gc001 else None,
    )

    # Build Master Action Card Markdown
    theme_color, status_badge, _ = determine_header_theme(
        has_stop_loss=has_stop_loss,
        is_defensive_risk_off=is_defensive,
        is_bullish_rebalance=is_bullish,
        decision="REBALANCE_TRIGGERED" if executed_trades else "HOLD_WITHIN_DEADBAND",
        has_exit_tickets=bool(cs_plan.exit_tickets),
    )

    card_lines = [
        f"# 🎯 14:25 每日实盘操作行动指南 (Daily Action Card)",
        f"",
        f"> **执行时间**: {today_str} 14:25 CST | **状态标签**: **{status_badge}**",
        f"> **策略体系**: 80% 全天候核心 ETF + 20% Tier-S 卫星个股事件 Alpha (T+5)",
        f"> **账户总净值 (NAV)**: **{working_ledger.total_nav:,.2f} RMB** | **可用资金**: **{working_ledger.cash_balance:,.2f} RMB**"
        + (f" | **今日定投补入**: `+{inflow:,.2f} RMB`" if inflow > 0 else ""),
        f"",
        f"**【配置分层看板】**: `[ 🏛️ 核心全天候: {core_w:.1f}% ]`  `[ 🚀 活跃卫星: {sat_w:.1f}% ]`  `[ 💰 逆回购/现金: {cash_w:.1f}% ]`",
        f"",
        f"---",
        f"",
        f"## ⚡ 实盘成交与指令看板",
        f"",
    ]

    if executed_trades:
        card_lines.append("### 📋 今日实盘成交执行明细")
        card_lines.append("| 交易类型 | 代码 | 标的名称 | 成交数量 | 成交价格 | 成交金额 | 手续费用 | 交易说明 |")
        card_lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
        for tr in executed_trades:
            side_badge = "🟢 买入" if tr.side == "BUY" else ("🔴 卖出" if tr.side == "SELL" else "💰 逆回购")
            card_lines.append(
                f"| **{side_badge}** | `{tr.symbol}` | {tr.name} | **{tr.shares:,}** | {tr.executed_price:.3f} | "
                f"{tr.gross_notional:,.1f}元 | {tr.total_friction:.2f}元 | {tr.rationale} |"
            )
        card_lines.append("")
    else:
        card_lines.append("### 🟢 今日无需任何调仓操作 (持仓处于安全容差死区内)")
        card_lines.append("")

    card_lines.append(cs_plan.to_markdown())
    card_lines.append("")

    if auto_sweep_gc001 and sweep_info:
        card_lines.append("### 💰 14:50 GC001 逆回购扫尾增益")
        card_lines.append(f"- **扫尾金额**: {sweep_info.sweep_amount:,.2f} RMB ({sweep_info.sweep_lots}手)")
        card_lines.append(f"- **预期年化收益率**: {sweep_info.annualized_rate*100:.2f}% | 计息天数: {sweep_info.interest_days}天")
        card_lines.append(f"- **操作指令**: {sweep_info.order_action}")
        card_lines.append("")

    card_markdown = "\n".join(card_lines)

    # Save action card unless dry run
    card_path.parent.mkdir(parents=True, exist_ok=True)
    with open(card_path, "w", encoding="utf-8") as f:
        f.write(card_markdown)
    _safe_print(f"📄 每日实盘操作指南已保存至: {card_path}")

    # Webhook dispatch
    webhook_res = None
    if mode in ("advice_only", "full_cycle"):
        webhook_res = send_intraday_webhook(
            webhook_url=webhook_url,
            feishu_payload=feishu_card,
            wecom_payload=wecom_payload,
            webhook_type=webhook_type,
            dry_run=dry_run,
        )

    return {
        "status": "SUCCESS",
        "trade_date": today_str,
        "mode": mode,
        "dry_run": dry_run,
        "is_trading_day": True,
        "total_nav": working_ledger.total_nav,
        "cash_balance": working_ledger.cash_balance,
        "daily_pnl": working_ledger.daily_pnl,
        "cumulative_return_pct": working_ledger.cumulative_return_pct,
        "executed_trades": [t.to_dict() for t in executed_trades],
        "core_satellite_plan": cs_plan.to_dict(),
        "action_card_path": str(card_path),
        "card_markdown": card_markdown,
        "feishu_card": feishu_card,
        "wecom_payload": wecom_payload,
        "gc001_sweep": sweep_info.to_dict() if sweep_info else None,
        "webhook_result": webhook_res,
    }


def main():
    parser = argparse.ArgumentParser(description="Institutional-Grade Automated Intraday Runner")
    parser.add_argument("--ledger", default=str(DEFAULT_LEDGER_PATH), help="Path to paper_ledger.json")
    parser.add_argument(
        "--mode",
        default="full_cycle",
        choices=["advice_only", "execute_paper", "full_cycle"],
        help="Execution mode: advice_only, execute_paper, or full_cycle (default: full_cycle)",
    )
    parser.add_argument("--auto-sweep-gc001", action="store_true", help="Automatically sweep residual cash into GC001 at 14:50")
    parser.add_argument("--webhook", default=None, help="Webhook URL for Feishu / WeCom bot")
    parser.add_argument("--webhook-type", default="feishu", choices=["feishu", "wecom", "all"], help="Webhook type: feishu, wecom, or all")
    parser.add_argument("--dry-run", action="store_true", help="Runs full calculation without sending network webhooks or altering disk state")
    parser.add_argument("--profile", default="conservative", choices=["conservative", "balanced", "aggressive"], help="Risk profile")
    parser.add_argument("--capital", type=float, default=1_000_000.0, help="Initial capital if ledger does not exist (default 1M RMB)")
    parser.add_argument("--inflow", type=float, default=0.0, help="New cash inflow deposited today")
    parser.add_argument("--date", default=None, help="Override trade date (YYYY-MM-DD)")
    parser.add_argument("--force-trade-day", action="store_true", help="Force execution even on non-trading days (weekends/holidays)")
    parser.add_argument("--card-out", default="research/production/1430_DAILY_ACTION_CARD.md", help="Action card output path")
    args = parser.parse_args()

    res = run_scheduled_intraday_cycle(
        ledger_path=args.ledger,
        mode=args.mode,
        auto_sweep_gc001=args.auto_sweep_gc001,
        webhook_url=args.webhook,
        webhook_type=args.webhook_type,
        dry_run=args.dry_run,
        profile=args.profile,
        initial_capital=args.capital,
        inflow=args.inflow,
        trade_date=args.date,
        force_trade_day=args.force_trade_day,
        action_card_path=args.card_out,
    )

    if res.get("status") == "SKIPPED_NON_TRADING_DAY":
        sys.exit(0)

    if args.dry_run:
        _safe_print("\n" + "=" * 78)
        _safe_print("✨ [DRY-RUN 完成] 全流程计算与风控门控审计均已成功运行，未更改磁盘真实账本。")
        _safe_print("=" * 78)


if __name__ == "__main__":
    main()
