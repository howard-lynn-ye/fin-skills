#!/usr/bin/env python3
"""Automated 14:25 Intraday Pipeline Runner & Satellite Lifecycle Manager.

End-to-End Orchestrator for China Market Intraday (14:25 - 14:55) Decision Making:
1. Pre-Flight State Audit:
   - Loads `my_holdings.json` (portfolio state, total NAV, available cash).
   - Loads `satellite_ledger.json` (active T+5 satellite stock positions).
2. Live Market Intelligence & Quotes:
   - Pulls live Eastmoney level-1 snapshots for 8 Core ETFs + Tier-S/A Alpha Candidates.
   - Evaluates real-time 60-day volatility and Eastmoney sectoral capital flows.
3. Intelligence Modules & Pre-Trade Guards:
   - `StockPredictabilityStratifier`: Filters candidates into Tier S/A vs. intercepts Tier C noise.
   - `KOLCredibilityRegistry`: Calibrates sentiment with Bayesian track records and contrarian inversion.
   - `SignalReconciler`: Resolves conflicting bull/bear viewpoints.
   - `QDIIPremiumGuard`: Blocks secondary-market bubble entry on cross-border ETFs (513100/513500/510900).
   - `BoardLotFeasibilityGuard`: Audits 100-share integer lot tracking error.
   - `DeadbandRebalancer`: Prevents wasteful churning via drift deadbands and Inflow-First balancing.
4. Satellite Lifecycle Management (T+5 Strategy):
   - Matured holding audit: If holding days >= 5 -> trigger `SELL_EXIT` (target horizon reached).
   - Take-Profit audit: If unrealized gain >= +8.0% -> trigger `SELL_EXIT` (take profit lock-in).
   - Stop-Loss audit: If unrealized loss <= -5.0% -> trigger `SELL_EXIT` (hard stop-loss cut).
   - In-progress tracking: Displays countdown of remaining holding days.
5. Post-Trade Cash Management & GC001 Auto-Sweep:
   - Prescribes 14:50 - 15:30 GC001 (国债逆回购) sweep placement or 511010 Treasury Bond parking.
   - `--auto-sweep-gc001`: Automatically calculates idle cash post-trade and allocates 100% of residual balance.
6. Multi-Channel Webhook Delivery & Dry-Run Simulator:
   - Feishu / Lark Interactive CardKit v2 (dynamic colors, visual pills, blocked noise alert, trade table).
   - WeChat Work (企业微信) Markdown with emoji status tags and structured formatting.
   - `--dry-run`: Full calculation & card generation without making network requests.

Usage:
    # 1. Standard dry-run (14:25 inspection):
    python3 research/production/run_intraday_pipeline.py --dry-run

    # 2. Run with new cash inflow and auto-sweep GC001:
    python3 research/production/run_intraday_pipeline.py --inflow 10000 --auto-sweep-gc001 --dry-run

    # 3. Send rich CardKit v2 card to Feishu webhook:
    python3 research/production/run_intraday_pipeline.py --webhook https://open.feishu.cn/open-apis/bot/v2/hook/xxx --webhook-type feishu

    # 4. Confirm execution (apply recommended trades to local ledger):
    python3 research/production/run_intraday_pipeline.py --confirm --auto-sweep-gc001
"""
from __future__ import annotations

import argparse
import json
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
    SatelliteLifecycleManager,
    generate_core_satellite_plan,
)
from fin_skills.china.stock_predictability_stratifier import (
    StockPredictabilityStratifier,
)
from research.production.live_advisor_bot import run_live_advisor
from research.production.webhook_cards import (
    GC001SweepInstruction,
    build_feishu_card_v2,
    build_wecom_markdown,
    calculate_gc001_auto_sweep,
    determine_header_theme,
    send_intraday_webhook,
)

# Re-export key card builders for external and test usage
__all__ = [
    "DEFAULT_ALPHA_CANDIDATE_POOL",
    "run_intraday_pipeline",
    "calculate_gc001_auto_sweep",
    "build_feishu_card_v2",
    "build_wecom_markdown",
    "determine_header_theme",
    "send_intraday_webhook",
]

# Production Tier-S/A Watchlist for daily 14:25 evaluation
DEFAULT_ALPHA_CANDIDATE_POOL = [
    {
        "symbol": "SZ300760",
        "name": "迈瑞医疗",
        "tier": "TIER_S_HIGH_PREDICTABILITY",
        "base_weight": 0.04,
        "posts": [
            {
                "author": "阿尔法工场",
                "text": "迈瑞医疗PE估值进入近10年极值低位区间，海外高端超声与微创外科订单保持高速放量，具备极高安全边际。",
                "polarity": 0.88,
                "verified": True,
            },
            {
                "author": "钟华守正出奇",
                "text": "医疗集采利空出尽还是深渊？彻底割肉清仓！",
                "polarity": -0.85,
                "verified": True,
            },
        ],
    },
    {
        "symbol": "SH600036",
        "name": "招商银行",
        "tier": "TIER_S_HIGH_PREDICTABILITY",
        "base_weight": 0.04,
        "kol_weighted_sentiment": 0.82,
        "kol_trigger_summary": "高股息与零售资产质量共振看多",
    },
    {
        "symbol": "SZ000963",
        "name": "华东医药",
        "tier": "TIER_S_HIGH_PREDICTABILITY",
        "base_weight": 0.04,
        "kol_weighted_sentiment": 0.78,
        "kol_trigger_summary": "创新药管线放量催化",
    },
    {
        "symbol": "BEKE",
        "name": "贝壳",
        "tier": "TIER_A_MODERATE_PREDICTABILITY",
        "base_weight": 0.03,
        "kol_weighted_sentiment": 0.75,
        "kol_trigger_summary": "核心一线城市存量房流动性底部企稳",
    },
    # Intentionally include Tier-C noisy candidates to test automatic interception & ETF substitution
    {
        "symbol": "02015",
        "name": "理想汽车-W",
        "tier": "TIER_C_NOISY_RANDOM_WALK",
        "raw_signal": 0.85,
        "kol_weighted_sentiment": 0.85,
    },
    {
        "symbol": "SZ300014",
        "name": "亿纬锂能",
        "tier": "TIER_C_NOISY_RANDOM_WALK",
        "raw_signal": 0.80,
        "kol_weighted_sentiment": 0.80,
    },
]


def run_intraday_pipeline(
    holdings_path: str | Path = "research/production/my_holdings.json",
    ledger_path: str | Path = "research/production/satellite_ledger.json",
    profile: str = "conservative",
    inflow: float = 0.0,
    force_rebalance: bool = False,
    confirm_execution: bool = False,
    webhook_url: str | None = None,
    webhook_type: str = "feishu",
    dry_run: bool = False,
    auto_sweep_gc001: bool = False,
    action_card_path: str | Path = "research/production/1430_DAILY_ACTION_CARD.md",
    initial_capital_if_empty: float = 100000.0,
    candidate_signals: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run full 14:25 intraday decision, satellite lifecycle audit, and rich webhook card delivery."""
    h_path = Path(holdings_path)
    l_path = Path(ledger_path)
    card_path = Path(action_card_path)

    candidates = candidate_signals if candidate_signals is not None else DEFAULT_ALPHA_CANDIDATE_POOL

    mode_tag = " [DRY-RUN 模拟模式]" if dry_run else ""
    _safe_print("=" * 78)
    _safe_print(f"🕒 启动 14:25 盘中自适应全天候与卫星执行管线{mode_tag}")
    _safe_print("=" * 78)

    # 1. Run live advisor (pass webhook_url=None so run_intraday_pipeline controls rich card delivery)
    advisor_res = run_live_advisor(
        holdings_path=h_path,
        ledger_path=l_path,
        profile=profile,
        inflow=inflow,
        force_rebalance=force_rebalance,
        webhook_url=None,
        confirm_execution=confirm_execution,
        initial_capital_if_empty=initial_capital_if_empty,
        sentiment_tilt=True,
        candidate_stock_signals=candidates,
    )

    state = advisor_res["state"]
    plan = advisor_res["plan"]
    cs_plan = advisor_res["core_satellite_plan"]
    cash_plan = advisor_res["cash_plan"]
    today_str = advisor_res["today"]

    # 2. Compute accurate post-trade residual cash
    post_cash = float(plan.post_rebalance_cash)

    # Account for Satellite Exits cash inflow
    for ex in cs_plan.exit_tickets:
        exit_price = ex.current_price or ex.entry_price
        fee = ex.shares * exit_price * 0.0006
        post_cash += (ex.shares * exit_price - fee)

    # Account for Satellite Buys cash outflow
    for st in cs_plan.satellite_tickets:
        fee = st.shares_to_buy * st.entry_price * 0.0001
        cost = st.shares_to_buy * st.entry_price + fee
        post_cash -= cost

    post_cash = max(0.0, post_cash)

    # 3. Calculate 14:50 GC001 Reverse Repo Auto-Sweep
    sweep_info = calculate_gc001_auto_sweep(
        post_trade_cash=post_cash,
        trade_date=today_str,
    )

    # 4. Risk triggers & dynamic color classification
    has_stop_loss = any(
        ex.lifecycle_status == "EXIT_STOP_LOSS" or "止损" in ex.exit_reason
        for ex in cs_plan.exit_tickets
    )
    is_defensive = (profile == "conservative" and plan.decision == "REBALANCE_TRIGGERED")
    is_bullish = (plan.decision in ("INFLOW_ONLY_BALANCING", "REBALANCE_TRIGGERED") and not is_defensive)

    # Weight percentages
    core_w = 80.0
    sat_w = sum(cs_plan.satellite_stock_weights.values()) * 100.0
    cash_w = max(0.0, 100.0 - core_w - sat_w)

    # 5. Build Multi-Channel Webhook Payloads
    feishu_card = build_feishu_card_v2(
        today_str=today_str,
        total_nav=state.total_nav,
        cash=state.cash,
        inflow=inflow,
        plan_decision=plan.decision,
        decision_reason=plan.reason,
        core_weight_pct=core_w,
        satellite_weight_pct=sat_w,
        cash_weight_pct=cash_w,
        has_stop_loss=has_stop_loss,
        is_defensive_risk_off=is_defensive,
        is_bullish_rebalance=is_bullish,
        exit_tickets=cs_plan.exit_tickets,
        trade_tickets=plan.trade_tickets,
        satellite_tickets=cs_plan.satellite_tickets,
        blocked_noisy_stocks=cs_plan.blocked_noisy_stocks,
        sweep_instruction=sweep_info if auto_sweep_gc001 else None,
    )

    wecom_payload = build_wecom_markdown(
        today_str=today_str,
        total_nav=state.total_nav,
        cash=state.cash,
        inflow=inflow,
        plan_decision=plan.decision,
        decision_reason=plan.reason,
        core_weight_pct=core_w,
        satellite_weight_pct=sat_w,
        cash_weight_pct=cash_w,
        has_stop_loss=has_stop_loss,
        is_defensive_risk_off=is_defensive,
        is_bullish_rebalance=is_bullish,
        exit_tickets=cs_plan.exit_tickets,
        trade_tickets=plan.trade_tickets,
        satellite_tickets=cs_plan.satellite_tickets,
        blocked_noisy_stocks=cs_plan.blocked_noisy_stocks,
        sweep_instruction=sweep_info if auto_sweep_gc001 else None,
    )

    # 6. Build Master Executive Action Card (Markdown)
    theme_color, status_badge, _ = determine_header_theme(
        has_stop_loss=has_stop_loss,
        is_defensive_risk_off=is_defensive,
        is_bullish_rebalance=is_bullish,
        decision=plan.decision,
        has_exit_tickets=bool(cs_plan.exit_tickets),
    )

    card_lines = [
        f"# 🎯 14:25 每日实盘操作行动指南 (Daily Action Card)",
        f"",
        f"> **生成时间**: {today_str} 14:25 CST | **状态标签**: **{status_badge}**",
        f"> **策略体系**: 80% 全天候核心 ETF + 20% Tier-S 卫星个股事件 Alpha (T+5)",
        f"> **账户总净值 (NAV)**: **{state.total_nav:,.2f} RMB** | **可用资金**: **{state.cash:,.2f} RMB**"
        + (f" | **今日定投补入**: `+{inflow:,.2f} RMB`" if inflow > 0 else ""),
        f"",
        f"**【配置分层看板】**: `[ 🏛️ 核心全天候: {core_w:.1f}% ]`  `[ 🚀 活跃卫星: {sat_w:.1f}% ]`  `[ 💰 逆回购/现金: {cash_w:.1f}% ]`",
        f"",
        f"---",
        f"",
        f"## ⚡ 核心结论与执行看板",
        f"",
    ]

    has_etf_trades = bool(plan.trade_tickets)
    has_satellite_exits = bool(cs_plan.exit_tickets)
    has_satellite_buys = bool(cs_plan.satellite_tickets)

    if not has_etf_trades and not has_satellite_exits and not has_satellite_buys:
        card_lines.append("### 🟢 今日无需任何股票/ETF交易操作！")
        card_lines.append(f"- **核心ETF**: 资产偏离处于安全死区以内 (`{plan.decision}`)，无换手损耗。")
        card_lines.append("- **卫星个股**: 现有持仓未触及 T+5 到期或止盈止损线；今日无新触发的高胜率共振信号。")
        card_lines.append(f"- **闲置资金**: 剩余 **{post_cash:,.2f} RMB** 将在 14:50 自动进行现金增益管理。")
    else:
        card_lines.append("### 🔴 今日存在明确操作指令（请于 14:30 - 14:55 依序执行）:")
        seq = 1
        if has_satellite_exits:
            card_lines.append(f"{seq}. **【优先平仓】卖出到期或止盈止损卫星个股** (回笼资金保障流动性)")
            seq += 1
        if has_etf_trades:
            card_lines.append(f"{seq}. **【核心再平衡】执行核心 ETF 再平衡挂单** (控制大类资产风险中枢)")
            seq += 1
        if has_satellite_buys:
            card_lines.append(f"{seq}. **【卫星建仓】买入新触发 Tier-S/A 高胜率狙击单** (严格执行 100股整手与 T+5 纪律)")
            seq += 1
        card_lines.append(f"{seq}. **【尾盘增益】14:50 执行 GC001 逆回购或买入 511010** (拒绝现金拖累)")

    card_lines.append("")
    card_lines.append("---")
    card_lines.append("")

    # Section 1: Satellite Exits
    if cs_plan.exit_tickets:
        card_lines.append("## ⏰ 1. 卫星仓平仓卖出清单 (收盘前市价/限价清仓)")
        card_lines.append("| 序号 | 代码 | 标的名称 | 买入日期 | 持仓天数 | 成本价 | 最新价 | 浮动盈亏 | 卖出数量 | 平仓触发理由 |")
        card_lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
        for i, ex in enumerate(cs_plan.exit_tickets, 1):
            lots = int(ex.shares // 100)
            lot_label = f"**{lots}手** ({ex.shares:,}股)" if lots > 0 else f"**{ex.shares:,}股**"
            card_lines.append(
                f"| {i} | `{ex.symbol}` | **{ex.name}** | {ex.entry_date} | {ex.current_holding_days}天 | "
                f"{ex.entry_price:.2f} | {ex.current_price:.2f} | **{ex.unrealized_pnl_pct*100:+.2f}%** | "
                f"🔴 {lot_label} | {ex.exit_reason} |"
            )
        card_lines.append("")

    # Section 2: Core ETF Rebalance
    if plan.trade_tickets:
        card_lines.append("## 🔄 2. 核心全天候 ETF 再平衡交易清单")
        card_lines.append("| 序号 | 方向 | 代码 | 资产名称 | 委托数量 | 参考价格 | 预估金额 | 预估手续费 | 调仓理由 |")
        card_lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
        for i, t in enumerate(plan.trade_tickets, 1):
            badge = "🟢 **买入 (BUY)**" if t.side == "BUY" else "🔴 **卖出 (SELL)**"
            card_lines.append(
                f"| {i} | {badge} | `{t.code}` | {t.name} | **{t.lots}手** ({t.shares}股) | {t.price:.3f} | {t.amount:,.1f}元 | {t.est_fee:.2f}元 | {t.rationale} |"
            )
        card_lines.append("")
        card_lines.append(f"> *总换手金额: {plan.gross_turnover_rmb:,.1f} 元 | 交易手续费: {plan.total_friction_rmb:.2f} 元 (ETF免征印花税)*")
        card_lines.append("")

    # Section 3: Satellite New Entries
    if cs_plan.satellite_tickets:
        card_lines.append("## 🚀 3. 卫星仓今日新开仓清单 (单票上限 5% | 严格 T+5 策略)")
        card_lines.append("| 序号 | 代码 | 标的名称 | 分层 | 触发共振信号 | 情绪分 | 建议买入 | 参考买入价 | 止盈 / 止损 | 策略逻辑 |")
        card_lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
        for i, t in enumerate(cs_plan.satellite_tickets, 1):
            s_lots = int(t.shares_to_buy // 100)
            lot_str = f"**{s_lots}手** ({t.shares_to_buy:,}股)" if s_lots > 0 else f"**{t.shares_to_buy:,}股**"
            card_lines.append(
                f"| {i} | `{t.symbol}` | **{t.name}** | `{t.tier}` | {t.kol_trigger_summary} | {t.kol_weighted_sentiment:+.2f} | "
                f"🟢 {lot_str} | {t.entry_price:.2f} | "
                f"+{t.take_profit_pct*100:.1f}% / {t.stop_loss_pct*100:.1f}% | {t.rationale} |"
            )
        card_lines.append("")

    # Section 4: Active Holding Watchlist
    if cs_plan.active_positions:
        card_lines.append("## 📋 4. 在持卫星仓生命周期跟踪 (T+5 观察中)")
        card_lines.append("| 代码 | 标的名称 | 买入日期 | 已持天数 | 成本价 | 最新价 | 浮动盈亏 | 状态 | 距到期剩余 |")
        card_lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
        for ap in cs_plan.active_positions:
            days_left = max(0, ap.target_holding_days - ap.current_holding_days)
            card_lines.append(
                f"| `{ap.symbol}` | {ap.name} | {ap.entry_date} | {ap.current_holding_days}天 | "
                f"{ap.entry_price:.2f} | {ap.current_price:.2f} | {ap.unrealized_pnl_pct*100:+.2f}% | "
                f"🟢 正常持有 | 还剩 **{days_left}** 交易日 |"
            )
        card_lines.append("")

    # Section 5: Blocked Noisy Stocks Interception
    if cs_plan.blocked_noisy_stocks:
        etf_name_map = {
            "510900": "恒生ETF (港股核心)",
            "510500": "中证500ETF (A股中盘)",
            "510300": "沪深300ETF (A股大盘)",
            "511010": "国债ETF (避险压舱)",
        }
        card_lines.append("## 🛡️ 5. 噪声个股拦截与宽基 ETF 替代明细")
        card_lines.append("| 被拦截标的 | 名称 | 分层定性 | 拦截原因 | 自动降级替换的宽基 ETF | 替代 ETF 名称 |")
        card_lines.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
        for b in cs_plan.blocked_noisy_stocks:
            sub_code = b.get("etf_substitute", "510900")
            sub_name = b.get("etf_name") or etf_name_map.get(sub_code, "宽基ETF")
            reason = b.get("explanation") or b.get("reason", "低可预测性噪音标的")
            card_lines.append(
                f"| `{b['symbol']}` | {b['name']} | `{b['tier']}` | {reason} | `{sub_code}` | **{sub_name}** |"
            )
        card_lines.append("")

    # Section 6: Cash Sweep Action
    card_lines.append("## 💰 6. 14:50 - 15:30 现金增益管理执行")
    card_lines.append(f"- **未投资闲置资金**: **{post_cash:,.2f} RMB**")

    if auto_sweep_gc001:
        th_bonus = " (🎁 周四出借享3天利息，周五资金正常使用)" if sweep_info.is_thursday_multiplier else ""
        card_lines.append(f"- **自动扫尾模式**: **已启用 (`--auto-sweep-gc001`)**")
        card_lines.append(f"- **主力扫尾标的**: `{sweep_info.primary_vehicle}`")
        card_lines.append(f"- **融出/买入金额**: **{sweep_info.sweep_amount:,.2f} RMB** ({sweep_info.sweep_lots}手)")
        card_lines.append(f"- **预期年化利率**: ~{sweep_info.annualized_rate*100:.2f}% (计息 **{sweep_info.interest_days}天**{th_bonus})")
        card_lines.append(f"- **建议操作指令**: **{sweep_info.order_action}**")
        card_lines.append(f"- **执行时效窗口**: {sweep_info.execution_window}")
        card_lines.append(f"- **增益细节要点**: {sweep_info.notes}")
    else:
        card_lines.append(f"- **推荐操作**: **{cash_plan.order_action}**")
        card_lines.append(f"- **执行时效**: {cash_plan.urgency_deadline}")
        card_lines.append(f"- **增益细节**: {cash_plan.notes}")

    card_lines.append("")
    card_lines.append("---")
    card_lines.append(f"*注: 本行动指南依据全套量化防御守则 (QDII溢价、整手颗粒度、死区再平衡、逆回购扫尾) 自动计算生成。*")

    card_markdown = "\n".join(card_lines)

    # 7. Write Action Card and JSON Payload Files
    card_path.parent.mkdir(parents=True, exist_ok=True)
    with open(card_path, "w", encoding="utf-8") as f:
        f.write(card_markdown)
    _safe_print(f"📄 每日实盘操作指南已保存至: {card_path}")

    feishu_file = card_path.parent / "feishu_card.json"
    with open(feishu_file, "w", encoding="utf-8") as f:
        json.dump(feishu_card, f, ensure_ascii=False, indent=2)

    wecom_file = card_path.parent / "wecom_card.json"
    with open(wecom_file, "w", encoding="utf-8") as f:
        json.dump(wecom_payload, f, ensure_ascii=False, indent=2)

    # 8. Dispatch Webhook
    webhook_res = send_intraday_webhook(
        webhook_url=webhook_url,
        feishu_payload=feishu_card,
        wecom_payload=wecom_payload,
        webhook_type=webhook_type,
        dry_run=dry_run,
    )

    return {
        "today": today_str,
        "action_card_path": str(card_path),
        "advisor_result": advisor_res,
        "card_markdown": card_markdown,
        "feishu_card": feishu_card,
        "wecom_payload": wecom_payload,
        "gc001_sweep": sweep_info.to_dict(),
        "webhook_result": webhook_res,
        "dry_run": dry_run,
        "auto_sweep_gc001": auto_sweep_gc001,
    }


def main():
    parser = argparse.ArgumentParser(description="Automated 14:25 Intraday Pipeline Runner & Webhook Delivery")
    parser.add_argument("--holdings", default="research/production/my_holdings.json", help="Path to holdings.json")
    parser.add_argument("--ledger", default="research/production/satellite_ledger.json", help="Path to satellite_ledger.json")
    parser.add_argument("--profile", default="conservative", choices=["conservative", "balanced", "aggressive"], help="Risk profile")
    parser.add_argument("--inflow", type=float, default=0.0, help="New cash inflow deposited today (e.g. 5000 RMB)")
    parser.add_argument("--capital", type=float, default=100000.0, help="Initial capital if holdings.json does not exist")
    parser.add_argument("--force", action="store_true", help="Force rebalance bypassing deadbands")
    parser.add_argument("--confirm", action="store_true", help="Confirm execution and update holdings.json and ledger")
    parser.add_argument("--webhook", default=None, help="Webhook URL (Feishu / WeCom / DingTalk)")
    parser.add_argument("--webhook-type", default="feishu", choices=["feishu", "wecom", "all"], help="Webhook type: feishu, wecom, or all")
    parser.add_argument("--dry-run", action="store_true", help="Runs full calculations, outputs card to console and file without making network requests")
    parser.add_argument("--auto-sweep-gc001", action="store_true", help="Automatically calculates idle cash post-trade and allocates 100% of residual balance to GC001 or 511010")
    parser.add_argument("--card-out", default="research/production/1430_DAILY_ACTION_CARD.md", help="Action card output path")
    args = parser.parse_args()

    res = run_intraday_pipeline(
        holdings_path=args.holdings,
        ledger_path=args.ledger,
        profile=args.profile,
        inflow=args.inflow,
        force_rebalance=args.force,
        confirm_execution=args.confirm,
        webhook_url=args.webhook,
        webhook_type=args.webhook_type,
        dry_run=args.dry_run,
        auto_sweep_gc001=args.auto_sweep_gc001,
        action_card_path=args.card_out,
        initial_capital_if_empty=args.capital,
    )

    _safe_print("\n" + res["card_markdown"])
    if args.dry_run:
        _safe_print("\n" + "=" * 78)
        _safe_print("✨ [DRY-RUN 检查完毕] Feishu CardKit v2 与 WeChat Work 交互卡片已成功生成并保存。")
        _safe_print("=" * 78)


if __name__ == "__main__":
    main()
