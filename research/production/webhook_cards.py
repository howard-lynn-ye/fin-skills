"""Multi-Channel Webhook CardKit Generators and GC001 Auto-Sweep Manager.

Provides:
1. `calculate_gc001_auto_sweep`: Computes 14:50 GC001 reverse repo placement or 511010 parking.
2. `determine_header_theme`: Selects dynamic color (Green / Orange / Red) based on market & risk triggers.
3. `build_feishu_card_v2`: Generates Feishu / Lark Interactive CardKit v2 JSON payloads.
4. `build_wecom_markdown`: Generates WeChat Work (企业微信) structured markdown payloads.
5. `send_intraday_webhook`: Dispatches payloads to multi-channel webhooks with dry-run support.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime
import json
import logging
from pathlib import Path
import sys
from typing import Any
import urllib.request

# Ensure UTF-8 / ASCII-safe console output across environments
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logger = logging.getLogger("webhook_cards")


def _safe_print(msg: str, file=None) -> None:
    """Print text safely without crashing on ASCII-only terminals."""
    target = file or sys.stdout
    try:
        print(msg, file=target)
    except UnicodeEncodeError:
        enc = getattr(target, "encoding", None) or "ascii"
        safe_msg = msg.encode(enc, errors="replace").decode(enc, errors="replace")
        print(safe_msg, file=target)


def _get_interest_days(d: date | datetime | str | None = None) -> int:
    """Return number of interest-accruing days for 1-day repo on date d.

    Thursday earns 3 days (Fri/Sat/Sun). Other weekdays earn 1 day.
    """
    if d is None:
        dt = date.today()
    elif isinstance(d, str):
        try:
            dt = datetime.strptime(d[:10], "%Y-%m-%d").date()
        except Exception:
            dt = date.today()
    elif isinstance(d, datetime):
        dt = d.date()
    else:
        dt = d

    # weekday: Monday is 0, Sunday is 6. Thursday is 3.
    if dt.weekday() == 3:
        return 3
    return 1


@dataclass
class GC001SweepInstruction:
    """Actionable instruction for 14:50 GC001 reverse repo auto-sweep."""

    post_trade_cash: float
    sweep_amount: float
    sweep_lots: int
    residual_cash: float
    interest_days: int
    is_thursday_multiplier: bool
    annualized_rate: float
    primary_vehicle: str
    secondary_vehicle: str | None
    secondary_amount: float
    execution_window: str
    order_action: str
    status: str
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def calculate_gc001_auto_sweep(
    post_trade_cash: float,
    trade_date: date | datetime | str | None = None,
    demand_rate: float = 0.0030,
    expected_repo_rate: float = 0.0210,
) -> GC001SweepInstruction:
    """Calculate 14:50 - 15:10 GC001 reverse repo or 511010 Treasury ETF sweep allocation.

    Rules:
    - Minimum GC001 lot is 1,000 RMB (1手 = 1,000元).
    - If cash >= 1,000 RMB: Sweep int(cash // 1000) * 1000 into 204001 (GC001).
      Remaining sub-1,000 cash can park in 511010/511880 if >= 100 RMB.
    - If 100 <= cash < 1,000 RMB: Park directly into 511010 (国债ETF) or 511880 (银华日利).
    - If cash < 100 RMB: Keep in broker account as liquid reserve.
    """
    cash = max(0.0, float(post_trade_cash))
    interest_days = _get_interest_days(trade_date)
    is_thursday = (interest_days == 3)
    thursday_bonus = " (周四3天计息特权已激活!)" if is_thursday else ""

    if cash < 100.0:
        return GC001SweepInstruction(
            post_trade_cash=cash,
            sweep_amount=0.0,
            sweep_lots=0,
            residual_cash=cash,
            interest_days=interest_days,
            is_thursday_multiplier=is_thursday,
            annualized_rate=demand_rate,
            primary_vehicle="CASH_RESERVE",
            secondary_vehicle=None,
            secondary_amount=0.0,
            execution_window="N/A",
            order_action="保持在账户中作为微额备用金",
            status="RESERVE_HELD",
            notes=f"尾盘闲置资金仅 {cash:.2f} 元，低于调拨门槛，留存账户活期。",
        )

    if cash < 1000.0:
        # 100 ~ 1,000 RMB: Below GC001 threshold, deploy to 511010 / 511880
        return GC001SweepInstruction(
            post_trade_cash=cash,
            sweep_amount=cash,
            sweep_lots=0,
            residual_cash=0.0,
            interest_days=interest_days,
            is_thursday_multiplier=is_thursday,
            annualized_rate=expected_repo_rate,
            primary_vehicle="511010.SH (国债ETF) / 511880.SH (银华日利)",
            secondary_vehicle=None,
            secondary_amount=0.0,
            execution_window="14:50 - 15:00 CST",
            order_action=f"买入 511010/511880 投入全部剩余闲置资金 {cash:,.2f} 元",
            status="ETF_PARKED",
            notes=f"闲置资金 {cash:.2f} 元未达1,000元GC001门槛，买入场内流动性ETF消除现金拖累。",
        )

    # cash >= 1,000 RMB: Standard GC001 sweep
    lots = int(cash // 1000.0)
    sweep_amount = lots * 1000.0
    residual = cash - sweep_amount

    secondary_v = "511010.SH (国债ETF)" if residual >= 100.0 else None
    secondary_amt = residual if residual >= 100.0 else 0.0

    return GC001SweepInstruction(
        post_trade_cash=cash,
        sweep_amount=sweep_amount,
        sweep_lots=lots,
        residual_cash=residual,
        interest_days=interest_days,
        is_thursday_multiplier=is_thursday,
        annualized_rate=expected_repo_rate,
        primary_vehicle="204001.SH (GC001) / 131810.SZ (R-001)",
        secondary_vehicle=secondary_v,
        secondary_amount=secondary_amt,
        execution_window="14:50 - 15:10 CST",
        order_action=f"卖出(融出) GC001，委托金额 {sweep_amount:,.0f} 元 ({lots}手)",
        status="GC001_SWEEP_SCHEDULED",
        notes=(
            f"融出 {sweep_amount:,.0f} 元 GC001 逆回购，锁定 {interest_days} 天 ~{expected_repo_rate*100:.2f}% 年化收益{thursday_bonus}。"
            f"资金次日 09:15 100% 可用，无任何买股流动性限制。"
            + (f" 剩余零头 {residual:.2f} 元泊入 511010 国债ETF。" if secondary_v else "")
        ),
    )


def determine_header_theme(
    has_stop_loss: bool,
    is_defensive_risk_off: bool = False,
    is_bullish_rebalance: bool = False,
    decision: str = "",
    has_exit_tickets: bool = False,
) -> tuple[str, str, str]:
    """Determine header color, status badge, and descriptive title.

    Colors:
    - Red: Stop Loss Triggered (highest priority risk event).
    - Orange: Defensive Risk Off / Deadband breach / Volatility reduction.
    - Green: Bullish Rebalance / Inflow-First Balancing / Normal Deadband Hold / Take-Profit.
    """
    if has_stop_loss:
        return "red", "🚨 止损触发", "硬止损触发 | 立即清仓截断风险"
    elif is_defensive_risk_off or (decision == "REBALANCE_TRIGGERED" and not is_bullish_rebalance):
        return "orange", "🛡️ 防御再平衡", "偏离突破 / 防御性再平衡触发"
    elif decision == "INFLOW_ONLY_BALANCING":
        return "green", "💵 定投补平", "增量定投补入低配 | 零换手损耗"
    elif has_exit_tickets:
        return "green", "💰 止盈平仓", "卫星达标止盈 | 资金全额归集"
    elif decision == "HOLD_WITHIN_DEADBAND":
        return "green", "🟢 安全持仓", "资产偏离处于安全死区以内 | 无需调仓"
    else:
        return "green", "🟢 多头再平衡", "自适应多头再平衡 | 优化风险中枢"


def build_feishu_card_v2(
    today_str: str,
    total_nav: float,
    cash: float,
    inflow: float,
    plan_decision: str,
    decision_reason: str,
    core_weight_pct: float,
    satellite_weight_pct: float,
    cash_weight_pct: float,
    has_stop_loss: bool,
    is_defensive_risk_off: bool,
    is_bullish_rebalance: bool,
    exit_tickets: list[Any],
    trade_tickets: list[Any],
    satellite_tickets: list[Any],
    blocked_noisy_stocks: list[dict[str, Any]],
    sweep_instruction: GC001SweepInstruction | None = None,
) -> dict[str, Any]:
    """Generate Feishu / Lark Interactive CardKit v2 JSON structure.

    Features:
    - Header with dynamic color (Green, Orange, Red).
    - Visual pills displaying Core Allocation (80%) vs Active Satellite T+5 Tickets (up to 20%).
    - Blocked Tier C noisy stocks notification.
    - Clear table of recommended trades with 100-share board lot rounding and 14:50 GC001 sweep.
    """
    color, badge, title_desc = determine_header_theme(
        has_stop_loss=has_stop_loss,
        is_defensive_risk_off=is_defensive_risk_off,
        is_bullish_rebalance=is_bullish_rebalance,
        decision=plan_decision,
        has_exit_tickets=bool(exit_tickets),
    )

    # 1. Header
    header = {
        "title": {
            "tag": "plain_text",
            "content": f"🎯 14:25 盘中实盘操作指南 | {badge}",
        },
        "subtitle": {
            "tag": "plain_text",
            "content": f"{today_str} 14:25 CST | 80% 全天候核心 + 20% 卫星 Alpha (T+5)",
        },
        "template": color,
    }

    elements: list[dict[str, Any]] = []

    # 2. Visual Pills & Portfolio Summary
    inflow_str = f" | **今日定投补入**: `+{inflow:,.2f} RMB`" if inflow > 0 else ""
    pills_content = (
        f"**【配置分层结构】**\n"
        f"<text_tag color='blue'>核心全天候 ETF ({core_weight_pct:.1f}%)</text_tag> "
        f"<text_tag color='violet'>卫星 T+5 狙击 ({satellite_weight_pct:.1f}%)</text_tag> "
        f"<text_tag color='green'>GC001扫尾/现金 ({cash_weight_pct:.1f}%)</text_tag>\n\n"
        f"💰 **账户总净值 (NAV)**: **{total_nav:,.2f} RMB** | **可用资金**: **{cash:,.2f} RMB**{inflow_str}\n"
        f"> 状态标签: **{badge}** | {title_desc}\n"
        f"> 决策依据: {decision_reason}"
    )
    elements.append({
        "tag": "div",
        "text": {"tag": "lark_md", "content": pills_content},
    })
    elements.append({"tag": "hr"})

    # 3. Blocked Noisy Stocks Interception (Tier C)
    if blocked_noisy_stocks:
        etf_name_map = {
            "510900": "恒生ETF (港股核心)",
            "510500": "中证500ETF (A股中盘)",
            "510300": "沪深300ETF (A股大盘)",
            "511010": "国债ETF (避险压舱)",
        }
        blocked_lines = ["🛡️ **噪声个股拦截与宽基 ETF 替代明细 (Tier C Defense)**:"]
        for b in blocked_noisy_stocks:
            sub_code = b.get("etf_substitute", "510900")
            sub_name = b.get("etf_name") or etf_name_map.get(sub_code, "宽基ETF")
            reason = b.get("explanation") or b.get("reason", "低可预测性噪音标的")
            blocked_lines.append(
                f"• **{b['symbol']} {b['name']}** (`{b['tier']}`): {reason} ➔ 自动降级替换为 **`{sub_code}` {sub_name}**"
            )
        elements.append({
            "tag": "div",
            "text": {"tag": "lark_md", "content": "\n".join(blocked_lines)},
        })
        elements.append({"tag": "hr"})
    else:
        elements.append({
            "tag": "div",
            "text": {
                "tag": "lark_md",
                "content": "🛡️ **噪声个股拦截**: 今日候选池无 Tier C 噪声标的触发，全部信号符合分层准入纪律。",
            },
        })
        elements.append({"tag": "hr"})

    # 4. Recommended Trades Table (100-share board lot rounding & GC001 sweep)
    has_trades = bool(exit_tickets or trade_tickets or satellite_tickets or (sweep_instruction and sweep_instruction.sweep_amount > 0))

    if not has_trades:
        elements.append({
            "tag": "div",
            "text": {
                "tag": "lark_md",
                "content": (
                    "🟢 **今日无需进行任何股票/ETF调仓操作！**\n"
                    "• **核心ETF**: 资产偏离处于安全死区以内，无需换手交易。\n"
                    "• **卫星个股**: 现有在持标的未触及到期或止盈止损线。\n"
                    "• **尾盘管理**: 闲置资金将于 14:50 自动进行逆回购扫尾增益。"
                ),
            },
        })
    else:
        trade_rows = [
            "📋 **今日建议交易执行清单 (14:30 - 14:55 执行，严格100股整手)**",
            "",
            "| 序号 | 类别 | 方向 | 代码 | 标的名称 | 数量 (整手) | 参考价 | 预估金额 | 挂单时效 |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ]
        seq = 1

        # A. Satellite Exits
        for ex in exit_tickets:
            shares = ex.shares
            lots = int(shares // 100)
            lot_label = f"**{lots}手** ({shares}股)" if lots > 0 else f"{shares}股"
            exit_price = ex.current_price or ex.entry_price
            amt = shares * exit_price
            trade_rows.append(
                f"| {seq} | 卫星平仓 | 🔴 **卖出** | `{ex.symbol}` | **{ex.name}** | {lot_label} | {exit_price:.2f} | {amt:,.1f}元 | 14:30 市价 |"
            )
            seq += 1

        # B. Core ETF Rebalances
        for t in trade_tickets:
            side_badge = "🟢 **买入**" if t.side == "BUY" else "🔴 **卖出**"
            trade_rows.append(
                f"| {seq} | 核心调仓 | {side_badge} | `{t.code}` | {t.name} | **{t.lots}手** ({t.shares}股) | {t.price:.3f} | {t.amount:,.1f}元 | 14:30-14:55 |"
            )
            seq += 1

        # C. Satellite Entries
        for st in satellite_tickets:
            s_shares = st.shares_to_buy
            s_lots = int(s_shares // 100)
            lot_str = f"**{s_lots}手** ({s_shares}股)" if s_lots > 0 else f"{s_shares}股"
            amt = s_shares * st.entry_price
            trade_rows.append(
                f"| {seq} | 卫星建仓 | 🟢 **买入** | `{st.symbol}` | **{st.name}** | {lot_str} | {st.entry_price:.2f} | {amt:,.1f}元 | 14:35 限价 |"
            )
            seq += 1

        # D. 14:50 GC001 Sweep row
        if sweep_instruction and sweep_instruction.sweep_amount > 0:
            sweep_lots_label = f"**{sweep_instruction.sweep_lots}手**" if sweep_instruction.sweep_lots > 0 else "全额"
            th_tag = " (3天计息)" if sweep_instruction.is_thursday_multiplier else " (1天计息)"
            trade_rows.append(
                f"| {seq} | 尾盘扫尾 | 💰 **逆回购** | `204001` | **GC001逆回购** | {sweep_lots_label} ({sweep_instruction.sweep_amount:,.0f}元) | ~{sweep_instruction.annualized_rate*100:.2f}% | {sweep_instruction.sweep_amount:,.0f}元 | 14:50-15:10{th_tag} |"
            )
            seq += 1

        trade_rows.append("")
        trade_rows.append("> *📌 整手纪律: A股场内股票与ETF买入严格遵守 100股 (1手) 整数倍，杜绝碎股摩擦。*")

        elements.append({
            "tag": "div",
            "text": {"tag": "lark_md", "content": "\n".join(trade_rows)},
        })

    # 5. GC001 Auto-Sweep Detail Box
    if sweep_instruction:
        elements.append({"tag": "hr"})
        thursday_note = " 🎁 **[周四3天计息特权已激活]** (周四出借享五/六/日3天利息，周五资金正常使用)" if sweep_instruction.is_thursday_multiplier else ""
        sweep_content = (
            f"💰 **14:50 - 15:10 GC001 国债逆回购自动扫尾 (消除现金拖累)**\n"
            f"• **推荐标的**: `{sweep_instruction.primary_vehicle}`\n"
            f"• **融出金额**: **{sweep_instruction.sweep_amount:,.2f} RMB** ({sweep_instruction.sweep_lots}手)\n"
            f"• **预期年化**: ~{sweep_instruction.annualized_rate*100:.2f}% | **计息天数**: **{sweep_instruction.interest_days} 天**{thursday_note}\n"
            f"• **操作指引**: `{sweep_instruction.order_action}`\n"
            f"• **流动性保障**: 明日 09:15 资金 100% 自动回款可用，绝不耽误开盘交易。\n"
            f"• **细节说明**: {sweep_instruction.notes}"
        )
        elements.append({
            "tag": "div",
            "text": {"tag": "lark_md", "content": sweep_content},
        })

    # 6. Note Footer
    elements.append({"tag": "hr"})
    elements.append({
        "tag": "note",
        "elements": [
            {
                "tag": "plain_text",
                "content": f"fin-skills 自动化巡检引擎 | 触发时间: {today_str} 14:25 CST | 严格执行死区再平衡与逆回购扫尾",
            }
        ],
    })

    return {
        "msg_type": "interactive",
        "card": {
            "schema": "2.0",
            "config": {
                "wide_screen_mode": True,
                "enable_forward": True,
            },
            "header": header,
            "body": {
                "elements": elements,
            },
            "elements": elements,
        },
    }


def build_wecom_markdown(
    today_str: str,
    total_nav: float,
    cash: float,
    inflow: float,
    plan_decision: str,
    decision_reason: str,
    core_weight_pct: float,
    satellite_weight_pct: float,
    cash_weight_pct: float,
    has_stop_loss: bool,
    is_defensive_risk_off: bool,
    is_bullish_rebalance: bool,
    exit_tickets: list[Any],
    trade_tickets: list[Any],
    satellite_tickets: list[Any],
    blocked_noisy_stocks: list[dict[str, Any]],
    sweep_instruction: GC001SweepInstruction | None = None,
) -> dict[str, Any]:
    """Generate WeChat Work (企业微信) structured markdown payload with emoji status tags."""
    color, badge, title_desc = determine_header_theme(
        has_stop_loss=has_stop_loss,
        is_defensive_risk_off=is_defensive_risk_off,
        is_bullish_rebalance=is_bullish_rebalance,
        decision=plan_decision,
        has_exit_tickets=bool(exit_tickets),
    )

    lines: list[str] = [
        f"### 🎯 14:25 每日实盘操作指南 | {badge}",
        f"> **生成时间**: {today_str} 14:25 CST | **策略**: 80/20 核心全天候+卫星Alpha",
        "",
        f"<font color=\"info\">【核心底仓 {core_weight_pct:.1f}%】</font> <font color=\"warning\">【卫星狙击 {satellite_weight_pct:.1f}%】</font> <font color=\"comment\">【GC001扫尾 {cash_weight_pct:.1f}%】</font>",
        "",
        f"💰 **账户总净值 (NAV)**: **{total_nav:,.2f} RMB** | **可用资金**: **{cash:,.2f} RMB**"
        + (f" | **定投补入**: `+{inflow:,.2f} RMB`" if inflow > 0 else ""),
        f"> ⚡ **决策看板**: {title_desc}",
        f"> 📋 **执行依据**: {decision_reason}",
        "",
    ]

    # Blocked Tier C Noisy Stocks
    if blocked_noisy_stocks:
        lines.append("🛡️ **Tier C 噪声个股拦截与宽基替换**:")
        for b in blocked_noisy_stocks:
            sub = b.get("etf_substitute", "510900")
            lines.append(f"• `{b['symbol']}` {b['name']} (`{b['tier']}`) ➔ 拦截降级替换为 **`{sub}`**")
        lines.append("")

    # Trade table
    has_trades = bool(exit_tickets or trade_tickets or satellite_tickets or (sweep_instruction and sweep_instruction.sweep_amount > 0))
    if not has_trades:
        lines.append("🟢 <font color=\"info\">**今日无需调仓**</font>: 偏离度在安全死区以内，在持标的未触发平仓线。")
        lines.append("")
    else:
        lines.append("📋 **推荐执行清单 (100股整手纪律)**:")
        seq = 1

        for ex in exit_tickets:
            shares = ex.shares
            lots = int(shares // 100)
            lot_lbl = f"{lots}手 ({shares}股)" if lots > 0 else f"{shares}股"
            exit_price = ex.current_price or ex.entry_price
            amt = shares * exit_price
            lines.append(
                f"{seq}. <font color=\"warning\">🔴 卖出平仓</font> `{ex.symbol}` {ex.name} | {lot_lbl} @ {exit_price:.2f}元 (~{amt:,.0f}元) | 原因: {ex.exit_reason}"
            )
            seq += 1

        for t in trade_tickets:
            if t.side == "BUY":
                side_tag = "<font color=\"info\">🟢 买入建仓</font>"
            else:
                side_tag = "<font color=\"warning\">🔴 卖出减仓</font>"
            lines.append(
                f"{seq}. {side_tag} `{t.code}` {t.name} | {t.lots}手 ({t.shares}股) @ {t.price:.3f}元 (~{t.amount:,.0f}元) | {t.rationale}"
            )
            seq += 1

        for st in satellite_tickets:
            s_shares = st.shares_to_buy
            s_lots = int(s_shares // 100)
            lot_str = f"{s_lots}手 ({s_shares}股)" if s_lots > 0 else f"{s_shares}股"
            amt = s_shares * st.entry_price
            lines.append(
                f"{seq}. <font color=\"info\">🟢 卫星买入</font> `{st.symbol}` {st.name} | {lot_str} @ {st.entry_price:.2f}元 (~{amt:,.0f}元) | T+5策略"
            )
            seq += 1

        if sweep_instruction and sweep_instruction.sweep_amount > 0:
            th_text = " [周四3天息]" if sweep_instruction.is_thursday_multiplier else ""
            lines.append(
                f"{seq}. <font color=\"comment\">💰 逆回购扫尾</font> `204001` GC001 | {sweep_instruction.sweep_lots}手 ({sweep_instruction.sweep_amount:,.0f}元) | ~{sweep_instruction.annualized_rate*100:.2f}%{th_text} | 14:50 融出"
            )
            seq += 1

        lines.append("")

    # GC001 Sweep Highlight
    if sweep_instruction:
        lines.append("💰 **14:50 - 15:10 GC001 逆回购自动扫尾**:")
        lines.append(f"- 标的代码: `{sweep_instruction.primary_vehicle}`")
        lines.append(f"- 融出金额: **{sweep_instruction.sweep_amount:,.0f} RMB** ({sweep_instruction.sweep_lots}手)")
        lines.append(f"- 计息天数: **{sweep_instruction.interest_days}天** {'(周四特权)' if sweep_instruction.is_thursday_multiplier else ''}")
        lines.append(f"- 挂单指引: {sweep_instruction.order_action}")
        lines.append(f"- 流动性: 次日 09:15 资金100%自动可用")
        lines.append("")

    lines.append("> <font color=\"comment\">fin-skills 自动巡检引擎 | 严格执行整手交易与尾盘扫尾纪律</font>")

    return {
        "msg_type": "markdown",
        "markdown": {
            "content": "\n".join(lines),
        },
    }


def send_intraday_webhook(
    webhook_url: str | None,
    feishu_payload: dict[str, Any],
    wecom_payload: dict[str, Any],
    webhook_type: str = "feishu",
    dry_run: bool = False,
    timeout: int = 10,
) -> dict[str, Any]:
    """Dispatch webhook payload to Feishu, WeCom, or both, with dry-run support."""
    result: dict[str, Any] = {
        "sent": False,
        "dry_run": dry_run,
        "webhook_type": webhook_type,
        "status": "initialized",
    }

    if dry_run or not webhook_url:
        result["status"] = "dry_run_simulated"
        _safe_print(f"ℹ️ [DRY-RUN] Webhook network delivery bypassed (type={webhook_type}).")
        return result

    target_payload = feishu_payload
    if webhook_type == "wecom":
        target_payload = wecom_payload
    elif webhook_type == "all":
        # Determine based on URL structure
        if "weixin.qq.com" in webhook_url or "qyapi" in webhook_url:
            target_payload = wecom_payload
        else:
            target_payload = feishu_payload

    headers = {"Content-Type": "application/json"}
    try:
        req = urllib.request.Request(
            webhook_url,
            data=json.dumps(target_payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            is_ok = resp.status in (200, 201)
            result["sent"] = is_ok
            result["status"] = "success" if is_ok else f"http_status_{resp.status}"
            if is_ok:
                _safe_print(f"✅ Webhook card successfully sent to {webhook_url[:35]}...")
            return result
    except Exception as e:
        logger.warning("Webhook delivery failed: %s", e)
        result["status"] = f"error: {e}"
        _safe_print(f"⚠️ Webhook delivery failed: {e}", file=sys.stderr)
        return result
