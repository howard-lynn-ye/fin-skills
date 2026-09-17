"""Unit tests for production paper trading engine and scheduled intraday runner."""
from __future__ import annotations

import csv
from datetime import date
import json
from pathlib import Path
import sys
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fin_skills.china.core_satellite_advisor import (
    CoreSatellitePlan,
    SatelliteAlphaTicket,
    SatellitePositionRecord,
)
from research.production.paper_trading_engine import (
    PaperLedger,
    PaperTradingEngine,
    Position,
    TradeRecord,
    initialize_ledger,
    is_etf_code,
    load_ledger,
    mark_to_market,
    save_ledger,
)
from research.production.scheduled_intraday_runner import (
    is_china_trading_day,
    run_scheduled_intraday_cycle,
)


# =============================================================================
# 1. Test Ledger Creation and JSON Schema Validation
# =============================================================================

def test_initialize_ledger_and_schema_validation(tmp_path: Path):
    """Test ledger creation and strict validation of all required JSON schema fields."""
    ledger_file = tmp_path / "test_ledger.json"
    engine = PaperTradingEngine(ledger_path=ledger_file)

    ledger = engine.initialize_ledger(
        initial_capital=1_000_000.0,
        ledger_path=ledger_file,
        account_id="PAPER_TEST_001",
        start_date="2026-09-16",
    )

    # 1. In-memory dataclass assertions
    assert ledger.account_id == "PAPER_TEST_001"
    assert ledger.initial_capital == 1_000_000.0
    assert ledger.total_nav == 1_000_000.0
    assert ledger.cash_balance == 1_000_000.0
    assert ledger.positions == {}
    assert ledger.daily_pnl == 0.0
    assert ledger.cumulative_return_pct == 0.0
    assert ledger.last_trade_date == "2026-09-16"

    # 2. File persistence and JSON schema assertions
    assert ledger_file.exists()
    with open(ledger_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    required_fields = [
        "account_id",
        "updated_at",
        "total_nav",
        "cash_balance",
        "positions",
        "daily_pnl",
        "cumulative_return_pct",
    ]
    for field in required_fields:
        assert field in data, f"Required field '{field}' missing from ledger JSON"

    assert data["account_id"] == "PAPER_TEST_001"
    assert data["total_nav"] == 1000000.0
    assert data["cash_balance"] == 1000000.0
    assert isinstance(data["positions"], dict)

    # 3. Re-load from disk
    loaded = engine.load_ledger(ledger_path=ledger_file)
    assert loaded.account_id == ledger.account_id
    assert loaded.total_nav == ledger.total_nav
    assert loaded.cash_balance == ledger.cash_balance


# =============================================================================
# 2. Test Buy Order Execution: Board Lot, Commission Min-5-RMB, Slippage
# =============================================================================

def test_buy_order_board_lot_commission_and_slippage(tmp_path: Path):
    """Test buy order with 100-share rounding, commission min-5-RMB, slippage, and T+1 locking."""
    l_file = tmp_path / "ledger.json"
    t_file = tmp_path / "tx.csv"
    engine = PaperTradingEngine(
        ledger_path=l_file,
        transactions_csv_path=t_file,
        commission_rate=0.0002,   # 2 bps
        min_commission=5.0,       # 5 RMB min
        transfer_fee_rate=0.00001,# 0.1 bps
        default_slippage_bps=5.0, # 5 bps
    )
    ledger = engine.initialize_ledger(initial_capital=100_000.0, ledger_path=l_file)

    # 1. Board lot violation (non-multiple of 100 shares must raise ValueError)
    with pytest.raises(ValueError, match="board lot violation"):
        engine.execute_buy(
            ledger=ledger,
            symbol="SH600036",
            name="招商银行",
            shares=150,  # Invalid: not integer multiple of 100
            price=36.50,
            trade_date="2026-09-16",
        )

    with pytest.raises(ValueError, match="Buy shares must be positive"):
        engine.execute_buy(
            ledger=ledger,
            symbol="SH600036",
            name="招商银行",
            shares=0,
            price=36.50,
            trade_date="2026-09-16",
        )

    # 2. Valid buy: 100 shares of SH600036 @ 36.50
    # Slippage: +5 bps -> 36.50 * 1.0005 = 36.51825 -> rounded to 36.5183
    # Gross notional = 100 * 36.5183 = 3651.83
    # Commission = max(3651.83 * 0.0002, 5.0) = 5.0 RMB (dominated by 5 RMB minimum fee!)
    # Transfer fee = 3651.83 * 0.00001 = 0.0365 RMB
    # Stamp duty on BUY = 0.0
    # Total friction = 5.0 + 0.0365 = 5.0365 RMB
    # Total cost = 3651.83 + 5.0365 = 3656.8665 RMB
    rec = engine.execute_buy(
        ledger=ledger,
        symbol="SH600036",
        name="招商银行",
        shares=100,
        price=36.50,
        trade_date="2026-09-16",
    )

    assert rec.shares == 100
    assert rec.requested_price == 36.50
    assert rec.slippage_bps == 5.0
    assert rec.executed_price == pytest.approx(36.5183, abs=0.001)
    assert rec.commission == 5.0  # Dominant min-5-RMB rule
    assert rec.transfer_fee == pytest.approx(0.0365, abs=0.005)
    assert rec.stamp_duty == 0.0  # Buy side pays NO stamp duty
    assert rec.total_friction == pytest.approx(5.0365, abs=0.01)

    # Check ledger state
    assert ledger.cash_balance == pytest.approx(100_000.0 - rec.gross_notional - rec.total_friction, abs=0.1)
    assert "SH600036" in ledger.positions
    pos = ledger.positions["SH600036"]
    assert pos.shares == 100
    assert pos.cost_basis == rec.executed_price
    assert pos.locked_shares_t1 == 100  # Locked today under T+1
    assert pos.unlocked_shares == 0     # Cannot be sold today!

    # Check CSV transaction audit trail
    assert t_file.exists()
    with open(t_file, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["symbol"] == "SH600036"
    assert rows[0]["side"] == "BUY"
    assert float(rows[0]["commission"]) == 5.0


# =============================================================================
# 3. Test Sell Order Execution: 0.05% Stamp Duty and Slippage
# =============================================================================

def test_sell_order_stamp_duty_and_slippage(tmp_path: Path):
    """Test sell order execution with 0.05% stamp duty on stocks and exemption on ETFs."""
    l_file = tmp_path / "ledger.json"
    t_file = tmp_path / "tx.csv"
    engine = PaperTradingEngine(
        ledger_path=l_file,
        transactions_csv_path=t_file,
        commission_rate=0.0002,   # 2 bps
        min_commission=5.0,       # 5 RMB min
        transfer_fee_rate=0.00001,# 0.1 bps
        stamp_duty_rate=0.0005,   # 0.05%
        default_slippage_bps=5.0, # 5 bps
    )
    ledger = engine.initialize_ledger(initial_capital=50_000.0, ledger_path=l_file)

    # Setup unlocked stock holding: 200 shares of SZ300760 (迈瑞医疗)
    ledger.positions["SZ300760"] = Position(
        symbol="SZ300760",
        name="迈瑞医疗",
        shares=200,
        cost_basis=250.0,
        current_price=250.0,
        market_value=50000.0,
        unrealized_pnl=0.0,
        pnl_pct=0.0,
        locked_shares_t1=0,
        unlocked_shares=200,  # Fully unlocked
    )

    # 1. Sell 100 shares of stock @ 260.00
    # Slippage: -5 bps -> 260.00 * 0.9995 = 259.87
    # Gross notional = 100 * 259.87 = 25987.0 RMB
    # Commission: 25987.0 * 0.0002 = 5.1974 RMB (> 5 RMB)
    # Transfer fee: 25987.0 * 0.00001 = 0.25987 RMB
    # Stamp duty (0.05% on stock sell): 25987.0 * 0.0005 = 12.9935 RMB
    # Total friction = 5.1974 + 0.25987 + 12.9935 = 18.45077 RMB
    # Net cash flow = 25987.0 - 18.45077 = 25968.54923 RMB
    rec_stock = engine.execute_sell(
        ledger=ledger,
        symbol="SZ300760",
        shares=100,
        price=260.00,
        trade_date="2026-09-16",
    )

    assert rec_stock.shares == 100
    assert rec_stock.executed_price == pytest.approx(259.87, abs=0.01)
    assert rec_stock.stamp_duty == pytest.approx(12.9935, abs=0.01)  # 0.05% stamp duty levied!
    assert rec_stock.commission == pytest.approx(5.1974, abs=0.01)
    assert rec_stock.transfer_fee == pytest.approx(0.2599, abs=0.01)
    assert rec_stock.net_cash_flow == pytest.approx(25987.0 - 18.4508, abs=0.1)

    # 2. Sell ETF holding: 510300 (沪深300ETF) -> MUST be stamp duty exempt!
    ledger.positions["510300"] = Position(
        symbol="510300",
        name="沪深300ETF",
        shares=10000,
        cost_basis=4.00,
        current_price=4.00,
        market_value=40000.0,
        unrealized_pnl=0.0,
        pnl_pct=0.0,
        locked_shares_t1=0,
        unlocked_shares=10000,
    )

    rec_etf = engine.execute_sell(
        ledger=ledger,
        symbol="510300",
        shares=5000,
        price=4.50,
        trade_date="2026-09-16",
    )

    assert rec_etf.stamp_duty == 0.0  # ETF sale is strictly EXEMPT from stamp duty in China!
    assert rec_etf.shares == 5000


# =============================================================================
# 4. Test A-Share T+1 Liquidity Enforcement
# =============================================================================

def test_t_plus_1_lock_prevents_same_day_sale(tmp_path: Path):
    """Test T+1 lock prevents same-day selling and unlocks on the next trading day."""
    engine = PaperTradingEngine(ledger_path=tmp_path / "ledger.json")
    ledger = engine.initialize_ledger(initial_capital=200_000.0, start_date="2026-09-15")

    # Day 1: Buy 500 shares of SZ000963 on 2026-09-15
    engine.execute_buy(
        ledger=ledger,
        symbol="SZ000963",
        name="华东医药",
        shares=500,
        price=32.00,
        trade_date="2026-09-15",
    )

    pos = ledger.positions["SZ000963"]
    assert pos.shares == 500
    assert pos.locked_shares_t1 == 500
    assert pos.unlocked_shares == 0

    # Attempt to sell on same day (2026-09-15): MUST be rejected with T+1 violation error
    with pytest.raises(ValueError, match="T\\+1 Liquidity Violation"):
        engine.execute_sell(
            ledger=ledger,
            symbol="SZ000963",
            shares=100,
            price=33.00,
            trade_date="2026-09-15",
        )

    # Advance to Day 2 (2026-09-16)
    engine.rollover_trading_day(ledger, "2026-09-16")
    assert pos.locked_shares_t1 == 0
    assert pos.unlocked_shares == 500

    # Selling 300 shares on Day 2 should now succeed
    rec_sell = engine.execute_sell(
        ledger=ledger,
        symbol="SZ000963",
        shares=300,
        price=33.50,
        trade_date="2026-09-16",
    )
    assert rec_sell.shares == 300
    assert pos.shares == 200
    assert pos.unlocked_shares == 200

    # Buy another 200 shares on Day 2
    engine.execute_buy(
        ledger=ledger,
        symbol="SZ000963",
        name="华东医药",
        shares=200,
        price=33.50,
        trade_date="2026-09-16",
    )
    # Total shares: 400 (200 old unlocked + 200 new locked)
    assert pos.shares == 400
    assert pos.unlocked_shares == 200
    assert pos.locked_shares_t1 == 200

    # Selling up to unlocked 200 shares succeeds
    engine.execute_sell(
        ledger=ledger,
        symbol="SZ000963",
        shares=200,
        price=33.50,
        trade_date="2026-09-16",
    )

    # Attempting to sell remaining 200 shares on Day 2 fails because they are locked
    with pytest.raises(ValueError, match="T\\+1 Liquidity Violation"):
        engine.execute_sell(
            ledger=ledger,
            symbol="SZ000963",
            shares=100,
            price=33.50,
            trade_date="2026-09-16",
        )


# =============================================================================
# 5. Test GC001 Overnight Repo Yield Calculation & Thursday Multiplier
# =============================================================================

def test_gc001_overnight_repo_yield_calculation(tmp_path: Path):
    """Test GC001 reverse repo auto-sweep interest calculation and Thursday 3-day multiplier."""
    engine = PaperTradingEngine(
        ledger_path=tmp_path / "ledger.json",
        transactions_csv_path=tmp_path / "tx.csv",
        gc001_annual_rate=0.018,  # 1.8%
    )
    ledger = engine.initialize_ledger(initial_capital=100_000.0)

    # 1. Standard weekday (Wednesday 2026-09-16 -> 1 day interest)
    # 100,000 RMB * (0.018 / 365) * 1 = 4.9315 RMB -> rounded to 4.93 RMB
    rec_wed = engine.execute_gc001_sweep(
        ledger=ledger,
        trade_date="2026-09-16",
    )
    assert rec_wed is not None
    assert rec_wed.shares == 100  # 100 lots of 1,000 RMB
    assert rec_wed.gross_notional == 100_000.0
    assert rec_wed.net_cash_flow == pytest.approx(4.93, abs=0.02)
    assert ledger.cash_balance == pytest.approx(100_004.93, abs=0.02)

    # 2. Thursday multiplier (Thursday 2026-09-17 -> 3 days interest: Fri, Sat, Sun)
    # 100,000 RMB * (0.018 / 365) * 3 = 14.7945 RMB -> rounded to 14.79 RMB
    ledger.cash_balance = 100_000.0
    rec_thu = engine.execute_gc001_sweep(
        ledger=ledger,
        trade_date="2026-09-17",
    )
    assert rec_thu is not None
    assert rec_thu.net_cash_flow == pytest.approx(14.79, abs=0.03)
    assert ledger.cash_balance == pytest.approx(100_014.79, abs=0.03)

    # 3. Sub-1000 cash balance: below minimum lot (1,000 RMB) -> returns None, no sweep
    ledger.cash_balance = 650.0
    rec_small = engine.execute_gc001_sweep(ledger=ledger, trade_date="2026-09-16")
    assert rec_small is None
    assert ledger.cash_balance == 650.0


# =============================================================================
# 6. Test Mark to Market NAV Updating and CSV Trajectory Recording
# =============================================================================

def test_mark_to_market_and_csv_trajectory(tmp_path: Path):
    """Test mark to market NAV updating and persistence in daily_nav_history.csv."""
    l_file = tmp_path / "ledger.json"
    nav_file = tmp_path / "nav.csv"
    engine = PaperTradingEngine(
        ledger_path=l_file,
        nav_history_csv_path=nav_file,
    )
    ledger = engine.initialize_ledger(initial_capital=500_000.0, ledger_path=l_file)
    ledger.cash_balance = 100_000.0

    # Add positions
    ledger.positions["510300"] = Position(
        symbol="510300",
        name="沪深300ETF",
        shares=50000,
        cost_basis=4.00,
        current_price=4.00,
        market_value=200000.0,
        unrealized_pnl=0.0,
        pnl_pct=0.0,
        locked_shares_t1=0,
        unlocked_shares=50000,
    )
    ledger.positions["SZ300760"] = Position(
        symbol="SZ300760",
        name="迈瑞医疗",
        shares=800,
        cost_basis=250.0,
        current_price=250.0,
        market_value=200000.0,
        unrealized_pnl=0.0,
        pnl_pct=0.0,
        locked_shares_t1=0,
        unlocked_shares=800,
    )
    ledger.total_nav = 500_000.0

    # Day 1: Market goes up!
    # 510300: 4.00 -> 4.20 (+5%) -> 50,000 * 4.20 = 210,000
    # SZ300760: 250 -> 265 (+6%) -> 800 * 265 = 212,000
    # Total market val = 422,000 + cash 100,000 = 522,000 NAV (+22,000 PnL)
    prices_day1 = {"510300": 4.20, "SZ300760": 265.0}
    nav1 = engine.mark_to_market(ledger, prices_day1, trade_date="2026-09-16")

    assert nav1 == 522_000.0
    assert ledger.total_nav == 522_000.0
    assert ledger.daily_pnl == 22_000.0
    assert ledger.cumulative_return_pct == pytest.approx(4.40, abs=0.01)  # 22k / 500k = 4.4%

    # Verify daily_nav_history.csv row
    assert nav_file.exists()
    with open(nav_file, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["date"] == "2026-09-16"
    assert float(rows[0]["total_nav"]) == 522_000.0
    assert float(rows[0]["daily_pnl"]) == 22_000.0
    assert float(rows[0]["cumulative_return_pct"]) == pytest.approx(4.40, abs=0.01)

    # Day 2: Market slight dip
    prices_day2 = {"510300": 4.15, "SZ300760": 260.0}
    nav2 = engine.mark_to_market(ledger, prices_day2, trade_date="2026-09-17")
    # 50,000 * 4.15 = 207,500 + 800 * 260 = 208,000 -> 415,500 + 100,000 = 515,500
    assert nav2 == 515_500.0
    assert ledger.daily_pnl == -6500.0  # 515,500 - 522,000

    with open(nav_file, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert rows[1]["date"] == "2026-09-17"
    assert float(rows[1]["total_nav"]) == 515_500.0


# =============================================================================
# 7. Test Execute Plan on CoreSatellitePlan
# =============================================================================

def test_execute_plan_core_satellite(tmp_path: Path):
    """Test executing a full CoreSatellitePlan with exit tickets, core rebalancing, and buys."""
    l_file = tmp_path / "ledger.json"
    engine = PaperTradingEngine(ledger_path=l_file)
    ledger = engine.initialize_ledger(initial_capital=500_000.0, ledger_path=l_file)

    # Setup 1 position to exit (matured satellite)
    ledger.positions["SZ300760"] = Position(
        symbol="SZ300760",
        name="迈瑞医疗",
        shares=200,
        cost_basis=250.0,
        current_price=270.0,
        market_value=54000.0,
        unrealized_pnl=4000.0,
        pnl_pct=0.08,
        locked_shares_t1=0,
        unlocked_shares=200,
    )
    ledger.cash_balance = 446_000.0
    ledger.total_nav = 500_000.0

    plan = CoreSatellitePlan(
        core_weight_budget=0.80,
        satellite_weight_budget=0.20,
        core_etf_weights={"510300": 0.40, "511010": 0.40},
        satellite_stock_weights={"SH600036": 0.05},
        exit_tickets=[
            SatellitePositionRecord(
                symbol="SZ300760",
                name="迈瑞医疗",
                shares=200,
                entry_price=250.0,
                entry_date="2026-09-01",
                current_price=270.0,
                action="SELL_EXIT",
                exit_reason="达标止盈 (+8.0%)",
            )
        ],
        satellite_tickets=[
            SatelliteAlphaTicket(
                symbol="SH600036",
                name="招商银行",
                tier="TIER_S_HIGH_PREDICTABILITY",
                kol_trigger_summary="头部大V共振",
                kol_weighted_sentiment=0.85,
                action="BUY",
                shares_to_buy=600,
                entry_price=36.50,
                rationale="高股息价值龙头",
            )
        ],
    )

    prices = {"SZ300760": 270.0, "510300": 4.50, "511010": 140.0, "SH600036": 36.50}
    executed = engine.execute_plan(ledger, plan, prices, trade_date="2026-09-16")

    assert len(executed) >= 3
    # 1. SZ300760 was sold
    assert any(t.symbol == "SZ300760" and t.side == "SELL" for t in executed)
    assert "SZ300760" not in ledger.positions  # Fully exited

    # 2. SH600036 was bought in 100-share board lot
    assert any(t.symbol == "SH600036" and t.side == "BUY" and t.shares == 600 for t in executed)
    assert "SH600036" in ledger.positions
    assert ledger.positions["SH600036"].locked_shares_t1 == 600

    # 3. Core ETFs bought
    assert any(t.symbol == "510300" and t.side == "BUY" for t in executed)
    assert any(t.symbol == "511010" and t.side == "BUY" for t in executed)


# =============================================================================
# 8. Test Chinese Trading Day Calendar Validator
# =============================================================================

def test_chinese_trading_day_calendar():
    """Verify weekend filtering and Chinese statutory holiday closures."""
    # Weekends are never trading days
    assert is_china_trading_day("2026-09-19") is False  # Saturday
    assert is_china_trading_day("2026-09-20") is False  # Sunday

    # Chinese statutory holidays
    assert is_china_trading_day("2026-01-01") is False  # 元旦
    assert is_china_trading_day("2026-02-17") is False  # 春节
    assert is_china_trading_day("2026-05-01") is False  # 劳动节
    assert is_china_trading_day("2026-10-01") is False  # 国庆节

    # Regular trading days
    assert is_china_trading_day("2026-09-16") is True   # Wednesday
    assert is_china_trading_day("2026-09-17") is True   # Thursday
    assert is_china_trading_day("2026-09-18") is True   # Friday


# =============================================================================
# 9. Test Scheduled Intraday Runner Integration and Dry-Run
# =============================================================================

def test_scheduled_intraday_runner_dry_run(tmp_path: Path):
    """Test full cycle scheduled intraday runner in dry-run mode."""
    l_file = tmp_path / "paper_ledger.json"
    card_file = tmp_path / "1430_DAILY_ACTION_CARD.md"

    res = run_scheduled_intraday_cycle(
        ledger_path=l_file,
        mode="full_cycle",
        auto_sweep_gc001=True,
        dry_run=True,
        profile="conservative",
        initial_capital=500_000.0,
        trade_date="2026-09-17",  # Thursday
        action_card_path=card_file,
    )

    assert res["status"] == "SUCCESS"
    assert res["dry_run"] is True
    assert res["total_nav"] > 0
    assert len(res["executed_trades"]) > 0
    assert card_file.exists()
    card_text = card_file.read_text(encoding="utf-8")
    assert "# 🎯 14:25 每日实盘操作行动指南" in card_text
    assert "今日实盘成交执行明细" in card_text
    assert "14:50 GC001 逆回购扫尾增益" in card_text


def test_scheduled_intraday_runner_skips_non_trading_day(tmp_path: Path):
    """Test that runner automatically skips execution on non-trading days."""
    l_file = tmp_path / "paper_ledger.json"

    # Sunday
    res_weekend = run_scheduled_intraday_cycle(
        ledger_path=l_file,
        mode="full_cycle",
        dry_run=False,
        force_trade_day=False,
        trade_date="2026-09-20",  # Sunday
    )
    assert res_weekend["status"] == "SKIPPED_NON_TRADING_DAY"
    assert res_weekend["is_trading_day"] is False
