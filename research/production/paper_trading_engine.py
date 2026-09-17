#!/usr/bin/env python3
"""Institutional-Grade Virtual Paper Trading Engine for Core-Satellite Portfolios.

Provides realistic simulation of China A-share market mechanics:
1. Persistent JSON ledger at `research/production/paper_ledger.json`.
2. A-Share Friction Simulator:
   - 100-share integer board lots for buying (fractions rejected).
   - 0.05% stamp duty (印花税) on sales only (reflecting 2023-08-28 halving; ETFs exempt).
   - Brokerage commission: 0.02% (2 bps) with a 5 RMB minimum fee per trade.
   - Transfer fee: 0.001% (0.1 bps).
   - Bid-ask slippage modeling (configurable, default 5 bps).
   - A-Share T+1 Liquidity Enforcement: shares bought today are locked in `locked_shares_t1`
     and cannot be sold until the next trading day.
   - 14:50 GC001 Treasury Reverse Repo cash sweep: idle cash accrues 1-day interest
     (or 3-day on Thursday) at GC001 rate (~1.8% annual / 365).
3. Persistent Audit Logging:
   - `paper_trade_transactions.csv`: complete audit trail of every executed trade.
   - `daily_nav_history.csv`: daily mark-to-market NAV, PnL, and performance trajectory.
"""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
import json
import logging
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

# Setup logging
logger = logging.getLogger("paper_trading_engine")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fin_skills.china.core_satellite_advisor import (
    CoreSatellitePlan,
    SatelliteAlphaTicket,
    SatellitePositionRecord,
)

DEFAULT_LEDGER_PATH = Path("research/production/paper_ledger.json")
DEFAULT_TRANSACTIONS_PATH = Path("research/production/paper_trade_transactions.csv")
DEFAULT_NAV_HISTORY_PATH = Path("research/production/daily_nav_history.csv")


def is_etf_code(symbol: str) -> bool:
    """Check whether an instrument is an ETF/LOF fund (exempt from stamp duty)."""
    clean = str(symbol).strip().upper()
    for prefix in ("SH", "SZ", "BJ"):
        if clean.startswith(prefix):
            clean = clean[len(prefix):]
    for suffix in (".SH", ".SZ", ".SS", ".BJ", ".XSHG", ".XSHE"):
        if clean.endswith(suffix):
            clean = clean[:-len(suffix)]
    clean = clean.strip()
    # Shanghai ETFs: 51xxxx, 56xxxx, 58xxxx
    # Shenzhen ETFs: 15xxxx, 16xxxx (LOF)
    return clean.startswith(("51", "15", "56", "58", "16"))


@dataclass
class Position:
    """Single asset holding in the paper trading ledger."""

    symbol: str
    name: str
    shares: int
    cost_basis: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    pnl_pct: float
    locked_shares_t1: int
    unlocked_shares: int
    last_buy_date: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "shares": int(self.shares),
            "cost_basis": round(float(self.cost_basis), 4),
            "current_price": round(float(self.current_price), 4),
            "market_value": round(float(self.market_value), 2),
            "unrealized_pnl": round(float(self.unrealized_pnl), 2),
            "pnl_pct": round(float(self.pnl_pct), 4),
            "locked_shares_t1": int(self.locked_shares_t1),
            "unlocked_shares": int(self.unlocked_shares),
        }

    @classmethod
    def from_dict(cls, symbol: str, data: dict[str, Any]) -> Position:
        shares = int(data.get("shares", 0))
        locked = int(data.get("locked_shares_t1", 0))
        unlocked = int(data.get("unlocked_shares", max(0, shares - locked)))
        return cls(
            symbol=symbol,
            name=str(data.get("name", symbol)),
            shares=shares,
            cost_basis=float(data.get("cost_basis", 0.0)),
            current_price=float(data.get("current_price", 0.0)),
            market_value=float(data.get("market_value", 0.0)),
            unrealized_pnl=float(data.get("unrealized_pnl", 0.0)),
            pnl_pct=float(data.get("pnl_pct", 0.0)),
            locked_shares_t1=locked,
            unlocked_shares=unlocked,
            last_buy_date=data.get("last_buy_date"),
        )


@dataclass
class TradeRecord:
    """Audit record for an executed trade or cash sweep."""

    trade_id: str
    trade_date: str
    timestamp: str
    symbol: str
    name: str
    side: str  # 'BUY', 'SELL', 'GC001_SWEEP'
    shares: int
    requested_price: float
    slippage_bps: float
    executed_price: float
    gross_notional: float
    commission: float
    transfer_fee: float
    stamp_duty: float
    total_friction: float
    net_cash_flow: float
    post_cash_balance: float
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PaperLedger:
    """Persistent account ledger for virtual paper trading."""

    account_id: str
    updated_at: str
    total_nav: float
    cash_balance: float
    positions: dict[str, Position] = field(default_factory=dict)
    daily_pnl: float = 0.0
    cumulative_return_pct: float = 0.0
    initial_capital: float = 1_000_000.0
    last_trade_date: str = ""
    cumulative_friction_rmb: float = 0.0
    trade_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "account_id": self.account_id,
            "updated_at": self.updated_at,
            "total_nav": round(self.total_nav, 2),
            "cash_balance": round(self.cash_balance, 2),
            "positions": {sym: pos.to_dict() for sym, pos in self.positions.items()},
            "daily_pnl": round(self.daily_pnl, 2),
            "cumulative_return_pct": round(self.cumulative_return_pct, 4),
            "initial_capital": round(self.initial_capital, 2),
            "last_trade_date": self.last_trade_date,
            "cumulative_friction_rmb": round(self.cumulative_friction_rmb, 2),
            "trade_count": self.trade_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PaperLedger:
        positions_raw = data.get("positions", {})
        positions = {
            sym: Position.from_dict(sym, p_data)
            for sym, p_data in positions_raw.items()
        }
        return cls(
            account_id=str(data.get("account_id", "PAPER_CORE_SATELLITE_001")),
            updated_at=str(data.get("updated_at", datetime.now().isoformat())),
            total_nav=float(data.get("total_nav", 1_000_000.0)),
            cash_balance=float(data.get("cash_balance", 1_000_000.0)),
            positions=positions,
            daily_pnl=float(data.get("daily_pnl", 0.0)),
            cumulative_return_pct=float(data.get("cumulative_return_pct", 0.0)),
            initial_capital=float(data.get("initial_capital", data.get("total_nav", 1_000_000.0))),
            last_trade_date=str(data.get("last_trade_date", "")),
            cumulative_friction_rmb=float(data.get("cumulative_friction_rmb", 0.0)),
            trade_count=int(data.get("trade_count", 0)),
        )


class PaperTradingEngine:
    """Production Paper Trading Engine with realistic A-Share frictions and T+1 liquidity."""

    def __init__(
        self,
        commission_rate: float = 0.0002,  # 0.02% (2 bps)
        min_commission: float = 5.0,      # 5 RMB minimum fee per trade
        transfer_fee_rate: float = 0.00001,  # 0.001% (0.1 bps)
        stamp_duty_rate: float = 0.0005,  # 0.05% on stock sales only
        default_slippage_bps: float = 5.0,  # 5 bps bid-ask slippage
        gc001_annual_rate: float = 0.018,  # ~1.8% annual GC001 repo rate
        ledger_path: str | Path | None = None,
        transactions_csv_path: str | Path | None = None,
        nav_history_csv_path: str | Path | None = None,
    ):
        self.commission_rate = commission_rate
        self.min_commission = min_commission
        self.transfer_fee_rate = transfer_fee_rate
        self.stamp_duty_rate = stamp_duty_rate
        self.default_slippage_bps = default_slippage_bps
        self.gc001_annual_rate = gc001_annual_rate

        self.ledger_path = Path(ledger_path) if ledger_path else DEFAULT_LEDGER_PATH
        self.transactions_csv_path = Path(transactions_csv_path) if transactions_csv_path else DEFAULT_TRANSACTIONS_PATH
        self.nav_history_csv_path = Path(nav_history_csv_path) if nav_history_csv_path else DEFAULT_NAV_HISTORY_PATH

    def initialize_ledger(
        self,
        initial_capital: float = 1_000_000.0,
        ledger_path: str | Path | None = None,
        account_id: str = "PAPER_CORE_SATELLITE_001",
        start_date: str | None = None,
    ) -> PaperLedger:
        """Create and initialize a fresh paper trading ledger."""
        path = Path(ledger_path) if ledger_path else self.ledger_path
        today_str = start_date or date.today().strftime("%Y-%m-%d")
        now_str = datetime.now().isoformat()

        ledger = PaperLedger(
            account_id=account_id,
            updated_at=now_str,
            total_nav=float(initial_capital),
            cash_balance=float(initial_capital),
            positions={},
            daily_pnl=0.0,
            cumulative_return_pct=0.0,
            initial_capital=float(initial_capital),
            last_trade_date=today_str,
            cumulative_friction_rmb=0.0,
            trade_count=0,
        )
        self.save_ledger(ledger, ledger_path=path)
        return ledger

    def load_ledger(self, ledger_path: str | Path | None = None) -> PaperLedger:
        """Load persistent ledger from JSON file, initializing if not present."""
        path = Path(ledger_path) if ledger_path else self.ledger_path
        if not path.exists():
            logger.info(f"Ledger file {path} does not exist. Initializing fresh 1,000,000 RMB ledger.")
            return self.initialize_ledger(ledger_path=path)

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return PaperLedger.from_dict(data)

    def save_ledger(self, ledger: PaperLedger, ledger_path: str | Path | None = None) -> None:
        """Persist paper trading ledger to JSON file."""
        path = Path(ledger_path) if ledger_path else self.ledger_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(ledger.to_dict(), f, indent=2, ensure_ascii=False)

    def rollover_trading_day(self, ledger: PaperLedger, new_trade_date: str) -> bool:
        """Advance trading day, unlocking T+1 locked shares from previous sessions."""
        if not ledger.last_trade_date:
            ledger.last_trade_date = new_trade_date
            return False

        if new_trade_date > ledger.last_trade_date:
            logger.info(f"Advancing trading day from {ledger.last_trade_date} to {new_trade_date}. Unlocking T+1 shares.")
            for pos in ledger.positions.values():
                if pos.locked_shares_t1 > 0:
                    pos.unlocked_shares += pos.locked_shares_t1
                    pos.locked_shares_t1 = 0
            ledger.last_trade_date = new_trade_date
            return True
        return False

    def calculate_fees(
        self,
        side: str,
        gross_notional: float,
        symbol: str,
        is_etf: bool | None = None,
    ) -> tuple[float, float, float, float]:
        """Calculate A-share friction line items: (commission, transfer_fee, stamp_duty, total_friction)."""
        side_norm = side.upper().strip()
        etf_flag = is_etf if is_etf is not None else is_etf_code(symbol)

        # Brokerage Commission: 0.02%, 5 RMB minimum
        commission = max(round(gross_notional * self.commission_rate, 4), self.min_commission)

        # Transfer Fee: 0.001%
        transfer_fee = round(gross_notional * self.transfer_fee_rate, 4)

        # Stamp Duty: 0.05% on sales only, ETFs exempt
        if side_norm == "SELL" and not etf_flag:
            stamp_duty = round(gross_notional * self.stamp_duty_rate, 4)
        else:
            stamp_duty = 0.0

        total_friction = round(commission + transfer_fee + stamp_duty, 4)
        return commission, transfer_fee, stamp_duty, total_friction

    def execute_buy(
        self,
        ledger: PaperLedger,
        symbol: str,
        name: str,
        shares: int,
        price: float,
        trade_date: str,
        slippage_bps: float | None = None,
        is_etf: bool | None = None,
        rationale: str = "",
        dry_run: bool = False,
    ) -> TradeRecord:
        """Execute a BUY order with 100-share board lot enforcement, slippage, and T+1 locking."""
        self.rollover_trading_day(ledger, trade_date)

        # 1. 100-share integer board lots enforcement
        if shares <= 0:
            raise ValueError(f"Buy shares must be positive, got {shares}")
        if shares % 100 != 0:
            raise ValueError(
                f"A-Share board lot violation: buy order for {symbol} must be an integer multiple "
                f"of 100 shares (1手), got {shares} shares."
            )

        # 2. Slippage modeling
        slip_bps = self.default_slippage_bps if slippage_bps is None else float(slippage_bps)
        exec_price = round(price * (1.0 + (slip_bps / 10000.0)), 4)
        gross_notional = round(shares * exec_price, 4)

        # 3. Frictions
        comm, trans, stamp, total_fric = self.calculate_fees("BUY", gross_notional, symbol, is_etf)
        total_cost = round(gross_notional + total_fric, 4)

        # 4. Solvency check
        if ledger.cash_balance < total_cost:
            raise ValueError(
                f"Insufficient cash balance for BUY {shares} {symbol}: required {total_cost:,.2f} RMB "
                f"(notional {gross_notional:,.2f} + fees {total_fric:.2f}), but available cash is "
                f"only {ledger.cash_balance:,.2f} RMB."
            )

        # 5. Apply to ledger
        ledger.cash_balance = round(ledger.cash_balance - total_cost, 2)
        ledger.cumulative_friction_rmb = round(ledger.cumulative_friction_rmb + total_fric, 2)
        ledger.trade_count += 1

        if symbol in ledger.positions:
            pos = ledger.positions[symbol]
            old_shares = pos.shares
            old_cost_total = pos.cost_basis * old_shares
            new_shares = old_shares + shares
            new_cost_basis = (old_cost_total + gross_notional) / new_shares
            pos.shares = new_shares
            pos.cost_basis = round(new_cost_basis, 4)
            pos.current_price = exec_price
            pos.market_value = round(new_shares * exec_price, 2)
            pos.unrealized_pnl = round(pos.market_value - (new_shares * pos.cost_basis), 2)
            pos.pnl_pct = round((exec_price - pos.cost_basis) / pos.cost_basis, 4)
            pos.locked_shares_t1 += shares
            pos.unlocked_shares = pos.shares - pos.locked_shares_t1
            pos.last_buy_date = trade_date
        else:
            ledger.positions[symbol] = Position(
                symbol=symbol,
                name=name,
                shares=shares,
                cost_basis=exec_price,
                current_price=exec_price,
                market_value=round(shares * exec_price, 2),
                unrealized_pnl=0.0,
                pnl_pct=0.0,
                locked_shares_t1=shares,  # T+1 lock
                unlocked_shares=0,
                last_buy_date=trade_date,
            )

        now_iso = datetime.now().isoformat()
        ledger.updated_at = now_iso
        self._update_total_nav_internal(ledger)

        # 6. Audit Record
        trade_id = f"TRD_{trade_date.replace('-', '')}_{ledger.trade_count:04d}_BUY"
        record = TradeRecord(
            trade_id=trade_id,
            trade_date=trade_date,
            timestamp=now_iso,
            symbol=symbol,
            name=name,
            side="BUY",
            shares=shares,
            requested_price=round(price, 4),
            slippage_bps=slip_bps,
            executed_price=exec_price,
            gross_notional=gross_notional,
            commission=comm,
            transfer_fee=trans,
            stamp_duty=stamp,
            total_friction=total_fric,
            net_cash_flow=-total_cost,
            post_cash_balance=ledger.cash_balance,
            rationale=rationale or f"买入 {shares} 股 {name}",
        )
        if not dry_run:
            self._append_transaction_csv(record)
        return record

    def execute_sell(
        self,
        ledger: PaperLedger,
        symbol: str,
        shares: int,
        price: float,
        trade_date: str,
        slippage_bps: float | None = None,
        is_etf: bool | None = None,
        rationale: str = "",
        dry_run: bool = False,
    ) -> TradeRecord:
        """Execute a SELL order respecting T+1 liquidity, stamp duty, and slippage."""
        self.rollover_trading_day(ledger, trade_date)

        if shares <= 0:
            raise ValueError(f"Sell shares must be positive, got {shares}")

        if symbol not in ledger.positions:
            raise ValueError(f"Cannot sell {symbol}: position not held in ledger.")

        pos = ledger.positions[symbol]

        # T+1 Liquidity Enforcement
        if shares > pos.unlocked_shares:
            raise ValueError(
                f"A-Share T+1 Liquidity Violation: cannot sell {shares} shares of {symbol} ({pos.name}). "
                f"Available unlocked shares: {pos.unlocked_shares}, locked shares from today (T+1): {pos.locked_shares_t1}."
            )

        # Slippage modeling
        slip_bps = self.default_slippage_bps if slippage_bps is None else float(slippage_bps)
        exec_price = round(price * (1.0 - (slip_bps / 10000.0)), 4)
        gross_notional = round(shares * exec_price, 4)

        # Frictions: includes 0.05% stamp duty on stock sales
        comm, trans, stamp, total_fric = self.calculate_fees("SELL", gross_notional, symbol, is_etf)
        net_proceeds = round(gross_notional - total_fric, 4)

        # Apply cash proceeds
        ledger.cash_balance = round(ledger.cash_balance + net_proceeds, 2)
        ledger.cumulative_friction_rmb = round(ledger.cumulative_friction_rmb + total_fric, 2)
        ledger.trade_count += 1

        # Position deduction
        pos.shares -= shares
        pos.unlocked_shares -= shares
        if pos.shares <= 0:
            del ledger.positions[symbol]
        else:
            pos.current_price = exec_price
            pos.market_value = round(pos.shares * exec_price, 2)
            pos.unrealized_pnl = round(pos.market_value - (pos.shares * pos.cost_basis), 2)
            pos.pnl_pct = round((exec_price - pos.cost_basis) / pos.cost_basis, 4)

        now_iso = datetime.now().isoformat()
        ledger.updated_at = now_iso
        self._update_total_nav_internal(ledger)

        # Audit Record
        trade_id = f"TRD_{trade_date.replace('-', '')}_{ledger.trade_count:04d}_SELL"
        record = TradeRecord(
            trade_id=trade_id,
            trade_date=trade_date,
            timestamp=now_iso,
            symbol=symbol,
            name=pos.name,
            side="SELL",
            shares=shares,
            requested_price=round(price, 4),
            slippage_bps=slip_bps,
            executed_price=exec_price,
            gross_notional=gross_notional,
            commission=comm,
            transfer_fee=trans,
            stamp_duty=stamp,
            total_friction=total_fric,
            net_cash_flow=net_proceeds,
            post_cash_balance=ledger.cash_balance,
            rationale=rationale or f"卖出 {shares} 股 {pos.name}",
        )
        if not dry_run:
            self._append_transaction_csv(record)
        return record

    def execute_gc001_sweep(
        self,
        ledger: PaperLedger,
        trade_date: str,
        annual_rate: float | None = None,
        interest_days: int | None = None,
        dry_run: bool = False,
    ) -> TradeRecord | None:
        """Sweep idle cash at 14:50 into GC001 Treasury Reverse Repo and accrue interest."""
        rate = self.gc001_annual_rate if annual_rate is None else float(annual_rate)

        # Determine interest days (Thursday earns 3 days: Fri, Sat, Sun)
        if interest_days is None:
            try:
                dt = datetime.strptime(trade_date[:10], "%Y-%m-%d").date()
            except Exception:
                dt = date.today()
            interest_days = 3 if dt.weekday() == 3 else 1

        # Minimum lot for GC001 is 1,000 RMB (1手 = 1,000元)
        sweep_amount = int(ledger.cash_balance // 1000) * 1000
        if sweep_amount < 1000:
            logger.info(f"Idle cash {ledger.cash_balance:.2f} RMB is below GC001 1,000 RMB lot. No sweep executed.")
            return None

        daily_interest = round(sweep_amount * (rate / 365.0) * interest_days, 2)
        ledger.cash_balance = round(ledger.cash_balance + daily_interest, 2)
        ledger.trade_count += 1

        now_iso = datetime.now().isoformat()
        ledger.updated_at = now_iso
        self._update_total_nav_internal(ledger)

        trade_id = f"TRD_{trade_date.replace('-', '')}_{ledger.trade_count:04d}_GC001"
        record = TradeRecord(
            trade_id=trade_id,
            trade_date=trade_date,
            timestamp=now_iso,
            symbol="204001.SH",
            name="GC001国债逆回购",
            side="GC001_SWEEP",
            shares=sweep_amount // 1000,
            requested_price=rate * 100.0,
            slippage_bps=0.0,
            executed_price=rate * 100.0,
            gross_notional=float(sweep_amount),
            commission=0.0,
            transfer_fee=0.0,
            stamp_duty=0.0,
            total_friction=0.0,
            net_cash_flow=daily_interest,
            post_cash_balance=ledger.cash_balance,
            rationale=f"14:50 GC001 自动扫尾: {sweep_amount:,} 元逆回购，年化 {rate*100:.2f}%，计息 {interest_days} 天，获得利息 +{daily_interest:.2f} 元",
        )
        if not dry_run:
            self._append_transaction_csv(record)
        return record

    def execute_plan(
        self,
        ledger: PaperLedger,
        plan: CoreSatellitePlan,
        market_prices: dict[str, float],
        trade_date: str,
        dry_run: bool = False,
    ) -> list[TradeRecord]:
        """Execute recommended adjustments from a CoreSatellitePlan respecting T+1 and board lots.

        Order of execution:
        1. Satellite Exits (SELL) to recover liquidity.
        2. Core ETF sells (if any rebalancing down).
        3. Core ETF buys (board lot 100 shares).
        4. Satellite new entries (BUY, board lot 100 shares, T+1 locked).
        """
        self.rollover_trading_day(ledger, trade_date)
        records: list[TradeRecord] = []

        # ---------------------------------------------------------------------
        # 1. Satellite Exits (SELL)
        # ---------------------------------------------------------------------
        for ex in plan.exit_tickets:
            sym = ex.symbol
            if sym in ledger.positions:
                pos = ledger.positions[sym]
                shares_to_sell = min(ex.shares, pos.unlocked_shares)
                if shares_to_sell > 0:
                    price = float(market_prices.get(sym, ex.current_price or ex.entry_price or 10.0))
                    rec = self.execute_sell(
                        ledger=ledger,
                        symbol=sym,
                        shares=shares_to_sell,
                        price=price,
                        trade_date=trade_date,
                        rationale=ex.exit_reason or f"卫星仓平仓卖出: {ex.name}",
                        dry_run=dry_run,
                    )
                    records.append(rec)

        # ---------------------------------------------------------------------
        # 2. Core ETF Rebalancing (Sells first, then Buys)
        # ---------------------------------------------------------------------
        if plan.core_etf_weights:
            total_nav = ledger.total_nav
            # Evaluate target values for each Core ETF
            core_deltas: list[tuple[str, int, float, str]] = []  # (symbol, shares_delta, price, name)
            for code, target_w in plan.core_etf_weights.items():
                price = float(market_prices.get(code, 3.0))
                if price <= 0:
                    continue
                target_value = total_nav * target_w
                current_shares = ledger.positions.get(code, Position(code, code, 0, 0, 0, 0, 0, 0, 0, 0)).shares
                current_value = current_shares * price
                diff_val = target_value - current_value

                raw_shares_diff = diff_val / price
                # Round to 100-share board lots
                lot_shares_diff = int(round(raw_shares_diff / 100.0)) * 100
                if abs(lot_shares_diff) >= 100:
                    name = code
                    core_deltas.append((code, lot_shares_diff, price, name))

            # Execute Core ETF SELLS
            for code, shares_diff, price, name in core_deltas:
                if shares_diff < 0:
                    sell_shares = abs(shares_diff)
                    if code in ledger.positions:
                        sell_shares = min(sell_shares, ledger.positions[code].unlocked_shares)
                        if sell_shares >= 100:
                            # Round down to 100-share lots
                            sell_shares = (sell_shares // 100) * 100
                            rec = self.execute_sell(
                                ledger=ledger,
                                symbol=code,
                                shares=sell_shares,
                                price=price,
                                trade_date=trade_date,
                                is_etf=True,
                                rationale=f"核心ETF再平衡卖出低配超额: {code}",
                                dry_run=dry_run,
                            )
                            records.append(rec)

            # Execute Core ETF BUYS
            for code, shares_diff, price, name in core_deltas:
                if shares_diff > 0:
                    buy_shares = (shares_diff // 100) * 100
                    # Check cash affordability
                    est_cost = buy_shares * price * 1.0006
                    if est_cost > ledger.cash_balance:
                        # Scale down buy shares
                        affordable_lots = int(ledger.cash_balance // (price * 100 * 1.0006))
                        buy_shares = affordable_lots * 100
                    if buy_shares >= 100:
                        rec = self.execute_buy(
                            ledger=ledger,
                            symbol=code,
                            name=name,
                            shares=buy_shares,
                            price=price,
                            trade_date=trade_date,
                            is_etf=True,
                            rationale=f"核心ETF再平衡买入补齐低配: {code}",
                            dry_run=dry_run,
                        )
                        records.append(rec)

        # ---------------------------------------------------------------------
        # 3. Satellite New Entries (BUY)
        # ---------------------------------------------------------------------
        for ticket in plan.satellite_tickets:
            sym = ticket.symbol
            price = float(market_prices.get(sym, ticket.entry_price or 10.0))
            shares = (ticket.shares_to_buy // 100) * 100
            if shares >= 100:
                est_cost = shares * price * 1.0006
                if est_cost > ledger.cash_balance:
                    affordable_lots = int(ledger.cash_balance // (price * 100 * 1.0006))
                    shares = affordable_lots * 100
                if shares >= 100:
                    rec = self.execute_buy(
                        ledger=ledger,
                        symbol=sym,
                        name=ticket.name,
                        shares=shares,
                        price=price,
                        trade_date=trade_date,
                        is_etf=False,
                        rationale=ticket.rationale or f"卫星事件Alpha建仓: {ticket.name}",
                        dry_run=dry_run,
                    )
                    records.append(rec)

        if not dry_run:
            self.save_ledger(ledger)

        return records

    def mark_to_market(
        self,
        ledger: PaperLedger,
        current_prices: dict[str, float],
        trade_date: str,
        dry_run: bool = False,
    ) -> float:
        """Mark portfolio to market, update NAV, daily PnL, and append to daily_nav_history.csv."""
        self.rollover_trading_day(ledger, trade_date)

        total_market_val = 0.0
        for sym, pos in ledger.positions.items():
            if sym in current_prices:
                pos.current_price = round(float(current_prices[sym]), 4)
            pos.market_value = round(pos.shares * pos.current_price, 2)
            pos.unrealized_pnl = round(pos.market_value - (pos.shares * pos.cost_basis), 2)
            if pos.cost_basis > 0:
                pos.pnl_pct = round((pos.current_price - pos.cost_basis) / pos.cost_basis, 4)
            else:
                pos.pnl_pct = 0.0
            total_market_val += pos.market_value

        new_total_nav = round(ledger.cash_balance + total_market_val, 2)
        old_nav = ledger.total_nav
        daily_pnl = round(new_total_nav - old_nav, 2)
        cum_pnl = round(new_total_nav - ledger.initial_capital, 2)
        cum_ret_pct = round((cum_pnl / ledger.initial_capital) * 100.0, 4) if ledger.initial_capital > 0 else 0.0
        daily_ret_pct = round((daily_pnl / old_nav) * 100.0, 4) if old_nav > 0 else 0.0

        ledger.total_nav = new_total_nav
        ledger.daily_pnl = daily_pnl
        ledger.cumulative_return_pct = cum_ret_pct
        ledger.updated_at = datetime.now().isoformat()
        ledger.last_trade_date = trade_date

        if not dry_run:
            self.save_ledger(ledger)
            self._append_daily_nav_csv(
                trade_date=trade_date,
                total_nav=new_total_nav,
                cash_balance=ledger.cash_balance,
                market_value=total_market_val,
                daily_pnl=daily_pnl,
                daily_ret_pct=daily_ret_pct,
                cum_ret_pct=cum_ret_pct,
                positions_count=len(ledger.positions),
            )

        return new_total_nav

    def _update_total_nav_internal(self, ledger: PaperLedger) -> None:
        """Internal helper to recompute total NAV based on current position valuations."""
        pos_val = sum(p.market_value for p in ledger.positions.values())
        ledger.total_nav = round(ledger.cash_balance + pos_val, 2)
        if ledger.initial_capital > 0:
            ledger.cumulative_return_pct = round(
                ((ledger.total_nav - ledger.initial_capital) / ledger.initial_capital) * 100.0,
                4,
            )

    def _append_transaction_csv(self, record: TradeRecord) -> None:
        """Append trade execution record to paper_trade_transactions.csv."""
        self.transactions_csv_path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = self.transactions_csv_path.exists()

        fieldnames = [
            "trade_id", "trade_date", "timestamp", "symbol", "name", "side",
            "shares", "requested_price", "slippage_bps", "executed_price",
            "gross_notional", "commission", "transfer_fee", "stamp_duty",
            "total_friction", "net_cash_flow", "post_cash_balance", "rationale",
        ]

        with open(self.transactions_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(record.to_dict())

    def _append_daily_nav_csv(
        self,
        trade_date: str,
        total_nav: float,
        cash_balance: float,
        market_value: float,
        daily_pnl: float,
        daily_ret_pct: float,
        cum_ret_pct: float,
        positions_count: int,
    ) -> None:
        """Append daily NAV record to daily_nav_history.csv, deduplicating same-day entries."""
        self.nav_history_csv_path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = self.nav_history_csv_path.exists()

        fieldnames = [
            "date", "total_nav", "cash_balance", "market_value",
            "daily_pnl", "daily_return_pct", "cumulative_return_pct", "positions_count",
        ]

        rows: list[dict[str, Any]] = []
        if file_exists:
            with open(self.nav_history_csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    if r.get("date") != trade_date:
                        rows.append(r)

        new_row = {
            "date": trade_date,
            "total_nav": f"{total_nav:.2f}",
            "cash_balance": f"{cash_balance:.2f}",
            "market_value": f"{market_value:.2f}",
            "daily_pnl": f"{daily_pnl:.2f}",
            "daily_return_pct": f"{daily_ret_pct:.4f}",
            "cumulative_return_pct": f"{cum_ret_pct:.4f}",
            "positions_count": str(positions_count),
        }
        rows.append(new_row)

        with open(self.nav_history_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


# =============================================================================
# Standalone Module-Level Convenience Functions
# =============================================================================

_DEFAULT_ENGINE = PaperTradingEngine()


def initialize_ledger(
    initial_capital: float = 1_000_000.0,
    ledger_path: str | Path | None = None,
    account_id: str = "PAPER_CORE_SATELLITE_001",
    start_date: str | None = None,
) -> PaperLedger:
    """Initialize a new persistent paper trading ledger."""
    engine = PaperTradingEngine(ledger_path=ledger_path)
    return engine.initialize_ledger(
        initial_capital=initial_capital,
        ledger_path=ledger_path,
        account_id=account_id,
        start_date=start_date,
    )


def load_ledger(ledger_path: str | Path | None = None) -> PaperLedger:
    """Load paper trading ledger from file."""
    engine = PaperTradingEngine(ledger_path=ledger_path)
    return engine.load_ledger(ledger_path=ledger_path)


def save_ledger(ledger: PaperLedger, ledger_path: str | Path | None = None) -> None:
    """Save paper trading ledger to file."""
    engine = PaperTradingEngine(ledger_path=ledger_path)
    engine.save_ledger(ledger, ledger_path=ledger_path)


def execute_plan(
    ledger: PaperLedger,
    plan: CoreSatellitePlan,
    market_prices: dict[str, float],
    trade_date: str,
    ledger_path: str | Path | None = None,
    dry_run: bool = False,
) -> list[TradeRecord]:
    """Execute adjustments from CoreSatellitePlan on paper ledger."""
    engine = PaperTradingEngine(ledger_path=ledger_path)
    return engine.execute_plan(
        ledger=ledger,
        plan=plan,
        market_prices=market_prices,
        trade_date=trade_date,
        dry_run=dry_run,
    )


def mark_to_market(
    ledger: PaperLedger,
    current_prices: dict[str, float],
    trade_date: str,
    ledger_path: str | Path | None = None,
    dry_run: bool = False,
) -> float:
    """Mark ledger to market, update NAV, and record daily history."""
    engine = PaperTradingEngine(ledger_path=ledger_path)
    return engine.mark_to_market(
        ledger=ledger,
        current_prices=current_prices,
        trade_date=trade_date,
        dry_run=dry_run,
    )
