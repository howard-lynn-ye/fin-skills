"""10-Year Simulation Models & Strategy Arms.

Models:
  1. Benchmark HS300 (100% 510300)
  2. Global Equity 60/40 (30% A-Share, 30% US, 40% Bonds)
  3. Classic All-Weather (30% Domestic Eq, 40% Bonds, 15% Gold, 15% Global Eq)
  4. Dynamic Risk Parity (Inverse realized vol weighting, monthly rebalance)
  5. Smart All-Weather (Risk Parity + Momentum tilt + Target Vol Scaling + Deadband)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Constants for A-Share ETF Broker Execution
STAMP_DUTY = 0.0000        # 0% for ETFs in China
COMMISSION_RATE = 0.0002   # 0.02%
COMMISSION_MIN = 2.0       # 2 RMB per trade
SLIPPAGE = 0.0002          # 0.02%
BOARD_LOT = 100
RISK_FREE_ANNUAL = 0.02    # 2.0%
TRADING_DAYS = 252


class BrokerSim:
    """Accurate cash broker simulation for A-share listed ETFs."""

    def __init__(self, initial_capital: float = 1_000_000.0):
        self.initial_capital = float(initial_capital)
        self.cash = float(initial_capital)
        self.positions: dict[str, dict] = {}  # code -> {"shares", "cost"}
        self.comm_paid = 0.0
        self.slip_paid = 0.0
        self.tax_paid = 0.0
        self.gross_traded = 0.0
        self.n_trades = 0

    @property
    def friction_paid(self) -> float:
        return self.comm_paid + self.slip_paid + self.tax_paid

    def market_value(self, prices: dict[str, float]) -> float:
        return sum(pos["shares"] * prices.get(c, pos["cost"]) for c, pos in self.positions.items())

    def equity(self, prices: dict[str, float]) -> float:
        return self.cash + self.market_value(prices)

    def shares_of(self, code: str) -> int:
        return self.positions.get(code, {}).get("shares", 0)

    def accrue_interest(self, daily_rf: float) -> None:
        self.cash *= (1.0 + daily_rf)

    def sell(self, code: str, shares: int, price: float) -> int:
        if code not in self.positions or price <= 0:
            return 0
        fill = min(int(shares), self.positions[code]["shares"])
        fill = (fill // BOARD_LOT) * BOARD_LOT if fill < self.positions[code]["shares"] else fill
        if fill <= 0:
            return 0
        gross = fill * price
        comm = max(COMMISSION_MIN, gross * COMMISSION_RATE)
        slip = gross * SLIPPAGE
        self.cash += (gross - comm - slip)
        self.comm_paid += comm
        self.slip_paid += slip
        self.gross_traded += gross
        self.n_trades += 1
        self.positions[code]["shares"] -= fill
        if self.positions[code]["shares"] <= 0:
            del self.positions[code]
        return fill

    def buy(self, code: str, shares: int, price: float) -> int:
        if price <= 0:
            return 0
        fill = (int(shares) // BOARD_LOT) * BOARD_LOT
        while fill >= BOARD_LOT:
            gross = fill * price
            comm = max(COMMISSION_MIN, gross * COMMISSION_RATE)
            slip = gross * SLIPPAGE
            if gross + comm + slip <= self.cash:
                break
            fill -= BOARD_LOT
        if fill < BOARD_LOT:
            return 0
        gross = fill * price
        comm = max(COMMISSION_MIN, gross * COMMISSION_RATE)
        slip = gross * SLIPPAGE
        self.cash -= (gross + comm + slip)
        self.comm_paid += comm
        self.slip_paid += slip
        self.gross_traded += gross
        self.n_trades += 1
        if code not in self.positions:
            self.positions[code] = {"shares": fill, "cost": price}
        else:
            old_s = self.positions[code]["shares"]
            old_c = self.positions[code]["cost"]
            self.positions[code] = {"shares": old_s + fill, "cost": (old_s * old_c + gross) / (old_s + fill)}
        return fill

    def rebalance(self, targets: dict[str, float], prices: dict[str, float], deadband: float = 0.02) -> None:
        eq = self.equity(prices)
        if eq <= 0:
            return
        # Sells first
        for code in list(self.positions.keys()):
            tgt_val = targets.get(code, 0.0) * eq
            px = prices.get(code, 0.0)
            if px <= 0:
                continue
            cur_val = self.shares_of(code) * px
            # Deadband check for trimming
            if tgt_val > 0 and abs(tgt_val - cur_val) / max(tgt_val, 1.0) < deadband:
                continue
            if cur_val > tgt_val:
                delta = cur_val - tgt_val
                want_sell = int(delta / px)
                if targets.get(code, 0.0) == 0.0:
                    want_sell = self.shares_of(code)
                self.sell(code, want_sell, px)

        # Buys second
        for code, w in targets.items():
            if w <= 0:
                continue
            px = prices.get(code, 0.0)
            if px <= 0:
                continue
            cur_val = self.shares_of(code) * px
            tgt_val = w * eq
            if cur_val > 0 and abs(tgt_val - cur_val) / max(tgt_val, 1.0) < deadband:
                continue
            if tgt_val > cur_val:
                delta = tgt_val - cur_val
                want_buy = int(delta / px)
                self.buy(code, want_buy, px)


class StrategyBase:
    name: str = ""
    description: str = ""

    def get_target_weights(self, step: int, hist_close: pd.DataFrame) -> dict[str, float]:
        raise NotImplementedError

    def should_rebalance(self, step: int) -> bool:
        return step % 20 == 0  # default monthly


class BenchmarkHS300(StrategyBase):
    name = "benchmark_hs300"
    description = "100% 沪深300ETF (A股核心大盘基准)，买入并长期持有"

    def get_target_weights(self, step: int, hist_close: pd.DataFrame) -> dict[str, float]:
        return {"510300": 1.0}

    def should_rebalance(self, step: int) -> bool:
        return step == 0


class ClassicAllWeather(StrategyBase):
    name = "classic_all_weather"
    description = "静态全天候：30% 股票(300+红利) + 40% 国债 + 15% 黄金 + 15% 美股(标普+纳指)，月度再平衡"

    def get_target_weights(self, step: int, hist_close: pd.DataFrame) -> dict[str, float]:
        return {
            "510300": 0.15,  # 沪深300
            "510880": 0.15,  # 红利
            "511010": 0.40,  # 国债
            "518880": 0.15,  # 黄金
            "513100": 0.075, # 纳指100
            "513500": 0.075, # 标普500
        }


class GlobalEquityBalanced(StrategyBase):
    name = "global_60_40"
    description = "全球股债60/40：30% A股(300+红利) + 30% 美股(标普+纳指) + 40% 国债，月度再平衡"

    def get_target_weights(self, step: int, hist_close: pd.DataFrame) -> dict[str, float]:
        return {
            "510300": 0.15,
            "510880": 0.15,
            "513100": 0.15,
            "513500": 0.15,
            "511010": 0.40,
        }


class DynamicRiskParity(StrategyBase):
    name = "dynamic_risk_parity"
    description = "动态风险平价：权重与资产过去60日已实现波动率成反比 (1/vol)，月度动态再平衡"

    def get_target_weights(self, step: int, hist_close: pd.DataFrame) -> dict[str, float]:
        if len(hist_close) < 60:
            return ClassicAllWeather().get_target_weights(step, hist_close)
        rets = hist_close.iloc[-60:].pct_change().dropna()
        vols = rets.std() * np.sqrt(TRADING_DAYS)
        inv_vols = 1.0 / np.maximum(vols, 0.02)
        weights = inv_vols / inv_vols.sum()
        return weights.to_dict()


class SmartAllWeather(StrategyBase):
    name = "smart_all_weather"
    description = "自适应全天候：风险平价 + 20日动量温和偏离 + 10%目标波动率缩放 + 5%换手死区"

    def get_target_weights(self, step: int, hist_close: pd.DataFrame) -> dict[str, float]:
        if len(hist_close) < 60:
            return ClassicAllWeather().get_target_weights(step, hist_close)

        # 1. Base Inverse Volatility Weights (60d)
        rets = hist_close.iloc[-60:].pct_change().dropna()
        vols = rets.std() * np.sqrt(TRADING_DAYS)
        inv_vols = 1.0 / np.maximum(vols, 0.02)
        base_w = inv_vols / inv_vols.sum()

        # 2. Gentle Momentum Tilt (20d return, +-20% relative tilt)
        mom = (hist_close.iloc[-1] / hist_close.iloc[-20] - 1.0)
        mom_rank = mom.rank(pct=True) - 0.5  # centered in [-0.5, +0.5]
        tilted_w = base_w * (1.0 + 0.30 * mom_rank)
        tilted_w = tilted_w / tilted_w.sum()

        # 3. Overall Portfolio Volatility Target (10% annual vol target)
        # Moreira & Muir (2017) Volatility Scaling
        recent_port_rets = (rets * tilted_w).sum(axis=1)
        realized_port_vol = float(recent_port_rets.std() * np.sqrt(TRADING_DAYS))
        target_vol = 0.10
        exposure = min(1.0, target_vol / max(realized_port_vol, 0.02))

        final_w = (tilted_w * exposure).to_dict()
        return final_w

    def should_rebalance(self, step: int) -> bool:
        return step % 20 == 0


def get_all_models() -> list[StrategyBase]:
    return [
        BenchmarkHS300(),
        GlobalEquityBalanced(),
        ClassicAllWeather(),
        DynamicRiskParity(),
        SmartAllWeather(),
    ]
