"""Small Budget (50K / 200K RMB) ETF Asset Allocation & Rotation Simulation.

Evaluates how small retail capital behaves when trading broad-market ETFs instead
of individual stocks. In Chinese mainland markets:
  1. ETFs are EXEMPT from the 0.05% stamp duty (印花税 = 0%).
  2. Share prices are 1~5 RMB (一手 100~500 元), eliminating the discrete lot
     barrier that locks small accounts out of high-priced stocks like Kweichow
     Moutai (1,400 RMB/sh, 140,000 RMB per lot).
  3. National Debt ETFs (国债ETF) allow T+0 intraday trading.
  4. Cross-asset diversification (Equities, Bonds, Gold, Dividends) is fully
     accessible at 50,000 RMB.

Universe:
  - 510300: 华泰柏瑞沪深300ETF (Large-cap core)
  - 510500: 南方中证500ETF (Mid-cap growth)
  - 159915: 易方达创业板ETF (High-beta tech)
  - 510880: 华泰柏瑞红利ETF (High dividend value)
  - 518880: 华安黄金ETF (Gold / commodity safe haven)
  - 511010: 国泰上证5年期国债ETF (Sovereign bonds)
  - 588000: 华夏上证科创板50ETF (STAR 50 innovation)
"""
from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CACHE_DIR = HERE / "_cache_etf"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

ETF_UNIVERSE = {
    "510300": ("1.510300", "沪深300ETF", "大盘核心"),
    "510500": ("1.510500", "中证500ETF", "中盘成长"),
    "159915": ("0.159915", "创业板ETF", "创业板"),
    "510880": ("1.510880", "红利ETF", "高股息价值"),
    "518880": ("1.518880", "黄金ETF", "商品避险"),
    "511010": ("1.511010", "国债ETF", "固定收益"),
    "588000": ("1.588000", "科创50ETF", "硬科技"),
}

# Transaction Fee Model for A-Share ETFs
STAMP_DUTY = 0.0000        # ETFs are EXEMPT from stamp duty in China
COMMISSION_RATE = 0.0002   # 0.02% (万2) standard brokerage rate
COMMISSION_MIN = 2.0       # 2 RMB min per order (many retail brokers waive the 5 RMB floor on ETFs)
SLIPPAGE = 0.0002          # 0.02% (ETFs enjoy tight order books)
BOARD_LOT = 100            # 100 shares per lot
RISK_FREE_ANNUAL = 0.02    # 2.0% cash yield
TRADING_DAYS = 252


def fetch_etf_bars() -> dict[str, pd.DataFrame]:
    """Fetch and cache daily bars from EastMoney API."""
    cache_file = CACHE_DIR / "etf_bars_2024_2026.json"
    if cache_file.exists():
        with open(cache_file, encoding="utf-8") as f:
            raw = json.load(f)
    else:
        raw = {}
        for code, (secid, name, _) in ETF_UNIVERSE.items():
            url = f"http://push2his.eastmoney.com/api/qt/stock/kline/get?secid={secid}&fields1=f1,f2,f3,f4,f5,f6&fields2=f51,f52,f53,f54,f55,f56,f57&klt=101&fqt=1&beg=20240101&end=20260911"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
                raw[code] = data["data"]["klines"]
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(raw, f, ensure_ascii=False)

    # Parse into DataFrames
    records = []
    for code, klines in raw.items():
        for line in klines:
            parts = line.split(",")
            records.append({
                "date": parts[0],
                "code": code,
                "open": float(parts[1]),
                "close": float(parts[2]),
                "high": float(parts[3]),
                "low": float(parts[4]),
                "volume": float(parts[5]),
                "amount": float(parts[6]),
            })
    df = pd.DataFrame(records)
    return {
        "open": df.pivot(index="date", columns="code", values="open").ffill().bfill(),
        "close": df.pivot(index="date", columns="code", values="close").ffill().bfill(),
        "high": df.pivot(index="date", columns="code", values="high").ffill().bfill(),
        "low": df.pivot(index="date", columns="code", values="low").ffill().bfill(),
    }


class ETFBrokerAccount:
    """Cash brokerage account tailored to A-share ETF rules."""
    def __init__(self, initial_cash: float):
        self.initial_cash = float(initial_cash)
        self.cash = float(initial_cash)
        self.positions: dict[str, dict] = {}  # code -> {"shares", "cost"}
        self.comm_paid = 0.0
        self.slip_paid = 0.0
        self.tax_paid = 0.0                   # always 0 for ETFs
        self.gross_traded = 0.0
        self.n_trades = 0

    @property
    def friction_paid(self) -> float:
        return self.comm_paid + self.slip_paid

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

    def rebalance(self, targets: dict[str, float], prices: dict[str, float]) -> None:
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
            if tgt_val > cur_val:
                delta = tgt_val - cur_val
                want_buy = int(delta / px)
                self.buy(code, want_buy, px)


def run_etf_experiment():
    print("=" * 88)
    print(" 🚀 小资金（5万 / 20万）A股 ETF 资产配置与轮动实盘模拟")
    print("=" * 88)
    bars = fetch_etf_bars()
    close_df = bars["close"]
    open_df = bars["open"]
    dates = close_df.index.tolist()
    print(f"数据周期: {dates[0]} 至 {dates[-1]} (共 {len(dates)} 个交易日)")
    print(f"ETF资产池: {list(ETF_UNIVERSE.keys())}")
    print("印花税率: 0.00% (ETF免印花税) | 佣金率: 万分之二 (保底2元) | 滑点: 万分之二\n")

    BUDGETS = [50_000.0, 200_000.0, 1_000_000.0]
    daily_rf = (1.0 + RISK_FREE_ANNUAL) ** (1.0 / TRADING_DAYS) - 1.0

    # 4 ETF Strategies:
    # 1. hs300_buy_and_hold: 100% 510300
    # 2. stock_bond_60_40: 60% 510300 + 40% 511010, monthly rebalance
    # 3. all_weather: 25% 510300 + 20% 510880 + 35% 511010 + 20% 518880, monthly rebalance
    # 4. etf_momentum_rotation: Bi-weekly hold top 2 ETFs by 20d momentum
    strat_names = ["hs300_hold", "stock_bond_60_40", "all_weather", "momentum_rotation"]

    results = {}

    for b in BUDGETS:
        accounts = {s: ETFBrokerAccount(b) for s in strat_names}
        nav_hist = {s: [] for s in strat_names}

        for t in range(len(dates)):
            cur_date = dates[t]
            today_open = open_df.iloc[t].to_dict()
            today_close = close_df.iloc[t].to_dict()
            hist_close = close_df.iloc[:t]  # strict causality: up to t-1

            for s in strat_names:
                acc = accounts[s]
                acc.accrue_interest(daily_rf)

                # Day 0: initial allocation
                if t == 0:
                    if s == "hs300_hold":
                        acc.rebalance({"510300": 1.0}, today_open)
                    elif s == "stock_bond_60_40":
                        acc.rebalance({"510300": 0.60, "511010": 0.40}, today_open)
                    elif s == "all_weather":
                        acc.rebalance({"510300": 0.25, "510880": 0.20, "511010": 0.35, "518880": 0.20}, today_open)
                    elif s == "momentum_rotation":
                        acc.rebalance({"510300": 0.50, "511010": 0.50}, today_open)

                # Rebalance rules
                elif t % 20 == 0:  # Monthly (every 20 trading days)
                    if s == "stock_bond_60_40":
                        acc.rebalance({"510300": 0.60, "511010": 0.40}, today_open)
                    elif s == "all_weather":
                        acc.rebalance({"510300": 0.25, "510880": 0.20, "511010": 0.35, "518880": 0.20}, today_open)

                if t % 10 == 0 and t >= 20:  # Bi-weekly momentum rotation
                    if s == "momentum_rotation":
                        # Compute 20-day momentum on hist_close (t-1)
                        mom = (hist_close.iloc[-1] / hist_close.iloc[-20] - 1.0)
                        top2 = mom.sort_values(ascending=False).head(2).index.tolist()
                        acc.rebalance({top2[0]: 0.50, top2[1]: 0.50}, today_open)

                # End of day valuation
                eq = acc.equity(today_close)
                nav_hist[s].append(eq)

        # Compute metrics for this budget
        b_res = {}
        for s in strat_names:
            acc = accounts[s]
            nav = pd.Series(nav_hist[s], index=dates)
            rets = nav.pct_change().dropna()
            total_ret = nav.iloc[-1] / b - 1.0
            years = len(dates) / TRADING_DAYS
            cagr = (nav.iloc[-1] / b) ** (1.0 / years) - 1.0
            vol = float(rets.std() * np.sqrt(TRADING_DAYS))
            rf_daily = (1.0 + RISK_FREE_ANNUAL) ** (1.0 / TRADING_DAYS) - 1.0
            excess = rets - rf_daily
            sharpe = float(excess.mean() / excess.std() * np.sqrt(TRADING_DAYS)) if excess.std() > 0 else 0.0
            cummax = nav.cummax()
            max_dd = float(((nav - cummax) / cummax).min())
            b_res[s] = {
                "final_equity": nav.iloc[-1],
                "total_return": total_ret,
                "cagr": cagr,
                "annual_vol": vol,
                "max_drawdown": max_dd,
                "sharpe_excess": sharpe,
                "friction_total": acc.friction_paid,
                "friction_pct": acc.friction_paid / b,
                "trades_count": acc.n_trades,
            }
        results[int(b)] = b_res

    # Print Comparison Table
    for b in BUDGETS:
        print(f"\n==================== 资金规模: {b/10000:.0f} 万元 ====================")
        print(f"{'策略名称':24s} | {'累计收益':8s} | {'年化CAGR':8s} | {'最大回撤':8s} | {'超额夏普':8s} | {'摩擦费用':9s} | {'费用占比':8s}")
        print("-" * 88)
        for s in strat_names:
            m = results[int(b)][s]
            s_label = {
                "hs300_hold": "沪深300单资产持有",
                "stock_bond_60_40": "股债平衡 (60/40)",
                "all_weather": "全天候大类资产配置",
                "momentum_rotation": "双周动量轮动",
            }[s]
            print(f"{s_label:20s} | {m['total_return']:+7.2%} | {m['cagr']:+7.2%} | {m['max_drawdown']:7.2%} | {m['sharpe_excess']:8.3f} | {m['friction_total']:8.1f}元 | {m['friction_pct']:7.3%}")

    # Save results to json
    res_path = HERE / "results_etf_simulation.json"
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存至: {res_path}")


if __name__ == "__main__":
    run_etf_experiment()
