#!/usr/bin/env python3
"""Multi-Budget Real Market Paper Trading Simulation with fin-skills.

Simulates a realistic broker account:
- T+1 settlement enforcement
- 100-share board lot constraint
- Exact cash debit/credit
- Stamp duty (0.05% seller-only) + commission (0.025%) + slippage (0.05%)
- Evaluates across different budgets: 50K, 200K, 1M, 5M RMB.
"""
from __future__ import annotations

import baostock as bs
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

UNIVERSE = {
    "sh.600519": {"name": "贵州茅台", "sector": "消费"},
    "sz.300750": {"name": "宁德时代", "sector": "新能源"},
    "sz.002594": {"name": "比亚迪",   "sector": "汽车/制造"},
    "sz.300308": {"name": "中际旭创", "sector": "算力通信"},
    "sh.601138": {"name": "工业富联", "sector": "算力硬件"},
    "sz.002475": {"name": "立讯精密", "sector": "消费电子"},
    "sh.600036": {"name": "招商银行", "sector": "金融"},
    "sh.601318": {"name": "中国平安", "sector": "金融"},
    "sh.601899": {"name": "紫金矿业", "sector": "有色资源"},
    "sz.002415": {"name": "海康威视", "sector": "科技制造"}
}

BUY_COMMISSION = 0.00025
BUY_SLIPPAGE = 0.0005
SELL_STAMP_DUTY = 0.0005
SELL_COMMISSION = 0.00025
SELL_SLIPPAGE = 0.0005

def fetch_market_data(start_date: str = "2026-01-01", end_date: str = "2026-09-13"):
    lg = bs.login()
    price_frames = []
    for code in UNIVERSE:
        rs = bs.query_history_k_data_plus(
            code,
            "date,code,open,high,low,close,preclose,volume,amount,pctChg",
            start_date=start_date,
            end_date=end_date,
            frequency="d",
            adjustflag="3"
        )
        rows = []
        while (rs.error_code == '0') & rs.next():
            rows.append(rs.get_row_data())
        df = pd.DataFrame(rows, columns=rs.fields)
        if len(df) > 0:
            for c in ["open", "high", "low", "close", "preclose", "volume", "amount"]:
                df[c] = df[c].astype(float)
            df["pctChg"] = df["pctChg"].astype(float) / 100.0
            price_frames.append(df)
    bs.logout()
    combined = pd.concat(price_frames, ignore_index=True)
    pivot_close = combined.pivot(index="date", columns="code", values="close")
    pivot_open = combined.pivot(index="date", columns="code", values="open")
    pivot_pct = combined.pivot(index="date", columns="code", values="pctChg")
    pivot_preclose = combined.pivot(index="date", columns="code", values="preclose")
    return pivot_close, pivot_open, pivot_pct, pivot_preclose

def hrp_weights(returns: pd.DataFrame) -> pd.Series:
    corr = returns.corr().fillna(0.0)
    cov = returns.cov().fillna(0.0)
    dist = np.sqrt((1.0 - corr) / 2.0).to_numpy(copy=True)
    np.fill_diagonal(dist, 0.0)
    condensed_dist = squareform(dist, checks=False)
    link = linkage(condensed_dist, method='single')
    
    def get_quasi_diag(linkage_mat):
        linkage_mat = linkage_mat.astype(int)
        num_items = linkage_mat.shape[0] + 1
        clusters = {i: [i] for i in range(num_items)}
        for i, row in enumerate(linkage_mat):
            clusters[num_items + i] = clusters[row[0]] + clusters[row[1]]
        return clusters[num_items * 2 - 2]
    
    sort_ix = get_quasi_diag(link)
    sorted_items = [returns.columns[i] for i in sort_ix]
    weights = pd.Series(1.0, index=sorted_items)
    clusters = [sorted_items]
    while len(clusters) > 0:
        new_clusters = []
        for cluster in clusters:
            if len(cluster) > 1:
                c1 = cluster[:len(cluster)//2]
                c2 = cluster[len(cluster)//2:]
                var1 = cov.loc[c1, c1].values.sum()
                var2 = cov.loc[c2, c2].values.sum()
                alpha = 1.0 - var1 / (var1 + var2) if (var1 + var2) > 0 else 0.5
                weights[c1] *= alpha
                weights[c2] *= (1.0 - alpha)
                new_clusters.extend([c1, c2])
        clusters = new_clusters
    return weights / weights.sum()

class RealisticBrokerAccount:
    """Simulates an actual brokerage cash trading account under A-Share rules."""
    def __init__(self, initial_cash: float):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        # positions: code -> {"total_shares": int, "settled_shares": int, "avg_cost": float}
        self.positions = {}
        self.total_tax_paid = 0.0
        self.total_comm_paid = 0.0
        self.total_trades_count = 0
        self.daily_nav = []
        
    def get_market_value(self, current_prices: dict[str, float]) -> float:
        val = 0.0
        for code, pos in self.positions.items():
            val += pos["total_shares"] * current_prices.get(code, pos["avg_cost"])
        return val

    def get_total_equity(self, current_prices: dict[str, float]) -> float:
        return self.cash + self.get_market_value(current_prices)

    def start_of_day_settlement(self):
        """T+1 rule: shares bought on t-1 become sellable today."""
        for code, pos in self.positions.items():
            pos["settled_shares"] = pos["total_shares"]

    def execute_sell(self, code: str, shares: int, price: float) -> bool:
        if code not in self.positions:
            return False
        pos = self.positions[code]
        available_shares = pos["settled_shares"]
        actual_shares = min(shares, available_shares)
        if actual_shares <= 0:
            return False
        
        gross_proceeds = actual_shares * price
        stamp_tax = gross_proceeds * SELL_STAMP_DUTY
        commission = max(5.0, gross_proceeds * SELL_COMMISSION)
        slippage = gross_proceeds * SELL_SLIPPAGE
        net_proceeds = gross_proceeds - stamp_tax - commission - slippage
        
        self.cash += net_proceeds
        self.total_tax_paid += stamp_tax
        self.total_comm_paid += (commission + slippage)
        self.total_trades_count += 1
        
        pos["total_shares"] -= actual_shares
        pos["settled_shares"] -= actual_shares
        if pos["total_shares"] == 0:
            del self.positions[code]
        return True

    def execute_buy(self, code: str, shares: int, price: float) -> bool:
        # A-share: must be 100-share board lot
        shares = (shares // 100) * 100
        if shares <= 0:
            return False
        
        gross_cost = shares * price
        commission = max(5.0, gross_cost * BUY_COMMISSION)
        slippage = gross_cost * BUY_SLIPPAGE
        total_required = gross_cost + commission + slippage
        
        # Check if cash is enough, reduce shares if needed
        while total_required > self.cash and shares >= 100:
            shares -= 100
            gross_cost = shares * price
            commission = max(5.0, gross_cost * BUY_COMMISSION)
            slippage = gross_cost * BUY_SLIPPAGE
            total_required = gross_cost + commission + slippage
            
        if shares < 100 or total_required > self.cash:
            return False
            
        self.cash -= total_required
        self.total_comm_paid += (commission + slippage)
        self.total_trades_count += 1
        
        if code not in self.positions:
            self.positions[code] = {
                "total_shares": shares,
                "settled_shares": 0,  # T+1: cannot sell today
                "avg_cost": price
            }
        else:
            pos = self.positions[code]
            new_tot = pos["total_shares"] + shares
            pos["avg_cost"] = (pos["total_shares"] * pos["avg_cost"] + gross_cost) / new_tot
            pos["total_shares"] = new_tot
            # settled_shares remains unchanged until next day
        return True


def run_simulation():
    print("=========================================================================")
    print(" 🚀 正在拉取真实 A 股行情，启动【多预算实盘模拟交易撮合实验】...")
    print("=========================================================================")
    close_df, open_df, pct_df, preclose_df = fetch_market_data("2026-01-01", "2026-09-13")
    dates = close_df.index.tolist()
    codes = close_df.columns.tolist()
    
    warmup = 20
    eval_dates = dates[warmup:]
    print(f"模拟时间窗口: {eval_dates[0]} 至 {eval_dates[-1]} (共 {len(eval_dates)} 个交易日)\n")

    # 4 Budgets: 50K (小散户), 200K (普通散户), 1M (中户/量化), 5M (大户)
    BUDGETS = [50_000, 200_000, 1_000_000, 5_000_000]
    accounts = {b: RealisticBrokerAccount(b) for b in BUDGETS}
    
    # Also create Equal Weight Buy & Hold accounts for comparison
    bh_accounts = {b: RealisticBrokerAccount(b) for b in BUDGETS}
    
    # Initial Day Buy & Hold allocation
    first_day_close = close_df.iloc[warmup].to_dict()
    for b, bh_acc in bh_accounts.items():
        alloc_per_stock = b / len(codes)
        for code, px in first_day_close.items():
            shares = int((alloc_per_stock / px) // 100) * 100
            if shares >= 100:
                bh_acc.execute_buy(code, shares, px)

    # Daily simulation loop
    TARGET_DAILY_VOL = 0.12 / np.sqrt(252)

    for t in range(warmup, len(dates)):
        curr_date = dates[t]
        hist_pct = pct_df.iloc[t-warmup:t]
        today_open = open_df.iloc[t].to_dict()
        today_close = close_df.iloc[t].to_dict()
        today_preclose = preclose_df.iloc[t].to_dict()

        # Step 1: Start of day settlement (T+1 unlocks)
        for acc in accounts.values():
            acc.start_of_day_settlement()
        for bh_acc in bh_accounts.values():
            bh_acc.start_of_day_settlement()

        # Step 2: AI Alpha Decision Model calculates target portfolio weights
        # Multifactor: 20-day trend + 3-day mean reversion
        mom20 = hist_pct.mean()
        vol20 = hist_pct.std().replace(0, 1e-4)
        sharpe_raw = mom20 / vol20
        rev3 = -hist_pct.iloc[-3:].mean() / vol20
        alpha = 0.7 * sharpe_raw + 0.3 * rev3
        
        selected = alpha[alpha > 0].index
        if len(selected) < 3:
            selected = alpha.nlargest(3).index
            
        sub_returns = hist_pct[selected]
        sub_weights = hrp_weights(sub_returns)
        
        port_hist_vol = hist_pct[selected].dot(sub_weights).std()
        leverage = min(1.0, TARGET_DAILY_VOL / port_hist_vol) if port_hist_vol > 1e-4 else 0.5
        
        target_weights = pd.Series(0.0, index=codes)
        target_weights[selected] = (sub_weights * leverage).clip(upper=0.25)
        
        # Step 3: Trade Execution for each budget account
        for b, acc in accounts.items():
            total_equity = acc.get_total_equity(today_open)
            
            # 3.1 First: Process SELLS (to free up cash)
            for code in list(acc.positions.keys()):
                current_shares = acc.positions[code]["total_shares"]
                target_w = target_weights.get(code, 0.0)
                px = today_open[code]
                target_shares = int((total_equity * target_w / px) // 100) * 100
                
                # Check limit down (cannot sell if open == limit down)
                limit_down = round(today_preclose[code] * 0.90, 2)
                if abs(px - limit_down) < 0.01:
                    continue  # Stuck at limit down
                    
                if target_shares < current_shares:
                    # Turnover deadband: only sell if weight difference > 2% or target is 0
                    current_w = (current_shares * px) / total_equity
                    if (current_w - target_w > 0.02) or (target_w == 0.0):
                        shares_to_sell = current_shares - target_shares
                        acc.execute_sell(code, shares_to_sell, px)

            # 3.2 Second: Process BUYS
            for code, target_w in target_weights.items():
                if target_w <= 0:
                    continue
                px = today_open[code]
                # Check limit up (cannot buy if open == limit up)
                limit_up = round(today_preclose[code] * 1.10, 2)
                if abs(px - limit_up) < 0.01:
                    continue  # Stuck at limit up
                    
                current_shares = acc.positions.get(code, {}).get("total_shares", 0)
                target_shares = int((total_equity * target_w / px) // 100) * 100
                
                if target_shares > current_shares:
                    current_w = (current_shares * px) / total_equity
                    if (target_w - current_w > 0.02):
                        shares_to_buy = target_shares - current_shares
                        acc.execute_buy(code, shares_to_buy, px)
                        
            # Record Daily NAV at market close
            day_end_equity = acc.get_total_equity(today_close)
            acc.daily_nav.append(day_end_equity)

        # Record Buy & Hold daily NAV
        for b, bh_acc in bh_accounts.items():
            bh_acc.daily_nav.append(bh_acc.get_total_equity(today_close))

    # Compile Evaluation Results
    print("=========================================================================================================")
    print(f" 📊 【不同预算下实盘模拟交易盈亏总表】({eval_dates[0]} 至 {eval_dates[-1]})")
    print("=========================================================================================================")
    results = []
    for b in BUDGETS:
        acc = accounts[b]
        bh = bh_accounts[b]
        
        final_equity = acc.daily_nav[-1]
        profit_rmb = final_equity - b
        ret_pct = (final_equity / b - 1.0) * 100
        
        # NAV series
        s = pd.Series(acc.daily_nav, index=eval_dates)
        daily_ret = s.pct_change().dropna()
        ann_vol = daily_ret.std() * np.sqrt(252) * 100
        cummax = s.cummax()
        max_dd = ((s - cummax) / cummax).min() * 100
        
        bh_final = bh.daily_nav[-1]
        bh_ret_pct = (bh_final / b - 1.0) * 100
        
        results.append({
            "初始预算": f"{b/10000:.0f} 万元",
            "AI实盘期末净资产": f"{final_equity:,.0f} 元",
            "AI净盈亏金额": f"{profit_rmb:+,.0f} 元",
            "AI实战收益率": f"{ret_pct:+.2f}%",
            "被动持有基准收益": f"{bh_ret_pct:+.2f}%",
            "实战超额 Alpha": f"{ret_pct - bh_ret_pct:+.2f}%",
            "最大回撤": f"{max_dd:.2f}%",
            "印花税+佣金损耗": f"{(acc.total_tax_paid + acc.total_comm_paid):,.0f} 元",
            "累计交易笔数": f"{acc.total_trades_count} 笔"
        })
    df_res = pd.DataFrame(results)
    print(df_res.to_string(index=False))
    print("=========================================================================================================\n")

if __name__ == "__main__":
    run_simulation()
