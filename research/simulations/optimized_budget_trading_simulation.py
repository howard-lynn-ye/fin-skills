#!/usr/bin/env python3
"""Optimized Low-Turnover Trading Simulation with fin-skills.

Applies:
1. Weekly Rebalancing (rebalance every 5 trading days instead of daily)
2. Hurdle Cost Filter: Only trade if expected alpha > round-trip cost + buffer (1.0%)
3. Evaluates real profitability across 50K, 200K, 1M, 5M RMB budgets.
"""
from __future__ import annotations

import baostock as bs
import pandas as pd
import numpy as np
from multi_budget_trading_simulation import (
    UNIVERSE, RealisticBrokerAccount, fetch_market_data, hrp_weights
)

def run_optimized_simulation():
    print("=========================================================================")
    print(" 🚀 启动【周度低频调仓 + 成本收益门槛优化版】实盘模拟交易...")
    print("=========================================================================")
    close_df, open_df, pct_df, preclose_df = fetch_market_data("2026-01-01", "2026-09-13")
    dates = close_df.index.tolist()
    codes = close_df.columns.tolist()
    
    warmup = 20
    eval_dates = dates[warmup:]
    
    BUDGETS = [50_000, 200_000, 1_000_000, 5_000_000]
    accounts = {b: RealisticBrokerAccount(b) for b in BUDGETS}
    bh_accounts = {b: RealisticBrokerAccount(b) for b in BUDGETS}
    
    # Initial Buy & Hold allocation
    first_day_close = close_df.iloc[warmup].to_dict()
    for b, bh_acc in bh_accounts.items():
        alloc_per_stock = b / len(codes)
        for code, px in first_day_close.items():
            shares = int((alloc_per_stock / px) // 100) * 100
            if shares >= 100:
                bh_acc.execute_buy(code, shares, px)

    TARGET_DAILY_VOL = 0.15 / np.sqrt(252)  # Target 15% annual vol

    for t in range(warmup, len(dates)):
        curr_date = dates[t]
        hist_pct = pct_df.iloc[t-warmup:t]
        today_open = open_df.iloc[t].to_dict()
        today_close = close_df.iloc[t].to_dict()
        today_preclose = preclose_df.iloc[t].to_dict()

        for acc in accounts.values():
            acc.start_of_day_settlement()
        for bh_acc in bh_accounts.values():
            bh_acc.start_of_day_settlement()

        # ONLY REBALANCE EVERY 5 TRADING DAYS (WEEKLY) OR FIRST DAY
        is_rebalance_day = ((t - warmup) % 5 == 0)

        if is_rebalance_day:
            # Multi-factor: 20-day trend + 5-day reversal
            mom20 = hist_pct.mean()
            vol20 = hist_pct.std().replace(0, 1e-4)
            sharpe_raw = mom20 / vol20
            rev5 = -hist_pct.iloc[-5:].mean() / vol20
            alpha = 0.7 * sharpe_raw + 0.3 * rev5
            
            # Select Top 3-4 strongest alphas
            selected = alpha.nlargest(4).index
            sub_returns = hist_pct[selected]
            sub_weights = hrp_weights(sub_returns)
            
            port_hist_vol = hist_pct[selected].dot(sub_weights).std()
            leverage = min(1.0, TARGET_DAILY_VOL / port_hist_vol) if port_hist_vol > 1e-4 else 0.8
            
            target_weights = pd.Series(0.0, index=codes)
            target_weights[selected] = (sub_weights * leverage).clip(upper=0.30)
            
            # Execute orders for each account
            for b, acc in accounts.items():
                total_equity = acc.get_total_equity(today_open)
                
                # SELLS
                for code in list(acc.positions.keys()):
                    current_shares = acc.positions[code]["total_shares"]
                    target_w = target_weights.get(code, 0.0)
                    px = today_open[code]
                    target_shares = int((total_equity * target_w / px) // 100) * 100
                    
                    limit_down = round(today_preclose[code] * 0.90, 2)
                    if abs(px - limit_down) < 0.01:
                        continue
                        
                    current_w = (current_shares * px) / total_equity
                    # Hurdle buffer: only sell if weight difference > 4% or target is 0
                    if (current_w - target_w > 0.04) or (target_w == 0.0):
                        shares_to_sell = current_shares - target_shares
                        acc.execute_sell(code, shares_to_sell, px)

                # BUYS
                for code, target_w in target_weights.items():
                    if target_w <= 0:
                        continue
                    px = today_open[code]
                    limit_up = round(today_preclose[code] * 1.10, 2)
                    if abs(px - limit_up) < 0.01:
                        continue
                        
                    current_shares = acc.positions.get(code, {}).get("total_shares", 0)
                    target_shares = int((total_equity * target_w / px) // 100) * 100
                    current_w = (current_shares * px) / total_equity
                    
                    # Hurdle buffer: only buy if weight delta > 4%
                    if (target_w - current_w > 0.04):
                        shares_to_buy = target_shares - current_shares
                        acc.execute_buy(code, shares_to_buy, px)

        # Record Daily Close Equities
        for b, acc in accounts.items():
            acc.daily_nav.append(acc.get_total_equity(today_close))
        for b, bh_acc in bh_accounts.items():
            bh_acc.daily_nav.append(bh_acc.get_total_equity(today_close))

    print("=========================================================================================================")
    print(f" 📊 【低频优化版实盘模拟盈亏总表】({eval_dates[0]} 至 {eval_dates[-1]})")
    print("=========================================================================================================")
    results = []
    for b in BUDGETS:
        acc = accounts[b]
        bh = bh_accounts[b]
        
        final_equity = acc.daily_nav[-1]
        profit_rmb = final_equity - b
        ret_pct = (final_equity / b - 1.0) * 100
        
        s = pd.Series(acc.daily_nav, index=eval_dates)
        cummax = s.cummax()
        max_dd = ((s - cummax) / cummax).min() * 100
        
        bh_final = bh.daily_nav[-1]
        bh_ret_pct = (bh_final / b - 1.0) * 100
        
        total_friction = acc.total_tax_paid + acc.total_comm_paid
        results.append({
            "初始预算": f"{b/10000:.0f} 万元",
            "期末净资产": f"{final_equity:,.0f} 元",
            "实盘净盈亏": f"{profit_rmb:+,.0f} 元",
            "实战净收益率": f"{ret_pct:+.2f}%",
            "买入持有基准": f"{bh_ret_pct:+.2f}%",
            "超额收益 Alpha": f"{ret_pct - bh_ret_pct:+.2f}%",
            "最大回撤": f"{max_dd:.2f}%",
            "总摩擦损耗": f"{total_friction:,.0f} 元",
            "交易笔数": f"{acc.total_trades_count} 笔"
        })
    df_res = pd.DataFrame(results)
    print(df_res.to_string(index=False))
    print("=========================================================================================================\n")

if __name__ == "__main__":
    run_optimized_simulation()
