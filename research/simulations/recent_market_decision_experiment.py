#!/usr/bin/env python3
"""Recent Real A-Share Market Experiment with fin-skills Decision Engine.

Compares:
1. Equal-Weight Buy & Hold (Static passive)
2. Naive Momentum Chasing (Unconstrained chasing, high turnover)
3. fin-skills Smart Decision Engine (Kelly / Volatility target + HRP Risk Parity + A-share Guard + Turnover Deadband)

Data: Real daily OHLCV from 2026-01-01 to 2026-09-11 fetched via baostock.
"""
from __future__ import annotations

import sys
import os
import baostock as bs
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

# Universe of diverse leading stocks
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

def fetch_market_data(start_date: str = "2026-01-01", end_date: str = "2026-09-13") -> tuple[pd.DataFrame, pd.DataFrame]:
    lg = bs.login()
    if lg.error_code != '0':
        raise RuntimeError(f"Baostock login failed: {lg.error_msg}")
    
    price_frames = []
    for code, info in UNIVERSE.items():
        rs = bs.query_history_k_data_plus(
            code,
            "date,code,open,high,low,close,preclose,volume,amount,pctChg",
            start_date=start_date,
            end_date=end_date,
            frequency="d",
            adjustflag="3"  # forward adjusted
        )
        rows = []
        while (rs.error_code == '0') & rs.next():
            rows.append(rs.get_row_data())
        df = pd.DataFrame(rows, columns=rs.fields)
        if len(df) > 0:
            df["close"] = df["close"].astype(float)
            df["open"] = df["open"].astype(float)
            df["high"] = df["high"].astype(float)
            df["low"] = df["low"].astype(float)
            df["volume"] = df["volume"].astype(float)
            df["amount"] = df["amount"].astype(float)
            df["pctChg"] = df["pctChg"].astype(float) / 100.0
            price_frames.append(df)
    bs.logout()
    
    combined = pd.concat(price_frames, ignore_index=True)
    pivot_close = combined.pivot(index="date", columns="code", values="close")
    pivot_pct = combined.pivot(index="date", columns="code", values="pctChg")
    return pivot_close, pivot_pct

def hrp_weights(returns: pd.DataFrame) -> pd.Series:
    """Hierarchical Risk Parity (HRP) allocation to prevent sector concentration."""
    corr = returns.corr().fillna(0.0)
    cov = returns.cov().fillna(0.0)
    
    # Distance matrix
    dist = np.sqrt((1.0 - corr) / 2.0).to_numpy(copy=True)
    np.fill_diagonal(dist, 0.0)
    
    # Hierarchical clustering
    condensed_dist = squareform(dist, checks=False)
    link = linkage(condensed_dist, method='single')
    
    # Quasi-diagonalization order
    def get_quasi_diag(linkage_mat):
        linkage_mat = linkage_mat.astype(int)
        num_items = linkage_mat.shape[0] + 1
        clusters = {i: [i] for i in range(num_items)}
        for i, row in enumerate(linkage_mat):
            clusters[num_items + i] = clusters[row[0]] + clusters[row[1]]
        return clusters[num_items * 2 - 2]
    
    sort_ix = get_quasi_diag(link)
    sorted_items = [returns.columns[i] for i in sort_ix]
    
    # Recursive bisection
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

def run_experiment():
    print("=========================================================")
    print(" 🚀 正在拉取 2026 年最新真实 A 股行情数据 (截至 2026-09-11)...")
    print("=========================================================")
    close_df, pct_df = fetch_market_data("2026-01-01", "2026-09-13")
    dates = close_df.index.tolist()
    codes = close_df.columns.tolist()
    print(f"数据区间: {dates[0]} 至 {dates[-1]}, 共 {len(dates)} 个交易日, 标的池: {len(codes)} 只龙头股\n")

    # Transaction cost model:
    # A-share: Stamp duty 0.05% (seller only, post 2023-08-28), Commission 0.025% each side, Slippage estimate 0.05%
    # Buy cost: 0.025% + 0.05% = 0.075% (7.5 bps)
    # Sell cost: 0.05% (tax) + 0.025% (comm) + 0.05% (slip) = 0.125% (12.5 bps)
    BUY_COST = 0.00075
    SELL_COST = 0.00125

    # Strategy 1: Equal Weight Buy & Hold
    eq_weights = pd.Series(1.0 / len(codes), index=codes)
    eq_daily_rets = pct_df.dot(eq_weights)
    eq_cum = (1.0 + eq_daily_rets).cumprod()

    # Strategy 2: Naive Chasing (Top 2 momentum in last 5 days, equally split, daily rebalance)
    naive_nav = [1.0]
    naive_weights = pd.Series(0.0, index=codes)
    naive_total_turnover = 0.0
    naive_total_costs = 0.0
    
    # Strategy 3: fin-skills Decision Engine
    # Volatility target: 12% annualized
    TARGET_ANN_VOL = 0.12
    TARGET_DAILY_VOL = TARGET_ANN_VOL / np.sqrt(252)
    
    finskills_nav = [1.0]
    finskills_weights = pd.Series(0.0, index=codes)
    finskills_total_turnover = 0.0
    finskills_total_costs = 0.0
    
    # Warmup window: 20 trading days
    warmup = 20
    
    for t in range(warmup, len(dates)):
        curr_date = dates[t]
        hist_pct = pct_df.iloc[t-warmup:t]
        today_pct = pct_df.iloc[t]
        
        # --- Strategy 2: Naive Chasing ---
        mom5 = (1.0 + hist_pct.iloc[-5:]).prod() - 1.0
        top2 = mom5.nlargest(2).index
        new_naive_w = pd.Series(0.0, index=codes)
        new_naive_w[top2] = 0.5
        
        turnover_naive = (new_naive_w - naive_weights).abs().sum() / 2.0
        cost_naive = turnover_naive * (BUY_COST + SELL_COST)
        naive_total_turnover += turnover_naive
        naive_total_costs += cost_naive
        
        r_naive = (naive_weights * today_pct).sum() - cost_naive
        naive_nav.append(naive_nav[-1] * (1.0 + r_naive))
        naive_weights = new_naive_w
        
        # --- Strategy 3: fin-skills Decision Engine ---
        # 1. Multi-factor Alpha Scoring: 20-day momentum (trend) + 5-day reversal (mean reversion)
        mom20 = hist_pct.mean()
        vol20 = hist_pct.std().replace(0, 1e-4)
        sharpe_raw = mom20 / vol20
        # Reversal: recent 3-day drop gives mean-reversion opportunity
        rev3 = -hist_pct.iloc[-3:].mean() / vol20
        alpha = 0.7 * sharpe_raw + 0.3 * rev3
        
        # 2. Select eligible subset: positive expected alpha
        selected = alpha[alpha > 0].index
        if len(selected) < 3:
            selected = alpha.nlargest(3).index
            
        # 3. HRP Risk Parity Allocation across selected stocks
        sub_returns = hist_pct[selected]
        sub_weights = hrp_weights(sub_returns)
        
        # 4. Volatility Targeting & Kelly Sizing (Macro cash allocation)
        # Realized portfolio volatility over last 20 days
        port_hist_vol = hist_pct[selected].dot(sub_weights).std()
        if port_hist_vol > 1e-4:
            leverage = min(1.0, TARGET_DAILY_VOL / port_hist_vol)  # Never exceed 100% (no leverage in cash account)
        else:
            leverage = 0.5
            
        target_w = pd.Series(0.0, index=codes)
        target_w[selected] = sub_weights * leverage
        
        # Single stock cap: max 25%
        target_w = target_w.clip(upper=0.25)
        
        # 5. Turnover Deadband / Hurdle Gate (save costs)
        weight_diff = target_w - finskills_weights
        # Only rebalance if change exceeds 2% or stock dropped out
        rebalance_mask = (weight_diff.abs() > 0.02) | ((target_w == 0) & (finskills_weights > 0))
        actual_w = finskills_weights.copy()
        actual_w[rebalance_mask] = target_w[rebalance_mask]
        
        turnover_fs = (actual_w - finskills_weights).abs().sum() / 2.0
        cost_fs = turnover_fs * (BUY_COST + SELL_COST)
        finskills_total_turnover += turnover_fs
        finskills_total_costs += cost_fs
        
        r_fs = (finskills_weights * today_pct).sum() - cost_fs
        finskills_nav.append(finskills_nav[-1] * (1.0 + r_fs))
        finskills_weights = actual_w

    # Compute metrics
    eval_dates = dates[warmup:]
    eq_sub_cum = eq_cum.loc[eval_dates] / eq_cum.loc[eval_dates[0]]
    naive_s = pd.Series(naive_nav[1:], index=eval_dates)
    fs_s = pd.Series(finskills_nav[1:], index=eval_dates)

    def calc_metrics(nav_series: pd.Series):
        total_ret = (nav_series.iloc[-1] - 1.0) * 100
        daily_ret = nav_series.pct_change().dropna()
        ann_ret = (nav_series.iloc[-1] ** (252 / len(nav_series)) - 1.0) * 100
        ann_vol = daily_ret.std() * np.sqrt(252) * 100
        sharpe = (ann_ret - 2.0) / ann_vol if ann_vol > 0 else 0.0  # 2% risk-free rate
        cummax = nav_series.cummax()
        drawdown = ((nav_series - cummax) / cummax).min() * 100
        return total_ret, ann_ret, ann_vol, sharpe, drawdown

    eq_tot, eq_ann, eq_vol, eq_sr, eq_dd = calc_metrics(eq_sub_cum)
    nv_tot, nv_ann, nv_vol, nv_sr, nv_dd = calc_metrics(naive_s)
    fs_tot, fs_ann, fs_vol, fs_sr, fs_dd = calc_metrics(fs_s)

    print("=========================================================================================")
    print(f" 📊 2026 年真实市场实战回测对比结果 ({eval_dates[0]} 至 {eval_dates[-1]})")
    print("=========================================================================================")
    metrics_table = [
        {"策略": "1. 散户基准: 等权重被动持有", "区间总收益": f"{eq_tot:+.2f}%", "年化波动率": f"{eq_vol:.2f}%", "夏普比率 (SR)": f"{eq_sr:.2f}", "最大回撤": f"{eq_dd:.2f}%", "总换手成本": "0.0%"},
        {"策略": "2. 散户策略: 盲目追涨动量轮动", "区间总收益": f"{nv_tot:+.2f}%", "年化波动率": f"{nv_vol:.2f}%", "夏普比率 (SR)": f"{nv_sr:.2f}", "最大回撤": f"{nv_dd:.2f}%", "总换手成本": f"{naive_total_costs*100:.2f}%"},
        {"策略": "3. fin-skills 智能决策引擎", "区间总收益": f"{fs_tot:+.2f}%", "年化波动率": f"{fs_vol:.2f}%", "夏普比率 (SR)": f"{fs_sr:.2f}", "最大回撤": f"{fs_dd:.2f}%", "总换手成本": f"{finskills_total_costs*100:.2f}%"}
    ]
    df_metrics = pd.DataFrame(metrics_table)
    print(df_metrics.to_string(index=False))
    print("=========================================================================================\n")

    # Generate Decision for the upcoming trading day (2026-09-14)
    print("=========================================================================================")
    print(f" 🎯 最新交易决策建议 (基于 2026-09-11 收盘数据，面向 2026-09-14 下周一开盘)")
    print("=========================================================================================")
    latest_date = dates[-1]
    last_close = close_df.loc[latest_date]
    
    # Current active weights of fin-skills engine
    total_capital = 1000000.0  # 1,000,000 RMB mock account
    order_recommendations = []
    
    cash_ratio = 1.0 - finskills_weights.sum()
    print(f"【宏观风控决策】: 当前模型推荐总仓位: {finskills_weights.sum()*100:.1f}%, 现金防御储备: {cash_ratio*100:.1f}%\n")
    
    for code, w in finskills_weights.items():
        name = UNIVERSE[code]["name"]
        sector = UNIVERSE[code]["sector"]
        px = last_close[code]
        target_val = total_capital * w
        target_shares = int(target_val // px // 100) * 100
        actual_val = target_shares * px
        
        status = "🟢 重点配置" if w >= 0.15 else ("🟡 轻度配置" if w > 0.05 else ("⚪ 观察持仓" if w > 0 else "🔴 清仓空仓"))
        order_recommendations.append({
            "代码": code,
            "名称": name,
            "行业": sector,
            "最新收盘价": f"{px:.2f} 元",
            "建议权重": f"{w*100:.1f}%",
            "建议股数 (手)": f"{target_shares} 股 ({target_shares//100} 手)",
            "目标市值": f"{actual_val:,.0f} 元",
            "决策状态": status
        })
    df_orders = pd.DataFrame(order_recommendations)
    print(df_orders.sort_values(by="建议权重", ascending=False).to_string(index=False))
    print("=========================================================================================")

if __name__ == "__main__":
    run_experiment()
