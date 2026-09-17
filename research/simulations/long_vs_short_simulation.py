#!/usr/bin/env python3
"""Long-Term vs Short-Term Horizon Simulation on Objective HS300 Constituents.

Strict Causality, Zero Cherry-Picking:
- Horizon: 2024-01-01 to 2026-09-13 (nearly 3 full years across different market regimes)
- Universe: 30 objective HS300 constituent stocks fetched directly via baostock.
- Strategy 1: Short-term Momentum/Reversal (Weekly / 5-day rebalance, high turnover)
- Strategy 2: Long-term Trend + Macro Regime Gate (Monthly / 20-day rebalance, 60-day trend, Macro cash filter)
- Benchmark: Equal-weight Buy & Hold
"""
from __future__ import annotations

import baostock as bs
import pandas as pd
import numpy as np
from multi_budget_trading_simulation import RealisticBrokerAccount, hrp_weights

def get_hs300_universe(n_stocks: int = 30) -> list[str]:
    lg = bs.login()
    rs = bs.query_hs300_stocks()
    stocks = []
    while (rs.error_code == '0') & rs.next():
        stocks.append(rs.get_row_data())
    bs.logout()
    df = pd.DataFrame(stocks, columns=rs.fields)
    return df['code'].head(n_stocks).tolist()

def fetch_multi_year_data(codes: list[str], start_date: str = "2024-01-01", end_date: str = "2026-09-13"):
    lg = bs.login()
    price_frames = []
    for code in codes:
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
            for c in ["open", "high", "low", "close", "preclose", "volume", "amount", "pctChg"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df["pctChg"] = df["pctChg"] / 100.0
            price_frames.append(df)
            
    # Also fetch HS300 Index (sh.000300) for Macro Regime Filter
    rs_idx = bs.query_history_k_data_plus(
        "sh.000300",
        "date,code,close",
        start_date=start_date,
        end_date=end_date,
        frequency="d",
        adjustflag="3"
    )
    idx_rows = []
    while (rs_idx.error_code == '0') & rs_idx.next():
        idx_rows.append(rs_idx.get_row_data())
    idx_df = pd.DataFrame(idx_rows, columns=rs_idx.fields)
    idx_df["close"] = pd.to_numeric(idx_df["close"], errors="coerce")
    bs.logout()

    combined = pd.concat(price_frames, ignore_index=True)
    pivot_close = combined.pivot(index="date", columns="code", values="close").ffill().bfill()
    pivot_open = combined.pivot(index="date", columns="code", values="open").ffill().bfill()
    pivot_pct = combined.pivot(index="date", columns="code", values="pctChg").fillna(0.0)
    pivot_preclose = combined.pivot(index="date", columns="code", values="preclose").ffill().bfill()
    
    idx_series = idx_df.set_index("date")["close"].ffill().bfill()
    return pivot_close, pivot_open, pivot_pct, pivot_preclose, idx_series

def run_simulation():
    print("=========================================================================")
    print(" 🚀 正在拉取 2024-2026 近 3 年真实行情，进行【长线 vs 短线因果律盲测】...")
    print("=========================================================================")
    codes = get_hs300_universe(30)
    close_df, open_df, pct_df, preclose_df, idx_close = fetch_multi_year_data(codes)
    
    valid_codes = close_df.dropna(axis=1).columns.tolist()
    dates = [d for d in close_df.index if d in idx_close.index]
    
    close_df = close_df.loc[dates, valid_codes]
    open_df = open_df.loc[dates, valid_codes]
    pct_df = pct_df.loc[dates, valid_codes]
    preclose_df = preclose_df.loc[dates, valid_codes]
    idx_close = idx_close.loc[dates]
    
    print(f"有效股票数量: {len(valid_codes)} 只, 交易日总跨度: {len(dates)} 天 ({dates[0]} 至 {dates[-1]})\n")

    # Warmup for long-term indicators: 60 trading days (approx 3 months)
    warmup = 60
    eval_dates = dates[warmup:]

    INITIAL_CAPITAL = 1_000_000.0
    acc_short = RealisticBrokerAccount(INITIAL_CAPITAL)
    acc_long = RealisticBrokerAccount(INITIAL_CAPITAL)
    acc_benchmark = RealisticBrokerAccount(INITIAL_CAPITAL)

    # Initial allocation for Benchmark
    first_px = close_df.iloc[warmup].to_dict()
    alloc_per_stock = INITIAL_CAPITAL / len(valid_codes)
    for c in valid_codes:
        px = first_px[c]
        sh = int((alloc_per_stock / px) // 100) * 100
        if sh >= 100:
            acc_benchmark.execute_buy(c, sh, px)

    # Simulation loop
    for t in range(warmup, len(dates)):
        curr_date = dates[t]
        today_open = open_df.iloc[t].to_dict()
        today_close = close_df.iloc[t].to_dict()
        today_preclose = preclose_df.iloc[t].to_dict()

        acc_short.start_of_day_settlement()
        acc_long.start_of_day_settlement()
        acc_benchmark.start_of_day_settlement()

        # ----------------------------------------------------
        # STRATEGY 1: SHORT-TERM (Weekly rebalance, 5-day cycle)
        # ----------------------------------------------------
        if (t - warmup) % 5 == 0:
            hist_pct_20 = pct_df.iloc[t-20:t]
            mom20 = hist_pct_20.mean()
            vol20 = hist_pct_20.std().replace(0, 1e-4)
            rev5 = -pct_df.iloc[t-5:t].mean() / vol20
            score_short = 0.6 * (mom20 / vol20) + 0.4 * rev5
            
            top_short = score_short.nlargest(5).index
            sub_w = hrp_weights(hist_pct_20[top_short])
            target_w_short = pd.Series(0.0, index=valid_codes)
            target_w_short[top_short] = sub_w.clip(upper=0.25)
            
            tot_eq_s = acc_short.get_total_equity(today_open)
            # Sells
            for c in list(acc_short.positions.keys()):
                curr_sh = acc_short.positions[c]["total_shares"]
                tw = target_w_short.get(c, 0.0)
                px = today_open[c]
                t_sh = int((tot_eq_s * tw / px) // 100) * 100
                if (curr_sh * px / tot_eq_s - tw > 0.03) or (tw == 0.0):
                    acc_short.execute_sell(c, curr_sh - t_sh, px)
            # Buys
            for c, tw in target_w_short.items():
                if tw > 0:
                    px = today_open[c]
                    curr_sh = acc_short.positions.get(c, {}).get("total_shares", 0)
                    t_sh = int((tot_eq_s * tw / px) // 100) * 100
                    if (tw - curr_sh * px / tot_eq_s > 0.03):
                        acc_short.execute_buy(c, t_sh - curr_sh, px)

        # ----------------------------------------------------
        # STRATEGY 2: LONG-TERM (Monthly rebalance, 20-day cycle)
        # Macro Regime Gate + 60-Day Low-Volatility Trend
        # ----------------------------------------------------
        if (t - warmup) % 20 == 0:
            # 1. Macro Regime Gate: Check HS300 Index vs 60-day moving average
            idx_hist_60 = idx_close.iloc[t-60:t]
            idx_ma60 = idx_hist_60.mean()
            idx_current = idx_hist_60.iloc[-1]
            is_bull_or_neutral = (idx_current >= idx_ma60 * 0.98) # Tolerance band
            
            tot_eq_l = acc_long.get_total_equity(today_open)
            
            if not is_bull_or_neutral:
                # BEAR REGIME: Clear all positions, hold 100% CASH to avoid market crash!
                for c in list(acc_long.positions.keys()):
                    curr_sh = acc_long.positions[c]["total_shares"]
                    acc_long.execute_sell(c, curr_sh, today_open[c])
            else:
                # BULL/RECOVERY REGIME: Pick long-term winners (60-day return / 60-day vol)
                hist_pct_60 = pct_df.iloc[t-60:t]
                mom60 = (1.0 + hist_pct_60).prod() - 1.0
                vol60 = hist_pct_60.std().replace(0, 1e-4) * np.sqrt(252)
                sharpe_long = mom60 / vol60
                
                # Pick top 5 resilient long-term leaders
                top_long = sharpe_long.nlargest(5).index
                sub_w_l = hrp_weights(hist_pct_60[top_long])
                target_w_long = pd.Series(0.0, index=valid_codes)
                target_w_long[top_long] = sub_w_l.clip(upper=0.25)
                
                # Sells
                for c in list(acc_long.positions.keys()):
                    curr_sh = acc_long.positions[c]["total_shares"]
                    tw = target_w_long.get(c, 0.0)
                    px = today_open[c]
                    t_sh = int((tot_eq_l * tw / px) // 100) * 100
                    if (curr_sh * px / tot_eq_l - tw > 0.04) or (tw == 0.0):
                        acc_long.execute_sell(c, curr_sh - t_sh, px)
                # Buys
                for c, tw in target_w_long.items():
                    if tw > 0:
                        px = today_open[c]
                        curr_sh = acc_long.positions.get(c, {}).get("total_shares", 0)
                        t_sh = int((tot_eq_l * tw / px) // 100) * 100
                        if (tw - curr_sh * px / tot_eq_l > 0.04):
                            acc_long.execute_buy(c, t_sh - curr_sh, px)

        # End of day valuation
        acc_short.daily_nav.append(acc_short.get_total_equity(today_close))
        acc_long.daily_nav.append(acc_long.get_total_equity(today_close))
        acc_benchmark.daily_nav.append(acc_benchmark.get_total_equity(today_close))

    # Compile Final Metrics
    def evaluate(acc, name):
        final_eq = acc.daily_nav[-1]
        profit = final_eq - INITIAL_CAPITAL
        ret_pct = (final_eq / INITIAL_CAPITAL - 1.0) * 100
        s = pd.Series(acc.daily_nav, index=eval_dates)
        max_dd = ((s - s.cummax()) / s.cummax()).min() * 100
        friction = acc.total_tax_paid + acc.total_comm_paid
        return {
            "策略": name,
            "期末总资产": f"{final_eq:,.0f} 元",
            "净盈亏": f"{profit:+,.0f} 元",
            "近3年总回报": f"{ret_pct:+.2f}%",
            "最大回撤": f"{max_dd:.2f}%",
            "印花税+佣金损耗": f"{friction:,.0f} 元",
            "交易笔数": f"{acc.total_trades_count} 笔"
        }

    res = [
        evaluate(acc_benchmark, "1. 被动基准: 沪深300买入持有"),
        evaluate(acc_short, "2. 短线高频: 周度动量反转 (5日调仓)"),
        evaluate(acc_long, "3. 长线价值趋势: 月度调仓 + 宏观避险门")
    ]
    df_res = pd.DataFrame(res)
    print("=========================================================================================================")
    print(f" 📊 【近 3 年 (2024~2026) 客观全池盲测：长线 vs 短线终极对决】")
    print("=========================================================================================================")
    print(df_res.to_string(index=False))
    print("=========================================================================================================\n")

if __name__ == "__main__":
    run_simulation()
