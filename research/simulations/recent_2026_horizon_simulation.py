#!/usr/bin/env python3
"""Recent 2026 Realistic Horizon Experiment: Short-Term (Weekly) vs Medium/Long-Term (Bi-Monthly).

Time window: 2026-01-01 to 2026-09-13 (Recent 7-8 months).
Objective Universe: Top 30 HS300 constituent stocks fetched directly via baostock.
Account: 1,000,000 RMB cash broker simulation with strict T+1 and fees.
"""
from __future__ import annotations

import baostock as bs
import pandas as pd
import numpy as np
from multi_budget_trading_simulation import RealisticBrokerAccount, hrp_weights

def get_universe(n_stocks: int = 30) -> list[str]:
    lg = bs.login()
    rs = bs.query_hs300_stocks()
    stocks = []
    while (rs.error_code == '0') & rs.next():
        stocks.append(rs.get_row_data())
    bs.logout()
    return pd.DataFrame(stocks, columns=rs.fields)['code'].head(n_stocks).tolist()

def fetch_data(codes: list[str]):
    lg = bs.login()
    price_frames = []
    for code in codes:
        rs = bs.query_history_k_data_plus(
            code,
            "date,code,open,high,low,close,preclose,volume,amount,pctChg",
            start_date="2025-10-01",  # Warmup from late 2025
            end_date="2026-09-13",
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
    bs.logout()
    
    combined = pd.concat(price_frames, ignore_index=True)
    pivot_close = combined.pivot(index="date", columns="code", values="close").ffill().bfill()
    pivot_open = combined.pivot(index="date", columns="code", values="open").ffill().bfill()
    pivot_pct = combined.pivot(index="date", columns="code", values="pctChg").fillna(0.0)
    pivot_preclose = combined.pivot(index="date", columns="code", values="preclose").ffill().bfill()
    return pivot_close, pivot_open, pivot_pct, pivot_preclose

def run():
    print("=========================================================================")
    print(" 🚀 正在拉取 2026 年最新近半年真实行情进行【短线(周频) vs 中长线(双月频)】盲测...")
    print("=========================================================================")
    codes = get_universe(30)
    close_df, open_df, pct_df, preclose_df = fetch_data(codes)
    
    dates = close_df.index.tolist()
    valid_codes = close_df.dropna(axis=1).columns.tolist()
    
    # 2026-01-01 cutoff
    start_2026_idx = next(i for i, d in enumerate(dates) if d >= "2026-01-01")
    eval_dates = dates[start_2026_idx:]
    print(f"评测区间: {eval_dates[0]} 至 {eval_dates[-1]} (共 {len(eval_dates)} 个交易日, 约 8 个月)")
    print(f"客观标的: 沪深 300 官方成分股 {len(valid_codes)} 只\n")

    INITIAL = 1_000_000.0
    acc_bench = RealisticBrokerAccount(INITIAL)
    acc_short = RealisticBrokerAccount(INITIAL)
    acc_midlong = RealisticBrokerAccount(INITIAL)

    # Initial Buy & Hold allocation on 2026 first trading day
    first_px = close_df.iloc[start_2026_idx].to_dict()
    alloc_per = INITIAL / len(valid_codes)
    for c in valid_codes:
        px = first_px[c]
        sh = int((alloc_per / px) // 100) * 100
        if sh >= 100:
            acc_bench.execute_buy(c, sh, px)

    for t in range(start_2026_idx, len(dates)):
        curr_date = dates[t]
        today_open = open_df.iloc[t].to_dict()
        today_close = close_df.iloc[t].to_dict()
        today_preclose = preclose_df.iloc[t].to_dict()

        acc_bench.start_of_day_settlement()
        acc_short.start_of_day_settlement()
        acc_midlong.start_of_day_settlement()

        # ----------------------------------------------------------
        # 策略 1: 短线周度波段 (每周一调仓一次，周期 5 天)
        # ----------------------------------------------------------
        if (t - start_2026_idx) % 5 == 0:
            hist_20 = pct_df.iloc[t-20:t]
            mom20 = hist_20.mean()
            vol20 = hist_20.std().replace(0, 1e-4)
            rev5 = -pct_df.iloc[t-5:t].mean() / vol20
            score_short = 0.6 * (mom20 / vol20) + 0.4 * rev5
            
            # Select top 5
            sel_s = score_short.nlargest(5).index
            sub_w_s = hrp_weights(hist_20[sel_s])
            tw_s = pd.Series(0.0, index=valid_codes)
            tw_s[sel_s] = sub_w_s.clip(upper=0.25)
            
            tot_s = acc_short.get_total_equity(today_open)
            # Sells
            for c in list(acc_short.positions.keys()):
                curr_sh = acc_short.positions[c]["total_shares"]
                target_w = tw_s.get(c, 0.0)
                px = today_open[c]
                t_sh = int((tot_s * target_w / px) // 100) * 100
                if (curr_sh * px / tot_s - target_w > 0.03) or (target_w == 0.0):
                    acc_short.execute_sell(c, curr_sh - t_sh, px)
            # Buys
            for c, target_w in tw_s.items():
                if target_w > 0:
                    px = today_open[c]
                    curr_sh = acc_short.positions.get(c, {}).get("total_shares", 0)
                    t_sh = int((tot_s * target_w / px) // 100) * 100
                    if (target_w - curr_sh * px / tot_s > 0.03):
                        acc_short.execute_buy(c, t_sh - curr_sh, px)

        # ----------------------------------------------------------
        # 策略 2: 中长线稳健波段 (每 40 个交易日调仓一次，约双月/季度调仓)
        # 选长期低波动、稳健趋势标的，极低换手
        # ----------------------------------------------------------
        if (t - start_2026_idx) % 40 == 0:
            hist_60 = pct_df.iloc[t-60:t]
            # Long-term Sharpe = 60-day return / 60-day vol
            mom60 = (1.0 + hist_60).prod() - 1.0
            vol60 = hist_60.std().replace(0, 1e-4)
            sharpe_long = mom60 / vol60
            
            sel_l = sharpe_long.nlargest(5).index
            sub_w_l = hrp_weights(hist_60[sel_l])
            tw_l = pd.Series(0.0, index=valid_codes)
            tw_l[sel_l] = sub_w_l.clip(upper=0.25)
            
            tot_l = acc_midlong.get_total_equity(today_open)
            # Sells
            for c in list(acc_midlong.positions.keys()):
                curr_sh = acc_midlong.positions[c]["total_shares"]
                target_w = tw_l.get(c, 0.0)
                px = today_open[c]
                t_sh = int((tot_l * target_w / px) // 100) * 100
                if (curr_sh * px / tot_l - target_w > 0.04) or (target_w == 0.0):
                    acc_midlong.execute_sell(c, curr_sh - t_sh, px)
            # Buys
            for c, target_w in tw_l.items():
                if target_w > 0:
                    px = today_open[c]
                    curr_sh = acc_midlong.positions.get(c, {}).get("total_shares", 0)
                    t_sh = int((tot_l * target_w / px) // 100) * 100
                    if (target_w - curr_sh * px / tot_l > 0.04):
                        acc_midlong.execute_buy(c, t_sh - curr_sh, px)

        # Record daily close NAV
        acc_bench.daily_nav.append(acc_bench.get_total_equity(today_close))
        acc_short.daily_nav.append(acc_short.get_total_equity(today_close))
        acc_midlong.daily_nav.append(acc_midlong.get_total_equity(today_close))

    def evaluate(acc, name):
        final_eq = acc.daily_nav[-1]
        profit = final_eq - INITIAL
        ret_pct = (final_eq / INITIAL - 1.0) * 100
        s = pd.Series(acc.daily_nav, index=eval_dates)
        max_dd = ((s - s.cummax()) / s.cummax()).min() * 100
        friction = acc.total_tax_paid + acc.total_comm_paid
        return {
            "投资策略方案": name,
            "期末总资产": f"{final_eq:,.0f} 元",
            "净盈亏金额": f"{profit:+,.0f} 元",
            "2026年净回报率": f"{ret_pct:+.2f}%",
            "最大回撤": f"{max_dd:.2f}%",
            "印花税+佣金损耗": f"{friction:,.0f} 元",
            "实际交易笔数": f"{acc.total_trades_count} 笔"
        }

    res = [
        evaluate(acc_bench, "1. 被动持有: 沪深300买入不折腾"),
        evaluate(acc_short, "2. 短线波段: 每周调仓轮动 (5日持有)"),
        evaluate(acc_midlong, "3. 中长线稳健: 双月/季度调仓 (40日持有)")
    ]
    df_res = pd.DataFrame(res)
    print("=========================================================================================================")
    print(f" 📊 【2026 年近 8 个月实操对比：短线(周频) vs 中长线(双月频)】")
    print("=========================================================================================================")
    print(df_res.to_string(index=False))
    print("=========================================================================================================\n")

if __name__ == "__main__":
    run()
