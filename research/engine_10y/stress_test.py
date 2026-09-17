"""10-Year (2015-2026) Comprehensive Stress Testing & Comparative Simulation.

Runs full 10-year backtests across 5 models, isolates 4 historical crisis periods,
and records drawdowns, Calmar ratios, and friction metrics.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from data_loader import fetch_10y_etf_data
from models import BrokerSim, get_all_models

TRADING_DAYS = 252
RISK_FREE_ANNUAL = 0.02

# 4 Historical Black Swan / Crisis Periods
CRISIS_PERIODS = {
    "2015_crash_and_circuit_breaker": {
        "name": "2015 股灾与千股跌停熔断",
        "start": "2015-06-12",
        "end": "2016-01-29",
        "desc": "A股5178点见顶，经历三次去杠杆踩踏暴跌与2016年1月熔断测试",
    },
    "2018_us_china_trade_war": {
        "name": "2018 中美贸易战单边大熊市",
        "start": "2018-01-29",
        "end": "2018-12-28",
        "desc": "中美贸易摩擦全面爆发，去杠杆叠加外部冲击，A股全年单边阴跌至2440点",
    },
    "2020_covid_liquidity_shock": {
        "name": "2020 疫情全球流动性闪崩",
        "start": "2020-01-20",
        "end": "2020-03-23",
        "desc": "国内疫情暴发开盘千股跌停，美股10天熔断4次，全球大类资产流动性无差别抛售",
    },
    "2022_fed_tightening_stagflation": {
        "name": "2022 美联储激进加息与股债双杀",
        "start": "2022-01-04",
        "end": "2022-10-31",
        "desc": "俄乌冲突引发全球通胀危机，美联储单次加息75bps，全球权益与债券同时大幅杀估值",
    },
}


def run_10y_simulation():
    print("=" * 88)
    print(" 🚀 启动 10 年（2015-2026）全球大类资产穿越牛熊极端压力测试")
    print("=" * 88)
    data = fetch_10y_etf_data()
    open_df = data["open"]
    close_df = data["close"]
    dates = data["dates"]
    print(f"历史数据跨度: {dates[0]} 至 {dates[-1]} (共 {len(dates)} 个交易日, ~11.7 年)")

    models = get_all_models()
    initial_cap = 1_000_000.0
    accounts = {m.name: BrokerSim(initial_cap) for m in models}
    nav_series = {m.name: {} for m in models}

    daily_rf = (1.0 + RISK_FREE_ANNUAL) ** (1.0 / TRADING_DAYS) - 1.0

    # 1. Main Simulation Loop
    for step in range(len(dates)):
        cur_date = dates[step]
        today_open = open_df.iloc[step].to_dict()
        today_close = close_df.iloc[step].to_dict()
        hist_close = close_df.iloc[:step]  # strict causality: up to step-1

        for m in models:
            acc = accounts[m.name]
            acc.accrue_interest(daily_rf)

            # Rebalance decision
            if step == 0 or m.should_rebalance(step):
                targets = m.get_target_weights(step, hist_close)
                acc.rebalance(targets, today_open, deadband=0.03)

            # Mark to market at close
            eq = acc.equity(today_close)
            nav_series[m.name][cur_date] = eq

    nav_df = pd.DataFrame(nav_series)
    nav_df.to_csv(HERE / "nav_10y.csv", index_label="date")
    print("  -> 10年每日净值矩阵已落盘至 nav_10y.csv")

    # 2. Compute 10-Year Full Period Metrics
    metrics_10y = {}
    years = len(dates) / TRADING_DAYS

    for m in models:
        s = nav_df[m.name]
        rets = s.pct_change().dropna()
        tot_ret = float(s.iloc[-1] / initial_cap - 1.0)
        cagr = float((s.iloc[-1] / initial_cap) ** (1.0 / years) - 1.0)
        vol = float(rets.std() * np.sqrt(TRADING_DAYS))
        cummax = s.cummax()
        max_dd = float(((s - cummax) / cummax).min())
        excess_rets = rets - daily_rf
        sharpe = float(excess_rets.mean() / excess_rets.std() * np.sqrt(TRADING_DAYS)) if excess_rets.std() > 0 else 0.0
        calmar = float(cagr / abs(max_dd)) if max_dd != 0 else 0.0
        acc = accounts[m.name]

        metrics_10y[m.name] = {
            "name": m.name,
            "description": m.description,
            "initial_capital": initial_cap,
            "final_equity": float(s.iloc[-1]),
            "total_return": tot_ret,
            "cagr": cagr,
            "annual_vol": vol,
            "max_drawdown": max_dd,
            "excess_sharpe": sharpe,
            "calmar_ratio": calmar,
            "friction_total": acc.friction_paid,
            "friction_pct": acc.friction_paid / initial_cap,
            "trades_count": acc.n_trades,
        }

    # 3. Compute Segmented Stress Test Results
    stress_results = {}
    for c_key, c_info in CRISIS_PERIODS.items():
        c_start, c_end = c_info["start"], c_info["end"]
        sub_dates = [d for d in dates if c_start <= d <= c_end]
        if not sub_dates:
            continue
        c_nav = nav_df.loc[sub_dates]
        period_metrics = {}
        for m in models:
            series = c_nav[m.name]
            p_ret = float(series.iloc[-1] / series.iloc[0] - 1.0)
            p_cummax = series.cummax()
            p_maxdd = float(((series - p_cummax) / p_cummax).min())
            period_metrics[m.name] = {
                "period_return": p_ret,
                "period_maxdd": p_maxdd,
            }
        stress_results[c_key] = {
            "meta": c_info,
            "trading_days": len(sub_dates),
            "metrics": period_metrics,
        }

    # Save to JSON
    with open(HERE / "metrics_10y.json", "w", encoding="utf-8") as f:
        json.dump(metrics_10y, f, indent=2, ensure_ascii=False)
    with open(HERE / "stress_tests.json", "w", encoding="utf-8") as f:
        json.dump(stress_results, f, indent=2, ensure_ascii=False)

    # 4. Display 10-Year Full Performance Table
    print("\n" + "=" * 105)
    print(f" 📊 【10年全周期跨越牛熊表现总表】(2015-01-05 至 2026-09-11, 2843 个交易日)")
    print("=" * 105)
    print(f"{'策略模型':22s} | {'累计收益':9s} | {'年化CAGR':8s} | {'最大回撤':8s} | {'年化波动':8s} | {'超额夏普':8s} | {'卡玛比率':8s} | {'总手续费':9s}")
    print("-" * 105)
    for m in models:
        r = metrics_10y[m.name]
        print(f"{m.name:22s} | {r['total_return']:+8.2%} | {r['cagr']:+7.2%} | {r['max_drawdown']:7.2%} | {r['annual_vol']:7.2%} | {r['excess_sharpe']:8.3f} | {r['calmar_ratio']:8.2f} | {r['friction_total']:8.0f}元")
    print("=" * 105)

    # 5. Display Stress Testing Crisis Matrix
    print("\n" + "=" * 105)
    print(" 🌩️ 【四大历史黑天鹅极端压力测试矩阵】")
    print("=" * 105)
    for c_key, c_data in stress_results.items():
        meta = c_data["meta"]
        print(f"\n▶ 危机情景: {meta['name']} ({meta['start']} 至 {meta['end']}, 共 {c_data['trading_days']} 交易日)")
        print(f"  背景说明: {meta['desc']}")
        print(f"  {'-' * 80}")
        print(f"  {'策略分支':24s} | {'危机期区间收益':14s} | {'危机期最大回撤':14s}")
        print(f"  {'-' * 80}")
        for m in models:
            pm = c_data["metrics"][m.name]
            print(f"  {m.name:24s} | {pm['period_return']:+13.2%} | {pm['period_maxdd']:13.2%}")

    print("\n" + "=" * 105)


if __name__ == "__main__":
    run_10y_simulation()
