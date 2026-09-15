"""Entry point for the Causal Blind-Test Benchmark.

Executes the 6 pre-registered strategies on the point-in-time HS300 universe
with strict causality, broker fee/limit realism, and self-audit guards.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Guarantee local imports work regardless of CWD
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# Also ensure fin_skills is importable
FIN_SKILLS_ROOT = os.path.abspath(os.path.join(HERE, "../../.."))
if FIN_SKILLS_ROOT not in sys.path:
    sys.path.insert(0, FIN_SKILLS_ROOT)

from broker import BrokerAccount
from config import (BENCHMARK_INDEX, DATA_START, EVAL_END, EVAL_START,
                    INITIAL_CAPITAL, REBALANCE_DEADBAND, RISK_FREE_ANNUAL,
                    TRADING_DAYS, UNIVERSE_ASOF, UNIVERSE_SIZE, WARMUP_DAYS)
from data import load_index, load_prices, resolve_universe
from fin_skills.api.guards.ashare_rules import AShareRulesGuard
from fin_skills.api.guards.assert_causal import CausalityGuard
from fin_skills.api.guards.trial_ledger import TrialLedgerGuard
from fin_skills.core.trial_ledger import TrialLedger
from metrics import summarise
from strategies import all_strategies


def main():
    print("=" * 80)
    print(" 🚀 A-Share Causal Blind-Test Benchmark (A股因果盲测基准)")
    print("=" * 80)
    print(f"Point-in-Time As-Of Date : {UNIVERSE_ASOF}")
    print(f"Universe Size            : {UNIVERSE_SIZE} stocks (from HS300 on {UNIVERSE_ASOF})")
    print(f"Evaluation Window        : {EVAL_START} -> {EVAL_END}")
    print(f"Initial Capital          : {INITIAL_CAPITAL:,.2f} RMB per arm")
    print(f"Cash Risk-Free Rate      : {RISK_FREE_ANNUAL:.1%} annual (idle cash earns interest)")
    print("-" * 80)

    # 1. Resolve PIT Universe & Load Prices
    print("[1/5] Resolving point-in-time universe and loading daily price panels...")
    codes = resolve_universe(UNIVERSE_ASOF, UNIVERSE_SIZE)
    print(f"  -> Universe resolved ({len(codes)} stocks, e.g. {codes[:4]} ... {codes[-1]})")

    prices = load_prices(codes, start=DATA_START, end=EVAL_END)
    index_close = load_index(code=BENCHMARK_INDEX, start=DATA_START, end=EVAL_END)
    dates = prices["close"].index.tolist()
    print(f"  -> Market data loaded ({len(dates)} total daily bars from {dates[0]} to {dates[-1]})")

    eval_dates = [d for d in dates if EVAL_START <= d <= EVAL_END]
    if not eval_dates:
        raise RuntimeError(f"No dates found in evaluation window {EVAL_START} to {EVAL_END}")
    warmup_idx = dates.index(eval_dates[0])
    if warmup_idx < WARMUP_DAYS:
        raise RuntimeError(f"Insufficient warmup: {warmup_idx} bars available, {WARMUP_DAYS} required")
    print(f"  -> Warmup period: {warmup_idx} bars ({dates[0]} to {dates[warmup_idx-1]})")
    print(f"  -> Evaluation period: {len(eval_dates)} bars ({eval_dates[0]} to {eval_dates[-1]})")

    # 2. Pre-register all trials in TrialLedger BEFORE execution
    print("\n[2/5] Pre-registering strategy arms in TrialLedger (Pre-registration Protocol)...")
    results_dir = Path(HERE) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = results_dir / "trials.jsonl"
    if ledger_path.exists():
        ledger_path.unlink()  # fresh run for clean trial accounting
    ledger = TrialLedger(ledger_path)

    strategies = all_strategies()
    trial_ids = {}
    for strat in strategies:
        tid = ledger.record(
            strategy=strat.name,
            params={
                "description": strat.description,
                "universe_asof": UNIVERSE_ASOF,
                "universe_size": UNIVERSE_SIZE,
                "eval_start": EVAL_START,
                "eval_end": EVAL_END,
            },
            note="Pre-registered benchmark arm"
        )
        trial_ids[strat.name] = tid
        print(f"  -> Registered trial [{tid[:8]}] {strat.name:22s}: {strat.description[:45]}...")

    # 3. Daily Causality Simulation Loop
    print("\n[3/5] Simulating daily trading with strict causality & broker fee/limit checks...")
    accounts = {s.name: BrokerAccount(INITIAL_CAPITAL) for s in strategies}
    nav_history = {s.name: {} for s in strategies}
    daily_rf = (1.0 + RISK_FREE_ANNUAL) ** (1.0 / TRADING_DAYS) - 1.0

    for step, t in enumerate(range(warmup_idx, len(dates))):
        cur_date = dates[t]
        
        # Strict causality: data available before today's open is STRICTLY <= t-1 close!
        hist_close = prices["close"].iloc[:t]
        hist_idx = index_close.loc[hist_close.index]

        today_open = prices["open"].iloc[t].to_dict()
        today_preclose = prices["preclose"].iloc[t].to_dict()
        today_close = prices["close"].iloc[t].to_dict()

        for strat in strategies:
            acc = accounts[strat.name]
            acc.start_of_day_settlement()
            acc.accrue_cash_interest(daily_rf)

            if strat.trades_today(step):
                targets = strat.weights(step, hist_close, hist_idx)
                acc.rebalance_to(targets, today_open, today_preclose, deadband=REBALANCE_DEADBAND)

            # Mark to market at today's close
            equity = acc.equity(today_close)
            nav_history[strat.name][cur_date] = equity

    # 4. Metrics & Performance Summaries
    print("\n[4/5] Computing metrics, friction breakdowns, and multiple-testing statistics...")
    metrics_summary = {}
    nav_df = pd.DataFrame(nav_history)
    nav_df.to_csv(results_dir / "nav_series.csv", index_label="date")

    for strat in strategies:
        s_name = strat.name
        acc = accounts[s_name]
        s_nav = nav_df[s_name]
        m = summarise(s_nav, acc, len(eval_dates))
        metrics_summary[s_name] = {
            "name": s_name,
            "description": strat.description,
            "metrics": m,
        }
        # Complete the pre-registered trial in the ledger
        ledger.complete(
            trial_ids[s_name],
            metrics={
                "sharpe": m["sharpe_excess"],
                "n_obs": len(eval_dates),
                "total_return": m["total_return"],
                "cagr": m["cagr"],
                "max_drawdown": m["max_drawdown"],
                "friction_total": m["friction"]["total"],
                "turnover_annual": m["turnover_one_way_annual"],
            },
            note="Completed benchmark run"
        )

    # 5. Audit with fin-skills Guards
    print("\n[5/5] Executing fin-skills self-audit guards...")
    audit_results = {}

    # Guard 1: Causality Guard (Assert Causal on momentum & breadth indicators)
    sample_df = prices["close"].iloc[:200]
    causal_guard = CausalityGuard()
    
    def mom20_indicator(df: pd.DataFrame) -> pd.DataFrame:
        return df.pct_change(20)

    def breadth_indicator(df: pd.DataFrame) -> pd.Series:
        rolling_mean = df.rolling(50).mean()
        return (df > rolling_mean).mean(axis=1)

    c_outcome = causal_guard.check(
        fn={"momentum_20d": mom20_indicator, "breadth_50d": breadth_indicator},
        df=sample_df,
        k=150
    )
    def outcome_to_dict(outcome) -> dict:
        return {
            "passed": outcome.passed,
            "findings": [str(f) for f in outcome.findings],
            "errors": [f.message for f in outcome.findings if f.severity == "error"],
            "warnings": [f.message for f in outcome.findings if f.severity == "warning"],
            "info": [f.message for f in outcome.findings if f.severity == "info"],
        }

    audit_results["causality_guard"] = outcome_to_dict(c_outcome)
    print(f"  -> CausalityGuard: {'PASSED' if c_outcome.passed else 'FAILED'}")

    # Guard 2: Trial Ledger Guard (Deflated Sharpe on registered trials)
    trial_guard = TrialLedgerGuard()
    best_strat = max(strategies, key=lambda s: metrics_summary[s.name]["metrics"]["sharpe_excess"])
    best_sharpe = metrics_summary[best_strat.name]["metrics"]["sharpe_excess"]
    t_outcome = trial_guard.check(
        best_sharpe=best_sharpe,
        n_obs=len(eval_dates),
        ledger=ledger,
        alpha=0.05
    )
    audit_results["trial_ledger_guard"] = {
        "best_strategy": best_strat.name,
        "best_sharpe": best_sharpe,
        **outcome_to_dict(t_outcome),
    }
    print(f"  -> TrialLedgerGuard (best={best_strat.name}, Sharpe={best_sharpe:.3f}): {'PASSED' if t_outcome.passed else 'FAILED'}")

    # Save results.json and audit_report.json
    with open(results_dir / "results.json", "w") as fh:
        json.dump(metrics_summary, fh, indent=2)
    with open(results_dir / "audit_report.json", "w") as fh:
        json.dump(audit_results, fh, indent=2, default=str)

    # Print Formatted Results Table
    print("\n" + "=" * 96)
    print(f"{'Strategy Arm':24s} | {'Return':8s} | {'CAGR':7s} | {'MaxDD':8s} | {'ExcessSh':8s} | {'Friction':10s} | {'Turnover/yr':11s}")
    print("-" * 96)
    for s in strategies:
        m = metrics_summary[s.name]["metrics"]
        print(f"{s.name:24s} | {m['total_return']:+7.2%} | {m['cagr']:+6.2%} | {m['max_drawdown']:7.2%} | {m['sharpe_excess']:8.3f} | {m['friction']['total']:10,.0f} | {m['turnover_one_way_annual']:10.2f}x")
    print("=" * 96)

    # Print Friction Breakdown Table
    print("\n" + "=" * 96)
    print(f"{'Strategy Arm':24s} | {'Stamp Duty':12s} | {'Commission':12s} | {'Slippage':12s} | {'Total Friction':14s} | {'Friction/Cap':12s}")
    print("-" * 96)
    for s in strategies:
        f = metrics_summary[s.name]["metrics"]["friction"]
        print(f"{s.name:24s} | {f['stamp_duty']:12,.2f} | {f['commission']:12,.2f} | {f['slippage']:12,.2f} | {f['total']:14,.2f} | {f['pct_of_initial']:12.2%}")
    print("=" * 96)


if __name__ == "__main__":
    main()
