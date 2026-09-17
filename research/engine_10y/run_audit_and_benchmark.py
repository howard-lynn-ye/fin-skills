"""10-Year Benchmark Audit Script integrating fin-skills guards.

Guards Executed:
  1. CausalityGuard (Assert Causal on 60d realized vol & momentum)
  2. TrialLedgerGuard (Deflated Sharpe Ratio on registered 10-year arms)
  3. CostCurveGuard (Cost sensitivity & breakeven bps under ETF friction)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
FIN_SKILLS_ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(FIN_SKILLS_ROOT) not in sys.path:
    sys.path.insert(0, str(FIN_SKILLS_ROOT))

from data_loader import fetch_10y_etf_data
from fin_skills.api.guards.assert_causal import CausalityGuard
from fin_skills.api.guards.cost_curve import CostCurveGuard
from fin_skills.api.guards.trial_ledger import TrialLedgerGuard
from fin_skills.core.trial_ledger import TrialLedger
from models import BrokerSim, get_all_models

TRADING_DAYS = 252
RISK_FREE_ANNUAL = 0.02


def main():
    print("=" * 88)
    print(" 🛡️ 启动 10 年大类资产配置基准全量自动化守卫审计")
    print("=" * 88)

    ledger_path = HERE / "trials_10y.jsonl"
    if ledger_path.exists():
        ledger_path.unlink()
    ledger = TrialLedger(ledger_path)

    models = get_all_models()
    trial_ids = {}
    for m in models:
        tid = ledger.record(
            strategy=m.name,
            params={"description": m.description, "sample_period": "2015-01-05 to 2026-09-11"},
            note="Pre-registered 10-year multi-asset benchmark arm"
        )
        trial_ids[m.name] = tid
        print(f"  -> 已预注册试验 [{tid[:8]}] {m.name:22s}")

    # Load data
    data = fetch_10y_etf_data()
    open_df, close_df = data["open"], data["close"]
    dates = data["dates"]
    daily_rf = (1.0 + RISK_FREE_ANNUAL) ** (1.0 / TRADING_DAYS) - 1.0

    initial_cap = 1_000_000.0
    accounts = {m.name: BrokerSim(initial_cap) for m in models}
    nav_series = {m.name: {} for m in models}

    for step in range(len(dates)):
        cur_date = dates[step]
        today_open = open_df.iloc[step].to_dict()
        today_close = close_df.iloc[step].to_dict()
        hist_close = close_df.iloc[:step]

        for m in models:
            acc = accounts[m.name]
            acc.accrue_interest(daily_rf)
            if step == 0 or m.should_rebalance(step):
                targets = m.get_target_weights(step, hist_close)
                acc.rebalance(targets, today_open, deadband=0.03)
            nav_series[m.name][cur_date] = acc.equity(today_close)

    nav_df = pd.DataFrame(nav_series)
    years = len(dates) / TRADING_DAYS
    completed_metrics = {}

    for m in models:
        s = nav_df[m.name]
        rets = s.pct_change().dropna()
        excess = rets - daily_rf
        sharpe = float(excess.mean() / excess.std() * np.sqrt(TRADING_DAYS)) if excess.std() > 0 else 0.0
        cagr = float((s.iloc[-1] / initial_cap) ** (1.0 / years) - 1.0)
        cummax = s.cummax()
        max_dd = float(((s - cummax) / cummax).min())

        completed_metrics[m.name] = {
            "sharpe": sharpe,
            "cagr": cagr,
            "max_drawdown": max_dd,
            "final_equity": float(s.iloc[-1]),
        }
        ledger.complete(
            trial_ids[m.name],
            metrics={"sharpe": sharpe, "cagr": cagr, "max_drawdown": max_dd, "n_obs": len(dates)},
            note="10-year completed evaluation"
        )

    # 1. Causality Guard Check
    print("\n[Guard 1/3] 运行因果守卫 (CausalityGuard)...")
    sample_close = close_df.iloc[:300]
    
    def vol_60d_fn(df: pd.DataFrame) -> pd.DataFrame:
        return df.pct_change().rolling(60).std() * np.sqrt(252)

    def mom_20d_fn(df: pd.DataFrame) -> pd.DataFrame:
        return df.pct_change(20)

    cg = CausalityGuard()
    c_res = cg.check(
        fn={"volatility_60d": vol_60d_fn, "momentum_20d": mom_20d_fn},
        df=sample_close,
        k=200
    )
    print(f"  -> CausalityGuard: {'PASSED' if c_res.passed else 'FAILED'}")

    # 2. Trial Ledger Guard Check
    print("\n[Guard 2/3] 运行实验账本通缩夏普守卫 (TrialLedgerGuard)...")
    tlg = TrialLedgerGuard()
    best_m = max(models, key=lambda m: completed_metrics[m.name]["sharpe"])
    best_sh = completed_metrics[best_m.name]["sharpe"]
    t_res = tlg.check(
        best_sharpe=best_sh,
        n_obs=len(dates),
        ledger=ledger,
        alpha=0.05
    )
    print(f"  -> TrialLedgerGuard (最优={best_m.name}, 夏普={best_sh:.3f}): {'PASSED' if t_res.passed else 'FAILED'}")

    # 3. Cost Curve Guard Check on Dynamic Risk Parity
    print("\n[Guard 3/3] 运行摩擦成本敏感度守卫 (CostCurveGuard)...")
    ccg = CostCurveGuard()
    drp_rets = nav_df["dynamic_risk_parity"].pct_change().dropna()
    drp_turnover = accounts["dynamic_risk_parity"].gross_traded / (2.0 * nav_df["dynamic_risk_parity"].mean() * len(dates))
    cc_res = ccg.check(
        returns=drp_rets,
        turnover=drp_turnover,
        bps=(0, 4, 10, 20, 50),
        cost_bps=4.0,  # ETF roundtrip: 0.02% comm + 0.02% slip = 4 bps
        benchmark=0.02
    )
    print(f"  -> CostCurveGuard (ETF往返成本4bps): {'PASSED' if cc_res.passed else 'FAILED'}")

    # Save manifest
    manifest = {
        "audit_timestamp": pd.Timestamp.now().isoformat(),
        "evaluation_period": f"{dates[0]} to {dates[-1]} ({len(dates)} days)",
        "models_audited": [m.name for m in models],
        "completed_metrics": completed_metrics,
        "causality_guard": {
            "passed": c_res.passed,
            "findings": [str(f) for f in c_res.findings]
        },
        "trial_ledger_guard": {
            "passed": t_res.passed,
            "best_strategy": best_m.name,
            "best_sharpe": best_sh,
            "findings": [str(f) for f in t_res.findings]
        },
        "cost_curve_guard": {
            "passed": cc_res.passed,
            "findings": [str(f) for f in cc_res.findings]
        }
    }
    with open(HERE / "audit_manifest_10y.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\n自动化守卫审计清单已保存至: {HERE / 'audit_manifest_10y.json'}")


if __name__ == "__main__":
    main()
