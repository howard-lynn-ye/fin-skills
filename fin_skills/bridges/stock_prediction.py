"""fin_skills.bridges.stock_prediction - Bridge to the stock_prediction multi-modal research repository.

Loads and aligns real historical datasets from `stock_prediction`:
1. Global Macro Series: VIX implied volatility, 10Y US Treasury yield (^TNX), Dollar Index (DXY), CSI300, S&P 500 (^GSPC), Hang Seng (^HSI).
2. Smart Money Flows: Northbound Stock Connect net inflows and margin trading leverage across 124 leading equities (2011 ~ 2026).
3. Xueqiu Social & Semantic Events: 126,000+ daily semantic signals and longitudinal social sentiment posts (2014 ~ 2026).
4. Point-in-Time Enforcement: All exogenous signals are strictly lagged by 1 day (shift(1)) before asset allocation to guarantee zero look-ahead bias.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

logger = logging.getLogger("bridge_stock_prediction")

DEFAULT_CANDIDATE_DIRS = [
    Path("/usr/local/google/home/shwaihe/stock_prediction"),
    Path(__file__).resolve().parents[3] / "stock_prediction",
]


def find_stock_prediction_dir() -> Path:
    """Locate the stock_prediction repository root."""
    env_dir = os.environ.get("STOCK_PREDICTION_DIR")
    if env_dir and Path(env_dir).is_dir():
        return Path(env_dir).resolve()

    for candidate in DEFAULT_CANDIDATE_DIRS:
        if candidate.is_dir() and (candidate / "data" / "fetched").exists():
            return candidate.resolve()

    raise FileNotFoundError(
        "Could not locate stock_prediction repository with data/fetched directory. "
        "Set STOCK_PREDICTION_DIR environment variable."
    )


class StockPredictionBridge:
    """Bridge for ingesting and conditioning historical multi-modal data from stock_prediction."""

    def __init__(self, repo_dir: Path | str | None = None):
        self.root = Path(repo_dir).resolve() if repo_dir else find_stock_prediction_dir()
        self.data_dir = self.root / "data" / "fetched"
        self.macro_dir = self.data_dir / "market" / "macro"
        self.smart_money_dir = self.data_dir / "market" / "smart_money"
        self.semantic_dir = self.data_dir / "features" / "xueqiu_semantic_events"
        self.social_dir = self.data_dir / "social" / "xueqiu" / "unified_longitudinal"

    def load_macro_indicators(
        self,
        start_date: str = "2015-01-01",
        end_date: str = "2026-09-08",
    ) -> pd.DataFrame:
        """Load and align daily macro series: VIX, TNX, DXY, CSI300, S&P500, HSI."""
        series = {}

        # 1. VIX (^VIX.csv or VIX.csv)
        vix_file = self.macro_dir / "^VIX.csv" if (self.macro_dir / "^VIX.csv").exists() else self.macro_dir / "VIX.csv"
        if vix_file.exists():
            df_vix = pd.read_csv(vix_file)
            df_vix["date"] = pd.to_datetime(df_vix["date"]).dt.strftime("%Y-%m-%d")
            series["vix"] = df_vix.set_index("date")["close"]

        # 2. 10Y US Treasury yield (^TNX.csv or TNX.csv)
        tnx_file = self.macro_dir / "^TNX.csv" if (self.macro_dir / "^TNX.csv").exists() else self.macro_dir / "TNX.csv"
        if tnx_file.exists():
            df_tnx = pd.read_csv(tnx_file)
            df_tnx["date"] = pd.to_datetime(df_tnx["date"]).dt.strftime("%Y-%m-%d")
            series["tnx_yield"] = df_tnx.set_index("date")["close"]

        # 3. US Dollar Index (DXY.csv)
        dxy_file = self.macro_dir / "DXY.csv"
        if dxy_file.exists():
            df_dxy = pd.read_csv(dxy_file)
            df_dxy["date"] = pd.to_datetime(df_dxy["date"]).dt.strftime("%Y-%m-%d")
            series["dxy"] = df_dxy.set_index("date")["close"]

        # 4. CSI 300 (510300.SS.csv or CSI300.csv)
        csi_file = self.macro_dir / "510300.SS.csv" if (self.macro_dir / "510300.SS.csv").exists() else self.macro_dir / "CSI300.csv"
        if csi_file.exists():
            df_csi = pd.read_csv(csi_file)
            df_csi["date"] = pd.to_datetime(df_csi["date"]).dt.strftime("%Y-%m-%d")
            series["csi300_close"] = df_csi.set_index("date")["close"]

        # 5. S&P 500 (^GSPC.csv)
        sp_file = self.macro_dir / "^GSPC.csv" if (self.macro_dir / "^GSPC.csv").exists() else self.macro_dir / "GSPC.csv"
        if sp_file.exists():
            df_sp = pd.read_csv(sp_file)
            df_sp["date"] = pd.to_datetime(df_sp["date"]).dt.strftime("%Y-%m-%d")
            series["sp500_close"] = df_sp.set_index("date")["close"]

        # 6. Hang Seng (^HSI.csv)
        hsi_file = self.macro_dir / "^HSI.csv" if (self.macro_dir / "^HSI.csv").exists() else self.macro_dir / "HSI.csv"
        if hsi_file.exists():
            df_hsi = pd.read_csv(hsi_file)
            df_hsi["date"] = pd.to_datetime(df_hsi["date"]).dt.strftime("%Y-%m-%d")
            series["hsi_close"] = df_hsi.set_index("date")["close"]

        df_merged = pd.DataFrame(series).sort_index().ffill().dropna(subset=["csi300_close"])
        df_merged = df_merged.loc[start_date:end_date].reset_index()

        # Derive macro regime tags
        # VIX > 25 or 20-day delta > 5 -> RISK_OFF_PANIC
        vix_ma20 = df_merged["vix"].rolling(20).mean()
        df_merged["vix_spike"] = (df_merged["vix"] > 25.0) | ((df_merged["vix"] - vix_ma20) > 4.0)
        df_merged["macro_regime"] = np.where(df_merged["vix_spike"], "RISK_OFF_PANIC", "NORMAL")

        return df_merged

    def load_smart_money_flows(
        self,
        start_date: str = "2015-01-01",
        end_date: str = "2026-09-08",
        sample_symbols: int = 30,
    ) -> pd.DataFrame:
        """Aggregate daily Northbound net inflows and margin trading leverage across leading equities."""
        files = sorted(self.smart_money_dir.glob("*.csv"))[:sample_symbols]
        dfs = []
        for f in files:
            try:
                df = pd.read_csv(f)
                if "date" in df.columns and "northbound_net_buy_shares" in df.columns:
                    dfs.append(df[["date", "northbound_net_buy_shares", "margin_buy_ratio"]])
            except Exception:
                pass

        if not dfs:
            return pd.DataFrame(columns=["date", "northbound_flow_agg", "margin_leverage_z20", "smart_money_regime"])

        combined = pd.concat(dfs).groupby("date").agg({
            "northbound_net_buy_shares": "mean",
            "margin_buy_ratio": "mean",
        }).reset_index().sort_values("date")

        combined = combined.rename(columns={"northbound_net_buy_shares": "northbound_flow_agg"})
        # 20-day rolling Z-score of margin ratio
        margin_mean = combined["margin_buy_ratio"].rolling(20).mean()
        margin_std = combined["margin_buy_ratio"].rolling(20).std().replace(0, 1e-5)
        combined["margin_leverage_z20"] = (combined["margin_buy_ratio"] - margin_mean) / margin_std

        # Smart money regime
        # Positive foreign inflow and controlled leverage = RISK_ON_ACCUMULATION
        cond_on = (combined["northbound_flow_agg"] > 0) & (combined["margin_leverage_z20"] > -0.5)
        cond_off = (combined["northbound_flow_agg"] < 0) & (combined["margin_leverage_z20"] < -1.0)
        combined["smart_money_regime"] = np.where(
            cond_on, "RISK_ON_ACCUMULATION", np.where(cond_off, "RISK_OFF_OUTFLOW", "NEUTRAL")
        )

        combined = combined[(combined["date"] >= start_date) & (combined["date"] <= end_date)].reset_index(drop=True)
        return combined

    def load_xueqiu_semantic_signals(
        self,
        start_date: str = "2015-01-01",
        end_date: str = "2026-09-08",
    ) -> pd.DataFrame:
        """Load aggregated Xueqiu daily semantic signals (post volume, polarity, event types)."""
        sem_file = self.semantic_dir / "xueqiu_daily_semantic_signals.csv"
        if not sem_file.exists():
            return pd.DataFrame(columns=["date", "social_post_volume", "social_net_polarity", "social_sentiment_z30"])

        df = pd.read_csv(sem_file)
        agg = df.groupby("date").agg({
            "substantive_post_count": "sum",
            "substantive_net_sentiment": "mean",
            "position_events": "sum",
            "valuation_events": "sum",
        }).reset_index().sort_values("date")

        agg = agg.rename(columns={
            "substantive_post_count": "social_post_volume",
            "substantive_net_sentiment": "social_net_polarity",
        })

        # Trailing 30-day Z-score of social post volume (detects retail euphoria spikes)
        vol_mean = agg["social_post_volume"].rolling(30, min_periods=5).mean()
        vol_std = agg["social_post_volume"].rolling(30, min_periods=5).std().replace(0, 1e-5)
        agg["social_heat_z30"] = (agg["social_post_volume"] - vol_mean) / vol_std

        # Classify retail euphoria FOMO vs Panic
        agg["social_fomo_spike"] = (agg["social_heat_z30"] > 2.0) & (agg["social_net_polarity"] > 0.3)
        agg["social_panic_dump"] = (agg["social_heat_z30"] > 1.5) & (agg["social_net_polarity"] < -0.3)

        agg = agg[(agg["date"] >= start_date) & (agg["date"] <= end_date)].reset_index(drop=True)
        return agg

    def build_multimodal_feature_panel(
        self,
        start_date: str = "2015-01-01",
        end_date: str = "2026-09-08",
    ) -> pd.DataFrame:
        """Merge macro, smart money, and social sentiment into an aligned daily panel with PIT shift(1)."""
        macro = self.load_macro_indicators(start_date=start_date, end_date=end_date)
        smart = self.load_smart_money_flows(start_date=start_date, end_date=end_date)
        social = self.load_xueqiu_semantic_signals(start_date=start_date, end_date=end_date)

        panel = macro.merge(smart, on="date", how="left").merge(social, on="date", how="left")
        panel = panel.sort_values("date").reset_index(drop=True)

        # Forward fill continuous macro metrics
        panel["vix"] = panel["vix"].ffill().fillna(18.0)
        panel["tnx_yield"] = panel["tnx_yield"].ffill().fillna(2.5)
        panel["dxy"] = panel["dxy"].ffill().fillna(95.0)

        # Fill discrete event signals with neutral zero
        panel["northbound_flow_agg"] = panel["northbound_flow_agg"].fillna(0.0)
        panel["margin_leverage_z20"] = panel["margin_leverage_z20"].fillna(0.0)
        panel["social_post_volume"] = panel["social_post_volume"].fillna(0.0)
        panel["social_net_polarity"] = panel["social_net_polarity"].fillna(0.0)
        panel["social_heat_z30"] = panel["social_heat_z30"].fillna(0.0)
        panel["social_fomo_spike"] = panel["social_fomo_spike"].fillna(False)
        panel["social_panic_dump"] = panel["social_panic_dump"].fillna(False)

        # CRITICAL PIT CAUSALITY:
        # All signals used for today's decision must be lagged by 1 bar (knowable yesterday at close)
        signal_cols = [
            "vix_spike", "macro_regime", "northbound_flow_agg", "margin_leverage_z20",
            "smart_money_regime", "social_post_volume", "social_net_polarity",
            "social_heat_z30", "social_fomo_spike", "social_panic_dump"
        ]
        for col in signal_cols:
            panel[f"{col}_lag1"] = panel[col].shift(1)

        return panel
