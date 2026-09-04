"""Append-only trial ledger — the input the Deflated Sharpe Ratio actually needs.

DSR's n_trials is the honest count of EVERY configuration tried: every barrier multiple,
lookback, threshold, feature set, universe filter and rebalance frequency, including the
ones you abandoned and everything an automated search evaluated. No library can recover
it for you; you have to record it as you go.

Record BEFORE looking at out-of-sample results. That ordering is what makes the count
credible.

Usage:
    led = TrialLedger("research/trials.jsonl")
    tid = led.record(strategy="ma_cross", params={"fast": 10, "slow": 50})
    ...run the backtest...
    led.complete(tid, metrics={"sharpe": 1.2, "n_obs": 1260})
    print(led.summary())
    print(led.deflated_sharpe(best_sharpe=1.2, n_obs=1260))
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


class TrialLedger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _append(self, rec: dict) -> None:
        # append-only: open in 'a', never rewrite history
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, sort_keys=True, default=str) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def record(self, strategy: str, params: dict, note: str = "") -> str:
        """Register a trial BEFORE evaluating it. Returns the trial id."""
        payload = json.dumps({"strategy": strategy, "params": params}, sort_keys=True)
        tid = hashlib.sha256(payload.encode()).hexdigest()[:16]
        self._append({
            "event": "registered", "trial_id": tid, "strategy": strategy,
            "params": params, "note": note,
            "ts": datetime.now(timezone.utc).isoformat(),
        })
        return tid

    def complete(self, trial_id: str, metrics: dict, note: str = "") -> None:
        self._append({
            "event": "completed", "trial_id": trial_id, "metrics": metrics,
            "note": note, "ts": datetime.now(timezone.utc).isoformat(),
        })

    def abandon(self, trial_id: str, reason: str) -> None:
        """An abandoned trial STILL COUNTS. That is the entire point of the ledger."""
        self._append({
            "event": "abandoned", "trial_id": trial_id, "reason": reason,
            "ts": datetime.now(timezone.utc).isoformat(),
        })

    def read(self) -> list[dict]:
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def summary(self) -> dict:
        recs = self.read()
        registered = {r["trial_id"] for r in recs if r["event"] == "registered"}
        completed = [r for r in recs if r["event"] == "completed"]
        sharpes = [r["metrics"].get("sharpe") for r in completed
                   if isinstance(r.get("metrics"), dict) and r["metrics"].get("sharpe") is not None]
        return {
            "n_trials": len(registered),          # <- this is DSR's n_trials
            "n_completed": len(completed),
            "n_abandoned": sum(1 for r in recs if r["event"] == "abandoned"),
            "sharpe_variance": float(np.var(sharpes, ddof=1)) if len(sharpes) > 1 else None,
            "best_sharpe": max(sharpes) if sharpes else None,
        }

    def deflated_sharpe(self, best_sharpe: float, n_obs: int,
                        skew: float = 0.0, kurt: float = 3.0) -> dict:
        """DSR using THIS ledger's honest trial count and observed Sharpe variance.

        Formula per Bailey & Lopez de Prado (2014). Verify against the source paper
        before publishing. kurt is NON-excess kurtosis (3.0 = normal).
        """
        from scipy.stats import norm

        s = self.summary()
        n_trials = max(s["n_trials"], 1)
        var_sr = s["sharpe_variance"]
        if var_sr is None or n_trials < 2:
            return {"error": "need >=2 completed trials with a 'sharpe' metric",
                    "n_trials": n_trials}

        e = np.euler_gamma
        sr0 = np.sqrt(var_sr) * (
            (1 - e) * norm.ppf(1 - 1 / n_trials)
            + e * norm.ppf(1 - 1 / (n_trials * np.e))
        )
        num = (best_sharpe - sr0) * np.sqrt(n_obs - 1)
        den = np.sqrt(1 - skew * best_sharpe + ((kurt - 1) / 4) * best_sharpe ** 2)
        return {
            "n_trials": n_trials,
            "expected_max_sharpe_from_noise": float(sr0),
            "observed_sharpe": float(best_sharpe),
            "deflated_sharpe_ratio": float(norm.cdf(num / den)),
            "verdict": "survives" if norm.cdf(num / den) > 0.95 else "NOT distinguishable from noise",
        }


if __name__ == "__main__":
    import tempfile

    led = TrialLedger(Path(tempfile.mkdtemp()) / "trials.jsonl")
    rng = np.random.default_rng(0)
    # 50 variants of a moving-average crossover, all pure noise
    for fast in range(5, 55, 5):
        for slow in range(60, 160, 20):
            tid = led.record("ma_cross", {"fast": fast, "slow": slow})
            led.complete(tid, {"sharpe": float(rng.normal(0, 0.45)), "n_obs": 1260})
    s = led.summary()
    print("summary:", s)
    print("DSR    :", led.deflated_sharpe(best_sharpe=s["best_sharpe"], n_obs=1260))
    print("\nThe best of 50 noise strategies looks good in isolation and dies under DSR.")
