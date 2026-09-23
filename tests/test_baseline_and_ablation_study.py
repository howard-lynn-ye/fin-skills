"""Unit tests for external baseline comparison and 7-arm component-wise ablation study."""
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_baseline_and_ablation_json_schema_and_parity():
    path = ROOT / "benchmarks/BASELINE_AND_COMPONENT_ABLATION_RESULTS.json"
    assert path.exists(), "Missing BASELINE_AND_COMPONENT_ABLATION_RESULTS.json"
    data = json.loads(path.read_text(encoding="utf-8"))

    assert data["seeds"] == [11, 23, 37, 42, 73]
    assert data["evaluated_panel_rows"] == 233138
    assert len(data["external_baselines"]) == 9
    assert len(data["component_ablations"]) == 7

    ours = data["external_baselines"]["FinSkills_Full_Closed_Loop_Ours"]
    finagent = data["external_baselines"]["FinAgent_Multimodal_Tool_Reflection"]
    finmem = data["external_baselines"]["FinMem_Layered_FAISS_Memory"]
    tradingagents = data["external_baselines"]["TradingAgents_Bull_Bear_Debate"]

    assert ours["net_sharpe_mean"] > finagent["net_sharpe_mean"]
    assert ours["net_sharpe_mean"] > finmem["net_sharpe_mean"]
    assert ours["net_sharpe_mean"] > tradingagents["net_sharpe_mean"]
    assert ours["leak_rate_mean_pct"] == 0.0
    assert ours["mean_prompt_tokens"] == 4039

    full_arm = data["component_ablations"]["0_Full_FinSkills_Architecture"]
    for key, arm in data["component_ablations"].items():
        if key != "0_Full_FinSkills_Architecture":
            assert arm["net_sharpe_mean"] < full_arm["net_sharpe_mean"]
            assert arm["delta_sharpe_vs_full"] < 0.0
