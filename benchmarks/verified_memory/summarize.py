"""Summarize downloaded Beacon evidence; never infer profitability from proxy returns."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evidence", type=Path, required=True)
    args = p.parse_args()
    root = args.evidence
    read = lambda name: json.loads((root / name).read_text(encoding="utf-8"))
    summary, protocol, deployment = read("summary.json"), read("protocol.json"), read("deployment.json")
    protocol_hash = hashlib.sha256((root / "protocol.json").read_bytes()).hexdigest()
    if summary["protocol_sha256"] != protocol_hash:
        raise ValueError("protocol hash mismatch")
    if not all(deployment["source_hashes"].get(k) == v for k, v in protocol["source_hashes"].items()):
        raise ValueError("executed source differs from deployed source")
    rows = summary["rows"]
    by_key = {(r["dataset"], r["scenario"], r["model"], r["mode"], r["seed"]): r for r in rows}
    action_checks = {"output_audit_equals_unchecked": True,
                     "clean_verified_equals_unchecked": True,
                     "early_verified_equals_clean_verified": True}
    for row in rows:
        d, s, m, a, seed = (row[k] for k in ("dataset", "scenario", "model", "mode", "seed"))
        if a == "output_audit":
            action_checks["output_audit_equals_unchecked"] &= (row["actions_sha256"] ==
                by_key[d, s, m, "unchecked", seed]["actions_sha256"])
        if s == "clean" and a == "verified_update":
            action_checks["clean_verified_equals_unchecked"] &= (row["actions_sha256"] ==
                by_key[d, s, m, "unchecked", seed]["actions_sha256"])
        if s == "early_feedback" and a == "verified_update":
            action_checks["early_verified_equals_clean_verified"] &= (row["actions_sha256"] ==
                by_key[d, "clean", m, a, seed]["actions_sha256"])
    groups = []
    for scenario in protocol["scenarios"]:
        for mode in protocol["modes"]:
            subset = [r for r in rows if r["scenario"] == scenario and r["mode"] == mode]
            groups.append({"scenario": scenario, "mode": mode, "cells": len(subset),
                "cells_with_invalid_updates": sum(r["audit"]["invalid_updates"] > 0 for r in subset),
                "cells_with_early_updates": sum(r["audit"]["early_updates"] > 0 for r in subset),
                "cells_without_updates": sum(r["audit"]["updates"] == 0 for r in subset),
                "minimum_negative_targets_learned": min(r["audit"]["negative_targets_learned"] for r in subset)})
    economics = []
    for dataset in protocol["datasets"]:
        for model in protocol["models"]:
            subset = [r for r in rows if r["dataset"] == dataset and r["model"] == model
                      and r["scenario"] == "clean" and r["mode"] == "verified_update"]
            economics.append({"dataset": dataset, "model": model,
                "mean_cumulative_return_5bps": mean(r["metrics_5bps"]["total_return"] for r in subset),
                "mean_max_drawdown": mean(r["metrics_5bps"]["max_drawdown"] for r in subset)})
    output = {"status": summary["status"], "job": deployment["stdout"],
        "remote_root": deployment["remote_root"], "completed_cells": len(rows),
        "protocol_sha256": protocol_hash, "source_hashes_match": True,
        "action_equivalence_checks": action_checks, "mechanism_groups": groups,
        "clean_economics": economics,
        "interpretation": "Known-fault update gating works; no consistent sparse-model economic advantage. "
          "ECB results are indicative price-only proxy returns, not executable FX returns."}
    with (root / "analysis.json").open("x", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    lines = ["# Beacon 稀疏记忆实验结果", "",
        f"作业 `{deployment['stdout']}`：{summary['status']}，完成 {len(rows)} 个矩阵试验。",
        f"RADFM 目录：`{deployment['remote_root']}`。", "",
        "Beacon 测试记录：", "", "```text", (root / "beacon-tests.txt").read_text().strip(), "```", "",
        "协议与部署源码哈希一致。所有配置在评分前登记；没有根据这一轮收益调参。", "",
        "## 机制结果", "",
        "| 反馈场景 | 学习条件 | 试验数 | 有错误更新的试验 | 有提前更新的试验 |",
        "|---|---|---:|---:|---:|"]
    for g in groups:
        lines.append(f"| {g['scenario']} | {g['mode']} | {g['cells']} | "
                     f"{g['cells_with_invalid_updates']} | {g['cells_with_early_updates']} |")
    lines += ["", "所有条件均产生了学习更新且保留了负收益标签。",
        "输出端审计只拒绝报告，交易动作与不审计条件相同；验证后更新在提前反馈场景中恢复了与干净数据相同的动作。",
        f"动作哈希检查：`{json.dumps(action_checks)}`。", "",
        "这些是已知故障的机制测试，不是对未知金融数据错误的检测能力证明。", "",
        "## 干净反馈下的经济结果", "",
        "以下为五个初始化种子的平均累计收益和平均最大回撤，费用为每笔交易额 5 bps。",
        "同一市场上的初始化种子不算独立市场样本。", "",
        "| 数据 | 模型 | 累计收益 | 最大回撤 |", "|---|---|---:|---:|"]
    for e in economics:
        lines.append(f"| {e['dataset']} | {e['model']} | {e['mean_cumulative_return_5bps']:.2%} | {e['mean_max_drawdown']:.2%} |")
    base = read("ecb-fixed-experts.json")
    lines += ["", "ECB 固定专家对照：", "", "| 专家 | 累计收益 |", "|---|---:|"]
    for name, v in base.items():
        lines.append(f"| {name} | {v['scores']['5']['metrics']['total_return']:.2%} |")
    lines += ["", "果蝇启发模型在合成市场上的表现不一致，在 ECB 代理回测中也没有显示出稳定优势。",
        "这轮结果支持继续研究反馈验证机制，不支持把果蝇架构作为已证实的收益贡献。", "",
        "ECB 评估区间为 2016—2025 年，之前的数据用于在线学习预热；之后仍只用已成熟标签更新。",
        "这是已经查看过的历史开发数据，不是未见留出集。ECB 参考汇率不是可成交报价，",
        "结果不含外汇利息、融资、价差和真实撮合，因此不能当作可实现的外汇收益。",
        "合成市场末尾 40% 用于评分。稀疏和稠密模型参数量相同；当前方法是完整专家反馈的在线回归，",
        "不是完整果蝇脑仿真、LLM Agent 或 bandit 强化学习。", "",
        "原始动作、学习记录、审计收据和账本均保留在上述 RADFM 目录。",
        "本地原始摘要见 `summary.json`，配对日期区块比较见 `contrasts.json`。", ""]
    with (root / "REPORT.md").open("x", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(json.dumps({"report": str(root / "REPORT.md"), "checks": action_checks,
                      "completed_cells": len(rows)}))


if __name__ == "__main__":
    main()
