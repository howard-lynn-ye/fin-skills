"""Render a source-checked, descriptive report of the fruit-fly circuit experiment."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    root = args.evidence
    read = lambda name: json.loads((root / name).read_text(encoding="utf-8"))
    summary, protocol, deployment = read("summary.json"), read("protocol.json"), read("deployment.json")
    if summary["protocol_sha256"] != hashlib.sha256((root / "protocol.json").read_bytes()).hexdigest():
        raise ValueError("protocol hash mismatch")
    if not all(deployment["source_hashes"].get(k) == v for k, v in protocol["source_hashes"].items()):
        raise ValueError("source hash mismatch")
    cue_groups, market_groups, gate_groups = [], [], []
    variants = sorted({r["variant"] for r in summary["cue_rows"]})
    for task in ("acquisition", "retention", "reversal"):
        for variant in variants:
            rows = [r for r in summary["cue_rows"] if r["task"] == task and r["variant"] == variant]
            cue_groups.append({"task": task, "variant": variant, "n": len(rows),
                **{key: mean(r["scores"][key] for r in rows) for key in
                   ("last_200_choice_accuracy", "final_probe_accuracy", "first_group_retention")}})
    datasets = sorted(protocol["datasets"])
    for dataset in datasets:
        for variant in variants:
            rows = [r for r in summary["market_rows"] if r["dataset"] == dataset
                    and r["variant"] == variant and r["scenario"] == "clean"
                    and r["mode"] == "verified_update"]
            market_groups.append({"dataset": dataset, "variant": variant, "n": len(rows),
                **{f"mean_return_{cost}bps": mean(r["metrics_by_cost"][str(cost)]["total_return"] for r in rows)
                   for cost in (0, 5, 20)},
                "mean_drawdown_5bps": mean(r["metrics_by_cost"]["5"]["max_drawdown"] for r in rows),
                "mean_turnover": mean(r["metrics_by_cost"]["5"]["total_turnover"] for r in rows)})
    for scenario in ("clean", "early_feedback", "wrong_payoff"):
        for mode in ("unchecked", "verified_update"):
            rows = [r for r in summary["market_rows"] if r["variant"] == "fly_trace"
                    and r["scenario"] == scenario and r["mode"] == mode]
            gate_groups.append({"scenario": scenario, "mode": mode, "n": len(rows),
                "cells_using_invalid_receipts": sum(r["audit"]["invalid_updates"] > 0 for r in rows),
                "cells_using_early_receipts": sum(r["audit"]["early_updates"] > 0 for r in rows),
                "minimum_negative_labels": min(r["audit"]["negative_labels"] for r in rows)})
    lookup = {(r["dataset"], r["scenario"], r["mode"], r["seed"]): r
              for r in summary["market_rows"] if r["variant"] == "fly_trace"}
    clean_same = all(r["actions_sha256"] == lookup[d, s, "unchecked", seed]["actions_sha256"]
                     for (d, s, mode, seed), r in lookup.items() if s == "clean" and mode == "verified_update")
    result = {"status": summary["status"], "job": deployment["stdout"],
              "remote_root": deployment["remote_root"], "hashes_match": True,
              "clean_gates_same_actions": clean_same, "cue_groups": cue_groups,
              "market_groups": market_groups, "gate_groups": gate_groups,
              "failures": summary["failures"]}
    with (root / "analysis.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    lines = ["# 果蝇电路 v2：Beacon 实验结果", "",
        f"作业 `{deployment['stdout']}`，状态 `{summary['status']}`。",
        f"完成 {len(summary['cue_rows'])} 个机制实验、{len(summary['market_rows'])} 个市场实验。",
        f"远端目录：`{deployment['remote_root']}`。所有输出位于 RADFM。", "",
        "源码及协议哈希与部署记录一致。试验配置在评分前冻结；没有用结果选择参数。", "",
        "## 电路范围", "",
        "实现了稀疏 KC 编码、相反效价的输出池、局部奖励预测误差更新和绑定原决策的延迟记忆。",
        "它是速率模型，不是完整连接组、脉冲脑仿真或已发表模型的精确复现。",
        "各对照分别移除稀疏化、原决策记忆、可塑性或奖励预测误差；另有简单线性对照。", "",
        "## 已知奖励任务", "",
        "表内为初始化重复的均值。末段选择含固定探索；探针不学习、以并列最优数校正。",
        "这些是自建机制任务，不能解释为已复现论文基准。", "",
        "| 任务 | 模型 | 末 200 次选择正确率 | 最终探针正确率 | 初始组映射保留率 |",
        "|---|---|---:|---:|---:|"]
    for row in cue_groups:
        lines.append(f"| {row['task']} | {row['variant']} | {row['last_200_choice_accuracy']:.1%} | "
                     f"{row['final_probe_accuracy']:.1%} | {row['first_group_retention']:.1%} |")
    lines += ["", "反转任务中，旧映射保留率高不是好事；需要学会新的映射。", "",
        "## 市场代理结果", "",
        "只展示干净反馈、验证后更新的模型比较。表内累计收益为初始化重复的均值。",
        "0/5/20 bps 使用相同动作重放，仅衡量成本敏感性，不是重新训练。", "",
        "| 数据 | 模型 | 0 bps | 5 bps | 20 bps | 5 bps 最大回撤 |",
        "|---|---|---:|---:|---:|---:|"]
    for row in market_groups:
        lines.append(f"| {row['dataset']} | {row['variant']} | {row['mean_return_0bps']:.2%} | "
                     f"{row['mean_return_5bps']:.2%} | {row['mean_return_20bps']:.2%} | "
                     f"{row['mean_drawdown_5bps']:.2%} |")
    lines += ["", "ECB 为 2016—2025 年历史参考汇率代理，先前已参与项目开发；不含外汇利息、融资与实际撮合。",
        "不能把这些结果当成可实现收益或未见数据上的证据。版本 2 仅学习被选动作的奖励，",
        "版本 1 学习全部专家收益，所以版本间收益差异不能单独归因于架构。", "",
        "## 与 fin-skills 的连接", "",
        "| 反馈场景 | 条件 | 试验数 | 使用无效收据的试验 | 使用提前反馈的试验 |",
        "|---|---|---:|---:|---:|"]
    for row in gate_groups:
        lines.append(f"| {row['scenario']} | {row['mode']} | {row['n']} | "
                     f"{row['cells_using_invalid_receipts']} | {row['cells_using_early_receipts']} |")
    lines += ["", f"干净反馈中，审计开关前后动作哈希一致：`{clean_same}`。",
        "无效收据以整个反馈记录为单位拒绝，包括错误位于未选专家的情况；",
        "因此这个计数不是实际被污染的标量奖励数量。这里检测的仍是预设错误。", "",
        "## 验证记录", "", "```text", (root / "beacon-tests.txt").read_text().strip(), "```", "",
        "所有原始动作、奖励更新、机制探针和独立账本均保留在 RADFM。",
        "原始摘要：`summary.json`；配对日期区块比较：`contrasts.json`。", ""]
    with (root / "REPORT.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines))
    print(json.dumps({"report": str(root / "REPORT.md"), "hashes_match": True,
                      "clean_gates_same_actions": clean_same}))


if __name__ == "__main__":
    main()
