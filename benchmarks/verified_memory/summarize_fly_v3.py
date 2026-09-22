"""Summarize source-checked stateful fly results without selecting a winning model."""
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
    groups = []
    for dataset in sorted(protocol["datasets"]):
        for arm in [*protocol["arms"], "fly_v2_reference"]:
            rows = [r for r in summary["rows"] if r["dataset"] == dataset and r["arm"] == arm]
            if not rows:
                groups.append({"dataset": dataset, "arm": arm, "completed": 0})
                continue
            group = {"dataset": dataset, "arm": arm, "completed": len(rows),
                **{f"return_{c}bps": mean(r["metrics_by_cost"][str(c)]["total_return"] for r in rows)
                   for c in (0, 5, 20)},
                "drawdown": mean(r["metrics_by_cost"]["5"]["max_drawdown"] for r in rows),
                "turnover": mean(r["metrics_by_cost"]["5"]["total_turnover"] for r in rows),
                "risky_exposure": mean(r["metrics_by_cost"]["5"]["mean_risky_exposure"] for r in rows),
                "hold_fraction": mean(r["audit"]["hold_fraction"] for r in rows) if arm != "fly_v2_reference" else None}
            groups.append(group)
    stateful = [r for r in summary["rows"] if r["arm"] != "fly_v2_reference"]
    checks = {"hashes_match": True,
              "max_nav_gap": max(r["audit"]["max_nav_gap"] for r in stateful),
              "max_turnover_gap": max(r["audit"]["max_turnover_gap"] for r in stateful),
              "max_reward_compounding_gap": max(r["audit"]["reward_compounding_gap"] for r in stateful),
              "blocked_receipts": sum(r["audit"]["blocked_receipts"] for r in stateful),
              "early_updates": sum(r["audit"]["early_updates"] for r in stateful)}
    result = {"status": summary["status"], "job": deployment["stdout"],
              "remote_root": deployment["remote_root"], "checks": checks,
              "groups": groups, "baselines": read("baselines.json"), "failures": summary["failures"]}
    with (root / "analysis.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    lines = ["# 果蝇交易控制器 v3：持仓、HOLD 与真实账户反馈", "",
        f"作业 `{deployment['stdout']}`，状态 `{summary['status']}`。",
        f"完成 {len(summary['rows'])}/{summary['planned_cells']} 个模型试验，失败 {len(summary['failures'])} 个。",
        f"全部远端输出在 `{deployment['remote_root']}`，未使用 Beacon HOME。", "",
        "## 本轮改变", "",
        "电路与学习参数保持不变。新增当前资产/现金权重、距上次实际交易的时间、预计调仓费用，",
        "以及保留原有股数和现金的 HOLD 动作。费用在成交时收取；HOLD 不重新平衡漂移后的权重。",
        "训练奖励来自持续账户净值，包含该动作的成交费和持仓盈亏，严格在结果可知后更新。",
        "hold_advantage 对照使用同区间相对原仓位不动的增量结果。", "",
        "## 账本与时序验证", "", "```json", json.dumps(checks, indent=2), "```", "",
        "每次动作已先冻结，再由独立现金/股数账本评分。还检查了奖励区间复利与最终净值的一致性。", "",
        "## 结果", "",
        "表内为五个初始化重复的均值，不是独立市场样本。收益为累计收益；换手为整个评估区间累计值。",
        "费用为每笔交易额 0/5/20 bps。费用敏感性重放相同动作，没有重新训练或重算状态依赖目标。", "",
        "| 数据 | 模型 | 0 bps 收益 | 5 bps 收益 | 最大回撤 | 累计换手 | 平均风险资产仓位 | HOLD 比例 |",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    for g in groups:
        if not g["completed"]:
            lines.append(f"| {g['dataset']} | {g['arm']} | 无完成试验 | — | — | — | — | — |")
            continue
        hold = "—" if g["hold_fraction"] is None else f"{g['hold_fraction']:.1%}"
        lines.append(f"| {g['dataset']} | {g['arm']} | {g['return_0bps']:.2%} | {g['return_5bps']:.2%} | "
                     f"{g['drawdown']:.2%} | {g['turnover']:.2f} | {g['risky_exposure']:.1%} | {hold} |")
    lines += ["", "## 固定策略参照（5 bps）", "", "| 数据 | 策略 | 累计收益 |", "|---|---|---:|"]
    for row in result["baselines"]:
        lines.append(f"| {row['dataset']} | {row['arm']} | {row['metrics_by_cost']['5']['total_return']:.2%} |")
    lines += ["", "## 解释边界", "",
        "这是看过 v2 结果后的开发迭代。ECB 2016—2025 年参考汇率及合成路径均不是未见留出数据。",
        "ECB 不包含外汇利息、融资和实际成交价差。降低换手或增加现金可以减亏，但本身不证明预测能力。",
        "v2 和 v3 同时改变输入、动作和奖励；两版差异不能单独归因于某一个机制。",
        "本轮没有依据收益挑选或重调参数；保留所有预先列出的对照和失败。", "",
        "## 测试记录", "", "```text", (root / "beacon-tests.txt").read_text().strip(), "```", "",
        "原始摘要：`summary.json`；配对日期区块比较：`contrasts.json`；完整轨迹保留在 RADFM。", ""]
    with (root / "REPORT.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines))
    print(json.dumps({"report": str(root / "REPORT.md"), "checks": checks}))


if __name__ == "__main__":
    main()
