# 量化策略、决策与算法目录

浏览 [完整目录](../catalog/QUANT_METHODS.md)，或读取
[JSON 数据](../catalog/quant_methods.json)。这是持续扩展的研究目录，不声称穷尽所有方法。

当前导出包含 **203 个条目**：51 个策略、37 个决策规则、115 个算法或模型适配器。
其中 63 个已注册适配器、123 个有外部实现、17 个是研究或方法参考。
这些数字由 `scripts/export_quant_methods.py` 在 Beacon 上生成；包含不同后端的条目，
不能解释为 203 个独立策略，也不是本轮新增了 63 个适配器。

收录范围包括因子、趋势、套息、相对价值、事件驱动、期权组合、做市、加密套利、
组合配置、再平衡、交易执行、止盈止损、时间序列、机器学习、强化学习和评估方法。
果蝇记忆模型通过现有模型注册表纳入。每个参考条目记录中英文名称、所需数据、
适用资产类别、注意事项、来源和核查日期；已接入条目继承原有适配器元数据。

## 找到方法，再选择工具

```python
from fin_skills.algorithms import search_methods, get_method, method_coverage

search_methods("协整", kind="strategy")
search_methods("carry", asset_class="fx")
search_methods(family="options", status="external", limit=20)
card = get_method("cointegration_pairs")
coverage = method_coverage()
```

`search_methods` 对中英文名称、数据字段和注意事项做关键词检索，多个词同时匹配。
它返回总数和 `next_offset`，默认只展示一页，最多每页 100 条。
排名反映关键词匹配程度，不代表适用性、预期收益或投资建议。
资产过滤为精确匹配；`multi_asset` 和 `input_dependent` 需要单独查询，
不会自动声称某个算法可以处理所有资产的数据。

模型可调用 `quant_method_coverage` 查看分类与缺口，使用 `search_quant_methods`
检索，再调用 `get_quant_method` 读取完整条目。这些工具已加入 AutoGen 桥接的默认工具集。
模型自主决定是否继续调用 `list_models`、`run_model` 或会话中的训练、预测、回测工具。
目录不强制某种市场状态必须选择某个策略。

## 三种收录状态

| 状态 | 含义 | 如何使用 |
|---|---|---|
| `integrated` | 本库已注册适配器 | 查看 `availability`、`operations` 和 `model_id` |
| `external` | 上游文档或代码列出实现 | 按来源进入对应项目；本库尚无该条目的完整接口 |
| `reference` | 文献或机构方法介绍 | 可用于研究设计；不能直接当作执行接口 |

已接入条目使用 `model:` 前缀。例如 `get_method("model:engle_granger")`
返回协整检验适配器；`get_method("cointegration_pairs")` 返回配对策略条目。
后者的 `related_model_ids` 指向可复用的检验组件，不代表已有完整的配对交易执行器。

`availability="ready"` 只表示注册表能够发现所需依赖；它不证明当前输入合适、
所有后端版本兼容、策略盈利或已经拥有预训练权重。静态导出不保存某台机器的依赖状态，
运行时调用 `model_catalog()` 查询。

## 来源与证据

本轮核查日期为 **2026-09-21**，参考来源包括原作者研究、项目官方文档和官方代码。
例如 [French 数据库](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html)、
[LEAN 策略目录](https://www.quantconnect.com/docs/v2/writing-algorithms/strategy-library)、
[Cvxportfolio 决策策略](https://www.cvxportfolio.com/en/stable/policies.html) 和
[Hummingbot 实现](https://hummingbot.org/strategies/)。完整链接保存在各条目中。

来源核查只验证方法或实现的存在；没有复现这些来源中的收益。数据要求和注意事项是
本库整理的研究约束，不是逐条引用原作者结论。没有复制上游策略代码、整本书或付费数据库。
目录元数据遵循本库许可；运行、复制或改编外部实现仍需核对该项目及数据的许可。

## 扩充和维护

编辑 `fin_skills/algorithms/_knowledge_data.py`，为新方法指定稳定 ID、类别、资产、
数据要求、区别性约束和可访问的原始来源。能找到成熟实现时优先链接或增加薄适配器。
相关模型链接不能充当“已实现”的证据。真正接入后在模型注册表登记，目录会自动纳入。

```bash
python scripts/build_index.py
python scripts/build_package.py
python scripts/export_quant_methods.py
python scripts/validate.py
python -m pytest -q tests/test_quant_methods.py tests/test_tools.py
```

导出的 Markdown 和 JSON 由脚本维护，不要手改。`validate.py` 检查它们是否过期。
新增适配器的数值正确性、时间隔离和成本处理需要独立测试；目录一致性测试不能替代它们。

目录末尾和 `method_coverage()["gaps"]` 保留尚未系统覆盖的领域，包括宏观事件、
更多固定收益套利、微观结构执行、上下文 bandit 和另类数据策略。
“全部收集”应作为长期覆盖目标；本次交付是有来源、可检索、明确标注实现状态的首批总目录。

## 本轮验证

2026-09-21，在 Beacon 的 `/beacon-projects/radfm/wy891/fin-skills-model-zoo-20260921`
运行目录、JSON 工具和复用工作流测试：**119 passed**。
检查覆盖中英文检索、资产筛选、分页、重复 ID、适配器关联、引用与执行能力区分、
可变返回值隔离，以及导出一致性。`validate.py` 通过；保留原有 fin-libraries 发现预算警告。
这些结果验证软件行为，不是 203 个方法的收益实验。
