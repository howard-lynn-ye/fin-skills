# 🛠️ fin-skills 终端操作手册与标准作业程序（Operations Guide & SOP）

> [!NOTE] **核心说明：本工具箱纯属确定性数学/统计审计工具，自身完全不训练任何模型**  
> `fin-skills` 包含的 32 个 Python 守卫（Guards）均基于 `numpy`/`pandas` 执行时序因果性断言、预热期分析与费率推演，**没有模型训练、没有参数拟合、不需要 GPU**。  
> 无论您的回测策略是纯规则因子（如均线交叉、Alpha 101）还是复杂模型输出，`fin-skills` 均作为外部中立质检器，在回测后耗时 `<0.05s` 验证结果真实性。

---

## 📋 1. 环境初始化与安装操作（首次配置或换机时执行）

### 1.1 安装 Python 可执行防伪包 (`fin_skills`)
将仓库以可编辑模式（Editable Mode）安装到当前 Python 环境中，确保任何代码修改实时生效：

```bash
cd /usr/local/google/home/shwaihe/fin-skills
pip install --user --break-system-packages -e .
```

**验证安装是否成功：**
```bash
python3 -c "import fin_skills; print('Skills count:', len(fin_skills.catalog()))"
# 预期输出: Skills count: 114
```

### 1.2 验证 API 核心单元测试套件
```bash
cd /usr/local/google/home/shwaihe/fin-skills
python3 -m pytest -q tests/test_api.py
# 预期输出: 79 passed in 3.3s
```

---

## ⚡ 2. 日常量化投研与防伪体检操作命令（CLI Cheat Sheet）

### 操作 A：对真实数据与回测产物进行一键防伪体检
在每次因子计算更新、特征工程调整或策略回测产出净值序列后，执行以下命令生成五维防伪体检报告（因果性、预热期、盈亏平衡成本、无风险利率口径、数据窗口验证）：

```bash
cd /usr/local/google/home/shwaihe/stock_prediction
python3 scripts/audit_with_fin_skills.py
```

### 操作 B：在命令行快速查询某个领域的防伪规范
无需打开浏览器翻找文件，直接在终端打印任意 Skill 的完整规范或检索关键词：

```bash
# 1. 检索同时包含 "survivorship" 和 "universe" 的技能
python3 -c "import fin_skills; print(fin_skills.find('survivorship', 'universe'))"

# 2. 直接在终端查看 A 股印花税与红利税计算规范
python3 -c "import fin_skills; print(fin_skills.load('china-ashare-trading-taxes'))"

# 3. 查看所有 32 个可执行守卫（Guards）及其所需的数据槽位（Slots）
python3 -c "from fin_skills.api import slots; import pprint; pprint.pprint(slots())"
```

### 操作 C：运行 12 类量化作弊检出基准 (`leak_bench`)
验证当前环境下的 13 个核心守卫能否在 `<0.07s` 内 100% 捕获 12 种典型回测造假且保持 0 误报：

```bash
cd /usr/local/google/home/shwaihe/fin-skills
python3 benchmarks/leak_bench.py --quick
```

---

## 🔄 3. 维护者标准作业程序（修改或新增 Skill 后的 SOP）

> [!IMPORTANT]
> **零漂移铁律（Zero-Drift Rule）**：`fin-skills` 的 `catalog/index.json`、`README.md` 中的技能统计表以及 `fin_skills/` 下的 Python 模块**全部由脚本自动生成**。如果您修改了任何 `plugins/*/skills/*/SKILL.md` 或 `scripts/*.py`，**必须按顺序执行以下三步命令**，否则 CI 校验 (`validate.py`) 将直接报错拦截！

### 标准三步编译与校验命令：
```bash
cd /usr/local/google/home/shwaihe/fin-skills

# 第 1 步：重新扫描所有 SKILL.md，更新 catalog/index.json 与 README 统计计数
python3 scripts/build_index.py

# 第 2 步：将各插件下的 scripts/*.py 编译同步至 fin_skills/ Python 包目录
python3 scripts/build_package.py

# 第 3 步：运行 6 字段规范检查、引用完整性检查与零漂移校验（必须输出 OK）
python3 scripts/validate.py
```

---

## 🌐 4. GitHub 双仓库 `v2` 分支同步操作命令

为了避免与上游 `master`/`main` 分支产生合并冲突，所有定制化文档、中文手册与桥接脚本均统一维护在 **`v2`** 分支上。

### 4.1 同步 `fin-skills` 仓库 (`howard-lynn-ye/fin-skills`)
```bash
cd /usr/local/google/home/shwaihe/fin-skills

# 确认当前处于 v2 分支
git branch -vv

# 执行构建校验（确保零漂移）
python3 scripts/build_index.py && python3 scripts/build_package.py && python3 scripts/validate.py

# 提交并推送到远端 v2 分支
git add .
git commit -m "docs: update documentation and skills on v2"
git push origin v2
```

### 4.2 同步 `stock_prediction` 仓库 (`Shwai-He/stock-prediction`)
```bash
cd /usr/local/google/home/shwaihe/stock_prediction

# 确认当前处于 v2 分支
git branch -vv

# 先运行防伪审计脚本确认全绿通过
python3 scripts/audit_with_fin_skills.py

# 提交代码/文档并推送到远端 v2 分支
git add scripts/audit_with_fin_skills.py docs/FIN_SKILLS_INTEGRATION_MANUAL.md README.md
git commit -m "feat(audit): update fin-skills integration audit report"
git push origin v2
```

---

## 🚑 5. 常见报错与快速排查手册（Troubleshooting）

| 报错现象 / 提示信息 | 原因分析 | 一键解决命令 / 方案 |
| :--- | :--- | :--- |
| `validate.py` 报错：<br>`fin_skills/ is out of date - run scripts/build_package.py` | 修改了 `plugins/` 下的脚本或 `SKILL.md` 后未重新编译 Python 包。 | 运行 `python3 scripts/build_index.py && python3 scripts/build_package.py` |
| `Bundle` 报错：<br>`ValueError: unknown slot name '...'` | 传入 `Bundle(...)` 的参数名不在 146 个标准词汇表内。 | 运行 `python3 -c "from fin_skills.api import slots; print(list(slots().keys()))"` 查看合法槽位名。 |
| `contamination_probe` 报错：<br>`INVALID: the entire backtest window predates training cutoff` | 您评测 LLM（如 Qwen3/GPT-4）的回测时间段早于该模型的预训练语料截止日。 | 将 `test_start` 调整至模型 Cutoff 日期之后（如 `2025-01-01` 以后），或使用未见过该时期数据的盲测集。 |
| `assert_causal` 报错：<br>`FAIL: LOOK-AHEAD ... cells before index k changed` | 您的特征计算函数中存在未来数据泄露（如 `shift(-1)`、居中窗口或全样本归一化）。 | 检查特征代码中的 `rolling()`、`shift()` 或 `StandardScaler()`，确保仅在历史过去窗口上计算。 |
