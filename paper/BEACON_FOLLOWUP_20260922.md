# Beacon 补充实验：安排与执行状态

日期：2026-09-22。当前状态：**本地准备完成，远端尚未提交**。

## 最值得补的实验

| 顺序 | 实验 | 要回答的问题 | 主要测量 |
|---|---|---|---|
| 1 | Agent 四条件可行性重跑，再扩展主实验 | 真实模型的产物能否执行、冻结并独立评分？技能和检查分别改变什么？ | 正确完成、错误接受、不可评分、超时与缺失；token、工具调用、耗时 |
| 2 | 固定技能文本、BM25、向量检索及 Jev 重排对照 | 内置检索和重排是否真正帮助使用者完成任务？ | Recall@k、nDCG@k、引用的实际支持性、下游正确率及成本 |
| 3 | 普通执行反馈、guard 反馈与最终审计的拆分 | 改进来自一般自调试、具体缺陷反馈，还是提交门槛？ | 同预算修复率、错误接受率、正确产物误拒率、修复轮数 |
| 4 | 果蝇记忆的缺失配对对照及统一 API 验证 | 风险门控下的收益变化能否归因于学习？库的模型接口是否保留学习行为？ | 相同门控下学习／冻结配对、延迟反馈、保存恢复一致性、普通学习器基线 |

第 2 项先冻结独立标注的问题、相关片段、知识快照及上下文预算；在相同候选集合上
比较 Jev 重排的增量。Jev 必须有真实服务响应与模型版本记录，mock 结果不能代替。
引用格式合法不等于内容得到来源支持。历史查询的语料和检索统计要使用相同时间截断。

第 4 项不重复已有复现。已有果蝇研究包含原方程数值核对、反馈机制和市场开发实验；
`fly_reuse/RESULTS.md` 明确记录缺少“冻结记忆＋相同风险门控”的组合。这一缺口比
继续增加训练种子更有助于判断学习的贡献。已有单一行情上的多种随机种子不能当作
多个独立市场。相关依据见 [果蝇接入结果](../benchmarks/fly_reuse/RESULTS.md) 与
[原代码及接口复现](../benchmarks/fly_paper/RESULTS.md)。

## 首轮冻结范围

只重跑 `Qwen/Qwen2.5-Coder-14B-Instruct`，版本
`aedcc2d42b622764e023cf882b6652e646b95671`；此版本来自保存的模型清单。
市场种子为 11、23、37，每个市场四条件各一次，共 12 次。
四条件保留为无库指导、技能文本、可选 guards、强制最终审计；每次仍为 16 轮、
每轮最多 2,048 个输出 token。模型加载失败和未完成产物也保留，不补写成功结果。

这些都是公开生成器的开发市场，目的是恢复可评分性，不能称为独立留出主实验。
只有检查完整结果后，才决定正式样本量、独立市场及后续模型；没有因旧结果较差而
重复采样直到成功，也没有把 32B 推理故障解释为模型能力。

## 本轮修复

保存的 14B 记录有 7 次接受，但独立有效评分为 0，见
[原始可行性汇总](../benchmarks/agent_study/BEACON_FEASIBILITY_RESULTS.json)。
本地保存的错误日志显示，多次提交按 `date` 读取日期，而矩阵 CSV 的日期列没有名称。

现在导出的三个矩阵都明确使用 `date`，TASK.md 与 manifest 给出同一读取接口。
所有条件收到相同的接口说明，并明确先写代码和报告再调用会读取二者的检查工具。
任务接口版本记为 2，每次推理前额外冻结任务说明、manifest 和数据文件的 SHA-256。
没有修改 oracle 的数值评分标准、强制审计门槛或原来的模型结果。

本轮同时改变了接口说明，因此不能与旧运行直接合并，宣称成绩变化全部来自技能。

## 输出位置与运行包

预定远端目录：

```text
/beacon-projects/radfm/wy891/fin-skills-followup-20260922-v1/
```

源代码、结果、应用缓存、临时文件、pytest 缓存和 Slurm 标准输出／错误均配置到
RADFM。只以离线模式复用旧 RADFM 目录的模型缓存及 Python 环境。提交脚本显式设置
绝对 `--chdir`、`--output` 和 `--error`，运行器拒绝非 RADFM 路径及输出目录中的符号链接。
没有修改 HOME 或设置把 HOME 当实验根目录的回退路径。

本地运行包位于 `D:/fin_skill/runs/beacon-followup-20260922-v1/`，其中有
`source.tar.gz`、`bundle-manifest.json`、`submit.sh` 和 `plan/protocol.json`。
运行包只包含允许的源码及所需测试；不包含凭据、旧实验输出或论文归档。
`bundle-manifest.json` 保存实际源码哈希及工作树状态，标为 `prepared_not_submitted`。

准备入口为 [prepare_followup.py](../benchmarks/agent_study/prepare_followup.py)，它只在
本地生成运行包，不登录、上传或提交。运行入口为
[beacon_followup.sh](../benchmarks/agent_study/beacon_followup.sh)。

部署入口为 [deploy_beacon_followup.py](../scripts/deploy_beacon_followup.py)。它先逐一核验
压缩包文件及哈希，登录时核对固定主机指纹，只上传三个明确列出的运行包文件，并在
提交前后保存状态回执。远端目录或本地部署回执已存在时拒绝盲目重试；连接中断后的
`submission_in_progress` 状态必须先检查远端作业，不能直接再提交。

在本机 PowerShell 中从库目录运行：

```powershell
Set-Location D:/fin_skill
python -m scripts.deploy_beacon_followup runs/beacon-followup-20260922-v1
```

密码只在本机终端的隐藏输入提示中输入，不写进命令、论文或回执。该入口不会搜索旧
会话或自动查找密码文件。添加 `--dry-run` 只做离线校验，不登录、不写文件、不提交。
实际提交成功才会生成含 Slurm 作业号的 `deployment-receipt.json`；它不能代替模型结果。

## 已验证与尚未完成

有序生成与库校验通过：129 个 skills；保留现有的 `fin-libraries` 发现预算警告。
本轮相关测试 **23 项通过**，覆盖任务 CSV 接口、真实检查调用、汇总分母及运行包哈希／
路径验证。两个 Bash 脚本的语法检查通过；只生成计划的命令确认 12 个实验单元。
这不证明 Beacon 环境、GPU 分配或真实模型重跑已成功。

最新只读连接检查已到达 Beacon SSH 端点，但服务器拒绝现有密钥，返回
`Permission denied (publickey,gssapi-keyex,gssapi-with-mic,password)`。认证未通过，
远端检查命令没有执行。随后，从旧任务记录定位已有登录配置路径的读取被自动安全
审核拦截，返回原因是无法确定请求的安全状态。没有通过其他读取途径绕过这次拦截。
因此没有新的远端登录、上传、Slurm 作业号或模型结果。当前任务不能记为已在 Beacon
开跑。新结果尚未形成，也未加入论文结果表或同步到 Overleaf。

续接核验确认了以下状态：

- 原完整回归在默认 Anaconda 环境结束：3,304 项通过、43 项失败、284 项跳过、
  51 项未选择，另有 10 项错误。该环境的 Pandas 为 2.1.4，低于项目要求的 2.2.2；
  不能据此宣称受支持环境的完整测试通过，也不能把这些错误计入模型实验结果。
- 在 `.venv-professional` 的 Pandas 2.3.3 / NumPy 2.4.6 / SciPy 1.17.1 环境，
  上述四组 Agent 测试共 23 项通过；生成器还报告了一处旧 `Q` 日期频率警告。
- v1 压缩包的 550 个源码文件逐一通过哈希核验，压缩包哈希也与 manifest 一致。
  Bash 运行入口没有 CRLF 换行；本地没有提交回执。受支持环境的完整回归仍需另行记录。
- 部署入口离线核验仍得到同一 v1 压缩包哈希，550 个源码文件、12 个计划单元。
  新增部署检查与引擎检查共 21 项通过；这些是本地测试，不是远端运行证据。
- 已修正引擎依赖检查中的标准库白名单，使其接受日期兼容逻辑使用的 `re`。
  受支持环境的完整回归已启动，最后一次成功读取的日志进度为 81%，当时未显示失败。
  后续读取运行回执的调用被自动安全审核拦截（无法确定安全状态），本次未取得结束
  回执，不能宣称完整回归通过。日志为 `full-regression-supported.log`。

原失败日志保留在 `runs/beacon-followup-20260922-v1/full-regression.log`；受支持环境的
Agent 测试日志为同目录的 `targeted-supported.log`。v1 保持冻结；任何后续源码修复
都必须生成不同版本的运行包，不能覆盖 v1。

路径选项的依据为 [Slurm sbatch 文档](https://slurm.schedmd.com/sbatch.html)；模型缓存与
离线选项的依据为 [Hugging Face 环境变量文档](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables)，
查阅于 2026-09-22。这些文档不证明当前 Beacon 的资源与账户配置已经重新核实。
