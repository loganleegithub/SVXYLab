# SVXYLab：本地研究工作区

日期：2026-09-07。

**P4 联合预测已实现，等待阶段验收；当前开发段的主要预测损失未优于无特征 M0。** 真实数据冻结覆盖 2019-01-02 至 2026-09-04 的 1930 个股票交易日；本阶段只重放 2020–2023 决策日期，不使用 2024 年后的标签值评分或选参。查看 `reports/predictions.html` 的 M0/M1/M2 预测、校准、分位覆盖和实际测试；阶段进度见 STATUS.md。

本轮数据请求与研究（含预热、训练和目标）统一从 **2019-01-01** 起，至 P1 冻结时最近共同完成交易日，原因是 SVXY 在 2018 年发生目标倍数变更。要求至少 400 行完整成熟训练样本，首次可预测决策日为 **2020-09-11**，此前 175 个请求决策日没有预测。当前定义见 RESEARCH_SPEC.md、experiment.toml，变更原因与影响见 runs/p1/experiment_change.json。

## 用户只需要做什么

1. 双击 `reports/predictions.html` 查看逐期预测、分时期损失、概率校准、交互作用与核对结果。
2. 核对 STATUS.md 后验收 P4；本轮停在此阶段。
3. 重新计算使用 `.venv/bin/python -m svxylab predictions --open`，只读取已验收的冻结数据。首个合资格月试运行加 `--pilot`。实际命令和普通测试说明见 AGENTS.md。

## 文件

- AGENTS.md：简短工程协作说明。
- RESEARCH_SPEC.md：研究/数据/模型/回测/持仓协议。
- FEATURES.json：19 个核心特征、2 个扩展特征、6 个交互及语义。
- experiment.toml：研究默认值。不是最佳参数或资金授权。
- SOURCES.md：官方资料、论文、附件和证据边界。
- TASKS.md：P0～P7 分阶段施工、审查和恢复提示词。
- STATUS.md：当前工程进度、研究状态、真实数据覆盖与验收状态。
- src/svxylab / tests：单一 Python 包与普通 pytest；合成夹具不进入真实数据。
- reports/predictions.html：P4 预测报告；reports/features.html、reports/baselines.html、reports/data.html、reports/environment.html 保留 P3/P2/P1/P0 快照。
- data/raw / data/clean：原始响应与清洗 CSV；下载时间、来源和 SHA-256 在 data/raw/downloads.jsonl。
- runs/p0 / runs/p1 / runs/p2 / runs/p3 / runs/p4：实际命令、失败、核验、冻结和实验变更记录。
- data/clean/p2/<UTC时间>：每次运行的总回报、时钟、标签、六条账户 CSV 与真实净值图。
- data/clean/p3/<UTC时间>：每次运行的核心/扩展特征、输入/来源/窗口/可用性、覆盖/缺失、相关性 CSV 与图。
- data/clean/p4/<UTC时间>：每次滚动训练成员、模型/变换快照、逐日预测、年度内层候选及评价结果；先前运行完整保留。
- requirements-lock.txt：本次项目虚拟环境的实际依赖版本。

SVXY/SPY 正式原件来自 Yahoo Chart，指数和 VX 来自 Cboe。Yahoo 供应商 OHLCV 已拆分调整；原值与复原到当时份额单位的列分开保留并明确标记。QuantConnect 已在外部浏览器实际核对，完整云端文件未取得到本地，不计入本地覆盖。SKEW 缺 2019-07-05、2024-11-29 两日，不阻塞核心。

P0 本地 Git 检查点为 `f3cc00f`，P1 为 `dc3d4ff`，P2 为 `5912f3b`，P3 为 `541178b`，无远端。P2 页面净值与汇总、P3 相关性只展示 2019–2023，全段 CSV 保留；封存模型评价仍属 P7。当前目录没有 START_HERE.html 或 inputs/reference；未重建或声称读取缺失附件。

数字预测由固定计算程序产生，不由 LLM 日常自由解释。数据不完整、模型没有优势和代码尚未完成，是不同状态，必须分开报告。历史精确发布时间尚未取得，继续使用明确的下一交易日可用假设，不声称完整历史 PIT。
