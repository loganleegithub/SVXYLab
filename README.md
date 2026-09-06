# SVXYLab：本地研究工作区

日期：2026-09-07。

**P1 数据取得与核对已完成；Mac 锁屏，页面显示确认待解锁后完成。没有回测结果或已验证因子。** 核心真实数据覆盖 2019-01-02 至 2026-09-04 的全部 1930 个股票交易日。实际结果、源间差异和剩余限制见 `reports/data.html` 与 STATUS.md。

本轮数据请求与研究（含预热、训练和目标）统一从 **2019-01-01** 起，至 P1 冻结时最近共同完成交易日，原因是 SVXY 在 2018 年发生目标倍数变更。仍要求至少 400 行完整成熟训练样本，因此实际首次预测晚于 2020 年初，日期待真实数据核验后确定。当前定义见 RESEARCH_SPEC.md、experiment.toml，变更原因与影响见 runs/p1/experiment_change.json。

## 用户只需要做什么

1. 双击 `reports/data.html` 查看真实数据覆盖、三个时点、换月边界、拆分核对和来源差异。
2. 核对 STATUS.md 后验收 P1。只有验收后才进入 P2 的总回报和持仓账本。
3. 需要重新计算时，唯一入口为 `.venv/bin/python -m svxylab data --open`；已有原件优先走缓存。实际命令和测试说明见 AGENTS.md。

## 文件

- AGENTS.md：简短工程协作说明。
- RESEARCH_SPEC.md：研究/数据/模型/回测/持仓协议。
- FEATURES.json：19 个核心特征、2 个扩展特征、6 个交互及语义。
- experiment.toml：研究默认值。不是最佳参数或资金授权。
- SOURCES.md：官方资料、论文、附件和证据边界。
- TASKS.md：P0～P7 分阶段施工、审查和恢复提示词。
- STATUS.md：当前工程进度、研究状态、真实数据覆盖与验收状态。
- src/svxylab / tests：单一 Python 包与普通 pytest；合成夹具不进入真实数据。
- reports/data.html：P1 数据报告；reports/environment.html 保留 P0 环境快照。
- data/raw / data/clean：原始响应与清洗 CSV；下载时间、来源和 SHA-256 在 data/raw/downloads.jsonl。
- runs/p0 / runs/p1：实际命令、失败、核验、冻结和实验变更记录。
- requirements-lock.txt：本次项目虚拟环境的实际依赖版本。

SVXY/SPY 正式原件来自 Yahoo Chart，指数和 VX 来自 Cboe。Yahoo 供应商 OHLCV 已拆分调整；原值与复原到当时份额单位的列分开保留并明确标记。QuantConnect 已在外部浏览器实际核对，完整云端文件未取得到本地，不计入本地覆盖。SKEW 缺 2019-07-05、2024-11-29 两日，不阻塞核心。

P0 本地 Git 检查点为 `f3cc00f`，无远端。当前目录没有 START_HERE.html 或 inputs/reference；未重建或声称读取缺失附件。

数字预测由未来实现的固定计算程序产生，不由 LLM 日常自由解释。数据不完整、模型没有优势和代码尚未完成，是不同状态，必须分开报告。
