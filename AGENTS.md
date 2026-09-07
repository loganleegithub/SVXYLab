# SVXYLab：给 Codex 的项目说明

用户是中文交易员，不会编程；本项目从零开始。此前 VVIX 等零散实验不是已验证因子库；不得默认复用 MatVIX、MatSHIX、Optimatrix 的方法或成果。

先读本文件、RESEARCH_SPEC.md、FEATURES.json、experiment.toml 和 TASKS.md 的当前阶段。SOURCES.md 是资料索引，不是新的一层权威。只有一份当前研究规格，不新增宪法、权限平台或证明系统。

## 做什么
建设本地、日频、SVXY/CASH 的联合研究程序：真实数据→联合特征→滚动预测→真实时钟持仓回测→中文报告。LLM 负责开发和研究协助，不在每天临场解释后改权重。

本轮数据请求与研究（含特征预热、训练和目标）统一从 2019-01-01 起；这是用户因 2018 年 SVXY 目标倍数变更而确认的选择。边界与首次预测条件见 RESEARCH_SPEC.md 和 experiment.toml，变更记录见 runs/p1/experiment_change.json。

## 怎么做
- 每轮只完成用户指定的一个阶段。先给不超过五点的执行计划，然后自主完成常规实现、运行和修复；阶段完成后停下让用户验收。
- 先检查现有 Python 与依赖；优先成熟库。用一个 Python 包、一个实验配置、一套普通测试、本地 CSV/JSON 和 HTML 报告。首版不建数据库服务、Web 后端、Docker、自治 Agent 平台或实时行情系统。
- 本地单写入线程。可让只读子智能体核对资料/时钟/计算；不让多个代理同时改同一工作区。
- 模型、公式、训练窗口、时点、成本、样本区间和指标调整必须写入实验记录，不能为改善结果而悄悄改目标。没有优势也要交付真实结果。
- 保留原始数据与来源、下载时间、摘要。数据抓取失败不合成行情、不改成代理冒充真实 VX。模拟数据只允许在明确标记的单元测试夹具中使用。
- 标准化、样条节点、参数选择都只在当时训练样本拟合；标签必须成熟。不可把概率、历史分位、模型分数混称。
- 基金真实收益已包含的费用不能重复扣。回测按下一交易日收盘及明确成本代理执行，不得同日收盘偷跑。
- 网络重试有界。失败后定位 DNS/权限/HTTP/许可/字段层面的确切原因，不无限换数据源。
- 不读交易账户，不发订单，不上传付费原始行情，不打印密钥；不自动购买或改系统级安全设置。
- 新增规则只针对实际重复错误。不要写测试锁定散文措辞，不用降低断言来掩盖计算错误。

## 交付与进度
每阶段交付真实可打开的报告、实际运行命令和结果、最重要的未解决项。更新 STATUS.md。分开写“工程完成/未完成”“研究结果”“真实数据覆盖”，不使用一个 PASS 代替三件事。
完成后请用户验收；验收后再保存本地 Git 检查点。远端 GitHub、PR、云运行不作为首版前置。
断线后读 STATUS.md 和现有文件继续，不重新 clone、不重新建架构、不删除既有原始数据。

## 命令
以下命令在 `/Users/logan/SVXYLab` 执行。复用本机 Python 3.12.13 和已有 `.venv`；不重复建环境。唯一项目入口为 `.venv/bin/python -m svxylab`。

P1 数据复算与报告：
```sh
.venv/bin/python -m svxylab data
```
P1 历史运行退出码 0；核心共同样本 1930 日，冻结至 2026-09-04，当时普通 pytest 18 项通过。macOS 打开报告命令返回 0；当时的工具锁屏记录保留。用户现已明确验收 P1，检查点为 `dc3d4ff`。每次运行记录与 pytest XML 在 `runs/p1/<UTC时间>/`。优先验证并复用原件缓存，不刷新冻结截止日。

P2 总回报、时钟、标签与连续账本：
```sh
.venv/bin/python -m svxylab baselines
```
2026-09-07 实际退出码 0，普通 pytest 38 项通过；六个对照各 1930 行，五日标签 1924 行完整、6 行尚未成熟。加 `--open` 生成后由 macOS 打开 `reports/baselines.html`。只读取 P1 冻结文件，不调用取数或模型；每次 CSV/图保存在 `data/clean/p2/<UTC时间>/`，运行 JSON 与测试结果在 `runs/p2/<UTC时间>/`，已有结果不覆盖。页面净值与汇总展示 2019–2023；2024 年后仅用于公司行动核账和时钟示例。

P3 原始联合特征与信息重叠：
```sh
.venv/bin/python -m svxylab features --open
```
仅读取冻结 P1 文件、已验收 P2 总回报与时钟；不读标签进行特征计算，不取数或拟合。19 项核心 CSV 与两项 SKEW 扩展 CSV 分开保存，保留每个交易日及缺失。当前核心完整 1909 日，核心加扩展共同完整 1905 日，首个完整日均为 2019-02-01。2026-09-07 首次运行退出 0，普通 pytest 69 项通过；每次输出在 `data/clean/p3/<UTC时间>/`，命令/测试/审计在 `runs/p3/<UTC时间>/`。报告为 `reports/features.html`，相关性仅展示 2019–2023；公式、窗口与展示记录见 `runs/p3/experiment_record.json`。

独立普通测试：
```sh
.venv/bin/python -m pytest -q
```
合成夹具只检查复利算术、日期/合约选择、字段/公司行动解析、账本、特征公式及时间边界，不是真实回测；真实数据核对另见各阶段报告。

P0 已验证的环境报告入口：
```sh
.venv/bin/python -m svxylab environment --open
```
历史 P0 运行退出码 0、4 项合成测试通过；原报告和 `runs/p0/` 保留。今后再运行会检查当前环境，并引用最新 P1 覆盖记录，不用文件数量推定行情覆盖。

依赖锁定在 `pyproject.toml` 和实际版本快照 `requirements-lock.txt`；P2 在同一虚拟环境加入 Matplotlib 3.11.1。已成功执行的本地包元数据刷新命令：
```sh
.venv/bin/python -m pip --isolated --disable-pip-version-check install --no-build-isolation --no-deps --no-index -e .
```
原始安装、失败及成功结果在 `runs/p0/setup.json`、`runs/p1/dependency_install.json`、`runs/p1/yfinance_install_failure.json`、`runs/p1/yfinance_install_cached.json` 和 `runs/p1/editable_refresh.json`。

正式 ETF 仅用 Yahoo Chart 原件；`provider_*` 为供应商值，OHLCV 当时份额复原字段有明确标记。保留 Nasdaq/QC 参考差异以及 P2 的 SPY 分红总回报与供应商复权参考差异。P3 已实现原始特征，未实现联合模型；停在 P3，验收后再继续。P0 检查点 `f3cc00f`、P1 检查点 `dc3d4ff`、P2 检查点 `5912f3b`，无 Git 远端，未改全局 Git 配置。P2 验收仅代表交付完整、真实与可复算，不代表模型有收益优势。
