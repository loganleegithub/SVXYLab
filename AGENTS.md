# SVXYLab：给 Codex 的项目说明

用户是中文交易员，不会编程；本项目从零开始。此前 VVIX 等零散实验不是已验证因子库；不得默认复用 MatVIX、MatSHIX、Optimatrix 的方法或成果。

先读本文件、RESEARCH_SPEC.md、FEATURES.json、experiment.toml 和 TASKS.md 的当前阶段。SOURCES.md 是资料索引，不是新的一层权威。只有一份当前研究规格，不新增宪法、权限平台或证明系统。

## 做什么
建设本地、日频、SVXY/CASH 的联合研究程序：真实数据→联合特征→滚动预测→真实时钟持仓回测→中文报告。LLM 负责开发和研究协助，不在每天临场解释后改权重。

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
P0 已于 2026-09-07 在 `/Users/logan/SVXYLab` 实际验证。复用本机 Python 3.12.13，项目虚拟环境为 `.venv`。以下命令均在项目根目录执行，用户无需手写代码。

唯一项目命令入口（检查环境、运行 pytest、生成并打开报告）：
```sh
.venv/bin/python -m svxylab environment --open
```
实际结果：退出码 0；生成并打开 `reports/environment.html`。每次运行的 JSON 与 pytest XML 保存在 `runs/p0/<UTC时间>/`。

独立运行普通测试：
```sh
.venv/bin/python -m pytest -q
```
实际结果：4 passed。全部是明确标记的合成算术测试，不是真实回测。

本次已运行的环境建立命令（环境已存在时不要重复创建）：
```sh
/Users/logan/.pyenv/versions/3.12.13/bin/python3 -m venv .venv
.venv/bin/python -m pip --isolated --disable-pip-version-check install --no-cache-dir --retries 1 --timeout 15 --index-url https://pypi.org/simple/ -r requirements-dev.txt
.venv/bin/python -m pip --isolated --disable-pip-version-check install --no-build-isolation --no-deps --no-index -e .
```
安装命令均成功；实际版本见环境报告。P0 仅有环境报告与纯算术功能，研究配置、特征、模型和行情处理尚未实现。Git 已在本目录初始化，无远端；待用户验收后再保存本地提交。
