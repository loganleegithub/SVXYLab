# 当前进度

更新日期：2026-09-07（Asia/Shanghai）。

状态：P0_ACCEPTED。用户已验收 P0 交付完整、真实且可复算；验收不代表模型有收益优势。本轮已授权先保存本地 Git 检查点，再仅执行 P1。

## 工程完成 / 未完成

P0 工程已完成：
- 实测 CPU 与 Python 进程均为 arm64，macOS 26.6.2（25G83）；复用现有 pyenv Python 3.12.13。系统另有 Python 3.9.6，本项目使用 `.venv` 中的 3.12.13。
- 建立一个 `src/svxylab` Python 包、项目虚拟环境、唯一 `python -m svxylab` 命令入口、普通 pytest。未安装系统软件。
- 已安装 svxylab 0.0.1、pip 25.0.1、setuptools 84.0.0、pytest 8.4.2、iniconfig 2.3.0、packaging 26.0、pluggy 1.6.0、Pygments 2.19.2；`pip check` 退出码 0。
- 独立执行 `.venv/bin/python -m pytest -q`：4 passed，退出码 0。包含合成指数 100→120→100、每日 -0.5x 理想化产品 100→90→97.5 的完整路径断言，以及 3 个无效输入检查；全部标记为 synthetic。
- 执行 `.venv/bin/python -m svxylab environment --open`：退出码 0，生成 `reports/environment.html`，内部再次运行 pytest 得到 4 项通过。
- `/usr/bin/open` 返回 0；浏览器清单确认 Chrome 已打开该本地文件，标题为“SVXYLab · P0 环境检查”。浏览器工具的 URL 策略不允许进一步读取本地 file 页面，未绕过；改用本地 HTML 解析检查文件链接，无脚本或假净值图。
- AGENTS.md 已补充实际验证的运行、测试与环境建立命令。
- Git 2.50.1（Apple Git-155）已在本目录初始化，分支 main，无远端、无提交。未修改全局 Git 配置；用户验收后再保存本地 Git 检查点。

后续工程尚未实现：真实行情适配、特征、时钟 / 持仓账本、模型、历史评估及日报。不以 P0 完成宣称整个研究程序完成。

## 研究结果

尚未开展真实研究或回测，没有有效性、收益或风险结论。100→90→97.5 只验证合成夹具的每日复利运算。

`RESEARCH_SPEC.md`、`FEATURES.json`、`experiment.toml`、`SOURCES.md` 保持原研究定义；实验配置仍为 DESIGN_ONLY。未复用旧项目或旧因子库，未调整模型、公式、训练窗口、时点、成本、样本区间或研究指标。

## 真实数据覆盖

尚未取得。`data/raw` 和 `data/clean` 实际行情文件均为 0，仅有目录占位文件；原始用户附件仍在 `inputs/reference`。

本次 Mac 实测网络：PyPI 索引、包文件域名、S03 Cboe 官方历史入口均为 HTTP 200；`https://cdn.cboe.com/` 根地址为 HTTP 403，已收到服务器响应，不能记作本次 DNS 失败。每个端点仅做 HEAD 检查，0 次重试，连接上限 5 秒，总上限 15 秒。

入口可达和 CDN 根地址拒绝均不能证明具体行情文件、字段、覆盖或许可。未下载行情正文、未请求付费数据、未读取交易账户。SOURCES.md 中旧沙盒 DNS 失败保留为历史记录，本机当前观测以上述结果为准。

## 交付位置与下一步

- 可打开报告：`reports/environment.html`。
- 本次检查、版本与配置 / 代码 SHA-256：`runs/p0/20260906T172649588287Z/environment.json`。
- pytest 结果：同目录 `pytest.xml`。
- 环境建立与初始网络检查记录：`runs/p0/setup.json`。
- 实际运行 / 测试命令：AGENTS.md 的“命令”部分。

P0 唯一真正技术阻塞项：无。真实数据尚未取得属于 P1 工作，不阻塞 P0 验收。

用户已验收 P0，正在将当前工作区保存为首个本地 Git 检查点；提交后在本文件记录提交号，再开始 P1。

本轮读取现状：TASKS.md 的 P1 请求起点为 2019 年，研究规格与实验配置仍为 2013 年，已向用户核对，暂不修改研究参数。当前目录没有 START_HERE.html 或 inputs/reference；本轮未删除、移动或重建这些文件，仅保存当前实际存在的交付。既有 runs/p0 记录与环境报告保持保留。
