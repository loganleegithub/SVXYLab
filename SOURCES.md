# 公开来源与证据边界

核对日期：2026-09-07。以下为公开资料与方法来源，不是本系统的验证结果。

本轮数据请求与研究计算统一从 2019-01-01 起（含特征预热和训练），截至 P1 冻结时最近共同完成交易日。下列来源自身的历史覆盖、基金制度变更日期与本轮研究起点分别记录；来源提供更早历史不扩大本轮样本。

## S01｜ProShares：SVXY 产品定义与 2018 年目标倍数变更

https://www.proshares.com/our-etfs/strategic/svxy

用途与边界：每日 -0.5 倍目标、实际基金与指数区别。2026-09-07 复核官方产品页：目标倍数于 2018-02-27 收盘后从 -1x 改为 -0.5x，2018-02-28 以前的收益反映旧目标。用户据此将本轮起点统一为 2019-01-01；不支持任何预测收益承诺。

## S02｜S&P VIX Futures Indices Methodology

https://www.spglobal.com/spdji/en/documents/methodologies/methodology-sp-vix-futures-indices.pdf

用途与边界：最近两个月度合约的每日滚动规则；尤其第 5 页。不得把日历插值 F30 当成该指数。

## S03｜Cboe：VIX 与其他波动率指数历史数据入口

https://www.cboe.com/tradable-products/vix/vix-historical-data/

用途与边界：VIX、VVIX、VIX9D 的官方历史下载入口；P1 已解析页面实际链接并下载、核验 CSV。

## S04｜Cboe：期货历史行情入口

https://www.cboe.com/markets/us/futures/market-statistics/historical-data/futures/

用途与边界：来源索引原记录为页面列明 2013 年至今部分期货产品逐合约价格/成交数据，这是供应商覆盖说明；本项目只请求并研究 2019-01-01 起的数据。不是本项目已拥有完整历史的证明。

## S05｜Cboe：期货每日结算价格

https://www.cboe.com/markets/us/futures/market-statistics/settlement/futures/daily/

用途与边界：核对结算字段、不同合约与日历；最终结算值与平日结算价不得混用。

## S06｜Cboe：VVIX 定义

https://www.cboe.com/us/indices/dashboard/vvix/

用途与边界：来自 VIX 期权的 30 天波动率之波动率定价；不是 SVXY 下跌概率。

## S07｜Cboe：SKEW 定义

https://www.cboe.com/us/indices/dashboard/skew/

用途与边界：尾部偏斜定价信息；不能直接读成现实世界崩盘概率。

## S08｜Nasdaq：Nations 指数符号历史通知

https://www.nasdaqtrader.com/TraderNews.aspx?id=dtn2017-9

用途与边界：确认 VOLI/VolDex、SDEX/SkewDex、TDEX/TailDex 的名称映射。不是现时连续历史、许可或方法版本证明。

## S09｜Nations：VolDex 方法介绍

https://nationsindexes.com/indexes/voldex/

用途与边界：平值附近期权的隐含波动率指标；本方案不接受网页对自身优越性的宣传作为验证。

## S10｜Nations：当前指数阅读说明

https://nationsindexes.com/education/reading-the-nations-indexes/

用途与边界：TailDex 等现行定义。不得未经方法版本核对，把当前 RiskDex 接到历史 SDEX 上。

## S11｜Cboe：做市商 Gamma 敞口研究

https://www.cboe.com/insights/posts/volatility-insights-evaluating-the-market-impact-of-spx-0-dte-options/

用途与边界：净 Gamma 需要净头寸信息，不能用市场总 OI 自动推定做市商买卖方向。

## S12｜Cboe DataShop：数据常见问题

https://datashop.cboe.com/faqs

用途与边界：OI 使用前夜 OCC 数据，次日更新；不能把日终 OI 当成同日盘中事实。

## S13｜Alpha Vantage：官方 API 文档

https://www.alphavantage.co/documentation/

用途与边界：TIME_SERIES_DAILY_ADJUSTED 提供原始 OHLCV、复权收盘、拆分/分红；该接口标为 Premium。P1 demo 对 SVXY/SPY 均返回 HTTP 200 授权提示，没有日线。未购买；正式 ETF 数据实际选用 S32。

## S14｜Johnson (2017)：Risk Premia and the VIX Term Structure

https://www.cambridge.org/core/journals/journal-of-financial-and-quantitative-analysis/article/risk-premia-and-the-vix-term-structure/56572D1F060448571BD8F597C732D9C3

用途与边界：期限结构风险定价机制依据，不是对本方案经济性的证明。

## S15｜Corsi (2009)：A Simple Approximate Long-Memory Model of Realized Volatility

https://academic.oup.com/jfec/article-abstract/7/2/174/856522

用途与边界：多时间尺度波动记忆依据。本方案日收盘估计只是低频代理，不冒充原论文高频 realized volatility。

## S16｜Gu, Kelly, Xiu：Empirical Asset Pricing via Machine Learning

https://www.nber.org/papers/w25398

用途与边界：非线性与交互的一般研究依据；股票研究结果不能直接搬为 SVXY 成功证据。

## S17｜Cheng：The VIX Premium

https://utoronto.scholaris.ca/items/1cfdb03f-e696-46b1-959c-15bcb51163d5

用途与边界：VIX 期货价格与物理预期之间的溢价，与 VIX² 减历史 RV² 不同。

## S18｜scikit-learn：SplineTransformer

https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.SplineTransformer.html

用途与边界：成熟样条基函数实现；参数在本包中是研究设计，不是官方交易建议。

## S19｜scikit-learn：QuantileRegressor

https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.QuantileRegressor.html

用途与边界：线性分位数回归与 pinball 损失的成熟实现。

## S20｜OpenAI：GPT-6 Astra

https://openai.com/index/gpt-6-astra/

用途与边界：模型发布与能力介绍，不作为量化有效性保证。

## S21｜OpenAI：ChatGPT Work and Codex

https://help.openai.com/en/articles/20001275-chatgpt-work-and-codex

用途与边界：新版桌面左上角进入 Codex；与普通 Chat 历史不同；本地工作不代表所有任务上下文不上云。

## S22｜OpenAI：桌面应用与更新说明

https://help.openai.com/en/articles/6825453-chatgpt-release-notes

用途与边界：旧 Codex app 更新到新版 ChatGPT desktop；Astra 为逐步开放。

## S23｜OpenAI：Models

https://learn.chatgpt.com/docs/models

用途与边界：输入框下方模型选择、reasoning effort、Max/Ultra 与子智能体；以实际账号可见选项为准。

## S24｜OpenAI：Codex environments

https://learn.chatgpt.com/docs/environments/modes

用途与边界：Local、Worktree、Cloud；Local 与 Worktree 在自己的电脑执行。

## S25｜OpenAI：Best practices

https://learn.chatgpt.com/guides/best-practices

用途与边界：明确目标、复杂任务先计划、简短实用 AGENTS.md、测试验证。

## S26｜OpenAI：Code review

https://learn.chatgpt.com/docs/code-review

用途与边界：/review、Git 仓库前提、审查未提交改动、Detached 审查线程。

## S27｜OpenAI：Permissions

https://learn.chatgpt.com/docs/permissions

用途与边界：权限按工作区和任务控制，不建议为下载数据而开放整台机器。

## S28｜OpenAI：Computer Use

https://learn.chatgpt.com/docs/computer-use

用途与边界：仅图形界面工作确有需要时安装官方插件并授权；非本项目首轮必需。

## S29｜OpenAI：AGENTS.md

https://learn.chatgpt.com/docs/agent-configuration/agents-md

用途与边界：项目指导文件的读取和作用。

## 附件对应

P01：`inputs/reference/晴雨表量化引擎.pdf`，4 页。重点第 3 页 Data Engine 图片与期限结构切换描述。
P02：`inputs/reference/年化75%，回撤不到4%：一个被99%投资者忽略的量化策略.pdf`，3 页。重点第 2 页机器学习流程与 23.8% 仪表盘。

PDF 未披露完整特征公式、模型、训练窗口、百分比含义、执行时点、成本或样本外协议。本包是独立研究设计，不是原模型复刻。

以上为规格包原有附件索引。P1 开始时当前目录没有 inputs/reference，本次没有重新读取或复用这些附件中的结果。

## 数据验证现状

规格包准备期的旧观测：沙盒尝试直连 cdn.cboe.com 时 DNS 解析失败，当时尚未取得完整行情。此为历史失败，不能代表本轮 Mac 的网络结果。P0 HEAD 实测官方入口 HTTP 200、CDN 根地址 HTTP 403；P1 对具体 CSV 的实际请求已成功，失败与成功正文均保留。

P1 已从 S03、VIX3M 官方页面 https://www.cboe.com/us/indices/dashboard/vix3m/ 及 SKEW 官方页面解析并核验五个历史 CSV。四个核心指数均为 1930/1930 个股票交易日；SKEW 1928/1930，缺 2019-07-05 与 2024-11-29。VX 前三月度合约完整 1930 日；SVXY/SPY 各 1930 日。实际请求起点 2019-01-01，首日 2019-01-02，冻结末日 2026-09-04。报告与原件清单是本次具体取得情况的依据。

## S30｜Cboe 标准月度 VX 官方合约目录

https://www-api.cboe.com/us/futures/market_statistics/historical_data/product/list/VX/

P1 用 futures_root=VX、duration_type=M 和官方 expire_date 选出 95 份月度合约（2019-01 至 2026-11）。本机 `/Users/logan/DATA/VIX` 的逐合约原件先校验既有清单摘要；三个到期月与本次官方下载逐字节一致，缺少的合约另从官方地址下载。研究范围内保留 16910 行逐合约结算，使用 Settle，不能将逐合约行数当作样本日数。导入时间与未知的历史下载时间分开保存。

## S31｜ProShares：2024 年真实公司行动

https://www.proshares.com/press-releases/proshares-announces-etf-share-splits2

SVXY 在公告第二批名单，2:1 拆分生效于 2024-04-11 开盘前；第一批其他基金的 4 月 10 日不能套用到 SVXY。P1 已与主源拆分事件及 QuantConnect RAW 日线抽查相符。只核对一日数据算术，完整总回报属于 P2。

## S32｜Yahoo Chart 原件与 yfinance 客户端

https://pypi.org/project/yfinance/1.7.0/

https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html

https://in.help.yahoo.com/kb/adjusted-close-sln28256.html

本轮唯一正式 ETF 适配器。原生 curl 曾 HTTP 429；成熟客户端实际请求 query2.finance.yahoo.com/v8/finance/chart/SVXY 和 SPY 返回完整原始 JSON。显式关闭 auto_adjust、back_adjust、repair，开启 actions、keepna；请求包含 div,splits,capitalGains，避免将未请求字段的缺键误记为零。两只 ETF 各 1930 日；SVXY 一次生效拆分、无报告分红，SPY 30 次分红、无拆分。

quote OHLCV 已拆分调整，provider_* 保留供应商值，清洗 OHLCV 明确复原到当时份额单位。adjclose 只作复权参考，不与其他口径混除。客户端开源与公开可读不等于获得再分发授权；本轮用于用户明确的个人非商业研究。决策和源间价格/成交量差异见 runs/p1/source_decision.json，不推定数据绝对无误。

## S33｜QuantConnect：外部浏览器实际核对

https://www.quantconnect.com/datasets/algoseek-us-equities

https://www.quantconnect.com/datasets/quantconnect-us-equity-security-master

用户明确要求后，复用现有浏览器登录，在 P1 Research 项目实际运行 RAW/ADJUSTED 日线及公司行动查询，两只 ETF 各 1930 日；未运行模型、回测或交易。研究容器生成文件的两条常规链接均无法下载到本地；未绕过 Object Store 权限。有限浏览器核对与云端摘要保存在 runs/p1/quantconnect_observation.json，代码在 runs/p1/quantconnect_probe.py 和 quantconnect_export.py，不能把这些摘要冒充本地完整原件。

## S34｜Nasdaq：ETF 参考日线

https://www.nasdaq.com/market-activity/etf/svxy/historical

https://www.nasdaq.com/market-activity/etf/spy/historical

外部浏览器确认公开表格；其实际 historical 接口由 curl_cffi 取得两份 1930 日参考响应，保留在 data/raw。原生 curl 的 HTTP/2 错误、诊断超时、小窗口零行均保留。分红接口明确返回非 Nasdaq 标的历史不可用；2026-04-20 两只 ETF 成交量为 N/A、OHLC 全同。这些参考异常不覆盖正式主源，逐字段差异在 data/clean/ETF_source_comparison.csv。

## 数据购买边界

本包不包含付费授权，不默认用户订阅任何行情。本轮已取得核心所需的本地原件，无需购买；没有读取交易账户或修改系统权限。以后若确需付费数据，仍先核实具体覆盖与字段再提出唯一明确决定。不得自动购买，不在聊天/日志/Git 中粘贴 API key；已有合法数据不重复购买。
