# 公开来源与证据边界

核对日期：2026-09-07。以下为公开资料与方法来源，不是本系统的验证结果。

## S01｜ProShares：SVXY 产品定义与 2018 年目标倍数变更

https://www.proshares.com/our-etfs/strategic/svxy

用途与边界：每日 -0.5 倍目标、实际基金与指数区别；不支持任何预测收益承诺。

## S02｜S&P VIX Futures Indices Methodology

https://www.spglobal.com/spdji/en/documents/methodologies/methodology-sp-vix-futures-indices.pdf

用途与边界：最近两个月度合约的每日滚动规则；尤其第 5 页。不得把日历插值 F30 当成该指数。

## S03｜Cboe：VIX 与其他波动率指数历史数据入口

https://www.cboe.com/tradable-products/vix/vix-historical-data/

用途与边界：VIX、VVIX、VIX9D 的官方历史下载入口；进入数据阶段仍须实际下载验真。

## S04｜Cboe：期货历史行情入口

https://www.cboe.com/markets/us/futures/market-statistics/historical-data/futures/

用途与边界：页面列明 2013 年至今部分期货产品逐合约价格/成交数据。不是本项目已拥有完整历史的证明。

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

用途与边界：TIME_SERIES_DAILY_ADJUSTED 提供原始 OHLCV、复权收盘、拆分/分红；该接口标为 Premium。具体 SVXY/SPY 覆盖与授权仍需核实。

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

## 数据验证现状

本次已通过网页核对官方数据入口与文档，尚未取得/核验本项目所需完整行情。当前沙盒尝试直连 cdn.cboe.com 时 DNS 解析失败；这不能证明用户 Mac 上无法下载，也不能证明任何端点已成功返回历史数据。不要把“来源已核实”写成“数据已取得”。P1 必须在用户 Mac 实际执行。

候选 Cboe 日线文件名为 VIX_History.csv、VVIX_History.csv、VIX9D_History.csv、VIX3M_History.csv、SKEW_History.csv。先从官方页面/指数页面解析当前下载地址。前面三个有官方历史页链接；后两个须进一步用各自官方指数页和真实响应验证。禁止只按名字拼接成功就宣称完整。

## 数据购买边界

本包不包含付费授权，不默认用户订阅任何行情。优先验证 Cboe 官方公开文件。ETF 历史价的首选正式接口候选为 S13，但只有在确认具体标的、历史日期、字段、下载权限和现价后，才向用户提出一次明确的数据购买决定。不得自动购买，不在聊天/日志/ Git 中粘贴 API key。用户已有合法数据可直接适配，不重复购买。
