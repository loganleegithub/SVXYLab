# E1 本地变更清单（等待验收）

新增实现：
- `panic_retreat.toml`：主B、A/C/D3/D5、三个期限、执行和十项单因素邻域。
- `src/svxylab/panic_retreat.py`：因果信号、as-of截面、复用会计、真实计算、紧凑证据。
- `src/svxylab/panic_retreat_report.py`：中文静态报告和真实路径图。
- `tests/test_panic_retreat.py`：16项行为/数值测试；合成夹具不写入真实数据。
- `reports/panic_retreat.html`：当前中文报告；验收注记与计算字段分开。
- `runs/panic_retreat/`：各次实验/输入/命令/核算、pytest XML、最终复算检查脚本与记录。

仅追加本轮入口与交付状态：`AGENTS.md`、`STATUS.md`、`README.md`；原历史正文保留。

`data/clean/panic_retreat/`在原有忽略规则下保存所有真实逐事件CSV、长账本、图表、JSON和历史截面；没有修改.gitignore或加入Git索引。最终结果目录为`20260909T171926132199Z`。此前初版和前次复算目录继续保留。

未修改原experiment.toml、RESEARCH_SPEC.md、__main__.py、release.py、ledger.py、returns.py、timing.py、旧冻结源码或旧报告。只验证实际使用的107份输入身份，不声称重新核验全部旧研究数千份产物。未重新拟合M2；没有新行情下载、模型/参数晋升、期权价格研究、监控服务、账户操作或下一实验。

验收状态：尚未验收。Git HEAD保持d04b2ebb3ae2599aea3c24e2c8c8d7d5046fdaa3；没有提交或推送。
