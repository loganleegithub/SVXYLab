"""N1一次性计算前记录；不改P7冻结版本。"""
from datetime import datetime, timezone
from pathlib import Path
import json, hashlib, subprocess
r=Path.cwd(); h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
record={
 'stage':'N1','recorded_at_utc':datetime.now(timezone.utc).isoformat(),
 'starting_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
 'authorization':'仅修复已有反例的日报推理后截止复查；资本与经济基准；期权仅资料核实；不购买、不连接账户、不扩展策略。',
 'research_attribute':'2024—2026已在P7揭示；N1是揭示后的描述性经济基准与路径归因，不是新封存检验或规则有效性验证。',
 'capital_usd':100000,'cost_per_traded_asset_one_way':0.0005,
 'periods':{'since_2019':['2019-01-03','2026-09-04'],'development_common':['2020-09-11','2023-12-29'],'revealed_2024':['2024-01-02','2026-09-04'],'continuous_common':['2020-09-11','2026-09-04']},
 'period_reason':'全历史简单基准含2020压力；开发/P7原共同执行起点用于逐项核对；连续共同段不在2024重置资本。2019-01-02仅作为第一信息/现金锚点，不读2018。',
 'existing_baselines':'保留现金0、每日固定25/50/75/100%、期限结构，以及原M0/M1/M2 main/risk-only；2019全段只比较简单对照，不能填造模型服务。',
 'new_fixed_weights':[0.05,0.10],
 'new_benchmarks':'100% SPY总回报、100% BIL短债总回报；5/10% SVXY分别配零收益现金或BIL，每日固定扣费后比例。',
 'risk_pool':'最初5/10%资本含首次费用划入独立SVXY买入持有池，余款独立现金0或BIL；禁止外部入金、储备向风险池划款或亏后恢复原比例。不借贷、不卖空，不按结果调整。收益留各自池；SVXY本段无分配。',
 'cash_proxy':'先选BIL：2007成立、持有1—3月国库券、月度分配，历史长于SGOV且与既有Yahoo价格解析兼容。仅实际每日价格和分配；不使用当前收益率倒填。除息日确认并收盘再投资为统一TR代理，非支付日到账、非用户实际利息；基金费不重复扣、税和汇率不建模。',
 'execution':'全部账户同资本与执行日期；上一真实交易日信息、当日正式股票收盘成交代理；旧持仓先记本日损益再交易；首日不取得入场前收益。期末市值不强平。双ETF再平衡两腿都收费。',
 'metrics':['净收益/CAGR','最大收盘回撤及峰谷日期','最差1/5交易日路径','最大回撤恢复时间与最长水下时间（含未恢复删失）','持有期平均敞口及收盘敞口','美元交易成本与换手','超过同日起跑BIL净收益的百分点及相对财富差'],
 'attribution':'复用原M2账本，按上一收盘实际持有敞口将每日持有涨/跌、空仓错失上涨/避免下跌、费用分开；用财富递推桥接精确合计期末差，不能把日收益相加当复利。列完整连续持有/空仓区间及每类最大两项路径，选择仅展示、不是新规则。比较risk-only/固定50%/BIL；同目标毛净只查成本。',
 'data_boundary':'SVXY/SPY/时钟/模型均复用P1—P7原件与验收产物；仅BIL新增2019-01-01至2026-09-04请求。无期权价格模拟。',
 'network':'BIL只走Yahoo Chart、最多2次有界请求；官方基金资料用于核实。无授权不购买数据。',
 'stop':'交付N1等待验收；不提交Git/推送、不自动开始保护性看跌回测。',
 'preserved_sha256':{}
}
for d in ['runs/p7','data/clean/p7','data/clean/forward/predictions']:
 for p in (r/d).rglob('*'):
  if p.is_file() and '__pycache__' not in p.parts:record['preserved_sha256'][str(p.relative_to(r))]=h(p)
record['preserved_sha256']['reports/latest.html']=h(r/'reports/latest.html')
p=r/'runs/n1/experiment_record.json'
assert not p.exists();p.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
print(record['recorded_at_utc'],len(record['preserved_sha256']))
