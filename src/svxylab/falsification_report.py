"""D5中文交付：只报告唯一alpha100反证，保留全部较差区间。"""
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from time import perf_counter
import os
import re
import subprocess
import sys
import traceback
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from svxylab.falsification import compute, NAME
from svxylab.mechanism import read_csv
from svxylab.features_report import write_json
from svxylab.release import digest

LABELS={'development':'开发段2020—2023','revealed':'已揭示段2024—2026','original_M2':'原M2主账户',
        'D5':'D5 α100主账户','M2_risk':'M2/D5仅风险','M0':'原M0主账户','fixed50':'固定50%','fixed100':'固定100%',
        'all':'全段','fixed_delete':'固定原变换删除','full_delete':'完整变换删除诊断'}

def pct(x):return '—' if pd.isna(x) else f'{100*x:+.2f}%'
def pp(x):return '—' if pd.isna(x) else f'{100*x:+.2f}个百分点'
def number(x):return '—' if pd.isna(x) else f'{x:,.2f}'

def table(df,columns,percent=(),scientific=()):
    out=df[list(columns)].copy()
    for c in out:
        if c in percent:out[c]=out[c].map(pct)
        elif c in scientific:out[c]=out[c].map(lambda x:'—' if pd.isna(x) else f'{x:.6g}')
        elif pd.api.types.is_float_dtype(out[c]):out[c]=out[c].map(number)
        else:out[c]=out[c].map(lambda x:LABELS.get(x,x))
    out.columns=[columns[k] for k in out]
    return '<div class="scroll">'+out.to_html(index=False,border=0,escape=True,na_rep='—')+'</div>'

def plots(output):
    fig,axes=plt.subplots(2,2,figsize=(14,8.5),layout='constrained')
    palette={'original_M2':'#b96947','D5':'#126e79','fixed50':'#92998d'}
    for j,scope in enumerate(['development','revealed']):
        for name,color in palette.items():
            ledger=read_csv(output/'ledgers'/f'{scope}_{name}.csv')
            x=pd.to_datetime(ledger.as_of_session)
            axes[0,j].plot(x,ledger.equity_end/100000,color=color,label=name,lw=1.5)
            axes[1,j].plot(x,ledger.drawdown*100,color=color,lw=1.2)
        axes[0,j].set_title(scope+' | same initial capital');axes[0,j].set_ylabel('Equity / initial capital')
        axes[1,j].set_ylabel('Drawdown (%)');axes[0,j].legend(frameon=False)
        for ax in axes[:,j]:
            ax.grid(alpha=.17);ax.spines[['top','right']].set_visible(False)
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    fig.savefig(output/'D5_equity.png',dpi=160);plt.close(fig)


def render_report(root,output,validation,tests):
    pred=read_csv(output/'prediction_metrics.csv'); account=read_csv(output/'account_metrics.csv')
    pair=read_csv(output/'paired_predictions.csv'); sensitivity=read_csv(output/'cluster_sensitivity.csv')
    ci=read_csv(output/'paired_block_intervals.csv'); design=read_csv(output/'design_stability.csv')
    chain=read_csv(output/'paired_failure_chain.csv')
    r=pred.loc[pred.scope.eq('revealed') & pred.period_type.eq('all')].iloc[0]
    allacct=account.loc[account.period_type.eq('all')]
    old=allacct.loc[allacct.scope.eq('revealed')&allacct.account.eq('original_M2')].iloc[0]
    new=allacct.loc[allacct.scope.eq('revealed')&allacct.account.eq('D5')].iloc[0]
    y24=pred.loc[pred.period_type.eq('year')&pred.period.eq('2024')].iloc[0]
    changes=sensitivity.loc[sensitivity.original_alpha.eq(1)].copy()
    changes['max_sensitivity_change']=changes.d5_max_abs_delta-changes.original_max_abs_delta
    pm=pred[['scope','period_type','period','MSE_delta']]
    am=account.loc[account.account.isin(['original_M2','D5'])].pivot(index=['scope','period_type','period'],columns='account',values='net_return').reset_index()
    contrasts=pm.merge(am,on=['scope','period_type','period']);contrasts['net_return_delta']=contrasts.D5-contrasts.original_M2
    contrasts['classification']=np.select([contrasts.MSE_delta.lt(-1e-16)&contrasts.net_return_delta.gt(1e-12),
        contrasts.MSE_delta.lt(-1e-16)&contrasts.net_return_delta.le(1e-12),
        contrasts.MSE_delta.abs().le(1e-16)&contrasts.net_return_delta.abs().le(1e-12)],
        ['误差/收益同向改善','误差改善、收益未改善','预测及区间回报不变'],default='未同时改善／连续状态差异')
    contrasts.to_csv(output/'period_contrasts.csv',index=False)
    largest=changes.sort_values('original_max_abs_delta',ascending=False).iloc[0]
    largestfull=changes.loc[changes.fit_id.eq(largest.fit_id)&changes.event_id.eq(largest.event_id)&changes.variant.eq('full_delete')].iloc[0]
    q1=contrasts.loc[contrasts.period.eq('2024Q1')].iloc[0]
    fixed50=allacct.loc[allacct.scope.eq('revealed')&allacct.account.eq('fixed50')].iloc[0]
    finding={'attribute':'POST_REVEAL_LIMITED_FALSIFICATION','candidate':NAME,
        'revealed_MSE_relative_change':r.MSE_relative_change,'2024_MSE_relative_change':y24.MSE_relative_change,
        'revealed_original_net_return':old.net_return,'revealed_D5_net_return':new.net_return,
        'revealed_net_return_delta':new.net_return-old.net_return,
        'changed_alpha_deletion_comparisons':len(changes),
        'lower_max_sensitivity_count':int(changes.max_sensitivity_change.lt(-1e-12).sum()),
        'higher_max_sensitivity_count':int(changes.max_sensitivity_change.gt(1e-12).sum()),
        'revealed_gate_crossings':[int(r.original_crossings),int(r.d5_crossings)],
        'revealed_gate_disagreements':int(r.gate_disagreements),
        'control_prediction_days':int(pair.original_alpha.eq(100).sum()),'next_intervention_started':False,
        'largest_original_event_counterexample':largest.to_dict(),
        'largest_event_full_delete_counterexample':largestfull.to_dict(),
        '2024Q1_counterexample':q1.to_dict(),
        'interpretation':'Partial mean-error and typical sensitivity improvement; strongest event sensitivity, gate churn and extreme portfolio tail not repaired. No trading rescue or independent edge.'}
    write_json(output/'findings.json',finding)
    def link(path,label):return '<a href="'+escape(os.path.relpath(path,root/'reports'))+'">'+escape(label)+'</a>'
    sections=['<header><div class="eyebrow">SVXYLab · D5 · RESEARCH ONLY</div><h1>固定收益头收缩，能修复哪一段？</h1><p class="sub">M2_MU_ALPHA100_D5 · 单一机制的历史有限反证 · 截至2026-09-04</p></header>']
    sections.append('<nav><a href="#verdict">判断</a><a href="#forecast">预测与稳定性</a><a href="#accounts">连续账户</a><a href="#counter">未改善区间</a><a href="#chain">逐日链条</a><a href="#audit">验证与口径</a></nav>')
    sections.append('<section id="verdict"><h2>确定事实、支持范围与保留反例</h2><div class="cards">'
        '<article><h3>唯一改动已隔离</h3><p>73个月全部核对原成熟训练成员并重拟收益头。仅2024的12个月从α=1改成100；其余'+str(finding['control_prediction_days'])+'个预测日精确不变。Q90/P10、原变换、输入、时钟和交易规则保留。</p></article>'
        '<article><h3>均值误差</h3><p>2024 MSE变化 '+pct(y24.MSE_relative_change)+'；整个已揭示段变化 '+pct(r.MSE_relative_change)+'。MSE是五日收益预测平方误差，不能当成美元亏损。</p></article>'
        '<article><h3>事件敏感性</h3><p>在2024的'+str(len(changes))+'组“月模型×事件簇×删除方法”配对中，'+str(finding['lower_max_sensitivity_count'])+'组最大预测影响下降，'+str(finding['higher_max_sensitivity_count'])+'组上升。相关诊断共享历史事件，不能当成这么多次独立危机。</p></article>'
        '<article><h3>经济结果</h3><p>已揭示段净收益 '+pct(old.net_return)+' → '+pct(new.net_return)+'，变化 '+pp(new.net_return-old.net_return)+'；最大回撤 '+pct(old.max_drawdown)+' → '+pct(new.max_drawdown)+'。仍须查看后续不变预测年份与最差路径。</p></article></div>'
        '<p class="notice">这是一项在2024—2026已经揭示后选择并运行的反证。变化支持到何种机制，由误差、敏感性和经济结果共同判断；不构成新独立样本优势，也不量化全部原损害的唯一因果份额。收益改善不能单独证明均值估计修复。完成D5即停，没有第二个候选或自动晋升。</p>'
        '<h3>支持到这里：部分误差及典型敏感性可由收缩改善</h3><p>2024全年误差下降、多数事件删除影响减小，支持α变化参与了原失误机制。但收益提升不能全部归为估计稳定性的修复，门槛路径仍是非线性的。</p>'
        '<h3>被反例削弱：恢复α=100足以消除事件依赖或交易失效</h3><p>原影响最大的2024年12月模型删除2020年压力簇，固定变换最大|Δμ|从 '+pct(largest.original_max_abs_delta)+' 增至 '+pct(largest.d5_max_abs_delta)+'；完整变换从 '+pct(largestfull.original_max_abs_delta)+' 增至 '+pct(largestfull.d5_max_abs_delta)+'。最强事件依赖没有被修好。门槛切换 '+str(int(r.original_crossings))+' → '+str(int(r.d5_crossings))+' 次，费用 $'+number(old.fees)+' → $'+number(new.fees)+'；最差五日 '+pct(old.worst_five)+' → '+pct(new.worst_five)+'，尾部损失几乎不变。</p>'
        '<h3>仍无法确认：稳定经济优势与其余错误来源</h3><p>2024Q1误差上升、净收益减少 '+pp(-q1.net_return_delta)+'；2025的原α100预测保持不变，全年仍亏2.85%。D5全段 '+pct(new.net_return)+' 仍低于固定50%的 '+pct(fixed50.net_return)+'，回撤也更深。已揭示全段MSE差的63日块区间跨0，净收益差的21/63日块区间均跨0，不能宣称稳定优势。</p>'
        '<p>本次足以保留“收益头收缩影响部分历史错误”的机制研究线索，尚不足以支持新主动机会策略。仍不能区分其余错误来自信息本身、模型偏差、有限样本还是门槛；原D1—D4的五日持有反例、风险信息不稳定及Q/P冲突继续保留。</p></section>')
    sections.append('<section id="forecast"><h2>同日期预测与事件影响</h2><p>原R5条件均值目标、SSE + α‖β‖²和不惩罚截距均保持。变换仍是每个原月份当时训练得到的19输入、63设计；没有重新选节点或网格。原已封存跨界标签和当前未成熟尾部继续不评分。</p>'+table(pred.loc[pred.period_type.isin(['all','year'])],
        {'scope':'区段','period':'时期','prediction_days':'预测日','score_days':'评分日','original_MSE':'原MSE','d5_MSE':'D5 MSE','MSE_relative_change':'MSE相对变化','original_crossings':'原门槛切换','d5_crossings':'D5切换','gate_disagreements':'两者门槛不同日'},
        percent=['MSE_relative_change'],scientific=['original_MSE','d5_MSE']))
    d24=design.loc[design.original_alpha.eq(1)]
    sections.append('<p>2024有效自由度（不含截距）从 '+number(d24.original_effective_df.min())+'–'+number(d24.original_effective_df.max())+' 降至 '+number(d24.d5_effective_df.min())+'–'+number(d24.d5_effective_df.max())+'。这是实际设计奇异值下的收缩变化，不能只凭alpha绝对值推论所有样本都足够稳定。</p>')
    tops=changes.sort_values('original_max_abs_delta',ascending=False).head(12)
    sections.append('<h3>按原影响排序的12项（完整结果可下载）</h3>'+table(tops,{'fit_id':'月模型','event_id':'已成熟训练事件','variant':'删除方式','removed_rows':'删除行数','original_max_abs_delta':'原最大|Δμ|','d5_max_abs_delta':'D5最大|Δμ|','original_gate_changes':'原门槛改变数','d5_gate_changes':'D5门槛改变数'},percent=['original_max_abs_delta','d5_max_abs_delta'])+
        '<p>固定变换删除只移除训练行；完整变换删除诊断采用D3已保存的对应删除变换。这些操作只测敏感性，正式D5预测仍使用全部原训练行和原变换，所有后来评价危机均保留。删除影响是在同一收益头自己的基准预测上计算，不能把预测水平下降误称敏感性下降。</p>'+link(output/'cluster_sensitivity.csv','全部1,260个事件簇的两种配对删除')+' · '+link(output/'design_stability.csv','每月自由度和条件数')+' · '+link(output/'paired_predictions.csv','全部同日期原值与D5预测')+'</section>')
    sections.append('<section id="accounts"><h2>同起点、连续持仓的净结果</h2><p>每段各账户以10万美元现金、相同实际收盘起跑；单边费用5bp，B=5%，μ&gt;2c才通过门槛，容量min(1,B/Q90)，现金收益0。先结算旧实际份额当日盈亏，再按前一信息日预测成交。没有期末强制平仓或年初重新注资。M2与D5仅风险策略完全相同，合用一份账本。</p>'+table(allacct,
        {'scope':'区段','account':'账户','net_return':'净收益','max_drawdown':'最大回撤','mean_exposure':'平均旧仓','fees':'费用美元','turnover':'总换手','worst_day':'最差日','worst_five':'最差连续五日'},percent=['net_return','max_drawdown','mean_exposure','worst_day','worst_five']))
    plots(output)
    sections.append('<img class="figure" src="'+os.path.relpath(output/'D5_equity.png',root/'reports')+'" alt="两个区段同起点净值与回撤"><p class="caption">开发段两条M2曲线重合。每段图内账户同资本；跨年延续实际权益。原收益序列已含基金费用，不再重复扣除管理费。</p>')
    selected=ci.loc[ci.period_type.eq('all') & ci.metric.isin(['MSE_delta','net_return_delta','max_drawdown_delta'])]
    sections.append('<h3>配对日历块区间</h3>'+table(selected,{'scope':'区段','block_sessions':'块长','metric':'D5减原M2','point':'点估计','lower95':'95%下界','upper95':'95%上界','valid_draws':'有效抽样'},scientific=['point','lower95','upper95'])+
        '<p>MSE差为负表示误差下降；净收益差为正表示改善；回撤以负值记录，回撤差为正表示减轻。21/63日各1000次配对循环块，不压缩缺失标签。经济区间条件于已实现净回报/交易成本序列，没有在抽样路径上重做可执行交易；不是实盘置信保证。多组揭示后比较没有提供新的独立验证。</p></section>')
    yearly=account.loc[account.period_type.eq('year')&account.account.isin(['original_M2','D5'])]
    sections.append('<section id="counter"><h2>所有年份及季度，保留未改善区间</h2>'+table(yearly,
        {'scope':'区段','period':'年份','account':'账户','net_return':'区间净收益','max_drawdown':'区间局部回撤','inherited_drawdown':'承接全段回撤','mean_exposure':'平均旧仓','fees':'费用美元','worst_day':'最差日','worst_five':'最差五日'},
        percent=['net_return','max_drawdown','inherited_drawdown','mean_exposure','worst_day','worst_five'])+
        '<p>区间局部回撤以该区间期初实际权益为锚；承接回撤继续使用全段此前高点。最差五日只在区间内滚动五个净回报。2025—2026预测不变，但2024期末权益/股数不同会传入以后，绝不能把金额差解释为这些年份获得了新的预测信息。</p>'+
        '<details><summary>展开全部季度的误差与净收益配对（含零变化）</summary>'+table(contrasts.loc[contrasts.period_type.eq('quarter')],{'scope':'区段','period':'季度','MSE_delta':'MSE差','original_M2':'原净收益','D5':'D5净收益','net_return_delta':'收益差','classification':'对照判断'},percent=['original_M2','D5','net_return_delta'],scientific=['MSE_delta'])+'</details>')
    bad=pair.loc[pair.score_observed].copy();bad['MSE_delta']=bad.d5_squared_error-bad.original_squared_error
    sections.append('<details><summary>展开误差退步最多的10个日期</summary>'+table(bad.sort_values('MSE_delta',ascending=False).head(10),{'decision_session':'决策日','mu5':'原μ5','d5_mu5':'D5 μ5','R5':'后来R5','MSE_delta':'平方误差差','original_alpha':'原α'},percent=['mu5','d5_mu5','R5'],scientific=['MSE_delta'])+'</details>'+link(output/'account_metrics.csv','全部账户/年/季度指标及最差路径日期')+' · '+link(output/'paired_block_intervals.csv','全部区间，包括样本短于块长的缺失记录')+'</section>')
    sections.append('<section id="chain"><h2>逐日输入 → 预测 → 决策 → 持仓 → 损益</h2><p>完整1503个同日期有服务预测均可查看，含2024与2025原失败日期。旧仓当日盈亏对应持仓账本；后来R5对应执行后的五日，两者不混为同一笔结果。没有服务的175个原请求日另保存在完整服务CSV。</p><div class="controls"><button id="prev">前一日</button><select id="day" aria-label="决策日期"></select><button id="next">后一日</button></div><div id="flow" class="cards"></div><details><summary>当日19项原输入</summary><div id="features"></div></details></section>')
    sections.append('<section id="audit"><h2>工程、真实覆盖与停止点</h2><p>普通测试 '+str(tests['tests'])+' 项通过。'+str(validation['account_count'])+'个账户、'+str(validation['account_rows'])+'行独立现金/股数核账；最大金额差 '+f"{validation['account_max_error']:.3g}"+' 美元。73份月模型训练身份、原预测重建、全部2520次删除的独立SVD解已核对。'+str(validation['protected_files'])+'份旧数据/产物/报告/运行记录摘要保持一致。</p>'
        '<p>真实行情1,930日，2019-01-02至2026-09-04，核心输入完整1,909日。预测831/672日、评分826/667日；不下载新行情。新计算是揭示后研究，不声称原始供应商值满足完整历史PIT。Q90不是最坏损失，P10未用于交易；原Q/P冲突和真实新交易日日报下载验证继续保留。</p>'
        '<p>HAC标准误只条件于固定设计和alpha，忽略估计偏差、选择和变换不确定性。只保留既定单一干预；没有足够依据自动晋升为主动机会策略。D5工程完成可被验收，经济研究可以是负结果。请验收D5，随后才保存本阶段本地Git检查点。</p>'
        '<p>运行 <code>.venv/bin/python -m svxylab falsification --open</code> 或双击“M2有限反证.command”。每次新建目录。HTML按本地文件交付，内置浏览器URL安全策略禁止页面预览，未完成整页截图/交互实测；图表、链接、HTML/JS另行核验，不能把打开命令成功等同视觉验收。</p>'
        '<p>'+link(root/'runs/m2_d5/experiment_record.json','计算前问题与全部口径')+' · '+link(root/'runs/m2_mechanism/acceptance.json','D1—D4验收')+' · '+link(root/'runs/m2_mechanism/D5_proposal.md','原批准方案')+' · '+link(output/'calculation_validation.json','实际计算验证')+' · '+link(output,'完整输出目录')+' · '+link(root/'runs/m2_d5/runtime_compatibility.json','三项入口/规格兼容变更')+'</p></section>')
    payload=chain.to_json(orient='records',force_ascii=False).replace('</','<\\/')
    (output/'daily_chain.js').write_text('const D5_RECORDS='+payload+';\n')
    js='''const records=D5_RECORDS,day=document.querySelector('#day');
for(const r of records){const o=document.createElement('option');o.value=r.decision_session;o.textContent=r.decision_session;day.append(o)}
function pc(v){return v==null?'未评分/未成熟':(100*v).toFixed(2)+'%'}function money(v){return v==null?'—':v.toLocaleString('zh-CN',{maximumFractionDigits:2})}
function draw(){const r=records.find(r=>r.decision_session===day.value);const card=(h,s)=>'<article><h3>'+h+'</h3>'+s+'</article>';
document.querySelector('#flow').innerHTML=card('1 · 原信息','<p>信息日 '+r.as_of_session+'<br>拟合 '+r.fit_id+'<br>原α '+r.original_alpha+' → 100</p>')+card('2 · 两份预测','<p>原μ '+pc(r.mu5)+'<br>D5 μ '+pc(r.d5_mu5)+'<br>同Q90 '+pc(r.q90)+'<br>同P10 '+pc(r.p10)+'</p>')+card('3 · 门槛和目标','<p>门槛0.10%<br>原目标 '+pc(r.original_target)+'<br>D5目标 '+pc(r.d5_target)+'<br>容量 '+pc(r.risk_capacity)+'</p>')+card('4 · 原实际账户','<p>旧仓 '+pc(r.original_M2_old_weight)+'<br>持有损益 $'+money(r.original_M2_holding_pnl)+'<br>费用 $'+money(r.original_M2_cost)+'<br>权益 $'+money(r.original_M2_equity_end)+'</p>')+card('5 · D5实际账户','<p>旧仓 '+pc(r.D5_old_weight)+'<br>持有损益 $'+money(r.D5_holding_pnl)+'<br>费用 $'+money(r.D5_cost)+'<br>权益 $'+money(r.D5_equity_end)+'<br>后来五日R5 '+pc(r.R5)+'</p>');
const keys=['A01','A02','A03','A04','A05','B01','B02','B03','C01','C02','C03','D01','D02','D03','E01','F01','F02','F03','F04'];document.querySelector('#features').textContent=keys.map(k=>k+': '+r[k].toPrecision(6)).join(' ｜ ')}
day.value='2024-07-29';day.onchange=draw;document.querySelector('#prev').onclick=()=>{day.selectedIndex=Math.max(0,day.selectedIndex-1);draw()};document.querySelector('#next').onclick=()=>{day.selectedIndex=Math.min(day.options.length-1,day.selectedIndex+1);draw()};draw();'''
    css='''*{box-sizing:border-box}body{margin:0;background:#f4f2ed;color:#25363c;font:16px/1.75 -apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif}main{max-width:1320px;margin:auto;padding:34px 28px 70px}header{padding:20px 0;border-bottom:3px solid #315865}h1{font-size:40px;line-height:1.3}h2{font-size:25px}h3{font-size:18px}.eyebrow{font-size:12px;letter-spacing:2px}.sub{font-size:18px;color:#617178}section{background:white;padding:28px;margin:24px 0;border:1px solid #dfe4df;border-radius:8px}nav{display:flex;gap:22px;flex-wrap:wrap;padding:20px 0}a{color:#176476;text-underline-offset:3px}.cards{display:grid;grid-template-columns:repeat(2,1fr);gap:15px}.cards article{background:#eff5f4;padding:18px;border-top:3px solid #70989c}.notice{background:#fff2e8;border-left:3px solid #b4774e;padding:20px}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px;white-space:nowrap}th,td{text-align:left;padding:10px 12px;border-bottom:1px solid #dee5e3}th{background:#edf3f3}tr:nth-child(even){background:#f9fbfb}.figure{width:100%;height:auto}.caption{font-size:13px;color:#697b80}details{padding:15px;background:#f7f9f8;margin:15px 0}summary{cursor:pointer;font-weight:600}.controls{display:flex;gap:12px}button,select{font:inherit;padding:7px 12px;background:#fff;border:1px solid #97acb1;border-radius:5px}code{overflow-wrap:anywhere}#features{overflow-wrap:anywhere}@media(max-width:760px){main{padding:18px}section{padding:18px}h1{font-size:30px}.cards{grid-template-columns:1fr}}'''
    html='<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>D5有限反证 · SVXYLab</title><style>'+css+'</style></head><body><main>'+''.join(sections)+'</main><script src="'+os.path.relpath(output/'daily_chain.js',root/'reports')+'"></script><script>'+js+'</script></body></html>'
    report=root/'reports/falsification.html';report.write_text(html)
    def relative(m):
        attr,ref=m.group(1),m.group(2)
        if ref.startswith(('http','#')):return m.group(0)
        return attr+'="'+os.path.relpath((root/'reports'/ref).resolve(),output)+'"'
    (output/'report.html').write_text(re.sub(r'(href|src)="([^"]+)"',relative,html))
    return report


def build_falsification_report(root,*,open_report=False):
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    run=root/'runs/m2_d5'/stamp;run.mkdir(parents=True)
    output=root/'data/clean/m2_d5'/stamp
    started={'stage':'M2_D5','command':[sys.executable,'-m','svxylab','falsification']+(['--open'] if open_report else []),
        'output_dir':str(output.relative_to(root)),'run_dir':str(run.relative_to(root)),
        'started_at_utc':datetime.now(timezone.utc).isoformat(),
        'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        'source_sha256':{str(p.relative_to(root)):digest(p) for p in sorted((root/'src').rglob('*.py'))},
        'experiment_sha256':digest(root/'runs/m2_d5/experiment_record.json')}
    write_json(run/'started.json',started);write_json(root/'runs/m2_d5/latest_run.json',started)
    t=perf_counter()
    try:
        validation=compute(root,output)
        check=subprocess.run([sys.executable,'-m','pytest','-q',f'--junitxml={run/"pytest.xml"}'],cwd=root,text=True,capture_output=True)
        (run/'pytest.log').write_text(check.stdout+check.stderr)
        if check.returncode:raise ValueError('普通测试失败：'+str(run/'pytest.log'))
        suite=ET.parse(run/'pytest.xml').getroot().find('testsuite')
        tests={k:int(suite.attrib[k]) for k in ['tests','failures','errors','skipped']}
        report=render_report(root,output,validation,tests)
        result={**started,'elapsed_seconds':perf_counter()-t,'validation':validation,'ordinary_tests':tests,
                'report':str(report.relative_to(root)),'report_sha256':digest(report),
                'output_sha256':{str(p.relative_to(output)):digest(p) for p in sorted(output.rglob('*')) if p.is_file()}}
        if open_report:result['open_returncode']=subprocess.run(['open',str(report)]).returncode
        write_json(run/'falsification_run.json',result);write_json(root/'runs/m2_d5/latest_run.json',result)
        print(f'D5有限反证完成：{report}\n普通测试{tests["tests"]}项；{validation["account_count"]}账户/{validation["account_rows"]}行。等待D5验收。',flush=True)
        return 0
    except Exception:
        (run/'failure.txt').write_text(traceback.format_exc());raise
