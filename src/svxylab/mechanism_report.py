"""M2机制诊断的中文本地报告；图表/结论读取本轮CSV，D5仅提出待决定方案。"""
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from time import perf_counter
import json
import os
import re
import subprocess
import sys
import traceback

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from svxylab.mechanism import compute, read_csv, save_csv, distribution
from svxylab.features_report import write_json, verify_hashes
from svxylab.release import digest, read_json

LABELS={'development':'开发段 2020–2023','revealed':'已揭示段 2024–2026',
        'mu0_q0':'μ0 + Q0','mu2_q0':'μ2 + Q0','mu0_q2':'μ0 + Q2','mu2_q2':'μ2 + Q2',
        'M0_risk':'M0 仅风险','M2_risk':'M2 仅风险',
        'pre_execution_log':'信息完成→执行（不可计入收益）','after_day1_log':'执行后第1日',
        'after_rest4_log':'执行后其余4日','after_full5_log':'执行后完整5日','past5_log':'信息前已发生5日',
        'high_up5':'训练分位高位／5日仍上升','high_down5':'训练分位高位／5日已下降',
        'lower_up5':'非高位／5日上升','lower_down5':'非高位／5日下降',
        'V_SELF21':'仅自身21日波动','V_JOINT63':'联合63列（风险目标）',
        'raw_outside':'至少一原始输入越界','raw_inside_joint_sparse':'原始逐列在界内／联合状态稀少',
        'raw_inside_joint_dense':'原始逐列在界内／联合近邻较密'}


def pct(x):return '—' if pd.isna(x) else f'{100*x:+.2f}%'
def pp(x):return '—' if pd.isna(x) else f'{100*x:+.2f} 个百分点'
def num(x):return '—' if pd.isna(x) else f'{x:,.2f}'


def table(df, columns, percent=(), scientific=()):
    out=df[list(columns)].copy()
    for col in out:
        if col in percent:out[col]=out[col].map(pct)
        elif col in scientific:out[col]=out[col].map(lambda v:'—' if pd.isna(v) else f'{v:.6g}')
        elif pd.api.types.is_float_dtype(out[col]):out[col]=out[col].map(num)
        else:out[col]=out[col].map(lambda v:LABELS.get(str(v),str(v)))
    out=out.rename(columns=columns)
    return '<div class="table-scroll">'+out.to_html(index=False,border=0,escape=True,na_rep='—')+'</div>'


def link(root,path,label):
    href=os.path.relpath(path,root/'reports')
    return f'<a href="{escape(href,quote=True)}">{escape(label)}</a>'


def plots(output):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    d=read_csv(output/'D1_accounts.csv')
    for i,(scope,label) in enumerate([('development','2020-2023'),('revealed','2024-2026')]):
        g=d.loc[d.scope.eq(scope)].iloc[:4]
        ax[0,0].bar(np.arange(4)+(i-.5)*.34,g.total_return*100,width=.34,label=label,color=['#879cab','#9d482f'][i])
    ax[0,0].set_xticks(np.arange(4),['mu0+q0','mu2+q0','mu0+q2','mu2+q2']);ax[0,0].axhline(0,color='#999',lw=.7)
    ax[0,0].set_ylabel('Net return (%)');ax[0,0].set_title('D1 / Frozen component replacement');ax[0,0].legend()
    d=read_csv(output/'D2_holding_accounts.csv');g=d.loc[d.scope.eq('revealed')&d.account.eq('M2_main')]
    for cadence,label,color in [('hold5','Hold 5 sessions','#9d482f'),('daily','Daily, matched start','#486b82')]:
        q=g.loc[g.cadence.eq(cadence)];ax[0,1].plot(q.offset,q.total_return*100,'o-',label=label,color=color)
    ax[0,1].set_xticks(range(5));ax[0,1].set_xlabel('Independent starting offset');ax[0,1].set_ylabel('Net return (%)')
    ax[0,1].set_title('D2 / Every offset retained (2024-2026)');ax[0,1].legend()
    d=read_csv(output/'D3_design.csv');date=pd.to_datetime(d.fit_id.str[:10])
    ax[1,0].plot(date,d.ridge_df_including_intercept,color='#486b82');ax[1,0].set_ylabel('Ridge effective degrees of freedom')
    ax[1,0].axvspan(pd.Timestamp('2024-01-01'),pd.Timestamp('2025-01-01'),color='#c77c56',alpha=.15)
    ax[1,0].set_title('D3 / 2024 alpha=1; other months alpha=100')
    d=read_csv(output/'D4_metrics.csv');d=d.loc[d.period_type.eq('year')]
    a=d.loc[d.model.eq('V_JOINT63')].set_index('period');b=d.loc[d.model.eq('V_SELF21')].set_index('period')
    delta=a.qlike-b.qlike
    ax[1,1].bar(delta.index.astype(str),delta,color=np.where(delta<0,'#486b82','#9d482f'))
    ax[1,1].axhline(0,color='#999',lw=.7);ax[1,1].set_ylabel('QLIKE: joint minus self (lower is better)')
    ax[1,1].set_title('D4 / Risk intensity is a separate target')
    fig.savefig(output/'mechanisms.png',dpi=170);plt.close(fig)


def findings(output):
    chain=read_csv(output/'D1_D2_complete_decision_diagnostics.csv')
    accounts=read_csv(output/'D1_accounts.csv');factor=read_csv(output/'D1_factorial.csv')
    switch=read_csv(output/'D1_switch_summary.csv');sw=switch.loc[switch.scope.eq('revealed')&switch.model.eq('M2')&switch.period_type.eq('all')].iloc[0]
    r=factor.loc[factor.scope.eq('revealed')].iloc[0]
    d3=read_csv(output/'D3_design.csv');infl=read_csv(output/'D3_cluster_influence.csv')
    hold=read_csv(output/'D2_holding_accounts.csv');h=hold.loc[hold.scope.eq('revealed')&hold.account.eq('M2_main')&hold.cadence.eq('hold5')]
    dist=[];tail=[]
    for scope,g in chain.loc[chain.model.eq('M2')].groupby('scope'):
        for name,col in [('实际R5','R5'),('未来收益预测残差','mu_prediction_residual')]:dist.append({'scope':scope,'variable':name,**distribution(g[col])})
        mask=g.Y10.eq(1)|g.R5.abs().ge(.1);err=(g.R5-g.mu5)**2
        tail.append({'scope':scope,'scored_rows':int(g.score_observed.sum()),'tail_rows':int(mask.sum()),
            'tail_sse_share':err[mask].sum()/err.sum(),'tail_definition':'Y10=1 OR abs(R5)>=10%; overlapping rows, not independent events'})
    save_csv(output,'D3_evaluation_distributions',dist);save_csv(output,'D3_evaluation_tail_concentration',tail)
    result={
      'engineering':'D1-D4 computed; D5 not executed; awaiting user acceptance',
      'facts':[
        '原标签时钟正确；Ridge/Logistic/Pinball分别估计均值、事件概率与分位，P10不参与交易。',
        f'已揭示段四组合：收益头部件替换在Q0下改变净收益{pp(r.mu_effect_with_q0)}，风险头在μ0下为{pp(r.q_effect_with_mu0)}；交互为{pp(r.interaction)}。',
        '2024原年度选择采用μ alpha=1、P10 C=10、Q90 alpha=.001；2025/2026回到100/.1/.01。这是原结果，不是本轮改参。'],
      'supported':[
        f'收益头进入硬门槛后有可复算损害。已揭示段切换{int(sw.gate_crossings)}次，切换前门槛距离中位数{sw.median_abs_margin_before_crossing*10000:.1f}bp，目标跳变中位数{sw.median_target_jump_on_crossing*100:.1f}%。',
        '原训练事件对后来均值预测有真实影响；固定变换精确删除与完整变换重拟合均显示这种敏感性。2024弱收缩的有效自由度及惩罚后条件数明显上升。',
        '尾部错误集中，联合状态稀少时平均误差更大；这些机制存在，但都不能单独解释全部失败。'],
      'rejected_specific_explanations':[
        f'“原五日预测只需每五日交易即可获救”：本段五偏移净收益范围{pct(h.total_return.min())}至{pct(h.total_return.max())}，全部低于同偏移每日账户。',
        '“只有原始输入超出训练min/max才会失败”：2024-07-29属于逐列在界内、联合近邻较密，却是最大均值预测误差日。',
        f'“失败只是10bp门槛附近的细小抖动”：只有{int(sw.crossings_from_within_10bp)}/{int(sw.gate_crossings)}次切换从这个邻域跨出；门槛放大存在，纯窄邻域解释不足。'],
      'unresolved':[
        '时延耗尽信息：执行前排序仅弱正相关，门槛两侧前置收益差区间跨0；没有证据把全部失败归于一个交易日延迟，也不能给出全局半衰期。',
        '共线性、有限事件、联合外推、年度选参及训练尺度共同作用；本轮不是随机因果实验，无法分配唯一根因百分比。',
        '联合信息在方差目标上有点估计改善，但总QLIKE差的21/63日区间均跨0，2024还退步；未确认稳健风险信息优势，更不能晋升收益择时。'],
      'next_direction':'优先对收益头的正则机制做一次有限反证；目前不足以支持新主动机会策略或资金使用。',
      'D5_executed':False,
      'review_baseline':'983c0da5659425e02812c73731f15e38072e89e6',
    }
    write_json(output/'findings.json',result)
    return result


def proposal_text(output):
    return '''# D5 单一机制假说与最小干预（待用户决定；未执行）

本轮D1—D4均为2024—2026揭示后的研究。以下不构成新策略验收或资金授权。

| 假说 | 本轮证据 | 当前判断 |
|---|---|---|
| 收益估计不稳，经门槛放大 | 收益头部件替换损害30.43–37.00个百分点；训练事件删除可大幅改变后来μ；2024 α=1、有效自由度约58.7 | 支持存在；不能断言全部损害来自α |
| 仅持有期限不匹配 | 五个独立五日偏移全部比同起点每日M2更差 | 不支持简单五日持有救援 |
| 信息被原执行时延耗尽 | 执行前排序弱正、门槛前置差区间跨0；主要负向区分在执行后其余四日 | 目前无法确立时延主因 |
| 原始输入/联合状态越界 | 稀少状态误差更高；最大误差日又在较密支持区 | 局部支持，不能作为唯一根因 |
| 只有风险信息，没有收益信息 | 新方差比较点估计较好但区间跨0、2024反例保留 | 值得继续观察，尚未确认稳定增量 |

**主假说：** 原年度选参在2024把收益头的α从100降到1，使63列均值估计对有限历史事件更敏感；硬门槛把这种误差转成大幅持仓变化。这是可反证的机制，尚未证明恢复α就能修复收益。

**唯一干预：** 另名 `M2_MU_ALPHA100_D5`，全部原月度成熟训练集、原变换、原R5均值目标不变，仅把μ5头的Ridge α固定为100。Q90/P10原发布值、19输入、63设计、时钟、训练窗口、仓位公式、资本和成本均不变。无新网格，无第二个候选。2020—2023及2025—2026原本α=100的月份作为预测不变对照；连续账户仍承接前期持仓/净值差异。2024为主要预测变化区间。不是把M2直接替换为M0。

**验收方法：** 保存同日期μ变化及分位/概率不变核验；比较μ的MSE、事件删除影响和门槛切换，再展示同起点连续净账本、敞口、费用、最大回撤及最差日/五日；保留各年/季度的未改善区间。配对21/63日块区间只作揭示后的不确定性描述。只有收益改善而均值误差/稳定性不改善，不能归因于估计被修复；只有MSE改善而经济性不改善，也不能称交易救回。若α=100仍同样敏感，或原较差区间不改善，就削弱该机制假说。拒绝自动再试第二个干预。

**决定边界：** D5尚未运行。用户可决定执行上述唯一试验、修改任务或停止；D1—D4可以在没有新策略的情况下验收。'''


def render_report(root, output, validation, tests=None):
    f=findings(output);plots(output)
    (output/'D5_proposal.md').write_text(proposal_text(output))
    read=lambda n:read_csv(output/(n+'.csv'))
    chain=read('D1_D2_complete_decision_diagnostics');acc=read('D1_accounts');fac=read('D1_factorial')
    switch=read('D1_switch_summary');switch=switch.loc[switch.period_type.eq('all')]
    holds=read('D2_holding_accounts');holds=holds.loc[holds.account.eq('M2_main')&holds.cadence.eq('hold5')]
    corr=read('D2_path_correlations');corr=corr.loc[corr.model.eq('M2')&corr.period_type.eq('all')&corr['head'].eq('mu5')]
    pathint=read('D2_gate_path_intervals');pathint=pathint.loc[pathint.model.eq('M2')&pathint.metric.isin(['pre_execution_log','after_day1_log','after_rest4_log','after_full5_log'])]
    design=read('D3_design');selection=read('D3_parameter_selection');selection=selection.loc[selection.model.eq('M2')]
    infl=read('D3_cluster_influence').nlargest(8,'fixed_delete_max_mu_delta')
    var=read('D4_metrics');vint=read('D4_paired_intervals');vint=vint.loc[vint.state.eq('all')&vint.metric.eq('qlike')]
    states=read('D4_original_state_increment');states=states.loc[states.scope.eq('revealed')&states.block.eq(63)]
    family=[]
    for stage in ['p6','p7']:
        d=read_csv(output/f'reused/{stage}/paired_prediction_periods.csv')
        d=d.loc[d.period_type.eq('all')&d['left'].str.startswith('M2_DROP_')&d.right.eq('M2')]
        for name,g in d.groupby('left'):
            family.append({'stage':stage,'removed':name.replace('M2_DROP_',''),
                **{r.metric:r.relative_change for r in g.itertuples() if r.metric in ['mse','log_loss','pinball']}})
    d3summary=design.assign(year=design.fit_id.str[:4]).groupby(['scope','year','alpha'],as_index=False).agg(
        months=('fit_id','size'),rows_min=('training_rows','min'),rows_max=('training_rows','max'),
        df_min=('ridge_df_including_intercept','min'),df_max=('ridge_df_including_intercept','max'),
        condition_max=('ridge_penalized_condition','max'),L1_nonzero_min=('q90_L1_nonzero','min'),L1_nonzero_max=('q90_L1_nonzero','max'))
    testtext=f"{tests['tests']}项普通测试，失败{tests['failures']}、错误{tests['errors']}" if tests else '普通测试结果另见运行记录'
    sections=[]
    sections.append('<header><p class="eyebrow">SVXYLAB · D1—D4 · 2026-09-07</p><h1>M2 为什么失败？</h1><p class="subtitle">机制归因与有限反证准备｜已揭示历史研究｜等待验收</p><p>本轮从附件基准 <code>983c0da</code> 继续，初始工作区干净、无后续提交差异。只读复用原模型、预测、标签和账户；D5仅提出方案。</p></header>')
    sections.append('<nav><a href="#verdict">结论边界</a><a href="#chain">逐日失败链</a><a href="#d1">预测→持仓</a><a href="#d2">时效与期限</a><a href="#d3">估计与支持</a><a href="#d4">信息功能</a><a href="#d5">下一项反证</a><a href="#delivery">验收与文件</a></nav>')
    cards=[]
    for key,title in [('facts','确定事实'),('supported','被支持的机制'),('rejected_specific_explanations','被否定的具体解释'),('unresolved','当前无法区分')]:
        cards.append('<article><h3>'+title+'</h3><ul>'+''.join('<li>'+escape(t)+'</li>' for t in f[key])+'</ul></article>')
    sections.append('<section id="verdict"><h2>有多条失败环节，没有唯一根因比例</h2><div class="verdict-grid">'+''.join(cards)+'</div><p class="notice">研究方向：'+escape(f['next_direction'])+'</p></section>')
    sections.append('<section id="chain"><h2>一张可逐日复算的失败链</h2><p>用日期选择查看全部1,503个可执行决策日。当天损益先由旧持仓承担，新目标只从本次收盘后生效。未来五日标签用于后来检验，未用于当天选择。初始显示2024-08-05只是事后案例，不是状态定义。</p><div class="controls"><label>模型 <select id="model"><option>M2</option><option>M0</option></select></label><button id="prev" type="button">前一日</button><label>决策日 <select id="day"></select></label><button id="next" type="button">后一日</button></div><div id="flow" class="flow"></div><p id="decomposition"></p><details><summary>查看当天19项原始输入</summary><div id="features"></div></details><p>'+link(root,output/'D1_D2_complete_decision_diagnostics.csv','完整日期、特征、预测、时钟、持仓、损益与支持 CSV')+' · '+link(root,output/'D1_switches.csv','全部实际切换 CSV')+'</p></section>')
    sections.append('<section><img class="figure" src="'+os.path.relpath(output/'mechanisms.png',root/'reports')+'" alt="四项机制诊断图：部件替换、持有期限、有效自由度与方差目标"/><p class="caption">每个比较保留全部预定组合。2020—2023与2024—2026分别以10万美元从共同执行收盘建立账户；两段收益不可直接相加。</p></section>')
    sections.append('<section id="d1"><h2>D1｜损害主要经过收益头与门槛，但风险头也有问题</h2><p><b>问题：</b>原预测的哪一部分转成了账户损害？<b>控制：</b>原预测、原资本、现金收益0、下一交易日收盘、B=5%、单边5bp。<b>只变：</b>μ和Q的四个固定组合。P10不进入交易。</p>'+table(acc,{'scope':'时期','account':'组件/账户','total_return':'净收益','max_drawdown':'最大回撤','mean_exposure':'平均旧敞口','cost_dollars':'费用美元'},percent=['total_return','max_drawdown','mean_exposure']))
    sections.append('<p>2024—2026把μ0换成μ2，在Q0和Q2下分别损害37.00和30.43个百分点；把Q0换成Q2，在μ0和μ2下分别损害7.21和0.64个百分点。交互项+6.58个百分点，故不能把两条单独损害直接相加后声称唯一归因。开发段部件替换贡献为正，是“收益头在所有时期都只会伤害”的反例；原M2主映射仍低于它的risk-only。</p>'+table(fac,{'scope':'时期','mu_effect_with_q0':'μ替换/Q0','mu_effect_with_q2':'μ替换/Q2','q_effect_with_mu0':'Q替换/μ0','q_effect_with_mu2':'Q替换/μ2','interaction':'部件交互','joint_difference':'M2−M0净收益'},percent=['mu_effect_with_q0','mu_effect_with_q2','q_effect_with_mu0','q_effect_with_mu2','interaction','joint_difference']))
    sections.append('<h3>目标变化与实际交易分开</h3><p>令g为均值过门槛与否、r为风险容量，精确分解为 <code>Δw = r旧Δg + g旧Δr + ΔgΔr</code>；再加“前一目标−实际旧仓位”和“实际旧仓位−执行前漂移仓位”，才得到实际再平衡缺口。启动现金单独列出。所有项是有方向的权重变化，不是可以唯一分配的因果损益比例。</p>'+table(switch,{'scope':'时期','model':'模型','gate_crossings':'门槛切换','median_abs_margin_before_crossing':'切换前距离中位数','median_target_jump_on_crossing':'目标跳变中位数','crossings_from_within_10bp':'从10bp邻域跨出','crossings_from_within_50bp':'从50bp邻域跨出','gate_crossings_near_monthly_fit':'月更±2日切换','distribution_conflicts':'Q/P冲突日'},percent=['median_abs_margin_before_crossing','median_target_jump_on_crossing']))
    sections.append('<p>已揭示段23/106次切换位于月更±2日，而该邻域占163/672日；并未显示切换都集中在月更。月更同一输入下的预测变化与新输入变化另存精确桥接。106次切换中只有18次从10bp邻域跨出，不能把所有失败归于极窄门槛抖动。持仓持续时间、尾部截断及门槛各固定区间均保留。</p><p>Q90/P10分别拟合会发生分布解释冲突：已揭示段44个决策日、其中43个已评分日。判据为Q90低于10%而P10高于10%，或反向严格不等式；等号不分类。正均值同时有高尾部概率本身不是矛盾。Q90是第90百分位，不是尾部最坏损失，也不保证账户最大亏损≤B。</p><p><b>成本依赖旧仓位：</b>2c是原五日静态筛选近似，不是本次实际成本。真实成本始终按执行前股数到目标股数的成交额计算。MSE衡量均值预测误差，不是金融亏损；同样MSE可以对应不同日期、敞口、成本和复利路径。</p><p>'+link(root,output/'D1_terminal_wealth_bridges.csv','部件替换的逐日期末财富恒等桥接')+' · '+link(root,output/'D1_monthly_update_bridge.csv','月更同输入桥接')+' · '+link(root,output/'D1_holding_spells.csv','持仓持续时间/未平仓截断')+' · '+link(root,output/'D1_margin_groups.csv','预定门槛分组')+'</p></section>')
    sections.append('<section id="d2"><h2>D2｜五日不是一日；时延不能被一句反弹故事解释</h2><p><b>问题：</b>信息主要对应已发生、前置段，还是执行后的特定期限？<b>控制：</b>同一原信息与五日预测。路径用总回报的对数拆为前置1日、执行后1日、其余4日，后两段之和严格等于原5日。全部按相同成熟日期比较，年度/季度完整保留。</p>'+table(corr,{'scope':'时期','segment':'路径段','matched_rows':'同日期样本','spearman':'μ排序/Spearman'},scientific=['spearman']))
    sections.append('<p>μ对已发生五日及未来完整五日都呈弱负排序，执行前和执行后第一日只呈很弱的正排序；没有强证据说明原μ主要准确解释了过去，更不能据此宣称把成交提前一天就能获救。已揭示段过门槛日相对未过门槛日的执行后完整五日平均对数收益低1.803个百分点，主要来自其余四日（低1.604个百分点）。这是一种事后条件关系；两侧样本、重叠五日路径与状态差异均需保留。</p>'+table(pathint,{'scope':'时期','metric':'路径段','block':'块长','estimate':'过门槛−未过','lower':'95%下端','upper':'95%上端'},percent=['estimate','lower','upper']))
    sections.append('<h3>五个不重叠持有偏移，全部交付</h3><p>每个偏移每五个真实交易日接收一次新目标，中间保留实际份额和现金。六种控制器各自独立，资本上限100%；不把五个账户相加成杠杆。每日对照从同偏移收盘现金重建；末端不足五日只按实际最后收盘计价，不虚构未来退出。</p>'+table(holds,{'scope':'时期','offset':'偏移','total_return':'五日持有净收益','total_return_minus_daily':'相对同起点每日','max_drawdown':'最大回撤','mean_exposure':'平均敞口','cost_dollars':'费用美元'},percent=['total_return','total_return_minus_daily','max_drawdown','mean_exposure']))
    sections.append('<p>2024—2026全部偏移更差，虽费用显著下降，仍未修复损害；开发段4/5偏移更差、1个略好。只能否定本次固定五日持有救援，不能证明所有动态控制设计都无效。原五日预测并不自动估计下一日收益。</p><p>压力分类只使用月模型成熟训练集的VIX第75百分位与已发生1/5日变化。另记录SVXY 21日平方回报窗口的“进入项−离开项”；这解释了压力已下降而慢窗口仍高/仍上升的具体来源。没有使用未来峰值定义状态。更早交易的数据可得性未核实，本轮没有按当日收盘或次日开盘回测；若以后改时钟，必须重建标签、成熟时间、训练选择和执行价格。</p><p>'+link(root,output/'D2_holding_accounts.csv','全部120个持有/每日同起点账户指标')+' · '+link(root,output/'D2_holding_periods.csv','全部账户年度敞口与费用')+' · '+link(root,output/'D2_nonoverlap_blocks.csv','独立五日区间明细')+' · '+link(root,output/'D2_D4_stress_states.csv','按当时状态及连续时期的诊断')+'</p></section>')
    sections.append('<section id="d3"><h2>D3｜有限事件、有效复杂度与训练支持共同作用</h2><p><b>问题：</b>大尾部是否通过杠杆、收缩与设计列真正影响估计？<b>控制：</b>全部73份原月模型、原成熟训练成员、原选参。<b>只变：</b>在各当时训练集内，对Y10=1或|R5|≥10%的行合并重叠标签事件；逐簇做目标值替换、固定变换删行、完整变换重拟合。后续评价价格和极端亏损全部保留。这是影响诊断，不能事前知道哪些事件应删除。</p><h3>实际分布与误差集中</h3>'+table(read('D3_evaluation_distributions'),{'scope':'时期','variable':'变量','rows':'成熟行','std':'标准差','skewness':'偏度','excess_kurtosis':'超额峰度','largest1pct_squared_share':'最大1%行平方占比','min':'最小','max':'最大'},percent=['std','largest1pct_squared_share','min','max'])+table(read('D3_evaluation_tail_concentration'),{'scope':'时期','scored_rows':'成熟行','tail_rows':'预定极端行','tail_sse_share':'这些行占μ平方误差'},percent=['tail_sse_share']))
    sections.append('<p>已揭示段34个极端行占μ平方误差51.68%，但五日重叠使这些行不是34个独立事件；约一半误差仍在其他行。有限样本显示厚尾与集中，不能推断左尾指数、正态性或无限方差。收益头保留原条件均值目标；Huber/MAE未运行，偏态下换损失可能改成另一个统计目标，不能把它当作无代价的均值修复。</p><h3>有误差影响，也有真实参数与预测影响</h3>'+table(infl,{'fit_id':'当月原模型','event_id':'当时训练事件簇','removed_rows':'删除行','sse_share':'原训练SSE占比','target_only_max_mu_delta':'只替目标最大μ变化','fixed_delete_max_mu_delta':'固定变换删行最大μ变化','full_delete_max_mu_delta':'全变换删行最大μ变化','full_minus_fixed_max_mu_delta':'全变换−固定最大差'},percent=['sse_share','target_only_max_mu_delta','fixed_delete_max_mu_delta','full_delete_max_mu_delta','full_minus_fixed_max_mu_delta']))
    sections.append('<p>每条训练行另存残差、SSE占比、Ridge杠杆、梯度范数、对后来月内预测的微小加权导数。1,260个“月模型×历史事件簇”都执行固定变换与全变换精确重拟合，并用独立SVD解交叉核对。目标替换只把该簇目标改成原训练均值；删除仅删除已标记行，不删除评价事件或未来价格。固定与全变换之差包含尺度、节点、乘积及重拟合的交互；样条/乘积预测贡献可加复算，但坐标改变后不宣称唯一因果分配。</p><h3>63列满秩，不等于估计稳定</h3>'+table(d3summary,{'scope':'时期','year':'年','alpha':'原μ alpha','months':'月数','rows_min':'训练行下限','rows_max':'上限','df_min':'有效自由度下限','df_max':'上限','condition_max':'惩罚后条件数上限','L1_nonzero_min':'Q非零列下限','L1_nonzero_max':'上限'},scientific=['condition_max']))
    sections.append('<p>全部月份原始19列与63列设计均为数值满秩，不能称线性代数奇异导致程序算错。标准化也没有消除共线性：2024年α=1时惩罚后条件数约2.15万–2.22万，而α=100月份约90–244。有效自由度按 <code>1+Σ s²/(s²+α)</code> 计算，包含不受惩罚截距；样本数增加也令同一SSE目标下α/n下降。真正要看各期谱和选参，不能只看α的绝对数字。</p><p>原Ridge数值解与保存预测重建误差约1e-13，说明这里讨论的是估计敏感性，不是优化器把方程算错。每月19/63列Pearson、Spearman矩阵、奇异值、逐列分布、L1精确非零及交互越界均已保存。逐列在min/max内并不保证联合支持；本轮以训练期排除±5日邻域后的第五近邻距离第95百分位作为固定诊断参考。</p>'+table(read('D3_support_error_groups'),{'scope':'时期','support_bucket':'支持类别','rows':'决策行','score_rows':'评分行','mu_mse_delta_M0':'μ MSE−M0','mu_rmse':'μ未来残差RMSE','interaction_outside_rows':'交互越界日','mean_mu_HAC_SE21':'均值条件SE21'},percent=['mu_rmse','mean_mu_HAC_SE21'],scientific=['mu_mse_delta_M0']))
    sections.append('<p>已揭示段有27日原始输入逐列在界内但联合近邻稀少；稀少/越界组平均误差更高。然而595个已评分“较密且逐列在界内”日的MSE仍比M0差，说明支持不足只解释一部分。维度、相关结构、事件数量与正则选择尚无法唯一拆开。</p><h3>均值不确定性不是未来收益残差</h3><p>M2条件HAC标准误中位数在开发/已揭示段约1.18%/1.09%，未来残差RMSE约4.62%/5.85%，两者不是一个量。已揭示段约86.6%的μ距门槛落在±1.96×SE21内；这提示决策对估计扰动敏感。该SE固定原变换与α，使用21/63个真实日历滞后的Newey-West协方差，忽略正则偏差、选参和变换不确定性；不是已校准置信区间，更不输出“盈利概率”。M0未估这项SE。</p><h3>原内层选择全部保留，不再扩大网格</h3>'+table(selection,{'stage':'来源','selection_id':'选择日','head':'预测头','chosen':'原选择','at_strongest_boundary':'最强收缩端','numerically_tied_alternatives':'数值并列数','relative_nearest_gap':'最近替代相对损失差'},percent=['relative_nearest_gap']))
    sections.append('<p>P4严格胜出/并列判断直接复用原收尾记录，未冒充新发现。P7的2024 M2三个头均离开最强收缩端，μ最接近替代的验证损失差约2.18%，高于本轮事先规定的1%“微小差”描述线；数值并列线仍为1e-12。2025/2026选择重新变强，但2025主账户仍负收益，是“只要强收缩就保证赚钱”的反例。</p><p>'+link(root,output/'D3_training_influence.csv','逐训练行梯度/杠杆/影响')+' · '+link(root,output/'D3_changed_predictions.csv','全部1,260簇的三类预测扰动')+' · '+link(root,output/'D3_design.csv','完整各月设计与正则谱摘要')+' · '+link(root,output/'D3_prediction_support.csv','逐预测近邻与交互越界')+' · '+link(root,output/'D3_correlations','全部原始/设计相关矩阵')+' · '+link(root,output/'original_candidates','原候选得分与选择快照')+'</p></section>')
    sections.append('<section id="d4"><h2>D4｜信息可以描述状态，风险信息与净收益信息要分别检验</h2><p><b>问题：</b>19项输入是当前描述、未来风险强度信息，还是未来净收益信息？先复用P6/P7全部家族删列。E01来自B/D；删除B仍经E保留部分信息，因此列消融与原始来源的因果贡献不同。下表为删除相对原M2的损失变化，负值表示删除后损失更低，不表示该家族在所有模型或时期都无用。</p>'+table(pd.DataFrame(family),{'stage':'原阶段','removed':'删除列家族','mse':'μ MSE变化','log_loss':'P10损失变化','pinball':'Q90损失变化'},percent=['mse','log_loss','pinball']))
    sections.append('<h3>只新增一个日线风险目标、两个固定方法</h3><p>V5是原执行后五个每日收盘对数总回报的平方均值，<b>不是高频已实现方差，也不是减去均值后的条件方差</b>。比较一个仅用自身21日波动的简约方法与一个原63列受约束联合方法。两者同成熟训练行、月度重估、log-link、固定L2=1，无选参网格；以QLIKE目标拟合正的平方回报均值，预测下限1e-12且本次投影0行。没有对数目标回归后直接反变换的均值偏差。</p>'+table(var.loc[var.period_type.eq('all')],{'scope':'时期','model':'方法','rows':'同日期成熟行','qlike':'QLIKE','mse':'V5 MSE','mean_observed_V5':'实际V5均值','mean_predicted_V5':'预测V5均值'},scientific=['qlike','mse','mean_observed_V5','mean_predicted_V5'])+table(vint,{'scope':'时期','block':'块长','joint_minus_self':'联合−简约 QLIKE','lower':'95%下端','upper':'95%上端'},scientific=['joint_minus_self','lower','upper']))
    sections.append('<p>QLIKE使用 <code>log(v̂)+V5/v̂</code>，可为负值，应比较差值，不能把它直接转成损失百分比。两段总体点估计均较低，但21/63日配对区间都跨0。2024联合QLIKE与MSE都略差；2025/2026点估计较好。结论是“风险强度有值得追查的线索，尚未确认稳定增量”；这不改变原收益头MSE、Q90和P10较差的事实。</p><h3>同一过去可定义状态中的原预测增量</h3>'+table(states,{'state':'当时状态','metric':'原目标评分','rows':'成熟行','M2_minus_M0':'M2−M0损失','lower':'63日块95%下端','upper':'上端'},scientific=['M2_minus_M0','lower','upper']))
    sections.append('<p>状态边界来自当时训练，不是按未来赚钱日期挑选；稀少状态区间尤其宽。原R5/L5/Y10和静态五日全仓净收益代理均保留。该净收益代理扣入/出场两侧费用，不能把重叠五日值累加为账户业绩，实际经济性仍看连续账本。A/B/C描述当前定价和压力，D/F描述已发生路径，E是它们的派生差；这些明确功能不等于收益择时有效。单列相关表只作功能描述，不是SHAP因果排名或准入标准。</p><p>更快状态的具体缺口已经可以指出：C03五日变化无法区分五日仍升而最近一日已降，F04的21日窗口混合新冲击与旧冲击滚出。本轮逐日表已把两者分开，但没有据此新增交易输入、改模型或搜索新状态。</p><p>'+link(root,output/'D4_metrics.csv','全部年度/季度/状态方差评分')+' · '+link(root,output/'D4_paired_intervals.csv','全部风险目标配对区间')+' · '+link(root,output/'D4_original_state_increment.csv','原三目标状态配对区间')+' · '+link(root,output/'D4_feature_functions.csv','19列功能相关诊断')+' · '+link(root,output/'reused','原P6/P7/N1结果副本与索引')+'</p></section>')
    sections.append('<section id="d5"><h2>D5｜只提出一个最小干预，尚未执行</h2><p>主假说是收益头在2024显著减弱正则后，对有限事件的估计更敏感，再由硬门槛放大为持仓变化。建议唯一试验 <code>M2_MU_ALPHA100_D5</code>：保持均值目标、原输入/变换/时钟/训练集/映射/成本不变，只固定μ头α=100；Q90/P10原预测不变。它不会退化成M0。</p><p>该干预只会改变原α不同的2024月份，其他年份预测作为不变对照，连续账户仍承接前期持仓/净值差异；既检验MSE与估计稳定性，也检验同日期成本、敞口和尾部亏损。若机制被推翻，保留结果并停止；不自动开始第二轮搜索。当前证据足够提出这项有限反证，尚不足以支持新主动机会策略或资本使用。</p><p>'+link(root,output/'D5_proposal.md','不超过一页的假说比较、单一干预与反证标准')+'</p><p class="notice">D5未运行。D1—D4在此停止，等待用户验收及是否执行D5的单独决定。</p></section>')
    methods=read_json(root/'runs/m2_mechanism/methods_sources.json')['sources']
    methodlinks=''.join('<li><a href="'+escape(m['url'],quote=True)+'">'+escape(m['name'])+'</a>：'+escape(m['use'])+'（'+escape(m['retrieval'])+'）</li>' for m in methods)
    sections.append('<section id="delivery"><h2>交付、验证与可复算边界</h2><div class="verdict-grid"><article><h3>工程</h3><p>D1—D4计算完成；'+escape(testtext)+'。132账户/99,090行（含现金锚点）按独立股数与现金方程复核；最大金额差 '+f"{validation['account_max_error']:.3g}"+' 美元。73份M2原训练身份吻合；μ保存系数与独立正规方程吻合；1,260簇固定/全变换检查及146项方差拟合完成。</p></article><article><h3>研究</h3><p>存在收益头→门槛损害、有限事件敏感性及局部支持不足；没有唯一可加根因，也没有得到新策略。全部正负比较保留，D5未执行。事后块区间不覆盖人类研究选择的不确定性。</p></article><article><h3>真实数据</h3><p>复用2019-01-02至2026-09-04的1,930个交易日；核心19列完整1,909日。两段各831/672个预测日、826/667个评分日；原开发跨2024与末端未成熟五日均不补值。新增行情下载0；账户连接0。</p></article><article><h3>实际限制</h3><p>历史来源仍采用已验收次日可得假设，缺乏完整历史版本/PIT。日线收盘是成交代理，不含日内路径。现金回报0；基金费不重复扣。日报真实新日下载验证与Q/P一致性问题保留。</p></article></div>')
    sections.append('<p>运行：<code>.venv/bin/python -m svxylab mechanism --open</code>，或双击项目内“M2机制归因.command”。每次新建输出目录。原请求自2020-01-02开始，前175日无预测服务；其完整请求记录在输出original_prediction_service.csv中，未补造预测。原7,589个数据/产物文件已逐一核对摘要；原日报算法、模型、变换、预测、标签和账户不改写。仅命令分发与兼容读取各改一处源码入口，当前规格追加本轮边界；原P7冻结和N1记录保持。内置浏览器安全策略禁止本地HTML预览，页面截图未核验；本地链接、脚本语法与生成图表另行检查。</p><p>'+link(root,output/'experiment_record.json','计算前问题、控制变量与刻度')+' · '+link(root,output/'calculation_validation.json','真实计算验证')+' · '+link(root,root/'runs/m2_mechanism/refinements_after_first_run.json','首轮后发现的摘要修正记录')+' · '+link(root,root/'runs/m2_mechanism/runtime_compatibility.json','入口兼容变更')+' · '+link(root,root/'runs/m2_mechanism/baseline.json','原产物摘要与提交基准')+' · '+link(root,output,'本次完整输出目录')+'</p><details><summary>方法来源及实际查阅范围</summary><p>来源用于核对统计目标、评分与动态成本概念，不证明SVXY机制。部分出版社正文未取得，查阅范围如实保留。</p><ul>'+methodlinks+'</ul></details></section>')
    cols=['model','scope','decision_session','as_of_session','fit_id','execution_at','decision_at','mu5','q90_raw','q90','p10','mu_margin','risk_capacity','gate','old_weight','pre_trade_weight','target_weight','holding_pnl','cost','equity_end','R5','L5','score_observed','label_end_session','gate_component','risk_component','gate_risk_interaction','price_drift_rebalance','raw_outside_count','design_joint_sparse',*['A01','A02','A03','A04','A05','B01','B02','B03','C01','C02','C03','D01','D02','D03','E01','F01','F02','F03','F04']]
    payload=chain[cols].to_json(orient='records',force_ascii=False).replace('</','<\\/')
    # Full derived daily inputs remain in the ignored local data directory, not in a Git-bound HTML report.
    (output/'failure_chain_data.js').write_text('const M2_FAILURE_RECORDS='+payload+';\n')
    script='''const records=DATA;const model=document.querySelector('#model'),day=document.querySelector('#day');
function money(x){return x==null?'未成熟':x.toLocaleString('zh-CN',{maximumFractionDigits:2})}function pc(x){return x==null?'未成熟':(100*x).toFixed(2)+'%'}
function populate(){let current=day.value||'2024-08-05';day.innerHTML='';records.filter(r=>r.model===model.value).forEach(r=>{let o=document.createElement('option');o.value=r.decision_session;o.textContent=r.decision_session;day.append(o)});day.value=current;if(!day.value)day.selectedIndex=0;draw()}
function draw(){const r=records.find(r=>r.model===model.value&&r.decision_session===day.value);const card=(h,t)=>'<article><h3>'+h+'</h3>'+t+'</article>';
document.querySelector('#flow').innerHTML=card('1 · 当时输入','<p>信息日 '+r.as_of_session+'</p><p>VIX '+Math.exp(r.B01).toFixed(2)+' · VIX5日 '+pc(Math.expm1(r.C03))+'</p><p>SVXY5日 '+pc(Math.expm1(r.F03))+' · 21日波动 '+pc(r.F04)+'</p>')+card('2 · 原预测','<p>μ5 '+pc(r.mu5)+'<br>Q90 '+pc(r.q90)+'（原值 '+pc(r.q90_raw)+'）<br>P10 '+pc(r.p10)+'</p><small>'+r.fit_id+'</small>')+card('3 · 决策转换','<p>距2c门槛 '+pc(r.mu_margin)+'<br>风险容量 '+pc(r.risk_capacity)+'<br>主目标 '+pc(r.target_weight)+'</p><small>截止 '+r.decision_at+'<br>执行 '+r.execution_at+'</small>')+card('4 · 实际持仓','<p>旧仓位 '+pc(r.old_weight)+'<br>执行前漂移 '+pc(r.pre_trade_weight)+'<br>新仓位 '+pc(r.target_weight)+'</p>')+card('5 · 损益与后来结果','<p>旧仓当日损益 $'+money(r.holding_pnl)+'<br>本次费用 $'+money(r.cost)+'<br>期末 $'+money(r.equity_end)+'</p><p>随后R5 '+pc(r.R5)+'<br>随后L5 '+pc(r.L5)+'</p>');
document.querySelector('#decomposition').textContent='权重分解：门槛项 '+pc(r.gate_component)+'，风险项 '+pc(r.risk_component)+'，交互项 '+pc(r.gate_risk_interaction)+'，价格漂移再平衡项 '+pc(r.price_drift_rebalance)+'。后来结果截至 '+r.label_end_session+'；当天旧仓损益不等于随后五日标签。';
let keys=['A01','A02','A03','A04','A05','B01','B02','B03','C01','C02','C03','D01','D02','D03','E01','F01','F02','F03','F04'];document.querySelector('#features').innerHTML='<div class="input-grid">'+keys.map(k=>'<p><b>'+k+'</b> '+r[k].toPrecision(7)+'</p>').join('')+'</div>'}
model.addEventListener('change',populate);day.addEventListener('change',draw);document.querySelector('#prev').onclick=()=>{day.selectedIndex=Math.max(0,day.selectedIndex-1);draw()};document.querySelector('#next').onclick=()=>{day.selectedIndex=Math.min(day.options.length-1,day.selectedIndex+1);draw()};populate();'''.replace('DATA','M2_FAILURE_RECORDS',1)
    css='''*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#f3f1eb;color:#222d32;font:16px/1.75 -apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif}main{max-width:1320px;margin:auto;padding:32px 32px 70px}header{padding:25px 0 20px;border-bottom:3px solid #223b49}.eyebrow{letter-spacing:2px;font-size:12px;color:#5c747d}h1{font-size:42px;line-height:1.25;margin:10px 0}h2{font-size:25px;margin:0 0 20px}h3{font-size:18px;margin:0 0 12px}.subtitle{color:#64747c;font-size:19px}nav{display:flex;flex-wrap:wrap;gap:18px;padding:20px 0}a{color:#185d79;text-underline-offset:3px}section{scroll-margin-top:18px;background:#fff;padding:30px;margin:20px 0;border:1px solid #dfe3e1;border-radius:8px}p{margin:10px 0 18px}.verdict-grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.verdict-grid article{background:#f6f8f8;padding:20px;border-left:3px solid #6e8d9a}ul{margin:0;padding-left:20px}li{margin:9px 0}.notice{background:#faf0e7;padding:18px;border-left:3px solid #a45734}.table-scroll{overflow:auto;max-width:100%;margin:20px 0}table{border-collapse:collapse;font-size:13px;line-height:1.6;width:100%}th,td{padding:10px 12px;text-align:left;border-bottom:1px solid #e1e5e5;white-space:nowrap}th{background:#edf2f3;color:#284854}tr:nth-child(even){background:#fafbfb}.figure{width:100%;height:auto}.caption,small{font-size:12px;color:#66777e}code{background:#eef1f2;padding:2px 5px;border-radius:4px;font-size:14px;overflow-wrap:anywhere}.controls{display:flex;flex-wrap:wrap;align-items:center;gap:12px;margin:20px 0}select,button{font:inherit;padding:8px 12px;border:1px solid #9db0b8;background:#fff;border-radius:4px}button{cursor:pointer}.flow{display:grid;grid-template-columns:repeat(5,1fr);gap:10px}.flow article{min-width:0;background:#edf3f5;border-top:3px solid #567f93;padding:15px}.flow h3{font-size:16px}.flow p{font-size:14px}.flow small{overflow-wrap:anywhere}.input-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:5px}details{padding:15px;background:#f8faf9}summary{cursor:pointer;font-weight:600}@media(max-width:900px){main{padding:16px}section{padding:18px}.flow{grid-template-columns:1fr 1fr}.verdict-grid{grid-template-columns:1fr}h1{font-size:34px}.input-grid{grid-template-columns:1fr 1fr}}@media print{body{background:#fff}main{max-width:none}section{break-inside:auto;border:0}.table-scroll{overflow:visible}nav,.controls{display:none}.verdict-grid{display:block}.flow{grid-template-columns:repeat(3,1fr)}}'''
    data_src=os.path.relpath(output/'failure_chain_data.js',root/'reports')
    html='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>M2机制归因 · SVXYLab</title><style>'+css+'</style><body><main>'+''.join(sections)+'</main><script src="'+data_src+'"></script><script>'+script+'</script></body></html>'
    report=root/'reports/mechanism.html';report.write_text(html)
    def snapshot_link(match):
        attr,ref=match.group(1),match.group(2)
        if ref.startswith(('http','#')):return match.group(0)
        return attr+'=\"'+os.path.relpath((root/'reports'/ref).resolve(),output)+'\"'
    (output/'report.html').write_text(re.sub(r'(href|src)="([^"]+)"',snapshot_link,html))
    return report


def build_mechanism_report(root, *, open_report=False):
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    run=root/'runs/m2_mechanism'/stamp;run.mkdir(parents=True)
    output=root/'data/clean/m2_mechanism'/stamp
    started={'stage':'M2_D1_D4','command':[sys.executable,'-m','svxylab','mechanism']+(['--open'] if open_report else []),
        'output_dir':str(output.relative_to(root)),'run_dir':str(run.relative_to(root)),
        'started_at_utc':datetime.now(timezone.utc).isoformat(),
        'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        'source_sha256':{str(p.relative_to(root)):digest(p) for p in sorted((root/'src').rglob('*.py'))},
        'experiment_sha256':digest(root/'runs/m2_mechanism/experiment_record.json')}
    write_json(run/'started.json',started);write_json(root/'runs/m2_mechanism/latest_run.json',started)
    t=perf_counter()
    try:
        validation=compute(root,output)
        pytest=subprocess.run([sys.executable,'-m','pytest','-q',f'--junitxml={run / "pytest.xml"}'],cwd=root,text=True,capture_output=True)
        (run/'pytest.log').write_text(pytest.stdout+pytest.stderr)
        if pytest.returncode:raise ValueError('普通测试失败，见'+str(run/'pytest.log'))
        import xml.etree.ElementTree as ET
        suite=ET.parse(run/'pytest.xml').getroot().find('testsuite')
        tests={k:int(suite.attrib[k]) for k in ['tests','failures','errors','skipped']}
        report=render_report(root,output,validation,tests)
        result={**started,'elapsed_seconds':perf_counter()-t,'validation':validation,'ordinary_tests':tests,
            'report':str(report.relative_to(root)),'report_sha256':digest(report),'D5_executed':False,
            'output_sha256':{str(p.relative_to(output)):digest(p) for p in sorted(output.rglob('*')) if p.is_file()}}
        if open_report:result['open_returncode']=subprocess.run(['open',str(report)]).returncode
        write_json(run/'mechanism_run.json',result);write_json(root/'runs/m2_mechanism/latest_run.json',result)
        print(f'机制归因完成：{report}\n普通测试{tests["tests"]}项；{validation["account_count"]}账户、{validation["account_rows"]}行；D5未执行。',flush=True)
        return 0
    except Exception:
        (run/'failure.txt').write_text(traceback.format_exc());raise
