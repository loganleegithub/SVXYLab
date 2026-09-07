"""P6中文静态报告，沿用本地入口、CSV/JSON与实际运行记录。"""

from datetime import datetime, timezone
from hashlib import sha256
from html import escape
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter
import traceback
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from svxylab.data_report import table
from svxylab.diagnostics import run_diagnostics
from svxylab.diagnostics_audit import validate
from svxylab.economics_report import money, percent, points
from svxylab.environment import run_command
from svxylab.features_report import write_json

NAMES={'M0':'M0历史常数','M1':'M1线性','M2':'M2原主版本','M2_REPLAY':'M2原规格复算',
       **{f'M2_DROP_{f}':f'M2删除{f}列' for f in ['A','B','C','D','E','F','BDE']},
       'M3_RECENCY':'M3历史权重','M2_SKEW':'M2加SKEW','M2_SKEW_REFERENCE':'M2同SKEW样本对照',
       'cash':'现金','fixed_25':'固定25%','fixed_50':'固定50%','fixed_75':'固定75%','fixed_100':'固定100%','curve':'F2>F1'}


def name(value):
    for suffix,label in [('_risk_only',' risk-only'),('_main',' main')]:
        if value.endswith(suffix):return NAMES.get(value[:-len(suffix)],value[:-len(suffix)])+label
    return NAMES.get(value,value)


def number(v):
    return f'{v:.7g}' if pd.notna(v) else '—'


def plots(root,output):
    os.environ.setdefault('MPLCONFIGDIR',str(root/'.cache/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font=FontProperties(fname='/System/Library/Fonts/STHeiti Light.ttc')
    ci=pd.read_csv(output/'bootstrap_intervals.csv')
    selected=ci.loc[ci.domain.eq('prediction') & ci.block_sessions.eq(21)]
    labels=list(dict.fromkeys(zip(selected.left,selected.right)))
    fig,axes=plt.subplots(1,3,figsize=(15,9),layout='constrained',sharey=True)
    for ax,field,title in zip(axes,['mse','log_loss','pinball'],['均值MSE差','概率对数损失差','Q90 Pinball差']):
        rows=selected.loc[selected.metric.eq(field)].set_index(['left','right']).loc[labels]
        for y,(_,r) in enumerate(rows.iterrows()):
            ax.plot([r.lower_95,r.upper_95],[y,y],color='#52788b');ax.scatter(r.point_difference,y,color='#a13d3d',s=24)
        ax.axvline(0,color='#999',linewidth=1);ax.set_title(title,fontproperties=font);ax.grid(axis='x',alpha=.2)
        ax.ticklabel_format(axis='x',style='sci',scilimits=(0,0),useMathText=True)
    axes[0].set_yticks(range(len(labels)),[name(a)+' − '+name(b) for a,b in labels],fontproperties=font,fontsize=9)
    axes[0].invert_yaxis();fig.suptitle('共同日期配对差：负值表示左侧损失较低；21日时间块的逐项95%区间',fontproperties=font)
    fig.savefig(output/'prediction_uncertainty.png',dpi=140);plt.close(fig)
    for cohort,models in [('core_common',['M0','M1','M2','M3_RECENCY']),('skew_common',['M2','M2_SKEW_REFERENCE','M2_SKEW'])]:
        fig,axes=plt.subplots(2,1,figsize=(13,8),layout='constrained',sharex=True)
        for ax,policy in zip(axes,['main','risk_only']):
            for model in models:
                strategy=model+'_'+policy;v=pd.read_csv(output/f'ledgers/{cohort}_{strategy}.csv')
                ax.plot(pd.to_datetime(v.as_of_session),v.equity_end,label=name(strategy),linewidth=1.4)
            v=pd.read_csv(output/f'ledgers/{cohort}_fixed_100.csv')
            ax.plot(pd.to_datetime(v.as_of_session),v.equity_end,label='固定100%',color='#777',linewidth=1.)
            ax.set_title(policy+' · 同资本、同收盘时点重建',fontproperties=font);ax.legend(prop=font,ncols=3);ax.grid(alpha=.2)
            ax.set_ylabel('账户美元',fontproperties=font)
            ax.set_xlim(pd.to_datetime(v.as_of_session.iloc[0]),pd.to_datetime(v.as_of_session.iloc[-1]))
        fig.savefig(output/f'{cohort}_equity.png',dpi=140);plt.close(fig)


def account_table(frame):
    return table(['账户','实际起跑日','日数','净收益','CAGR','最大回撤','最差单日','最差五日','平均敞口','现金日比例','周转倍数','费用 $'],
        [[name(r.strategy),r.first_execution,r.account_days,percent(r.total_return),percent(r.cagr),percent(r.max_drawdown),
          percent(r.worst_day)+' / '+r.worst_day_session,percent(r.worst_five_days)+' / '+r.worst_five_start+' → '+r.worst_five_end,
          percent(r.mean_exposure),percent(r.cash_fraction),f'{r.total_turnover:.2f}',money(r.cost_dollars)] for r in frame.itertuples()])


def render(root,data):
    result=data['result']; output=root/result['output_dir']; rel=result['output_dir']; read=lambda f:pd.read_csv(output/f)
    metrics=read('prediction_metrics.csv');paired=read('paired_prediction_periods.csv');accounts=read('account_metrics.csv')
    ci=read('bootstrap_intervals.csv');runs=read('variant_runs.csv');specs=json.loads((output/'variants.json').read_text())
    all_pairs=paired.loc[paired.period_type.eq('all')]
    paired_table=table(['左侧','配对对照','共同评分日','均值MSE相对变化','对数损失相对变化','Pinball相对变化'],
        [[name(left),name(right),int(g.rows.iloc[0]),*[percent(g.loc[g.metric.eq(f),'relative_change'].item()) for f in ['mse','log_loss','pinball']]]
         for (left,right),g in all_pairs.groupby(['left','right'],sort=False)])
    base=metrics.loc[metrics.period_type.eq('all')].set_index('model')
    headlines=[]
    for model,reference in [('M3_RECENCY','M2'),('M2_SKEW','M2_SKEW_REFERENCE')]:
        g=all_pairs.loc[all_pairs.left.eq(model)&all_pairs.right.eq(reference)]
        changes=[percent(g.loc[g.metric.eq(f),'relative_change'].item()) for f in ['mse','log_loss','pinball']]
        headlines.append(f"{name(model)}相对{name(reference)}的MSE / 对数损失 / Pinball分别变化{' / '.join(changes)}；负值表示损失较低。")
    core=accounts.loc[accounts.cohort.eq('core_common')].set_index('strategy'); skew=accounts.loc[accounts.cohort.eq('skew_common')].set_index('strategy')
    for model,cohort in [('M3_RECENCY',core),('M2_SKEW',skew)]:
        headlines.append(f"{name(model)} main净收益{percent(cohort.loc[model+'_main','total_return'])}、最大回撤{percent(cohort.loc[model+'_main','max_drawdown'])}、平均敞口{percent(cohort.loc[model+'_main','mean_exposure'])}；"
            f"risk-only净收益{percent(cohort.loc[model+'_risk_only','total_return'])}、最大回撤{percent(cohort.loc[model+'_risk_only','max_drawdown'])}。")
    variant_table=table(['试验','作用','删除列','删除交互','保留输入列数','首次预测执行日','有预测 / 无预测日'],
        [[name(v['name']),v['role'],', '.join(v['removed_columns']) or '—',', '.join(v['removed_interactions']) or '—',len(v['columns']),
          runs.loc[runs.variant.eq(v['name']),'first_forecast'].item(),
          str(runs.loc[runs.variant.eq(v['name']),'forecast_sessions'].item())+' / '+str(1006-runs.loc[runs.variant.eq(v['name']),'forecast_sessions'].item())] for v in specs])
    weight=pd.read_csv(output/'variants/M3_RECENCY/fits.csv')
    weights_table=table(['月拟合执行日','名义训练行','有效训练行（权重ESS）','阳性行','重叠事件簇'],
        [[r.fit_session,r.training_rows,f'{r.effective_training_rows:.2f}',r.positive_count,r.event_clusters] for r in weight.itertuples()])
    loss_ci=ci.loc[ci.domain.eq('prediction')]
    uncertainty_notes=[]
    for left,right,metric in [('M2','M0','mse'),('M3_RECENCY','M2','pinball'),('M3_RECENCY','M2','mse'),('M2_SKEW','M2_SKEW_REFERENCE','mse')]:
        g=loss_ci.loc[loss_ci.left.eq(left)&loss_ci.right.eq(right)&loss_ci.metric.eq(metric)]
        text='两种块长的区间均高于0，指向左侧该项损失较高' if g.lower_95.gt(0).all() else '两种块长的区间均低于0，指向左侧该项损失较低' if g.upper_95.lt(0).all() else '区间包含0或对块长敏感，不能把点差直接称为稳定增量'
        uncertainty_notes.append(f'{name(left)} − {name(right)}的{metric}：{text}。')
    improved=[]
    for left,g in all_pairs.loc[all_pairs.left.str.startswith('M2_DROP_')&all_pairs.metric.isin(['mse','log_loss','pinball'])].groupby('left'):
        if g.difference.lt(0).all():improved.append(name(left))
    ablation_note='、'.join(improved)+'后，三项平均损失点值均下降；这只描述该表示方式下的冗余或失配，不能证明对应原始来源无用，也没有据此修改主版本。'
    m2_year=paired.loc[paired.left.eq('M2')&paired.right.eq('M0')&paired.period_type.eq('year')&paired.metric.eq('mse')]
    year_note='M2相对M0的MSE逐年变化为'+ '、'.join(str(r.period)+'年'+percent(r.relative_change) for r in m2_year.itertuples())+'。年份间方向与幅度不同，不能只看总体或某一次危机。'
    drop2022=paired.loc[paired.left.eq('M2')&paired.right.eq('M0')&paired.period_type.eq('leave_year_out')&paired.period.astype(str).eq('2022')&paired.metric.eq('pinball'),'relative_change'].item()
    year_note+=f'仅移除2022评分行后，M2相对M0的Pinball变为{percent(drop2022)}；这说明总体差异依赖该年，不是另一次重新训练的结果。'
    ci_table=table(['左侧','右侧','指标','块长','共同评分行','点差','95%下界','95%上界'],
        [[name(r.left),name(r.right),r.metric,r.block_sessions,r.paired_rows,number(r.point_difference),number(r.lower_95),number(r.upper_95)] for r in loss_ci.itertuples()])
    econ_ci=ci.loc[ci.domain.eq('economics')&ci.metric.isin(['total_return','max_drawdown'])]
    econ_ci_table=table(['口径','左侧','右侧','指标','块长','点差','95%下界','95%上界'],
        [[r.cohort,name(r.left),name(r.right),r.metric,r.block_sessions,points(r.point_difference),points(r.lower_95),points(r.upper_95)] for r in econ_ci.itertuples()])
    years=paired.loc[paired.period_type.isin(['year','leave_year_out'])&paired.metric.isin(['mse','log_loss','pinball'])]
    year_table=table(['区间口径','年份','左侧','右侧','指标','配对行','损失差','相对变化'],
        [[r.period_type,r.period,name(r.left),name(r.right),r.metric,r.rows,number(r.difference),percent(r.relative_change)] for r in years.itertuples()])
    events=read('positive_events.csv');event_table=table(['事件','首个执行日','重叠区间终点','阳性行数'],
        [[r.event_id,r.start,r.end,r.positive_rows] for r in events.itertuples()])
    event_accounts=read('account_periods.csv')
    selected=event_accounts.loc[event_accounts.cohort.eq('core_common')&event_accounts.period_type.eq('event')&event_accounts.strategy.isin(['M0_main','M2_main','M2_risk_only','M3_RECENCY_main','fixed_100'])]
    event_paths=table(['事件','账户','连续账户路径净变化','路径收盘回撤','平均敞口','现金日数','费用 $'],
        [[r.period,name(r.strategy),percent(r.continuous_path_return),percent(r.path_max_drawdown),percent(r.mean_exposure),r.cash_days,money(r.cost_dollars)] for r in selected.itertuples()])
    quarterly=event_accounts.loc[event_accounts.cohort.eq('core_common')&event_accounts.period_type.eq('quarter')&event_accounts.strategy.isin(['M0_main','M2_main','M2_risk_only','M3_RECENCY_main','fixed_100'])]
    quarters=table(['季度','账户','连续净变化','路径回撤','平均敞口','无服务日'],
        [[r.period,name(r.strategy),percent(r.continuous_path_return),percent(r.path_max_drawdown),percent(r.mean_exposure),r.startup_unserved_days] for r in quarterly.itertuples()])
    prediction_all=table(['模型','评分日','阳性行','MSE','对数损失','Pinball','Brier','Q90覆盖'],
        [[name(k),int(r.rows),int(r.events),number(r.mse),number(r.log_loss),number(r.pinball),number(r.brier),percent(r.q90_coverage)] for k,r in base.iterrows()])
    links=' · '.join(f"<a href='../{rel}/{p.relative_to(output).as_posix()}'>{escape(p.name)}</a>" for p in sorted(output.glob('*.csv')))
    ledger_links=' · '.join(f"<a href='../{rel}/{r.ledger_file}'>{escape(r.cohort+' '+name(r.strategy))}</a>" for r in accounts.itertuples())
    variant_links=' · '.join(f"<a href='../{rel}/variants/{v['name']}/predictions.csv'>{escape(name(v['name']))}预测</a> · <a href='../{rel}/variants/{v['name']}/training_rows.csv'>训练成员/权重</a> · <a href='../{rel}/variants/{v['name']}/cv_scores.csv'>候选分数</a>" for v in specs)
    commands=''.join(f"<details><summary>退出{c['exit_code']} · {escape(c['command'])}</summary><pre>{escape(c['output'])}</pre></details>" for c in data['commands'])
    audit=data['independent_audit']; ev=result['evaluation']
    doc=f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'><title>SVXYLab · P6 诊断与不确定性</title>
<style>body{{margin:0;background:#fff;color:#243440;font:16px/1.75 -apple-system,BlinkMacSystemFont,'PingFang SC',sans-serif}}main{{max-width:1450px;margin:auto;padding:28px 24px 70px}}h1{{font-size:28px}}h2{{font-size:22px;margin-top:32px}}a{{color:#176785}}.note{{background:#f2f6f8;border-left:4px solid #52788b;padding:12px 18px}}.scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{border:1px solid #d4dce2;padding:7px;text-align:left;vertical-align:top}}th{{background:#f3f6f8}}img{{max-width:100%;height:auto}}details{{padding:9px 0;border-bottom:1px solid #ddd}}summary{{cursor:pointer}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}</style></head>
<body><main><h1>SVXYLab · P6 信息家族、挑战者与时间块不确定性</h1><p>{data['generated_at']} · RESEARCH_ONLY · P5验收检查点 {result['accepted_p5_checkpoint'][:7]}</p>
<div class='note'><p><strong>工程执行：</strong>{'完成' if data['engineering_complete'] else '未完成'}。{data['tests_passed']}项普通测试通过；7项列消融、2个挑战者和2项复算/样本对照实际完成。原P4预测和P5账本保留，未改模型目标、网格或仓位公式。</p>
<p><strong>研究结果：</strong>下列差异均为2019起积累训练样本后的2020～2023开发段诊断。按共同日期配对，逐项给出较好和较差结果；没有一个综合分数或自动晋升。</p>
<p><strong>真实覆盖：</strong>完整请求1006个股票交易日，核心共同账户从{ev['cohorts']['core_common']['first_execution']}开始，SKEW配对账户从{ev['cohorts']['skew_common']['first_execution']}开始。此前初始现金属于无服务，不能宣称预警疫情。</p>
<p><strong>下一阶段的封存主版本仍是预定M2。</strong>未执行P7，未评价2024年后模型或账户结果。</p></div>
<h2>本次真实结果</h2>{''.join('<p>'+x+'</p>' for x in headlines)}
<p>M1/M2在P4的主要平均预测损失未优于M0，这一负结果保持不变。经济回报须与敞口、回撤、费用一起看；本轮结果不能证明P10有效或可用于资金。</p>
<h2>限定消融与两个挑战者</h2>{variant_table}
<p>A～F逐组删除后重新拟合，同步删除引用这些列的交互；训练和预测仍要求原19项核心完整，避免因删列自动获得更多样本。B+D+E额外联合删除，防止E保留B/D的关键变换。</p>
<p><strong>这些都是删列，不是删除完整原始来源。</strong>B+D+E删除后，A03、C03仍包含VIX，F01/F02仍包含SPY价格信息。因此不能推论“VIX或SPY原始来源无用”。E由B/D派生，不是独立来源。M2与M1的差异同时含样条主效应和交互，不是纯交互因果效应；本轮没有额外搜索交互组合。</p>
<p>M2_REPLAY从原训练输入重新拟合，只核对P6实现能复现已验收M2，逐日6个原始/发布字段最大差{result['m2_full_training_refit_max_prediction_difference']:.3g}；它不是新候选。</p>
<h3>M3：504个完整股票交易日半衰期</h3><p>年龄按完整交易日日历计，不因缺失压缩。每个内外层训练集的权重归一化到和为训练行数。只有三个预测头的训练损失加权；标准化、样条节点和交互尺度保持原训练期无权变换，以单独观察损失重加权效果。验证损失仍逐行等权，原三档网格和更强收缩并列规则不变。单类别概率使用加权阳性与Jeffreys平滑。</p>
<details><summary>全部40个月的名义样本、有效样本和阳性支持</summary>{weights_table}</details>
<h3>SKEW：相同可用样本，单独增加两列</h3><p>沿用已验收P3的Cboe真实SKEW原件、G01水平/G02五日变化和ASSUMED_NEXT_SESSION约定。开发段预热后2019-07-05、2019-07-12的缺失保留，不插值；这使首个可拟合日期及训练成员有所变化。</p>
<p>M2_SKEW_REFERENCE与M2_SKEW的训练、验证成员和预测可用日期完全相同，前者只有原核心列，后者新增两个样条主效应；不添加新交互或近期权重。它是样本覆盖对照，不是第三个挑战者。另报该对照相对原M2的差异，区分样本变化和增加SKEW列。历史精确published_at仍未知，不能宣称完整历史PIT。</p>
<h2>预测损失：共同日期配对</h2>{paired_table}<p>相对变化=(左侧损失/右侧损失)−1，负值较低；不是收益率。不同SKEW样本不得拿未配对总体数直接相减。没有按结果扩大正则网格或修改目标。</p>
<p>{ablation_note}</p>
<details><summary>全部模型自身总体指标（单独标明样本数）</summary>{prediction_all}</details>
<h2>连续账户：同资本、同一实际收盘起点</h2><p>保留P5基础B=.05、单边c=.0005及main/risk-only，所有账户以100,000美元现金重新开始。信号t在下一股票交易日收盘执行；旧仓位承担此前损益，缺预测保持实际份额/现金，逐笔成交额扣费。持仓不读score_observed或未来标签，不以五日重叠标签叠加交易。</p>
<h3>核心共同起点</h3>{account_table(accounts.loc[accounts.cohort.eq('core_common')])}<img src='../{rel}/core_common_equity.png' alt='核心共同起点连续账户'>
<h3>SKEW共同起点，全部重新起跑</h3>{account_table(accounts.loc[accounts.cohort.eq('skew_common')])}<img src='../{rel}/skew_common_equity.png' alt='SKEW共同起点连续账户'>
<details><summary>完整2020～2023请求日历（包含各模型无服务时段）</summary>{account_table(accounts.loc[accounts.cohort.eq('full')])}</details>
<p>最差五日由连续5个账户日收益复合；平均敞口是承担当日收益的旧仓位，现金时间和实际换手/费用同时报告。末日按收盘市值计价，不强制卖出；跨2024标签保持为空，2023尾部预测仍可交易。Q90不是最大损失保证，B=5%不等于最大回撤5%。</p>
<h2>21/63日配对时间块，1000次</h2><p>在完整连续股票交易日日历上随机抽取连续块，非环形、不把末年接回首年；拼接后截取原日历长度。同一次抽样的左右两条序列使用相同索引。缺预测/不可评分仍占原日期、作为NaN，仅对共同有效行计算损失差，不压缩缺失制造“连续”样本。</p>
<p>固定种子1707+块长。区间为逐项2.5%/97.5%经验百分位；未作多重比较校正。抽样中差值小于0的比例仅为重复样本描述，不是模型优胜的后验概率。时间块保留部分局部依赖，不能覆盖全部研究选择或未知危机；核心评分仅826日、33阳性行对应11个重叠区间。</p>
<img src='../{rel}/prediction_uncertainty.png' alt='配对预测损失差与21日时间块区间'>
<p>{' '.join(uncertainty_notes)}</p><p>同一M2−M1的Q90损失差也会随21/63日块长改变区间是否跨0，不能挑选较好看的块长作为最终结论。</p>
<details><summary>全部预测指标的21/63日区间</summary>{ci_table}</details>
<p>经济抽样使用已经实现的配对净日收益、敞口、费用比例与换手，重新计算复合收益和含初始资本锚点的回撤。它条件于已拟合预测和实际历史执行；没有在每个重抽样世界重新训练/生成指令，不能称为一条可实际交易的新账本。收益差正值较高；回撤以负数记录，所以回撤差正值代表幅度较小。</p>
<details><summary>全部配对净收益和最大回撤区间</summary>{econ_ci_table}</details>
<p>CAGR、最差单日/五日、平均敞口、年化周转和平均日费用比例的区间也完整保存在CSV；没有只保留显得有利的区间。</p>
<h2>年份、季度与压力事件依赖</h2><p>全部年份/季度都保留。leave_year_out仅移除对应年的评分行后描述已保存预测损失，不重新拟合；不能冒充新的样本外试验。区间账户收益是原连续账户中的路径片段，以上一收盘净值为分母，并非每季/每个事件重置的新账户。</p>
<p>{year_note}</p>
<details><summary>全部逐年与删一年后的配对预测损失</summary>{year_table}</details>
<details><summary>核心主版本/挑战者及对照的全部季度账户路径</summary>{quarters}</details>
<p>原M2可评分阳性33行合并为以下11个重叠区间；仅为事后压力描述，不是33次独立崩盘，也不以事件定义新规则。各家族、两个挑战者在这些区间内外的预测和账户数据完整保存。</p>{event_table}
<details><summary>11个事件中的主要账户完整比较</summary>{event_paths}</details>
<h2>工程复算与数据限制</h2><p>从保存的训练期变换、样条和系数重建{audit['saved_state_prediction_rows']}行预测，最大差{audit['max_saved_state_prediction_error']:.3g}；核对{audit['saved_candidate_losses_checked']}个候选验证损失和全部训练/验证时钟。此项是保存系数重建，不是独立重新优化。</p>
<p>M3 Ridge另用加权正规方程独立求解，系数最大差{audit['m3_independent_ridge_coef_max_error']:.3g}。独立成交方程核对{audit['accounts']}个账户、{audit['account_rows']}行，最大金额差{audit['max_account_money_error']:.3g}美元。已保存bootstrap的{audit['bootstrap_scalar_draws_recomputed']}个标量抽样及{audit['intervals_checked']}个区间再次复算。</p>
<p>首次运行完成所有拟合/账户/抽样后，审计读取NumPy字符串日期时发生类型转换错误；修复仅将其显式转为Python字符串，原计算结果未改。<a href='../runs/p6/date_conversion_fix.json'>最小反例与修复记录</a>及原失败运行均保留。</p>
<p>冻结行情未重新下载，源摘要保持；历史发布时间和收盘成交代理限制延续P4/P5，P1源间差异及P2总回报口径限制保留。只使用开发段，未查看2024之后模型表现；没有账户连接、真实订单、远端上传或自动晋升。</p>
<p><a href='../{data['run_record']}'>实际运行/测试/依赖/源码与产物摘要</a> · <a href='../{rel}/independent_validation.json'>独立数值核对</a> · <a href='../runs/p6/experiment_record.json'>拟合前口径记录</a> · <a href='../{rel}/variants.json'>完整删除列与来源依赖</a> · <a href='../{rel}/skew_availability.json'>SKEW来源/可用性</a> · <a href='../STATUS.md'>当前状态</a></p>
<details><summary>预测、指标、配对区间与全部1000次抽样</summary><p>{links}</p></details>
<details><summary>全部72条逐日账户账本</summary><p>{ledger_links}</p></details>
<details><summary>每个消融/挑战者的预测、训练权重与候选分数</summary><p>{variant_links}</p></details>
<p>模型/变换与选择快照位于各variant目录的models和selections；完整文件清单及摘要在运行记录。入口：<code>.venv/bin/python -m svxylab diagnostics --open</code>，每次新建时间戳目录。</p>{commands}
<p><strong>当前停在P6待验收。下一阶段主版本仍为原M2；挑战者或某段路径较好不触发晋升。</strong></p></main></body></html>"""
    (root/'reports/diagnostics.html').write_text(doc)


def build_diagnostics_report(root,*,open_report=False):
    root=root.resolve()
    if Path(sys.prefix).resolve()!=(root/'.venv').resolve():raise ValueError('请使用项目.venv解释器')
    now=datetime.now(timezone.utc);stamp=now.strftime('%Y%m%dT%H%M%S%fZ')
    directory=root/'runs/p6'/stamp;directory.mkdir(parents=True);output=root/'data/clean/p6'/stamp
    files=sorted((root/'src').rglob('*.py'))+sorted((root/'tests').rglob('*.py'))
    files += [root/p for p in ['RESEARCH_SPEC.md','FEATURES.json','experiment.toml','pyproject.toml','requirements-lock.txt','runs/p6/experiment_record.json','runs/p5/acceptance.json']]
    hashes={p.relative_to(root).as_posix():sha256(p.read_bytes()).hexdigest() for p in files}
    write_json(directory/'started.json',{'stage':'P6','started_at':now.isoformat(),'source_sha256':hashes})
    started=perf_counter()
    try:
        result=run_diagnostics(root,output);write_json(directory/'computation.json',result)
        audit=validate(root,result);write_json(output/'independent_validation.json',audit)
        plots(root,output)
    except Exception:
        (directory/'failure.txt').write_text(traceback.format_exc());raise
    commands=[run_command(c,root,timeout=120) for c in [[sys.executable,'-m','pip','check'],
        [sys.executable,'-m','pytest','-q','--junitxml',str(directory/'pytest.xml')],['git','diff','--check'],['git','rev-parse','HEAD'],['git','remote']]]
    tests=list(ET.parse(directory/'pytest.xml').getroot().iter('testcase'))
    passing=sum(t.find('failure') is None and t.find('error') is None and t.find('skipped') is None for t in tests)
    success=bool(tests) and passing==len(tests) and all(c['exit_code']==0 for c in commands) and audit['passed']
    data={'stage':'P6','generated_at':now.isoformat(),'invocation':'.venv/bin/python -m svxylab diagnostics'+(' --open' if open_report else ''),
        'result':result,'independent_audit':audit,'elapsed_seconds':perf_counter()-started,'engineering_complete':success,
        'tests_passed':passing,'tests_failed':len(tests)-passing,'commands':commands,'source_sha256':hashes,
        'artifact_sha256':{p.relative_to(root).as_posix():sha256(p.read_bytes()).hexdigest() for p in sorted(output.rglob('*')) if p.is_file()},
        'python':sys.version,'platform':platform.platform(),'packages':{d.metadata['Name']:d.version for d in metadata.distributions()},
        'run_record':(directory/'diagnostics_run.json').relative_to(root).as_posix()}
    write_json(root/data['run_record'],data);render(root,data)
    if open_report:
        opened=run_command(['/usr/bin/open',str(root/'reports/diagnostics.html')],root)
        commands.append(opened);success=success and opened['exit_code']==0;data['engineering_complete']=success
        write_json(root/data['run_record'],data);render(root,data)
    print(f"P6：{root/'reports/diagnostics.html'}；普通pytest {passing}通过；独立审计完成。",flush=True)
    return 0 if success else 1
