"""P5本地中文报告及真实运行记录，复用项目入口与静态报告方式。"""

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

import pandas as pd

from svxylab.data_report import table
from svxylab.economics import NAMES, run_economics
from svxylab.economics_audit import validate_economics
from svxylab.environment import run_command
from svxylab.features_report import write_json


def percent(value):
    return f"{value:.2%}" if pd.notna(value) else '—'


def money(value):
    return f"{value:,.2f}" if pd.notna(value) else '—'


def points(value):
    return f"{value*100:+.2f}个百分点" if pd.notna(value) else '—'


def plots(root, output):
    os.environ.setdefault('MPLCONFIGDIR', str(root/'.cache/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font = FontProperties(fname='/System/Library/Fonts/STHeiti Light.ttc')
    for scope in ['full', 'common']:
        for policy in ['main', 'risk_only']:
            strategies = [f'{m}_{policy}' for m in ['M0','M1','M2']] + ['cash','fixed_25','fixed_50','fixed_75','fixed_100','curve']
            fig,axes = plt.subplots(2,1,figsize=(13,8),sharex=True,layout='constrained',height_ratios=[2,1])
            colors = ['#176785','#517b42','#b14440','#8b8f97','#c6ced1','#9eaab0','#77858f','#4f5963','#b48731']
            for strategy,color in zip(strategies,colors,strict=True):
                ledger = pd.read_csv(output/f'ledgers/{scope}_base_{strategy}.csv')
                dates = pd.to_datetime(ledger.as_of_session)
                for ax,field,scale in [(axes[0],'equity_end',1.),(axes[1],'drawdown',100.)]:
                    ax.plot(dates,ledger[field]*scale,label=NAMES[strategy],color=color,
                            linewidth=1.6 if strategy.startswith('M') else 1.,alpha=1 if strategy.startswith('M') else .8)
            axes[0].set_title(('完整请求日历' if scope=='full' else '共同起点重建账户')+' · '+('主映射' if policy=='main' else '仅风险容量')+' · 真实连续净账本',fontproperties=font)
            axes[0].set_ylabel('期末资产（美元）',fontproperties=font)
            axes[1].set_ylabel('收盘回撤（%）',fontproperties=font)
            axes[0].legend(prop=font,ncols=3,fontsize=9)
            for ax in axes:
                ax.grid(alpha=.25);ax.spines[['top','right']].set_visible(False)
            fig.savefig(output/f'{scope}_{policy}.png',dpi=140)
            fig.savefig(output/f'{scope}_{policy}.svg',metadata={'Date':None})
            plt.close(fig)


def metric_table(frame):
    return table(['控制器/对照','期末资产 $','净收益','CAGR','最大收盘回撤','最差单日','最差5日','平均持有敞口','现金日比例','成交额周转倍数','费用 $'],
                 [[NAMES.get(r.strategy,r.strategy),money(r.ending_equity),percent(r.total_return),percent(r.cagr),percent(r.max_drawdown),
                   f'{percent(r.worst_day)}（{r.worst_day_session}）',
                   f'{percent(r.worst_five_days)}（{r.worst_five_start} → {r.worst_five_end}）',
                   percent(r.mean_exposure),percent(r.cash_fraction),f'{r.total_turnover:.2f}',money(r.cost_dollars)] for r in frame.itertuples()])


def render(root, data):
    result = data['result']; output=root/result['output_dir']; rel=result['output_dir']
    read = lambda f: pd.read_csv(output/f)
    all_metrics=read('account_metrics.csv'); base=all_metrics.loc[all_metrics.scenario.eq('base')]
    common=base.loc[base.scope.eq('common')].set_index('strategy')
    full=base.loc[base.scope.eq('full')].set_index('strategy')
    attribution=read('attribution.csv'); matches=read('post_hoc_exposure_matches.csv'); gross=read('same_target_gross_net.csv')
    sensitivity=read('single_factor_sensitivities.csv'); risk=read('risk_capacity_diagnostics.csv')
    risk_table=[]
    for model,g in risk.groupby('model'):
        observed=g.loc[g.path_complete_in_development]
        increased=observed.loc[observed.q90_lower_and_capacity_increased]
        risk_table.append([model,len(g),int(g.q90_zero.sum()),int(g.capacity_at_cap.sum()),int(g.q90_lower_and_capacity_increased.sum()),
                           int(observed.loss_exceeds_q90.sum()),len(observed),int(observed.static_budget_exceeded.sum()),
                           int(increased.loss_exceeds_q90.sum())])
    overview=[]
    for model in ['M0','M1','M2']:
        main=common.loc[model+'_main']; risk_only=common.loc[model+'_risk_only']; reference=common.loc['M0_main']
        matched=matches.loc[matches.scope.eq('common') & matches.controller.eq(model+'_main')].iloc[0]
        overview.append(f"{model}主映射净收益{percent(main.total_return)}，最大回撤{percent(main.max_drawdown)}，平均敞口{percent(main.mean_exposure)}；"
                        f"同模型risk-only净收益{percent(risk_only.total_return)}，主映射减risk-only为{points(main.total_return-risk_only.total_return)}；"
                        f"相对M0主映射为{points(main.total_return-reference.total_return)}，相对事后敞口匹配为{points(main.total_return-matched.total_return)}。")
    cases=read('cases.csv'); case_daily=read('case_daily_ledgers.csv'); case_info=read('case_information_predictions.csv'); case_accounts=read('case_account_summary.csv')
    cases_html=[]
    for case in cases.itertuples():
        info=case_info.loc[case_info.case_id.eq(case.case_id)]
        daily=case_daily.loc[case_daily.case_id.eq(case.case_id) & case_daily.strategy.isin(['M2_main','M2_risk_only','fixed_100'])]
        info_table=table(['模型','信息日','原决策UTC','μ5','Q90','曲线A01','VIX水平','VVIX五日对数变化','SVXY五日动量','SVXY二十一日波动'],
                         [[r.model,r.as_of_session,r.simulation_decision_at,percent(r.mu5),percent(r.q90),f'{r.A01:.5f}',f'{__import__("math").exp(r.B01):.2f}',
                           percent(r.C02),percent(r.F03),percent(r.F04)] for r in info.itertuples()])
        path_table=table(['执行日','账户','本日用到的信息日','μ5','Q90','旧仓位','SVXY当日回报','目标','实际成交 $','费用 $','期末资产 $','归一化路径'],
                         [[r.as_of_session,NAMES[r.strategy],r.source_information_session,percent(getattr(r,'source_mu5',float('nan'))),
                           percent(getattr(r,'source_q90',float('nan'))),percent(r.old_weight),percent(r.asset_return),percent(r.target_weight),
                           money(r.signed_trade),money(r.cost),money(r.equity_end),f'{r.normalized_equity:.5f}'] for r in daily.itertuples()])
        summary=case_accounts.loc[case_accounts.case_id.eq(case.case_id)]
        mpath=daily.loc[daily.strategy.eq('M2_main')]
        rreturn=summary.loc[summary.strategy.eq('M2_risk_only'),'path_return_including_entry_fee'].item()
        m_info=info.loc[info.model.eq('M2')].iloc[0]
        start_capacity=min(1.,.05/m_info.q90) if m_info.q90 else 1.
        explanation=(f"入场信号μ5={percent(m_info.mu5)}，未超过0.10%固定门槛；Q90={percent(m_info.q90)}本可提供{percent(start_capacity)}风险容量，但main目标为现金。"
                     +("此后各日μ5仍未过门槛，主映射整个窗口保持现金。" if mpath.target_weight.eq(0).all()
                       else "后续μ5跨过门槛又回落，发生多次买入/卖出；具体旧仓位、目标和费用逐行列出，不能只用入场预测解释整个窗口。")
                     +f"同窗口risk-only净变化{percent(rreturn)}。"
                     +( "这次过滤减少了下跌损失；不代表过滤在全部样本中有效。" if case.kind=='favorable'
                        else "这次过滤使账户错过反弹；并不是缺预测导致无服务。"))
        summary_table=table(['账户','五日连续账户净变化（含入场换仓费）','入场相对最差收盘损失','入场后平均敞口','路径成交费 $'],
                            [[NAMES[r.strategy],percent(r.path_return_including_entry_fee),percent(r.worst_entry_relative_account_close_loss),
                              percent(r.mean_old_weight_after_entry),money(r.fees_during_path)] for r in summary.itertuples()])
        cases_html.append(f"<h3>{'有利' if case.kind=='favorable' else '不利'}案例：{case.entry} → {case.end}</h3>"
            f"<p>M2主映射路径净变化{percent(case.main_return)}，全仓对照{percent(case.fixed100_return)}，差{points(case.difference)}。"
            "从各自入场换仓前资产归一到1，只描述既有连续账户这一段；未在入场日重置真实账户。入场当天此前损益归旧仓位，所列新仓位从入场收盘之后承担风险；此后仍按每天新信息换仓。</p>"
            +f'<p>{explanation}</p>'+info_table+path_table+f'<details><summary>该案例全部12个账户的完整比较</summary>{summary_table}</details>')
    exposure=read('exposure_distribution.csv')
    exposure_table=table(['口径','账户','持有敞口区间','交易日数','比例'],[[r.scope,NAMES[r.strategy],r.bucket,r.days,percent(r.fraction)] for r in exposure.itertuples()])
    attr_table=table(['口径','诊断','左账户','右账户','复合净收益差','平均敞口差','逐日敞口贡献之和','逐日费用贡献之和'],
                     [[r.scope,r.comparison,NAMES.get(r.left,r.left),NAMES.get(r.right,r.right),points(r.compounded_total_return_difference),
                       points(r.mean_exposure_difference),points(r.arithmetic_exposure_contribution),points(r.arithmetic_cost_contribution)] for r in attribution.itertuples()])
    match_table=table(['口径','控制器','事后固定目标','匹配后实际平均敞口','控制器净收益','匹配账户净收益','匹配账户回撤'],
                      [[r.scope,NAMES[r.controller],percent(r.post_hoc_fixed_target),percent(r.mean_exposure),percent(r.controller_total_return),
                        percent(r.total_return),percent(r.max_drawdown)] for r in matches.itertuples()])
    gross_table=table(['口径','账户','净收益','基础目标不变的零费用毛收益','费用引起的复合收益差'],
                      [[r.scope,NAMES[r.strategy],percent(r.net_total_return),percent(r.same_targets_gross_return),points(r.compounded_fee_drag)] for r in gross.itertuples()])
    sense_table=table(['口径','情景','账户','起跑日','比较基准','净收益','净收益差','最大回撤','回撤差','平均敞口','费用 $'],
                      [[r.scope,r.scenario,NAMES[r.strategy],r.first_execution,r.reference_scenario,percent(r.total_return),points(r.total_return_delta_reference),
                        percent(r.max_drawdown),points(r.max_drawdown_delta_reference),percent(r.mean_exposure),money(r.cost_dollars)] for r in sensitivity.itertuples()])
    delays=all_metrics.loc[all_metrics.scenario.eq('delay_aligned_base')]
    rebound=read('rebound_windows.csv')
    rebound=pd.concat([g.nlargest(6,'return_shortfall') for _,g in rebound.groupby('scope')])
    rebound_table=table(['口径','窗口入场','窗口终点','标的反弹','M2主映射','全仓对照','少获得的回报','后续平均敞口','整段模型无服务'],
                        [[r.scope,r.entry,r.end,percent(r.asset_return),percent(r.main_return),percent(r.full100_return),points(r.return_shortfall),
                          percent(r.mean_exposure_after_entry),'是' if r.startup_unserved_entire_window else '否'] for r in rebound.itertuples()])
    opportunity=table(['口径','账户','现金日数','现金时遇上涨日','少获得的上涨日贡献之和','减少的下跌日损失之和'],
                      [[r.scope,NAMES[r.strategy],r.cash_days,r.cash_on_asset_up_days,percent(r.missed_positive_day_arithmetic_return),
                        percent(r.avoided_negative_day_arithmetic_loss)] for r in base.itertuples()])
    tails=read('last_five_2023_execution_days.csv')
    tail_table=table(['执行日','账户','信息日','μ5','Q90','旧仓位','本日持有损益 $','收盘新仓位','成交费 $'],
                     [[r.as_of_session,NAMES[r.strategy],r.source_information_session,percent(r.source_mu5),percent(r.source_q90),percent(r.old_weight),
                       money(r.holding_pnl),percent(r.new_weight),money(r.cost)] for r in tails.loc[tails.scope.eq('common')].itertuples()])
    ledger_links=' · '.join(f"<a href='../{rel}/{r.ledger_file}'>{escape(r.scope+' '+NAMES[r.strategy])}</a>" for r in base.itertuples())
    other_links=' · '.join(f"<a href='../{rel}/{p.name}'>{p.name}</a>" for p in sorted(output.glob('*.csv')))
    commands=''.join(f"<details><summary>退出{r['exit_code']} · {escape(r['command'])}</summary><pre>{escape(r['output'])}</pre></details>" for r in data['commands'])
    audit=data['independent_audit']
    doc=f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>
<title>SVXYLab · P5 经济映射与连续净账本</title><style>
body{{margin:0;background:#fff;color:#22313d;font:16px/1.7 -apple-system,BlinkMacSystemFont,'PingFang SC',sans-serif}}main{{max-width:1400px;margin:auto;padding:28px 24px 70px}}
h1{{font-size:28px}}h2{{font-size:22px;margin-top:34px}}h3{{font-size:18px}}a{{color:#176785}}.note{{background:#f2f6f8;border-left:4px solid #537587;padding:12px 18px}}
.scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{border:1px solid #d4dce2;padding:7px;vertical-align:top;text-align:left}}th{{background:#f3f6f8}}
img{{max-width:100%;height:auto}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}}details{{padding:9px 0;border-bottom:1px solid #ddd}}summary{{cursor:pointer}}
</style></head><body><main><h1>SVXYLab · P5 经济映射与连续净账本</h1><p>{data['generated_at']} · RESEARCH_ONLY · P4验收检查点 {result['p4_accepted_checkpoint'][:7]}</p>
<div class='note'><p><strong>工程执行：</strong>{'完成' if data['engineering_complete'] else '未完成'}。普通pytest {data['tests_passed']}项通过，独立核对{audit['accounts_checked']}条账户、{audit['ledger_rows_checked']}行。
只使用已验收P4预测；没有重新训练、调参或按经济结果晋升模型。</p>
<p><strong>经济研究结果：</strong>共同起点M2主映射净收益{percent(common.loc['M2_main','total_return'])}、最大回撤{percent(common.loc['M2_main','max_drawdown'])}；
全仓对照净收益{percent(common.loc['fixed_100','total_return'])}、最大回撤{percent(common.loc['fixed_100','max_drawdown'])}。
平均敞口分别为{percent(common.loc['M2_main','mean_exposure'])}、{percent(common.loc['fixed_100','mean_exposure'])}。收益、风险和敞口须一起比较，单个优劣不证明模型预测有效。</p>
<p><strong>真实覆盖：</strong>完整请求{result['requested_first_execution']}至{result['last_account_close']}，模型前175个请求日无预测；共同起点{result['base_common_first_execution']}。
没有读取2024以后的收益补尾段。P4负预测结果原样保留。</p></div>
<h2>固定映射与真实执行</h2><p>w_risk=1（Q90=0），否则min(1,B/Q90)；main仅在μ5&gt;2c时采用w_risk，否则为现金。基础B=.05、单边c=.0005。
M0/M1/M2各有main与risk-only，共6条控制器，另有现金、固定25/50/75/100%和F2&gt;F1曲线对照。<strong>P10不进入仓位公式，经济结果不能证明P10有效。</strong></p>
<p>信号t只在下一股票交易日收盘执行；旧份额先承担当日真实价格/公司行动损益。成交费用按实际成交金额逐笔计算，换仓后达到目标比例。
缺预测时不发新目标，保留实际份额和现金，照记每日损益；不前填仓位自动再平衡，不叠加五日标签交易。最后收盘按市值计价，不强制卖出。</p>
<h2>两种账户起点，分别重建</h2><p>所有账户研究资本均为100,000美元。完整请求日历的现金锚点为2019-12-31，首次执行2020-01-02；共同起点的锚点为2020-09-10、首次执行2020-09-11。
锚点仅支持第一笔真实收盘执行，不取得此前账户收益。共同起点是从现金重跑，绝非从已运行净值中间截取再缩放。</p>
<p>完整请求前175日模型未提供服务，初始现金不能称为成功预警疫情。同期全仓对照净值经历真实涨跌；表内完整起点收益包含这一覆盖差异。</p>
<h3>完整请求日历</h3>{metric_table(base.loc[base.scope.eq('full')])}
<img src='../{rel}/full_main.png' alt='完整请求日历主映射净值与回撤'><details><summary>完整请求日历risk-only净值与回撤</summary><img src='../{rel}/full_risk_only.png' alt='完整请求risk-only'></details>
<h3>共同可交易起点</h3>{metric_table(base.loc[base.scope.eq('common')])}
<img src='../{rel}/common_main.png' alt='共同起点主映射净值与回撤'><details><summary>共同起点risk-only净值与回撤</summary><img src='../{rel}/common_risk_only.png' alt='共同起点risk-only'></details>
<p>最大回撤含初始资本锚点及第一笔交易费用；最差五日是连续5个账户日收益复合。平均敞口使用承担本日收益的旧仓位，第一执行日收盘前仍为现金；并另存收盘敞口。
周转倍数为每日实际成交额/换仓前资产之和，双向买卖均计；费用已包含在净收益，不再扣基金管理费。</p>
<h2>模型增量、敞口与μ5过滤</h2>{''.join('<p>'+v+'</p>' for v in overview)}
<p class='note'>本段的μ5过滤拖累三个模型的总收益；M2主映射比risk-only少41.95个百分点，且回撤更大。
M2主映射相对事后同敞口账户多2.59个百分点，但最大回撤32.98%对17.66%；M2 risk-only反而比同敞口账户少5.24个百分点。
这不足以据此宣布稳定的模型信息增量；部分经济差异由承担的敞口不同解释。P4没有显示主要预测损失优势，因此也不能先认定“预测有用、只是翻译无用”。</p>
<p>这些是开发段经济描述。M1/M2相对同映射M0的差异可能来自敞口和交易时机；P4主要预测损失未优于M0的结论不因此改写。
main与risk-only的差异隔离既定μ5过滤的经济效果；是否拖累或改善按下表真实结果判断，不临时删除较差控制器。</p>{attr_table}
<p>复合净收益差是完整账户终值之差。逐日敞口贡献与费用贡献只是可精确相加的<strong>日收益差算术分解</strong>，两者之和等于逐日收益差之和，不能当作复合净收益差。
这不提供因果权重、显著性或P6消融结论。</p>
<h3>事后平均敞口匹配诊断</h3>{match_table}<p>固定目标由整段控制器实际平均旧仓位推得，并从同一资本/时点重跑，实际平均敞口匹配误差&lt;1e-12。
匹配使用了整段未来账户信息，仅为<strong>事后诊断</strong>，不是当年可执行基准。不能把单纯加减敞口带来的收益/回撤变化称为择时优势。</p>
<details><summary>全部基础账户的敞口分布</summary>{exposure_table}</details>
<h2>Q90低估与风险容量</h2>{table(['模型','有预测日','Q90=0日','风险容量=100%日','Q90下降且容量增加日','随后L5超过Q90日','可核对路径','静态容量×L5超过B日','增仓且后来低估日'],risk_table)}
<p>以上L5只从2023内完整的未来五个收盘计算，纯属事后诊断，不反馈仓位。固定B下，Q90下降会提高风险容量，直到100%上限；若Q90低估，容量会偏大。
“静态容量×L5”只检验当时容量与随后标的路径，实际账户仍逐日更新、扣费，不能混为同一个损失。
<strong>Q90不是最大损失保证，B=5%不是账户最大回撤5%。</strong></p>
<p>例如信息日2021-01-19的M2 Q90为4.98%，较前次下降，风险容量升至100%；μ5为0.1954%，也通过主映射门槛，所以2021-01-20收盘目标为100%。
随后到2021-01-27的标的入场相对最差收盘损失为12.43%，超过当时Q90与5%静态预算。实际账户随后仍会随每日预测换仓，这12.43%不能直接当作连续账户损失。</p>
<h2>交易费用与预定单因素敏感性</h2><p>仅运行成本0/5/15/30bp、预算2.5/5/10%、额外1股票交易日延迟；不做全部组合，不搜索赢家。
成本情景按固定公式同时更新2c过滤阈值与实际费率，因此另给基础目标保持不变的零费用毛账本，单独衡量费用拖累。</p>{gross_table}
<p>额外延迟将已保存整条预测/目标推迟1个股票交易日使用，不重新拟合。完整请求起点仍为2020-01-02。
延迟后的共同可交易起点为2020-09-14；为隔离延迟，基础对照也从2020-09-14现金重建（delay_aligned_base），下表延迟差异用这个同日起点比较。
年底尚未到延迟执行日的指令不在2023成交，不读取2024补算。</p>
<p><a href='../{rel}/pending_delayed_orders.csv'>年底尚未执行的12个延迟目标</a>仅保存已知指令与未来日历时间，不含2024成交或收益。</p>
<p>共同起点M2主映射保持基础目标的毛收益为118.23%，扣真实成交费用后104.33%，相差13.90个百分点。
成本情景从0bp至30bp的净收益为127.44%、104.33%、74.57%、7.85%；它同时改变过滤门槛，不能全解释为手续费。
额外延迟1日后M2主映射净收益40.83%，比同日起跑基础账户少63.50个百分点；risk-only为147.75%，相对同日起跑基础多3.69个百分点。延迟影响对两个控制器不同，均保留。</p>
<details><summary>全部144行单因素敏感性及配对差异</summary>{sense_table}</details>
<details><summary>延迟比较的同日起跑基础账户</summary>{metric_table(delays)}</details>
<h2>现金时间与错失反弹</h2>{opportunity}<p>上涨日少获得的贡献=sum((1−旧仓位)×正的标的日收益)，减少的下跌贡献同理。
这是算术机会诊断，不是另一条净值或复合绩效；完整日历中的无服务阶段单独识别。</p>{rebound_table}
<p>表中为M2主映射少获得回报最多的正向五日窗口，窗口可以重叠，不能当成多笔独立交易相加。全部窗口和前10个上涨日见CSV。</p>
<h2>最有利、最不利各两个完整案例</h2><p>事后按预定主版本M2 main相对固定100%的5日连续账户净变化差选择，各取最大/最小两个，排除所选四个之间的重叠。
排名在本次经济计算前记录；它们是解释案例，不是新的入场规则，也不代表独立事件样本。</p>{''.join(cases_html)}
<h2>2023最后5个有预测执行日</h2>{tail_table}<p>对应P4跨2024标签的5个预测日，预测仍可在2023实际执行；交易从未依赖score_observed。
当日损益由旧份额承担，新仓位影响随后仍在2023内的路径；最后收盘后的2024损益保持未评价。</p>
<h2>工程核对、数据/PIT限制与阶段状态</h2><p>直接从已验收预测和价格另行求解买卖成交方程，逐日重建{audit['accounts_checked']}条账户、{audit['ledger_rows_checked']}行，
现金/股值/成交/净资产最大差{audit['max_cash_equity_trade_error_dollars']:.3g}美元。模型、冻结预测摘要保持不变；无封存收益、P10、标签成熟/可评分状态进入交易。</p>
<p>历史精确发布时间仍未知，继续使用ASSUMED_NEXT_SESSION / NO_FULL_HISTORICAL_PIT_CLAIM；延迟敏感性不能补成历史精确PIT。
P1源间差异、P2总回报口径限制保留。无盘中路径、交易账户、真实订单、现金利息或杠杆；交易成本是统一代理，研究持仓非投资建议。</p>
<p>当前仅交付P5。P6消融、两个挑战者和时间块不确定性尚未执行；没有消耗2024之后封存评价，也没有自动晋升任何模型。</p>
<p><a href='../{data['run_record']}'>本次实际运行/测试/源码及产物摘要</a> · <a href='../{rel}/independent_validation.json'>独立逐笔审计</a> ·
<a href='../runs/p4/acceptance.json'>P4验收与预测摘要</a> · <a href='../runs/p5/experiment_record.json'>经济计算前口径记录</a> · <a href='../STATUS.md'>STATUS</a></p>
<h3>基础逐日账本（含实际份额、现金、费用与信号来源）</h3><p>{ledger_links}</p><details><summary>全部指标、敏感性、案例与交易CSV</summary><p>{other_links}</p></details>
<p>实际入口：<code>.venv/bin/python -m svxylab economics --open</code>。每次生成新时间戳目录，旧产物保留。Python {escape(data['python'])}。</p>{commands}
<p><strong>工程完成不要求跑赢简单对照。停在P5，等待用户验收。</strong></p></main></body></html>"""
    (root/'reports/economics.html').write_text(doc)


def build_economics_report(root, *, open_report=False):
    root=root.resolve()
    if Path(sys.prefix).resolve() != (root/'.venv').resolve():
        raise ValueError('请使用项目解释器.venv/bin/python')
    now=datetime.now(timezone.utc); stamp=now.strftime('%Y%m%dT%H%M%S%fZ')
    run_dir=root/'runs/p5'/stamp;run_dir.mkdir(parents=True)
    output=root/'data/clean/p5'/stamp
    sources=sorted((root/'src').rglob('*.py'))+sorted((root/'tests').rglob('*.py'))
    sources += [root/p for p in ['experiment.toml','FEATURES.json','RESEARCH_SPEC.md','pyproject.toml','requirements-lock.txt','runs/p4/acceptance.json','runs/p5/experiment_record.json']]
    hashes={p.relative_to(root).as_posix():sha256(p.read_bytes()).hexdigest() for p in sources}
    write_json(run_dir/'started.json',{'stage':'P5','started_at':now.isoformat(),'source_sha256':hashes})
    start=perf_counter()
    try:
        result=run_economics(root,output)
        independent=validate_economics(root,result)
        write_json(output/'independent_validation.json',independent)
        result['compute_and_audit_seconds']=perf_counter()-start
        write_json(run_dir/'computation.json',result)
        plots(root,output)
    except Exception:
        (run_dir/'failure.txt').write_text(traceback.format_exc())
        raise
    commands=[run_command(c,root,timeout=120) for c in [[sys.executable,'-m','pip','check'],
        [sys.executable,'-m','pytest','-q','--junitxml',str(run_dir/'pytest.xml')],['git','diff','--check'],['git','rev-parse','HEAD'],['git','remote']]]
    cases=[]
    for case in ET.parse(run_dir/'pytest.xml').getroot().iter('testcase'):
        passed=case.find('failure') is None and case.find('error') is None and case.find('skipped') is None
        cases.append({'name':case.get('name'),'passed':passed})
    passed=bool(cases) and all(r['passed'] for r in cases) and all(c['exit_code']==0 for c in commands) and independent['passed']
    data={'stage':'P5','generated_at':now.isoformat(),'invocation':'.venv/bin/python -m svxylab economics'+(' --open' if open_report else ''),
          'result':result,'independent_audit':independent,'engineering_complete':passed,'tests_passed':sum(r['passed'] for r in cases),
          'tests_failed':sum(not r['passed'] for r in cases),'test_cases':cases,'commands':commands,'source_sha256':hashes,
          'artifact_sha256':{p.relative_to(root).as_posix():sha256(p.read_bytes()).hexdigest() for p in sorted(output.rglob('*')) if p.is_file()},
          'python':sys.version,'platform':platform.platform(),'packages':{d.metadata['Name']:d.version for d in metadata.distributions()},
          'run_record':(run_dir/'economics_run.json').relative_to(root).as_posix()}
    write_json(root/data['run_record'],data);render(root,data)
    if open_report:
        opened=run_command(['/usr/bin/open',str(root/'reports/economics.html')],root)
        commands.append(opened);passed=passed and opened['exit_code']==0;data['engineering_complete']=passed
        write_json(root/data['run_record'],data);render(root,data)
    print(f"P5：{root/'reports/economics.html'}；普通pytest {data['tests_passed']}通过；独立账本{independent['accounts_checked']}条。",flush=True)
    return 0 if passed else 1
