"""统一中文交付：数据、真实前向预报、冻结研究；仅本地静态文件。"""
from datetime import datetime, timezone
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
from svxylab.diagnostics_report import plots, account_table, name, number
from svxylab.economics_report import percent, money
from svxylab.environment import run_command
from svxylab.features_report import write_json
from svxylab.release import digest, read_json, source_files, verify_release, run_release
from svxylab.release_audit import validate_release


def dataframe_table(frame, columns=None):
    frame=frame if columns is None else frame[columns]
    def fmt(v):
        if isinstance(v,(float,np.floating)):return number(v)
        return str(v) if pd.notna(v) else '—'
    return table(list(frame.columns),[[fmt(v) for v in r] for r in frame.itertuples(index=False,name=None)])


def artifact_links(output, root):
    paths=[p for p in sorted(output.rglob('*')) if p.is_file()]
    return ''.join(f'<li><a href="../{escape(str(p.relative_to(root)))}">{escape(str(p.relative_to(output)))}</a></li>' for p in paths)


def render_latest(root):
    release=read_json(root/read_json(root/'runs/p7/latest_run.json')['run_record'])
    result=release['result'];output=root/result['output_dir'];rel=result['output_dir']
    daily=read_json(root/read_json(root/'runs/p7/latest_daily.json')['run_record']) if (root/'runs/p7/latest_daily.json').exists() else None
    now=datetime.now(timezone.utc).isoformat();forecast='<p>尚未生成前向记录。历史回放不冒充当时发出的预报。</p>'
    if daily:
        status=daily['status'];forecast=f'<p>日报状态：<strong>{escape(status)}</strong>；本次检查 {escape(daily["started_at"])}。</p>'
        if 'forecast_file' in daily:
            issued=read_json(root/daily['forecast_file']);t=issued['timing']
            models=read_json(root/issued['model_state_file'])
            rows=[]
            for p in issued['predictions']:
                fit=models[p['model']]['fit']
                rows.append([name(p['model']),percent(p['mu5']),percent(p['q90']),percent(p['p10']),
                    percent(p['main']),percent(p['risk_only']),fit['training_rows'],fit['positive_count'],fit['event_clusters'],
                    fit['fit_session'],', '.join(p['outside_training_range']) or '无'])
            forecast+=f'''<p>真实生成时间 <strong>{issued['created_at_utc']}</strong>；数据已取得至 {issued['inputs']['actual_received_at_max']}。
信息日 <strong>{t['as_of_session']}</strong>，纽约决策截止 <strong>{t['decision_at_new_york']}</strong>，计划执行 <strong>{t['execution_at']}</strong>（UTC）。
第5个未来收盘/标签成熟 <strong>{t['label_matures_at']}</strong>（UTC）。重复运行保留首次记录及时间。</p>'''
            forecast+=table(['模型','μ5','Q90(L5)','P10','main参考','risk-only参考','训练行','阳性训练行','重叠事件簇','拟合执行月起点','超出训练范围的列'],rows)
            features=issued['core_features'];forecast+=f'<details><summary>这条前向预报的19项原始特征</summary>{dataframe_table(pd.DataFrame([features]))}</details>'
            forecast+=f'<p><a href="../{daily["forecast_file"]}">不可回写的原始前向记录</a> · <a href="../{issued["model_state_file"]}">所用模型/变换</a> · <a href="../{daily["run_record"]}">日报实际运行</a>。R5/L5/Y10在原预测记录中均为空；成熟后只在单独结果记录中追加。</p>'
        else:forecast+=f'<p>本次没有新预测或目标仓位：{escape(daily.get("error",status))}。研究模拟保持已有实际份额和现金，继续计算持有损益；不以旧目标自动再平衡。</p>'
    metrics=pd.read_csv(output/'prediction_metrics.csv');paired=pd.read_csv(output/'paired_prediction_periods.csv')
    accounts=pd.read_csv(output/'account_metrics.csv');ci=pd.read_csv(output/'bootstrap_intervals.csv')
    base=metrics.loc[metrics.period_type.eq('all')]
    rows=[[name(r.model),r.rows,r.events,number(r.mse),number(r.log_loss),number(r.pinball),number(r.brier),percent(r.q90_coverage)] for r in base.itertuples()]
    prediction_table=table(['模型','评分日','阳性行','μ5 MSE','P10对数损失','Q90 Pinball','P10 Brier','Q90覆盖'],rows)
    pairs=paired.loc[paired.period_type.eq('all') & paired.metric.isin(['mse','log_loss','pinball'])]
    paired_table=table(['模型','对照','头/损失','配对日','损失差','相对变化'],[[name(r.left),name(r.right),r.metric,r.rows,number(r.difference),percent(r.relative_change)] for r in pairs.itertuples()])
    core=accounts.loc[accounts.cohort.eq('core_common') & accounts.strategy.isin([f'{m}_{p}' for m in ['M0','M1','M2','M3_RECENCY'] for p in ['main','risk_only']]+['cash','fixed_25','fixed_50','fixed_75','fixed_100','curve'])]
    skew=accounts.loc[accounts.cohort.eq('skew_common')]
    s=output/'sensitivity';sensitivity=pd.read_csv(s/'account_metrics.csv')
    cases=pd.read_csv(s/'cases.csv');caseinfo=pd.read_csv(s/'case_information_predictions.csv');casepaths=pd.read_csv(s/'case_account_summary.csv')
    case_html=''
    for c in cases.itertuples():
        ps=caseinfo.loc[caseinfo.case_id.eq(c.case_id)];path=casepaths.loc[casepaths.case_id.eq(c.case_id)]
        p=ps.loc[ps.model.eq('M2')].iloc[0]
        case_html+=f'''<details><summary>{'最有利' if c.kind=='favorable' else '最不利'}案例 {c.case_id}：{c.entry} → {c.end}，M2 main相对全仓路径差 {percent(c.difference)}</summary>
<p>已知信息日 {p.as_of_session}，决策 {p.simulation_decision_at}，实际执行代理 {p.execution_at}；μ5={percent(p.mu5)}、Q90={percent(p.q90)}、P10={percent(p.p10)}。
当时输入及三个模型预测完整列在下表；日内最大损失未知。路径收益包含入场当笔费用和之后真实连续账户变仓，非五日标签叠加。</p>
{dataframe_table(ps,['model','as_of_session','mu5','q90','p10','A01','A02','B01','B02','C01','C02','D02','E01','F01','F03'])}
{dataframe_table(path,['strategy','entry_new_weight','path_return_including_entry_fee','worst_entry_relative_account_close_loss','mean_old_weight_after_entry','fees_during_path'])}
<p><a href='../{rel}/sensitivity/case_daily_ledgers.csv'>这四例的全部逐日账本</a> · <a href='../{rel}/sensitivity/case_information_predictions.csv'>完整当时信息</a></p></details>'''
    risk=pd.read_csv(s/'risk_capacity_diagnostics.csv');risk_summary=[]
    for model,g in risk.groupby('model'):
        observed=g.loc[g.path_complete_in_historical_evaluation]
        risk_summary.append([model,len(g),len(observed),int(g.q90_zero.sum()),int(g.capacity_at_cap.sum()),int(g.q90_lower_and_capacity_increased.sum()),int(observed.loss_exceeds_q90.sum()),int(observed.static_budget_exceeded.sum())])
    events=read_json(output/'positive_events.json');event_table=table(['区间','开始','结束','阳性行'],[[e['event_id'],e['start'],e['end'],e['positive_rows']] for e in events])
    a=release['independent_audit'];c=result['core_run'];end=c['processed_decision_last']
    primary=pairs.loc[pairs.left.eq('M2') & pairs.right.eq('M0')]
    primary_note='；'.join(f'{r.metric} {percent(r.relative_change)}' for r in primary.itertuples())
    ci_main=ci.loc[ci.domain.eq('prediction') & ci.left.eq('M2') & ci.right.eq('M0') & ci.metric.isin(['mse','log_loss','pinball'])]
    main=core.loc[core.strategy.eq('M2_main')].iloc[0];full=core.loc[core.strategy.eq('fixed_100')].iloc[0]
    doc=f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SVXYLab · 数据、预报与研究</title>
<style>body{{margin:0;background:#f1f4f5;color:#1f3039;font:16px/1.7 -apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif}}main{{max-width:1360px;margin:auto;padding:32px}}h1{{font-size:32px}}h2{{margin-top:44px;border-top:2px solid #d4dfe1;padding-top:22px}}a{{color:#166675}}table{{border-collapse:collapse;display:block;overflow-x:auto;background:white;font-size:13px;margin:18px 0}}td,th{{text-align:right;border-bottom:1px solid #dce2e5;padding:8px 12px;white-space:nowrap}}th{{background:#e4eeee}}td:first-child,th:first-child{{text-align:left}}details{{margin:18px 0;padding:14px;background:#fff;border:1px solid #dce2e5}}summary{{cursor:pointer;font-weight:600}}img{{max-width:100%}}.notice{{padding:20px;background:#fff0d9;border-left:5px solid #a77d31}}.nav{{display:flex;gap:24px;flex-wrap:wrap}}code{{overflow-wrap:anywhere}}li{{overflow-wrap:anywhere}}</style></head><body><main>
<h1>SVXYLab · 数据、预报与研究</h1><p>页面生成 {now} · 本机日频研究 · <strong>RESEARCH_ONLY</strong></p>
<nav class="nav"><a href="#data">数据覆盖</a><a href="#forecast">前向预报</a><a href="#research">历史研究</a><a href="#audit">运行与复算</a></nav>
<div class="notice">主版本仍为原 M2；所有预定对照、消融、挑战者均保留。不连接账户、不下单、不自动晋升。Q90不是最大损失保证，B=.05不是最大回撤5%。交付验收与投资有效性是两件事。</div>
<h2 id="data">一、真实数据与限制</h2>
<p>P1原始冻结范围 2019-01-01 → <strong>{end}</strong>，前置特征/训练也统一从2019开始。开发研究为2020–2023；本页单独报告2024年至冻结日，准确名称是<strong>“本轮未参与选择的历史评估”</strong>，不称绝对未见。月度滚动可按当时成熟条件消化之前的结果。</p>
<p>真实Yahoo Chart SVXY/SPY份额价格、分红/拆分总回报；Cboe指数及真实标准月度VX结算，按原特征字典计算。P1冻结原件/摘要保持，日报的新下载独立保存，不能默认为公开再分发许可。行情与派生CSV仅本地审查，未推送GitHub；供应商复权参考和P1/P2已报告的源间差异仍保留。</p>
<p>历史缺乏完整发布时间与历史版本库，沿用 ASSUMED_NEXT_SESSION 和 NO_FULL_HISTORICAL_PIT_CLAIM。下一交易日收盘是成本代理，非保证成交价；费用按成交额逐笔扣除，现金收益0。SPY/VIX/VX的更新时间差异与SVXY目标倍数、路径和跟踪误差限制仍适用。</p>
<p><a href="data.html">P1数据</a> · <a href="baselines.html">P2时钟/账本</a> · <a href="features.html">P3特征</a> · <a href="../FEATURES.json">特征字典</a> · <a href="../experiment.toml">唯一配置</a> · <a href="../runs/p7/freeze.json">揭示前冻结版本</a></p>
<h2 id="forecast">二、真实前向预报</h2>{forecast}
<p><strong>μ5</strong>：下一交易日收盘入场后未来5交易日累计收益的预测均值；<strong>Q90(L5)</strong>：同一入场点至未来5个日收盘最大入场相对损失的条件90%分位；<strong>P10</strong>：该路径损失至少10%的预测概率。训练阳性行重叠，不是独立危机次数；概率不等于历史排名。</p>
<p>研究参考：Q90=0时w_risk=1，否则min(1,B/Q90)；main仅在μ5&gt;2c时采用w_risk，否则现金。B=.05，单边c=.0005；risk-only不做均值过滤。P10从未进入仓位公式。前向运行几天或20天只能验证流程，不能证明收益优势。</p>
<p>双击项目里的 <a href="../更新日报.command">更新日报.command</a>，只检查新数据、复用本月模型或在合资格的新月份按冻结程序拟合，不重跑历史账户。关闭市场时按真实交易日历定位下一次执行。缺数据、过时或过了截止则无新目标；重开页面要检查时间，静态页不会自动刷新。</p>
<h2 id="research">三、固定历史评估结果</h2>
<p><strong>{c['requested_decision_first']} → {end}</strong>：请求{c['request_sessions']}个决策日，核心有预测{c['forecast_sessions']}日，无服务{c['no_forecast_sessions']}日；{c['monthly_fit_cycles']}个月度拟合周期、{c['selection_cycles']}次年度选择。最后未成熟标签保持空，但可执行预测继续计入截至冻结日的实际持有损益。</p>
<p>所有账户在各自共同起点以10万美元现金重新起跑，保留前一交易日信息；完整请求口径另行全量保存，不截取运行中账户冒充同资本起点。开发段前175个无服务决策日仍属无服务，不能归功为疫情预警。<a href="predictions.html">已验收P4预测/负结果</a> · <a href="economics.html">已验收P5</a> · <a href="diagnostics.html">已验收P6</a>。</p>
<p>当前段M2相对M0平均损失：<strong>{primary_note}</strong>。M2 main净收益{percent(main.total_return)}、最大收盘回撤{percent(main.max_drawdown)}；同日起跑固定100%为{percent(full.total_return)}、{percent(full.max_drawdown)}。这些是各自完整数值；经济路径优劣不能证明P10有效。</p>
{prediction_table}<details><summary>全部14项预定配对预测比较</summary>{paired_table}</details>
<p>负损失差表示左侧较低；家族删除只是列/表示方式诊断，不能声称移除了整个VIX/SPY信息来源。M3仅对三个训练目标使用504日半衰期权重，变换仍无权；SKEW另与相同训练/验证样本的核心M2配对，避免把样本变化混称新增信息。</p>
<img src="../{rel}/prediction_uncertainty.png" alt="各预定配对预测损失差及21日时间块区间">
<details><summary>M2相对M0：21/63交易日块的全部主要损失区间</summary>{dataframe_table(ci_main,['metric','block_sessions','paired_rows','point_difference','lower_95','upper_95'])}</details>
<p>21/63日时间块各1000次，沿完整交易日日历配对，缺失日不压缩成相邻日；逐项95%百分位描述区间未做多重选择修正，也不是投资获利概率。经济区间条件于已拟合预测/实际路径，重新复合抽到的日收益，不是重新训练的一条可执行账本。全部抽样、指标及较差区间保留。</p>
<h3>相同资本/执行起点的原六条控制器与简单对照</h3>{account_table(core)}
<img src="../{rel}/core_common_equity.png" alt="核心共同起点main与risk-only连续净值">
<details><summary>SKEW覆盖相同起点的完整对照</summary>{account_table(skew)}<img src="../{rel}/skew_common_equity.png" alt="SKEW相同起点账户"></details>
<details><summary>完整请求、核心共同、SKEW共同的全部账户</summary>{account_table(accounts)}</details>
<details><summary>每年/季度/事件的连续账户路径</summary>{dataframe_table(pd.read_csv(output/'account_periods.csv'))}</details>
<h3>敞口、均值过滤、Q90容量与摩擦</h3>
<p>main减risk-only对照分离μ5过滤的影响；原模型对M0配对损失衡量状态信息增量。固定25/50/75/100%对照用于识别单纯敞口大小的影响。事后用平均敞口匹配固定仓位只作追加描述，选这个仓位用了整段信息，不能作为事前可执行基准。</p>
<details><summary>μ5过滤：实际日收益的敞口项与费用项（不是复合收益的可加分解）</summary>{dataframe_table(pd.read_csv(s/'mu_filter_attribution.csv'))}</details>
<details><summary>完整敞口分布、换手、成本、现金时间、错失上涨</summary>{dataframe_table(accounts)}</details>
{table(['模型','预测日','完整后续路径','Q90=0日','风险容量100%日','Q90降低且容量增加日','后续损失超过Q90行','静态容量×损失超过B行'],risk_summary)}
<p>Q90降低会提高研究仓位，若随后风险被低估会放大暴露。表内静态容量×五日损失只是风险诊断，实际账户每天可换仓，与完整最大回撤不同。<a href='../{rel}/sensitivity/risk_capacity_diagnostics.csv'>每个原始日期和后续路径</a>。</p>
<details><summary>原定单因素敏感性：成本、预算、额外一交易日延迟</summary>{dataframe_table(sensitivity,['cohort','strategy','scenario','budget','cost_rate','delay','first_execution','total_return','max_drawdown','worst_day','worst_five_days','mean_exposure','cost_dollars'])}</details>
<p>只有原规格的单因素，没有扫描新阈值或全组合找赢家。成本敏感性同时使用该成本的2c均值门槛与实际逐笔费用；下面固定同一目标的毛/净对照单独识别摩擦。延迟仍由旧实际仓位承担换仓前损益。</p>
<details><summary>同目标毛/净与事后敞口匹配</summary>{dataframe_table(pd.read_csv(s/'same_target_gross_net.csv'))}{dataframe_table(pd.read_csv(s/'post_hoc_exposure_matches.csv'))}</details>
<h3>预定最有利、最不利各两个完整案例</h3><p>按P5既定五日账户路径差排序，剔除相互重叠窗口，事后选择用于解释；不改变训练、权重或起止规则。</p>{case_html}
<details><summary>全部阳性行合并的重叠事件区间</summary>{event_table}</details>
<h2 id="audit">工程执行、复算及交付边界</h2>
<p>工程状态：{'完成' if release['engineering_complete'] else '未完成'}。普通pytest <strong>{release['tests_passed']}通过、{release['tests_failed']}失败</strong>。核心保存系数重建{a['core_saved_state_rows']}行，最大误差{a['core_saved_state_max_error']:.3g}；变体{a['saved_state_prediction_rows']}行，最大误差{a['max_saved_state_prediction_error']:.3g}。候选验证损失复核{a['core_saved_candidate_losses']+a['saved_candidate_losses_checked']}个。
逐笔独立方程核对{a['accounts']}个账户、{a['account_rows']}行，最大金额误差{a['max_account_money_error']:.3g}美元；复算{a['bootstrap_scalar_draws_recomputed']}个bootstrap标量抽样和{a['intervals_checked']}个区间。</p>
<p>M2另按原训练输入独立走同一冻结拟合流程作参照，最大预测差{result['m2_refit_reference_max_error']:.3g}。它是从输入重新拟合；上述保存系数重建没有再次优化全部模型，二者不混称。M3均值头另用加权正规方程校验。</p>
<p>双击 <a href='../重新运行研究.command'>重新运行研究.command</a> 可按同一冻结版本重新计算当前历史评估，输出新目录保留旧产物。源码/依赖、配置、预测、日损失、候选分数、选择、训练/验证成员、变换/模型、全部账本与测试记录均在本地；报告通过相对路径链接。</p>
<p><a href='../{release['run_record']}'>本次实际运行、测试、依赖及全部SHA摘要</a> · <a href='../{rel}/independent_validation.json'>独立数值审计</a> · <a href='../runs/p7/experiment_record.json'>揭示前口径</a> · <a href='../runs/p7/freeze.json'>源码冻结清单</a> · <a href='../runs/p6/accepted_checkpoint.json'>P6本地验收检查点</a> · <a href='../STATUS.md'>当前进度</a></p>
<details><summary>全部本地研究产物、逐日账本、预测及模型快照</summary><ul>{artifact_links(output,root)}</ul></details>
<p><strong>停在P7等待最后验收。</strong>原开发段负结果保持。没有要求跑赢M0；没有按本页结果新增规则、改模型或自动晋升。</p>
</main></body></html>'''
    (root/'reports/latest.html').write_text(doc)


def build_release(root, *, open_report=False):
    root=root.resolve()
    if Path(sys.prefix).resolve()!=(root/'.venv').resolve():raise ValueError('请使用项目.venv')
    freeze=verify_release(root);now=datetime.now(timezone.utc);stamp=now.strftime('%Y%m%dT%H%M%S%fZ')
    directory=root/'runs/p7'/stamp;directory.mkdir(parents=True)
    output=root/'data/clean/p7'/stamp;started=perf_counter()
    write_json(directory/'started.json',{'stage':'P7','started_at':now.isoformat(),'freeze_sha256':digest(root/'runs/p7/freeze.json'),'source_sha256':freeze['source_sha256']})
    try:
        result=run_release(root,output);write_json(directory/'computation.json',result)
        audit=validate_release(root,result);write_json(output/'independent_validation.json',audit)
        plots(root,output)
    except Exception:
        (directory/'failure.txt').write_text(traceback.format_exc());raise
    commands=[run_command(c,root,timeout=120) for c in [[sys.executable,'-m','pip','check'],
        [sys.executable,'-m','pytest','-q','--junitxml',str(directory/'pytest.xml')],['git','diff','--check'],['git','rev-parse','HEAD']]]
    tests=list(ET.parse(directory/'pytest.xml').getroot().iter('testcase'))
    passing=sum(t.find('failure') is None and t.find('error') is None and t.find('skipped') is None for t in tests)
    success=bool(tests) and passing==len(tests) and all(c['exit_code']==0 for c in commands) and audit['passed']
    data={'stage':'P7','generated_at':now.isoformat(),'result':result,'independent_audit':audit,'engineering_complete':success,
        'tests_passed':passing,'tests_failed':len(tests)-passing,'commands':commands,'elapsed_seconds':perf_counter()-started,
        'python':sys.version,'platform':platform.platform(),'packages':{d.metadata['Name']:d.version for d in metadata.distributions()},
        'source_sha256':freeze['source_sha256'],'freeze_sha256':digest(root/'runs/p7/freeze.json'),
        'artifact_sha256':{str(p.relative_to(root)):digest(p) for p in sorted(output.rglob('*')) if p.is_file()},
        'run_record':str((directory/'research_run.json').relative_to(root))}
    write_json(root/data['run_record'],data)
    if not success:return 1
    write_json(root/'runs/p7/latest_run.json',{'run_record':data['run_record']})
    from svxylab.daily import update_daily
    daily_code=update_daily(root,open_report=False);data['daily_exit_code']=daily_code
    if open_report:commands.append(run_command(['/usr/bin/open',str(root/'reports/latest.html')],root))
    data['engineering_complete']=success and daily_code==0 and all(c['exit_code']==0 for c in commands)
    write_json(root/data['run_record'],data);render_latest(root)
    print(f"P7：{root/'reports/latest.html'}；pytest {passing}通过；全部历史计算、审计和日报结束。",flush=True)
    return 0 if data['engineering_complete'] else 1
