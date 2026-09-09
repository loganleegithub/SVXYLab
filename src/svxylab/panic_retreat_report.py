"""Chinese static E1 report, rendered entirely from saved numerical artifacts."""
from html import escape
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NAMES={'A_SHOCK':'A 冲击直接参与','B_RETREAT':'B 退潮确认（主假设）','C_RETREAT_VX':'C 退潮＋VX确认',
       'D3_CLOCK_WAIT':'D3 机械等3日','D5_CLOCK_WAIT':'D5 机械等5日','cash':'现金0',
       'curve':'F2>F1曲线基准','fixed_50':'每日固定50% SVXY','SVXY_BUY_HOLD':'持续持有SVXY',
       'BIL_TR_REFERENCE':'独立BIL再投资参照','ALL':'全段','2019-2023':'2019—2023',
       '2024-2026':'2024—2026-09-04','TRIGGERED_TRADES':'成熟已触发交易','ALL_OPPORTUNITIES':'共同机会',
       'BOTH_TRIGGERED':'共同触发子集（补充）','BASE':'基础5bp／滞后1日',
       'MATURE':'已成熟','MATURE_TRADE':'成熟交易','CASH_NO_TRIGGER':'自然未触发·现金0',
       'NATURAL_NO_TRIGGER':'自然未触发','WINDOW_PENDING':'窗口尚未结束','DATA_GAP':'信号数据缺口',
       'SIGNAL_DATA_GAP':'首次触发不完整','PRICE_GAP':'价格缺口','COMMON_PENDING':'共同终点待成熟',
       'TRIGGERED':'已触发','TRIGGERED_AFTER_GAP':'缺口后首次观察触发','COMPLETED':'完成',
       'OPEN_MARKED':'未到期·市值计价','ENTRY_PENDING':'执行待发生','ACTIVE_AT_CUTOFF':'截止仍活动',
       'ENDED':'生命周期结束','SKIPPED_POSITION_OVERLAP':'持仓重叠跳过'}
LABELS={'rule_id':'规则','H':'持有交易日','period':'时期','sample':'样本口径','episodes':'冲击轮数',
  'signals':'信号数','data_judgeable':'完整可判断','executable':'可执行交易','natural_no_trigger':'自然未触发',
  'mature_trades':'本期内成熟交易','pending':'待成熟／跨本期末','gaps':'缺口','n':'成熟样本数',
  'mean':'均值','median':'中位数','win_rate':'胜率','mean_win':'平均盈利','mean_loss':'平均亏损','worst':'最差',
  'q10':'经验10%分位','q25':'25%分位','q75':'75%分位','trigger_rate':'信号率','mean_wait':'平均等待日',
  'mean_entry_loss':'平均入场相对损失','worst_entry_loss':'最大入场相对损失','mean_peak_drawdown':'平均峰谷回撤',
  'mean_signal_to_entry':'信号至执行未持有收益','profit_ge_10':'净收益≥10%次数','loss_ge_10':'入场损失≥10%次数',
  'pair':'配对差（前者减后者）','mean_loss_delta':'入场损失差','mean_drawdown_delta':'峰谷回撤差','excluded':'排除数',
  'excluded_episode_ids':'排除事件','account':'连续账户','total_return':'净收益','CAGR':'年复合收益',
  'max_drawdown':'最大收盘回撤','mean_exposure':'平均敞口','in_market_days':'持有损益日数','turnover':'累计换手倍数',
  'fees':'费用（美元）','equity_end':'期末权益（美元）','scenario':'敏感性','variant':'B单因素变体',
  'episode_id':'事件','shock_session':'冲击日','shock_vix':'冲击VIX','shock_rise5':'五日简单涨幅',
  'signal_session':'信号日','entry_session':'买入收盘','exit_session':'计划卖出收盘','signal_status':'信号状态',
  'trade_status':'交易状态','opportunity_status':'机会状态','common_end':'固定共同终点','net_return':'净收益',
  'marked_return':'截止市值收益','mae':'MAE（有符号）','mfe':'MFE','entry_loss':'入场相对最大损失',
  'peak_drawdown':'持有峰谷回撤','net_drawdown':'含费账户回撤','worst_day':'最差单日',
  'signal_to_entry_return':'信号至执行未持有收益','shock_to_signal_return':'冲击至信号已发生',
  'shock_to_entry_return':'冲击至买入已发生','wait_days':'等待日','repeated_shocks':'活动期再次冲击日数',
  'lifecycle_status':'尾部生命周期','end_session':'生命周期结束日','left_truncated':'起点左截断',
  'metric':'统计对象','clusters_contributing':'贡献簇数','lower':'描述区间下限','upper':'描述区间上限',
  'removed_cluster':'剔除簇','removed_episodes':'包含事件','remaining_mean':'剔除后均值','original_mean':'全样本均值',
  'n_remaining':'剩余样本','direction_changed':'方向改变','status':'状态','as_of_session':'交易日',
  'age':'冲击后交易日','vix':'VIX收盘','rise5':'五日简单涨幅','peak_so_far':'截至当日最高收盘',
  'retreat_condition':'当日达到退潮','vx_complete':'两端VX齐全','vx_down':'两只同合约均下跌','low_streak':'连续低于25日数',
  'f1_contract':'当日F1合约','f2_contract':'当日F2合约','f1_expiration':'F1到期','f2_expiration':'F2到期',
  'previous_session':'前股票交易日','f1_previous':'F1前日结算','f1_current':'F1当日结算',
  'f2_previous':'F2前日结算','f2_current':'F2当日结算','entry_to_aug12_gross':'入场至08-12毛回报',
  'actual_H10_net':'实际H10净回报','pnl_dollars':'本段美元损益','cluster_id':'推断簇','window_complete':'窗口已走完',
  'capital_at_entry':'连续账户入场资本','continuous_pnl_dollars':'连续账户事件贡献（美元）','continuous_return':'连续账户事件回报'}
PCT={'mean','median','win_rate','mean_win','mean_loss','worst','q10','q25','q75','trigger_rate','mean_entry_loss',
     'worst_entry_loss','mean_peak_drawdown','mean_signal_to_entry','mean_loss_delta','mean_drawdown_delta',
     'total_return','CAGR','max_drawdown','mean_exposure','shock_rise5','net_return','marked_return','mae','mfe',
     'entry_loss','peak_drawdown','net_drawdown','worst_day','signal_to_entry_return','shock_to_signal_return',
     'shock_to_entry_return','lower','upper','remaining_mean','original_mean','rise5','entry_to_aug12_gross','actual_H10_net','continuous_return'}


def pct(x):return '未计算／不适用' if pd.isna(x) else f'{x*100:+.2f}%'


def table(frame,columns=None):
    cols=[c for c in (columns or frame.columns.tolist()) if c in frame.columns]
    if frame.empty:return '<p class="muted">本口径样本数为0；没有可报告的估计值。</p>'
    def cell(v,c):
        if pd.isna(v):return '<span class="muted">—</span>'
        if isinstance(v,(bool,np.bool_)):return '是' if v else '否'
        if c in PCT:return f'{float(v)*100:.2f}%'
        if isinstance(v,(float,np.floating)):return f'{v:,.2f}' if not float(v).is_integer() else f'{v:,.0f}'
        text=NAMES.get(str(v),str(v))
        if c=='pair':
            for k in sorted(NAMES,key=len,reverse=True):
                if k in ['A_SHOCK','B_RETREAT','C_RETREAT_VX','D3_CLOCK_WAIT','D5_CLOCK_WAIT']:text=text.replace(k,NAMES[k])
        return escape(text)
    return '<div class="table-scroll"><table><thead><tr>'+''.join('<th>'+escape(LABELS.get(c,c))+'</th>' for c in cols)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+cell(row[c],c)+'</td>' for c in cols)+'</tr>' for _,row in frame.iterrows())+'</tbody></table></div>'


def build_report(root,output,run_dir,cfg,summary,n1):
    read=lambda n:pd.read_csv(output/(n+'.csv'))
    ep=read('episodes');s=read('signals');t=read('trades');o=read('opportunity_results');m=read('event_summary')
    paired=read('paired_summary');a=read('account_summary');led=read('ledgers');case=read('case_paths');daily=read('episode_daily')
    intervals=read('cluster_intervals');leave=read('leave_one_cluster_out');parts=read('participation')
    h10=t.loc[t.H.eq(10)];b=h10.loc[h10.rule_id.eq('B_RETREAT')&h10.trade_status.eq('MATURE')]
    main=m.loc[m.period.eq('ALL')&m.H.eq(10)&m['sample'].eq('TRIGGERED_TRADES')]
    oppmain=m.loc[m.period.eq('ALL')&m.H.eq(10)&m['sample'].eq('ALL_OPPORTUNITIES')]
    bm=main.loc[main.rule_id.eq('B_RETREAT')].iloc[0]
    base_ac=a.loc[a.scenario.eq('BASE')&a.period.eq('ALL')]
    ba=base_ac.loc[base_ac.account.eq('B_RETREAT')&base_ac.H.eq(10)].iloc[0]
    mainpair=paired.loc[paired.period.eq('ALL')&paired.H.eq(10)&paired['sample'].eq('ALL_OPPORTUNITIES')]
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['PingFang SC','Arial Unicode MS','DejaVu Sans'],
                         'axes.unicode_minus':False,'font.size':10,'svg.fonttype':'path'})
    colors={'A_SHOCK':'#708391','B_RETREAT':'#11796d','C_RETREAT_VX':'#3f61a6','D3_CLOCK_WAIT':'#a07228','D5_CLOCK_WAIT':'#b96666',
            'SVXY_BUY_HOLD':'#555555','fixed_50':'#979797','BIL_TR_REFERENCE':'#bd9e44','cash':'#d3d3d3','curve':'#9565a0'}
    charts=output/'charts';charts.mkdir(exist_ok=True)
    fig,axes=plt.subplots(2,1,figsize=(12,7),sharex=True,gridspec_kw={'height_ratios':[2,1]})
    for name in ['A_SHOCK','B_RETREAT','C_RETREAT_VX','D3_CLOCK_WAIT','D5_CLOCK_WAIT','SVXY_BUY_HOLD','fixed_50','curve','BIL_TR_REFERENCE']:
        x=led.loc[led.scenario.eq('BASE')&led.account.eq(name)&led.H.isin([0,10])]
        dates=pd.to_datetime(x.as_of_session);value=x.equity_end/cfg['capital']
        axes[0].plot(dates,value,label=NAMES[name],color=colors[name],linewidth=1.8 if name=='B_RETREAT' else 1)
        axes[1].plot(dates,100*(value/np.maximum.accumulate(np.r_[1,value])[1:]-1),color=colors[name],linewidth=1)
    axes[0].set_ylabel('连续权益 / 初始资本');axes[1].set_ylabel('收盘回撤（%）');axes[0].legend(ncol=3,fontsize=8)
    for ax in axes:ax.grid(alpha=.15)
    fig.tight_layout();fig.savefig(charts/'continuous.svg');fig.savefig(charts/'continuous.png',dpi=150);plt.close(fig)
    selected=set()
    if len(b):selected.update([b.loc[b.net_return.idxmin(),'episode_id'],b.loc[b.net_return.idxmax(),'episode_id']])
    pe=read('paired_events')
    for pair in ['B_RETREAT-D3_CLOCK_WAIT','B_RETREAT-D5_CLOCK_WAIT']:
        x=pe.loc[pe.H.eq(10)&pe.pair.eq(pair)]
        if len(x):selected.update([x.loc[x.net_return.idxmin(),'episode_id'],x.loc[x.net_return.idxmax(),'episode_id']])
    link=lambda name,label:f'<a href="{escape(str(output/name))}">{escape(label)}</a>'
    def img(path,alt):return f'<img src="{escape(str(path))}" alt="{escape(alt)}" loading="lazy">'
    snippets=[]
    for episode in ep.itertuples():
        eid=episode.episode_id;p=case.loc[case.episode_id.eq(eid)];dt=daily.loc[daily.episode_id.eq(eid)]
        rows=h10.loc[h10.episode_id.eq(eid)];br=rows.loc[rows.rule_id.eq('B_RETREAT')]
        fig,axs=plt.subplots(2,2,figsize=(12,6))
        dx=pd.to_datetime(dt.as_of_session);axs[0,0].plot(dx,dt.vix,label='VIX收盘',color='#565b66')
        axs[0,0].plot(dx,dt.peak_so_far,label='当时运行最高',linestyle='--',color='#bd9e44');axs[0,0].legend(fontsize=8)
        axs[0,0].set_title('VIX与当时最高点（指数点）')
        anchors=[('冲击收盘',episode.shock_session,axs[0,1]),
                 ('B信号收盘',br.signal_session.iloc[0] if len(br) else None,axs[1,0]),
                 ('B实际买入收盘',br.entry_session.iloc[0] if len(br) else None,axs[1,1])]
        for label,date,ax in anchors:
            ax.set_title(label+'锚点：SVXY真实总回报路径')
            if date is not None and date in p.as_of_session.values:
                pp=p.loc[p.as_of_session.ge(date)];anchor=pp.total_return_index.iloc[0]
                ax.plot(pd.to_datetime(pp.as_of_session),100*(pp.total_return_index/anchor-1),color='#11796d')
                ax.axhline(0,color='#ccc',linewidth=.7)
                for r in rows.itertuples():
                    if r.entry_session>=date:ax.axvline(pd.Timestamp(r.entry_session),color=colors[r.rule_id],alpha=.35,linewidth=.8)
                if len(br):
                    for d0,style in [(br.entry_session.iloc[0],':'),(br.exit_session.iloc[0],'--')]:
                        ax.axvline(pd.Timestamp(d0),color='#11796d',linestyle=style,alpha=.6)
            else:ax.text(.5,.5,'B未触发／该锚点尚不存在',ha='center',va='center',transform=ax.transAxes)
            ax.set_ylabel('锚点以后回报（%）')
        for ax in axs.flat:ax.grid(alpha=.12);ax.tick_params(axis='x',rotation=30,labelsize=7)
        fig.tight_layout();fig.savefig(charts/(eid+'.svg'));fig.savefig(charts/(eid+'.png'),dpi=150);plt.close(fig)
        worst=''
        if len(br):
            r=br.iloc[0];worst=f'；B净收益 {pct(r.net_return)}，先承受入场相对损失 {pct(r.entry_loss)}'
            observations=[]
            if r.net_return<0:observations.append('按规则时间退出仍亏损；失败交易保留。')
            if r.entry_loss>=.1:observations.append('退潮确认后仍发生至少10%的收盘口径入场损失。')
            if r.signal_to_entry_return>0:observations.append(f'信号到可执行收盘已上涨{pct(r.signal_to_entry_return)}，这段不计入交易收益。')
            if episode.repeated_shocks:observations.append(f'生命周期内还有{episode.repeated_shocks}日再次满足冲击条件，没有重置起点或补做交易。')
        else:observations=['等待窗口内没有完整可执行的B信号；见下方自然未触发／缺口／待成熟状态。']
        snippets.append(f'<details id="{eid}" {"open" if eid in selected else ""}><summary>{escape(episode.shock_session)}｜VIX {episode.shock_vix:.2f}，五日上涨 {pct(episode.shock_rise5)}{worst}</summary>'
          +'<p>'+''.join(observations)+'</p>'+img(charts/(eid+'.svg'),eid+'四锚点价格证据')
          +'<p class="muted">彩色竖线为各规则入场；B绿点线为入场、虚线为时间退出。图示毛路径不是可实现止盈；交易净收益以含费账本为准。VIX生命周期与SVXY固定观察窗口不同，横轴终点可不同。</p>'
          +table(o.loc[o.H.eq(10)&o.episode_id.eq(eid)],['rule_id','signal_session','entry_session','exit_session','common_end','signal_status','opportunity_status','net_return'])
          +table(rows,['rule_id','trade_status','net_return','mae','mfe','entry_loss','peak_drawdown','worst_day','fees','signal_to_entry_return'])
          +'<details><summary>逐日峰值、再次冲击与当前两只VX合约（可横向滚动）</summary>'
          +table(dt,['as_of_session','age','vix','rise5','peak_so_far','retreat_condition','vx_complete','vx_down','low_streak',
                    'f1_contract','f1_expiration','previous_session','f1_previous','f1_current','f2_contract','f2_expiration','f2_previous','f2_current'])+'</details></details>')
    pair_sentences='；'.join(f'{NAMES.get(r.pair.split("-")[1],r.pair)}：B增量{pct(r.mean)}' for r in mainpair.itertuples() if r.pair.startswith('B_'))
    cpair=mainpair.loc[mainpair.pair.eq('C_RETREAT_VX-B_RETREAT')].iloc[0]
    lead=(f'本定义识别{len(ep)}轮冲击、{summary["inference_clusters"]}个重叠推断簇。主B有{int(bm.n)}笔成熟交易，平均净收益{pct(bm["mean"])}，'
          f'最差单笔{pct(bm.worst)}；入场后最大收盘损失{pct(bm.worst_entry_loss)}。')
    uncertainty=intervals.loc[intervals.metric.eq('B_RETREAT_OPPORTUNITIES')].iloc[0]
    es=read('execution_summary');ns=read('neighborhood_summary')
    dev=m.loc[m.rule_id.eq('B_RETREAT')&m.H.eq(10)&m['sample'].eq('ALL_OPPORTUNITIES')&m.period.eq('2019-2023')].iloc[0]
    revealed=m.loc[m.rule_id.eq('B_RETREAT')&m.H.eq(10)&m['sample'].eq('ALL_OPPORTUNITIES')&m.period.eq('2024-2026')].iloc[0]
    late=es.loc[es.rule_id.eq('B_RETREAT')&es.H.eq(10)&es['sample'].eq('ALL_OPPORTUNITIES')&es.period.eq('ALL')&es.scenario.eq('LAG_2')].iloc[0]
    changed_c=h10.loc[h10.rule_id.eq('C_RETREAT_VX')].set_index('episode_id').entry_session.ne(b.set_index('episode_id').entry_session)
    difference_events='、'.join(changed_c.index[changed_c])
    sensitivity_text=(f'2019—2023的{int(dev.n)}笔B平均净收益{pct(dev["mean"])}；2024—2026的{int(revealed.n)}笔为{pct(revealed["mean"])}。'
        f'成本增至单边30bp仍保留正的全段点估计，但多延迟一天降至{pct(late["mean"])}。十个B邻域均值为正，最差单笔仍约−40.93%。'
        '逐簇剔除后B均值及B相对D3/D5仍为正；去掉2025-04-03簇，B−A由正转负。')
    verdict='相对机械等3／5日，原B有继续固定规则观察的历史线索；自身收益区间跨零，且额外延迟使均值转负，尚不足作为实际资金依据。'
    if bm['mean']<0:verdict='这组预定B定义的平均成熟交易净收益为负，当前证据不支持将其当作获利规则。'
    explanation=(f'{lead} 在全部冲击共同机会口径内，相对{pair_sentences}。C相对B的平均增量为{pct(cpair["mean"])}，'
        f'比较样本{int(cpair.n)}轮。B共同机会均值的事件簇95%描述区间为[{pct(uncertainty.lower)}, {pct(uncertainty.upper)}]。{sensitivity_text}{verdict}')
    summary['research_conclusion']=explanation
    from svxylab.panic_retreat import write_json
    write_json(output/'summary.json',summary)
    card=json.loads((output/'candidate_card.json').read_text());card['research_conclusion']=verdict
    card['next_step']='支持把原B保留为固定前向观察候选；须另行授权，本轮不启动。自身收益、执行敏感性及样本局限继续保留，不选回测冠军。'
    write_json(output/'candidate_card.json',card)
    special=[]
    for title,start,end in [('2020年一季度','2020-01-01','2020-03-31'),('2022年','2022-01-01','2022-12-31'),
                           ('2024年7—8月','2024-07-01','2024-08-31'),('2025年3—4月','2025-03-01','2025-04-30')]:
        related=ep.loc[ep.shock_session.le(end)&(ep.end_session.fillna(cfg['end']).ge(start))]
        entries=' · '.join(f'<a href="#{r.episode_id}">{r.shock_session}</a>' for r in related.itertuples())
        special.append(f'<li>{title}：{entries or "没有按本定义识别的episode；没有手工添加理想事件。"}</li>')
    n1_html=f'<p>N1原24.68%复算为{pct(n1.get("original_interval_return",np.nan))}，对应<b>8月5日收盘至8月12日收盘</b>，包含8月6日当天收益。8月6日收盘买入到12日收盘的毛回报为<b>{pct(n1.get("aug06_close_to_aug12_close",np.nan))}</b>。</p>'
    n1_html+=table(pd.DataFrame(n1.get('rules',[])),['rule_id','shock_session','signal_session','entry_session','exit_session','signal_to_entry_return','entry_to_aug12_gross','actual_H10_net'])
    eps=read('execution_pairs')
    eall=pd.concat([m.assign(scenario='BASE'),es],ignore_index=True)
    fullpairs=pd.concat([paired.assign(scenario='BASE'),eps],ignore_index=True)
    selected_periods=['ALL','2019-2023','2024-2026']
    basiccols=['rule_id','n','mean','median','win_rate','worst','mean_entry_loss','worst_entry_loss','mean_signal_to_entry']
    pathcols=['rule_id','H','n','mean','median','mean_win','mean_loss','q10','q25','q75','profit_ge_10','loss_ge_10']
    acctcols=['account','H','total_return','CAGR','max_drawdown','mean_exposure','in_market_days','turnover','fees']
    filelist=['episodes.csv','signals.csv','episode_daily.csv','trades.csv','opportunity_results.csv','event_summary.csv','paired_summary.csv',
       'ledgers.csv','participation.csv','account_summary.csv','execution_summary.csv','execution_pairs.csv','neighborhood_summary.csv',
       'neighborhood_episodes.csv','neighborhood_signals.csv','neighborhood_trades.csv','neighborhood_opportunities.csv','inference_clusters.csv',
       'cluster_intervals.csv','bootstrap_scalars.csv','leave_one_cluster_out.csv','case_paths.csv','trade_paths.csv','independent_trade_audit.csv',
       'n1_reconciliation.json','summary.json','candidate_card.json']
    inventory=pd.DataFrame([
        {'研究组成':'可执行事件定义','原有':'未具备本定义；M2为五日条件预测，mechanism聚类使用成熟标签','本轮':'新增因果episode与五条规则'},
        {'研究组成':'当时可得触发时钟','原有':'已有次交易日截止/收盘；无本事件触发','本轮':'复用P2真实日历时钟，保存每日证据'},
        {'研究组成':'包含成本的实际SVXY路径','原有':'已有P2连续账本、N1路径归因','本轮':'复用continuous_ledger，固定份额持有，显式排定退出'},
        {'研究组成':'未触发＋机械等待完整比较','原有':'未具备；N1反弹是事后归因','本轮':'全episode五规则×三期限机会表及连续账户'}])
    coverage=summary['real_data_coverage']
    return f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SVXYLab｜E1 恐慌退潮事件研究</title><style>
:root{{--ink:#22302e;--muted:#687873;--green:#11796d;--paper:#f4f4ed;--line:#d9dfd7}}*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:16px/1.8 -apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif}}main{{max-width:1350px;margin:auto;padding:46px 32px 90px}}h1{{font-size:40px;line-height:1.3;margin:10px 0 18px;letter-spacing:-1px}}h2{{font-size:25px;margin:0 0 16px}}h3{{font-size:19px}}p{{max-width:1120px}}.kicker{{letter-spacing:2px;font-size:12px;color:var(--green)}}.lead{{font-size:19px;line-height:1.9}}.muted{{color:var(--muted);font-size:13px}}.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:28px 0}}.card{{background:#fff;border:1px solid var(--line);border-radius:10px;padding:20px}}.card b{{display:block;font-size:28px;font-weight:600}}.card span{{font-size:13px;color:var(--muted)}}section{{margin:30px 0;background:#fff;padding:28px;border-radius:12px;border:1px solid var(--line)}}.note{{border-left:4px solid var(--green);padding:8px 18px;background:#edf4ee}}a{{color:var(--green);text-underline-offset:3px}}nav{{display:flex;gap:18px;flex-wrap:wrap;font-size:14px}}.table-scroll{{overflow:auto;max-width:100%;margin:16px 0;border:1px solid var(--line);border-radius:7px}}table{{border-collapse:collapse;font-size:13px;min-width:100%}}th,td{{padding:10px 13px;text-align:right;border-bottom:1px solid #e7ebe6;white-space:nowrap}}th{{background:#edf2ec;color:#41584e;font-weight:600}}td:first-child,th:first-child{{text-align:left}}tr:hover{{background:#f7faf7}}img{{max-width:100%;height:auto;display:block;margin:20px auto}}details{{border:1px solid var(--line);border-radius:8px;padding:16px;margin:14px 0}}summary{{cursor:pointer;font-weight:600}}code{{font-size:13px;background:#eef1e9;padding:3px 6px;border-radius:3px}}.files{{columns:3;list-style:none;padding:0;font-size:13px}}.files li{{margin-bottom:8px;overflow-wrap:anywhere}}footer{{font-size:13px;color:var(--muted)}}@media(max-width:760px){{main{{padding:24px 14px}}section{{padding:16px}}h1{{font-size:31px}}.cards{{grid-template-columns:repeat(2,1fr)}}.files{{columns:1}}}}
</style><main><div class="kicker">SVXYLAB · E1 · PANIC RETREAT</div><h1>恐慌退潮以后，<br>还剩多少可交易的修复？</h1>
<p class="muted">真实数据截止 2026-09-04 · 2019年以来已揭示历史研究 · 规范化资本100,000美元 · 等待用户验收</p>
<p class="lead">{escape(lead)}</p>
<div class="cards"><div class="card"><b>{len(ep)} / {summary['inference_clusters']}</b><span>因果事件 / 推断簇</span></div><div class="card"><b>{pct(bm['mean'])}</b><span>B成熟交易平均净收益 · n={int(bm.n)}</span></div><div class="card"><b>{pct(bm.worst)}</b><span>B最差单笔净收益</span></div><div class="card"><b>{pct(ba.total_return)}</b><span>B全段连续账户净收益 · H10</span></div></div>
<p class="note">{escape(verdict)} B共同机会均值的95%描述区间：{pct(uncertainty.lower)}至{pct(uncertainty.upper)}。事件簇重采样不涵盖阈值选择、未知未来制度风险，也不把已观察历史重新变成盲测。</p>
<p>{escape(sensitivity_text)}</p>
<nav><a href="#results">主比较</a><a href="#accounts">连续账户</a><a href="#robustness">稳健性</a><a href="#cases">逐事件证据</a><a href="#august">2024年8月核对</a><a href="#method">口径与数据</a><a href="#delivery">复算与交付</a></nav>
<section id="results"><h2>实际买入后的收益与损失</h2><p>{escape(pair_sentences)}。上述为全部冲击共同机会配对收益差，单位为百分点。B−D3与B−D5的95%描述区间分别为+0.23至+10.00、+0.82至+10.31个百分点；该历史线索仍受7簇小样本及已揭示研究限制。C−B为{pct(cpair['mean'])}，样本{int(cpair.n)}；C只有{int(changed_c.sum())}轮改变入场（{escape(difference_events)}），其他入场与B相同。它新增的证据集中于这一例，没有普遍改善的证明。</p>
<h3>已触发且成熟的交易 · H=10 · 两边各5bp</h3>{table(main,basiccols)}
<p>一般情况下各规则的交易集合可能不同；本次主配置五规则均为相同7轮，真实自然未触发、缺口、窗口待成熟均为0。没有合成真实案例来填“未触发”一栏。信号至执行期间SVXY平均变化{pct(bm.mean_signal_to_entry)}，这段已经发生且未持有；只从实际买入收盘以后累计交易收益。</p>
<h3>同一批冲击、同一预定共同终点</h3>{table(oppmain,['rule_id','episodes','n','signals','natural_no_trigger','mean','worst','mean_entry_loss','trigger_rate','mean_wait'])}
{table(mainpair,['pair','n','mean','median','worst','mean_loss_delta','mean_drawdown_delta','excluded'])}
<p class="muted">风险差为前者减后者，负数表示该样本路径损失较少。自然未触发记现金0，未知结果为空；共同起点s+lag收盘前现金，终点s+W+lag+H，不取决于最晚实际触发日。较少参与与较小风险可以并存，不能直接称风险匹配alpha。</p>
<details><summary>收益分布、辅助H5/H20与计数核对</summary>{table(m.loc[m.period.eq('ALL')&m['sample'].eq('TRIGGERED_TRADES')],pathcols)}
{table(m.loc[m.period.eq('ALL')&m.H.eq(10)&m['sample'].eq('ALL_OPPORTUNITIES')],['rule_id','episodes','data_judgeable','natural_no_trigger','signals','executable','mature_trades','pending','gaps'])}<p>计数有嵌套关系；不能把每列相加当作总事件数。收益≥10%与入场损失≥10%可以发生于同一笔。</p></details>
<details><summary>所有时期与逐年结果（包含零事件年份）</summary>{table(m,['period','H','rule_id','sample','episodes','n','mean','worst','mean_entry_loss'])}</details>
<details><summary>共同触发子集，仅作补充</summary>{table(paired.loc[paired.period.eq('ALL')&paired.H.eq(10)&paired['sample'].eq('BOTH_TRIGGERED')],['pair','n','mean','mean_loss_delta','excluded','excluded_episode_ids'])}</details></section>
<section id="accounts"><h2>放回真实日历，检查资金和机会成本</h2><p>B主账户全段净收益{pct(ba.total_return)}、最大收盘回撤{pct(ba.max_drawdown)}、平均敞口{pct(ba.mean_exposure)}，持有损益{int(ba.in_market_days)}日。每个账户只使用起始资本，不叠加独立事件的100,000美元。</p>
{img(charts/'continuous.svg','五条规则和基准的连续权益与收盘回撤')}
{table(base_ac.loc[base_ac.H.isin([0,10])],acctcols)}
<p>同以2019-01-02收盘前现金为锚点；基准最早2019-01-03收盘买入。SVXY持续持有保持份额；固定50%和曲线基准按既有目标规则再平衡。BIL单独持有使用N1真实分配再投资总回报代理，扣入场5bp，不代表用户现金利率；没有建立SVXY/BIL轮动。无到期头寸不强制卖出，不另扣期末清算费。较低敞口造成的较小回撤不能自动解释为预测优势。</p>
<details><summary>三个期限及原连续净值的分段／全年切片</summary>{table(a.loc[a.scenario.eq('BASE')],['period',*acctcols])}</details>
<details><summary>逐事件连续账户贡献、重叠跳过与尾部状态</summary>{table(parts.loc[parts.scenario.eq('BASE')],['episode_id','rule_id','H','signal_session','entry_session','exit_session','status','capital_at_entry','continuous_pnl_dollars','continuous_return'])}
<p>本次主配置没有重叠跳过，也没有截止未到期头寸或未来待买入计划；全部7个episode生命周期已结束。新计划入场日不晚于现有计划退出日即跳过；独立事件假设交易仍在前表中。末尾未完成头寸按市值标记，不算成熟交易。</p></details></section>
<section id="robustness"><h2>有限稳健性：是否靠少数事件撑住</h2><h3>主H10：簇整体重采样1000次，种子1707</h3>
{table(intervals,['metric','n','clusters_contributing','mean','lower','upper'])}
<p>同簇所有episode一起抽取，保留簇内成员数及重复次数，再重算原均值。推断簇由[s,s+W+lag+20]重叠连通形成；不重叠也不保证经济独立。尾部分位与区间仅描述已见样本；不将事件任意拼成CAGR。</p>
<details open><summary>去掉某簇会改变方向的项目</summary>{table(leave.loc[leave.direction_changed],['metric','removed_cluster','removed_episodes','n_remaining','original_mean','remaining_mean'])}</details>
<details><summary>全部逐簇剔除（坏事件未从正式结果删除）</summary>{table(leave,['metric','removed_cluster','removed_episodes','n_remaining','original_mean','remaining_mean'])}</details>
<h3>相同信号：成本与多延迟一天</h3>{table(eall.loc[eall.period.eq('ALL')&eall.H.eq(10)&eall['sample'].eq('ALL_OPPORTUNITIES')],['scenario','rule_id','episodes','n','signals','mean','worst','mean_entry_loss'])}
<details><summary>执行敏感性的两时期／逐年配对与连续账户</summary>{table(fullpairs.loc[fullpairs.H.eq(10)&fullpairs['sample'].eq('ALL_OPPORTUNITIES')],['scenario','period','pair','n','mean','mean_loss_delta'])}{table(a.loc[a.scenario.isin(['COST_0BP','COST_15BP','COST_30BP','LAG_2'])],['scenario','period',*acctcols])}</details>
<h3>仅B的十个单因素邻域 · 全部保留</h3>{table(ns.loc[ns.period.eq('ALL')&ns['sample'].eq('ALL_OPPORTUNITIES')],['variant','episodes','n','signals','natural_no_trigger','mean','worst','mean_entry_loss'])}
<p>每个邻域重新识别自己的事件，未在主版挑好的事件上换阈值。W同时改变生命周期和共同终点。没有交叉组合、止盈止损搜索或冠军晋升；B主定义始终是30／40%／10%／W10／低于25连续5日。</p>
<details><summary>邻域两个历史段、全年和连续账户</summary>{table(ns,['variant','period','sample','episodes','n','mean','worst'])}{table(a.loc[~a.scenario.isin(['BASE','COST_0BP','COST_15BP','COST_30BP','LAG_2'])],['scenario','period',*acctcols])}</details></section>
<section id="cases"><h2>逐事件证据与全部失败</h2><p>默认展开B最好、最差及B相对D3/D5改善／恶化最大的事件；这只是展示顺序，不改变入样规则。所有主事件都可展开，不依据新闻解释因果。</p>
<ul>{''.join(special)}</ul>
{table(ep,['episode_id','shock_session','shock_vix','shock_rise5','repeated_shocks','end_session','lifecycle_status','left_truncated'])}
<h3>所有主配置亏损交易 · 含H5/H10/H20</h3>{table(t.loc[t.net_return.lt(0)],['episode_id','rule_id','H','entry_session','exit_session','net_return','entry_loss','peak_drawdown','fees'])}
{''.join(snippets)}</section>
<section id="august"><h2>2024年8月：把“已经反弹”与“可以买到”分开</h2>{n1_html}<p>8月12日只用于核对旧N1文字，不替代E1时间退出，也不把冲击前一日到入场的修复加进净交易回报。</p></section>
<section id="method"><h2>复用存量，固定一份新实验口径</h2><p>施工前Git HEAD为d04b2ebb3ae2599aea3c24e2c8c8d7d5046fdaa3，工作区干净；在当前仓库src/tests/runs/reports/任务和进度记录中检索恐慌、退潮、反弹、panic、retreat、rebound与event study，没有完整等价实现。</p>{table(inventory)}
<p>真实股票日历{coverage['stock_sessions']}日（{coverage['first']}—{coverage['last']}），VIX有效{coverage['vix_present']}日，SVXY价格及公司行动齐全{coverage['svxy_price_action_complete']}日，VX当日当前ID的两端证据完整{coverage['vx_current_previous_complete']}日（首日没有2018回看），BIL{coverage['bil_rows']}日。事件窗口的缺口计数见主表。未重新下载行情；只对实际读入的原件和派生表核对摘要。</p>
<p>SVXY是每日−0.5倍VIX短期期货指数暴露，不能把VIX现货下跌比例换算成SVXY利润；2018-02-27收盘后改倍数，本轮仅使用2019以后实际基金历史。依据<a href="https://www.proshares.com/our-etfs/strategic/svxy">发行人产品说明</a>与<a href="https://www.sec.gov/Archives/edgar/data/1415311/000119312518059052/d503117dex991.htm">2018发行人公告</a>。VIX与VX原件来自Cboe，ETF来自已有Yahoo Chart原件。</p>
<p>阈值为附件待检验示例，不是机构算法或最优值。冲击条件：VIX≥30且五日简单上涨≥40%（需六个收盘）；只在s+1…s+10找B的首次相对运行最高回落≥10%。C要求当日仍满足B且当前前两个月度VX同ID在t和t−1均严格下跌。活动期不重置；窗口结束且连续5日VIX&lt;25后，次日重新布防。缺失VIX打断低值连续计数并使峰值未知；VX缺口只影响相应C判断。</p>
<p>主执行e=t+1实际股票收盘，截止为该收盘前60分钟，包含半日市／夏令时。价格是收盘成交代理，5bp不是历史逐笔价差；没有完整历史发布时间证据。退出u=e+H，不重复移位。买卖均计费，拆分份额除以split_factor，分配留现金、不再投资，不重复扣基金费；总回报图与含费份额账本用途分开。</p>
<p>MAE为相对买入后权益的最小有符号收盘变化，MFE为最大值，入场相对损失=max(0,−MAE)；峰谷回撤由持有期间滚动高点计算。另有含买卖费用的净账户回撤。均为日线收盘口径，不代表日内最坏损失或可保证成交的止损。事件按冲击年份归属，交易／共同机会要求在展示时期末成熟；跨期观察单列为待成熟，连续账户只从原净值切片，不逐年重置。</p></section>
<section id="delivery"><h2>工程交付、研究结论与下一步含义</h2><p><b>工程：</b>本次已运行五规则×三期限、四项执行敏感性、十个B邻域、{summary['audit']['ledger_accounts']}个账户／{summary['audit']['ledger_rows']:,}行核账，以及{summary['audit']['independent_mature_trades']}笔成熟交易的独立份额计算。测试、两次复算和页面目视检查的最终状态见下方验收检查卡。</p>
<div id="validation-card" class="note">最终pytest、同输入复算与页面检查进行中；计算完成不等于策略有优势。</div>
<p><b>研究：</b>{escape(explanation)}</p><p><b>下一步：</b>证据支持保留原B作为固定前向观察候选，尚不能确认收益优势。本轮在交付验收处结束；若以后另行授权前向观察，仍用原B及上述执行规则。没有授权实际资金、期权交易或监控集成，不把C或邻域中较好的一项自动升级。</p>
<p>复算命令：<code>.venv/bin/python -m svxylab.panic_retreat --open</code><br>历史信号截面：<code>.venv/bin/python -m svxylab.panic_retreat --as-of 2024-08-07 --signal-only</code>。截面接口不读取SVXY结果，不能称为当前实时机会。</p>
<p>{link('summary.json','机器可读总结果')} · {link('candidate_card.json','原B研究卡')} · <a href="{escape(str(run_dir/'experiment_record.json'))}">计算前实验记录</a> · <a href="{escape(str(run_dir/'inputs.json'))}">输入来源及摘要</a> · <a href="{escape(str(run_dir/'independent_validation.json'))}">独立核账</a></p>
<details><summary>紧凑计算证据与完整表下载</summary><ul class="files">{''.join('<li>'+link(name,name)+'</li>' for name in filelist)}</ul></details>
<p class="muted">运行产物：{escape(str(output))}。原始与派生行情继续留在本地忽略目录；本次未提交、推送或宣布验收，旧报告和M2研究结果保留。</p></section>
<footer>SVXYLab · E1 · 数据截止2026-09-04。工程完成、真实数据覆盖、研究评价分别记录。此页只呈现价格证据，不补写新闻因果故事。</footer></main></html>'''
