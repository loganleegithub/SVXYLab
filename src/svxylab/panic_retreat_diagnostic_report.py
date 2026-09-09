"""Chinese, local, static report; retrospective plotting lives only here."""
from __future__ import annotations

from html import escape
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from svxylab.panic_retreat import RULES, write_json

LABELS = {
 'episode_id':'事件', 'rule_id':'原规则', 'H':'持有期限', 'cell':'四格',
 'entry_session':'入场日', 'exit_session':'退出日', 'signal_session':'原信号日',
 'information_session':'信息日', 'as_of_session':'日期', 'vix':'VIX',
 'vix_change':'VIX日变化', 'vix_down_day':'当日回落', 'vix_up_day':'当日升压',
 'peak_so_far':'当时运行最高值', 'b_state':'B状态', 'c_state':'C状态',
 'b_renewed':'B重新成立', 'c_renewed':'C重新成立', 'b_state_class':'B状态类别',
 'window_expired':'窗口已结束', 'one_trade_used':'一轮一笔已用',
 'position_blocks_H10':'原H10仓位阻挡', 'lifecycle_end':'生命周期结束', 'rearmed':'重新布防',
 'signal_state':'原信号状态','pre_execution_state':'成交前可用状态','fill_day_final_state':'成交日日线后状态',
 'window':'窗口', 'state_class':'状态类别', 'candidate_days':'候选日数',
 'observed_days':'观察日数', 'false_days':'条件不满足日', 'unknown_condition_days':'条件未知日',
 'mature':'已成熟', 'price_gaps':'行情缺口', 'immature':'未成熟', 'status':'状态',
 'positive':'正收益日数', 'negative':'负收益日数', 'mean_return':'日路径平均收益',
 'median_return':'日路径中位收益', 'min_return':'收益下界', 'max_return':'收益上界',
 'max_entry_loss':'最大入场损失', 'max_peak_drawdown':'最大峰谷回撤',
 'net_return':'含费收益', 'entry_loss':'入场相对损失', 'peak_drawdown':'峰谷回撤',
 'net_drawdown':'含费净值回撤', 'fees':'费用（美元）', 'actual_holding_sessions':'实际持有交易日',
 'R00':'原执行', 'R10':'只晚入场', 'R01':'只晚退出', 'R11':'两者都晚',
 'total_change':'总变化', 'entry_contribution':'入场贡献', 'exit_contribution':'退出贡献',
 'interaction':'交互项', 'entry_first':'先动入场的变化', 'entry_after_exit':'后动入场的变化',
 'exit_first':'先动退出的变化', 'exit_after_entry':'后动退出的变化',
 'first_false_session':'首次不满足信息日', 'earliest_action_session':'最早行动日',
 'pnl_at_information':'该信息日累计损益（美元）',
 'pnl_through_earliest_action':'至最早行动日累计损益（美元）',
 'remaining_original_pnl':'此后原账本损益（美元）', 'final_pnl':'原最终损益（美元）',
 'false_after_reset':'不满足时已重置', 'actionable_before_planned_exit':'行动日不晚于原退出',
 'unknown_days':'未知日数', 'original_condition':'原条件', 'complete_new_daily_rows':'新增完成日线数',
 'lag':'执行延迟', 'stage':'观察阶段', 'execution_decision_at':'成交前决策截止（UTC）',
 'state_observable_at_assumed_deadline':'该状态假定可得截止（UTC）',
 'execution_at':'执行收盘（UTC）', 'historical_publication_proven':'历史发布时间已证明',
 'from_H':'起始持有日', 'to_H':'终止持有日', 'start_session':'区间起日', 'end_session':'区间止日',
 'holding_pnl':'区间持有损益（美元）', 'extra_fees':'新增费用（美元）',
 'net_return_change':'净收益变化', 'identity_residual_dollars':'恒等式误差（美元）',
 'trade_status':'路径状态', 'reference':'原结果对照', 'trades':'笔数', 'max_numeric_error':'最大数值误差',
 'window_blocks':'窗口阻挡', 'one_trade_blocks':'一轮一笔阻挡', 'position_blocks':'原仓位阻挡',
 'shock_session':'首次冲击', 'end_session':'结束日', 'window_end_session':'窗口结束',
 'rearm_session':'重新布防', 'price_end_session':'价格评价终日', 'outside_true':'窗口外B满足日',
 'later_renewed':'原B之后重新成立日', 'later_sustained':'原B之后持续满足日',
 'later_false':'原B之后不满足日', 'new_running_high':'再创新高',
}
VALUES = {'TRUE':'满足','FALSE':'不满足','UNKNOWN':'未知','NOT_APPLICABLE':'不适用',
 'SUSTAINED':'持续满足','RENEWED':'重新成立','PRIOR_UNKNOWN':'前日未知',
 'INSIDE':'窗口内','OUTSIDE':'窗口外','MATURE':'已成熟','PRICE_GAP':'行情缺口',
 'OPEN_MARKED':'持有未成熟','ENTRY_PENDING':'入场未到','NO_CANDIDATE':'无候选',
 'COMPLETE':'完整','INCOMPLETE':'不完整','FALSE_OBSERVED':'已观察不满足',
 'NO_FALSE_OBSERVED':'未观察到不满足','SIGNAL_FORMED':'原信号形成',
 'PRE_EXECUTION_CUTOFF':'成交前截止','EXECUTION_FINAL_DAILY':'成交日日线完成后',
 '00':'原执行','10':'只晚入场','01':'只晚退出','11':'两者都晚'}
PERCENT = {k for k in LABELS if 'return' in k or 'drawdown' in k or 'entry_loss' in k} | {
 'R00','R10','R01','R11','total_change','entry_contribution','exit_contribution','interaction',
 'entry_first','entry_after_exit','exit_first','exit_after_entry'}


def pct(x):
    return '—' if pd.isna(x) else f'{x:+.2%}'


def table(df, columns=None):
    frame = df[columns].copy() if columns else df.copy()
    def format_value(x, col):
        if pd.isna(x): return '—'
        if isinstance(x, (bool,np.bool_)): return '是' if x else '否'
        if col in PERCENT: return pct(x)
        if isinstance(x, (float,np.floating)): return f'{x:.2f}' if abs(x)>1e-7 else f'{x:.2g}'
        return VALUES.get(x,x) if isinstance(x,str) else str(x)
    for col in frame:
        frame[col] = frame[col].map(lambda x: format_value(x,col))
    frame = frame.rename(columns=LABELS)
    return '<div class="table-scroll">'+frame.to_html(index=False,escape=True,border=0)+'</div>'


def details(title, body, opened=False):
    return f'<details {"open" if opened else ""}><summary>{escape(title)}</summary>{body}</details>'


def plot_episode(ep, tables, market, prices, out):
    daily = tables['timeline'].query('episode_id == @ep.episode_id and in_lifecycle').copy()
    sig = tables['signals'].query('episode_id == @ep.episode_id')
    later = tables['candidates'].query('episode_id == @ep.episode_id and H == 10')
    full_later = tables['candidates'].query('episode_id == @ep.episode_id')
    max_exit = int(full_later.u.max()) if len(full_later) else int(ep.end_i)+1
    end = min(max(max_exit,int(ep.end_i)+1,int(sig.t.max())+21),len(prices)-1)
    start = int(ep.s); x = np.arange(start,end+1)
    fig, axes = plt.subplots(5,1,figsize=(13,11),sharex=True,
                            gridspec_kw={'height_ratios':[2.2,1.6,1.7,1.8,1.5]},layout='constrained')
    ax = axes[0]
    ax.plot(daily.i,daily.vix,color='#304b70',lw=1.2,label='VIX 日线')
    ax.scatter(daily.i,daily.vix,c=np.where(daily.vix_down_day,'#258377','#bb5947'),s=13,zorder=4,
               label='绿色：当日下降；红色：持平／上升')
    ax.step(daily.i,daily.peak_so_far,where='post',color='#798491',ls='--',label='当时运行最高值')
    ax.step(daily.i,daily.peak_so_far*.9,where='post',color='#bb8b38',alpha=.7,label='原 B 退潮界线')
    # Retrospective peak is computed only in this downstream display function.
    peak = daily.loc[daily.vix.idxmax()]
    ax.annotate(f'整轮最高 {peak.vix:.2f} · {peak.as_of_session}\n仅事后注释',
                xy=(peak.i,peak.vix),xytext=(8,12),textcoords='offset points',fontsize=8)
    ax.set_ylim(top=daily.vix.max()*1.28); ax.set_ylabel('压力')
    ax.legend(loc='upper right',fontsize=7,ncol=2)
    lanes = [('vix_down_day','VIX 当天回落'),('b_state','B 条件满足'),('b_renewed','B 重新成立'),
             ('c_state','C 条件满足'),('c_renewed','C 重新成立'),('shock_condition','再次冲击条件')]
    for j,(key,label) in enumerate(lanes):
        hit = daily[key].eq('TRUE') if key.endswith('state') else daily[key].eq(True)
        if key == 'shock_condition': hit &= daily.age.gt(0)
        axes[1].scatter(daily.loc[hit,'i'],np.full(hit.sum(),j),marker='s',s=14,color='#2f7c87')
    axes[1].set_yticks(range(len(lanes)),[v for _,v in lanes],fontsize=8); axes[1].invert_yaxis()
    colors = ['#8495a8','#277f83','#bb8147','#807aa4','#b07387']
    for j,r in enumerate(sig.itertuples()):
        t = int(r.t); e=t+1
        axes[2].plot([t,e+20],[j,j],color=colors[j],alpha=.5,lw=1)
        axes[2].scatter(t,j,marker='o',s=25,color=colors[j])
        axes[2].scatter(e,j,marker='>',s=30,color=colors[j])
        for h in (5,10,20):
            axes[2].scatter(e+h,j,marker='x',s=25,color=colors[j])
            axes[2].annotate(str(h),(e+h,j),xytext=(0,5),textcoords='offset points',fontsize=7)
    axes[2].set_yticks(range(5),['A','B','C','等3日','等5日'],fontsize=8); axes[2].invert_yaxis()
    axes[2].set_title('原交易时钟：圆点信号 → 三角成交 × H5／10／20计划退出',fontsize=9,loc='left')
    price = prices.iloc[start:end+1]
    axes[3].plot(x,price.close/price.close.iloc[0]-1,color='#304b70')
    axes[3].yaxis.set_major_formatter(PercentFormatter(1)); axes[3].set_ylabel('SVXY 相对冲击日')
    axes[3].axhline(0,color='#c0c6ce',lw=.7)
    for cls,mark,col in [('SUSTAINED','o','#5f8c99'),('RENEWED','D','#c38338')]:
        g=later[later.state_class.eq(cls)]
        axes[4].scatter(g.information_i,g.net_return,marker=mark,s=20,color=col,label=VALUES[cls])
    axes[4].axhline(0,color='#8a98a6',lw=.7)
    axes[4].yaxis.set_major_formatter(PercentFormatter(1)); axes[4].set_ylabel('候选日的 H10 净收益')
    axes[4].legend(loc='upper right',fontsize=8)
    for a in axes:
        a.axvline(start,color='#b44e41',lw=.8)
        a.axvline(ep.window_end_i,color='#c09045',ls='--',lw=.8)
        a.axvline(ep.end_i,color='#637c99',ls=':',lw=1)
        a.axvline(ep.end_i+1,color='#32887b',ls=':',lw=.8)
        a.axvspan(ep.window_end_i+.3,ep.end_i,color='#e8d6b7',alpha=.15)
        a.grid(alpha=.12); a.spines[['right','top']].set_visible(False)
    ticks = np.unique(np.r_[np.linspace(start,end,9).astype(int),int(ep.end_i)])
    axes[-1].set_xticks(ticks,[market.as_of_session.iloc[i] for i in ticks],rotation=30,ha='right',fontsize=8)
    fig.suptitle(f'{ep.shock_session} — 完整生命周期与独立候选路径\n虚线：十日窗口结束；点线：生命周期结束／次日布防；阴影：窗口外',fontsize=12)
    target = out/f'timeline_{ep.shock_session}'
    fig.savefig(target.with_suffix('.png'),dpi=145)
    fig.savefig(target.with_suffix('.svg'),metadata={'Date':None})
    plt.close(fig)
    return dict(episode_id=ep.episode_id,shock_session=ep.shock_session,
        window_end_session=market.as_of_session.iloc[int(ep.window_end_i)],end_session=ep.end_session,
        rearm_session=market.as_of_session.iloc[int(ep.end_i)+1],price_end_session=market.as_of_session.iloc[end],
        retrospective_peak_session=peak.as_of_session,retrospective_peak=peak.vix,
        outside_true=int((daily.b_state.eq('TRUE') & daily.window_expired).sum()),
        later_renewed=int((daily.after_original_b & daily.b_renewed).sum()),
        later_sustained=int((daily.after_original_b & daily.b_state_class.eq('SUSTAINED')).sum()),
        later_false=int((daily.after_original_b & daily.b_state.eq('FALSE')).sum()))


def cards(tables, capital):
    info = tables['availability']; h = tables['holding_outcomes']; p = tables['candidates']
    affected = info.query('stage == "PRE_EXECUTION_CUTOFF" and lag == 2 and original_condition == "FALSE"')
    b = h.query('rule_id == "B_RETREAT" and H == 10'); bc = h.query('rule_id in ["B_RETREAT", "C_RETREAT_VX"] and H == 10')
    f20 = b.query('episode_id == "E_2020-02-27"').iloc[0]
    p20 = p.query('episode_id == "E_2020-02-27" and H == 10 and window == "OUTSIDE"')
    return [
      dict(title='入场前失效检查',
        support=f'原 lag=2 的 B/C 交易有 {len(affected)} 笔在成交前最新完成日线已不满足原条件，其中包括 2020 年 2 月事件的 B 和 C。状态检查存在可识别对象。',
        against='原 lag=1 没有新增完整日线可复查。2020 年 3 月 3 日 B 不满足是该日日线完成后才知道，不能据此声称可撤销 3 月 3 日收盘买入。延后执行才看到的不满足也出现在后来盈利的 2021 年 12 月、2022 年 4 月等交易中。',
        clock='lag=2 只能按既有假定在 t+2 收盘前60分钟检查 t+1 完成数据；实际历史发布／接收时刻仍未证明。无撤销交易或撤销收益模拟。',
        next_question='下一实验须先锁定使用 lag=1 还是 lag=2、具体可得时点、数据缺口处置、撤销后是否永远放弃该事件。不能同时更快成交和新增撤销门控。'),
      dict(title='持仓失效退出',
        support=f'2020 年 B 的首次不满足信息日是 {f20.first_false_session}，按原时钟最早 {f20.earliest_action_session} 行动。至该日原账本累计 {f20.pnl_through_earliest_action:+,.2f} 美元，此后原计划仍产生 {f20.remaining_original_pnl:+,.2f} 美元损益，说明风险并非已全部发生在可行动前。',
        against=f'H10 的 B/C 有 {int((bc.remaining_original_pnl>1e-6).sum())} 笔在该最早行动日之后仍获得正损益；2021 年 12 月 B 后续修复约 +6,931 美元，提前放弃也有代价。C 在重置后才不满足的情形单列，不能把低波重置自动当作坏信号。',
        clock='不满足在信息日 q 完成数据中形成，原时钟最早 q+1 收盘行动；所列损益来自原持有账本，包含等待该次收盘承担的波动，不是假设平仓收益，也未扣新增平仓费。',
        next_question='下一实验须锁定使用 B 还是 C 的持续条件、首次不满足与未知状态的区别、退出成交时点及重置处理；保留 2020，同时衡量放弃后续修复的代价。'),
      dict(title='同一压力事件中的后续退潮',
        support=f'2020 年窗口外确有 {len(p20)} 个原 B 条件满足日，其中 {int(p20.state_class.eq("RENEWED").sum())} 日为重新成立。H10 独立路径有 {int(p20.net_return.gt(0).sum())} 日为正；原规则确实排除了后续阶段。不能据原 E1 宣称后续没有可交易路径。',
        against=f'同一批窗口外路径仍有 {int(p20.net_return.lt(0).sum())} 日亏损，最差 {pct(p20.net_return.min())}；窗口内 3 月 4 日、3 月 10 日重新成立后的 H10 路径均大亏。持续符合运行峰值下方10%与新退潮不是同一件事，也没有可事先选到最佳日期的证据。',
        clock='候选按当日之前的运行峰值与当日完成 VIX 生成，次交易日收盘参与。97 日属于高度重叠的路径描述，不是97个独立信号或样本。',
        next_question='下一实验须先锁定持续满足是否可作为机会，还是只用 FALSE→TRUE；再锁定再次入场的等待、原持仓冲突、事件边界。不能一并调整撤销与持仓退出。')]


def render_report(root,out,run,tables,market,prices,cfg,validation):
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['PingFang SC','Arial Unicode MS','DejaVu Sans'],
                         'axes.unicode_minus':False,'svg.fonttype':'path','svg.hashsalt':'E1-diagnostic-v1'})
    report = root/'reports/panic_retreat_diagnostic.html'
    link = lambda p: escape(os.path.relpath(p,report.parent),quote=True)
    summaries = [plot_episode(ep,tables,market,prices,out) for ep in tables['episodes'].itertuples()]
    overview = pd.DataFrame(summaries); overview.to_csv(out/'episode_coverage.csv',index=False)
    mechanism_cards = cards(tables,cfg['capital']); write_json(out/'mechanism_cards.json',mechanism_cards)
    attr = tables['attribution']; b10=attr.query('rule_id == "B_RETREAT" and H == 10')
    p20=tables['candidates'].query('episode_id == "E_2020-02-27" and H == 10 and window == "OUTSIDE"')
    body = [f'''<header><div class="eyebrow">SVXYLab · E1 补充诊断 · 揭示后历史研究</div>
    <h1>第一次缓和之后，原规则错过了什么？</h1>
    <p class="lead">七轮事件保持不变。将阶段覆盖、入场／退出归因与条件可得时点分开，终点是三张机制证据卡。</p>
    <div class="numbers"><div><strong>7 / 35</strong>原事件／信号</div><div><strong>420</strong>四格交易路径</div>
    <div><strong>{len(tables['candidates'])}</strong>后续独立假设路径</div><div><strong>0</strong>机制修改</div></div></header>
    <nav><a href="#findings">诊断结论</a><a href="#timeline">七轮时间轴</a><a href="#execution">执行归因</a>
    <a href="#availability">信息时点</a><a href="#validation">验证与文件</a><a href="#cards">三项证据</a></nav>
    <section id="findings"><h2>能回答的三件事</h2>
    <p><b>2020 年后续路径确实被规则排除，但不是所有参与都能修复损失。</b>原窗口 3 月 12 日结束，生命周期 8 月 4 日结束，8 月 5 日重新布防。
    窗口外 97 个 B 满足日由 95 个持续满足日和 2 个重新成立日构成。其 H10 次日收盘路径有 {int(p20.net_return.gt(0).sum())} 正、{int(p20.net_return.lt(0).sum())} 负；最低 {pct(p20.net_return.min())}。
    这些路径高度重叠；不是97次独立退潮，不构成一套能事先选择获利日期的再入场策略。</p>
    <p><b>原 B 的 H10 延迟损失同时来自两端。</b>七事件等权，原执行平均 {pct(b10.R00.mean())}，两端各晚一天平均 {pct(b10.R11.mean())}。
    总变化 {b10.total_change.mean()*100:+.3f} 个百分点中，入场贡献 {b10.entry_contribution.mean()*100:+.3f}，退出贡献 {b10.exit_contribution.mean()*100:+.3f}。
    这只解释原有延迟对照，不能当作更快执行已经可得或更优的证据。</p>
    <p><b>原 lag=1 的成交前复查没有新增完整日线。</b>成交当日最终数据不满足，不能倒推为收盘前已能撤销。
    lag=2 多出一日可按原约定检查，但历史实际发布时间证据仍缺失。状态失效不属于收益分解中的第三项。</p>
    <p class="note">真实冻结行情：2019-01-02—2026-09-04，1930 个股票交易日；原35信号×H5/H10/H20，单边5bp，独立起点10万美元，现金收益0。
    基金价格已含的费用不重复扣除；拆股按原份额会计，现金分配留现金。本报告无参数寻优、无新盲测、无期权数据采购。</p></section>''']
    body.append('<section id="timeline"><h2>七轮完整生命周期与后续覆盖</h2><p>阶段标记允许重叠。绿色日线点只代表当天VIX下降；B状态依据从首次冲击起的运行最高值；“重新成立”严格要求上一日不满足、本日满足。事件结束当日仍纳入覆盖，下一交易日布防行不产生候选。</p><p>仓位阻挡按候选的次日成交日判断：不晚于原计划退出日即阻挡，等于退出日也阻挡，沿用原规则。窗口结束、一轮一笔与仓位阻挡可同时成立。入场相对损失以买入费后入场权益为基准；峰谷回撤采用买入后固定份额的日收盘路径，末日取卖出费前权益；含费净值回撤另列。日线不能显示盘中尾部。</p>')
    body.append(table(overview,['episode_id','window_end_session','end_session','rearm_session','price_end_session','outside_true','later_renewed','later_sustained','later_false']))
    body.append('<p>图中价格仅用于解释；本样本各事件绘图区间内无SVXY公司行动，故价格相对值与同期固定份额毛路径一致。所有收益评价仍使用份额／现金会计。整轮最高点只在绘图函数中计算，未传入候选生成函数。</p>')
    for ep in tables['episodes'].itertuples():
        eid = ep.episode_id
        g=tables['candidate_summary'].query('episode_id == @eid and H == 10')
        day=tables['coverage_days'].query('episode_id == @eid')
        p=tables['candidates'].query('episode_id == @eid')
        renew=p.query('state_class == "RENEWED"')
        daily_cols=['as_of_session','vix','vix_change','peak_so_far','b_state','c_state','b_state_class',
                    'window_expired','one_trade_used','position_blocks_H10','lifecycle_end']
        outcome=p.query('H == 10')[['information_session','net_return','entry_loss','peak_drawdown','trade_status']]
        all_days=day.merge(outcome,left_on='as_of_session',right_on='information_session',how='left')
        content = f'<img class="chart" src="{link(out/f"timeline_{ep.shock_session}.svg")}" alt="{ep.shock_session} 完整时间轴、原交易时钟和所有H10候选结果">'
        content += '<h3>H10 覆盖汇总</h3><p>先按本事件，再按状态与窗口分组；极值只描述全体范围，不选最佳日期。无候选显示为空收益，不补零。观察日／不满足日属于窗口总数，在状态分组间重复展示，不能再相加。</p>'
        content += table(g,['window','state_class','candidate_days','mature','price_gaps','immature','positive','negative','mean_return','median_return','min_return','max_return','max_entry_loss','max_peak_drawdown','status'])
        content += details('所有重新成立日：H5／H10／H20，不挑最优日',table(renew,['information_session','H','window','position_blocks','entry_session','exit_session','net_return','entry_loss','peak_drawdown']),eid=='E_2020-02-27')
        content += details('原B之后全部日期（包括不满足日）＋ H10 路径',table(all_days,daily_cols+['net_return','entry_loss','peak_drawdown','trade_status']))
        content += details('H5／H20 全部候选路径',table(p[p.H.ne(10)],['information_session','H','state_class','window','position_blocks','entry_session','exit_session','net_return','fees','entry_loss','peak_drawdown','trade_status']))
        body.append(details(ep.shock_session+' 事件：完整图与逐日明细',content,ep.shock_session=='2020-02-27'))
    body.append('</section><section id="execution"><h2>把入场与退出分开</h2><p>原信号不变。令原成交为 e，原退出为 u=e+H；四格分别为 (e,u)、(e+1,u)、(e,u+1)、(e+1,u+1)，实际持有 H、H−1、H+1、H 日。各自同资本从 e−1 现金起步，统一比较至 u+1，提前退出后持现金。</p><p>入场贡献 = ½[(R10−R00)+(R11−R01)]；退出贡献 = ½[(R01−R00)+(R11−R10)]。两项严格加总为 R11−R00。交互项 R11−R10−R01+R00 已平分进入两项贡献，单列观察，不能重复加一次。表中收益是百分比，差值的单位是百分点。</p>')
    body.append('<h3>主 B／H10：七事件逐笔归因</h3>'+table(b10,['episode_id','R00','R10','R01','R11','total_change','entry_contribution','exit_contribution','interaction']))
    for h in (10,5,20):
        g=tables['four_cells'].query('H == @h')
        content=table(g,['episode_id','rule_id','cell','entry_session','exit_session','actual_holding_sessions','net_return','fees','entry_loss','peak_drawdown','net_drawdown','pre_execution_state','fill_day_final_state','trade_status'])
        content+=table(attr.query('H == @h'),['episode_id','rule_id','entry_first','entry_after_exit','exit_first','exit_after_entry','entry_contribution','exit_contribution','interaction','total_change'])
        body.append(details(f'H{h}：全部35个原信号的四格、费用、路径风险与两种变更顺序',content))
    body.append('<h3>同一入场：期限差异来自哪些持有区间</h3><p>固定原入场及相同初始份额，拆出入场→第5日、第5→10日、第10→20日的原持有损益。区间损益减实际退出费用差，等于两期限净收益的美元差；第一段还扣入场费。不把不同起点的区间百分比直接相加。</p>')
    body.append(details('全部35个信号：三个连续持有区间',table(tables['holding_segments'],['episode_id','rule_id','from_H','to_H','start_session','end_session','holding_pnl','extra_fees','net_return_change','identity_residual_dollars'])))
    body.append('</section><section id="availability"><h2>何时知道条件不再满足</h2><p>B／C只检查原条件；A首次冲击与机械等待没有持续确认条件，标为不适用。C包含每日同合约VX双降，所以“C不满足”不等价于VIX恐慌重新升级。缺失数据属于未知，不改写成不满足。</p>')
    info=tables['availability']
    affected=info.query('stage == "PRE_EXECUTION_CUTOFF" and lag == 2 and original_condition == "FALSE"')
    body.append('<h3>lag=2 成交前已观察不满足的原信号（只记录状态）</h3>'+table(affected,['episode_id','rule_id','information_session','entry_session','original_condition','execution_decision_at','historical_publication_proven']))
    body.append('<p>lag=1 成交前检查仍只能看到原信号信息日 t；t+1 最终日线属于成交后信息。lag=2 的截止可按原可得性约定看到 t+1，但这里的“可得”是历史研究假定，不是已经拿到发布日志。数据库下载时间不能补足这一证明。</p>')
    body.append(details('全部35信号×两种延迟×三个观察时点',table(info,['episode_id','rule_id','lag','stage','information_session','entry_session','original_condition','complete_new_daily_rows','execution_decision_at','state_observable_at_assumed_deadline','historical_publication_proven'])))
    body.append('<h3>持有期：确认之前已承担什么，之后又发生什么</h3><p>从入场当日完成数据起，记录首次原条件不满足的信息日 q，最早行动日为 q+1。累计损益按原账本，包含买入费用及等到该收盘的涨跌；“此后”仍是原计划持有结果，没有模拟提前清仓。行动日恰好等于原退出日时，没有可提前的剩余区间。首次不满足只在原退出日完成后可见时，行动已经晚于原退出，前后切分留空。低波重置单独标记，不作为新退出条件。</p>')
    cols=['episode_id','rule_id','first_false_session','earliest_action_session','pnl_at_information','pnl_through_earliest_action','remaining_original_pnl','final_pnl','false_after_reset','status']
    for h in (10,5,20):
        g=tables['holding_outcomes'].query('H == @h and condition_applicable')
        body.append(details(f'H{h}：B／C全部14笔状态与原账本损益',table(g,cols),h==10))
    evidence=[['同日收盘','最终VIX及同合约VX信号须在订单截止前发布并实际收到；若使用未最终值，须锁定该版本定义','历史信号发布／接收日志、VX初值及修订版本、订单截止与同日成交报价','未取得；未模拟'],
              ['盘中','锁定盘中VIX和VX可交易／结算字段；不能把日终最终值放到盘中','带时间戳的信号快照、收取延迟、成交盘口及成本','未取得；未模拟'],
              ['次日开盘','前日日终信号须在开盘订单截止前实际可得','历史发布及接收时间、VX结算版本、可执行开盘价／竞价及费用','未取得；未模拟']]
    body.append('<h3>更快执行尚缺的证据</h3>'+table(pd.DataFrame(evidence,columns=['方式','必须锁定的信息时点','所需证据','当前状态'])))
    body.append('</section><section id="validation"><h2>实际验证与可复算文件</h2>')
    body.append(f'<p>计算完成：{validation["audited_accounts"]} 个独立路径账户、{validation["audited_rows"]} 行逐日份额／现金核账；最大金额误差 {validation["max_account_error"]:.3g} 美元。105笔原执行与35笔既有H10延迟结果均对齐。真实数据未下载或替换，原输入摘要全部保持一致。</p>')
    body.append(table(tables['original_reconciliation']))
    body.append(f'<p>本次运行：<code>{escape(str(run.relative_to(root)))}</code>。普通测试、因果扰动和两次复算的最终记录见下方独立交付验证；运行内 validation.json 只陈述该次计算验证，不冒称完成尚未执行的验收。</p>')
    body.append('<p class="note">页面展示限制：本会话的浏览器工具此前已按安全策略拒绝本地HTML。未改用其他入口绕过。报告由macOS打开的结果单独记录；图表PNG可直接目视检查，HTML资源和锚点可静态核验。整页截图、折叠交互及完整页面布局尚未由工具核验。</p>')
    files=sorted(out.glob('*.csv'))+sorted(out.glob('*.json'))+[run/'experiment_record.json']
    body.append('<div class="downloads">'+''.join(f'<a href="{link(p)}">{escape(p.name)}</a>' for p in files)+'</div>')
    body.append(f'<p><a href="{link(root/"runs/panic_retreat_diagnostic/final_validation.json")}">最终交付验证记录</a> · <a href="{link(root/"runs/panic_retreat_diagnostic/reproducibility.json")}">两次复算对照</a> · <a href="{link(root/"reports/panic_retreat.html")}">原E1报告（保持原样）</a></p>')
    body.append('<p>运行：<code>.venv/bin/python -m svxylab.panic_retreat_diagnostic --open</code>。每次新建运行目录；原E1入口、模型M2、配置及旧报告不变。新增重叠候选不做独立日抽样，也没有新的显著性主张。</p></section>')
    body.append('<section id="cards"><h2>三张证据卡：平列，等待选择</h2><p>三者解决不同问题。本轮不排序、不推荐冠军，不把多个修改混合后的收益归功于同一机制。2020反例保留。</p><div class="cards">')
    for card in mechanism_cards:
        body.append('<article><h3>'+escape(card['title'])+'</h3>'+''.join(f'<h4>{label}</h4><p>{escape(card[key])}</p>' for key,label in [('support','支持证据'),('against','反对证据与代价'),('clock','可得时点与证据缺口'),('next_question','下一实验必须锁定的问题')])+'</article>')
    body.append('</div><p class="stop">诊断终点已到：等待你选择机制及验收。本轮没有执行机制修改、开始下一实验、提交或推送。</p></section>')
    style='''body{margin:0;background:#f4f5f3;color:#213343;font:16px/1.65 -apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif}main{max-width:1280px;margin:auto;padding:28px}header,section{background:white;border:1px solid #dfe5e5;border-radius:12px;padding:28px;margin:18px 0}header{background:#18384d;color:#f5f8f8}h1{font-size:34px;line-height:1.3}h2{font-size:25px;margin:0 0 20px}h3{font-size:20px}h4{font-size:15px;margin-bottom:6px;color:#507080}.eyebrow{font-size:13px;letter-spacing:.1em;color:#bad3d7}.lead{max-width:920px;color:#d7e2e9}.numbers{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-top:28px}.numbers strong{display:block;font-size:28px;color:#a4d1c9}nav{display:flex;gap:20px;flex-wrap:wrap;padding:14px}a{color:#246c7b;text-decoration:none}a:hover{text-decoration:underline}.note,.stop{padding:18px;background:#f3f5ee;border-left:4px solid #a3b38b}.chart{width:100%;height:auto}.table-scroll{overflow:auto;max-height:640px;border:1px solid #e1e7e8;margin:14px 0}table{border-collapse:collapse;width:100%;font-size:13px;white-space:nowrap}th{position:sticky;top:0;background:#eaf0f2;color:#314f62;text-align:left}td,th{padding:9px 12px;border-bottom:1px solid #e3e8eb}tr:nth-child(even){background:#f8fafb}details{border:1px solid #dce4e7;border-radius:7px;margin:14px 0;padding:12px}summary{cursor:pointer;font-weight:600;color:#275b72;padding:7px}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:18px}.cards article{border:1px solid #d7e1e4;border-top:4px solid #648a91;border-radius:8px;padding:20px}.cards p{font-size:14px}.downloads{display:flex;flex-wrap:wrap;gap:8px 18px;font-size:13px}code{overflow-wrap:anywhere;font-size:13px}@media(max-width:800px){main{padding:8px}header,section{padding:18px}.cards{grid-template-columns:1fr}.numbers{grid-template-columns:repeat(2,1fr)}h1{font-size:27px}}'''
    report.write_text('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>E1补充诊断 · SVXYLab</title><style>'+style+'</style></head><body><main>'+''.join(body)+'</main></body></html>')
    return report
