"""可直接打开的N1中文报告；不覆盖P7交付快照。"""
from datetime import datetime, timezone
from pathlib import Path
import base64
import html
import io
import json
import os
import subprocess
import time
import traceback
import pandas as pd
from svxylab.capital import run_capital, NAMES, PERIOD_NAMES
from svxylab.capital_audit import audit_capital
from svxylab.features_report import write_json, verify_hashes
from svxylab.release import read_json, digest


def pct(x):return f'{100*x:+.2f}%'
def money(x):return f'${x:,.2f}'
def number(x):return f'{x:,.0f}'
def esc(x):return html.escape(str(x))

def table(rows,headers):
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+esc(h)+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(v)+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'

def image_chart(root,output,ledgers,period,names,filename):
    os.environ.setdefault('MPLCONFIGDIR',str(root/'.cache/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font=FontProperties(fname='/System/Library/Fonts/STHeiti Light.ttc')
    fig,axes=plt.subplots(2,1,figsize=(11,6.2),sharex=True,gridspec_kw={'height_ratios':[2,1]})
    colors=['#a73935','#1b677c','#b18724','#446f4c','#8a659a','#688398']
    for name,color in zip(names,colors):
        a=ledgers[period,name];dates=pd.to_datetime(a.as_of_session)
        axes[0].plot(dates,a.equity_end/1000,label=NAMES[name],color=color,linewidth=1.5)
        axes[1].plot(dates,100*a.drawdown,color=color,linewidth=1.15)
    axes[0].set_title(PERIOD_NAMES[period]+' · 同起始10万美元连续净账本',fontproperties=font)
    axes[0].set_ylabel('资产（千美元）',fontproperties=font)
    axes[1].set_ylabel('收盘回撤（%）',fontproperties=font)
    axes[0].legend(prop=font,fontsize=8,ncols=2,loc='best')
    for ax in axes:ax.grid(alpha=.15);ax.spines[['top','right']].set_visible(False)
    fig.autofmt_xdate();fig.tight_layout();path=output/filename;fig.savefig(path,dpi=140)
    plt.close(fig)
    return '<img alt="'+esc(PERIOD_NAMES[period])+'净值和回撤" src="data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode()+'">'


def render_capital(root,output,metrics,ledgers,inputs,audit):
    def get(period,name):return metrics.loc[metrics.period.eq(period)&metrics.strategy.eq(name)].iloc[0]
    p='revealed_2024';m=get(p,'M2_main');spy=get(p,'SPY_100');bil=get(p,'BIL_100')
    f5=get(p,'fixed_5_BIL');f10=get(p,'fixed_10_BIL');risk=get(p,'M2_risk_only')
    relative='../'+str(output.relative_to(root))+'/'
    def link(path,label):return '<a href="'+relative+path+'">'+esc(label)+'</a>'
    parts=['''<header><div class="eyebrow">SVXYLAB · N1 · 2026-09-07</div><h1>先问资本是否值得承担风险</h1>
    <p class="lead">资本与经济基准 · 真实连续账本 · 保护性看跌的数据准备</p>
    <div class="notice"><b>研究属性：</b>2024—2026 已在 P7 揭示。本轮是揭示后的经济基准比较与路径归因，不是新封存检验、未来收益证明或资金使用授权。</div></header>''']
    parts.append(f'''<section><h2>目前的答案</h2><p><b>现有证据不支持为 M2 主映射承担当前这一级风险。</b>
    在 2024-01-02 至 2026-09-04，同起始10万美元，M2主映射亏损 {money(-m.total_return*100000)}，净收益 {pct(m.total_return)}，最大收盘回撤 {abs(m.max_drawdown)*100:.2f}%，截至期末仍未恢复。同期 BIL 代理净收益 {pct(bil.total_return)}，M2少 {abs(m.excess_over_bil_pp):.2f} 个百分点；SPY总回报账户为 {pct(spy.total_return)}、回撤 {abs(spy.max_drawdown)*100:.2f}%。</p>
    <p>固定5%/10% SVXY、其余配置BIL，分别为 {pct(f5.total_return)} / {pct(f10.total_return)}，比全BIL多 {f5.excess_over_bil_pp:.2f} / {f10.excess_over_bil_pp:.2f} 个百分点，仍需承受 {abs(f5.max_drawdown)*100:.2f}% / {abs(f10.max_drawdown)*100:.2f}% 的最大收盘回撤。<b>这是小额风险承担的描述性结果，尚不足以证明这点超额回报在未来值得。</b>没有用这些结果挑新权重、改规则或晋升挑战者。</p>
    <p>SPY是广义机会成本对照，其资产风险不同，不能把其历史胜出称为严格风险匹配后的模型增量。BIL是短债ETF市场总回报代理，有交易费用和价格波动；不是无风险常数，也不是用户实际拿到的现金利息。</p></section>''')
    primary=['M2_main','M2_risk_only','cash','fixed_5','fixed_10','fixed_5_BIL','fixed_10_BIL','pool_5_BIL','pool_10_BIL','fixed_50','SPY_100','BIL_100']
    def performance(frame):
        return table([[esc(r.name_zh),pct(r.total_return),f'{r.excess_over_bil_pp:+.2f}',f'{abs(r.max_drawdown)*100:.2f}%',pct(r.worst_five_days),f'{r.mean_svxy_exposure*100:.2f}%',money(r.cost_dollars)] for r in frame.itertuples()],
           ['账户','净收益','超BIL（百分点）','最大回撤','最差5日','平均SVXY敞口','交易成本'])
    parts.append('<section><h2>同资本、同日期、同收盘执行：已揭示段</h2>'+performance(metrics.loc[metrics.period.eq(p)].set_index('strategy').loc[primary].reset_index()))
    parts.append('<p class="muted">平均敞口取每个收益区间开始时实际持仓；首个执行日尚未入场，旧敞口为0。SPY/BIL的SVXY敞口均为0，各自资产敞口见完整表。费用已从净值扣除，不应再次扣除。</p>')
    parts.append(image_chart(root,output,ledgers,p,['M2_main','SPY_100','fixed_50','BIL_100'],'revealed_accounts.png'))
    parts.append(image_chart(root,output,ledgers,p,['fixed_5_BIL','fixed_10_BIL','pool_5_BIL','pool_10_BIL','BIL_100'],'revealed_small_allocations.png')+'</section>')
    parts.append('''<section><h2>仓位比例与风险池：两个不同问题</h2>
    <p><b>固定比例再平衡：</b>每日收盘把SVXY恢复为当时总资产的5%或10%。跌后可能从现金/BIL买入补足，涨后卖出；全过程自融资，但储备会反复承担损失。因此5%仓位不是累计损失或最大回撤上限。若连续多期都损失一半，忽略费用的固定10%账户每期乘0.95，三期累计亏损14.26%；这是代数示例，不是模拟行情回测。</p>
    <p><b>严格不追加资金的风险池：</b>最初只划出5,000或10,000美元，首次费用也从该池支付；池内买入持有SVXY，储备另持现金0或BIL。没有外部入金，没有储备向风险池转账，也不恢复最初比例。盈利留在池内，SVXY比重自然漂移。</p>
    <p>在仅持多头、无借贷/额外负债且储备为零收益现金的定义下，风险池本身损失不会超过最初划入金额。但<b>整个账户从后来盈利高点计算的回撤仍可超过5%/10%</b>；BIL储备自身也可能亏损。初始本金损失边界、峰谷回撤和未来花费的保护权利金必须分别计算。</p>''')
    pool_rows=[]
    for period in ['since_2019','revealed_2024','continuous_common']:
        for name in ['pool_5_cash','pool_10_cash','pool_5_BIL','pool_10_BIL']:
            r=get(period,name);l=ledgers[period,name]
            pool_rows.append([esc(PERIOD_NAMES[period]),esc(NAMES[name]),pct(r.total_return),f'{abs(r.max_drawdown)*100:.2f}%',
              f'{r.mean_svxy_exposure*100:.2f}%',f'{l.new_weight.iloc[-1]*100:.2f}%',money(l.risk_pool_equity.iloc[-1]),'$0'])
    parts.append(table(pool_rows,['区间','账户','净收益','最大回撤','平均SVXY','期末SVXY','期末风险池','储备补资'])+'</section>')
    parts.append('''<section><h2>需要忍受的路径与等待</h2><p>回撤只在每日收盘观测，不代表日内最差损失或可退出价格。最差1日/5日以连续账户净值计算，包含该期实际费用；不是把五日标签交易叠加。恢复时间从峰值到首次重新达到峰值，另列谷底到恢复；“未恢复”按冻结日右删失，不能当成已恢复用时。</p>''')
    for period in ['revealed_2024','since_2019','development_common','continuous_common']:
        frame=metrics.loc[metrics.period.eq(period)]
        wanted=['M2_main','M2_risk_only','fixed_5','fixed_10','fixed_5_BIL','fixed_10_BIL','SPY_100','BIL_100']
        rows=[]
        for r in frame.loc[frame.strategy.isin(wanted)].itertuples():
            recovered=esc(r.max_dd_recovered) if pd.notna(r.max_dd_recovered) else ('未恢复至2026-09-04' if r.max_dd_censored else '无回撤')
            sessions=f'{r.max_dd_recovery_sessions}/{r.max_dd_trough_recovery_sessions}'
            if r.max_dd_censored:sessions+='（观察下限）'
            rows.append([esc(r.name_zh),f'{pct(r.worst_day)}<br>{r.worst_day_session}',f'{pct(r.worst_five_days)}<br>{r.worst_five_start} → {r.worst_five_end}',
              f'{r.max_dd_peak} → {r.max_dd_trough}',recovered,sessions,f'{r.max_dd_recovery_calendar_days}天',
              f'{r.longest_underwater_sessions}'+('（未恢复）' if r.longest_underwater_censored else '')])
        parts.append('<details '+('open' if period=='revealed_2024' else '')+'><summary>'+esc(PERIOD_NAMES[period])+'</summary>'+table(rows,
            ['账户','最差单日/日期','最差5日/区间','最大回撤峰 → 谷','恢复日期','峰/谷至恢复（交易日）','峰至恢复/期末（日历）','最长水下（交易日）'])+'</details>')
    parts.append('<p>2019起的基准包含2020冲击；模型首个合资格日是2020-09-11，不能用模型当时尚无服务的现金起点声称避开了2020崩盘。新5%/10%＋BIL在2019起全段的最大回撤分别为 '+f'{abs(get("since_2019","fixed_5_BIL").max_drawdown)*100:.2f}% / {abs(get("since_2019","fixed_10_BIL").max_drawdown)*100:.2f}%，峰至恢复分别377/410个交易日。'+'</p></section>')
    attribution=pd.read_csv(output/'attribution.csv');paths=pd.read_csv(output/'holding_intervals.csv')
    states={'held_up':'持有且SVXY上涨','held_down':'持有且SVXY下跌','flat_up':'空仓且SVXY上涨','flat_down':'空仓且SVXY下跌','initial_execution':'首次执行费用'}
    own=attribution.loc[attribution.period.eq(p)&attribution.against.eq('fixed50')]
    up=own.loc[own.state.eq('held_up'),'actual_holding_pnl_dollars'].sum();down=own.loc[own.state.eq('held_down'),'actual_holding_pnl_dollars'].sum()
    parts.append(f'''<section><h2>M2的损害从哪里来</h2><p>原账本的持有上涨日合计贡献 {money(up)}，持有下跌日合计 {money(down)}，再扣 {money(m.cost_dollars)} 交易费用，精确得到期末净损益 {money(m.ending_equity-100000)}。这些是连续账户的美元流水加总，不是独立满仓交易，也不是最大回撤。</p>
    <p>相对risk-only，主映射少赚 {100*(risk.total_return-m.total_return):.2f} 个百分点。下表按<b>当天损益发生前的实际持仓</b>分类；即使当天收盘已经退出，该天此前的损失仍归持有。空仓期间可以避免下跌，也会错过反弹，而且零收益现金落后于BIL。</p>''')
    rows=[]
    for state in states:
        row=[states[state]]
        for comparison in ['risk_only','fixed50','BIL']:
            v=attribution.loc[attribution.period.eq(p)&attribution.against.eq(comparison)&attribution.state.eq(state),'terminal_difference_pp'].sum()
            row.append(f'{v:+.3f}')
        rows.append(row)
    rows.append(['合计',f'{100*(m.total_return-risk.total_return):+.3f}',f'{100*(m.total_return-get(p,"fixed_50").total_return):+.3f}',f'{m.excess_over_bil_pp:+.3f}'])
    parts.append(table(rows,['路径类型','对risk-only期末差（百分点）','对固定50%期末差','对BIL期末差']))
    parts.append('''<p class="muted">这是精确复利归因：每天差异先按对照当日期初资本计金额，再按M2后续实际增长传递至期末。大额正负项可以相互抵消；它们不是可独立获得的机会收益，也不意味着某路径有可重复的因果作用。公式为 Σ[E对照,日初 × (rM2−r对照) × M2后续增长倍数]，合计严格等于期末财富差；成本包含在各日内。</p>''')
    selected=[]
    for state in ['held','flat']:
        subset=paths.loc[paths.period.eq(p)&paths.state.eq(state)].sort_values('net_pnl_dollars' if state=='held' else 'terminal_delta_vs_risk_only')
        for label,group in [('最不利',subset.head(2)),('最有利',subset.tail(2).iloc[::-1])]:
            for r in group.itertuples():
                selected.append([('持有' if state=='held' else '空仓')+' · '+label,f'{r.first} → {r.last}',str(r.sessions),pct(r.svxy_path_return),pct(r.path_return),money(r.net_pnl_dollars),money(r.terminal_delta_vs_risk_only)])
    parts.append(table(selected,['路径（事后展示）','区间','天数','SVXY同期','M2区间净收益','M2实际净损益','对risk-only期末桥接']))
    parts.append('<p>持有区间按M2自身实际净损益展示最大正负各两项；空仓区间按对risk-only的期末差展示最大正负各两项。这两种排序分别回答“实际亏在哪里”和“过滤错过/避免了什么”，均为事后归因。2024-07-19至08-05，M2连续持有路径亏损38,872.47美元（−32.69%）；随后08-06至08-12空仓时SVXY反弹24.68%，对risk-only期末差再贡献−15,693.39美元。2025-03-26至04-25另一持有路径实际亏损13,658.02美元（−18.59%）。风险容量限制未阻止这些实际损失，收益过滤也未捕获随后的完整修复；这些路径不用于重选规则。</p>')
    parts.append('<p>空仓路径最后一天可以在收盘重新买入，因此会有入场费用，却不取得当天入场前的反弹收益。全部持有/空仓区间均保留，正反两类案例各展示最大两项，仅作归因，未据此改动收益过滤或仓位规则。'+link('holding_intervals.csv','全部连续区间')+' · '+link('attribution.csv','全部区间汇总')+'</p>')
    # 原同目标毛净，保持原预测和过滤，不把变成本场景当成纯费用。
    old_gross=pd.read_csv(root/inputs['p7_dir']/'sensitivity/same_target_gross_net.csv')
    old_m=old_gross.loc[old_gross.cohort.eq('common')&old_gross.strategy.eq('M2_main')].iloc[0]
    parts.append(f'<p>原P7同一目标的毛收益为 {pct(old_m.same_target_gross_return)}、净收益 {pct(old_m.net_return)}；费用对最终复利收益的拖累约 {100*(old_m.same_target_gross_return-old_m.net_return):.2f} 个百分点。毛收益仍为负，问题不能全归咎于成本。该比较保留同一组目标，与改变2c过滤的零成本敏感性不同。</p></section>')
    cash=inputs['cash_proxy']
    parts.append(f'''<section><h2>现金代理究竟是什么</h2><p>选择BIL，是因为它在2007年成立、投资1—3个月美国国库券并按月分配，覆盖本轮2019起点；该选择在N1计算前记录。<a href="{cash['fund_document']}">State Street基金资料</a></p>
    <p>本轮实际取得Yahoo Chart原件，2019-01-02至2026-09-04共 <b>{cash['rows']}日</b>，{cash['distribution_events']}项现金分配事件、{cash['split_events']}项拆分，与原SVXY/SPY股票日历逐日一致。没有用现在的利率倒填。按原始收盘价和实际分配重建总回报；与同供应商复权参考的最大单日差 {cash['max_daily_difference_to_provider_adjusted_bps']:.5f} bp，全段总回报差 {100*(cash['full_tr_return']-cash['provider_adjusted_reference_return']):+.5f} 个百分点，保留差异。<a href="{html.escape(cash['source_url'],quote=True)}">Yahoo原请求</a> · {link('inputs/BIL_returns.csv','本地逐日复算')}</p>
    <p>统一沿用P2口径：分配在除息日确认，收盘再投资，并对实际再投资交易扣费。<b>这不是支付日到账的实盘现金路径</b>；官方区分除息日、登记日与支付日，故不能说用户已经获得这些利息。<a href="https://www.ssga.com/us/en/intermediary/resources/documents/etf-dividend-distributions">发行人分配说明</a>。BIL与SPY原始基金收益已含基金层费用，不再扣当前费率；本轮未建税后、汇率、账户利率或实际委托模型。Yahoo历史并非完整历史发布时间/版本库；正式分配全量发行人交叉核对仍未完成。</p>
    <p>所有账户从同一执行收盘买入，首日以前的收益不归账户；期末仅按市值计价、不强制卖出。单边5bp为统一成交代理；双ETF换仓两腿都收费。超BIL显示净收益相差的百分点，完整CSV另给期末财富比值，不能混称年化超额。</p></section>''')
    parts.append('''<section><h2>保护性看跌下一步必须过的简单对照</h2>
    <p><b>只比原M2更好不够。</b>保护研究必须同时对比全BIL、同资本的5%/10% SVXY＋BIL每日固定比例，以及严格不补资的5%/10%风险池；原现金/25/50/75/100%与SPY总回报对照继续保留。同预算未保护SVXY是测量保护本身代价的直接对照。</p>
    <p>在相同起点、成本、现金收益与终点口径下，保护方案需要证明：额外权利金、买卖价差、滚动和行权成本换来了值得的尾部损失/回撤/恢复时间改善；或在相近风险下保留更多超过BIL的净收益。不能通过比较不同总资本、不同SVXY美元头寸或忽略储备收益制造优势，也不能靠加大仓位挽救权利金拖累。本轮不选择期限、执行价、滚动规则或效用阈值。</p>
    <p>若保护费和后续加仓可不断从储备补给，就不再是本报告“不补资风险池”。整数张期权、每张实际交割股数和未覆盖零股必须逐笔记账。保护到期后的再建仓、跳空和流动性成本尚未检验，不能先声称存在总损失硬上限。</p></section>''')
    sources=read_json(root/'runs/n1/options_data_review.json')
    parts.append('''<section><h2>真实期权数据：文档已核实，SVXY样本尚未验收</h2>
    <p><b>已取得且验证的SVXY历史期权报价：0行。</b>没有购买、注册、联系供应商或生成假价。Cboe公开样本链接访问失败，一次只读产品页请求HTTP403后停止；这不表示供应商没有数据。当前不能计算保护权利金和经济收益。</p>''')
    parts.append(table([
      ['Cboe DataShop','Quotes / EOD当前产品页声明2012-01起；覆盖OPRA美股/ETF/指数期权。','分钟/自定义NBBO、size；EOD为15:45及收市。产品覆盖不等于SVXY某日某张合约有可交易报价。','<a href="https://datashop.cboe.com/option-quote-intervals">Quotes</a> · <a href="https://datashop.cboe.com/option-eod-summary">EOD</a>'],
      ['OptionMetrics IvyDB US','EOD声明1996起；日内产品声明2018起。','日内仅10:00/14:00/15:45三个快照；缺少本轮已验收SVXY样本和客户字段手册。','<a href="https://optionmetrics.com/united-states/">EOD</a> · <a href="https://optionmetrics.com/united-states-intraday/">Intraday</a>']],['供应商','文档覆盖声明','本项目尚缺/不能推定','来源']))
    parts.append('''<p>Cboe的quote_datetime表示区间结束的美国东部时间，成交OHLC为零可能只是没有交易；IV/Greeks不能替代bid/ask。日内文件交付延迟15分钟，早收市某些16:00标签实际对应13:00收市；2026-06-22起size改为最近价格变化时的数量。不能把这些时点当成15:00已经可得的输入。逐条最后报价更新时间及严格陈旧度字段仍待确认。<a href="https://datashop.cboe.com/documents/Option_Quotes_Layout.pdf">字段说明</a> · <a href="https://datashop.cboe.com/option-quote-intervals">时点/数量政策</a></p>''')
    parts.append(table([
     ['合约身份','永久标的/合约ID、历史root、put标记、原始执行价、真实到期日/最后交易日、上市退市日期、标准/调整合约标志'],
     ['可执行报价','事件时间/时区/DST、bid/ask及各自size、报价条件、更新时间/陈旧度、同期标的未复权价格及NBBO；成交量和OI的实际可得日'],
     ['调整与交割','权利金美元乘数、合约数量倍数、每张交割股数/现金、执行价变化、有效日/OCC公告、美式行权及历史交割周期'],
     ['来源与许可','原件、请求范围、下载时间、摘要、字段/修订版本、本地研究与保存/处理/报告发布权限']],['必须有的真实字段','验收内容']))
    parts.append('''<p><b>两次SVXY调整不能只靠复权股价处理：</b>2024-04-11两拆一，OCC规定合约数量×2、执行价÷2，每张仍100股、权利金美元乘数仍100；2018-09-18反拆遗留SVXY1合约交割25股但美元乘数仍100，需查明是否存续到2019。研究从2019开始不等于可以忽略其历史身份。<a href="https://infomemo.theocc.com/infomemos?number=54366">OCC #54366</a> · <a href="https://infomemo.theocc.com/infomemos?number=43647">OCC #43647</a></p>
    <p>标准ETF期权为美式、实物交割；历史交割周期在2024-05-28由T+2变T+1。不能把当前规则回填所有年份。<a href="https://www.theocc.com/clearance-and-settlement/clearing/etf-options">OCC规格</a> · <a href="https://infomemo.theocc.com/infomemos?number=53901">转换公告</a></p>
    <p><b>许可仍是待完成项：</b>Cboe内部研究、保存与再分发受产品/订单/合同约束，可还原报价的派生表也需核对；购买不自动授予上传GitHub或第三方云端处理的权利。OptionMetrics网站条款不是IvyDB客户数据合同，本轮未确认其客户许可，也未执行许可接受。<a href="https://datashop.cboe.com/documents/Cboe_LiveVol_DataShop_License_Agreement.pdf">Cboe许可</a> · <a href="https://datashop.cboe.com/documents/DataShop_Policies_for_Historical_Data_Services.pdf">历史数据政策</a> · <a href="https://optionmetrics.com/terms-of-use/">OptionMetrics网站条款</a></p>
    <p>下一阶段先验收真实样本：2019起点/可能遗留合约、2020压力、2024拆股和交割转换、2026数量口径变化及冻结末日。逐日列出完整链覆盖、价差、零bid、单边/交叉/陈旧报价与缺口原因，验证合约交割价值连续，明确整数张数和未覆盖零股。之后才可预先记录保护规则并研究，2024—2026仍必须标为已揭示历史。</p></section>''')
    parts.append('<section><h2>完整比较与下载</h2><p>没有只留下赢家。下列四种区间各自同资本重建；“连续共同段”从2020-09-11起持有到2026-09-04，中间不在2024清零或重新入金。分段收益不能直接相加。2019首个信息日为1月2日、执行从1月3日，未读取2018行情。</p>')
    for period in PERIOD_NAMES:
        subset=metrics.loc[metrics.period.eq(period)]
        parts.append('<details><summary>'+esc(PERIOD_NAMES[period])+' · '+str(len(subset))+'个账户</summary>'+performance(subset))
        parts.append(table([[esc(r.name_zh),pct(r.cagr),f'{r.mean_svxy_exposure*100:.2f}%',f'{r.mean_spy_exposure*100:.2f}%',f'{r.mean_bil_exposure*100:.2f}%',f'{r.average_cash_weight*100:.2f}%',f'{r.total_turnover:.2f}倍',link(r.ledger_file,'逐日账本')] for r in subset.itertuples()],
            ['账户','CAGR','平均SVXY','平均SPY','平均BIL','平均现金0','累计成交额/当日净值','明细'])+'</details>')
    parts.append('<p>'+link('metrics.csv','全部指标CSV')+' · '+link('drawdown_episodes.csv','全部回撤与恢复区间')+' · '+link('input_verification.json','输入与真实数据摘要')+' · '+link('independent_validation.json','独立方程核账')+' · '+link('original_ledger_parity.json','原P5/P7账本逐日对照')+'</p></section>')
    parts.append(f'''<section><h2>工程、研究结果和数据覆盖分别交付</h2>
    <p><b>工程：</b>日报截止修复通过已有真实输入反例的三个合成时钟边界：晚100毫秒和恰好截止都不发新预测；早1毫秒正常生成，时间为推理及元数据完成之后，9个预测头与原预报完全相同。历史预测/模型/标签/持仓规则/P7产物和首条前向预报摘要保持不变。N1独立方程核对 {audit['accounts']} 个账户、{audit['account_rows_including_anchors']:,} 行（含锚点），最大绝对数值差 {audit['maximum_absolute_numerical_difference']:.3g}；两资产两腿费用、风险池不补资、恢复删失、复利归因均核对。</p>
    <p><b>研究：</b>原M2负结果保留；5%/10%仅为本轮预定研究规模，没有优化。原开发段M2表现较好不能抵消已揭示段损害。N1没有新增显著性或获利概率声明，旧P7不确定性结果继续保留；本轮事后比较不能提供新的独立统计验证。</p>
    <p><b>真实数据：</b>SVXY/SPY复用原1930日冻结总回报，BIL新增真实1930日；历史报价可用性仍存在次交易日可得假设，非完整PIT。期权真实报价0行，样本和许可待补。日报未来新交易日下载分支仍待实际前向检验；Q90/P10一致性限制未改。</p>
    <p>运行入口：<code>.venv/bin/python -m svxylab capital --open</code>，或双击项目内“资本与经济基准.command”。每次输出新目录，保留原P7报告和本轮失败/修正记录。<a href="../runs/n1/experiment_record.json">计算前记录</a> · <a href="../runs/n1/deadline_after.json">截止回归</a> · <a href="../runs/n1/options_data_review.json">期权资料明细</a> · <a href="latest.html">原P7报告快照</a></p>
    <div class="notice"><b>N1交付后停止，等待用户验收。</b>不连接交易账户，不发订单，不购买期权数据，不启动保护回测，不自动扩展策略。Git检查点在用户验收后再保存。</div></section>''')
    css='''*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#f5f6f4;color:#233438;font:16px/1.8 -apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif}main{max-width:1320px;margin:auto;padding:36px 30px 90px}header{padding:22px 0 28px}.eyebrow{color:#52716b;font-size:13px;letter-spacing:2px}h1{font-size:38px;line-height:1.35;margin:15px 0}h2{font-size:24px;line-height:1.5;margin:0 0 18px}.lead{font-size:19px;color:#617371}section{background:#fff;border:1px solid #dfe5df;border-radius:10px;padding:28px;margin:24px 0}p{margin:14px 0}.notice{background:#e8eee7;border-left:4px solid #617f66;padding:16px 20px}.muted{font-size:14px;color:#667672}a{color:#206678;text-underline-offset:3px}.scroll{overflow:auto}table{width:100%;border-collapse:collapse;font-size:13px;line-height:1.6}th,td{padding:10px 11px;border-bottom:1px solid #e0e5e2;text-align:left;vertical-align:top}th{background:#edf1ee;white-space:nowrap}td:first-child{min-width:135px}tbody tr:nth-child(even){background:#fafbf9}img{display:block;width:100%;height:auto;margin:22px 0}details{margin:14px 0;padding:12px 0;border-bottom:1px solid #d8e1dc}summary{cursor:pointer;font-weight:600;padding:8px 0}code{background:#edf1ee;padding:3px 6px;overflow-wrap:anywhere}@media(max-width:700px){main{padding:20px 12px}h1{font-size:29px}section{padding:18px 14px}h2{font-size:21px}table{min-width:820px}}'''
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SVXYLab N1 · 资本与经济基准</title><style>'+css+'</style><body><main>'+''.join(parts)+'</main></body></html>'
    (output/'report.html').write_text(page,encoding='utf-8')
    report=root/'reports/capital.html';report.write_text(page,encoding='utf-8')
    # 输出目录副本需同样支持数据链接。
    (output/'report.html').write_text(page.replace(relative,'./').replace('../runs/n1/','../../../../runs/n1/').replace('href="latest.html"','href="../../../../reports/latest.html"'),encoding='utf-8')
    return report


def build_capital_report(root, *, open_report=False):
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');start=time.perf_counter()
    directory=root/'runs/n1'/stamp;directory.mkdir(parents=True)
    output=root/'data/clean/n1'/stamp
    record={'stage':'N1','command':'.venv/bin/python -m svxylab capital'+(' --open' if open_report else ''),
      'started_at_utc':datetime.now(timezone.utc).isoformat(),'output_dir':str(output.relative_to(root)),
      'experiment_record_sha256':digest(root/'runs/n1/experiment_record.json'),
      'source_sha256':{str(p.relative_to(root)):digest(p) for p in [*sorted((root/'src/svxylab').glob('*.py')),root/'experiment.toml',root/'FEATURES.json',root/'requirements-lock.txt']}}
    write_json(directory/'started.json',record)
    try:
        metrics,ledgers,inputs=run_capital(root,output)
        audit=audit_capital(root,output,inputs)
        report=render_capital(root,output,metrics,ledgers,inputs,audit)
        preserved=read_json(root/'runs/n1/experiment_record.json')['preserved_sha256'];verify_hashes(root,preserved)
        record.update(status='ENGINEERING_COMPLETE_RESEARCH_DESCRIPTIVE_OPTIONS_DATA_PENDING',accounts=len(metrics),
          independent_audit={k:v for k,v in audit.items() if k!='records'},preserved_files_verified=len(preserved),
          report=str(report.relative_to(root)),report_sha256=digest(report),elapsed_seconds=time.perf_counter()-start,
          artifacts_sha256={str(p.relative_to(root)):digest(p) for p in output.rglob('*') if p.is_file()},exit_code=0)
        if open_report:
            completed=subprocess.run(['/usr/bin/open',str(report)],capture_output=True,text=True,timeout=20)
            record['open_command']={'args':['/usr/bin/open',str(report)],'exit_code':completed.returncode,'stderr':completed.stderr}
        write_json(directory/'capital_run.json',record);write_json(root/'runs/n1/latest_run.json',{'run_record':str((directory/'capital_run.json').relative_to(root))})
        print(f'N1完成：{report}；{len(metrics)}个账户；原P7保留；等待验收',flush=True)
        return 0
    except Exception:
        (directory/'failure.txt').write_text(traceback.format_exc());record.update(status='FAILED',exit_code=1,elapsed_seconds=time.perf_counter()-start)
        write_json(directory/'capital_run.json',record);raise
