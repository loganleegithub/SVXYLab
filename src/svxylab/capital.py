"""N1资本基准与事后路径归因：复用原账本，不训练、不改控制器。"""
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from svxylab.economics import run_account, model_targets, account_metrics
from svxylab.ledger import baseline_targets
from svxylab.features_report import write_json
from svxylab.release import read_json

NAMES={
 'cash':'现金（收益0）','fixed_5':'SVXY固定5%＋现金0','fixed_10':'SVXY固定10%＋现金0',
 **{f'fixed_{w}':f'SVXY固定{w}%＋现金0' for w in [25,50,75,100]},
 'curve':'原期限结构对照','SPY_100':'SPY 100%总回报','BIL_100':'BIL 100%短债代理',
 **{f'{m}_{p}':f'{m} '+('主映射' if p=='main' else '仅风险容量') for m in ['M0','M1','M2'] for p in ['main','risk_only']},
 **{f'fixed_{w}_BIL':f'SVXY固定{w}%＋BIL' for w in [5,10]},
 **{f'pool_{w}_{s}':f'{w}%独立风险池＋'+('现金0' if s=='cash' else 'BIL') for w in [5,10] for s in ['cash','BIL']}}
PERIOD_NAMES={'since_2019':'2019起简单基准（含2020压力）','development_common':'原开发共同段',
 'revealed_2024':'2024—2026已揭示历史','continuous_common':'2020—2026连续共同段（不重置资本）'}


def combine_pool(stock, reserve, capital):
    """两个起始划定的自融资池相加；没有跨池转账。"""
    assert stock.as_of_session.tolist()==reserve.as_of_session.tolist()
    result=stock.copy()
    for col in ['equity_start','equity_before_trade','equity_end','old_cash','price_pnl','distribution_income','holding_pnl',
                'cash_interest','signed_trade','trade_notional','cost']:
        result[col]=stock[col]+reserve[col]
    result['old_weight']=stock.old_weight*stock.equity_start/result.equity_start
    result['new_weight']=stock.new_weight*stock.equity_end/result.equity_end
    result['pre_trade_weight']=stock.pre_trade_weight*stock.equity_before_trade/result.equity_before_trade
    result['bil_old_weight']=reserve.old_weight*reserve.equity_start/result.equity_start
    result['bil_new_weight']=reserve.new_weight*reserve.equity_end/result.equity_end
    result['cash_end']=stock.cash_end+reserve.cash_end
    result['old_bil_shares']=reserve.old_shares
    result['bil_shares_end']=reserve.shares_end
    result['bil_close']=reserve.close
    result['bil_split_factor']=reserve.split_factor
    result['svxy_signed_trade']=stock.signed_trade
    result['bil_signed_trade']=reserve.signed_trade
    result['svxy_cost']=stock.cost
    result['bil_cost']=reserve.cost
    result['target_weight']=np.nan  # 初始比例不是每日再平衡指令。
    result['signal_missing']=False
    result['portfolio_return']=result.equity_end/result.equity_start-1
    result['turnover']=result.trade_notional/result.equity_before_trade
    result['drawdown']=result.equity_end/result.equity_end.cummax()-1
    result['risk_pool_equity']=stock.equity_end
    result['reserve_equity']=reserve.equity_end
    result['reserve_to_risk_transfer']=0.0
    result['external_capital_flow']=0.0
    result['reason']='独立池不互转；初始费用包含于各池预算，分配只留在原池'
    return result


def fixed_with_bil(stock, bil, clock, start, weight, capital, cost):
    """每日固定两只ETF扣费后比例，两腿都计成交费用；只用于N1新基准。"""
    i=stock.as_of_session.tolist().index(start)-1
    a,b=stock.iloc[i:].reset_index(drop=True),bil.iloc[i:].reset_index(drop=True)
    t=clock.iloc[i:].reset_index(drop=True)
    if a.as_of_session.tolist()!=b.as_of_session.tolist():raise ValueError('双资产日期不符')
    shares=np.zeros(2);cash=float(capital);equity=float(capital);rows=[];w=np.array([weight,1-weight])
    for j in range(len(a)):
        closes=np.array([a.close.iloc[j],b.close.iloc[j]])
        factors=np.array([a.split_factor.iloc[j],b.split_factor.iloc[j]])
        divs=np.array([a.cash_dividend.iloc[j]+a.capital_gain_distribution.iloc[j],b.cash_dividend.iloc[j]+b.capital_gain_distribution.iloc[j]])
        old_shares=shares.copy();old_cash=cash;old_equity=equity
        old_values=shares*np.array([a.close.iloc[j-1],b.close.iloc[j-1]]) if j else np.zeros(2)
        shares=shares/factors
        income=float(shares@divs) if j else 0.;cash+=income
        values=shares*closes;pre=float(values.sum()+cash)
        if j:
            if t.execution_session.iloc[j-1]!=a.as_of_session.iloc[j]:raise ValueError('执行日期不一致')
            after=brentq(lambda v:v+cost*np.abs(w*v-values).sum()-pre,pre*(1-cost)/(1+cost),pre,xtol=1e-9)
            trades=w*after-values;fees=cost*np.abs(trades);shares=w*after/closes;cash=0.;equity=after
        else:trades=np.zeros(2);fees=np.zeros(2);equity=pre
        rows.append({'as_of_session':a.as_of_session.iloc[j],'is_anchor':j==0,
          'signal_information_session':None if not j else a.as_of_session.iloc[j-1],
          'decision_at':None if not j else t.decision_at.iloc[j-1],
          'execution_at':None if not j else t.execution_at.iloc[j-1],
          'equity_start':old_equity,'equity_before_trade':pre,'equity_end':equity,
          'old_weight':old_values[0]/old_equity,'bil_old_weight':old_values[1]/old_equity,
          'new_weight':shares[0]*closes[0]/equity,'bil_new_weight':shares[1]*closes[1]/equity,
          'target_weight':weight if j else np.nan,'old_cash':old_cash,'cash_end':cash,
          'old_shares':old_shares[0],'old_bil_shares':old_shares[1],
          'shares_end':shares[0],'bil_shares_end':shares[1],
          'svxy_signed_trade':trades[0],'bil_signed_trade':trades[1],
          'price_pnl':float(values.sum()-old_values.sum()),'distribution_income':income,
          'holding_pnl':pre-old_equity,'cash_interest':0.,'asset_return':a.simple_return.iloc[j],
          'cost':float(fees.sum()),'svxy_cost':fees[0],'bil_cost':fees[1],
          'signed_trade':float(trades.sum()),'trade_notional':float(np.abs(trades).sum()),
          'turnover':float(np.abs(trades).sum()/pre),'portfolio_return':equity/old_equity-1,
          'signal_missing':j==0,'ever_forecast_executed':True,
          'reason':'两资产每日固定扣费后比例，SVXY和BIL交易各收5bp'})
    result=pd.DataFrame(rows)
    result['drawdown']=result.equity_end/result.equity_end.cummax()-1
    return result


def recovery_metrics(ledger):
    values=ledger.equity_end.to_numpy();dates=ledger.as_of_session.tolist()
    episodes=[];peak=0;j=1
    while j<len(values):
        if values[j]>=values[peak]:peak=j;j+=1;continue
        begin=peak;k=j
        while k<len(values) and values[k]<values[begin]:k+=1
        end=k if k<len(values) else len(values)-1
        trough=j+int(np.argmin(values[j:end+1]))
        episodes.append({'peak':dates[begin],'trough':dates[trough],'recovery':dates[k] if k<len(values) else None,
          'end_observed':dates[end],'depth':float(values[trough]/values[begin]-1),
          'peak_to_recovery_or_end_sessions':end-begin,
          'trough_to_recovery_or_end_sessions':end-trough,
          'calendar_days':(pd.Timestamp(dates[end])-pd.Timestamp(dates[begin])).days,
          'censored':k==len(values)})
        if k==len(values):break
        peak=k;j=k+1
    if not episodes:
        return {'max_dd_peak':None,'max_dd_trough':None,'max_dd_recovered':None,'max_dd_recovery_sessions':0,
          'max_dd_trough_recovery_sessions':0,'max_dd_recovery_calendar_days':0,'max_dd_censored':False,
          'longest_underwater_sessions':0,'longest_underwater_censored':False,'underwater_sessions':0},episodes
    worst=min(episodes,key=lambda x:x['depth']);longest=max(episodes,key=lambda x:x['peak_to_recovery_or_end_sessions'])
    return {'max_dd_peak':worst['peak'],'max_dd_trough':worst['trough'],'max_dd_recovered':worst['recovery'],
      'max_dd_recovery_sessions':worst['peak_to_recovery_or_end_sessions'],
      'max_dd_trough_recovery_sessions':worst['trough_to_recovery_or_end_sessions'],
      'max_dd_recovery_calendar_days':worst['calendar_days'],'max_dd_censored':worst['censored'],
      'longest_underwater_sessions':longest['peak_to_recovery_or_end_sessions'],
      'longest_underwater_censored':longest['censored'],
      'underwater_sessions':int((values<np.maximum.accumulate(values)).sum())},episodes


def exact_attribution(left, right, capital):
    """D_T = Σ E_B,i-1 (r_A,i-r_B,i) Π_{j>i}(1+r_A,j)，精确复利桥接。"""
    a=left.iloc[1:].reset_index(drop=True);b=right.iloc[1:].reset_index(drop=True)
    assert a.as_of_session.tolist()==b.as_of_session.tolist()
    future=np.r_[np.cumprod((1+a.portfolio_return.to_numpy()[::-1]))[::-1][1:],1.]
    bridge=b.equity_start*(a.portfolio_return-b.portfolio_return)*future
    state=np.where(a.old_weight.gt(1e-12),np.where(a.asset_return.ge(0),'held_up','held_down'),
                   np.where(a.asset_return.ge(0),'flat_up','flat_down')).astype(object)
    state[0]='initial_execution'
    detail=pd.DataFrame({'as_of_session':a.as_of_session,'state':state,'actual_old_weight':a.old_weight,
        'svxy_return':a.asset_return,'actual_holding_pnl_dollars':a.holding_pnl,
        'actual_cost_dollars':a.cost,'daily_return_difference':a.portfolio_return-b.portfolio_return,
        'terminal_difference_dollars':bridge,'terminal_difference_pp':100*bridge/capital})
    assert abs(bridge.sum()-(a.equity_end.iloc[-1]-b.equity_end.iloc[-1]))<2e-6
    return detail


def holding_paths(ledger, fixed, risk, bil, capital):
    a=ledger.iloc[1:].reset_index(drop=True)
    bridges={k:exact_attribution(ledger,x,capital) for k,x in [('fixed50',fixed),('risk_only',risk),('BIL',bil)]}
    state=np.where(a.old_weight.gt(1e-12),'held','flat').astype(object);state[0]='initial_execution'
    group=pd.Series(state).ne(pd.Series(state).shift()).cumsum();rows=[]
    for _,indices in a.groupby(group).groups.items():
        lo,hi=min(indices),max(indices);block=a.loc[lo:hi]
        row={'state':state[lo],'first':block.as_of_session.iloc[0],'last':block.as_of_session.iloc[-1],
             'sessions':len(block),'equity_start':block.equity_start.iloc[0],'equity_end':block.equity_end.iloc[-1],
             'net_pnl_dollars':float(block.equity_end.iloc[-1]-block.equity_start.iloc[0]),
             'path_return':float(block.equity_end.iloc[-1]/block.equity_start.iloc[0]-1),
             'holding_pnl_dollars':float(block.holding_pnl.sum()),'cost_dollars':float(block.cost.sum()),
             'svxy_path_return':float(np.prod(1+block.asset_return)-1),'mean_exposure':float(block.old_weight.mean()),
             'held_up_pnl_dollars':float(block.loc[block.holding_pnl.ge(0),'holding_pnl'].sum()),
             'held_down_pnl_dollars':float(block.loc[block.holding_pnl.lt(0),'holding_pnl'].sum())}
        for k,d in bridges.items():row['terminal_delta_vs_'+k]=float(d.loc[lo:hi,'terminal_difference_dollars'].sum())
        rows.append(row)
    return pd.DataFrame(rows),bridges


def run_capital(root, output):
    from svxylab.capital_data import load_capital_inputs
    import tomllib
    cfg=tomllib.loads((root/'experiment.toml').read_text());spec=cfg['n1']
    prices,clock,curve,pred,inputs=load_capital_inputs(root)
    output.mkdir(parents=True);(output/'ledgers').mkdir();(output/'paths').mkdir();(output/'inputs').mkdir()
    write_json(output/'input_verification.json',inputs)
    prices['BIL'].to_csv(output/'inputs/BIL_returns.csv',index=False)
    write_json(output/'inputs/BIL_actions.json',inputs['bil_actions'])
    config=read_json(root/'runs/n1/experiment_record.json')
    capital=spec['capital'];cost=spec['cost_one_way_bps']/10000
    model,meta=model_targets(pred,prices['SVXY'].as_of_session.tolist(),cfg['portfolio']['loss_budget_base'],cfg['portfolio']['cost_one_way_bps_base']/10000)
    base=baseline_targets(curve,cfg['evaluation']['fixed_weight_baselines']+spec['fixed_svxy_weights'])
    all_metrics=[];attribution=[];path_summary=[];parity=[];all_ledgers={};episode_rows=[]
    for period,(start,end) in config['periods'].items():
        pr={s:p.loc[p.as_of_session.le(end)].reset_index(drop=True) for s,p in prices.items()}
        cl=clock.loc[clock.as_of_session.le(end)].reset_index(drop=True)
        index=pd.Index(pr['SVXY'].as_of_session,name='as_of_session');one=pd.Series(1.,index=index)
        targets={**base,**(model if period!='since_2019' else {})};ledgers={}
        for name,target in targets.items():
            ledgers[name]=run_account(pr['SVXY'],cl,target.reindex(index),start,capital,cost,metadata=meta.get(name))
        for symbol in ['SPY','BIL']:
            ledgers[symbol+'_100']=run_account(pr[symbol],cl,one,start,capital,cost)
        for weight in spec['fixed_svxy_weights']:
            w=round(weight*100)
            ledgers[f'fixed_{w}_BIL']=fixed_with_bil(pr['SVXY'],pr['BIL'],cl,start,weight,capital,cost)
            stock=run_account(pr['SVXY'],cl,one,start,weight*capital,cost)
            for reserve in ['cash','BIL']:
                rest=run_account(pr['BIL'],cl,one if reserve=='BIL' else one*0,start,(1-weight)*capital,cost)
                total=combine_pool(stock,rest,capital)
                if reserve=='cash':total[['bil_old_weight','bil_new_weight']]=0.
                ledgers[f'pool_{w}_{reserve}']=total
                # 组件保留便于独立核对风险池无补资。
                stock.to_csv(output/'ledgers'/f'{period}_pool_{w}_{reserve}_risk_component.csv',index=False)
                rest.to_csv(output/'ledgers'/f'{period}_pool_{w}_{reserve}_reserve_component.csv',index=False)
        cash_return=account_metrics(ledgers['BIL_100'],capital)['total_return']
        for name,ledger in ledgers.items():
            path=f'ledgers/{period}_{name}.csv';ledger.to_csv(output/path,index=False)
            metrics=account_metrics(ledger,capital);rec,episodes=recovery_metrics(ledger)
            metrics['average_cash_weight']=(ledger.old_cash/ledger.equity_start).iloc[1:].mean()
            bil_exposure=ledger.iloc[1:].get('bil_old_weight',pd.Series(0.)).mean() if name!='BIL_100' else metrics['mean_exposure']
            all_metrics.append({'period':period,'strategy':name,'name_zh':NAMES[name],'ledger_file':path,**metrics,**rec,
              'excess_over_bil_pp':100*(metrics['total_return']-cash_return),
              'relative_wealth_vs_bil':(1+metrics['total_return'])/(1+cash_return)-1,
              'mean_bil_exposure':bil_exposure,
              'mean_svxy_exposure':0. if name in ['SPY_100','BIL_100'] else metrics['mean_exposure'],
              'mean_spy_exposure':metrics['mean_exposure'] if name=='SPY_100' else 0.})
            episode_rows += [{'period':period,'strategy':name,**x} for x in episodes]
            all_ledgers[period,name]=ledger
        if period!='since_2019':
            paths,bridges=holding_paths(ledgers['M2_main'],ledgers['fixed_50'],ledgers['M2_risk_only'],ledgers['BIL_100'],capital)
            paths.insert(0,'period',period);paths.to_csv(output/'paths'/f'{period}_M2_intervals.csv',index=False)
            path_summary.append(paths)
            for against,d in bridges.items():
                d.to_csv(output/'paths'/f'{period}_M2_vs_{against}_daily.csv',index=False)
                totals=d.groupby('state')[['actual_holding_pnl_dollars','actual_cost_dollars','terminal_difference_dollars','terminal_difference_pp']].sum().reset_index()
                totals.insert(0,'against',against);totals.insert(0,'period',period);attribution.append(totals)
        # 同资本/起点逐日与已验收原账本比较；不以新回测替换旧文件。
        if period in ['development_common','revealed_2024']:
            prior=root/inputs['p5_dir'] if period=='development_common' else root/inputs['p7_dir']/'sensitivity'
            for name in list(base)[:6]+list(model)+['curve']:
                if name in ['fixed_5','fixed_10']:continue
                old=pd.read_csv(prior/'ledgers'/f'common_base_{name}.csv',float_precision='round_trip')
                new=ledgers[name]
                assert old.as_of_session.tolist()==new.as_of_session.tolist()
                cols=['equity_end','cost','old_weight','shares_end','cash_end','target_weight']
                np.testing.assert_allclose(old[cols],new[cols],rtol=2e-13,atol=2e-8,equal_nan=True)
                parity.append({'period':period,'strategy':name,'rows':len(old),'max_equity_difference':float((old.equity_end-new.equity_end).abs().max())})
    metrics=pd.DataFrame(all_metrics);metrics.to_csv(output/'metrics.csv',index=False)
    pd.DataFrame(episode_rows).to_csv(output/'drawdown_episodes.csv',index=False)
    pd.concat(attribution,ignore_index=True).to_csv(output/'attribution.csv',index=False)
    pd.concat(path_summary,ignore_index=True).to_csv(output/'holding_intervals.csv',index=False)
    write_json(output/'original_ledger_parity.json',parity)
    return metrics,all_ledgers,inputs
