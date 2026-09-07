"""N1独立方程核账；不调用交易生成函数。"""
import numpy as np
import pandas as pd
from svxylab.features_report import write_json


def audit_capital(root,output,inputs):
    from svxylab.release import stage_output
    p2=stage_output(root,'p2')
    prices={s:pd.read_csv(p2/f'{s}_returns.csv',float_precision='round_trip').set_index('as_of_session') for s in ['SVXY','SPY']}
    prices['BIL']=pd.read_csv(output/'inputs/BIL_returns.csv',float_precision='round_trip').set_index('as_of_session')
    metrics=pd.read_csv(output/'metrics.csv');records=[];maximum=0.
    def close(a,b):
        nonlocal maximum
        a,b=np.asarray(a,float),np.asarray(b,float)
        maximum=max(maximum,float(np.max(np.abs(a-b))))
        np.testing.assert_allclose(a,b,rtol=3e-13,atol=2e-7)
    def single(path,symbol):
        a=pd.read_csv(path,float_precision='round_trip');p=prices[symbol].loc[a.as_of_session]
        old=a.old_shares.to_numpy();split=p.split_factor.to_numpy();px=p.close.to_numpy()
        income=old/split*(p.cash_dividend+p.capital_gain_distribution).to_numpy()
        income[0]=0.
        stock=old/split*px
        before=a.old_cash.to_numpy()+stock+income
        trades=a.shares_end.to_numpy()*px-stock
        fees=.0005*np.abs(trades)
        close(a.distribution_income,income);close(a.cost,fees);close(a.equity_before_trade,before)
        close(a.cash_end,a.old_cash+income-trades-fees)
        close(a.equity_end,a.shares_end*px+a.cash_end)
        close(a.equity_end,before-fees)
        close(a.old_weight,np.r_[0.,a.shares_end.to_numpy()[:-1]*px[:-1]]/a.equity_start)
        return a
    for row in metrics.itertuples():
        path=output/row.ledger_file
        if row.strategy.startswith('pool_'):
            risk=single(path.with_name(path.stem+'_risk_component.csv'),'SVXY')
            reserve=single(path.with_name(path.stem+'_reserve_component.csv'),'BIL')
            a=pd.read_csv(path,float_precision='round_trip')
            close(a.equity_end,risk.equity_end+reserve.equity_end)
            close(a.cost,risk.cost+reserve.cost)
            close(a.old_cash,risk.old_cash+reserve.old_cash)
            close(a.equity_end,a.shares_end*a.close+a.bil_shares_end*a.bil_close+a.cash_end)
            close(a.cash_end,a.old_cash+a.distribution_income-a.svxy_signed_trade-a.bil_signed_trade-a.cost)
            close(a.old_weight,risk.old_weight*risk.equity_start/a.equity_start)
            assert a.reserve_to_risk_transfer.eq(0).all() and a.external_capital_flow.eq(0).all()
            fraction=int(row.strategy.split('_')[1])/100
            close(risk.equity_start.iloc[0],100000*fraction)
            close(reserve.equity_start.iloc[0],100000*(1-fraction))
            # 实际股数只随公司行动调整；本段SVXY无分配且不补仓。
            p=prices['SVXY'].loc[a.as_of_session]
            close(risk.shares_end.iloc[2:].to_numpy(),risk.shares_end.iloc[1:-1].to_numpy()/p.split_factor.iloc[2:].to_numpy())
        elif row.strategy.endswith('_BIL'):
            a=pd.read_csv(path,float_precision='round_trip');p=prices['SVXY'].loc[a.as_of_session];b=prices['BIL'].loc[a.as_of_session]
            old=np.column_stack([a.old_shares,a.old_bil_shares]);f=np.column_stack([p.split_factor,b.split_factor]);px=np.column_stack([p.close,b.close])
            dist=np.column_stack([p.cash_dividend+p.capital_gain_distribution,b.cash_dividend+b.capital_gain_distribution])
            stock=old/f*px;income=(old/f*dist).sum(axis=1);income[0]=0.
            trades=np.column_stack([a.svxy_signed_trade,a.bil_signed_trade])
            fees=.0005*np.abs(trades).sum(axis=1)
            close(a.distribution_income,income);close(a.cost,fees)
            close(a.equity_end,a.old_cash+stock.sum(axis=1)+income-fees)
            close(a.cash_end,a.old_cash+income-trades.sum(axis=1)-fees)
            end=np.column_stack([a.shares_end,a.bil_shares_end])*px
            close(end,stock+trades);close(a.equity_end,end.sum(axis=1)+a.cash_end)
            close(a.new_weight.iloc[1:],int(row.strategy.split('_')[1])/100)
        else:
            symbol=row.strategy[:3] if row.strategy in ['BIL_100','SPY_100'] else 'SVXY'
            a=single(path,symbol)
        assert len(a)-1==row.account_days
        assert a.as_of_session.iloc[1]==row.first_execution and a.as_of_session.iloc[-1]==row.last_close
        assert a.signal_information_session.iloc[1:].tolist()==a.as_of_session.iloc[:-1].tolist()
        close(a.equity_start.iloc[1:],a.equity_end.iloc[:-1])
        close(a.portfolio_return,a.equity_end/a.equity_start-1)
        close(row.total_return,a.equity_end.iloc[-1]/100000-1)
        close(row.max_drawdown,(a.equity_end/a.equity_end.cummax()-1).min())
        close(row.worst_five_days,a.equity_end.pct_change(5,fill_method=None).min())
        close(row.worst_day,a.portfolio_return.iloc[1:].min())
        close(row.mean_exposure,a.old_weight.iloc[1:].mean());close(row.cost_dollars,a.cost.sum())
        close(row.average_cash_weight,(a.old_cash/a.equity_start).iloc[1:].mean())
        close(row.average_cash_weight+row.mean_svxy_exposure+row.mean_spy_exposure+row.mean_bil_exposure,1.)
        # 独立逐日找峰谷与恢复，不调用报告的分段函数。
        dd=a.equity_end/a.equity_end.cummax()-1
        if dd.min()<-1e-14:
            trough=int(dd.idxmin());peak=int(a.equity_end.iloc[:trough+1].idxmax())
            hits=np.flatnonzero(a.equity_end.iloc[trough+1:].to_numpy()>=a.equity_end.iloc[peak])
            recovered=trough+1+hits[0] if len(hits) else None
            assert row.max_dd_trough==a.as_of_session.iloc[trough]
            assert row.max_dd_peak==a.as_of_session.iloc[peak]
            assert bool(row.max_dd_censored)==(recovered is None)
            assert row.max_dd_recovery_sessions==(recovered if recovered is not None else len(a)-1)-peak
        records.append({'period':row.period,'strategy':row.strategy,'rows':len(a)})
    for _,group in metrics.groupby('period'):
        b=group.loc[group.strategy.eq('BIL_100')].iloc[0]
        close(group.excess_over_bil_pp,100*(group.total_return-b.total_return))
    intervals=pd.read_csv(output/'holding_intervals.csv')
    for period,g in intervals.groupby('period'):
        main=metrics.loc[metrics.period.eq(period)&metrics.strategy.eq('M2_main')].iloc[0]
        close(g.net_pnl_dollars.sum(),main.ending_equity-100000)
        assert g.sessions.sum()==main.account_days
        for suffix,strategy in [('fixed50','fixed_50'),('risk_only','M2_risk_only'),('BIL','BIL_100')]:
            other=metrics.loc[metrics.period.eq(period)&metrics.strategy.eq(strategy)].iloc[0]
            close(g['terminal_delta_vs_'+suffix].sum(),main.ending_equity-other.ending_equity)
    result={'accounts':len(records),'account_rows_including_anchors':sum(r['rows'] for r in records),
      'maximum_absolute_numerical_difference':maximum,'equations':'shares, splits, distributions, both-leg costs, cash conservation, no pool transfers, metrics, recovery, terminal attribution',
      'records':records}
    write_json(output/'independent_validation.json',result)
    return result
