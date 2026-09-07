"""明确合成夹具；检查资本分池、双腿费用、复利归因和恢复删失。"""
import numpy as np
import pandas as pd
import pytest
from svxylab.capital import fixed_with_bil, combine_pool, recovery_metrics, exact_attribution
from svxylab.economics import run_account
from svxylab.returns import total_return
from svxylab.timing import session_clock
import exchange_calendars as xcals

pytestmark=pytest.mark.synthetic


def synthetic_prices(values, distributions=None):
    dates=['2024-01-02','2024-01-03','2024-01-04','2024-01-05']
    return total_return(pd.DataFrame({'as_of_session':dates,'close':values,'split_factor':[1.,1.,.5,1.],
         'cash_dividend':distributions or [0.,0.,0.,0.],'capital_gain_distribution':0.}))


def test_two_asset_ledger_split_dividend_and_two_leg_cost():
    a=synthetic_prices([100,100,40,60]);b=synthetic_prices([100,100,50,49],[0,0,0,1])
    clock=session_clock(a.as_of_session,xcals.get_calendar('XNYS').schedule)
    x=fixed_with_bil(a,b,clock,'2024-01-03',.1,100000,.0005)
    assert x.equity_end.iloc[1]==pytest.approx(100000/1.0005)
    assert x.cost.iloc[1]==pytest.approx(100000-100000/1.0005)
    for i in range(1,len(x)):
        old=np.array([x.old_shares.iloc[i],x.old_bil_shares.iloc[i]])
        before_shares=old/np.array([a.split_factor.iloc[i],b.split_factor.iloc[i]])
        closes=np.array([a.close.iloc[i],b.close.iloc[i]])
        dist=before_shares@np.array([a.cash_dividend.iloc[i],b.cash_dividend.iloc[i]])
        trades=np.array([x.svxy_signed_trade.iloc[i],x.bil_signed_trade.iloc[i]])
        assert x.cost.iloc[i]==pytest.approx(.0005*np.abs(trades).sum())
        assert x.equity_end.iloc[i]==pytest.approx(x.old_cash.iloc[i]+before_shares@closes+dist-x.cost.iloc[i])
        assert x.new_weight.iloc[i]==pytest.approx(.1)
        assert x.bil_new_weight.iloc[i]==pytest.approx(.9)
        assert x.old_cash.iloc[i]+dist-trades.sum()-x.cost.iloc[i]==pytest.approx(0,abs=1e-8)


def test_pool_never_refills_and_weight_is_not_loss_cap_for_rebalancing():
    a=synthetic_prices([100,100,25,12.5]);b=synthetic_prices([100,100,50,50])
    clock=session_clock(a.as_of_session,xcals.get_calendar('XNYS').schedule)
    one=pd.Series(1.,index=pd.Index(a.as_of_session,name='as_of_session'))
    stock=run_account(a,clock,one,'2024-01-03',10000,0)
    cash=run_account(b,clock,one*0,'2024-01-03',90000,0)
    pool=combine_pool(stock,cash,100000)
    fixed=run_account(a,clock,one*.1,'2024-01-03',100000,0)
    assert pool.equity_end.iloc[-1]==pytest.approx(92500)
    assert fixed.equity_end.iloc[-1]==pytest.approx(90250)
    assert pool.reserve_to_risk_transfer.eq(0).all()
    assert pool.reserve_equity.eq(90000).all()
    assert pool.old_cash.iloc[0]==100000
    assert pool.new_weight.iloc[-1]<.1
    # 再经历一次50%损失，固定10%累计损失将超过初始10%；不需改历史行情做证明。
    assert fixed.equity_end.iloc[-1]*.95<90000


def test_recovery_dates_include_unrecovered_tail():
    x=pd.DataFrame({'as_of_session':pd.date_range('2024-01-01',periods=7).strftime('%Y-%m-%d'),
                    'equity_end':[100,90,80,100,110,85,90]})
    metric,episodes=recovery_metrics(x)
    assert metric['max_dd_peak']=='2024-01-05'
    assert metric['max_dd_trough']=='2024-01-06'
    assert metric['max_dd_recovered'] is None and metric['max_dd_censored']
    assert metric['max_dd_recovery_sessions']==2
    assert metric['max_dd_trough_recovery_sessions']==1
    assert episodes[0]['recovery']=='2024-01-04'
    assert metric['longest_underwater_sessions']==3


def test_compounded_bridge_sums_to_terminal_difference():
    def frame(returns):
        end=100*np.cumprod(1+np.array(returns));start=np.r_[100,end[:-1]]
        return pd.DataFrame({'as_of_session':['a','b','c','d'],'portfolio_return':returns,
          'equity_end':end,'equity_start':start,'old_weight':[0,0,.5,.5],
          'asset_return':[0,.2,-.3,.4],'holding_pnl':end-start,'cost':0})
    a=frame([0,.1,-.2,.3]);b=frame([0,.2,-.1,.1]);d=exact_attribution(a,b,100)
    assert d.terminal_difference_dollars.sum()==pytest.approx(a.equity_end.iloc[-1]-b.equity_end.iloc[-1])
    assert d.terminal_difference_pp.sum()==pytest.approx(114.4-118.8)
    assert d.state.iloc[0]=='initial_execution'
