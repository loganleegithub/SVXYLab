"""Synthetic fixtures test behavior; actual frozen history is audited separately."""
from pathlib import Path
import tomllib
import numpy as np
import pandas as pd
import pytest
from svxylab.panic_retreat import scan_signals, vx_evidence, evaluate, account, inference_clusters
from svxylab.timing import session_clock
from svxylab.returns import total_return

pytestmark = pytest.mark.synthetic
CFG = tomllib.loads((Path(__file__).parents[1]/'panic_retreat.toml').read_text())

def fixture(vix, prices=None):
    n=len(vix)
    dates=pd.bdate_range('2020-01-02', periods=n+70).strftime('%Y-%m-%d').tolist()
    schedule=pd.DataFrame({'close':pd.to_datetime(dates,utc=True)+pd.Timedelta(hours=21)},index=pd.to_datetime(dates))
    cl=session_clock(dates[:n],schedule)
    m=pd.DataFrame({'as_of_session':dates[:n],'vix':vix,'vx_down':True,'vx_complete':True})
    p=total_return(pd.DataFrame({'as_of_session':dates[:n],'close':prices if prices is not None else np.arange(n)+100.,
        'split_factor':1.,'cash_dividend':0.,'capital_gain_distribution':0.}))
    return m,p,cl,dates

def sig(rows,rule):
    return rows.loc[rows.rule_id.eq(rule)].iloc[0]

def test_equal_boundaries_simple_change_and_first_peak():
    m,_,_,_=fixture([30/1.4]*5+[30,27,45,40.5]+[20]*35)
    ep,s,d=scan_signals(m,CFG)
    assert ep.iloc[0].s==5
    assert sig(s,'B_RETREAT').t==6
    assert d.loc[d.i.eq(6),'peak_so_far'].iloc[0]==30
    assert sig(s,'C_RETREAT_VX').t==6
    # A simple +40% shock has log-change below .40.
    assert np.log(30/(30/1.4))<.40

def test_lifecycle_window_natural_and_pending():
    m,_,_,_=fixture([20]*5+[32]*13+[20]*7+[40,36]+[20]*35)
    ep,s,_=scan_signals(m,CFG)
    assert len(ep)==2
    assert sig(s,'B_RETREAT').signal_status=='NATURAL_NO_TRIGGER'
    _,short,_=scan_signals(m.iloc[:10],CFG)
    assert sig(short,'B_RETREAT').signal_status=='WINDOW_PENDING'
    assert ep.iloc[1].s==25

def test_vix_gap_breaks_lows_and_peak():
    m,_,_,_=fixture([20]*5+[40,39,np.nan,35]+[20]*40)
    ep,s,d=scan_signals(m,CFG)
    assert sig(s,'B_RETREAT').signal_status=='DATA_GAP'
    assert d.loc[d.i.eq(8),'peak_so_far'].isna().all()
    assert ep.iloc[0].end_i>=15

def test_c_requires_current_retreat_and_preserves_b():
    m,p,cl,dates=fixture([20]*5+[40,36,45,40]+[20]*35)
    m.loc[6,['vx_down','vx_complete']]=[False,False]
    ep,s,_=scan_signals(m,CFG)
    assert sig(s,'B_RETREAT').t==6
    assert sig(s,'C_RETREAT_VX').t==8
    assert sig(s,'C_RETREAT_VX').signal_status=='TRIGGERED_AFTER_GAP'
    _,o,_=evaluate(ep,s,p,cl,dates,CFG,horizon=10)
    assert sig(o,'C_RETREAT_VX').opportunity_status=='SIGNAL_DATA_GAP'
    assert pd.isna(sig(o,'C_RETREAT_VX').net_return)
    assert pd.notna(sig(o,'B_RETREAT').net_return)
    led,part=account(s,p,cl,CFG,rule='C_RETREAT_VX',horizon=10)
    assert part.status.iloc[0]=='COMPLETED'  # implementable skip-unknown-day policy

def test_current_contract_both_ends_at_roll():
    dates=['2020-01-21','2020-01-22']
    curve=pd.DataFrame({'as_of_session':dates,'f1_contract':['old','new'],'f2_contract':['new','next'],
       'f1_expiration':['2020-01-22','2020-02-19'],'f2_expiration':['2020-02-19','2020-03-18']})
    vx=pd.DataFrame([{'contract_id':k,'as_of_session':d,'value':v,'duration_type':'M'} for k,d,v in
       [('old',dates[0],40),('new',dates[0],20),('new',dates[1],21),('next',dates[0],19),('next',dates[1],18)]])
    e=vx_evidence(dates,curve,vx)
    assert e.iloc[1].f1_previous==20
    assert not e.iloc[1].vx_down  # spliced 40 -> 21 is a false decline
    vx=vx.loc[~((vx.contract_id=='new')&(vx.as_of_session==dates[0]))]
    assert not vx_evidence(dates,curve,vx).iloc[1].vx_complete

def test_trade_horizon_fees_shares_and_distribution():
    m,p,cl,dates=fixture([20]*5+[40,36]+[20]*40, np.full(47,100.))
    p.loc[9:,'close']=50.;p.loc[9,'split_factor']=.5;p.loc[10,'cash_dividend']=1.
    p=total_return(p)
    ep,s,_=scan_signals(m,CFG)
    t,o,paths=evaluate(ep,s,p,cl,dates,CFG,horizon=5)
    r=sig(t,'B_RETREAT');q=100000/1.0005/100
    expected=q*2*50*.9995+q*2
    assert r.e==7 and r.u==12
    assert r.net_return==pytest.approx(expected/100000-1)
    assert r.fees==pytest.approx(q*100*.0005+q*2*50*.0005)
    assert sig(o,'B_RETREAT').common_end_i==5+10+1+5

def test_opportunity_denominator_zero_null_gap():
    m,p,cl,dates=fixture([20]*5+[40]*12+[20]*30)
    ep,s,_=scan_signals(m,CFG)
    t,o,_=evaluate(ep,s,p,cl,dates,CFG,horizon=10)
    assert len(o)==5 and len(t)==3
    assert sig(o,'B_RETREAT').net_return==0
    _,o2,_=evaluate(ep,s,p.iloc[:20],cl.iloc[:20],dates,CFG,horizon=10)
    assert pd.isna(sig(o2,'B_RETREAT').net_return)
    p.loc[8,'close']=np.nan
    t,o,_=evaluate(ep,s,p,cl,dates,CFG,horizon=10)
    assert sig(o,'A_SHOCK').opportunity_status=='PRICE_GAP'

def test_overlap_and_unrealized_tail():
    m,p,cl,dates=fixture([20]*40)
    s=pd.DataFrame([{'episode_id':str(k),'rule_id':'B_RETREAT','t':i,'signal_status':'TRIGGERED'} for k,i in enumerate([5,8,15,37])])
    led,part=account(s,p,cl,CFG,rule='B_RETREAT',horizon=10)
    assert part.status.tolist()==['COMPLETED','SKIPPED_POSITION_OVERLAP','SKIPPED_POSITION_OVERLAP','OPEN_MARKED']
    assert led.shares_end.iloc[-1]>0
    assert led.cost.gt(0).sum()==3

def test_prefix_future_price_and_market_perturbation():
    m,p,cl,dates=fixture([20]*5+[40,36]+[20]*28+[40,36]+[20]*30)
    cutoff=20
    ep,s,d=scan_signals(m,CFG);ep2,s2,d2=scan_signals(m.iloc[:cutoff+1],CFG)
    pd.testing.assert_frame_equal(d.loc[d.i.le(cutoff)].reset_index(drop=True),d2,check_dtype=False)
    cols=['episode_id','rule_id','t']
    pd.testing.assert_frame_equal(s.loc[s.t.le(cutoff),cols].reset_index(drop=True),s2.loc[s2.t.notna(),cols].reset_index(drop=True),check_dtype=False)
    l,_=account(s,p,cl,CFG,rule='B_RETREAT',horizon=20)
    m.loc[cutoff+1:,'vix']=100.;m.loc[cutoff+1:,'vx_down']=False
    p.loc[cutoff+1:,'close']*=.2
    _,changed,_=scan_signals(m,CFG)
    l2,_=account(changed,p,cl,CFG,rule='B_RETREAT',horizon=20)
    pd.testing.assert_frame_equal(l.iloc[:cutoff+1],l2.iloc[:cutoff+1])

def test_risk_metrics_not_equivalent():
    m,p,cl,dates=fixture([20]*5+[40,36]+[20]*30)
    p.loc[7:12,'close']=[100,120,90,110,100,115]
    ep,s,_=scan_signals(m,CFG)
    t,_,_=evaluate(ep,s,p,cl,dates,CFG,horizon=5)
    r=sig(t,'B_RETREAT')
    assert r.mae==pytest.approx(-.1)
    assert r.mfe==pytest.approx(.2)
    assert r.peak_drawdown==pytest.approx(.25)
    assert r.loss_ge_10 and r.profit_ge_10

def test_clusters_connected_overlap():
    e=pd.DataFrame({'episode_id':['a','b','c'],'s':[0,30,100]})
    c=inference_clusters(e,CFG)
    assert c.cluster_id.tolist()==['K001','K001','K002']


def test_zero_episodes_and_single_cluster_no_fake_interval():
    from svxylab.panic_retreat import summaries, cluster_uncertainty
    m,p,cl,dates=fixture([20]*40)
    ep,s,_=scan_signals(m,CFG);t,o,_=evaluate(ep,s,p,cl,dates,CFG,horizon=10)
    assert len(ep)==len(t)==len(o)==0
    report=summaries(t,o,CFG)
    assert report.n.eq(0).all() and report['mean'].isna().all()
    m,p,cl,dates=fixture([20]*5+[40,36]+[20]*35)
    ep,s,_=scan_signals(m,CFG);t,o,_=evaluate(ep,s,p,cl,dates,CFG,horizon=10)
    from svxylab.panic_retreat import paired_results
    _,pairs=paired_results(o,CFG)
    ci,boot,_=cluster_uncertainty(ep,inference_clusters(ep,CFG),t,o,pairs,CFG)
    assert ci.lower.isna().all() and boot.empty


def test_real_exchange_clock_weekend_holiday_half_day_dst():
    import exchange_calendars as xcals
    cal=xcals.get_calendar('XNYS',start='2024-01-01',end='2024-12-31')
    dates=cal.sessions.strftime('%Y-%m-%d').tolist()
    cl=session_clock(['2024-07-02','2024-07-03','2024-03-08'],cal.schedule)
    assert cl.execution_session.tolist()==['2024-07-03','2024-07-05','2024-03-11']
    assert cl.decision_at_new_york.iloc[0].endswith('12:00:00-04:00')
    assert cl.execution_at.iloc[0]=='2024-07-03T17:00:00+00:00'
    assert cl.decision_at_new_york.iloc[2].endswith('15:00:00-04:00')


def test_delay_preserves_signal_and_exit_horizon():
    m,p,cl,dates=fixture([20]*5+[40,36]+[20]*40)
    ep,s,_=scan_signals(m,CFG)
    one,_,_=evaluate(ep,s,p,cl,dates,CFG,horizon=10)
    two,_,_=evaluate(ep,s,p,cl,dates,CFG,horizon=10,lag=2)
    assert one.t.tolist()==two.t.tolist()
    assert (two.e-one.e).eq(1).all() and (two.u-two.e).eq(10).all()


def test_signal_only_never_opens_price_or_outcome_files(monkeypatch,tmp_path):
    from svxylab.panic_retreat import load_inputs, write_json, digest
    import exchange_calendars as xcals
    root=tmp_path/'SYNTHETIC_TEST_ONLY';clean=root/'data/clean';clean.mkdir(parents=True)
    cfg={**CFG,'start':'2024-07-01','end':'2024-07-10'}
    cal=xcals.get_calendar('XNYS',start=cfg['start'],end=cfg['end'])
    dates=cal.sessions.strftime('%Y-%m-%d').tolist()
    pd.DataFrame({'as_of_session':dates,'close':cal.schedule.close.astype(str).tolist()}).to_csv(clean/'equity_sessions.csv',index=False)
    pd.DataFrame({'as_of_session':dates,'value':20.}).to_csv(clean/'VIX_daily.csv',index=False)
    pd.DataFrame({'as_of_session':dates,'f1_contract':'a','f2_contract':'b',
        'f1_expiration':'2024-07-17','f2_expiration':'2024-08-21'}).to_csv(clean/'VX_front_three.csv',index=False)
    pd.DataFrame([{'as_of_session':d,'contract_id':cid,'duration_type':'M','value':22.} for d in dates for cid in ['a','b']]).to_csv(clean/'VX_contract_daily.csv',index=False)
    write_json(root/'runs/p1/reproducibility.json',{'clean_sha256':{str(p.relative_to(root)):digest(p) for p in clean.iterdir()}})
    (root/'data/raw').mkdir();(root/'data/raw/downloads.jsonl').write_text('')
    opened=[];original=Path.open
    def checked(path,*args,**kwargs):
        rel=str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
        assert not any(token in rel for token in ['SVXY_returns','BIL_returns','runs/p2/','runs/n1/','data/clean/p2/'])
        opened.append(rel)
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',checked)
    market,prices,clock,_,_,bil,_,record=load_inputs(root,cfg,as_of='2024-07-03',signal_only=True)
    assert prices is clock is bil is None
    assert market.as_of_session.max()=='2024-07-03'
    assert record['coverage']['signal_only'] and opened


def test_resampling_preserves_whole_unequal_clusters():
    from svxylab.panic_retreat import cluster_uncertainty
    episodes=pd.DataFrame({'episode_id':['a','b','c'],'s':[0,1,100]})
    clusters=inference_clusters(episodes,CFG)
    rows=pd.DataFrame({'episode_id':['a','b','c'],'rule_id':'B_RETREAT','H':10,'net_return':[.1,.3,-.2]})
    pair_details=pd.DataFrame(columns=['episode_id','H','pair','net_return'])
    _,boot,_=cluster_uncertainty(episodes,clusters,rows,rows,pair_details,CFG)
    x=boot.loc[boot.metric.eq('B_RETREAT_OPPORTUNITIES')]
    assert set(x.n)=={2,3,4}
    expected={2:-.2,3:(.1+.3-.2)/3,4:.2}
    for n,value in expected.items():assert np.allclose(x.loc[x.n.eq(n),'mean'],value)
