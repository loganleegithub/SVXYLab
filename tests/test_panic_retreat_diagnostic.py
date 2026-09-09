"""Synthetic fixtures test timing/accounting; they are not historical results."""
from pathlib import Path
import tomllib

import numpy as np
import pandas as pd
import pytest

from svxylab.panic_retreat_diagnostic import (
    availability, attribution, causal_timeline, candidates, candidate_summary,
    condition, four_cells, holding_outcomes, holding_segments, holding_states,
    independent_audit, reference_states, trade,
)
from svxylab.returns import total_return
from svxylab.timing import session_clock

pytestmark = pytest.mark.synthetic

CFG=tomllib.loads((Path(__file__).resolve().parents[1]/'panic_retreat.toml').read_text())


def fixture(vix=None):
    vix=([20.]*6+[40.,35.,34.,39.,35.,34.,60.,52.,53.,52.]+[22.]*60) if vix is None else vix
    n=len(vix)
    dates=pd.bdate_range('2020-01-02',periods=n+70).strftime('%Y-%m-%d').tolist()
    schedule=pd.DataFrame({'close':pd.to_datetime(dates,utc=True)+pd.Timedelta(hours=21)},index=pd.to_datetime(dates))
    clock=session_clock(dates[:n],schedule)
    market=pd.DataFrame(dict(as_of_session=dates[:n],vix=vix,vx_complete=True,vx_down=True))
    prices=total_return(pd.DataFrame(dict(as_of_session=dates[:n],close=100+np.arange(n)*.7,
                           split_factor=1.,cash_dividend=0.,capital_gain_distribution=0.)))
    return market,prices,clock


def test_causal_daily_and_candidate_dates_survive_prefix_and_future_shock():
    m,p,c=fixture(); _,_,full=causal_timeline(m,CFG)
    for stop in (7,8,9,10,12,15,20,25,45):
        _,_,short=causal_timeline(m.iloc[:stop],CFG)
        pd.testing.assert_frame_equal(full[full.i<stop].reset_index(drop=True),short,check_dtype=False)
        changed=m.copy(); changed.loc[stop:,'vix']=500.
        changed.loc[stop:,'vx_down']=False
        _,_,future=causal_timeline(changed,CFG)
        pd.testing.assert_frame_equal(full[full.i<stop].reset_index(drop=True),future[future.i<stop].reset_index(drop=True),check_dtype=False)


def test_sustained_renewed_and_down_day_are_different():
    m,p,c=fixture(); _,s,t=causal_timeline(m,CFG)
    d=t[t.in_lifecycle].set_index('i')
    assert d.loc[7,'b_renewed'] and d.loc[8,'b_state_class']=='SUSTAINED'
    assert d.loc[9,'b_state']=='FALSE' and d.loc[10,'b_renewed']
    assert d.loc[14,'b_state']=='TRUE' and not d.loc[14,'vix_down_day']
    cover,paths,_=candidates(t,p,c,CFG)
    assert 9 in cover.i.tolist() and 9 not in paths.information_i.tolist()
    assert 7 not in paths.information_i.tolist()  # Original B signal excluded.
    assert paths.query('information_i==10').one_trade_blocks.all()
    assert paths.query('information_i==10').position_blocks.all()


def test_reset_row_is_next_session_and_never_candidate():
    m,p,c=fixture(); eps,_,t=causal_timeline(m,CFG)
    for ep in eps.itertuples():
        last=t.query('episode_id==@ep.episode_id')
        assert last.rearmed.sum()==1
        assert last[last.rearmed].i.iloc[0]==ep.end_i+1
        assert not last[last.rearmed].candidate.any()


def test_information_cutoff_has_no_same_day_daily_leak():
    m,p,c=fixture(); _,s,_=causal_timeline(m,CFG)
    a=availability(s,m,c,CFG)
    lag1=a.query('stage=="PRE_EXECUTION_CUTOFF" and lag==1')
    assert lag1.complete_new_daily_rows.eq(0).all()
    assert lag1.query('rule_id in ["B_RETREAT","C_RETREAT_VX"]').original_condition.eq('TRUE').all()
    post=a.query('stage=="EXECUTION_FINAL_DAILY"')
    assert not post.usable_before_this_fill.any()
    assert (pd.to_datetime(post.state_observable_at_assumed_deadline)>pd.to_datetime(post.execution_at)).all()
    assert a.query('stage=="PRE_EXECUTION_CUTOFF" and lag==2').complete_new_daily_rows.eq(1).all()
    assert not a.historical_publication_proven.any()


def test_availability_and_holding_state_prefix_invariance():
    m,p,c=fixture(); ep,s,_=causal_timeline(m,CFG)
    full=availability(s,m,c,CFG); hs=holding_states(s,ep,m,c,CFG)
    stop=14
    e2,s2,_=causal_timeline(m.iloc[:stop],CFG)
    a=availability(s2,m.iloc[:stop],c,CFG)
    pd.testing.assert_frame_equal(full[full.observation_i<stop].reset_index(drop=True),a,check_dtype=False)
    h2=holding_states(s2,e2,m.iloc[:stop],c,CFG)
    pd.testing.assert_frame_equal(hs[hs.information_i<stop].reset_index(drop=True),h2,check_dtype=False)


def test_unknown_is_not_false_and_c_uses_original_vx_condition():
    m,p,c=fixture(); m.loc[8,'vx_complete']=False
    state=reference_states(m,6,10,CFG)
    assert condition(state.loc[8],'B_RETREAT')=='TRUE'
    assert condition(state.loc[8],'C_RETREAT_VX')=='UNKNOWN'
    assert condition(state.loc[9],'C_RETREAT_VX')=='FALSE'
    m.loc[8,'vix']=np.nan
    state=reference_states(m,6,10,CFG)
    assert condition(state.loc[10],'B_RETREAT')=='UNKNOWN'
    assert condition(state.loc[10],'A_SHOCK')=='NOT_APPLICABLE'


def test_four_cells_horizons_common_cash_and_two_order_identity():
    m,p,c=fixture(); _,s,_=causal_timeline(m,CFG)
    cells,led,common=four_cells(s,p,c,CFG)
    assert cells.groupby('cell').size().eq(len(s)*3).all()
    for cell,delta in [('00',0),('10',-1),('01',1),('11',0)]:
        g=cells[cells.cell.eq(cell)]
        assert (g.actual_holding_sessions==g.H+delta).all()
    for _,g in cells.groupby(['episode_id','rule_id','H']):
        assert g.common_start_i.nunique()==g.common_end_i.nunique()==1
        for r in g.itertuples():
            path=common[common.trade_id.eq(r.trade_id)]
            assert path.iloc[0].equity_end==CFG['capital']
            assert path.iloc[-1].shares_end==0
            assert path.iloc[-1].equity_end/CFG['capital']-1==pytest.approx(r.net_return)
    a=attribution(cells)
    np.testing.assert_allclose(a.entry_contribution+a.exit_contribution,a.total_change,atol=1e-15)
    np.testing.assert_allclose(a.entry_after_exit-a.entry_first,a.interaction,atol=1e-15)
    np.testing.assert_allclose(a.exit_after_entry-a.exit_first,a.interaction,atol=1e-15)


def test_independent_shares_and_segment_identity_with_actions_and_fees():
    m,p,c=fixture()
    # Synthetic 2:1 split and cash distributions during holding, not on entry.
    p.loc[11:,'close']/=2; p.loc[11,'split_factor']=.5
    p.loc[13,'cash_dividend']=1.25; p.loc[15,'capital_gain_distribution']=.4
    p=total_return(p.drop(columns=['simple_return','log_return','total_return_index'],errors='ignore'))
    _,s,_=causal_timeline(m,CFG)
    cells,led,_=four_cells(s,p,c,CFG)
    audit=independent_audit(cells,led,p,CFG['capital'])
    assert len(audit)==len(cells) and audit.max_account_error.max()<2e-7
    assert cells.distribution_cash.gt(0).any()
    seg=holding_segments(cells,led,CFG['capital'])
    assert seg.identity_residual_dollars.abs().max()<2e-7
    assert cells.fees.gt(0).all()


def test_holding_first_false_next_close_and_remaining_original_pnl():
    m,p,c=fixture(); ep,s,_=causal_timeline(m,CFG)
    cells,led,_=four_cells(s,p,c,CFG)
    states=holding_states(s,ep,m,c,CFG)
    h=holding_outcomes(states,cells,led,CFG)
    b=h.query('rule_id=="B_RETREAT" and H==10').iloc[0]
    assert b.first_false_session==m.as_of_session.iloc[9]
    assert b.earliest_action_session==m.as_of_session.iloc[10]
    assert b.pnl_through_earliest_action+b.remaining_original_pnl==pytest.approx(b.final_pnl)
    assert h.query('rule_id=="A_SHOCK"').status.eq('NOT_APPLICABLE').all()


def test_missing_prices_and_immaturity_remain_null():
    m,p,c=fixture(); _,s,_=causal_timeline(m,CFG)
    p.loc[10,'close']=np.nan
    row,_=trade(p,c,8,18,CFG,trade_id='GAP')
    assert row['trade_status']=='PRICE_GAP' and np.isnan(row['net_return'])
    row,_=trade(p.iloc[:9],c.iloc[:9],8,18,CFG,trade_id='OPEN')
    assert row['trade_status']=='OPEN_MARKED' and np.isnan(row['net_return'])
    row,_=trade(p.iloc[:8],c.iloc[:8],8,18,CFG,trade_id='PENDING')
    assert row['trade_status']=='ENTRY_PENDING' and np.isnan(row['net_return'])


def test_no_candidates_separate_from_zero_return():
    m,p,c=fixture([20.]*6+[40.,35.]+[40.]*5)
    ep,s,t=causal_timeline(m,CFG); cover,paths,led=candidates(t,p,c,CFG)
    summary=candidate_summary(ep,cover,paths)
    assert len(cover)>0 and len(paths)==0
    assert summary.status.eq('NO_CANDIDATE').all()
    assert summary.mean_return.isna().all()


def test_perturbing_prices_changes_outcomes_without_candidate_selection():
    m,p,c=fixture(); _,_,t=causal_timeline(m,CFG)
    cover,paths,_=candidates(t,p,c,CFG)
    changed=p.copy(); changed.loc[12:,'close']*=.2
    cover2,paths2,_=candidates(t,changed,c,CFG)
    pd.testing.assert_frame_equal(cover,cover2)
    assert paths.trade_id.tolist()==paths2.trade_id.tolist()
    assert not np.allclose(paths.net_return,paths2.net_return)
