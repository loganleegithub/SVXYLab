"""P6合成算术与时钟测试；不是真实预测或收益证据。"""

from copy import deepcopy
import json
from pathlib import Path
import tomllib

import numpy as np
import pandas as pd
import pytest

from svxylab.diagnostic_models import (VariantDesign, fit_variant, sample_weights,
    variant_eligibility, variants, weighted_head)
from svxylab.diagnostics import losses, moving_blocks, return_statistics
from svxylab.features import CORE_IDS
from svxylab.models import Design, fit_joint
from test_models import synthetic_prediction_data

pytestmark=pytest.mark.synthetic


def specifications():
    root=Path(__file__).parents[1]
    return {v['name']:v for v in variants(json.loads((root/'FEATURES.json').read_text()))}


def test_column_ablation_removes_only_specified_columns_and_dependent_interactions():
    v=specifications()['M2_DROP_BDE']
    assert v['removed_columns']==['B01','B02','B03','D01','D02','D03','E01']
    assert v['removed_interactions']==['I02','I03','I04']
    assert 'A03' in v['columns'] and 'C03' in v['columns'] and 'F01' in v['columns']
    assert all(set(i['features'])<=set(v['columns']) for i in v['interactions'])


def test_recency_age_uses_uncompressed_equity_sessions_and_normalized_count():
    train=pd.DataFrame(index=['a','b','c']); positions={'a':0,'b':504,'c':1008}
    w=sample_weights(train,positions,1014,504)
    np.testing.assert_allclose(w,[3/7,6/7,12/7])
    assert w.sum()==pytest.approx(3)


def test_p6_full_design_and_fit_reproduce_original_p4(synthetic_prediction_data):
    frame,c,interactions=synthetic_prediction_data; v=specifications()['M2_REPLAY']
    train=frame.iloc[:60]; params={'mu5':100.,'q90':.01,'p10':.1}
    fitted,_=fit_variant(train,v,params,c,{d:i for i,d in enumerate(frame.index)},60)
    original=fit_joint('M2',train,params,c,interactions)
    for k,value in original.predict(frame[CORE_IDS].iloc[60:65].to_numpy()).items():
        np.testing.assert_allclose(fitted.predict(frame[CORE_IDS].iloc[60:65].to_numpy())[k],value,atol=1e-12)


def test_skew_cohort_filters_train_validation_and_current_without_fill(synthetic_prediction_data):
    frame,c,_=synthetic_prediction_data; frame=frame.assign(G01=100.,G02=0.)
    frame.loc[frame.index[40],'G01']=np.nan
    specs=specifications()
    a,sa,ra=variant_eligibility(frame,80,c,specs['M2_SKEW'],True)
    b,sb,rb=variant_eligibility(frame,80,c,specs['M2_SKEW_REFERENCE'],True)
    assert frame.index[40] not in a.index and a.index.equals(b.index) and ra==rb=='ELIGIBLE'
    assert all(x[0].index.equals(y[0].index) and x[1].index.equals(y[1].index) for x,y in zip(sa,sb))
    frame.loc[frame.index[80],'G02']=np.nan
    assert variant_eligibility(frame,80,c,specs['M2_SKEW'],False)[2]=='MISSING_COHORT_FEATURES'


def test_deleted_missing_core_feature_does_not_expand_comparison_cohort(synthetic_prediction_data):
    frame,c,_=synthetic_prediction_data; frame.loc[frame.index[40],'B01']=np.nan
    t,_,_=variant_eligibility(frame,80,c,specifications()['M2_DROP_B'],True)
    assert frame.index[40] not in t.index


def test_weighted_ridge_matches_direct_penalized_normal_equation(synthetic_prediction_data):
    _,c,_=synthetic_prediction_data
    x=np.array([[-2.,1.],[-1.,0.],[0.,2.],[1.,1.],[2.,-1.]])
    y=np.array([1.,2.,0.,3.,5.]); w=np.array([.2,.4,.8,1.2,2.4]); alpha=2.
    head=weighted_head('mu5',x,y,alpha,c,w)
    xm=np.average(x,axis=0,weights=w); ym=np.average(y,weights=w)
    xc=x-xm; coef=np.linalg.solve(xc.T@(w[:,None]*xc)+alpha*np.eye(2),xc.T@(w*(y-ym)))
    np.testing.assert_allclose(head['coef'],coef,atol=1e-12)
    assert head['intercept']==pytest.approx(ym-xm@coef)


def test_weighted_quantile_and_probability_fallback(synthetic_prediction_data):
    _,c,_=synthetic_prediction_data
    x=np.zeros((3,1)); y=np.array([0.,.1,.2]); w=np.array([.1,.1,2.8])
    head=weighted_head('q90',x,y,.01,c,w)
    assert head['intercept']==pytest.approx(.2)
    fallback=weighted_head('p10',x,np.zeros(3),.1,c,w)
    assert fallback['value']==pytest.approx(.5/4)


def test_future_perturbation_cannot_change_m3_training_or_transform(synthetic_prediction_data):
    frame,c,_=synthetic_prediction_data; v=specifications()['M3_RECENCY']; position=75
    original,_,_=variant_eligibility(frame,position,c,v,False)
    changed=frame.copy(); cutoff=frame.decision_at.iloc[position]
    changed.loc[changed.label_matures_at>=cutoff,['R5','L5','Y10']]=[.9,.9,1]
    changed.loc[changed.index>frame.index[position],CORE_IDS]=100000.
    after,_,_=variant_eligibility(changed,position,c,v,False)
    pd.testing.assert_frame_equal(original,after)
    positions={d:i for i,d in enumerate(frame.index)}; p={'mu5':100.,'q90':.01,'p10':.1}
    a,aw=fit_variant(original,v,p,c,positions,position); b,bw=fit_variant(after,v,p,c,positions,position)
    np.testing.assert_array_equal(aw,bw)
    np.testing.assert_allclose(a.predict(frame[CORE_IDS].iloc[[position]].to_numpy())['mu5'],b.predict(frame[CORE_IDS].iloc[[position]].to_numpy())['mu5'])


def test_moving_blocks_preserve_calendar_gaps_and_identical_pair_zero():
    idx=moving_blocks(12,3,10,1707)
    assert idx.min()>=0 and idx.max()<12
    assert np.diff(idx.reshape(10,4,3),axis=2).tolist()==np.ones((10,4,2),dtype=int).tolist()
    a=np.arange(12,dtype=float); a[4]=np.nan
    assert np.isnan(a[idx]).any()
    np.testing.assert_allclose(np.nanmean((a-a)[idx],axis=1),0)


def test_unscorable_tail_coverage_stays_missing_not_zero():
    p=pd.DataFrame({'decision_session':['2023-12-21','2023-12-22'],'forecast_available':[True,True],
        'score_observed':[True,False],'R5':[.1,np.nan],'L5':[.1,np.nan],'Y10':[1,np.nan],
        'mu5':[0.,0.],'q90':[.2,.2],'p10':[.2,.2]})
    result=losses(p)
    assert result.q90_coverage.iloc[0]==1 and result.iloc[1].isna().all()


def test_resampled_account_statistics_include_capital_anchor_and_five_returns():
    r=np.array([-.1,.1,0.,0.,0.,0.]); z=np.zeros(6)
    stats=return_statistics(r,z,z,z,1.)
    assert stats['total_return'][0]==pytest.approx(-.01)
    assert stats['max_drawdown'][0]==pytest.approx(-.1)
    assert stats['worst_five_days'][0]==pytest.approx(-.01)


def test_saved_numpy_calendar_is_accepted_by_independent_bootstrap_audit(tmp_path):
    from svxylab.diagnostics_audit import audit_bootstrap
    from svxylab.diagnostics import ECON_FIELDS
    dates=pd.bdate_range('2020-09-11',periods=6).strftime('%Y-%m-%d').to_numpy(dtype='U10')
    idx=moving_blocks(6,3,1000,1707)
    np.savez_compressed(tmp_path/'indices.npz',dates=dates,indices=idx)
    (tmp_path/'ledgers').mkdir()
    ledger=pd.DataFrame({'as_of_session':['2020-09-10',*dates],'portfolio_return':[0.,.01,-.01,0.,0.,0.,0.],
        'old_weight':.5,'turnover':0.,'cost':0.,'equity_start':100.})
    for name in ['left','right']:ledger.to_csv(tmp_path/f'ledgers/test_{name}.csv',index=False)
    header={'domain':'economics','cohort':'test','left':'left','right':'right','block_sessions':3,'index_file':'indices.npz'}
    pd.DataFrame({**header,'draw':np.arange(1000),**{f:np.zeros(1000) for f in ECON_FIELDS}}).to_csv(tmp_path/'bootstrap_draws.csv',index=False)
    pd.DataFrame([{**header,'metric':f,'lower_95':0.,'upper_95':0.} for f in ECON_FIELDS]).to_csv(tmp_path/'bootstrap_intervals.csv',index=False)
    pd.DataFrame({'model':[]}).to_csv(tmp_path/'daily_losses.csv',index=False)
    result=audit_bootstrap(tmp_path)
    assert result['bootstrap_scalar_draws_recomputed']==8000 and result['bootstrap_max_difference']==0
