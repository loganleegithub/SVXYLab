"""明确合成夹具；验证诊断算术，不把这些数值称为市场证据。"""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from svxylab.mechanism import (switch_terms,log_paths,risk_capacity,offset_target,
    ridge_system,hac_prediction_se,fit_variance,variance_predict,training_clusters,circular_blocks)
pytestmark=pytest.mark.synthetic


def test_switch_interaction_and_drift_identity():
    # Gate turns on as risk capacity falls: interaction is negative and must remain.
    t=switch_terms(False,.8,True,.4,0.,0.)
    assert t['gate_component']==pytest.approx(.8)
    assert t['gate_risk_interaction']==pytest.approx(-.4)
    assert sum(t.values())==pytest.approx(.4)
    t=switch_terms(True,.5,True,.5,.5,.6)
    assert sum(t.values())==pytest.approx(-.1)
    assert t['price_drift_rebalance']==pytest.approx(-.1)
    np.testing.assert_allclose(risk_capacity([0,.025,.1,1]),[1,1,.5,.05])


def test_paths_exclude_information_to_execution_and_keep_partial_tail():
    logs=pd.Series(np.arange(12,dtype=float)**2*.001)
    out=log_paths(np.exp(logs))
    assert out.pre_execution_log.iloc[0]==pytest.approx(.001)
    assert out.after_day1_log.iloc[0]==pytest.approx(.003)
    assert out.after_rest4_log.iloc[0]==pytest.approx(.032)
    assert out.after_full5_log.iloc[0]==pytest.approx(.035)
    assert out.V5.iloc[0]==pytest.approx(np.mean(np.array([.003,.005,.007,.009,.011])**2))
    np.testing.assert_allclose(out.after_day1_log+out.after_rest4_log,out.after_full5_log,equal_nan=True)
    assert np.isnan(out.after_full5_log.iloc[-6]) and np.isfinite(out.after_day1_log.iloc[-6])


def test_offset_uses_calendar_and_does_not_forward_fill_target():
    dates=[f'd{i:02d}' for i in range(20)]
    clock=pd.DataFrame({'as_of_session':dates,'execution_session':dates[1:]+['d20']})
    t=pd.Series(np.arange(20)/20,index=pd.Index(dates,name='as_of_session'))
    sparse,start=offset_target(t,clock,'d02',3,'d19')
    assert start=='d05'
    assert sparse.dropna().index.tolist()==['d04','d09','d14']
    assert sparse.iloc[5:9].isna().all()


def test_ridge_influence_exact_equations_with_unpenalized_intercept():
    rng=np.random.default_rng(13);X=rng.normal(size=(80,5));X[:,4]=X[:,0]+X[:,1]
    y=X@np.array([1.,0.,2.,0.,0.])+2+rng.normal(size=80)
    theta,inv,Z=ridge_system(X,y,100.)
    fit=Ridge(alpha=100,solver='svd').fit(X,y)
    np.testing.assert_allclose(theta,np.r_[fit.intercept_,fit.coef_],atol=2e-13)
    j=3;res=y-Z@theta;eps=1e-5
    weight=np.ones(80);weight[j]+=eps
    fit2=Ridge(alpha=100,solver='svd').fit(X,y,sample_weight=weight)
    deriv=inv@Z[j]*res[j]
    np.testing.assert_allclose((np.r_[fit2.intercept_,fit2.coef_]-theta)/eps,deriv,rtol=1e-5,atol=1e-8)


def test_hac_is_parameter_uncertainty_and_keeps_calendar_gaps():
    X=np.zeros((4,1));y=np.array([1.,-1.,1.,-1.]);theta,inv,Z=ridge_system(X,y,1.)
    V=np.array([[1.,0.]])
    se0=hac_prediction_se(Z,y,V,inv,np.array([0,1,2,3]),0)
    assert se0.item()==pytest.approx(.5) # RMSE=1, mean standard error=.5
    contiguous=hac_prediction_se(Z,y,V,inv,np.arange(4),1)
    gapped=hac_prediction_se(Z,y,V,inv,np.arange(4)*10,1)
    assert contiguous.item()<gapped.item()
    assert gapped.item()==pytest.approx(se0.item())


def test_variance_qlike_constant_targets_arithmetic_mean_not_geomean():
    X=np.zeros((6,1));y=np.array([0.,1.,1.,2.,4.,16.])*.0001
    model=fit_variance(X,y);raw,p=variance_predict(model,X)
    np.testing.assert_allclose(p,y.mean(),rtol=1e-7)
    assert model['target_zero_rows']==1
    assert np.all(p>0)


def test_events_merge_only_mature_training_rows_and_keep_positive_tail():
    train=pd.DataFrame({'Y10':[1,0,0,0],'R5':[-.05,.15,.01,-.15],
        'execution_session':['d01','d04','d08','d20'],'label_end_session':['d06','d09','d13','d25']},index=['t0','t1','t2','t3'])
    e=training_clusters(train)
    assert [v['members'] for v in e]==[['t0','t1'],['t3']]
    assert e[0]['end']=='d09'


def test_blocks_wrap_and_preserve_missing_positions():
    index=circular_blocks(70,21,draws=12)
    assert index.shape==(12,70)
    assert index.min()>=0 and index.max()<70
    np.testing.assert_array_equal((np.diff(index[:,:21],axis=1)%70),np.ones((12,20)))


def test_mechanism_compatibility_requires_exact_prior_n1_record(tmp_path):
    import json
    from test_release_compatibility import fixture_root
    from svxylab.release import digest,verify_release
    write,source,model,freeze,original=fixture_root(tmp_path)
    source.write_text('N1 state')
    amendment=write('runs/n1/runtime_compatibility.json',json.dumps({'p7_freeze_sha256':digest(freeze),
        'changed_frozen_files':{'src/svxylab/daily.py':{'before':original['src/svxylab/daily.py'],'after':digest(source)}}}))
    before=digest(source)
    write('runs/m2_mechanism/baseline_source/src/svxylab/daily.py','N1 state')
    source.write_text('synthetic second entrypoint state')
    record={'p7_freeze_sha256':digest(freeze),'n1_amendment_sha256':digest(amendment),
        'changed_frozen_files':{'src/svxylab/daily.py':{'before':before,'after':digest(source)}}}
    p=write('runs/m2_mechanism/runtime_compatibility.json',json.dumps(record))
    assert verify_release(tmp_path)['source_sha256']['src/svxylab/daily.py']==digest(source)
    record['n1_amendment_sha256']='incorrect'
    p.write_text(json.dumps(record))
    with pytest.raises(ValueError):verify_release(tmp_path)
