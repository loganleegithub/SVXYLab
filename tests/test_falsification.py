"""合成夹具：D5同alpha发布控制、不同alpha均值、连续区间及兼容门控。"""
import json
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from svxylab.falsification import fixed_mean, return_path_metrics, segment_metrics
from svxylab.release import digest, verify_release

pytestmark = pytest.mark.synthetic


def test_same_alpha_preserves_published_precision_at_gate():
    X=np.array([[-1.],[0.],[1.]]); y=np.full(3,.001)
    raw=np.array([np.nextafter(.001,np.inf),.001]);published=raw.copy()
    _,_,_,result,pub,_=fixed_mean(X,y,np.array([[0.],[1.]]),100.,raw,published)
    assert np.array_equal(result,raw) and np.array_equal(pub,published)
    assert pub[0]>.001 and not pub[1]>.001


def test_alpha_intervention_matches_independent_svd_and_preserves_mean_target():
    rng=np.random.default_rng(90);X=rng.normal(size=(85,6));y=X[:,0]*.03+rng.normal(size=85)*.04;y[0]=-.5
    V=rng.normal(size=(9,6));old=Ridge(alpha=1,solver='svd').fit(X,y).predict(V)
    theta,_,_,new,pub,_=fixed_mean(X,y,V,1.,old,old)
    expected=Ridge(alpha=100,solver='svd').fit(X,y)
    np.testing.assert_allclose(new,expected.predict(V),atol=1e-14)
    assert np.max(np.abs(new-old))>.005
    assert abs(theta[0]+X.mean(axis=0)@theta[1:]-y.mean())<1e-14
    np.testing.assert_array_equal(pub,np.maximum(new,-1))


def test_unchanged_control_rejects_different_original_prediction():
    with pytest.raises(AssertionError):fixed_mean(np.array([[-1.],[1.]]),np.array([0.,0.]),np.array([[0.]]),100.,np.array([.1]),np.array([.1]))


def test_period_metrics_use_carried_capital_and_opening_peak():
    returns=np.array([-.2,.1,.05,-.1,.03]); wealth=200000*np.cumprod(1+returns)
    g=pd.DataFrame({'portfolio_return':returns,'equity_start':np.r_[200000,wealth[:-1]],'equity_end':wealth,
        'drawdown':wealth/300000-1,'old_weight':.5,'cost':3.,'turnover':.1,'holding_pnl':0.,
        'as_of_session':['d1','d2','d3','d4','d5']})
    m=segment_metrics(g)
    assert m['opening_equity']==200000 and m['ending_equity']==wealth[-1]
    assert abs(m['net_return']-(wealth[-1]/200000-1))<1e-15
    assert abs(m['max_drawdown']+.2)<1e-15
    assert m['inherited_drawdown']<m['max_drawdown']
    assert abs(m['worst_five']-m['net_return'])<1e-15
    r,d=return_path_metrics(returns[None,:]);assert r[0]==m['net_return'] and d[0]==m['max_drawdown']


def test_d5_compatibility_rejects_wrong_previous_revision(tmp_path):
    from test_release_compatibility import fixture_root
    write,source,model,freeze,original=fixture_root(tmp_path)
    n1=write('runs/n1/runtime_compatibility.json',json.dumps({'p7_freeze_sha256':digest(freeze),'changed_frozen_files':{}}))
    mechanism=write('runs/m2_mechanism/runtime_compatibility.json',json.dumps({'p7_freeze_sha256':digest(freeze),'n1_amendment_sha256':digest(n1),'changed_frozen_files':{}}))
    before=digest(source);write('runs/m2_d5/baseline_source/src/svxylab/daily.py',source.read_text());source.write_text('synthetic D5 entry')
    item={'p7_freeze_sha256':digest(freeze),'mechanism_amendment_sha256':digest(mechanism),'changed_frozen_files':{'src/svxylab/daily.py':{'before':before,'after':digest(source)}}}
    p=write('runs/m2_d5/runtime_compatibility.json',json.dumps(item))
    assert verify_release(tmp_path)['source_sha256']['src/svxylab/daily.py']==digest(source)
    item['mechanism_amendment_sha256']='incorrect';p.write_text(json.dumps(item))
    with pytest.raises(ValueError):verify_release(tmp_path)
