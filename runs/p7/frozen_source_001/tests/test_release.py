"""P7合成测试：前向时钟、月度状态、不可覆写的冻结历史。"""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from svxylab.daily import market_time, issue_status, append_after_freeze, fit_forward_if_due, record_outcomes
from svxylab.models import fit_joint
from svxylab.features import CORE_IDS
from svxylab.release import runtime_config
from svxylab.release_audit import saved_prediction
from test_models import synthetic_prediction_data

pytestmark=pytest.mark.synthetic


def test_runtime_adapter_changes_only_evaluation_bounds(synthetic_prediction_data):
    _,config,_=synthetic_prediction_data;old=deepcopy(config)
    c=runtime_config(config,'2026-09-04')
    assert config==old
    assert c['data']['outer_test_requested_start']==old['data']['locked_historical_start']
    assert c['data']['locked_historical_start']=='2026-09-05'
    c['data']=old['data'];assert c==old


@pytest.mark.parametrize('model',['M0','M1','M2'])
def test_saved_forward_prediction_reproduces_original_estimator(synthetic_prediction_data,model):
    frame,config,interactions=synthetic_prediction_data
    fit=fit_joint(model,frame.iloc[:65],{'mu5':10.,'q90':.001,'p10':1.},config,interactions)
    raw=frame[CORE_IDS].iloc[65:70].to_numpy(copy=True);raw[-1]*=15
    result=saved_prediction(fit.snapshot(),raw)
    for field,value in result.items():np.testing.assert_allclose(value,fit.predict(raw)[field],atol=2e-11,rtol=2e-10)


def test_real_exchange_holiday_clock_selects_future_execution():
    clock=market_time('2026-09-07T06:00:00Z')
    assert clock['as_of_session']=='2026-09-04'
    assert clock['decision_session']==clock['execution_session']=='2026-09-08'
    assert clock['decision_at_new_york']=='2026-09-08T15:00:00-04:00'
    assert clock['label_end_session']=='2026-09-15'


def test_early_close_deadline_and_late_generation_are_not_backdated():
    timing=market_time('2026-11-27T16:00:00Z')
    assert timing['decision_at_new_york']=='2026-11-27T12:00:00-05:00'
    assert issue_status(timing,timing['as_of_session'],True,'2026-11-27T15:00Z','2026-11-27T17:00Z')=='MISSED_DECISION_DEADLINE'


def test_freshness_and_received_time_are_independent_of_label_scoring():
    t=market_time('2026-09-07T06:00Z')
    assert issue_status(t,'2026-09-04',True,'2026-09-06T18:00Z','2026-09-07T06:00Z')=='FORWARD_FORECAST_AVAILABLE'
    assert issue_status(t,'2026-09-03',True,'2026-09-06T18:00Z','2026-09-07T06:00Z')=='STALE_INPUTS_NO_FORECAST'
    assert issue_status(t,'2026-09-04',False,'2026-09-06T18:00Z','2026-09-07T06:00Z')=='MISSING_CORE_INPUTS_NO_FORECAST'
    assert issue_status(t,'2026-09-04',True,'2026-09-08T18:00Z','2026-09-07T06:00Z')=='INPUT_NOT_YET_RECEIVED'


def test_update_cannot_revise_frozen_historical_values():
    original=pd.DataFrame({'as_of_session':['2026-09-03','2026-09-04'],'close':[9.,10.]})
    incoming=pd.DataFrame({'as_of_session':['2026-09-04','2026-09-08'],'close':[999.,11.]})
    result=append_after_freeze(original,incoming,'2026-09-04')
    assert result.close.tolist()==[9.,10.,11.]
    with pytest.raises(ValueError,match='重复'):append_after_freeze(original,pd.concat([incoming,incoming]),'2026-09-04')


def test_same_month_daily_reuses_state_and_never_calls_optimizer(monkeypatch,tmp_path):
    states={m:{'fit':{'fit_session':'2026-09-01'}} for m in ['M0','M1','M2']}
    def forbidden(*args,**kwargs):raise AssertionError('不应拟合')
    monkeypatch.setattr('svxylab.daily.fit_joint',forbidden)
    got,due=fit_forward_if_due(None,None,None,states,{'decision_session':'2026-09-08'},'2026-09-07T06:00Z',tmp_path)
    assert got is states and due is False


def test_forward_outcome_stays_empty_until_complete_and_never_rewrites_prediction(tmp_path):
    directory=tmp_path/'data/clean/forward/predictions';directory.mkdir(parents=True)
    clock=market_time('2026-09-07T06:00Z')
    record={'timing':clock,'R5':None};path=directory/'2026-09-08.json';path.write_text(json.dumps(record))
    original=path.read_bytes()
    prices=pd.DataFrame({'as_of_session':['2026-09-08','2026-09-09','2026-09-10','2026-09-11','2026-09-14','2026-09-15'],
        'total_return_index':[100.,101.,90.,91.,92.,94.]})
    assert record_outcomes(tmp_path,prices,'2026-09-14T22:00Z',{})==0
    assert record_outcomes(tmp_path,prices.iloc[:-1],'2026-09-16T00:00Z',{})==0
    assert record_outcomes(tmp_path,prices,'2026-09-16T00:00Z',{})==1
    outcome=json.loads((tmp_path/'data/clean/forward/outcomes/2026-09-08.json').read_text())
    assert outcome['Y10']==1 and outcome['L5']==pytest.approx(.1) and outcome['R5']==pytest.approx(-.06)
    assert record_outcomes(tmp_path,prices,'2026-09-17T00:00Z',{})==0
    assert path.read_bytes()==original


def test_forward_monthly_refit_uses_actual_run_cutoff_not_future_deadline(synthetic_prediction_data,tmp_path):
    frame,config,interactions=synthetic_prediction_data;pos=85;row=frame.iloc[pos]
    now=row.decision_at-pd.Timedelta(hours=2)
    timing={k:row[k] for k in ['decision_session']};timing['as_of_session']=frame.index[pos]
    seed=fit_joint('M2',frame.iloc[:50],{'mu5':100.,'q90':.01,'p10':.1},config,interactions).snapshot()
    saved={m:{**deepcopy(seed),'fit':{'fit_session':'2019-01-02','selection_id':'2019-01-02'}} for m in ['M0','M1','M2']}
    a=tmp_path/'a';b=tmp_path/'b';a.mkdir();b.mkdir()
    first,due=fit_forward_if_due(frame,config,{'interactions':interactions},saved,timing,now,a)
    assert due
    members=pd.read_csv(a/'forward_training_members_and_values.csv')
    assert pd.to_datetime(members.label_matures_at,utc=True).lt(now).all()
    assert pd.to_datetime(members.label_available_at,utc=True).le(now).all()
    changed=frame.copy();changed.loc[changed.label_available_at.gt(now),['R5','L5','Y10']]=[.8,.9,1]
    second,_=fit_forward_if_due(changed,config,{'interactions':interactions},saved,timing,now,b)
    raw=frame[CORE_IDS].iloc[[pos]].to_numpy()
    for m in first:
        for k,v in saved_prediction(first[m],raw).items():np.testing.assert_array_equal(v,saved_prediction(second[m],raw)[k])
