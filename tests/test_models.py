"""合成模型/时钟测试，不是市场收益或真实预测验证。"""

from copy import deepcopy
from pathlib import Path
import json
import math
import tomllib

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import pytest

from svxylab.features import CORE_IDS
from svxylab.models import Design, fit_joint, prediction_loss, publish, select_parameters
from svxylab.prediction_data import development_csv, eligibility, inner_splits, training_rows
from svxylab.predictions import run_predictions
from svxylab.timing import session_clock


@pytest.fixture
def synthetic_prediction_data():
    root = Path(__file__).resolve().parents[1]
    config = tomllib.loads((root / "experiment.toml").read_text())
    config = deepcopy(config)
    config["data"].update({"initial_train_end": "2019-01-02", "outer_test_requested_start": "2019-02-01", "locked_historical_start": "2020-01-01"})
    config["training"].update({"max_window_sessions": 90, "min_outer_training_rows": 32, "min_inner_training_rows": 8,
                               "inner_validation_blocks": 3, "inner_validation_block_sessions": 5})
    interactions = json.loads((root / "FEATURES.json").read_text())["interactions"]
    schedule = xcals.get_calendar("XNYS", start="2019-01-01", end="2019-12-31").schedule
    days = schedule.index.strftime("%Y-%m-%d")[:110].tolist()
    rng = np.random.default_rng(1707)
    raw = rng.normal(size=(len(days),len(CORE_IDS)))
    frame = pd.DataFrame(raw,index=pd.Index(days,name="as_of_session"),columns=CORE_IDS)
    clock = session_clock(days,schedule).set_index("as_of_session")
    frame = frame.join(clock)
    for c in ["information_close_at", "decision_at", "execution_at", "label_matures_at", "label_available_at"]:
        frame[c] = pd.to_datetime(frame[c],utc=True)
    frame["assumed_available_at"] = frame.decision_at
    frame["R5"] = .01*raw[:,0]+.005*raw[:,1]*raw[:,2]+rng.normal(0,.01,len(days))
    frame["L5"] = np.clip(.05+.03*raw[:,3]+.02*raw[:,4],0,.3)
    frame["Y10"] = (frame.L5 >= .10).astype(int)
    frame["observed"] = True
    frame["reason"] = "SYNTHETIC_LABEL"
    return frame,config,interactions


def test_synthetic_maturity_cutoff_equality_and_future_labels(synthetic_prediction_data):
    frame,c,_ = synthetic_prediction_data
    position = 50
    cutoff = frame.decision_at.iloc[position]
    allowed = training_rows(frame,position,c)
    assert allowed.index[-1] == frame.index[position-6]
    assert allowed.label_available_at.max() == cutoff
    before = training_rows(frame,position,c,cutoff=cutoff-pd.Timedelta(nanoseconds=1))
    assert before.index[-1] == frame.index[position-7]
    changed = frame.copy()
    changed.loc[changed.label_matures_at >= cutoff, ["R5","L5","Y10"]] = [.9,.9,1]
    changed.loc[changed.label_available_at > cutoff, ["R5","L5","Y10"]] = [-.9,.8,1]
    pd.testing.assert_frame_equal(training_rows(changed,position,c),allowed)
    with pytest.raises(ValueError):
        training_rows(frame,position,c,cutoff="2019-03-01")


def test_synthetic_calendar_window_not_last_n_complete_rows(synthetic_prediction_data):
    frame,c,_ = synthetic_prediction_data
    c["training"]["max_window_sessions"] = 30
    frame.loc[frame.index[60:65],"A01"] = np.nan
    allowed = training_rows(frame,80,c)
    assert allowed.index[0] == frame.index[51]
    assert not allowed.index.isin(frame.index[60:65]).any()
    assert len(allowed) == 19


def test_synthetic_inner_blocks_purge_and_preserve_calendar_gaps(synthetic_prediction_data):
    frame,c,_ = synthetic_prediction_data
    frame.loc[frame.index[66],"B01"] = np.nan
    outer = training_rows(frame,80,c)
    splits = inner_splits(frame,outer,c)
    assert len(splits) == 3
    assert [info["validation_information_first"] for _,_,info in splits] == [frame.index[i] for i in [60,65,70]]
    assert [len(v) for _,v,_ in splits] == [5,4,5]
    for train,validation,info in splits:
        cutoff = pd.Timestamp(info["validation_cutoff"])
        assert train.label_matures_at.lt(cutoff).all()
        assert train.label_available_at.le(cutoff).all()
        assert set(train.index).isdisjoint(validation.index)
        assert train.index[-1] < validation.index[0]


def test_synthetic_outer_and_inner_requirements_are_both_required(synthetic_prediction_data):
    frame,c,_ = synthetic_prediction_data
    c["training"]["min_outer_training_rows"] = 400
    _,_,reason = eligibility(frame,70,c,need_selection=True)
    assert reason == "INSUFFICIENT_OUTER_TRAINING_ROWS"
    c["training"]["min_outer_training_rows"] = 32
    c["training"]["min_inner_training_rows"] = 60
    _,_,reason = eligibility(frame,70,c,need_selection=True)
    assert reason == "INSUFFICIENT_INNER_TRAINING_ROWS"


@pytest.mark.parametrize("model,columns", [("M1",19),("M2",63)])
def test_synthetic_transform_fits_only_training_and_preserves_interactions(synthetic_prediction_data,model,columns):
    frame,c,interactions = synthetic_prediction_data
    raw = frame[CORE_IDS].to_numpy()
    design = Design(model,c,interactions)
    design.fit(raw[:50])
    snapshot = json.dumps(design.snapshot(),sort_keys=True)
    output = design.transform(raw[50:55]*1000)
    assert output.shape == (5,columns)
    assert np.allclose(design.raw_scaler.mean_,raw[:50].mean(axis=0))
    assert json.dumps(design.snapshot(),sort_keys=True) == snapshot
    if model == "M2":
        z = design.raw_scaler.transform(raw[:10])
        recovered = design.transform(raw[:10])*design.output_scaler.scale_+design.output_scaler.mean_
        for i,(a,b) in enumerate(design.pairs):
            assert np.allclose(recovered[:,57+i],z[:,a]*z[:,b])
        assert [x["id"] for x in design.interactions] == [f"I0{i}" for i in range(1,7)]


def test_synthetic_spline_extrapolation_is_linear_and_reported(synthetic_prediction_data):
    frame,c,interactions = synthetic_prediction_data
    raw = frame[CORE_IDS].iloc[:50].to_numpy()
    design = Design("M2",c,interactions)
    design.fit(raw)
    values = np.repeat(raw.mean(axis=0)[None,:],3,axis=0)
    values[:,0] = design.raw_max[0]+design.raw_scaler.scale_[0]*np.array([1,2,3])
    changed = design.transform(values)
    assert np.allclose(changed[2]-changed[1],changed[1]-changed[0],atol=1e-12)
    outside,distance = design.support(values)
    assert outside[:,0].all()
    assert np.allclose(distance,[1,2,3])


def test_synthetic_m0_uses_same_mean_linear_quantile_and_jeffreys(synthetic_prediction_data):
    frame,c,interactions = synthetic_prediction_data
    train = frame.iloc[:40]
    fitted = fit_joint("M0",train,{},c,interactions)
    predictions = fitted.predict(frame[CORE_IDS].iloc[45:48].to_numpy())
    assert np.allclose(predictions["mu5"],train.R5.mean())
    assert np.allclose(predictions["q90"],np.quantile(train.L5,.9,method="linear"))
    assert np.allclose(predictions["p10"],(train.Y10.sum()+.5)/41)


@pytest.mark.parametrize("single_class",[0,1])
def test_synthetic_single_class_logistic_fallback_is_marked(synthetic_prediction_data,single_class):
    frame,c,interactions = synthetic_prediction_data
    train = frame.iloc[:40].copy();train.Y10 = single_class
    fitted = fit_joint("M1",train,{"mu5":10,"q90":.01,"p10":.1},c,interactions)
    assert fitted.heads["p10"]["fallback"] == "SINGLE_CLASS_JEFFREYS"
    result = fitted.predict(frame[CORE_IDS].iloc[45:48].to_numpy())
    assert np.allclose(result["p10"],(single_class*40+.5)/41)


def test_synthetic_projection_and_inner_scoring_use_same_values():
    raw = {"mu5":np.array([-2.,.3]),"q90":np.array([-.2,1.2]),"p10":np.array([-100.,100.])}
    result = publish(raw)
    assert np.array_equal(result["mu5"],[-1,.3])
    assert np.array_equal(result["q90"],[0,1])
    assert np.array_equal(result["q_projected"],[True,True])
    assert prediction_loss("mu5",np.array([-1.,.3]),raw["mu5"]) == 0
    assert prediction_loss("q90",np.array([0.,1.]),raw["q90"]) == 0
    with pytest.raises(ValueError):
        publish({**raw,"q90":np.array([np.inf,1])})


def test_synthetic_grid_ties_prefer_stronger_shrinkage(synthetic_prediction_data):
    frame,c,interactions = synthetic_prediction_data
    frame[["R5","L5","Y10"]] = 0
    outer = training_rows(frame,65,c)
    splits = inner_splits(frame,outer,c)
    selected,scores,totals,transforms = select_parameters(splits,"M1",c,interactions)
    assert selected == {"mu5":100.,"q90":.01,"p10":.1}
    assert len(scores) == 27 and len(totals) == 9 and len(transforms) == 3
    for transform,(train,_,_) in zip(transforms,splits,strict=True):
        assert np.allclose(transform["transform"]["raw_scaler"]["mean"],train[CORE_IDS].mean())


def test_synthetic_locked_outcome_is_never_parsed_or_exposed(tmp_path):
    path = tmp_path / "labels.csv"
    path.write_text("as_of_session,label_end_session,R5,L5,Y10,observed,reason\n2023-12-20,2023-12-29,.01,.02,0,True,test\n2023-12-21,2024-01-02,DO_NOT_PARSE,DO_NOT_PARSE,DO_NOT_PARSE,True,test\n2024-01-02,2024-01-10,DO_NOT_PARSE,DO_NOT_PARSE,DO_NOT_PARSE,True,test\n")
    result = development_csv(path,"2024-01-01",labels=True)
    assert result.index.tolist() == ["2023-12-20","2023-12-21"]
    assert result.loc["2023-12-21",["R5","L5","Y10"]].tolist() == ["","",""]
    assert result.loc["2023-12-21","observed"] == "False"


def test_synthetic_future_perturbation_leaves_prior_rolling_predictions_unchanged(synthetic_prediction_data,tmp_path):
    frame,c,interactions = synthetic_prediction_data
    frame = frame.iloc[:85].copy()
    first = tmp_path / "first"
    run_predictions(frame,c,interactions,first)
    cutoff_position = 65
    cutoff = frame.decision_at.iloc[cutoff_position]
    changed = frame.copy()
    changed.loc[changed.decision_at > cutoff,CORE_IDS] *= 100
    changed.loc[changed.label_available_at > cutoff,["R5","L5","Y10"]] = [-.8,.8,1]
    second = tmp_path / "changed"
    run_predictions(changed,c,interactions,second)
    before,after = [pd.read_csv(d / "predictions.csv") for d in [first,second]]
    columns = [c for c in before if c not in ["computed_at_utc","R5","L5","Y10"]]
    mask = before.decision_session <= frame.decision_session.iloc[cutoff_position]
    pd.testing.assert_frame_equal(before.loc[mask,columns],after.loc[mask,columns],check_exact=True)
    for directory in [first,second]:
        fits = pd.read_csv(directory / "fits.csv")
        for model,subset in fits.groupby("model"):
            assert subset.fit_session.str[:7].is_unique
        selected = pd.read_csv(directory / "selected_parameters.csv")
        assert selected.selection_id.nunique() == 1
