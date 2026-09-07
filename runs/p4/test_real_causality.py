"""明示合成未来扰动夹具：复制真实开发输入，仅供 pytest，不是新行情/回测。

从 2021-06-16 起人为改变价格；原件不写入，扰动输出置于 SYNTHETIC_TEST_ONLY 目录。
"""

import csv
from datetime import datetime,timezone
from hashlib import sha256
import json
from pathlib import Path
import tomllib

import numpy as np
import pandas as pd
import pytest

from svxylab.features import CORE_IDS,build_inputs,calculate_features
from svxylab.prediction_data import load_development_data,development_csv
from svxylab.predictions import run_predictions
from svxylab.returns import total_return
from svxylab.timing import five_day_labels

ROOT=Path(__file__).resolve().parents[2]


def input_rows(name,numeric):
    with (ROOT / "data/clean" / name).open(newline="") as stream:
        rows=[r for r in csv.DictReader(stream) if "2019-01-01" <= r["as_of_session"] < "2024-01-01"]
    frame=pd.DataFrame(rows)
    for key in numeric:
        frame[key]=pd.to_numeric(frame[key].replace("",np.nan))
    return frame


@pytest.fixture
def synthetic_future_from_real_inputs():
    run=sorted(p for p in (ROOT / "runs/p4").glob("*/predictions_run.json") if not json.loads(p.read_text())["result"]["pilot"])[-1]
    record=json.loads(run.read_text())
    config=tomllib.loads((ROOT / "experiment.toml").read_text())
    definitions=json.loads((ROOT / "FEATURES.json").read_text())
    baseline,verified=load_development_data(ROOT,config)
    cutoff_information="2021-06-15"
    cutoff_decision=baseline.loc[cutoff_information,"decision_session"]
    curve=input_rows("VX_front_three.csv",["f1_settle","f2_settle","f3_settle"])
    vx=input_rows("VX_contract_daily.csv",["value"])
    indices={s:input_rows(f"{s}_daily.csv",["value"]) for s in ["VIX","VVIX","VIX9D","VIX3M","SKEW"]}
    fields=["close","split_factor","cash_dividend","capital_gain_distribution"]
    prices={s:input_rows(f"{s}_daily.csv",fields)[["as_of_session"]+fields].copy() for s in ["SPY","SVXY"]}
    original_returns={s:total_return(p) for s,p in prices.items()}
    x=build_inputs(baseline.index,curve,vx,indices,original_returns,history_start="2019-01-01")
    original_features=calculate_features(x)
    for s,factor in [("SPY",.85),("SVXY",.70)]:
        prices[s].loc[prices[s].as_of_session > cutoff_information,"close"] *= factor
    for s in ["VIX","VVIX","VIX9D","VIX3M"]:
        indices[s].loc[indices[s].as_of_session > cutoff_information,"value"] *= 1.6
    for k in (1,2,3):
        curve.loc[curve.as_of_session > cutoff_information,f"f{k}_settle"] *= 1.4
    vx.loc[vx.as_of_session > cutoff_information,"value"] *= 1.4
    altered_returns={s:total_return(p) for s,p in prices.items()}
    altered_features=calculate_features(build_inputs(baseline.index,curve,vx,indices,altered_returns,history_start="2019-01-01"))
    pd.testing.assert_frame_equal(original_features.loc[:cutoff_information],altered_features.loc[:cutoff_information],check_exact=True)
    changed=baseline.copy()
    future=changed.index > cutoff_information
    changed.loc[future,CORE_IDS]=altered_features.loc[future,CORE_IDS]
    p2_run=json.loads((ROOT / verified["p2_accepted_run"]).read_text())
    clock=development_csv(ROOT / p2_run["result"]["output_dir"] / "clock.csv","2024-01-01").reset_index()
    altered_labels=five_day_labels(altered_returns["SVXY"],clock).set_index("as_of_session")
    affected=changed.label_end_session > cutoff_information
    for key in ["R5","L5","Y10","observed","reason"]:
        changed.loc[affected,key]=altered_labels.loc[affected,key]
    changed.loc[changed.label_end_session >= "2024-01-01",["R5","L5","Y10"]]=np.nan
    changed.loc[changed.label_end_session >= "2024-01-01","observed"]=False
    return run,record,config,definitions["interactions"],baseline,changed,cutoff_information,cutoff_decision


def test_synthetic_future_quotes_do_not_change_prior_real_replay(synthetic_future_from_real_inputs):
    run,record,config,interactions,baseline,changed,cutoff_info,cutoff_decision=synthetic_future_from_real_inputs
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    fixture_dir=run.parent / "SYNTHETIC_TEST_ONLY" / stamp
    fixture_dir.mkdir(parents=True)
    (fixture_dir / "README.txt").write_text("合成未来扰动单元测试夹具；不是取得的新行情，不可用于收益结论。原始行情从未写入。\n")
    result=run_predictions(changed,config,interactions,fixture_dir / "outputs")
    original=pd.read_csv(ROOT / record["result"]["output_dir"] / "predictions.csv")
    altered=pd.read_csv(fixture_dir / "outputs/predictions.csv")
    columns=["as_of_session","decision_session","simulation_decision_at","model","fit_id","forecast_available","forecast_status",
             "mu5_raw","mu5","q90_raw","q90","p10_logit","p10","mu_projected","q_projected","outside_feature_count","outside_feature_ids","max_outside_distance_training_sd"]
    mask=original.decision_session <= cutoff_decision
    pd.testing.assert_frame_equal(original.loc[mask,columns],altered.loc[mask,columns],check_exact=True)
    later=(original.decision_session > cutoff_decision)&original.forecast_available
    assert not original.loc[later,["mu5","q90","p10"]].equals(altered.loc[later,["mu5","q90","p10"]])
    assert (original.loc[mask,columns].shape[0]>0)
    for group in ["p1_clean_sha256","p2_artifact_sha256","p3_artifact_sha256"]:
        for path,wanted in record["input_verification"][group].items():
            assert sha256((ROOT / path).read_bytes()).hexdigest()==wanted,path
    passed={"fixture_kind":"SYNTHETIC_TEST_ONLY","baseline_run":run.relative_to(ROOT).as_posix(),"test_source_sha256":sha256(Path(__file__).read_bytes()).hexdigest(),
            "quote_mutation_begins_after_information_session":cutoff_info,"last_unchanged_decision_session":cutoff_decision,
            "earlier_rows_compared":int(mask.sum()),"earlier_actual_forecast_rows":int((mask&original.forecast_available).sum()),
            "earlier_predictions_identical":True,"later_predictions_changed":True,"past_features_identical":True,
            "frozen_input_hashes_unchanged":True,"elapsed_replay_seconds":result["elapsed_seconds"],"fixture_dir":fixture_dir.relative_to(ROOT).as_posix(),"passed":True}
    (run.parent / "real_future_perturbation.json").write_text(json.dumps(passed,ensure_ascii=False,indent=2)+"\n")
