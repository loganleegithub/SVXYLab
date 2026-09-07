"""真实 P4 审计：不用生产模型/评分/训练选择函数，从保存的数值独立复算。"""

import csv
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import tomllib

import numpy as np
import pandas as pd
from scipy.interpolate import BSpline

parser = argparse.ArgumentParser(description="保存系数的独立数值审计；不重新拟合")
parser.add_argument("--bundle", type=Path, help="本地离线包根目录；仅使用包内已截断开发输入")
args = parser.parse_args()
ROOT = args.bundle.resolve() if args.bundle else Path(__file__).resolve().parents[2]
BUNDLE = json.loads((ROOT / "bundle_manifest.json").read_text()) if args.bundle else None
RUN = ROOT / BUNDLE["baseline_run"] if BUNDLE else sorted(
    p for p in (ROOT / "runs/p4").glob("*/predictions_run.json")
    if not json.loads(p.read_text())["result"]["pilot"])[-1]
RECORD = json.loads(RUN.read_text())
OUTPUT = ROOT / RECORD["result"]["output_dir"]
CONFIG = tomllib.loads((ROOT / "experiment.toml").read_text())
DICTIONARY = json.loads((ROOT / "FEATURES.json").read_text())
IDS = [r["id"] for r in DICTIONARY["features"] if r["tier"] == "core"]
INTERACTIONS = DICTIONARY["interactions"]
TARGETS = ["R5", "L5", "Y10"]


def bounded_rows(path):
    result = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["as_of_session"] >= CONFIG["data"]["locked_historical_start"]:
                break
            result[row["as_of_session"]] = row
    return result


if BUNDLE:
    input_paths = {k: ROOT / v for k, v in BUNDLE["development_inputs"].items()}
else:
    p3_accept = json.loads((ROOT / "runs/p3/acceptance.json").read_text())
    p3 = json.loads((ROOT / p3_accept["accepted_run"]).read_text())
    p2_accept = json.loads((ROOT / "runs/p2/acceptance.json").read_text())
    p2 = json.loads((ROOT / p2_accept["accepted_run"]).read_text())
    input_paths = {"features": ROOT / p3["result"]["output_dir"] / "core_features.csv",
                   "availability": ROOT / p3["result"]["output_dir"] / "feature_availability.csv",
                   "labels": ROOT / p2["result"]["output_dir"] / "labels.csv"}
raw = bounded_rows(input_paths["features"])
availability = bounded_rows(input_paths["availability"])
labels = bounded_rows(input_paths["labels"])
days = list(raw)
positions = {d:i for i,d in enumerate(days)}


def inputs(dates):
    return np.asarray([[float(raw[d][k]) if raw[d][k] else np.nan for k in IDS] for d in dates])


def target(dates,key):
    assert all(labels[d]["label_end_session"] < CONFIG["data"]["locked_historical_start"] for d in dates)
    return np.asarray([float(labels[d][key]) for d in dates])


def allowed(dates,cutoff):
    found = []
    cutoff = datetime.fromisoformat(cutoff)
    for day in dates:
        r = labels[day]
        if (all(raw[day][k] for k in IDS) and r["observed"] == "True" and r["label_end_session"] < "2024-01-01"
                and datetime.fromisoformat(r["label_matures_at"]) < cutoff
                and datetime.fromisoformat(r["label_available_at"]) <= cutoff
                and datetime.fromisoformat(availability[day]["assumed_available_at"]) <= cutoff):
            found.append(day)
    return found


errors = {k:0.0 for k in ["raw_scaler", "spline_knots", "output_scaler", "predictions", "cv_losses", "metrics", "m0_constants", "support"]}


def track(kind,observed,expected):
    a,b = np.asarray(observed,dtype=float),np.asarray(expected,dtype=float)
    assert a.shape == b.shape,(kind,a.shape,b.shape)
    assert np.isfinite(a).all() and np.isfinite(b).all(),kind
    delta = float(np.max(np.abs(a-b))) if a.size else 0.0
    errors[kind] = max(errors[kind],delta)
    assert np.allclose(a,b,atol=2e-11,rtol=2e-10),(kind,delta)


def unscaled_basis(transform,x):
    z = (x-np.asarray(transform["raw_scaler"]["mean"]))/np.asarray(transform["raw_scaler"]["scale"])
    if transform["model"] == "M1":
        return z
    out=[]
    for i,state in enumerate(transform["bsplines"]):
        spline = BSpline(np.asarray(state["knots"]),np.asarray(state["coefficients"]),state["degree"])
        left,right = spline.t[spline.k],spline.t[-spline.k-1]
        at = np.clip(z[:,i],left,right)
        values = spline(at)
        # 直接在边界接一阶切线；避免把 BSpline 默认多项式外推误当线性。
        values += (z[:,i]-at)[:,None]*spline(at,nu=1)
        out.append(values[:,:transform["basis_per_feature"]])
    out.append(np.column_stack([z[:,IDS.index(r["features"][0])]*z[:,IDS.index(r["features"][1])] for r in INTERACTIONS]))
    return np.column_stack(out)


def matrix(transform,x):
    base=unscaled_basis(transform,x)
    if transform["model"] == "M1":
        return base
    return (base-np.asarray(transform["output_scaler"]["mean"]))/np.asarray(transform["output_scaler"]["scale"])


def validate_transform(transform,dates):
    x=inputs(dates)
    track("raw_scaler",transform["raw_scaler"]["mean"],x.mean(axis=0))
    sd=x.std(axis=0,ddof=0);sd[sd==0]=1
    track("raw_scaler",transform["raw_scaler"]["scale"],sd)
    track("raw_scaler",transform["raw_min"],x.min(axis=0))
    track("raw_scaler",transform["raw_max"],x.max(axis=0))
    if transform["model"] == "M2":
        assert transform["basis_per_feature"] == 3 and len(transform["columns"]) == 63
        z=(x-x.mean(axis=0))/sd
        for i,s in enumerate(transform["bsplines"]):
            track("spline_knots",s["knots"][s["degree"]:-s["degree"]],np.quantile(z[:,i],[0,.5,1],method="linear"))
        base=unscaled_basis(transform,x)
        track("output_scaler",transform["output_scaler"]["mean"],base.mean(axis=0))
        scale=base.std(axis=0,ddof=0);scale[scale==0]=1
        track("output_scaler",transform["output_scaler"]["scale"],scale)


def raw_score(head,fit,z,n):
    if fit["kind"] == "constant":
        value = math.log(fit["value"]/(1-fit["value"])) if head == "p10" else fit["value"]
        return np.full(n,value)
    return z@np.asarray(fit["coef"])+fit["intercept"]


def prediction(head,score):
    if head == "mu5":return np.maximum(score,-1)
    if head == "q90":return np.clip(score,0,1)
    return 1/(1+np.exp(-score))


def loss(head,y,p):
    if head=="mu5":return math.fsum((a-b)**2 for a,b in zip(y,p))/len(y)
    if head=="q90":return math.fsum(max(.9*(a-b),-.1*(a-b)) for a,b in zip(y,p))/len(y)
    eps=np.finfo(float).eps
    return math.fsum(-(a*math.log(max(eps,min(1-eps,b)))+(1-a)*math.log1p(-max(eps,min(1-eps,b)))) for a,b in zip(y,p))/len(y)


predictions=pd.read_csv(OUTPUT / "predictions.csv")
membership=pd.read_csv(OUTPUT / "training_rows.csv")
inner=pd.read_csv(OUTPUT / "inner_rows.csv")
fit_count,forecast_rows,inner_fits,selection_count=0,0,0,0
checked_outer_dates={}
for path in sorted((OUTPUT / "models").glob("*.json")):
    state=json.loads(path.read_text());fit=state["fit"]
    position=positions[fit["fit_information_session"]]
    dates=allowed(days[max(0,position-CONFIG["training"]["max_window_sessions"]+1):position+1],fit["simulation_fit_at"])
    observed=membership.loc[membership.fit_session.eq(fit["fit_session"]),"as_of_session"].tolist()
    assert dates==observed and len(dates)==fit["training_rows"] >= 400
    assert dates[-1]==fit["training_last"] and dates[0]==fit["training_first"]
    assert fit["positive_count"]==sum(target(dates,"Y10"))
    checked_outer_dates[fit["fit_session"]]=dates
    if state["transform"]:
        validate_transform(state["transform"],dates)
    else:
        track("m0_constants",[state["heads"]["mu5"]["value"],state["heads"]["q90"]["value"],state["heads"]["p10"]["value"]],
              [np.mean(target(dates,"R5")),np.quantile(target(dates,"L5"),.9,method="linear"),(sum(target(dates,"Y10"))+.5)/(len(dates)+1)])
    rows=predictions.loc[predictions.fit_id.eq(fit["fit_id"]) & predictions.forecast_available]
    month_rows=predictions.loc[predictions.forecast_available & predictions.model.eq(state["model"])
                              & predictions.decision_session.str[:7].eq(fit["fit_session"][:7])]
    assert fit["fit_session"]==month_rows.decision_session.min()
    x=inputs(rows.as_of_session.tolist())
    z=matrix(state["transform"],x) if state["transform"] else None
    assert (rows.simulation_decision_at >= fit["simulation_fit_at"]).all()
    for head in ["mu5","q90","p10"]:
        score=raw_score(head,state["heads"][head],z,len(x))
        track("predictions",rows[head],prediction(head,score))
        raw_field={"mu5":"mu5_raw","q90":"q90_raw","p10":"p10_logit"}[head]
        track("predictions",rows[raw_field],score)
    if state["transform"]:
        t=state["transform"];lo,hi=np.asarray(t["raw_min"]),np.asarray(t["raw_max"])
        outside=(x<lo)|(x>hi)
        track("support",rows.outside_feature_count,outside.sum(axis=1))
        track("support",rows.max_outside_distance_training_sd,
              (np.maximum(np.maximum(lo-x,x-hi),0)/np.asarray(t["raw_scaler"]["scale"])).max(axis=1))
    forecast_rows+=len(rows);fit_count+=1

for path in sorted((OUTPUT / "selections").glob("*.json")):
    selection=json.loads(path.read_text());key=selection["selection_id"]
    outer=checked_outer_dates[key]
    end=positions[outer[-1]]
    expected_blocks=[days[end+1-189+b*63:end+1-189+(b+1)*63] for b in range(3)]
    for model,saved in selection["models"].items():
        fold_transforms={r["block"]:r["transform"] for r in saved["fold_transforms"]}
        for block,dates in enumerate(expected_blocks,1):
            cutoff=labels[dates[0]]["decision_at"]
            expected_train=allowed([d for d in outer if d<dates[0]],cutoff)
            expected_val=[d for d in dates if d in outer]
            t=inner.loc[inner.selection_id.eq(key)&inner.block.eq(block)&inner.role.eq("train"),"as_of_session"].tolist()
            v=inner.loc[inner.selection_id.eq(key)&inner.block.eq(block)&inner.role.eq("validation"),"as_of_session"].tolist()
            assert t==expected_train and v==expected_val and len(t)>=200 and len(v)==63
            transform=fold_transforms[block];validate_transform(transform,t)
            z=matrix(transform,inputs(v))
            for score in [r for r in saved["scores"] if r["block"]==block]:
                head=score["head"];label={"mu5":"R5","q90":"L5","p10":"Y10"}[head]
                p=prediction(head,raw_score(head,score["fit"],z,len(v)))
                track("cv_losses",[score["loss"]],[loss(head,target(v,label),p)])
                inner_fits+=1
        for head in ["mu5","q90","p10"]:
            totals=[]
            for item in [r for r in saved["totals"] if r["head"]==head]:
                chunks=[r for r in saved["scores"] if r["head"]==head and r["parameter"]==item["parameter"]]
                expected=math.fsum(r["loss"]*r["validation_rows"] for r in chunks)/sum(r["validation_rows"] for r in chunks)
                track("cv_losses",[item["loss"]],[expected]);totals.append((expected,item["parameter"]))
            preferred=sorted(totals,key=lambda t:t[1],reverse=head!="p10")
            best_score,best_parameter=float("inf"),None
            for score,parameter in preferred:
                if score<best_score-1e-12:best_score,best_parameter=score,parameter
            assert saved["parameters"][head]==best_parameter
    selection_count+=1

metrics=pd.read_csv(OUTPUT / "metrics.csv")
scored=predictions.loc[predictions.forecast_available & predictions.score_observed]
for row in metrics.itertuples():
    subset=scored.loc[scored.model.eq(row.model)]
    if row.period_type=="year":subset=subset.loc[subset.decision_session.str[:4].eq(row.period)]
    if row.period_type=="quarter":subset=subset.loc[pd.to_datetime(subset.decision_session).dt.to_period("Q").astype(str).eq(row.period)]
    dates=subset.as_of_session.tolist()
    yret,yloss,event=[target(dates,k) for k in TARGETS]
    expected=[loss("mu5",yret,subset.mu5),loss("p10",event,subset.p10),loss("mu5",event,subset.p10),loss("q90",yloss,subset.q90),
              float(np.mean(yloss<=subset.q90.to_numpy())),float(np.mean(subset.p10)),float(np.mean(event)),float(np.mean(subset.q90))]
    observed=[row.mse,row.log_loss,row.brier,row.pinball,row.q90_coverage,row.mean_probability,row.event_rate,row.mean_q90]
    track("metrics",observed,expected)
    assert row.rows==len(subset) and row.events==sum(event)

assert not predictions.loc[predictions.forecast_available & ~predictions.score_observed,TARGETS].notna().any().any()
assert scored.label_end_session.lt("2024-01-01").all()
calibration=pd.read_csv(OUTPUT / "calibration.csv")
for row in calibration.itertuples():
    subset=scored.loc[scored.model.eq(row.model)]
    subset=subset.loc[(np.minimum((subset.p10*5).astype(int),4)+1)==row.bin]
    assert row.rows==len(subset) and row.events==subset.Y10.sum()
    if len(subset):track("metrics",[row.mean_probability,row.event_rate],[subset.p10.mean(),subset.Y10.mean()])

result={"checked_at":datetime.now(timezone.utc).isoformat(),"run_record":RUN.relative_to(ROOT).as_posix(),
        "audit_source_sha256":sha256(Path(__file__).read_bytes()).hexdigest(),
        "method":"直接从P3原始特征/P2标签选择成熟成员，独立重建训练尺度、样条、交互和已保存系数的全部预测；逐标量重算候选损失及评估，无生产模型/评分函数。",
        "monthly_models_checked":fit_count,"forecast_rows_replayed":forecast_rows,"prediction_values_checked":forecast_rows*6,
        "annual_selections_checked":selection_count,"inner_head_fits_checked":inner_fits,"metric_rows_checked":len(metrics),
        "training_cutoffs_and_membership_match":True,"no_locked_labels_used":True,"max_absolute_errors":errors,"passed":True}
audit_output = ROOT / "audit_results" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") if BUNDLE else RUN.parent
audit_output.mkdir(parents=True, exist_ok=True)
(audit_output / "independent_validation.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
print(json.dumps(result,ensure_ascii=False,indent=2))
