"""按真实决策月份重放 P4；预测、拟合、候选失败与未服务日期分别保存。"""

import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, mean_pinball_loss, mean_squared_error

from svxylab.features import CORE_IDS
from svxylab.features_report import write_json
from svxylab.models import HEADS, MODELS, fit_joint, select_parameters
from svxylab.prediction_data import TARGETS, eligibility, event_clusters, training_rows


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def training_info(train, cutoff, position, frame, config):
    return {"training_first": train.index[0], "training_last": train.index[-1], "training_rows": len(train),
            "positive_count": int(train.Y10.sum()), "event_clusters": event_clusters(train),
            "calendar_window_first": frame.index[max(0, position-config["training"]["max_window_sessions"]+1)],
            "calendar_window_last": frame.index[position], "simulation_fit_at": cutoff.isoformat(),
            "latest_label_matures_at": train.label_matures_at.max().isoformat(),
            "latest_label_available_at": train.label_available_at.max().isoformat(),
            "latest_feature_available_at": train.assumed_available_at.max().isoformat(),
            "training_values_sha256": sha256(train[CORE_IDS+TARGETS].to_csv().encode()).hexdigest()}


def run_predictions(frame, config, interactions, output: Path, *, pilot=False):
    started = perf_counter()
    output.mkdir(parents=True, exist_ok=False)
    (output / "models").mkdir()
    (output / "selections").mkdir()
    forecasts, service, fits, memberships, inner_memberships = [], [], [], [], []
    cv_scores, cv_totals, selected_parameters = [], [], []
    active, model_by_fit, fit_by_id, annual_parameters = {}, {}, {}, {}
    first_month, active_month, last_selection_year = None, None, None
    prediction_fields = ["as_of_session", "decision_session", "simulation_decision_at", "execution_at", "computed_at_utc", "model", "fit_id",
                         "forecast_available", "forecast_status", "mu5_raw", "mu5", "q90_raw", "q90", "p10_logit", "p10", "mu_projected", "q_projected",
                         "outside_feature_count", "outside_feature_ids", "max_outside_distance_training_sd", "R5", "L5", "Y10", "score_observed", "score_status", "label_end_session", "label_matures_at", "label_available_at"]
    with (output / "predictions.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=prediction_fields)
        writer.writeheader()
        for position, row in enumerate(frame.itertuples()):
            if not config["data"]["outer_test_requested_start"] <= row.decision_session < config["data"]["locked_historical_start"]:
                continue
            month, year = row.decision_session[:7], row.decision_session[:4]
            if pilot and first_month is not None and month != first_month:
                break
            due = active_month != month
            need_selection = last_selection_year != year
            train, splits, reason = eligibility(frame, position, config, need_selection=due and need_selection)
            if due and reason == "ELIGIBLE":
                computed = utc_now()
                cycle_start = perf_counter()
                if need_selection:
                    selection_id = row.decision_session
                    selection = {"selection_id": selection_id, "simulation_selection_at": row.decision_at.isoformat(),
                                 "computed_at_utc": computed, "training": training_info(train,row.decision_at,position,frame,config),
                                 "models": {}}
                    for inner_train, validation, info in splits:
                        for role, subset in [("train", inner_train), ("validation", validation)]:
                            for d,r in subset.iterrows():
                                inner_memberships.append({"selection_id": selection_id, "block": info["block"], "role": role,
                                                          "as_of_session": d, "validation_cutoff": info["validation_cutoff"],
                                                          "label_matures_at": r.label_matures_at.isoformat(), "label_available_at": r.label_available_at.isoformat(),
                                                          "feature_available_at": r.assumed_available_at.isoformat()})
                    for model in ["M1", "M2"]:
                        parameters, scores, totals, transforms = select_parameters(splits, model, config, interactions)
                        annual_parameters[model] = parameters
                        selection["models"][model] = {"parameters": parameters, "scores": scores, "totals": totals, "fold_transforms": transforms}
                        cv_scores.extend({"selection_id": selection_id, **{k:v for k,v in s.items() if k != "fit"},
                                          "single_class_fallback": s["fit"].get("fallback")} for s in scores)
                        cv_totals.extend({"selection_id": selection_id, **s} for s in totals)
                        selected_parameters.append({"selection_id": selection_id, "model": model, **parameters})
                        # 任一后续拟合失败时，已经完成的候选结果仍在本地。
                        write_json(output / "selections" / f"{selection_id}.json", selection)
                    selection["elapsed_seconds"] = perf_counter()-cycle_start
                    write_json(output / "selections" / f"{selection_id}.json", selection)
                    last_selection_year = year
                common = training_info(train, row.decision_at, position, frame, config)
                for d,r in train.iterrows():
                    memberships.append({"fit_session": row.decision_session, "as_of_session": d,
                                        "label_matures_at": r.label_matures_at.isoformat(), "label_available_at": r.label_available_at.isoformat(),
                                        "feature_available_at": r.assumed_available_at.isoformat(), "Y10": int(r.Y10)})
                for model in MODELS:
                    fit_start = perf_counter()
                    fitted = fit_joint(model, train, annual_parameters.get(model, {}), config, interactions)
                    fit_id = f"{row.decision_session}_{model}"
                    summary = {"fit_id": fit_id, "model": model, "fit_session": row.decision_session, "fit_information_session": row.Index,
                               "selection_id": selection_id if model != "M0" else "", **common, "computed_at_utc": utc_now(),
                               "elapsed_seconds": perf_counter()-fit_start, "design_columns": len(fitted.design.columns) if fitted.design else 0,
                               "mu5_parameter": annual_parameters.get(model, {}).get("mu5"),
                               "q90_parameter": annual_parameters.get(model, {}).get("q90"),
                               "p10_parameter": annual_parameters.get(model, {}).get("p10"),
                               "p10_fallback": fitted.heads["p10"].get("fallback"),
                               "warning_count": sum(len(h["warnings"]) for h in fitted.heads.values())}
                    state = {**fitted.snapshot(), "fit": summary}
                    write_json(output / "models" / f"{fit_id}.json", state)
                    active[model] = (fit_id, fitted)
                    model_by_fit[fit_id] = fitted
                    fit_by_id[fit_id] = summary
                    fits.append(summary)
                active_month = month
                first_month = first_month or month
                print(f"拟合 {row.decision_session}：成熟训练 {len(train)} 行、阳性 {int(train.Y10.sum())}、重叠事件簇 {common['event_clusters']}；"
                      f"{'年度选参 + ' if need_selection else ''}M0/M1/M2 完成，{perf_counter()-cycle_start:.2f} 秒。", flush=True)
                pd.DataFrame(fits).to_csv(output / "fits.csv", index=False)
            complete_features = np.isfinite(frame.loc[row.Index, CORE_IDS].to_numpy(dtype=float)).all()
            feature_ready = row.assumed_available_at <= row.decision_at
            usable = bool(active) and complete_features and feature_ready
            status = "FORECAST_AVAILABLE" if usable else reason if not active else "MISSING_CORE_FEATURES"
            service.append({"as_of_session": row.Index, "decision_session": row.decision_session,
                            "simulation_decision_at": row.decision_at.isoformat(), "eligible_training_rows": len(train),
                            "core_complete": bool(complete_features), "forecast_available": usable, "reason": status,
                            "fit_session_used": active["M0"][0].rsplit("_",1)[0] if usable else ""})
            raw = frame.loc[[row.Index], CORE_IDS].to_numpy(dtype=float)
            for model in MODELS:
                record = {k: None for k in prediction_fields}
                record.update({"as_of_session": row.Index, "decision_session": row.decision_session,
                               "simulation_decision_at": row.decision_at.isoformat(), "execution_at": row.execution_at.isoformat(),
                               "computed_at_utc": utc_now(), "model": model, "forecast_available": usable, "forecast_status": status,
                               "fit_id": active[model][0] if usable else ""})
                if usable:
                    fitted = active[model][1]
                    prediction = fitted.predict(raw)
                    record.update({k: v[0].item() for k,v in prediction.items()})
                    if fitted.design:
                        outside, distance = fitted.design.support(raw)
                        record.update({"outside_feature_count": int(outside.sum()), "outside_feature_ids": "|".join(np.array(CORE_IDS)[outside[0]]),
                                       "max_outside_distance_training_sd": float(distance[0])})
                observed = bool(row.observed and np.isfinite([row.R5, row.L5, row.Y10]).all())
                record.update({k: float(getattr(row,k)) if observed else None for k in TARGETS})
                record.update({"score_observed": observed, "score_status": "DEVELOPMENT_LABEL_COMPLETE" if observed else row.reason,
                               "label_end_session": row.label_end_session, "label_matures_at": row.label_matures_at.isoformat(),
                               "label_available_at": row.label_available_at.isoformat()})
                writer.writerow(record)
                forecasts.append(record)
            stream.flush()
    if not active:
        raise ValueError("开发区间内没有满足全部条件的训练时点；保留无预测日期，不降低样本门槛")
    predictions = pd.DataFrame(forecasts)
    services = pd.DataFrame(service)
    services.to_csv(output / "prediction_service.csv", index=False)
    pd.DataFrame(memberships).to_csv(output / "training_rows.csv", index=False)
    pd.DataFrame(inner_memberships).to_csv(output / "inner_rows.csv", index=False)
    pd.DataFrame(cv_scores).to_csv(output / "cv_scores.csv", index=False)
    pd.DataFrame(cv_totals).to_csv(output / "cv_totals.csv", index=False)
    pd.DataFrame(selected_parameters).to_csv(output / "selected_parameters.csv", index=False)
    interaction_diagnostics(frame, predictions, model_by_fit, fit_by_id, output)
    evaluation = evaluate_predictions(predictions, output, config["targets"]["loss_quantile"])
    valid = services.loc[services.forecast_available]
    return {"pilot": pilot, "elapsed_seconds": perf_counter()-started, "requested_decision_first": services.decision_session.iloc[0],
            "processed_decision_last": services.decision_session.iloc[-1], "request_sessions": len(services),
            "forecast_sessions": len(valid), "no_forecast_sessions": int((~services.forecast_available).sum()),
            "first_prediction_information_session": valid.as_of_session.iloc[0], "first_prediction_decision_session": valid.decision_session.iloc[0],
            "last_prediction_information_session": valid.as_of_session.iloc[-1], "model_fits": len(fits),
            "monthly_fit_cycles": len(fits)//3, "selection_cycles": len(selected_parameters)//2,
            "first_fit": fits[0], "last_fit": fits[-1], "evaluation": evaluation,
            "fitting_warning_count": sum(r["warning_count"] for r in fits),
            "outer_single_class_fallbacks": sum(bool(r["p10_fallback"]) for r in fits),
            "inner_single_class_fallbacks": sum(bool(r["single_class_fallback"]) for r in cv_scores)}


def metrics_for(frame, quantile):
    return {"rows": len(frame), "events": int(frame.Y10.sum()), "first_decision": frame.decision_session.iloc[0],
            "last_decision": frame.decision_session.iloc[-1], "mse": float(mean_squared_error(frame.R5,frame.mu5)),
            "log_loss": float(log_loss(frame.Y10,frame.p10,labels=[0,1])), "brier": float(brier_score_loss(frame.Y10,frame.p10)),
            "pinball": float(mean_pinball_loss(frame.L5,frame.q90,alpha=quantile)),
            "q90_coverage": float((frame.L5 <= frame.q90).mean()), "mean_probability": float(frame.p10.mean()),
            "event_rate": float(frame.Y10.mean()), "mean_q90": float(frame.q90.mean())}


def evaluate_predictions(predictions, output, quantile):
    scored = predictions.loc[predictions.forecast_available & predictions.score_observed].copy()
    all_metrics = []
    for model in MODELS:
        subset = scored.loc[scored.model.eq(model)].sort_values("decision_session")
        all_metrics.append({"model": model, "period_type": "all", "period": "all", **metrics_for(subset, quantile)})
        for kind, key in [("year", subset.decision_session.str[:4]), ("quarter", pd.to_datetime(subset.decision_session).dt.to_period("Q").astype(str))]:
            for period, group in subset.groupby(key, sort=True):
                all_metrics.append({"model": model, "period_type": kind, "period": period, **metrics_for(group,quantile)})
    metrics = pd.DataFrame(all_metrics)
    for metric in ["mse", "log_loss", "brier", "pinball"]:
        reference = metrics.loc[metrics.model.eq("M0")].set_index(["period_type", "period"])[metric]
        metrics[f"{metric}_delta_M0"] = [row[metric]-reference.loc[(row.period_type,row.period)] for _,row in metrics.iterrows()]
        metrics[f"{metric}_relative_change_M0"] = [row[metric]/reference.loc[(row.period_type,row.period)]-1
                                                      if reference.loc[(row.period_type,row.period)] > 0 else np.nan for _,row in metrics.iterrows()]
    metrics.to_csv(output / "metrics.csv", index=False)
    losses = scored[["as_of_session", "decision_session", "model"]].copy()
    losses["squared_error"] = (scored.R5-scored.mu5).pow(2)
    p = np.clip(scored.p10.to_numpy(dtype=float), np.finfo(float).eps, 1-np.finfo(float).eps)
    losses["log_loss"] = -(scored.Y10*np.log(p)+(1-scored.Y10)*np.log1p(-p))
    losses["brier"] = (scored.Y10-scored.p10).pow(2)
    error = scored.L5-scored.q90
    losses["pinball"] = np.maximum(quantile*error,(quantile-1)*error)
    losses.to_csv(output / "daily_losses.csv", index=False)
    calibration, risk, projections, support = [], [], [], []
    for model in MODELS:
        subset = scored.loc[scored.model.eq(model)]
        bin_ids = np.minimum((subset.p10.to_numpy()*5).astype(int), 4)
        for b in range(5):
            group = subset.loc[bin_ids == b]
            calibration.append({"model": model, "bin": b+1, "lower": b/5, "upper": (b+1)/5, "rows": len(group),
                                "events": int(group.Y10.sum()), "mean_probability": float(group.p10.mean()) if len(group) else None,
                                "event_rate": float(group.Y10.mean()) if len(group) else None})
        bounds = np.unique(np.quantile(subset.q90, np.linspace(0,1,6), method="linear"))
        groups = pd.cut(subset.q90, bounds, include_lowest=True, duplicates="drop") if len(bounds)>1 else pd.Series("constant",index=subset.index)
        for i, (interval, group) in enumerate(subset.groupby(groups,observed=True,sort=True),1):
            risk.append({"model": model, "group": i, "q90_range": str(interval), "rows": len(group), "events": int(group.Y10.sum()),
                         "mean_predicted_q90": float(group.q90.mean()), "observed_loss_quantile90": float(np.quantile(group.L5,quantile,method="linear")),
                         "observed_mean_loss": float(group.L5.mean()), "observed_event_rate": float(group.Y10.mean()),
                         "coverage": float((group.L5<=group.q90).mean())})
        all_predictions = predictions.loc[predictions.model.eq(model) & predictions.forecast_available]
        projections.append({"model": model, "forecast_rows": len(all_predictions), "scored_rows": len(subset),
                            "unscored_rows": len(all_predictions)-len(subset), "mu_projected": int(all_predictions.mu_projected.sum()),
                            "q_projected": int(all_predictions.q_projected.sum()), "min_raw_mu5": float(all_predictions.mu5_raw.min()),
                            "min_raw_q90": float(all_predictions.q90_raw.min()), "max_raw_q90": float(all_predictions.q90_raw.max())})
        if model != "M0":
            for year, group in all_predictions.groupby(all_predictions.decision_session.str[:4]):
                support.append({"model": model, "year": year, "rows": len(group),
                                "outside_rows": int(group.outside_feature_count.gt(0).sum()),
                                "max_outside_features": int(group.outside_feature_count.max()),
                                "max_outside_distance_training_sd": float(group.max_outside_distance_training_sd.max())})
    pd.DataFrame(calibration).to_csv(output / "calibration.csv", index=False)
    pd.DataFrame(risk).to_csv(output / "risk_groups.csv", index=False)
    pd.DataFrame(projections).to_csv(output / "projections.csv", index=False)
    pd.DataFrame(support).to_csv(output / "support.csv", index=False)
    paired = []
    one = scored.loc[scored.model.eq("M1")].set_index("decision_session")
    two = scored.loc[scored.model.eq("M2")].set_index("decision_session")
    if not one.index.equals(two.index):
        raise ValueError("M1/M2 预测日期不一致，不能未经配对比较")
    for field in ["mu5", "q90", "p10"]:
        delta = two[field]-one[field]
        paired.append({"output": field, "rows": len(delta), "mean_M2_minus_M1": float(delta.mean()),
                       "mean_absolute_difference": float(delta.abs().mean()), "max_absolute_difference": float(delta.abs().max())})
    pd.DataFrame(paired).to_csv(output / "model_disagreement.csv", index=False)
    return {"overall": metrics.loc[metrics.period_type.eq("all")].to_dict("records"),
            "projections": projections, "support": support, "model_disagreement": paired,
            "calibration": calibration, "risk_groups": risk}


def interaction_diagnostics(frame, predictions, models, fits, output):
    conditional, local = [], []
    for fit_id, fitted in models.items():
        if fitted.model != "M2":
            continue
        d = fitted.design
        base = len(CORE_IDS)*d.basis_per_feature
        for j, interaction in enumerate(d.interactions):
            a,b = d.pairs[j]
            for head in HEADS:
                state = fitted.heads[head]
                beta = state["coef"][base+j] if state["kind"] != "constant" else 0.0
                for q, partner in zip([.25,.5,.75], d.z_quantiles[:,b], strict=True):
                    conditional.append({"fit_id": fit_id, "fit_session": fits[fit_id]["fit_session"], "interaction": interaction["id"],
                                        "head": head, "changed_feature": CORE_IDS[a], "partner_feature": CORE_IDS[b],
                                        "partner_training_quantile": q, "partner_z": float(partner),
                                        "interaction_slope_per_training_sd": float(beta*partner/d.output_scaler.scale_[base+j]),
                                        "unit": "log_odds" if head=="p10" else "raw_decimal_prediction"})
        dates = predictions.loc[predictions.fit_id.eq(fit_id), "as_of_session"].tolist()
        for day in sorted(set([dates[0], dates[len(dates)//2], dates[-1]])):
            raw = frame.loc[[day],CORE_IDS].to_numpy(dtype=float)
            z = d.raw_scaler.transform(raw)[0]
            plus, minus = np.repeat(raw, len(CORE_IDS), axis=0), np.repeat(raw, len(CORE_IDS), axis=0)
            for i in range(len(CORE_IDS)):
                plus[i,i] += 1e-4*d.raw_scaler.scale_[i]
                minus[i,i] -= 1e-4*d.raw_scaler.scale_[i]
            up, down = fitted.scores(plus), fitted.scores(minus)
            for i, feature in enumerate(CORE_IDS):
                for head in HEADS:
                    derivative = float((up[head][i]-down[head][i])/2e-4)
                    state = fitted.heads[head]
                    interaction_part = 0.0
                    if state["kind"] != "constant":
                        for j, (a,b) in enumerate(d.pairs):
                            if i in (a,b):
                                interaction_part += state["coef"][base+j]*z[b if i==a else a]/d.output_scaler.scale_[base+j]
                    local.append({"fit_id": fit_id, "as_of_session": day, "feature": feature, "head": head,
                                  "raw_feature_value": float(raw[0,i]), "training_z": float(z[i]),
                                  "local_total_slope_per_training_sd": derivative, "interaction_component": float(interaction_part),
                                  "spline_component": derivative-interaction_part,
                                  "unit": "log_odds" if head=="p10" else "raw_decimal_prediction"})
    pd.DataFrame(conditional).to_csv(output / "interaction_conditions.csv", index=False)
    pd.DataFrame(local).to_csv(output / "local_effects.csv", index=False)
