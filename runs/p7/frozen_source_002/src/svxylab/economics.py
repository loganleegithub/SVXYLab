"""P5：冻结预测到连续账户；不拟合模型、不读取封存收益或评分标签。"""

import csv
from hashlib import sha256
import json
from pathlib import Path
import tomllib

import numpy as np
import pandas as pd

from svxylab.baselines import NAMES as BASELINE_NAMES, verify_ledger
from svxylab.features_report import verify_hashes, write_json
from svxylab.ledger import baseline_targets, continuous_ledger

MODELS = ("M0", "M1", "M2")
POLICIES = ("main", "risk_only")
NAMES = {**{f"{m}_{p}": f"{m} {'主映射' if p == 'main' else '仅风险容量'}" for m in MODELS for p in POLICIES}, **BASELINE_NAMES}
PRICE_FIELDS = ["as_of_session", "close", "split_factor", "cash_dividend", "capital_gain_distribution", "simple_return", "total_return_index"]
CLOCK_FIELDS = ["as_of_session", "information_close_at", "decision_session", "decision_at", "execution_session", "execution_at"]
PREDICTION_FIELDS = ["as_of_session", "decision_session", "simulation_decision_at", "execution_at", "model", "fit_id", "forecast_available", "mu5", "q90"]


def bounded_csv(path, columns, locked="2024-01-01"):
    """过滤行后才选字段/转数值；不将2024以后结果加载为研究数值。"""
    rows = []
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["as_of_session"] >= locked:
                break
            rows.append({key: row[key] for key in columns})
    return pd.DataFrame(rows, columns=columns)


def load_inputs(root):
    config = tomllib.loads((root / "experiment.toml").read_text())
    accepted = json.loads((root / "runs/p4/acceptance.json").read_text())
    verify_hashes(root, {accepted["accepted_run"]: accepted["accepted_run_sha256"]})
    verify_hashes(root, accepted["accepted_artifact_sha256"])
    verify_hashes(root, accepted["model_source_sha256"])
    stages, hashes = {}, dict(accepted["accepted_artifact_sha256"])
    for stage in ["p2", "p3"]:
        a = json.loads((root / f"runs/{stage}/acceptance.json").read_text())
        verify_hashes(root, {a["accepted_run"]: a["accepted_run_sha256"]})
        verify_hashes(root, a["accepted_artifact_sha256"])
        stages[stage] = json.loads((root / a["accepted_run"]).read_text())["result"]["output_dir"]
        hashes.update(a["accepted_artifact_sha256"])
    p1 = json.loads((root / "runs/p1/reproducibility.json").read_text())
    curve_path = "data/clean/VX_front_three.csv"
    verify_hashes(root, {curve_path: p1["clean_sha256"][curve_path]})
    files = {"prices": stages["p2"]+"/SVXY_returns.csv", "clock": stages["p2"]+"/clock.csv",
             "predictions": accepted["accepted_prediction_file"], "curve": curve_path,
             "features": stages["p3"]+"/core_features.csv"}
    prices = bounded_csv(root / files["prices"], PRICE_FIELDS)
    clock = bounded_csv(root / files["clock"], CLOCK_FIELDS)
    predictions = bounded_csv(root / files["predictions"], PREDICTION_FIELDS)
    curve = bounded_csv(root / files["curve"], ["as_of_session", "f1_settle", "f2_settle"])
    dictionary = json.loads((root / "FEATURES.json").read_text())
    ids = [f["id"] for f in dictionary["features"] if f["tier"] == "core"]
    features = bounded_csv(root / files["features"], ["as_of_session", *ids])
    for frame, columns in [(prices, PRICE_FIELDS[1:]), (predictions, ["mu5", "q90"]),
                           (curve, ["f1_settle", "f2_settle"]), (features, ids)]:
        for column in columns:
            frame[column] = pd.to_numeric(frame[column].replace("", np.nan), errors="raise")
    predictions["forecast_available"] = predictions.forecast_available.eq("True")
    dates = prices.as_of_session.tolist()
    if not (dates == clock.as_of_session.tolist() == curve.as_of_session.tolist() == features.as_of_session.tolist()):
        raise ValueError("P5需要同一完整开发段日历，不允许跳过缺口")
    if len(set(dates)) != len(dates) or dates != sorted(dates):
        raise ValueError("P5日历必须唯一递增")
    if config["portfolio"]["cash_annual_return_base"] != 0:
        raise ValueError("当前账本仅实现规格中现金收益0")
    for model in MODELS:
        rows = predictions.loc[predictions.model.eq(model)]
        if rows.as_of_session.duplicated().any():
            raise ValueError("同模型同信息日只能有一份已验收预测")
        timing = clock.set_index("as_of_session").loc[rows.as_of_session]
        if not (np.array_equal(rows.decision_session, timing.decision_session)
                and np.array_equal(pd.to_datetime(rows.execution_at, utc=True), pd.to_datetime(timing.execution_at, utc=True))):
            raise ValueError("冻结预测与已验收执行时钟不一致")
    input_record = {"files": files, "sha256": {p: sha256((root / p).read_bytes()).hexdigest() for p in files.values()},
                    "p4_acceptance_sha256": sha256((root / "runs/p4/acceptance.json").read_bytes()).hexdigest(),
                    "prediction_fields_read": PREDICTION_FIELDS, "labels_or_score_observed_read": False,
                    "locked_returns_read": False, "model_fit_or_selection_performed": False,
                    "first_information": dates[0], "last_information": dates[-1], "information_rows": len(dates)}
    return prices, clock, predictions, curve, features, config, input_record


def model_targets(predictions, sessions, budget, filter_cost):
    """缺预测返回NaN而非前填仓位；结果评分字段/P10从未参与。"""
    if budget <= 0 or not 0 <= filter_cost < 1:
        raise ValueError("预算/成本无效")
    index = pd.Index(sessions, name="as_of_session")
    targets, metadata = {}, {}
    for model in MODELS:
        selected = predictions.loc[predictions.model.eq(model)].set_index("as_of_session").reindex(index)
        available = selected.forecast_available.fillna(False).astype(bool)
        valid = selected.loc[available, ["mu5", "q90"]]
        if not np.isfinite(valid.to_numpy(float)).all() or not valid.q90.between(0, 1).all():
            raise ValueError("可用预测必须为有限发布值，Q90在[0,1]")
        risk = pd.Series(np.nan, index=index)
        q = selected.loc[available, "q90"]
        risk.loc[available] = np.minimum(1.0, budget / q.replace(0, np.nan)).fillna(1.0)
        main = risk.copy()
        main.loc[available & selected.mu5.le(2*filter_cost)] = 0.0
        for policy, target in [("main", main), ("risk_only", risk)]:
            name = f"{model}_{policy}"
            targets[name] = target
            metadata[name] = pd.DataFrame({"source_information_session": index,
                "source_prediction_decision_at": selected.simulation_decision_at,
                "source_forecast_available": available, "source_fit_id": selected.fit_id,
                "source_mu5": selected.mu5, "source_q90": selected.q90,
                "risk_capacity": risk, "mu_filter_pass": available & selected.mu5.gt(2*filter_cost),
                "planned_target": target}, index=index)
    return targets, metadata


def scenarios(config):
    p = config["portfolio"]
    base = {"budget": p["loss_budget_base"], "cost_rate": p["cost_one_way_bps_base"]/10000, "delay": 0}
    result = [{"scenario": "base", **base, "factor": "base"}]
    result += [{"scenario": f"cost_{v:g}bp", **base, "cost_rate": v/10000, "factor": "cost"} for v in p["cost_one_way_bps_sensitivities"]]
    result += [{"scenario": f"budget_{v*100:g}pct", **base, "budget": v, "factor": "budget"} for v in p["loss_budget_sensitivities"]]
    result.append({"scenario": "delay_1session", **base, "delay": config["clock"]["extra_delay_sensitivity_sessions"], "factor": "delay"})
    return result


def run_account(prices, clock, target, start, capital, rate, *, metadata=None, delay=0):
    """先在完整信息索引上延迟，然后从执行日前的现金锚点重建账户。"""
    dates = prices.as_of_session.tolist()
    first = dates.index(start)
    if first < 1:
        raise ValueError("首次执行前必须保留一个真实信息交易日")
    delayed = target.shift(delay)
    sliced_prices = prices.iloc[first-1:].reset_index(drop=True)
    sliced_clock = clock.iloc[first-1:].reset_index(drop=True)
    ledger = continuous_ledger(sliced_prices, sliced_clock, delayed.iloc[first-1:], capital=capital, cost_rate=rate)
    ledger["is_anchor"] = np.arange(len(ledger)) == 0
    ledger["execution_delay_sessions"] = delay
    if metadata is not None:
        # ledger每行用前一信息行的计划；元数据必须延迟完全相同的交易日数。
        used = metadata.shift(delay+1).iloc[first-1:].reset_index(drop=True)
        used.iloc[0] = None
        for c in used:
            ledger[c] = used[c]
        ledger["ever_forecast_executed"] = ledger.source_forecast_available.fillna(False).astype(bool).cummax()
    else:
        ledger["source_information_session"] = pd.Series(dates).shift(delay+1).iloc[first-1:].reset_index(drop=True)
        ledger.loc[0, "source_information_session"] = None
        ledger["source_forecast_available"] = ~ledger.signal_missing
        ledger["ever_forecast_executed"] = True
    return ledger


def account_metrics(ledger, capital):
    rows = ledger.loc[~ledger.is_anchor]
    n = len(rows)
    years = (pd.Timestamp(rows.as_of_session.iloc[-1])-pd.Timestamp(rows.as_of_session.iloc[0])).days/365.25
    five = ledger.equity_end.pct_change(5, fill_method=None)
    worst5 = five.idxmin() if five.notna().any() else None
    worst1 = rows.portfolio_return.idxmin()
    exposure = rows.old_weight
    return {"first_execution": rows.as_of_session.iloc[0], "last_close": rows.as_of_session.iloc[-1], "account_days": n,
            "anchor": ledger.as_of_session.iloc[0], "initial_capital": capital, "ending_equity": rows.equity_end.iloc[-1],
            "total_return": rows.equity_end.iloc[-1]/capital-1,
            "cagr": (rows.equity_end.iloc[-1]/capital)**(1/years)-1 if years else np.nan,
            "max_drawdown": ledger.drawdown.min(), "worst_day": rows.portfolio_return.min(),
            "worst_day_session": ledger.loc[worst1, "as_of_session"],
            "worst_five_days": five.min(), "worst_five_start": ledger.loc[worst5-5, "as_of_session"] if worst5 else None,
            "worst_five_end": ledger.loc[worst5, "as_of_session"] if worst5 else None,
            "mean_exposure": exposure.mean(), "mean_closing_exposure": rows.new_weight.mean(),
            "exposure_p25": exposure.quantile(.25), "exposure_median": exposure.median(), "exposure_p75": exposure.quantile(.75),
            "full_exposure_days": int(exposure.ge(1-1e-12).sum()), "cash_days": int(exposure.le(1e-12).sum()),
            "cash_fraction": exposure.le(1e-12).mean(), "average_cash_weight": (1-exposure).mean(),
            "total_turnover": rows.turnover.sum(), "annual_turnover": rows.turnover.sum()/years if years else np.nan,
            "trade_notional_dollars": rows.trade_notional.sum(), "cost_dollars": rows.cost.sum(),
            "cost_over_initial_capital": rows.cost.sum()/capital, "trade_days": int(rows.trade_notional.gt(1e-8).sum()),
            "missing_signal_days": int(rows.signal_missing.sum()),
            "startup_no_service_days": int((~rows.ever_forecast_executed).sum()),
            "missed_positive_day_arithmetic_return": ((1-exposure)*rows.asset_return.clip(lower=0)).sum(),
            "avoided_negative_day_arithmetic_loss": ((1-exposure)*(-rows.asset_return).clip(lower=0)).sum(),
            "cash_on_asset_up_days": int((exposure.le(1e-12) & rows.asset_return.gt(0)).sum()),
            "unserved_asset_return": np.prod(1+rows.loc[~rows.ever_forecast_executed, "asset_return"])-1}


def arithmetic_attribution(left, right):
    a, b = left.loc[~left.is_anchor], right.loc[~right.is_anchor]
    assert a.as_of_session.tolist() == b.as_of_session.tolist()
    exposure = float(((a.old_weight-b.old_weight)*a.asset_return).sum())
    friction = float((b.cost/b.equity_start-a.cost/a.equity_start).sum())
    delta = float((a.portfolio_return-b.portfolio_return).sum())
    assert abs(exposure+friction-delta) < 2e-11
    return {"arithmetic_exposure_contribution": exposure, "arithmetic_cost_contribution": friction,
            "arithmetic_daily_return_difference": delta,
            "compounded_total_return_difference": a.equity_end.iloc[-1]/left.equity_end.iloc[0]-b.equity_end.iloc[-1]/right.equity_end.iloc[0],
            "mean_exposure_difference": a.old_weight.mean()-b.old_weight.mean()}


def risk_diagnostics(prices, predictions, budget, cost):
    prices = prices.set_index("as_of_session")
    dates = prices.index.tolist()
    rows = []
    for model in MODELS:
        previous_q, previous_w = None, None
        for p in predictions.loc[predictions.model.eq(model) & predictions.forecast_available].itertuples():
            i = dates.index(p.decision_session)
            w = 1.0 if p.q90 == 0 else min(1.0, budget/p.q90)
            complete = i+5 < len(dates)
            path = prices.total_return_index.iloc[i:i+6].to_numpy() if complete else None
            loss = max(0, 1-min(path[1:])/path[0]) if complete else np.nan
            rows.append({"model": model, "as_of_session": p.as_of_session, "execution_session": p.decision_session,
                         "mu5": p.mu5, "q90": p.q90, "risk_capacity": w, "main_capacity": w if p.mu5 > 2*cost else 0,
                         "q90_zero": p.q90 == 0, "capacity_at_cap": w == 1,
                         "q90_lower_and_capacity_increased": previous_q is not None and p.q90 < previous_q and w > previous_w,
                         "path_complete_in_development": complete, "path_end": dates[i+5] if complete else None,
                         "subsequent_entry_relative_loss": loss, "loss_exceeds_q90": bool(loss > p.q90) if complete else None,
                         "static_risk_capacity_times_loss": w*loss,
                         "static_budget_exceeded": bool(w*loss > budget) if complete else None})
            previous_q, previous_w = p.q90, w
    return pd.DataFrame(rows)


def select_cases(ledgers, prices, predictions, features, output):
    """按计算前记录的M2主映射相对全仓的五日账户路径选4例；仅事后展示。"""
    main = ledgers[("common", "base", "M2_main")]
    full = ledgers[("common", "base", "fixed_100")]
    candidates = []
    for i in range(1, len(main)-5):
        m = main.equity_end.iloc[i+5]/main.equity_before_trade.iloc[i]-1
        b = full.equity_end.iloc[i+5]/full.equity_before_trade.iloc[i]-1
        candidates.append({"entry_index": i, "entry": main.as_of_session.iloc[i], "end": main.as_of_session.iloc[i+5],
                           "main_return": m, "fixed100_return": b, "difference": m-b})
    selected = []
    for kind, reverse in [("favorable", True), ("unfavorable", False)]:
        count = 0
        for candidate in sorted(candidates, key=lambda r: r["difference"], reverse=reverse):
            if any(candidate["entry"] <= prior["end"] and candidate["end"] >= prior["entry"] for prior in selected):
                continue
            selected.append({"case_id": f"{kind}_{count+1}", "kind": kind, **candidate})
            count += 1
            if count == 2:
                break
    assert len(selected) == 4
    case_rows, case_summary, info = [], [], []
    for case in selected:
        i = case["entry_index"]
        signal = main.source_information_session.iloc[i]
        raw = features.set_index("as_of_session").loc[signal]
        for m in MODELS:
            p = predictions.loc[predictions.model.eq(m) & predictions.as_of_session.eq(signal)].iloc[0]
            info.append({"case_id": case["case_id"], **p.to_dict(), **raw.to_dict()})
        for strategy in NAMES:
            ledger = ledgers[("common", "base", strategy)]
            path = ledger.iloc[i:i+6].copy()
            initial = path.equity_before_trade.iloc[0]
            path["normalized_equity"] = path.equity_end/initial
            path["case_id"], path["strategy"] = case["case_id"], strategy
            case_rows.append(path)
            case_summary.append({"case_id": case["case_id"], "strategy": strategy,
                                 "path_return_including_entry_fee": path.normalized_equity.iloc[-1]-1,
                                 "worst_entry_relative_account_close_loss": max(0, 1-path.normalized_equity.min()),
                                 "mean_old_weight_after_entry": path.old_weight.iloc[1:].mean(),
                                 "fees_during_path": path.cost.sum(), "entry_pretrade_equity": initial,
                                 "entry_new_weight": path.new_weight.iloc[0]})
    pd.DataFrame(selected).to_csv(output / "cases.csv", index=False)
    pd.concat(case_rows, ignore_index=True).to_csv(output / "case_daily_ledgers.csv", index=False)
    pd.DataFrame(case_summary).to_csv(output / "case_account_summary.csv", index=False)
    pd.DataFrame(info).to_csv(output / "case_information_predictions.csv", index=False)
    return selected


def run_economics(root, output):
    prices, clock, predictions, curve, features, config, inputs = load_inputs(root)
    output.mkdir(parents=True, exist_ok=False)
    account_dir = output / "ledgers"
    account_dir.mkdir()
    capital = config["portfolio"]["research_capital"]
    request_start = config["data"]["outer_test_requested_start"]
    available = predictions.loc[predictions.forecast_available]
    common = max(available.loc[available.model.eq(m), "decision_session"].min() for m in MODELS)
    dates = prices.as_of_session.tolist()
    preset = scenarios(config)
    summaries, audits, all_ledgers, trade_rows, gross_rows = [], [], {}, [], []
    base_targets = baseline_targets(curve, config["evaluation"]["fixed_weight_baselines"])
    for scenario in preset:
        targets, metas = model_targets(predictions, dates, scenario["budget"], scenario["cost_rate"])
        targets.update(base_targets)
        common_first = dates[dates.index(common)+scenario["delay"]]
        for scope, first in [("full", request_start), ("common", common_first)]:
            for name, target in targets.items():
                ledger = run_account(prices, clock, target, first, capital, scenario["cost_rate"],
                                     metadata=metas.get(name), delay=scenario["delay"])
                key = (scope, scenario["scenario"], name)
                audit = verify_ledger(ledger, capital, scenario["cost_rate"])
                if not all(audit["checks"].values()):
                    raise ValueError(f"真实P5账本对账失败：{key} {audit}")
                # 直接核对持有回报+实际费率；不能用同一累计净值恒等式掩盖时钟错误。
                rows = ledger.iloc[1:]
                np.testing.assert_allclose(rows.portfolio_return,
                    rows.old_weight*rows.asset_return-rows.cost/rows.equity_start, atol=2e-12, rtol=2e-10)
                path = f"ledgers/{scope}_{scenario['scenario']}_{name}.csv"
                ledger.to_csv(output / path, index=False)
                metric = {"scope": scope, **scenario, "strategy": name, "ledger_file": path, **account_metrics(ledger, capital)}
                summaries.append(metric)
                audits.append({"scope": scope, "scenario": scenario["scenario"], "strategy": name, **audit})
                all_ledgers[key] = ledger
                trades = ledger.loc[ledger.trade_notional.gt(1e-8)].copy()
                trades["scope"], trades["scenario"], trades["strategy"] = key
                trade_rows.append(trades)
                if scenario["scenario"] == "base":
                    gross = run_account(prices, clock, target, first, capital, 0, metadata=metas.get(name))
                    gross_path = f"ledgers/{scope}_base_same_targets_gross_{name}.csv"
                    gross.to_csv(output / gross_path, index=False)
                    gm = account_metrics(gross, capital)
                    gross_rows.append({"scope": scope, "strategy": name, "net_total_return": metric["total_return"],
                                       "same_targets_gross_return": gm["total_return"],
                                       "compounded_fee_drag": gm["total_return"]-metric["total_return"],
                                       "gross_ledger_file": gross_path})
            print(f"P5 {scope} / {scenario['scenario']}：{first}起12条独立账户完成。", flush=True)
    # 延迟比较必须在相同可执行起点重建基础账户；不是截取已运行账户。
    delay = next(s for s in preset if s["factor"] == "delay")
    first = dates[dates.index(common)+delay["delay"]]
    base = preset[0]
    targets, metas = model_targets(predictions, dates, base["budget"], base["cost_rate"])
    targets.update(base_targets)
    for name, target in targets.items():
        ledger = run_account(prices, clock, target, first, capital, base["cost_rate"], metadata=metas.get(name))
        path = f"ledgers/common_delay_aligned_base_{name}.csv"
        ledger.to_csv(output / path, index=False)
        all_ledgers[("common", "delay_aligned_base", name)] = ledger
        trades = ledger.loc[ledger.trade_notional.gt(1e-8)].copy()
        trades["scope"], trades["scenario"], trades["strategy"] = 'common', 'delay_aligned_base', name
        trade_rows.append(trades)
        summaries.append({"scope": "common", **base, "scenario": "delay_aligned_base", "factor": "start_alignment_reference",
                          "strategy": name, "ledger_file": path, **account_metrics(ledger, capital)})
        a = verify_ledger(ledger, capital, base["cost_rate"])
        assert all(a["checks"].values())
        audits.append({"scope": "common", "scenario": "delay_aligned_base", "strategy": name, **a})
    summary = pd.DataFrame(summaries)
    summary.to_csv(output / "account_metrics.csv", index=False)
    pd.concat(trade_rows, ignore_index=True).to_csv(output / "trades.csv", index=False)
    pd.DataFrame(gross_rows).to_csv(output / "same_target_gross_net.csv", index=False)
    write_json(output / "ledger_audit.json", audits)
    attributions, matched, exposure_rows, missed_rows, tails = [], [], [], [], []
    for scope in ["full", "common"]:
        for name in NAMES:
            ledger = all_ledgers[(scope, "base", name)]
            rows = ledger.iloc[1:]
            bins = np.where(rows.old_weight.le(1e-12), "cash", np.where(rows.old_weight.le(.25), "(0,25%]",
                   np.where(rows.old_weight.le(.5), "(25%,50%]", np.where(rows.old_weight.le(.75), "(50%,75%]", "(75%,100%]"))))
            for bucket in ["cash", "(0,25%]", "(25%,50%]", "(50%,75%]", "(75%,100%]"]:
                exposure_rows.append({"scope": scope, "strategy": name, "bucket": bucket,
                                      "days": int((bins == bucket).sum()), "fraction": float((bins == bucket).mean())})
            missed = rows.loc[rows.asset_return.gt(0)].copy()
            missed["unearned_asset_return"] = (1-missed.old_weight)*missed.asset_return
            missed = missed.nlargest(10, "unearned_asset_return")
            missed["scope"], missed["strategy"] = scope, name
            missed_rows.append(missed)
            if not name.startswith("M"):
                continue
            tail = rows.tail(5).copy()
            tail["scope"], tail["strategy"] = scope, name
            tails.append(tail)
            n = len(rows)
            weight = float(rows.old_weight.mean()*n/(n-1))
            assert -1e-12 <= weight <= 1+1e-12
            weight = min(1., max(0., weight))
            target = pd.Series(weight, index=pd.Index(dates, name="as_of_session"))
            match = run_account(prices, clock, target, rows.as_of_session.iloc[0], capital, base["cost_rate"])
            assert abs(match.old_weight.iloc[1:].mean()-rows.old_weight.mean()) < 1e-12
            match_path = f"ledgers/{scope}_POST_HOC_EXPOSURE_MATCH_{name}.csv"
            match.to_csv(output / match_path, index=False)
            matched.append({"scope": scope, "controller": name, "post_hoc_fixed_target": weight,
                            "controller_total_return": rows.equity_end.iloc[-1]/capital-1,
                            "ledger_file": match_path, **account_metrics(match, capital)})
            attributions.append({"scope": scope, "comparison": "POST_HOC_EXPOSURE_MATCH", "left": name,
                                 "right": f"matched_{name}", **arithmetic_attribution(ledger, match)})
        for model in MODELS:
            left, right = f"{model}_main", f"{model}_risk_only"
            attributions.append({"scope": scope, "comparison": "MU_FILTER", "left": left, "right": right,
                                 **arithmetic_attribution(all_ledgers[(scope, "base", left)], all_ledgers[(scope, "base", right)])})
        for model in ["M1", "M2"]:
            for policy in POLICIES:
                left, right = f"{model}_{policy}", f"M0_{policy}"
                attributions.append({"scope": scope, "comparison": "MODEL_VS_M0_SAME_MAPPING", "left": left, "right": right,
                                     **arithmetic_attribution(all_ledgers[(scope, "base", left)], all_ledgers[(scope, "base", right)])})
    pd.DataFrame(matched).to_csv(output / "post_hoc_exposure_matches.csv", index=False)
    pd.DataFrame(attributions).to_csv(output / "attribution.csv", index=False)
    pd.DataFrame(exposure_rows).to_csv(output / "exposure_distribution.csv", index=False)
    pd.concat(missed_rows, ignore_index=True).to_csv(output / "missed_up_days.csv", index=False)
    pd.concat(tails, ignore_index=True).to_csv(output / "last_five_2023_execution_days.csv", index=False)
    sensitivity = []
    for row in summary.loc[~summary.scenario.isin(["base", "delay_aligned_base"])].to_dict("records"):
        ref_id = "delay_aligned_base" if row["scope"] == "common" and row["factor"] == "delay" else "base"
        ref = summary.loc[summary.scope.eq(row["scope"]) & summary.scenario.eq(ref_id) & summary.strategy.eq(row["strategy"])].iloc[0]
        assert row["first_execution"] == ref.first_execution
        sensitivity.append({**row, "reference_scenario": ref_id,
                            **{f"{metric}_delta_reference": row[metric]-ref[metric] for metric in ["total_return", "max_drawdown", "mean_exposure", "cost_dollars"]}})
    pd.DataFrame(sensitivity).to_csv(output / "single_factor_sensitivities.csv", index=False)
    pending=[]
    for name,target in targets.items():
        planned = target.shift(delay['delay']).iloc[-1]
        effective = clock.iloc[-1]
        if pd.notna(planned):
            pending.append({'scenario':'delay_1session','strategy':name,
                            'source_information_session':dates[-1-delay['delay']],
                            'effective_decision_at':effective.decision_at,'scheduled_execution_session':effective.execution_session,
                            'scheduled_execution_at':effective.execution_at,'planned_target':planned,
                            'status':'OUTSIDE_2023_ACCOUNT_NO_EXECUTION_OR_OUTCOME_EVALUATED'})
    pd.DataFrame(pending).to_csv(output / 'pending_delayed_orders.csv',index=False)
    risk = risk_diagnostics(prices, predictions, base["budget"], base["cost_rate"])
    risk.to_csv(output / "risk_capacity_diagnostics.csv", index=False)
    selected_cases = select_cases(all_ledgers, prices, predictions, features, output)
    # 错失反弹的五日窗口也从连续账户提取，含基准共同时间，不在标签交易上叠加资金。
    rebound_rows = []
    for scope in ["full", "common"]:
        m = all_ledgers[(scope, "base", "M2_main")]
        b = all_ledgers[(scope, "base", "fixed_100")]
        for i in range(1, len(m)-5):
            asset = prices.set_index("as_of_session").total_return_index
            gain = asset[m.as_of_session.iloc[i+5]]/asset[m.as_of_session.iloc[i]]-1
            if gain <= 0:
                continue
            main_return = m.equity_end.iloc[i+5]/m.equity_before_trade.iloc[i]-1
            full_return = b.equity_end.iloc[i+5]/b.equity_before_trade.iloc[i]-1
            rebound_rows.append({"scope": scope, "entry": m.as_of_session.iloc[i], "end": m.as_of_session.iloc[i+5],
                                 "asset_return": gain, "main_return": main_return, "full100_return": full_return,
                                 "return_shortfall": full_return-main_return,
                                 "mean_exposure_after_entry": m.old_weight.iloc[i+1:i+6].mean(),
                                 "startup_unserved_entire_window": not m.ever_forecast_executed.iloc[i:i+6].any()})
    pd.DataFrame(rebound_rows).to_csv(output / "rebound_windows.csv", index=False)
    write_json(output / "input_verification.json", inputs)
    return {"output_dir": output.relative_to(root).as_posix(), "inputs": inputs, "base_common_first_execution": common,
            "delayed_common_first_execution": first, "requested_first_execution": request_start,
            "last_account_close": dates[-1], "accounts": len(summary), "same_targets_gross_accounts": len(gross_rows),
            "post_hoc_matching_accounts": len(matched), "scenarios": preset, "cases": selected_cases,
            "p4_forecast_sessions": int(len(available)/3), "p4_unserved_request_sessions": int((~predictions.forecast_available).sum()/3),
            "all_account_checks_passed": True, "p4_model_unchanged": True,
            "p4_accepted_checkpoint": json.loads((root / "runs/p5/experiment_record.json").read_text())["accepted_p4_checkpoint"],
            "locked_evaluation_performed": False, "p6_performed": False}
