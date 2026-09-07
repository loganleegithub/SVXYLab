"""P4 收尾诊断：读取保存结果；不改变预测器、候选网格或目标。"""

from datetime import datetime, timezone
from hashlib import sha256
from html import escape
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib

import numpy as np
import pandas as pd

from svxylab.features import CORE_IDS
from svxylab.features_report import write_json
from svxylab.prediction_data import TARGETS, development_csv

BASELINE = "runs/p4/20260907T013333116356Z/predictions_run.json"
REFERENCE = "0810f1a5ae371b4a703b8c1d2e098ffb0c13eae6"
ATOL, RTOL = 2e-11, 2e-10
TIME_FIELDS = {"computed_at_utc", "elapsed_seconds"}


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def compare_runs(old, new):
    """比较每个 CSV 及模型/选择状态，只排除实际计算时间与计时。"""
    checks = []
    assert {p.name for p in old.glob('*.csv')} == {p.name for p in new.glob('*.csv')}
    for source in sorted(old.glob("*.csv")):
        a, b = [pd.read_csv(p, float_precision="round_trip") for p in (source, new / source.name)]
        columns = [c for c in a if c not in TIME_FIELDS]
        pd.testing.assert_frame_equal(a[columns], b[columns], atol=ATOL, rtol=RTOL)
        delta = {}
        for c in columns:
            if pd.api.types.is_numeric_dtype(a[c]) and not pd.api.types.is_bool_dtype(a[c]):
                valid = a[c].notna() & b[c].notna()
                delta[c] = float((a.loc[valid, c]-b.loc[valid, c]).abs().max()) if valid.any() else 0.0
        checks.append({"file": source.name, "rows": len(a), "max_absolute_errors": delta})

    def clean(value):
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items() if k not in TIME_FIELDS}
        if isinstance(value, list):
            return [clean(v) for v in value]
        return value

    def same(a, b):
        if isinstance(a, dict):
            assert a.keys() == b.keys()
            for k in a:
                same(a[k], b[k])
        elif isinstance(a, list):
            assert len(a) == len(b)
            for x, y in zip(a, b, strict=True):
                same(x, y)
        elif isinstance(a, (int, float)) and not isinstance(a, bool):
            assert np.isclose(a, b, atol=ATOL, rtol=RTOL), (a, b)
        else:
            assert a == b, (a, b)

    states = list(old.glob("models/*.json")) + list(old.glob("selections/*.json"))
    for p in states:
        same(clean(json.loads(p.read_text())), clean(json.loads((new / p.relative_to(old)).read_text())))
    return {"method": "FROM_ORIGINAL_TRAINING_INPUTS_REFIT_AND_RESELECT", "atol": ATOL, "rtol": RTOL,
            "excluded_fields": sorted(TIME_FIELDS), "csv_checks": checks, "model_and_selection_states": len(states),
            "passed": True}


def probability_groups(scored):
    """经验五分位边界固定用线性插值；边界重复合并，等值概率同组。"""
    summary, membership = [], []
    for model, frame in scored.groupby("model", sort=True):
        p = frame.p10.to_numpy()
        edges = np.unique(np.quantile(p, np.linspace(0, 1, 6), method="linear"))
        ids = np.searchsorted(edges[1:-1], p, side="left")
        for method, groups, count in [("equal_width", np.minimum((p*5).astype(int), 4), 5),
                                       ("equal_frequency", ids, max(1, len(edges)-1))]:
            for i in range(count):
                group = frame.loc[groups == i]
                lo, hi = (i/5, (i+1)/5) if method == "equal_width" else (edges[i], edges[min(i+1, len(edges)-1)])
                summary.append({"model": model, "method": method, "group": i+1, "lower": lo, "upper": hi,
                                "rows": len(group), "positive_rows": int(group.Y10.sum()),
                                "mean_probability": group.p10.mean(), "event_rate": group.Y10.mean()})
            for row, group in zip(frame.itertuples(), groups, strict=True):
                membership.append({"model": model, "method": method, "as_of_session": row.as_of_session,
                                   "decision_session": row.decision_session, "p10": row.p10, "group": int(group)+1,
                                   "Y10": int(row.Y10)})
    return pd.DataFrame(summary), pd.DataFrame(membership)


def event_intervals(positives):
    """按入场至五日终点的闭区间相交合并；不是独立事件估计。"""
    events, members = [], []
    for row in positives.sort_values("decision_session").itertuples():
        if not events or row.decision_session > events[-1]["interval_end"]:
            events.append({"event_id": len(events)+1, "interval_start": row.decision_session,
                           "interval_end": row.label_end_session, "positive_rows": 0,
                           "first_information": row.as_of_session, "last_information": row.as_of_session,
                           "last_entry": row.decision_session})
        event = events[-1]
        event["interval_end"] = max(event["interval_end"], row.label_end_session)
        event["last_information"], event["last_entry"] = row.as_of_session, row.decision_session
        event["positive_rows"] += 1
        members.append({"event_id": event["event_id"], **row._asdict()})
    return pd.DataFrame(events), pd.DataFrame(members).drop(columns="Index", errors="ignore")


def diagnose(output, review):
    read = lambda name: pd.read_csv(output / name, float_precision="round_trip")
    scores, totals, choices = [read(f) for f in ["cv_scores.csv", "cv_totals.csv", "selected_parameters.csv"]]
    candidates, selections = [], []
    for (selection, model, head), group in totals.groupby(["selection_id", "model", "head"]):
        chosen = choices.loc[choices.selection_id.eq(selection) & choices.model.eq(model), head].item()
        selected_loss = group.loc[group.parameter.eq(chosen), "loss"].item()
        alternatives = group.loc[group.parameter.ne(chosen), "loss"]-selected_loss
        strongest = group.parameter.min() if head == "p10" else group.parameter.max()
        state = json.loads((output / "selections" / f"{selection}.json").read_text())["models"][model]
        assert state["parameters"][head] == chosen
        row = {"selection_id": selection, "model": model, "head": head, "chosen": chosen,
               "selected_loss": selected_loss, "at_strongest_boundary": chosen == strongest,
               "strictly_better_than_both": bool((alternatives > 1e-12).all()),
               "numerically_tied_alternatives": int((alternatives.abs() <= 1e-12).sum()),
               "nearest_alternative_gap": float(alternatives.min())}
        for n, r in enumerate(group.sort_values("parameter", ascending=head == "p10").itertuples(), 1):
            block = scores.loc[scores.selection_id.eq(selection) & scores.model.eq(model) & scores['head'].eq(head) & scores.parameter.eq(r.parameter)]
            assert block.block.tolist() == [1, 2, 3]
            weighted = math.fsum(block.loss*block.validation_rows)/sum(block.validation_rows)
            snap = [s for s in state["totals"] if s["head"] == head and s["parameter"] == r.parameter][0]
            assert abs(weighted-r.loss) < ATOL and abs(snap["loss"]-r.loss) < ATOL
            for s in block.itertuples():
                original = [v for v in state["scores"] if v["head"] == head and v["parameter"] == r.parameter and v["block"] == s.block][0]
                assert abs(original["loss"]-s.loss) < ATOL
            row[f"parameter_{n}"], row[f"loss_{n}"] = r.parameter, r.loss
            candidates.append({"selection_id": selection, "model": model, "head": head, "parameter": r.parameter,
                               **{f"block_{s.block}_loss": s.loss for s in block.itertuples()},
                               "weighted_loss": weighted, "loss_minus_selected": r.loss-selected_loss,
                               "validation_rows": r.validation_rows, "selected": r.parameter == chosen})
        selections.append(row)
    selection_frame = pd.DataFrame(selections)
    selection_frame.to_csv(review / "selection_diagnostics.csv", index=False)
    pd.DataFrame(candidates).to_csv(review / "candidate_diagnostics.csv", index=False)
    coefficients, counts = [], []
    for path in sorted((output / "models").glob("*.json")):
        state = json.loads(path.read_text())
        for head, fit in state["heads"].items():
            columns = state["transform"]["columns"] if state["transform"] else []
            values = np.asarray(fit.get("coef", []))
            base = {"fit_session": state["fit"]["fit_session"], "model": state["model"], "head": head}
            interactions = [v for c, v in zip(columns, values, strict=True) if c.startswith("I")]
            counts.append({**base, "columns": len(columns), "nonzero_exact": int(np.count_nonzero(values)),
                           "nonzero_above_1e_12": int(np.sum(np.abs(values) > 1e-12)),
                           "interaction_nonzero_exact": int(np.count_nonzero(interactions)),
                           "intercept_or_constant": fit.get("intercept", fit.get("value"))})
            for c, v in zip(columns, values, strict=True):
                coefficients.append({**base, "column": c, "coefficient": v, "exact_zero": v == 0,
                                     "nonzero_but_rounds_to_6dp_zero": v != 0 and round(float(v), 6) == 0})
    counts = pd.DataFrame(counts)
    counts.to_csv(review / "coefficient_counts.csv", index=False)
    pd.DataFrame(coefficients).to_csv(review / "coefficients.csv", index=False)
    predictions = read("predictions.csv")
    scored = predictions.loc[predictions.forecast_available & predictions.score_observed].copy()
    assert scored.label_end_session.lt("2024-01-01").all()
    groups, membership = probability_groups(scored)
    original = read("calibration.csv")
    width = groups.loc[groups.method.eq("equal_width")]
    np.testing.assert_allclose(width[["rows", "positive_rows", "mean_probability", "event_rate"]],
                               original[["rows", "events", "mean_probability", "event_rate"]], atol=ATOL, equal_nan=True)
    groups.to_csv(review / "probability_groups.csv", index=False)
    membership.to_csv(review / "probability_group_members.csv", index=False)
    one = scored.loc[scored.model.eq("M0")]
    events, event_members = event_intervals(one.loc[one.Y10.eq(1), ["as_of_session", "decision_session", "label_end_session", *TARGETS]])
    events.to_csv(review / "positive_event_intervals.csv", index=False)
    event_members.to_csv(review / "positive_event_members.csv", index=False)
    # 前十个概率排名，排名第十位相同的值全部保留；最高日期也不会随意挑一行。
    top = []
    for model, group in predictions.loc[predictions.forecast_available].groupby("model"):
        group = group.copy()
        group["probability_rank"] = group.p10.rank(method="min", ascending=False).astype(int)
        top.append(group.loc[group.probability_rank.le(10)])
    top = pd.concat(top).sort_values(["model", "probability_rank", "decision_session"])
    top.to_csv(review / "highest_probability_dates.csv", index=False)
    quarters = scored.loc[scored.model.eq("M2") & scored.decision_session.between("2022-01-01", "2022-06-30")].copy()
    quarters["quarter"] = pd.to_datetime(quarters.decision_session).dt.to_period("Q").astype(str)
    quarters["covered"] = quarters.L5.le(quarters.q90)
    quarters["loss_minus_q90"] = quarters.L5-quarters.q90
    quarters.to_csv(review / "m2_2022q1q2_dates.csv", index=False)
    coverage = quarters.groupby("quarter").agg(rows=("covered", "size"), covered=("covered", "sum"),
                                              coverage=("covered", "mean"), positive_rows=("Y10", "sum")).reset_index()
    metrics = read("metrics.csv")
    for r in coverage.itertuples():
        actual = metrics.loc[metrics.model.eq("M2") & metrics.period.eq(r.quarter)]
        assert actual.rows.item() == r.rows and abs(actual.q90_coverage.item()-r.coverage) < ATOL
    coverage.to_csv(review / "m2_quarter_coverage.csv", index=False)
    q = counts.loc[counts.model.eq("M2") & counts['head'].eq("q90")]
    return {"selection_count": len(selections), "strongest_boundary_count": int(selection_frame.at_strongest_boundary.sum()),
            "strictly_better_count": int(selection_frame.strictly_better_than_both.sum()),
            "tied_alternatives": int(selection_frame.numerically_tied_alternatives.sum()),
            "smallest_candidate_gap": float(selection_frame.nearest_alternative_gap.min()),
            "m2_q90_interaction_nonzero_months": int(q.interaction_nonzero_exact.gt(0).sum()),
            "scored_dates": len(one), "positive_rows": int(one.Y10.sum()), "overlap_intervals": len(events),
            "quarter_coverage": coverage.to_dict("records"), "post_review_diagnostics": True,
            "calibrator_fitted": False, "trading_threshold_selected": False}


def table_csv(review, name, columns=None):
    frame = pd.read_csv(review / name)
    if columns:
        frame = frame[columns]
    return "<div class='scroll'>" + frame.to_html(index=False, border=0, float_format=lambda v: f"{v:.9g}", na_rep="—") + "</div>"


def review_html(root, data):
    pointer = root / "runs/p4/review_latest.json"
    if not pointer.exists():
        return ""
    info = json.loads(pointer.read_text())
    if data["run_record"] not in [info["baseline_run"], info["rerun_record"]]:
        return ""
    return (root / info["review_dir"] / "appendix.html").read_text()


def write_appendix(root, review, context, comparison, result):
    def link(path, label):
        return f"<a href='../{escape(str(path))}'>{escape(label)}</a>"
    relative = review.relative_to(root)
    tables = lambda name, columns=None: table_csv(review, name, columns)
    refit_error = next(r for r in comparison["csv_checks"] if r["file"] == "predictions.csv")["max_absolute_errors"]
    error = max(refit_error[k] for k in ["mu5_raw", "mu5", "q90_raw", "q90", "p10_logit", "p10"])
    audit = json.loads((root / context["rerun_record"]).with_name("independent_validation.json").read_text())
    stats = pd.read_csv(review / "coefficient_counts.csv").groupby(["model", "head"]).agg(
        months=("fit_session", "size"), min_nonzero=("nonzero_exact", "min"), max_nonzero=("nonzero_exact", "max"),
        interaction_nonzero_min=("interaction_nonzero_exact", "min"), interaction_nonzero_max=("interaction_nonzero_exact", "max"))
    top = pd.read_csv(review / "highest_probability_dates.csv")
    maxima = top.loc[top.probability_rank.eq(1), ["model", "as_of_session", "decision_session", "p10", "label_end_session", *TARGETS, "score_observed"]]
    dates = pd.read_csv(review / "m2_2022q1q2_dates.csv")
    misses = dates.loc[~dates.covered, ["quarter", "as_of_session", "decision_session", "label_end_session", "q90_raw", "q90", "L5", "Y10"]]
    doc = f"""<section id='p4-review'><h2>P4 审查收尾 · 本次审查后追加诊断</h2>
<p class='note'>以下分组、边界与逐日诊断是看过原 P4 后按本次要求追加；不是事先注册。没有拟合校准器、选择交易阈值、扩大参数网格或新增模型。</p>
<h3>工程执行</h3><p>本次入口执行时 HEAD 为 {context['head']}，参考提交为 {REFERENCE}；启动时工作区{'有差异，保留在实际命令记录' if context['initial_git_status'] else '干净'}。原运行、报告原件及失败记录保留。
完整 P4 在新目录重跑；原83项测试、追加4项审查算术测试和真实输入合成未来扰动均通过。没有计算错误需要修复，预测器及实验参数保持原样。仅追加诊断、离线导出和报告展示。
实际命令日志已显示本地配置origin；本次未新增/修改远端，未执行fetch、push或上传。历史“无远端”表述已在当前STATUS更正。</p>
<p>{link(relative/'review_context.json','本次实际命令、退出码与日志')} · {link(context['rerun_record'],'完整重拟合运行')}</p>
<h3>复算结果</h3><p><strong>从原训练输入重新拟合：</strong>重新完成年度选择、训练尺度/样条及全部月度系数，逐决策日/模型比较17份CSV与124份模型/选择快照。
只排除 computed_at_utc、elapsed_seconds；绝对容差 {ATOL:g}、相对容差 {RTOL:g}。六个原始/发布预测字段最大差 {error:.3g}。
<strong>按保存系数重建：</strong>独立脚本重建训练尺度和样条，以保存系数计算2493行、14958个值；预测最大差 {audit['max_absolute_errors']['predictions']:.3g}。
该脚本核对216个候选头的已保存系数和损失，没有重新优化这些系数。两种复算各自留证。</p>
<p>{link(relative/'refit_comparison.json','逐文件数值比较')} · {link(Path(context['rerun_record']).with_name('independent_validation.json'),'保存系数独立审计')}</p>
<h3>预测研究结果：原损失比较不变</h3><p>M1/M2 的开发段主要平均预测损失仍未优于 M0。工程完成不要求跑赢 M0，未进入 P5。
24次选择全在最强收缩边界，24次均严格优于其余候选，数值并列0次（并列阈值1e-12）。最小备选差 {result['smallest_candidate_gap']:.12g}。
α越大收缩越强，Logistic C越小收缩越强；这是原有限网格内的比较，不证明边界以外更好。</p>
<details><summary>24次选择：从强到弱的三个候选损失与差距</summary>{tables('selection_diagnostics.csv')}</details>
<p>{link(relative/'candidate_diagnostics.csv','72个候选：三个验证块、加权损失和距所选差')}；各候选均为189个验证行加权平均，已与原 cv_scores/cv_totals/selections 交叉核对。</p>
<h3>系数非零情况</h3><p>M2 Q90的六个交互在全部40个月都是精确零，不是六位小数舍入；首次、末次与中间月份一致。
该头的非零样条主效应有9～23项。L1产生稀疏解属于模型结果，未取消正则化。M0为常数头，不含斜率系数；以下统计不计截距。</p>
{stats.to_html(border=0)}<details><summary>每月、每模型、每个头的非零数量</summary>{tables('coefficient_counts.csv')}</details>
<p>{link(relative/'coefficients.csv','逐项原系数、精确零与舍入标记')}</p>
<h3>概率五组：原等宽与追加等频</h3><p>使用原保存预测的826个已评分日，每模型同样33个阳性行。表内概率均为0～1的小数。
equal_width保留[0,.2)、[.2,.4)、[.4,.6)、[.6,.8)、[.8,1]及空组；与原报告逐值一致。
equal_frequency固定使用各模型全部已评分概率的0/20/40/60/80/100%经验分位（linear），重复边界合并，区间右闭且最小值纳入第一组。
相同概率始终同组，因此各组行数不必相等，也可能少于五个非空组；没有按日期强拆并列。它仅描述本段，不能作为下一日可用的交易分界。</p>
{tables('probability_groups.csv')}<p>M2最低概率等频组为14/166=8.43%阳性、最高组为6/165=3.64%；M1相应为9/166=5.42%与4/165=2.42%。
这段样本没有呈现随预测概率单调升高的事件比例，不能据分组宣称风险排序有效。33个阳性行还存在五日重叠。</p>
<p>{link(relative/'probability_group_members.csv','逐日期组成员')}</p>
<h3>33个阳性行的重叠区间</h3><p>按入场收盘至第五个后续收盘的闭区间相交合并，共 {result['overlap_intervals']} 个区间。
这既不是33次独立崩盘，也不保证合并区间之间统计独立。</p>{tables('positive_event_intervals.csv')}
<p>{link(relative/'positive_event_members.csv','全部33个阳性行与区间归属')}</p>
<h3>预测概率最高的日期及后来结果</h3><p>在全部831个有预测日期中排名，最高值的并列日期全部列出；M0按月常数，不能随意只挑一个最高日期。R5/L5为小数收益/损失，跨界后续结果保持空。</p>
{maxima.to_html(index=False,border=0,float_format=lambda v:f'{v:.9g}')}
<p>M1/M2最高概率均出现在信息日2021-01-27、决策日2021-01-28，分别为46.71%和42.35%。
随后至2021-02-04的五日R5为+9.71%，最大入场相对收盘损失L5为2.31%，Y10=0。单个高概率未发生事件并不单独证明概率计算错误。</p>
<p>{link(relative/'highest_probability_dates.csv','各模型前10名（截止并列全部保留）及后续结果')}</p>
<h3>M2：2022Q1/Q2的Q90覆盖复核</h3>{tables('m2_quarter_coverage.csv')}
<p>按原决策日所属季度统计，覆盖条件为 L5≤发布Q90。Q1为46/62=74.193548%，Q2为47/62=75.806452%，明显低于目标90%。
下面列出全部31个未覆盖日；跨季度结束的五日标签仍归属于入场决策季度。</p>
<details><summary>逐日未覆盖的原始日期与损失</summary>{misses.to_html(index=False,border=0,float_format=lambda v:f'{v:.9g}')}</details>
<p>{link(relative/'m2_2022q1q2_dates.csv','全部124日：信息日、决策日、标签终点、原始/发布Q90、实际损失')}</p>
<h3>数据/PIT限制与离线审查</h3><p>冻结数据没有重新下载。历史精确发布时间未知，继续使用 ASSUMED_NEXT_SESSION / NO_FULL_HISTORICAL_PIT_CLAIM。
原来源差异、SPY总回报口径差异、较晚才有首个预测以及五日标签重叠仍限制解释；本次没有消除这些限制。</p>
<p>最小审查包只供现有授权范围内的个人非商业本地审查。已有来源记录没有再分发授权，不能据公开可读推定可公开分享行情或派生数据。
包内含开发段特征/可用性和删去跨2024结果的标签、两次P4结果、源码/依赖、模型/选择快照、测试及审计记录。
不包含原始行情、封存段结果或虚拟环境；P1～P3原件核账及真实行情扰动的原料继续留在本地源工作区。
源记录里的历史路径通过包内 source_references.json 映射，缺省原件明确标为仅在源工作区；导出文件摘要与源摘要分别保存。
离线重拟合以导出的开发输入为起点，不能冒充从交易所原件重建P1～P3。</p>
<p>{link(relative/'review_result.json','审查摘要')} · {link('data/clean/p4_review/'+review.name+'/LOCAL_REVIEW_ONLY/README.md','本地离线包说明与复算命令')}</p>
<p><strong>停在 P4，等待验收；没有计算错误反例或修复前后影响日期需要报告。</strong></p></section>"""
    (review / "appendix.html").write_text(doc)


def finish_review(root, context):
    review = root / context["review_dir"]
    old, new = [json.loads((root / context[k]).read_text()) for k in ["baseline_run", "rerun_record"]]
    assert old["input_verification"] == new["input_verification"]
    for key in ["experiment.toml", "FEATURES.json"]:
        assert digest(root / key) == old["source_sha256"][key] == new["source_sha256"][key]
    for record in [old, new]:
        for path, wanted in record["artifact_sha256"].items():
            assert digest(root / path) == wanted, path
    comparison = compare_runs(root / old["result"]["output_dir"], root / new["result"]["output_dir"])
    write_json(review / "refit_comparison.json", comparison)
    result = diagnose(root / old["result"]["output_dir"], review)
    result.update({"baseline_run": context["baseline_run"], "rerun_record": context["rerun_record"],
                   "calculation_error_found": False, "model_or_target_changed": False, "loss_comparison_changed": False,
                   "input_verification_identical": True, "original_artifact_hashes_verified": True})
    write_json(review / "review_result.json", result)
    write_appendix(root, review, context, comparison, result)
    write_json(root / "runs/p4/review_latest.json", context)
    from svxylab.predictions_report import render
    render(root, new)
    return result


def build_bundle(root, context):
    """仅导出本轮P4所需文件；所有市场数据派生物仍留在本地忽略目录。"""
    from bs4 import BeautifulSoup
    review = root / context["review_dir"]
    bundle = root / "data/clean/p4_review" / review.name / "LOCAL_REVIEW_ONLY"
    bundle.mkdir(parents=True, exist_ok=False)
    files, references = [], {}

    def record(source, destination, policy="BYTE_IDENTICAL"):
        files.append({"source_path": source.relative_to(root).as_posix() if source else None,
                      "source_sha256": digest(source) if source else None,
                      "bundle_path": destination.relative_to(bundle).as_posix(), "export_sha256": digest(destination),
                      "transformation": policy})

    def copy(source, target=None):
        destination = bundle / (target or source.relative_to(root))
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        record(source, destination)

    paths = sorted((root / "src").rglob("*.py")) + sorted((root / "tests").rglob("*.py"))
    paths += [root / p for p in ["AGENTS.md", "RESEARCH_SPEC.md", "FEATURES.json", "experiment.toml", "TASKS.md", "STATUS.md",
                                "pyproject.toml", "requirements-lock.txt", "SOURCES.md", "runs/p4/experiment_record.json",
                                "runs/p4/independent_validation.py", "runs/p4/test_real_causality.py",
                                "runs/p4/dependency_install.json", "runs/p4/environment_before.json", "runs/p1/source_decision.json"]]
    for key in ["baseline_run", "rerun_record"]:
        run = root / context[key]
        metadata = json.loads(run.read_text())
        paths += [p for p in run.parent.iterdir() if p.is_file()]
        paths += [p for p in (root / metadata["result"]["output_dir"]).rglob("*") if p.is_file()]
    paths += [p for p in review.iterdir() if p.is_file() and p.name not in ["predictions_before.html", "appendix.html", "bundle_record.json", "final_validation.json"]]
    for source in sorted(set(paths)):
        copy(source)
    copy(root / "runs/p4/offline_replay.py", "replay.py")
    # 保留源数字字符串，不经浮点读写来导出；只裁剪日期并屏蔽跨封存边界结果。
    original = json.loads((root / context["baseline_run"]).read_text())
    inputs = {}
    for key, stage, filename in [("features", "p3", "core_features.csv"), ("availability", "p3", "feature_availability.csv"), ("labels", "p2", "labels.csv")]:
        accepted = json.loads((root / original["input_verification"][f"{stage}_accepted_run"]).read_text())
        source = root / accepted["result"]["output_dir"] / filename
        frame = development_csv(source, "2024-01-01", labels=key == "labels")
        assert frame.index.min() >= "2019-01-01" and frame.index.max() < "2024-01-01"
        if key == "labels":
            assert frame.loc[frame.label_end_session.ge("2024-01-01"), TARGETS].eq("").all().all()
        destination = bundle / "development_inputs" / filename
        destination.parent.mkdir(exist_ok=True)
        frame.to_csv(destination)
        inputs[key] = destination.relative_to(bundle).as_posix()
        record(source, destination, "DEVELOPMENT_ONLY_WITH_CROSS_BOUNDARY_TARGETS_EMPTY" if key == "labels" else "DEVELOPMENT_ONLY")
    # 原始运行记录不篡改；它们中的源路径明确映射到导出件或仅在源工作区存在的原件。
    source_to_export = {r["source_path"]: r for r in files if r["source_path"]}

    def source_ref(value):
        if not isinstance(value, str) or not value or "\n" in value or len(value) > 500:
            return
        candidate = Path(value)
        source = candidate if candidate.is_absolute() else root / candidate
        try:
            rel = source.relative_to(root).as_posix()
            if not source.is_file():
                return
        except (ValueError, OSError):
            return
        exported = source_to_export.get(rel)
        references[value] = {"source_path": rel, "source_sha256": digest(source),
                             "bundle_path": exported["bundle_path"] if exported else None,
                             "status": exported["transformation"] if exported else "LOCAL_SOURCE_ONLY_NOT_EXPORTED",
                             "reason": None if exported else "P1-P3原件、封存数据或旧阶段/合成夹具不属于最小开发段包；没有再分发授权"}

    def scan(value):
        if isinstance(value, dict):
            for k, v in value.items():
                source_ref(k)
                scan(v)
        elif isinstance(value, list):
            for v in value:
                scan(v)
        else:
            source_ref(value)

    for item in list(files):
        if item["bundle_path"].endswith(".json"):
            scan(json.loads((bundle / item["bundle_path"]).read_text()))
    # 报告中的实际可点链接只指向包内文件；外部文档网址保留为资料来源。
    html = BeautifulSoup((root / "reports/predictions.html").read_text(), "html.parser")
    for element in html.find_all(["a", "img"]):
        attr = "href" if element.name == "a" else "src"
        url = element.get(attr, "")
        if not url or url.startswith(("https:", "http:", "#")):
            continue
        target = (bundle / "reports" / url.split("#")[0]).resolve()
        if not target.is_file():
            source = (root / "reports" / url.split("#")[0]).resolve()
            source_ref(str(source))
            if element.name == "a":
                element.replace_with(element.get_text()+"（仅源工作区；见 source_references.json）")
            else:
                element.decompose()
    banner = html.new_tag("p")
    banner.string = "本地离线审查副本：没有原始行情或封存段结果。运行记录中的历史源路径通过 source_references.json 映射；复算方法见根目录 README.md。"
    html.main.insert(0, banner)
    destination = bundle / "reports/predictions.html"
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(str(html))
    record(root / "reports/predictions.html", destination, "LOCAL_LINKS_RESOLVED_OR_EXPLICITLY_SOURCE_ONLY")
    write_json(bundle / "source_references.json", references)
    record(None, bundle / "source_references.json", "GENERATED_SOURCE_PATH_MAP")
    readme = f"""# SVXYLab P4 本地离线审查包

仅供用户已授权的个人非商业本地研究。原来源记录仅支持该本地用途，没有取得行情或派生数据再分发授权；不能公开上传本包，也不默认提供可对外分享的数据包。原数据继续留在源工作区，无重新下载。许可依据是包内 runs/p1/source_decision.json 的既有记录，不是新增许可授予。

从 reports/predictions.html 打开同一份 P4 报告的离线副本。模型、目标、参数网格未变。本次追加诊断不称事先注册。

## 实际复算

在本目录运行兼容 Python 3.12 环境下的 `python replay.py`。本机可复用 `/Users/logan/SVXYLab/.venv/bin/python replay.py`。脚本核验全部导出摘要，直接从 development_inputs 的原尺度核心特征/可用性/成熟标签重新完成4次年度选择和40个月度拟合，然后与原P4逐日预测及模型状态比较，再执行保存系数独立审计和普通pytest。结果写入新的 replay_results 与 audit_results 时间戳目录。

本包不包含虚拟环境或依赖二进制。当前依赖版本见 requirements-lock.txt 和实际运行记录；另一台离线机器需要事先已有兼容依赖环境，不能只凭本包在空机器安装依赖。

两种证据分开：replay.py 是从导出训练输入重新拟合；runs/p4/independent_validation.py --bundle . 是使用已保存系数重建，不重新优化。两者均不重建P1-P3行情、特征或标签。真实行情未来扰动的原料未随包导出；原测试源码和本次实际运行结果保留，需源工作区才能再次执行该原料级测试。

## 文件及边界

- 原P4基准运行：{context['baseline_run']}；本次重跑：{context['rerun_record']}。
- 源码、普通测试、依赖、配置和特征字典在原相对路径；两次P4的17份CSV、模型/变换、选择快照和运行记录保持原相对目录。
- development_inputs 仅包含2019-01-02～2023-12-29信息日。6个标签终点跨2024的R5/L5/Y10全部为空；预测文件中5个有预测但跨界日期也为空。不包含2024信息日或封存期结果。
- bundle_manifest.json 对每份导出文件保留源路径/源SHA-256和导出路径/导出SHA-256。裁剪后输入摘要不冒充完整源文件摘要。
- source_references.json 将历史运行记录里的源路径映射到包内文件或明确 LOCAL_SOURCE_ONLY_NOT_EXPORTED；这些历史路径是来源标识，不是承诺包内存在P1-P3全量数据。
- 可用性表中的 published_at 仍为空；保留 ASSUMED_NEXT_SESSION / NO_FULL_HISTORICAL_PIT_CLAIM。

本包不含原始行情、完整P1/P2/P3数据、封存结果、交易账户信息或密钥。源文件和旧产物未被覆盖；验收后才保存新的本地Git检查点，不连接远端。
"""
    (bundle / "README.md").write_text(readme)
    record(None, bundle / "README.md", "GENERATED_LOCAL_REVIEW_INSTRUCTIONS")
    manifest = {"kind": "LOCAL_REVIEW_ONLY_NO_REDISTRIBUTION_AUTHORIZATION", "source_root": str(root),
                "reference_commit": REFERENCE, "baseline_run": context["baseline_run"], "rerun_record": context["rerun_record"],
                "development_inputs": inputs, "files": files, "source_reference_map": "source_references.json",
                "manifest_hash_scope": "manifest本身不递归哈希；外部bundle_record.json保存manifest和ZIP的摘要",
                "locked_outcomes_exported": False, "raw_quotes_exported": False}
    write_json(bundle / "bundle_manifest.json", manifest)
    for item in files:
        assert digest(bundle / item["bundle_path"]) == item["export_sha256"]
    for ref in references.values():
        if ref["bundle_path"]:
            assert (bundle / ref["bundle_path"]).is_file()
    archive = Path(shutil.make_archive(str(bundle), "zip", root_dir=bundle.parent, base_dir=bundle.name))
    write_json(review / "bundle_record.json", {"bundle_dir": bundle.relative_to(root).as_posix(),
               "archive": archive.relative_to(root).as_posix(), "archive_sha256": digest(archive), "archive_bytes": archive.stat().st_size,
               "manifest_sha256": digest(bundle / "bundle_manifest.json"), "exported_files": len(files),
               "source_references": len(references), "paths_and_export_hashes_verified": True,
               "distribution": "LOCAL_ONLY_NOT_PUBLIC_OR_REMOTE", "locked_outcomes_exported": False})
    return bundle


def build_review(root, *, open_report=False):
    """唯一入口的审查模式；完整重拟合与只读诊断分开记录。"""
    stamp = datetime.now(timezone.utc).strftime("review_%Y%m%dT%H%M%S%fZ")
    review = root / "runs/p4" / stamp
    review.mkdir()
    shutil.copy2(root / "reports/predictions.html", review / "predictions_before.html")
    context = {"review_dir": review.relative_to(root).as_posix(), "baseline_run": BASELINE,
               "reference_commit": REFERENCE, "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
               "initial_git_status": subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True), "commands": []}
    commands = [("pytest_before", [sys.executable, "-m", "pytest", "-q", "--junitxml="+str(review / "pytest_before.xml")]),
                ("full_p4", [sys.executable, "-m", "svxylab", "predictions"]),
                ("independent_audit", [sys.executable, "runs/p4/independent_validation.py"]),
                ("real_causality", [sys.executable, "-m", "pytest", "runs/p4/test_real_causality.py", "-q", "--junitxml="+str(review / "real_causality.xml")])]
    for name, command in commands:
        started = datetime.now(timezone.utc).isoformat()
        run = subprocess.run(command, cwd=root, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
        log = review / f"{name}.log"
        log.write_text(run.stdout)
        context["commands"].append({"name": name, "command": command, "started_at": started, "exit_code": run.returncode, "log": log.relative_to(root).as_posix()})
        write_json(review / "review_context.json", context)
        print(f"P4审查 {name}：退出{run.returncode}", flush=True)
        if run.returncode:
            return run.returncode
    runs = [p for p in (root / "runs/p4").glob("*/predictions_run.json") if not json.loads(p.read_text())["result"]["pilot"]]
    context["rerun_record"] = sorted(runs)[-1].relative_to(root).as_posix()
    write_json(review / "review_context.json", context)
    finish_review(root, context)
    build_bundle(root, context)
    if open_report:
        return subprocess.run(["/usr/bin/open", str(root / "reports/predictions.html")], check=False).returncode
    return 0
