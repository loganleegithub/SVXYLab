"""P4 本地中文预测报告；分别呈现工程、预测结果和实际覆盖。"""

from datetime import datetime, timezone
from hashlib import sha256
from html import escape
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import sys
import tomllib
import traceback
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from svxylab.data_report import table
from svxylab.environment import run_command
from svxylab.features_report import write_json
from svxylab.models import MODELS
from svxylab.prediction_data import load_development_data
from svxylab.predictions import run_predictions

NAMES = {"M0": "M0 历史常数", "M1": "M1 联合线性", "M2": "M2 样条与交互"}


def percent(value, digits=2):
    return f"{value:.{digits}%}" if pd.notna(value) else "—"


def plot_predictions(root, output):
    os.environ.setdefault("MPLCONFIGDIR",str(root / ".cache/matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font = FontProperties(fname="/System/Library/Fonts/STHeiti Light.ttc")
    predictions = pd.read_csv(output / "predictions.csv")
    valid = predictions.loc[predictions.forecast_available]
    colors = {"M0":"#737d87", "M1":"#2678a5", "M2":"#ba583d"}
    fig, axes = plt.subplots(3,1,figsize=(12,9),sharex=True,layout="constrained")
    for ax,(field,label,actual) in zip(axes,[("mu5","五日均值预测 μ5", "R5"),("q90","损失第90分位 Q90", "L5"),("p10","损失达10%的概率 P10", None)],strict=True):
        if actual:
            observed = valid.loc[valid.model.eq("M0") & valid.score_observed]
            ax.plot(pd.to_datetime(observed.decision_session),observed[actual]*100,color="#ced4d9",linewidth=.6,label="随后成熟的真实结果")
        for model in MODELS:
            group = valid.loc[valid.model.eq(model)]
            ax.plot(pd.to_datetime(group.decision_session),group[field]*100,label=NAMES[model],color=colors[model],linewidth=1)
        ax.set_ylabel(label+"（%）",fontproperties=font)
        ax.grid(alpha=.25,linewidth=.5)
        ax.spines[["top","right"]].set_visible(False)
    axes[0].set_title("按实际决策日排列的历史重放预测 · 参数按月更新",fontproperties=font)
    axes[0].legend(prop=font,ncols=2,fontsize=8)
    axes[-1].set_xlim(pd.Timestamp(valid.decision_session.min()),pd.Timestamp(valid.decision_session.max()))
    fig.savefig(output / "predictions.png",dpi=145)
    fig.savefig(output / "predictions.svg",metadata={"Date":None})
    plt.close(fig)
    calibration = pd.read_csv(output / "calibration.csv")
    fig, axes = plt.subplots(1,3,figsize=(12,4),sharex=True,sharey=True,layout="constrained")
    for ax,model in zip(axes,MODELS,strict=True):
        group = calibration.loc[calibration.model.eq(model) & calibration.rows.gt(0)]
        ax.plot([0,1],[0,1],"--",color="#999",linewidth=.8)
        ax.plot(group.mean_probability,group.event_rate,"o-",color=colors[model])
        for row in group.itertuples():
            ax.annotate(f"n={row.rows}",(row.mean_probability,row.event_rate),xytext=(4,5),textcoords="offset points",fontsize=8)
        ax.set_title(NAMES[model],fontproperties=font)
        ax.set_xlabel("组内平均预测概率",fontproperties=font)
        ax.grid(alpha=.25)
        ax.set_xlim(-.02,1.02);ax.set_ylim(-.02,1.02)
    axes[0].set_ylabel("随后成熟的事件比例",fontproperties=font)
    fig.suptitle("五个固定概率区间的校准诊断 · 空组保留在表中",fontproperties=font)
    fig.savefig(output / "calibration.png",dpi=145)
    plt.close(fig)


def metric_table(frame):
    return table(["时期", "模型", "行数 / 事件", "MSE", "MSE相对M0", "对数损失", "对数损失相对M0", "Brier", "Pinball", "Pinball相对M0", "Q90覆盖"],
                 [[r.period,NAMES[r.model],f"{r.rows} / {r.events}",f"{r.mse:.7f}",percent(r.mse_relative_change_M0),
                   f"{r.log_loss:.6f}",percent(r.log_loss_relative_change_M0),f"{r.brier:.6f}",f"{r.pinball:.6f}",
                   percent(r.pinball_relative_change_M0),percent(r.q90_coverage)] for r in frame.itertuples()])


def render(root: Path, data: dict):
    result = data["result"]
    output = root / result["output_dir"]
    predictions = pd.read_csv(output / "predictions.csv")
    metrics = pd.read_csv(output / "metrics.csv")
    fits = pd.read_csv(output / "fits.csv")
    params = pd.read_csv(output / "selected_parameters.csv")
    dictionary = json.loads((root / "FEATURES.json").read_text())
    names = {r["id"]:r["name_zh"] for r in dictionary["features"]}
    calibration = table(["模型", "固定概率组", "行数", "事件数", "平均预测概率", "实际事件比例"],
                        [[NAMES[r['model']],f"[{r['lower']:.0%}, {r['upper']:.0%}"+("]" if r['bin']==5 else ")"),r['rows'],r['events'],
                          percent(r['mean_probability']),percent(r['event_rate'])] for r in result['evaluation']['calibration']])
    projections = table(["模型", "预测日数", "已评分", "未评分", "均值下界投影", "Q90投影", "原始Q90最小 / 最大"],
                        [[NAMES[r['model']],r['forecast_rows'],r['scored_rows'],r['unscored_rows'],r['mu_projected'],r['q_projected'],
                          f"{r['min_raw_q90']:.4f} / {r['max_raw_q90']:.4f}"] for r in result['evaluation']['projections']])
    support = table(["模型", "年份", "预测日数", "至少一项超出训练范围", "单日最多超出列数", "最大越界距离（训练标准差）"],
                    [[NAMES[r['model']],r['year'],r['rows'],r['outside_rows'],r['max_outside_features'],f"{r['max_outside_distance_training_sd']:.3f}"] for r in result['evaluation']['support']])
    disagreement = table(["输出", "配对日数", "M2−M1 平均差", "平均绝对差", "最大绝对差"],
                         [[r['output'],r['rows'],percent(r['mean_M2_minus_M1']),percent(r['mean_absolute_difference']),percent(r['max_absolute_difference'])] for r in result['evaluation']['model_disagreement']])
    fit_rows = [[r.fit_session,NAMES[r.model],r.training_first,r.training_last,r.training_rows,r.positive_count,r.event_clusters,
                 r.design_columns,r.selection_id if pd.notna(r.selection_id) else "—",f"{r.elapsed_seconds:.3f}"] for r in fits.itertuples()]
    fit_table = table(["拟合决策日", "模型", "训练首日", "训练末日", "行数", "阳性行", "重叠事件簇", "设计列数", "选参日期", "秒"],fit_rows)
    param_table = table(["年度选择日期", "模型", "Ridge α", "Quantile α", "Logistic C"],
                        [[r.selection_id,NAMES[r.model],r.mu5,r.q90,r.p10] for r in params.itertuples()])
    conditions = pd.read_csv(output / "interaction_conditions.csv")
    periods = [conditions.fit_session.min(), conditions.fit_session.max()]
    rows = []
    for day in sorted(set(periods)):
        for interaction in dictionary["interactions"]:
            subset = conditions.loc[conditions.fit_session.eq(day) & conditions.interaction.eq(interaction["id"])]
            slopes = []
            for head in ["mu5","q90","p10"]:
                for q in [.25,.75]:
                    values = subset.loc[subset['head'].eq(head) & subset.partner_training_quantile.eq(q),"interaction_slope_per_training_sd"]
                    slopes.append(f"{values.iloc[0]:+.6f}")
            a,b = interaction["features"]
            rows.append([day,interaction["id"],names[a]+" × "+names[b],*slopes])
    interaction_table = table(["拟合日", "交互", "本变量 × 伙伴", "μ斜率 Q25", "μ斜率 Q75", "Q90原值斜率 Q25", "Q90原值斜率 Q75", "log-odds斜率 Q25", "log-odds斜率 Q75"],rows)
    local = pd.read_csv(output / "local_effects.csv")
    last_id = local.fit_id.max()
    local_rows = local.loc[local.fit_id.eq(last_id) & local.feature.isin(["A01","B01","F03"]) & local['head'].eq("p10")]
    local_table = table(["信息日", "变量", "当前训练 z", "总局部斜率", "交互部分", "样条部分"],
                        [[r.as_of_session,names[r.feature],f"{r.training_z:.3f}",f"{r.local_total_slope_per_training_sd:+.5f}",
                          f"{r.interaction_component:+.5f}",f"{r.spline_component:+.5f}"] for r in local_rows.itertuples()])
    risks = table(["模型", "事后Q90组", "行数 / 事件", "平均预测Q90", "实际L5的90分位", "实际事件比例", "组内Q90覆盖"],
                  [[NAMES[r['model']],r['group'],f"{r['rows']} / {r['events']}",percent(r['mean_predicted_q90']),percent(r['observed_loss_quantile90']),
                    percent(r['observed_event_rate']),percent(r['coverage'])] for r in result['evaluation']['risk_groups']])
    first = result["first_fit"]
    available = predictions.loc[predictions.forecast_available]
    shown_dates = sorted(set([available.decision_session.min(),available.decision_session.max()]))
    samples = available.loc[available.decision_session.isin(shown_dates)]
    sample_table = table(["信息日", "决策日", "模型", "μ5", "Q90", "P10", "标签终点", "后来结果", "实际计算UTC"],
                         [[r.as_of_session,r.decision_session,NAMES[r.model],percent(r.mu5),percent(r.q90),percent(r.p10),r.label_end_session,
                           f"R5={percent(r.R5)}, L5={percent(r.L5)}, Y10={int(r.Y10)}" if r.score_observed else "跨开发边界，未评分",r.computed_at_utc]
                          for r in samples.itertuples()])
    commands = "".join(f"<details><summary>退出码 {c['exit_code']} · <code>{escape(c['command'])}</code></summary><pre>{escape(c['output'] or '（无输出）')}</pre></details>" for c in data["commands"])
    test_table = table(["普通 pytest", "实际结果"],[[r["name"],r["result"]] for r in data["test_cases"]])
    links = " · ".join(f"<a href='../{escape(p)}'>{escape(Path(p).name)}</a>" for p in data["artifact_sha256"] if Path(p).parent == output.relative_to(root) and p.endswith(".csv"))
    audit_file = (root / data["run_record"]).with_name("independent_validation.json")
    audit_html = ""
    if audit_file.exists():
        audit = json.loads(audit_file.read_text())
        audit_html = (f"<p>本次已保存模型与真实数据的独立核对：{'通过' if audit['passed'] else '未通过'}。"
                      f"按保存系数重建 {audit['forecast_rows_replayed']} 行模型预测，核对 {audit['inner_head_fits_checked']} 个内层候选头的已保存预测损失及 {audit['metric_rows_checked']} 行时期指标；没有重新拟合系数；"
                      f"预测最大绝对误差 {audit['max_absolute_errors']['predictions']:.3g}。"
                      f"<a href='../{audit_file.relative_to(root).as_posix()}'>数值与时钟核对结果</a>。</p>")
    future_html = ""
    evidence_path = root / "runs/p4/reproducibility.json"
    if evidence_path.exists():
        evidence = json.loads(evidence_path.read_text())
        if evidence.get("final_run") == data["run_record"]:
            test_path = root / evidence["future_perturbation_record"]
            test = json.loads(test_path.read_text())
            future_html = (f"<p>真实输入上的合成未来扰动 pytest 已通过：人为改变 {test['quote_mutation_begins_after_information_session']} 之后的ETF、VX和隐含指数，"
                           f"重新计算特征、标签和滚动模型，较早 {test['earlier_actual_forecast_rows']} 行模型预测逐值不变，较晚预测发生变化。"
                           f"测试针对首次完整重放；最终复算的预测数值与该运行相同。原始数据摘要不变，合成夹具与真实结果分开保存。"
                           f"<a href='../{evidence['future_perturbation_record']}'>实际扰动记录</a> · <a href='../runs/p4/reproducibility.json'>最终复算比较</a>。</p>")
    pilot_html = ""
    if (root / "runs/p4/pilot_summary.json").exists():
        pilot_summary = json.loads((root / "runs/p4/pilot_summary.json").read_text())
        pilot_html = (f"<p>先完成首个合资格月的试运行：{pilot_summary['forecast_sessions']} 个预测日，计算 {pilot_summary['elapsed_seconds']:.3f} 秒（不含绘图与测试）；"
                      f"随后执行完整开发段。<a href='../runs/p4/pilot_summary.json'>试运行摘要</a> · <a href='../runs/p4/pilot_invocation.json'>实际命令与退出结果</a>。</p>")
    headline = "试运行完成，完整 P4 尚待运行" if result["pilot"] else ("工程完成，等待 P4 验收" if data["engineering_complete"] else "工程未完成，存在实际失败")
    overall = {r["model"]:r for r in result["evaluation"]["overall"]}
    comparison = []
    for model in ["M1","M2"]:
        m = overall[model]
        comparison.append(f"{model} 相对 M0：均值 MSE {m['mse_relative_change_M0']:+.1%}、概率对数损失 {m['log_loss_relative_change_M0']:+.1%}、分位 Pinball {m['pinball_relative_change_M0']:+.1%}。")
    docs = " · ".join(f"<a href='{url}'>{label}</a>" for label,url in [
        ("SplineTransformer","https://scikit-learn.org/1.9/modules/generated/sklearn.preprocessing.SplineTransformer.html"),
        ("Ridge","https://scikit-learn.org/1.9/modules/generated/sklearn.linear_model.Ridge.html"),
        ("LogisticRegression","https://scikit-learn.org/1.9/modules/generated/sklearn.linear_model.LogisticRegression.html"),
        ("QuantileRegressor","https://scikit-learn.org/1.9/modules/generated/sklearn.linear_model.QuantileRegressor.html")])
    from svxylab.prediction_review import review_html
    review_appendix = review_html(root, data)
    doc = f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>
<title>SVXYLab · P4 联合预测</title><style>
body{{margin:0;font:16px/1.7 -apple-system,BlinkMacSystemFont,'PingFang SC',sans-serif;color:#22313d;background:#fff}}
main{{max-width:1320px;margin:auto;padding:28px 24px 70px}}h1{{font-size:28px}}h2{{font-size:22px;margin-top:34px}}h3{{font-size:18px}}
a{{color:#165b8a}}.status,.note{{background:#f3f6f8;padding:12px 18px;border-left:4px solid #607d8b}}.scroll{{overflow:auto}}
table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{border:1px solid #d6dfe6;padding:8px;text-align:left;vertical-align:top}}th{{background:#f3f6f8}}
img{{max-width:100%;height:auto}}pre,code{{font:13px/1.6 ui-monospace,monospace}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}details{{border-bottom:1px solid #ddd;padding:9px 0}}summary{{cursor:pointer;overflow-wrap:anywhere}}
</style></head><body><main><h1>SVXYLab · P4 联合预测</h1><p>{escape(data['generated_at'])} · RESEARCH_ONLY · 已验收 P3 检查点 541178b</p>
<div class='status'><p><strong>工程：</strong>{headline}。普通 pytest {data['tests_passed']} 项通过、{data['tests_failed']} 项失败/错误；本次重放耗时 {result['elapsed_seconds']:.2f} 秒。</p>
<p><strong>预测研究结果：</strong>{' '.join(comparison)} 负数表示损失较低。结果按预定规则保留，没有按好坏改目标、删模型或扩大网格；本阶段没有经济仓位或收益优势结论。</p>
<p><strong>实际覆盖：</strong>本次处理 {result['requested_decision_first']} 至 {result['processed_decision_last']} 的 {result['request_sessions']} 个决策日，{result['forecast_sessions']} 日有三模型预测，{result['no_forecast_sessions']} 日没有预测。封存段未参与拟合或评分。</p></div>
{review_appendix}
<h2>从何时真的能够预测</h2><p>首个可预测信息日 <strong>{result['first_prediction_information_session']}</strong>，决策日 <strong>{result['first_prediction_decision_session']}</strong>；截止 {first['simulation_fit_at']}。
训练来自 {first['training_first']} 至 {first['training_last']}，{first['training_rows']} 行、{first['positive_count']} 行阳性、{first['event_clusters']} 个合并的重叠事件簇。
2019 年起累积数据，须等到 400 行完整成熟样本和全部内层条件满足；此前无预测日期与原因保存在 prediction_service.csv，没有为疫情初期补入旧制度数据或降低门槛。</p>
{pilot_html}
<p>请求区间按实际决策/执行交易日计。信息 t → t+1 收盘前 60 分钟决策 → t+1 收盘执行；标签是 t+1 到 t+6 收盘。
仅用 label_end_session&lt;2024-01-01 的后来结果评分；跨界尾部仍保留预测，结果为空。computed_at_utc 是本次真实计算时间，simulation_decision_at 是历史重放时钟，不声称预测当年已实际生成。</p>{sample_table}
<h2>三个输出与预定模型</h2><p>μ5 为未来五日平均总回报估计；Q90 为这五日相对入场点的每日收盘最差损失的第90分位估计；P10 为同一损失达到10%的概率。
Q90 不是最坏可能损失或盘中回撤，P10 不是历史分位。三个头分别训练和评分，不把它们混成一个总分。</p>
<p>M0 用完全相同成熟样本的历史均值、线性插值90分位和 Jeffreys 事件率 (阳性行+0.5)/(n+1)。M1 为19项标准化核心输入的联合线性模型。
M2 为57个二次样条基加6个指定交互，共63列；三个分位节点、线性外推、无冗余bias。标准化均值/尺度、样条节点、交互与输出列尺度都只在对应训练部分拟合。
均值使用 Ridge，概率使用 L2 LogisticRegression，分位使用成熟库的 L1 QuantileRegressor；没有类权重、SMOTE或额外校准搜索。</p>
<img src='../{result['output_dir']}/predictions.png' alt='三个模型在有效预测日期的五日均值、损失分位和事件概率'><p>灰线是后来成熟的真实结果，只供回看比较；它没有在预测时进入当期拟合。
<a href='../{result['output_dir']}/predictions.svg'>矢量图</a>。图中没有账户净值或仓位回测。</p>
<h2>同日期的预测损失与连续时期</h2><p>MSE、对数损失、Brier、Pinball 都是越低越好。相对变化=(本模型损失/M0损失)−1，按同日配对；Q90覆盖应结合目标90%与样本数看。
五日标签相互重叠，事件行数不是独立崩盘次数；时间块不确定性分析留在 P6，当前不作显著性或因果宣称。</p>
{metric_table(metrics.loc[metrics.period_type.eq('all')])}<h3>按年</h3>{metric_table(metrics.loc[metrics.period_type.eq('year')])}
<details><summary>按连续季度的完整结果</summary>{metric_table(metrics.loc[metrics.period_type.eq('quarter')])}</details>
<h2>五组概率校准与损失分位</h2><p>固定五个概率区间，不为结果更好看而改分组。空组如实保留；一组的平均预测概率与实际事件比例不同，表示这组存在校准偏差。稀疏组不能证明概率精确。</p>
<img src='../{result['output_dir']}/calibration.png' alt='五个固定概率区间的预测概率与真实事件比例'>{calibration}
<h3>Q90 事后风险分组</h3><p>按各模型实际预测Q90经验五分位描述结果，仅用于评分展示；重复边界会减少分组数，不分拆相同预测制造区别。这里的“实际L5的90分位”是该组历史统计，不是当时模型输出。</p>{risks}
<h2>M1/M2 分歧、投影与超出训练范围</h2>{disagreement}<p>差值以小数收益/损失/概率转为百分比展示；P10 的差可读为概率百分点差，不能称收益。</p>{projections}
<p>三模型、内外层评分均使用相同处理：μ5下界−1，Q90投影到[0,1]；保存全部原始值与投影标记。投影次数多可能反映模型失配，不能将其隐藏。概率头在单类训练时用同一 Jeffreys 常数并记录；外层发生 {result['outer_single_class_fallbacks']} 次，内层候选拟合发生 {result['inner_single_class_fallbacks']} 次。</p>
{support}<p>支持范围按每次训练中19项原值的最小/最大值判断；越界距离用当期训练标准差计，不是概率。M1/M2使用同一训练日期，因此范围一致；不裁剪输入或悄悄删掉这些日期。</p>
<h2>六个交互的条件作用</h2><p>下表展示首次和最后一次月度拟合。固定其他输入，仅看交互这一部分：伙伴变量处于当期训练的25%或75%分位时，本变量上升一个训练标准差所对应的局部斜率。
μ和Q90在原始小数预测尺度；概率头在log-odds尺度，不能直接当概率百分点。这不是因果权重，也不是离开真实联合状态进行收益优化。</p>{interaction_table}
<h3>同月三个真实联合状态的局部总作用</h3><p>最后一次月度拟合参数不变，选该月首、中、末预测信息日；以下为概率头在log-odds尺度的局部总斜率，并分出交互与样条部分。
当前状态不同，局部作用可以不同。全部19项、三个头、每月三个日期保存在 local_effects.csv；没有凭系数排名淘汰特征。</p>{local_table}
<h2>每次训练与年度内层选择</h2><p>本次共 {result['monthly_fit_cycles']} 个月度拟合周期、{result['model_fits']} 份三模型快照、{result['selection_cycles']} 次年度选择。
每块验证前剔除 label_matures_at≥截止或 label_available_at&gt;截止的训练行，特征也须已到假设可用时间；最大1260个股票交易日窗口不按有效行压缩。
三个63交易日块的候选损失逐行平均，均值用MSE、概率用对数损失、分位用Pinball；并列选择更强收缩。全部候选得分、窗口、事件数、变换参数与系数均保存。</p>
{param_table}<details><summary>全部月度拟合记录</summary>{fit_table}</details>
<p><a href='../{result['output_dir']}/training_rows.csv'>逐次外层训练成员</a> · <a href='../{result['output_dir']}/inner_rows.csv'>逐块训练/验证成员</a> ·
<a href='../{result['output_dir']}/models/{fits.iloc[0].fit_id}.json'>第一份模型快照</a> · <a href='../{result['output_dir']}/models/{fits.iloc[-1].fit_id}.json'>最后一份模型快照</a>。
其余快照由 fits.csv 的 fit_id 对应同目录 models/&lt;fit_id&gt;.json；年度完整候选在 selections/&lt;selection_id&gt;.json。</p>
<h2>真实数据边界、实际命令与验证</h2><p>P1 原始清单 {data['input_verification']['raw_records_verified']} 条、P1/P2/P3 已验收产物摘要复核一致，没有重取行情。未使用SKEW扩展。
历史精确 published_at 尚未取得，继续使用 ASSUMED_NEXT_SESSION / NO_FULL_HISTORICAL_PIT_CLAIM；P1来源差异和P2 SPY总回报方法差异仍保留。</p>
<p>普通合成测试覆盖标签成熟边界、只在训练拟合变换、内层清除、样条外推、单类后备、投影和未来扰动不改变更早滚动预测；夹具不是行情或收益证据。</p>{audit_html}
{future_html}
<details><summary>本次 {len(data['test_cases'])} 项普通测试的实际结果</summary>{test_table}</details>{commands}
<p>唯一入口 <code>.venv/bin/python -m svxylab predictions --open</code>；首月试运行加 <code>--pilot</code>。每次保存到新时间戳目录；已有原始数据和失败不删除。
Python {escape(data['python'])}，{escape(data['platform'])}。</p><details><summary>实际依赖版本</summary><pre>{escape(json.dumps(data['packages'],ensure_ascii=False,indent=2))}</pre></details>
<p>成熟库实现依据：{docs}。各头的正则尺度和优化器按本次记录固定，不跨库比较α的数值大小。</p>
<h2>复算文件与阶段状态</h2><p>{links}</p><p><a href='../{data['run_record']}'>本次运行与代码/配置/数据/产物摘要</a> · <a href='../runs/p4/experiment_record.json'>拟合前实验记录</a> · <a href='../runs/p4/eligibility_preflight.json'>首次预测条件核对</a> · <a href='../runs/p4/dependency_install.json'>依赖安装结果</a> · <a href='../STATUS.md'>STATUS.md</a></p>
<p>停在 P4。工程完成不要求优于 M0；资本或经济收益优势尚未验证，P5持仓映射及2024年后封存评估均未执行。</p></main></body></html>"""
    (root / "reports/predictions.html").write_text(doc)


def build_predictions_report(root: Path, *, open_report=False, pilot=False):
    root = root.resolve()
    if Path(sys.prefix).resolve() != (root / ".venv").resolve():
        raise ValueError("请使用项目解释器 .venv/bin/python")
    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = root / "runs/p4" / stamp
    run_dir.mkdir(parents=True)
    output = root / "data/clean/p4" / stamp
    config = tomllib.loads((root / "experiment.toml").read_text())
    definitions = json.loads((root / "FEATURES.json").read_text())
    interactions = [x for x in definitions["interactions"] if x["id"] in config["models"]["interaction_ids"]]
    source_paths = sorted((root / "src").rglob("*.py")) + sorted((root / "tests").rglob("*.py"))
    source_paths += [root / p for p in ["experiment.toml","FEATURES.json","RESEARCH_SPEC.md","pyproject.toml","requirements-lock.txt","runs/p4/experiment_record.json","runs/p4/independent_validation.py","runs/p4/test_real_causality.py"]]
    hashes = {p.relative_to(root).as_posix():sha256(p.read_bytes()).hexdigest() for p in source_paths}
    write_json(run_dir / "started.json",{"stage":"P4","pilot":pilot,"started_at":now.isoformat(),"source_sha256":hashes,"config":config})
    try:
        frame, inputs = load_development_data(root,config)
        result = run_predictions(frame,config,interactions,output,pilot=pilot)
        result["output_dir"] = output.relative_to(root).as_posix()
        write_json(run_dir / "computation.json",result)
        plot_predictions(root,output)
    except Exception:
        (run_dir / "failure.txt").write_text(traceback.format_exc())
        raise
    commands = [run_command(args,root,timeout=60) for args in (
        [sys.executable,"-m","pip","--disable-pip-version-check","check"],
        [sys.executable,"-m","pytest","-q","--junitxml",str(run_dir / "pytest.xml")],
        ["git","diff","--check"],["git","rev-parse","HEAD"],["git","remote"],
    )]
    cases=[]
    if (run_dir / "pytest.xml").exists():
        for case in ET.parse(run_dir / "pytest.xml").getroot().iter("testcase"):
            state="失败" if case.find("failure") is not None or case.find("error") is not None else ("跳过" if case.find("skipped") is not None else "通过")
            cases.append({"name":case.get("name"),"result":state})
    passed = bool(cases) and all(c["result"]=="通过" for c in cases) and all(c["exit_code"]==0 for c in commands)
    data={"stage":"P4","generated_at":now.astimezone().isoformat(),"result":result,"input_verification":inputs,
          "invocation":".venv/bin/python -m svxylab predictions"+(" --pilot" if pilot else "")+(" --open" if open_report else ""),
          "engineering_complete":passed and not pilot,"pilot_complete":passed if pilot else None,
          "tests_passed":sum(c["result"]=="通过" for c in cases),"tests_failed":sum(c["result"]=="失败" for c in cases),
          "test_cases":cases,"commands":commands,"source_sha256":hashes,
          "artifact_sha256":{p.relative_to(root).as_posix():sha256(p.read_bytes()).hexdigest() for p in sorted(output.rglob("*")) if p.is_file()},
          "python":sys.version,"platform":platform.platform(),
          "packages":{d.metadata["Name"]:d.version for d in sorted(metadata.distributions(),key=lambda d:d.metadata["Name"].lower())},
          "run_record":(run_dir / "predictions_run.json").relative_to(root).as_posix(),"locked_period_evaluated":False,"portfolio_mapping_performed":False}
    def save():
        write_json(root / data["run_record"],data)
        render(root,data)
    save()
    if open_report:
        opened=run_command(["/usr/bin/open",str(root / "reports/predictions.html")],root)
        commands.append(opened)
        passed=passed and opened["exit_code"]==0
        data["engineering_complete"]=passed and not pilot
        save()
    print(f"P4 {'首月试运行' if pilot else '开发段重放'}：{root / 'reports/predictions.html'}\n"
          f"{result['forecast_sessions']} 日有预测，{result['no_forecast_sessions']} 日无预测；{result['model_fits']} 份模型。\n"
          f"普通 pytest：{data['tests_passed']} 通过，{data['tests_failed']} 失败。",flush=True)
    return 0 if passed else 1
