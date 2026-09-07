"""P3 冻结输入、原始特征、中文状态卡与相关性诊断的本地报告。"""

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
from svxylab.features import (CORE_IDS, EXTENSION_IDS, INPUT_COLUMNS, LOOKBACK, UNITS, build_inputs,
                             calculate_features, feature_availability, missing_reasons)


def write_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def verify_hashes(root: Path, hashes: dict):
    for path, expected in hashes.items():
        if sha256((root / path).read_bytes()).hexdigest() != expected:
            raise ValueError(f"已验收文件摘要改变：{path}；不会覆盖或重新下载")


def prepare_features(root: Path, output: Path) -> dict:
    config = tomllib.loads((root / "experiment.toml").read_text())
    specification = json.loads((root / "FEATURES.json").read_text())
    experiment = json.loads((root / "runs/p3/experiment_record.json").read_text())
    if ([f["id"] for f in specification["features"] if f["tier"] == "core"] != CORE_IDS or
            [f["id"] for f in specification["features"] if f["tier"] == "extension"] != EXTENSION_IDS):
        raise ValueError("P3 特征列与当前 FEATURES.json 不一致")
    start = config["data"]["feature_history_requested_start"]
    if start != experiment["history_start"]:
        raise ValueError("研究起点改变，必须先记录实验变更")
    frozen = json.loads((root / "runs/p1/freeze.json").read_text())
    p1 = json.loads((root / "runs/p1/reproducibility.json").read_text())
    acceptance = json.loads((root / "runs/p2/acceptance.json").read_text())
    verify_hashes(root, {acceptance["accepted_run"]: acceptance["accepted_run_sha256"]})
    p2 = json.loads((root / acceptance["accepted_run"]).read_text())
    verify_hashes(root, p1["clean_sha256"])
    verify_hashes(root, acceptance["accepted_artifact_sha256"])
    manifest = root / "data/raw/downloads.jsonl"
    raw_records = [json.loads(line) for line in manifest.read_text().splitlines()]
    verify_hashes(root, {r["raw_file"]: r["sha256"] for r in raw_records})
    clean = root / "data/clean"
    p2_dir = root / p2["result"]["output_dir"]
    sessions = pd.read_csv(clean / "equity_sessions.csv").as_of_session.tolist()
    if sessions[-1] != frozen["requested_end"] or sessions[-1] != experiment["frozen_last_session"]:
        raise ValueError("P3 必须继续使用已冻结截止日")
    indices = {s: pd.read_csv(clean / f"{s}_daily.csv") for s in ["VIX", "VVIX", "VIX9D", "VIX3M", "SKEW"]}
    returns = {s: pd.read_csv(p2_dir / f"{s}_returns.csv") for s in ["SPY", "SVXY"]}
    # 只读取总回报与时钟，不加载 labels.csv 或账本的收益结果。
    inputs = build_inputs(sessions, pd.read_csv(clean / "VX_front_three.csv"), pd.read_csv(clean / "VX_contract_daily.csv"),
                          indices, returns, history_start=start)
    values = calculate_features(inputs)
    reasons = missing_reasons(values, inputs)
    availability = feature_availability(values, pd.read_csv(p2_dir / "clock.csv"))
    output.mkdir(parents=True, exist_ok=False)
    values[CORE_IDS].to_csv(output / "core_features.csv")
    values[EXTENSION_IDS].to_csv(output / "extension_features.csv")
    inputs.to_csv(output / "feature_inputs.csv")
    availability.to_csv(output / "feature_availability.csv")
    reasons.to_csv(output / "missing_reasons.csv")
    coverage = []
    observations = []
    for key in values:
        good = values[key].notna()
        missing_dates = values.index[~good].tolist()
        coverage.append({"feature": key, "tier": "core" if key in CORE_IDS else "extension", "unit": UNITS[key],
                         "expected": len(values), "available": int(good.sum()), "missing": int((~good).sum()),
                         "first_valid": values[key].first_valid_index(), "last_valid": values[key].last_valid_index(),
                         "warmup_missing": int(reasons[key].eq("INSUFFICIENT_POST_2019_HISTORY").sum()),
                         "other_missing": int(((~good) & ~reasons[key].eq("INSUFFICIENT_POST_2019_HISTORY")).sum()),
                         "missing_sessions": json.dumps(missing_dates)})
        obs = pd.DataFrame({"feature": key, "value": values[key], "unit": UNITS[key],
                            "window_start_session": pd.Series(values.index, index=values.index).shift(LOOKBACK[key]),
                            "latest_input_session": values.index, "assumed_available_at": availability.assumed_available_at,
                            "available_under_assumption": good, "missing_reason": reasons[key],
                            "input_rows_file": "feature_inputs.csv", "definition_file": "feature_metadata.json"})
        observations.append(obs.reset_index())
    pd.DataFrame(coverage).to_csv(output / "coverage.csv", index=False)
    pd.concat(observations, ignore_index=True).to_csv(output / "feature_observations.csv", index=False)
    vx_sources = ["data/clean/VX_front_three.csv", "data/clean/VX_contract_daily.csv", "data/clean/VX_contract_calendar.csv"]
    actual_sources = {
        "A01": vx_sources, "A02": vx_sources, "A03": vx_sources + ["data/clean/VIX_daily.csv"],
        "A04": vx_sources, "A05": vx_sources,
        "B01": ["data/clean/VIX_daily.csv"], "B02": ["data/clean/VIX9D_daily.csv", "data/clean/VIX_daily.csv"],
        "B03": ["data/clean/VIX_daily.csv", "data/clean/VIX3M_daily.csv"], "C01": ["data/clean/VVIX_daily.csv"],
        "C02": ["data/clean/VVIX_daily.csv"], "C03": ["data/clean/VIX_daily.csv"],
        **{k: [f"{p2['result']['output_dir']}/SPY_returns.csv", "data/clean/SPY_daily.csv", "data/clean/SPY_corporate_actions.csv"]
           for k in ["D01", "D02", "D03", "E01", "F01", "F02"]},
        **{k: [f"{p2['result']['output_dir']}/SVXY_returns.csv", "data/clean/SVXY_daily.csv", "data/clean/SVXY_corporate_actions.csv"]
           for k in ["F03", "F04"]},
        "G01": ["data/clean/SKEW_daily.csv"], "G02": ["data/clean/SKEW_daily.csv"],
    }
    actual_sources["E01"] += ["data/clean/VIX_daily.csv"]
    dictionary = {"definition_status": specification["definition_status"], "feature_values": "RAW_UNSTANDARDIZED",
                  "published_at": None, "availability_policy": "ASSUMED_NEXT_SESSION", "pit_status": "NO_FULL_HISTORICAL_PIT_CLAIM",
                  "provenance": "逐日输入保存在 feature_inputs.csv，含各来源原件路径、sha256、实际收到时间与未知 published_at。窗口中的各日沿同一股票交易日索引取原值；A04 另存两端同合约结算和原件。",
                  "window_rule": experiment["rolling_windows"], "features": []}
    for feature in specification["features"]:
        key = feature["id"]
        dictionary["features"].append({**feature, "unit": UNITS[key], "input_columns": INPUT_COLUMNS[key],
                                       "required_prior_sessions": LOOKBACK[key], "actual_source_files": actual_sources[key],
                                       "source_sha256": {p: sha256((root / p).read_bytes()).hexdigest() for p in actual_sources[key]}})
    write_json(output / "feature_metadata.json", dictionary)
    masks = {"core": availability.core_complete, "extension_only": availability.extension_complete,
             "core_plus_extension": availability.core_plus_extension_complete}
    common = {}
    for name, mask in masks.items():
        selected = values.index[mask]
        common[name] = {"rows": len(selected), "first": selected[0] if len(selected) else None,
                        "last": selected[-1] if len(selected) else None, "missing_sessions": values.index[~mask].tolist()}
    correlation = {}
    for name, columns, mask in [("core", CORE_IDS, masks["core"]),
                                ("with_skew", CORE_IDS + EXTENSION_IDS, masks["core_plus_extension"])]:
        frame = values.loc[mask & (values.index <= "2023-12-31"), columns]
        corr = frame.corr(method="pearson")
        corr.to_csv(output / f"correlation_{name}_2019_2023.csv", index_label="feature")
        pairs = [{"feature_1": a, "feature_2": b, "pearson": float(corr.loc[a, b]), "common_rows": len(frame)}
                 for i, a in enumerate(columns) for b in columns[i+1:]]
        pd.DataFrame(pairs).to_csv(output / f"correlation_pairs_{name}_2019_2023.csv", index=False)
        correlation[name] = {"rows": len(frame), "first": frame.index[0], "last": frame.index[-1],
                             "largest_absolute_pairs": sorted(pairs, key=lambda p: abs(p["pearson"]), reverse=True)[:8]}
    cards = []
    for day in experiment["state_card_dates"]:
        cards.append({"as_of_session": day, "inputs": {k: float(inputs.loc[day, k]) for k in ["VIX", "VVIX", "F1", "F2", "F3", "VIX9D", "VIX3M", "tau1"]},
                      "features": {k: float(values.loc[day, k]) for k in values},
                      "lag5_session": inputs.loc[day, "lag5_session"],
                      "decision_at_new_york": availability.loc[day, "decision_at_new_york"],
                      "execution_at": availability.loc[day, "execution_at"]})
    write_json(output / "state_cards.json", cards)
    roll_day = "2019-03-19"
    i = inputs.index.get_loc(roll_day)
    roll = {"as_of_session": roll_day, "lag5_session": inputs.lag5_session.loc[roll_day],
            "contract_at_t": inputs.f1_contract.loc[roll_day], "front_at_lag5": inputs.f1_contract.iloc[i-5],
            "current_settle": float(inputs.a04_current_settle.loc[roll_day]),
            "same_contract_lag5_settle": float(inputs.a04_lag5_settle.loc[roll_day]),
            "old_front_lag5_settle": float(inputs.F1.iloc[i-5]), "A04": float(values.A04.loc[roll_day]),
            "incorrect_front_splice_log_change": float(np.log(inputs.F1.loc[roll_day]/inputs.F1.iloc[i-5]))}
    write_json(output / "real_roll_check.json", roll)
    return {"first": values.index[0], "last": values.index[-1], "sessions": len(values), "coverage": coverage,
            "common": common, "correlation": correlation, "cards": cards, "roll": roll,
            "p1_clean_sha256": p1["clean_sha256"], "p2_artifact_sha256": acceptance["accepted_artifact_sha256"],
            "accepted_p2_run": acceptance["accepted_run"], "raw_records_verified": len(raw_records),
            "raw_manifest_sha256": sha256(manifest.read_bytes()).hexdigest(),
            "output_dir": output.relative_to(root).as_posix()}


def plot_correlation(root: Path, output: Path, rows: int):
    os.environ.setdefault("MPLCONFIGDIR", str(root / ".cache/matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font = FontProperties(fname="/System/Library/Fonts/STHeiti Light.ttc")
    corr = pd.read_csv(output / "correlation_core_2019_2023.csv", index_col="feature")
    fig, ax = plt.subplots(figsize=(11, 10), layout="constrained")
    shown = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(CORE_IDS)), CORE_IDS, rotation=45, ha="right")
    ax.set_yticks(range(len(CORE_IDS)), CORE_IDS)
    for i in range(len(CORE_IDS)):
        for j in range(len(CORE_IDS)):
            value = corr.iloc[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(value) > .6 else "#222222")
    ax.set_title(f"19 项原始特征的 Pearson 相关性 · 2019–2023\n共同完整样本 {rows} 日 · 仅描述信息重叠", fontproperties=font, pad=18)
    fig.colorbar(shown, ax=ax, shrink=.75, label="Pearson r")
    fig.savefig(output / "correlation.png", dpi=150)
    fig.savefig(output / "correlation.svg", metadata={"Date": None})
    plt.close(fig)


def render(root: Path, data: dict):
    result = data["result"]
    output = root / result["output_dir"]
    definitions = json.loads((output / "feature_metadata.json").read_text())["features"]
    names = {f["id"]: f["name_zh"] for f in definitions}
    coverage = table(["特征", "名称", "单位", "实有 / 应有", "预热缺失", "其他缺失", "首个有效日"],
                     [[r["feature"], names[r["feature"]], r["unit"], f"{r['available']} / {r['expected']}",
                       r["warmup_missing"], r["other_missing"], r["first_valid"]] for r in result["coverage"]])
    formulas = table(["特征", "家族 / 名称", "原公式", "输入列 / 最大回看", "实际源文件"],
                     [[f["id"], f"{f['family']} / {f['name_zh']}", f["formula"],
                       f"{', '.join(f['input_columns'])}；向前 {f['required_prior_sessions']} 个股票交易日",
                       "; ".join(f["actual_source_files"])] for f in definitions])
    cards = []
    for card in result["cards"]:
        x, f = card["inputs"], card["features"]
        relief = "下降，正在缓解" if f["C03"] < 0 else "上升，正在加压"
        vvix = "增强" if f["C02"] > 0 else "减弱"
        curve = "F2 高于 F1，曲线向上" if x["F2"] > x["F1"] else "F2 低于或等于 F1，近端曲线倒挂或持平"
        if card["as_of_session"] == "2020-04-06":
            headline = "压力水平仍高，变化方向正在缓解"
        elif card["as_of_session"] == "2020-02-21":
            headline = "曲线仍向上，同时压力指标上升"
        else:
            headline = "水平、曲线与自身路径各自描述一件事"
        rows = [[k, names[k], f"{f[k]:.8f}", UNITS[k]] for k in CORE_IDS+EXTENSION_IDS]
        cards.append(f"<article><h3>{card['as_of_session']} · {headline}</h3>"
                     f"<p>VIX {x['VIX']:.2f} 点；相对 {card['lag5_session']}，VIX 变化 {np.expm1(f['C03']):+.2%}（C03={f['C03']:+.6f}），{relief}。"
                     f"VVIX {x['VVIX']:.2f} 点，五日变化 {np.expm1(f['C02']):+.2%}，其不确定性定价{vvix}。</p>"
                     f"<p>真实 VX 结算 F1={x['F1']:.4f}、F2={x['F2']:.4f}、F3={x['F3']:.4f}；{curve}，A01={f['A01']:+.6f}。"
                     f"同时隐含期限比 B02={f['B02']:+.6f}、B03={f['B03']:+.6f}；两组曲线来自不同标的。</p>"
                     f"<p>SPY 过去五日总回报 {np.expm1(f['F01']):+.2%}，21 日收盘波动 {f['D02']:.2%}，下跌半方差占比 {f['D03']:.2%}；"
                     f"SVXY 过去五日总回报 {np.expm1(f['F03']):+.2%}、21 日收盘波动 {f['F04']:.2%}。"
                     f"E01={f['E01']:+.6f} 是隐含与历史方差差。</p>"
                     f"<p>假设可用于 {escape(card['decision_at_new_york'])} 的决策，执行收盘为 {escape(card['execution_at'])}。"
                     f"这些是当日及向后信息的描述；没有模型概率、仓位建议或此后收益。</p>"
                     f"<details><summary>展开当日全部 19 项核心与 2 项扩展原值</summary>{table(['ID','名称','原值','单位'],rows)}</details></article>")
    pairs = table(["特征 1", "特征 2", "Pearson r", "共同日数"],
                  [[p['feature_1']+' '+names[p['feature_1']], p['feature_2']+' '+names[p['feature_2']],
                    f"{p['pearson']:.6f}", p['common_rows']] for p in result['correlation']['core']['largest_absolute_pairs']])
    missing = table(["特征", "预热以外的缺失交易日"],
                    [[r["feature"], ", ".join(json.loads(r["missing_sessions"])[r["warmup_missing"]:])]
                     for r in result["coverage"] if r["other_missing"]])
    commands = "".join(f"<details><summary>退出码 {c['exit_code']} · <code>{escape(c['command'])}</code></summary><pre>{escape(c['output'] or '（无输出）')}</pre></details>" for c in data["commands"])
    tests = table(["普通 pytest", "实际结果"], [[c["name"], c["result"]] for c in data["test_cases"]])
    artifacts = " · ".join(f"<a href='../{escape(path)}'>{escape(Path(path).name)}</a>" for path in data["artifact_sha256"] if not path.endswith((".png", ".svg")))
    audit_file = (root / data["run_record"]).with_name("independent_recalculation.json")
    audit_html = ""
    if audit_file.exists():
        audit = json.loads(audit_file.read_text())
        if audit["run_record"] == data["run_record"]:
            largest = max(v["max_absolute_error"] for v in audit["per_feature"].values())
            audit_html = (f"<p><strong>本次真实数据独立复算：{'通过' if audit['passed'] else '未通过'}</strong>。"
                          f"仅用 Python 标准库，从 P1 价格/公司行动重建总回报，从官方到期日历重选合约，逐标量核对 {audit['feature_cells_checked']} 格（包括空值）、"
                          f"{audit['front_contract_selection_checks']} 次合约选择。差异 {audit['mismatch_count']} 格，最大绝对数值误差 {largest:.3g}。"
                          f"两套相关性的 171 / 210 对也独立复算；没有调用生产特征或总回报函数。</p>"
                          f"<p>实际审计命令 <code>.venv/bin/python runs/p3/independent_recalculation.py</code>，"
                          f"<a href='../{audit_file.relative_to(root).as_posix()}'>本次独立结果</a> · <a href='../runs/p3/independent_recalculation.py'>审计算式</a>。"
                          f"这是本轮审计证据；日常复算继续使用唯一项目入口。</p>")
    common, corr, roll = result["common"], result["correlation"], result["roll"]
    doc = f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>
<title>SVXYLab · P3 联合信息表</title><style>
body{{margin:0;font:16px/1.7 -apple-system,BlinkMacSystemFont,'PingFang SC',sans-serif;color:#22313d;background:white}}
main{{max-width:1240px;margin:auto;padding:28px 24px 70px}}h1{{font-size:28px}}h2{{font-size:22px;margin-top:32px}}h3{{font-size:18px}}a{{color:#165b8a}}
.status,article{{background:#f3f6f8;padding:12px 18px;margin:18px 0;border-left:4px solid #607d8b}}.scroll{{overflow:auto}}
table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{border:1px solid #d6dfe6;padding:8px;text-align:left;vertical-align:top}}th{{background:#f3f6f8}}
img{{max-width:100%;height:auto}}code,pre{{font:13px/1.6 ui-monospace,monospace}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}details{{border-bottom:1px solid #ddd;padding:9px 0}}summary{{cursor:pointer;overflow-wrap:anywhere}}
</style></head><body><main><h1>SVXYLab · P3 联合信息表</h1><p>{escape(data['generated_at'])} · RESEARCH_ONLY · 已验收 P2 检查点 5912f3b</p>
<div class='status'><p><strong>工程：</strong>{'完成' if data['engineering_complete'] else '未完成'}；19 项核心和 2 项独立扩展已计算，普通 pytest {data['tests_passed']} 项通过，{data['tests_failed']} 项失败或错误。</p>
<p><strong>研究结果：</strong>只交付原始特征、覆盖、时钟与信息重叠。定义仍为 DESIGN_NOT_EMPIRICALLY_VALIDATED；没有拟合、标准化、筛选特征或收益优势结论。</p>
<p><strong>真实数据：</strong>{result['first']} 至 {result['last']}，{result['sessions']} 个股票交易日。核心完整 {common['core']['rows']} 日；扩展两项单独完整 {common['extension_only']['rows']} 日；核心加扩展共同完整 {common['core_plus_extension']['rows']} 日。</p></div>
<p><a href='../{result['output_dir']}/core_features.csv'>下载 19 列核心 CSV</a> · <a href='../{result['output_dir']}/extension_features.csv'>下载 2 列 SKEW 扩展 CSV</a>。文件另含 as_of_session 日期索引；保留全部交易日与空值，不删失去完整特征的日期。</p>
<h2>覆盖、预热与缺失</h2><p>研究输入及预热统一从 2019-01-01 开始。核心首个完整日 {common['core']['first']}，此后至冻结日的完整行数为 {common['core']['rows']}。
21 日波动使用 21 个对数回报，需要 22 个 TR 水平；F02 的 21 日回撤只使用含当日在内 21 个 TR 水平。
所有滞后都沿完整股票日历计算，不按删掉空值后的行数移动；窗口不全不插值，D03 分母为零保留空并单列原因。</p>
{coverage}<h3>预热以外的真实缺口</h3>{missing}
<p>SKEW 缺值只影响扩展；G02 还受同一交易日历上五日前端点缺值影响。覆盖 CSV 列出全部缺失日期，missing_reasons.csv 逐格给原因。核心 19 项原始数值保留独立输出。</p>
<h2>三张真实日期状态卡</h2><p>三个日期在计算前指定，只从当日及过去数值说明同时存在的状态。曲线向上可以和压力加速共存；高 VIX 水平也可以和五日下降共存。
“有利曲线”仅指 F2&gt;F1 的定价形态，不保证展期收益；变化率的正负不等于对未来方向的预测。</p>{''.join(cards)}
<h2>A04 的真实换月核对</h2><p>{roll['as_of_session']} 当天前月为 {roll['contract_at_t']}；五个股票交易日前 {roll['lag5_session']} 的前月还是 {roll['front_at_lag5']}。
正确 A04 = ln({roll['current_settle']:.6f} / {roll['same_contract_lag5_settle']:.6f}) = {roll['A04']:+.8f}，两端使用当天选定的同一个合约。
如果错误拼接前月序列，则会算成 ln({roll['current_settle']:.6f} / {roll['old_front_lag5_settle']:.6f}) = {roll['incorrect_front_splice_log_change']:+.8f}。
两条数值和原合约身份都保存在 real_roll_check.json；计算没有采用错误拼接。</p>
<h2>相关性：描述共同信息，不淘汰特征</h2><p>核心 Pearson 矩阵只使用 {corr['core']['first']} 至 {corr['core']['last']} 的 {corr['core']['rows']} 个完整共同日；所有格子使用同一日期集合。
加 SKEW 的独立 21 列矩阵使用 {corr['with_skew']['rows']} 个共同日。2024 年后的特征原值与覆盖照常保存，相关性诊断不使用这一段，不查看标签或封存模型绩效。</p>
<img src='../{result['output_dir']}/correlation.png' alt='19项真实原始特征在2019至2023共同日期的Pearson相关性矩阵'><p><a href='../{result['output_dir']}/correlation.svg'>下载矢量图</a>。</p>
{pairs}<p>表列绝对相关性最大的八对，仅帮助看重叠，19 列全部保留；完整矩阵与逐对样本数均可下载。E01 由 VIX 和 D02 派生，与 B/D 共享原始信息。
相关性不能证明独立增量或因果关系，也没有计算单特征收益或显著性来筛选。A 家族是不同到期的真实 VX 期货；B 家族是股票期权隐含波动期限，不能互相冒充。</p>
<h2>逐项公式、单位与来源</h2><p>以下公式来自唯一特征字典 FEATURES.json。D01/D02/F04 是对数回报平方均值的年化平方根，不是去均值后的样本标准差；波动为小数。
E01 先把 VIX 点数除 100 再平方，减去 SPY 已发生的年化历史方差，因此不是已测得的真实前瞻 VRP。
SKEW 是尾部相对定价指数，不是物理崩盘概率。</p>{formulas}
<p>feature_metadata.json 逐项保存经济问题、单位、最大回看、实际源文件及摘要；feature_inputs.csv 保存每个交易日的未标准化输入、真实合约与原件路径/摘要/收到时间。
feature_observations.csv 将每项值、公式窗口最早日、最新输入日和假设可用时间连在一起。多日窗口的各日原件可按日期回到输入表核对；A04 明列两端同合约原件。</p>
<h2>时间边界与可复算范围</h2><p>信息日 t 收盘数据，假设在 t+1 股票收盘前 60 分钟可得，再于 t+1 收盘执行。逐日时钟沿用已验收 P2，包含夏令时、半日市及节假日。
历史真实 published_at 尚未取得，因此是 ASSUMED_NEXT_SESSION，NO_FULL_HISTORICAL_PIT_CLAIM。原始收到时间不能冒充当年的发布时间；额外一交易日延迟敏感性仍按规格留在 P5。</p>
<p>沿用 P2 从分红与拆分计算的 SPY/SVXY 总回报；SPY 与供应商复权参考最大单日差约 2.520235bp 的既有方法差异仍保留，详见 <a href='baselines.html'>P2 报告</a>。
本次验证 P1 的 {len(result['p1_clean_sha256'])} 份清洗文件、P2 的 {len(result['p2_artifact_sha256'])} 份已验收产物及 {result['raw_records_verified']} 条原始记录摘要；没有重新取数或覆盖失败。</p>
<h2>实际命令与测试</h2><p>复算并打开：<code>.venv/bin/python -m svxylab features --open</code>；普通测试：<code>.venv/bin/python -m pytest -q</code>。</p>
<p>合成夹具仅检查 21 个公式、换月、预热、缺失、零分母、未来篡改/追加不改过去，以及 t+1 时钟；它们不是真实行情或回测。</p>
{audit_html}
<details><summary>展开本次全部 {len(data['test_cases'])} 项测试</summary>{tests}</details>{commands}
<p>Python {escape(data['python'])}；平台 {escape(data['platform'])}；本阶段没有安装新依赖。</p>
<details><summary>实际软件依赖版本</summary><pre>{escape(json.dumps(data['packages'],ensure_ascii=False,indent=2))}</pre></details>
<h2>文件与阶段状态</h2><p>{artifacts}</p><p><a href='../{data['run_record']}'>本次运行记录及数据/代码/结果摘要</a> · <a href='../runs/p3/experiment_record.json'>计算前实验记录</a> · <a href='../runs/p2/acceptance.json'>P2 验收记录</a> · <a href='../STATUS.md'>STATUS.md</a></p>
<p>当前停在 P3，等待验收；联合预测和训练属于 P4，尚未执行。工程完成不代表模型有收益优势。</p></main></body></html>"""
    (root / "reports/features.html").write_text(doc)


def build_features_report(root: Path, *, open_report: bool = False) -> int:
    root = root.resolve()
    if Path(sys.prefix).resolve() != (root / ".venv").resolve():
        raise ValueError("请使用项目解释器 .venv/bin/python")
    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = root / "runs/p3" / stamp
    run_dir.mkdir(parents=True)
    output = root / "data/clean/p3" / stamp
    try:
        result = prepare_features(root, output)
        plot_correlation(root, output, result["correlation"]["core"]["rows"])
    except Exception:
        (run_dir / "failure.txt").write_text(traceback.format_exc())
        raise
    commands = [run_command(args, root, timeout=60) for args in (
        [sys.executable, "-m", "pip", "--disable-pip-version-check", "check"],
        [sys.executable, "-m", "pytest", "-q", "--junitxml", str(run_dir / "pytest.xml")],
        ["git", "diff", "--check"], ["git", "rev-parse", "HEAD"], ["git", "remote"],
    )]
    cases = []
    if (run_dir / "pytest.xml").exists():
        for case in ET.parse(run_dir / "pytest.xml").getroot().iter("testcase"):
            state = "失败" if case.find("failure") is not None or case.find("error") is not None else ("跳过" if case.find("skipped") is not None else "通过")
            cases.append({"name": case.get("name"), "result": state})
    complete = bool(cases) and all(c["result"] == "通过" for c in cases) and all(c["exit_code"] == 0 for c in commands)
    paths = sorted((root / "src").rglob("*.py")) + sorted((root / "tests").rglob("*.py"))
    paths += [root / p for p in ["experiment.toml", "FEATURES.json", "RESEARCH_SPEC.md", "pyproject.toml", "requirements-lock.txt", "runs/p3/experiment_record.json", "runs/p3/independent_recalculation.py"]]
    data = {"stage": "P3", "generated_at": now.astimezone().isoformat(), "result": result,
            "invocation": ".venv/bin/python -m svxylab features" + (" --open" if open_report else ""),
            "engineering_complete": complete, "tests_passed": sum(c["result"] == "通过" for c in cases),
            "tests_failed": sum(c["result"] == "失败" for c in cases), "test_cases": cases, "commands": commands,
            "python": sys.version, "platform": platform.platform(),
            "packages": {d.metadata["Name"]: d.version for d in sorted(metadata.distributions(), key=lambda d:d.metadata["Name"].lower())},
            "source_sha256": {p.relative_to(root).as_posix(): sha256(p.read_bytes()).hexdigest() for p in paths},
            "artifact_sha256": {p.relative_to(root).as_posix(): sha256(p.read_bytes()).hexdigest() for p in sorted(output.iterdir())},
            "labels_loaded_for_calculation": False, "model_fitted": False, "feature_selection": False,
            "run_record": (run_dir / "features_run.json").relative_to(root).as_posix()}
    def save():
        write_json(root / data["run_record"], data)
        render(root, data)
    save()
    if open_report:
        opened = run_command(["/usr/bin/open", str(root / "reports/features.html")], root)
        commands.append(opened)
        complete = data["engineering_complete"] = complete and opened["exit_code"] == 0
        save()
    print(f"P3 报告：{root / 'reports/features.html'}\n普通 pytest：{data['tests_passed']} 通过，{data['tests_failed']} 失败\n"
          f"核心完整 {result['common']['core']['rows']} 日；核心加扩展共同完整 {result['common']['core_plus_extension']['rows']} 日。")
    return 0 if complete else 1
