"""P2 中文账本报告、真实净值图与每次运行记录。"""

from datetime import datetime, timezone
from hashlib import sha256
from html import escape
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import sys
import traceback
import xml.etree.ElementTree as ET

import pandas as pd

from svxylab.baselines import NAMES, prepare_baselines
from svxylab.data_report import table
from svxylab.environment import run_command


def money(value):
    return "尚未成熟" if pd.isna(value) else f"{float(value):,.4f}"


def percent(value):
    return "尚未成熟" if pd.isna(value) else f"{float(value):.4%}"


def plot_ledgers(root: Path, output: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(root / ".cache/matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font = FontProperties(fname="/System/Library/Fonts/STHeiti Light.ttc")
    colors = ["#64748b", "#5c8bad", "#26887b", "#c88728", "#bd4b43", "#725aa1"]
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, layout="constrained", height_ratios=[2, 1])
    for (name, label), color in zip(NAMES.items(), colors, strict=True):
        ledger = pd.read_csv(output / f"ledger_{name}.csv")
        shown = ledger.loc[ledger.as_of_session <= "2023-12-31"]
        dates = pd.to_datetime(shown.as_of_session)
        axes[0].plot(dates, shown.equity_end, label=label, color=color, linewidth=1.3)
        axes[1].plot(dates, shown.drawdown * 100, color=color, linewidth=1.1)
    axes[0].set_title("预定对照的真实日线账本 · 2019–2023 · 单边成本 5bp", fontproperties=font)
    axes[0].set_ylabel("期末资产（美元）", fontproperties=font)
    axes[1].set_ylabel("收盘回撤（%）", fontproperties=font)
    axes[1].set_xlim(dates.iloc[0], dates.iloc[-1])
    axes[0].legend(prop=font, ncols=3, fontsize=9, loc="upper left")
    for ax in axes:
        ax.grid(True, linewidth=.5, alpha=.3)
        ax.spines[["right", "top"]].set_visible(False)
    fig.savefig(output / "baselines.png", dpi=150)
    fig.savefig(output / "baselines.svg", metadata={"Date": None})
    plt.close(fig)


def hand_examples(output: Path) -> list:
    ledgers = {name: pd.read_csv(output / f"ledger_{name}.csv") for name in ("fixed_50", "curve", "fixed_100")}
    normal = ledgers["fixed_50"].loc[lambda d: d.as_of_session.between("2019-07-01", "2019-07-05")]
    curve = ledgers["curve"]
    exits = curve.loc[curve.as_of_session.str.startswith("2020") & curve.pre_trade_weight.gt(.9) & curve.target_weight.eq(0)]
    center = int(exits.index[0])
    switching = curve.iloc[center - 2:center + 3]
    company = ledgers["fixed_100"].loc[lambda d: d.as_of_session.between("2024-04-09", "2024-04-12")]
    examples = []
    for title, frame, highlight, detail in (
        ("普通行情：固定 50%，每天按实际偏离再平衡", normal, normal.iloc[0], "目标虽然连续为 50%，价格波动会改变执行前权重，因此可能产生实际换手和成本。"),
        ("换仓：2020 年首次从持有转为现金", switching, curve.loc[center], "退出当日的收盘前损益仍归旧持仓；上一交易日曲线决定本次退出。"),
        ("公司行动：2024-04-11 的 2:1 拆分", company, company.loc[company.as_of_session.eq("2024-04-11")].iloc[0],
         "拆分让份额翻倍、每份价格约减半；不能把价格减半记作亏损 50%。这四行仅核对公司行动，不进行封存模型评价。"),
    ):
        columns = ["as_of_session", "equity_start", "old_weight", "asset_return", "holding_pnl", "cost", "new_weight", "equity_end"]
        examples.append({"title": title, "detail": detail, "rows": frame[columns].to_dict("records"),
                         "highlight": highlight.to_dict()})
    return examples


def render(root: Path, data: dict):
    result = data["result"]
    output = root / result["output_dir"]
    report = root / "reports/baselines.html"
    metrics_rows = []
    for name, m in result["metrics"].items():
        metrics_rows.append([NAMES[name], money(m["ending_equity"]), percent(m["cagr"]), percent(m["max_drawdown"]),
                             percent(m["worst_day"]), percent(m["worst_five_days"]), percent(m["mean_exposure"]),
                             f"{m['annual_turnover']:.4f}", money(m["cost_dollars"])])
    examples_html = []
    for example in data["examples"]:
        rows = [[r["as_of_session"], money(r["equity_start"]), percent(r["old_weight"]), percent(r["asset_return"]),
                 money(r["holding_pnl"]), money(r["cost"]), percent(r["new_weight"]), money(r["equity_end"])] for r in example["rows"]]
        h = example["highlight"]
        arithmetic = (f"{h['as_of_session']}：期初 {money(h['equity_start'])} + 持有损益 {money(h['holding_pnl'])}"
                      f" − 成本 {money(h['cost'])} = 期末 {money(h['equity_end'])} 美元。"
                      f"执行前实际权重 {percent(h['pre_trade_weight'])}，上一信息日目标 {percent(h['target_weight'])}；"
                      f"成交额 {money(h['trade_notional'])} × 0.0005 = 成本 {money(h['cost'])}。")
        if h["split_factor"] != 1:
            arithmetic += (f" 拆分前份额 {money(h['old_shares'])} ÷ {h['split_factor']}"
                           f" = 拆分后份额 {money(h['shares_after_split_before_trade'])}，当日每份收盘 {money(h['close'])}。")
        examples_html.append(f"<h3>{escape(example['title'])}</h3><p>{escape(example['detail'])}</p>"
                             + table(["日期", "期初资产 $", "旧仓位", "SVXY 当日总回报", "旧持仓损益 $", "成本 $", "新仓位", "期末资产 $"], rows)
                             + f"<p class='calculation'>{escape(arithmetic)}</p>")
    labels = pd.read_csv(output / "labels.csv").set_index("as_of_session")
    label_rows = [[d, labels.loc[d, "execution_session"], labels.loc[d, "label_end_session"],
                   percent(labels.loc[d, "R5"]), percent(labels.loc[d, "L5"]),
                   "尚未成熟" if pd.isna(labels.loc[d, "Y10"]) else int(labels.loc[d, "Y10"]),
                   labels.loc[d, "label_matures_at"], labels.loc[d, "label_available_at"]]
                  for d in ("2019-01-02", "2019-07-01", "2020-03-16", result["last"])]
    clock = pd.read_csv(output / "clock.csv").set_index("as_of_session")
    clock_rows = [[date, clock.loc[date, "decision_session"], clock.loc[date, "decision_at_new_york"],
                   clock.loc[date, "execution_at"]] for date in ("2024-03-08", "2024-11-27")]
    spy = pd.read_csv(output / "SPY_returns.csv")
    event = spy.loc[spy.cash_dividend.gt(0)].iloc[0]
    comparison_rows = [[s, v["largest_difference_session"], f"{v['max_absolute_daily_difference_bps']:.6f}",
                        percent(v["computed_return"]), percent(v["provider_reference_return"])]
                       for s, v in result["reference_return_comparison"].items()]
    audits = [[NAMES[name], "全部对账通过" if all(a["checks"].values()) else "存在失败",
               f"{a['max_equity_error_dollars']:.3g}", f"{a['max_cash_error_dollars']:.3g}", f"{a['max_cost_error_dollars']:.3g}"]
              for name, a in result["audit"].items()]
    commands = "".join(f"<details><summary>退出码 {r['exit_code']} · <code>{escape(r['command'])}</code></summary>"
                        f"<pre>{escape(r['output'] or '（无输出）')}</pre></details>" for r in data["commands"])
    tests = table(["普通 pytest 测试", "实际结果"], [[r["name"], r["result"]] for r in data["test_cases"]])
    links = " · ".join(f"<a href='../{escape(p)}'>{escape(Path(p).name)}</a>" for p in data["artifact_sha256"] if p.endswith(".csv"))
    m = next(iter(result["metrics"].values()))
    document = f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>
<title>SVXYLab · P2 时钟与连续账本</title><style>
body{{margin:0;font:16px/1.7 -apple-system,BlinkMacSystemFont,'PingFang SC',sans-serif;color:#22313d;background:white}}
main{{max-width:1240px;margin:auto;padding:28px 24px 70px}}h1{{font-size:28px}}h2{{font-size:22px;margin-top:32px}}h3{{font-size:18px}}
a{{color:#165b8a}}.status,.calculation{{background:#f2f5f7;padding:12px 18px;border-left:4px solid #5d7786}}
.scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{border:1px solid #d6dfe6;padding:8px;text-align:left;vertical-align:top}}th{{background:#f3f6f8}}
img{{max-width:100%;height:auto}}code,pre{{font:13px/1.6 ui-monospace,monospace}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}details{{border-bottom:1px solid #ddd;padding:9px 0}}summary{{cursor:pointer;overflow-wrap:anywhere}}
</style></head><body><main><h1>SVXYLab · P2 时钟与连续账本</h1><p>{escape(data['generated_at'])} · RESEARCH_ONLY · 已验收 P1 检查点 dc3d4ff</p>
<div class='status'><p><strong>工程：</strong>{'完成' if data['engineering_complete'] else '未完成'}；普通 pytest {len(data['test_cases'])} 项，通过 {data['tests_passed']}，失败/错误 {data['tests_failed']}。六条连续账户已按资金、份额、现金与成交成本对账。</p>
<p><strong>研究结果：</strong>本页只呈现预定对照的真实账本，不包含联合模型、训练或参数优选；这些结果不证明模型有收益优势。</p>
<p><strong>真实数据：</strong>{result['first']} 至 {result['last']}，{len(result['sessions'])} 个交易日；复用 P1 冻结原件，核对 {result['raw_records_verified']} 条原始保存记录。SKEW 的两日缺失不参与本阶段账本。</p></div>
<h2>预定对照净值：{m['first']} 至 {m['last']}</h2>
<p>净值图与汇总采用 {m['rows']} 个交易日，起始现金 {result['capital']:,.0f} 美元，单边成本 {result['rate'] * 10000:g}bp，现金收益 0。
所有账户最早在 2019-01-03 收盘换仓。固定权重每日收盘再平衡；曲线规则用上一交易日 F2>F1 决定 100% 持有，否则现金。
2024 年后的全段账本已保存在 CSV，供会计核对；本页不做封存模型评分或按其结果调参，模型封存评估仍在 P7。</p>
<img src='../{result['output_dir']}/baselines.png' alt='根据真实逐日账户生成的六个预定对照净值与收盘回撤图'>
<p><a href='../{result['output_dir']}/baselines.svg'>下载矢量图</a>。图由 Matplotlib 根据下方可复算 CSV 生成，无模拟行情。</p>
{table(['对照','期末资产 $','年化收益','最大收盘回撤','最差单日','最差连续五日','平均旧仓位','年均单边换手倍数','累计成本 $'], metrics_rows)}
<p>年化收益按首末收盘实际日历跨度 / 365.25 换算；回撤从连续账户历史收盘峰值计算。平均敞口采用每日损益发生前的旧权重。
换手为实际成交额 / 执行前资产，年均值除以相同年数；不是相邻目标权重之差。五日账户损益只作路径统计，不产生额外交易。末日保留持仓按市值计价，没有虚构清仓或现金利息。</p>
<p>扣费公式：执行前资产为 E、实际股仓为 u、扣费后目标为 w、单边费率 c=0.0005。
买入时 E_new=E×(1+c×u)/(1+c×w)；卖出时 E_new=E×(1−c×u)/(1−c×w)。
新股市值为 w×E_new，费用为 c×|w×E_new−u×E|；余款为现金。这个公式保证费用来自实际成交额，并使新权重在扣费后仍等于目标。</p>
<h2>三段可手算的真实账本</h2><p>每段不超过五行。表内金额显示四位小数，CSV 保留计算精度；“旧仓位”来自上一收盘，“新仓位”是本次收盘扣费后的实际权重。</p>
{''.join(examples_html)}
<h2>总回报与公司行动口径</h2><p>每天总回报 = (当日收盘 + 每份现金分红 + 每份资本利得分配) / (拆分价格因子 × 前日收盘) − 1。
首日 TR 锚定为 100，首日以前的回报保持空；没有使用 2019 年前数据。总回报按除息日确认分配并在收盘再投资，是研究计算口径，不声称当天已收到实际现金。</p>
<p>SPY 仅建立后续输入需要的总回报，不进入 SVXY/CASH 账户。例如 {event.as_of_session}，前收盘 {money(event.previous_close)}、当日收盘 {money(event.close)}、每份分红 {money(event.cash_dividend)} 美元：
({money(event.close)} + {money(event.cash_dividend)}) / {money(event.previous_close)} − 1 = {percent(event.simple_return)}。
本段 SVXY 没有报告现金分配；其 2024 年拆分已在上方真实账本核对。基金费用已在真实基金价格中体现，不再扣管理费。</p>
{table(['标的','与供应商复权参考差异最大的日子','最大日差（bp）','本项目总回报','供应商复权收盘变动'],comparison_rows)}
<p>主计算使用 P1 当时份额价格与公司行动，provider_adjusted_close 只用于交叉核对。供应商复权方法、事件金额精度及修订可产生差异，全部保留在 returns CSV；不强行令两条方法相等或覆盖源值。
P1 的来源差异和历史发布时间限制仍然适用，详见 <a href='data.html'>数据报告</a>。</p>
<p>2020-03-20 的 SPY 方法诊断：以 C_t/(C_前日−分红)−1 重算，与供应商复权参考只差约 0.000988bp；
主算法仍为 (C_t+分红)/C_前日−1，即除息日收盘现金再投资的总回报。这是该日差异的可复算解释，不声称恢复了供应商全部历史算法。
<a href='../runs/p2/return_reference_diagnostic.json'>原值与两种算术</a>。</p>
<h2>真实交易时钟与五日标签</h2><p>信息 t → t+1 收盘前 60 分钟决策 → t+1 收盘执行；R5 从 t+1 收盘到 t+6 收盘。
L5 是相对入场点的未来每日收盘最差损失，下限 0；不取盘中最低价或期间峰谷回撤。Y10 表示这种损失达到 10%，不是模型概率。</p>
{table(['信息日','入场收盘日','五日终点','R5','L5','Y10','结果发生/成熟时间 UTC','假设可用时间 UTC'],label_rows)}
<p>已取得完整标签 {result['labels_observed']} 行，尚不完整 {result['labels_pending']} 行；最后一个完整标签的信息日为 {result['last_observed_information_session']}。
尾部标签保持空，未来日期只来自交易日历，不填造收益。成熟指结果已发生；历史 published_at 不明，所以另存保守的次交易日截止可用时间。训练只能读取已成熟且已到假设可用时间的标签，本阶段没有训练。</p>
{table(['信息日','下一交易日','决策时间（纽约）','执行收盘 UTC'],clock_rows)}
<p>日历使用 exchange_calendars {result['calendar_version']}，逐日核对 P1 原排程；夏令时、周末、节假日和半日市按实际股票交易日处理。
缺失信号时保留已有股数和现金，继续计持仓损益；持有价格缺失则报错，不跳过该日制造净值。</p>
<h2>真实账本对账与普通测试</h2>
{table(['对照','实际结果','资产等式最大误差 $','现金等式最大误差 $','成本等式最大误差 $'],audits)}
<p>零成本全仓账户与从首次执行收盘起的总回报路径独立对照，最大金额差为 {result['zero_cost_full_holding_vs_total_return_max_error_dollars']:.3g} 美元。
pytest 的合成夹具覆盖拆分、分红、执行先后、实际换手、未来扰动、缺信号仍记损益、标签成熟边界及半日市；夹具不写入真实行情或净值。</p>
<details><summary>展开本次 {len(data['test_cases'])} 项普通 pytest 的逐项结果</summary>{tests}</details>
<h2>复算文件、命令和状态</h2><p>{links}</p>
<p>唯一入口：<code>.venv/bin/python -m svxylab baselines --open</code>；独立测试：<code>.venv/bin/python -m pytest -q</code>。</p>
{commands}<p>Python {escape(data['python'])}；平台 {escape(data['platform'])}。</p>
<details><summary>实际依赖版本</summary><pre>{escape(json.dumps(data['packages'],ensure_ascii=False,indent=2))}</pre></details>
<p><a href='../{data['run_record']}'>本次运行记录与数据/代码/结果摘要</a> · <a href='../runs/p2/experiment_record.json'>计算前记录的实现口径</a> · <a href='../runs/p2/dependency_install.json'>实际依赖安装记录</a> · <a href='../STATUS.md'>STATUS.md</a></p>
<p>工程完成不代表预测或经济研究完成。仍未实现 P3 特征、联合预测或日常交易系统。停在 P2，等待用户验收。</p></main></body></html>"""
    report.write_text(document)


def build_baselines_report(root: Path, *, open_report: bool = False) -> int:
    root = root.resolve()
    if Path(sys.prefix).resolve() != (root / ".venv").resolve():
        raise ValueError("请使用项目解释器 .venv/bin/python")
    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = root / "runs/p2" / stamp
    run_dir.mkdir(parents=True)
    output = root / "data/clean/p2" / stamp
    try:
        result = prepare_baselines(root, output)
        plot_ledgers(root, output)
        examples = hand_examples(output)
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
    paths += [root / p for p in ("experiment.toml", "FEATURES.json", "RESEARCH_SPEC.md", "pyproject.toml", "requirements-lock.txt", "runs/p2/experiment_record.json")]
    data = {"stage": "P2", "generated_at": now.astimezone().isoformat(), "result": result,
            "invocation": ".venv/bin/python -m svxylab baselines" + (" --open" if open_report else ""),
            "engineering_complete": complete, "examples": examples,
            "tests_passed": sum(c["result"] == "通过" for c in cases), "tests_failed": sum(c["result"] == "失败" for c in cases),
            "test_cases": cases, "commands": commands, "python": sys.version, "platform": platform.platform(),
            "packages": {d.metadata["Name"]: d.version for d in sorted(metadata.distributions(), key=lambda d:d.metadata["Name"].lower())},
            "source_sha256": {p.relative_to(root).as_posix(): sha256(p.read_bytes()).hexdigest() for p in paths},
            "artifact_sha256": {p.relative_to(root).as_posix(): sha256(p.read_bytes()).hexdigest() for p in sorted(output.iterdir())},
            "training_cutoff": None, "predictions": None, "model_fitted": False,
            "run_record": (run_dir / "baselines_run.json").relative_to(root).as_posix()}
    def save():
        (root / data["run_record"]).write_text(json.dumps(data,ensure_ascii=False,indent=2) + "\n")
        render(root, data)
    save()
    if open_report:
        opened = run_command(["/usr/bin/open", str(root / "reports/baselines.html")], root)
        commands.append(opened)
        complete = data["engineering_complete"] = complete and opened["exit_code"] == 0
        save()
    print(f"P2 报告：{root / 'reports/baselines.html'}\n普通 pytest：{data['tests_passed']} 通过，{data['tests_failed']} 失败\n"
          f"真实账本：6 × {len(result['sessions'])} 行；标签 {result['labels_observed']} 完整，{result['labels_pending']} 未成熟。")
    return 0 if complete else 1
