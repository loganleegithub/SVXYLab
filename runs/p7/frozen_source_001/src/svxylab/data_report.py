"""P1 静态数据核对报告与可复算运行记录。"""

from contextlib import redirect_stdout
from datetime import datetime, timezone
from hashlib import sha256
from html import escape
from importlib import metadata
import io
import json
from pathlib import Path
import platform
import sys
import traceback
import xml.etree.ElementTree as ET

import pandas as pd
import numpy as np

from svxylab.data import prepare_data
from svxylab.downloads import DownloadStore
from svxylab.environment import run_command


def table(headers, rows):
    head = "".join(f"<th>{escape(str(x))}</th>" for x in headers)
    body = "".join("<tr>" + "".join(f"<td>{escape(str(x))}</td>" for x in row) + "</tr>" for row in rows)
    return f"<div class='scroll'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def number(value):
    return "尚未取得" if value is None or pd.isna(value) else f"{float(value):.6g}"


def spot_checks(root):
    frames = {}
    for symbol in ("VIX", "VVIX", "VIX9D", "VIX3M", "SVXY", "SPY"):
        path = root / f"data/clean/{symbol}_daily.csv"
        if path.exists():
            frames[symbol] = pd.read_csv(path).set_index("as_of_session")
    curve = pd.read_csv(root / "data/clean/VX_front_three.csv").set_index("as_of_session")
    rows = []
    for date, context in (("2019-07-01", "普通时点"), ("2020-03-16", "2020 年压力时点"), ("2025-04-04", "2025 年压力时点")):
        row = [date, context]
        row += [number(frames[s].loc[date, "value"]) if s in frames and date in frames[s].index else "尚未取得"
                for s in ("VIX", "VVIX", "VIX9D", "VIX3M")]
        row += [number(curve.loc[date, f"f{i}_settle"]) for i in (1, 2, 3)]
        row += [number(frames[s].loc[date, "close"]) if s in frames and date in frames[s].index else "尚未取得" for s in ("SVXY", "SPY")]
        rows.append(row)
    roll = [[date] + [str(curve.loc[date, f"f{i}_expiration"]) + " / " + number(curve.loc[date, f"f{i}_settle"])
                      for i in (1, 2, 3)] for date in ("2019-03-18", "2019-03-19")]
    company = {"verified": False, "reason": "正式 ETF 日线 / 公司行动尚未取得"}
    if "SVXY" in frames and "2024-04-11" in frames["SVXY"].index:
        previous, current = frames["SVXY"].loc["2024-04-10"], frames["SVXY"].loc["2024-04-11"]
        factor = float(current.split_factor)
        adjusted_return = current.provider_adjusted_close / previous.provider_adjusted_close - 1
        raw_corrected = current.close / factor / previous.close - 1
        company = {"verified": factor == 0.5, "date": "2024-04-11", "official_ratio": "2:1",
                   "previous_raw_close": float(previous.close), "raw_close": float(current.close),
                   "split_factor": factor, "split_corrected_change": float(raw_corrected),
                   "provider_adjusted_change": float(adjusted_return),
                   "difference_bps": float((raw_corrected - adjusted_return) * 10000),
                   "reason": "与发行人公告日期及 2:1 拆分核对；这是一日数据检查，不是策略回测"}
    return {"spots": rows, "roll": roll, "company_action": company}


def audit_etf_sources(root, downloads):
    """参考源缺失与价差原样报告；不覆盖主源，不用一致性阈值挑价格。"""
    results, comparisons = {}, []
    for symbol in ("SVXY", "SPY"):
        record = next((r for r in reversed(downloads) if r["label"] == symbol + "_history_candidate" and r["source"] == "nasdaq"), None)
        path = root / f"data/clean/{symbol}_daily.csv"
        if not record or not path.exists():
            continue
        raw = json.loads((root / record["raw_file"]).read_text())["data"]
        if raw["symbol"] != symbol:
            raise ValueError("Nasdaq 参考原件标识错误")
        other = pd.DataFrame(raw["tradesTable"]["rows"])
        other["as_of_session"] = pd.to_datetime(other.date, format="%m/%d/%Y").dt.strftime("%Y-%m-%d")
        missing = other.loc[other.eq("N/A").any(axis=1), "as_of_session"].tolist()
        for c in ("open", "high", "low", "close", "volume"):
            other[c] = pd.to_numeric(other[c].str.replace(",", "", regex=False).replace("N/A", np.nan), errors="raise")
        joined = pd.read_csv(path).merge(other, on="as_of_session", suffixes=("_then", "_nasdaq"), validate="one_to_one")
        differences = {c: float((joined["provider_" + c] - joined[c + "_nasdaq"]).abs().max()) for c in ("open", "high", "low", "close", "volume")}
        results[symbol] = {"common_rows": len(joined), "nasdaq_missing_field_dates": missing,
                           "max_absolute_difference_split_adjusted_units": differences,
                           "nasdaq_raw_file": record["raw_file"], "nasdaq_sha256": record["sha256"]}
        for _, row in joined.iterrows():
            comparison = {"symbol": symbol, "as_of_session": row.as_of_session}
            for field in ("open", "high", "low", "close", "volume"):
                comparison["yahoo_split_adjusted_" + field] = row["provider_" + field]
                comparison["nasdaq_split_adjusted_" + field] = row[field + "_nasdaq"]
                comparison[field + "_difference"] = row["provider_" + field] - row[field + "_nasdaq"]
            comparisons.append(comparison)
    pd.DataFrame(comparisons).to_csv(root / "data/clean/ETF_source_comparison.csv", index=False)
    observation = root / "runs/p1/quantconnect_observation.json"
    qc = json.loads(observation.read_text()) if observation.exists() else {}
    qc_rows = []
    for symbol, spots in qc.get("spot_closes", {}).items():
        path = root / f"data/clean/{symbol}_daily.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path).set_index("as_of_session")
        for date, value in spots.items():
            if date not in frame.index:
                continue
            close = float(frame.loc[date, "close"])
            qc_rows.append([symbol, date, number(close), value, number(close - value)])
    result = {"nasdaq": results, "qc_spot_comparison": qc_rows, "quantconnect_observation": qc}
    (root / "runs/p1/source_comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def render(data, report):
    summary = data["data"]
    coverage_rows, gaps = [], []
    for name, c in summary["coverage"].items():
        coverage_rows.append([name, c["first"] or "尚未取得", c["last"] or "尚未取得", c["expected"], c["observed"], c["missing_count"], len(c["extra_sessions"])])
        if c["missing_sessions"]:
            dates = "、".join(c["missing_sessions"]) if c["observed"] else "整个请求区间尚未取得"
            gaps.append(f"<details><summary>{escape(name)}：缺 {c['missing_count']} 日</summary><p>{escape(dates)}</p></details>")
        if c["extra_sessions"]:
            gaps.append(f"<details><summary>{escape(name)}：{len(c['extra_sessions'])} 个非股票交易日记录</summary><p>{escape('、'.join(c['extra_sessions']))}</p></details>")
    network = []
    for r in data["downloads"]:
        result = r["error_layer"] or ("本机原件导入，未声称本次 HTTP 下载" if r.get("method") == "LOCAL_IMPORT" else "HTTP 正文已保存")
        if r["source"] == "alphavantage" and "daily" in r["label"].lower():
            result = "HTTP 200 仅返回 demo 授权提示；没有日线数据"
        network.append([r["source"] + "/" + r["label"], r["http_status"] or "—", r["bytes"], r["retrieved_at"], result])
    commands = "".join(f"<details><summary>退出码 {r['exit_code']} · <code>{escape(r['command'])}</code></summary>"
                        f"<p>{escape(r['started_at'])}</p><pre>{escape(r['output'] or '（无输出）')}</pre></details>" for r in data["commands"])
    checks = data["checks"]
    comparison = data["source_comparison"]
    company = checks["company_action"]
    company_labels = {"verified": "拆分日期与比例已核对", "date": "生效日期", "official_ratio": "发行人拆分比例",
                      "previous_raw_close": "前日收盘（复原到当时美元/份）", "raw_close": "当日收盘（当时美元/份）",
                      "split_factor": "拆分价格因子", "split_corrected_change": "份额变化校正后的一日变动（小数）",
                      "provider_adjusted_change": "供应商复权收盘的一日变动（小数）", "difference_bps": "两种算法差（bp）", "reason": "核对范围"}
    company_html = table(["字段", "实际核对"], [(company_labels.get(k, k), v) for k, v in company.items()])
    source_links = [
        ("Cboe 官方指数入口", "https://www.cboe.com/tradable-products/vix/vix-historical-data/"),
        ("Cboe 官方 VX 合约目录", "https://www-api.cboe.com/us/futures/market_statistics/historical_data/product/list/VX/"),
        ("ProShares 2024 年拆分公告", "https://www.proshares.com/press-releases/proshares-announces-etf-share-splits2"),
        ("QuantConnect 曾尝试的论坛导出流程（本次未成功）", "https://www.quantconnect.com/forum/discussion/19781"),
        ("US Equities 数据说明", "https://www.quantconnect.com/datasets/algoseek-us-equities"),
        ("拆分与分红数据说明", "https://www.quantconnect.com/datasets/quantconnect-us-equity-security-master"),
        ("Yahoo 复权字段说明", "https://in.help.yahoo.com/kb/adjusted-close-sln28256.html"),
        ("yfinance 数据请求选项", "https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html"),
    ]
    links = " · ".join(f"<a href='{escape(url, quote=True)}'>{escape(label)}</a>" for label, url in source_links)
    clean_links = " · ".join(f"<a href='../{escape(p, quote=True)}'>{escape(Path(p).name)}</a>" for p in data["clean_sha256"])
    core_count = len(summary["core_common_sessions"])
    ready = "核心完整覆盖" if summary["full_core_data_ready"] else "核心覆盖未完整"
    engineering = "已完成本次取得、核验和报告流程" if data["engineering_complete"] else "未完成，见失败命令"
    document = f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width, initial-scale=1'><title>SVXYLab · P1 真实数据核对</title>
<style>body{{margin:0;background:#fff;color:#1e2b36;font:16px/1.7 -apple-system,BlinkMacSystemFont,'PingFang SC',sans-serif}}
main{{max-width:1160px;margin:auto;padding:32px 24px 64px}}h1{{font-size:28px}}h2{{font-size:21px;margin-top:32px}}a{{color:#165b8a}}
.status{{background:#f2f5f7;border-left:4px solid #526d82;padding:12px 20px}}.scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:14px}}
td,th{{border:1px solid #dce2e7;padding:9px;text-align:left;vertical-align:top;overflow-wrap:anywhere}}th{{background:#f3f6f8}}
code,pre{{font:13px/1.6 ui-monospace,monospace}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f6f7f8;padding:12px}}
details{{border-bottom:1px solid #ddd;padding:10px 0}}summary{{cursor:pointer;overflow-wrap:anywhere}}</style></head><body><main>
<h1>SVXYLab · P1 真实数据核对</h1><p>生成：{escape(data['generated_at'])} · 数据请求：{summary['requested_start']} 至 {summary['requested_end']}</p>
<div class='status'><p><strong>工程：</strong>{engineering}。{data['tests']['total']} 项普通 pytest，失败 {data['tests']['failures']}，错误 {data['tests']['errors']}。</p>
<p><strong>真实数据：</strong>{ready}；核心共同样本 {core_count} / {len(summary['sessions'])} 个交易日，实际共同截至日 {summary['core_frozen_cutoff'] or '尚未取得'}。</p>
<p><strong>研究结果：</strong>尚未训练、预测或回测；数据齐全和测试通过均不代表有收益优势。</p></div>
<h2>覆盖与冻结范围</h2>{table(['数据集','实际首日','实际末日','应有日数','实有日数','缺失日数','非股票交易日记录'],coverage_rows)}
<p>应有日数采用 exchange_calendars {summary['calendar_version']} 的 XNYS 日历，包含实际半日市。起点 2019-01-01 是假日，首个应有日为 2019-01-02。
VIX 额外日期保留在原件和日值表，股票研究以交易日历连接；不将额外记录凑成更多样本。SKEW 不参与核心共同样本。</p>
<p>Cboe 四指数与 F1/F2/F3 的共同样本：{len(summary['public_common_sessions'])} 日。月度 VX 合约 {summary['monthly_contracts']} 份，区间内逐合约结算记录 {summary['vx_rows']} 行；逐合约行数不能当作研究样本日数。</p>
{''.join(gaps)}<p><a href='../runs/p1/freeze.json'>冻结日期</a> · <a href='../runs/p1/data_summary.json'>完整覆盖与缺失日期 JSON</a> · <a href='../runs/p1/experiment_change.json'>2019 年起点变更记录</a></p>
<h2>三个真实时点抽查</h2>{table(['日期','背景','VIX','VVIX','VIX9D','VIX3M','VX F1结算','VX F2结算','VX F3结算','SVXY 当时份额收盘','SPY 当时份额收盘'],checks['spots'])}
<p>表中均为取得的真实日值；指数及 VX 单位为点，ETF 为美元/份。日期仅用于核对数据，不用于选择模型或展示预测成功。</p>
<h2>真实换月边界</h2>{table(['信息日收盘','F1 到期日 / 结算','F2 到期日 / 结算','F3 到期日 / 结算'],checks['roll'])}
<p>官方目录标明 2019 年 3 月合约于 3 月 19 日（星期二）到期。当日收盘已到期合约退出，F1 必须是 4 月 17 日合约。
选择只接受 VX、标准月度 M、到期日严格大于信息日；读取 Settle，保留 Close 原字段。缺失应有近月会记缺失，不顺移名称。</p>
<h2>真实公司行动</h2><p>发行人公告将 SVXY 列在第二批：2:1 拆分于 2024-04-11 开盘前生效；不可误用第一批的 4 月 10 日。{links}</p>{company_html}
<p>split_factor 是份额倍数的倒数，0.5 表示一份变两份。Yahoo quote 中 OHLCV 已按拆分调整，即使 auto_adjust=False 也如此。
provider_open/high/low/close/volume 保留供应商原值；open/high/low/close 乘之后已报告的拆分份额倍数，volume 除该倍数，复原到当时份额单位。
这些列明确标记为拆分复原值，不冒充供应商直接给出的未复权逐笔记录。分红、资本利得分配和 provider_adjusted_close 分列保留。没有自动修价或填补缺行。</p>
<p>例如 4 月 10 日供应商 Close 为约 54.585，乘 2 得约 109.17；4 月 11 日约 55.08 已是拆分后的当时价格。完整总回报与持仓账本属于 P2，本阶段未计算整段策略净值。</p>
<h2>交叉来源差异</h2><p>正式 ETF 原件统一采用 Yahoo Chart（yfinance 1.7.0，auto_adjust=False、back_adjust=False、repair=False、actions=True、keepna=True）。
Nasdaq 两份参考日线也保存到本地，但其非 Nasdaq 挂牌 ETF 分红接口明确返回不可用；QuantConnect 云端实际取得两只 ETF 各 1930 根 RAW / ADJUSTED 日线，SVXY 一次生效拆分、SPY 30 次分红。
研究容器文件的两条标准链接都被编辑器报为文件不存在，未取得该完整云端文件到本地，未将云端行数冒充本地原件。</p>
{table(['标的','日期','主源当时份额收盘','QC RAW 浏览器抽查','主源减 QC（美元）'],comparison['qc_spot_comparison'])}
<p>QC 抽查属于浏览器可见数值记录。2020-03-16 的 SVXY 主源约 28.23，QC 为 28.27，差约 -0.04 美元；原差异保留，没有按结果择价。
Nasdaq 的 2026-04-20 成交量为 N/A 且 OHLC 相同；主源当日有非零成交量和完整 OHLC。其余价格、成交量差异逐日保留在比较 CSV，不以“完全一致”掩盖来源差异与供应商精度。
QC 在拆分前后的成交量与主源同样有差异：2024-04-10 的 SVXY 主源复原为 1,898,100 份，QC RAW 为 1,813,029 份；4 月 11 日分别为 956,800 与 887,384 份。没有把成交量差异解释成完全可由拆分消除。</p>
<details><summary>参考源逐字段差异摘要</summary><pre>{escape(json.dumps(comparison['nasdaq'],ensure_ascii=False,indent=2))}</pre></details>
<p><a href='../runs/p1/source_decision.json'>本轮来源选择及口径记录</a> · <a href='../runs/p1/quantconnect_observation.json'>QC 云端有限核对记录</a> · <a href='../runs/p1/source_comparison.json'>差异 JSON</a></p>
<h2>原件、来源和时间边界</h2>
<p>本机 /Users/logan/DATA/VIX 中可用的月度 VX 原件先核对既有清单 SHA-256，再对照本次官方目录；2019-03-19、2020-03-18、2026-07-22 三份与本次官方下载逐字节一致。
本机已有的 Yahoo SVXY JSON 仅为参考，缺少明确分红字段，未冒充完整正式 ETF 数据。没有复用旧模型、因子、预测或回测结果。</p>
<p>原始获取时间无法由旧文件名证明，导入记录的 original_downloaded_at 保持空；retrieved_at 表示本项目收到该文件的时间。
历史真实 published_at 未取得，保持空。所有历史日线只声明“假设次股票交易日收盘前 60 分钟可得”，不宣称完整历史 PIT。
CFE 结算与隐含指数收盘不完全同步，价格类型保留。</p>
<p>原件 SHA-256 核对：{len(data['downloads'])} 条保存记录，摘要不符 {len(data['raw_integrity_errors'])} 条。
<a href='../data/raw/downloads.jsonl'>追加式下载 / 导入清单</a>含来源 URL、收到时间、响应状态、摘要及原始文件位置；失败正文同样保留。</p>
<p>本次按用户明确的个人非商业研究需求使用公开 Cboe 文件、Yahoo API 和已有数据；QC 使用现有免费账户的云端 History。
yfinance 是第三方开源客户端，其文档将 Yahoo API 用途限定为个人研究，不能由此推定其他分发权。没有购买套餐或取得对外分发数据的授权，行情不上传 Git 远端。</p>
<h2>实际网络与取数响应</h2><p>每个 HTTP 请求最多 2 次尝试；仅网络错误或 429/500/502/503/504 重试一次。每次连接 8 秒、总时限 30 秒、TLS 校验开启。
早期 Yahoo curl 请求 HTTP 429，后来 yfinance 客户端返回完整原件；Nasdaq 原生 curl 出现 HTTP/2 错误，curl_cffi 请求成功，但历史小窗口曾返回零行、分红字段不可用。
Alpha Vantage demo 返回授权提示。HTTP 200 与数据验证成功分开记录。yfinance 每次最多 8 个包括会话初始化的请求，每标的最多 2 个 chart 请求，每请求 12 秒；无批量取数线程。重复运行优先使用摘要验证过的缓存。</p>
<details><summary>展开全部保存记录（{len(network)} 条）</summary>{table(['来源 / 文件','HTTP','字节','本项目收到时间 UTC','实际结果'],network)}</details>
<h2>复算文件与实际命令</h2><p>数据表：{clean_links}</p><p>唯一入口：<code>.venv/bin/python -m svxylab data --open</code>；独立测试：<code>.venv/bin/python -m pytest -q</code>。</p>
<p>本次包版本：{escape(', '.join(k+' '+v for k,v in data['packages'].items()))}。Python {escape(data['python'])}；平台 {escape(data['platform'])}。</p>
{commands}<p><a href='../{data['run_record']}'>本次运行 JSON（代码、配置、原件与清洗文件摘要）</a> · <a href='../runs/p1/reproducibility.json'>缓存复算核验</a> · <a href='../runs/p1/development_failures.json'>开发及连接失败记录</a> · <a href='../runs/p1/yfinance_install_cached.json'>客户端安装命令及结果</a> · <a href='../runs/p1/yfinance_install_failure.json'>保留的安装失败</a> · <a href='../STATUS.md'>STATUS.md</a></p>
<p>合成夹具仅用于 pytest 算术和解析边界；不进入上述真实数据表。P0 的 100→90→97.5 仍只是合成复利运算。</p>
<h2>尚缺什么</h2><p>{escape(summary['summary'])} 最重要的未解决项：{escape(data['blocker'])}。</p>
<pre>{escape(json.dumps(summary['failures'],ensure_ascii=False,indent=2))}</pre><p>停在 P1，等待阶段验收；未进入 P2。</p>
</main></body></html>"""
    report.write_text(document)


def build_data_report(root: Path, *, open_report: bool = False) -> int:
    root = root.resolve()
    if Path(sys.prefix).resolve() != (root / ".venv").resolve():
        raise ValueError("请使用项目解释器 .venv/bin/python")
    now = datetime.now(timezone.utc)
    run_dir = root / "runs/p1" / now.strftime("%Y%m%dT%H%M%S%fZ")
    run_dir.mkdir(parents=True)
    commands = []
    output = io.StringIO()
    with redirect_stdout(output):
        try:
            summary = prepare_data(root)
        except Exception:
            (run_dir / "failure.txt").write_text(traceback.format_exc())
            raise
    commands.append({"command": ".venv/bin/python -m svxylab data" + (" --open" if open_report else ""),
                     "started_at": now.isoformat(), "exit_code": 0, "output": output.getvalue()})
    for args in ([sys.executable, "-m", "pip", "--disable-pip-version-check", "check"],
                 [sys.executable, "-m", "pytest", "-q", "--junitxml", str(run_dir / "pytest.xml")],
                 ["git", "diff", "--check"], ["git", "remote"], ["git", "rev-parse", "HEAD"]):
        commands.append(run_command(args, root, timeout=60))
    tests = {"total": 0, "failures": 0, "errors": 0, "skipped": 0}
    if (run_dir / "pytest.xml").exists():
        for suite in ET.parse(run_dir / "pytest.xml").getroot().iter("testsuite"):
            for key, attr in (("total", "tests"), ("failures", "failures"), ("errors", "errors"), ("skipped", "skipped")):
                tests[key] += int(suite.get(attr, "0"))
    downloads = DownloadStore(root).records
    integrity = [r["raw_file"] for r in downloads if not (root / r["raw_file"]).is_file()
                 or sha256((root / r["raw_file"]).read_bytes()).hexdigest() != r["sha256"]]
    sources = sorted((root / "src").rglob("*.py")) + sorted((root / "tests").rglob("*.py"))
    sources += [root / p for p in ("experiment.toml", "FEATURES.json", "RESEARCH_SPEC.md", "pyproject.toml", "requirements-lock.txt",
                                  "runs/p1/source_decision.json", "runs/p1/quantconnect_observation.json") if (root / p).exists()]
    sources += sorted((root / "runs/p1").glob("*.py"))
    complete = all(c["exit_code"] == 0 for c in commands) and bool(tests["total"]) and not integrity
    comparisons = audit_etf_sources(root, downloads)
    data = {"stage": "P1", "generated_at": now.astimezone().isoformat(), "data": summary,
            "engineering_complete": complete, "tests": tests, "commands": commands, "downloads": downloads,
            "raw_integrity_errors": integrity, "checks": spot_checks(root), "source_comparison": comparisons, "python": sys.version,
            "platform": platform.platform(), "training_cutoff": None, "predictions": None, "portfolio_ledger": None,
            "packages": {d.metadata["Name"]: d.version for d in sorted(metadata.distributions(), key=lambda d:d.metadata["Name"].lower())},
            "source_sha256": {p.relative_to(root).as_posix(): sha256(p.read_bytes()).hexdigest() for p in sources},
            "clean_sha256": {p.relative_to(root).as_posix(): sha256(p.read_bytes()).hexdigest() for p in sorted((root / "data/clean").glob("*.csv"))},
            "run_record": (run_dir / "data_run.json").relative_to(root).as_posix(),
            "blocker": "无。仍保留来源价差、拆分复原与发布时间假设；不能据此声明模型有效" if summary["full_core_data_ready"] else "SVXY / SPY 正式完整 OHLCV、拆分分红及复权参考尚未取得"}
    report = root / "reports/data.html"
    report.parent.mkdir(exist_ok=True)
    def save():
        (root / data["run_record"]).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
        render(data, report)
    save()
    if open_report:
        opened = run_command(["/usr/bin/open", str(report)], root)
        commands.append(opened)
        complete = data["engineering_complete"] = complete and opened["exit_code"] == 0
        save()
    print(f"数据报告：{report}\npytest：{tests['total']} 项；工程流程：{'完成' if complete else '未完成'}\n{summary['summary']}")
    return 0 if complete else 1
