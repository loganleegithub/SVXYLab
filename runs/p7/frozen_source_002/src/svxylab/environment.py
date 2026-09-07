"""P0 环境检查与静态 HTML 报告；只使用 Python 标准库和本机命令。"""

from datetime import datetime, timezone
from hashlib import sha256
from html import escape
from importlib import metadata
import json
import platform
from pathlib import Path
import shlex
import ssl
import subprocess
import sys
import time
import tomllib
import xml.etree.ElementTree as ET


def run_command(argv: list[str], root: Path, timeout: int = 30) -> dict:
    started = datetime.now(timezone.utc).isoformat()
    before = time.monotonic()
    try:
        result = subprocess.run(argv, cwd=root, capture_output=True, text=True, timeout=timeout)
        code, output = result.returncode, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        code, output = 124, f"超过 {timeout} 秒上限；子进程已终止。"
    except OSError as error:
        code, output = 127, str(error)
    return {"command": shlex.join(argv), "started_at": started, "exit_code": code,
            "elapsed_seconds": round(time.monotonic() - before, 3), "output": output}


def check_network(url: str, root: Path) -> dict:
    record = run_command([
        "/usr/bin/curl", "--disable", "--head", "--location", "--max-redirs", "3",
        "--connect-timeout", "5", "--max-time", "15", "--retry", "0",
        "--silent", "--show-error", "--output", "/dev/null", "--write-out",
        "http=%{http_code}\\ndns_seconds=%{time_namelookup}\\n"
        "total_seconds=%{time_total}\\nssl_verify=%{ssl_verify_result}\\nurl=%{url_effective}\\n",
        "--url", url,
    ], root, timeout=20)
    fields = dict(line.split("=", 1) for line in record["output"].splitlines() if "=" in line)
    code = record["exit_code"]
    http = fields.get("http", "000")
    if code:
        result = {5: "代理 DNS 解析失败", 6: "DNS 解析失败", 7: "连接失败",
                  28: "连接或响应超时", 35: "TLS 握手失败", 60: "TLS 证书校验失败",
                  124: "命令超时", 127: "本机命令不可执行"}.get(code, f"curl 失败（{code}）")
    elif http.startswith("2"):
        result = f"HTTP {http}：本次 HEAD 请求成功"
    else:
        result = f"HTTP {http}：已收到服务器响应，端点未返回成功状态"
    return {"url": url, "result": result, "fields": fields, "execution": record}


def render_report(data: dict, report: Path, record_path: Path) -> None:
    def table(headers, rows):
        headings = "".join(f"<th scope='col'>{escape(str(h))}</th>" for h in headers)
        body = "".join("<tr>" + "".join(f"<td>{escape(str(c))}</td>" for c in row) + "</tr>" for row in rows)
        return f"<div class='table'><table><thead><tr>{headings}</tr></thead><tbody>{body}</tbody></table></div>"

    commands = "".join(
        f"<details><summary>退出码 {r['exit_code']} · <code>{escape(r['command'])}</code></summary>"
        f"<p>开始时间（UTC）：{escape(r['started_at'])}</p><pre>{escape(r['output'] or '（无输出）')}</pre></details>"
        for r in data["setup_commands"] + data["commands"]
    )
    engineering = "本地环境检查已完成；当前阶段与验收状态见 STATUS.md。" if data["engineering_checks_complete"] else "本地环境检查未完成；见下方命令的失败结果。"
    software = table(["软件 / 环境", "实际观测"], data["software"].items())
    packages = table(["项目虚拟环境中的包", "实际版本"], data["packages"].items())
    network = table(["检查地址", "实际结果"], [(n["url"], n["result"]) for n in data["network"]])
    missing = table(["项目", "当前状态"], [
        ("P0 环境 / 包 / 测试", "完成" if data["engineering_checks_complete"] else "未完成，见命令结果"),
        ("真实 VX、VIX、VVIX、VIX9D、VIX3M、SVXY / SPY 行情", data["real_market_data"]),
        ("SKEW 扩展数据", data.get("skew_status", "尚未取得；不阻塞核心研究")),
        ("特征、训练、持仓回测与模型有效性", "尚未实现 / 尚未研究；属于后续阶段"),
        ("P0 真正技术阻塞项", "无" if data["engineering_checks_complete"] else "本地工程检查尚未通过"),
        ("本地 Git 检查点", data["software"]["Git 提交"]),
    ])
    raw_counts = data["data_files"]
    tests = data["tests"]
    record_link = "../" + record_path.relative_to(report.parent.parent).as_posix()
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>SVXYLab · P0 环境检查</title>
<style>
body{{font:16px/1.7 -apple-system,BlinkMacSystemFont,'PingFang SC',sans-serif;color:#17212b;background:#fff;margin:0}}
main{{max-width:1000px;margin:auto;padding:32px 24px 64px}}h1{{font-size:28px}}h2{{font-size:21px;margin-top:32px}}
.status{{border-left:4px solid #426078;padding:12px 20px;background:#f3f6f8}}.table{{overflow-x:auto}}
table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{text-align:left;vertical-align:top;padding:10px;border:1px solid #dce2e6;overflow-wrap:anywhere}}
th{{background:#f3f6f8}}code,pre{{font-family:ui-monospace,monospace;font-size:13px}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f7f8fa;padding:12px}}
summary{{cursor:pointer;overflow-wrap:anywhere}}details{{padding:10px 0;border-bottom:1px solid #dce2e6}}a{{color:#145b94}}
</style></head><body><main>
<h1>SVXYLab · P0 环境检查</h1>
<p>生成时间：{escape(data['generated_at_local'])} · 仅限本地环境与纯算术验证</p>
<div class="status"><p><strong>工程状态：</strong>{engineering}</p>
<p><strong>研究结果：</strong>尚未开展；本页没有真实回测结果。</p>
<p><strong>真实数据覆盖：</strong>{escape(data['real_market_data'])}。data/raw 文件数 {raw_counts['raw']}，data/clean 文件数 {raw_counts['clean']}（不计占位文件）。</p></div>
<h2>实际软件与环境</h2>{software}<p>复用本机已有 Python 3.12.13；项目虚拟环境隔离安装打包与测试依赖，运行逻辑使用标准库。</p>{packages}
<h2>本机网络检查</h2>{network}
<p>每个地址仅一次 HEAD 请求，无重试；连接上限 5 秒，总上限 15 秒，最多 3 次重定向。TLS 校验保持开启。结果仅代表本机当前网络路径。</p>
<p>CDN 根地址的 HTTP 403 是端点拒绝响应，不是 DNS 失败，也不能据此判断具体 CSV 文件的访问权限。入口 HTTP 200 不等于取得行情、验证字段或拥有数据授权。P0 未请求行情正文。</p>
<h2>实际测试结果</h2>
<p>pytest：共 {tests['total']} 项，失败 {tests['failures']}，错误 {tests['errors']}，跳过 {tests['skipped']}；进程退出码 {tests['exit_code']}。</p>
<p><strong>合成测试 / SYNTHETIC：</strong>夹具中的指数 100 → 120 → 100；每日 -0.5x 理想化复利应为 100 → 90 → 97.5。</p>
<p>首日指数 +20%，理想化产品 -10%；次日指数 -1/6，理想化产品 +1/12，90 × 13/12 = 97.5。此例只检查每日复利运算，不是历史行情、真实 SVXY 净值或真实回测。</p>
<h2>已运行命令与结果</h2><p>执行目录：<code>{escape(data['project_root'])}</code>。下方保留实际输出及退出码；Git 初始化前的“无仓库”和验收前的“无提交”是已记录的状态。</p>{commands}
<h2>缺少什么</h2>{missing}
<p>报告生成入口：<code>.venv/bin/python -m svxylab environment --open</code><br>独立测试：<code>.venv/bin/python -m pytest -q</code></p>
<p><a href="{escape(record_link, quote=True)}">本次环境 JSON 记录（含配置 / 代码 SHA-256）</a> · <a href="../STATUS.md">阶段进度</a></p>
</main></body></html>"""
    report.write_text(document, encoding="utf-8")


def build_environment_report(root: Path, *, open_report: bool = False) -> int:
    root = root.resolve()
    if not (root / "pyproject.toml").is_file() or not (root / "experiment.toml").is_file():
        raise ValueError("请在 SVXYLab 项目目录运行此命令")
    if Path(sys.prefix).resolve() != (root / ".venv").resolve() or sys.prefix == sys.base_prefix:
        raise ValueError("请使用项目解释器 .venv/bin/python")
    config = tomllib.loads((root / "experiment.toml").read_text(encoding="utf-8"))
    json.loads((root / "FEATURES.json").read_text(encoding="utf-8"))
    now = datetime.now(timezone.utc)
    run_dir = root / "runs" / "p0" / now.strftime("%Y%m%dT%H%M%S%fZ")
    run_dir.mkdir(parents=True)
    report = root / "reports" / "environment.html"
    report.parent.mkdir(exist_ok=True)
    commands = []

    def check(argv, timeout=30):
        record = run_command(argv, root, timeout)
        commands.append(record)
        return record

    machine = check(["/usr/bin/uname", "-m"])
    macos = check(["/usr/bin/sw_vers"])
    git = check(["git", "--version"])
    curl = check(["/usr/bin/curl", "--version"])
    repo = check(["git", "rev-parse", "--show-toplevel"])
    remotes = check(["git", "remote"])
    head = check(["git", "rev-parse", "--verify", "-q", "HEAD"])
    check(["git", "status", "--short"])
    pip_check = check([sys.executable, "-m", "pip", "--disable-pip-version-check", "check"])
    network = [check_network(url, root) for url in (
        "https://pypi.org/simple/pytest/", "https://files.pythonhosted.org/",
        "https://www.cboe.com/tradable-products/vix/vix-historical-data/", "https://cdn.cboe.com/",
    )]
    commands.extend(item["execution"] for item in network)
    junit = run_dir / "pytest.xml"
    pytest = check([sys.executable, "-m", "pytest", "-q", "--junitxml", str(junit)], timeout=60)
    tests = {"total": 0, "failures": 0, "errors": 0, "skipped": 0, "exit_code": pytest["exit_code"]}
    if junit.exists():
        for suite in ET.parse(junit).getroot().iter("testsuite"):
            for key, attribute in (("total", "tests"), ("failures", "failures"), ("errors", "errors"), ("skipped", "skipped")):
                tests[key] += int(suite.get(attribute, "0"))
    packages = {d.metadata["Name"]: d.version for d in sorted(metadata.distributions(), key=lambda d: d.metadata["Name"].lower())}
    source_paths = [root / name for name in ("RESEARCH_SPEC.md", "FEATURES.json", "experiment.toml", "pyproject.toml", "requirements-dev.txt")]
    source_paths += sorted((root / "src").rglob("*.py")) + sorted((root / "tests").rglob("*.py"))
    setup_path = root / "runs" / "p0" / "setup.json"
    market_status, skew_status = "尚未取得", "尚未取得；不阻塞核心研究"
    summary_path = root / "runs/p1/data_summary.json"
    if summary_path.exists():
        p1 = json.loads(summary_path.read_text())
        market_status = (f"最近一次 P1 记录（{p1['generated_at']}）：核心共同样本 {len(p1['core_common_sessions'])} 日，"
                         f"截至 {p1['core_frozen_cutoff'] or '尚未取得'}；本环境命令不重新核验行情，详见 reports/data.html")
        skew = p1["coverage"].get("SKEW")
        if skew:
            skew_status = f"最近 P1 记录 {skew['observed']} / {skew['expected']} 日，缺 {skew['missing_count']} 日；详见数据报告"
    complete = (pytest["exit_code"] == 0 and tests["total"] > 0 and pip_check["exit_code"] == 0
                and repo["exit_code"] == 0 and repo["output"].strip() == str(root)
                and remotes["exit_code"] == 0 and not remotes["output"].strip()
                and machine["exit_code"] == 0 and macos["exit_code"] == 0)
    data = {
        "stage": "P0", "generated_at_local": now.astimezone().isoformat(), "project_root": str(root),
        "invocation": shlex.join([sys.executable, "-m", "svxylab", *sys.argv[1:]]),
        "engineering_checks_complete": complete, "research_result": "尚未开展", "real_market_data": market_status,
        "skew_status": skew_status,
        "software": {"CPU 架构": machine["output"].strip(), "Python 进程架构": platform.machine(),
                     "macOS": macos["output"].strip(), "Python": sys.version, "项目解释器": sys.executable,
                     "复用的基础 Python": sys.base_prefix, "虚拟环境": sys.prefix,
                     "Python TLS": ssl.OPENSSL_VERSION, "Git": git["output"].strip(),
                     "curl": curl["output"].splitlines()[0], "Git 远端": remotes["output"].strip() or "无",
                     "Git 提交": head["output"].strip() if head["exit_code"] == 0 else "尚未提交；等待 P0 验收"},
        "packages": packages, "network": network, "tests": tests,
        "data_files": {kind: sum(1 for p in (root / "data" / kind).rglob("*") if p.is_file() and p.name != ".gitkeep") for kind in ("raw", "clean")},
        "config_status": config["status"], "training_cutoff": None, "predictions": None, "portfolio_ledger": None,
        "sha256": {p.relative_to(root).as_posix(): sha256(p.read_bytes()).hexdigest() for p in source_paths},
        "setup_commands": json.loads(setup_path.read_text(encoding="utf-8")) if setup_path.exists() else [],
        "commands": commands,
    }
    record_path = run_dir / "environment.json"

    def save():
        record_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        render_report(data, report, record_path)

    save()
    if open_report:
        opened = check(["/usr/bin/open", str(report)])
        data["open_exit_code"] = opened["exit_code"]
        if opened["exit_code"] != 0:
            complete = data["engineering_checks_complete"] = False
        save()
    print(f"环境报告：{report}\npytest：{tests['total']} 项，退出码 {tests['exit_code']}\n真实行情：{market_status}")
    return 0 if complete else 1
