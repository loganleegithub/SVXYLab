"""本轮真实数据审计：只用标准库逐标量复算，不调用生产特征函数。

日常入口仍为 python -m svxylab features；此脚本保存本轮独立算术证据。
"""

import csv
from datetime import date, datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUN = sorted((ROOT / "runs/p3").glob("*/features_run.json"))[-1]
RECORD = json.loads(RUN.read_text())
OUTPUT = ROOT / RECORD["result"]["output_dir"]


def rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def by_date(path):
    return {r["as_of_session"]: r for r in rows(path)}


sessions = [r["as_of_session"] for r in rows(ROOT / "data/clean/equity_sessions.csv")]
expirations = sorted(r["expire_date"] for r in rows(ROOT / "data/clean/VX_contract_calendar.csv") if r["duration_type"] == "M")
settles = {(r["contract_id"], r["as_of_session"]): float(r["value"]) for r in rows(ROOT / "data/clean/VX_contract_daily.csv")}
idx = {s: by_date(ROOT / f"data/clean/{s}_daily.csv") for s in ["VIX", "VVIX", "VIX9D", "VIX3M", "SKEW"]}
curve = by_date(ROOT / "data/clean/VX_front_three.csv")
tr = {}
log_returns = {}
for symbol in ["SPY", "SVXY"]:
    prices = by_date(ROOT / f"data/clean/{symbol}_daily.csv")
    wealth = [100.0]
    for i, day in enumerate(sessions[1:], 1):
        p, before = prices[day], prices[sessions[i-1]]
        gross = (float(p["close"]) + float(p["cash_dividend"]) + float(p["capital_gain_distribution"])) / (float(p["split_factor"]) * float(before["close"]))
        wealth.append(wealth[-1] * gross)
    tr[symbol] = wealth
    log_returns[symbol] = [None] + [math.log(wealth[i]/wealth[i-1]) for i in range(1, len(wealth))]

actual_core = by_date(OUTPUT / "core_features.csv")
actual_extension = by_date(OUTPUT / "extension_features.csv")
feature_ids = [c for c in actual_core[sessions[0]] if c != "as_of_session"] + ["G01", "G02"]
checks = {key: {"checked": 0, "finite_values": 0, "missing_values": 0, "max_absolute_error": 0.0} for key in feature_ids}
mismatches = []
front_selection_checks = 0


def index_value(symbol, i):
    if i < 0:
        return None
    raw = idx[symbol].get(sessions[i], {}).get("value")
    return float(raw) if raw else None


def log_change(now, previous):
    return math.log(now/previous) if now is not None and previous is not None else None


for i, day in enumerate(sessions):
    expiration = [d for d in expirations if d > day][:3]
    contracts = ["VX_" + d for d in expiration]
    F1, F2, F3 = [settles[c, day] for c in contracts]
    tau = [(date.fromisoformat(d) - date.fromisoformat(day)).days for d in expiration]
    for k, contract in enumerate(contracts, 1):
        assert curve[day][f"f{k}_contract"] == contract
        front_selection_checks += 1
    vix, vvix, v9, v3, skew = [index_value(s, i) for s in ["VIX", "VVIX", "VIX9D", "VIX3M", "SKEW"]]
    spy, svxy = tr["SPY"], tr["SVXY"]
    expected = {
        "A01": 30*math.log(F2/F1)/(tau[1]-tau[0]),
        "A02": 30*math.log(F3/F2)/(tau[2]-tau[1])-30*math.log(F2/F1)/(tau[1]-tau[0]),
        "A03": math.log(F1/vix), "A04": log_change(F1, settles.get((contracts[0], sessions[i-5]))) if i >= 5 else None,
        "A05": tau[0], "B01": math.log(vix), "B02": math.log(v9/vix), "B03": math.log(vix/v3),
        "C01": math.log(vvix), "C02": log_change(vvix, index_value("VVIX", i-5)),
        "C03": log_change(vix, index_value("VIX", i-5)),
        "D01": None, "D02": None, "D03": None, "E01": None,
        "F01": math.log(spy[i]/spy[i-5]) if i >= 5 else None,
        "F02": spy[i]/max(spy[i-20:i+1])-1 if i >= 20 else None,
        "F03": math.log(svxy[i]/svxy[i-5]) if i >= 5 else None, "F04": None,
        "G01": skew, "G02": skew-index_value("SKEW", i-5) if skew is not None and index_value("SKEW", i-5) is not None else None,
    }
    if i >= 5:
        expected["D01"] = math.sqrt(252*math.fsum(r*r for r in log_returns["SPY"][i-4:i+1])/5)
    if i >= 21:
        window = log_returns["SPY"][i-20:i+1]
        denom = math.fsum(r*r for r in window)
        expected["D02"] = math.sqrt(252*denom/21)
        expected["D03"] = math.fsum(r*r for r in window if r < 0)/denom if denom else None
        expected["E01"] = (vix/100)**2 - 252*denom/21
        expected["F04"] = math.sqrt(252*math.fsum(r*r for r in log_returns["SVXY"][i-20:i+1])/21)
    for key, value in expected.items():
        raw = (actual_core if key not in ["G01", "G02"] else actual_extension)[day][key]
        observed = float(raw) if raw else None
        item = checks[key]
        item["checked"] += 1
        if value is None:
            item["missing_values"] += 1
            ok = observed is None
        else:
            item["finite_values"] += 1
            ok = observed is not None and math.isclose(value, observed, rel_tol=1e-10, abs_tol=1e-11)
            if observed is not None:
                item["max_absolute_error"] = max(item["max_absolute_error"], abs(value-observed))
        if not ok:
            mismatches.append({"as_of_session": day, "feature": key, "expected": value, "observed": observed})

# 标准库 Pearson，使用生产矩阵声明的共同日期，直接核对全部 171 / 210 对。
correlation_checks = {}
for name, ids in [("core", feature_ids[:-2]), ("with_skew", feature_ids)]:
    records = [{**actual_core[d], **actual_extension[d]} for d in sessions if d <= "2023-12-31"]
    complete = [r for r in records if all(r[k] for k in ids)]
    values = {k: [float(r[k]) for r in complete] for k in ids}
    centers = {k: math.fsum(v)/len(v) for k, v in values.items()}
    centered = {k: [v-centers[k] for v in vs] for k, vs in values.items()}
    variance_sums = {k: math.fsum(v*v for v in vs) for k, vs in centered.items()}
    error = 0.0
    pairs = rows(OUTPUT / f"correlation_pairs_{name}_2019_2023.csv")
    for pair in pairs:
        a, b = pair["feature_1"], pair["feature_2"]
        expected = math.fsum(x*y for x,y in zip(centered[a],centered[b])) / math.sqrt(variance_sums[a]*variance_sums[b])
        error = max(error, abs(expected-float(pair["pearson"])))
        assert int(pair["common_rows"]) == len(complete)
    correlation_checks[name] = {"common_rows": len(complete), "pairs": len(pairs), "max_absolute_error": error, "passed": error < 1e-12}

result = {"audited_at": datetime.now(timezone.utc).isoformat(), "run_record": RUN.relative_to(ROOT).as_posix(),
          "audit_source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
          "method": "标准库逐标量：从 P1 当时份额价格/公司行动独立构造 TR；从官方月度到期日历重选 F1/F2/F3；未调用生产特征或总回报函数。",
          "sessions": len(sessions), "feature_cells_checked": sum(v["checked"] for v in checks.values()),
          "front_contract_selection_checks": front_selection_checks, "per_feature": checks,
          "correlation_checks": correlation_checks, "mismatch_count": len(mismatches), "mismatches": mismatches,
          "passed": not mismatches and all(v["passed"] for v in correlation_checks.values())}
destination = RUN.parent / "independent_recalculation.json"
destination.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
print(json.dumps({k: result[k] for k in ["passed", "sessions", "feature_cells_checked", "front_contract_selection_checks", "mismatch_count", "correlation_checks"]},ensure_ascii=False))
raise SystemExit(0 if result["passed"] else 1)
