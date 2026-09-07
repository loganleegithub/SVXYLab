"""P1：真实日线、官方月度合约身份与覆盖核对；不计算模型或持仓。"""

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import tomllib

import exchange_calendars as xcals
import numpy as np
import pandas as pd

from svxylab.downloads import DownloadStore
from svxylab.etf import acquire_etfs

AVAILABILITY = "ASSUMED_NEXT_SESSION_60MIN_BEFORE_CLOSE"
INDEX_PAGES = {
    "VIX": ("vix_history_page", "https://www.cboe.com/tradable-products/vix/vix-historical-data/"),
    "VVIX": ("vix_history_page", "https://www.cboe.com/tradable-products/vix/vix-historical-data/"),
    "VIX9D": ("vix_history_page", "https://www.cboe.com/tradable-products/vix/vix-historical-data/"),
    "VIX3M": ("vix3m_dashboard", "https://www.cboe.com/us/indices/dashboard/vix3m/"),
    "SKEW": ("skew_dashboard", "https://www.cboe.com/us/indices/dashboard/skew/"),
}
CATALOG_URL = "https://www-api.cboe.com/us/futures/market_statistics/historical_data/product/list/VX/"


def with_provenance(frame: pd.DataFrame, record: dict, symbol: str, price_type: str) -> pd.DataFrame:
    frame = frame.copy()
    frame["source"] = record["source"]
    frame["source_symbol"] = symbol
    frame["retrieved_at"] = record["retrieved_at"]
    frame["availability_policy"] = AVAILABILITY
    frame["published_at"] = None
    frame["raw_file"] = record["raw_file"]
    frame["raw_sha256"] = record["sha256"]
    frame["price_type"] = price_type
    return frame


def read_index(root: Path, record: dict, symbol: str) -> pd.DataFrame:
    frame = pd.read_csv(root / record["raw_file"])
    value_column = symbol if symbol in {"VVIX", "SKEW"} else "CLOSE"
    if not {"DATE", value_column}.issubset(frame.columns):
        raise ValueError(f"{symbol} 缺少官方日线字段")
    frame["as_of_session"] = pd.to_datetime(frame["DATE"], format="%m/%d/%Y").dt.strftime("%Y-%m-%d")
    frame["value"] = pd.to_numeric(frame[value_column], errors="raise")
    if frame.as_of_session.duplicated().any() or not np.isfinite(frame.value).all() or (frame.value <= 0).any():
        raise ValueError(f"{symbol} 存在重复日期或无效日值")
    frame["unit"] = "index_points"
    return with_provenance(frame.sort_values("as_of_session"), record, symbol, "OFFICIAL_INDEX_DAILY_CLOSE")


def monthly_contracts(catalog: dict, start: str, end: str) -> list[dict]:
    contracts = sorted([c for rows in catalog.values() for c in rows
                        if c["futures_root"] == "VX" and c["duration_type"] == "M"
                        and c["expire_date"] >= start], key=lambda c: c["expire_date"])
    future = [c for c in contracts if c["expire_date"] > end]
    if len(future) < 3:
        raise ValueError("官方目录未给出截至日后完整的前三个月度 VX 合约")
    contracts = [c for c in contracts if c["expire_date"] <= future[2]["expire_date"]]
    months = [c["expire_date"][:7] for c in contracts]
    expected = pd.period_range(start[:7], future[2]["expire_date"][:7], freq="M").astype(str).tolist()
    if months != expected:
        raise ValueError("官方月度目录有缺口或重复；不能把次近月冒充缺失近月")
    return contracts


def read_vx(root: Path, record: dict, contract: dict) -> pd.DataFrame:
    frame = pd.read_csv(root / record["raw_file"])
    if not {"Trade Date", "Futures", "Settle"}.issubset(frame.columns):
        raise ValueError(f"{contract['expire_date']} 缺少 VX 结算字段")
    code = contract["product_display"].split("/")[-1][0]
    expiry = pd.Timestamp(contract["expire_date"])
    expected_name = f"{code} ({expiry.strftime('%b %Y')})"
    if set(frame.Futures) != {expected_name}:
        raise ValueError(f"合约内部身份与官方月度目录不一致：{contract['expire_date']}")
    frame["as_of_session"] = pd.to_datetime(frame["Trade Date"], format="%Y-%m-%d").dt.strftime("%Y-%m-%d")
    if frame.as_of_session.duplicated().any() or (frame.as_of_session > contract["expire_date"]).any():
        raise ValueError(f"合约存在重复日期或到期后记录：{contract['expire_date']}")
    frame["value"] = pd.to_numeric(frame["Settle"], errors="raise")
    if not np.isfinite(frame.value).all() or (frame.value <= 0).any():
        raise ValueError(f"合约存在无效结算价：{contract['expire_date']}")
    frame["contract_id"] = "VX_" + contract["expire_date"]
    frame["expiration_date"] = contract["expire_date"]
    frame["duration_type"] = "M"
    frame["unit"] = "index_points"
    return with_provenance(frame, record, frame.contract_id.iloc[0], "CFE_DAILY_SETTLEMENT")


def select_front_three(sessions: list[str], contracts: list[dict], values: pd.DataFrame) -> pd.DataFrame:
    lookup = values.set_index(["as_of_session", "expiration_date"]).value.to_dict()
    rows = []
    for session in sessions:
        required = [c for c in contracts if c["expire_date"] > session][:3]
        row = {"as_of_session": session, "missing": ""}
        missing = []
        for rank in range(1, 4):
            c = required[rank - 1] if len(required) >= rank else None
            expiry = c["expire_date"] if c else None
            value = lookup.get((session, expiry))
            row[f"f{rank}_contract"] = "VX_" + expiry if expiry else None
            row[f"f{rank}_expiration"] = expiry
            row[f"f{rank}_settle"] = value
            if value is None:
                missing.append(f"F{rank}")
        row["missing"] = ",".join(missing)
        row["complete"] = not missing
        rows.append(row)
    return pd.DataFrame(rows)


def coverage(frame: pd.DataFrame, sessions: list[str]) -> dict:
    actual = set(frame.as_of_session) if not frame.empty else set()
    expected = set(sessions)
    present = sorted(actual & expected)
    return {"expected": len(expected), "observed": len(present), "missing_count": len(expected - actual),
            "first": present[0] if present else None, "last": present[-1] if present else None,
            "missing_sessions": sorted(expected - actual), "extra_sessions": sorted(actual - expected)}


def acquire_cboe(root: Path, store: DownloadStore, start: str, end: str) -> tuple[dict, pd.DataFrame, list, list]:
    indices, failures = {}, []
    for symbol, (label, url) in INDEX_PAGES.items():
        try:
            page = store.fetch(url, source="cboe", label=label)
            body = (root / page["raw_file"]).read_text()
            links = sorted(set(re.findall(r"https://cdn\.cboe\.com/[^\s\"<>\\]+/" + symbol + r"_History\.csv", body)))
            if len(links) != 1:
                raise ValueError("官方页面没有唯一历史 CSV 链接")
            raw = store.fetch(links[0], source="cboe", label=symbol + "_history")
            if not raw["ok"]:
                raise ValueError(f"{raw['error_layer']} / HTTP {raw['http_status']}")
            data = read_index(root, raw, symbol)
            indices[symbol] = data.loc[data.as_of_session.between(start, end)].copy()
        except (ValueError, KeyError, OSError) as error:
            failures.append({"dataset": symbol, "reason": str(error)})
    record = store.fetch(CATALOG_URL, source="cboe", label="VX_contract_catalog")
    contracts = monthly_contracts(json.loads((root / record["raw_file"]).read_text()), start, end)
    frames = []
    for contract in contracts:
        try:
            raw = store.fetch("https://cdn.cboe.com/" + contract["path"], source="cboe", label="VX_" + contract["expire_date"])
            if not raw["ok"]:
                raise ValueError(f"{raw['error_layer']} / HTTP {raw['http_status']}")
            frame = read_vx(root, raw, contract)
            frames.append(frame.loc[frame.as_of_session.between(start, end)].copy())
        except (ValueError, KeyError, OSError) as error:
            failures.append({"dataset": "VX_" + contract["expire_date"], "reason": str(error)})
    values = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["as_of_session", "expiration_date", "value"])
    return indices, values, contracts, failures


def prepare_data(root: Path) -> dict:
    config = tomllib.loads((root / "experiment.toml").read_text())
    start = config["data"]["feature_history_requested_start"]
    now = datetime.now(timezone.utc)
    freeze_path = root / "runs/p1/freeze.json"
    prior = json.loads(freeze_path.read_text()) if freeze_path.exists() else None
    calendar = xcals.get_calendar("XNYS", start=start, end=now.date().isoformat())
    completed = calendar.schedule.loc[calendar.schedule.close < pd.Timestamp(now)]
    end = prior["requested_end"] if prior else completed.index[-1].strftime("%Y-%m-%d")
    # 请求起点可以是假日（本轮为元旦）；按实际排程切片，不将其当作交易日查找。
    sessions = calendar.schedule.loc[start:end].index.strftime("%Y-%m-%d").tolist()
    store = DownloadStore(root)
    indices, values, contracts, failures = acquire_cboe(root, store, start, end)
    curve = select_front_three(sessions, contracts, values)
    cleaned = root / "data/clean"
    cleaned.mkdir(parents=True, exist_ok=True)
    for name, frame in indices.items():
        frame.to_csv(cleaned / f"{name}_daily.csv", index=False)
    values.to_csv(cleaned / "VX_contract_daily.csv", index=False)
    curve.to_csv(cleaned / "VX_front_three.csv", index=False)
    pd.DataFrame(contracts).to_csv(cleaned / "VX_contract_calendar.csv", index=False)
    calendar.schedule.loc[start:end].to_csv(cleaned / "equity_sessions.csv", index_label="as_of_session")
    stats = {name: coverage(indices.get(name, pd.DataFrame()), sessions) for name in INDEX_PAGES}
    stats["VX_F1_F2_F3"] = coverage(curve.loc[curve.complete], sessions)
    public_common = set(sessions)
    for name in ["VIX", "VVIX", "VIX9D", "VIX3M", "VX_F1_F2_F3"]:
        public_common -= set(stats.get(name, {"missing_sessions": sessions})["missing_sessions"])
    etfs, actions, etf_records, etf_failures = acquire_etfs(root, store, start, end)
    failures.extend(etf_failures)
    for symbol, frame in etfs.items():
        etfs[symbol] = with_provenance(frame, etf_records[symbol], symbol, "UNADJUSTED_EQUITY_RECONSTRUCTED_FROM_SPLIT_ADJUSTED_QUOTES")
    core_common = public_common.copy()
    for symbol in ("SVXY", "SPY"):
        frame = etfs.get(symbol, pd.DataFrame())
        stats[symbol] = coverage(frame, sessions)
        core_common -= set(stats[symbol]["missing_sessions"])
        if not frame.empty:
            frame.to_csv(cleaned / f"{symbol}_daily.csv", index=False)
            pd.DataFrame(actions[symbol], columns=["as_of_session", "kind", "date", "splitRatio", "shares_multiplier", "amount", "per_then_share_amount"]).to_csv(
                cleaned / f"{symbol}_corporate_actions.csv", index=False)
    full_core = len(core_common) == len(sessions)
    result = {"stage": "P1", "generated_at": now.isoformat(), "requested_start": start, "requested_end": end,
              "calendar": "exchange_calendars.XNYS", "calendar_version": xcals.__version__,
              "sessions": sessions, "coverage": stats, "failures": failures,
              "monthly_contracts": len(contracts), "vx_rows": len(values), "public_common_sessions": sorted(public_common),
              "core_common_sessions": sorted(core_common), "core_frozen_cutoff": max(core_common) if core_common else None,
              "research_started": False, "corporate_actions": actions,
              "full_core_data_ready": full_core, "historical_availability": AVAILABILITY,
              "summary": "核心数据覆盖完整；尚未训练模型。" if full_core else "核心数据尚不完整；缺口见覆盖表。"}
    freeze = {"stage": "P1", "requested_start": start, "requested_end": end,
              "public_common_cutoff": max(public_common) if public_common else None,
              "core_frozen_cutoff": max(core_common) if core_common else None, "complete_core_available": full_core,
              "created_at": prior["created_at"] if prior else now.isoformat()}
    freeze_path.write_text(json.dumps(freeze, ensure_ascii=False, indent=2) + "\n")
    (root / "runs/p1/data_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"P1 Cboe：{len(contracts)} 月度合约；{len(public_common)}/{len(sessions)} 个公共核心共同交易日。")
    return result
