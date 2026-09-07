"""唯一正式 ETF 适配：Yahoo Chart 原件，显式撤销拆分调整到当时份额单位。"""

from datetime import datetime, timedelta
from hashlib import sha256
import json
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from svxylab.downloads import DownloadStore


def validate_ohlcv(frame: pd.DataFrame, symbol: str) -> None:
    numbers = frame[["open", "high", "low", "close", "volume"]]
    if (frame.as_of_session.duplicated().any() or not np.isfinite(numbers).all().all()
            or (numbers.iloc[:, :4] <= 0).any().any() or (numbers.volume < 0).any()
            or (numbers.high < numbers[["open", "close", "low"]].max(axis=1)).any()
            or (numbers.low > numbers[["open", "close", "high"]].min(axis=1)).any()):
        raise ValueError(f"{symbol} 日期或 OHLCV 数值关系异常；不自动修价")


def read_yahoo(root: Path, record: dict, symbol: str, start: str, end: str) -> tuple[pd.DataFrame, list]:
    query = parse_qs(urlsplit(record["url"]).query)
    if not {"div", "splits", "capitalGains"}.issubset(set(query.get("events", [""])[0].split(","))):
        raise ValueError(f"{symbol} 请求未明确包含分红、拆分和资本利得，不能把缺键当作零")
    response = json.loads((root / record["raw_file"]).read_text())
    chart = response["chart"]
    if chart.get("error") or len(chart.get("result") or []) != 1:
        raise ValueError(f"{symbol} Yahoo 没有返回唯一有效行情结果")
    item = chart["result"][0]
    meta = item["meta"]
    if (meta["symbol"] != symbol or meta["currency"] != "USD" or meta["dataGranularity"] != "1d"
            or meta["exchangeTimezoneName"] != "America/New_York"):
        raise ValueError(f"{symbol} 身份、币种、时区或日线粒度不符")
    frame = pd.DataFrame(item["indicators"]["quote"][0])
    frame["timestamp"] = item["timestamp"]
    frame["as_of_session"] = pd.to_datetime(frame.timestamp, unit="s", utc=True).dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
    frame = frame.sort_values("as_of_session").reset_index(drop=True)
    validate_ohlcv(frame, symbol)
    adjusted = dict(zip(item["timestamp"], item["indicators"]["adjclose"][0]["adjclose"], strict=True))
    frame["provider_adjusted_close"] = frame.timestamp.map(adjusted)
    if not np.isfinite(frame.provider_adjusted_close).all() or (frame.provider_adjusted_close <= 0).any():
        raise ValueError(f"{symbol} 复权参考缺少日值")
    frame["unadjustment_shares_multiplier"] = 1.0
    frame["split_factor"] = 1.0
    frame["cash_dividend"] = 0.0
    frame["capital_gain_distribution"] = 0.0
    events = item.get("events", {})
    actions = []
    for event in events.get("splits", {}).values():
        date = datetime.fromtimestamp(event["date"], ZoneInfo("America/New_York")).date().isoformat()
        ratio = float(event["numerator"]) / float(event["denominator"])
        if not np.isfinite(ratio) or ratio <= 0 or not frame.as_of_session.eq(date).any():
            raise ValueError(f"{symbol} 拆分数值或日期无效")
        frame.loc[frame.as_of_session.lt(date), "unadjustment_shares_multiplier"] *= ratio
        frame.loc[frame.as_of_session.eq(date), "split_factor"] /= ratio
        actions.append(dict(event, as_of_session=date, kind="split", shares_multiplier=ratio))
    for field in ("open", "high", "low", "close", "volume"):
        frame["provider_" + field] = frame[field]
        # Yahoo quote OHLC 与成交量已经按拆分调整；auto_adjust=False 也不会撤销它。
        frame[field] = frame[field] / frame.unadjustment_shares_multiplier if field == "volume" else frame[field] * frame.unadjustment_shares_multiplier
    for kind, column in (("dividends", "cash_dividend"), ("capitalGains", "capital_gain_distribution")):
        for event in events.get(kind, {}).values():
            date = datetime.fromtimestamp(event["date"], ZoneInfo("America/New_York")).date().isoformat()
            amount = float(event["amount"])
            if not np.isfinite(amount) or amount <= 0 or not frame.as_of_session.eq(date).any():
                raise ValueError(f"{symbol} 分配数值或日期无效")
            multiplier = frame.loc[frame.as_of_session.eq(date), "unadjustment_shares_multiplier"].iloc[0]
            frame.loc[frame.as_of_session.eq(date), column] += amount * multiplier
            actions.append(dict(event, as_of_session=date, kind=kind, per_then_share_amount=amount * multiplier))
    if len({(e["as_of_session"], e["kind"]) for e in actions}) != len(actions):
        raise ValueError(f"{symbol} 同日重复公司行动")
    frame["value"] = frame.close
    frame["unit"] = "USD_per_then_share"
    frame["volume_unit"] = "then_shares_reconstructed_from_provider_split_adjusted_volume"
    frame["provider_quote_basis"] = "SPLIT_ADJUSTED"
    frame["normalization"] = "UNADJUSTED_UNITS_RECONSTRUCTED_FROM_REPORTED_SPLITS"
    return frame.loc[frame.as_of_session.between(start, end)].copy(), actions


def acquire_etfs(root: Path, store: DownloadStore, start: str, end: str):
    def cached(symbol):
        for record in reversed(store.records):
            if record["label"] != symbol + "_history_full" or not record["ok"]:
                continue
            params = parse_qs(urlsplit(record["url"]).query)
            requested_start = datetime.fromtimestamp(int(params["period1"][0]), ZoneInfo("America/New_York")).date().isoformat()
            requested_end = datetime.fromtimestamp(int(params["period2"][0]), ZoneInfo("America/New_York")).date().isoformat()
            path = root / record["raw_file"]
            if requested_start == start and requested_end > end and path.is_file() and sha256(path.read_bytes()).hexdigest() == record["sha256"]:
                return record
        return None
    selected = {s: cached(s) for s in ("SVXY", "SPY")}
    missing = [s for s, r in selected.items() if not r]
    failures = []
    if missing:
        import yfinance as yf
        from curl_cffi.requests import Session
        yf.set_tz_cache_location(str(root / ".cache/yfinance"))
        class RecordedSession(Session):
            def __init__(self):
                super().__init__(impersonate="chrome")
                self.count, self.chart_counts = 0, {}
            def request(self, method, url, *args, **kwargs):
                self.count += 1
                if self.count > 8:
                    raise RuntimeError("ETF 客户端总请求上限 8")
                kwargs["timeout"] = min(kwargs.get("timeout", 12) or 12, 12)
                path = urlsplit(url).path
                chart_request = "/v8/finance/chart/" in path
                if chart_request:
                    self.chart_counts[path] = self.chart_counts.get(path, 0) + 1
                    if self.chart_counts[path] > 2:
                        raise RuntimeError("单标的 chart 请求上限 2")
                try:
                    response = super().request(method, url, *args, **kwargs)
                except Exception as error:
                    if chart_request:
                        store.save_response(b"", url=url, source="yahoo_client", label=path.rsplit("/", 1)[-1] + "_transport_failure",
                                            http_status=0, client="yfinance_1.7.0", error_layer=type(error).__name__)
                    raise
                if chart_request:
                    symbol = path.rsplit("/", 1)[-1]
                    tag = "history_full" if "period1" in (kwargs.get("params") or {}) else "timezone_probe"
                    store.save_response(response.content, url=response.url, source="yahoo_client", label=symbol + "_" + tag,
                                        http_status=response.status_code, client="yfinance_1.7.0_curl_cffi")
                return response
        session = RecordedSession()
        exclusive = (datetime.fromisoformat(end) + timedelta(days=1)).date().isoformat()
        for symbol in missing:
            try:
                yf.Ticker(symbol, session=session).history(start=start, end=exclusive, auto_adjust=False, back_adjust=False,
                                                          repair=False, actions=True, keepna=True, timeout=12)
                selected[symbol] = cached(symbol)
            except Exception as error:
                failures.append({"dataset": symbol, "reason": type(error).__name__})
    frames, actions = {}, {}
    for symbol, record in selected.items():
        try:
            if not record:
                raise ValueError("尚未取得所需完整请求的原始响应")
            frames[symbol], actions[symbol] = read_yahoo(root, record, symbol, start, end)
        except (ValueError, KeyError, OSError) as error:
            failures.append({"dataset": symbol, "reason": str(error)})
    return frames, actions, selected, failures
