"""真实 Yahoo 小样本；不自动修价，不打印 cookie / crumb，不写项目外缓存。"""
from datetime import datetime, timezone
import json
from pathlib import Path
from urllib.parse import urlsplit

from curl_cffi.requests import Session
import yfinance as yf

from svxylab.downloads import DownloadStore

root = Path.cwd()
store = DownloadStore(root)
yf.set_tz_cache_location(str(root / ".cache/yfinance"))


class RecordingSession(Session):
    def __init__(self, tag="probe"):
        super().__init__(impersonate="chrome")
        self.count = 0
        self.chart_counts = {}
        self.tag = tag

    def request(self, method, url, *args, **kwargs):
        self.count += 1
        if self.count > 8:
            raise RuntimeError("本次客户端总请求上限 8，停止重试")
        kwargs["timeout"] = min(kwargs.get("timeout", 12) or 12, 12)
        path = urlsplit(url).path
        if "/v8/finance/chart/" in path:
            self.chart_counts[path] = self.chart_counts.get(path, 0) + 1
            if self.chart_counts[path] > 2:
                raise RuntimeError("单个标的 chart 请求上限 2，停止重试")
        response = super().request(method, url, *args, **kwargs)
        if "/v8/finance/chart/" in path:
            symbol = path.rsplit("/", 1)[-1]
            window = self.tag if "period1" in kwargs.get("params", {}) else "timezone_probe"
            store.save_response(response.content, url=response.url, source="yahoo_client", label=symbol + "_" + window,
                                http_status=response.status_code, client="yfinance_1.7.0_curl_cffi")
        return response


if __name__ == "__main__":
    session = RecordingSession()
    result = {"started_at": datetime.now(timezone.utc).isoformat(), "yfinance_version": yf.__version__,
              "auto_adjust": False, "back_adjust": False, "repair": False, "actions": True, "keepna": True}
    try:
        ticker = yf.Ticker("SVXY", session=session)
        frame = ticker.history(start="2024-04-08", end="2024-04-13", auto_adjust=False,
                               back_adjust=False, repair=False, actions=True, keepna=True, timeout=12)
        frame.to_csv(root / "runs/p1/yahoo_probe_frame.csv")
        print(frame.to_string())
        result["rows"] = len(frame)
        result["columns"] = list(frame.columns)
    except Exception as error:
        # 异常类型足以记录失败；避免某些客户端异常夹带认证参数。
        result["error_type"] = type(error).__name__
        print("Yahoo 客户端小样本失败：", type(error).__name__)
    result["requests"] = session.count
    (root / "runs/p1/yahoo_probe_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
