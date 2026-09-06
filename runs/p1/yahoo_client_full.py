"""在已通过小样本连接和字段检查后，取得本轮两只 ETF 的完整供应商原件。"""
from datetime import datetime, timezone
import json
from pathlib import Path

import yfinance as yf
from yahoo_client_probe import RecordingSession

root = Path.cwd()
session = RecordingSession(tag="history_full")
result = {"stage": "P1", "started_at": datetime.now(timezone.utc).isoformat(), "yfinance_version": yf.__version__,
          "requested_start": "2019-01-01", "requested_end_exclusive": "2026-09-05",
          "auto_adjust": False, "back_adjust": False, "repair": False, "actions": True, "keepna": True, "tickers": {}}
for symbol in ("SVXY", "SPY"):
    try:
        frame = yf.Ticker(symbol, session=session).history(start="2019-01-01", end="2026-09-05", auto_adjust=False,
                                                         back_adjust=False, repair=False, actions=True, keepna=True, timeout=12)
        result["tickers"][symbol] = {"rows": len(frame), "first": str(frame.index.min()), "last": str(frame.index.max()),
                                      "columns": list(frame.columns), "dividend_events": int(frame.Dividends.ne(0).sum()),
                                      "split_events": int(frame["Stock Splits"].ne(0).sum())}
        print(symbol, result["tickers"][symbol], flush=True)
    except Exception as error:
        result["tickers"][symbol] = {"error_type": type(error).__name__}
        print(symbol, "FAILED", type(error).__name__, flush=True)
result["requests"] = session.count
(root / "runs/p1/yahoo_full_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
