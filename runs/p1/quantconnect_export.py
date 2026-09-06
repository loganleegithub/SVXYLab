# Execute in the P1 QuantConnect Research notebook after quantconnect_probe.py.
# Real data export only: no orders, models, strategy, or performance calculation.
from datetime import datetime, timezone
from pathlib import Path
import json
import sys

start, end = datetime(2019, 1, 1), datetime(2026, 9, 5)


def history_bars(symbol):
    return [{"time": str(b.time), "end_time": str(b.end_time), "open": float(b.open),
             "high": float(b.high), "low": float(b.low), "close": float(b.close),
             "volume": float(b.volume), "fill_forward": b.is_fill_forward}
            for b in qb.history[TradeBar](symbol, start, end, Resolution.DAILY)]


bundle = {"schema": "SVXYLAB_QC_P1_V1", "requested_start": "2019-01-01",
          "requested_end_exclusive": "2026-09-05", "normalization": "RAW",
          "time_zone": "America/New_York", "fill_forward": False,
          "daily_precise_end_time": True, "python_version": sys.version,
          "source": "QuantConnect US Equities / US Equity Security Master",
          "lean_engine_version_from_UI": "2.5.0.0.18057", "tickers": {}}
for ticker, symbol in symbols.items():
    qb.securities[symbol].set_data_normalization_mode(DataNormalizationMode.RAW)
    raw = history_bars(symbol)
    splits = [{"time": str(x.time), "type": str(x.type), "split_factor": float(x.split_factor),
               "reference_price": float(x.reference_price)}
              for x in qb.history[Split](symbol, start, end, Resolution.DAILY)]
    dividends = [{"time": str(x.time), "distribution": float(x.distribution),
                  "reference_price": float(x.reference_price)}
                 for x in qb.history[Dividend](symbol, start, end, Resolution.DAILY)]
    qb.securities[symbol].set_data_normalization_mode(DataNormalizationMode.ADJUSTED)
    adjusted = history_bars(symbol)
    qb.securities[symbol].set_data_normalization_mode(DataNormalizationMode.RAW)
    bundle["tickers"][ticker] = {"raw": raw, "adjusted": adjusted, "splits": splits, "dividends": dividends}
    print(ticker, "RAW", len(raw), "ADJUSTED", len(adjusted), "SPLIT_EVENTS", len(splits), "DIVIDENDS", len(dividends))
bundle["exported_at_utc"] = datetime.now(timezone.utc).isoformat()
directory = Path("output")
directory.mkdir(exist_ok=True)
full_path = directory / "svxylab_p1_etf_20190101_20260904.json"
full_path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2))
(directory / "svxylab_p1_probe.json").write_text(json.dumps(probe, indent=2))
print("P1_EXPORT_FILE", str(full_path.resolve()), "BYTES", full_path.stat().st_size)
