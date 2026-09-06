# P1 real-data probe in QuantConnect Research; no model, orders, or strategy backtest.
from AlgorithmImports import *
from QuantConnect.Research import QuantBook
import json

qb = QuantBook()
qb.set_time_zone(TimeZones.NEW_YORK)
qb.settings.daily_precise_end_time = True
symbols = {ticker: qb.add_equity(ticker, Resolution.DAILY, fill_forward=False,
    data_normalization_mode=DataNormalizationMode.RAW).symbol for ticker in ["SVXY", "SPY"]}
probe = {}
for ticker, symbol in symbols.items():
    start, end = datetime(2024, 4, 8), datetime(2024, 4, 13)
    bars = [{"time": str(b.time), "end_time": str(b.end_time), "open": float(b.open),
             "high": float(b.high), "low": float(b.low), "close": float(b.close),
             "volume": float(b.volume), "fill_forward": b.is_fill_forward}
            for b in qb.history[TradeBar](symbol, start, end, Resolution.DAILY)]
    splits = [{"time": str(x.time), "type": str(x.type), "split_factor": float(x.split_factor),
               "reference_price": float(x.reference_price)}
              for x in qb.history[Split](symbol, start, end, Resolution.DAILY)]
    dividends = [{"time": str(x.time), "distribution": float(x.distribution),
                  "reference_price": float(x.reference_price)}
                 for x in qb.history[Dividend](symbol, start, end, Resolution.DAILY)]
    probe[ticker] = {"bars": bars, "splits": splits, "dividends": dividends}
print("P1_REAL_DATA_PROBE=" + json.dumps(probe))
