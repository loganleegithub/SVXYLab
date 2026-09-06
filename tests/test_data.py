"""明确标记的合成小夹具；验证数据处理，不是真实行情或回测。"""

from hashlib import sha256
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from svxylab.data import coverage, monthly_contracts, read_index, read_vx, select_front_three
from svxylab.downloads import DownloadStore
from svxylab.etf import read_yahoo

pytestmark = pytest.mark.synthetic


def record(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return {"source": "SYNTHETIC_TEST", "retrieved_at": "2026-09-06T00:00:00+00:00",
            "raw_file": name, "sha256": sha256(path.read_bytes()).hexdigest()}


def contract(expiry, code="H"):
    return {"expire_date": expiry, "duration_type": "M", "futures_root": "VX",
            "product_display": f"VX+VXT/{code}9"}


def test_expiration_day_rolls_and_does_not_relabel_missing_front():
    contracts = [contract(d) for d in ["2019-03-19", "2019-04-17", "2019-05-22", "2019-06-19"]]
    values = pd.DataFrame([
        ("2019-03-18", "2019-03-19", 12.0),
        ("2019-03-18", "2019-04-17", 14.0),
        ("2019-03-18", "2019-05-22", 16.0),
        ("2019-03-19", "2019-03-19", 11.0),
        ("2019-03-19", "2019-05-22", 15.0),
        ("2019-03-19", "2019-06-19", 17.0),
    ], columns=["as_of_session", "expiration_date", "value"])
    curve = select_front_three(["2019-03-18", "2019-03-19"], contracts, values)
    assert curve.iloc[0].f1_settle == 12.0
    assert curve.iloc[1].f1_expiration == "2019-04-17"
    assert pd.isna(curve.iloc[1].f1_settle)
    assert curve.iloc[1].f2_settle == 15.0
    assert curve.iloc[1].missing == "F1"
    assert not curve.iloc[1].complete


def test_catalog_uses_only_monthly_and_rejects_gap():
    monthly = [contract(d) for d in ["2019-01-16", "2019-02-13", "2019-03-19", "2019-04-17"]]
    weekly = dict(contract("2019-01-09"), duration_type="W")
    assert monthly_contracts({"2019": monthly + [weekly]}, "2019-01-01", "2019-01-31") == monthly
    with pytest.raises(ValueError, match="缺口"):
        monthly_contracts({"2019": monthly[1:]}, "2019-01-01", "2019-01-31")


def test_vx_uses_settlement_and_checks_identity(tmp_path):
    raw = record(tmp_path, "synthetic.csv", "Trade Date,Futures,Close,Settle\n2019-03-18,H (Mar 2019),99,12.5\n")
    assert read_vx(tmp_path, raw, contract("2019-03-19")).value.tolist() == [12.5]
    with pytest.raises(ValueError, match="身份"):
        read_vx(tmp_path, raw, contract("2019-04-17", "J"))


@pytest.mark.parametrize("rows", ["01/02/2019,12\n01/02/2019,13\n", "01/02/2019,nan\n", "01/02/2019,-1\n"])
def test_reject_duplicate_or_invalid_index(tmp_path, rows):
    raw = record(tmp_path, "synthetic.csv", "DATE,VVIX\n" + rows)
    with pytest.raises(ValueError):
        read_index(tmp_path, raw, "VVIX")


def test_coverage_does_not_count_off_calendar_values():
    frame = pd.DataFrame({"as_of_session": ["2019-01-01", "2019-01-02"]})
    result = coverage(frame, ["2019-01-02", "2019-01-03"])
    assert result["observed"] == 1
    assert result["missing_sessions"] == ["2019-01-03"]
    assert result["extra_sessions"] == ["2019-01-01"]


def test_local_import_checks_hash_and_preserves_original(tmp_path):
    original = tmp_path / "synthetic.csv"
    original.write_text("SYNTHETIC_TEST\n")
    digest = sha256(original.read_bytes()).hexdigest()
    store = DownloadStore(tmp_path)
    with pytest.raises(ValueError, match="摘要"):
        store.import_local(original, url="https://example.invalid/synthetic.csv", source="synthetic",
                           label="fixture", expected_sha256="bad")
    imported = store.import_local(original, url="https://example.invalid/synthetic.csv", source="synthetic",
                                  label="fixture", expected_sha256=digest)
    assert (tmp_path / imported["raw_file"]).read_bytes() == original.read_bytes()
    assert imported["original_downloaded_at"] is None
    assert len([json.loads(x) for x in store.manifest.read_text().splitlines()]) == 1


def test_empty_content_type_http_failure_is_preserved(tmp_path, monkeypatch):
    monkeypatch.setattr("svxylab.downloads.subprocess.run", lambda *a, **k:
                        SimpleNamespace(returncode=0, stdout=json.dumps({"http_code": 403, "content_type": None})))
    store = DownloadStore(tmp_path)
    response = store.fetch("https://example.invalid/synthetic", source="synthetic", label="fixture")
    assert not response["ok"]
    assert response["http_status"] == 403
    assert response["error_layer"] == "HTTP"
    assert len(store.records) == 1
    assert (tmp_path / response["raw_file"]).exists()


def synthetic_yahoo_bundle():
    timestamps = [int(pd.Timestamp(d + " 09:30", tz="America/New_York").timestamp())
                  for d in ["2024-04-10", "2024-04-11"]]
    return {"chart": {"error": None, "result": [{
        "meta": {"symbol": "SVXY", "currency": "USD", "dataGranularity": "1d", "exchangeTimezoneName": "America/New_York"},
        "timestamp": timestamps,
        "indicators": {"quote": [{"open": [50, 51], "high": [51, 52], "low": [49, 50], "close": [50, 51], "volume": [2000, 1000]}],
                       "adjclose": [{"adjclose": [50, 51]}]},
        "events": {"splits": {str(timestamps[1]): {"date": timestamps[1], "numerator": 2, "denominator": 1, "splitRatio": "2:1"}}}
    }]}}


def yahoo_record(tmp_path, bundle):
    raw = record(tmp_path, "synthetic.json", json.dumps(bundle))
    raw["url"] = "https://example.invalid/SYNTHETIC?events=div,splits,capitalGains"
    return raw


def test_yahoo_preserves_provider_values_and_restores_then_share_units(tmp_path):
    raw = yahoo_record(tmp_path, synthetic_yahoo_bundle())
    frame, actions = read_yahoo(tmp_path, raw, "SVXY", "2019-01-01", "2026-09-04")
    assert frame.provider_close.tolist() == [50, 51]
    assert frame.close.tolist() == [100, 51]
    assert frame.volume.tolist() == [1000, 1000]
    assert frame.split_factor.tolist() == [1, .5]
    assert len(actions) == 1


def test_absent_dividend_request_is_not_zero_dividends(tmp_path):
    raw = yahoo_record(tmp_path, synthetic_yahoo_bundle())
    raw["url"] = "https://example.invalid/SYNTHETIC?events=splits"
    with pytest.raises(ValueError, match="分红"):
        read_yahoo(tmp_path, raw, "SVXY", "2019-01-01", "2026-09-04")


def test_dividend_is_preserved_in_then_share_units(tmp_path):
    bundle = synthetic_yahoo_bundle()
    item = bundle["chart"]["result"][0]
    date = item["timestamp"][0]
    item["events"]["dividends"] = {str(date): {"date": date, "amount": 0.25}}
    frame, actions = read_yahoo(tmp_path, yahoo_record(tmp_path, bundle), "SVXY", "2019-01-01", "2026-09-04")
    assert frame.cash_dividend.tolist() == [0.5, 0.0]
    dividend = next(a for a in actions if a["kind"] == "dividends")
    assert dividend["amount"] == 0.25
    assert dividend["per_then_share_amount"] == 0.5


def test_yahoo_rejects_missing_bar_without_filling(tmp_path):
    bundle = synthetic_yahoo_bundle()
    bundle["chart"]["result"][0]["indicators"]["quote"][0]["close"][1] = None
    raw = yahoo_record(tmp_path, bundle)
    with pytest.raises(ValueError, match="OHLCV"):
        read_yahoo(tmp_path, raw, "SVXY", "2019-01-01", "2026-09-04")


def test_yahoo_rejects_wrong_symbol(tmp_path):
    raw = yahoo_record(tmp_path, synthetic_yahoo_bundle())
    with pytest.raises(ValueError, match="身份"):
        read_yahoo(tmp_path, raw, "SPY", "2019-01-01", "2026-09-04")
