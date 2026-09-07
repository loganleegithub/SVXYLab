"""明确的合成测试夹具，仅检查公式/日期/缺失/时钟；不是市场回测。"""

import math

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import pytest

from svxylab.features import (CORE_IDS, EXTENSION_IDS, LOOKBACK, build_inputs, calculate_features,
                             feature_availability, features_at_cutoff, missing_reasons)
from svxylab.timing import session_clock


@pytest.fixture
def synthetic_market():
    schedule = xcals.get_calendar("XNYS", start="2024-01-01", end="2024-04-30").schedule
    sessions = schedule.index.strftime("%Y-%m-%d")[:36].tolist()
    expirations = [sessions[25], "2024-03-19", "2024-04-17", "2024-05-22"]
    contracts = [f"VX_{d}" for d in expirations]
    curve, vx = [], []
    for i, day in enumerate(sessions):
        prices = [15 + k * 2 + i * (.03 + k * .01) for k in range(4)]
        front = 0 if i < 25 else 1
        row = {"as_of_session": day}
        for j in (1, 2, 3):
            k = front + j - 1
            row.update({f"f{j}_contract": contracts[k], f"f{j}_expiration": expirations[k], f"f{j}_settle": prices[k]})
        curve.append(row)
        vx.extend({"as_of_session": day, "contract_id": contracts[k], "duration_type": "M", "value": prices[k]} for k in range(4))
    indices = {s: pd.DataFrame({"as_of_session": sessions, "value": [base + i * .2 for i in range(36)]})
               for s, base in [("VIX", 18), ("VVIX", 85), ("VIX9D", 16), ("VIX3M", 21), ("SKEW", 125)]}
    returns = {}
    for s, changes in [("SPY", [.01, -.02, .005, -.003, .007]), ("SVXY", [.02, -.03, .004, -.015, .002])]:
        values = [100.0]
        for i in range(35):
            values.append(values[-1] * math.exp(changes[i % len(changes)]))
        returns[s] = pd.DataFrame({"as_of_session": sessions, "total_return_index": values})
    return {"sessions": sessions, "curve": pd.DataFrame(curve), "vx": pd.DataFrame(vx), "indices": indices,
            "returns": returns, "schedule": schedule}


def compute(market, *, history_start="2019-01-01"):
    x = build_inputs(*(market[k] for k in ["sessions", "curve", "vx", "indices", "returns"]), history_start=history_start)
    return x, calculate_features(x)


@pytest.mark.parametrize("feature", CORE_IDS + EXTENSION_IDS)
def test_synthetic_every_dictionary_formula(synthetic_market, feature):
    m = synthetic_market
    x, f = compute(m)
    i = 30
    row = m["curve"].iloc[i]
    F1, F2, F3 = [row[f"f{k}_settle"] for k in (1, 2, 3)]
    tau = [(pd.Timestamp(row[f"f{k}_expiration"]) - pd.Timestamp(m["sessions"][i])).days for k in (1, 2, 3)]
    idx = {s: frame.value.tolist() for s, frame in m["indices"].items()}
    spy, svxy = [m["returns"][s].total_return_index.tolist() for s in ["SPY", "SVXY"]]
    rs = [math.log(spy[j] / spy[j-1]) for j in range(i-20, i+1)]
    rv = [math.log(svxy[j] / svxy[j-1]) for j in range(i-20, i+1)]
    d02 = math.sqrt(252 * sum(r*r for r in rs) / 21)
    a01 = 30 * math.log(F2/F1) / (tau[1]-tau[0])
    prior = m["vx"].loc[(m["vx"].contract_id == row.f1_contract) &
                         (m["vx"].as_of_session == m["sessions"][i-5]), "value"].item()
    expected = {
        "A01": a01, "A02": 30*math.log(F3/F2)/(tau[2]-tau[1])-a01,
        "A03": math.log(F1/idx["VIX"][i]), "A04": math.log(F1/prior), "A05": tau[0],
        "B01": math.log(idx["VIX"][i]), "B02": math.log(idx["VIX9D"][i]/idx["VIX"][i]),
        "B03": math.log(idx["VIX"][i]/idx["VIX3M"][i]), "C01": math.log(idx["VVIX"][i]),
        "C02": math.log(idx["VVIX"][i]/idx["VVIX"][i-5]), "C03": math.log(idx["VIX"][i]/idx["VIX"][i-5]),
        "D01": math.sqrt(252*sum(r*r for r in rs[-5:])/5), "D02": d02,
        "D03": sum(min(r,0)**2 for r in rs)/sum(r*r for r in rs), "E01": (idx["VIX"][i]/100)**2-d02**2,
        "F01": math.log(spy[i]/spy[i-5]), "F02": spy[i]/max(spy[i-20:i+1])-1,
        "F03": math.log(svxy[i]/svxy[i-5]), "F04": math.sqrt(252*sum(r*r for r in rv)/21),
        "G01": idx["SKEW"][i], "G02": idx["SKEW"][i]-idx["SKEW"][i-5],
    }
    assert f.iloc[i][feature] == pytest.approx(expected[feature], abs=1e-12)


def test_synthetic_warmup_distinguishes_returns_from_levels(synthetic_market):
    x, f = compute(synthetic_market)
    for key, lag in LOOKBACK.items():
        assert f[key].first_valid_index() == f.index[lag]
        assert f[key].isna().sum() == lag
    assert pd.isna(x.r_SPY.iloc[0])
    assert pd.isna(f.D02.iloc[20]) and not pd.isna(f.F02.iloc[20])


def test_synthetic_a04_roll_uses_current_contract_at_both_endpoints(synthetic_market):
    x, f = compute(synthetic_market)
    i = 25
    assert x.f1_contract.iloc[i] != x.f1_contract.iloc[i-5]
    assert f.A04.iloc[i] == pytest.approx(math.log(x.F1.iloc[i]/x.a04_lag5_settle.iloc[i]))
    assert abs(f.A04.iloc[i] - math.log(x.F1.iloc[i]/x.F1.iloc[i-5])) > .05
    m = synthetic_market
    missing_key = (m["vx"].as_of_session == m["sessions"][i-5]) & (m["vx"].contract_id == x.f1_contract.iloc[i])
    m["vx"] = m["vx"].loc[~missing_key]
    _, changed = compute(m)
    assert pd.isna(changed.A04.iloc[i])
    assert not pd.isna(changed.A01.iloc[i])


def test_synthetic_skew_gap_does_not_shrink_core_or_shift_calendar(synthetic_market):
    x, original = compute(synthetic_market)
    m = synthetic_market
    gap = 27
    m["indices"]["SKEW"] = m["indices"]["SKEW"].drop(gap)
    x, f = compute(m)
    pd.testing.assert_frame_equal(f[CORE_IDS], original[CORE_IDS])
    assert f.G01.iloc[gap:gap+6].isna().sum() == 1
    assert pd.isna(f.G02.iloc[gap]) and pd.isna(f.G02.iloc[gap+5])
    assert f.G02.iloc[gap+1] == original.G02.iloc[gap+1]
    assert len(f) == len(original)


def test_synthetic_missing_core_window_never_skips_or_fills(synthetic_market):
    m = synthetic_market
    m["returns"]["SPY"].loc[26, "total_return_index"] = np.nan
    m["indices"]["VVIX"] = m["indices"]["VVIX"].drop(25)
    x, f = compute(m)
    assert f.D01.iloc[26:32].isna().all()
    assert pd.isna(f.C02.iloc[25]) and pd.isna(f.C02.iloc[30])
    assert not pd.isna(f.C02.iloc[26])
    assert len(f) == 36


def test_synthetic_zero_semivariance_denominator_is_missing(synthetic_market):
    synthetic_market["returns"]["SPY"].total_return_index = 100.0
    x, f = compute(synthetic_market)
    assert f.D03.isna().all()
    assert (f.D02.iloc[21:] == 0).all()
    assert (missing_reasons(f, x).D03.iloc[21:] == "ZERO_VARIANCE_DENOMINATOR").all()


def test_synthetic_history_boundary_blocks_earlier_warmup(synthetic_market):
    m = synthetic_market
    start = m["sessions"][10]
    _, f = compute(m, history_start=start)
    assert f.index[0] == start
    assert f.D02.first_valid_index() == m["sessions"][31]
    assert f.A04.first_valid_index() == m["sessions"][15]


def test_synthetic_future_mutation_and_appending_leave_past_features_unchanged(synthetic_market):
    m = synthetic_market
    x, baseline = compute(m)
    cut = m["sessions"][27]
    prefix = {**m, "sessions": m["sessions"][:28]}
    _, truncated = compute(prefix)
    pd.testing.assert_frame_equal(truncated, baseline.loc[:cut])
    for k in (1, 2, 3):
        m["curve"].loc[m["curve"].as_of_session > cut, f"f{k}_settle"] *= 3
    m["vx"].loc[m["vx"].as_of_session > cut, "value"] *= 3
    for frame in m["indices"].values():
        frame.loc[frame.as_of_session > cut, "value"] *= 1.7
    for frame in m["returns"].values():
        frame.loc[frame.as_of_session > cut, "total_return_index"] *= 2.3
    _, changed = compute(m)
    pd.testing.assert_frame_equal(changed.loc[:cut], baseline.loc[:cut])
    assert not changed.iloc[28:].equals(baseline.iloc[28:])


def test_synthetic_feature_clock_enforces_next_close_minus_hour(synthetic_market):
    m = synthetic_market
    _, f = compute(m)
    clock = session_clock(m["sessions"], m["schedule"])
    availability = feature_availability(f, clock)
    i = 25
    deadline = pd.Timestamp(availability.assumed_available_at.iloc[i])
    before = features_at_cutoff(f, availability, deadline - pd.Timedelta(microseconds=1))
    at = features_at_cutoff(f, availability, deadline)
    assert f.index[i] not in before.index and f.index[i] in at.index
    assert pd.Timestamp(availability.execution_at.iloc[i]) - deadline == pd.Timedelta(hours=1)
    same_day = features_at_cutoff(f, availability, pd.Timestamp(availability.information_close_at.iloc[i]))
    assert f.index[i] not in same_day.index
    with pytest.raises(ValueError):
        features_at_cutoff(f, availability, "2024-02-01")
    bad = clock.copy()
    bad.loc[i, "decision_at"] = bad.loc[i, "information_close_at"]
    with pytest.raises(ValueError):
        feature_availability(f, bad)


@pytest.mark.parametrize("day,ny_hour", [("2024-03-08", 15), ("2024-11-27", 12)])
def test_synthetic_feature_clock_dst_and_half_day(day, ny_hour):
    schedule = xcals.get_calendar("XNYS", start="2024-01-01", end="2024-12-31").schedule
    clock = session_clock([day], schedule)
    f = pd.DataFrame(1.0, index=pd.Index([day], name="as_of_session"), columns=CORE_IDS+EXTENSION_IDS)
    available = feature_availability(f, clock)
    actual = pd.Timestamp(available.decision_at.iloc[0]).tz_convert("America/New_York")
    assert actual.hour == ny_hour
    assert actual.date().isoformat() == ("2024-03-11" if day == "2024-03-08" else "2024-11-29")
