"""合成手算夹具，仅检验会计和标签时钟，不代表真实市场表现。"""

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import pytest

from svxylab.ledger import baseline_targets, continuous_ledger, rebalance
from svxylab.returns import total_return
from svxylab.timing import five_day_labels, mature_labels, session_clock

pytestmark = pytest.mark.synthetic


def synthetic_prices(closes, splits=None, dividends=None):
    return pd.DataFrame({"as_of_session": pd.bdate_range("2024-04-01", periods=len(closes)).strftime("%Y-%m-%d"),
                         "close": closes, "split_factor": splits or [1.0] * len(closes),
                         "cash_dividend": dividends or [0.0] * len(closes),
                         "capital_gain_distribution": 0.0})


def synthetic_clock(frame):
    schedule = xcals.get_calendar("XNYS", start="2024-04-01", end="2024-06-28").schedule
    return session_clock(frame.as_of_session.tolist(), schedule)


def account(closes, weights, *, capital=100.0, cost=0.0, splits=None, dividends=None):
    frame = total_return(synthetic_prices(closes, splits, dividends))
    target = pd.Series(weights, index=pd.Index(frame.as_of_session, name="as_of_session"), dtype=float)
    return continuous_ledger(frame, synthetic_clock(frame), target, capital=capital, cost_rate=cost)


def test_split_creates_shares_not_return():
    frame = total_return(synthetic_prices([100, 50, 55], splits=[1, .5, 1]))
    assert np.isnan(frame.simple_return.iloc[0])
    assert frame.total_return_index.tolist() == pytest.approx([100, 100, 110])
    ledger = account([100, 100, 50, 55], [1, 1, 1, 1], splits=[1, 1, .5, 1])
    assert ledger.equity_end.tolist() == pytest.approx([100, 100, 100, 110])
    assert ledger.shares_end.iloc[2] == 2
    assert ledger.cost.iloc[2] == 0


def test_dividend_is_not_lost_or_double_counted():
    frame = total_return(synthetic_prices([100, 98, 99], dividends=[0, 2, 0]))
    assert frame.total_return_index.tolist() == pytest.approx([100, 100, 100 * 99 / 98])
    ledger = account([100, 100, 98, 99], [1, np.nan, np.nan, np.nan], dividends=[0, 0, 2, 0])
    assert ledger.equity_end.tolist() == pytest.approx([100, 100, 100, 101])
    assert ledger.distribution_income.iloc[2] == 2
    assert ledger.cash_end.iloc[2] == 2
    assert ledger.shares_end.iloc[2] == 1


def test_split_and_dividend_use_same_current_share_units():
    frame = total_return(synthetic_prices([100, 49], splits=[1, .5], dividends=[0, 1]))
    assert frame.total_return_index.tolist() == pytest.approx([100, 100])


def test_old_position_owns_execution_day_profit():
    ledger = account([100, 110, 121, 60.5], [1, 0, 1, 1])
    # 首个 +10% 尚未买入；次个 +10% 归旧股；退出后的 -50% 不归新买入股。
    assert ledger.holding_pnl.tolist() == pytest.approx([0, 0, 10, 0])
    assert ledger.equity_end.tolist() == pytest.approx([100, 100, 110, 110])
    assert ledger.old_weight.tolist() == pytest.approx([0, 0, 1, 0])


def test_exact_buy_cost_and_post_fee_weight():
    trade = rebalance(100, 0, 1, .01)
    assert trade["equity"] == pytest.approx(100 / 1.01)
    assert trade["cost"] == pytest.approx(100 - 100 / 1.01)
    assert trade["cash"] == 0


def test_exact_sell_cost():
    trade = rebalance(100, 100, 0, .01)
    assert trade["equity"] == 99
    assert trade["cost"] == 1
    assert trade["cash"] == 99


def test_fixed_target_drift_still_has_real_turnover():
    ledger = account([100, 100, 120], [.5, .5, .5], cost=.01)
    # 前日 E=100/1.005、股值=50/1.005；上涨后 u=6/11，不是 0.5。
    pre_equity = 110 / 1.005
    expected = pre_equity * (1 - .01 * (6 / 11)) / (1 - .01 * .5)
    assert ledger.pre_trade_weight.iloc[2] == pytest.approx(6 / 11)
    assert ledger.equity_end.iloc[2] == pytest.approx(expected)
    assert ledger.trade_notional.iloc[2] > 0
    assert ledger.cost.iloc[2] == pytest.approx(.01 * ledger.trade_notional.iloc[2])
    assert ledger.new_weight.iloc[2] == pytest.approx(.5)


def test_missing_signal_holds_shares_and_keeps_losses():
    ledger = account([100, 100, 80, 88], [.5, np.nan, np.nan, 0])
    assert ledger.equity_end.tolist() == pytest.approx([100, 100, 90, 94])
    assert ledger.shares_end.iloc[1:].tolist() == pytest.approx([.5, .5, .5])
    assert ledger.new_weight.iloc[2] == pytest.approx(40 / 90)
    assert ledger.trade_notional.iloc[2] == 0


def test_missing_front_signal_is_not_a_cash_exit():
    curve = pd.DataFrame({"as_of_session": ["2024-04-01", "2024-04-02", "2024-04-03"],
                          "f1_settle": [10, np.nan, 12], "f2_settle": [11, 11, 12]})
    targets = baseline_targets(curve, [.25])
    assert targets["curve"].iloc[0] == 1
    assert np.isnan(targets["curve"].iloc[1])
    assert targets["curve"].iloc[2] == 0


def test_future_prices_and_signals_do_not_change_prior_account():
    original = account([100, 110, 121, 100, 90, 120, 110, 100], [.5] * 8, cost=.0005)
    changed = account([100, 110, 121, 100, 90, 20, 200, 1], [.5] * 5 + [0, 1, 0], cost=.0005)
    pd.testing.assert_frame_equal(original.iloc[:5], changed.iloc[:5])


def test_label_uses_next_close_through_sixth_close_and_entry_relative_loss():
    # t 的 50 不作入场价；入场 100，未来最高 120 不作损失基准。
    frame = total_return(synthetic_prices([50, 100, 120, 110, 90, 105, 110, 120]))
    labels = five_day_labels(frame, synthetic_clock(frame))
    assert labels.R5.iloc[0] == pytest.approx(.10)
    assert labels.L5.iloc[0] == pytest.approx(.10)
    assert labels.Y10.iloc[0] == 1  # 正好 10% 必须计入，非峰谷 25%。
    assert labels.label_end_session.iloc[0] == frame.as_of_session.iloc[6]
    assert labels[["R5", "L5", "Y10"]].iloc[-6:].isna().all().all()


def test_missing_future_path_does_not_become_zero_label():
    frame = total_return(synthetic_prices([100] * 9))
    frame.loc[3, "total_return_index"] = np.nan
    labels = five_day_labels(frame, synthetic_clock(frame))
    assert not labels.observed.iloc[0]
    assert np.isnan(labels.Y10.iloc[0])


def test_label_maturity_and_assumed_availability_are_separate():
    frame = total_return(synthetic_prices([100] * 12))
    labels = five_day_labels(frame, synthetic_clock(frame))
    first = labels.iloc[0]
    assert mature_labels(labels, first.label_matures_at).empty
    before = (pd.Timestamp(first.label_available_at) - pd.Timedelta(seconds=1)).isoformat()
    assert mature_labels(labels, before).empty
    assert mature_labels(labels, first.label_available_at).index.tolist() == [0]


def test_future_changes_do_not_change_already_matured_labels():
    frame = total_return(synthetic_prices([100] * 15))
    clock = synthetic_clock(frame)
    original = five_day_labels(frame, clock)
    frame.loc[10:, "total_return_index"] = [10, 20, 30, 40, 50]
    changed = five_day_labels(frame, clock)
    pd.testing.assert_frame_equal(original.iloc[:4], changed.iloc[:4])


def test_real_early_close_and_dst_clock():
    schedule = xcals.get_calendar("XNYS", start="2024-03-01", end="2024-12-31").schedule
    clock = session_clock(["2024-03-08", "2024-11-27"], schedule).set_index("as_of_session")
    # 周末切换夏令时；感恩节后周五 13:00 收盘，12:00 决策。
    assert clock.loc["2024-03-08", "decision_at"] == "2024-03-11T19:00:00+00:00"
    assert clock.loc["2024-11-27", "decision_session"] == "2024-11-29"
    assert clock.loc["2024-11-27", "decision_at"] == "2024-11-29T17:00:00+00:00"
    assert clock.loc["2024-11-27", "execution_at"] == "2024-11-29T18:00:00+00:00"


def test_continuous_account_has_no_five_day_overlapping_leverage():
    ledger = account([100, 100, 110, 121, 133.1, 146.41, 161.051, 177.1561], [1] * 8)
    assert ledger.equity_end.iloc[-1] == pytest.approx(177.1561)
    assert ledger.shares_end.iloc[1:].tolist() == pytest.approx([1] * 7)
    assert ledger.trade_notional.sum() == pytest.approx(100)


def test_ledger_reconciles_cash_shares_cost_and_equity():
    ledger = account([100, 95, 105, 70, 120, 100], [.25, .75, .5, 0, 1, 0], cost=.0005)
    assert np.allclose(ledger.equity_end, ledger.equity_start + ledger.holding_pnl - ledger.cost)
    assert np.allclose(ledger.equity_end, ledger.cash_end + ledger.shares_end * ledger.close)
    assert np.allclose(ledger.cost, ledger.trade_notional * .0005)
    assert (ledger.cash_end >= 0).all()


@pytest.mark.parametrize("bad_close", [np.nan, 0, -1])
def test_missing_or_invalid_held_price_is_rejected(bad_close):
    with pytest.raises(ValueError, match="价格"):
        total_return(synthetic_prices([100, bad_close]))
