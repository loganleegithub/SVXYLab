"""P5合成夹具：仓位/真实执行时钟/缺服务/起点/费用，不是真实收益。"""

import numpy as np
import pandas as pd
import pytest

from svxylab.economics import account_metrics, bounded_csv, model_targets, run_account, scenarios
from svxylab.returns import total_return
from svxylab.timing import session_clock
import exchange_calendars as xcals

pytestmark = pytest.mark.synthetic


def fixture(closes, mu=None, q=None, available=None):
    dates = pd.bdate_range("2020-09-08", periods=len(closes)).strftime("%Y-%m-%d").tolist()
    prices = total_return(pd.DataFrame({"as_of_session": dates, "close": closes, "split_factor": 1.,
                                       "cash_dividend": 0., "capital_gain_distribution": 0.}))
    clock = session_clock(dates, xcals.get_calendar("XNYS", start="2020-09-01", end="2020-12-31").schedule)
    predictions = pd.concat([pd.DataFrame({"as_of_session": dates, "model": m,
        "forecast_available": available if available is not None else True,
        "mu5": mu if mu is not None else .02, "q90": q if q is not None else .1,
        "simulation_decision_at": clock.decision_at, "fit_id": "SYNTHETIC", "score_observed": False, "P10": 1.}) for m in ["M0", "M1", "M2"]])
    return dates, prices, clock, predictions


def test_fixed_formula_strict_mu_boundary_q_zero_and_capacity_cap():
    dates, _, _, p = fixture([100]*4, mu=[.001, .0010001, -.1, .01], q=[0, .1, .01, .2])
    t, _ = model_targets(p, dates, .05, .0005)
    assert t['M2_risk_only'].tolist() == pytest.approx([1., .5, 1., .25])
    assert t['M2_main'].tolist() == pytest.approx([0., .5, 0., .25])


def test_future_score_and_probability_cannot_gate_tail_trades():
    dates, prices, clock, p = fixture([100, 100, 80, 88])
    targets, metas = model_targets(p, dates, .05, .0005)
    p['score_observed'], p['P10'] = True, 0.
    changed, _ = model_targets(p, dates, .05, .0005)
    pd.testing.assert_series_equal(targets['M2_main'], changed['M2_main'])
    ledger = run_account(prices, clock, targets['M2_main'], dates[1], 100, 0, metadata=metas['M2_main'])
    assert ledger.shares_end.iloc[1] == .5
    assert ledger.holding_pnl.iloc[2] == -10


def test_missing_prediction_keeps_real_shares_and_cash_without_rebalancing():
    dates, prices, clock, p = fixture([100, 100, 80, 88], available=[True, False, False, True])
    targets, metas = model_targets(p, dates, .05, .0005)
    ledger = run_account(prices, clock, targets['M1_main'], dates[1], 100, 0, metadata=metas['M1_main'])
    assert ledger.equity_end.tolist() == pytest.approx([100, 100, 90, 94])
    assert ledger.shares_end.iloc[1:].tolist() == pytest.approx([.5, .5, .5])
    assert ledger.cash_end.iloc[1:].tolist() == pytest.approx([50]*3)
    assert ledger.trade_notional.iloc[2:].sum() == 0
    assert ledger.new_weight.iloc[2] != .5


def test_common_start_is_cash_reset_with_first_executable_order():
    dates, prices, clock, p = fixture([100, 100, 120, 60, 72])
    targets, metas = model_targets(p, dates, .05, .0005)
    full = run_account(prices, clock, targets['M2_main'], dates[1], 100, 0, metadata=metas['M2_main'])
    fresh = run_account(prices, clock, targets['M2_main'], dates[3], 100, 0, metadata=metas['M2_main'])
    assert fresh.equity_end.iloc[0] == fresh.equity_before_trade.iloc[1] == 100
    assert fresh.old_weight.iloc[1] == 0
    assert fresh.shares_end.iloc[1] == pytest.approx(50/60)
    assert fresh.equity_end.iloc[1] != full.equity_end.iloc[3]
    assert fresh.source_information_session.iloc[1] == dates[2]


def test_one_more_session_delay_preserves_original_signal_and_no_early_profit():
    dates, prices, clock, p = fixture([100, 110, 121, 60.5], mu=[.02, -.1, .02, .02], q=[.05]*4)
    t, m = model_targets(p, dates, .05, 0)
    ledger = run_account(prices, clock, t['M0_main'], dates[1], 100, 0, metadata=m['M0_main'], delay=1)
    assert ledger.new_weight.tolist() == pytest.approx([0, 0, 1, 0])
    assert ledger.equity_end.tolist() == pytest.approx([100, 100, 100, 50])
    assert ledger.source_information_session.iloc[2] == dates[0]
    assert ledger.signal_information_session.iloc[2] == dates[1]


def test_gross_same_targets_does_not_change_mu_filter():
    dates, prices, clock, p = fixture([100, 100, 120], mu=[.0008]*3)
    base, _ = model_targets(p, dates, .05, .0005)
    zero_cost_scenario, _ = model_targets(p, dates, .05, 0)
    gross = run_account(prices, clock, base['M2_main'], dates[1], 100, 0)
    assert gross.equity_end.iloc[-1] == 100
    assert zero_cost_scenario['M2_main'].iloc[0] == .5


def test_metrics_include_initial_fee_and_exclude_anchor_from_exposure():
    dates, prices, clock, p = fixture([100]*7, q=[.05]*7)
    t, m = model_targets(p, dates, .05, .0005)
    ledger = run_account(prices, clock, t['M0_main'], dates[1], 100, .01, metadata=m['M0_main'])
    result = account_metrics(ledger, 100)
    assert result['max_drawdown'] == pytest.approx(1/1.01-1)
    assert result['mean_exposure'] == pytest.approx(5/6)
    assert result['worst_five_days'] == pytest.approx(1/1.01-1)


def test_bounded_input_never_converts_or_exposes_locked_outcome(tmp_path):
    p = tmp_path/'synthetic.csv'
    p.write_text('as_of_session,close,FORBIDDEN\n2023-12-29,100,unknown\n2024-01-02,not_a_number,secret_future\n')
    result = bounded_csv(p, ['as_of_session', 'close'])
    assert result.as_of_session.tolist() == ['2023-12-29']
    assert result.columns.tolist() == ['as_of_session', 'close']
    assert pd.to_numeric(result.close).item() == 100


def test_future_price_or_prediction_mutation_cannot_change_earlier_account():
    dates, prices, clock, p = fixture([100, 100, 90, 120, 130, 125])
    t, m = model_targets(p, dates, .05, .0005)
    original = run_account(prices, clock, t['M1_main'], dates[1], 100, .0005, metadata=m['M1_main'])
    prices.loc[4:, 'close'] *= 2
    prices.loc[4:, 'simple_return'] = [.8, -.5]
    p.loc[p.as_of_session.ge(dates[4]), 'mu5'] = -.1
    t, m = model_targets(p, dates, .05, .0005)
    changed = run_account(prices, clock, t['M1_main'], dates[1], 100, .0005, metadata=m['M1_main'])
    pd.testing.assert_frame_equal(original.iloc[:4], changed.iloc[:4])


def test_sensitivity_is_only_original_single_factors():
    import tomllib
    from pathlib import Path
    config = tomllib.loads((Path(__file__).parents[1]/'experiment.toml').read_text())
    grid = scenarios(config)
    assert len(grid) == 7
    baseline = grid[0]
    for row in grid[1:]:
        assert sum(row[k] != baseline[k] for k in ['budget', 'cost_rate', 'delay']) == 1
