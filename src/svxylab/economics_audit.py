"""P5独立逐笔复核：从已验收预测/价格代入公式，不调用生产映射或账本。"""

import csv
from hashlib import sha256
import json
import math

import numpy as np
import pandas as pd


def validate_economics(root, result):
    output = root / result['output_dir']
    files = result['inputs']['files']
    def bounded(name):
        values = []
        columns = {'prices':['as_of_session','close','split_factor','cash_dividend','capital_gain_distribution'],
                   'curve':['as_of_session','f1_settle','f2_settle'],
                   'predictions':['as_of_session','model','forecast_available','mu5','q90']}[name]
        with (root/files[name]).open(newline='') as stream:
            for row in csv.DictReader(stream):
                if row['as_of_session'] >= '2024-01-01':
                    break
                values.append({key:row[key] for key in columns})
        return values
    price = {r['as_of_session']: r for r in bounded('prices')}
    dates = list(price); positions = {d:i for i,d in enumerate(dates)}
    curve = {r['as_of_session']: r for r in bounded('curve')}
    predictions = {(r['model'], r['as_of_session']): r for r in bounded('predictions')}
    metrics = pd.read_csv(output/'account_metrics.csv', float_precision='round_trip')
    gross = pd.read_csv(output/'same_target_gross_net.csv', float_precision='round_trip')
    matches = pd.read_csv(output/'post_hoc_exposure_matches.csv', float_precision='round_trip')
    specs = [{'path':r.ledger_file, 'strategy':r.strategy, 'rate':r.cost_rate, 'filter_rate':r.cost_rate,
              'B':r.budget, 'delay':int(r.delay), 'constant':None, 'capital':r.initial_capital} for r in metrics.itertuples()]
    specs += [{'path':r.gross_ledger_file, 'strategy':r.strategy, 'rate':0, 'filter_rate':.0005, 'B':.05,
               'delay':0, 'constant':None, 'capital':100000.} for r in gross.itertuples()]
    specs += [{'path':r.ledger_file, 'strategy':r.controller, 'rate':.0005, 'filter_rate':.0005, 'B':.05,
               'delay':0, 'constant':r.post_hoc_fixed_target, 'capital':r.initial_capital} for r in matches.itertuples()]
    checked, maximum = 0, 0.
    for spec in specs:
        ledger = pd.read_csv(output/spec['path'], float_precision='round_trip')
        assert ledger.as_of_session.lt('2024-01-01').all()
        start = positions[ledger.as_of_session.iloc[0]]
        assert ledger.as_of_session.tolist() == dates[start:]
        cash, shares = spec['capital'], 0.
        expected_equity = []
        for i, row in enumerate(ledger.itertuples()):
            p = price[row.as_of_session]
            close = float(p['close']); split = float(p['split_factor'])
            shares /= split
            cash += shares*(float(p['cash_dividend'])+float(p['capital_gain_distribution'])) if i else 0
            stock = shares*close
            before = cash+stock
            target = None
            if i:
                source = dates[positions[row.as_of_session]-spec['delay']-1]
                if spec['constant'] is not None:
                    target = spec['constant']
                elif spec['strategy'].startswith('M'):
                    model, policy = spec['strategy'].split('_', 1)
                    p = predictions.get((model, source))
                    if p and p['forecast_available'] == 'True':
                        q, mu = float(p['q90']), float(p['mu5'])
                        target = 1. if q == 0 else min(1., spec['B']/q)
                        if policy == 'main' and mu <= 2*spec['filter_rate']:
                            target = 0.
                    if p:
                        assert row.source_information_session == source
                elif spec['strategy'] == 'cash':
                    target = 0.
                elif spec['strategy'].startswith('fixed_'):
                    target = float(spec['strategy'].split('_')[1])/100
                elif spec['strategy'] == 'curve':
                    p = curve[source]
                    if p['f1_settle'] and p['f2_settle']:
                        target = float(float(p['f2_settle']) > float(p['f1_settle']))
                assert pd.Timestamp(row.decision_at) < pd.Timestamp(row.execution_at)
                assert row.signal_information_session == dates[positions[row.as_of_session]-1]
            if target is None:
                assert math.isnan(row.target_weight)
                fee = 0.
                equity = before
                trade = 0.
            else:
                assert abs(row.target_weight-target) < 2e-12
                # 独立求交易额：买入/卖出各自解自融资现金方程，不调用rebalance。
                rate = spec['rate']
                if target*before >= stock:
                    trade = (target*before-stock)/(1+target*rate)
                else:
                    trade = (target*before-stock)/(1-target*rate)
                fee = abs(trade)*rate
                cash -= trade+fee
                shares += trade/close
                equity = cash+shares*close
            comparisons = [(row.equity_before_trade,before), (row.equity_end,equity), (row.cash_end,cash),
                           (row.shares_end*close,shares*close), (row.cost,fee), (row.signed_trade,trade)]
            error = max(abs(a-b) for a,b in comparisons)
            maximum = max(maximum, error)
            assert error < 2e-10*spec['capital'], (spec['path'],row.as_of_session,error)
            expected_equity.append(equity); checked += 1
        if spec['path'] in metrics.ledger_file.values:
            summary = metrics.loc[metrics.ledger_file.eq(spec['path'])].iloc[0]
            wealth = np.asarray(expected_equity)
            assert abs(summary.total_return-(wealth[-1]/spec['capital']-1)) < 2e-10
            assert abs(summary.max_drawdown-(wealth/np.maximum.accumulate(wealth)-1).min()) < 2e-10
            assert abs(summary.worst_five_days-min(wealth[5:]/wealth[:-5]-1)) < 2e-10
    accepted = json.loads((root/'runs/p4/acceptance.json').read_text())
    assert sha256((root/accepted['accepted_prediction_file']).read_bytes()).hexdigest() == accepted['accepted_prediction_sha256']
    for p,h in accepted['model_source_sha256'].items():
        assert sha256((root/p).read_bytes()).hexdigest() == h
    # 所有base共同账户在同一时点以同一资本重启；延迟差异另有相同起点基准。
    for scenario, rows in metrics.loc[metrics.scope.eq('common')].groupby('scenario'):
        assert rows.first_execution.nunique() == rows.anchor.nunique() == rows.initial_capital.nunique() == 1
        assert len(rows) == 12
    tail = pd.read_csv(output/'last_five_2023_execution_days.csv')
    assert tail.as_of_session.nunique() == 5 and tail.source_forecast_available.all()
    assert tail.new_weight.gt(0).any()
    risk = pd.read_csv(output/'risk_capacity_diagnostics.csv')
    assert risk.loc[~risk.path_complete_in_development, 'subsequent_entry_relative_loss'].isna().all()
    for row in risk.itertuples():
        i = positions[row.execution_session]
        assert row.path_complete_in_development == (i+5 < len(dates))
        if i+5 >= len(dates):
            continue
        wealth, smallest = 1., 1.
        previous_close = float(price[dates[i]]['close'])
        for date in dates[i+1:i+6]:
            p = price[date]
            close = float(p['close'])
            wealth *= (close+float(p['cash_dividend'])+float(p['capital_gain_distribution']))/(float(p['split_factor'])*previous_close)
            smallest = min(smallest,wealth)
            previous_close = close
        loss = 1-smallest
        assert abs(row.subsequent_entry_relative_loss-loss) < 2e-12
        assert abs(row.static_risk_capacity_times_loss-row.risk_capacity*loss) < 2e-12
    return {'method':'independent_from_accepted_predictions_and_prices_solve_trade_equations', 'accounts_checked':len(specs),
            'ledger_rows_checked':checked, 'max_cash_equity_trade_error_dollars':maximum,
            'target_source_execution_and_missing_holding_checked':True, 'common_accounts_restarted':True,
            'p4_predictions_and_model_hashes_unchanged':True, 'tail_forecasts_tradable_without_labels':True,
            'risk_diagnostic_paths_rebuilt_from_prices':True, 'no_locked_economic_outcomes_used':True, 'passed':True}
