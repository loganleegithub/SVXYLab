"""Finite, post-revelation E1 diagnostic. No strategy or production changes.

The causal layer takes only dated signal evidence. Prices and final-episode
annotations enter separate, downstream functions. Each hypothetical path is an
independent cash-funded account, never a re-entry portfolio.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import tomllib

import numpy as np
import pandas as pd

from svxylab.panic_retreat import ROOT, RULES, digest, fixed_trade, scan_signals, write_json

HORIZONS = (5, 10, 20)
CELLS = {'00': (0, 0), '10': (1, 0), '01': (0, 1), '11': (1, 1)}
METRICS = ['net_return', 'marked_return', 'mae', 'mfe', 'entry_loss', 'peak_drawdown',
           'net_drawdown', 'worst_day', 'fees', 'entry_price', 'exit_price',
           'entry_shares', 'exit_shares_before_sale', 'distribution_cash']


def load_inputs(root=ROOT):
    acceptance = json.loads((root / 'runs/panic_retreat/acceptance.json').read_text())
    old = root / acceptance['accepted_output_dir']
    identity = dict(acceptance['accepted_source_sha256'])
    identity.update(acceptance['accepted_artifact_sha256'])
    identity[acceptance['accepted_report']] = acceptance['accepted_report_sha256']
    previous = json.loads((root / acceptance['accepted_run']).with_name('inputs.json').read_text())
    identity.update(previous['sha256'])
    for name, expected in identity.items():
        if digest(root / name) != expected:
            raise ValueError(f'Accepted E1 input changed: {name}')
    cfg = tomllib.loads((root / 'panic_retreat.toml').read_text())
    market = pd.read_csv(old / 'signal_inputs.csv')
    price_name = next(p for p in identity if p.startswith('data/clean/p2/') and p.endswith('/SVXY_returns.csv'))
    clock_name = next(p for p in identity if p.startswith('data/clean/p2/') and p.endswith('/clock.csv'))
    prices, clock = pd.read_csv(root / price_name), pd.read_csv(root / clock_name)
    if not (market.as_of_session.tolist() == prices.as_of_session.tolist() == clock.as_of_session.tolist()):
        raise ValueError('Frozen market, prices and clock must align exactly')
    episodes, signals, daily = scan_signals(market, cfg)
    for name, actual in [('episodes', episodes), ('signals', signals), ('episode_daily', daily)]:
        expected = pd.read_csv(old / f'{name}.csv')
        pd.testing.assert_frame_equal(actual.reset_index(drop=True), expected[actual.columns],
                                      check_dtype=False, check_exact=False, atol=1e-12, rtol=1e-12)
    if len(episodes) != 7 or len(signals) != 35 or signals.t.isna().any():
        raise ValueError('This finite diagnostic requires the accepted seven episodes / 35 signals')
    return cfg, market, prices, clock, old, identity


def condition(row, rule):
    if rule not in ('B_RETREAT', 'C_RETREAT_VX'):
        return 'NOT_APPLICABLE'
    if not row['retreat_known']:
        return 'UNKNOWN'
    if not row['retreat_condition']:
        return 'FALSE'
    if rule == 'B_RETREAT':
        return 'TRUE'
    if not row['vx_complete']:
        return 'UNKNOWN'
    return 'TRUE' if row['vx_down'] else 'FALSE'


def causal_timeline(market, cfg):
    """No prices or retrospective annotations accepted by this API."""
    episodes, signals, daily = scan_signals(market, cfg)
    rows = []
    for ep in episodes.itertuples():
        observed = signals[signals.episode_id.eq(ep.episode_id)]
        triggers = {r.rule_id: int(r.t) for r in observed.itertuples() if pd.notna(r.t)}
        previous_b = previous_c = 'UNKNOWN'
        previous_peak = np.nan
        for r in daily[daily.episode_id.eq(ep.episode_id)].to_dict('records'):
            i = r['i']; b = condition(r, 'B_RETREAT'); c = condition(r, 'C_RETREAT_VX')
            prev_vix = market.vix.iloc[i-1] if i else np.nan
            delta = r['vix'] - prev_vix
            t = triggers.get('B_RETREAT')
            after = t is not None and i > t
            r.update(b_state=b, c_state=c, vix_change=delta,
                     vix_down_day=bool(np.isfinite(delta) and delta < 0),
                     vix_up_day=bool(np.isfinite(delta) and delta > 0),
                     new_running_high=bool(r['age'] > 0 and np.isfinite(previous_peak) and r['vix'] > previous_peak),
                     b_renewed=b == 'TRUE' and previous_b == 'FALSE',
                     c_renewed=c == 'TRUE' and previous_c == 'FALSE',
                     b_state_class=('SUSTAINED' if previous_b == 'TRUE' else
                                    'RENEWED' if previous_b == 'FALSE' else 'PRIOR_UNKNOWN') if b == 'TRUE' else b,
                     after_original_b=after, candidate=after and b == 'TRUE',
                     window_expired=r['age'] > cfg['window'],
                     window_end=r['age'] == cfg['window'],
                     one_trade_used=t is not None and i >= t,
                     lifecycle_end=bool(pd.notna(ep.end_i) and i == ep.end_i),
                     rearmed=False, in_lifecycle=True)
            for rule in RULES:
                tr = triggers.get(rule)
                # Even a pre-known mechanical schedule is only marked after its signal forms.
                r[f'{rule}_signal'] = tr is not None and i == tr
                r[f'{rule}_entry'] = tr is not None and i == tr + 1
                for h in HORIZONS:
                    r[f'{rule}_exit_H{h}'] = tr is not None and i == tr + 1 + h
            for h in HORIZONS:
                # Original overlap policy: candidate entry <= planned exit blocks re-entry.
                r[f'original_b_holding_H{h}'] = t is not None and t + 1 <= i <= t + 1 + h
                r[f'position_blocks_H{h}'] = after and i + 1 <= t + 1 + h
            rows.append(r)
            previous_b, previous_c, previous_peak = b, c, r['peak_so_far']
        if pd.notna(ep.end_i) and ep.end_i + 1 < len(market):
            i = int(ep.end_i) + 1
            # This row belongs to the preceding episode's display, not its candidate set.
            r = {key: False for key in rows[-1] if isinstance(rows[-1][key], (bool, np.bool_))}
            r.update(episode_id=ep.episode_id, i=i, as_of_session=market.as_of_session.iloc[i],
                     age=i-ep.s, vix=market.vix.iloc[i], rearmed=True, in_lifecycle=False,
                     b_state='NOT_APPLICABLE', c_state='NOT_APPLICABLE', b_state_class='NOT_APPLICABLE')
            rows.append(r)
    return episodes, signals, pd.DataFrame(rows)


def reference_states(market, s, last, cfg):
    """Original condition relative to original shock, also after lifecycle reset.

    Continued reference is diagnostic only; no new event or entry is created.
    """
    peak = 0.; known = True; rows = []
    for i in range(s, min(last + 1, len(market))):
        r = market.iloc[i]; x = r.vix
        if not np.isfinite(x) or x <= 0: known = False
        if known: peak = max(peak, x)
        row = {'i': i, 'as_of_session': r.as_of_session, 'vix': x,
               'peak_so_far': peak if known else np.nan, 'retreat_known': known,
               'retreat_condition': bool(known and x <= peak * (1-cfg['retreat']) + 1e-12),
               'vx_complete': bool(r.vx_complete), 'vx_down': bool(r.vx_down)}
        row.update(b_state=condition(row, 'B_RETREAT'), c_state=condition(row, 'C_RETREAT_VX'))
        rows.append(row)
    return pd.DataFrame(rows).set_index('i', drop=False)


def availability(signals, market, clock, cfg):
    rows = []
    for r in signals.itertuples():
        if pd.isna(r.t): continue
        t = int(r.t)
        states = reference_states(market, int(r.s), t+2, cfg)
        for lag in (1, 2):
            e = t + lag
            for stage, q in [('SIGNAL_FORMED', t), ('PRE_EXECUTION_CUTOFF', e-1), ('EXECUTION_FINAL_DAILY', e)]:
                if q >= len(market): continue
                st = condition(states.loc[q], r.rule_id)
                rows.append(dict(episode_id=r.episode_id, rule_id=r.rule_id, lag=lag,
                    stage=stage, observation_i=q, information_session=market.as_of_session.iloc[q],
                    entry_i=e, entry_session=clock.execution_session.iloc[e-1] if e-1 < len(clock) else None,
                    original_signal_session=r.signal_session, original_condition=st,
                    complete_new_daily_rows=q-t,
                    usable_before_this_fill=stage != 'EXECUTION_FINAL_DAILY',
                    state_observable_at_assumed_deadline=clock.decision_at.iloc[q],
                    execution_decision_at=clock.decision_at.iloc[e-1] if e-1 < len(clock) else None,
                    execution_at=clock.execution_at.iloc[e-1] if e-1 < len(clock) else None,
                    historical_publication_proven=False,
                    source_policy='ASSUMED_NEXT_SESSION_60MIN_BEFORE_CLOSE'))
    return pd.DataFrame(rows)


def trade(prices, clock, e, u, cfg, **identity):
    status, metrics, ledger = fixed_trade(prices, clock, e, u, cfg['capital'], cfg['cost'])
    date = lambda i: prices.as_of_session.iloc[i] if 0 <= i < len(prices) else None
    row = dict(identity, e=e, u=u, actual_holding_sessions=u-e,
               entry_session=date(e), exit_session=date(u), trade_status=status,
               cost_rate=cfg['cost'], decision_at=clock.decision_at.iloc[e-1] if e-1 < len(clock) else None,
               execution_at=clock.execution_at.iloc[e-1] if e-1 < len(clock) else None,
               **{k: metrics.get(k, np.nan) for k in METRICS})
    return row, ledger


def common_cash_path(row, ledger, prices, start, end, capital):
    """Explicit common cash start/end. Never fakes a missing live holding value."""
    rows = []; by_date = ledger.set_index('as_of_session') if len(ledger) else pd.DataFrame()
    for i in range(start, min(end+1, len(prices))):
        date = prices.as_of_session.iloc[i]
        cash_only = i < row['e'] or (i > row['u'] and row['trade_status'] == 'MATURE')
        if len(by_date) and date in by_date.index and i >= row['e']:
            r = by_date.loc[date]
            equity, shares, cash, fee = r.equity_end, r.shares_end, r.cash_end, r.cost
        elif cash_only:
            equity = capital if i < row['e'] else capital * (1+row['net_return'])
            shares, cash, fee = 0., equity, 0.
        else:
            equity = shares = cash = fee = np.nan
        rows.append(dict(trade_id=row['trade_id'], i=i, as_of_session=date,
                         equity_end=equity, shares_end=shares, cash_end=cash, cost=fee,
                         phase='CASH' if cash_only else 'TRADE_PATH'))
    return pd.DataFrame(rows)


def four_cells(signals, prices, clock, cfg):
    rows = []; ledgers = []; common = []
    for r in signals.itertuples():
        if pd.isna(r.t): continue
        e = int(r.t)+1
        for h in HORIZONS:
            for cell, (de, du) in CELLS.items():
                uid = f'{r.episode_id}_{r.rule_id}_H{h}_{cell}'
                row, ledger = trade(prices, clock, e+de, e+h+du, cfg,
                    episode_id=r.episode_id, rule_id=r.rule_id, signal_session=r.signal_session,
                    H=h, cell=cell, trade_id=uid, common_start_i=e-1, common_end_i=e+h+1)
                rows.append(row)
                if len(ledger): ledgers.append(ledger.assign(trade_id=uid))
                common.append(common_cash_path(row, ledger, prices, e-1, e+h+1, cfg['capital']))
    return pd.DataFrame(rows), pd.concat(ledgers, ignore_index=True), pd.concat(common, ignore_index=True)


def attribution(cells):
    rows = []
    for keys, g in cells.groupby(['episode_id', 'rule_id', 'H'], sort=True):
        r = g.set_index('cell').net_return
        a, b, c, d = (r[k] for k in CELLS)
        rows.append(dict(zip(['episode_id', 'rule_id', 'H'], keys)) | dict(
            R00=a, R10=b, R01=c, R11=d, total_change=d-a,
            entry_first=b-a, entry_after_exit=d-c, exit_first=c-a, exit_after_entry=d-b,
            entry_contribution=((b-a)+(d-c))/2, exit_contribution=((c-a)+(d-b))/2,
            interaction=d-b-c+a,
            status='COMPLETE' if g.trade_status.eq('MATURE').all() else 'INCOMPLETE'))
    return pd.DataFrame(rows)


def candidates(timeline, prices, clock, cfg):
    coverage = timeline[timeline.in_lifecycle & timeline.after_original_b].copy()
    rows = []; ledgers = []
    for r in coverage[coverage.candidate].itertuples():
        for h in HORIZONS:
            uid = f'{r.episode_id}_LATER_{r.as_of_session}_H{h}'
            row, ledger = trade(prices, clock, r.i+1, r.i+1+h, cfg,
                trade_id=uid, episode_id=r.episode_id, information_i=r.i,
                information_session=r.as_of_session, H=h, state_class=r.b_state_class,
                window='OUTSIDE' if r.window_expired else 'INSIDE',
                window_blocks=r.window_expired, one_trade_blocks=r.one_trade_used,
                position_blocks=getattr(r, f'position_blocks_H{h}'), vix_down_day=r.vix_down_day)
            rows.append(row)
            if len(ledger): ledgers.append(ledger.assign(trade_id=uid))
    cols = ['trade_id','episode_id','information_i','information_session','H','state_class','window',
            'window_blocks','one_trade_blocks','position_blocks','vix_down_day','trade_status', *METRICS]
    return coverage, pd.DataFrame(rows) if rows else pd.DataFrame(columns=cols), \
        pd.concat(ledgers, ignore_index=True) if ledgers else pd.DataFrame()


def candidate_summary(episodes, coverage, paths):
    rows = []
    for ep in episodes.episode_id:
        for window in ('INSIDE', 'OUTSIDE'):
            pool = coverage[coverage.episode_id.eq(ep) & coverage.window_expired.eq(window == 'OUTSIDE')]
            for cls in ('SUSTAINED', 'RENEWED', 'PRIOR_UNKNOWN'):
                for h in HORIZONS:
                    g = paths[paths.episode_id.eq(ep) & paths.window.eq(window) & paths.state_class.eq(cls) & paths.H.eq(h)]
                    mature = g[g.trade_status.eq('MATURE')]
                    rows.append(dict(episode_id=ep, window=window, state_class=cls, H=h,
                        observed_days=len(pool), false_days=int(pool.b_state.eq('FALSE').sum()),
                        unknown_condition_days=int(pool.b_state.eq('UNKNOWN').sum()), candidate_days=len(g),
                        mature=len(mature), price_gaps=int(g.trade_status.eq('PRICE_GAP').sum()),
                        immature=int(g.trade_status.isin(['ENTRY_PENDING','OPEN_MARKED']).sum()),
                        status='NO_CANDIDATE' if g.empty else 'COMPLETE' if len(g)==len(mature) else 'INCOMPLETE',
                        positive=int(mature.net_return.gt(0).sum()), negative=int(mature.net_return.lt(0).sum()),
                        mean_return=mature.net_return.mean(), median_return=mature.net_return.median(),
                        min_return=mature.net_return.min(), max_return=mature.net_return.max(),
                        max_entry_loss=mature.entry_loss.max(), max_peak_drawdown=mature.peak_drawdown.max()))
    return pd.DataFrame(rows)


def holding_states(signals, episodes, market, clock, cfg):
    rows = []
    ends = episodes.set_index('episode_id').end_i.to_dict()
    for r in signals.itertuples():
        if pd.isna(r.t): continue
        e = int(r.t)+1
        states = reference_states(market, int(r.s), e+max(HORIZONS), cfg)
        for h in HORIZONS:
            for q in range(e, min(e+h+1, len(market))):
                rows.append(dict(episode_id=r.episode_id, rule_id=r.rule_id, H=h,
                    information_i=q, information_session=market.as_of_session.iloc[q],
                    original_condition=condition(states.loc[q], r.rule_id),
                    earliest_action_i=q+1, earliest_action_session=clock.execution_session.iloc[q],
                    assumed_observable_at=clock.decision_at.iloc[q],
                    earliest_execution_at=clock.execution_at.iloc[q],
                    by_original_exit=q+1 <= e+h,
                    reset_already_observed=bool(pd.notna(ends[r.episode_id]) and q >= ends[r.episode_id]),
                    historical_publication_proven=False))
    return pd.DataFrame(rows)


def holding_outcomes(states, cells, ledgers, cfg):
    rows = []
    for r in cells[cells.cell.eq('00')].itertuples():
        g = states[states.episode_id.eq(r.episode_id) & states.rule_id.eq(r.rule_id) & states.H.eq(r.H)]
        false = g[g.original_condition.eq('FALSE')]
        base = dict(episode_id=r.episode_id, rule_id=r.rule_id, H=r.H,
                    entry_session=r.entry_session, exit_session=r.exit_session, net_return=r.net_return,
                    condition_applicable=r.rule_id in ('B_RETREAT','C_RETREAT_VX'),
                    unknown_days=int(g.original_condition.eq('UNKNOWN').sum()),
                    first_false_session=None, earliest_action_session=None,
                    false_after_reset=False, actionable_before_planned_exit=False,
                    pnl_at_information=np.nan, pnl_through_earliest_action=np.nan,
                    remaining_original_pnl=np.nan, final_pnl=cfg['capital']*r.net_return,
                    status='NOT_APPLICABLE' if r.rule_id not in ('B_RETREAT','C_RETREAT_VX') else
                           'NO_FALSE_OBSERVED' if false.empty else 'FALSE_OBSERVED')
        if len(false):
            f = false.iloc[0]
            led = ledgers[ledgers.trade_id.eq(r.trade_id)].set_index('as_of_session')
            obs = f.information_session
            action = f.earliest_action_session
            base.update(first_false_session=obs, earliest_action_session=action,
                        false_after_reset=bool(f.reset_already_observed),
                        actionable_before_planned_exit=bool(f.by_original_exit))
            if obs in led.index:
                base['pnl_at_information'] = led.loc[obs].equity_end-cfg['capital']
            if f.by_original_exit and action in led.index and r.trade_status == 'MATURE':
                through = led.loc[action].equity_end-cfg['capital']
                base.update(pnl_through_earliest_action=through,
                            remaining_original_pnl=cfg['capital']*r.net_return-through)
        rows.append(base)
    return pd.DataFrame(rows)


def holding_segments(cells, ledgers, capital):
    rows = []
    for keys, g in cells[cells.cell.eq('00')].groupby(['episode_id','rule_id'], sort=True):
        by_h = g.set_index('H'); long = by_h.loc[20]
        path = ledgers[ledgers.trade_id.eq(long.trade_id)].set_index('holding_day')
        for a, b in [(0,5),(5,10),(10,20)]:
            if a not in path.index or b not in path.index:
                pnl = fee_delta = delta = residual = np.nan
            else:
                pnl = path.loc[a+1:b].holding_pnl.sum()
                if a == 0:
                    fee_delta = by_h.loc[b].fees
                    delta = by_h.loc[b].net_return*capital
                else:
                    fee_delta = by_h.loc[b].fees-by_h.loc[a].fees
                    delta = (by_h.loc[b].net_return-by_h.loc[a].net_return)*capital
                residual = delta-(pnl-fee_delta)
            rows.append(dict(episode_id=keys[0], rule_id=keys[1], from_H=a, to_H=b,
                start_session=path.loc[a].as_of_session if a in path.index else None,
                end_session=path.loc[b].as_of_session if b in path.index else None,
                holding_pnl=pnl, extra_fees=fee_delta, net_return_change=delta/capital,
                identity_residual_dollars=residual,
                status='COMPLETE' if np.isfinite(residual) else 'INCOMPLETE'))
    return pd.DataFrame(rows)


def independent_audit(trades, ledgers, prices, capital):
    """Scalar share/cash reconstruction, never calls E1 ledger/rebalance."""
    rows = []; indexed = {key: g for key,g in ledgers.groupby('trade_id', sort=False)}
    for r in trades.itertuples():
        if r.trade_status != 'MATURE': continue
        q = capital/(1+r.cost_rate)/prices.close.iloc[r.e]
        cash = 0.; fees = q*prices.close.iloc[r.e]*r.cost_rate
        shape = []; max_error = 0.; rows_checked = 0
        saved = indexed[r.trade_id].set_index('as_of_session')
        for i in range(r.e, r.u+1):
            p = prices.iloc[i]
            if i > r.e:
                q /= p.split_factor
                cash += q*(p.cash_dividend+p.capital_gain_distribution)
            shape.append(q*p.close+cash)
            if i == r.u:
                sale_fee = q*p.close*r.cost_rate
                fees += sale_fee; cash += q*p.close-sale_fee; q = 0.
            actual = saved.loc[p.as_of_session]
            expected = np.array([q, cash, q*p.close+cash])
            error = np.max(np.abs(expected-actual[['shares_end','cash_end','equity_end']].to_numpy(float)))
            max_error = max(max_error, error); rows_checked += 1
        shape = np.asarray(shape)
        errors = [abs(cash/capital-1-r.net_return), abs(fees-r.fees)/capital,
                  abs(max(0.,1-shape.min()/shape[0])-r.entry_loss),
                  abs((1-shape/np.maximum.accumulate(shape)).max()-r.peak_drawdown)]
        if max_error > 2e-7 or max(errors) > 2e-11:
            raise AssertionError(f'Independent accounting mismatch: {r.trade_id}')
        rows.append(dict(trade_id=r.trade_id, rows_checked=rows_checked,
                         max_account_error=max_error, max_metric_error=max(errors)))
    return pd.DataFrame(rows)


def original_reconciliation(cells, old):
    rows = []
    for name, expected, actual in [
        ('E1_MAIN_105', pd.read_csv(old/'trades.csv'), cells[cells.cell.eq('00')]),
        ('E1_LAG2_H10_35', pd.read_csv(old/'execution_trades.csv').query('scenario == "LAG_2"'),
         cells[cells.cell.eq('11') & cells.H.eq(10)])]:
        keys = ['episode_id','rule_id','H']
        a, b = expected.set_index(keys).sort_index(), actual.set_index(keys).sort_index()
        pd.testing.assert_index_equal(a.index,b.index)
        for col in ['e','u','entry_session','exit_session','trade_status','decision_at','execution_at']:
            if a[col].tolist() != b[col].tolist(): raise AssertionError((name,col))
        error = np.nanmax(np.abs(a[METRICS].to_numpy(float)-b[METRICS].to_numpy(float)))
        if error > 2e-8: raise AssertionError((name,error))
        rows.append(dict(reference=name,trades=len(a),max_numeric_error=error))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description='E1有限补充诊断：不修改机制')
    parser.add_argument('--open', action='store_true')
    args = parser.parse_args()
    cfg, market, prices, clock, old, identity = load_inputs()
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    run = ROOT/'runs/panic_retreat_diagnostic'/run_id
    out = ROOT/'data/clean/panic_retreat_diagnostic'/run_id
    run.mkdir(parents=True); out.mkdir(parents=True)
    record = dict(task='E1_SUPPLEMENTAL_DIAGNOSTIC', history_attribute='揭示后历史研究；不是新盲测或独立OOS',
        original_output=str(old.relative_to(ROOT)), input_sha256=identity,
        original_config=cfg, frozen_events=7, frozen_signals=35, horizons=HORIZONS,
        cells=CELLS, common_interval='cash at e-1 through u+1; cash after exit',
        candidate_policy='Every original-B-true date strictly after original B signal through lifecycle end; no best-date selection or portfolio',
        state_classes='TRUE after FALSE=RENEWED; TRUE after TRUE=SUSTAINED; previous UNKNOWN separate',
        attribution='Two change orders averaged; entry + exit = total; interaction shown separately; condition state not additive attribution',
        holding_policy='First FALSE from entry final daily through scheduled exit final daily; action next session; original ledger P&L only; no liquidation simulation',
        segment_policy='Same initial shares; additive dollar holding P&L less incremental actual exit fees; not sum of percentage interval returns',
        availability='Original assumed next-session deadline; lag1 no new completed daily observation; actual historical publishing/receipt timestamps absent',
        future_peak_policy='Downstream retrospective chart annotation only; absent from causal APIs',
        uncertainty='Overlapping candidate days are descriptive; no independent-day inference; no new significance claims',
        stop='Three unranked evidence cards; user selection and acceptance; no mechanism, commit or push')
    write_json(run/'experiment_record.json', record)
    print(f'计算前口径已保存：{run.relative_to(ROOT)}', flush=True)
    episodes, signals, timeline = causal_timeline(market, cfg)
    info = availability(signals, market, clock, cfg)
    states = holding_states(signals, episodes, market, clock, cfg)
    # Price-free outputs are materialized before any new outcome calculation.
    tables = dict(episodes=episodes, signals=signals, timeline=timeline, availability=info, holding_states=states)
    for name, df in tables.items(): df.to_csv(out/f'{name}.csv', index=False)
    cells, cell_ledgers, common = four_cells(signals, prices, clock, cfg)
    info_index = info.set_index(['episode_id','rule_id','lag','stage'])
    for column, stage in [('signal_state','SIGNAL_FORMED'),('pre_execution_state','PRE_EXECUTION_CUTOFF'),
                          ('fill_day_final_state','EXECUTION_FINAL_DAILY')]:
        cells[column] = [info_index.loc[(r.episode_id,r.rule_id,1 if r.cell in ('00','01') else 2,stage),
                                        'original_condition'] for r in cells.itertuples()]
    cells['new_complete_daily_rows_before_fill'] = cells.cell.isin(['10','11']).astype(int)
    coverage, later, later_ledgers = candidates(timeline, prices, clock, cfg)
    attr = attribution(cells)
    segments = holding_segments(cells, cell_ledgers, cfg['capital'])
    if not np.allclose(attr.entry_contribution+attr.exit_contribution, attr.total_change, atol=1e-14, equal_nan=True):
        raise AssertionError('Attribution does not add up')
    if segments.identity_residual_dollars.abs().max() > 2e-7: raise AssertionError('Holding segments do not reconcile')
    audit = independent_audit(pd.concat([cells,later],ignore_index=True),
                              pd.concat([cell_ledgers,later_ledgers],ignore_index=True),prices,cfg['capital'])
    reconciliation = original_reconciliation(cells, old)
    tables.update(coverage_days=coverage, candidates=later, candidate_ledgers=later_ledgers,
        candidate_summary=candidate_summary(episodes,coverage,later), four_cells=cells,
        cell_ledgers=cell_ledgers, common_paths=common, attribution=attr, holding_segments=segments,
        holding_outcomes=holding_outcomes(states,cells,cell_ledgers,cfg),
        independent_audit=audit, original_reconciliation=reconciliation)
    for name, df in tables.items(): df.to_csv(out/f'{name}.csv', index=False)
    validation = dict(engineering='COMPUTED; ordinary tests/reproducibility/page checks recorded separately at final delivery',
        research='Descriptive diagnostic complete; no mechanism tested or selected',
        coverage=dict(sessions=len(market),start=market.as_of_session.iloc[0],end=market.as_of_session.iloc[-1],
            episodes=len(episodes),original_signals=len(signals),four_cells=len(cells),
            later_dates=int(timeline.candidate.sum()),later_paths=len(later),
            candidate_statuses=later.trade_status.value_counts().to_dict()),
        audited_accounts=len(audit),audited_rows=int(audit.rows_checked.sum()),
        max_account_error=audit.max_account_error.max(),
        reconciliation=reconciliation.to_dict('records'),
        attribution_max_error=(attr.entry_contribution+attr.exit_contribution-attr.total_change).abs().max(),
        segment_max_error_dollars=segments.identity_residual_dollars.abs().max(),
        old_input_files_unchanged=all(digest(ROOT/p)==h for p,h in identity.items()),
        historical_actual_publication_proven=False,downloaded_data=False,
        mechanism_selected=None,mechanism_executed=False,commit_or_push=False)
    if not validation['old_input_files_unchanged']: raise AssertionError('Old input changed')
    write_json(out/'validation.json',validation)
    from svxylab.panic_retreat_diagnostic_report import render_report
    report = render_report(ROOT,out,run,tables,market,prices,cfg,validation)
    opened = subprocess.run(['open',str(report)],capture_output=True,text=True).returncode if args.open else None
    result = dict(output_dir=str(out.relative_to(ROOT)),report=str(report.relative_to(ROOT)),
                  open_exit_code=opened,validation=validation,
                  source_sha256={str(p.relative_to(ROOT)):digest(p) for p in sorted((ROOT/'src/svxylab').glob('panic_retreat_diagnostic*.py'))},
                  csv_sha256={p.name:digest(p) for p in sorted(out.glob('*.csv'))})
    write_json(run/'run.json',result)
    write_json(ROOT/'runs/panic_retreat_diagnostic/latest.json',dict(run=str(run.relative_to(ROOT)),**result))
    print(json.dumps(result,ensure_ascii=False,default=str,indent=2))


if __name__ == '__main__':
    main()
