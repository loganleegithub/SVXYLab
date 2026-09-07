"""M2 D1-D4：揭示后有界机制诊断；原预测、训练身份及账本只读。

所有新口径见 runs/m2_mechanism/experiment_record.json；D5没有执行入口。
"""
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json
import shutil
import tomllib

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.stats import skew, kurtosis
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from svxylab.diagnostics_audit import matrix
from svxylab.economics import run_account, account_metrics
from svxylab.features import CORE_IDS
from svxylab.features_report import write_json, verify_hashes
from svxylab.models import Design
from svxylab.prediction_data import training_rows
from svxylab.release import read_json, digest, stage_output, verify_release

COST = .0005
BUDGET = .05
CAPITAL = 100000.
PERIODS = [('development', '2020-09-11', '2023-12-29'),
           ('revealed', '2024-01-02', '2026-09-04')]
MARGIN_EDGES = [-np.inf, -.01, -.005, -.0025, -.001, 0, .001, .0025, .005, .01, np.inf]


def read_csv(path, **kwargs):
    return pd.read_csv(path, float_precision='round_trip', **kwargs)


def save_csv(output, name, rows, **kwargs):
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(output / (name + '.csv'), index=False, **kwargs)
    return frame


def periods(frame, column='decision_session'):
    yield 'all', 'all', frame
    for kind, labels in [('year', frame[column].str[:4]),
                         ('quarter', pd.to_datetime(frame[column]).dt.to_period('Q').astype(str))]:
        for label, group in frame.groupby(labels, sort=True):
            yield kind, label, group


def risk_capacity(q, budget=BUDGET):
    q = np.asarray(q, dtype=float)
    if not np.isfinite(q).all() or np.any((q < 0) | (q > 1)):
        raise ValueError('Q90必须为原发布值[0,1]')
    result = np.ones(q.shape)
    np.divide(budget, q, out=result, where=q > 0)
    return np.minimum(1., result)


def switch_terms(g0, r0, g1, r1, old_actual, pretrade):
    """精确权重恒等式；有方向的交互项保留，不按绝对金额强制分摊。"""
    dg, dr = float(g1) - float(g0), r1 - r0
    return {'gate_component': r0 * dg, 'risk_component': float(g0) * dr,
            'gate_risk_interaction': dg * dr,
            'prior_target_minus_old_actual': float(g0) * r0 - old_actual,
            'price_drift_rebalance': old_actual - pretrade}


def log_paths(total_return):
    """索引为信息日t；当日执行不被当作可交易比较。"""
    z = np.log(total_return.astype(float))
    daily = z.diff()
    out = pd.DataFrame(index=z.index)
    out['past5_log'] = z - z.shift(5)
    out['pre_execution_log'] = z.shift(-1) - z
    out['after_day1_log'] = z.shift(-2) - z.shift(-1)
    out['after_rest4_log'] = z.shift(-6) - z.shift(-2)
    out['after_full5_log'] = z.shift(-6) - z.shift(-1)
    out['V5'] = daily.pow(2).rolling(5, min_periods=5).mean().shift(-6)
    return out


def offset_target(target, clock, start, offset, end):
    """完整股票日历每5日给一次新目标；中间NaN令实际份额保持。"""
    execution = clock.execution_session
    dates = clock.as_of_session.tolist()
    start_pos = dates.index(start) + offset
    actual_start = dates[start_pos]
    positions = np.arange(len(dates)) + 1
    due = (positions >= start_pos) & ((positions - start_pos) % 5 == 0) & execution.le(end)
    return target.where(due.to_numpy()), actual_start


def ridge_system(X, y, alpha):
    Z = np.column_stack([np.ones(len(X)), X])
    penalty = np.diag(np.r_[0., np.full(X.shape[1], alpha)])
    A = Z.T @ Z + penalty
    inverse = np.linalg.solve(A, np.eye(A.shape[0]))
    theta = inverse @ Z.T @ y
    return theta, inverse, Z


def hac_prediction_se(Z, residual, V, inverse, positions, lag):
    """条件于原设计/alpha的Newey-West方差；用真实日历滞后，不压缩缺失日。"""
    weights = V @ inverse @ Z.T * residual[None, :]
    variance = np.sum(weights ** 2, axis=1)
    lookup = {int(p): i for i, p in enumerate(positions)}
    for k in range(1, lag + 1):
        left, right = [], []
        for i, p in enumerate(positions):
            if int(p) - k in lookup:
                left.append(i); right.append(lookup[int(p) - k])
        if left:
            variance += 2 * (1 - k / (lag + 1)) * np.sum(weights[:, left] * weights[:, right], axis=1)
    if variance.min(initial=0) < -1e-12:
        raise ValueError('HAC方差出现实质负值')
    return np.sqrt(np.maximum(variance, 0))


def training_clusters(train):
    marked = train.loc[train.Y10.eq(1) | train.R5.abs().ge(.10)]
    events = []
    for date, row in marked.iterrows():
        if not events or row.execution_session > events[-1]['end']:
            events.append({'start': row.execution_session, 'end': row.label_end_session, 'members': [date]})
        else:
            events[-1]['end'] = max(events[-1]['end'], row.label_end_session)
            events[-1]['members'].append(date)
    return events


def distribution(values):
    x = np.asarray(values, float); x = x[np.isfinite(x)]
    if not len(x):
        return {'rows': 0}
    energy = x ** 2
    top = max(1, int(np.ceil(.01 * len(x))))
    qs = np.quantile(x, [0, .01, .05, .25, .5, .75, .95, .99, 1])
    return {'rows': len(x), 'mean': x.mean(), 'std': x.std(),
            'skewness': skew(x, bias=False) if x.std() > 0 else np.nan,
            'excess_kurtosis': kurtosis(x, bias=False) if x.std() > 0 else np.nan,
            'largest1pct_squared_share': np.sort(energy)[-top:].sum() / energy.sum() if energy.sum() else 0,
            **dict(zip(['min','p01','p05','p25','median','p75','p95','p99','max'], qs, strict=True))}


def load_inputs(root, output):
    baseline = read_json(root / 'runs/m2_mechanism/baseline.json')
    verify_hashes(root, baseline['preserved_sha256'])
    verify_release(root)
    config = tomllib.loads((root / 'experiment.toml').read_text())
    paths = {s: stage_output(root, s) for s in ['p2','p3','p4','p5','p6','p7']}
    paths['n1'] = root / read_json(root / 'runs/n1/acceptance.json')['accepted_output_dir']
    p2, p3 = paths['p2'], paths['p3']
    prices = read_csv(p2 / 'SVXY_returns.csv')
    clock = read_csv(p2 / 'clock.csv')
    features = read_csv(p3 / 'core_features.csv').set_index('as_of_session')
    labels = read_csv(p2 / 'labels.csv').set_index('as_of_session')
    available = read_csv(p3 / 'feature_availability.csv').set_index('as_of_session')
    frame = features.join(labels).join(available[['assumed_available_at']])
    for k in ['information_close_at','decision_at','execution_at','label_matures_at','label_available_at','assumed_available_at']:
        frame[k] = pd.to_datetime(frame[k], utc=True)
    assert frame.index.tolist() == prices.as_of_session.tolist() == clock.as_of_session.tolist()
    assert prices.as_of_session.min() >= '2019-01-01' and prices.as_of_session.max() == '2026-09-04'
    path = log_paths(prices.set_index('as_of_session').total_return_index)
    frame = frame.join(path)
    np.testing.assert_allclose(np.expm1(frame.after_full5_log), frame.R5, atol=2e-14, equal_nan=True)
    predictions = pd.concat([read_csv(paths['p4'] / 'predictions.csv').assign(source_stage='p4'),
                             read_csv(paths['p7'] / 'core/predictions.csv').assign(source_stage='p7')], ignore_index=True)
    assert not predictions.duplicated(['model','as_of_session']).any()
    predictions['scope'] = np.where(predictions.source_stage.eq('p4'), 'development', 'revealed')
    save_csv(output, 'original_prediction_service', predictions)
    frame.to_csv(output / 'complete_information_and_paths.csv')
    prices.to_csv(output / 'price_path.csv', index=False)
    return frame, prices, clock, predictions, config, paths


def audit_account(ledger, target, prices, clock, rate=COST):
    """独立按成交额的现金方程解股数，不调用rebalance核算。"""
    px = prices.set_index('as_of_session'); date_pos = {d:i for i,d in enumerate(prices.as_of_session)}
    cash, shares, previous_equity = CAPITAL, 0., CAPITAL
    maximum = 0.
    for i, row in enumerate(ledger.itertuples()):
        p = px.loc[row.as_of_session]
        shares /= p.split_factor
        if i:
            cash += shares * (p.cash_dividend + p.capital_gain_distribution)
        stock = shares * p.close; before = cash + stock; fee = 0.
        if i:
            position = date_pos[row.as_of_session]
            signal = prices.as_of_session.iloc[position-1]
            assert clock.execution_session.iloc[position-1] == row.as_of_session
            w = target.loc[signal]
            if pd.notna(w):
                # Trade d satisfies (stock+d)/(before-c|d|)=w.
                signed = (w * before - stock) / (1 + rate*w if w*before >= stock else 1-rate*w)
                fee = rate * abs(signed)
                shares += signed / p.close
                cash -= signed + fee
        equity = shares*p.close + cash
        maximum = max(maximum, abs(equity-row.equity_end), abs(cash-row.cash_end),
                      abs(shares-row.shares_end), abs(fee-row.cost))
        assert abs(row.portfolio_return - (equity/previous_equity-1)) < 2e-11
        assert cash >= -1e-8 and shares >= -1e-9
        previous_equity = equity
    if maximum > 2e-7:
        raise ValueError(f'独立账户核验失败 {maximum}')
    return {'rows': len(ledger), 'max_amount_error': maximum}


def d1_accounts(frame, prices, clock, predictions, paths, output):
    (output / 'ledgers').mkdir()
    metrics, audits, chains, switches, spells, factorial, daily_bridges = [], [], [], [], [], [], []
    targets_all, ledgers_all = {}, {}
    for scope, start, end in PERIODS:
        px = prices.loc[prices.as_of_session.le(end)].reset_index(drop=True)
        cl = clock.iloc[:len(px)].copy()
        idx = pd.Index(px.as_of_session, name='as_of_session')
        preds = {}
        for m in ['M0','M2']:
            p = predictions.loc[predictions.model.eq(m) & predictions.scope.eq(scope) & predictions.forecast_available].copy()
            preds[m] = p.set_index('as_of_session').reindex(idx)
        targets = {}
        for mu, q in [('M0','M0'),('M2','M0'),('M0','M2'),('M2','M2')]:
            good = preds[mu].forecast_available.fillna(False) & preds[q].forecast_available.fillna(False)
            t = pd.Series(np.nan, index=idx)
            t.loc[good] = (preds[mu].loc[good,'mu5'].gt(2*COST).to_numpy() * risk_capacity(preds[q].loc[good,'q90']))
            targets[f'mu{mu[-1]}_q{q[-1]}'] = t
        for m in ['M0','M2']:
            good = preds[m].forecast_available.fillna(False)
            t = pd.Series(np.nan, index=idx)
            t.loc[good] = risk_capacity(preds[m].loc[good,'q90'])
            targets[m+'_risk'] = t
        for name, t in targets.items():
            ledger = run_account(px, cl, t, start, CAPITAL, COST)
            audits.append({'scope':scope,'account':name,**audit_account(ledger,t,px,cl)})
            ledger_file = f'ledgers/D1_{scope}_{name}.csv'
            ledger.to_csv(output/ledger_file,index=False)
            metrics.append({'scope':scope,'account':name,'ledger_file':ledger_file,**account_metrics(ledger,CAPITAL)})
            targets_all[scope,name] = t
            ledgers_all[scope,name] = ledger
            if name not in ['mu0_q0','mu2_q2','M0_risk','M2_risk']:
                continue
            oldname = {'mu0_q0':'M0_main','mu2_q2':'M2_main','M0_risk':'M0_risk_only','M2_risk':'M2_risk_only'}[name]
            olddir = paths['p5'] if scope=='development' else paths['p7']/'sensitivity'
            original = read_csv(olddir/f'ledgers/common_base_{oldname}.csv')
            fields = ['old_weight','target_weight','cost','equity_end','shares_end','cash_end','portfolio_return']
            assert original.as_of_session.tolist() == ledger.as_of_session.tolist()
            np.testing.assert_allclose(original[fields],ledger[fields],atol=2e-7,rtol=2e-12,equal_nan=True)
            if name not in ['mu0_q0','mu2_q2']:
                continue
            model = 'M'+name[2]
            p = predictions.loc[predictions.model.eq(model)&predictions.scope.eq(scope)&predictions.forecast_available].copy()
            l = ledger.iloc[1:].rename(columns={'as_of_session':'decision_session'})
            chain = l.merge(p.drop(columns=['execution_at']),on='decision_session',validate='one_to_one')
            chain = chain.merge(frame[CORE_IDS].reset_index(),on='as_of_session',validate='one_to_one')
            chain['account'] = name
            chain['mu_margin'] = chain.mu5-2*COST
            chain['gate'] = chain.mu_margin.gt(0)
            chain['risk_capacity'] = risk_capacity(chain.q90)
            chain['gate_crossing'] = chain.gate.ne(chain.gate.shift(1))
            chain.loc[0,'gate_crossing'] = False
            chain['fit_updated'] = chain.fit_id.ne(chain.fit_id.shift(1))
            chain['days_since_fit'] = np.arange(len(chain))-np.maximum.accumulate(np.where(chain.fit_updated,np.arange(len(chain)),0))
            update_pos = np.flatnonzero(chain.fit_updated)
            chain['within2_model_update'] = np.min(np.abs(np.arange(len(chain))[:,None]-update_pos),axis=1)<=2
            chain['distribution_conflict'] = ((chain.q90.lt(.10)&chain.p10.gt(.10))|(chain.q90.gt(.10)&chain.p10.lt(.10)))
            chain['positive_mu_and_p10_over10pct'] = chain.gate & chain.p10.gt(.10)
            chain['target_change'] = chain.target_weight.diff()
            chain['actual_rebalance_gap'] = chain.target_weight-chain.pre_trade_weight
            for i, row in chain.iterrows():
                if i==0:
                    terms={k:0. for k in ['gate_component','risk_component','gate_risk_interaction','prior_target_minus_old_actual','price_drift_rebalance']}
                    terms['startup_component'] = row.actual_rebalance_gap
                else:
                    previous=chain.iloc[i-1]
                    terms=switch_terms(previous.gate,previous.risk_capacity,row.gate,row.risk_capacity,row.old_weight,row.pre_trade_weight)
                    terms['startup_component']=0.
                assert abs(sum(terms.values())-row.actual_rebalance_gap)<2e-12
                for k,v in terms.items():chain.loc[i,k]=v
            switches.append(chain.loc[chain.trade_notional.gt(1e-8)].copy())
            held = chain.new_weight.gt(1e-12)
            group = held.ne(held.shift()).cumsum()
            for _, block in chain.loc[held].groupby(group):
                i,j=block.index[0],block.index[-1]
                closed=j+1<len(chain)
                spells.append({'scope':scope,'model':model,'entry':block.decision_session.iloc[0],
                    'exit':chain.decision_session.iloc[j+1] if closed else '', 'last_mark':block.decision_session.iloc[-1],
                    'sessions':int(j-i+1),'right_censored':not closed,
                    'entry_margin':block.mu_margin.iloc[0],'entry_weight':block.new_weight.iloc[0]})
            chains.append(chain)
        w = {n:ledgers_all[scope,n].equity_end.iloc[-1]/CAPITAL-1 for n in targets}
        factorial.append({'scope':scope, 'mu_effect_with_q0':w['mu2_q0']-w['mu0_q0'],
            'mu_effect_with_q2':w['mu2_q2']-w['mu0_q2'],
            'q_effect_with_mu0':w['mu0_q2']-w['mu0_q0'],
            'q_effect_with_mu2':w['mu2_q2']-w['mu2_q0'],
            'interaction':w['mu2_q2']-w['mu2_q0']-w['mu0_q2']+w['mu0_q0'],
            'joint_difference':w['mu2_q2']-w['mu0_q0']})
        ref = ledgers_all[scope,'mu0_q0'].iloc[1:]
        for name in ['mu2_q0','mu0_q2','mu2_q2']:
            candidate=ledgers_all[scope,name].iloc[1:]
            future=np.r_[np.cumprod((1+ref.portfolio_return.to_numpy())[::-1])[::-1][1:],1.]
            holding=(candidate.old_weight.to_numpy()-ref.old_weight.to_numpy())*ref.asset_return.to_numpy()
            friction=ref.cost.to_numpy()/ref.equity_start.to_numpy()-candidate.cost.to_numpy()/candidate.equity_start.to_numpy()
            scale=candidate.equity_start.to_numpy()/CAPITAL*future
            assert abs(np.sum((holding+friction)*scale)-(w[name]-w['mu0_q0']))<2e-11
            daily_bridges.append(pd.DataFrame({'scope':scope,'account':name,'decision_session':candidate.as_of_session,
                'holding_return_difference':holding,'fee_return_difference':friction,
                'terminal_holding_contribution':holding*scale,'terminal_fee_contribution':friction*scale}))
        print(f'D1 {scope}: 六账户原预测比较完成；原四账户逐日吻合。',flush=True)
    save_csv(output,'D1_accounts',metrics); save_csv(output,'D1_factorial',factorial)
    save_csv(output,'D1_failure_chain',pd.concat(chains,ignore_index=True))
    save_csv(output,'D1_switches',pd.concat(switches,ignore_index=True));save_csv(output,'D1_holding_spells',spells)
    save_csv(output,'D1_terminal_wealth_bridges',pd.concat(daily_bridges,ignore_index=True))
    return pd.concat(chains,ignore_index=True), targets_all, ledgers_all, audits


def d2_holding(prices, clock, targets, output):
    metrics, audits, block_rows, period_rows = [], [], [], []
    for scope,start,end in PERIODS:
        px=prices.loc[prices.as_of_session.le(end)].reset_index(drop=True);cl=clock.iloc[:len(px)].copy()
        idx=pd.Index(px.as_of_session,name='as_of_session')
        names={'M0_main':targets[scope,'mu0_q0'],'M2_main':targets[scope,'mu2_q2'],
               'M0_risk':targets[scope,'M0_risk'],'M2_risk':targets[scope,'M2_risk'],
               'fixed50':pd.Series(.5,index=idx),'fixed100':pd.Series(1.,index=idx)}
        for offset in range(5):
            for name,target in names.items():
                sparse,actual_start=offset_target(target,cl,start,offset,end)
                for cadence,t in [('hold5',sparse),('daily',target)]:
                    l=run_account(px,cl,t,actual_start,CAPITAL,COST)
                    audits.append({'scope':scope,'account':name,'offset':offset,'cadence':cadence,**audit_account(l,t,px,cl)})
                    path=f'ledgers/D2_{scope}_{offset}_{cadence}_{name}.csv';l.to_csv(output/path,index=False)
                    metrics.append({'scope':scope,'offset':offset,'cadence':cadence,'account':name,
                        'ledger_file':path,'terminal_block_complete':False,
                        'terminal_holding_observed_sessions':(len(l)-2)%5 if cadence=='hold5' else 0,
                        **account_metrics(l,CAPITAL)})
                    for year, g in l.iloc[1:].groupby(l.as_of_session.iloc[1:].str[:4]):
                        wealth=np.r_[1.,np.cumprod(1+g.portfolio_return.to_numpy())]
                        period_rows.append({'scope':scope,'offset':offset,'cadence':cadence,'account':name,
                            'year':year,'rows':len(g),'return':wealth[-1]-1,
                            'within_period_drawdown':np.min(wealth/np.maximum.accumulate(wealth)-1),
                            'mean_exposure':g.old_weight.mean(),'cost_dollars':g.cost.sum()})
                    if cadence=='hold5':
                        for i in range(1,len(l),5):
                            j=min(i+5,len(l)-1)
                            # End is before the next rebalance; entry fee belongs to this block.
                            ending=l.equity_before_trade.iloc[j] if j==i+5 else l.equity_end.iloc[j]
                            block_rows.append({'scope':scope,'offset':offset,'account':name,
                                'entry':l.as_of_session.iloc[i],'end':l.as_of_session.iloc[j],
                                'full_five_sessions':j-i==5,'target':l.target_weight.iloc[i],
                                'block_return_with_entry_fee':ending/l.equity_before_trade.iloc[i]-1,
                                'entry_cost':l.cost.iloc[i],'posttrade_exposure':l.new_weight.iloc[i]})
        print(f'D2 {scope}: 五偏移、六对照、五日/每日同起点共60账户完成。',flush=True)
    m=save_csv(output,'D2_holding_accounts',metrics)
    save_csv(output,'D2_nonoverlap_blocks',block_rows)
    save_csv(output,'D2_holding_periods',period_rows)
    for field in ['total_return','max_drawdown','mean_exposure','cost_dollars','worst_five_days']:
        lookup=m.loc[m.cadence.eq('daily')].set_index(['scope','offset','account'])[field]
        m[field+'_minus_daily']=[r[field]-lookup.loc[(r.scope,r.offset,r.account)] for _,r in m.iterrows()]
    save_csv(output,'D2_holding_accounts',m)
    return audits


def fit_variance(X, y, alpha=1.):
    """正值log-link QLIKE；E[r²]目标，非E[log(r²)]，没有模型网格。"""
    if not np.isfinite(y).all() or (y<0).any() or y.mean()<=0:
        raise ValueError('方差代理训练目标无效')
    scale=float(y.mean()); z=y/scale; Z=np.column_stack([np.ones(len(X)),X])
    def objective(theta):
        eta=Z@theta
        if np.max(np.abs(eta))>600:
            return np.inf,np.full(len(theta),np.nan)
        ratio=z*np.exp(-eta)
        loss=np.mean(eta+ratio)+.5*alpha*np.sum(theta[1:]**2)
        gradient=Z.T@(1-ratio)/len(Z)+np.r_[0.,alpha*theta[1:]]
        return loss,gradient
    result=minimize(objective,np.zeros(Z.shape[1]),jac=True,method='L-BFGS-B',
                    options={'ftol':1e-14,'gtol':1e-9,'maxiter':2000,'maxls':50})
    gradient=float(np.max(np.abs(objective(result.x)[1])))
    if not result.success or gradient>2e-6:
        raise ValueError(f'QLIKE拟合未收敛 {result.message}, gradient={gradient}')
    return {'theta':result.x.tolist(),'target_scale':scale,'alpha':alpha,
            'objective':float(result.fun),'gradient_max':gradient,'iterations':int(result.nit),
            'target_zero_rows':int((y==0).sum())}


def variance_predict(state, X):
    eta=np.column_stack([np.ones(len(X)),X])@np.asarray(state['theta'])
    raw=state['target_scale']*np.exp(eta)
    if not np.isfinite(raw).all():raise ValueError('方差预测非有限')
    return raw,np.maximum(raw,1e-12)


def nearest_support(train, future, positions):
    dist=cdist(train,train)/np.sqrt(train.shape[1])
    dist[np.abs(positions[:,None]-positions[None,:])<=5]=np.inf
    ref=np.partition(dist,4,axis=1)[:,4]
    threshold=float(np.quantile(ref,.95))
    q=np.partition(cdist(future,train)/np.sqrt(train.shape[1]),4,axis=1)[:,4]
    return q,threshold,ref


def d3_and_variance(frame, predictions, config, paths, output):
    design_rows, distributions, train_rows, support_rows, influence, clusters_out = [],[],[],[],[],[]
    coefficient_rows, variance_rows, state_rows, singular_rows = [],[],[],[]
    source_refs={}; audit={'original_prediction_max_error':0.,'ridge_coefficient_max_error':0.,
        'training_identity_fits':0,'cluster_fixed_refits':0,'cluster_full_refits':0,'variance_fits':0}
    for folder in ['D3_correlations','D3_refits','D4_models','original_candidates']:(output/folder).mkdir()
    positions={d:i for i,d in enumerate(frame.index)}
    # Read the project's unchanged dictionary, without changing the original Design implementation.
    root=paths['p2'].parents[3]
    interactions=read_json(root/'FEATURES.json')['interactions']
    m2=predictions.loc[predictions.model.eq('M2')&predictions.forecast_available].copy()
    for count,((stage,fit_id),pred) in enumerate(m2.groupby(['source_stage','fit_id'],sort=True),1):
        source_dir=paths['p4'] if stage=='p4' else paths['p7']/'core'
        source=source_dir/'models'/f'{fit_id}.json';saved=read_json(source)
        source_refs[str(source.relative_to(root))]=digest(source)
        fit=saved['fit'];state=saved['transform'];scope=pred.scope.iloc[0]
        pos=positions[fit['fit_information_session']]
        train=training_rows(frame,pos,config,cutoff=fit['simulation_fit_at'])
        memberships=read_csv(source_dir/'training_rows.csv')
        actual=memberships.loc[memberships.fit_session.eq(fit['fit_session']),'as_of_session'].tolist()
        assert train.index.tolist()==actual and len(train)==fit['training_rows']
        assert train.label_matures_at.lt(pd.Timestamp(fit['simulation_fit_at'])).all()
        raw=train[CORE_IDS].to_numpy(float);future_raw=frame.loc[pred.as_of_session,CORE_IDS].to_numpy(float)
        X=matrix(state,raw);V=matrix(state,future_raw); y=train.R5.to_numpy(float)
        original=np.r_[saved['heads']['mu5']['intercept'],saved['heads']['mu5']['coef']]
        alpha=saved['heads']['mu5']['parameter']
        theta,inverse,Z=ridge_system(X,y,alpha);VF=np.column_stack([np.ones(len(V)),V])
        error=float(np.max(np.abs(VF@theta-pred.mu5_raw.to_numpy())))
        coef_error=float(np.max(np.abs(theta-original)))
        audit['original_prediction_max_error']=max(error,audit['original_prediction_max_error'])
        audit['ridge_coefficient_max_error']=max(coef_error,audit['ridge_coefficient_max_error'])
        assert error<2e-10 and coef_error<2e-10
        audit['training_identity_fits']+=1
        fitted=Z@theta;residual=y-fitted
        leverage=np.sum((Z@inverse)*Z,axis=1)
        derivative=(VF@inverse@Z.T)*residual[None,:]
        train_pos=np.array([positions[d] for d in train.index])
        se={k:hac_prediction_se(Z,residual,VF,inverse,train_pos,k) for k in [21,63]}
        centered=X-X.mean(axis=0);singular=np.linalg.svd(centered,compute_uv=False)
        tol=max(centered.shape)*np.finfo(float).eps*singular[0]
        rawz=(raw-np.asarray(state['raw_scaler']['mean']))/np.asarray(state['raw_scaler']['scale'])
        futurez=(future_raw-np.asarray(state['raw_scaler']['mean']))/np.asarray(state['raw_scaler']['scale'])
        raw_s=np.linalg.svd(rawz-rawz.mean(axis=0),compute_uv=False)
        rank=int((singular>tol).sum());rawrank=int(np.linalg.matrix_rank(rawz-rawz.mean(axis=0)))
        shrink=singular**2/(singular**2+alpha)
        qcoef=np.asarray(saved['heads']['q90'].get('coef',np.zeros(63)))
        design_rows.append({'scope':scope,'fit_id':fit_id,'training_rows':len(train),'alpha':alpha,
            'alpha_over_n':alpha/len(train),'raw_columns':19,'design_columns':X.shape[1],'raw_rank':rawrank,
            'design_rank':rank,'ridge_df_including_intercept':1+shrink.sum(),
            'ridge_penalized_condition':(singular[0]**2+alpha)/(singular[-1]**2+alpha),
            'unpenalized_positive_condition':singular[0]/singular[rank-1],
            'largest_mode_shrink':shrink[0],'median_mode_shrink':np.median(shrink),
            'q90_L1_nonzero':int(np.count_nonzero(qcoef)),
            'q90_L1_interaction_nonzero':int(np.count_nonzero(qcoef[-6:])),
            'R5_train_rmse':np.sqrt(np.mean(residual**2)),
            'training_event_clusters':len(training_clusters(train))})
        for typ,ss in [('raw19',raw_s),('design63',singular)]:
            for j,value in enumerate(ss):singular_rows.append({'scope':scope,'fit_id':fit_id,'kind':typ,'mode':j+1,'singular_value':value,'ridge_mode_shrink':value**2/(value**2+alpha) if typ=='design63' else np.nan})
        for kind,data,columns in [('raw19',raw,CORE_IDS),('design63',X,state['columns'])]:
            df=pd.DataFrame(data,columns=columns)
            for method in ['pearson','spearman']:
                df.corr(method=method).to_csv(output/'D3_correlations'/f'{fit_id}_{kind}_{method}.csv')
            for j,col in enumerate(columns):distributions.append({'scope':scope,'fit_id':fit_id,'kind':kind,'column':col,**distribution(data[:,j])})
        for kind,values in [('R5',y),('training_residual',residual)]:
            distributions.append({'scope':scope,'fit_id':fit_id,'kind':kind,'column':kind,**distribution(values)})
        evs=training_clusters(train);event_map={d:f'{e["start"]}_{e["end"]}' for e in evs for d in e['members']}
        for j,date in enumerate(train.index):
            train_rows.append({'scope':scope,'fit_id':fit_id,'as_of_session':date,'R5':y[j],
                'fitted_mu5':fitted[j],'residual':residual[j],'squared_error':residual[j]**2,
                'sse_share':residual[j]**2/np.sum(residual**2),'ridge_leverage':leverage[j],
                'row_gradient_norm':2*abs(residual[j])*np.linalg.norm(Z[j]),
                'later_month_upweight_derivative_rms':np.sqrt(np.mean(derivative[:,j]**2)),
                'later_month_upweight_derivative_max':np.max(np.abs(derivative[:,j])),
                'training_event_id':event_map.get(date,'')})
        for head in ['mu5','q90','p10']:
            h=saved['heads'][head]
            for col,coef in zip(state['columns'],h.get('coef',np.zeros(63)),strict=True):
                coefficient_rows.append({'scope':scope,'fit_id':fit_id,'head':head,'column':col,'coef':coef})
        rawnear,rawcut,_=nearest_support(rawz,futurez,train_pos)
        designnear,designcut,_=nearest_support(X,V,train_pos)
        q25,q50,q75=np.quantile(np.maximum(fitted,-1),[.25,.5,.75])
        high=float(train.B01.quantile(.75))
        for j,p in enumerate(pred.itertuples()):
            rawoutside=(future_raw[j]<raw.min(axis=0))|(future_raw[j]>raw.max(axis=0))
            ioutside=(V[j,-6:]<X[:,-6:].min(axis=0))|(V[j,-6:]>X[:,-6:].max(axis=0))
            support_rows.append({'scope':scope,'fit_id':fit_id,'as_of_session':p.as_of_session,'decision_session':p.decision_session,
                'mu_HAC_SE21':se[21][j],'mu_HAC_SE63':se[63][j],
                'gate_within_1_96_SE21':abs(p.mu5-2*COST)<=1.96*se[21][j],
                'gate_within_1_96_SE63':abs(p.mu5-2*COST)<=1.96*se[63][j],
                'raw_outside_count':int(rawoutside.sum()),'interaction_outside_count':int(ioutside.sum()),
                'outside_interaction_ids':'|'.join(np.asarray(state['columns'][-6:])[ioutside]),
                **{f'interaction_{k}_design_value':V[j,-6+k] for k in range(6)},
                'raw5nn':rawnear[j],'raw5nn_train95':rawcut,'raw_joint_sparse':rawnear[j]>rawcut,
                'design5nn':designnear[j],'design5nn_train95':designcut,'design_joint_sparse':designnear[j]>designcut,
                'mu_train_quartile':int(np.searchsorted([q25,q50,q75],p.mu5,side='right')+1),
                'mu_train_p25':q25,'mu_train_p50':q50,'mu_train_p75':q75,
                'vix_high_cutoff_log':high,'vix_high':frame.loc[p.as_of_session,'B01']>=high})
        for event in evs:
            event_id=f'{event["start"]}_{event["end"]}'
            selected=train.index.isin(event['members']);keep=~selected
            yy=y.copy();yy[selected]=y.mean()
            target_only=inverse@Z.T@yy
            fixed,_,_=ridge_system(X[keep],y[keep],alpha)
            fresh=Design('M2',config,interactions);XX=fresh.fit(raw[keep]);VV=fresh.transform(future_raw)
            full,_,_=ridge_system(XX,y[keep],alpha)
            # Independent sklearn SVD fit checks all exact deletion systems.
            sk=Ridge(alpha=alpha,solver='svd').fit(X[keep],y[keep])
            np.testing.assert_allclose(fixed,np.r_[sk.intercept_,sk.coef_],atol=2e-10,rtol=2e-10)
            skfull=Ridge(alpha=alpha,solver='svd').fit(XX,y[keep])
            np.testing.assert_allclose(full,np.r_[skfull.intercept_,skfull.coef_],atol=2e-10,rtol=2e-10)
            write_json(output/'D3_refits'/f'{fit_id}_{event_id}.json',{
                'original_fit':str(source.relative_to(root)),'original_fit_sha256':digest(source),
                'removed_training_information_dates':event['members'],'last_removed_matures_at':train.loc[selected,'label_matures_at'].max().isoformat(),
                'fit_cutoff':fit['simulation_fit_at'],'alpha':alpha,
                'target_mean_replace_theta':target_only.tolist(),'fixed_delete_theta':fixed.tolist(),
                'full_delete_theta':full.tolist(),'full_delete_transform':fresh.snapshot()})
            variants={'target_mean_replace':(target_only,V),'fixed_delete':(fixed,V),'full_delete':(full,VV)}
            deltas={}
            for variant,(coef,design) in variants.items():
                vp=np.column_stack([np.ones(len(design)),design])@coef
                delta=vp-pred.mu5_raw.to_numpy();deltas[variant]=delta
                for j,p in enumerate(pred.itertuples()):
                    spl_delta=design[j,:57]@coef[1:58]-V[j,:57]@theta[1:58]
                    int_delta=design[j,57:]@coef[58:]-V[j,57:]@theta[58:]
                    assert abs((coef[0]-theta[0])+spl_delta+int_delta-delta[j])<2e-10
                    influence.append({'scope':scope,'fit_id':fit_id,'training_event_id':event_id,
                        'variant':variant,'as_of_session':p.as_of_session,'decision_session':p.decision_session,
                        'original_mu5_raw':p.mu5_raw,'changed_mu5_raw':vp[j],'changed_mu5':max(-1.,vp[j]),
                        'mu_delta':delta[j],'intercept_delta':coef[0]-theta[0],
                        'spline_component_delta':spl_delta,'interaction_component_delta':int_delta,
                        'gate_changed':(max(-1.,vp[j])>2*COST)!=(p.mu5>2*COST),
                        'original_R5':p.R5,'original_score_observed':p.score_observed})
            clusters_out.append({'scope':scope,'fit_id':fit_id,'event_id':event_id,'removed_rows':int(selected.sum()),
                'event_start':event['start'],'event_end':event['end'],'event_R5_mean':y[selected].mean(),
                'sse_share':np.sum(residual[selected]**2)/np.sum(residual**2),
                'leverage_sum':leverage[selected].sum(),
                'target_only_max_mu_delta':np.max(np.abs(deltas['target_mean_replace'])),
                'fixed_delete_max_mu_delta':np.max(np.abs(deltas['fixed_delete'])),
                'full_delete_max_mu_delta':np.max(np.abs(deltas['full_delete'])),
                'full_minus_fixed_max_mu_delta':np.max(np.abs(deltas['full_delete']-deltas['fixed_delete'])),
                'fixed_delete_gate_changes':int(np.sum((np.maximum(pred.mu5_raw.to_numpy()+deltas['fixed_delete'],-1)>2*COST)!=pred.mu5.gt(2*COST).to_numpy())),
                'full_delete_gate_changes':int(np.sum((np.maximum(pred.mu5_raw.to_numpy()+deltas['full_delete'],-1)>2*COST)!=pred.mu5.gt(2*COST).to_numpy()))})
            audit['cluster_fixed_refits']+=1;audit['cluster_full_refits']+=1
        own=np.log(np.maximum(train.F04.to_numpy()**2/252,1e-12))[:,None]
        own_future=np.log(np.maximum(frame.loc[pred.as_of_session,'F04'].to_numpy()**2/252,1e-12))[:,None]
        scaler=StandardScaler().fit(own)
        for model,TX,PX in [('V_SELF21',scaler.transform(own),scaler.transform(own_future)),('V_JOINT63',X,V)]:
            fittedvar=fit_variance(TX,train.V5.to_numpy(float),1.)
            rawvar,vhat=variance_predict(fittedvar,PX)
            write_json(output/'D4_models'/f'{fit_id}_{model}.json',{
                **fittedvar,'model':model,'original_fit_id':fit_id,'training_information_dates':train.index.tolist(),
                'train_last_mature_at':train.label_matures_at.max().isoformat(),'simulation_fit_at':fit['simulation_fit_at'],
                'transform_source':str(source.relative_to(root)),
                'own_scaler_mean':scaler.mean_.tolist(),'own_scaler_scale':scaler.scale_.tolist()})
            state_rows.append({'scope':scope,'fit_id':fit_id,'model':model,'training_rows':len(train),**{k:v for k,v in fittedvar.items() if k!='theta'}})
            for j,p in enumerate(pred.itertuples()):
                v=frame.loc[p.as_of_session,'V5'] if p.score_observed else np.nan
                variance_rows.append({'scope':scope,'fit_id':fit_id,'model':model,'as_of_session':p.as_of_session,
                    'decision_session':p.decision_session,'observed':p.score_observed,
                    'V5':v,'vhat_raw':rawvar[j],'vhat':vhat[j],'floored':rawvar[j]<1e-12,
                    'qlike':np.log(vhat[j])+v/vhat[j] if np.isfinite(v) else np.nan,
                    'mse':(v-vhat[j])**2 if np.isfinite(v) else np.nan})
            audit['variance_fits']+=1
        if count%6==0 or count==1:print(f'D3/D4 {count}/73个月：{fit_id} 原状态重建、支持/影响及两项方差诊断完成。',flush=True)
    for name,rows in [('D3_design',design_rows),('D3_distributions',distributions),('D3_training_influence',train_rows),
                      ('D3_prediction_support',support_rows),('D3_cluster_influence',clusters_out),
                      ('D3_changed_predictions',influence),('D3_coefficients',coefficient_rows),
                      ('D3_singular_values',singular_rows),('D4_predictions',variance_rows),('D4_fits',state_rows)]:save_csv(output,name,rows)
    write_json(output/'D3_source_models.json',source_refs)
    return pd.DataFrame(support_rows),pd.DataFrame(variance_rows),audit


def circular_blocks(n, width, draws=1000, seed=1707):
    if width>n:raise ValueError('块长超过完整日历')
    starts=np.random.default_rng(seed+width).integers(0,n,size=(draws,int(np.ceil(n/width))))
    return ((starts[:,:,None]+np.arange(width))%n).reshape(draws,-1)[:,:n]


def interval(values):
    x=np.asarray(values,float);x=x[np.isfinite(x)]
    return {'lower':float(np.quantile(x,.025)),'upper':float(np.quantile(x,.975)),
            'valid_draws':len(x)} if len(x) else {'lower':np.nan,'upper':np.nan,'valid_draws':0}


def bootstrap_mean(values, index):
    x=np.asarray(values,float)[index];valid=np.isfinite(x);n=valid.sum(axis=1)
    return np.divide(np.nansum(x,axis=1),n,out=np.full(len(x),np.nan),where=n>0)


def summarize_diagnostics(frame, chains, support, variance, paths, output):
    data=chains.merge(support.drop(columns=['scope','fit_id']),on=['as_of_session','decision_session'],how='left',validate='many_to_one')
    pathcols=['past5_log','pre_execution_log','after_day1_log','after_rest4_log','after_full5_log','V5']
    data=data.merge(frame[pathcols].reset_index(),on='as_of_session',validate='many_to_one')
    # Support and state cutoffs reference the same M2 design for both models; its SE is not M0's SE.
    data.loc[data.model.ne('M2'),['mu_HAC_SE21','mu_HAC_SE63']]=np.nan
    for col in ['gate_within_1_96_SE21','gate_within_1_96_SE63']:
        data[col]=data[col].where(data.model.eq('M2'))
    data['uncertainty_reference']=np.where(data.model.eq('M2'),'M2_FIXED_DESIGN_ALPHA_HAC','NOT_ESTIMATED_FOR_M0')
    # Preserve the original P4 score boundary; no new use of its cross-2024 outcomes in period comparisons.
    data.loc[~data.score_observed,['after_day1_log','after_rest4_log','after_full5_log','V5']]=np.nan
    data['vix_change1']=data.as_of_session.map(frame.B01.diff())
    data['vix_change5']=data.C03
    data['svxy_change1']=data.as_of_session.map(np.log1p(read_csv(output/'price_path.csv').set_index('as_of_session').simple_return))
    data['svxy_change5']=data.F03
    px=read_csv(output/'price_path.csv').set_index('as_of_session');r=np.log(px.total_return_index).diff()
    data['svxy_change21']=data.as_of_session.map(np.log(px.total_return_index).diff(21))
    data['own21_var_change']=data.as_of_session.map((frame.F04**2/252).diff())
    data['own21_incoming_contribution']=data.as_of_session.map(r.pow(2)/21)
    data['own21_outgoing_contribution']=data.as_of_session.map(r.shift(21).pow(2)/21)
    np.testing.assert_allclose(data.own21_var_change,
        data.own21_incoming_contribution-data.own21_outgoing_contribution,atol=3e-17)
    data['stress_state']=np.where(data.vix_high,'high','lower')+np.where(data.vix_change5.gt(0),'_up5','_down5')
    data['one_day_turn_vs_five']=np.sign(data.vix_change1)!=np.sign(data.vix_change5)
    data['high_falling_fast_but_own21_rising']=data.vix_high&data.vix_change1.lt(0)&data.own21_var_change.gt(0)
    data['five_day_full_position_net_proxy']=(1+data.R5)*(1-COST)/(1+COST)-1
    data['mu_prediction_residual']=data.R5-data.mu5
    data['margin_bin']=pd.cut(data.mu_margin,MARGIN_EDGES,include_lowest=True).astype(str)
    save_csv(output,'D1_D2_complete_decision_diagnostics',data)
    summary, margins, correlations, segment_groups, bootstrap, states, sparse, error_dist=[],[],[],[],[],[],[],[]
    for (scope,model),group in data.groupby(['scope','model'],sort=True):
        for kind,period,g in periods(group):
            sw=g.loc[g.gate_crossing]
            summary.append({'scope':scope,'model':model,'period_type':kind,'period':period,'rows':len(g),
                'gate_crossings':len(sw),'median_abs_margin_before_crossing':float(group.mu_margin.shift().reindex(sw.index).abs().median()),
                'median_abs_margin_after_crossing':sw.mu_margin.abs().median(),
                'median_target_jump_on_crossing':sw.target_change.abs().median(),
                'gate_crossings_near_monthly_fit':int(sw.within2_model_update.sum()),
                'all_days_near_monthly_fit':int(g.within2_model_update.sum()),
                'distribution_conflicts':int(g.distribution_conflict.sum()),
                'mu_positive_p10_high':int(g.positive_mu_and_p10_over10pct.sum()),
                'pure_price_drift_trade_days':int((g.target_change.abs().lt(1e-12)&g.trade_notional.gt(1e-8)).sum()),
                'gate_risk_joint_change_days':int(g.gate_risk_interaction.abs().gt(1e-12).sum()),
                'conditional_SE21_median':g.mu_HAC_SE21.median() if model=='M2' else np.nan,
                'conditional_SE63_median':g.mu_HAC_SE63.median() if model=='M2' else np.nan,
                'gate_within_1_96_SE21_fraction':g.gate_within_1_96_SE21.mean() if model=='M2' else np.nan,
                **{f'crossings_from_within_{bp}bp':int((group.mu_margin.shift().reindex(sw.index).abs()<=bp/10000).sum()) for bp in [10,25,50]}})
            error_dist.append({'scope':scope,'model':model,'period_type':kind,'period':period,**distribution(g.mu_prediction_residual)})
            for b,gg in g.groupby('margin_bin',sort=True):
                margins.append({'scope':scope,'model':model,'period_type':kind,'period':period,'margin_bin':b,
                    'rows':len(gg),'crossings':int(gg.gate_crossing.sum()),'mean_R5':gg.R5.mean(),
                    'mean_full5_net_proxy':gg.five_day_full_position_net_proxy.mean(),
                    'mean_target':gg.target_weight.mean(),'mean_absolute_trade':gg.turnover.mean()})
            common=g.loc[g[pathcols[:5]].notna().all(axis=1)]
            for segment in pathcols[:5]:
                for head in ['mu5','q90']:
                    corr=common[[head,segment]].corr(method='spearman').iloc[0,1] if len(common)>2 else np.nan
                    correlations.append({'scope':scope,'model':model,'period_type':kind,'period':period,
                        'head':head,'segment':segment,'matched_rows':len(common),'spearman':corr})
                groupings=[('gate',common.gate)]
                if model=='M2':groupings.append(('past_train_mu_quartile',common.mu_train_quartile))
                for gtype,labels in groupings:
                    for label,gg in common.groupby(labels,sort=True):
                        segment_groups.append({'scope':scope,'model':model,'period_type':kind,'period':period,
                            'group_type':gtype,'group':str(label),'segment':segment,'rows':len(gg),
                            'mean_log_return':gg[segment].mean(),'median_log_return':gg[segment].median()})
            for st,gg in g.groupby('stress_state',sort=True):
                states.append({'scope':scope,'model':model,'period_type':kind,'period':period,'stress_state':st,'rows':len(gg),
                    'mu_mse':np.mean(gg.mu_prediction_residual**2),'mean_R5':gg.R5.mean(),
                    'mean_net5_proxy':gg.five_day_full_position_net_proxy.mean(),'mean_L5':gg.L5.mean(),
                    'mean_V5':gg.V5.mean(),'q90_coverage':(gg.loc[gg.score_observed,'L5']<=gg.loc[gg.score_observed,'q90']).mean(),
                    'mean_mu5':gg.mu5.mean(),'mean_q90':gg.q90.mean(),'gate_pass_rate':gg.gate.mean(),
                    'one_vs_five_turn_fraction':gg.one_day_turn_vs_five.mean(),
                    'own21_rising_fraction':gg.own21_var_change.gt(0).mean()})
        for width in [21,63]:
            ix=circular_blocks(len(group),width)
            for segment in pathcols[:5]:
                common=group[pathcols[:5]].notna().all(axis=1)
                plus=group[segment].where(group.gate&common).to_numpy()
                minus=group[segment].where(~group.gate&common).to_numpy()
                diff=bootstrap_mean(plus,ix)-bootstrap_mean(minus,ix)
                bootstrap.append({'scope':scope,'model':model,'block':width,'question':'gate_above_minus_below','metric':segment,
                    'estimate':np.nanmean(plus)-np.nanmean(minus) if np.isfinite(plus).any() and np.isfinite(minus).any() else np.nan,
                    **interval(diff)})
    m2=data.loc[data.model.eq('M2')].copy()
    m0=data.loc[data.model.eq('M0')].set_index('decision_session')
    m2['mu_mse_delta_M0']=m2.mu_prediction_residual.pow(2).to_numpy()-(m0.loc[m2.decision_session,'mu_prediction_residual'].to_numpy()**2)
    m2['support_bucket']=np.where(m2.raw_outside_count.gt(0),'raw_outside',np.where(m2.design_joint_sparse,'raw_inside_joint_sparse','raw_inside_joint_dense'))
    for (scope,b),g in m2.groupby(['scope','support_bucket']):
        sparse.append({'scope':scope,'support_bucket':b,'rows':len(g),'score_rows':int(g.score_observed.sum()),
            'mu_mse_delta_M0':g.mu_mse_delta_M0.mean(),'mu_rmse':np.sqrt(g.mu_prediction_residual.pow(2).mean()),
            'conflict_fraction':g.distribution_conflict.mean(),'interaction_outside_rows':int(g.interaction_outside_count.gt(0).sum()),
            'mean_mu_HAC_SE21':g.mu_HAC_SE21.mean()})
    for name,rows in [('D1_switch_summary',summary),('D1_margin_groups',margins),('D2_path_correlations',correlations),
                      ('D2_path_groups',segment_groups),('D2_gate_path_intervals',bootstrap),('D2_D4_stress_states',states),
                      ('D3_support_error_groups',sparse),('D3_future_residual_distribution',error_dist)]:save_csv(output,name,rows)
    # Risk-intensity comparison: same dates, unchanged information states, no trading account.
    var=variance.merge(m2[['decision_session','stress_state','gate','raw_outside_count','design_joint_sparse']],on='decision_session',validate='many_to_one')
    vm,vi=[],[]
    for (scope,model),g in var.groupby(['scope','model']):
        for kind,p,gg in periods(g):
            valid=gg.loc[gg.observed]
            vm.append({'scope':scope,'model':model,'period_type':kind,'period':p,'state':'all',
                'rows':len(valid),'qlike':valid.qlike.mean(),'mse':valid.mse.mean(),
                'mean_observed_V5':valid.V5.mean(),'mean_predicted_V5':valid.vhat.mean(),'floored':int(gg.floored.sum())})
        for st,gg in g.groupby('stress_state'):
            valid=gg.loc[gg.observed]
            vm.append({'scope':scope,'model':model,'period_type':'state','period':st,'state':st,
                'rows':len(valid),'qlike':valid.qlike.mean(),'mse':valid.mse.mean(),
                'mean_observed_V5':valid.V5.mean(),'mean_predicted_V5':valid.vhat.mean(),'floored':int(gg.floored.sum())})
    for scope,_,_ in PERIODS:
        a=var.loc[var.scope.eq(scope)&var.model.eq('V_JOINT63')].set_index('decision_session')
        b=var.loc[var.scope.eq(scope)&var.model.eq('V_SELF21')].set_index('decision_session')
        assert a.index.equals(b.index)
        for state in ['all',*sorted(a.stress_state.unique())]:
            mask=np.ones(len(a),bool) if state=='all' else a.stress_state.eq(state).to_numpy()
            for metric in ['qlike','mse']:
                delta=(a[metric]-b[metric]).where(mask).to_numpy()
                for width in [21,63]:
                    ix=circular_blocks(len(delta),width)
                    vi.append({'scope':scope,'state':state,'metric':metric,'block':width,
                        'rows':int(np.isfinite(delta).sum()),'joint_minus_self':np.nanmean(delta),
                        **interval(bootstrap_mean(delta,ix))})
    save_csv(output,'D4_metrics',vm);save_csv(output,'D4_paired_intervals',vi)
    feature_functions=[]
    for scope,_,_ in PERIODS:
        g=m2.loc[m2.scope.eq(scope)&m2.score_observed]
        for feature in CORE_IDS:
            for target in ['past5_log','V5','L5','R5','five_day_full_position_net_proxy']:
                feature_functions.append({'scope':scope,'feature':feature,'family':feature[0],'target':target,
                    'rows':len(g),'spearman':g[[feature,target]].corr(method='spearman').iloc[0,1]})
    save_csv(output,'D4_feature_functions',feature_functions)
    original_state_increment(data,output)
    return data


def reuse_candidates_and_ablations(paths, output):
    copied={};selection=[]
    root=paths['p2'].parents[3]
    for stage,directory in [('p4',paths['p4']),('p7',paths['p7']/'core')]:
        dest=output/'original_candidates'/stage;dest.mkdir()
        for name in ['cv_scores.csv','cv_totals.csv','selected_parameters.csv','inner_rows.csv','interaction_conditions.csv','local_effects.csv']:
            shutil.copy2(directory/name,dest/name);copied[str((directory/name).relative_to(root))]=digest(directory/name)
        shutil.copytree(directory/'selections',dest/'selections')
        if stage=='p4':
            review=root/read_json(root/'runs/p4/review_latest.json')['review_dir']
            for name in ['selection_diagnostics.csv','candidate_diagnostics.csv']:
                shutil.copy2(review/name,dest/name);copied[str((review/name).relative_to(root))]=digest(review/name)
            old=read_csv(review/'selection_diagnostics.csv')
            # These are original P4 review conclusions, augmented with an explicitly new relative-gap descriptor.
            for r in old.to_dict('records'):
                selection.append({'stage':stage,**r,'relative_nearest_gap':r['nearest_alternative_gap']/r['selected_loss'],
                                  'tiny_strict_gap_le1pct':1e-12<r['nearest_alternative_gap']<=.01*r['selected_loss'],
                                  'original_review_reused':True})
        else:
            totals=read_csv(directory/'cv_totals.csv');chosen=read_csv(directory/'selected_parameters.csv')
            for (sid,model,head),g in totals.groupby(['selection_id','model','head']):
                parameter=chosen.loc[chosen.selection_id.eq(sid)&chosen.model.eq(model),head].item()
                value=g.loc[g.parameter.eq(parameter),'loss'].item();gap=g.loc[g.parameter.ne(parameter),'loss']-value
                selection.append({'stage':stage,'selection_id':sid,'model':model,'head':head,'chosen':parameter,
                    'selected_loss':value,'at_strongest_boundary':parameter==(g.parameter.min() if head=='p10' else g.parameter.max()),
                    'strictly_better_than_both':bool((gap>1e-12).all()),'numerically_tied_alternatives':int((gap.abs()<=1e-12).sum()),
                    'nearest_alternative_gap':gap.min(),'relative_nearest_gap':gap.min()/value,
                    'tiny_strict_gap_le1pct':1e-12<gap.min()<=.01*value,'original_review_reused':False})
        for p in (directory/'selections').glob('*.json'):copied[str(p.relative_to(root))]=digest(p)
    save_csv(output,'D3_parameter_selection',selection)
    for stage in ['p6','p7']:
        dest=output/'reused'/stage;dest.mkdir(parents=True)
        for name in ['prediction_metrics.csv','paired_prediction_periods.csv','bootstrap_intervals.csv',
                     'account_metrics.csv','paired_account_periods.csv','variant_runs.csv']:
            shutil.copy2(paths[stage]/name,dest/name);copied[str((paths[stage]/name).relative_to(root))]=digest(paths[stage]/name)
    n1=output/'reused/n1';n1.mkdir()
    for name in ['metrics.csv','attribution.csv','holding_intervals.csv']:
        shutil.copy2(paths['n1']/name,n1/name);copied[str((paths['n1']/name).relative_to(root))]=digest(paths['n1']/name)
    write_json(output/'reused_source_sha256.json',copied)


def compute(root, output):
    output.mkdir(parents=True,exist_ok=False)
    shutil.copy2(root/'runs/m2_mechanism/experiment_record.json',output/'experiment_record.json')
    with threadpool_limits(limits=1):
        frame,prices,clock,predictions,config,paths=load_inputs(root,output)
        chains,targets,ledgers,audits=d1_accounts(frame,prices,clock,predictions,paths,output)
        audits.extend(d2_holding(prices,clock,targets,output))
        support,variance,numeric=d3_and_variance(frame,predictions,config,paths,output)
        summarize_diagnostics(frame,chains,support,variance,paths,output)
        monthly_update_bridge(frame,predictions,paths,output)
        reuse_candidates_and_ablations(paths,output)
    verify_hashes(root,read_json(root/'runs/m2_mechanism/baseline.json')['preserved_sha256'])
    validation={'account_count':len(audits),'account_rows':sum(a['rows'] for a in audits),
        'account_max_error':max(a['max_amount_error'] for a in audits),'accounts':audits,**numeric,
        'preserved_files_verified':len(read_json(root/'runs/m2_mechanism/baseline.json')['preserved_sha256']),
        'new_market_downloads':0,'D5_executed':False,'original_daily_logic_modified':False}
    write_json(output/'calculation_validation.json',validation)
    return validation


def original_state_increment(data, output):
    rows=[]
    for scope,_,_ in PERIODS:
        a=data.loc[data.scope.eq(scope)&data.model.eq('M2')].set_index('decision_session')
        b=data.loc[data.scope.eq(scope)&data.model.eq('M0')].set_index('decision_session')
        assert a.index.equals(b.index)
        loss={}
        for model,g in [('M2',a),('M0',b)]:
            prob=np.clip(g.p10,np.finfo(float).eps,1-np.finfo(float).eps)
            loss[model]={'mu_mse':(g.R5-g.mu5)**2,
                'q_pinball':np.maximum(.9*(g.L5-g.q90),-.1*(g.L5-g.q90)),
                'p_logloss':-g.Y10*np.log(prob)-(1-g.Y10)*np.log1p(-prob)}
        for st in ['all',*sorted(a.stress_state.unique())]:
            mask=np.ones(len(a),bool) if st=='all' else a.stress_state.eq(st).to_numpy()
            for metric in loss['M2']:
                delta=(loss['M2'][metric]-loss['M0'][metric]).where(mask).to_numpy()
                for width in [21,63]:
                    ix=circular_blocks(len(a),width)
                    rows.append({'scope':scope,'state':st,'metric':metric,'block':width,
                        'rows':int(np.isfinite(delta).sum()),'M2_minus_M0':np.nanmean(delta),
                        **interval(bootstrap_mean(delta,ix))})
    save_csv(output,'D4_original_state_increment',rows)


def monthly_update_bridge(frame, predictions, paths, output):
    from svxylab.release_audit import saved_prediction
    records=[];prior_state=None;prior_p=None
    rows=predictions.loc[predictions.model.eq('M2')&predictions.forecast_available].sort_values('decision_session')
    for p in rows.itertuples():
        directory=paths['p4'] if p.source_stage=='p4' else paths['p7']/'core'
        if prior_p is None or p.fit_id!=prior_p.fit_id:
            state=read_json(directory/'models'/f'{p.fit_id}.json')
            if prior_p is not None:
                x=frame.loc[[p.as_of_session],CORE_IDS].to_numpy(float)
                same_input_old=saved_prediction(prior_state,x)
                same_input_new=saved_prediction(state,x)
                row={'decision_session':p.decision_session,'scope':p.scope,'prior_fit_id':prior_p.fit_id,'new_fit_id':p.fit_id,
                     'new_input_information_session':p.as_of_session,'prior_input_information_session':prior_p.as_of_session}
                for head in ['mu5','q90','p10']:
                    old_x=float(same_input_old[head][0]);new_x=float(same_input_new[head][0]);before=float(getattr(prior_p,head))
                    assert abs(new_x-getattr(p,head))<2e-10
                    row.update({f'{head}_prior_published':before,f'{head}_old_fit_new_input':old_x,
                        f'{head}_new_published':new_x,f'{head}_input_move_under_old_fit':old_x-before,
                        f'{head}_fit_update_at_same_input':new_x-old_x})
                records.append(row)
            prior_state=state
        prior_p=p
    save_csv(output,'D1_monthly_update_bridge',records)
