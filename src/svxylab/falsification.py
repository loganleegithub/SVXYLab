"""D5唯一干预：原变换/成熟训练成员不动，只固定R5收益头alpha=100。"""
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from svxylab.diagnostics_audit import matrix
from svxylab.economics import run_account
from svxylab.features import CORE_IDS
from svxylab.features_report import verify_hashes, write_json
from svxylab.mechanism import (BUDGET, CAPITAL, COST, PERIODS, audit_account,
    circular_blocks, hac_prediction_se, load_inputs, periods, read_csv,
    ridge_system, risk_capacity, save_csv, training_clusters)
from svxylab.prediction_data import training_rows
from svxylab.release import digest, read_json

NAME = 'M2_MU_ALPHA100_D5'
ALPHA = 100.


def fixed_mean(X, y, future, original_alpha, original_raw, original_published):
    """所有月份重拟核对；同alpha发布值保持原精度，避免算术噪声触发门槛。"""
    theta, inverse, Z = ridge_system(X, y, ALPHA)
    sk = Ridge(alpha=ALPHA, solver='svd').fit(X, y)
    error = float(np.max(np.abs(theta-np.r_[sk.intercept_, sk.coef_])))
    np.testing.assert_allclose(theta, np.r_[sk.intercept_, sk.coef_], atol=2e-10, rtol=2e-10)
    calculated = np.column_stack([np.ones(len(future)), future]) @ theta
    if original_alpha == ALPHA:
        np.testing.assert_allclose(calculated, original_raw, atol=2e-10, rtol=2e-10)
        raw = np.array(original_raw, copy=True)
        published = np.array(original_published, copy=True)
    else:
        assert original_alpha == 1., '原均值alpha应只有1或100；不可静默扩展比较'
        raw = calculated
        published = np.maximum(raw, -1.)
    return theta, inverse, Z, raw, published, error


def build_predictions(root, frame, predictions, config, paths, accepted, output):
    pair = predictions.loc[predictions.model.eq('M2') & predictions.forecast_available].copy()
    pair = pair.sort_values('decision_session').reset_index(drop=True)
    forecasts, designs, impacts, perturbations, memberships = [], [], [], [], []
    audit = {'fit_count':0, 'changed_alpha_fits':0, 'training_memberships_verified':0,
             'refit_coefficient_max_error':0., 'original_prediction_max_error':0.,
             'event_cluster_count':0, 'deletion_refits':0, 'deletion_svd_max_error':0.}
    (output/'models').mkdir(); (output/'deletions').mkdir()
    positions = {d:i for i,d in enumerate(frame.index)}
    for (stage, fit_id), p in pair.groupby(['source_stage', 'fit_id'], sort=True):
        directory = paths['p4'] if stage == 'p4' else paths['p7']/'core'
        source = directory/'models'/f'{fit_id}.json'; saved = read_json(source)
        fit, state = saved['fit'], saved['transform']; alpha = saved['heads']['mu5']['parameter']
        train = training_rows(frame, positions[fit['fit_information_session']], config, cutoff=fit['simulation_fit_at'])
        prior = read_csv(directory/'training_rows.csv')
        ids = prior.loc[prior.fit_session.eq(fit['fit_session']), 'as_of_session'].tolist()
        assert ids == train.index.tolist() and len(train) == fit['training_rows']
        assert train.label_matures_at.lt(pd.Timestamp(fit['simulation_fit_at'])).all()
        raw = train[CORE_IDS].to_numpy(float); future_raw = frame.loc[p.as_of_session, CORE_IDS].to_numpy(float)
        X = matrix(state, raw); V = matrix(state, future_raw); y = train.R5.to_numpy(float)
        original_theta = np.r_[saved['heads']['mu5']['intercept'], saved['heads']['mu5']['coef']]
        VF = np.column_stack([np.ones(len(V)), V])
        original_error = float(np.max(np.abs(VF@original_theta-p.mu5_raw.to_numpy())))
        assert original_error < 2e-10
        theta, inverse, Z, new_raw, new_pub, error = fixed_mean(X,y,V,alpha,p.mu5_raw.to_numpy(),p.mu5.to_numpy())
        pp = p.assign(d5_mu5_raw=new_raw,d5_mu5=new_pub,original_alpha=alpha,d5_alpha=ALPHA)
        train_pos = np.array([positions[d] for d in train.index])
        for lag in [21,63]:
            pp[f'd5_HAC_SE{lag}'] = hac_prediction_se(Z, y-Z@theta, VF, inverse, train_pos, lag)
        forecasts.append(pp)
        singular = np.linalg.svd(X-X.mean(axis=0), compute_uv=False)
        designs.append({'scope':p.scope.iloc[0],'fit_id':fit_id,'training_rows':len(train),
                        'original_alpha':alpha,'d5_alpha':ALPHA,
                        'original_effective_df':np.sum(singular**2/(singular**2+alpha)),
                        'd5_effective_df':np.sum(singular**2/(singular**2+ALPHA)),
                        'original_penalized_condition':(singular[0]**2+alpha)/(singular[-1]**2+alpha),
                        'd5_penalized_condition':(singular[0]**2+ALPHA)/(singular[-1]**2+ALPHA),
                        'max_abs_mu_change':np.max(np.abs(new_pub-p.mu5.to_numpy()))})
        write_json(output/'models'/f'{fit_id}.json',{'candidate':NAME,'original_fit':str(source.relative_to(root)),
            'original_fit_sha256':digest(source),'fit':fit,'alpha':ALPHA,'theta':theta.tolist(),
            'training_information_dates':ids,'published_unchanged_control':alpha==ALPHA,
            'transform_source':'Unchanged original_fit.transform','q90_p10_source':'Unchanged original published prediction rows'})
        memberships.append(train.reset_index()[['as_of_session','label_matures_at','label_available_at']].assign(fit_id=fit_id))
        for event in training_clusters(train):
            event_id = f'{event["start"]}_{event["end"]}'
            reference = accepted/'D3_refits'/f'{fit_id}_{event_id}.json'; old = read_json(reference)
            assert old['removed_training_information_dates'] == event['members']
            assert old['alpha'] == alpha and old['fit_cutoff'] == fit['simulation_fit_at']
            keep = ~train.index.isin(event['members'])
            states = {'fixed_delete':(X[keep], V),
                      'full_delete':(matrix(old['full_delete_transform'],raw[keep]),
                                     matrix(old['full_delete_transform'],future_raw))}
            saved_deletions = {}
            for variant,(XX,VV) in states.items():
                coef,_,_ = ridge_system(XX,y[keep],ALPHA)
                sk = Ridge(alpha=ALPHA,solver='svd').fit(XX,y[keep])
                err = float(np.max(np.abs(coef-np.r_[sk.intercept_,sk.coef_])))
                assert err < 2e-10
                if alpha == ALPHA:
                    np.testing.assert_allclose(coef,old[variant+'_theta'],atol=2e-10,rtol=2e-10)
                    coef = np.asarray(old[variant+'_theta'])
                VV1 = np.column_stack([np.ones(len(VV)),VV])
                old_deleted = VV1 @ np.asarray(old[variant+'_theta']); new_deleted = VV1 @ coef
                old_delta = old_deleted-p.mu5_raw.to_numpy(); new_delta = new_deleted-new_raw
                common = {'scope':p.scope.iloc[0],'fit_id':fit_id,'original_alpha':alpha,
                          'event_id':event_id,'variant':variant,'removed_rows':len(event['members'])}
                impacts.append({**common,'later_prediction_rows':len(p),
                    'original_max_abs_delta':np.max(np.abs(old_delta)),'d5_max_abs_delta':np.max(np.abs(new_delta)),
                    'original_rms_delta':np.sqrt(np.mean(old_delta**2)),'d5_rms_delta':np.sqrt(np.mean(new_delta**2)),
                    'original_gate_changes':np.count_nonzero((np.maximum(old_deleted,-1)>2*COST)!=p.mu5.gt(2*COST)),
                    'd5_gate_changes':np.count_nonzero((np.maximum(new_deleted,-1)>2*COST)!=(new_pub>2*COST))})
                perturbations.append(pd.DataFrame({**common,'as_of_session':p.as_of_session.to_numpy(),
                    'decision_session':p.decision_session.to_numpy(),'original_mu5_raw':p.mu5_raw.to_numpy(),
                    'd5_mu5_raw':new_raw,'original_deleted_mu5_raw':old_deleted,'d5_deleted_mu5_raw':new_deleted,
                    'original_delete_delta':old_delta,'d5_delete_delta':new_delta}))
                saved_deletions[variant+'_theta'] = coef.tolist()
                audit['deletion_svd_max_error'] = max(audit['deletion_svd_max_error'],err)
                audit['deletion_refits'] += 1
            write_json(output/'deletions'/f'{fit_id}_{event_id}.json',{
                'accepted_D3_reference':str(reference.relative_to(root)),
                'accepted_D3_sha256':digest(reference),'d5_alpha':ALPHA,**saved_deletions})
            audit['event_cluster_count'] += 1
        audit['fit_count'] += 1; audit['changed_alpha_fits'] += int(alpha != ALPHA)
        audit['training_memberships_verified'] += 1
        audit['refit_coefficient_max_error'] = max(audit['refit_coefficient_max_error'],error)
        audit['original_prediction_max_error'] = max(audit['original_prediction_max_error'],original_error)
        if audit['fit_count'] % 12 == 0: print(f'D5：已核对 {audit["fit_count"]}/73 个月及全部当时事件簇',flush=True)
    result = pd.concat(forecasts).sort_values('decision_session').reset_index(drop=True)
    assert result.as_of_session.tolist() == pair.as_of_session.tolist()
    pd.testing.assert_frame_equal(result[pair.columns],pair,check_exact=True)
    assert np.array_equal(result.loc[result.original_alpha.eq(ALPHA),'d5_mu5'],result.loc[result.original_alpha.eq(ALPHA),'mu5'])
    assert result.loc[result.original_alpha.ne(ALPHA),'decision_session'].str.startswith('2024').all()
    result['mu_change'] = result.d5_mu5-result.mu5
    result['original_gate'] = result.mu5.gt(2*COST); result['d5_gate'] = result.d5_mu5.gt(2*COST)
    result['gate_disagreement'] = result.original_gate.ne(result.d5_gate)
    result['risk_capacity'] = risk_capacity(result.q90)
    result['original_target'] = result.risk_capacity*result.original_gate
    result['d5_target'] = result.risk_capacity*result.d5_gate
    result['original_squared_error'] = ((result.mu5-result.R5)**2).where(result.score_observed)
    result['d5_squared_error'] = ((result.d5_mu5-result.R5)**2).where(result.score_observed)
    for scope,g in result.groupby('scope'):
        for prefix in ['original','d5']:
            cross = g[prefix+'_gate'].ne(g[prefix+'_gate'].shift()); cross.iloc[0] = False
            result.loc[g.index,prefix+'_crossing'] = cross
            result.loc[g.index,prefix+'_target_jump'] = g[prefix+'_target'].diff().fillna(0)
    # The accepted complete chain is authoritative for conditional SE and support annotations.
    support = read_csv(accepted/'D1_D2_complete_decision_diagnostics.csv')
    support = support.loc[support.model.eq('M2'),['as_of_session','mu_HAC_SE21','mu_HAC_SE63','raw_outside_count','design_joint_sparse']]
    result = result.merge(support,on='as_of_session',validate='one_to_one')
    save_csv(output,'paired_predictions',result)
    save_csv(output,'design_stability',designs); save_csv(output,'cluster_sensitivity',impacts)
    save_csv(output,'deletion_predictions',pd.concat(perturbations,ignore_index=True))
    save_csv(output,'training_memberships',pd.concat(memberships,ignore_index=True))
    audit.update(prediction_days=len(result),score_days=int(result.score_observed.sum()),
                 unchanged_alpha_prediction_days=int(result.original_alpha.eq(ALPHA).sum()),
                 changed_alpha_prediction_days=int(result.original_alpha.ne(ALPHA).sum()),
                 unchanged_Q90_P10_rows=len(result))
    return result,audit


def segment_metrics(g):
    """连续账本区间，不注资；局部回撤以区间期初实际权益为锚。"""
    r = g.portfolio_return.to_numpy(float)
    wealth = np.r_[1.,np.cumprod(1+r)]
    dd = wealth/np.maximum.accumulate(wealth)-1
    five = pd.Series(r).add(1).rolling(5,min_periods=5).apply(np.prod,raw=True)-1
    worst5 = int(five.idxmin()) if five.notna().any() else None
    return {'rows':len(g),'opening_equity':g.equity_start.iloc[0],'ending_equity':g.equity_end.iloc[-1],
        'net_return':wealth[-1]-1,'max_drawdown':dd.min(),'inherited_drawdown':g.drawdown.min(),
        'mean_exposure':g.old_weight.mean(),'fees':g.cost.sum(),'turnover':g.turnover.sum(),
        'worst_day':r.min(),'worst_day_session':g.as_of_session.iloc[int(np.argmin(r))],
        'worst_five':five.min(),'worst_five_first':g.as_of_session.iloc[worst5-4] if worst5 is not None else None,
        'worst_five_last':g.as_of_session.iloc[worst5] if worst5 is not None else None,
        'holding_pnl':g.holding_pnl.sum(),'cash_days':int(g.old_weight.le(1e-12).sum())}


def accounts(frame,prices,clock,predictions,pair,accepted,output):
    (output/'ledgers').mkdir(); metrics,audits,chains = [],[],[]; ledger_map = {}
    for scope,start,end in PERIODS:
        px = prices.loc[prices.as_of_session.le(end)].reset_index(drop=True);cl = clock.iloc[:len(px)]
        idx = pd.Index(px.as_of_session,name='as_of_session')
        p = pair.loc[pair.scope.eq(scope)].set_index('as_of_session').reindex(idx)
        m0 = predictions.loc[predictions.model.eq('M0') & predictions.scope.eq(scope)].set_index('as_of_session').reindex(idx)
        valid0 = m0.forecast_available.eq(True)
        q0 = pd.Series(np.nan,index=idx)
        q0.loc[valid0] = risk_capacity(m0.loc[valid0,'q90'])
        targets = {'original_M2':p.original_target,'D5':p.d5_target,'M2_risk':p.risk_capacity,
                   'M0':q0.where(m0.mu5.gt(2*COST),0).where(m0.forecast_available.eq(True)),
                   'fixed50':pd.Series(.5,index=idx),'fixed100':pd.Series(1.,index=idx)}
        for name,target in targets.items():
            ledger = run_account(px,cl,target,start,CAPITAL,COST)
            path = output/'ledgers'/f'{scope}_{name}.csv';ledger.to_csv(path,index=False)
            audit = audit_account(ledger,target,px,cl)
            audits.append({'scope':scope,'account':name,**audit})
            oldname = {'original_M2':'mu2_q2','M2_risk':'M2_risk','M0':'mu0_q0'}.get(name)
            if oldname:
                old = read_csv(accepted/'ledgers'/f'D1_{scope}_{oldname}.csv')
                assert old.as_of_session.tolist()==ledger.as_of_session.tolist()
                np.testing.assert_allclose(old.equity_end,ledger.equity_end,atol=2e-7,rtol=0)
            ledger_map[scope,name] = ledger
            for kind,period,g in periods(ledger.loc[~ledger.is_anchor],column='as_of_session'):
                metrics.append({'scope':scope,'account':name,'period_type':kind,'period':period,**segment_metrics(g)})
        g = pair.loc[pair.scope.eq(scope)].copy()
        for name in ['original_M2','D5']:
            ledger = ledger_map[scope,name].set_index('as_of_session')
            fields = ['old_weight','pre_trade_weight','target_weight','holding_pnl','cost','trade_notional','equity_start','equity_end','portfolio_return','drawdown','shares_end','cash_end']
            for field in fields:g[name+'_'+field] = g.decision_session.map(ledger[field])
        for col in CORE_IDS:g[col] = g.as_of_session.map(frame[col])
        g['holding_pnl_delta'] = g.D5_holding_pnl-g.original_M2_holding_pnl
        g['fee_delta'] = g.D5_cost-g.original_M2_cost
        g['equity_delta'] = g.D5_equity_end-g.original_M2_equity_end
        # Both paths start with equal cash; level PnL differences telescope exactly.
        np.testing.assert_allclose(g.equity_delta,np.cumsum(g.holding_pnl_delta-g.fee_delta),atol=3e-8,rtol=0)
        chains.append(g)
    save_csv(output,'account_metrics',metrics);save_csv(output,'paired_failure_chain',pd.concat(chains,ignore_index=True))
    return ledger_map,audits


def summarize(pair,output):
    summary=[]
    for scope,group in pair.groupby('scope'):
        for kind,period,g in periods(group):
            r={'scope':scope,'period_type':kind,'period':period,'prediction_days':len(g),
               'score_days':int(g.score_observed.sum()),'gate_disagreements':int(g.gate_disagreement.sum()),
               'max_abs_mu_change':g.mu_change.abs().max()}
            for prefix,mu,se in [('original','mu5','mu_HAC_SE'),('d5','d5_mu5','d5_HAC_SE')]:
                r.update({prefix+'_MSE':g[prefix+'_squared_error'].mean(),
                          prefix+'_crossings':int(g[prefix+'_crossing'].sum()),
                          prefix+'_median_abs_target_jump_on_crossing':g.loc[g[prefix+'_crossing'].eq(True),prefix+'_target_jump'].abs().median(),
                          prefix+'_median_SE21':g[se+'21'].median(),prefix+'_median_SE63':g[se+'63'].median()})
            r['MSE_delta']=r['d5_MSE']-r['original_MSE']
            r['MSE_relative_change']=r['d5_MSE']/r['original_MSE']-1
            summary.append(r)
    save_csv(output,'prediction_metrics',summary)


def return_path_metrics(returns):
    wealth = np.concatenate([np.ones((len(returns),1)),np.cumprod(1+returns,axis=1)],axis=1)
    return wealth[:,-1]-1,(wealth/np.maximum.accumulate(wealth,axis=1)-1).min(axis=1)


def bootstrap(pair,ledgers,output):
    (output/'bootstrap_draws').mkdir(); results=[]; scalar_count=0
    for scope,group in pair.groupby('scope'):
        for kind,period,g in periods(group):
            original = ledgers[scope,'original_M2'].set_index('as_of_session').loc[g.decision_session,'portfolio_return'].to_numpy()
            new = ledgers[scope,'D5'].set_index('as_of_session').loc[g.decision_session,'portfolio_return'].to_numpy()
            loss = (g.d5_squared_error-g.original_squared_error).to_numpy()
            for width in [21,63]:
                if width > len(g):
                    for metric in ['MSE_delta','daily_net_return_delta','net_return_delta','max_drawdown_delta']:
                        results.append({'scope':scope,'period_type':kind,'period':period,'block_sessions':width,
                            'metric':metric,'point':np.nan,'lower95':np.nan,'upper95':np.nan,'valid_draws':0,
                            'reason':'PERIOD_SHORTER_THAN_BLOCK'})
                    continue
                ix = circular_blocks(len(g),width)
                selected = loss[ix];valid = np.isfinite(selected).sum(axis=1)
                mse = np.divide(np.nansum(selected,axis=1),valid,out=np.full(len(ix),np.nan),where=valid>0)
                oret,odd = return_path_metrics(original[ix]);nret,ndd = return_path_metrics(new[ix])
                draws = {'MSE_delta':mse,'daily_net_return_delta':(new[ix]-original[ix]).mean(axis=1),
                         'net_return_delta':nret-oret,'max_drawdown_delta':ndd-odd}
                raw = pd.DataFrame({'draw':np.arange(len(ix)),'scored_selections':valid,**draws})
                save_csv(output/'bootstrap_draws',f'{scope}_{kind}_{period}_{width}',raw)
                op,od = return_path_metrics(original[None,:]);np_,nd = return_path_metrics(new[None,:])
                points={'MSE_delta':np.nanmean(loss),'daily_net_return_delta':np.mean(new-original),
                        'net_return_delta':float(np_[0]-op[0]),'max_drawdown_delta':float(nd[0]-od[0])}
                for metric,values in draws.items():
                    finite = values[np.isfinite(values)]; lo,hi = np.quantile(finite,[.025,.975])
                    results.append({'scope':scope,'period_type':kind,'period':period,'block_sessions':width,
                        'metric':metric,'point':points[metric],'lower95':lo,'upper95':hi,'valid_draws':len(finite)})
                    scalar_count += len(finite)
    save_csv(output,'paired_block_intervals',results)
    return scalar_count


def compute(root,output):
    output.mkdir(parents=True)
    baseline = read_json(root/'runs/m2_d5/baseline.json')
    verify_hashes(root,baseline['preserved_sha256'])
    accepted = root/read_json(root/'runs/m2_mechanism/acceptance.json')['accepted_output_dir']
    shutil.copyfile(root/'runs/m2_d5/experiment_record.json',output/'experiment_record.json')
    frame,prices,clock,predictions,config,paths = load_inputs(root,output)
    pair,validation = build_predictions(root,frame,predictions,config,paths,accepted,output)
    ledger_map,audits = accounts(frame,prices,clock,predictions,pair,accepted,output)
    summarize(pair,output)
    validation['bootstrap_valid_scalars'] = bootstrap(pair,ledger_map,output)
    validation.update(account_count=len(audits),account_rows=sum(a['rows'] for a in audits),
        account_max_error=max(a['max_amount_error'] for a in audits),accounts=audits,
        protected_files=len(baseline['preserved_sha256']),real_sessions=len(frame),
        real_first=frame.index.min(),real_last=frame.index.max(),market_downloads=0,
        D5_executed=True,candidate=NAME,second_intervention=False)
    verify_hashes(root,baseline['preserved_sha256'])
    write_json(output/'calculation_validation.json',validation)
    return validation
