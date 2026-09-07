"""P6独立数值核对：保存状态重建预测、成交方程和已保存配对抽样。"""

from hashlib import sha256
import json

import numpy as np
import pandas as pd
from scipy.interpolate import BSpline
from scipy.special import expit, logit
from threadpoolctl import threadpool_limits

from svxylab.features import CORE_IDS


def basis(state, raw):
    z=(raw-np.asarray(state['raw_scaler']['mean']))/np.asarray(state['raw_scaler']['scale'])
    values=[]
    for i,s in enumerate(state['bsplines']):
        spline=BSpline(s['knots'],s['coefficients'],s['degree'])
        at=np.clip(z[:,i],spline.t[spline.k],spline.t[-spline.k-1])
        y=spline(at)+(z[:,i]-at)[:,None]*spline(at,nu=1)
        values.append(y[:,:state['basis_per_feature']])
    for interaction in state['interactions']:
        a,b=[state['input_columns'].index(k) for k in interaction['features']]
        values.append((z[:,a]*z[:,b])[:,None])
    return np.column_stack(values)


def matrix(state, raw):
    return (basis(state,raw)-np.asarray(state['output_scaler']['mean']))/np.asarray(state['output_scaler']['scale'])


def check_transform(state, raw):
    np.testing.assert_allclose(state['raw_scaler']['mean'],raw.mean(axis=0),atol=2e-11,rtol=2e-10)
    sd=raw.std(axis=0);sd[sd==0]=1
    np.testing.assert_allclose(state['raw_scaler']['scale'],sd,atol=2e-11,rtol=2e-10)
    z=(raw-raw.mean(axis=0))/sd
    for i,s in enumerate(state['bsplines']):
        np.testing.assert_allclose(s['knots'][s['degree']:-s['degree']],np.quantile(z[:,i],[0,.5,1]),atol=2e-11,rtol=2e-10)
    b=basis(state,raw);scale=b.std(axis=0);scale[scale==0]=1
    np.testing.assert_allclose(state['output_scaler']['mean'],b.mean(axis=0),atol=2e-11,rtol=2e-10)
    np.testing.assert_allclose(state['output_scaler']['scale'],scale,atol=2e-11,rtol=2e-10)


def score(head, fitted, X):
    if fitted['kind']=='constant':
        return np.full(len(X),logit(fitted['value']) if head=='p10' else fitted['value'])
    return X@np.asarray(fitted['coef'])+fitted['intercept']


def audit_models(frame, config, output):
    positions={d:i for i,d in enumerate(frame.index)}
    count=candidates=0; error=0.; ridge_error=0.
    for directory in sorted((output/'variants').iterdir()):
        variant=json.loads((directory/'variant.json').read_text())
        predictions=pd.read_csv(directory/'predictions.csv',float_precision='round_trip')
        members=pd.read_csv(directory/'training_rows.csv',float_precision='round_trip')
        inner=pd.read_csv(directory/'inner_rows.csv',float_precision='round_trip')
        for path in sorted((directory/'models').glob('*.json')):
            saved=json.loads(path.read_text()); state=saved['transform']; fit=saved['fit']
            rows=members.loc[members.fit_id.eq(fit['fit_id'])]
            cutoff=pd.Timestamp(fit['simulation_fit_at']); pos=positions[fit['fit_information_session']]
            window=frame.iloc[max(0,pos-config['training']['max_window_sessions']+1):pos+1]
            columns=CORE_IDS+['R5','L5','Y10']+(['G01','G02'] if variant['skew_cohort'] else [])
            mask=(np.isfinite(window[columns].to_numpy(float)).all(axis=1)&window.observed&window.label_matures_at.lt(cutoff)
                  &window.label_available_at.le(cutoff)&window.assumed_available_at.le(cutoff))
            expected=window.loc[mask]
            assert rows.as_of_session.tolist()==expected.index.tolist() and len(rows)>=400
            if variant['recency']:
                w=np.exp2((np.array([positions[d] for d in expected.index])-pos)/504);w*=len(w)/w.sum()
            else:
                w=np.ones(len(rows))
            np.testing.assert_allclose(rows.sample_weight,w,atol=2e-12,rtol=2e-12)
            assert abs(fit['weight_sum']-len(rows))<2e-10
            assert abs(fit['effective_training_rows']-w.sum()**2/(w@w))<2e-10
            raw=expected[state['input_columns']].to_numpy(float);check_transform(state,raw)
            assert sha256(expected[variant['columns']+['R5','L5','Y10']].to_csv().encode()).hexdigest()==fit['variant_training_values_sha256']
            selected=predictions.loc[predictions.fit_id.eq(fit['fit_id'])]
            X=matrix(state,frame.loc[selected.as_of_session,state['input_columns']].to_numpy(float))
            for head,raw_field,published in [('mu5','mu5_raw','mu5'),('q90','q90_raw','q90'),('p10','p10_logit','p10')]:
                value=score(head,saved['heads'][head],X)
                transformed=np.maximum(value,-1) if head=='mu5' else np.clip(value,0,1) if head=='q90' else expit(value)
                for field,v in [(raw_field,value),(published,transformed)]:
                    difference=np.max(np.abs(selected[field].to_numpy()-v));error=max(error,float(difference))
                    np.testing.assert_allclose(selected[field],v,atol=2e-11,rtol=2e-10)
            if variant['recency']:
                # Separate closed-form weighted Ridge solution checks use of training weights.
                X=matrix(state,raw);y=expected.R5.to_numpy();xm=np.average(X,axis=0,weights=w);ym=np.average(y,weights=w)
                xc=X-xm;alpha=saved['heads']['mu5']['parameter']
                with threadpool_limits(limits=1):
                    coef=np.linalg.solve(xc.T@(w[:,None]*xc)+alpha*np.eye(X.shape[1]),xc.T@(w*(y-ym)))
                ridge_error=max(ridge_error,float(np.max(np.abs(coef-saved['heads']['mu5']['coef']))))
                np.testing.assert_allclose(saved['heads']['mu5']['coef'],coef,atol=2e-11,rtol=2e-10)
            count+=len(selected)
        for path in sorted((directory/'selections').glob('*.json')):
            saved=json.loads(path.read_text())
            outer=members.loc[members.fit_session.eq(saved['selection_id'])]
            last=positions[outer.as_of_session.iloc[-1]]
            first=last+1-3*63
            for fold in saved['fold_transforms']:
                rows=inner.loc[inner.selection_id.eq(saved['selection_id'])&inner.block.eq(fold['block'])]
                train=rows.loc[rows.role.eq('train')];validation=rows.loc[rows.role.eq('validation')]
                cutoff=pd.Timestamp(fold['validation_cutoff'])
                block_dates=frame.index[first+(fold['block']-1)*63:first+fold['block']*63]
                outer_frame=frame.loc[outer.as_of_session]
                expected_train=outer_frame.loc[(outer_frame.index<block_dates[0]) & outer_frame.label_matures_at.lt(cutoff)
                    & outer_frame.label_available_at.le(cutoff) & outer_frame.assumed_available_at.le(cutoff)]
                assert train.as_of_session.tolist()==expected_train.index.tolist()
                assert validation.as_of_session.tolist()==outer_frame.index[outer_frame.index.isin(block_dates)].tolist()
                assert fold['validation_information_first']==block_dates[0] and fold['validation_information_last']==block_dates[-1]
                assert len(train)>=200 and pd.to_datetime(train.label_matures_at,utc=True).lt(cutoff).all()
                assert pd.to_datetime(train.label_available_at,utc=True).le(cutoff).all()
                assert pd.to_datetime(train.feature_available_at,utc=True).le(cutoff).all()
                assert set(train.as_of_session).isdisjoint(validation.as_of_session)
                assert train.as_of_session.max()<validation.as_of_session.min()
                state=fold['transform']; raw=frame.loc[train.as_of_session,state['input_columns']].to_numpy(float)
                check_transform(state,raw)
                if variant['recency']:
                    w=np.exp2((np.array([positions[d] for d in train.as_of_session])-fold['weight_reference_position'])/504);w*=len(w)/w.sum()
                else:w=np.ones(len(train))
                np.testing.assert_allclose(train.sample_weight,w,atol=2e-12,rtol=2e-12)
                X=matrix(state,frame.loc[validation.as_of_session,state['input_columns']].to_numpy(float))
                for candidate in [c for c in saved['scores'] if c['block']==fold['block']]:
                    head=candidate['head'];v=score(head,candidate['fit'],X)
                    y=frame.loc[validation.as_of_session,{'mu5':'R5','q90':'L5','p10':'Y10'}[head]].to_numpy()
                    if head=='mu5':loss=np.mean((y-np.maximum(v,-1))**2)
                    elif head=='q90':
                        d=y-np.clip(v,0,1);loss=np.maximum(.9*d,-.1*d).mean()
                    else:
                        p=np.clip(expit(v),np.finfo(float).eps,1-np.finfo(float).eps);loss=np.mean(-y*np.log(p)-(1-y)*np.log1p(-p))
                    assert abs(loss-candidate['loss'])<2e-11
                    candidates+=1
            for head,selected in saved['parameters'].items():
                totals=[r for r in saved['totals'] if r['head']==head]
                best=float('inf');expected=None
                for total in sorted(totals,key=lambda r:r['parameter'],reverse=head!='p10'):
                    rows=[r for r in saved['scores'] if r['head']==head and r['parameter']==total['parameter']]
                    loss=sum(r['loss']*r['validation_rows'] for r in rows)/sum(r['validation_rows'] for r in rows)
                    assert abs(loss-total['loss'])<2e-12
                    if loss<best-1e-12:best,expected=loss,total['parameter']
                assert selected==expected
        assert predictions.loc[~predictions.score_observed,['R5','L5','Y10']].isna().all().all()
    # Identical SKEW cohorts isolate added columns from reduced training availability.
    for filename,columns in [('training_rows.csv',['fit_session','as_of_session','sample_weight']),('inner_rows.csv',['selection_id','block','role','as_of_session','sample_weight'])]:
        a=pd.read_csv(output/'variants/M2_SKEW'/filename)[columns]
        b=pd.read_csv(output/'variants/M2_SKEW_REFERENCE'/filename)[columns]
        pd.testing.assert_frame_equal(a,b)
    return {'saved_state_prediction_rows':count,'saved_candidate_losses_checked':candidates,
            'max_saved_state_prediction_error':error,'m3_independent_ridge_coef_max_error':ridge_error}


def audit_accounts(output, prices):
    price=prices.set_index('as_of_session');metrics=pd.read_csv(output/'account_metrics.csv')
    reference=pd.read_csv(output/'accepted_reference_predictions.csv',float_precision='round_trip')
    predictions={m:g.set_index('as_of_session') for m,g in reference.groupby('model')}
    for path in (output/'variants').glob('*/predictions.csv'):
        p=pd.read_csv(path,float_precision='round_trip')
        predictions[p.model.iloc[0]]=p.set_index('as_of_session')
    dates=price.index.tolist();positions={d:i for i,d in enumerate(dates)}
    error=0.; count=0
    for item in metrics.itertuples():
        ledger=pd.read_csv(output/item.ledger_file,float_precision='round_trip')
        cash=100000.;shares=0.
        for i,row in enumerate(ledger.itertuples()):
            p=price.loc[row.as_of_session];shares/=p.split_factor
            if i:cash+=shares*(p.cash_dividend+p.capital_gain_distribution)
            stock=shares*p.close;before=cash+stock;trade=fee=0.
            if i and pd.notna(row.target_weight):
                w=row.target_weight;denom=1+w*.0005 if w*before>=stock else 1-w*.0005
                trade=(w*before-stock)/denom;fee=abs(trade)*.0005
                cash-=trade+fee;shares+=trade/p.close
            observed=[row.cash_end,row.shares_end*p.close,row.equity_end,row.signed_trade,row.cost]
            expected=[cash,shares*p.close,cash+shares*p.close,trade,fee]
            delta=float(np.max(np.abs(np.asarray(observed)-expected)));error=max(error,delta)
            assert delta<2e-5
            if i and item.strategy.startswith('M'):
                model=item.strategy.removesuffix('_main').removesuffix('_risk_only')
                source=dates[positions[row.as_of_session]-1]
                assert row.source_information_session==source
                p_source=predictions[model].loc[source] if source in predictions[model].index else None
                available=row.source_forecast_available==True
                assert available==bool(p_source is not None and p_source.forecast_available)
                if available:
                    np.testing.assert_allclose([row.source_mu5,row.source_q90],[p_source.mu5,p_source.q90],atol=2e-12,rtol=2e-12)
                    assert row.source_fit_id==p_source.fit_id
                    w=1 if row.source_q90==0 else min(1,.05/row.source_q90)
                    if item.strategy.endswith('_main') and row.source_mu5<=.001:w=0.
                    assert abs(w-row.target_weight)<2e-12
                else:assert pd.isna(row.target_weight) and trade==0
            assert row.as_of_session<'2024-01-01';count+=1
        assert abs((cash+shares*p.close)/100000-1-item.total_return)<2e-10
    for _,g in metrics.groupby('cohort'):
        assert g.first_execution.nunique()==g.anchor.nunique()==g.initial_capital.nunique()==1
    return {'account_rows':count,'accounts':len(metrics),'max_account_money_error':error}


def audit_bootstrap(output):
    draws=pd.read_csv(output/'bootstrap_draws.csv',float_precision='round_trip')
    intervals=pd.read_csv(output/'bootstrap_intervals.csv',float_precision='round_trip')
    daily=pd.read_csv(output/'daily_losses.csv',float_precision='round_trip')
    keys=['domain','cohort','left','right','block_sessions']
    count=0;maximum=0.
    for key,group in draws.groupby(keys):
        domain,cohort,left,right,width=key
        sample=np.load(output/group.index_file.iloc[0]);idx=sample['indices'];dates=sample['dates']
        assert len(group)==1000 and idx.shape==(1000,len(dates)) and idx.min()>=0 and idx.max()<len(dates)
        starts=idx[:,::width]
        assert (starts<=len(dates)-width).all()
        for j in range(0,len(dates),width):
            assert (np.diff(idx[:,j:min(j+width,len(dates))],axis=1)==1).all()
        group=group.sort_values('draw')
        if domain=='prediction':
            a=daily.loc[daily.model.eq(left)].set_index('decision_session').loc[dates]
            b=daily.loc[daily.model.eq(right)].set_index('decision_session').loc[dates]
            mask=a.mse.notna()&b.mse.notna()
            values={f:np.nanmean((a[f]-b[f]).where(mask).to_numpy()[idx],axis=1) for f in ['mse','log_loss','pinball','brier','q90_coverage']}
        else:
            values={}
            for name,sign in [(left,1),(right,-1)]:
                ledger=pd.read_csv(output/f'ledgers/{cohort}_{name}.csv',float_precision='round_trip').iloc[1:]
                assert ledger.as_of_session.tolist()==dates.tolist()
                r=ledger.portfolio_return.to_numpy()[idx]
                wealth=np.cumprod(1+r,axis=1);anchored=np.concatenate([np.ones((1000,1)),wealth],axis=1)
                years=(pd.Timestamp(str(dates[-1]))-pd.Timestamp(str(dates[0]))).days/365.25
                stats={'total_return':wealth[:,-1]-1,'cagr':wealth[:,-1]**(1/years)-1,
                    'max_drawdown':np.min(anchored/np.maximum.accumulate(anchored,axis=1)-1,axis=1),
                    'worst_day':r.min(axis=1),'worst_five_days':(anchored[:,5:]/anchored[:,:-5]-1).min(axis=1),
                    'mean_exposure':ledger.old_weight.to_numpy()[idx].mean(axis=1),
                    'annual_turnover':ledger.turnover.to_numpy()[idx].sum(axis=1)/years,
                    'mean_fee_fraction':(ledger.cost/ledger.equity_start).to_numpy()[idx].mean(axis=1)}
                for field,v in stats.items():values[field]=values.get(field,0)+sign*v
        selected=intervals
        for k,v in zip(keys,key):selected=selected.loc[selected[k].eq(v)]
        for field,values in values.items():
            error=float(np.max(np.abs(values-group[field].to_numpy())));maximum=max(maximum,error)
            assert error<2e-10
            row=selected.loc[selected.metric.eq(field)].iloc[0]
            np.testing.assert_allclose([row.lower_95,row.upper_95],np.quantile(values,[.025,.975]),atol=2e-10,rtol=2e-10)
            count+=len(values)
    return {'bootstrap_scalar_draws_recomputed':count,'bootstrap_max_difference':maximum,'intervals_checked':len(intervals)}


def validate(root,result):
    from svxylab.diagnostics import prepare
    from svxylab.economics import load_inputs
    frame,config,_,_=prepare(root);prices,*_=load_inputs(root)
    output=root/result['output_dir']
    return {**audit_models(frame,config,output),**audit_accounts(output,prices),**audit_bootstrap(output),
        'methods':'Saved coefficient/transform reconstruction (not independent optimizer refit); independent M3 Ridge solve; separate trade equations; all saved bootstrap draws recomputed.',
        'passed':True,'locked_outcomes_used':False,'skew_training_validation_members_identical':True}
