"""D5只读独立复算：不调用D5拟合/变换/账户/指标/bootstrap实现。"""
from pathlib import Path
from hashlib import sha256
import json
import sys
import tomllib
import numpy as np
import pandas as pd
from scipy.interpolate import BSpline
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2]
R=lambda p:pd.read_csv(p,float_precision='round_trip')
J=lambda p:json.loads(p.read_text())
H=lambda p:sha256(p.read_bytes()).hexdigest()


def transform(s,raw):
    z=(raw-np.asarray(s['raw_scaler']['mean']))/np.asarray(s['raw_scaler']['scale']);parts=[]
    for j,b in enumerate(s['bsplines']):
        spline=BSpline(b['knots'],b['coefficients'],b['degree']);lower=b['knots'][b['degree']];upper=b['knots'][-b['degree']-1]
        at=np.clip(z[:,j],lower,upper)
        v=spline(at)+(z[:,j]-at)[:,None]*spline.derivative()(at)
        parts.append(v[:,:s['basis_per_feature']])
    for pair in s['interactions']:
        a,b=[s['input_columns'].index(k) for k in pair['features']];parts.append((z[:,a]*z[:,b])[:,None])
    return (np.concatenate(parts,axis=1)-np.asarray(s['output_scaler']['mean']))/np.asarray(s['output_scaler']['scale'])


def run(output):
    full=R(output/'complete_information_and_paths.csv').set_index('as_of_session');pred=R(output/'paired_predictions.csv')
    original=R(output/'original_prediction_service.csv');original=original.loc[original.model.eq('M2')&original.forecast_available].sort_values('decision_session').reset_index(drop=True)
    # Full request CSV includes blank no-service flags; the selected values remain bool.
    # Normalize only their known nullable type, retaining exact value comparisons.
    for col in ['mu_projected','q_projected']:
        assert original[col].map(type).eq(bool).all() and original[col].notna().all()
        original[col]=original[col].astype(bool)
    pd.testing.assert_frame_equal(pred[original.columns],original,check_exact=True)
    assert pred.loc[~pred.score_observed,['R5','L5','Y10','d5_squared_error','original_squared_error']].isna().all().all()
    cfg=tomllib.loads((ROOT/'experiment.toml').read_text());newfiterror=0.; pred_error=0.; seerror=0.;refitcount=0
    deletion=R(output/'deletion_predictions.csv');deletion_error=0.;deleted_rows=0
    by_deletion={k:g for k,g in deletion.groupby(['fit_id','event_id','variant'])}
    designcache={}
    for f in sorted((output/'models').glob('*.json')):
        saved=J(f);source=ROOT/saved['original_fit'];assert H(source)==saved['original_fit_sha256'];old=J(source)
        s=old['transform'];cols=s['input_columns'];fit=old['fit'];dates=full.index.tolist()
        pos=dates.index(fit['fit_information_session']);window=full.iloc[max(0,pos-cfg['training']['max_window_sessions']+1):pos+1]
        cutoff=pd.Timestamp(fit['simulation_fit_at'])
        valid=np.isfinite(window[cols+['R5','L5','Y10']].to_numpy(float)).all(axis=1)&window.observed.to_numpy()
        for k,strict in [('label_matures_at',True),('label_available_at',False),('assumed_available_at',False)]:
            tm=pd.to_datetime(window[k],utc=True);valid &= (tm<cutoff if strict else tm<=cutoff).to_numpy()
        train=window.loc[valid];assert train.index.tolist()==saved['training_information_dates']
        p=pred.loc[pred.fit_id.eq(f.stem)];X=transform(s,train[cols].to_numpy(float));V=transform(s,full.loc[p.as_of_session,cols].to_numpy(float));y=train.R5.to_numpy()
        independent=Ridge(alpha=100,solver='svd').fit(X,y);coef=np.r_[independent.intercept_,independent.coef_]
        newfiterror=max(newfiterror,float(np.max(np.abs(coef-saved['theta']))))
        result=independent.predict(V);pred_error=max(pred_error,float(np.max(np.abs(result-p.d5_mu5_raw))))
        np.testing.assert_allclose(result,p.d5_mu5_raw,atol=2e-10,rtol=0)
        np.testing.assert_allclose(np.maximum(result,-1),p.d5_mu5,atol=2e-10,rtol=0)
        if saved['published_unchanged_control']:
            np.testing.assert_array_equal(p.d5_mu5_raw,p.mu5_raw);np.testing.assert_array_equal(p.d5_mu5,p.mu5)
        # Direct autocovariance over the complete calendar, separately for each prediction.
        Z=np.column_stack([np.ones(len(X)),X]);VV=np.column_stack([np.ones(len(V)),V]);resid=y-independent.predict(X)
        A=Z.T@Z+np.diag(np.r_[0.,np.repeat(100,X.shape[1])]);weight=np.linalg.solve(A,VV.T).T@Z.T*resid
        calendar=np.zeros((len(V),len(full)));calendar[:,[dates.index(d) for d in train.index]]=weight
        for lag in [21,63]:
            variance=(calendar**2).sum(axis=1)
            for k in range(1,lag+1):variance+=2*(1-k/(lag+1))*(calendar[:,k:]*calendar[:,:-k]).sum(axis=1)
            expected=np.sqrt(np.maximum(variance,0));seerror=max(seerror,float(np.max(np.abs(expected-p[f'd5_HAC_SE{lag}']))))
        designcache[f.stem]=(s,V,full.loc[p.as_of_session,cols].to_numpy(float),p)
        refitcount+=1
    for f in sorted((output/'deletions').glob('*.json')):
        d=J(f);reference=ROOT/d['accepted_D3_reference'];assert H(reference)==d['accepted_D3_sha256'];old=J(reference)
        fit_id=Path(old['original_fit']).stem;event=f.stem[len(fit_id)+1:]
        state,V,raw,p=designcache[fit_id]
        for variant in ['fixed_delete','full_delete']:
            design=V if variant=='fixed_delete' else transform(old['full_delete_transform'],raw)
            rows=by_deletion[fit_id,event,variant]
            assert rows.decision_session.tolist()==p.decision_session.tolist()
            for key,coef in [('original',old[variant+'_theta']),('d5',d[variant+'_theta'])]:
                expected=np.column_stack([np.ones(len(design)),design])@np.asarray(coef)
                deletion_error=max(deletion_error,float(np.max(np.abs(expected-rows[key+'_deleted_mu5_raw']))))
                np.testing.assert_allclose(expected-rows[key+'_mu5_raw'],rows[key+'_delete_delta'],atol=3e-15,rtol=0)
            deleted_rows+=len(rows)
    assert newfiterror<2e-10 and deletion_error<2e-10 and seerror<2e-10
    px=R(output/'price_path.csv').set_index('as_of_session');date_pos={d:i for i,d in enumerate(px.index)}
    ledger_count=0;account_rows=0;account_error=0.;ledger_cache={}
    for f in sorted((output/'ledgers').glob('*.csv')):
        scope,name=f.stem.split('_',1);l=R(f);cash=100000.;shares=0.;previous=100000.
        forecast=pred.loc[pred.scope.eq(scope)].set_index('as_of_session');m0=R(output/'original_prediction_service.csv');m0=m0.loc[m0.scope.eq(scope)&m0.model.eq('M0')].set_index('as_of_session')
        for i,row in enumerate(l.itertuples()):
            price=px.loc[row.as_of_session];shares/=price.split_factor
            if not i:assert row.is_anchor and row.cost==0 and row.shares_end==0
            else:
                info=px.index[date_pos[row.as_of_session]-1]
                if name in ['fixed50','fixed100']:target=.5 if name=='fixed50' else 1.
                else:
                    p=m0.loc[info] if name=='M0' else forecast.loc[info]
                    q=p.q90;capacity=min(1.,.05/q) if q>0 else 1.
                    mu=p.d5_mu5 if name=='D5' else p.mu5
                    target=capacity if name=='M2_risk' or mu>.001 else 0.
                assert abs(row.target_weight-target)<1e-14
                cash+=shares*(price.cash_dividend+price.capital_gain_distribution)
                signed=(row.shares_end-shares)*price.close;fee=abs(signed)*.0005
                cash-=signed+fee;shares=row.shares_end
                account_error=max(account_error,abs(fee-row.cost))
                assert abs(shares*price.close/(shares*price.close+cash)-target)<1e-12
            equity=cash+shares*price.close
            account_error=max(account_error,abs(cash-row.cash_end),abs(equity-row.equity_end))
            assert abs(equity/previous-1-row.portfolio_return)<1e-12;previous=equity
        ledger_cache[scope,name]=l;ledger_count+=1;account_rows+=len(l)
    assert account_error<2e-7
    metrics=R(output/'account_metrics.csv');metric_error=0.
    for m in metrics.itertuples():
        l=ledger_cache[m.scope,m.account];g=l.loc[~l.is_anchor]
        if m.period_type=='year':g=g.loc[g.as_of_session.str.startswith(m.period)]
        if m.period_type=='quarter':g=g.loc[pd.to_datetime(g.as_of_session).dt.to_period('Q').astype(str).eq(m.period)]
        w=np.r_[g.equity_start.iloc[0],g.equity_end.to_numpy()]
        five=w[5:]/w[:-5]-1
        expected=[w[-1]/w[0]-1,(w/np.maximum.accumulate(w)-1).min(),five.min(),g.old_weight.mean(),g.cost.sum()]
        actual=[m.net_return,m.max_drawdown,m.worst_five,m.mean_exposure,m.fees]
        metric_error=max(metric_error,float(np.max(np.abs(np.array(expected)-actual))))
    assert metric_error<2e-10
    intervals=R(output/'paired_block_intervals.csv');bootstrap_error=0.;scalars=0;interval_error=0.
    for (scope,kind,period,width),gci in intervals.groupby(['scope','period_type','period','block_sessions']):
        group=pred.loc[pred.scope.eq(scope)]
        if kind=='year':group=group.loc[group.decision_session.str.startswith(period)]
        if kind=='quarter':group=group.loc[pd.to_datetime(group.decision_session).dt.to_period('Q').astype(str).eq(period)]
        n=len(group)
        if n<width:assert gci.valid_draws.eq(0).all();continue
        dates=group.decision_session
        old=ledger_cache[scope,'original_M2'].set_index('as_of_session').loc[dates,'portfolio_return'].to_numpy()
        new=ledger_cache[scope,'D5'].set_index('as_of_session').loc[dates,'portfolio_return'].to_numpy()
        losses=np.where(group.score_observed,(group.d5_mu5-group.R5)**2-(group.mu5-group.R5)**2,np.nan)
        rng=np.random.default_rng(1707+width);draws=[]
        for draw in range(1000):
            starts=rng.integers(0,n,size=int(np.ceil(n/width)))
            ix=np.array([(int(s)+k)%n for s in starts for k in range(width)])[:n]
            a=np.r_[0.,np.cumsum(np.log1p(old[ix]))];b=np.r_[0.,np.cumsum(np.log1p(new[ix]))]
            odd=np.expm1(a-np.maximum.accumulate(a)).min();ndd=np.expm1(b-np.maximum.accumulate(b)).min()
            draws.append([np.nanmean(losses[ix]),np.mean(new[ix]-old[ix]),np.expm1(b[-1])-np.expm1(a[-1]),ndd-odd])
        expected=np.array(draws);cols=['MSE_delta','daily_net_return_delta','net_return_delta','max_drawdown_delta']
        saved=R(output/'bootstrap_draws'/f'{scope}_{kind}_{period}_{width}.csv')
        bootstrap_error=max(bootstrap_error,float(np.nanmax(np.abs(saved[cols].to_numpy()-expected))))
        for j,key in enumerate(cols):
            low,high=np.nanquantile(expected[:,j],[.025,.975]);row=gci.loc[gci.metric.eq(key)].iloc[0]
            interval_error=max(interval_error,abs(low-row.lower95),abs(high-row.upper95))
        scalars+=int(np.isfinite(expected).sum())
    assert bootstrap_error<2e-11 and interval_error<2e-11
    protected=J(ROOT/'runs/m2_d5/baseline.json')['preserved_sha256']
    for path,checksum in protected.items():assert H(ROOT/path)==checksum,path
    result={'output_dir':str(output.relative_to(ROOT)),'fit_count':refitcount,'independent_svd_coefficient_max_error':newfiterror,
        'independent_mean_prediction_max_error':pred_error,'independent_calendar_HAC_max_error':seerror,
        'deletion_prediction_rows':deleted_rows,'deletion_prediction_max_error':deletion_error,
        'accounts':ledger_count,'account_rows':account_rows,'account_equation_max_error':account_error,
        'continuous_period_metrics_checked':len(metrics),'period_metric_max_error':metric_error,
        'bootstrap_scalar_values':scalars,'bootstrap_scalar_max_error':bootstrap_error,'bootstrap_interval_max_error':interval_error,
        'original_prediction_columns_exact':True,'Q90_P10_exact_rows':len(pred),'preserved_files':len(protected),
        'scope':'Independent spline tails, sklearn SVD, maturity membership, calendar-HAC, saved deletions, cash/share conservation, continuous period metrics and log-compounded paired bootstrap; no downloads.'}
    target=ROOT/'runs/m2_d5'/('independent_validation_'+output.name+'.json');target.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(result,ensure_ascii=False))
    return result

if __name__=='__main__':
    path=ROOT/(sys.argv[1] if len(sys.argv)>1 else J(ROOT/'runs/m2_d5/latest_run.json')['output_dir'])
    with threadpool_limits(limits=1):run(path)
