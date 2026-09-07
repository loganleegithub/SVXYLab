"""Read-only independent reconstruction of D1-D4 outputs from local accepted inputs.
No downloads, strategy change or D5 experiment. CLI run writes only its new validation JSON.
"""
from pathlib import Path
from hashlib import sha256
import json
import math
import sys
import numpy as np
import pandas as pd
from scipy.interpolate import BSpline
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2]
R=lambda p:pd.read_csv(p,float_precision='round_trip')
J=lambda p:json.loads(p.read_text())


def design_from_snapshot(state,raw):
    z=(raw-np.array(state['raw_scaler']['mean']))/np.array(state['raw_scaler']['scale'])
    chunks=[]
    for col,s in enumerate(state['bsplines']):
        f=BSpline(s['knots'],s['coefficients'],s['degree'])
        v=z[:,col];lo=s['knots'][s['degree']];hi=s['knots'][-s['degree']-1]
        at=np.minimum(np.maximum(v,lo),hi)
        result=f(at)+(v-at)[:,None]*f.derivative()(at)
        chunks.append(result[:,:state['basis_per_feature']])
    for pair in state['interactions']:
        a,b=[state['input_columns'].index(k) for k in pair['features']]
        chunks.append((z[:,a]*z[:,b])[:,None])
    features=np.concatenate(chunks,axis=1)
    return (features-np.array(state['output_scaler']['mean']))/np.array(state['output_scaler']['scale'])


def run(output):
    frame=R(output/'complete_information_and_paths.csv').set_index('as_of_session')
    px=R(output/'price_path.csv').set_index('as_of_session')
    prices=px.total_return_index.to_numpy();expected=np.full(len(prices),np.nan)
    for t in range(len(prices)-6):
        terms=[math.log(prices[k]/prices[k-1])**2 for k in range(t+2,t+7)]
        expected[t]=math.fsum(terms)/5
    np.testing.assert_allclose(frame.V5,expected,rtol=2e-11,atol=1e-16,equal_nan=True)
    vdict=dict(zip(frame.index,expected))
    chain=R(output/'D1_D2_complete_decision_diagnostics.csv')
    q=chain.q90.to_numpy();cap=np.array([1. if v==0 else min(1.,.05/v) for v in q])
    np.testing.assert_allclose(chain.target_weight,np.where(chain.mu5>.001,cap,0.),atol=1e-12)
    cols=['gate_component','risk_component','gate_risk_interaction','prior_target_minus_old_actual','price_drift_rebalance','startup_component']
    np.testing.assert_allclose(chain[cols].sum(axis=1),chain.target_weight-chain.pre_trade_weight,atol=2e-12)
    assert chain.loc[~chain.score_observed,['R5','L5','Y10']].isna().all().all()
    hold=R(output/'D2_holding_accounts.csv');held_rows=0
    for x in hold.loc[hold.cadence.eq('hold5')].itertuples():
        l=R(output/x.ledger_file)
        # Original target can be zero, but schedule itself cannot inspect a later outcome.
        finite=np.flatnonzero(l.target_weight.notna())
        np.testing.assert_array_equal(finite,np.arange(1,len(l),5))
        for i in range(2,len(l)):
            if i not in finite:
                assert abs(l.shares_end.iloc[i]-l.shares_end.iloc[i-1]/l.split_factor.iloc[i])<1e-8
                assert l.trade_notional.iloc[i]==0
        assert l.new_weight.between(-1e-10,1+1e-10).all();held_rows+=len(l)
    features=frame.loc[:,['A01','A02','A03','A04','A05','B01','B02','B03','C01','C02','C03','D01','D02','D03','E01','F01','F02','F03','F04']]
    forecasts=R(output/'D4_predictions.csv')
    variance_error=gradient_max=0.;variance_fit_count=0
    for path in sorted((output/'D4_models').glob('*.json')):
        s=J(path);dates=s['training_information_dates'];tr=frame.loc[dates]
        cutoff=pd.Timestamp(s['simulation_fit_at'])
        assert (pd.to_datetime(tr.label_matures_at,utc=True)<cutoff).all()
        assert (pd.to_datetime(tr.label_available_at,utc=True)<=cutoff).all()
        y=np.array([vdict[d] for d in dates]);scale=math.fsum(y)/len(y)
        assert abs(scale-s['target_scale'])<2e-15
        pred=forecasts[(forecasts.fit_id==s['original_fit_id'])&(forecasts.model==s['model'])]
        if s['model']=='V_SELF21':
            X=(np.log(np.maximum(tr.F04.to_numpy()**2/252,1e-12))[:,None]-s['own_scaler_mean'])/s['own_scaler_scale']
            V=(np.log(np.maximum(frame.loc[pred.as_of_session,'F04'].to_numpy()**2/252,1e-12))[:,None]-s['own_scaler_mean'])/s['own_scaler_scale']
        else:
            transform=J(ROOT/s['transform_source'])['transform']
            X=design_from_snapshot(transform,features.loc[dates].to_numpy())
            V=design_from_snapshot(transform,features.loc[pred.as_of_session].to_numpy())
        theta=np.array(s['theta']);estimate=scale*np.exp(theta[0]+V@theta[1:])
        variance_error=max(variance_error,float(np.max(np.abs(estimate-pred.vhat_raw))))
        np.testing.assert_allclose(estimate,pred.vhat_raw,rtol=1e-11,atol=2e-14)
        scaled_prediction=np.exp(theta[0]+X@theta[1:]);res=1-y/scale/scaled_prediction
        gradient=np.r_[np.mean(res),X.T@res/len(X)+s['alpha']*theta[1:]]
        gradient_max=max(gradient_max,float(np.max(np.abs(gradient))))
        assert gradient_max<2e-6
        for r in pred.itertuples():
            if r.observed:
                obs=vdict[r.as_of_session]
                assert abs(obs-r.V5)<2e-15
                assert abs(math.log(r.vhat)+obs/r.vhat-r.qlike)<2e-10
        variance_fit_count+=1
    changed=R(output/'D3_changed_predictions.csv')
    groups={(fit,event,variant):g for (fit,event,variant),g in changed.groupby(['fit_id','training_event_id','variant'],sort=False)}
    rebuilt=0;influence_error=0.
    for path in sorted((output/'D3_refits').glob('*.json')):
        s=J(path);source=J(ROOT/s['original_fit']);fit=source['fit']['fit_id']
        # A filename identifies only existing saved experiments; no selection or new deletion.
        event=path.stem[len(fit)+1:]
        assert pd.Timestamp(s['last_removed_matures_at'])<pd.Timestamp(s['fit_cutoff'])
        assert all(d<source['fit']['fit_information_session'] for d in s['removed_training_information_dates'])
        for variant,key,transform in [('target_mean_replace','target_mean_replace_theta',source['transform']),
            ('fixed_delete','fixed_delete_theta',source['transform']),('full_delete','full_delete_theta',s['full_delete_transform'])]:
            g=groups[fit,event,variant];V=design_from_snapshot(transform,features.loc[g.as_of_session].to_numpy())
            theta=np.array(s[key]);pred=theta[0]+V@theta[1:]
            influence_error=max(influence_error,float(np.max(np.abs(pred-g.changed_mu5_raw))))
            np.testing.assert_allclose(pred,g.changed_mu5_raw,rtol=1e-10,atol=2e-11)
            np.testing.assert_allclose(g.intercept_delta+g.spline_component_delta+g.interaction_component_delta,g.mu_delta,atol=2e-10)
            assert np.array_equal(g.gate_changed,(np.maximum(pred,-1)>.001)!=(g.original_mu5_raw>.001))
            rebuilt+=len(g)
    # Independent explicit circular-block draws and means for all 40 D4 QLIKE state intervals.
    interval_rows=R(output/'D4_paired_intervals.csv');state=chain.loc[chain.model.eq('M2')].set_index('decision_session').stress_state
    interval_error=0.;scalar_means=0
    for r in interval_rows.itertuples():
        a=forecasts[(forecasts.scope==r.scope)&(forecasts.model=='V_JOINT63')].set_index('decision_session')
        b=forecasts[(forecasts.scope==r.scope)&(forecasts.model=='V_SELF21')].set_index('decision_session')
        values=(a[r.metric]-b[r.metric]).to_numpy()
        if r.state!='all':values=np.where(state.loc[a.index].eq(r.state),values,np.nan)
        n=len(values);starts=np.random.default_rng(1707+r.block).integers(0,n,size=(1000,math.ceil(n/r.block)))
        draws=[]
        for start in starts:
            ids=np.concatenate([np.arange(int(v),int(v)+r.block)%n for v in start])[:n]
            vals=values[ids];vals=vals[np.isfinite(vals)]
            draws.append(math.fsum(vals)/len(vals) if len(vals) else np.nan)
        finite=np.array(draws)[np.isfinite(draws)]
        lo,hi=np.quantile(finite,[.025,.975]);scalar_means+=len(finite)
        err=max(abs(lo-r.lower),abs(hi-r.upper));interval_error=max(interval_error,err)
        assert err<2e-10
    protected=J(ROOT/'runs/m2_mechanism/baseline.json')['preserved_sha256']
    for rel,expected_hash in protected.items():
        assert sha256((ROOT/rel).read_bytes()).hexdigest()==expected_hash,rel
    return {'path_rows':len(frame),'failure_chain_rows':len(chain),'five_day_hold_rows':held_rows,
        'variance_fits_reconstructed':variance_fit_count,'variance_prediction_max_error':variance_error,
        'independent_QLIKE_gradient_max':gradient_max,'influence_predictions_reconstructed':rebuilt,
        'influence_prediction_max_error':influence_error,'bootstrap_scalar_means':scalar_means,
        'bootstrap_interval_max_error':interval_error,'preserved_files':len(protected),
        'D5_executed':False,'validation_scope':'Independent path arithmetic, decision identities, five-day share holding, QLIKE score/gradient, saved perturbation reconstruction, all D4 state block intervals; exact refits independently checked inside original computation.'}

if __name__=='__main__':
    output=ROOT/(sys.argv[1] if len(sys.argv)>1 else J(ROOT/'runs/m2_mechanism/latest_run.json')['output_dir'])
    with threadpool_limits(limits=1):result=run(output)
    target=ROOT/'runs/m2_mechanism/independent_validation.json'
    target.write_text(json.dumps({'output_dir':str(output.relative_to(ROOT)),**result},ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))
