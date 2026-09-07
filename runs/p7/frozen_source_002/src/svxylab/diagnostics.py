"""P6开发段诊断：有限重拟合、同日账户和配对时间块。"""

from hashlib import sha256
import json
import traceback
import tomllib

import numpy as np
import pandas as pd

from svxylab.diagnostic_models import run_variant, variants
from svxylab.economics import account_metrics, load_inputs, model_targets, run_account
from svxylab.features import CORE_IDS
from svxylab.features_report import verify_hashes, write_json
from svxylab.ledger import baseline_targets
from svxylab.prediction_data import development_csv, load_development_data

LOSS_FIELDS=['mse','log_loss','pinball','brier','q90_coverage']
ECON_FIELDS=['total_return','cagr','max_drawdown','worst_day','worst_five_days','mean_exposure','annual_turnover','mean_fee_fraction']


def prepare(root):
    config=tomllib.loads((root/'experiment.toml').read_text())
    accepted=json.loads((root/'runs/p5/acceptance.json').read_text())
    verify_hashes(root,{accepted['accepted_run']:accepted['accepted_run_sha256'],accepted['accepted_report']:accepted['accepted_report_sha256']})
    verify_hashes(root,accepted['accepted_artifact_sha256'])
    # P6 adds files/an entry; all original model and account computation remains unchanged.
    verify_hashes(root,{p:h for p,h in accepted['accepted_source_sha256'].items() if p!='src/svxylab/__main__.py'})
    frame,provenance=load_development_data(root,config)
    a=json.loads((root/'runs/p3/acceptance.json').read_text())
    directory=json.loads((root/a['accepted_run']).read_text())['result']['output_dir']
    extension=development_csv(root/directory/'extension_features.csv',config['data']['locked_historical_start'])
    availability=development_csv(root/directory/'feature_availability.csv',config['data']['locked_historical_start'])
    if not extension.index.equals(frame.index):
        raise ValueError('SKEW必须保留相同完整交易日日历')
    for key in ['G01','G02']:
        extension[key]=pd.to_numeric(extension[key].replace('',np.nan),errors='raise')
    frame=frame.join(extension)
    assert pd.to_datetime(availability.assumed_available_at,utc=True).equals(frame.assumed_available_at)
    metadata=json.loads((root/directory/'feature_metadata.json').read_text())
    if metadata['availability_policy']!='ASSUMED_NEXT_SESSION' or metadata['pit_status']!='NO_FULL_HISTORICAL_PIT_CLAIM':
        raise ValueError('SKEW历史时钟口径改变，需要另行记录')
    skew={'status':'AVAILABLE_UNDER_ACCEPTED_ASSUMPTION','published_at_known':False,
        'availability_policy':metadata['availability_policy'],'pit_status':metadata['pit_status'],
        'extension_file':directory+'/extension_features.csv','availability_file':directory+'/feature_availability.csv',
        'development_missing_after_core_warmup':frame.index[(frame.index>='2019-02-01')&frame[['G01','G02']].isna().any(axis=1)].tolist(),
        'feature_metadata':[f for f in metadata['features'] if f['id'] in ['G01','G02']]}
    return frame,config,provenance,skew


def losses(prediction):
    p=prediction.set_index('decision_session').sort_index()
    valid=p.forecast_available & p.score_observed
    result=pd.DataFrame(index=p.index)
    result['mse']=(p.R5-p.mu5)**2
    probability=np.clip(p.p10,np.finfo(float).eps,1-np.finfo(float).eps)
    result['log_loss']=-(p.Y10*np.log(probability)+(1-p.Y10)*np.log1p(-probability))
    error=p.L5-p.q90
    result['pinball']=np.maximum(.9*error,-.1*error)
    result['brier']=(p.Y10-p.p10)**2
    result['q90_coverage']=(p.L5<=p.q90).astype(float)
    result.loc[~valid,:]=np.nan
    result['event']=p.Y10.where(valid)
    return result


def moving_blocks(n, width, draws, seed):
    if not 1<=width<=n:
        raise ValueError('块长必须不超过完整交易日日历')
    starts=np.random.default_rng(seed+width).integers(0,n-width+1,size=(draws,int(np.ceil(n/width))))
    return (starts[:,:,None]+np.arange(width)[None,None,:]).reshape(draws,-1)[:,:n]


def return_statistics(returns, exposure, turnover, fees, years):
    r=np.atleast_2d(returns)
    wealth=np.column_stack([np.ones(len(r)),np.cumprod(1+r,axis=1)])
    return {'total_return':wealth[:,-1]-1,'cagr':wealth[:,-1]**(1/years)-1,
        'max_drawdown':(wealth/np.maximum.accumulate(wealth,axis=1)-1).min(axis=1),
        'worst_day':r.min(axis=1),'worst_five_days':(wealth[:,5:]/wealth[:,:-5]-1).min(axis=1),
        'mean_exposure':np.atleast_2d(exposure).mean(axis=1),'annual_turnover':np.atleast_2d(turnover).sum(axis=1)/years,
        'mean_fee_fraction':np.atleast_2d(fees).mean(axis=1)}


def positive_intervals(prediction):
    rows=prediction.loc[prediction.forecast_available & prediction.score_observed & prediction.Y10.eq(1)].sort_values('decision_session')
    result=[]
    for p in rows.itertuples():
        if not result or p.decision_session>result[-1]['end']:
            result.append({'event_id':f'event_{len(result)+1:02d}','start':p.decision_session,'end':p.label_end_session,'positive_rows':1,'members':[p.as_of_session]})
        else:
            result[-1]['end']=max(result[-1]['end'],p.label_end_session)
            result[-1]['positive_rows']+=1; result[-1]['members'].append(p.as_of_session)
    return result


def periods(dates, events):
    dt=pd.to_datetime(dates)
    result=[('all','all',np.ones(len(dates),bool))]
    for kind,labels in [('year',dt.year.astype(str)),('quarter',dt.to_period('Q').astype(str))]:
        result += [(kind,value,np.asarray(labels==value)) for value in sorted(set(labels))]
    result += [('event',e['event_id'],(dates>=e['start'])&(dates<=e['end'])) for e in events]
    return result


def comparison_pairs(names):
    core=[('M1','M0'),('M2','M0'),('M2','M1')]+[(n,'M2') for n in names if n.startswith('M2_DROP_') or n=='M3_RECENCY']
    skew=[('M2_SKEW','M2_SKEW_REFERENCE'),('M2_SKEW_REFERENCE','M2'),('M2_SKEW','M2')]
    return [('core_common',a,b) for a,b in core]+[('skew_common',a,b) for a,b in skew]


def evaluate(root, frame, config, predictions, output, *, account_inputs=None):
    # P7 supplies separately authorized frozen-period inputs; P6's default stays development-only.
    prices,clock,_,curve,_,_,inputs=load_inputs(root) if account_inputs is None else account_inputs
    dates=prices.as_of_session.tolist(); names=list(predictions)
    events=positive_intervals(predictions['M2']); write_json(output/'positive_events.json',events)
    event_table=pd.DataFrame(events); event_table['members']=event_table.members.apply('|'.join)
    event_table.to_csv(output/'positive_events.csv',index=False)
    daily={name:losses(p) for name,p in predictions.items()}
    calendar=next(iter(daily.values())).index
    loss_rows=[]; calibration=[]
    for name,l in daily.items():
        for kind,period,mask in periods(calendar,events):
            v=l.loc[mask]; good=v.mse.notna()
            loss_rows.append({'model':name,'period_type':kind,'period':period,'rows':int(good.sum()),'events':int(v.event.sum()),
                              **{k:float(v[k].mean()) for k in LOSS_FIELDS}})
        p=predictions[name]; p=p.loc[p.forecast_available & p.score_observed]
        bins=np.minimum((p.p10.to_numpy()*5).astype(int),4)
        for b in range(5):
            g=p.loc[bins==b]
            calibration.append({'model':name,'bin':b+1,'rows':len(g),'positive_rows':int(g.Y10.sum()),
                                'mean_probability':g.p10.mean(),'event_rate':g.Y10.mean()})
    pd.DataFrame(loss_rows).to_csv(output/'prediction_metrics.csv',index=False)
    pd.concat([v.assign(model=k).reset_index() for k,v in daily.items()],ignore_index=True).to_csv(output/'daily_losses.csv',index=False)
    pd.DataFrame(calibration).to_csv(output/'calibration.csv',index=False)
    pairs=comparison_pairs(names)
    paired_periods=[]
    for cohort,left,right in pairs:
        a,b=daily[left],daily[right]
        jointly_valid=a.mse.notna() & b.mse.notna()
        for kind,period,mask in periods(calendar,events)+[('leave_year_out',y,calendar.str[:4]!=y) for y in sorted(set(calendar.str[:4]))]:
            selected=jointly_valid & mask
            for field in LOSS_FIELDS:
                av,bv=a.loc[selected,field],b.loc[selected,field]
                paired_periods.append({'cohort':cohort,'left':left,'right':right,'period_type':kind,'period':period,'metric':field,
                    'rows':len(av),'left_value':av.mean(),'right_value':bv.mean(),'difference':(av-bv).mean(),
                    'relative_change':av.mean()/bv.mean()-1 if len(bv) and bv.mean()!=0 else np.nan})
    pd.DataFrame(paired_periods).to_csv(output/'paired_prediction_periods.csv',index=False)
    first={n:p.loc[p.forecast_available,'decision_session'].min() for n,p in predictions.items()}
    core_names=[n for n in names if n not in ['M2_SKEW','M2_SKEW_REFERENCE']]
    skew_names=['M2','M2_SKEW_REFERENCE','M2_SKEW']
    cohorts={'full':(config['data']['outer_test_requested_start'],names),
             'core_common':(max(first[n] for n in core_names),core_names),
             'skew_common':(max(first[n] for n in skew_names),skew_names)}
    targets,metas={},{}
    for name,p in predictions.items():
        # Mapping implementation is exactly P5; the temporary identifier only adapts its fixed M0/M1/M2 API.
        t,m=model_targets(p.assign(model='M2'),dates,.05,.0005)
        for policy in ['main','risk_only']:
            targets[f'{name}_{policy}']=t[f'M2_{policy}']; metas[f'{name}_{policy}']=m[f'M2_{policy}']
    baseline=baseline_targets(curve,config['evaluation']['fixed_weight_baselines']); targets.update(baseline)
    ledgers={}; summaries=[]; account_periods=[]
    (output/'ledgers').mkdir()
    for cohort,(start,members) in cohorts.items():
        for name in [f'{n}_{p}' for n in members for p in ['main','risk_only']]+list(baseline):
            ledger=run_account(prices,clock,targets[name],start,100000.,.0005,metadata=metas.get(name))
            path=f'ledgers/{cohort}_{name}.csv'; ledger.to_csv(output/path,index=False)
            summaries.append({'cohort':cohort,'strategy':name,'ledger_file':path,**account_metrics(ledger,100000.)})
            ledgers[cohort,name]=ledger
            rows=ledger.iloc[1:]; account_dates=pd.Index(rows.as_of_session)
            for kind,period,mask in periods(account_dates,events):
                p=rows.loc[mask]
                if p.empty:
                    continue
                wealth=np.r_[p.equity_start.iloc[0],p.equity_end.to_numpy()]
                account_periods.append({'cohort':cohort,'strategy':name,'period_type':kind,'period':period,
                    'first':p.as_of_session.iloc[0],'last':p.as_of_session.iloc[-1],'days':len(p),
                    'continuous_path_return':wealth[-1]/wealth[0]-1,'path_max_drawdown':(wealth/np.maximum.accumulate(wealth)-1).min(),
                    'mean_exposure':p.old_weight.mean(),'cash_days':int(p.old_weight.le(1e-12).sum()),
                    'startup_unserved_days':int((~p.ever_forecast_executed).sum()),'cost_dollars':p.cost.sum()})
    pd.DataFrame(summaries).to_csv(output/'account_metrics.csv',index=False)
    pd.DataFrame(account_periods).to_csv(output/'account_periods.csv',index=False)
    economic_pairs=[(cohort,f'{a}_{policy}',f'{b}_{policy}') for cohort,a,b in pairs for policy in ['main','risk_only']]
    economic_pairs += [(cohort,f'{n}_main',f'{n}_risk_only') for cohort,(_,members) in cohorts.items() if cohort!='full' for n in members]
    economic_pairs += [('core_common',f'M2_{policy}',b) for policy in ['main','risk_only'] for b in baseline]
    paired_economics=[]
    ap=pd.DataFrame(account_periods)
    for cohort,left,right in economic_pairs:
        a=ap.loc[ap.cohort.eq(cohort)&ap.strategy.eq(left)].set_index(['period_type','period'])
        b=ap.loc[ap.cohort.eq(cohort)&ap.strategy.eq(right)].set_index(['period_type','period'])
        assert a.index.equals(b.index)
        for index in a.index:
            for metric in ['continuous_path_return','path_max_drawdown','mean_exposure','cost_dollars']:
                paired_economics.append({'cohort':cohort,'left':left,'right':right,'period_type':index[0],'period':index[1],
                    'metric':metric,'left_value':a.loc[index,metric],'right_value':b.loc[index,metric],
                    'difference':a.loc[index,metric]-b.loc[index,metric]})
    pd.DataFrame(paired_economics).to_csv(output/'paired_account_periods.csv',index=False)
    bootstrap(root,config,daily,pairs,ledgers,economic_pairs,output)
    return {'cohorts':{k:{'first_execution':v[0],'models':v[1]} for k,v in cohorts.items()},
        'accounts':len(summaries),'prediction_comparisons':len(pairs),'economic_comparisons':len(economic_pairs),
        'positive_rows':sum(e['positive_rows'] for e in events),'overlap_events':len(events),'input_prices_clock':inputs}


def bootstrap(root,config,daily,pairs,ledgers,economic_pairs,output):
    (output/'bootstrap_indices').mkdir()
    cache={}; summaries=[]; draws=[]
    spec=config['evaluation']; widths=[spec['bootstrap_block_sessions'],spec['bootstrap_block_sensitivity_sessions']]
    def indices(dates,width):
        key=(tuple(dates),width)
        if key not in cache:
            value=moving_blocks(len(dates),width,spec['bootstrap_draws'],config['training']['seed'])
            path=f'bootstrap_indices/{dates[0]}_{dates[-1]}_{width}.npz'
            np.savez_compressed(output/path,dates=np.asarray(dates,dtype='U10'),indices=value)
            cache[key]=(value,path)
        return cache[key]
    def save(domain,cohort,left,right,width,values,point,index_file,count):
        header={'domain':domain,'cohort':cohort,'left':left,'right':right,'block_sessions':width,'index_file':index_file}
        for metric,v in values.items():
            low,high=np.quantile(v,[.025,.975],method='linear')
            summaries.append({**header,'metric':metric,'paired_rows':count,'point_difference':point[metric],
                'lower_95':low,'upper_95':high,'draw_fraction_below_zero':np.mean(v<0),'draws':len(v)})
        draws.append(pd.DataFrame({**header,'draw':np.arange(spec['bootstrap_draws']),**values}))
    for cohort,left,right in pairs:
        a,b=daily[left],daily[right]; mask=a.mse.notna()&b.mse.notna()
        dates=a.index[a.index>=a.index[mask][0]]
        delta=(a[LOSS_FIELDS]-b[LOSS_FIELDS]).where(mask).loc[dates]
        for width in widths:
            idx,path=indices(dates,width)
            values={f:np.nanmean(delta[f].to_numpy()[idx],axis=1) for f in LOSS_FIELDS}
            save('prediction',cohort,left,right,width,values,delta.mean().to_dict(),path,int(mask.sum()))
    for cohort,left,right in economic_pairs:
        a,b=ledgers[cohort,left].iloc[1:],ledgers[cohort,right].iloc[1:]
        assert a.as_of_session.tolist()==b.as_of_session.tolist()
        dates=a.as_of_session.to_numpy(); years=(pd.Timestamp(dates[-1])-pd.Timestamp(dates[0])).days/365.25
        vectors=lambda r:[r.portfolio_return.to_numpy(),r.old_weight.to_numpy(),r.turnover.to_numpy(),(r.cost/r.equity_start).to_numpy()]
        av,bv=vectors(a),vectors(b)
        pa,pb=return_statistics(*av,years),return_statistics(*bv,years)
        point={f:pa[f][0]-pb[f][0] for f in ECON_FIELDS}
        for width in widths:
            idx,path=indices(dates,width)
            da=return_statistics(*[v[idx] for v in av],years); db=return_statistics(*[v[idx] for v in bv],years)
            save('economics',cohort,left,right,width,{f:da[f]-db[f] for f in ECON_FIELDS},point,path,len(a))
    pd.DataFrame(summaries).to_csv(output/'bootstrap_intervals.csv',index=False)
    pd.concat(draws,ignore_index=True).to_csv(output/'bootstrap_draws.csv',index=False)
    print(f'P6配对时间块：{len(summaries)}个指标区间；21/63日各1000次，保留全部抽样。',flush=True)


def run_diagnostics(root,output):
    frame,config,provenance,skew=prepare(root)
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'input_verification.json',provenance); write_json(output/'skew_availability.json',skew)
    dictionary=json.loads((root/'FEATURES.json').read_text()); variants_spec=variants(dictionary)
    write_json(output/'variants.json',variants_spec)
    p4=json.loads((root/'runs/p4/acceptance.json').read_text())
    original=pd.read_csv(root/p4['accepted_prediction_file'])
    predictions={n:original.loc[original.model.eq(n)].copy() for n in ['M0','M1','M2']}
    results=[]; maximum=0.
    for variant in variants_spec:
        try:
            prediction,result=run_variant(frame,config,variant,output/'variants'/variant['name'])
        except Exception:
            write_json(output/'failure_context.json',{'variant':variant,'failure':traceback.format_exc()});raise
        results.append(result)
        if variant['name']=='M2_REPLAY':
            a,b=prediction.set_index('decision_session'),predictions['M2'].set_index('decision_session')
            assert a.index.equals(b.index) and a.forecast_available.equals(b.forecast_available)
            for field in ['mu5_raw','mu5','q90_raw','q90','p10_logit','p10']:
                av,bv=a[field].to_numpy(float),b[field].to_numpy(float)
                np.testing.assert_allclose(av,bv,atol=2e-11,rtol=2e-10,equal_nan=True)
                maximum=max(maximum,float(np.nanmax(np.abs(av-bv))))
        else:
            predictions[variant['name']]=prediction
    pd.DataFrame(results).to_csv(output/'variant_runs.csv',index=False)
    # Preserve explicit copies of the accepted reference predictions with their source manifest.
    original.to_csv(output/'accepted_reference_predictions.csv',index=False)
    evaluation=evaluate(root,frame,config,predictions,output)
    return {'output_dir':output.relative_to(root).as_posix(),'variant_runs':results,'evaluation':evaluation,
        'm2_full_training_refit_max_prediction_difference':maximum,'accepted_p4_prediction_sha256':p4['accepted_prediction_sha256'],
        'accepted_p5_checkpoint':json.loads((root/'runs/p6/experiment_record.json').read_text())['accepted_p5_checkpoint'],
        'skew_status':skew['status'],'next_locked_primary':'M2','automatic_promotion':False,
        'locked_outcomes_evaluated':False,'p7_performed':False}
