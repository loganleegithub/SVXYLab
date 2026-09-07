"""P7保存状态与真实时钟的独立复算；不另定义预测模型。"""
import json
import math

import numpy as np
import pandas as pd
from scipy.special import expit

from svxylab.diagnostics_audit import matrix, score, check_transform, audit_models, audit_bootstrap
from svxylab.features import CORE_IDS
from svxylab.release import load_historical, read_json


def saved_prediction(saved, raw):
    """用保存系数推理，无拟合；M2样条边界以一阶切线延伸。"""
    state=saved['transform']
    if state is None:X=np.empty((len(raw),0))
    elif state['model']=='M1':X=(raw-np.asarray(state['raw_scaler']['mean']))/np.asarray(state['raw_scaler']['scale'])
    else:X=matrix(state,raw)
    mu,q,p=[score(h,saved['heads'][h],X) for h in ['mu5','q90','p10']]
    return {'mu5_raw':mu,'mu5':np.maximum(mu,-1),'q90_raw':q,'q90':np.clip(q,0,1),'p10_logit':p,'p10':expit(p)}


def audit_core(frame, config, output):
    pred=pd.read_csv(output/'predictions.csv',float_precision='round_trip')
    members=pd.read_csv(output/'training_rows.csv');inner=pd.read_csv(output/'inner_rows.csv')
    count=0;error=0.;candidates=0
    for path in sorted((output/'models').glob('*.json')):
        s=read_json(path);fit=s['fit'];cutoff=pd.Timestamp(fit['simulation_fit_at'])
        pos=frame.index.get_loc(fit['fit_information_session']);window=frame.iloc[max(0,pos-1259):pos+1]
        mask=(np.isfinite(window[CORE_IDS+['R5','L5','Y10']].to_numpy()).all(axis=1)&window.observed&
              window.label_matures_at.lt(cutoff)&window.label_available_at.le(cutoff)&window.assumed_available_at.le(cutoff))
        train=window.loc[mask];actual=members.loc[members.fit_session.eq(fit['fit_session'])]
        assert actual.as_of_session.tolist()==train.index.tolist() and len(train)>=400
        raw=train[CORE_IDS].to_numpy();state=s['transform']
        if state is None:
            np.testing.assert_allclose([s['heads'][k]['value'] for k in ['mu5','q90','p10']],
                [train.R5.mean(),train.L5.quantile(.9),(train.Y10.sum()+.5)/(len(train)+1)],atol=2e-12)
        elif state['model']=='M1':
            np.testing.assert_allclose(state['raw_scaler']['mean'],raw.mean(axis=0),atol=2e-11)
            np.testing.assert_allclose(state['raw_scaler']['scale'],raw.std(axis=0),atol=2e-11)
        else:check_transform(state,raw)
        selected=pred.loc[pred.fit_id.eq(fit['fit_id']) & pred.forecast_available]
        rebuilt=saved_prediction(s,frame.loc[selected.as_of_session,CORE_IDS].to_numpy())
        for k,v in rebuilt.items():
            error=max(error,float(np.max(np.abs(v-selected[k]))));np.testing.assert_allclose(v,selected[k],atol=2e-11,rtol=2e-10)
        count+=len(selected)
    for path in sorted((output/'selections').glob('*.json')):
        s=read_json(path)
        for model,m in s['models'].items():
            for fold in m['fold_transforms']:
                selected=inner.loc[inner.selection_id.eq(s['selection_id'])&inner.block.eq(fold['block'])]
                validation=selected.loc[selected.role.eq('validation')]
                train=selected.loc[selected.role.eq('train')]
                cutoff=pd.Timestamp(fold['validation_cutoff'])
                assert len(train)>=200 and pd.to_datetime(train.label_available_at,utc=True).le(cutoff).all()
                raw=frame.loc[validation.as_of_session,CORE_IDS].to_numpy()
                state=fold['transform']
                X=(raw-np.asarray(state['raw_scaler']['mean']))/np.asarray(state['raw_scaler']['scale']) if model=='M1' else matrix(state,raw)
                for candidate in [r for r in m['scores'] if r['block']==fold['block']]:
                    head=candidate['head'];raw_score=score(head,candidate['fit'],X)
                    p=np.maximum(raw_score,-1) if head=='mu5' else np.clip(raw_score,0,1) if head=='q90' else expit(raw_score)
                    y=frame.loc[validation.as_of_session,{'mu5':'R5','q90':'L5','p10':'Y10'}[head]].to_numpy()
                    if head=='mu5':loss=np.mean((y-p)**2)
                    elif head=='q90':loss=np.mean(np.maximum(.9*(y-p),-.1*(y-p)))
                    else:
                        p=np.clip(p,np.finfo(float).eps,1-np.finfo(float).eps);loss=np.mean(-y*np.log(p)-(1-y)*np.log1p(-p))
                    assert abs(loss-candidate['loss'])<2e-11;candidates+=1
            for head in ['mu5','q90','p10']:
                totals=[t for t in m['totals'] if t['head']==head]
                totals.sort(key=lambda t:t['parameter'],reverse=head!='p10')
                best=math.inf;chosen=None
                for t in totals:
                    if t['loss']<best-1e-12:best=t['loss'];chosen=t['parameter']
                assert m['parameters'][head]==chosen
    assert pred.loc[~pred.score_observed,['R5','L5','Y10']].isna().all().all()
    return {'core_saved_state_rows':count,'core_saved_candidate_losses':candidates,'core_saved_state_max_error':error}


def audit_ledgers(output, prices, curve, predictions):
    metrics=pd.read_csv(output/'account_metrics.csv');specs=[]
    for r in metrics.to_dict('records'):
        specs.append({**r,'path':output/r['ledger_file'],'cost_rate':.0005,'filter_rate':.0005,'budget':.05,'delay':0})
    d=output/'sensitivity';sm=pd.read_csv(d/'account_metrics.csv')
    for r in sm.to_dict('records'):specs.append({**r,'path':d/r['ledger_file'],'filter_rate':r['cost_rate']})
    for r in pd.read_csv(d/'same_target_gross_net.csv').to_dict('records'):
        specs.append({**r,'path':d/r['gross_ledger_file'],'cost_rate':0.,'filter_rate':.0005,'budget':.05,'delay':0})
    for r in pd.read_csv(d/'post_hoc_exposure_matches.csv').to_dict('records'):
        specs.append({**r,'strategy':r['controller'],'path':d/r['ledger_file'],'cost_rate':.0005,'filter_rate':.0005,'budget':.05,'delay':0,'constant':r['post_hoc_fixed_target']})
    px=prices.set_index('as_of_session');cv=curve.set_index('as_of_session');dates=px.index.tolist();position={v:i for i,v in enumerate(dates)}
    pred={n:p.set_index('as_of_session') for n,p in predictions.items()};maximum=0.;count=0
    for item in specs:
        l=pd.read_csv(item['path'],float_precision='round_trip');cash=100000.;shares=0.;values=[]
        assert l.as_of_session.tolist()==dates[position[l.as_of_session.iloc[0]]:]
        for i,r in enumerate(l.itertuples()):
            p=px.loc[r.as_of_session];shares/=p.split_factor
            if i:cash+=shares*(p.cash_dividend+p.capital_gain_distribution)
            stock=shares*p.close;before=cash+stock;trade=fee=0.;target=None
            if i:
                source=dates[position[r.as_of_session]-int(item['delay'])-1];name=item['strategy']
                if 'constant' in item:target=item['constant']
                elif name.startswith('M'):
                    model=name.removesuffix('_main').removesuffix('_risk_only');q=pred[model].loc[source] if source in pred[model].index else None
                    if q is not None and q.forecast_available:
                        target=1 if q.q90==0 else min(1,item['budget']/q.q90)
                        if name.endswith('_main') and q.mu5<=2*item['filter_rate']:target=0.
                        assert r.source_fit_id==q.fit_id
                elif name=='cash':target=0.
                elif name.startswith('fixed_'):target=float(name.split('_')[1])/100
                elif cv.loc[source,['f1_settle','f2_settle']].notna().all():target=float(cv.loc[source,'f2_settle']>cv.loc[source,'f1_settle'])
                assert pd.Timestamp(r.decision_at)<pd.Timestamp(r.execution_at)
            if target is None:assert pd.isna(r.target_weight)
            else:
                assert abs(target-r.target_weight)<2e-12
                rate=item['cost_rate'];trade=(target*before-stock)/(1+target*rate if target*before>=stock else 1-target*rate)
                fee=abs(trade)*rate;cash-=trade+fee;shares+=trade/p.close
            expected=np.array([cash,shares*p.close,cash+shares*p.close,trade,fee])
            actual=np.array([r.cash_end,r.shares_end*p.close,r.equity_end,r.signed_trade,r.cost])
            error=float(np.abs(expected-actual).max());maximum=max(maximum,error);assert error<2e-5
            values.append(expected[2]);count+=1
        if 'total_return' in item:assert abs(values[-1]/100000-1-item['total_return'])<2e-10
    for _,g in metrics.groupby('cohort'):assert g.first_execution.nunique()==g.anchor.nunique()==g.initial_capital.nunique()==1
    return {'account_rows':count,'accounts':len(specs),'max_account_money_error':maximum}


def validate_release(root, result):
    frame,config,inputs=load_historical(root);output=root/result['output_dir']
    p=pd.read_csv(output/'core/predictions.csv',float_precision='round_trip')
    predictions={m:g for m,g in p.groupby('model')}
    for path in (output/'variants').glob('*/predictions.csv'):
        v=pd.read_csv(path,float_precision='round_trip');predictions[v.model.iloc[0]]=v
    cutoff=config['data']['locked_historical_start']
    for v in predictions.values():
        assert v.decision_session.ge(config['data']['outer_test_requested_start']).all() and v.decision_session.lt(cutoff).all()
        assert v.loc[v.score_observed,'label_end_session'].lt(cutoff).all()
    return {**audit_core(frame,config,output/'core'),**audit_models(frame,config,output),
        **audit_ledgers(output,inputs[0],inputs[3],predictions),**audit_bootstrap(output),
        'passed':True,'description':'保存系数/变换重建；M3均值头独立加权方程；逐笔自融资方程；保存bootstrap复算。不是独立重新优化全部模型。'}
