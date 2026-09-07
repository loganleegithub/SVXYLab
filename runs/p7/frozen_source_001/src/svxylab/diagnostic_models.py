"""P6限定的列消融/两个挑战者；复用P4训练时钟、预测头和发布规则。"""

from datetime import datetime, timezone
from hashlib import sha256
from time import perf_counter
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, QuantileRegressor, Ridge
from sklearn.preprocessing import SplineTransformer, StandardScaler
from threadpoolctl import threadpool_limits

from svxylab.features import CORE_IDS
from svxylab.features_report import write_json
from svxylab.models import (Design, GRID_FOR_HEAD, HEADS, JointModel, LABEL_FOR_HEAD,
                            fit_head, head_score, prediction_loss)
from svxylab.prediction_data import TARGETS, inner_splits, training_rows
from svxylab.predictions import training_info


def variants(dictionary):
    result = []
    for name, deleted, skew, recency, role in [
        ('M2_REPLAY', '', False, False, 'parity_reference'),
        *[(f'M2_DROP_{family}', family, False, False, 'column_ablation') for family in ['A','B','C','D','E','F','BDE']],
        ('M3_RECENCY', '', False, True, 'challenger'),
        ('M2_SKEW_REFERENCE', '', True, False, 'cohort_reference'),
        ('M2_SKEW', '', True, False, 'challenger')]:
        removed = [f for f in CORE_IDS if f[0] in deleted]
        columns = [f for f in CORE_IDS if f not in removed]
        if name == 'M2_SKEW':
            columns += ['G01','G02']
        interactions = [i for i in dictionary['interactions'] if not set(i['features']) & set(removed)]
        remaining = [f for f in dictionary['features'] if f['id'] in columns]
        result.append({'name':name, 'columns':columns, 'removed_columns':removed, 'interactions':interactions,
                       'removed_interactions':[i['id'] for i in dictionary['interactions'] if i not in interactions],
                       'skew_cohort':skew, 'recency':recency, 'role':role,
                       'remaining_raw_inputs':sorted(set(x for f in remaining for x in f['inputs'])),
                       'deletion_claim':'COLUMN_FAMILY_ONLY_NOT_ENTIRE_RAW_SOURCE' if removed else 'NONE'})
    return result


class VariantDesign(Design):
    """与原Design相同变换，输入列由本轮预定消融指定；P4源码保持不变。"""
    def __init__(self, variant, config):
        self.model, self.config = 'M2', config
        self.input_columns = variant['columns']
        self.interactions = variant['interactions']
        self.pairs = [(self.input_columns.index(i['features'][0]), self.input_columns.index(i['features'][1])) for i in self.interactions]
        self.raw_scaler, self.output_scaler = StandardScaler(), StandardScaler()

    def products(self, z):
        return np.column_stack([z[:,a]*z[:,b] for a,b in self.pairs]) if self.pairs else np.empty((len(z),0))

    def fit(self, raw):
        if not np.isfinite(raw).all():
            raise ValueError('P6训练特征有缺失或非有限值')
        self.raw_min, self.raw_max = raw.min(axis=0), raw.max(axis=0)
        z = self.raw_scaler.fit_transform(raw)
        self.z_quantiles = np.quantile(z,[.25,.5,.75],axis=0,method='linear')
        spec = self.config['models']
        self.spline = SplineTransformer(n_knots=spec['spline_knots'],degree=spec['spline_degree'],
            knots=spec['spline_knot_placement'],include_bias=spec['spline_include_bias'],extrapolation=spec['spline_extrapolation'])
        basis = self.spline.fit_transform(z)
        self.basis_per_feature = basis.shape[1]//len(self.input_columns)
        self.columns = [f'{f}_spline_{j}' for f in self.input_columns for j in range(self.basis_per_feature)]+[i['id'] for i in self.interactions]
        return self.output_scaler.fit_transform(np.column_stack([basis,self.products(z)]))

    def transform(self, raw):
        z = self.raw_scaler.transform(raw)
        return self.output_scaler.transform(np.column_stack([self.spline.transform(z),self.products(z)]))

    def snapshot(self):
        result = super().snapshot()
        result['input_columns'] = self.input_columns
        return result


def sample_weights(train, positions, reference, half_life):
    ages = reference-np.asarray([positions[d] for d in train.index])
    if (ages < 0).any() or half_life <= 0:
        raise ValueError('权重年龄必须来自过去的完整交易日日历')
    w = np.exp2(-ages/half_life)
    return w*len(w)/w.sum()


def weighted_head(head, X, y, parameter, config, weights):
    if weights is None:
        return fit_head(head,X,y,parameter,config)
    started = perf_counter()
    if head == 'p10' and len(np.unique(y)) < 2:
        return {'kind':'constant','value':float((weights@y+.5)/(weights.sum()+1)),
                'rows':len(y),'positive_count':int(y.sum()),'weighted_positive_count':float(weights@y),
                'parameter':float(parameter),'fallback':'WEIGHTED_SINGLE_CLASS_JEFFREYS',
                'warnings':[],'iterations':0,'elapsed_seconds':perf_counter()-started}
    if head == 'mu5':
        estimator = Ridge(alpha=parameter,solver='svd',fit_intercept=True)
    elif head == 'p10':
        estimator = LogisticRegression(C=parameter,l1_ratio=0.0,solver='lbfgs',class_weight=None,
            tol=1e-8,max_iter=3000,random_state=config['training']['seed'])
    else:
        estimator = QuantileRegressor(quantile=config['targets']['loss_quantile'],alpha=parameter,solver='highs',fit_intercept=True)
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
        warnings.simplefilter('always'); estimator.fit(X,y,sample_weight=weights)
    if any(issubclass(w.category,ConvergenceWarning) for w in caught):
        raise ValueError(f'P6 {head}加权拟合未收敛')
    coef=np.asarray(estimator.coef_).reshape(-1); intercept=float(np.asarray(estimator.intercept_).reshape(-1)[0])
    if not np.isfinite(coef).all() or not np.isfinite(intercept):
        raise ValueError('P6加权拟合出现非有限系数')
    iterations=getattr(estimator,'n_iter_',None)
    return {'kind':'logistic' if head=='p10' else 'linear','coef':coef.tolist(),'intercept':intercept,
            'parameter':float(parameter),'rows':len(y),'positive_count':int(y.sum()) if head=='p10' else None,
            'weighted_positive_count':float(weights@y) if head=='p10' else None,
            'iterations':int(np.max(iterations)) if iterations is not None else None,
            'warnings':[str(w.message) for w in caught],'fallback':None,'elapsed_seconds':perf_counter()-started}


def variant_eligibility(frame, position, config, variant, need_selection):
    train=training_rows(frame,position,config)  # All variants preserve complete-core cohort.
    required=CORE_IDS+(['G01','G02'] if variant['skew_cohort'] else [])
    if variant['skew_cohort']:
        train=train.loc[np.isfinite(train[['G01','G02']].to_numpy(float)).all(axis=1)]
    if not np.isfinite(frame[required].iloc[position].to_numpy(float)).all():
        return train,[],'MISSING_COHORT_FEATURES'
    if train.empty or train.index[-1]<config['data']['initial_train_end']:
        return train,[],'BEFORE_INITIAL_TRAIN_END'
    if len(train)<config['training']['min_outer_training_rows']:
        return train,[],'INSUFFICIENT_OUTER_TRAINING_ROWS'
    splits=inner_splits(frame,train,config) if need_selection else []
    if need_selection and (len(splits)!=config['training']['inner_validation_blocks'] or any(len(t)<config['training']['min_inner_training_rows'] or v.empty for t,v,_ in splits)):
        return train,splits,'INSUFFICIENT_INNER_TRAINING_ROWS'
    return train,splits,'ELIGIBLE'


def fit_variant(train, variant, parameters, config, positions, reference):
    design=VariantDesign(variant,config); X=design.fit(train[variant['columns']].to_numpy(float))
    weights=sample_weights(train,positions,reference,config['models']['challenger_recency_half_life_sessions']) if variant['recency'] else None
    heads={h:weighted_head(h,X,train[LABEL_FOR_HEAD[h]].to_numpy(float),parameters[h],config,weights) for h in HEADS}
    return JointModel(variant['name'],design,heads), weights


def select_variant(splits, variant, config, positions):
    scores,transforms,members=[] ,[],[]
    for train,validation,info in splits:
        design=VariantDesign(variant,config); X=design.fit(train[variant['columns']].to_numpy(float))
        V=design.transform(validation[variant['columns']].to_numpy(float))
        reference=positions[info['validation_information_first']]
        weights=sample_weights(train,positions,reference,config['models']['challenger_recency_half_life_sessions']) if variant['recency'] else None
        transforms.append({**info,'transform':design.snapshot(),'weight_reference_position':reference})
        for role,subset in [('train',train),('validation',validation)]:
            for i,(date,row) in enumerate(subset.iterrows()):
                members.append({'block':info['block'],'role':role,'as_of_session':date,'sample_weight':float(weights[i]) if role=='train' and weights is not None else 1.,
                    'validation_cutoff':info['validation_cutoff'],'label_matures_at':row.label_matures_at.isoformat(),
                    'label_available_at':row.label_available_at.isoformat(),'feature_available_at':row.assumed_available_at.isoformat()})
        for head in HEADS:
            for parameter in sorted(config['models'][GRID_FOR_HEAD[head]],reverse=head!='p10'):
                fitted=weighted_head(head,X,train[LABEL_FOR_HEAD[head]].to_numpy(float),parameter,config,weights)
                raw=head_score(head,fitted,V,len(V))
                scores.append({'model':variant['name'],'head':head,'parameter':float(parameter),**info,
                    'loss':prediction_loss(head,validation[LABEL_FOR_HEAD[head]].to_numpy(float),raw,config['targets']['loss_quantile']),
                    'fit':fitted})
    parameters,totals={},[]
    for head in HEADS:
        best=float('inf')
        for parameter in sorted(config['models'][GRID_FOR_HEAD[head]],reverse=head!='p10'):
            values=[s for s in scores if s['head']==head and s['parameter']==parameter]
            n=sum(s['validation_rows'] for s in values); value=sum(s['loss']*s['validation_rows'] for s in values)/n
            totals.append({'model':variant['name'],'head':head,'parameter':float(parameter),'loss':value,'validation_rows':n})
            if value<best-1e-12:
                parameters[head],best=float(parameter),value
    return parameters,scores,totals,transforms,members


def run_variant(frame, config, variant, output):
    output.mkdir(parents=True,exist_ok=False); (output/'models').mkdir(); (output/'selections').mkdir()
    write_json(output/'variant.json',variant)
    positions={d:i for i,d in enumerate(frame.index)}
    predictions,fits,members,inner,cv,totals,selections=[],[],[],[],[],[],[]
    active=None; active_month=None; selected_year=None; parameters=None
    started=perf_counter()
    for position,row in enumerate(frame.itertuples()):
        if not config['data']['outer_test_requested_start']<=row.decision_session<config['data']['locked_historical_start']:
            continue
        month,year=row.decision_session[:7],row.decision_session[:4]
        due=month!=active_month; choose=year!=selected_year
        train,splits,reason=variant_eligibility(frame,position,config,variant,due and choose)
        if due and reason=='ELIGIBLE':
            if choose:
                parameters,scores,aggregate,transforms,inner_members=select_variant(splits,variant,config,positions)
                selection_id=row.decision_session
                write_json(output/'selections'/f'{selection_id}.json',{'selection_id':selection_id,
                    'simulation_selection_at':row.decision_at.isoformat(),'parameters':parameters,'scores':scores,
                    'totals':aggregate,'fold_transforms':transforms,'computed_at_utc':datetime.now(timezone.utc).isoformat()})
                cv.extend({'selection_id':selection_id,**{k:v for k,v in s.items() if k!='fit'}} for s in scores)
                totals.extend({'selection_id':selection_id,**s} for s in aggregate)
                selections.append({'selection_id':selection_id,'model':variant['name'],**parameters})
                inner.extend({'selection_id':selection_id,**s} for s in inner_members)
                selected_year=year
            fit_start=perf_counter(); active,weights=fit_variant(train,variant,parameters,config,positions,position)
            fit_id=f"{row.decision_session}_{variant['name']}"
            w=weights if weights is not None else np.ones(len(train))
            summary={'fit_id':fit_id,'model':variant['name'],'fit_session':row.decision_session,'fit_information_session':row.Index,
                'selection_id':selection_id,**training_info(train,row.decision_at,position,frame,config),
                'variant_training_values_sha256':sha256(train[variant['columns']+TARGETS].to_csv().encode()).hexdigest(),
                'weight_sum':float(w.sum()),'effective_training_rows':float(w.sum()**2/(w@w)),
                'weight_reference_position':position,'design_columns':len(active.design.columns),
                'computed_at_utc':datetime.now(timezone.utc).isoformat(),'elapsed_seconds':perf_counter()-fit_start,
                'warning_count':sum(len(h['warnings']) for h in active.heads.values())}
            write_json(output/'models'/f'{fit_id}.json',{**active.snapshot(),'variant':variant,'fit':summary})
            fits.append(summary)
            for i,(date,r) in enumerate(train.iterrows()):
                members.append({'fit_id':fit_id,'fit_session':row.decision_session,'as_of_session':date,'sample_weight':float(w[i]),
                    'label_matures_at':r.label_matures_at.isoformat(),'label_available_at':r.label_available_at.isoformat(),
                    'feature_available_at':r.assumed_available_at.isoformat(),'Y10':int(r.Y10)})
            active_month=month
            pd.DataFrame(fits).to_csv(output/'fits.csv',index=False)
        required=CORE_IDS+(['G01','G02'] if variant['skew_cohort'] else [])
        usable=active is not None and np.isfinite(frame.loc[row.Index,required].to_numpy(float)).all() and row.assumed_available_at<=row.decision_at
        record={'as_of_session':row.Index,'decision_session':row.decision_session,'simulation_decision_at':row.decision_at.isoformat(),
            'execution_at':row.execution_at.isoformat(),'computed_at_utc':datetime.now(timezone.utc).isoformat(),'model':variant['name'],
            'forecast_available':bool(usable),'forecast_status':'FORECAST_AVAILABLE' if usable else reason if active is None else 'MISSING_COHORT_FEATURES',
            'fit_id':fit_id if usable else ''}
        if usable:
            raw=frame.loc[[row.Index],variant['columns']].to_numpy(float)
            record.update({k:v[0].item() for k,v in active.predict(raw).items()})
            outside,distance=active.design.support(raw)
            record.update({'outside_feature_count':int(outside.sum()),'outside_feature_ids':'|'.join(np.array(variant['columns'])[outside[0]]),
                           'max_outside_distance_training_sd':float(distance[0])})
        observed=bool(row.observed and np.isfinite([row.R5,row.L5,row.Y10]).all())
        record.update({k:float(getattr(row,k)) if observed else None for k in TARGETS})
        record.update({'score_observed':observed,'label_end_session':row.label_end_session})
        predictions.append(record)
    if active is None:
        raise ValueError(f"P6 {variant['name']}未满足首个训练条件；不降低门槛")
    for name,rows in [('predictions',predictions),('fits',fits),('training_rows',members),('inner_rows',inner),('cv_scores',cv),('cv_totals',totals),('selected_parameters',selections)]:
        pd.DataFrame(rows).to_csv(output/f'{name}.csv',index=False)
    result={'variant':variant['name'],'request_sessions':len(predictions),'forecast_sessions':sum(p['forecast_available'] for p in predictions),
        'first_forecast':next(p['decision_session'] for p in predictions if p['forecast_available']),
        'fits':len(fits),'selections':len(selections),'elapsed_seconds':perf_counter()-started,
        'warning_count':sum(f['warning_count'] for f in fits)}
    write_json(output/'result.json',result)
    print(f"P6 {variant['name']}：{result['forecast_sessions']}预测日、{len(fits)}月拟合、{len(selections)}年度选择，{result['elapsed_seconds']:.2f}秒。",flush=True)
    return pd.DataFrame(predictions),result
