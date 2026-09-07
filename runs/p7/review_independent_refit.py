"""P7审查独立重拟合：仅读本地真实输入，stdout输出，无项目写入。"""
from pathlib import Path
import json,numpy as np,pandas as pd
from sklearn.preprocessing import StandardScaler,SplineTransformer
from sklearn.linear_model import LogisticRegression,QuantileRegressor
from scipy.special import expit
from threadpoolctl import threadpool_limits
from svxylab.release import load_historical,read_json
from svxylab.features import CORE_IDS
from svxylab.diagnostics_audit import check_transform
root=Path.cwd();frame,cfg,_=load_historical(root);out=root/read_json(root/read_json(root/'runs/p7/latest_run.json')['run_record'])['result']['output_dir'];p=pd.read_csv(out/'core/predictions.csv',float_precision='round_trip');pairs=[tuple(CORE_IDS.index(k) for k in i['features']) for i in read_json(root/'FEATURES.json')['interactions']]
count=rows=0;maximum=0.
with threadpool_limits(limits=1):
 for path in sorted((out/'core/models').glob('*.json')):
  s=read_json(path);model=s['fit']['model']
  if model=='M0':continue
  f=s['fit'];cut=pd.Timestamp(f['simulation_fit_at']);pos=frame.index.get_loc(f['fit_information_session']);window=frame.iloc[max(0,pos-1259):pos+1]
  tr=window.loc[window.observed & np.isfinite(window[CORE_IDS+['R5','L5','Y10']]).all(axis=1) & window.label_matures_at.lt(cut) & window.label_available_at.le(cut) & window.assumed_available_at.le(cut)].copy()
  g=p.loc[p.fit_id.eq(f['fit_id'])];raw=tr[CORE_IDS].to_numpy(float);test=frame.loc[g.as_of_session,CORE_IDS].to_numpy(float);scaler=StandardScaler();X=scaler.fit_transform(raw);V=scaler.transform(test)
  if model=='M2':
   spline=SplineTransformer(n_knots=3,degree=2,knots='quantile',include_bias=False,extrapolation='linear');XB=spline.fit_transform(X);VB=spline.transform(V)
   products=np.column_stack([X[:,a]*X[:,b] for a,b in pairs]);testproducts=np.column_stack([V[:,a]*V[:,b] for a,b in pairs]);Xs=np.column_stack([XB,products]);Vs=np.column_stack([VB,testproducts]);scale=StandardScaler();X=scale.fit_transform(Xs);V=scale.transform(Vs)
  predictions={};params={h:s['heads'][h]['parameter'] for h in ['mu5','q90','p10']}
  xm=X.mean(axis=0);ym=tr.R5.mean();U,z,VT=np.linalg.svd(X-xm,full_matrices=False);coef=VT.T@((z/(z*z+params['mu5']))*(U.T@(tr.R5.to_numpy()-ym)));predictions['mu5_raw']=V@coef+ym-xm@coef
  lr=LogisticRegression(C=params['p10'],l1_ratio=0.,solver='lbfgs',class_weight=None,tol=1e-8,max_iter=3000,random_state=1707).fit(X,tr.Y10.to_numpy(float));predictions['p10_logit']=lr.decision_function(V)
  qr=QuantileRegressor(quantile=.9,alpha=params['q90'],solver='highs',fit_intercept=True).fit(X,tr.L5.to_numpy(float));predictions['q90_raw']=qr.predict(V)
  predictions.update(mu5=np.maximum(predictions['mu5_raw'],-1),q90=np.clip(predictions['q90_raw'],0,1),p10=expit(predictions['p10_logit']))
  for k,v in predictions.items():
   err=float(np.max(abs(v-g[k])));maximum=max(maximum,err);np.testing.assert_allclose(v,g[k],atol=2e-11,rtol=2e-10,err_msg=f['fit_id']+' '+k)
  count+=1;rows+=len(g)
inner=pd.read_csv(out/'core/inner_rows.csv');members=pd.read_csv(out/'core/training_rows.csv');folds=0
for path in (out/'core/selections').glob('*.json'):
 s=read_json(path);outer=members.loc[members.fit_session.eq(s['selection_id'])];last=frame.index.get_loc(outer.as_of_session.iloc[-1]);first=last+1-189;of=frame.loc[outer.as_of_session]
 for model,m in s['models'].items():
  for fold in m['fold_transforms']:
   dates=frame.index[first+(fold['block']-1)*63:first+fold['block']*63];cut=frame.loc[dates[0],'decision_at'];t=of.loc[(of.index<dates[0])&of.label_matures_at.lt(cut)&of.label_available_at.le(cut)&of.assumed_available_at.le(cut)];v=of.loc[of.index.isin(dates)];actual=inner.loc[inner.selection_id.eq(s['selection_id'])&inner.block.eq(fold['block'])]
   assert actual.loc[actual.role.eq('train'),'as_of_session'].tolist()==t.index.tolist();assert actual.loc[actual.role.eq('validation'),'as_of_session'].tolist()==v.index.tolist();raw=t[CORE_IDS].to_numpy();state=fold['transform']
   if model=='M2':check_transform(state,raw)
   else:np.testing.assert_allclose(state['raw_scaler']['mean'],raw.mean(axis=0),atol=2e-12);np.testing.assert_allclose(state['raw_scaler']['scale'],raw.std(axis=0),atol=2e-12)
   folds+=1
print(json.dumps({'independent_outer_refits':count,'refitted_heads':count*3,'forecast_rows':rows,'max_raw_or_published_prediction_error':maximum,'additional_core_inner_transforms_and_time_members_verified':folds,'writes':0}))
