"""真实保存预测的执行桥接反例；不拟合/选参，不改变原始产物。"""
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
from svxylab.release import load_historical, read_json
from svxylab.economics import model_targets, run_account, account_metrics
from svxylab.features_report import write_json

root=Path(__file__).resolve().parents[2]
frame,config,inputs=load_historical(root);prices,clock=inputs[:2]
source=root/'data/clean/p7/20260907T040716823855Z/core/predictions.csv'
predictions=pd.read_csv(source,float_precision='round_trip')
accepted=read_json(root/'runs/p4/acceptance.json')
prior=pd.read_csv(root/accepted['accepted_prediction_file'],float_precision='round_trip').groupby('model',sort=False).tail(1)
start=predictions.decision_session.min();dates=prices.as_of_session.tolist()
assert set(prior.as_of_session)=={dates[dates.index(start)-2]}
directory=root/'data/clean/p7_initialization_counterexample'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');directory.mkdir(parents=True)
old,mo=model_targets(predictions,dates,.05,.0005)
new,mn=model_targets(pd.concat([prior,predictions],ignore_index=True),dates,.05,.0005)
results=[]
for name in old:
    a=run_account(prices,clock,old[name],start,100000.,.0005,metadata=mo[name],delay=1)
    b=run_account(prices,clock,new[name],start,100000.,.0005,metadata=mn[name],delay=1)
    a.to_csv(directory/(name+'_before.csv'),index=False);b.to_csv(directory/(name+'_after.csv'),index=False)
    impact=a.as_of_session[(a.equity_end-b.equity_end).abs()>1e-6].tolist()
    results.append({'controller':name,'first_execution':start,'bridge_information_session':prior.as_of_session.iloc[0],
        'before_first_signal_missing':bool(a.signal_missing.iloc[1]),'after_first_signal_missing':bool(b.signal_missing.iloc[1]),
        'after_first_target':float(b.target_weight.iloc[1]),'first_fee_difference':float(b.cost.iloc[1]-a.cost.iloc[1]),
        'net_return_before':account_metrics(a,100000.)['total_return'],'net_return_after':account_metrics(b,100000.)['total_return'],
        'equity_level_affected_dates':impact})
write_json(root/'runs/p7/initialization_counterexample.json',{'data_directory':str(directory.relative_to(root)),
    'source_new_predictions':str(source.relative_to(root)),'source_prior_predictions':accepted['accepted_prediction_file'],
    'description':'Before is the faulty P7 initialization that drops existing P4 information, not an accepted P5 result. All model forecasts remain fixed. Only the one-session delayed account is affected.','results':results})
print(pd.DataFrame([{k:v for k,v in r.items() if k!='equity_level_affected_dates'} for r in results]).to_string(index=False))
