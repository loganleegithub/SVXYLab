"""P7：固定版本的历史评估；原阶段的开发边界和冻结文件保持不变。"""
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import shutil
import tomllib

import numpy as np
import pandas as pd

from svxylab.diagnostic_models import run_variant, variants
from svxylab.diagnostics import evaluate
from svxylab.economics import (PRICE_FIELDS, CLOCK_FIELDS, account_metrics, model_targets,
    run_account, scenarios, select_cases, arithmetic_attribution, risk_diagnostics)
from svxylab.features_report import write_json, verify_hashes
from svxylab.ledger import baseline_targets
from svxylab.prediction_data import load_development_data, development_csv
from svxylab.predictions import run_predictions


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def stage_output(root, stage):
    accepted=read_json(root/f'runs/{stage}/acceptance.json')
    return root/read_json(root/accepted['accepted_run'])['result']['output_dir']


def runtime_config(config, cutoff):
    adapted=deepcopy(config)
    adapted['data']['outer_test_requested_start']=config['data']['locked_historical_start']
    adapted['data']['locked_historical_start']=(pd.Timestamp(cutoff)+pd.Timedelta(days=1)).date().isoformat()
    return adapted


def source_files(root):
    return sorted((root/'src').rglob('*.py'))+sorted((root/'tests').rglob('*.py'))+[root/p for p in
        ['experiment.toml','FEATURES.json','RESEARCH_SPEC.md','requirements-lock.txt','pyproject.toml',
         '更新日报.command','重新运行研究.command','runs/p7/experiment_record.json']]


def verify_release(root):
    freeze=read_json(root/'runs/p7/freeze.json')
    verify_hashes(root,freeze['source_sha256'])
    for stage in ['p4','p5','p6']:
        a=read_json(root/f'runs/{stage}/acceptance.json')
        verify_hashes(root,{a['accepted_run']:a['accepted_run_sha256'],a['accepted_report']:a['accepted_report_sha256']})
        verify_hashes(root,a['accepted_artifact_sha256'])
    return freeze


def load_historical(root):
    freeze=verify_release(root)
    config=tomllib.loads((root/'experiment.toml').read_text())
    end=freeze['historical_cutoff']; adapted=runtime_config(config,end)
    # Original boundary reader is reused with explicit authorized evaluation bounds.
    frame,inputs=load_development_data(root,adapted)
    p2,p3=stage_output(root,'p2'),stage_output(root,'p3')
    ext=development_csv(p3/'extension_features.csv',adapted['data']['locked_historical_start'])
    frame=frame.join(ext.apply(pd.to_numeric,errors='raise'))
    prices=pd.read_csv(p2/'SVXY_returns.csv',usecols=PRICE_FIELDS)
    clock=pd.read_csv(p2/'clock.csv',usecols=CLOCK_FIELDS)
    curve=pd.read_csv(root/'data/clean/VX_front_three.csv')
    features=pd.read_csv(p3/'core_features.csv')
    prices=prices.loc[prices.as_of_session.le(end)].reset_index(drop=True)
    clock=clock.loc[clock.as_of_session.le(end)].reset_index(drop=True)
    curve=curve.loc[curve.as_of_session.le(end)].reset_index(drop=True)
    features=features.loc[features.as_of_session.le(end)].reset_index(drop=True)
    assert frame.index.tolist()==prices.as_of_session.tolist()==clock.as_of_session.tolist()==curve.as_of_session.tolist()
    inputs={k.replace('development_','historical_'):v for k,v in inputs.items()}
    inputs.update(label_values_from_locked_period_used=True,evaluation_name='本轮未参与选择的历史评估',
        actual_requested_start=config['data']['locked_historical_start'],last_executable_close=end,
        runtime_bounds=adapted['data'],freeze_record='runs/p7/freeze.json')
    return frame,adapted,(prices,clock,None,curve,features,adapted,inputs)


def economic_supplements(config, predictions, account_inputs, output):
    prices,clock,_,curve,features,_,_=account_inputs
    dates=prices.as_of_session.tolist(); capital=config['portfolio']['research_capital']
    first=max(p.loc[p.forecast_available,'decision_session'].min() for p in predictions.values())
    scopes={'full':min(predictions['M0'].decision_session),'common':first}
    combined=pd.concat([predictions[m].assign(model=m) for m in ['M0','M1','M2']],ignore_index=True)
    metrics=[]; ledgers={}; gross=[]; matches=[]; attributions=[]
    directory=output/'sensitivity';directory.mkdir();(directory/'ledgers').mkdir()
    for spec in scenarios(config):
        targets,meta=model_targets(combined,dates,spec['budget'],spec['cost_rate'])
        targets.update(baseline_targets(curve,config['evaluation']['fixed_weight_baselines']))
        for cohort,start in scopes.items():
            for strategy,target in targets.items():
                ledger=run_account(prices,clock,target,start,capital,spec['cost_rate'],metadata=meta.get(strategy),delay=spec['delay'])
                path=f'ledgers/{cohort}_{spec["scenario"]}_{strategy}.csv';ledger.to_csv(directory/path,index=False)
                metrics.append({'cohort':cohort,'strategy':strategy,**spec,'ledger_file':path,**account_metrics(ledger,capital)})
                ledgers[cohort,spec['scenario'],strategy]=ledger
                if spec['scenario']=='base' and strategy.startswith('M'):
                    uncosted=run_account(prices,clock,target,start,capital,0.,metadata=meta[strategy])
                    path_g=f'ledgers/{cohort}_same_target_gross_{strategy}.csv';uncosted.to_csv(directory/path_g,index=False)
                    gross.append({'cohort':cohort,'strategy':strategy,'gross_ledger_file':path_g,
                        'net_return':account_metrics(ledger,capital)['total_return'],'same_target_gross_return':account_metrics(uncosted,capital)['total_return']})
                    avg=ledger.iloc[1:].old_weight.mean()
                    match=run_account(prices,clock,pd.Series(avg,index=pd.Index(dates,name='as_of_session')),start,capital,spec['cost_rate'])
                    path_m=f'ledgers/{cohort}_post_hoc_{strategy}.csv';match.to_csv(directory/path_m,index=False)
                    matches.append({'cohort':cohort,'controller':strategy,'post_hoc_fixed_target':avg,'ledger_file':path_m,
                        'label':'POST_HOC_NON_EXECUTABLE_SELECTION',**account_metrics(match,capital)})
    for cohort in scopes:
        for model in ['M0','M1','M2']:
            a,b=ledgers[cohort,'base',model+'_main'],ledgers[cohort,'base',model+'_risk_only']
            attributions.append({'cohort':cohort,'model':model,'comparison':'main_minus_risk_only',**arithmetic_attribution(a,b)})
    pd.DataFrame(metrics).to_csv(directory/'account_metrics.csv',index=False)
    pd.DataFrame(gross).to_csv(directory/'same_target_gross_net.csv',index=False)
    pd.DataFrame(matches).to_csv(directory/'post_hoc_exposure_matches.csv',index=False)
    pd.DataFrame(attributions).to_csv(directory/'mu_filter_attribution.csv',index=False)
    risk=risk_diagnostics(prices,combined,.05,.0005)
    risk.rename(columns={'path_complete_in_development':'path_complete_in_historical_evaluation'}).to_csv(directory/'risk_capacity_diagnostics.csv',index=False)
    selected=select_cases(ledgers,prices,combined,features,directory)
    return {'scopes':scopes,'accounts':len(metrics),'same_target_gross_accounts':len(gross),
        'post_hoc_accounts':len(matches),'cases':selected}


def run_release(root, output):
    frame,config,account_inputs=load_historical(root)
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'input_verification.json',account_inputs[-1]);write_json(output/'runtime_config.json',config)
    dictionary=read_json(root/'FEATURES.json')
    core=run_predictions(frame,config,dictionary['interactions'],output/'core')
    core_rows=pd.read_csv(output/'core/predictions.csv',float_precision='round_trip')
    # The reused P4 text field names its original stage; clarify the authorized new scope without changing numbers.
    core_rows['score_status']=np.where(core_rows.score_observed,'HISTORICAL_EVALUATION_LABEL_COMPLETE',core_rows.score_status)
    core_rows.to_csv(output/'core/predictions.csv',index=False)
    predictions={m:core_rows.loc[core_rows.model.eq(m)].copy() for m in ['M0','M1','M2']}
    specs=variants(dictionary);write_json(output/'variants.json',specs)
    results=[]
    for variant in specs:
        directory=output/'variants'/variant['name']
        results.append(run_variant(frame,config,variant,directory))
        p=pd.read_csv(directory/'predictions.csv',float_precision='round_trip')
        if variant['name']=='M2_REPLAY':
            a=predictions['M2'];b=p
            assert a.as_of_session.tolist()==b.as_of_session.tolist()
            fields=['mu5_raw','mu5','q90_raw','q90','p10_logit','p10']
            np.testing.assert_allclose(a[fields],b[fields],atol=2e-11,rtol=2e-10,equal_nan=True)
            parity=float(np.nanmax(np.abs(a[fields].to_numpy()-b[fields].to_numpy())))
        else:predictions[variant['name']]=p
    evaluation=evaluate(root,frame,config,predictions,output,account_inputs=account_inputs)
    supplements=economic_supplements(config,{m:predictions[m] for m in ['M0','M1','M2']},account_inputs,output)
    return {'output_dir':output.relative_to(root).as_posix(),'core_run':core,'variant_runs':results,
        'evaluation':evaluation,'economic_supplements':supplements,'m2_refit_reference_max_error':parity,
        'evaluation_name':'本轮未参与选择的历史评估','primary':'M2','automatic_promotion':False}
