"""手动日报：冻结的月度程序与真实接收时钟，无账户接口或模型搜索。"""
from datetime import datetime, timezone
import json
from pathlib import Path
import tomllib
import traceback

import exchange_calendars as xcals
import numpy as np
import pandas as pd

from svxylab.data import (CATALOG_URL, monthly_contracts, read_index, read_vx, select_front_three, with_provenance)
from svxylab.downloads import DownloadStore
from svxylab.etf import acquire_etfs
from svxylab.features import CORE_IDS, build_inputs, calculate_features, feature_availability
from svxylab.features_report import write_json
from svxylab.models import fit_joint, select_parameters
from svxylab.prediction_data import eligibility
from svxylab.predictions import training_info
from svxylab.release import digest, load_historical, read_json, stage_output, verify_release
from svxylab.release_audit import saved_prediction
from svxylab.returns import total_return
from svxylab.timing import session_clock, five_day_labels


def market_time(now):
    now=pd.Timestamp(now)
    if now.tzinfo is None:raise ValueError('日报时间必须带时区')
    schedule=xcals.get_calendar('XNYS',start=(now-pd.Timedelta(days=30)).date().isoformat(),
        end=(now+pd.Timedelta(days=40)).date().isoformat()).schedule
    closed=schedule.loc[pd.to_datetime(schedule.close,utc=True).lt(now)]
    information=closed.index[-1].strftime('%Y-%m-%d')
    return session_clock([information],schedule).iloc[0].to_dict()


def issue_status(timing, latest_information, complete, received_at, now):
    now=pd.Timestamp(now)
    if latest_information!=timing['as_of_session']:return 'STALE_INPUTS_NO_FORECAST'
    if not complete:return 'MISSING_CORE_INPUTS_NO_FORECAST'
    if pd.Timestamp(received_at)>now:return 'INPUT_NOT_YET_RECEIVED'
    if now>=pd.Timestamp(timing['decision_at']):return 'MISSED_DECISION_DEADLINE'
    if now<=pd.Timestamp(timing['information_close_at']):return 'INFORMATION_SESSION_NOT_CLOSED'
    return 'FORWARD_FORECAST_AVAILABLE'


def append_after_freeze(original, updates, cutoff, keys=('as_of_session',)):
    """冻结历史绝不由当前版本静默回填或修订。"""
    new=updates.loc[updates.as_of_session.gt(cutoff)].copy()
    result=pd.concat([original,new],ignore_index=True).sort_values(list(keys)).reset_index(drop=True)
    if result.duplicated(list(keys)).any():raise ValueError('日报新增日期/合约重复')
    return result


def obtain_inputs(root, directory, now):
    frame,_,_=load_historical(root)
    cutoff=read_json(root/'runs/p7/freeze.json')['historical_cutoff']
    timing=market_time(now);end=timing['as_of_session']
    p2=stage_output(root,'p2');p3=stage_output(root,'p3')
    if end<cutoff:raise ValueError('系统时钟早于已冻结行情，不产生前向记录')
    manifest=[json.loads(line) for line in (root/'data/raw/downloads.jsonl').read_text().splitlines()]
    inputs={'mode':'VERIFIED_FROZEN_CACHE','latest_information':cutoff,'last_completed_equity_session':end,
        'actual_received_at_max':max(r['retrieved_at'] for r in manifest if r['ok']),
        'historical_pit':'NO_FULL_HISTORICAL_PIT_CLAIM','network_requests':0,
        'files':{str(p.relative_to(root)):digest(p) for p in [p2/'SVXY_returns.csv',p2/'labels.csv',p3/'core_features.csv',p3/'feature_availability.csv']}}
    if end==cutoff:
        return frame,pd.read_csv(p2/'SVXY_returns.csv'),timing,inputs
    # Each acquisition is immutable and separate from P1. A repeated successful daily call reuses its original prediction.
    source=directory/'sources';store=DownloadStore(source)
    # The daily primary uses 19 core columns; optional SKEW service cannot block it.
    indices={'SKEW':pd.read_csv(root/'data/clean/SKEW_daily.csv')}
    for symbol in ['VIX','VVIX','VIX9D','VIX3M']:
        original=pd.read_csv(root/f'data/clean/{symbol}_daily.csv')
        url=next(r['url'] for r in reversed(manifest) if r['label']==symbol+'_history' and r['ok'])
        r=store.fetch(url,source='cboe',label=symbol+'_history',refresh=True,attempts=1)
        if not r['ok']:raise ValueError(f"{symbol}: {r['error_layer']} HTTP{r['http_status']}")
        update=read_index(source,r,symbol)
        indices[symbol]=append_after_freeze(original,update.loc[update.as_of_session.le(end)],cutoff)
    cat=store.fetch(CATALOG_URL,source='cboe',label='VX_contract_catalog',refresh=True,attempts=1)
    if not cat['ok']:raise ValueError(f"VX目录: {cat['error_layer']} HTTP{cat['http_status']}")
    contracts=monthly_contracts(read_json(source/cat['raw_file']),'2019-01-01',end)
    newer=[]
    for c in contracts:
        if c['expire_date']<=cutoff:continue
        r=store.fetch('https://cdn.cboe.com/'+c['path'],source='cboe',label='VX_'+c['expire_date'],refresh=True,attempts=1)
        if not r['ok']:raise ValueError(f"VX {c['expire_date']}: {r['error_layer']} HTTP{r['http_status']}")
        newer.append(read_vx(source,r,c))
    vx=append_after_freeze(pd.read_csv(root/'data/clean/VX_contract_daily.csv'),
        pd.concat(newer).loc[lambda x:x.as_of_session.le(end)],cutoff,('as_of_session','contract_id'))
    acquired,actions,records,failures=acquire_etfs(source,store,'2019-01-01',end)
    if failures or set(acquired)!={'SVXY','SPY'}:raise ValueError('ETF更新失败：'+json.dumps(failures,ensure_ascii=False))
    schedule=xcals.get_calendar('XNYS',start='2019-01-01',end=(pd.Timestamp(end)+pd.Timedelta(days=40)).date().isoformat()).schedule
    sessions=schedule.loc[:end].index.strftime('%Y-%m-%d').tolist();returns={}
    for symbol,update in acquired.items():
        update=with_provenance(update,records[symbol],symbol,'THEN_SHARE_PRICE')
        prices=append_after_freeze(pd.read_csv(root/f'data/clean/{symbol}_daily.csv'),update,cutoff)
        if prices.as_of_session.tolist()!=sessions:raise ValueError(symbol+'价格有缺口，不跳过持有损益')
        returns[symbol]=total_return(prices)
    curve=select_front_three(sessions,contracts,vx)
    raw=build_inputs(sessions,curve,vx,indices,returns,history_start='2019-01-01')
    features=calculate_features(raw);clock=session_clock(sessions,schedule)
    available=feature_availability(features,clock);labels=five_day_labels(returns['SVXY'],clock)
    result=features.join(labels.set_index('as_of_session')).join(available[['assumed_available_at']])
    for key in ['information_close_at','decision_at','execution_at','label_matures_at','label_available_at','assumed_available_at']:
        result[key]=pd.to_datetime(result[key],utc=True)
    # Preserve exactly the accepted training feature values and already-observed labels.
    result.loc[frame.index,CORE_IDS+['G01','G02']]=frame[CORE_IDS+['G01','G02']]
    mature=frame.index[frame.observed]
    result.loc[mature,['R5','L5','Y10']]=frame.loc[mature,['R5','L5','Y10']]
    result.to_csv(directory/'daily_training_inputs.csv');raw.to_csv(directory/'raw_feature_inputs.csv')
    returns['SVXY'].to_csv(directory/'SVXY_returns.csv',index=False)
    records=store.records
    inputs.update(mode='APPENDED_PUBLIC_SOURCE_UPDATE',latest_information=end,network_requests=len(records),
        actual_received_at_max=max(r['retrieved_at'] for r in records if r['ok']),
        new_raw_root=str(source.relative_to(root)),new_raw_manifest=str(store.manifest.relative_to(root)),new_raw_manifest_sha256=digest(store.manifest),
        historical_values_preserved_through=cutoff,
        files={str(p.relative_to(root)):digest(p) for p in directory.glob('*.csv')})
    return result,returns['SVXY'],timing,inputs


def fit_forward_if_due(frame, config, dictionary, saved, timing, now, directory):
    """只在真实手动运行的合资格月份更新；无回补过去决策的前向声明。"""
    month=timing['decision_session'][:7];year=month[:4]
    if all(s['fit']['fit_session'][:7]==month for s in saved.values()):return saved,False
    view=frame.copy();pos=view.index.get_loc(timing['as_of_session'])
    view.loc[timing['as_of_session'],'decision_at']=pd.Timestamp(now)
    need_selection=saved['M2']['fit']['selection_id'][:4]!=year
    train,splits,reason=eligibility(view,pos,config,need_selection=need_selection)
    if reason!='ELIGIBLE':raise ValueError('月度更新未满足冻结条件：'+reason)
    params={m:{h:saved[m]['heads'][h]['parameter'] for h in ['mu5','q90','p10']} for m in ['M1','M2']}
    selection_id=timing['decision_session'] if need_selection else saved['M2']['fit']['selection_id']
    if need_selection:
        selections={}
        for m in ['M1','M2']:
            params[m],scores,totals,transforms=select_parameters(splits,m,config,dictionary['interactions'])
            selections[m]={'parameters':params[m],'scores':scores,'totals':totals,'fold_transforms':transforms}
        write_json(directory/'forward_annual_selection.json',{'actual_selected_at':str(now),'selection_id':selection_id,'models':selections})
    common=training_info(train,pd.Timestamp(now),pos,view,config)
    train.to_csv(directory/'forward_training_members_and_values.csv')
    result={}
    for model in ['M0','M1','M2']:
        fitted=fit_joint(model,train,params.get(model,{}),config,dictionary['interactions'])
        result[model]={**fitted.snapshot(),'fit':{**common,'fit_id':timing['decision_session']+'_'+model+'_FORWARD',
            'fit_session':timing['decision_session'],'fit_information_session':timing['as_of_session'],
            'selection_id':selection_id,'computed_at_utc':str(now),'actual_forward_refit':True}}
    return result,True


def record_outcomes(root, prices, now, inputs):
    directory=root/'data/clean/forward';(directory/'outcomes').mkdir(parents=True,exist_ok=True)
    index=prices.set_index('as_of_session');dates=index.index.tolist();count=0
    for path in sorted((directory/'predictions').glob('*.json')):
        p=read_json(path);target=directory/'outcomes'/path.name
        if target.exists() or pd.Timestamp(now)<pd.Timestamp(p['timing']['label_matures_at']):continue
        start=p['timing']['execution_session'];end=p['timing']['label_end_session']
        if start not in dates or end not in dates:continue
        values=index.loc[start:end,'total_return_index'].to_numpy()
        if len(values)!=6 or not np.isfinite(values).all():continue
        outcome={'prediction_file':str(path.relative_to(root)),'prediction_sha256':digest(path),'observed_at_utc':str(now),
            'R5':float(values[-1]/values[0]-1),'L5':float(max(0,1-values[1:].min()/values[0])),
            'Y10':int(values[1:].min()<=.9*values[0]),'source_inputs':inputs,'source_label':'FIRST_RECORDED_COMPLETE_PATH'}
        write_json(target,outcome);count+=1
    return count


def update_daily(root, *, open_report=False):
    from svxylab.release_report import render_latest
    from svxylab.environment import run_command
    verify_release(root)
    now=pd.Timestamp(datetime.now(timezone.utc));stamp=now.strftime('%Y%m%dT%H%M%S%fZ')
    directory=root/'data/clean/daily'/stamp;directory.mkdir(parents=True)
    record_dir=root/'runs/p7/daily'/stamp;record_dir.mkdir(parents=True)
    record={'stage':'P7_DAILY','started_at':now.isoformat(),'full_historical_backtest_run':False,'llm_called':False,'orders':False,
        'output_dir':str(directory.relative_to(root)),'status':'STARTED','run_record':str((record_dir/'daily_run.json').relative_to(root))}
    forward=root/'data/clean/forward';(forward/'predictions').mkdir(parents=True,exist_ok=True)
    try:
        timing=market_time(now);existing=forward/'predictions'/f"{timing['decision_session']}.json"
        if existing.exists():
            issued=read_json(existing)
            record.update(status='EXISTING_FORWARD_RECORD_PRESERVED',forecast_file=str(existing.relative_to(root)),
                forecast_created_at=issued['created_at_utc'],forecast_sha256=digest(existing),timing=timing,new_prediction_created=False)
        else:
            frame,prices,timing,inputs=obtain_inputs(root,directory,now)
            # Acquiring files can cross the deadline; use the actual completion time for issuance.
            issued_at=pd.Timestamp(datetime.now(timezone.utc))
            complete=frame.loc[timing['as_of_session'],CORE_IDS].notna().all() if timing['as_of_session'] in frame.index else False
            status=issue_status(timing,inputs['latest_information'],complete,inputs['actual_received_at_max'],issued_at)
            record.update(status=status,inputs=inputs,timing=timing,new_prediction_created=False)
            record['newly_matured_outcomes']=record_outcomes(root,prices,issued_at,inputs)
            if status=='FORWARD_FORECAST_AVAILABLE':
                pointer=root/'runs/p7/forward_model_state.json'
                if pointer.exists():
                    pointer_value=read_json(pointer)
                    if digest(root/pointer_value['model_state_file'])!=pointer_value['sha256']:raise ValueError('前向模型快照摘要改变')
                    saved=read_json(root/pointer_value['model_state_file'])
                else:
                    release=read_json(root/read_json(root/'runs/p7/latest_run.json')['run_record'])
                    model_dir=root/release['result']['output_dir']/'core/models'
                    saved={m:read_json(sorted(model_dir.glob('*_'+m+'.json'))[-1]) for m in ['M0','M1','M2']}
                config=tomllib.loads((root/'experiment.toml').read_text());dictionary=read_json(root/'FEATURES.json')
                saved,refit=fit_forward_if_due(frame,config,dictionary,saved,timing,issued_at,directory)
                finished=pd.Timestamp(datetime.now(timezone.utc))
                if finished>=pd.Timestamp(timing['decision_at']):raise ValueError('拟合完成已过决策截止，本次无新预测')
                model_file=directory/'model_state.json';write_json(model_file,saved)
                write_json(pointer,{'model_state_file':str(model_file.relative_to(root)),'sha256':digest(model_file)})
                raw=frame.loc[[timing['as_of_session']],CORE_IDS].to_numpy();pred=[]
                for model,state in saved.items():
                    values={k:float(v[0]) for k,v in saved_prediction(state,raw).items()}
                    q=values['q90'];w=1. if q==0 else min(1.,config['portfolio']['loss_budget_base']/q)
                    risk=w;main=w if values['mu5']>2*config['portfolio']['cost_one_way_bps_base']/10000 else 0.
                    transform=state['transform']
                    outside=[] if transform is None else [k for k,x,lo,hi in zip(CORE_IDS,raw[0],transform['raw_min'],transform['raw_max']) if x<lo or x>hi]
                    pred.append({'model':model,'fit_id':state['fit']['fit_id'],**values,'main':main,'risk_only':risk,
                        'mode':'RESEARCH_ONLY','outside_training_range':outside,'R5':None,'L5':None,'Y10':None,'score_observed':False})
                issued={'created_at_utc':finished.isoformat(),'timing':timing,'inputs':inputs,'predictions':pred,
                    'core_features':frame.loc[timing['as_of_session'],CORE_IDS].to_dict(),
                    'model_state_file':str(model_file.relative_to(root)),'model_state_sha256':digest(model_file),
                    'freeze_sha256':digest(root/'runs/p7/freeze.json'),'monthly_refit_this_run':refit,
                    'forecast_kind':'ACTUAL_FORWARD_RECORD','historical_pit_claim':False,'primary':'M2','orders':False}
                write_json(existing,issued)
                record.update(forecast_file=str(existing.relative_to(root)),forecast_sha256=digest(existing),
                    forecast_created_at=issued['created_at_utc'],new_prediction_created=True,monthly_refit=refit)
        record['finished_at']=datetime.now(timezone.utc).isoformat()
    except Exception as error:
        record.update(status='UPDATE_FAILED_NO_NEW_FORECAST',error_type=type(error).__name__,error=str(error))
        (record_dir/'failure.txt').write_text(traceback.format_exc())
    write_json(root/record['run_record'],record);write_json(root/'runs/p7/latest_daily.json',{'run_record':record['run_record']})
    render_latest(root)
    if open_report:
        opened=run_command(['/usr/bin/open',str(root/'reports/latest.html')],root);record['open_command']=opened
        write_json(root/record['run_record'],record)
    print('日报：'+record['status']+'；'+str(root/'reports/latest.html'),flush=True)
    return 0 if record['status'] in ['FORWARD_FORECAST_AVAILABLE','EXISTING_FORWARD_RECORD_PRESERVED'] else 1
