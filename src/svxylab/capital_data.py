"""N1唯一新增行情：BIL真实Yahoo Chart原件；旧输入只读验证。"""
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlencode
from zoneinfo import ZoneInfo
import json
import numpy as np
import pandas as pd
from svxylab.downloads import DownloadStore
from svxylab.etf import read_yahoo
from svxylab.returns import total_return
from svxylab.features_report import verify_hashes, write_json
from svxylab.release import digest, read_json, stage_output

BIL_DOCUMENT='https://www.ssga.com/us/en/intermediary/etfs/state-street-spdr-bloomberg-1-3-month-t-bill-etf-bil'


def obtain_bil(root, dates):
    base=root/'data/raw/n1';store=DownloadStore(base)
    start='2019-01-01';end='2026-09-04'
    ny=ZoneInfo('America/New_York')
    params={'period1':int(datetime.fromisoformat(start).replace(tzinfo=ny).timestamp()),
            'period2':int((datetime.fromisoformat(end)+timedelta(days=1)).replace(tzinfo=ny).timestamp()),
            'interval':'1d','events':'div,splits,capitalGains'}
    url='https://query1.finance.yahoo.com/v8/finance/chart/BIL?'+urlencode(params)
    candidates=[r for r in store.records if r['label']=='BIL_history_full' and r['url']==url and r['ok']]
    record=candidates[-1] if candidates else None
    if record:
        verify_hashes(base,{record['raw_file']:record['sha256']})
    else:
        from curl_cffi.requests import Session
        with Session(impersonate='chrome') as session:
            for attempt in range(2):
                try:
                    response=session.get(url,timeout=20)
                    record=store.save_response(response.content,url=url,source='yahoo',label='BIL_history_full',
                        http_status=response.status_code,client='curl_cffi_0.16.3')
                except Exception as e:
                    record=store.save_response(b'',url=url,source='yahoo',label='BIL_history_full',http_status=0,
                        client='curl_cffi_0.16.3',error_layer=type(e).__name__)
                if record['ok'] or record['http_status'] not in [0,429,500,502,503,504]:break
        if not record['ok']:
            raise ValueError(f"BIL真实数据未取得：{record['error_layer']} HTTP {record['http_status']}，不合成替代")
    prices,actions=read_yahoo(base,record,'BIL',start,end)
    if prices.as_of_session.tolist()!=dates:raise ValueError('BIL与冻结股票日历不完整一致；不填价或压缩日期')
    returns=total_return(prices)
    audit={'symbol':'BIL','rows':len(prices),'first':dates[0],'last':dates[-1],
        'raw_file':str((base/record['raw_file']).relative_to(root)),'raw_sha256':record['sha256'],
        'retrieved_at':record['retrieved_at'],'source_url':record['url'],'fund_document':BIL_DOCUMENT,
        'distribution_events':len([x for x in actions if x['kind']!='split']),
        'split_events':len([x for x in actions if x['kind']=='split']),
        'max_daily_difference_to_provider_adjusted_bps':float(returns.reference_difference_bps.abs().max()),
        'full_tr_return':float(returns.total_return_index.iloc[-1]/100-1),
        'provider_adjusted_reference_return':float(prices.provider_adjusted_close.iloc[-1]/prices.provider_adjusted_close.iloc[0]-1),
        'accounting':'EX_DATE_CLOSE_REINVESTMENT_PROXY_NOT_PAYABLE_DATE_CASH',
        'actual_user_cash_return':False,'today_yield_used':False,'full_historical_vintages_available':False,
        'redistribution_authorized':False,'original_fund_fees_deducted_again':False}
    return returns,actions,audit


def load_capital_inputs(root):
    preserved=read_json(root/'runs/n1/experiment_record.json')['preserved_sha256']
    verify_hashes(root,preserved)
    for stage in ['p1','p2','p3','p4','p5','p6','p7']:
        a=read_json(root/f'runs/{stage}/acceptance.json')
        if 'accepted_artifact_sha256' in a:verify_hashes(root,a['accepted_artifact_sha256'])
        if 'accepted_run_sha256' in a:verify_hashes(root,{a['accepted_run']:a['accepted_run_sha256']})
    p2=stage_output(root,'p2');p7=stage_output(root,'p7');p5=stage_output(root,'p5')
    prices={s:pd.read_csv(p2/f'{s}_returns.csv',float_precision='round_trip') for s in ['SVXY','SPY']}
    clock=pd.read_csv(p2/'clock.csv');curve=pd.read_csv(root/'data/clean/VX_front_three.csv')
    p4=read_json(root/'runs/p4/acceptance.json')['accepted_prediction_file']
    from svxylab.economics import PREDICTION_FIELDS
    prediction_files=[root/p4,p7/'core/predictions.csv']
    predictions=pd.concat([pd.read_csv(p,usecols=PREDICTION_FIELDS,float_precision='round_trip') for p in prediction_files],ignore_index=True)
    if predictions.duplicated(['model','as_of_session']).any():raise ValueError('跨2024拼接重复预测')
    dates=prices['SVXY'].as_of_session.tolist()
    assert dates==prices['SPY'].as_of_session.tolist()==clock.as_of_session.tolist()==curve.as_of_session.tolist()
    prices['BIL'],actions,cash_audit=obtain_bil(root,dates)
    paths=[p2/'SVXY_returns.csv',p2/'SPY_returns.csv',p2/'clock.csv',root/'data/clean/VX_front_three.csv',*prediction_files]
    return prices,clock,curve,predictions,{'sha256':{str(p.relative_to(root)):digest(p) for p in paths},'cash_proxy':cash_audit,'bil_actions':actions,'p5_dir':str(p5.relative_to(root)),'p7_dir':str(p7.relative_to(root)), 'labels_read':False,'models_refit':False}
