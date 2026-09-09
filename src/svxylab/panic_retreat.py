"""E1 causal panic episodes and fixed-horizon SVXY research, isolated from M2.

Signal functions receive no asset prices, outcome labels or predictions. Targets
passed to the existing ledger are placed at execution minus one, exactly once.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import tomllib

import exchange_calendars as xcals
import numpy as np
import pandas as pd

from svxylab.ledger import continuous_ledger, baseline_targets
from svxylab.returns import total_return

RULES = ['A_SHOCK', 'B_RETREAT', 'C_RETREAT_VX', 'D3_CLOCK_WAIT', 'D5_CLOCK_WAIT']
PAIRS = [('B_RETREAT','A_SHOCK'), ('B_RETREAT','D3_CLOCK_WAIT'),
         ('B_RETREAT','D5_CLOCK_WAIT'), ('C_RETREAT_VX','B_RETREAT')]
PRICE_FIELDS = ['close','split_factor','cash_dividend','capital_gain_distribution']
ROOT = Path(__file__).resolve().parents[2]


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k,v in value.items()}
    if isinstance(value, (list,tuple,np.ndarray)):
        return [jsonable(v) for v in value]
    if isinstance(value, (float,np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer): return int(value)
    if isinstance(value, np.bool_): return bool(value)
    if isinstance(value, Path): return str(value)
    return value


def write_json(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(jsonable(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def digest(path): return sha256(path.read_bytes()).hexdigest()


def read_bounded(path, end):
    # In signal-only mode no future numerical rows are deserialized.
    with path.open(newline='') as stream:
        rows=[r for r in csv.DictReader(stream) if '2019-01-02'<=r['as_of_session']<=end]
    return pd.DataFrame(rows)


def vx_evidence(dates, curve, vx):
    """Use each current unexpired monthly ID at BOTH adjacent stock sessions."""
    if curve.as_of_session.duplicated().any() or vx.duplicated(['contract_id','as_of_session']).any():
        raise ValueError('Duplicate curve or contract-date')
    if len(vx) and not vx.duration_type.eq('M').all(): raise ValueError('Only monthly VX')
    front=curve.set_index('as_of_session').reindex(dates)
    lookup=vx.set_index(['contract_id','as_of_session'])['value'].to_dict()
    rows=[]
    for i,date in enumerate(dates):
        row={'as_of_session':date,'previous_session':dates[i-1] if i else None}
        flags=[]
        expirations=[]
        for n in [1,2]:
            f=front.iloc[i]; cid=f.get(f'f{n}_contract'); expiry=f.get(f'f{n}_expiration')
            now=float(lookup.get((cid,date),np.nan)); prev=float(lookup.get((cid,dates[i-1]),np.nan)) if i else np.nan
            valid=(pd.notna(expiry) and expiry>date and np.isfinite([now,prev]).all() and min(now,prev)>0)
            if f'f{n}_settle' in f and np.isfinite(now) and pd.notna(f[f'f{n}_settle']):
                if not np.isclose(float(f[f'f{n}_settle']),now,rtol=0,atol=1e-10):
                    raise ValueError('Curve and same-contract settlement disagree')
            row.update({f'f{n}_contract':cid,f'f{n}_expiration':expiry,f'f{n}_current':now,
                        f'f{n}_previous':prev,f'f{n}_change':now/prev-1 if valid else np.nan})
            flags.append(bool(valid));expirations.append(expiry)
        complete=all(flags) and expirations[0]<expirations[1]
        row['vx_complete']=bool(complete)
        row['vx_down']=bool(complete and row['f1_current']<row['f1_previous'] and row['f2_current']<row['f2_previous'])
        rows.append(row)
    return pd.DataFrame(rows)


def scan_signals(market, cfg):
    """One forward pass; running peak and observed signals never rewritten.

    Episodes may receive a later lifecycle end, but each daily evidence row is
    immutable. Missing peak data prevents B/C until a known low streak rearms.
    """
    m=market.reset_index(drop=True)
    if m.as_of_session.duplicated().any() or not m.as_of_session.is_monotonic_increasing:
        raise ValueError('Expected complete increasing stock calendar')
    v=pd.to_numeric(m.vix,errors='coerce').to_numpy(float)
    episodes=[]; records=[]; active=None; low=0; seen_rearm=False; prior_gap=False
    for i,row in m.iterrows():
        x=v[i];valid=np.isfinite(x) and x>0
        low=low+1 if valid and x<cfg['rearm_level'] else 0
        if not valid: prior_gap=True
        if low>=cfg['rearm_days']: seen_rearm=True; prior_gap=False
        lb=cfg['lookback']
        lookback_ok=i>=lb and np.isfinite(v[i-lb:i+1]).all() and (v[i-lb:i+1]>0).all()
        rise=x/v[i-lb]-1 if lookback_ok else np.nan
        # Price-level multiplication preserves the exact decimal threshold boundary.
        shock=bool(lookback_ok and x>=cfg['level'] and x>=v[i-lb]*(1+cfg['rise'])-1e-12)
        if active is None and shock:
            active={'episode_id':'E_'+row.as_of_session,'s':i,'shock_session':row.as_of_session,
                'shock_vix':x,'shock_rise5':rise,'left_truncated':not seen_rearm,
                'detection_gap':prior_gap,'end_i':np.nan,'end_session':None,
                'signals':{r:None for r in RULES},'gaps':{r:False for r in RULES},
                'peak':x,'peak_known':True,'repeated_shocks':0}
            episodes.append(active)
        if active is None: continue
        age=i-active['s']
        if not valid: active['peak_known']=False
        if valid and active['peak_known']: active['peak']=max(active['peak'],x)
        peak=active['peak'] if active['peak_known'] else np.nan
        retreat_known=bool(valid and active['peak_known'])
        retreat=bool(retreat_known and x<=peak*(1-cfg['retreat'])+1e-12)
        vx_ok=bool(row.get('vx_complete',False));vx_down=bool(row.get('vx_down',False))
        if age and shock: active['repeated_shocks']+=1
        known={'A_SHOCK':True,'B_RETREAT':retreat_known,
               'C_RETREAT_VX':retreat_known and (not retreat or vx_ok),'D3_CLOCK_WAIT':True,'D5_CLOCK_WAIT':True}
        hit={'A_SHOCK':age==0,'B_RETREAT':1<=age<=cfg['window'] and retreat,
             'C_RETREAT_VX':1<=age<=cfg['window'] and retreat and vx_ok and vx_down,
             'D3_CLOCK_WAIT':age==3,'D5_CLOCK_WAIT':age==5}
        for rule in RULES:
            if active['signals'][rule] is None:
                if 1<=age<=cfg['window'] and not known[rule]: active['gaps'][rule]=True
                if hit[rule]:
                    active['signals'][rule]={'t':i,'signal_session':row.as_of_session,
                        'signal_status':'TRIGGERED_AFTER_GAP' if active['gaps'][rule] else 'TRIGGERED',
                        'signal_vix':x,'signal_peak':peak,'first_trigger_complete':not active['gaps'][rule]}
        record={'episode_id':active['episode_id'],'i':i,'as_of_session':row.as_of_session,'age':age,
                'vix':x,'rise5':rise,'shock_condition':shock,'peak_so_far':peak,'peak_known':active['peak_known'],
                'retreat_known':retreat_known,'retreat_condition':retreat,'low_streak':low,
                'in_waiting_window':1<=age<=cfg['window'],'vx_complete':vx_ok,'vx_down':vx_down}
        record.update({k:row[k] for k in row.index if k.startswith('f1_') or k.startswith('f2_') or k=='previous_session'})
        records.append(record)
        if age>=cfg['window'] and low>=cfg['rearm_days']:
            active['end_i']=i;active['end_session']=row.as_of_session;active=None
    eps=[];signals=[]
    for ep in episodes:
        end=ep['s']+cfg['window'];complete=end<len(m)
        eps.append({k:v for k,v in ep.items() if k not in ['signals','gaps','peak','peak_known']}|
                   {'window_end_i':end,'window_complete':complete,'lifecycle_status':'ENDED' if pd.notna(ep['end_i']) else 'ACTIVE_AT_CUTOFF'})
        for rule in RULES:
            result=ep['signals'][rule]
            if result is None:
                needed={'A_SHOCK':0,'D3_CLOCK_WAIT':3,'D5_CLOCK_WAIT':5}.get(rule,cfg['window'])
                status='WINDOW_PENDING' if ep['s']+needed>=len(m) else 'DATA_GAP' if ep['gaps'][rule] else 'NATURAL_NO_TRIGGER'
                result={'t':np.nan,'signal_session':None,'signal_status':status,'signal_vix':np.nan,
                        'signal_peak':np.nan,'first_trigger_complete':False if ep['gaps'][rule] else True}
            signals.append({'episode_id':ep['episode_id'],'s':ep['s'],'shock_session':ep['shock_session'],
                            'rule_id':rule,'window_complete':complete,'detection_gap':ep['detection_gap'],**result})
    epcols=['episode_id','s','shock_session','shock_vix','shock_rise5','left_truncated','detection_gap','end_i','end_session','repeated_shocks','window_end_i','window_complete','lifecycle_status']
    scols=['episode_id','s','shock_session','rule_id','window_complete','detection_gap','t','signal_session','signal_status','signal_vix','signal_peak','first_trigger_complete']
    return pd.DataFrame(eps,columns=epcols),pd.DataFrame(signals,columns=scols),pd.DataFrame(records)


def inference_clusters(episodes,cfg):
    rows=[];right=-1;cluster=0
    for ep in episodes.sort_values('s').itertuples():
        end=ep.s+cfg['window']+cfg['lag']+max(cfg['horizons'])
        if ep.s>right:cluster+=1
        right=max(right,end)
        rows.append({'episode_id':ep.episode_id,'cluster_id':f'K{cluster:03d}','start_i':ep.s,'end_i':end})
    return pd.DataFrame(rows,columns=['episode_id','cluster_id','start_i','end_i'])


def valid_prices(prices):
    x=prices[PRICE_FIELDS].to_numpy(float)
    return np.isfinite(x).all(axis=1)&(x[:,:2]>0).all(axis=1)&(x[:,2:]>=0).all(axis=1)


def fixed_trade(prices,clock,e,u,capital,cost):
    stop=min(u,len(prices)-1)
    if e>=len(prices):return 'ENTRY_PENDING',{},pd.DataFrame()
    if not valid_prices(prices.iloc[e:stop+1]).all():return 'PRICE_GAP',{},pd.DataFrame()
    # A missing anchor is also unpriceable under the inherited ledger contract.
    if not valid_prices(prices.iloc[e-1:e]).all():return 'PRICE_GAP',{},pd.DataFrame()
    p=prices.iloc[e-1:stop+1].reset_index(drop=True);cl=clock.iloc[e-1:stop+1].reset_index(drop=True)
    target=pd.Series(np.nan,index=pd.Index(p.as_of_session,name='as_of_session'))
    target.iloc[0]=1.
    if u<len(prices):target.iloc[-2]=0.
    ledger=continuous_ledger(p,cl,target,capital=capital,cost_rate=cost)
    # Shape is fixed-share close wealth before sale cost; distributions stay cash.
    path=ledger.equity_end.iloc[1:].to_numpy().copy()
    if u<len(prices):path[-1]+=ledger.cost.iloc[-1]
    relative=path/path[0]-1
    shape_dd=1-path/np.maximum.accumulate(path)
    net_path=np.r_[capital,ledger.equity_end.iloc[1:].to_numpy()]
    metrics={'net_return':ledger.equity_end.iloc[-1]/capital-1 if u<len(prices) else np.nan,
        'marked_return':ledger.equity_end.iloc[-1]/capital-1,'mae':min(0.,relative.min()),
        'mfe':max(0.,relative.max()),'entry_loss':max(0.,-relative.min()),
        'peak_drawdown':shape_dd.max(),'net_drawdown':(1-net_path/np.maximum.accumulate(net_path)).max(),
        'worst_day':np.min(path[1:]/path[:-1]-1) if len(path)>1 else np.nan,
        'profit_ge_10':bool(ledger.equity_end.iloc[-1]>=capital*1.1) if u<len(prices) else None,
        'loss_ge_10':bool(np.min(path)<=path[0]*.9+1e-8),
        'fees':ledger.cost.sum(),'entry_price':p.close.iloc[1],
        'exit_price':p.close.iloc[-1] if u<len(prices) else np.nan,
        'entry_shares':ledger.shares_end.iloc[1], 'exit_shares_before_sale':ledger.shares_after_split_before_trade.iloc[-1],
        'distribution_cash':ledger.distribution_income.sum()}
    ledger['holding_day']=np.arange(len(ledger))-1
    ledger['shape_return']=np.r_[np.nan,relative]
    ledger['reason']=np.where(ledger.trade_notional.gt(0),'计划买入或时间退出','计划持有份额／退出后现金')
    return ('MATURE' if u<len(prices) else 'OPEN_MARKED'),metrics,ledger


def evaluate(episodes,signals,prices,clock,calendar,cfg,*,horizon,lag=None,cost=None):
    lag=cfg['lag'] if lag is None else lag;cost=cfg['cost'] if cost is None else cost
    trades=[];opps=[];paths=[];n=len(prices)
    date=lambda i:calendar[int(i)] if pd.notna(i) and int(i)<len(calendar) else None
    for r in signals.to_dict('records'):
        s=int(r['s']);t=r['t']; common=s+cfg['window']+lag+horizon
        base={**r,'H':horizon,'lag':lag,'cost_rate':cost,'common_start_i':s+lag,
              'common_start':date(s+lag),'common_end_i':common,'common_end':date(common)}
        status=r['signal_status'];metrics={};entry=np.nan;end=np.nan
        if pd.notna(t):
            entry=int(t)+lag;end=entry+horizon
            ts,metrics,path=fixed_trade(prices,clock,entry,end,cfg['capital'],cost)
            missed=np.nan;shock_to_signal=np.nan;shock_to_entry=np.nan
            # Fixed-share, no-fee market descriptions, not profits of the trade.
            for a,b,key in [(int(t),entry,'signal_to_entry'),(s,int(t),'shock_to_signal'),(s,entry,'shock_to_entry')]:
                if b<n and valid_prices(prices.iloc[a:b+1]).all():
                    q=1.;cash=0.;start=prices.close.iloc[a]
                    for p in prices.iloc[a+1:b+1].itertuples():
                        q/=p.split_factor;cash+=q*(p.cash_dividend+p.capital_gain_distribution)
                    val=(q*prices.close.iloc[b]+cash)/start-1
                    if key=='signal_to_entry':missed=val
                    elif key=='shock_to_signal':shock_to_signal=val
                    else:shock_to_entry=val
            tr={**base,'e':entry,'u':end,'entry_session':date(entry),'exit_session':date(end),
                'decision_at':clock.decision_at.iloc[entry-1] if 0<entry<=len(clock) else None,
                'execution_at':clock.execution_at.iloc[entry-1] if 0<entry<=len(clock) else None,
                'wait_days':int(t)-s,'trade_status':ts,'signal_to_entry_return':missed,
                'shock_to_signal_return':shock_to_signal,'shock_to_entry_return':shock_to_entry,**metrics}
            trades.append(tr)
            if len(path):
                path['rule_signal_session']=r['signal_session']
                path['planned_entry_session']=date(entry);path['planned_exit_session']=date(end)
                paths.append(path.assign(episode_id=r['episode_id'],rule_id=r['rule_id'],H=horizon,lag=lag,cost_rate=cost))
            status='MATURE_TRADE' if ts=='MATURE' else ts
        elif status=='NATURAL_NO_TRIGGER':
            metrics={k:0. for k in ['net_return','entry_loss','peak_drawdown','net_drawdown','mae','mfe','fees']}
            status='CASH_NO_TRIGGER'
        if common>=n:status='COMMON_PENDING'
        if r['signal_status'] in ['DATA_GAP','TRIGGERED_AFTER_GAP'] or r['detection_gap']:status='SIGNAL_DATA_GAP'
        eligible=status in ['MATURE_TRADE','CASH_NO_TRIGGER']
        opps.append({**base,'e':entry,'u':end,'entry_session':date(entry),'exit_session':date(end),
                     'opportunity_status':status,'wait_days':int(t)-s if pd.notna(t) else np.nan,
                     **{k:metrics.get(k,np.nan) if eligible else np.nan for k in ['net_return','entry_loss','peak_drawdown','net_drawdown','mae','mfe','fees']}})
    tcols=['episode_id','rule_id','s','shock_session','t','signal_session','signal_status','H','lag','cost_rate','common_end','e','u','entry_session','exit_session','trade_status','net_return']
    ocols=tcols[:-2]+['window_complete','detection_gap','opportunity_status','net_return']
    return (pd.DataFrame(trades) if trades else pd.DataFrame(columns=tcols),
            pd.DataFrame(opps) if opps else pd.DataFrame(columns=ocols),pd.concat(paths,ignore_index=True) if paths else pd.DataFrame())


def account(signals,prices,clock,cfg,*,rule,horizon,lag=None,cost=None):
    lag=cfg['lag'] if lag is None else lag;cost=cfg['cost'] if cost is None else cost
    bad=np.flatnonzero(~valid_prices(prices));n=int(bad[0]) if len(bad) else len(prices)
    p=prices.iloc[:n].reset_index(drop=True);cl=clock.iloc[:n].reset_index(drop=True)
    target=pd.Series(np.nan,index=pd.Index(p.as_of_session,name='as_of_session'));exit_i=-1;part=[]
    for r in signals.loc[signals.rule_id.eq(rule)&signals.t.notna()].sort_values('t').itertuples():
        e=int(r.t)+lag;u=e+horizon
        if e<=exit_i:status='SKIPPED_POSITION_OVERLAP'
        else:
            exit_i=u
            status='ENTRY_PENDING' if e>=len(prices) else 'PRICE_GAP' if e>=n or u>=n and n<len(prices) else 'OPEN_MARKED' if u>=n else 'COMPLETED'
            if e<n:target.iloc[e-1]=1.
            if u<n:target.iloc[u-1]=0.
        part.append({'episode_id':r.episode_id,'rule_id':rule,'H':horizon,'lag':lag,'cost_rate':cost,
                     't':int(r.t),'e':e,'u':u,'status':status})
    if n==0:return pd.DataFrame(),pd.DataFrame(part)
    led=continuous_ledger(p,cl,target,capital=cfg['capital'],cost_rate=cost)
    led['reason']=np.where(led.trade_notional.gt(0),'计划买入或时间退出','计划持有份额／现金')
    led['signal_missing']=False
    led['unvalued_sessions_after_cutoff']=len(prices)-n
    for r in part:
        r['signal_session']=prices.as_of_session.iloc[r['t']] if r['t']<len(prices) else None
        r['entry_session']=prices.as_of_session.iloc[r['e']] if r['e']<len(prices) else None
        r['exit_session']=prices.as_of_session.iloc[r['u']] if r['u']<len(prices) else None
        if r['status'] in ['COMPLETED','OPEN_MARKED']:
            stop=min(r['u'],n-1);start_equity=led.equity_start.iloc[r['e']]
            r['capital_at_entry']=start_equity
            r['continuous_pnl_dollars']=led.equity_end.iloc[stop]-start_equity
            r['continuous_return']=led.equity_end.iloc[stop]/start_equity-1
        else:
            r['capital_at_entry']=np.nan;r['continuous_pnl_dollars']=np.nan;r['continuous_return']=np.nan
    return led,pd.DataFrame(part,columns=['episode_id','rule_id','H','lag','cost_rate','t','e','u','status',
            'signal_session','entry_session','exit_session','capital_at_entry','continuous_pnl_dollars','continuous_return'])


def periods(cfg):
    return [('ALL',cfg['start'],cfg['end']),('2019-2023',cfg['start'],'2023-12-31'),
        ('2024-2026','2024-01-01',cfg['end'])]+[(str(y),f'{y}-01-01',min(f'{y}-12-31',cfg['end'])) for y in range(2019,2027)]


def distribution(values):
    x=pd.Series(values,dtype=float).dropna()
    if x.empty:return {'n':0,**{k:np.nan for k in ['mean','median','win_rate','mean_win','mean_loss','worst','q10','q25','q75']}}
    return {'n':len(x),'mean':x.mean(),'median':x.median(),'win_rate':x.gt(0).mean(),
       'mean_win':x[x>0].mean(),'mean_loss':x[x<0].mean(),'worst':x.min(),
       'q10':x.quantile(.1),'q25':x.quantile(.25),'q75':x.quantile(.75)}


def summaries(trades,opps,cfg,rules=RULES):
    out=[]
    for per,start,end in periods(cfg):
        for h in sorted(opps.H.unique()) if len(opps) else cfg['horizons']:
            for rule in rules:
                o=opps.loc[opps.H.eq(h)&opps.rule_id.eq(rule)&opps.shock_session.between(start,end)]
                t=trades.loc[trades.H.eq(h)&trades.rule_id.eq(rule)&trades.shock_session.between(start,end)]
                for kind,df in [('TRIGGERED_TRADES',t.loc[t.trade_status.eq('MATURE')&t.exit_session.le(end)]),
                                ('ALL_OPPORTUNITIES',o.loc[o.common_end.le(end)&o.net_return.notna()])]:
                    stats=distribution(df.net_return)
                    out.append({'period':per,'H':h,'rule_id':rule,'sample':kind,'episodes':len(o),
                        'signals':int(o.t.notna().sum()),'mature_trades':int((t.trade_status.eq('MATURE')&t.exit_session.le(end)).sum()),
                        'natural_no_trigger':int(o.signal_status.eq('NATURAL_NO_TRIGGER').sum()),
                        'data_judgeable':int((~o.signal_status.isin(['DATA_GAP','TRIGGERED_AFTER_GAP','WINDOW_PENDING'])&~o.detection_gap.astype(bool)).sum()),
                        'executable':int(t.get('trade_status',pd.Series(dtype=str)).isin(['MATURE','OPEN_MARKED']).sum()),
                        'pending':int((o.common_end.gt(end)|o.opportunity_status.isin(['COMMON_PENDING','ENTRY_PENDING','OPEN_MARKED','WINDOW_PENDING'])).sum()),
                        'gaps':int(o.opportunity_status.str.contains('GAP').sum()),'trigger_rate':o.t.notna().mean(),
                        'mean_wait':(o.t-o.s).mean(),**stats,
                        'mean_entry_loss':df.get('entry_loss',pd.Series(dtype=float)).mean(),
                        'worst_entry_loss':df.get('entry_loss',pd.Series(dtype=float)).max(),
                        'mean_peak_drawdown':df.get('peak_drawdown',pd.Series(dtype=float)).mean(),
                        'mean_signal_to_entry':df.get('signal_to_entry_return',pd.Series(dtype=float)).mean(),
                        'profit_ge_10':int(df.net_return.ge(.1).sum()),
                        'loss_ge_10':int(df.get('entry_loss',pd.Series(dtype=float)).ge(.1-1e-12).sum())})
    return pd.DataFrame(out)


def paired_results(opps,cfg):
    rows=[];details=[]
    for per,start,end in periods(cfg):
        for h in sorted(opps.H.unique()):
            p=opps.loc[opps.H.eq(h)&opps.shock_session.between(start,end)&opps.common_end.le(end)]
            for a,b in PAIRS:
                left=p.loc[p.rule_id.eq(a)].set_index('episode_id');right=p.loc[p.rule_id.eq(b)].set_index('episode_id')
                both=left.index.intersection(right.index)
                l=left.loc[both];r=right.loc[both]
                valid=l.net_return.notna()&r.net_return.notna()
                for sample,mask in [('ALL_OPPORTUNITIES',valid),('BOTH_TRIGGERED',valid&l.t.notna()&r.t.notna())]:
                    delta=l.loc[mask,'net_return']-r.loc[mask,'net_return']
                    rows.append({'period':per,'H':h,'pair':a+'-'+b,'sample':sample,**distribution(delta),
                        'mean_loss_delta':(l.loc[mask,'entry_loss']-r.loc[mask,'entry_loss']).mean(),
                        'mean_drawdown_delta':(l.loc[mask,'peak_drawdown']-r.loc[mask,'peak_drawdown']).mean(),
                        'excluded':len(both)-int(mask.sum()),'excluded_episode_ids':';'.join(both[~mask])})
                if per=='ALL':
                    for eid in both[valid]:
                        details.append({'episode_id':eid,'H':h,'pair':a+'-'+b,'net_return':l.loc[eid,'net_return']-r.loc[eid,'net_return'],
                           'entry_loss_delta':l.loc[eid,'entry_loss']-r.loc[eid,'entry_loss'],
                           'drawdown_delta':l.loc[eid,'peak_drawdown']-r.loc[eid,'peak_drawdown']})
    return pd.DataFrame(rows),pd.DataFrame(details)


def account_summary(led,name,h,cfg):
    rows=[]
    for per,start,end in periods(cfg):
        x=led.loc[led.as_of_session.between(start,end)]
        if x.empty:continue
        initial=x.equity_start.iloc[0];wealth=np.r_[initial,x.equity_end.to_numpy()]
        net=wealth[-1]/initial-1
        days=(pd.Timestamp(x.as_of_session.iloc[-1])-pd.Timestamp(x.as_of_session.iloc[0])).days+1
        rows.append({'period':per,'account':name,'H':h,'sessions':len(x),'total_return':net,
             'CAGR':(1+net)**(365.25/days)-1,'max_drawdown':(1-wealth/np.maximum.accumulate(wealth)).max(),
             'worst_day':x.portfolio_return.min(),'mean_exposure':x.old_weight.mean(),'in_market_days':int(x.old_shares.gt(0).sum()),
             'turnover':x.turnover.sum(),'fees':x.cost.sum(),'equity_start':initial,'equity_end':wealth[-1],
             'pnl_dollars':wealth[-1]-initial,'terminal_shares':x.shares_end.iloc[-1]})
    return rows


def cluster_uncertainty(episodes,clusters,trades,opps,pair_details,cfg):
    """Resample entire inference clusters, retaining episode multiplicity."""
    source=[];h=cfg['main_horizon']
    for rule in RULES:
        for sample,frame in [('TRADES',trades),('OPPORTUNITIES',opps)]:
            x=frame.loc[frame.H.eq(h)&frame.rule_id.eq(rule)&frame.net_return.notna(),['episode_id','net_return']]
            source.append((rule+'_'+sample,x))
    for a,b in PAIRS:
        pair=a+'-'+b
        source.append((pair,pair_details.loc[pair_details.H.eq(h)&pair_details.pair.eq(pair),['episode_id','net_return']]))
    keys=sorted(clusters.cluster_id.unique());rng=np.random.default_rng(cfg['seed'])
    draws=rng.integers(0,len(keys),(cfg['bootstrap_repetitions'],len(keys))) if keys else np.empty((0,0),int)
    boot=[];intervals=[];leave=[]
    for label,frame in source:
        x=frame.merge(clusters[['episode_id','cluster_id']],on='episode_id')
        groups=[x.loc[x.cluster_id.eq(k),'net_return'].to_numpy(float) for k in keys]
        contributing=sum(len(g)>0 for g in groups);values=[]
        if contributing>=2:
            for rep,draw in enumerate(draws):
                picked=np.concatenate([groups[i] for i in draw]);val=picked.mean() if len(picked) else np.nan
                values.append(val);boot.append({'metric':label,'replicate':rep,'n':len(picked),'mean':val})
        tail=(1-cfg['interval'])/2
        intervals.append({'metric':label,'n':len(x),'clusters_all':len(keys),'clusters_contributing':contributing,
            'mean':x.net_return.mean(),'lower':np.nanquantile(values,tail) if values else np.nan,
            'upper':np.nanquantile(values,1-tail) if values else np.nan,
            'valid_replicates':int(np.isfinite(values).sum()),'status':'DESCRIPTIVE' if contributing>=2 else 'INSUFFICIENT_CLUSTERS'})
        for key in keys:
            remaining=x.loc[x.cluster_id.ne(key),'net_return'];mean=remaining.mean();original=x.net_return.mean()
            leave.append({'metric':label,'removed_cluster':key,'removed_episodes':';'.join(clusters.loc[clusters.cluster_id.eq(key),'episode_id']),
                          'n_remaining':len(remaining),'original_mean':original,'remaining_mean':mean,
                          'direction_changed':bool(np.isfinite(mean) and original*mean<0)})
    return pd.DataFrame(intervals),pd.DataFrame(boot),pd.DataFrame(leave)


def load_inputs(root,cfg,*,as_of=None,signal_only=False):
    end=min(as_of or cfg['end'],cfg['end']);hashes={};sources=[]
    p1=json.loads((root/'runs/p1/reproducibility.json').read_text())['clean_sha256']
    def read(rel,expected=None,bounded=False):
        path=root/rel;actual=digest(path)
        if expected is not None and actual!=expected:raise ValueError('Input digest mismatch: '+rel)
        hashes[rel]=actual
        return read_bounded(path,end) if bounded else pd.read_csv(path)
    def clean(name,bounded=False):
        rel='data/clean/'+name
        return read(rel,p1.get(rel),bounded)
    schedule=clean('equity_sessions.csv',signal_only)
    schedule=schedule.loc[schedule.as_of_session.between(cfg['start'],end)].reset_index(drop=True)
    dates=schedule.as_of_session.tolist()
    if len(set(dates))!=len(dates) or dates!=sorted(dates):raise ValueError('Invalid calendar')
    cal=xcals.get_calendar('XNYS',start=cfg['start'],end=str((pd.Timestamp(cfg['end'])+pd.Timedelta(days=130)).date()))
    future_dates=cal.sessions.strftime('%Y-%m-%d').tolist()
    if dates!=[d for d in future_dates if cfg['start']<=d<=end]:raise ValueError('Frozen calendar differs from XNYS')
    local_close=pd.to_datetime(schedule.close,utc=True)
    actual_close=cal.schedule.loc[pd.to_datetime(dates),'close'].reset_index(drop=True)
    if not np.array_equal(local_close,actual_close):raise ValueError('Frozen close clock differs')
    vix=clean('VIX_daily.csv',signal_only);curve=clean('VX_front_three.csv',signal_only)
    vx_path=root/'data/clean/VX_contract_daily.csv'
    if vx_path.exists():
        vx=clean('VX_contract_daily.csv',signal_only)
        vx=vx.loc[vx.as_of_session.between(cfg['start'],end)].copy()
        vx['value']=pd.to_numeric(vx.value)
    else:
        vx=pd.DataFrame(columns=['contract_id','as_of_session','value','duration_type'])
    vix=vix.loc[vix.as_of_session.between(cfg['start'],end)]
    market=vx_evidence(dates,curve,vx)
    if vix.as_of_session.duplicated().any():raise ValueError('Duplicate VIX dates')
    # Assign numerically by position, since the source series has a date index.
    market['vix']=pd.to_numeric(vix.set_index('as_of_session').reindex(dates).value).to_numpy()
    for frame in [vix,vx]:
        if 'raw_file' in frame:
            sources.extend(frame[['raw_file','raw_sha256','source','retrieved_at']].drop_duplicates().to_dict('records'))
    prices=clock=bil=n1=None
    if not signal_only:
        a2=json.loads((root/'runs/p2/acceptance.json').read_text())
        if digest(root/a2['accepted_run'])!=a2['accepted_run_sha256']:raise ValueError('P2 run identity')
        p2=json.loads((root/a2['accepted_run']).read_text())['result']['output_dir']
        rel=p2+'/SVXY_returns.csv';prices=read(rel,a2['accepted_artifact_sha256'].get(rel))
        rel=p2+'/clock.csv';clock=read(rel,a2['accepted_artifact_sha256'].get(rel))
        prices=prices.loc[prices.as_of_session.between(cfg['start'],end)].reset_index(drop=True)
        clock=clock.loc[clock.as_of_session.between(cfg['start'],end)].reset_index(drop=True)
        if prices.as_of_session.tolist()!=dates or clock.as_of_session.tolist()!=dates:raise ValueError('P2 alignment')
        raw_prices=clean('SVXY_daily.csv')
        sources.extend(raw_prices[['raw_file','raw_sha256','source','retrieved_at']].drop_duplicates().to_dict('records'))
        if valid_prices(prices).all():
            reconstructed=total_return(prices)
            if not np.allclose(reconstructed.total_return_index,prices.total_return_index,rtol=0,atol=1e-9):raise ValueError('P2 TR arithmetic')
        try:
            an=json.loads((root/'runs/n1/acceptance.json').read_text());n1dir=an['accepted_output_dir']
            rel=n1dir+'/inputs/BIL_returns.csv';bil=read(rel,an['accepted_artifact_sha256'].get(rel))
            rel=n1dir+'/holding_intervals.csv';n1=read(rel,an['accepted_artifact_sha256'].get(rel))
            if bil.as_of_session.tolist()!=dates:raise ValueError('BIL calendar differs')
            manifest=root/'data/raw/n1/data/raw/downloads.jsonl'
            records=[json.loads(line) for line in manifest.read_text().splitlines()]
            chosen=[r for r in records if r.get('label')=='BIL_history_full' and r.get('ok')]
            if not chosen:raise ValueError('BIL source manifest unavailable')
            record=chosen[-1]
            sources.append({'raw_file':str((root/'data/raw/n1'/record['raw_file']).relative_to(root)),
                'raw_sha256':record['sha256'],'source':record['source'],'retrieved_at':record['retrieved_at'],'source_url':record['url']})
        except FileNotFoundError:
            bil=None;n1=None
    manifest_records=[json.loads(line) for line in (root/'data/raw/downloads.jsonl').read_text().splitlines()]
    urls={r.get('raw_file'):r.get('url') for r in manifest_records}
    unique={r['raw_file']:{**r,'source_url':r.get('source_url',urls.get(r['raw_file']))} for r in sources}
    for rel,record in unique.items():
        if digest(root/rel)!=record['raw_sha256']:raise ValueError('Raw source identity: '+rel)
        hashes[rel]=record['raw_sha256']
    coverage={'first':dates[0] if dates else None,'last':dates[-1] if dates else None,'stock_sessions':len(dates),
        'vix_present':int(market.vix.notna().sum()),'vx_current_previous_complete':int(market.vx_complete.sum()),
        'vx_monthly_rows':len(vx),'svxy_rows':len(prices) if prices is not None else None,
        'svxy_price_action_complete':int(valid_prices(prices).sum()) if prices is not None else None,
        'bil_rows':len(bil) if bil is not None else None,'raw_sources_verified':len(unique),
        'downloaded_new_market_data':False,'signal_only':signal_only,'historical_availability':'ASSUMED_NEXT_SESSION; not full PIT'}
    return market,prices,clock,future_dates,curve,bil,n1,{'sha256':hashes,'sources':list(unique.values()),'coverage':coverage}


def audit_trades(trades,prices,cfg):
    """Independent scalar shares/cash, no rebalance/continuous_ledger calls."""
    rows=[]
    for r in trades.loc[trades.trade_status.eq('MATURE')].itertuples():
        cash=float(cfg['capital']);cost=r.cost_rate;e=int(r.e);u=int(r.u)
        shares=cash/(prices.close.iloc[e]*(1+cost));buy_fee=shares*prices.close.iloc[e]*cost
        cash-=shares*prices.close.iloc[e]+buy_fee;income=0.
        for p in prices.iloc[e+1:u+1].itertuples():
            shares/=p.split_factor;dist=shares*(p.cash_dividend+p.capital_gain_distribution);cash+=dist;income+=dist
        sale=shares*prices.close.iloc[u];sell_fee=sale*cost;cash+=sale-sell_fee
        error=abs(cash/cfg['capital']-1-r.net_return);fee_error=abs(buy_fee+sell_fee-r.fees)
        if error>1e-11 or fee_error>1e-7:raise AssertionError('Independent trade arithmetic')
        rows.append({'episode_id':r.episode_id,'rule_id':r.rule_id,'H':r.H,'lag':r.lag,'cost_rate':cost,
            'entry_session':r.entry_session,'exit_session':r.exit_session,'net_return':r.net_return,
            'independent_cash_end':cash,'independent_fees':buy_fee+sell_fee,'cash_distributions':income,
            'return_error':error,'fee_error_dollars':fee_error})
    return pd.DataFrame(rows)


def audit_ledger(led):
    previous_equity=led.equity_start.iloc[0];previous_shares=0.;previous_cash=previous_equity;errors=[]
    for r in led.itertuples():
        q=previous_shares/r.split_factor
        income=r.distribution_income
        expected_cash=previous_cash+income-r.signed_trade-r.cost
        expected_shares=q+r.signed_trade/r.close
        error=max(abs(expected_cash-r.cash_end),abs(expected_shares-r.shares_end)*r.close,
                  abs(r.cash_end+r.shares_end*r.close-r.equity_end),abs(previous_equity-r.equity_start))
        if error>1e-6 or r.cash_end< -1e-7 or r.new_weight>1+1e-12:raise AssertionError('Ledger conservation')
        errors.append(error);previous_equity=r.equity_end;previous_cash=r.cash_end;previous_shares=r.shares_end
    return {'rows':len(led),'max_error_dollars':max(errors)}


def n1_reconciliation(n1,prices,trades):
    if n1 is None:return {'status':'N1_INPUT_UNAVAILABLE'}
    x=n1.loc[n1.period.eq('revealed_2024')&n1['first'].eq('2024-08-06')&n1['last'].eq('2024-08-12')]
    if len(x)!=1:return {'status':'N1_INTERVAL_NOT_UNIQUE'}
    p=prices.set_index('as_of_session');old=x.svxy_path_return.iloc[0]
    inclusive=p.loc['2024-08-12','total_return_index']/p.loc['2024-08-05','total_return_index']-1
    close_buy=p.loc['2024-08-12','total_return_index']/p.loc['2024-08-06','total_return_index']-1
    if not np.isclose(old,inclusive,rtol=0,atol=1e-12):raise AssertionError('N1 interval definition')
    evidence=[]
    for r in trades.loc[trades.H.eq(10)&trades.shock_session.between('2024-07-01','2024-08-31')].itertuples():
        remaining=(p.loc['2024-08-12','total_return_index']/p.loc[r.entry_session,'total_return_index']-1) if r.entry_session<='2024-08-12' else np.nan
        evidence.append({'rule_id':r.rule_id,'shock_session':r.shock_session,'signal_session':r.signal_session,
            'entry_session':r.entry_session,'exit_session':r.exit_session,'signal_to_entry_return':r.signal_to_entry_return,
            'entry_to_aug12_gross':remaining,'actual_H10_net':r.net_return})
    return {'status':'RECONCILED','original_interval_return':old,'aug05_close_to_aug12_close':inclusive,
        'aug06_close_to_aug12_close':close_buy,'includes_aug06_return_from_aug05':True,'rules':evidence,
        'note':'08-12只是旧归因窗口对照终点；E1实际退出仍为入场后H日，不能用此日重选退出。'}


def real_causality(market,prices,clock,cfg):
    ep,s,d=scan_signals(market,cfg);checks=[]
    for date in ['2020-03-06','2020-03-20','2024-08-07','2025-04-09','2026-09-04']:
        idx=market.index[market.as_of_session.le(date)][-1]
        _,short,trace=scan_signals(market.iloc[:idx+1],cfg)
        cols=['episode_id','rule_id','t','signal_session','signal_status']
        pd.testing.assert_frame_equal(s.loc[s.t.le(idx),cols].reset_index(drop=True),short.loc[short.t.notna(),cols].reset_index(drop=True),check_dtype=False)
        pd.testing.assert_frame_equal(d.loc[d.i.le(idx)].reset_index(drop=True),trace,check_dtype=False)
        altered=market.copy();altered.loc[idx+1:,'vix']=99.;altered.loc[idx+1:,'vx_down']=False
        # Derived current-ID evidence is perturbed only in the synthetic future copy.
        for col in ['f1_current','f1_previous','f2_current','f2_previous']:altered.loc[idx+1:,col]=999.
        _,ss,_=scan_signals(altered,cfg)
        pp=prices.copy();pp.loc[idx+1:,'close']*=.1
        for rule in RULES:
            led,_=account(s,prices,clock,cfg,rule=rule,horizon=20)
            other,_=account(ss,pp,clock,cfg,rule=rule,horizon=20)
            pd.testing.assert_frame_equal(led.iloc[:idx+1],other.iloc[:idx+1])
        checks.append({'cutoff':date,'signal_and_daily_prefix_equal':True,'five_account_prefixes_equal':True})
    return {'fixture':'SYNTHETIC_FUTURE_PERTURBATION_OF_REAL_INPUT_IN_MEMORY_ONLY','checks':checks}


def run(root,cfg,run_id,run_dir,output,*,as_of=None,signal_only=False):
    market,prices,clock,calendar,curve,bil,n1,inputs=load_inputs(root,cfg,as_of=as_of,signal_only=signal_only)
    write_json(run_dir/'inputs.json',inputs)
    ep,signals,daily=scan_signals(market,cfg)
    save=lambda frame,name:frame.to_csv(output/(name+'.csv'),index=False,float_format='%.15g')
    save(ep,'episodes');save(signals,'signals');save(daily,'episode_daily');save(market,'signal_inputs')
    if signal_only:
        write_json(output/'snapshot.json',{'as_of':as_of,'last_information':market.as_of_session.iloc[-1],
            'historical_snapshot_only':True,'asset_prices_read':False,'outcomes_read':False,'episodes':len(ep),
            'signals':int(signals.t.notna().sum()),'current_buy_permission':False})
        return {'status':'SIGNAL_SNAPSHOT_COMPLETE','output_dir':str(output.relative_to(root))}
    print('E1 signals saved before outcome evaluation',len(ep),'episodes',flush=True)
    clusters=inference_clusters(ep,cfg);save(clusters,'inference_clusters')
    all_t=[];all_o=[];all_paths=[];ledgers=[];participation=[];acct=[];audits=[]
    def register(led,name,h,scenario='BASE'):
        check=audit_ledger(led);audits.append({'account':name,'H':h,'scenario':scenario,**check})
        ledgers.append(led.assign(account=name,H=h,scenario=scenario))
        acct.extend([{**r,'scenario':scenario} for r in account_summary(led,name,h,cfg)])
    for h in cfg['horizons']:
        t,o,path=evaluate(ep,signals,prices,clock,calendar,cfg,horizon=h)
        all_t.append(t);all_o.append(o);all_paths.append(path)
        for rule in RULES:
            led,part=account(signals,prices,clock,cfg,rule=rule,horizon=h)
            register(led,rule,h);participation.append(part.assign(scenario='BASE'))
    trades=pd.concat(all_t,ignore_index=True);opps=pd.concat(all_o,ignore_index=True)
    save(trades,'trades');save(opps,'opportunity_results');save(pd.concat(all_paths,ignore_index=True),'trade_paths')
    metrics=summaries(trades,opps,cfg);paired,details=paired_results(opps,cfg)
    save(metrics,'event_summary');save(paired,'paired_summary');save(details,'paired_events')
    # Same-day calendar anchor, baseline targets only execute on the next close.
    targets=baseline_targets(curve,[.5]);idx=pd.Index(prices.as_of_session,name='as_of_session')
    hold=pd.Series(np.nan,index=idx);hold.iloc[0]=1.
    targets['SVXY_BUY_HOLD']=hold
    for name,targ in targets.items():
        led=continuous_ledger(prices,clock,targ.reindex(idx),capital=cfg['capital'],cost_rate=cfg['cost'])
        register(led,name,0)
    if bil is not None:
        # Reinvestment reference: normalized actual N1 TR as price, no invented quote.
        proxy=bil.copy();proxy['close']=bil.total_return_index
        proxy['split_factor']=1.;proxy['cash_dividend']=0.;proxy['capital_gain_distribution']=0.
        led=continuous_ledger(proxy,clock,hold,capital=cfg['capital'],cost_rate=cfg['cost'])
        register(led,'BIL_TR_REFERENCE',0)
    interval,boot,leave=cluster_uncertainty(ep,clusters,trades,opps,details,cfg)
    save(interval,'cluster_intervals');save(boot,'bootstrap_scalars');save(leave,'leave_one_cluster_out')
    sensitivity_t=[];sensitivity_o=[];sensitivity_summary=[];sensitivity_pairs=[]
    scenarios=[(f'COST_{round(c*10000)}BP',cfg['lag'],c) for c in cfg['cost_sensitivities']]+[('LAG_2',cfg['delay_sensitivity'],cfg['cost'])]
    for name,lag,cost in scenarios:
        t,o,_=evaluate(ep,signals,prices,clock,calendar,cfg,horizon=10,lag=lag,cost=cost)
        sensitivity_t.append(t.assign(scenario=name));sensitivity_o.append(o.assign(scenario=name))
        sensitivity_summary.append(summaries(t,o,cfg).assign(scenario=name))
        pp,_=paired_results(o,cfg);sensitivity_pairs.append(pp.assign(scenario=name))
        for rule in RULES:
            led,part=account(signals,prices,clock,cfg,rule=rule,horizon=10,lag=lag,cost=cost)
            register(led,rule,10,name);participation.append(part.assign(scenario=name))
    save(pd.concat(sensitivity_t,ignore_index=True),'execution_trades');save(pd.concat(sensitivity_o,ignore_index=True),'execution_opportunities')
    save(pd.concat(sensitivity_summary,ignore_index=True),'execution_summary');save(pd.concat(sensitivity_pairs,ignore_index=True),'execution_pairs')
    print('E1 base, baselines, cluster intervals and execution sensitivities computed',flush=True)
    neighbor_ep=[];neighbor_s=[];neighbor_t=[];neighbor_o=[];neighbor_summary=[]
    for key,values in cfg['neighborhood'].items():
        for value in values:
            variant=f'{key}_{value:g}';v={**cfg,key:value}
            ee,ss,dd=scan_signals(market,v);ss=ss.loc[ss.rule_id.eq('B_RETREAT')]
            tt,oo,_=evaluate(ee,ss,prices,clock,calendar,v,horizon=10)
            neighbor_ep.append(ee.assign(variant=variant));neighbor_s.append(ss.assign(variant=variant))
            neighbor_t.append(tt.assign(variant=variant));neighbor_o.append(oo.assign(variant=variant))
            neighbor_summary.append(summaries(tt,oo,v,rules=['B_RETREAT']).assign(variant=variant))
            led,part=account(ss,prices,clock,v,rule='B_RETREAT',horizon=10)
            register(led,'B_RETREAT',10,variant);participation.append(part.assign(scenario=variant))
    for name,frames in [('neighborhood_episodes',neighbor_ep),('neighborhood_signals',neighbor_s),('neighborhood_trades',neighbor_t),
                        ('neighborhood_opportunities',neighbor_o),('neighborhood_summary',neighbor_summary)]:save(pd.concat(frames,ignore_index=True),name)
    ledger_table=pd.concat(ledgers,ignore_index=True);part_table=pd.concat(participation,ignore_index=True)
    save(ledger_table,'ledgers');save(part_table,'participation');save(pd.DataFrame(acct),'account_summary')
    evidence=[]
    for episode in ep.itertuples():
        s=int(episode.s);end=min(len(prices)-1,s+cfg['window']+cfg['lag']+max(cfg['horizons']))
        path=prices.iloc[s:end+1][['as_of_session','close','total_return_index']].copy()
        path['episode_id']=episode.episode_id;path['i']=np.arange(s,end+1)
        path['shock_anchor_return']=path.total_return_index/path.total_return_index.iloc[0]-1
        path=path.merge(market[['as_of_session','vix']],on='as_of_session',how='left')
        evidence.append(path)
    save(pd.concat(evidence,ignore_index=True),'case_paths')
    n1check=n1_reconciliation(n1,prices,trades);write_json(output/'n1_reconciliation.json',n1check)
    audited_trades=pd.concat([trades,*sensitivity_t,*neighbor_t],ignore_index=True)
    trade_audit=audit_trades(audited_trades,prices,cfg);save(trade_audit,'independent_trade_audit')
    assert len(opps)==len(ep)*len(RULES)*len(cfg['horizons'])
    assert not opps.duplicated(['episode_id','rule_id','H']).any()
    assert opps.loc[opps.opportunity_status.eq('CASH_NO_TRIGGER'),'net_return'].eq(0).all()
    causal=real_causality(market,prices,clock,cfg);write_json(run_dir/'causality.json',causal)
    audit={'ledgers':audits,'ledger_accounts':len(audits),'ledger_rows':sum(a['rows'] for a in audits),
        'independent_mature_trades':len(trade_audit),'max_trade_return_error':trade_audit.return_error.max(),
        'max_trade_fee_error_dollars':trade_audit.fee_error_dollars.max(),'episode_opportunity_count_reconciled':True,
        'positive_real_examples':trade_audit.loc[trade_audit.net_return.gt(0)].head(1).to_dict('records'),
        'negative_real_examples':trade_audit.loc[trade_audit.net_return.lt(0)].head(1).to_dict('records'),
        'natural_no_trigger_examples':opps.loc[opps.opportunity_status.eq('CASH_NO_TRIGGER')].head(1).to_dict('records'),
        'natural_no_trigger_note':'主配置实际0例；五规则均在7轮内触发。十个B邻域也0例。未制造真实案例；零现金分母由合成反例验证。'}
    write_json(run_dir/'independent_validation.json',audit)
    summary={'engineering':{'computation_complete':True,'tests':'pending final pytest','visual_review':'pending'},
        'real_data_coverage':inputs['coverage'],'history_attribute':cfg['history_attribute'],
        'main_hypothesis':'B_RETREAT','episodes':len(ep),'inference_clusters':clusters.cluster_id.nunique(),
        'main_results':metrics.loc[metrics.period.eq('ALL')&metrics.H.eq(10)].to_dict('records'),
        'paired_results':paired.loc[paired.period.eq('ALL')&paired.H.eq(10)&paired['sample'].eq('ALL_OPPORTUNITIES')].to_dict('records'),
        'intervals':interval.to_dict('records'),'audit':{k:v for k,v in audit.items() if k!='ledgers'},
        'research_conclusion':'由本次实际结果生成；主B不因辅助或邻域冠军而替换。',
        'auxiliary_rules':RULES,'open_positions_at_cutoff':part_table.loc[part_table.status.eq('OPEN_MARKED')].to_dict('records'),
        'pending_entries_at_cutoff':part_table.loc[part_table.status.eq('ENTRY_PENDING')].to_dict('records'),
        'acceptance':'WAITING_USER_ACCEPTANCE','data_cutoff':cfg['end']}
    write_json(output/'summary.json',summary)
    card={'research_object':'实际SVXY份额；B_RETREAT固定定义','current_buy_opportunity':None,
        'current_buy_permission':False,'live_data_available':False,'as_of_information':cfg['end'],
        'rule':{k:cfg[k] for k in ['level','rise','lookback','retreat','window','rearm_level','rearm_days','lag','main_horizon','cost']},
        'identification':'股票日历因果冲击；运行最高收盘；s+1至s+W首次退潮；活动期不重复开事件',
        'entry':'信号后下一真实股票交易日收盘；截止为该收盘前60分钟，假设数据此前可得',
        'exit':'入场收盘后第10个股票交易日；时间退出无附加滞后',
        'sizing':'100000美元规范化单一账户现金买入；持有固定份额，分配留现金；重叠信号跳过',
        'evidence':['signals.csv','trades.csv','opportunity_results.csv','paired_summary.csv','cluster_intervals.csv','ledgers.csv'],
        'research_evaluation':summary['main_results'],'limitations':['已揭示历史、稀少且可能相关的冲击','非完整历史PIT','收盘及5bp是代理；日内最坏损失未知','未验证期权价格或实盘收益'],
        'integration':'本轮不接入监控，不晋升辅助规则，不进入下一实验'}
    write_json(output/'candidate_card.json',card)
    from svxylab.panic_retreat_report import build_report
    report=build_report(root,output,run_dir,cfg,summary,n1check)
    (root/'reports/panic_retreat.html').write_text(report)
    (output/'report.html').write_text(report)
    print('E1 complete:',output,flush=True)
    return {'status':'COMPUTATION_COMPLETE','output_dir':str(output.relative_to(root)),
            'report':'reports/panic_retreat.html','episodes':len(ep),'clusters':clusters.cluster_id.nunique(),
            'accounts':len(audits),'independent_trades':len(trade_audit)}


def main():
    parser=argparse.ArgumentParser(description='E1 恐慌退潮：冻结历史事件研究')
    parser.add_argument('--open',action='store_true');parser.add_argument('--as-of')
    parser.add_argument('--signal-only',action='store_true')
    args=parser.parse_args()
    if args.signal_only and not args.as_of:parser.error('--signal-only requires --as-of YYYY-MM-DD')
    if args.as_of and not args.signal_only:parser.error('--as-of is supported with --signal-only')
    cfg=tomllib.loads((ROOT/'panic_retreat.toml').read_text())
    if args.as_of:
        try: datetime.strptime(args.as_of,'%Y-%m-%d')
        except ValueError:parser.error('Invalid as-of date')
        if not cfg['start']<=args.as_of<=cfg['end']:parser.error('Historical snapshot date outside frozen coverage')
    run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output=ROOT/'data/clean/panic_retreat'/run_id;run_dir=ROOT/'runs/panic_retreat'/run_id
    output.mkdir(parents=True);run_dir.mkdir(parents=True)
    record={'task_id':cfg['task_id'],'started_at_utc':datetime.now(timezone.utc).isoformat(),
        'command':[sys.executable,'-m','svxylab.panic_retreat',*sys.argv[1:]],'config':cfg,
        'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'registered_before_outcome_calculation':True,'history_already_revealed':True,
        'source_sha256':{str(p.relative_to(ROOT)):digest(p) for p in [ROOT/'panic_retreat.toml',Path(__file__),
           ROOT/'src/svxylab/panic_retreat_report.py',ROOT/'src/svxylab/ledger.py',ROOT/'src/svxylab/returns.py',ROOT/'src/svxylab/timing.py'] if p.exists()},
        'scope':'E1 only; no M2 fits or old release verification; stop for acceptance; no commit or push'}
    write_json(run_dir/'experiment_record.json',record)
    try:
        result=run(ROOT,cfg,run_id,run_dir,output,as_of=args.as_of,signal_only=args.signal_only)
        result['exit_code']=0
        if args.open and not args.signal_only:
            opened=subprocess.run(['open',str(ROOT/'reports/panic_retreat.html')],check=False)
            result['open_exit_code']=opened.returncode
    except Exception as exc:
        write_json(run_dir/'run.json',{'status':'FAILED','error_type':type(exc).__name__,'error':str(exc),'exit_code':1})
        raise
    result['finished_at_utc']=datetime.now(timezone.utc).isoformat()
    write_json(run_dir/'run.json',result)
    if not args.signal_only:write_json(ROOT/'runs/panic_retreat/latest.json',{'run_id':run_id,**result})
    print(json.dumps(jsonable(result),ensure_ascii=False))


if __name__=='__main__':main()
