"""Read-only real-input diagnostic checks; synthetic future changes stay in memory.

Run from the workspace: .venv/bin/python runs/panic_retreat_diagnostic/verify_delivery.py
"""
from html.parser import HTMLParser
import json
from pathlib import Path
from urllib.parse import unquote, urlsplit

import exchange_calendars as xcals
import numpy as np
import pandas as pd

from svxylab.panic_retreat_diagnostic import (
    ROOT, load_inputs, causal_timeline, availability, holding_states, candidates,
    independent_audit, original_reconciliation, write_json, digest,
)


def main():
    cfg,m,p,c,old,identity=load_inputs()
    latest=json.loads((ROOT/'runs/panic_retreat_diagnostic/latest.json').read_text())
    out=ROOT/latest['output_dir']
    ep,s,t=causal_timeline(m,cfg)
    full_info=availability(s,m,c,cfg)
    full_holding=holding_states(s,ep,m,c,cfg)
    # Every observed lifecycle or rearm day is a truncation checkpoint.
    stops=sorted(set(t.i.astype(int)+1))
    for stop in stops:
        _,_,prefix=causal_timeline(m.iloc[:stop],cfg)
        pd.testing.assert_frame_equal(t[t.i<stop].reset_index(drop=True),prefix,check_dtype=False)
    # Distinct observed signals, window ends, lifecycle ends and rearm dates.
    adaptive=sorted(set(s.t.astype(int)+1)|set(ep.window_end_i.astype(int)+1)|
                    set(ep.end_i.astype(int)+1)|set(ep.end_i.astype(int)+2))
    for stop in adaptive:
        changed=m.copy()
        changed.loc[stop:,'vix']=999.  # SYNTHETIC TEST ONLY: not persisted as data.
        changed.loc[stop:,'vx_complete']=False
        changed.loc[stop:,'vx_down']=False
        changed.loc[stop:,'f1_current']=999.
        changed.loc[stop:,'f2_current']=998.
        e2,s2,t2=causal_timeline(changed,cfg)
        pd.testing.assert_frame_equal(t[t.i<stop].reset_index(drop=True),t2[t2.i<stop].reset_index(drop=True),check_dtype=False)
        info2=availability(s2,changed,c,cfg)
        pd.testing.assert_frame_equal(full_info[full_info.observation_i<stop].reset_index(drop=True),
                                      info2[info2.observation_i<stop].reset_index(drop=True),check_dtype=False)
        hs2=holding_states(s2,e2,changed,c,cfg)
        pd.testing.assert_frame_equal(full_holding[full_holding.information_i<stop].reset_index(drop=True),
                                      hs2[hs2.information_i<stop].reset_index(drop=True),check_dtype=False)
        e3,s3,_=causal_timeline(m.iloc[:stop],cfg)
        ip=availability(s3,m.iloc[:stop],c,cfg)
        pd.testing.assert_frame_equal(full_info[full_info.observation_i<stop].reset_index(drop=True),ip,check_dtype=False)
    print(f'真实输入因果检查完成：{len(stops)}个截断、{len(adaptive)}个未来扰动／信息截面',flush=True)
    cells=pd.read_csv(out/'four_cells.csv',dtype={'cell':str})
    later=pd.read_csv(out/'candidates.csv')
    cell_ledger=pd.read_csv(out/'cell_ledgers.csv')
    later_ledger=pd.read_csv(out/'candidate_ledgers.csv')
    audit=independent_audit(pd.concat([cells,later],ignore_index=True),
        pd.concat([cell_ledger,later_ledger],ignore_index=True),p,cfg['capital'])
    original=original_reconciliation(cells,old)
    calendar=xcals.get_calendar('XNYS',start='2019-01-01',end='2026-09-04')
    schedule=calendar.schedule
    for r in pd.concat([cells,later],ignore_index=True).itertuples():
        assert r.u-r.e==r.actual_holding_sessions
        assert r.entry_session==p.as_of_session.iloc[r.e] and r.exit_session==p.as_of_session.iloc[r.u]
        close=pd.Timestamp(schedule.loc[r.entry_session,'close'])
        assert pd.Timestamp(r.execution_at)==close
        assert pd.Timestamp(r.decision_at)==close-pd.Timedelta(minutes=60)
    a=pd.read_csv(out/'attribution.csv')
    np.testing.assert_allclose(a.entry_contribution+a.exit_contribution,a.total_change,atol=1e-14)
    np.testing.assert_allclose(a.R11-a.R10-a.R01+a.R00,a.interaction,atol=1e-14)
    # Every common path starts in equal cash and ends on the same comparison date.
    common=pd.read_csv(out/'common_paths.csv')
    for r in cells.itertuples():
        path=common[common.trade_id.eq(r.trade_id)]
        assert path.i.iloc[0]==r.common_start_i and path.i.iloc[-1]==r.common_end_i
        assert path.equity_end.iloc[0]==cfg['capital']
        assert path.shares_end.iloc[-1]==0
        assert abs(path.equity_end.iloc[-1]/cfg['capital']-1-r.net_return)<1e-12
        cash=path[path.i>r.u]
        assert cash.shares_end.eq(0).all() and cash.cost.eq(0).all()
    first=t.query('episode_id=="E_2020-02-27" and in_lifecycle')
    assert first.as_of_session.iloc[-1]=='2020-08-04'
    assert t.query('episode_id=="E_2020-02-27" and rearmed').as_of_session.iloc[0]=='2020-08-05'
    outside=first[first.window_expired & first.b_state.eq('TRUE')]
    assert len(outside)==97 and outside.b_renewed.sum()==2
    assert outside.b_state_class.eq('SUSTAINED').sum()==95
    renewed=first[first.after_original_b & first.b_renewed].as_of_session.tolist()
    assert renewed==['2020-03-04','2020-03-10','2020-03-13','2020-03-19']
    coverage=pd.read_csv(out/'coverage_days.csv')
    expected=t[t.in_lifecycle & t.after_original_b]
    assert coverage.as_of_session.tolist()==expected.as_of_session.tolist()
    assert len(later)==3*expected.candidate.sum()==606
    assert later.groupby(['episode_id','information_session']).H.apply(set).map(lambda x:x=={5,10,20}).all()
    assert later.one_trade_blocks.all()
    for h in (5,10,20):
        g=later[later.H.eq(h)]
        assert g.position_blocks.tolist()==expected[expected.candidate][f'position_blocks_H{h}'].tolist()
    # Price perturbation can change outcomes but not the causal candidate set.
    changed=p.copy(); changed.loc[301:,'close']*=.37
    cover2,prices2,_=candidates(t,changed,c,cfg)
    assert prices2.trade_id.tolist()==later.trade_id.tolist()
    assert not np.allclose(prices2.net_return,later.net_return)
    class Links(HTMLParser):
        def __init__(self): super().__init__(); self.links=[]; self.ids=[]; self.images=[]
        def handle_starttag(self,tag,attrs):
            d=dict(attrs)
            if 'id' in d:self.ids.append(d['id'])
            if 'href' in d:self.links.append(d['href'])
            if 'src' in d:self.links.append(d['src']);self.images.append(d['src'])
    report=ROOT/'reports/panic_retreat_diagnostic.html'
    html=Links(); html.feed(report.read_text())
    pending=[]
    for url in html.links:
        parsed=urlsplit(url)
        if parsed.scheme: raise AssertionError('Unexpected external dependency')
        if not parsed.path:
            assert parsed.fragment in html.ids
        else:
            target=(report.parent/unquote(parsed.path)).resolve()
            if not target.exists():
                # Final validation/reproducibility are written after this audit.
                assert target.name in ('final_validation.json','reproducibility.json')
                pending.append(str(target.relative_to(ROOT)))
    assert len(html.images)==7 and len(html.ids)==len(set(html.ids))
    result=dict(status='COMPLETED',audited_output_dir=str(out.relative_to(ROOT)),
        real_input_prefixes=len(stops),future_signal_perturbations=len(adaptive),
        availability_prefix_and_future_checks=len(adaptive),price_perturbation_candidate_dates_unchanged=True,
        retrospective_peak_absent_from_causal_API=True,synthetic_values_persisted_as_real_data=False,
        original_reconciliation=original.to_dict('records'),independent_accounts=len(audit),
        independent_account_rows=int(audit.rows_checked.sum()),max_account_error=audit.max_account_error.max(),
        exchange_calendar_execution_checks=len(cells)+len(later),common_account_paths=len(cells),
        lifecycle_2020_complete=True,coverage_rows=len(coverage),later_dates=len(later)//3,
        outside_2020_true=97,outside_2020_sustained=95,outside_2020_renewed=2,
        historical_publish_timestamp_proven=False,
        static_html_links=len(html.links),chart_images=len(html.images),pending_final_record_links=pending,
        full_page_visual_check='NOT_VERIFIED_BROWSER_POLICY_BLOCK',
        protected_files=len(identity),protected_files_unchanged=all(digest(ROOT/f)==v for f,v in identity.items()))
    write_json(ROOT/'runs/panic_retreat_diagnostic/independent_validation.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
