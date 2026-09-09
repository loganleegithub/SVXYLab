"""E1 independent artifact checks. Run from the existing virtual environment.

Rechecks saved scalar bootstrap using cluster multiplicity matrix algebra,
compares two completed runs, and checks local HTML targets without a browser.
Does not re-fit old models or recursively hash historical artifacts.
"""
from collections import Counter
from hashlib import sha256
from html.parser import HTMLParser
import json
from pathlib import Path
import subprocess
import tomllib
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'runs/panic_retreat'
cfg=tomllib.loads((ROOT/'panic_retreat.toml').read_text())


def digest(p):return sha256(p.read_bytes()).hexdigest()


def read_json(p):return json.loads(p.read_text())


def main():
    successful=[]
    for p in sorted(BASE.glob('*/run.json')):
        r=read_json(p)
        if r['status']=='COMPUTATION_COMPLETE':successful.append((p.parent,r))
    (first,ar),(second,br)=successful[-2:]
    a=ROOT/ar['output_dir'];b=ROOT/br['output_dir']
    assert read_json(first/'experiment_record.json')['source_sha256']==read_json(second/'experiment_record.json')['source_sha256']
    assert read_json(first/'inputs.json')==read_json(second/'inputs.json')
    csvs=sorted(x.name for x in a.glob('*.csv'))
    assert csvs==sorted(x.name for x in b.glob('*.csv'))
    comparisons=[]
    for name in csvs:
        ha=digest(a/name);hb=digest(b/name)
        assert ha==hb,name
        comparisons.append({'file':name,'sha256':ha})
    calculation_jsons=['summary.json','candidate_card.json','n1_reconciliation.json']
    for name in calculation_jsons:
        ja,jb=read_json(a/name),read_json(b/name)
        if name=='summary.json':
            # Delivery annotations can be appended after numerical comparison.
            ja.pop('engineering',None);jb.pop('engineering',None)
        assert ja==jb,name
    inputs=read_json(second/'inputs.json')
    for rel,expected in inputs['sha256'].items():assert digest(ROOT/rel)==expected,rel
    read=lambda name:pd.read_csv(b/(name+'.csv'))
    ep=read('episodes');sig=read('signals');opp=read('opportunity_results');trades=read('trades')
    assert len(sig)==len(ep)*5 and len(opp)==len(ep)*5*3
    assert opp.groupby(['episode_id','H']).rule_id.nunique().eq(5).all()
    assert not opp.duplicated(['episode_id','rule_id','H']).any()
    assert (trades.u-trades.e).eq(trades.H).all()
    assert (trades.e-trades.t).eq(trades.lag).all()
    assert (opp.common_end_i-opp.s).eq(cfg['window']+opp.lag+opp.H).all()
    assert opp.loc[opp.opportunity_status.eq('CASH_NO_TRIGGER'),'net_return'].eq(0).all()
    # Every participated event contributes once to the same continuous capital.
    part=read('participation');accounts=read('account_summary')
    for (scenario,rule,h),g in part.groupby(['scenario','rule_id','H']):
        total=g.continuous_pnl_dollars.sum()
        account=accounts.loc[accounts.scenario.eq(scenario)&accounts.account.eq(rule)&accounts.H.eq(h)&accounts.period.eq('ALL')].iloc[0]
        assert abs(total-account.pnl_dollars)<1e-6
    # Independent bootstrap: sum cluster outcome/count with a multinomial count matrix.
    clusters=read('inference_clusters');keys=sorted(clusters.cluster_id.unique());mapping=dict(zip(clusters.episode_id,clusters.cluster_id))
    draws=np.random.default_rng(cfg['seed']).integers(0,len(keys),(cfg['bootstrap_repetitions'],len(keys)))
    counts=np.array([[Counter(row)[j] for j in range(len(keys))] for row in draws])
    boot=read('bootstrap_scalars');pairs=read('paired_events');max_error=0.
    for label,g in boot.groupby('metric',sort=False):
        if label.endswith('_TRADES'):
            rule=label.removesuffix('_TRADES');values=trades.loc[trades.H.eq(10)&trades.rule_id.eq(rule),['episode_id','net_return']]
        elif label.endswith('_OPPORTUNITIES'):
            rule=label.removesuffix('_OPPORTUNITIES');values=opp.loc[opp.H.eq(10)&opp.rule_id.eq(rule),['episode_id','net_return']]
        else:values=pairs.loc[pairs.H.eq(10)&pairs.pair.eq(label),['episode_id','net_return']]
        values=values.dropna();values['cluster']=values.episode_id.map(mapping)
        sums=np.array([values.loc[values.cluster.eq(k),'net_return'].sum() for k in keys])
        sizes=np.array([values.cluster.eq(k).sum() for k in keys])
        expected=(counts@sums)/(counts@sizes)
        actual=g.sort_values('replicate')['mean'].to_numpy()
        error=float(np.max(np.abs(expected-actual)));assert error<1e-12
        max_error=max(max_error,error)
    class Links(HTMLParser):
        def __init__(self):super().__init__();self.ids=set();self.targets=[]
        def handle_starttag(self,tag,attrs):
            attrs=dict(attrs)
            if 'id' in attrs:self.ids.add(attrs['id'])
            for k in ['href','src']:
                if k in attrs:self.targets.append(attrs[k])
    html=(ROOT/'reports/panic_retreat.html').read_text();parser=Links();parser.feed(html)
    for target in parser.targets:
        if target.startswith('#'):assert target[1:] in parser.ids,target
        elif not target.startswith(('http://','https://')):assert Path(target).exists(),target
    protected=['experiment.toml','RESEARCH_SPEC.md','src/svxylab/__main__.py','src/svxylab/release.py',
               'src/svxylab/ledger.py','src/svxylab/returns.py','src/svxylab/timing.py']
    for rel in protected:
        original=subprocess.check_output(['git','show','HEAD:'+rel],cwd=ROOT)
        assert (ROOT/rel).read_bytes()==original,rel
    xml=ET.parse(BASE/'pytest_final.xml').getroot()
    suites=xml.findall('.//testsuite')
    test_stats={k:sum(int(s.attrib.get(k,0)) for s in suites) for k in ['tests','failures','errors','skipped']}
    assert test_stats['failures']==test_stats['errors']==0
    result={'status':'VERIFIED','first_run':str(first.relative_to(ROOT)),'second_run':str(second.relative_to(ROOT)),
      'equal_csv_files':len(csvs),'equal_calculation_json_files':len(calculation_jsons),'csv_identity':comparisons,
      'non_calculation_json_annotations_excluded':{'summary.json':['engineering']},
      'inputs_verified':len(inputs['sha256']),'protected_legacy_files_unchanged':protected,
      'bootstrap_scalars_independently_recomputed':len(boot),'bootstrap_max_error':max_error,
      'episodes':len(ep),'signals':len(sig),'opportunity_rows':len(opp),'ordinary_pytest':test_stats,
      'local_html_targets_exist':True,'html_targets_checked':len(parser.targets),
      'test_source_sha256':digest(ROOT/'tests/test_panic_retreat.py'),
      'visual_check':{'browser_page':'BLOCKED_BY_LOCAL_FILE_URL_SECURITY_POLICY; no workaround attempted',
         'macos_open':'exit 0; not evidence of page inspection','chart_files':'continuous and worst 2020 case viewed directly; legible Chinese, axes and legends',
         'full_page_screenshot_verified':False},
      'no_commit_or_push':True,'user_acceptance':'PENDING'}
    (BASE/'final_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['csv_identity','protected_legacy_files_unchanged']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
