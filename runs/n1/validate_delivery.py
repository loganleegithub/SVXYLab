"""N1最终只读内容验证和两次复跑对照；仅写本轮验证结果。"""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
from datetime import datetime,timezone
import json,subprocess,tomllib
import xml.etree.ElementTree as ET
import pandas as pd
from svxylab.features_report import verify_hashes,write_json
from svxylab.release import read_json,digest,verify_release
r=Path.cwd();latest=read_json(r/read_json(r/'runs/n1/latest_run.json')['run_record']);out=r/latest['output_dir']
prior=r/'data/clean/n1/20260907T062451688773Z'
files=sorted(out.rglob('*.csv'));same=[]
for p in files:
 q=prior/p.relative_to(out)
 assert q.exists();assert p.read_bytes()==q.read_bytes(),p
 same.append(str(p.relative_to(out)))
for name in ['input_verification.json','original_ledger_parity.json','independent_validation.json']:
 assert read_json(out/name)==read_json(prior/name)
verify_hashes(r,latest['artifacts_sha256'])
preserved=read_json(r/'runs/n1/experiment_record.json')['preserved_sha256'];verify_hashes(r,preserved)
active=verify_release(r)
original=tomllib.loads((r/'runs/p7/frozen_source/experiment.toml').read_text());current=tomllib.loads((r/'experiment.toml').read_text());current.pop('n1');assert current==original
class Links(HTMLParser):
 def __init__(self):super().__init__();self.links=[];self.images=[]
 def handle_starttag(self,tag,attrs):
  d=dict(attrs)
  if tag=='a':self.links.append(d.get('href',''))
  if tag=='img':self.images.append(d.get('src',''))
link_checks=[]
for p in [r/'reports/capital.html',out/'report.html']:
 parser=Links();parser.feed(p.read_text());local=[u for u in parser.links if not urlsplit(u).scheme]
 missing=[u for u in local if not (p.parent/unquote(u)).resolve().exists()]
 assert not missing;assert len(parser.images)==2 and all(u.startswith('data:image/png;base64,') for u in parser.images)
 assert 'initial_e<' not in p.read_text()
 link_checks.append({'file':str(p.relative_to(r)),'verified_local_links':len(local),'embedded_charts':len(parser.images)})
suites=ET.parse(r/'runs/n1/pytest.xml').getroot();s=suites.find('testsuite')
assert s is not None and s.get('failures')=='0' and s.get('errors')=='0'
parity=read_json(out/'original_ledger_parity.json')
metrics=pd.read_csv(out/'metrics.csv');counts=metrics.groupby('period').account_days.first().to_dict()
result={'validated_at_utc':datetime.now(timezone.utc).isoformat(),'run_record':read_json(r/'runs/n1/latest_run.json')['run_record'],
 'original_p7_related_files_unchanged':len(preserved),'p7_freeze_unchanged_sha256':digest(r/'runs/p7/freeze.json'),
 'explicit_compatibility_changes':list(read_json(r/'runs/n1/runtime_compatibility.json')['changed_frozen_files']),
 'active_source_files_verified':len(active['source_sha256']),'original_config_sections_unchanged':True,
 'ordinary_tests':int(s.get('tests')),'ordinary_test_failures':0,'synthetic_clock_regressions':read_json(r/'runs/n1/deadline_after.json'),
 'reproducibility':{'earlier_dir':str(prior.relative_to(r)),'final_dir':latest['output_dir'],'csv_files_byte_identical':len(same),
  'ledger_csv_files_byte_identical':len([p for p in same if p.startswith('ledgers/')]),'three_audit_json_equal':True,
  'note':'两次完整计算均重新从真实已存输入生成82个账户；首轮N1报表语义问题修正之前的计算不冒充已修复版本。'},
 'original_p5_p7_accounts_compared':len(parity),'original_p5_p7_max_equity_difference':max(x['max_equity_difference'] for x in parity),
 'period_account_days':counts,'independent_audit':latest['independent_audit'],'html_local_links':link_checks,
 'open_command':latest['open_command'],'charts_visually_inspected':True,
 'html_browser_visual_check':'NOT_COMPLETED: browser URL security policy rejected file://; no bypass attempted. Native macOS open exit 0, local HTML links checked, PNG charts visually inspected.',
 'paid_purchases':0,'option_quote_rows':0,'accounts_connected':False,'orders':False,'git_committed_or_pushed':False}
write_json(r/'runs/n1/final_validation.json',result)
write_json(r/'runs/n1/reproducibility.json',result['reproducibility'])
print(json.dumps({k:result[k] for k in ['ordinary_tests','original_p7_related_files_unchanged','reproducibility','original_p5_p7_accounts_compared','original_p5_p7_max_equity_difference']},ensure_ascii=False,indent=2))
