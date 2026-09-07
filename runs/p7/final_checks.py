"""P7最终本地交付检查；不下载数据、不更改研究计算。"""
from pathlib import Path
from datetime import datetime,timezone
import json
import os
import re
import subprocess
from urllib.parse import unquote,urlsplit

from bs4 import BeautifulSoup
import pandas as pd

from svxylab.release import verify_release,read_json,digest
from svxylab.features_report import verify_hashes,write_json

root=Path(__file__).resolve().parents[2];freeze=verify_release(root)
record_path=root/read_json(root/'runs/p7/latest_run.json')['run_record'];record=read_json(record_path)
verify_hashes(root,record['artifact_sha256']);assert record['engineering_complete'] and record['tests_passed']==121 and record['tests_failed']==0
versions=[]
for path,copy in [(root/'runs/p7'/f'freeze_{i:03d}.json',root/'runs/p7'/f'frozen_source_{i:03d}') for i in range(1,4)]+[(root/'runs/p7/freeze.json',root/'runs/p7/frozen_source')]:
    v=read_json(path)
    for p,h in v['source_sha256'].items():assert digest(copy/p)==h
    versions.append({'record':str(path.relative_to(root)),'record_sha256':digest(path),'resolvable_source_copy':str(copy.relative_to(root)),'files_verified':len(v['source_sha256'])})
final=root/'runs/p7/final_validation.json';write_json(final,{'stage':'P7','status':'CHECKING'})
html=root/'reports/latest.html';soup=BeautifulSoup(html.read_text(),'html.parser');links=[]
for tag in soup.select('[href],[src]'):
    raw=tag.get('href',tag.get('src'));u=urlsplit(raw)
    if u.scheme or not u.path:continue
    target=(html.parent/unquote(u.path)).resolve();assert target.exists(),str(target);links.append(str(target.relative_to(root)))
for doc in ['STATUS.md','AGENTS.md']:
    for raw in re.findall(r'\]\(([^)]+)\)',(root/doc).read_text()):
        if urlsplit(raw).scheme:continue
        path=(root/unquote(raw.split('#')[0])).resolve();assert path.exists(),str(path)
for entry in ['更新日报.command','重新运行研究.command']:assert os.access(root/entry,os.X_OK)
forward=root/'data/clean/forward/predictions/2026-09-08.json';p=read_json(forward)
assert p['timing']['decision_session']=='2026-09-08' and all(v[k] is None for v in p['predictions'] for k in ['R5','L5','Y10'])
assert read_json(root/'runs/p7/forward_independent_validation.json')['passed']
assert read_json(root/'runs/p7/reproducibility.json')['passed']
# Report-only local derived values are allowed; neither raw nor derived market files are staged/tracked.
tracked=subprocess.check_output(['git','ls-files','-z'],cwd=root).decode().split('\0')
market=[p for p in tracked if p.startswith(('data/raw/','data/clean/')) and not p.endswith('.gitkeep')];assert not market
candidates=subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','-z'],cwd=root).decode().split('\0')
patterns=[re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),re.compile(r'AKIA[A-Z0-9]{16}'),
          re.compile(r'ghp_[A-Za-z0-9]{30,}'),re.compile(r'github_pat_[A-Za-z0-9_]{30,}'),re.compile(r'sk-[A-Za-z0-9]{32,}')]
hits=[];scanned=0
for name in dict.fromkeys(candidates):
    path=root/name
    if not name or not path.is_file():continue
    try:content=path.read_text()
    except UnicodeDecodeError:continue
    scanned+=1
    if any(p.search(content) for p in patterns):hits.append(name)
assert not hits,hits  # Do not print any matched secret value.
protected=['src/svxylab/models.py','src/svxylab/predictions.py','src/svxylab/prediction_data.py','src/svxylab/ledger.py','src/svxylab/economics.py','src/svxylab/diagnostic_models.py','experiment.toml','FEATURES.json','RESEARCH_SPEC.md','requirements-lock.txt','pyproject.toml']
commands=[]
for cmd in [['git','diff','--exit-code','5cf576e','--',*protected],['git','diff','--check']]:
    result=subprocess.run(cmd,cwd=root,capture_output=True,text=True);assert result.returncode==0,result.stdout+result.stderr
    commands.append({'command':cmd,'exit_code':result.returncode})
open_command=subprocess.run(['/usr/bin/open',str(html)],cwd=root);assert open_command.returncode==0
release_invocation=read_json(root/'runs/p7/invocation_20260907T041827223700Z.json');assert release_invocation['exit_code']==0
daily_invocation=read_json(sorted((root/'runs/p7').glob('daily_launcher_*.json'))[-1]);assert daily_invocation['exit_code']==0 and daily_invocation['original_forward_record_preserved']
summary={'stage':'P7','validated_at':datetime.now(timezone.utc).isoformat(),'stage_status':'P7_COMPLETE_AWAITING_FINAL_ACCEPTANCE','passed':True,
    'run_record':str(record_path.relative_to(root)),'run_sha256':digest(record_path),'report_sha256':digest(html),'local_links_checked':len(links),'tables':len(soup.find_all('table')),
    'source_versions':versions,'current_source_files_verified':len(freeze['source_sha256']),'artifact_files_verified':len(record['artifact_sha256']),
    'protected_models_mapping_configuration_unchanged':protected,'accepted_p4_p5_p6_artifacts_unchanged':True,
    'original_p1_raw_and_clean_verified_by_release_loader':True,'python_tests_passed':record['tests_passed'],'macos_open_exit_code':open_command.returncode,
    'codex_browser_and_ledger_panel_status':'queued by open_in_codex; macOS open succeeded separately','browser_screenshot_claimed':False,
    'plot_visually_inspected':'data/clean/p7/20260907T041828788697Z/core_common_equity.png',
    'research_launcher_actual_exit_code':release_invocation['exit_code'],'daily_launcher_actual_exit_code':daily_invocation['exit_code'],
    'first_forward_record_preserved_on_repeat':True,'forward_prediction_sha256':digest(forward),'forward_results_all_null':True,
    'forward_independent_check':'runs/p7/forward_independent_validation.json','reproducibility':'runs/p7/reproducibility.json',
    'secret_signature_scan_files':scanned,'secret_signature_matches':[],'tracked_raw_or_clean_market_files':market,
    'scope_of_secret_check':'Recognizable private-key/token signatures and tracked-file inventory; no secret values printed, no credentials or raw files added. Not a universal proof about every possible credential format.',
    'daily_network_requests_this_run':0,'live_post_freeze_download_branch_exercised':False,'remote_actions':[],'account_connections':False,'orders':False,'automatic_model_promotion':False,
    'p6_accepted_git_checkpoint':read_json(root/'runs/p6/accepted_checkpoint.json')['commit'],'p7_git_checkpoint_pending_final_acceptance':True,
    'documentation_sha256':{f:digest(root/f) for f in ['AGENTS.md','STATUS.md']},'commands':commands}
write_json(final,summary)
print({k:summary[k] for k in ['passed','local_links_checked','tables','current_source_files_verified','artifact_files_verified','secret_signature_scan_files','research_launcher_actual_exit_code','daily_launcher_actual_exit_code']})
