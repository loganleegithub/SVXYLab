"""记录本轮明确兼容改动；不改写P7冻结文件或旧源码。"""
from pathlib import Path
from datetime import datetime,timezone
import json,tomllib
from svxylab.release import digest,read_json,verify_release
from svxylab.features_report import write_json
r=Path.cwd();freeze=read_json(r/'runs/p7/freeze.json');changes={}
reasons={'src/svxylab/daily.py':'推理/元数据完成后重新取时并检查截止，准确生成时间；新增兼容记录摘要以区分修复运行版本。',
'src/svxylab/release.py':'保留原冻结清单和原源码副本，核对N1明确的旧/新摘要并返回实际运行源码摘要；不改模型或历史计算。',
'src/svxylab/__main__.py':'增加capital命令，旧命令路由不变。',
'experiment.toml':'仅新增[n1]资本基准参数，所有原节完全不变。',
'RESEARCH_SPEC.md':'更新阶段状态并追加N1用户授权范围，原P7规格副本不变。'}
for path,before in freeze['source_sha256'].items():
 after=digest(r/path)
 if after!=before:
  assert path in reasons,path
  assert digest(r/'runs/p7/frozen_source'/path)==before
  changes[path]={'before':before,'after':after,'reason':reasons[path]}
old=tomllib.loads((r/'runs/p7/frozen_source/experiment.toml').read_text());new=tomllib.loads((r/'experiment.toml').read_text());del new['n1'];assert old==new
record={'stage':'N1','recorded_at_utc':datetime.now(timezone.utc).isoformat(),'p7_freeze_sha256':digest(r/'runs/p7/freeze.json'),
 'authorization':'本轮用户已授权截止修复和N1；本记录不是新P7冻结或模型规格选择。',
 'changed_frozen_files':changes,'original_config_sections_equal':True,
 'historical_predictions_models_labels_ledgers_and_reports':'UNCHANGED_SHA_VERIFIED',
 'original_counterexample':'runs/n1/deadline_before.log','repaired_regression':'runs/n1/deadline_after.json'}
p=r/'runs/n1/runtime_compatibility.json'
if p.exists():
 assert read_json(p)['changed_frozen_files']==changes,'已记录兼容改动再次变化，需显式修订说明'
else:write_json(p,record)
verified=verify_release(r);print('explicit changed files',len(changes),'active sources',len(verified['source_sha256']))
