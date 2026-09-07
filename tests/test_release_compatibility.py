"""合成文件夹具：P7原冻结不变，明确兼容摘要不会放过额外源码改动。"""
import json
from pathlib import Path
import pytest
from svxylab.release import digest, verify_release

pytestmark=pytest.mark.synthetic


def fixture_root(root):
    def write(name,value):
        p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(value);return p
    source=write('src/svxylab/daily.py','old daily')
    model=write('src/svxylab/models.py','unchanged model')
    original={'src/svxylab/daily.py':digest(source),'src/svxylab/models.py':digest(model)}
    freeze=write('runs/p7/freeze.json',json.dumps({'source_sha256':original,'historical_cutoff':'2026-09-04'}))
    for stage in ['p4','p5','p6']:
        report=write(f'reports/{stage}.html','accepted')
        run=write(f'runs/{stage}/run.json','{}')
        write(f'runs/{stage}/acceptance.json',json.dumps({'accepted_report':str(report.relative_to(root)),
          'accepted_report_sha256':digest(report),'accepted_run':str(run.relative_to(root)),
          'accepted_run_sha256':digest(run),'accepted_artifact_sha256':{}}))
    write('runs/p7/frozen_source/src/svxylab/daily.py','old daily')
    return write,source,model,freeze,original


def test_explicit_amendment_keeps_original_and_rejects_unrecorded_change(tmp_path):
    write,source,model,freeze,original=fixture_root(tmp_path)
    assert verify_release(tmp_path)['source_sha256']==original
    source.write_text('deadline repair')
    with pytest.raises(ValueError):verify_release(tmp_path)
    write('runs/n1/runtime_compatibility.json',json.dumps({'p7_freeze_sha256':digest(freeze),
      'changed_frozen_files':{'src/svxylab/daily.py':{'before':original['src/svxylab/daily.py'],'after':digest(source)}}}))
    result=verify_release(tmp_path)
    assert result['source_sha256']['src/svxylab/daily.py']==digest(source)
    assert result['original_source_sha256']==original
    assert json.loads(freeze.read_text())['source_sha256']==original
    model.write_text('unauthorized change')
    with pytest.raises(ValueError):verify_release(tmp_path)


def test_amendment_cannot_reference_different_old_source(tmp_path):
    write,source,model,freeze,original=fixture_root(tmp_path)
    write('runs/n1/runtime_compatibility.json',json.dumps({'p7_freeze_sha256':digest(freeze),
      'changed_frozen_files':{'src/svxylab/daily.py':{'before':'incorrect','after':digest(source)}}}))
    with pytest.raises(ValueError,match='旧摘要'):verify_release(tmp_path)
