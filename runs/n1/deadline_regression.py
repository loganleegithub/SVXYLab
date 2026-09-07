"""真实已存行情与模型、仅合成时钟：全部日报写入拦截到内存。"""
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import json
import pandas as pd
import svxylab.daily as daily
import svxylab.release as release
import svxylab.release_report as report


def check(root, completion, expect_created):
    # 此处仅隔离源码版本门控；最终另行验证N1明确兼容清单。
    with patch.object(release,'verify_release',return_value=daily.read_json(root/'runs/p7/freeze.json')):
        frame,_,_=daily.load_historical(root)
    forecast=root/'data/clean/forward/predictions/2026-09-08.json'
    original=daily.read_json(forecast);timing=original['timing']
    prices=pd.read_csv(daily.stage_output(root,'p2')/'SVXY_returns.csv')
    clock={'now':pd.Timestamp('2026-09-08T18:59:59.900Z')};captured={}
    real_exists,real_predict,real_digest=Path.exists,daily.saved_prediction,daily.digest
    class SyntheticClock:
        @classmethod
        def now(cls,tz=None):return clock['now'].to_pydatetime()
    def predict(state,raw):
        result=real_predict(state,raw);clock['now']=pd.Timestamp(completion);return result
    def exists(path):return False if Path(path)==forecast else real_exists(path)
    def write(path,value):captured[str(Path(path).relative_to(root))]=deepcopy(value)
    def digest(path):
        key=str(Path(path).relative_to(root))
        return 'SYNTHETIC_MEMORY_ONLY' if key in captured else real_digest(path)
    with (patch.object(daily,'datetime',SyntheticClock),patch.object(daily,'verify_release'),
          patch.object(daily,'obtain_inputs',return_value=(frame,prices,timing,original['inputs'])),
          patch.object(daily,'record_outcomes',return_value=0),patch.object(daily,'saved_prediction',side_effect=predict),
          patch.object(daily,'write_json',side_effect=write),patch.object(daily,'digest',side_effect=digest),
          patch.object(report,'render_latest'),patch.object(Path,'mkdir'),patch.object(Path,'exists',exists),
          patch.object(Path,'write_text',side_effect=AssertionError('NO_DISK_WRITES'))):
        code=daily.update_daily(root)
    row=next(v for k,v in captured.items() if k.endswith('/daily_run.json'))
    key=str(forecast.relative_to(root))
    assert row['new_prediction_created']==expect_created
    assert (key in captured)==expect_created
    assert code==(0 if expect_created else 1)
    if expect_created:
        assert pd.Timestamp(captured[key]['created_at_utc'])==pd.Timestamp(completion)
        # 9个预测头与真实原始预报完全相同，时钟修复不改推理结果。
        assert captured[key]['predictions']==original['predictions']
    else:assert row['status']=='MISSED_DECISION_DEADLINE'
    assert daily.read_json(forecast)==original
    return {'completion':completion,'deadline':timing['decision_at'],'created':expect_created,'exit_code':code,
        'status':row['status'],'forecast_value_parity':expect_created,'project_writes':0,'network_requests':0}

if __name__=='__main__':
    r=Path.cwd();result=[check(r,t,ok) for t,ok in [
        ('2026-09-08T19:00:00.100Z',False),('2026-09-08T19:00:00.000Z',False),
        ('2026-09-08T18:59:59.999Z',True)]]
    (r/'runs/n1/deadline_after.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))
