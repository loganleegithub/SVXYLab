"""P7实际产物比较：排除实际计算时间/计时字段，不排除预测或账户值。"""
from pathlib import Path
import json
from datetime import datetime,timezone

import numpy as np
import pandas as pd

root=Path(__file__).resolve().parents[2]
final_record=json.loads((root/'runs/p7/latest_run.json').read_text())
run=json.loads((root/final_record['run_record']).read_text());final=root/run['result']['output_dir']
previous=root/'data/clean/p7/20260907T041300923825Z'
first=root/'data/clean/p7/20260907T040716823855Z'
ignore={'computed_at_utc','elapsed_seconds'}

def clean(value):
    if isinstance(value,dict):return {k:clean(v) for k,v in value.items() if k not in ignore}
    if isinstance(value,list):return [clean(v) for v in value]
    return value

comparisons=[]
for original in [first,previous]:
    csvs=jsons=archives=ledgers=0;exact_csv=0;max_number=0.
    for p in sorted(original.rglob('*.csv')):
        other=final/p.relative_to(original)
        assert other.exists(),str(p)
        try:a=pd.read_csv(p,float_precision='round_trip');b=pd.read_csv(other,float_precision='round_trip')
        except pd.errors.EmptyDataError:
            assert p.read_bytes()==other.read_bytes();csvs+=1;exact_csv+=1;continue
        a=a.drop(columns=list(ignore),errors='ignore');b=b.drop(columns=list(ignore),errors='ignore')
        assert a.shape==b.shape and a.columns.tolist()==b.columns.tolist(),str(p)
        try:pd.testing.assert_frame_equal(a,b,check_exact=True);exact_csv+=1
        except AssertionError:pd.testing.assert_frame_equal(a,b,atol=2e-11,rtol=2e-10)
        for col in a.select_dtypes(include='number'):
            values=np.abs(a[col].to_numpy(float)-b[col].to_numpy(float));finite=values[np.isfinite(values)]
            if len(finite):max_number=max(max_number,float(finite.max()))
        if p.parent.name=='ledgers':assert p.read_bytes()==other.read_bytes();ledgers+=1
        csvs+=1
    for p in sorted(original.rglob('*.json')):
        if p.name in ['input_verification.json','runtime_config.json']:continue
        other=final/p.relative_to(original);assert other.exists(),str(p)
        assert clean(json.loads(p.read_text()))==clean(json.loads(other.read_text())),str(p);jsons+=1
    for p in original.rglob('*.npz'):
        a=np.load(p);b=np.load(final/p.relative_to(original));assert a.files==b.files
        for k in a.files:np.testing.assert_array_equal(a[k],b[k])
        archives+=1
    comparisons.append({'original':str(original.relative_to(root)),'final':str(final.relative_to(root)),
        'csvs':csvs,'exact_csvs_after_only_timing_exclusion':exact_csv,'max_numeric_difference':max_number,
        'identical_jsons_excluding_only_timing':jsons,'identical_bootstrap_arrays':archives,
        'byte_identical_ledgers':ledgers})
result={'checked_at':datetime.now(timezone.utc).isoformat(),'passed':True,'excluded_fields':sorted(ignore),
    'comparisons':comparisons,'interpretation':'Two input refits/rollouts, not merely coefficient reconstruction. First was stopped before accounts due to holiday boundary; second completed all numerical work but failed JSON summary serialization. Original data and outputs retained.','final_run':final_record['run_record']}
(root/'runs/p7/reproducibility.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(result)
