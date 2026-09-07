"""P7审查反例：合成时钟仅存在内存；真实行情不改写、不下载。

断言确认被冻结版本的问题存在，不是修复后的普通回归测试。
从项目根目录运行；所有日报写入被拦截，只向stdout输出。
"""
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import json

import pandas as pd

import svxylab.daily as daily
import svxylab.release_report as report


def deadline_counterexample(root):
    frame, _, _ = daily.load_historical(root)
    forecast_path = root / 'data/clean/forward/predictions/2026-09-08.json'
    original = daily.read_json(forecast_path)
    timing = original['timing']
    prices = pd.read_csv(daily.stage_output(root, 'p2') / 'SVXY_returns.csv')
    clock = {'now': pd.Timestamp('2026-09-08T18:59:59.900Z')}
    captured = {}
    real_exists, real_predict, real_digest = Path.exists, daily.saved_prediction, daily.digest

    class SyntheticClock:
        @classmethod
        def now(cls, tz=None):
            return clock['now'].to_pydatetime()

    def predictor(state, raw):
        result = real_predict(state, raw)
        clock['now'] = pd.Timestamp('2026-09-08T19:00:00.100Z')
        return result

    def exists(path):
        return False if Path(path) == forecast_path else real_exists(path)

    def write(path, value):
        captured[str(Path(path).relative_to(root))] = deepcopy(value)

    def digest(path):
        key = str(Path(path).relative_to(root))
        return 'SYNTHETIC_MEMORY_ONLY' if key in captured else real_digest(path)

    with (
        patch.object(daily, 'datetime', SyntheticClock),
        patch.object(daily, 'verify_release'),
        patch.object(daily, 'obtain_inputs', return_value=(frame, prices, timing, original['inputs'])),
        patch.object(daily, 'record_outcomes', return_value=0),
        patch.object(daily, 'saved_prediction', side_effect=predictor),
        patch.object(daily, 'write_json', side_effect=write),
        patch.object(daily, 'digest', side_effect=digest),
        patch.object(report, 'render_latest'),
        patch.object(Path, 'mkdir'),
        patch.object(Path, 'exists', exists),
        patch.object(Path, 'write_text', side_effect=AssertionError('审查夹具禁止真实落盘')),
    ):
        code = daily.update_daily(root)

    issued = captured[str(forecast_path.relative_to(root))]
    record = next(value for key, value in captured.items() if key.endswith('/daily_run.json'))
    late = clock['now'] >= pd.Timestamp(timing['decision_at'])
    assert code == 0 and record['status'] == 'FORWARD_FORECAST_AVAILABLE'
    assert record['new_prediction_created'] and late
    assert pd.Timestamp(issued['created_at_utc']) < pd.Timestamp(timing['decision_at'])
    assert daily.read_json(forecast_path) == original
    return {
        'fixture': 'SYNTHETIC_DEADLINE_CROSSING_NO_WRITES',
        'exit_code': code,
        'status': record['status'],
        'recorded_creation_time': issued['created_at_utc'],
        'actual_inference_completion': clock['now'].isoformat(),
        'deadline': timing['decision_at'],
        'accepted_after_deadline': bool(late and record['new_prediction_created']),
        'project_writes': 0,
        'network_requests_actually_made': 0,
    }


def coherence_counterexample(root):
    run = daily.read_json(root / daily.read_json(root / 'runs/p7/latest_run.json')['run_record'])
    source = root / run['result']['output_dir'] / 'core/predictions.csv'
    predictions = pd.read_csv(source, float_precision='round_trip')
    scored = predictions.loc[predictions.forecast_available & predictions.score_observed].copy()
    low_q = (scored.q90 < .1 - 1e-8) & (scored.p10 > .1 + 1e-8)
    high_q = (scored.q90 > .1 + 1e-8) & (scored.p10 < .1 - 1e-8)
    counts = scored.assign(conflict=low_q | high_q).groupby('model').conflict.sum().astype(int).to_dict()
    assert counts == {'M0': 0, 'M1': 20, 'M2': 43}
    row = scored.loc[scored.model.eq('M2') & scored.decision_session.eq('2024-08-06')].iloc[0]
    assert row.q90 < .1 and row.p10 > .1
    return {
        'source': str(source.relative_to(root)),
        'source_csv_line': int(row.name + 2),
        'scored_rows_per_model': scored.groupby('model').size().to_dict(),
        'inconsistent_rows': counts,
        'example': {key: row[key] for key in ['decision_session', 'mu5', 'q90', 'p10']},
        'example_risk_only_target': min(1., .05 / row.q90),
        'example_main_target': min(1., .05 / row.q90) if row.mu5 > .001 else 0.,
    }


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[2]
    print('SYNTHETIC CLOCK FIXTURE: all forecast writes are intercepted in memory.')
    print(json.dumps({'deadline': deadline_counterexample(root), 'coherence': coherence_counterexample(root)},
                     ensure_ascii=False, indent=2))
