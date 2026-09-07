"""明确合成夹具：检查审查分组、重叠与差异检测，不是行情。"""

import numpy as np
import pandas as pd
import pytest

from svxylab.prediction_review import compare_runs, event_intervals, probability_groups

pytestmark = pytest.mark.synthetic


def test_equal_frequency_keeps_identical_probabilities_together():
    frame = pd.DataFrame({"model": ["M0"]*10, "p10": [.1]*7+[.4, .6, .9], "Y10": [0]*8+[1, 1],
                          "as_of_session": [str(i) for i in range(10)], "decision_session": [str(i) for i in range(10)]})
    groups, members = probability_groups(frame)
    freq = members.loc[members.method.eq("equal_frequency")]
    assert freq.groupby("p10").group.nunique().max() == 1
    summary = groups.loc[groups.method.eq("equal_frequency")]
    assert summary.rows.sum() == 10 and summary.positive_rows.sum() == 2
    assert summary.rows.max() >= 7


def test_constant_probability_is_one_frequency_group_and_five_width_groups():
    frame = pd.DataFrame({"model": ["M0"]*5, "p10": [.2]*5, "Y10": [0, 0, 0, 1, 1],
                          "as_of_session": list("abcde"), "decision_session": list("abcde")})
    groups, _ = probability_groups(frame)
    frequency = groups.loc[groups.method.eq("equal_frequency")]
    assert len(frequency) == 1 and frequency.rows.item() == 5 and frequency.event_rate.item() == .4
    width = groups.loc[groups.method.eq("equal_width")]
    assert len(width) == 5 and width.loc[width.group.eq(2), "rows"].item() == 5


def test_positive_intervals_merge_transitive_overlaps_but_not_gaps():
    frame = pd.DataFrame({"as_of_session": ["2020-01-01", "2020-01-04", "2020-01-08", "2020-01-20"],
                          "decision_session": ["2020-01-02", "2020-01-05", "2020-01-09", "2020-01-21"],
                          "label_end_session": ["2020-01-07", "2020-01-10", "2020-01-14", "2020-01-28"]})
    intervals, members = event_intervals(frame)
    assert intervals.positive_rows.tolist() == [3, 1]
    assert intervals.interval_end.tolist() == ["2020-01-14", "2020-01-28"]
    assert members.event_id.tolist() == [1, 1, 1, 2]


def test_refit_comparison_ignores_only_actual_timing_and_rejects_value_change(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    old.mkdir(); new.mkdir()
    a = pd.DataFrame({"decision_session": ["2020-01-02"], "model": ["M1"], "mu5": [.1], "computed_at_utc": ["first"]})
    a.to_csv(old / "predictions.csv", index=False)
    b = a.copy(); b["computed_at_utc"] = "second"
    b.to_csv(new / "predictions.csv", index=False)
    assert compare_runs(old, new)["passed"]
    b["mu5"] += .001
    b.to_csv(new / "predictions.csv", index=False)
    with pytest.raises(AssertionError):
        compare_runs(old, new)
