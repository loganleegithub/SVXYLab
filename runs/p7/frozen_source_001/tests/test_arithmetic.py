"""SYNTHETIC / 合成测试：仅验证运算，绝非 SVXY 历史回测。"""

import pytest

from svxylab.arithmetic import compound_daily

pytestmark = pytest.mark.synthetic


def test_synthetic_daily_minus_half_compounding(synthetic_round_trip):
    # 第一天 +20% 对应 -10%；第二天 -1/6 对应 +1/12；90 × 13/12 = 97.5。
    actual = compound_daily(synthetic_round_trip["index"], multiplier=-0.5)
    assert actual == pytest.approx(synthetic_round_trip["idealized_product"], abs=1e-12)


@pytest.mark.parametrize("levels", [[], [100.0, 0.0], [100.0, float("nan")]])
def test_synthetic_invalid_index_is_rejected(levels):
    with pytest.raises(ValueError):
        compound_daily(levels, multiplier=-0.5)
