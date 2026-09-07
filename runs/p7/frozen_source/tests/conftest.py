"""明确标记的合成测试夹具；这里的数值不是市场观测。"""

import pytest


@pytest.fixture
def synthetic_round_trip():
    return {"index": [100.0, 120.0, 100.0], "idealized_product": [100.0, 90.0, 97.5]}
