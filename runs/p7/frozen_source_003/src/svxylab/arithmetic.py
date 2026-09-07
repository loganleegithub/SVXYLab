"""每日重设倍数的理想化复利算术；不提供真实基金回测。"""

from collections.abc import Sequence
from math import isfinite


def compound_daily(
    index_levels: Sequence[float], *, multiplier: float, initial_value: float = 100.0
) -> list[float]:
    """逐日计算 V[t] = V[t-1] * (1 + multiplier * (I[t]/I[t-1] - 1))。

    不包含基金跟踪误差、费用、公司行动或成交时钟；仅供算术检查。
    """
    if not index_levels or any(not isfinite(x) or x <= 0 for x in index_levels):
        raise ValueError("指数水平必须是非空、有限、严格正数序列")
    if not isfinite(multiplier) or not isfinite(initial_value) or initial_value <= 0:
        raise ValueError("倍数须有限，初始值须有限且严格为正")
    values = [initial_value]
    for previous, current in zip(index_levels, index_levels[1:]):
        factor = 1 + multiplier * (current / previous - 1)
        value = values[-1] * factor
        if factor < 0 or not isfinite(value):
            raise ValueError("该输入超出非负、有限的理想化算术路径范围")
        values.append(value)
    return values
