"""从当时份额价格和公司行动计算收盘总回报，不填补缺失价格。"""

import numpy as np
import pandas as pd


def total_return(prices: pd.DataFrame) -> pd.DataFrame:
    frame = prices.sort_values("as_of_session").reset_index(drop=True).copy()
    fields = ["close", "split_factor", "cash_dividend", "capital_gain_distribution"]
    values = frame[fields].to_numpy(dtype=float)
    if (frame.empty or frame.as_of_session.duplicated().any() or not np.isfinite(values).all()
            or (values[:, :2] <= 0).any() or (values[:, 2:] < 0).any()):
        raise ValueError("价格/公司行动缺失、重复或无效；不能制造总回报")
    frame["previous_close"] = frame.close.shift()
    distributions = frame.cash_dividend + frame.capital_gain_distribution
    gross = (frame.close + distributions) / (frame.split_factor * frame.previous_close)
    frame["simple_return"] = gross - 1
    frame["log_return"] = np.log(gross)
    # 首日只是财富序列的锚点；首日之前的真实回报保持未知。
    frame["total_return_index"] = 100 * np.r_[1.0, gross.iloc[1:].to_numpy()].cumprod()
    if "provider_adjusted_close" in frame:
        frame["provider_reference_return"] = frame.provider_adjusted_close.pct_change(fill_method=None)
        frame["reference_difference_bps"] = (frame.simple_return - frame.provider_reference_return) * 10000
    return frame
