"""交易日时钟、结果成熟时间与固定五交易日标签。"""

import numpy as np
import pandas as pd


def session_clock(information_sessions: list[str], schedule: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    if horizon < 1 or len(set(information_sessions)) != len(information_sessions):
        raise ValueError("标签期限或信息日无效")
    dates = schedule.index.strftime("%Y-%m-%d").tolist()
    positions = {date: i for i, date in enumerate(dates)}
    close = pd.to_datetime(schedule["close"], utc=True)
    records = []
    for date in information_sessions:
        i = positions[date]
        if i + horizon + 2 >= len(dates):
            raise ValueError("交易日历未延伸至标签的假设可用时间")
        decision = close.iloc[i + 1] - pd.Timedelta(minutes=60)
        records.append({
            "as_of_session": date,
            "information_close_at": close.iloc[i].isoformat(),
            "decision_session": dates[i + 1], "decision_at": decision.isoformat(),
            "decision_at_new_york": decision.tz_convert("America/New_York").isoformat(),
            "execution_session": dates[i + 1], "execution_at": close.iloc[i + 1].isoformat(),
            "label_end_session": dates[i + horizon + 1],
            "label_matures_at": close.iloc[i + horizon + 1].isoformat(),
            "label_available_at": (close.iloc[i + horizon + 2] - pd.Timedelta(minutes=60)).isoformat(),
            "availability_policy": "ASSUMED_NEXT_SESSION_60MIN_BEFORE_CLOSE",
        })
    return pd.DataFrame(records)


def five_day_labels(returns: pd.DataFrame, clock: pd.DataFrame, horizon: int = 5,
                    loss_threshold: float = .10) -> pd.DataFrame:
    if not 0 < loss_threshold <= 1:
        raise ValueError("损失事件阈值必须在 (0,1]")
    if returns.as_of_session.tolist() != clock.as_of_session.tolist():
        raise ValueError("回报和时钟的信息日不一致")
    labels = clock.copy()
    path = returns.total_return_index.to_numpy(dtype=float)
    result = []
    for i in range(len(path)):
        entry, end = i + 1, i + 1 + horizon
        row = {"R5": np.nan, "L5": np.nan, "Y10": np.nan, "observed": False,
               "reason": "未来五日路径尚未完整取得"}
        if end < len(path):
            window = path[entry:end + 1]
            if np.isfinite(window).all() and (window > 0).all():
                worst = window[1:].min()
                row = {"R5": window[-1] / window[0] - 1,
                       "L5": max(0.0, (window[0] - worst) / window[0]),
                       # 以价格门槛比较，避免 1 - 90/100 的浮点误差漏记恰好 10%。
                       "Y10": int(worst <= window[0] * (1 - loss_threshold)),
                       "observed": True, "reason": "已取得完整未来收盘路径"}
            else:
                row["reason"] = "未来路径存在价格缺口"
        result.append(row)
    return pd.concat([labels, pd.DataFrame(result)], axis=1)


def mature_labels(labels: pd.DataFrame, decision_at: str) -> pd.DataFrame:
    cutoff = pd.Timestamp(decision_at)
    if cutoff.tzinfo is None:
        raise ValueError("决策截止必须带时区")
    return labels.loc[labels.observed
                      & (pd.to_datetime(labels.label_matures_at, utc=True) < cutoff)
                      & (pd.to_datetime(labels.label_available_at, utc=True) <= cutoff)].copy()
