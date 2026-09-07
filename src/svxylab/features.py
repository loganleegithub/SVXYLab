"""P3 原始联合特征：只做向后计算，不读取标签、拟合或筛选特征。"""

import numpy as np
import pandas as pd


CORE_IDS = ["A01", "A02", "A03", "A04", "A05", "B01", "B02", "B03", "C01", "C02", "C03",
            "D01", "D02", "D03", "E01", "F01", "F02", "F03", "F04"]
EXTENSION_IDS = ["G01", "G02"]
LOOKBACK = {key: 0 for key in CORE_IDS + EXTENSION_IDS}
LOOKBACK.update({key: 5 for key in ["A04", "C02", "C03", "D01", "F01", "F03", "G02"]})
LOOKBACK.update({key: 21 for key in ["D02", "D03", "E01", "F04"]})
LOOKBACK["F02"] = 20
UNITS = {key: "无量纲（自然对数或对数比）" for key in CORE_IDS}
UNITS.update({"A01": "无量纲（30日标准化对数斜率）", "A02": "无量纲（30日标准化斜率差）",
              "A05": "日历天", "D01": "年化波动（小数）", "D02": "年化波动（小数）",
              "D03": "比例（0至1）", "E01": "年化方差（小数平方）", "F02": "收盘回撤（小数）",
              "F04": "年化波动（小数）", "G01": "指数点", "G02": "指数点变化"})
INPUT_COLUMNS = {
    "A01": ["F1", "F2", "tau1", "tau2"], "A02": ["F1", "F2", "F3", "tau1", "tau2", "tau3"],
    "A03": ["F1", "VIX"], "A04": ["f1_contract", "a04_current_settle", "lag5_session", "a04_lag5_settle"],
    "A05": ["f1_expiration", "tau1"], "B01": ["VIX"], "B02": ["VIX9D", "VIX"], "B03": ["VIX", "VIX3M"],
    "C01": ["VVIX"], "C02": ["VVIX", "lag5_session"], "C03": ["VIX", "lag5_session"],
    "D01": ["TR_SPY", "r_SPY"], "D02": ["TR_SPY", "r_SPY"], "D03": ["TR_SPY", "r_SPY"],
    "E01": ["VIX", "TR_SPY", "r_SPY"], "F01": ["TR_SPY", "lag5_session"], "F02": ["TR_SPY"],
    "F03": ["TR_SVXY", "lag5_session"], "F04": ["TR_SVXY", "r_SVXY"], "G01": ["SKEW"], "G02": ["SKEW", "lag5_session"],
}


def aligned(frame: pd.DataFrame, sessions: pd.Index) -> pd.DataFrame:
    if frame.as_of_session.duplicated().any():
        raise ValueError("输入存在重复交易日，不能选择其中任意一行")
    return frame.set_index("as_of_session").reindex(sessions)


def log_positive(values: pd.Series) -> pd.Series:
    return np.log(values.where(values.gt(0) & np.isfinite(values)))


def build_inputs(sessions, curve: pd.DataFrame, vx: pd.DataFrame, indices: dict,
                 returns: dict, *, history_start: str) -> pd.DataFrame:
    """对齐完整股票日历；在计算任何 lag/rolling 之前截断研究起点。"""
    sessions = pd.Index(sessions, name="as_of_session")
    if sessions.has_duplicates or not sessions.is_monotonic_increasing:
        raise ValueError("股票交易日必须唯一且递增")
    sessions = sessions[sessions >= history_start]
    if sessions.empty:
        raise ValueError("研究边界内没有交易日")
    inputs = pd.DataFrame(index=sessions)
    front = aligned(curve, sessions)
    for n in (1, 2, 3):
        inputs[f"F{n}"] = front[f"f{n}_settle"]
        inputs[f"f{n}_contract"] = front[f"f{n}_contract"]
        inputs[f"f{n}_expiration"] = front[f"f{n}_expiration"]
        inputs[f"tau{n}"] = (pd.to_datetime(front[f"f{n}_expiration"]) - pd.to_datetime(sessions)).dt.days
    present = inputs[["tau1", "tau2", "tau3"]].notna().all(axis=1)
    valid = inputs.tau1.gt(0) & inputs.tau2.gt(inputs.tau1) & inputs.tau3.gt(inputs.tau2)
    if (present & ~valid).any():
        raise ValueError("VX 前三月度合约必须尚未到期且到期日严格递增")
    vx = vx.loc[vx.as_of_session.between(sessions[0], sessions[-1])].copy()
    if vx.duplicated(["contract_id", "as_of_session"]).any():
        raise ValueError("逐合约同日结算存在重复记录")
    if "duration_type" in vx and not vx.duration_type.eq("M").all():
        raise ValueError("P3 只使用标准月度 VX")
    lookup = vx.set_index(["contract_id", "as_of_session"])
    for n in (1, 2, 3):
        selected = lookup.reindex(pd.MultiIndex.from_arrays([inputs[f"f{n}_contract"], sessions]))
        for col in ("raw_file", "raw_sha256", "source", "retrieved_at", "published_at"):
            if col in selected:
                inputs[f"f{n}_{col}"] = selected[col].to_numpy()
    inputs["lag5_session"] = pd.Series(sessions, index=sessions).shift(5)
    for prefix, dates in [("a04_current", sessions), ("a04_lag5", inputs.lag5_session)]:
        keys = pd.MultiIndex.from_arrays([inputs.f1_contract, dates])
        selected = lookup.reindex(keys)
        inputs[f"{prefix}_settle"] = selected.value.to_numpy()
        for col in ("raw_file", "raw_sha256", "source", "retrieved_at", "published_at"):
            if col in selected:
                inputs[f"{prefix}_{col}"] = selected[col].to_numpy()
    both = inputs[["F1", "a04_current_settle"]].notna().all(axis=1)
    if not np.allclose(inputs.loc[both, "F1"], inputs.loc[both, "a04_current_settle"], rtol=0, atol=1e-12):
        raise ValueError("F1 与同日同一真实合约结算不一致")
    for symbol in ("VIX", "VVIX", "VIX9D", "VIX3M", "SKEW"):
        frame = aligned(indices[symbol], sessions)
        inputs[symbol] = frame.value
        for col in ("raw_file", "raw_sha256", "source", "retrieved_at", "published_at"):
            if col in frame:
                inputs[f"{symbol}_{col}"] = frame[col]
    for symbol in ("SPY", "SVXY"):
        frame = aligned(returns[symbol], sessions)
        inputs[f"TR_{symbol}"] = frame.total_return_index
        # 不使用供应商复权价格或 P2 的前置首日回报补热。
        inputs[f"r_{symbol}"] = log_positive(frame.total_return_index / frame.total_return_index.shift(1))
        for col in ("raw_file", "raw_sha256", "source", "retrieved_at", "published_at"):
            if col in frame:
                inputs[f"{symbol}_{col}"] = frame[col]
    return inputs


def calculate_features(inputs: pd.DataFrame) -> pd.DataFrame:
    """原始单位，与 FEATURES.json 一一对应；窗口缺失不填补。"""
    x = inputs
    f = pd.DataFrame(index=x.index)
    f["A01"] = 30 * log_positive(x.F2 / x.F1) / (x.tau2 - x.tau1).where(x.tau2.gt(x.tau1))
    f["A02"] = 30 * log_positive(x.F3 / x.F2) / (x.tau3 - x.tau2).where(x.tau3.gt(x.tau2)) - f.A01
    f["A03"] = log_positive(x.F1 / x.VIX)
    f["A04"] = log_positive(x.a04_current_settle / x.a04_lag5_settle)
    f["A05"] = x.tau1.where(x.tau1.gt(0))
    f["B01"] = log_positive(x.VIX)
    f["B02"] = log_positive(x.VIX9D / x.VIX)
    f["B03"] = log_positive(x.VIX / x.VIX3M)
    f["C01"] = log_positive(x.VVIX)
    f["C02"] = log_positive(x.VVIX / x.VVIX.shift(5))
    f["C03"] = log_positive(x.VIX / x.VIX.shift(5))
    spy_sq = x.r_SPY.pow(2)
    f["D01"] = np.sqrt(252 * spy_sq.rolling(5, min_periods=5).mean())
    f["D02"] = np.sqrt(252 * spy_sq.rolling(21, min_periods=21).mean())
    denom = spy_sq.rolling(21, min_periods=21).sum()
    f["D03"] = x.r_SPY.clip(upper=0).pow(2).rolling(21, min_periods=21).sum() / denom.where(denom.gt(0))
    f["E01"] = (x.VIX / 100).pow(2) - f.D02.pow(2)
    f["F01"] = log_positive(x.TR_SPY / x.TR_SPY.shift(5))
    f["F02"] = x.TR_SPY / x.TR_SPY.rolling(21, min_periods=21).max() - 1
    f["F03"] = log_positive(x.TR_SVXY / x.TR_SVXY.shift(5))
    f["F04"] = np.sqrt(252 * x.r_SVXY.pow(2).rolling(21, min_periods=21).mean())
    f["G01"] = x.SKEW
    f["G02"] = x.SKEW - x.SKEW.shift(5)
    return f[CORE_IDS + EXTENSION_IDS].replace([np.inf, -np.inf], np.nan)


def missing_reasons(features: pd.DataFrame, inputs: pd.DataFrame) -> pd.DataFrame:
    reasons = pd.DataFrame("", index=features.index, columns=features.columns)
    position = np.arange(len(features))
    denom = inputs.r_SPY.pow(2).rolling(21, min_periods=21).sum()
    for key in features:
        missing = features[key].isna()
        reasons.loc[missing, key] = "MISSING_OR_INVALID_INPUT"
        reasons.loc[missing & (position < LOOKBACK[key]), key] = "INSUFFICIENT_POST_2019_HISTORY"
    reasons.loc[features.D03.isna() & denom.eq(0), "D03"] = "ZERO_VARIANCE_DENOMINATOR"
    return reasons


def feature_availability(features: pd.DataFrame, clock: pd.DataFrame) -> pd.DataFrame:
    c = aligned(clock, features.index)
    dates = features.index.to_list()
    info = pd.to_datetime(c.information_close_at, utc=True)
    decision = pd.to_datetime(c.decision_at, utc=True)
    execution = pd.to_datetime(c.execution_at, utc=True)
    if (c.decision_session.iloc[:-1].to_list() != dates[1:] or
            not c.decision_session.equals(c.execution_session) or
            not (decision > info).all() or not (execution - decision).eq(pd.Timedelta(hours=1)).all() or
            not (c.decision_session > pd.Series(dates, index=features.index)).all()):
        raise ValueError("特征必须在 t+1 收盘前60分钟可用，不能使用同日收盘决策或错位日历")
    out = c[["information_close_at", "decision_session", "decision_at", "decision_at_new_york",
             "execution_session", "execution_at"]].copy()
    out["latest_input_session"] = features.index
    out["assumed_available_at"] = c.decision_at
    out["published_at"] = ""
    out["availability_policy"] = "ASSUMED_NEXT_SESSION"
    out["pit_status"] = "NO_FULL_HISTORICAL_PIT_CLAIM"
    out["core_complete"] = features[CORE_IDS].notna().all(axis=1)
    out["extension_complete"] = features[EXTENSION_IDS].notna().all(axis=1)
    out["core_plus_extension_complete"] = out.core_complete & out.extension_complete
    return out


def features_at_cutoff(features: pd.DataFrame, availability: pd.DataFrame, cutoff,
                       *, include_extension: bool = False) -> pd.DataFrame:
    """为后续阶段保留明确的时间门槛；此处没有模型或训练。"""
    cutoff = pd.Timestamp(cutoff)
    if cutoff.tzinfo is None:
        raise ValueError("决策截止必须带时区")
    a = availability.reindex(features.index)
    complete = "core_plus_extension_complete" if include_extension else "core_complete"
    mask = a[complete] & (pd.to_datetime(a.assumed_available_at, utc=True) <= cutoff)
    columns = CORE_IDS + EXTENSION_IDS if include_extension else CORE_IDS
    return features.loc[mask, columns].copy()
