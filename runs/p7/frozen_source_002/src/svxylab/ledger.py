"""规格 9.1 的连续自融资账本；每日一个目标，不叠加标签交易。"""

import numpy as np
import pandas as pd


def rebalance(equity: float, stock_value: float, target: float, cost_rate: float) -> dict:
    if (not np.isfinite([equity, stock_value, target, cost_rate]).all() or equity <= 0
            or not 0 <= stock_value <= equity or not 0 <= target <= 1 or not 0 <= cost_rate < 1):
        raise ValueError("换仓输入不允许非有限值、杠杆、负仓位或无效成本")
    current = stock_value / equity
    after = (equity * (1 + cost_rate * current) / (1 + cost_rate * target) if target >= current
             else equity * (1 - cost_rate * current) / (1 - cost_rate * target))
    new_stock = target * after
    signed_trade = new_stock - stock_value
    fee = cost_rate * abs(signed_trade)
    return {"equity": after, "stock_value": new_stock, "cash": after - new_stock,
            "signed_trade": signed_trade, "trade_notional": abs(signed_trade), "cost": fee}


def baseline_targets(curve: pd.DataFrame, fixed_weights: list[float]) -> dict[str, pd.Series]:
    index = pd.Index(curve.as_of_session, name="as_of_session")
    targets = {"cash": pd.Series(0.0, index=index)}
    targets.update({f"fixed_{round(w * 100)}": pd.Series(float(w), index=index) for w in fixed_weights})
    valid = (np.isfinite(curve[["f1_settle", "f2_settle"]]).all(axis=1)
             & curve[["f1_settle", "f2_settle"]].gt(0).all(axis=1))
    targets["curve"] = pd.Series(np.where(valid, (curve.f2_settle > curve.f1_settle).astype(float), np.nan), index=index)
    return targets


def continuous_ledger(prices: pd.DataFrame, clock: pd.DataFrame, targets: pd.Series, *,
                      capital: float = 100000.0, cost_rate: float = .0005) -> pd.DataFrame:
    if prices.as_of_session.tolist() != clock.as_of_session.tolist():
        raise ValueError("账本价格与信息时钟日期不一致")
    if not targets.index.equals(pd.Index(prices.as_of_session, name="as_of_session")):
        raise ValueError("目标必须按信息日逐日对齐；不隐式填充")
    fields = prices[["close", "split_factor", "cash_dividend", "capital_gain_distribution"]].to_numpy(float)
    if (not np.isfinite(fields).all() or (fields[:, :2] <= 0).any()
            or (fields[:, 2:] < 0).any() or capital <= 0):
        raise ValueError("真实价格或公司行动不可缺失；不能跳过持有损益")
    if prices.as_of_session.duplicated().any():
        raise ValueError("账本不能有重复市场日期")
    finite_targets = targets.dropna().to_numpy(float)
    if not np.isfinite(finite_targets).all() or ((finite_targets < 0) | (finite_targets > 1)).any():
        raise ValueError("目标不得超出 [0,1]")
    equity, cash, shares = float(capital), float(capital), 0.0
    rows = []
    for i, price in prices.iterrows():
        opening_equity, old_cash, old_shares = equity, cash, shares
        old_stock_value = old_shares * prices.iloc[i - 1].close if i else 0.0
        old_weight = old_stock_value / opening_equity
        shares /= price.split_factor
        distribution = shares * (price.cash_dividend + price.capital_gain_distribution) if i else 0.0
        cash += distribution
        stock_value = shares * price.close
        price_pnl = stock_value - old_stock_value
        pre_equity = cash + stock_value
        pre_weight = stock_value / pre_equity
        # 只读取上一信息日的目标；当日信号尚不可用于当日收盘执行。
        target = targets.iloc[i - 1] if i else np.nan
        decision = clock.iloc[i - 1] if i else None
        if decision is not None and decision.execution_session != price.as_of_session:
            raise ValueError("执行日必须是信息日的下一真实股票交易日")
        if pd.isna(target):
            equity = pre_equity
            trade = {"signed_trade": 0.0, "trade_notional": 0.0, "cost": 0.0}
            reason = "起点现金；尚无上一交易日信息" if i == 0 else "信号缺失；保留份额和现金，继续记损益"
        else:
            trade = rebalance(pre_equity, stock_value, float(target), cost_rate)
            equity, cash = trade["equity"], trade["cash"]
            shares = trade["stock_value"] / price.close
            reason = "按上一信息日目标在本日收盘换仓"
        rows.append({
            "as_of_session": price.as_of_session,
            "signal_information_session": decision.as_of_session if decision is not None else None,
            "decision_at": decision.decision_at if decision is not None else None,
            "execution_at": decision.execution_at if decision is not None else None,
            "equity_start": opening_equity, "old_cash": old_cash, "old_shares": old_shares,
            "old_weight": old_weight, "close": price.close, "split_factor": price.split_factor,
            "shares_after_split_before_trade": old_shares / price.split_factor,
            "asset_return": price.simple_return,
            "price_pnl": price_pnl, "distribution_income": distribution,
            "cash_interest": 0.0, "holding_pnl": price_pnl + distribution,
            "equity_before_trade": pre_equity, "pre_trade_weight": pre_weight,
            "target_weight": target, "signal_missing": pd.isna(target),
            "signed_trade": trade["signed_trade"], "trade_notional": trade["trade_notional"],
            "turnover": trade["trade_notional"] / pre_equity, "cost": trade["cost"],
            "shares_end": shares, "cash_end": cash,
            "new_weight": shares * price.close / equity, "equity_end": equity,
            "portfolio_return": equity / opening_equity - 1, "reason": reason,
        })
    result = pd.DataFrame(rows)
    result["drawdown"] = result.equity_end / result.equity_end.cummax() - 1
    return result
