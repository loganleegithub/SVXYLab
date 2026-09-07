"""P2 使用 P1 冻结文件计算，取数与拟合均不在此阶段调用。"""

from hashlib import sha256
import json
from pathlib import Path
import tomllib

import exchange_calendars as xcals
import numpy as np
import pandas as pd

from svxylab.ledger import baseline_targets, continuous_ledger
from svxylab.returns import total_return
from svxylab.timing import five_day_labels, session_clock

NAMES = {"cash": "现金", "fixed_25": "固定 25%", "fixed_50": "固定 50%",
         "fixed_75": "固定 75%", "fixed_100": "固定 100%", "curve": "F2>F1 持有，否则现金"}


def metrics(ledger: pd.DataFrame) -> dict:
    frame = ledger.loc[ledger.as_of_session <= "2023-12-31"]
    years = (pd.Timestamp(frame.as_of_session.iloc[-1]) - pd.Timestamp(frame.as_of_session.iloc[0])).days / 365.25
    return {"first": frame.as_of_session.iloc[0], "last": frame.as_of_session.iloc[-1], "rows": len(frame),
            "ending_equity": float(frame.equity_end.iloc[-1]),
            "total_return": float(frame.equity_end.iloc[-1] / frame.equity_start.iloc[0] - 1),
            "cagr": float((frame.equity_end.iloc[-1] / frame.equity_start.iloc[0]) ** (1 / years) - 1),
            "max_drawdown": float(frame.drawdown.min()), "worst_day": float(frame.portfolio_return.min()),
            "worst_five_days": float(frame.equity_end.pct_change(5, fill_method=None).min()),
            "mean_exposure": float(frame.old_weight.iloc[1:].mean()),
            "annual_turnover": float(frame.turnover.sum() / years), "cost_dollars": float(frame.cost.sum()),
            "trade_days": int((frame.trade_notional > 1e-8).sum())}


def verify_ledger(ledger: pd.DataFrame, capital: float, rate: float) -> dict:
    scale = max(capital, float(ledger.equity_end.max()))
    equity_error = float(np.abs(ledger.equity_end - (ledger.equity_start + ledger.holding_pnl - ledger.cost)).max())
    holdings_error = float(np.abs(ledger.equity_end - (ledger.cash_end + ledger.shares_end * ledger.close)).max())
    cash_error = float(np.abs(ledger.cash_end - (ledger.old_cash + ledger.distribution_income
                                                - ledger.signed_trade - ledger.cost)).max())
    cost_error = float(np.abs(ledger.cost - ledger.trade_notional * rate).max())
    available = ~ledger.signal_missing
    weight_error = float(np.abs(ledger.loc[available, "new_weight"] - ledger.loc[available, "target_weight"]).max())
    chronological = (pd.to_datetime(ledger.decision_at.iloc[1:], utc=True)
                     < pd.to_datetime(ledger.execution_at.iloc[1:], utc=True)).all()
    checks = {"equity_reconciles": equity_error <= 1e-10 * scale,
              "holdings_reconcile": holdings_error <= 1e-10 * scale,
              "cash_reconciles": cash_error <= 1e-10 * scale,
              "cost_reconciles": cost_error <= 1e-10 * scale,
              "post_fee_weights": weight_error < 1e-12, "decision_precedes_execution": bool(chronological),
              "no_leverage_or_negative_cash": bool((ledger.cash_end >= -1e-10 * scale).all()
                                                   and ledger.new_weight.between(-1e-12, 1 + 1e-12).all())}
    return {"checks": checks, "max_equity_error_dollars": equity_error,
            "max_holdings_error_dollars": holdings_error, "max_cash_error_dollars": cash_error,
            "max_cost_error_dollars": cost_error, "max_target_weight_error": weight_error}


def prepare_baselines(root: Path, output: Path) -> dict:
    config = tomllib.loads((root / "experiment.toml").read_text())
    frozen = json.loads((root / "runs/p1/freeze.json").read_text())
    accepted = json.loads((root / "runs/p1/reproducibility.json").read_text())
    if (not frozen["complete_core_available"] or config["data"]["feature_history_requested_start"] != frozen["requested_start"]
            or config["targets"]["horizon_sessions"] != 5 or config["portfolio"]["cash_annual_return_base"] != 0):
        raise ValueError("当前 P2 需要已冻结的完整 P1 数据、五日标签与零现金基准")
    clean_hashes = accepted["clean_sha256"]
    for path, expected in clean_hashes.items():
        if sha256((root / path).read_bytes()).hexdigest() != expected:
            raise ValueError(f"P1 已验收清洗原件改变：{path}；不自动重取或覆盖")
    manifest_path = root / "data/raw/downloads.jsonl"
    raw_records = [json.loads(line) for line in manifest_path.read_text().splitlines()]
    for record in raw_records:
        if sha256((root / record["raw_file"]).read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"P1 原始响应摘要不符：{record['raw_file']}")
    clean = root / "data/clean"
    original_schedule = pd.read_csv(clean / "equity_sessions.csv", index_col="as_of_session")
    sessions = original_schedule.index.tolist()
    future_end = (pd.Timestamp(frozen["requested_end"]) + pd.Timedelta(days=40)).date().isoformat()
    schedule = xcals.get_calendar("XNYS", start=frozen["requested_start"], end=future_end).schedule
    for field in ("open", "close"):
        current = schedule.loc[sessions[0]:sessions[-1], field]
        if current.index.strftime("%Y-%m-%d").tolist() != sessions or not np.array_equal(
                pd.to_datetime(current, utc=True).to_numpy(), pd.to_datetime(original_schedule[field], utc=True).to_numpy()):
            raise ValueError("实际交易日历与 P1 已核验日历不同")
    clock = session_clock(sessions, schedule)
    returns = {}
    for symbol in ("SVXY", "SPY"):
        prices = pd.read_csv(clean / f"{symbol}_daily.csv")
        if prices.as_of_session.tolist() != sessions:
            raise ValueError(f"{symbol} 必须覆盖冻结日历，不能跳过持有损益")
        returns[symbol] = total_return(prices)
    labels = five_day_labels(returns["SVXY"], clock, loss_threshold=config["targets"]["loss_event_threshold"])
    curve = pd.read_csv(clean / "VX_front_three.csv")
    if curve.as_of_session.tolist() != sessions:
        raise ValueError("VX 曲线与冻结交易日历不一致")
    targets = baseline_targets(curve, config["evaluation"]["fixed_weight_baselines"])
    capital = config["portfolio"]["research_capital"]
    rate = config["portfolio"]["cost_one_way_bps_base"] / 10000
    ledgers = {name: continuous_ledger(returns["SVXY"], clock, target, capital=capital, cost_rate=rate)
               for name, target in targets.items()}
    audit = {name: verify_ledger(ledger, capital, rate) for name, ledger in ledgers.items()}
    # 另一种零成本全仓表达：从首个执行收盘买入后，份额不因标签重叠而增加。
    gross = continuous_ledger(returns["SVXY"], clock, targets["fixed_100"], capital=capital, cost_rate=0)
    expected = capital * returns["SVXY"].total_return_index.iloc[1:] / returns["SVXY"].total_return_index.iloc[1]
    gross_difference = float(np.abs(gross.equity_end.iloc[1:].to_numpy() - expected.to_numpy()).max())
    all_valid = all(all(a["checks"].values()) for a in audit.values()) and gross_difference < capital * 1e-9
    if not all_valid:
        raise ValueError("真实连续账本复算未通过；必须先修复会计差异")
    output.mkdir(parents=True, exist_ok=True)
    clock.to_csv(output / "clock.csv", index=False)
    labels.to_csv(output / "labels.csv", index=False)
    for symbol, frame in returns.items():
        frame.to_csv(output / f"{symbol}_returns.csv", index=False)
    for name, ledger in ledgers.items():
        ledger.to_csv(output / f"ledger_{name}.csv", index=False)
    summaries = {name: metrics(frame) for name, frame in ledgers.items()}
    pd.DataFrame.from_dict(summaries, orient="index").rename_axis("baseline").to_csv(output / "metrics_2019_2023.csv")
    reference = {}
    for symbol, frame in returns.items():
        i = frame.reference_difference_bps.abs().idxmax()
        reference[symbol] = {"largest_difference_session": frame.loc[i, "as_of_session"],
                             "max_absolute_daily_difference_bps": float(abs(frame.loc[i, "reference_difference_bps"])),
                             "computed_return": float(frame.loc[i, "simple_return"]),
                             "provider_reference_return": float(frame.loc[i, "provider_reference_return"]),
                             "distribution_events": int(((frame.cash_dividend + frame.capital_gain_distribution) > 0).sum())}
    return {"sessions": sessions, "first": sessions[0], "last": sessions[-1], "capital": capital,
            "rate": rate, "metrics": summaries, "audit": audit,
            "labels_observed": int(labels.observed.sum()), "labels_pending": int((~labels.observed).sum()),
            "last_observed_information_session": labels.loc[labels.observed, "as_of_session"].iloc[-1],
            "zero_cost_full_holding_vs_total_return_max_error_dollars": gross_difference,
            "reference_return_comparison": reference, "p1_clean_sha256": clean_hashes,
            "raw_records_verified": len(raw_records), "raw_manifest_sha256": sha256(manifest_path.read_bytes()).hexdigest(),
            "calendar_version": xcals.__version__, "output_dir": output.relative_to(root).as_posix()}
