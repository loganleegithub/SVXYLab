"""一次独立审计：从 P1 原输入按股数/现金和有符号成交额重建账户。

不调用生产 rebalance/continuous_ledger；Decimal 40 位，核对全部真实行。
"""

from datetime import datetime, timezone
from decimal import Decimal, localcontext
from pathlib import Path
import json
import tomllib

import pandas as pd

root = Path(__file__).resolve().parents[2]
record_path = sorted(root.glob("runs/p2/*/baselines_run.json"))[-1]
record = json.loads(record_path.read_text())
output = root / record["result"]["output_dir"]
config = tomllib.loads((root / "experiment.toml").read_text())
prices = pd.read_csv(root / "data/clean/SVXY_daily.csv")
curve = pd.read_csv(root / "data/clean/VX_front_three.csv")
audit = {}
with localcontext() as context:
    context.prec = 40
    rate = Decimal(str(config["portfolio"]["cost_one_way_bps_base"])) / 10000
    capital = Decimal(str(config["portfolio"]["research_capital"]))
    for name in record["result"]["metrics"]:
        actual = pd.read_csv(output / f"ledger_{name}.csv")
        cash, quantity = capital, Decimal(0)
        max_equity_error, max_fee_error = Decimal(0), Decimal(0)
        for i, price in prices.iterrows():
            close = Decimal(str(price.close))
            quantity /= Decimal(str(price.split_factor))
            distribution = Decimal(str(price.cash_dividend)) + Decimal(str(price.capital_gain_distribution))
            if i:
                cash += quantity * distribution
            stock = quantity * close
            before = stock + cash
            weight = None
            if i:
                if name == "cash":
                    weight = Decimal(0)
                elif name.startswith("fixed_"):
                    weight = Decimal(name.split("_")[1]) / 100
                elif pd.notna(curve.iloc[i - 1].f1_settle) and pd.notna(curve.iloc[i - 1].f2_settle):
                    weight = Decimal(int(curve.iloc[i - 1].f2_settle > curve.iloc[i - 1].f1_settle))
            fee = Decimal(0)
            if weight is not None:
                # 从 T = w * (E - c*abs(T)) - S 解有符号成交额，而非直接套 E_new。
                difference = weight * before - stock
                signed_trade = difference / (1 + weight * rate if difference >= 0 else 1 - weight * rate)
                fee = rate * abs(signed_trade)
                quantity += signed_trade / close
                cash -= signed_trade + fee
            equity = quantity * close + cash
            max_equity_error = max(max_equity_error, abs(equity - Decimal(str(actual.equity_end.iloc[i]))))
            max_fee_error = max(max_fee_error, abs(fee - Decimal(str(actual.cost.iloc[i]))))
        audit[name] = {"rows": len(actual), "max_equity_error_dollars": float(max_equity_error),
                       "max_cost_error_dollars": float(max_fee_error),
                       "passed": max_equity_error < Decimal("0.000001") and max_fee_error < Decimal("0.000001")}

result = {"stage": "P2", "checked_at": datetime.now(timezone.utc).isoformat(),
          "command": ".venv/bin/python runs/p2/independent_recalculation.py",
          "run_record": record_path.relative_to(root).as_posix(), "method": "Decimal 40-digit signed-trade equation from original P1 inputs",
          "audit": audit, "passed": all(row["passed"] for row in audit.values())}
(root / "runs/p2/independent_recalculation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
if not result["passed"]:
    raise SystemExit(1)
