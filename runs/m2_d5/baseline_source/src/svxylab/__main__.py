"""唯一项目命令入口：.venv/bin/python -m svxylab。"""

import argparse
from pathlib import Path

from svxylab.environment import build_environment_report


def main() -> int:
    parser = argparse.ArgumentParser(description="SVXYLab 本地研究程序（P0 环境至 P7 交付）")
    parser.add_argument("command", choices=["environment", "data", "baselines", "features", "predictions", "economics", "diagnostics", "research", "daily", "capital", "mechanism"], help="生成当前阶段的本地研究报告")
    parser.add_argument("--open", action="store_true", help="生成后用 macOS 默认应用打开报告")
    parser.add_argument("--pilot", action="store_true", help="P4：只试运行首个符合条件的历史月")
    parser.add_argument("--review", action="store_true", help="P4：按原参数完整复算、追加审查诊断并导出本地离线包")
    args = parser.parse_args()
    if args.pilot and args.command != "predictions":
        parser.error("--pilot 仅用于 predictions")
    if args.review and (args.command != "predictions" or args.pilot):
        parser.error("--review 仅用于完整 predictions，不能与 --pilot 同用")
    try:
        if args.command == "mechanism":
            from svxylab.mechanism_report import build_mechanism_report
            return build_mechanism_report(Path.cwd(), open_report=args.open)
        if args.command == "capital":
            from svxylab.capital_report import build_capital_report
            return build_capital_report(Path.cwd(), open_report=args.open)
        if args.command == "research":
            from svxylab.release_report import build_release
            return build_release(Path.cwd(), open_report=args.open)
        if args.command == "daily":
            from svxylab.daily import update_daily
            return update_daily(Path.cwd(), open_report=args.open)
        if args.command == "diagnostics":
            from svxylab.diagnostics_report import build_diagnostics_report
            return build_diagnostics_report(Path.cwd(), open_report=args.open)
        if args.command == "economics":
            from svxylab.economics_report import build_economics_report
            return build_economics_report(Path.cwd(), open_report=args.open)
        if args.command == "predictions":
            if args.review:
                from svxylab.prediction_review import build_review
                return build_review(Path.cwd(), open_report=args.open)
            from svxylab.predictions_report import build_predictions_report
            return build_predictions_report(Path.cwd(), open_report=args.open, pilot=args.pilot)
        if args.command == "features":
            from svxylab.features_report import build_features_report
            return build_features_report(Path.cwd(), open_report=args.open)
        if args.command == "baselines":
            from svxylab.baselines_report import build_baselines_report
            return build_baselines_report(Path.cwd(), open_report=args.open)
        if args.command == "data":
            from svxylab.data_report import build_data_report
            return build_data_report(Path.cwd(), open_report=args.open)
        return build_environment_report(Path.cwd(), open_report=args.open)
    except (OSError, ValueError) as error:
        parser.exit(1, f"本次命令未完成：{error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
