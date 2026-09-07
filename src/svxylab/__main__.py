"""唯一项目命令入口：.venv/bin/python -m svxylab。"""

import argparse
from pathlib import Path

from svxylab.environment import build_environment_report


def main() -> int:
    parser = argparse.ArgumentParser(description="SVXYLab 本地研究程序（P0 环境 / P1 数据 / P2 账本）")
    parser.add_argument("command", choices=["environment", "data", "baselines"], help="生成环境、数据或连续账本报告")
    parser.add_argument("--open", action="store_true", help="生成后用 macOS 默认应用打开报告")
    args = parser.parse_args()
    try:
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
