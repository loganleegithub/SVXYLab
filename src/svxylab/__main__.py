"""唯一项目命令入口：.venv/bin/python -m svxylab。"""

import argparse
from pathlib import Path

from svxylab.environment import build_environment_report


def main() -> int:
    parser = argparse.ArgumentParser(description="SVXYLab 本地研究程序（P0 环境 / P1 数据核验）")
    parser.add_argument("command", choices=["environment", "data"], help="生成环境报告或真实数据核对报告")
    parser.add_argument("--open", action="store_true", help="生成后用 macOS 默认应用打开报告")
    args = parser.parse_args()
    try:
        if args.command == "data":
            from svxylab.data_report import build_data_report
            return build_data_report(Path.cwd(), open_report=args.open)
        return build_environment_report(Path.cwd(), open_report=args.open)
    except (OSError, ValueError) as error:
        parser.exit(1, f"本次命令未完成：{error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
