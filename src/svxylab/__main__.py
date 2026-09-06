"""唯一项目命令入口：.venv/bin/python -m svxylab environment。"""

import argparse
from pathlib import Path

from svxylab.environment import build_environment_report


def main() -> int:
    parser = argparse.ArgumentParser(description="SVXYLab 本地研究程序（当前仅 P0）")
    parser.add_argument("command", choices=["environment"], help="检查环境、运行 pytest 并生成环境报告")
    parser.add_argument("--open", action="store_true", help="生成后用 macOS 默认应用打开报告")
    args = parser.parse_args()
    try:
        return build_environment_report(Path.cwd(), open_report=args.open)
    except (OSError, ValueError) as error:
        parser.exit(1, f"P0 未完成：{error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
