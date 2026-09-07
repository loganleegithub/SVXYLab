#!/bin/zsh
cd -- "$(dirname -- "$0")" || exit 1
./.venv/bin/python -m svxylab capital --open
svxy_capital_exit=$?
if [[ -t 0 ]]; then
  print "资本与经济基准研究结束，退出码 $svxy_capital_exit。按回车关闭窗口。"
  read -r
fi
exit $svxy_capital_exit
