#!/bin/zsh
cd -- "$(dirname -- "$0")" || exit 1
./.venv/bin/python -m svxylab research --open
svxy_exit=$?
if [[ -t 0 ]]; then
  print "研究结束，退出码 $svxy_exit。按回车关闭窗口。"
  read -r
fi
exit $svxy_exit
