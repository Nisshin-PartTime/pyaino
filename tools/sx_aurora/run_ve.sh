#!/bin/bash
# pyainoのスクリプトをVE(NLCPy)で実行する
#   bash tools/sx_aurora/run_ve.sh <venv> script.py [args...]
# VE_NODE_NUMBER=1 などで使うVEを選べる
VENV=${1:?usage: run_ve.sh <venv> script.py [args...]}; shift
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
[ -f "$VENV/lib/libnlcpy_shim.so" ] && export LD_PRELOAD="$VENV/lib/libnlcpy_shim.so${LD_PRELOAD:+:$LD_PRELOAD}"
PYAINO_NP=nlcpy PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" exec "$VENV/bin/python" "$@"
