#!/bin/bash
# pyainoをSX-Aurora TSUBASA(VE)で使うためのPython環境を作る
#   bash tools/sx_aurora/setup_nlcpy.sh <venvの作成先>
# 前提: VEOS/AVEOなどVEの実行環境と、Python 3.12, g++ が入っていること
# 作成後は tools/sx_aurora/run_ve.sh <venv> script.py で実行する
set -e
VENV=${1:?usage: setup_nlcpy.sh <venv-dir>}
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PYTHON:-python3.12}
$PY -m venv "$VENV"
"$VENV/bin/pip" install -q "nlcpy==3.1.0" "numpy<2" matplotlib regex huggingface_hub
SITE=$("$VENV/bin/python" -c "import sysconfig; print(sysconfig.get_paths()['platlib'])")
# libstdc++が古い(GLIBCXX_3.4.29が無い)ときだけ互換処理をする
if ! strings "$(g++ -print-file-name=libstdc++.so.6)" | grep -q GLIBCXX_3.4.29; then
    g++ -O2 -shared -fPIC -o "$VENV/lib/libnlcpy_shim.so" "$HERE/nlcpy_shim.cpp"
    "$VENV/bin/python" "$HERE/nlcpy_glibcxx_compat.py" "$SITE/nlcpy"
fi
LD_PRELOAD=$( [ -f "$VENV/lib/libnlcpy_shim.so" ] && echo "$VENV/lib/libnlcpy_shim.so" ) \
    "$VENV/bin/python" -c "import nlcpy; a = nlcpy.arange(4.0); print('NLCPy', nlcpy.__version__, 'OK on', nlcpy.venode.VE(), float((a @ a)))"
