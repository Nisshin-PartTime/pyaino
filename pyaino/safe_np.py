# safe_np.py
# 20260523 A.Inoue

# ================================================================
# safe_np 使用ガイドライン（最新版・NumPy 問題対応版）
# ================================================================
#
# 本モジュールは、NumPy / CuPy 間の差異や版数依存、
# さらに NumPy が HDArray を壊す問題を吸収し、
# 実装の安定性を高めるための「互換レイヤ」である。
#
# safe_np は自動微分のための必須要素ではなく、
# あくまで「NumPy 側の危険箇所を安全化するための補助レイヤ」である。
#
# ------------------------------------------------
# 【基本方針】
# ------------------------------------------------
# - safe_np は「NumPy 側で問題が起きる箇所に限定して使用する」
# - CuPy 側は既存の safe_cp をそのまま使用する（今回問題なし）
# - HDArray → ndarray 変換は np.asarray() によるゼロコピーを基本とする
# - すべての np 呼び出しを置き換えることは目的としない
#
# ------------------------------------------------
# 【使用すべき箇所（必須）】
# ------------------------------------------------
# NumPy 側で HDArray が壊れる可能性が高い関数は必ず safe_np を使う。
#
# ● 軸を扱う高レベル関数（NumPy が HDArray を壊す代表例）
#   sum, mean, max, min, std, var, prod,
#   argmax, argmin, cumsum, cumprod,
#   any, all, einsum
#
# ● 比較演算後の集約（最も壊れやすい）
#     y = x == a
#     np.sum(y)   ← safe_np.sum に置換必須
#
# ● ブールマスク関連
#     np.sum(x > 0)
#     np.any(x < 0)
#
# ● 結合・分割・形状操作で HDArray が混ざる場合
#   concatenate, stack, hstack, vstack, dstack
#   split, tile, repeat, take, put
#   reshape, broadcast_to, moveaxis, transpose
#   expand_dims, squeeze
#
# ------------------------------------------------
# 【使用が望ましい箇所（任意）】
# ------------------------------------------------
# - reshape, transpose, broadcast_to, expand_dims などの基本操作
# - concatenate / split / where / reduction 系など頻出処理
# → 統一的に扱いたい場合のみ safe_np を使用する
#
# ------------------------------------------------
# 【使用しなくてよい箇所】
# ------------------------------------------------
# - 明確に NumPy / CuPy 固有処理を行う場合
# - 外部ライブラリ制約により直接呼び出しが必要な場合
# - 挙動が完全に安定しており、ラップの必要がない場合
#
# ● CuPy 側の関数（今回問題なし）
#   einsum, sort, argsort, where, clip,
#   expand_dims, squeeze, triu_indices, tril_indices
#
# ------------------------------------------------
# 【設計原則】
# ------------------------------------------------
# - safe_np は「すべてを包む層」ではなく、
#     NumPy が HDArray を壊す箇所だけを吸収する層とする
#
# - HDArray → ndarray 変換は np.asarray() によるゼロコピーを徹底する
#     → 計算グラフ情報（creator, generation 等）を持ち込まない
#     → NumPy の __array_function__ 問題を完全に回避
#
# - 問題が起きた際に、
#     backend（NumPy/CuPy） / safe_np / 上位層
#   のどこが原因か切り分けやすく保つ
#
# - 必要最小限にとどめることで、
#     保守性と可読性のバランスを維持する
#
# ------------------------------------------------
# 【transpose の扱い】
# ------------------------------------------------
# - x.T はそのまま使用してよい（NumPy / CuPy 間で安定）
# - np.transpose(x) は x.T に置き換える（axes 指定なしの場合）
# - 軸指定あり transpose は safe_np 経由を推奨
#     snp.transpose(x, axes=...)
#
# ------------------------------------------------
# 【reduction 系（sum, mean など）】
# ------------------------------------------------
# - NumPy 側で HDArray を壊すため、safe_np で統一することが必須
# - CuPy 側は既存の safe_cp をそのまま使用する（問題なし）
#
# ================================================================

from pyaino import Config
np = Config.np
from pyaino.nucleus import HDArray

if np.__name__ == "cupy":
    import cupy as _cupy
else:
    _cupy = None

try:
    import cupyx as _cupyx
except ImportError:
    _cupyx = None

import numpy as _numpy
import math as _math
from functools import lru_cache


"""
CuPy の高レベル API（cp.reshape, cp.transposeなど）は
内部で余計な引数を付けるので使いたくない
ndarray メソッド（gy.reshape(), gy.transpose()）は
pyaino の自動微分に捕まるので使えない
NumPy の API も使わない
だから CuPy の低レベル実装（core.core）を使う
NumPy のときは高レベル API (普通のもの np.reshape, np.transposeなど)
をそのまま使うが、逆にpyainoの自動微分を避ける（安全策）ため、ここで定義
（本来pyainoの自動微分は変数がHDArrayかどうかを見ているのでこれは安全策）

"""


def _resolve_attr(obj, path):
    """
    'a.b.c' のような属性パスを辿って取得する。
    見つからなければ None を返す。
    """
    for attr in path.split("."):
        obj = getattr(obj, attr, None)
        if obj is None:
            return None
    return obj


def _get_cupy12_modules():
    modules = []
    for path in ("_core.core", "core.core"):
        mod = _resolve_attr(_cupy, path)
        if mod is not None:
            modules.append(mod)
    return modules


def _get_cupy13_modules():
    modules = []
    core = getattr(_cupy, "_core", None)
    if core is not None:
        for attr in dir(core):
            if attr.startswith("_routines_"):
                mod = getattr(core, attr, None)
                if mod is not None:
                    modules.append(mod)
    return modules


@lru_cache(None)
def _load_cupy_func(name):
    """
    CuPy の内部関数を探索して返す（キャッシュ付き）。
    見つからなければ cupy.<name> を返す。

    - "reshape" → "reshape", "_reshape" の両方を探索
    - CuPy 12 / 13+ に対応
    """
    if _cupy is None:
        raise RuntimeError(
            f"_load_cupy_func('{name}') called but CuPy is not available")

    # 候補名
    candidate_names = (name, f"_{name}")

    # 探索対象モジュール
    modules = []
    modules.extend(_get_cupy12_modules())
    modules.extend(_get_cupy13_modules())

    # 探索
    for mod in modules:
        for cand in candidate_names:
            func = getattr(mod, cand, None)
            if callable(func):

                #print(f'###_load_cupy_func {func}')
                
                return func

    # フォールバック
    func = getattr(_cupy, name, None)
    if callable(func):
        return func

    raise AttributeError(f"{name} not found in CuPy internals or public API.")


# ------------------------------------------------------------
# CuPy の参照を import 時に固定
# ------------------------------------------------------------
if _cupy is not None:
    _cupy_ndarray = _cupy.ndarray

    _cupy_expand_dims = _cupy.expand_dims
    _cupy_asarray = _cupy.asarray
    _cupy_asnumpy = _cupy.asnumpy

    _cupy_sum = _cupy_ndarray.sum
    _cupy_mean = _cupy_ndarray.mean
    _cupy_std = _cupy_ndarray.std
    _cupy_var = _cupy_ndarray.var
    _cupy_prod = _cupy_ndarray.prod
    _cupy_max = _cupy_ndarray.max
    _cupy_min = _cupy_ndarray.min
    _cupy_argmax = _cupy_ndarray.argmax
    _cupy_argmin = _cupy_ndarray.argmin
    _cupy_cumsum = _cupy_ndarray.cumsum
    _cupy_cumprod = _cupy_ndarray.cumprod

    _cupy_reduction_methods = {
        "sum": _cupy_sum,
        "mean": _cupy_mean,
        "std": _cupy_std,
        "var": _cupy_var,
        "prod": _cupy_prod,
        "max": _cupy_max,
        "min": _cupy_min,
        "argmax": _cupy_argmax,
        "argmin": _cupy_argmin,
        "cumsum": _cupy_cumsum,
        "cumprod": _cupy_cumprod,
    }
else:
    _cupy_ndarray = None
    _cupy_expand_dims = None
    _cupy_asarray = None
    _cupy_asnumpy = None
    _cupy_reduction_methods = {}


def _cupy_reduction(a, method_name, axis, dtype, out, keepdims):
    """
    CuPy の reduction を ndarray メソッド経由で安全に実行する。

    safe_np では reduction 系の呼び出し形をある程度そろえるため、
    一部メソッドでも共通引数（axis, dtype, out, keepdims）を受ける。
    ただし CuPy ndarray メソッドで未対応の機能は以下のように扱う。

    - out はサポートしない（指定時は例外）
    - keepdims は必要に応じて safe_np 側で再現する
    - dtype を受けないメソッドでは無視または非使用とする

    """
    method = _cupy_reduction_methods.get(method_name, None)
    if method is None:
        raise AttributeError(f"CuPy ndarray method '{method_name}' is not available")

    # CuPy ndarray メソッドは out を受け取れない
    if out is not None:
        raise NotImplementedError(f"CuPy ndarray.{method_name} does not support 'out'")

    #print(f'###_cupy_reduction {method} {type(a)}')

    # ---- CuPy ndarray メソッド流儀で呼ぶ ----
    if method_name in ("sum", "mean", "std", "var", "prod"):
        # 元の実装と同様、位置引数で呼ぶ
        res = method(a, axis, dtype)

    elif method_name in ("max", "min"):
        # CuPy の max/min は dtype を受けない
        res = method(a, axis)

    elif method_name in ("argmax", "argmin"):
        # dtype も out も keepdims も元メソッドでは未対応
        res = method(a, axis)

    elif method_name in ("cumsum", "cumprod"):
        # axis, dtype は環境差が出やすいので分岐
        if axis is None and dtype is None:
            res = method(a)
        elif axis is None:
            res = method(a, None, dtype)
        elif dtype is None:
            res = method(a, axis)
        else:
            res = method(a, axis, dtype)

    else:
        raise ValueError(f"Unsupported reduction method: {method_name}")

    # --- keepdims の再現 ---
    if keepdims:
        if axis is None:
            # NumPy と同じ：全軸縮約 → (1,1,1,...)
            return res.reshape((1,) * a.ndim)
        else:
            # axis が単数か複数かを正規化
            axes = axis if isinstance(axis, tuple) else (axis,)
            axes = tuple(ax if ax >= 0 else ax + a.ndim for ax in axes)
            for ax in sorted(axes):
                res = _cupy_expand_dims(res, ax)

    return res


# ------------------------------------------------------------
# fallbackの用意
# ------------------------------------------------------------
def _fallback_add_at(x, indices, values):
    for idx, val in zip(indices, values):
        x[idx] += val
    return None  # x の破壊的更新


def _numpy_erf(x):
    """
    NumPy 配列/スカラーに対して math.erf をベクトル化して適用する。
    NumPy トップレベルに erf が無い環境でも動作するようにする。
    """
    vec_erf = _numpy.vectorize(_math.erf, otypes=[float])
    return vec_erf(x)


def _fallback_erf(x):
    """
    NumPy/CuPy 共通 fallback。
    CuPy 配列なら一度 CPU に戻してから math.erf ベースで処理し、
    最後に元の backend に戻す。
    """
    if _cupy is not None and isinstance(x, _cupy.ndarray):
        return _cupy_asarray(_numpy_erf(_cupy_asnumpy(x)))
    else:
        return _numpy_erf(x)

# ------------------------------------------------------------
# NumPy の場合の純粋ndarrayへの戻し
# ------------------------------------------------------------
def _to_numpy(x):
    """ HDArray をゼロコピーで ndarray に変換する """
    if isinstance(x, HDArray):
        return _numpy.asarray(x) # numpyならこれでndarrayに戻る
    return x

# ------------------------------------------------------------
# NumPy / CuPy の切り替え
# ------------------------------------------------------------
if np.__name__ == "cupy":
    reshape      = _load_cupy_func("reshape")
    broadcast_to = _load_cupy_func("broadcast_to")
    moveaxis     = _load_cupy_func("moveaxis")
    ascontiguousarray = _load_cupy_func("ascontiguousarray")
    transpose    = _load_cupy_func("transpose")

    # ここからは高レベル API を使う
    concatenate  = _cupy.concatenate
    stack        = _cupy.stack
    hstack       = _cupy.hstack
    vstack       = _cupy.vstack
    dstack       = _cupy.dstack
    split        = _cupy.split
    tile         = _cupy.tile
    repeat       = _cupy.repeat
    take         = _cupy.take
    put          = _cupy.put

    def sum(a, axis=None, dtype=None, out=None, keepdims=False):
        return _cupy_reduction(a, "sum", axis, dtype, out, keepdims)

    def mean(a, axis=None, dtype=None, out=None, keepdims=False):
        return _cupy_reduction(a, "mean", axis, dtype, out, keepdims)

    def std(a, axis=None, dtype=None, out=None, keepdims=False):
        return _cupy_reduction(a, "std", axis, dtype, out, keepdims)

    def var(a, axis=None, dtype=None, out=None, keepdims=False):
        return _cupy_reduction(a, "var", axis, dtype, out, keepdims)

    def prod(a, axis=None, dtype=None, out=None, keepdims=False):
        return _cupy_reduction(a, "prod", axis, dtype, out, keepdims)

    def max(a, axis=None, dtype=None, out=None, keepdims=False):
        return _cupy_reduction(a, "max", axis, dtype, out, keepdims)

    def min(a, axis=None, dtype=None, out=None, keepdims=False):
        return _cupy_reduction(a, "min", axis, dtype, out, keepdims)

    def argmax(a, axis=None, dtype=None, out=None, keepdims=False):
        return _cupy_reduction(a, "argmax", axis, dtype, out, keepdims)

    def argmin(a, axis=None, dtype=None, out=None, keepdims=False):
        return _cupy_reduction(a, "argmin", axis, dtype, out, keepdims)

    def cumsum(a, axis=None, dtype=None, out=None, keepdims=False):
        return _cupy_reduction(a, "cumsum", axis, dtype, out, keepdims)

    def cumprod(a, axis=None, dtype=None, out=None, keepdims=False):
        return _cupy_reduction(a, "cumprod", axis, dtype, out, keepdims)

    einsum       = _cupy.einsum

    sort         = _cupy.sort
    argsort      = _cupy.argsort

    where        = _cupy.where
    clip         = _cupy.clip
    expand_dims  = _cupy_expand_dims
    squeeze      = _cupy.squeeze

    triu_indices = _cupy.triu_indices
    tril_indices = _cupy.tril_indices

    # --------------------------------------------------------
    # add.at のフォールバック処理
    # --------------------------------------------------------
    if _cupyx is not None and hasattr(_cupyx, "scatter_add"):
        add_at = _cupyx.scatter_add
    elif _cupy is not None and hasattr(_cupy, "add") and hasattr(_cupy.add, "at"):
        add_at = _cupy.add.at
    else:
        add_at = _fallback_add_at

    if _cupyx is not None and hasattr(_cupyx, "scipy") and hasattr(_cupyx.scipy, "special"):
        erf = _cupyx.scipy.special.erf
    elif _cupy is not None and hasattr(_cupy, "erf"):
        # 将来版 or 一部環境対策
        erf = _cupy.erf
    else:
        erf = _fallback_erf

else: # numpy環境
    # ---------- 形状操作 ----------
    def reshape(x, *args, **kwargs):
        return _numpy.reshape(_to_numpy(x), *args, **kwargs)

    def broadcast_to(x, *args, **kwargs):
        return _numpy.broadcast_to(_to_numpy(x), *args, **kwargs)

    def moveaxis(x, *args, **kwargs):
        return _numpy.moveaxis(_to_numpy(x), *args, **kwargs)

    def ascontiguousarray(x, *args, **kwargs):
        return _numpy.ascontiguousarray(_to_numpy(x), *args, **kwargs)

    def transpose(x, *args, **kwargs):
        return _numpy.transpose(_to_numpy(x), *args, **kwargs)


    # ---------- 結合・分割 ----------
    def concatenate(xs, *args, **kwargs):
        xs = [_to_numpy(x) for x in xs]
        return _numpy.concatenate(xs, *args, **kwargs)

    def stack(xs, *args, **kwargs):
        xs = [_to_numpy(x) for x in xs]
        return _numpy.stack(xs, *args, **kwargs)

    def hstack(xs, *args, **kwargs):
        xs = [_to_numpy(x) for x in xs]
        return _numpy.hstack(xs, *args, **kwargs)

    def vstack(xs, *args, **kwargs):
        xs = [_to_numpy(x) for x in xs]
        return _numpy.vstack(xs, *args, **kwargs)

    def dstack(xs, *args, **kwargs):
        xs = [_to_numpy(x) for x in xs]
        return _numpy.dstack(xs, *args, **kwargs)

    def split(x, *args, **kwargs):
        return _numpy.split(_to_numpy(x), *args, **kwargs)

    def tile(x, *args, **kwargs):
        return _numpy.tile(_to_numpy(x), *args, **kwargs)

    def repeat(x, *args, **kwargs):
        return _numpy.repeat(_to_numpy(x), *args, **kwargs)

    def take(x, *args, **kwargs):
        return _numpy.take(_to_numpy(x), *args, **kwargs)

    def put(x, *args, **kwargs):
        return _numpy.put(_to_numpy(x), *args, **kwargs)


    # ---------- 集約 ----------
    def sum(x, *args, **kwargs):
        return _numpy.sum(_to_numpy(x), *args, **kwargs)

    def mean(x, *args, **kwargs):
        return _numpy.mean(_to_numpy(x), *args, **kwargs)

    def std(x, *args, **kwargs):
        return _numpy.std(_to_numpy(x), *args, **kwargs)

    def var(x, *args, **kwargs):
        return _numpy.var(_to_numpy(x), *args, **kwargs)

    def prod(x, *args, **kwargs):
        return _numpy.prod(_to_numpy(x), *args, **kwargs)

    def max(x, *args, **kwargs):
        return _numpy.max(_to_numpy(x), *args, **kwargs)

    def min(x, *args, **kwargs):
        return _numpy.min(_to_numpy(x), *args, **kwargs)

    def argmax(x, *args, **kwargs):
        return _numpy.argmax(_to_numpy(x), *args, **kwargs)

    def argmin(x, *args, **kwargs):
        return _numpy.argmin(_to_numpy(x), *args, **kwargs)

    def cumsum(x, *args, **kwargs):
        return _numpy.cumsum(_to_numpy(x), *args, **kwargs)

    def cumprod(x, *args, **kwargs):
        return _numpy.cumprod(_to_numpy(x), *args, **kwargs)

    def einsum(*args, **kwargs):
        # einsum は複数引数を取るので、すべて変換する
        new_args = [_to_numpy(a) for a in args]
        return _numpy.einsum(*new_args, **kwargs)


    # ---------- ソート ----------
    def sort(x, *args, **kwargs):
        return _numpy.sort(_to_numpy(x), *args, **kwargs)

    def argsort(x, *args, **kwargs):
        return _numpy.argsort(_to_numpy(x), *args, **kwargs)


    # ---------- 条件 ----------
    def where(*args, **kwargs):
        new_args = [_to_numpy(a) for a in args]
        return _numpy.where(*new_args, **kwargs)

    def clip(x, *args, **kwargs):
        return _numpy.clip(_to_numpy(x), *args, **kwargs)

    def expand_dims(x, *args, **kwargs):
        return _numpy.expand_dims(_to_numpy(x), *args, **kwargs)

    def squeeze(x, *args, **kwargs):
        return _numpy.squeeze(_to_numpy(x), *args, **kwargs)


    # ---------- インデックス ----------
    def triu_indices(*args, **kwargs):
        return _numpy.triu_indices(*args, **kwargs)

    def tril_indices(*args, **kwargs):
        return _numpy.tril_indices(*args, **kwargs)


    # ---------- add.at ----------
    def add_at(x, *args, **kwargs):
        if np.__name__ == 'nlcpy': # numpyのadd.atはnlcpy配列を扱えない
            return np.add.at(x, *args, **kwargs)
        return _numpy.add.at(_to_numpy(x), *args, **kwargs)

    # ---------- erf ----------
    if hasattr(_numpy, "erf"):
        erf = _numpy.erf
    else:
        erf = _numpy_erf


