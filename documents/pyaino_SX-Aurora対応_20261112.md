# pyaino を NEC SX-Aurora TSUBASA（VE）で動かす

2026-11-12

pyaino は `Config.np` を NLCPy（NEC が提供する、numpy互換で VE 上で計算するライブラリ）に切り替えると、VE 上で動く。Qwen2.5-0.5B の推論・生成・LoRA 学習は CPU と同じ結果を出し、CPU（16スレッド）より速い。

## 使い方

```bash
# 1. 環境を作る(1回だけ)。Python 3.12 と g++ を使う
bash tools/sx_aurora/setup_nlcpy.sh ~/venv_ve

# 2. VEで実行する(PYAINO_NP=nlcpy を設定して起動する)
bash tools/sx_aurora/run_ve.sh ~/venv_ve experiments/qwen/chat.py
bash tools/sx_aurora/run_ve.sh ~/venv_ve pyaino/test/test_Qwen_numpy_only.py
```

- スクリプト内で切り替える場合は、ほかの pyaino モジュールを import する前に `set_np('nlcpy')` を呼ぶ。
- 使う VE は `VE_NODE_NUMBER=1` などで選ぶ。

## 環境の制約と対処（`tools/sx_aurora/`）

| 問題 | 対処 |
|---|---|
| システムの NLCPy（`nec-python36-nlcpy-ve`）は Python 3.6 用。pyaino は 3.11 以降の文法（`x[*idx]`）を使う | PyPI の NLCPy 3.1.0（Python 3.12 用）を仮想環境に入れる。numpy は 1 系（`numpy<2`）にする |
| PyPI の NLCPy は GCC 11 以降でビルドされていて、RHEL8 系の libstdc++（GCC 8）にない `GLIBCXX_3.4.29` を要求する。yum で入るパッケージにも、これを含む libstdc++ は無い | 要求されるのは `std::__throw_bad_array_new_length()` の1つだけ。`nlcpy_glibcxx_compat.py` が拡張モジュールのこの記号の版指定を外し、版要求を weak にする。記号そのものは `nlcpy_shim.cpp` から作る小さな共有ライブラリで補い、`run_ve.sh` が `LD_PRELOAD` する |

システムの libstdc++ には手を加えない。新しい libstdc++ がある環境では、互換処理は自動で省かれる。

## pyaino 本体の変更

### NLCPy への対応
| 場所 | 内容 |
|---|---|
| `Config.set_np` | `'nlcpy'` を受け付ける。環境変数 `PYAINO_NP=numpy\|cupy\|nlcpy` で既定を選べる |
| `Initializer` | `scale` を float にする（NLCPy は 0 次元配列の書式指定や、乱数の引数に配列を渡すことを受け付けない） |
| `safe_np.add_at` | NLCPy では `nlcpy.add.at` を使う（numpy の `add.at` は NLCPy の配列を扱えない） |
| `cf.select_category` | `argmax` に `keepdims` を使わない（NLCPy に無い） |
| `cf.c2n_dictionary` / `n2c_dictionary` / `load_parameters`、`MNIST`、`Qwen.load_lora` | NLCPy を cupy と同じ「デバイスの配列」として扱う（保存時に numpy に変換する） |

### 高速化
NLCPy の `matmul` は、3 次元以上の積をホストの numpy に回す。毎回 VE からホストへ行列を移して計算し、VE に戻すので、これが実行時間の大半を占めていた（16 トークンの生成で 3,689 回）。

- `safe_np.matmul` を追加した。numpy・cupy では `np.matmul` そのもので、NLCPy では次のように計算する。
    - `(…,m,k)@(k,n)` は、2 次元の `dot` を 1 回呼ぶだけにする（例：x(8,122,896)@w(896,4864) で 47ms → 2.2ms）。
    - ブロードキャスト軸（`[:, :, None]`）は行の側に畳み込んで、1 回の積にする。
    - バッチ付きの積は、バッチごとの 2 次元 `dot` に分ける。
- 線形層（`ScaleDotLinear`、`MatMulLinear`）、注意機構（`Neuron` の `np.matmul` 全 43 か所）、Qwen を `snp.matmul` に置き換えた。
- GQA の注意機構の逆伝播では、グループ方向の和を行列積の縮約に含めた。計算結果は同じで、積と和の 2 段階が 1 回の積で済む。CPU でも処理が減る。

## 結果（Qwen2.5-0.5B-Instruct, float32, VE Type 10C 1 枚）

| 項目 | CPU（16 スレッド） | VE（高速化前） | VE（高速化後） |
|---|---|---|---|
| logits（CPU との相対誤差） | 基準 | 4.9e-6 | 6.2e-6（最大確率のトークンは 100% 一致） |
| 生成 | 4.6 トークン/秒 | 0.82 トークン/秒 | 約 8 トークン/秒 |
| LoRA 学習 1 ステップ（976 トークン） | 約 18 秒 | 約 15.5 秒 | 約 3.9 秒 |
| 順伝播（40 トークン） | 0.44 秒 | 1.7 秒 | 0.14 秒 |

`experiments/qwen/run_experiment.py --mode lora`（24 ステップ、バッチ 8）を CPU と同じ条件で実行した結果は次のとおり。

| 実行環境 | 学習時間 | 学習後の eval 損失 | 一般 NLL |
|---|---|---|---|
| CPU（16 スレッド） | 440 秒 | 0.755 | 1.063 |
| VE 1 枚 | 82 秒（約 5.4 倍速） | 0.750 | 1.081 |

LoRA の A の初期値は乱数で、numpy と NLCPy では乱数の系列が違うため、数値は完全には一致しない。

検証の内容は次のとおり。
- `test_Qwen_numpy_only.py` は VE 上でも合格した（float64 の勾配検査、KV キャッシュ、LoRA の保存・読込・マージ、`cf.save/load_parameters`）。
- CPU では、HF transformers との照合（logits、生成、全パラメタと LoRA の勾配）と、既存の注意機構のテストの結果が変わらないことを確認した。
- `test_ContextualSelfAttention.py`、`test_attention_unit.py`、`test_attention_unit_scale_and_mask.py` は、変更前の main でも同じエラーで失敗する（テストが古い引数の形を使っている）。

## 残る課題
- 学習の残りの時間の大半は、注意機構のバッチごとの小さな `dot` と線形層の積で、VE の性能を使い切れていない。注意機構を 1 つのカーネル（NLCPy の JIT、または VE 用の C）にまとめれば、さらに速くなる余地がある。
- 生成は、1 トークンごとに pyaino の計算グラフを組む Python 側の手間（`nucleus` など）が目立ってきている。
- `Functions.Erf` など、cupy の場合だけ分岐している箇所のうち、Qwen が使わないものは NLCPy で未確認。
- bf16 は使っていない（VE の NLCPy は float32 / float64 が中心）。
