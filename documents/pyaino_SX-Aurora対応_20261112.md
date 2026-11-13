# pyaino を NEC SX-Aurora TSUBASA（VE）で動かす

2026-11-12

pyaino は `Config.np` を NLCPy（NEC が提供する、numpy互換で VE 上で計算するライブラリ）に切り替えると、VE 上で動く。Qwen2.5-0.5B の推論・生成・LoRA 学習は CPU と同じ結果を出し、CPU（16スレッド）より速い（生成は約 12 倍、LoRA 学習は約 9.6 倍）。

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

### 高速化（その 2）：VE の融合カーネルと推論専用の順伝播
`matmul` を直したあとは、次の 3 つが目立った。

- NLCPy の演算は、配列が小さくても 1 回あたり数十 μs かかる（RMS 正規化だけで約 160 μs）。生成では、1 トークンに数千回の小さな演算が並ぶ。
- 右辺が転置した配列の積（`gy @ w.T` など）は、連続な配列の積の約 3 倍かかる。
- 学習の注意機構は、`(B,T,h,H)` と `(B,h,T,H)` の間の転置のコピーと、バッチごとの小さな積が多い。

それぞれ次のように対処した。数値は従来の経路と一致する（float64 で差 1e-14 程度。`test_Qwen_numpy_only.py` で確認）。

| 変更 | 内容 |
|---|---|
| `pyaino/ve_kernels.py`（新規） | VE 用の C の関数を、NLCPy の JIT（`nlcpy.jit`、ncc）で初回にコンパイルして使う。中身は、残差の加算と RMS 正規化、q/k/v のバイアス・q/k の正規化（Qwen3）・RoPE・KV キャッシュへの書き込み、生成用の注意、SiLU×up、学習用の RoPE と注意（順伝播・逆伝播）。学習用の注意は、(バッチ, 頭) ごとの小さな行列積を NLC の BLAS（逐次版）で計算する。頭は行の間隔で選ぶので、転置は要らない。numpy・cupy では使わず、ncc や NLC が無ければ従来の経路に戻る |
| `Qwen.QwenInference`（新規） | 生成専用の順伝播。計算グラフ（`Function` の入出力の記録など）を作らず、重みを直接使って学習時と同じ式を計算する。NLCPy では q|k|v と gate|up の重みを結合し、上の融合カーネルを使う。KV キャッシュは転置して (B, kv頭, H, 長さ) で持つ。`generate(fast=True)`（既定）で使う |
| `generate` の選択 | 確定的なときは logits の argmax を直接とる（結果は同じ）。確率的なときや repetition penalty を使うときは、logits をホストに移してから処理する。選択の乱数には `Config.np` の乱数を使うので、`np.random.seed` で再現できる |
| `LinearLayerLoRA.weight_t`、`LinearLayerCrossEntropyTied.weight_t` | NLCPy で凍結中の重みは、連続な転置を作って保持し、逆伝播と出力層の積に使う（VE のメモリを約 2GB 余分に使う）。`merge_lora` や重みの差し替えで作り直す |
| `QwenSelfAttention` | NLCPy では学習時の注意を融合カーネルで計算する（`use_kernels=False` で `nn.AttentionUnit` に戻る） |
| `LinearLayerCrossEntropyTied`（出力層と交差エントロピー） | NLCPy で埋め込みが凍結中（LoRA）なら、融合カーネル `lm_ce` で計算する。VE の各スレッドが 32 行ずつ logits を BLAS で求め、その場で softmax と損失に変え、続けて x の勾配 (softmax − one-hot)·E まで計算する。勾配は順伝播のときに求めておき、逆伝播では gy を掛けるだけにする。logits の積が 2 回（順伝播と逆伝播の再計算）から 1 回に減り、語彙 15 万の配列に対する NLCPy の演算（max、exp、sum など）も無くなる。976 トークンで約 0.38 秒が約 0.15 秒になった。評価（順伝播だけ）のときも勾配まで計算するので、その分は遅くなる。`use_kernels=False` で従来のタイル計算に戻る |
| `grad_norm` | パラメタごとにホストへ値を取り出さず、デバイス上で足してから 1 回だけ取り出す |

## 結果（Qwen2.5-0.5B-Instruct, float32, VE Type 10C 1 枚）

| 項目 | CPU（16 スレッド） | VE（高速化前） | VE（matmul 対応後） | VE（融合カーネル後） |
|---|---|---|---|---|
| logits（CPU との相対誤差） | 基準 | 4.9e-6 | 6.2e-6 | 6.2e-6（最大確率のトークンは 100% 一致） |
| 生成（貪欲） | 4.6 → 約 6 トークン/秒 ※ | 0.82 トークン/秒 | 約 8 トークン/秒 | 約 70 トークン/秒 |
| 生成（top-k/top-p, repetition penalty） | – | – | – | 約 39 トークン/秒 |
| LoRA 学習 1 ステップ（976 トークン） | 約 18 秒 | 約 15.5 秒 | 約 3.9 秒 | 約 1.8 秒 → 約 1.7 秒 ※2 |
| 順伝播（40 トークン） | 0.44 秒 | 1.7 秒 | 0.14 秒 | 0.10 秒 |

※2 出力層と交差エントロピーの融合カーネルを加えたあと。

※ CPU でも `QwenInference`（重みの結合と融合カーネルは使わない）で生成が速くなった。生成は HF transformers の貪欲生成と 24/24 トークン一致する（Qwen2.5-0.5B、Qwen3-0.6B）。

生成の 1 トークンは約 14 ms で、内訳はおおむね、各層の 4 回の行列ベクトル積（qkv、o、gate|up、down）、融合カーネル 4 回、出力層の行列ベクトル積（約 1.3 ms）である。

`experiments/qwen/run_experiment.py --mode lora`（24 ステップ、バッチ 8）を CPU と同じ条件で実行した結果は次のとおり。

| 実行環境 | 学習時間 | 学習後の eval 損失 | 一般 NLL |
|---|---|---|---|
| CPU（16 スレッド） | 440 秒 | 0.755 | 1.063 |
| VE 1 枚（matmul 対応後） | 82 秒（約 5.4 倍速） | 0.750 | 1.081 |
| VE 1 枚（融合カーネル後） | 62 秒（約 7.1 倍速） | 0.750 | 1.081 |
| VE 1 枚（出力層と交差エントロピーのカーネル後） | 46 秒（約 9.6 倍速） | 0.750 | 1.081 |

LoRA の A の初期値は乱数で、numpy と NLCPy では乱数の系列が違うため、数値は完全には一致しない。

検証の内容は次のとおり。
- `test_Qwen_numpy_only.py` は VE 上でも合格した（float64 の勾配検査、KV キャッシュ、LoRA の保存・読込・マージ、`cf.save/load_parameters`）。
- CPU では、HF transformers との照合（logits、生成、全パラメタと LoRA の勾配）と、既存の注意機構のテストの結果が変わらないことを確認した。
- `test_ContextualSelfAttention.py`、`test_attention_unit.py`、`test_attention_unit_scale_and_mask.py` は、変更前の main でも同じエラーで失敗する（テストが古い引数の形を使っている）。

## 残る課題
- 学習 1 ステップ（約 1.7 秒）の内訳は、線形層の積（順・逆で約 0.85 秒、ほぼ行列積の性能どおり）、AdamT の更新（約 0.3 秒。パラメタごとの小さな演算）、出力層と交差エントロピー（約 0.15 秒）である。更新は融合カーネルにすれば縮められる。
- 埋め込みも学習する場合（全パラメタ学習）は、出力層の重みの勾配が要るので、融合カーネルを使わず従来のタイル計算で行う。
- 融合カーネルには、VE 用の C コンパイラ（ncc）と NLC が必要。無い環境では警告を出して、従来の配列演算の経路で動く。
- `Functions.Erf` など、cupy の場合だけ分岐している箇所のうち、Qwen が使わないものは NLCPy で未確認。
- bf16 は使っていない（VE の NLCPy は float32 / float64 が中心）。
