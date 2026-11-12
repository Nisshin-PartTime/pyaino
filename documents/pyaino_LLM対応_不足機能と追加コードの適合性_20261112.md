# pyaino：LLM（Qwen小型モデル）対応で見えた不足機能と、追加コードの適合性

作成日: 2026-11-12
対象: Qwen2.5-0.5B-Instruct / Qwen3-0.6B を pyaino で推論・学習・LoRA した作業（`pyaino/Qwen.py`, `pyaino/QwenTokenizer.py`）

各項目には、実際に作業中に確認した根拠（ファイル・挙動）を付けた。優先度は「LLMの学習・推論を pyaino 本体で素直に書けるか」という観点でつけている。

---

## 1. 現状の pyaino に足りない機能

### 優先度 高（LoRA・LLM学習を本体だけで書くのに必須）

| # | 不足機能 | 現状と根拠 | 影響 |
|---|---|---|---|
| 1 | **パラメタの凍結（trainable フラグ）** | `LinearLayer.__backward__` は常に `grad_w` を計算する（`Neuron.py` `LinearLayer`, `F.ScaleDotLinear`）。凍結の仕組みがない | LoRA では基底重みの勾配は不要。計算量は順伝播と同程度、メモリは重みと同量の無駄が出る |
| 2 | **LoRA などのアダプタ** | 該当する実装はない | 小型LLMのファインチューニングで標準的な手法が使えない |
| 3 | **勾配の蓄積（gradient accumulation）** | `WeightsAndBiases.set_gradient(flush=True)` が既定で上書きする。バッチを分けて足し込む流れが用意されていない | CPUで大きな実効バッチを作れない |
| 4 | **全体勾配ノルムでのクリッピング** | `GradientClipping` はパラメタごとに個別に掛かる（`Optimizers.py`） | LLM学習で一般的な global-norm clipping ができない |
| 5 | **位置オフセット付き RoPE と KV キャッシュ** | `RoPE.get_cos_sin` は位置が常に 0 から始まる。`AttentionUnit` の causality は `Tq==Tk` 前提。`BigramLanguageModel.generate` は1トークンごとにブロック全体を再計算する | 逐次生成が O(T²) になり、実用速度が出ない |
| 6 | **RoPE の設定を AttentionUnit から渡せない** | `AttentionUnit(rope=True)` は内部で `RoPE()` を作るため、base=10000・interleaved に固定される。Qwen/Llama は base=1e6・half（rotate_half）方式 | 既存の学習済みモデルを `rope=True` のまま使えない（今回は外側で `RotaryTransformHalf` を当てて回避） |
| 7 | **ignore_index 付きの省メモリ LM ヘッド損失** | `LinearLayerCrossEntropy` はタイル計算で省メモリだが、`t=-100` のような無視ラベルに対応していない。`grad_w` も常に計算する。tied embedding（埋め込みの転置）にも対応しない | SFT（プロンプト部分をマスク）が書けない |

### 優先度 中（正確さ・互換性）

| # | 不足機能・問題 | 現状と根拠 | 影響 |
|---|---|---|---|
| 8 | **Softmax の分母に `+1e-7`** | `Activators.Softmax` と `StatelessSoftmax` は `exp/(sum+1e-7)` を返すが、`__backward__` は厳密な softmax の導関数を使う | 順伝播と逆伝播が僅かに不整合になる。float64 の有限差分検査で勾配誤差 1.8e-5、KVキャッシュとの差 4e-7 を確認した。厳密版にすると誤差はそれぞれ 4e-8、1e-15 まで下がる。fp32 では実用上ほぼ無害 |
| 9 | **重みのみの RMSNorm（Llama/Qwen 型）** | `RMSNormalization(scale_and_bias=True)` は gamma と beta の両方を持つ。eps の既定は 1e-12（Qwen は 1e-6）、最適化関数の既定は AdaGrad | HF の重みとそのまま対応しない |
| 10 | **HF 形式の重みの読み込み（safetensors）** | 読み込み経路は `torch_bridge` だけで、torch が必須 | 「NumPy だけで動く」方針と矛盾する（今回は `Qwen.load_safetensors` を numpy だけで実装した） |
| 11 | **byte-level BPE トークナイザ（tokenizer.json）とチャット雛形** | `cf.Tokenizer` は文字・語単位で、BPE がない | 既存 LLM の語彙が使えない（今回は `QwenTokenizer` を実装し、HF と完全一致を確認した） |
| 12 | **標準的なサンプリング（top-k / top-p / repetition penalty）** | 既存の `generate` は `stochastic`, `beta` による確率選択のみ | 生成品質の調整手段が少ない |
| 13 | **パラメタの命名・保存の仕組み** | `cf.export_parameters` は呼び出し元の変数名（`who_am_i`）を起点にし、属性名 `w`, `b`, `gamma`... を再帰的に拾う | 外部モデル（HF の階層名）との対応付けが壊れやすい |

### 優先度 低（性能・使い勝手）

| # | 不足機能 | 根拠 | 影響 |
|---|---|---|---|
| 14 | **低精度（bf16/fp16）での保持と混合精度** | numpy に bf16 がない。fp32 では 0.5B モデルで約 2GB、全パラメタ学習（AdamT）では約 10GB 使った | メモリが約2倍になる |
| 15 | **activation checkpointing（再計算）** | なし | 長い系列での学習でメモリが不足する |
| 16 | **BLAS スレッド数の制御** | 40 コア環境では既定の 40 スレッドが最も遅かった（デコード 221ms/token）。16 スレッドで 153〜175ms | 性能のための設定が利用者任せになっている |
| 17 | **import 時の依存と出力** | `Neuron` → `common_function` が先頭で `matplotlib` を import する。`Config` は import のたびに十数行を表示する | 描画を使わない推論でも matplotlib が必須になり、ログも埋まる |
| 18 | **SFT 用のデータ整形** | パディング、ラベルのマスク、チャット雛形の適用がない | 今回は `build_sft_example`, `collate` で補った |
| 19 | **config 駆動のモデル構築** | `config.json` からモデルを組み立てる仕組みがない | 既存アーキテクチャの再現に毎回コードが必要になる |

---

## 2. 追加したコードは pyaino の書き方に合っているか

**結論（2026-11-12 改修後）: pyaino の部品を継承して組み立てる形に書き直し、パラメタ管理・更新・保存は pyaino の流儀に合わせた。pyaino 本体（Neuron.py 等）は変更していない。**

改修前は独自の `Param` クラスとモデル単位の一括更新で書いていた。これを次のように改めた。

### 構成（pyaino の部品との対応）

| Qwen.py のクラス | 継承元・再利用 | 役割 |
|---|---|---|
| `QwenBlock` | `sbh.TransformerBlock` | `__forward__` / `__backward__` / `update` を**そのまま使用**。`ln1, sa, ln2, ffwd` を差し替えるだけ |
| `QwenMLP` | `sbh.SwiGLU` | `linear_g, linear_i, linear_o, act(A.Swish)` の構成と `update` を継承 |
| `LinearLayerLoRA` | `nn.LinearLayer` | `frozen` と LoRA を追加。LoRA の A, B もそれぞれ `nn.LinearLayer` |
| `EmbeddingQwen` | `nn.Embedding` | `frozen` と出力層との重み共有（tie）を追加 |
| `RoPEOffset` | `nn.RoPE` | 位置オフセットを追加（KVキャッシュ用）。`mode='half'` で使用 |
| 注意計算 | `nn.AttentionUnit` | GQA・因果マスク・RoPE を学習時も逐次生成時もこれで計算 |
| `ExactSoftmax` | `A.Softmax` | 分母の `+1e-7` を除いた順伝播のみ上書き |
| `RMSNormalizationGamma` | `Function` | gamma のみの RMS 正規化（既存のものは beta も持つため） |
| `LmHeadQwen` | `sbh.LmHead` | `unify=True`（損失まで一体処理）の構成として `forward` / `backward` / `update` を**そのまま使用** |
| `LinearLayerCrossEntropyTied` | `Function` | 重み共有の出力層＋タイル分割の交差エントロピー（ignore対応） |
| `QwenLanguageModel` | `BigramLanguageModel` に倣う | `embed`, `blocks`（`nn.Sequential`）, `lm_head` と `forward(idx, targets, mask, dropout)` / `backward()` / `update(eta, **kwargs)` / `generate(seed, max_tokens, stochastic, beta, skip_ids, end_ids)`（seedを含む全長を指定し、seedと結合した配列を返す。選択は `cf.select_category`） |
| `QwenInference` | （pyaino の部品を使わない） | 生成専用の順伝播。計算グラフを作らず各層の重みを直接使い、学習時と同じ式を計算する（数値の一致はテストで確認）。NLCPy では VE の融合カーネル（`ve_kernels.py`）を使う |
| `QwenTokenizer` | `cf.Tokenizer` に倣う | `encode`（既定で `np.array`）/ `decode` / `vocab_size()` / `token2id` / `id2token` / `split_text_into_tokens` |

### pyaino の流儀に合わせた点
- パラメタは `WeightsAndBiases`（`w`, `b`, `grad_w`, `grad_b`）、`ParametersForEmbedding`、`gamma` / `ggamma` の属性で持つ。`cf.save_parameters` / `cf.load_parameters` でそのまま保存・復元できる（numpyのみのテストで確認済み）。
- 最適化関数は `optimize='AdamT', w_decay=...` を kwargs で各層へ渡し、各層が `update(eta, **kwargs)` を持つ。`g_clip` などの既存のクリップ指定もそのまま通る。
- 設定だけで乱数初期化のモデルを組める（`WeightsAndBiases` の遅延初期化を使う）。学習済み重みは `from_pretrained` → `import_hf_weights` で設定する。
- 言語モデルの入出力は pyaino の流儀に従う。`forward(x, t)` の `t` は次トークンで、損失を返す。`t` を与えなければ logits を返す。
- LoRAの保存・読込とsafetensorsの書き出しは、`cf.save_parameters` と同じ `cf.c2n_dictionary` / `n2c_dictionary` / `astype_dictionary` でnumpy/cupyとdtypeを変換する。
- 学習率の調整は独自関数を持たず、`Optimizers.LinearGrowCosineDecay` などの既存の関数を使う（`experiments/qwen/run_experiment.py`）。
- `print('Initialize', ...)` の表示、`snp`（safe_np）の使用、日本語コメント、`# モジュール名` + `# 日付` のヘッダ。

### pyaino にない機能として追加したもの（本体ではなく Qwen.py 内）
- `frozen`（凍結）、LoRA（`apply_lora`, `merge_lora`, `save_lora` / `load_lora`（pickle 辞書）, `export_peft`）
- 全体の勾配ノルムでのクリップ（`update(eta, max_grad_norm=...)`）
- KVキャッシュ（`KVCache`）と、`generate` の top-k / top-p / repetition penalty、生成専用の順伝播（`QwenInference`）
- safetensors の読み書き、HF 名との対応表（`parameter_table`, `hf_state_dict`）

### 残る相違・制約
- 高階微分（`create_graph=True`）には非対応。逆伝播は数値計算として書いている。
- safetensors の読み込みとトークナイザはホスト側（numpy / 純Python）で行う。CuPy での動作は未検証。NLCPy（SX-Aurora）では動作を確認済み（`documents/pyaino_SX-Aurora対応_20261112.md`）。
- `QwenSelfAttention` は `nn.SelfAttention`（q,k,v を一つの線形層で作る）を継承していない。Qwen の q, k, v はバイアスの有無、Qwen3 の q_norm / k_norm、LoRA を個別に掛けることが HF / PEFT と一致させるのに必要なため、別々の線形層で持っている。
- `cf.save_parameters` / `load_parameters` は呼び出し元の変数名を名前の起点にするので、保存と読込で同じ変数名（例: `model`）を使う必要がある（pyaino 既存の仕様）。

### 本体へ取り込む場合の提案
`LinearLayerLoRA` の `frozen` / LoRA、`EmbeddingQwen` の `frozen` / tie、`RoPEOffset` の `offset`、`ExactSoftmax` は、それぞれ `nn.LinearLayer`, `nn.Embedding`, `nn.RoPE`, `A.Softmax` に既定値で今の挙動を保つオプションとして入れられる。入れれば Qwen.py 側の継承クラスは不要になる。ただし既存モデルにも関わるので、作者の判断で進めるのがよい。
