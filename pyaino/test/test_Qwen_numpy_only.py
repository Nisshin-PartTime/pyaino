# pyaino.Qwen の numpy だけで完結する検査(torch不要, 数秒)
#   小さな乱数初期化のQwen2/Qwen3構成で
#   1) 解析的勾配 vs 有限差分 (全パラメタ, LoRA)
#   2) KVキャッシュ有り逐次計算 == キャッシュ無し一括計算
#      生成用の推論専用順伝播(QwenInference, 重み結合の有無, VEの融合カーネル)も同じ値か
#      (NLCPyのみ)学習用の融合カーネル(注意、出力層+交差エントロピー)が
#      AttentionUnit・タイル計算と同じ損失・勾配か(padding mask, ignoreあり)
#   3) LoRA保存→読込, merge の一致
#   4) cf.save_parameters / cf.load_parameters による全パラメタの保存・復元
# 乱数の重み・入力はsafetensorsと同じくホスト側のnumpyで作るので、
# Config.np が cupy や nlcpy でもそのまま実行できる
import os, tempfile
import numpy
from pyaino.Config import *
set_dtype('float64')                     # 有限差分のため倍精度
from pyaino.Qwen import QwenLanguageModel
from pyaino import common_function as cf

def tiny(model_type, seed=0):
    np.random.seed(seed)
    model = QwenLanguageModel(vocab_size=40, emb_dim=16, n_layer=2, n_head=(4, 2),
                              head_dim=8 if model_type == 'qwen3' else None,
                              intermediate=24, rope_base=10000.0,
                              qkv_bias=(model_type == 'qwen2'),
                              qk_norm=(model_type == 'qwen3'))
    model.forward(np.zeros((1, 2), dtype=int))   # 遅延初期化を済ませる
    # 全パラメタを再現可能な乱数で設定(gammaは1の周り、biasも非零)
    rs = numpy.random.RandomState(seed)
    sd = {k: (1.0 if k.endswith('norm.weight') else 0.0) + rs.randn(*v.shape) * 0.3
          for k, v in model.hf_state_dict().items()}
    model.import_hf_weights(sd)
    return model

def check_grads(model, x, t, names, eps=1e-5, n_probe=2):
    model.forward(x, t)
    model.backward()
    table = model.parameter_table()
    lora = model.lora_parameters()
    lora_g = model.lora_parameters(grad=True)
    worst = 0.0
    for n in names:
        if n in lora:
            p, g = lora[n], lora_g[n]
        else:
            obj, attr, gattr, _ = table[n]
            p, g = getattr(obj, attr), getattr(obj, gattr)
        g = g.copy()
        for _ in range(n_probe):
            idx = tuple(int(rng.integers(s)) for s in p.shape)
            old = float(p[idx])
            p[idx] = old + eps; lp = float(model.forward(x, t))
            p[idx] = old - eps; lm = float(model.forward(x, t))
            p[idx] = old
            num = (lp - lm) / (2 * eps)
            gi = float(g[idx])
            err = abs(num - gi) / max(abs(num) + abs(gi), 1e-8)
            worst = max(worst, err)
    return worst

rng = numpy.random.default_rng(0)
for mt in ('qwen2', 'qwen3'):
    model = tiny(mt)
    ids = np.asarray(rng.integers(0, 40, (2, 8)))
    x, t = ids[:, :-1], ids[:, 1:].copy()
    t[:, :2] = -100

    # 1) 全パラメタの勾配
    e_full = check_grads(model, x, t, list(model.parameter_table()))

    # 1') LoRAの勾配(Bを非零にして全経路を検査)
    model.apply_lora(r=2, alpha=4, seed=1)
    for k, v in model.lora_parameters().items():
        if k.endswith('lora_B'):
            v[...] = np.asarray(rng.standard_normal(v.shape) * 0.3)
    e_lora = check_grads(model, x, t, list(model.lora_parameters()))

    # 2) KVキャッシュ(複数トークンのprefill → offset付き複数トークン → 1トークン)
    z_full = model.forward(x)
    for blk in model.blocks.layers:
        from pyaino.Qwen import KVCache
        blk.sa.kv = KVCache()
    z1 = model.forward(x[:, :4]); z2 = model.forward(x[:, 4:6]); z3 = model.forward(x[:, 6:])
    for blk in model.blocks.layers:
        blk.sa.kv = None
    e_cache = float(np.abs(np.concatenate([z1, z2, z3], axis=1) - z_full).max())
    from pyaino.Qwen import QwenInference
    e_infer = 0.0
    for fuse, kernels in ((False, False), (True, False), (True, None)): # None: NLCPyなら融合カーネル
        inf = QwenInference(model, fuse=fuse, kernels=kernels)
        zs = [inf.forward(x[:, :4]), inf.forward(x[:, 4:6]), inf.forward(x[:, 6:7])]
        for T, z in zip((4, 6, 7), zs):
            e_infer = max(e_infer, float(np.abs(z - z_full[:, T - 1]).max()))

    # 2') NLCPyの融合カーネル(学習用の注意, 出力層+交差エントロピー) == 従来の計算 (padding maskあり)
    e_kern = 0.0
    from pyaino import ve_kernels
    if ve_kernels.get_kernels(np, Config.dtype) is not None:
        mask = np.ones(x.shape); mask[1, -2:] = 0
        res = []
        for use in (True, False):
            for blk in model.blocks.layers:
                blk.sa.use_kernels = use
            model.lm_head.linear_layer.use_kernels = use
            loss = float(model.forward(x, t, mask=mask)); model.backward()
            res.append((loss, {k: v.copy() for k, v in model.lora_parameters(grad=True).items()}))
        (l1, g1), (l2, g2) = res
        e_kern = abs(l1 - l2) + max(float(np.abs(g1[k] - g2[k]).max()) for k in g1)

    with tempfile.TemporaryDirectory() as td:
        # 3) LoRA保存・読込・併合
        path = os.path.join(td, 'lora.pkl')
        model.save_lora(path)
        m2 = tiny(mt)
        m2.load_lora(path)
        e_load = float(np.abs(m2.forward(x) - z_full).max())
        m2.merge_lora()
        e_merge = float(np.abs(m2.forward(x) - z_full).max())

        # 4) pyainoの標準の保存・復元(LoRA併合済みのm2 -> 別の乱数初期化モデル)
        #    名前は呼び出し元の変数名が起点になるので、保存・読込で同じ変数名 m を使う
        m = m2; del m2
        cf.save_parameters(os.path.join(td, 'all.pkl'), m)
        m = tiny(mt, seed=123)
        cf.load_parameters(os.path.join(td, 'all.pkl'), m)
        e_pkl = float(np.abs(m.forward(x) - z_full).max())

    print(f'{mt}: grad(full) {e_full:.1e}  grad(LoRA) {e_lora:.1e}  cache {e_cache:.1e}  infer {e_infer:.1e}  VE-kernel {e_kern:.1e}  '
          f'lora-reload {e_load:.1e}  merge {e_merge:.1e}  save/load_parameters {e_pkl:.1e}')
    assert e_full < 1e-6 and e_lora < 1e-6, 'gradient check failed'
    assert e_cache < 1e-12 and e_infer < 1e-12 and e_kern < 1e-12 and e_load < 1e-12 and e_merge < 1e-10 and e_pkl < 1e-10
print('ALL OK')
