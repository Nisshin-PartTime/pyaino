# Qwen
# Qwen2/Qwen2.5/Qwen3(dense)系の小型LLMを pyaino で推論・学習・LoRAする
# 20261112
#
# 使い方
#   model = QwenLanguageModel.from_pretrained(model_dir) # HFのsafetensorsを直接読む
#   tok   = QwenTokenizer(model_dir)
#   model_dir には 'Qwen/Qwen2.5-0.5B-Instruct' のようなHFのリポジトリ名も渡せる
#   (その場合のみ huggingface_hub で取得。sklearn_datasets が sklearn に任せるのと同じ)
#   out   = model.generate(tok.encode(prompt), max_tokens=256) # seedを含む全長(pyainoの流儀)
#
#   model.apply_lora(r=8, alpha=16)              # 基底重みを凍結しLoRAを付加
#   loss = model.forward(x, t, dropout=0.05)     # tは次トークン(-100は無視), dropoutはLoRA経路
#   model.backward()
#   model.update(eta=2e-4, max_grad_norm=1.0)
#
#   cf.save_parameters('qwen.pkl', model) / cf.load_parameters('qwen.pkl', model) も使える
#
# 構成は pyaino の部品を継承して作っている
#   QwenBlock          <- stems_blocks_heads.TransformerBlock (順伝播・逆伝播・更新をそのまま使用)
#   QwenMLP            <- stems_blocks_heads.SwiGLU
#   LmHeadQwen         <- stems_blocks_heads.LmHead (順伝播・逆伝播・更新をそのまま使用)
#   LinearLayerLoRA    <- Neuron.LinearLayer  (凍結とLoRA。LoRAのA,BもLinearLayer)
#   EmbeddingQwen      <- Neuron.Embedding    (凍結と出力層との重み共有)
#   RoPEOffset         <- Neuron.RoPE         (位置オフセット。KVキャッシュでの逐次生成用)
#   注意計算は Neuron.AttentionUnit (GQA, 因果マスク, RoPE) をそのまま使う
#   生成の選択は cf.select_category、パラメタの numpy/cupy 変換は cf.c2n_dictionary 等、
#   学習率の調整は Optimizers の CosineDecay 等を使う
#   重みの向きは pyaino の規約 y = x @ w, w.shape = (入力, 出力)

import contextlib
import json
import math
import os
import pickle
import time
import warnings
import weakref

import numpy # ホスト側の処理(safetensorsの読み書き, 生成の選択)用

from pyaino.Config import *
from pyaino.nucleus import Function
from pyaino import Neuron as nn
from pyaino import Activators as A
from pyaino import Optimizers
from pyaino import stems_blocks_heads as sbh
from pyaino import common_function as cf
from pyaino import safe_np as snp
from pyaino import ve_kernels
from pyaino.QwenTokenizer import resolve_model_dir


#### safetensors の読み書き #########################################
_ST_DTYPES = {'F32': 'float32', 'F16': 'float16', 'F64': 'float64',
              'I64': 'int64', 'I32': 'int32', 'I16': 'int16', 'I8': 'int8',
              'U8': 'uint8', 'BOOL': 'bool'}

def load_safetensors(path, dtype=None):
    """ safetensorsファイル(またはそのディレクトリ)を {名前: numpy.ndarray} で読む """
    import numpy # ファイルの読み込みは常にホスト側(numpy)で行う
    if os.path.isdir(path):
        index = os.path.join(path, 'model.safetensors.index.json')
        if os.path.exists(index):
            with open(index) as f:
                files = sorted(set(json.load(f)['weight_map'].values()))
        else:
            files = sorted(f for f in os.listdir(path) if f.endswith('.safetensors'))
        tensors = {}
        for fn in files:
            tensors.update(load_safetensors(os.path.join(path, fn), dtype))
        return tensors

    with open(path, 'rb') as f:
        n = int.from_bytes(f.read(8), 'little')
        header = json.loads(f.read(n))
    mm = numpy.memmap(path, dtype=numpy.uint8, mode='r')
    base = 8 + n
    tensors = {}
    for name, info in header.items():
        if name == '__metadata__':
            continue
        s, e = info['data_offsets']
        raw = mm[base + s: base + e]
        if info['dtype'] == 'BF16': # bfloat16はfloat32の上位16bit
            a = (raw.view(numpy.uint16).astype(numpy.uint32) << 16).view(numpy.float32)
        else:
            a = raw.view(_ST_DTYPES[info['dtype']])
        a = a.reshape(info['shape'])
        if dtype is not None and a.dtype.kind == 'f':
            a = a.astype(dtype)
        else:
            a = numpy.array(a)
        tensors[name] = a
    return tensors

def save_safetensors(path, tensors, metadata=None):
    """ {名前: ndarray} をsafetensors形式で保存 """
    import numpy
    rev = {v: k for k, v in _ST_DTYPES.items()}
    tensors = cf.c2n_dictionary(dict(tensors)) # cupy配列ならnumpyへ
    header, chunks, offset = {}, [], 0
    for name in sorted(tensors):
        a = numpy.ascontiguousarray(tensors[name])
        b = a.tobytes()
        header[name] = {'dtype': rev[a.dtype.name], 'shape': list(a.shape),
                        'data_offsets': [offset, offset + len(b)]}
        chunks.append(b)
        offset += len(b)
    if metadata:
        header['__metadata__'] = {str(k): str(v) for k, v in metadata.items()}
    h = json.dumps(header).encode('utf-8')
    h += b' ' * ((8 - len(h) % 8) % 8)
    with open(path, 'wb') as f:
        f.write(len(h).to_bytes(8, 'little'))
        f.write(h)
        for b in chunks:
            f.write(b)

def _no_overflow_warning():
    """ SiLUのexp(-x)の桁あふれ警告を抑止(結果は正しく0になる) """
    return np.errstate(over='ignore') if hasattr(np, 'errstate') \
        else contextlib.nullcontext()


#### 凍結とLoRAに対応した線形層 #######################################
class LinearLayerLoRA(nn.LinearLayer):
    """
    y = x @ w + b + (DO(x) @ A) @ B * alpha/r

    frozen=True ならwとbの勾配を計算せず更新もしない(入力の勾配だけを返す)
    LoRAのA:(m,r), B:(r,n) はそれぞれバイアス無しのLinearLayerとして持つ
    forwardのdropoutはLoRA経路の入力だけに掛ける
    """
    def __init__(self, *configuration, frozen=False, **kwargs):
        kwargs['matmul'] = True
        super().__init__(*configuration, **kwargs)
        self.frozen = frozen
        self.lora_a = None
        self.lora_b = None
        self.lora_scale = 0.0
        self.lora_DO = nn.Dropout()
        self.lora_dropped = False
        self._wT = None # NLCPyで凍結中に保持する w の連続な転置 (元のw, w.T)

    def add_lora(self, r=8, alpha=16, **kwargs):
        """ LoRAを付加(Aは一様乱数 U(-1/√m, 1/√m)、Bは0で初期化) """
        m, n = self.config
        kwargs.setdefault('optimize', 'AdamT')
        kwargs.setdefault('w_decay', 0.0)
        self.lora_a = nn.LinearLayer(m, r, matmul=True, bias=False, **kwargs)
        self.lora_b = nn.LinearLayer(r, n, matmul=True, bias=False, **kwargs)
        bound = 1.0 / math.sqrt(m)
        self.lora_a.parameters.w = \
            np.random.uniform(-bound, bound, (m, r)).astype(Config.dtype)
        self.lora_b.parameters.w = np.zeros((r, n), dtype=Config.dtype)
        self.lora_scale = alpha / r

    def merge_lora(self):
        """ LoRAを基底の重みへ併合して取り除く """
        if self.lora_a is None:
            return
        w, b, gamma = self.parameters()
        w += self.lora_scale * np.matmul(self.lora_a.parameters.w, self.lora_b.parameters.w)
        self.lora_a = self.lora_b = None
        self._wT = None

    def weight_t(self):
        """
        wの転置。NLCPyでは右辺が転置した配列の積が遅い(連続な配列の約3倍)ので、
        凍結中は連続な転置を作って保持する(wが別の配列に置き換わったら作り直す。
        凍結中のwをその場で書き換えた場合は、merge_loraと同じく self._wT = None とすること)
        """
        w = self.parameters.w
        if not (self.frozen and np.__name__ == 'nlcpy'):
            return w.T
        if self._wT is None or self._wT[0] is not w:
            self._wT = (w, np.ascontiguousarray(w.T))
        return self._wT[1]

    def __forward__(self, x, *, dropout=0.0):
        y = super().__forward__(x)
        if self.lora_a is not None:
            self.lora_dropped = dropout > 0.0
            xd = self.lora_DO.forward(x, dropout=dropout) if self.lora_dropped else x
            u = self.lora_a.forward(xd)
            y += self.lora_b.forward(u) * self.lora_scale
        return y

    def __backward__(self, grad_y):
        if self.lora_a is not None:
            gu = self.lora_b.backward(grad_y * self.lora_scale)
            gxd = self.lora_a.backward(gu)
            if self.lora_dropped:
                gxd = self.lora_DO.backward(gxd)

        if self.frozen: # 入力の勾配だけ(重みの勾配は計算しない)
            grad_x = snp.matmul(grad_y, self.weight_t())
            self.dot_linear.x = self.dot_linear.y = None # 保持していた入出力を解放
        else:
            grad_x = super().__backward__(grad_y)

        if self.lora_a is not None:
            grad_x += gxd
        return grad_x

    def update(self, eta=0.001, **kwargs):
        if not self.frozen:
            super().update(eta=eta, **kwargs)
        if self.lora_a is not None:
            self.lora_a.update(eta=eta, **kwargs)
            self.lora_b.update(eta=eta, **kwargs)

    def gradients(self):
        """ 学習対象の(パラメタ, 勾配)の組 """
        if not self.frozen:
            yield self.parameters.w, self.parameters.grad_w
            if self.parameters.bias:
                yield self.parameters.b, self.parameters.grad_b
        if self.lora_a is not None:
            yield self.lora_a.parameters.w, self.lora_a.parameters.grad_w
            yield self.lora_b.parameters.w, self.lora_b.parameters.grad_w


#### 重みのみのRMS正規化(Llama/Qwen型) ###############################
class RMSNormalizationGamma(Function):
    """
    y = x / sqrt(mean(x^2) + eps) * gamma  (最終軸で正規化、betaは無い)
    Neuron.RMSNormalization(scale_and_bias=True)はbetaも持つため別に用意
    """
    def __init__(self, eps=1e-6, frozen=False, **kwargs):
        super().__init__()
        self.eps = eps
        self.frozen = frozen
        self.gamma, self.ggamma = None, None
        self.sigma = None
        optimize = kwargs.pop('optimize', 'AdamT')
        kwargs.pop('bias', None)
        self.OFg = cf.eval_in_module(optimize, Optimizers, bias=True, **kwargs) # w_decay対象外

    def init_parameters(self, shape):
        self.gamma = np.ones(shape[-1:], dtype=Config.dtype)

    def __forward__(self, x, **kwargs):
        if self.gamma is None:
            self.init_parameters(x.shape)
        meansquare = snp.mean(x * x, axis=-1, keepdims=True) # 二乗平均
        self.sigma = np.sqrt(meansquare + self.eps)
        self.z = x / self.sigma                              # 正規化出力
        return self.z * self.gamma

    def __backward__(self, gy):
        z = self.z
        if not self.frozen:
            self.ggamma = snp.sum(snp.reshape(gy * z, (-1, z.shape[-1])), axis=0)
        gz = gy * self.gamma
        gx = (gz - z * snp.mean(gz * z, axis=-1, keepdims=True)) / self.sigma
        self.z = None
        return gx

    def update(self, eta=0.001, **kwargs):
        if not self.frozen and self.ggamma is not None:
            self.OFg.update(self.gamma, self.ggamma, eta, **kwargs)

    def gradients(self):
        if not self.frozen:
            yield self.gamma, self.ggamma


#### 凍結と重み共有に対応したEmbedding ###############################
class EmbeddingQwen(nn.Embedding):
    """
    tied=True: 出力層(LinearLayerCrossEntropyTied)が先に同じ重みの勾配を設定するので、
    逆伝播ではそこへ加算する(flush=False)
    """
    def __init__(self, *configuration, frozen=False, tied=False, **kwargs):
        super().__init__(*configuration, **kwargs)
        self.frozen = frozen
        self.tied = tied

    def __backward__(self, gy):
        if self.frozen:
            return
        gx = gy if self.mask is None else gy * self.mask
        self.parameters.set_gradient(self.x, gx, flush=not self.tied)

    def update(self, eta=0.001, **kwargs):
        if not self.frozen:
            super().update(eta=eta, **kwargs)

    def gradients(self):
        if not self.frozen:
            yield self.parameters.w, self.parameters.grad_w


#### 位置オフセット付きRoPE ###########################################
class RoPEOffset(nn.RoPE):
    """ nn.RoPEの位置を offset から始める(KVキャッシュでの逐次生成用) """
    def __init__(self, base=10000.0, mode='half'):
        super().__init__(base=base, mode=mode)
        self.offset = 0

    def get_cos_sin(self, x):
        T = x.shape[-2]
        H = x.shape[-1]
        L = self.offset + T
        if self.inv_freq is None:
            if H % 2 != 0:
                raise ValueError(f"RoPE requires even head_dim, but got {H}")
            i = np.arange(H // 2, dtype=x.dtype)
            self.inv_freq = self.base ** (-2.0 * i / H)
        if self.cos_sin is None or len(self.cos_sin) < L:
            n = L if self.cos_sin is None else 2 * L       # 不足したら倍に拡張
            position = np.arange(n, dtype=x.dtype)
            theta = position[:, None] * self.inv_freq[None, :]
            self.cos_sin = np.stack((np.cos(theta), np.sin(theta)), axis=-1)
        cos = self.cos_sin[self.offset:L, :, 0]
        sin = self.cos_sin[self.offset:L, :, 1]
        shape = (1,) * (x.ndim - 2) + (T, H // 2)
        return cos.reshape(shape), sin.reshape(shape)


class ExactSoftmax(A.Softmax):
    """
    A.Softmaxは分母に1e-7を加えるため順伝播が厳密なsoftmaxから僅かにずれ、
    逆伝播(厳密なsoftmaxの導関数)と整合しない。これを除いたもの(逆伝播はA.Softmaxのまま)
    """
    def __forward__(self, x):
        x = x / self.temperature
        exp_a = np.exp(x - snp.max(x, axis=-1, keepdims=True))
        y = exp_a / snp.sum(exp_a, axis=-1, keepdims=True)
        self.y = y
        return y


#### KVキャッシュ ####################################################
class KVCache:
    """ 1層分のRoPE適用後のk,vを(B,T,C)で保持(容量は倍々で確保) """
    def __init__(self):
        self.k, self.v = None, None
        self.length = 0

    def append(self, k, v):
        L, T = self.length, k.shape[1]
        if self.k is None or self.k.shape[1] < L + T:
            cap = max(256, 2 * (L + T))
            nk = np.empty((k.shape[0], cap, k.shape[2]), dtype=k.dtype)
            nv = np.empty((v.shape[0], cap, v.shape[2]), dtype=v.dtype)
            if self.k is not None:
                nk[:, :L] = self.k[:, :L]
                nv[:, :L] = self.v[:, :L]
            self.k, self.v = nk, nv
        self.k[:, L:L + T] = k
        self.v[:, L:L + T] = v
        self.length = L + T
        return self.k[:, :L + T], self.v[:, :L + T]


#### 注意機構 ########################################################
class QwenSelfAttention(Function):
    """
    q,k,vを別々の線形層で作り(Qwen2はバイアス有り)、Qwen3ではq,kにRMS正規化を掛け、
    AttentionUnit(GQA, 因果マスク, half型RoPE)で注意を計算する
    self.kv にKVCacheを設定すると逐次生成モード(RoPEは位置オフセット付き)
    """
    def __init__(self, emb_dim=None, head_dim=None, n_head=1, rope_base=10000.0,
                 bias=True, qk_norm=False, eps=1e-6, **kwargs):
        super().__init__()
        if isinstance(n_head, int):
            n_head = n_head, n_head
        hq, hk = n_head
        if head_dim is None:
            head_dim = emb_dim // hq
        self.config = emb_dim, head_dim, n_head
        print('Initialize', self.__class__.__name__, self.config, kwargs)
        H = head_dim
        self.linear_q = LinearLayerLoRA(emb_dim, hq * H, bias=bias, **kwargs)
        self.linear_k = LinearLayerLoRA(emb_dim, hk * H, bias=bias, **kwargs)
        self.linear_v = LinearLayerLoRA(emb_dim, hk * H, bias=bias, **kwargs)
        self.linear_o = LinearLayerLoRA(hq * H, emb_dim, bias=False, **kwargs)
        if qk_norm:
            self.q_norm = RMSNormalizationGamma(eps=eps, **kwargs)
            self.k_norm = RMSNormalizationGamma(eps=eps, **kwargs)
        else:
            self.q_norm = self.k_norm = None

        # 一括処理(学習): RoPEもAttentionUnitの中で適用
        self.attention = nn.AttentionUnit(head=(hq, hk), causality=True)
        self.attention.rope = RoPEOffset(base=rope_base, mode='half')
        self.attention.softmax = ExactSoftmax()
        # 逐次生成: RoPEは外で適用してからキャッシュし、因果マスクはオフセット付き
        self.attention_cached = nn.AttentionUnit(head=(hq, hk), causality=True)
        self.attention_cached.softmax = ExactSoftmax()
        self.kv = None
        # NLCPyでは注意(RoPE含む)を融合カーネルで計算する(転置と小さな積の繰り返しを避ける)
        self.use_kernels = True
        self.kctx = None

    def __forward__(self, x, *, mask=None, dropout=0.0):
        emb_dim, H, (hq, hk) = self.config
        B, T, _ = x.shape
        q = self.linear_q.forward(x, dropout=dropout)
        k = self.linear_k.forward(x, dropout=dropout)
        v = self.linear_v.forward(x, dropout=dropout)
        if self.q_norm is not None: # 頭ごとに正規化
            q = self.q_norm.forward(q.reshape(B, T, hq, H)).reshape(B, T, hq * H)
            k = self.k_norm.forward(k.reshape(B, T, hk, H)).reshape(B, T, hk * H)

        if self.kv is not None:
            y = self.forward_cached(q, k, v)
        elif self.kernels() is not None:
            y = self.attention_kernels(q, k, v, mask)
        else:
            y = self.attention.forward(q, k, v, mask=mask)

        y = self.linear_o.forward(y, dropout=dropout)
        return y

    def forward_cached(self, q, k, v):
        """ 逐次生成: 位置offsetからRoPEを掛け、キャッシュと合わせて注意を計算 """
        emb_dim, H, (hq, hk) = self.config
        B, T, _ = q.shape
        offset = self.kv.length
        rope = self.attention.rope
        rope.offset = offset
        qh = q.reshape(B, T, hq, H).transpose(0, 2, 1, 3) # (B,hq,T,H)
        kh = k.reshape(B, T, hk, H).transpose(0, 2, 1, 3) # (B,hk,T,H)
        qh, kh = rope.forward(qh, kh)
        rope.offset = 0
        q = qh.transpose(0, 2, 1, 3).reshape(B, T, hq * H)
        k = kh.transpose(0, 2, 1, 3).reshape(B, T, hk * H)
        k, v = self.kv.append(k, v)
        Tk = k.shape[1]
        # 絶対位置 offset+i のqueryは位置 j<=offset+i のkeyに注意する
        tril = np.arange(Tk)[None, :] <= (np.arange(T)[:, None] + offset)
        self.attention_cached.tril = tril.astype(Config.dtype)[None, None]
        return self.attention_cached.forward(q, k, v)

    def kernels(self):
        if not self.use_kernels:
            return None
        return ve_kernels.get_kernels(np, Config.dtype)

    def attention_kernels(self, q, k, v, mask):
        """ AttentionUnit(GQA, 因果マスク, half型RoPE, 厳密なsoftmax)と同じ計算をVEの融合カーネルで """
        kern = self.kernels()
        emb_dim, H, (hq, hk) = self.config
        B, T, _ = q.shape
        cos, sin = self.attention.rope.get_cos_sin(np.empty((T, H), dtype=q.dtype))
        cos = np.ascontiguousarray(cos.reshape(T, -1))
        sin = np.ascontiguousarray(sin.reshape(T, -1))
        q, k, v = (np.ascontiguousarray(a) for a in (q, k, v))
        qr, kr = np.empty_like(q), np.empty_like(k)
        kern.rope(q, qr, cos, sin, hq)
        kern.rope(k, kr, cos, sin, hk)
        if mask is not None:
            mask = np.ascontiguousarray(np.asarray(mask).reshape(B, T).astype(q.dtype))
        P = np.empty((B, hq, T, T), dtype=q.dtype)
        y = np.empty_like(q)
        kern.attn_fwd(qr, kr, v, mask, P, y, hq, hk)
        self.kctx = qr, kr, v, P, cos, sin
        return y

    def attention_kernels_backward(self, gy):
        kern = self.kernels()
        emb_dim, H, (hq, hk) = self.config
        qr, kr, v, P, cos, sin = self.kctx
        self.kctx = None
        gy = np.ascontiguousarray(gy)
        gS = np.empty_like(P)
        gqr, gkr, gv = np.empty_like(qr), np.empty_like(kr), np.empty_like(v)
        kern.attn_bwd(gy, qr, kr, v, P, gS, gqr, gkr, gv, hq, hk)
        gq, gk = np.empty_like(qr), np.empty_like(kr)
        kern.rope(gqr, gq, cos, sin, hq, inverse=True)
        kern.rope(gkr, gk, cos, sin, hk, inverse=True)
        return gq, gk, gv

    def __backward__(self, gy):
        emb_dim, H, (hq, hk) = self.config
        B, T, _ = gy.shape
        gy = self.linear_o.backward(gy)
        if self.kctx is not None:
            gq, gk, gv = self.attention_kernels_backward(gy)
        else:
            gq, gk, gv = self.attention.backward(gy)
        if self.q_norm is not None:
            gq = self.q_norm.backward(gq.reshape(B, T, hq, H)).reshape(B, T, hq * H)
            gk = self.k_norm.backward(gk.reshape(B, T, hk, H)).reshape(B, T, hk * H)
        gx = self.linear_q.backward(gq)
        gx += self.linear_k.backward(gk)
        gx += self.linear_v.backward(gv)
        return gx

    def update(self, eta=0.001, **kwargs):
        for l in self.layers():
            l.update(eta=eta, **kwargs)

    def layers(self):
        ls = [self.linear_q, self.linear_k, self.linear_v, self.linear_o]
        if self.q_norm is not None:
            ls += [self.q_norm, self.k_norm]
        return ls


#### MLP(SwiGLU) ####################################################
class QwenMLP(sbh.SwiGLU):
    """
    y = linear_o(SiLU(linear_g(x)) * linear_i(x))   (HFの gate, up, down に対応)
    線形層をLinearLayerLoRAにしたSwiGLU。Qwenは出力にdropoutが無いので、
    forwardのdropoutはLoRA経路だけに渡す。updateはSwiGLUのものを使う
    """
    def __init__(self, emb_dim=None, intermediate=None, **kwargs):
        Function.__init__(self)
        self.config = emb_dim, intermediate, None
        print('Initialize', self.__class__.__name__, self.config, kwargs)
        self.linear_g = LinearLayerLoRA(emb_dim, intermediate, bias=False, **kwargs)
        self.linear_i = LinearLayerLoRA(emb_dim, intermediate, bias=False, **kwargs)
        self.linear_o = LinearLayerLoRA(intermediate, emb_dim, bias=False, **kwargs)
        self.act = A.Swish()

    def __forward__(self, x, *, dropout=0.0, **kwargs):
        with _no_overflow_warning():
            gate = self.act.forward(self.linear_g.forward(x, dropout=dropout))
        up = self.linear_i.forward(x, dropout=dropout)
        self.gate = gate
        self.up = up
        return self.linear_o.forward(gate * up, dropout=dropout)

    def __backward__(self, gy):
        gprod = self.linear_o.backward(gy)
        with _no_overflow_warning():
            ggate = self.act.backward(gprod * self.up)
        gup = gprod * self.gate
        self.gate = self.up = None
        gx = self.linear_i.backward(gup)
        gx += self.linear_g.backward(ggate)
        return gx

    def layers(self):
        return [self.linear_g, self.linear_i, self.linear_o]


#### デコーダ層 ######################################################
class QwenBlock(sbh.TransformerBlock):
    """
    x -> x + sa(ln1(x)) -> z + ffwd(ln2(z))
    TransformerBlockと同じ構成なので __forward__/__backward__/update はそのまま使う
    """
    def __init__(self, emb_dim, n_head, head_dim=None, intermediate=None,
                 rope_base=10000.0, eps=1e-6, qkv_bias=True, qk_norm=False, **kwargs):
        Function.__init__(self)
        self.sa = QwenSelfAttention(emb_dim, head_dim, n_head, rope_base=rope_base,
                                    bias=qkv_bias, qk_norm=qk_norm, eps=eps, **kwargs)
        self.ffwd = QwenMLP(emb_dim, intermediate, **kwargs)
        self.ln1 = RMSNormalizationGamma(eps=eps, **kwargs)
        self.ln2 = RMSNormalizationGamma(eps=eps, **kwargs)
        self.proj = None
        self.proj_used = False

    def layers(self):
        return [self.ln1, *self.sa.layers(), self.ln2, *self.ffwd.layers()]


#### 出力層(重み共有) + 交差エントロピー #############################
class LinearLayerCrossEntropyTied(Function):
    """
    logits = x @ E.T (EはEmbeddingの重み(V,D)を共有) と交差エントロピーを、
    行方向のタイルで計算する(語彙15万でもlogits全体を保持しない)
    t == ignore の位置は損失に含めない。損失はトークン平均
    逆伝播ではlogitsを再計算し、Eの勾配をEmbedding側へ設定する
    NLCPyでEが凍結中なら、VEの融合カーネル(ve_kernels.lm_ce)で損失とxの勾配を順伝播で一緒に求める
    (logitsの計算が1回で済む。逆伝播ではgyを掛けるだけ)
    """
    def __init__(self, embedding, tile_size=512, ignore=-100, use_kernels=True):
        super().__init__()
        self.embedding = weakref.ref(embedding) # パラメタの二重登録を避けるため弱参照
        self.tile_size = tile_size
        self.ignore = ignore
        self.use_kernels = use_kernels
        self._wT = None
        self.gx = None

    def weight(self):
        return self.embedding().parameters.w # (V, D)

    def weight_t(self):
        """ Eの転置(D,V)。LinearLayerLoRA.weight_tと同じく、NLCPyで凍結中は連続な転置を保持 """
        w = self.weight()
        if not (self.embedding().frozen and np.__name__ == 'nlcpy'):
            return w.T
        if self._wT is None or self._wT[0] is not w:
            self._wT = (w, np.ascontiguousarray(w.T))
        return self._wT[1]

    def __forward__(self, x, t=None):
        wT = self.weight_t()
        if t is None:                         # 推論: logitsを返す
            return snp.matmul(x, wT)
        xf = snp.reshape(x, (-1, x.shape[-1]))
        tf = t.reshape(-1)
        N = xf.shape[0]
        valid = tf != self.ignore
        self.n_valid = max(int(snp.sum(valid)), 1)
        kern = ve_kernels.get_kernels(np, xf.dtype) \
               if self.use_kernels and self.embedding().frozen else None
        if kern is not None:                  # 損失とxの勾配(gy=1)を一緒に求める
            lrow, self.gx = kern.lm_ce(np.ascontiguousarray(xf), self.weight(), wT,
                np.ascontiguousarray(np.asarray(tf), dtype='int64'),
                self.ignore, 1.0 / self.n_valid)
            self.x = x
            return snp.sum(lrow) / self.n_valid
        lse = np.zeros(N, dtype=xf.dtype)     # log Σ exp(logits)
        zt = np.zeros(N, dtype=xf.dtype)      # 正解のlogit
        for s in range(0, N, self.tile_size):
            e = min(s + self.tile_size, N)
            z = np.dot(xf[s:e], wT)
            m = snp.max(z, axis=-1)
            lse[s:e] = m + np.log(snp.sum(np.exp(z - m[:, None]), axis=-1))
            zt[s:e] = z[np.arange(e - s), np.where(valid[s:e], tf[s:e], 0)]
        loss = snp.sum((lse - zt) * valid) / self.n_valid
        self.x, self.tf, self.lse, self.valid = x, tf, lse, valid
        return loss

    def __backward__(self, gy=1.0):
        if self.gx is not None:               # 融合カーネルで順伝播のときに求めた勾配
            gx, x, self.gx, self.x = self.gx, self.x, None, None
            if float(gy) != 1.0:
                gx *= float(gy)
            return gx.reshape(x.shape)
        emb = self.embedding()
        w, wT = self.weight(), self.weight_t()
        x = self.x
        xf = snp.reshape(x, (-1, x.shape[-1]))
        tf, valid = self.tf, self.valid
        N = xf.shape[0]
        gscale = float(gy) / self.n_valid
        gx = np.empty_like(xf)
        gw = None if emb.frozen else np.zeros_like(w)
        for s in range(0, N, self.tile_size):
            e = min(s + self.tile_size, N)
            z = np.dot(xf[s:e], wT)
            p = np.exp(z - self.lse[s:e, None])               # softmax
            p[np.arange(e - s), np.where(valid[s:e], tf[s:e], 0)] -= 1.0
            p *= (valid[s:e] * gscale)[:, None].astype(p.dtype)
            gx[s:e] = np.dot(p, w)
            if gw is not None:
                gw += np.dot(p.T, xf[s:e])
        if gw is not None:
            emb.parameters.grad_w = gw  # Embeddingの逆伝播がここへ加算する
        self.x = None
        return gx.reshape(x.shape)

    def update(self, **kwargs):
        pass # 重みはEmbedding側が持ち、そちらで更新する


class LmHeadQwen(sbh.LmHead):
    """
    最終RMS正規化 + 重み共有の出力層(+損失)
    sbh.LmHead の unify=True(損失まで一体処理)と同じ構成なので forward/backward/update はそのまま使う
    linear_layer は targets無しでlogits、有りで損失を返す
    """
    def __init__(self, embedding, eps=1e-6, tile_size=512, ignore=-100, **kwargs):
        self.ln_f = RMSNormalizationGamma(eps=eps, **kwargs)
        self.linear_layer = LinearLayerCrossEntropyTied(embedding, tile_size, ignore)
        self.unify = True


#### 推論専用の順伝播(生成用) ########################################
class QwenInference:
    """
    生成のための推論専用の順伝播。計算グラフ(Functionの入出力の記録など)を作らず、
    モデルの重みを直接使って、学習時の順伝播と同じ式を計算する
    (RMS正規化、q,k,v(+バイアス, Qwen3はq_norm/k_norm)、half型RoPE、GQA、厳密なsoftmax、
     SwiGLU、LoRAは併合せずに x@A@B*scale を加える)
    KVキャッシュは層ごとに頭単位 (B,hk,容量,H) で持つ
    NLCPyでは q|k|v と gate|up の重みを結合して積の回数を減らし(生成のたびに作り直す)、
    小さな演算はVEの融合カーネル(ve_kernels)にまとめる
    """
    def __init__(self, model, fuse=None, kernels=None):
        self.model = model
        self.kern = None
        if kernels is None or kernels:
            self.kern = ve_kernels.get_kernels(np, Config.dtype)
            if kernels and self.kern is None:
                raise RuntimeError('VE kernels are not available.')
        self.fuse = (np.__name__ == 'nlcpy') if fuse is None else fuse
        self.fuse = self.fuse or self.kern is not None
        c = model.config
        self.eps = c['eps']
        self.layers = []
        for blk in model.blocks.layers:
            sa, ff = blk.sa, blk.ffwd
            emb_dim, H, (hq, hk) = sa.config
            L = dict(sa=sa, ff=ff, ln1=blk.ln1.gamma, ln2=blk.ln2.gamma,
                     H=H, hq=hq, hk=hk, k=None, v=None)
            if self.fuse:
                lq, lk, lv = sa.linear_q, sa.linear_k, sa.linear_v
                L['w_qkv'] = np.concatenate(
                    [lq.parameters.w, lk.parameters.w, lv.parameters.w], axis=1)
                L['b_qkv'] = np.concatenate(
                    [lq.parameters.b, lk.parameters.b, lv.parameters.b]) \
                    if lq.parameters.bias else None
                L['w_gu'] = np.concatenate(
                    [ff.linear_g.parameters.w, ff.linear_i.parameters.w], axis=1)
            self.layers.append(L)
        self.length = 0

    def rms(self, x, gamma):
        sigma = np.sqrt(snp.mean(x * x, axis=-1, keepdims=True) + self.eps)
        return x / sigma * gamma

    @staticmethod
    def lora(layer, x, y):
        """ y に LoRA の寄与を加える """
        if layer.lora_a is not None:
            u = snp.matmul(x, layer.lora_a.parameters.w)
            y += snp.matmul(u, layer.lora_b.parameters.w) * layer.lora_scale
        return y

    @classmethod
    def linear(cls, layer, x):
        p = layer.parameters
        y = snp.matmul(x, p.w)
        if p.bias:
            y += p.b
        return cls.lora(layer, x, y)

    def rope(self, sa, x, offset):
        """ x:(B,h,T,H) に位置 offset からの half型RoPEを掛ける(RotaryTransformHalfと同じ式) """
        rope = sa.attention.rope
        rope.offset = offset
        cos, sin = rope.get_cos_sin(x)
        rope.offset = 0
        d = x.shape[-1] // 2
        x0, x1 = x[..., :d], x[..., d:]
        return np.concatenate([x0 * cos - x1 * sin, x0 * sin + x1 * cos], axis=-1)

    def append_kv(self, L, k, v):
        B, hk, T, H = k.shape
        n = self.length
        if L['k'] is None or L['k'].shape[2] < n + T:
            cap = max(256, 2 * (n + T))
            nk = np.empty((B, hk, cap, H), dtype=k.dtype)
            nv = np.empty((B, hk, cap, H), dtype=v.dtype)
            if L['k'] is not None:
                nk[:, :, :n] = L['k'][:, :, :n]
                nv[:, :, :n] = L['v'][:, :, :n]
            L['k'], L['v'] = nk, nv
        L['k'][:, :, n:n + T] = k
        L['v'][:, :, n:n + T] = v
        return L['k'][:, :, :n + T], L['v'][:, :, :n + T]

    def attention(self, L, x):
        sa, H, hq, hk = L['sa'], L['H'], L['hq'], L['hk']
        B, T, _ = x.shape
        if self.fuse:
            qkv = snp.matmul(x, L['w_qkv'])
            if L['b_qkv'] is not None:
                qkv += L['b_qkv']
            q = qkv[..., :hq * H]
            k = qkv[..., hq * H:(hq + hk) * H]
            v = qkv[..., (hq + hk) * H:]
            q = self.lora(sa.linear_q, x, q)
            k = self.lora(sa.linear_k, x, k)
            v = self.lora(sa.linear_v, x, v)
        else:
            q = self.linear(sa.linear_q, x)
            k = self.linear(sa.linear_k, x)
            v = self.linear(sa.linear_v, x)
        q = q.reshape(B, T, hq, H)
        k = k.reshape(B, T, hk, H)
        if sa.q_norm is not None:                       # Qwen3: 頭ごとに正規化
            q = self.rms(q, sa.q_norm.gamma)
            k = self.rms(k, sa.k_norm.gamma)
        q = self.rope(sa, q.transpose(0, 2, 1, 3), self.length)   # (B,hq,T,H)
        k = self.rope(sa, k.transpose(0, 2, 1, 3), self.length)   # (B,hk,T,H)
        v = v.reshape(B, T, hk, H).transpose(0, 2, 1, 3)
        k, v = self.append_kv(L, k, v)                  # (B,hk,Tk,H)
        Tk = k.shape[2]
        g = hq // hk                                    # GQA: q頭 h は kv頭 h//g を使う
        q = np.reshape(q, (B, hk, g * T, H))
        a = snp.matmul(q, k.transpose(0, 1, 3, 2)) * (H ** -0.5)    # (B,hk,g*T,Tk)
        if T > 1:                                       # 位置 offset+i は j<=offset+i を見る
            a = a.reshape(B, hk, g, T, Tk)
            allow = np.arange(Tk)[None, :] <= (np.arange(T)[:, None] + self.length)
            a = np.where(allow, a, -np.inf)
            a = a.reshape(B, hk, g * T, Tk)
        a = np.exp(a - snp.max(a, axis=-1, keepdims=True))
        a /= snp.sum(a, axis=-1, keepdims=True)
        y = snp.matmul(a, v)                            # (B,hk,g*T,H)
        y = y.reshape(B, hq, T, H).transpose(0, 2, 1, 3).reshape(B, T, hq * H)
        return self.linear(sa.linear_o, y)

    def mlp(self, L, x):
        ff = L['ff']
        if self.fuse:
            gu = snp.matmul(x, L['w_gu'])
            n = gu.shape[-1] // 2
            gate = self.lora(ff.linear_g, x, gu[..., :n])
            up = self.lora(ff.linear_i, x, gu[..., n:])
        else:
            gate = self.linear(ff.linear_g, x)
            up = self.linear(ff.linear_i, x)
        with _no_overflow_warning():
            gate = gate * (1.0 / (1.0 + np.exp(-gate)))  # SiLU(A.Swishと同じ)
        return self.linear(ff.linear_o, gate * up)

    def forward(self, idx):
        """ idx:(B,T) を位置 self.length から順伝播し、最後の位置のlogits (B,V) を返す """
        if self.kern is not None:
            return self.forward_kernels(idx)
        model = self.model
        x = model.embed.parameters.w[idx]
        for L in self.layers:
            x = x + self.attention(L, self.rms(x, L['ln1']))
            x = x + self.mlp(L, self.rms(x, L['ln2']))
        self.length += idx.shape[1]
        h = self.rms(x[:, -1], model.lm_head.ln_f.gamma)  # (B,D)
        return self.logits(h)

    def logits(self, h):
        w = self.model.lm_head.linear_layer.weight()      # (V,D)
        if h.shape[0] == 1:
            return np.dot(w, h[0])[None]                 # 行列ベクトル積の方が速い
        return np.dot(h, w.T)

    def kv_buffers(self, L, B, hk, H, n):
        """ 融合カーネル用に、長さnまで入る転置したKVキャッシュ(B,hk,H,容量)を用意する """
        if L['k'] is None or L['k'].shape[3] < n:
            cap = max(256, 2 * n)
            nk = np.empty((B, hk, H, cap), dtype=Config.dtype)
            nv = np.empty((B, hk, H, cap), dtype=Config.dtype)
            if L['k'] is not None:
                nk[..., :self.length] = L['k'][..., :self.length]
                nv[..., :self.length] = L['v'][..., :self.length]
            L['k'], L['v'] = nk, nv
        return L['k'], L['v']

    def forward_kernels(self, idx):
        """ forwardと同じ計算を、小さな演算をVEの融合カーネルにまとめて行う """
        kern, model, eps = self.kern, self.model, self.eps
        B, T = idx.shape
        n, offset = B * T, self.length
        x = np.ascontiguousarray(model.embed.parameters.w[idx.reshape(-1)])  # (n,D)
        h = np.empty_like(x)
        d = None                                         # 前の層の出力(残差に加える)
        rope = self.layers[0]['sa'].attention.rope
        H0 = self.layers[0]['H']
        rope.offset = offset                             # RoPEの表は全層で共通
        cos, sin = rope.get_cos_sin(np.empty((T, H0), dtype=Config.dtype))
        rope.offset = 0
        cos = np.ascontiguousarray(cos.reshape(T, -1))
        sin = np.ascontiguousarray(sin.reshape(T, -1))
        for L in self.layers:
            sa, ff, H, hq, hk = L['sa'], L['ff'], L['H'], L['hq'], L['hk']
            kern.add_rms(x, d, L['ln1'], h, eps)         # x += d; h = rms(x)
            qkv = np.dot(h, L['w_qkv'])
            self.lora(sa.linear_q, h, qkv[:, :hq * H])
            self.lora(sa.linear_k, h, qkv[:, hq * H:(hq + hk) * H])
            self.lora(sa.linear_v, h, qkv[:, (hq + hk) * H:])
            kc, vc = self.kv_buffers(L, B, hk, H, offset + T)
            q = np.empty((B, hq, T, H), dtype=x.dtype)
            kern.qkv_rope(qkv, L['b_qkv'],
                          None if sa.q_norm is None else sa.q_norm.gamma,
                          None if sa.k_norm is None else sa.k_norm.gamma,
                          cos, sin, q, kc, vc, B, T, hq, hk, H, offset, eps)
            y = np.empty((n, hq * H), dtype=x.dtype)
            kern.attention(q, kc, vc, y, B, T, hq, hk, H, offset)
            d = self.linear(sa.linear_o, y)
            kern.add_rms(x, d, L['ln2'], h, eps)
            gu = np.dot(h, L['w_gu'])
            I = gu.shape[1] // 2
            self.lora(ff.linear_g, h, gu[:, :I])
            self.lora(ff.linear_i, h, gu[:, I:])
            a = np.empty((n, I), dtype=x.dtype)
            kern.silu_mul(gu, a)
            d = self.linear(ff.linear_o, a)
        self.length += T
        last = np.ascontiguousarray(x.reshape(B, T, -1)[:, -1])
        dl = np.ascontiguousarray(d.reshape(B, T, -1)[:, -1])
        hl = np.empty_like(last)
        kern.add_rms(last, dl, model.lm_head.ln_f.gamma, hl, eps)
        return self.logits(hl)


#### 言語モデル ######################################################
class QwenLanguageModel:
    """
    Qwen2/2.5/3系の因果言語モデル

    forward(idx)          : logits (B,T,V)
    forward(idx, targets) : 次トークン予測の平均損失(targetsはidxを1つずらしたもの、-100は無視)
    backward()            : 損失からの逆伝播
    update(eta=...)       : 学習対象のパラメタを各層のOptimizersで更新
    generate(seed, ...)   : KVキャッシュ付きの逐次生成
    """
    LORA_TARGETS = ('q_proj', 'k_proj', 'v_proj', 'o_proj',
                    'gate_proj', 'up_proj', 'down_proj')

    def __init__(self, vocab_size=151936, emb_dim=896, n_layer=24, n_head=(14, 2),
                 head_dim=None, intermediate=4864, rope_base=1e6, eps=1e-6,
                 qkv_bias=True, qk_norm=False, tie=True, optimize='AdamT', w_decay=0.0,
                 ignore=-100, tile_size=512, eos_ids=None, **kwargs):
        self.config = dict(vocab_size=vocab_size, emb_dim=emb_dim, n_layer=n_layer,
                           n_head=n_head, head_dim=head_dim, intermediate=intermediate,
                           rope_base=rope_base, eps=eps, qkv_bias=qkv_bias,
                           qk_norm=qk_norm, tie=tie)
        print('Initialize', self.__class__.__name__, self.config)
        kwargs['optimize'] = optimize
        kwargs['w_decay']  = w_decay

        # -- Embedding -------------------------------------------------
        self.embed = EmbeddingQwen(vocab_size, emb_dim, tied=tie, **kwargs)

        # -- Transformer blocks ----------------------------------------
        self.blocks = nn.Sequential(
            *[QwenBlock(emb_dim, n_head, head_dim, intermediate, rope_base, eps,
                        qkv_bias, qk_norm, **kwargs)
              for _ in range(n_layer)]
            )

        # -- Lm Head(tie=Falseなら出力層用の重みを別に持つ) ------------
        self.lm_weight = None if tie else EmbeddingQwen(vocab_size, emb_dim, **kwargs)
        self.lm_head = LmHeadQwen(self.embed if tie else self.lm_weight,
                                  eps, tile_size, ignore, **kwargs)

        # -- others ----------------------------------------------------
        self.vocab_size = vocab_size
        self.eos_ids = eos_ids
        self.lora_config = None

    # ---- HFの学習済みモデルから構築 ----
    @classmethod
    def from_pretrained(cls, model_dir, revision=None, **kwargs):
        """ model_dir: HF形式のディレクトリ、またはHFのリポジトリ名(要huggingface_hub) """
        model_dir = resolve_model_dir(model_dir, revision)
        with open(os.path.join(model_dir, 'config.json')) as f:
            c = json.load(f)
        mt = c.get('model_type', 'qwen2')
        if c.get('rope_scaling'):
            raise NotImplementedError('rope_scaling is not supported.')
        eos = c.get('eos_token_id')
        model = cls(vocab_size=c['vocab_size'], emb_dim=c['hidden_size'],
                    n_layer=c['num_hidden_layers'],
                    n_head=(c['num_attention_heads'],
                            c.get('num_key_value_heads', c['num_attention_heads'])),
                    head_dim=c.get('head_dim'), intermediate=c['intermediate_size'],
                    rope_base=float(c.get('rope_theta', 10000.0)),
                    eps=float(c.get('rms_norm_eps', 1e-6)),
                    qkv_bias=c.get('attention_bias', mt == 'qwen2'),
                    qk_norm=(mt == 'qwen3'),
                    tie=c.get('tie_word_embeddings', True),
                    eos_ids=eos if isinstance(eos, list) else [eos], **kwargs)
        t0 = time.time()
        model.import_hf_weights(load_safetensors(model_dir, dtype=Config.dtype))
        print(f'{cls.__name__}: loaded {mt} ({model.num_parameters()/1e6:.1f}M params)'
              f' in {time.time()-t0:.1f}s')
        return model

    def parameter_table(self):
        """ HFの名前 -> (保持オブジェクト, 属性名, 勾配属性名, 転置の要否) """
        table = {'model.embed_tokens.weight':
                 (self.embed.parameters, 'w', 'grad_w', False),
                 'model.norm.weight': (self.lm_head.ln_f, 'gamma', 'ggamma', False)}
        if self.lm_weight is not None:
            table['lm_head.weight'] = (self.lm_weight.parameters, 'w', 'grad_w', False)
        for name, layer in self.linear_layers().items():
            table[name + '.weight'] = (layer.parameters, 'w', 'grad_w', True)
            if layer.parameters.bias:
                table[name + '.bias'] = (layer.parameters, 'b', 'grad_b', False)
        for i, blk in enumerate(self.blocks.layers):
            p = f'model.layers.{i}.'
            table[p + 'input_layernorm.weight'] = (blk.ln1, 'gamma', 'ggamma', False)
            table[p + 'post_attention_layernorm.weight'] = (blk.ln2, 'gamma', 'ggamma', False)
            if blk.sa.q_norm is not None:
                table[p + 'self_attn.q_norm.weight'] = (blk.sa.q_norm, 'gamma', 'ggamma', False)
                table[p + 'self_attn.k_norm.weight'] = (blk.sa.k_norm, 'gamma', 'ggamma', False)
        return table

    def linear_layers(self):
        """ HFのモジュール名 -> LinearLayerLoRA """
        d = {}
        for i, blk in enumerate(self.blocks.layers):
            p = f'model.layers.{i}.'
            d[p + 'self_attn.q_proj'] = blk.sa.linear_q
            d[p + 'self_attn.k_proj'] = blk.sa.linear_k
            d[p + 'self_attn.v_proj'] = blk.sa.linear_v
            d[p + 'self_attn.o_proj'] = blk.sa.linear_o
            d[p + 'mlp.gate_proj'] = blk.ffwd.linear_g
            d[p + 'mlp.up_proj'] = blk.ffwd.linear_i
            d[p + 'mlp.down_proj'] = blk.ffwd.linear_o
        return d

    def import_hf_weights(self, tensors):
        """ HF(torch)の重み辞書を設定。Linearの(out,in)は(in,out)に転置する """
        table = self.parameter_table()
        done = 0
        for k, a in tensors.items():
            if k not in table:
                warnings.warn(k + ' skipped.')
                continue
            obj, attr, _, transpose = table[k]
            a = a.T if transpose else a
            setattr(obj, attr, np.ascontiguousarray(np.asarray(a, dtype=Config.dtype)))
            done += 1
        missing = [k for k in table if k not in tensors]
        if missing:
            warnings.warn(f'{len(missing)} parameters not given: {missing[:3]}...')
        return done

    def hf_state_dict(self, grad=False):
        """ HFの名前とレイアウトでパラメタ(grad=Trueなら勾配)を返す """
        d = {}
        for k, (obj, attr, gattr, transpose) in self.parameter_table().items():
            a = getattr(obj, gattr if grad else attr)
            if a is not None:
                d[k] = a.T if transpose else a
        return d

    # ---- パラメタ管理 ----
    def layers(self):
        """ 更新・凍結の対象となる全ての層 """
        ls = [self.embed]
        for blk in self.blocks.layers:
            ls += blk.layers()
        ls.append(self.lm_head.ln_f)
        if self.lm_weight is not None:
            ls.append(self.lm_weight)
        return ls

    def set_frozen(self, flag=True):
        for l in self.layers():
            l.frozen = flag

    def num_parameters(self, trainable_only=False):
        n = 0
        for k, (obj, attr, _, _) in self.parameter_table().items():
            a = getattr(obj, attr)
            if a is not None and not (trainable_only and self._owner_frozen(obj)):
                n += a.size
        if self.lora_config is not None:
            for layer in self.linear_layers().values():
                if layer.lora_a is not None:
                    n += layer.lora_a.parameters.w.size + layer.lora_b.parameters.w.size
        return n

    def _owner_frozen(self, obj):
        for l in self.layers():
            if obj is l or obj is getattr(l, 'parameters', None):
                return l.frozen
        return False

    # ---- 順伝播・逆伝播・更新 ----
    def forward(self, idx, targets=None, mask=None, dropout=0.0, last_only=False):
        idx = np.asarray(idx)
        if idx.ndim == 1:
            idx = idx[None]
        x = self.embed(idx)
        x = self.blocks.forward(x, mask=mask, dropout=dropout)
        if last_only:
            x = x[:, -1:]
        if targets is not None:
            targets = np.asarray(targets).reshape(idx.shape)
        return self.lm_head(x, targets) # targets無し:logits, 有り:損失

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def backward(self, gy=None):
        gx = self.lm_head.backward()
        gx = self.blocks.backward(gx)
        self.embed.backward(gx)

    def grad_norm(self):
        s = 0.0                  # デバイス上で足し込み、最後に一度だけ取り出す
        for l in self.layers():
            for p, g in l.gradients():
                if g is not None:
                    s = s + snp.sum(g.astype('float64') ** 2)
        return math.sqrt(float(s))

    def update(self, eta=0.001, max_grad_norm=None, **kwargs):
        """
        各層のupdateで更新。max_grad_normを与えると全体の勾配ノルムでクリップする
        (kwargsのg_clip等は従来どおり各パラメタ個別のクリップとして働く)
        """
        gnorm = self.grad_norm()
        if max_grad_norm is not None and gnorm > max_grad_norm:
            rate = max_grad_norm / (gnorm + 1e-6)
            for l in self.layers():
                for p, g in l.gradients():
                    if g is not None:
                        g *= rate
        self.embed.update(eta=eta, **kwargs)
        self.blocks.update(eta=eta, **kwargs)
        self.lm_head.update(eta=eta, **kwargs)
        if self.lm_weight is not None:
            self.lm_weight.update(eta=eta, **kwargs)
        return gnorm

    # ---- LoRA ----
    def apply_lora(self, r=8, alpha=16, targets=LORA_TARGETS, seed=None, **kwargs):
        """ 全パラメタを凍結し、対象の線形層にLoRAを付加する """
        if seed is not None:
            np.random.seed(seed)
        self.set_frozen(True)
        n = 0
        for name, layer in self.linear_layers().items():
            if name.split('.')[-1] in targets:
                layer.add_lora(r=r, alpha=alpha, **kwargs)
                n += 1
        self.lora_config = dict(r=r, alpha=alpha, targets=list(targets))
        print(f'LoRA applied to {n} linear layers: r={r} alpha={alpha} '
              f'trainable={self.num_parameters(True)/1e6:.3f}M')

    def merge_lora(self):
        for layer in self.linear_layers().values():
            layer.merge_lora()
        self.lora_config = None

    def lora_parameters(self, grad=False):
        """ HFのモジュール名.lora_A/lora_B -> 配列(pyainoの向き A:(in,r), B:(r,out)) """
        d = {}
        for name, layer in self.linear_layers().items():
            if layer.lora_a is not None:
                a, b = layer.lora_a.parameters, layer.lora_b.parameters
                d[name + '.lora_A'] = a.grad_w if grad else a.w
                d[name + '.lora_B'] = b.grad_w if grad else b.w
        return d

    def save_lora(self, file_name, dtype=None):
        """ LoRAの重みを cf.save_parameters と同じ手順(numpy化, dtype変換)でpickle辞書に保存 """
        params = cf.c2n_dictionary(self.lora_parameters())
        params = cf.astype_dictionary(params, dtype=dtype)
        params['lora_config'] = self.lora_config
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)
        print(self.__class__.__name__, 'LoRAのパラメータをファイルに記録しました=>', file_name)

    def load_lora(self, file_name, dtype=None):
        """ save_loraの保存物を読む(cf.load_parameters と同じく現在のnp, dtypeに変換) """
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        lc = params.pop('lora_config')
        if np.__name__ in ('cupy', 'nlcpy'):
            params = cf.n2c_dictionary(params)
        else:
            params = cf.c2n_dictionary(params)
        params = cf.astype_dictionary(params, dtype=dtype)
        if self.lora_config is None:
            self.apply_lora(r=lc['r'], alpha=lc['alpha'], targets=tuple(lc['targets']))
        layers = self.linear_layers()
        for k, v in params.items():
            name, which = k.rsplit('.', 1)
            layer = layers[name]
            holder = layer.lora_a if which == 'lora_A' else layer.lora_b
            holder.parameters.w = v
        print(self.__class__.__name__, 'LoRAのパラメータをファイルから取得しました<=', file_name)

    def export_peft(self, out_dir, base_model_name_or_path=''):
        """ HuggingFace PEFT 互換形式(adapter_model.safetensors + adapter_config.json) """
        os.makedirs(out_dir, exist_ok=True)
        sd = cf.c2n_dictionary({f'base_model.model.{k}.weight': v.T # torchの(out,in)
                                for k, v in self.lora_parameters().items()})
        save_safetensors(os.path.join(out_dir, 'adapter_model.safetensors'), sd)
        lc = self.lora_config
        with open(os.path.join(out_dir, 'adapter_config.json'), 'w') as f:
            json.dump({'peft_type': 'LORA', 'task_type': 'CAUSAL_LM',
                       'base_model_name_or_path': base_model_name_or_path,
                       'r': lc['r'], 'lora_alpha': lc['alpha'], 'lora_dropout': 0.0,
                       'target_modules': lc['targets'], 'bias': 'none',
                       'fan_in_fan_out': False, 'inference_mode': True}, f, indent=2)

    # ---- 生成 ----
    @staticmethod
    def _sample(z, beta, top_k, top_p):
        """
        z(ホストのnumpy配列)から確率 softmax(z)**beta に従って選ぶ(top_k, top_pで候補を絞る)
        選択は、numpyでは cf.select_category、cupy/NLCPyでは Config.np の一様乱数による
        逆関数法(np.random.choice と同じ方法。デバイスとの往復を避ける)。どちらも
        np.random.seed で再現できる
        """
        z = z * beta             # softmax(z)**beta に相当
        if top_k and top_k > 0:
            kth = numpy.partition(z, -top_k)[-top_k]
            z = numpy.where(z < kth, -numpy.inf, z)
        probs = numpy.exp(z - z.max())
        probs /= probs.sum()
        if top_p < 1.0:
            order = numpy.argsort(-probs)
            cum = numpy.cumsum(probs[order])
            probs[order[cum - probs[order] > top_p]] = 0.0
            probs /= probs.sum()
        if np is numpy:
            return int(cf.select_category(probs, True, 1)[0])
        cdf = numpy.cumsum(probs)
        u = float(np.random.random_sample()) * cdf[-1]
        return min(int(numpy.searchsorted(cdf, u, side='right')), len(cdf) - 1)

    def generate(self, seed, max_tokens=1000, stochastic=False, beta=2,
                 skip_ids=None, end_ids=None, top_k=0, top_p=1.0,
                 repetition_penalty=1.0, stream=None, fast=True):
        """
        入力されたidx列(seed)に続くidx列を生成し、seedと結合して返す
        (BigramLanguageModel, RnnLanguageModel と同じく max_tokens はseedを含む全長、
         end_idsが出たら打切り、それも含めて返す)
        stochastic=False: 確定的(argmax)
        stochastic=True : 確率 p**beta に従って選ぶ(cf.select_categoryと同じ。beta=1/温度)
        top_k, top_p    : 確率的に選ぶ際の候補の絞り込み
        repetition_penalty: 既出idxのlogitを抑える(HFと同じ方式)
        skip_ids        : 選択候補から除外するidx
        stream(idx)     : 1トークンごとに呼ばれる(逐次表示用)
        KVキャッシュを使い、seedは一度だけ、以後は1トークンずつ順伝播する
        fast=True       : 計算グラフを作らない推論専用の順伝播(QwenInference)を使う
                          (Falseなら学習と同じ層のforwardをKVCache付きで使う)
        """
        if not isinstance(seed, np.ndarray):
            seed = np.array(seed)
        seed = seed.reshape(-1).astype('int64')
        if len(seed) == 0:
            raise ValueError('seed must contain at least one token.')
        if max_tokens < len(seed):
            raise ValueError(
                f'max_tokens ({max_tokens}) must be >= seed length ({len(seed)}).')
        end_ids = self.eos_ids if end_ids is None else end_ids
        end_ids = set(int(i) for i in (end_ids or []))
        gen_data = [int(i) for i in seed.tolist()]

        if fast:
            infer = QwenInference(self)
            step = lambda idx: infer.forward(idx)                     # (1,V)
        else:
            for blk in self.blocks.layers:
                blk.sa.kv = KVCache()
            step = lambda idx: self.forward(idx, last_only=True)[:, -1] # (1,V)
        try:
            logits = step(seed.reshape(1, -1))
            for i in range(max_tokens - len(seed)):
                z = logits[0]
                if stochastic or repetition_penalty != 1.0 or skip_ids is not None:
                    # 選択の前処理はホストで行う(cupy/NLCPyでは小さな演算の繰り返しが遅い)
                    z = (z.get() if hasattr(z, 'get') else numpy.asarray(z)).astype('float64')
                if repetition_penalty != 1.0:
                    prev = numpy.array(sorted(set(gen_data)))
                    zp = z[prev]
                    z[prev] = numpy.where(zp > 0, zp / repetition_penalty,
                                          zp * repetition_penalty)
                if skip_ids is not None: # skip_idsの指すものを選択候補から除外
                    z[numpy.array(skip_ids)] = -numpy.inf
                if stochastic:
                    next_idx = self._sample(z, beta, top_k, top_p)
                else:                    # 確定的: softmaxを経ずにargmax(結果は同じ)
                    next_idx = int(z.argmax())
                gen_data.append(next_idx)
                if stream is not None:
                    stream(next_idx)
                if next_idx in end_ids: # end_idが出現したら打切り
                    break
                logits = step(np.array([[next_idx]]))
        finally:
            for blk in self.blocks.layers:
                blk.sa.kv = None
        return np.array(gen_data)


#### 学習用ユーティリティ ############################################
def build_sft_example(tokenizer, messages, max_len=None, train_on_prompt=False,
                      enable_thinking=None):
    """
    チャット形式の1例から (ids, labels) を作る(labelsはidsと同じ位置合わせ)
    最後のassistant発話と<|im_end|>だけを学習対象とし、他は-100でマスクする
    """
    assert messages[-1]['role'] == 'assistant'
    prompt = tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=True,
                                           enable_thinking=enable_thinking)
    full = tokenizer.apply_chat_template(messages, add_generation_prompt=False)
    if not full.startswith(prompt):   # Qwen3の<think>等で一致しない場合に補う
        full = prompt + messages[-1]['content'] + '<|im_end|>\n'
    # 生成時と同じく「プロンプトのトークン列 + 応答のトークン列」として分けて符号化
    p_ids = tokenizer.encode(prompt, ndarray=False)
    ids = p_ids + tokenizer.encode(full[len(prompt):], ndarray=False)
    labels = list(ids) if train_on_prompt else [-100] * len(p_ids) + ids[len(p_ids):]
    if tokenizer.decode(ids[-1:]) == '\n': # <|im_end|>の後の改行は学習しない
        labels[-1] = -100
    if max_len is not None:
        ids, labels = ids[:max_len], labels[:max_len]
    return ids, labels


def collate(batch, pad_id):
    """
    [(ids, labels), ...] を右詰めpaddingし、pyainoの言語モデルの流儀で
    入力 x = ids[:, :-1] と 目標 t = labels[:, 1:] (次トークン) を返す
    """
    import numpy
    T = max(len(i) for i, _ in batch)
    ids = numpy.full((len(batch), T), pad_id, dtype=numpy.int64)
    lab = numpy.full((len(batch), T), -100, dtype=numpy.int64)
    for b, (i, l) in enumerate(batch):
        ids[b, :len(i)] = i
        lab[b, :len(l)] = l
    return np.asarray(ids[:, :-1]), np.asarray(lab[:, 1:])

