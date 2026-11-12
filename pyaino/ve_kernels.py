# ve_kernels
# 20261112
# NLCPy(SX-Aurora)用の融合カーネル。生成(QwenInference)の1トークンあたりの演算回数を減らす
# NLCPyでは小さな配列の演算1回ごとに数十μsかかるので、RMS正規化やRoPE、注意の計算を
# 1つのVE関数(ncc でJITコンパイル)にまとめる。numpy/cupyでは使わない
import glob
import os
import warnings

_C_SOURCE = r'''
#include <stdint.h>
#include <math.h>
#include <cblas.h>

/* x += d (dが0でなければ); y = x / sqrt(mean(x^2) + eps) * g  (n行 × D列) */
void add_rms_SFX(uint64_t px, uint64_t pd, uint64_t pg, uint64_t py,
                 int64_t n, int64_t D, double eps)
{
    REAL *x = (REAL*)px; const REAL *d = (const REAL*)pd;
    const REAL *g = (const REAL*)pg; REAL *y = (REAL*)py;
#pragma omp parallel for
    for (int64_t i = 0; i < n; i++) {
        REAL *xi = x + i * D;
        if (d) {
            const REAL *di = d + i * D;
            for (int64_t j = 0; j < D; j++) xi[j] += di[j];
        }
        double s = 0.0;
        for (int64_t j = 0; j < D; j++) s += (double)xi[j] * xi[j];
        REAL sigma = (REAL)sqrt(s / D + eps);
        REAL *yi = y + i * D;
        for (int64_t j = 0; j < D; j++) yi[j] = xi[j] / sigma * g[j];
    }
}

/* qkv:(B*T, (hq+2hk)*H) にバイアスを加え、q,kに頭ごとのRMS正規化(qg,kgが0でなければ)と
   half型RoPE(cs,sn:(T,H/2))を掛けて、q:(B,hq,T,H)と、転置したキャッシュ kc,vc:(B,hk,H,cap)の
   位置offset+tへ書く */
void qkv_rope_SFX(uint64_t pqkv, uint64_t pb, uint64_t pqg, uint64_t pkg,
                  uint64_t pcs, uint64_t psn, uint64_t pq, uint64_t pkc, uint64_t pvc,
                  int64_t B, int64_t T, int64_t hq, int64_t hk, int64_t H,
                  int64_t cap, int64_t offset, double eps)
{
    const REAL *qkv = (const REAL*)pqkv, *bias = (const REAL*)pb;
    const REAL *qg = (const REAL*)pqg, *kg = (const REAL*)pkg;
    const REAL *cs = (const REAL*)pcs, *sn = (const REAL*)psn;
    REAL *q = (REAL*)pq, *kc = (REAL*)pkc, *vc = (REAL*)pvc;
    const int64_t nh = hq + 2 * hk, C = nh * H, h2 = H / 2;
#pragma omp parallel for collapse(2)
    for (int64_t r = 0; r < B * T; r++) {
        for (int64_t h = 0; h < nh; h++) {
            const int64_t b = r / T, t = r % T;
            REAL v[1024];                       /* head_dim <= 1024 */
            const REAL *src = qkv + r * C + h * H;
            for (int64_t j = 0; j < H; j++)
                v[j] = bias ? src[j] + bias[h * H + j] : src[j];
            if (h >= hq + hk) {                 /* v: そのまま書く */
                REAL *dst = vc + (b * hk + (h - hq - hk)) * H * cap + offset + t;
                for (int64_t j = 0; j < H; j++) dst[j * cap] = v[j];
                continue;
            }
            const REAL *gm = h < hq ? qg : kg;
            if (gm) {
                double s = 0.0;
                for (int64_t j = 0; j < H; j++) s += (double)v[j] * v[j];
                REAL sigma = (REAL)sqrt(s / H + eps);
                for (int64_t j = 0; j < H; j++) v[j] = v[j] / sigma * gm[j];
            }
            const REAL *c = cs + t * h2, *s = sn + t * h2;
            REAL *dst; int64_t st;              /* 書込み先と要素の間隔 */
            if (h < hq) { dst = q + ((b * hq + h) * T + t) * H; st = 1; }
            else { dst = kc + (b * hk + (h - hq)) * H * cap + offset + t; st = cap; }
            for (int64_t j = 0; j < h2; j++) {
                REAL x0 = v[j], x1 = v[j + h2];
                dst[j * st]        = x0 * c[j] - x1 * s[j];
                dst[(j + h2) * st] = x0 * s[j] + x1 * c[j];
            }
        }
    }
}

/* 因果的な注意(GQA, 厳密なsoftmax)。q:(B,hq,T,H), kc,vc:(B,hk,H,cap), y:(B,T,hq*H)
   位置offset+tのqueryは位置0..offset+tのkeyを見る
   (キャッシュを転置して持つと、keyの方向に連続した長いベクトルで計算できる) */
void attention_SFX(uint64_t pq, uint64_t pkc, uint64_t pvc, uint64_t py,
                   int64_t B, int64_t T, int64_t hq, int64_t hk, int64_t H,
                   int64_t cap, int64_t offset, double scale)
{
    const REAL *q = (const REAL*)pq, *kc = (const REAL*)pkc, *vc = (const REAL*)pvc;
    REAL *y = (REAL*)py;
    const int64_t g = hq / hk;
#pragma omp parallel for collapse(3)
        for (int64_t b = 0; b < B; b++)
        for (int64_t h = 0; h < hq; h++)
        for (int64_t t = 0; t < T; t++) {
            REAL a[offset + t + 1];             /* mallocはVEでは遅いのでスタックに置く */
            const REAL *qi = q + ((b * hq + h) * T + t) * H;
            const REAL *k = kc + (b * hk + h / g) * H * cap;
            const REAL *v = vc + (b * hk + h / g) * H * cap;
            const int64_t L = offset + t + 1;
            for (int64_t j = 0; j < L; j++) a[j] = 0;
            for (int64_t d = 0; d < H; d++) {
                const REAL qd = qi[d], *kd = k + d * cap;
                for (int64_t j = 0; j < L; j++) a[j] += qd * kd[j];
            }
            REAL m = -INFINITY;
            for (int64_t j = 0; j < L; j++) { a[j] *= (REAL)scale; if (a[j] > m) m = a[j]; }
            REAL sum = 0;
            for (int64_t j = 0; j < L; j++) { a[j] = EXP(a[j] - m); sum += a[j]; }
            REAL *yi = y + (b * T + t) * hq * H + h * H;
            for (int64_t d = 0; d < H; d++) {
                const REAL *vd = v + d * cap;
                REAL acc = 0;
                for (int64_t j = 0; j < L; j++) acc += a[j] * vd[j];
                yi[d] = acc / sum;
            }
        }
}

/* ---- 学習用(位置0から、q,k,v は (B,T,h*H) のまま扱い転置しない) ---- */

/* x:(B,T,h*H) に half型RoPE を掛けて y へ。sign=-1 なら逆回転(逆伝播用) */
void rope_SFX(uint64_t px, uint64_t py, uint64_t pcs, uint64_t psn,
              int64_t B, int64_t T, int64_t h, int64_t H, int64_t sign)
{
    const REAL *x = (const REAL*)px, *cs = (const REAL*)pcs, *sn = (const REAL*)psn;
    REAL *y = (REAL*)py;
    const int64_t h2 = H / 2;
#pragma omp parallel for
    for (int64_t r = 0; r < B * T; r++) {
        const int64_t t = r % T;
        const REAL *c = cs + t * h2, *s = sn + t * h2;
        for (int64_t i = 0; i < h; i++) {
            const REAL *xi = x + (r * h + i) * H;
            REAL *yi = y + (r * h + i) * H;
            for (int64_t j = 0; j < h2; j++) {
                REAL x0 = xi[j], x1 = xi[j + h2], sj = sign > 0 ? s[j] : -s[j];
                yi[j]      = x0 * c[j] - x1 * sj;
                yi[j + h2] = x0 * sj + x1 * c[j];
            }
        }
    }
}

/* 因果的な注意の順伝播。q:(B,T,hq*H), k,v:(B,T,hk*H), mask:(B,T)(0の位置のkeyは見ない。0なら無し)
   P:(B,hq,T,T) に注意の重み(見ない位置は0)、y:(B,T,hq*H)
   (b,頭)ごとに S = scale*Q·K^T, P = softmax(S), Y = P·V をBLASで計算する(行の間隔で頭を選ぶので転置不要) */
void attn_fwd_SFX(uint64_t pq, uint64_t pk, uint64_t pv, uint64_t pm, uint64_t pP, uint64_t py,
                  int64_t B, int64_t T, int64_t hq, int64_t hk, int64_t H, double scale)
{
    const REAL *q = (const REAL*)pq, *k = (const REAL*)pk, *v = (const REAL*)pv;
    const REAL *mask = (const REAL*)pm;
    REAL *P = (REAL*)pP, *y = (REAL*)py;
    const int64_t g = hq / hk, Cq = hq * H, Ck = hk * H;
#pragma omp parallel for collapse(2)
    for (int64_t b = 0; b < B; b++)
    for (int64_t h = 0; h < hq; h++) {
        REAL *S = P + (b * hq + h) * T * T;
        const REAL *Q = q + b * T * Cq + h * H;
        const REAL *K = k + b * T * Ck + (h / g) * H, *V = v + b * T * Ck + (h / g) * H;
        const REAL *mb = mask ? mask + b * T : 0;
        GEMM(CblasRowMajor, CblasNoTrans, CblasTrans, T, T, H,
             (REAL)scale, Q, Cq, K, Ck, (REAL)0, S, T);
        for (int64_t t = 0; t < T; t++) {
            REAL *a = S + t * T, m = -INFINITY, sum = 0;
            for (int64_t j = 0; j <= t; j++)
                if ((!mb || mb[j] != 0) && a[j] > m) m = a[j];
            for (int64_t j = 0; j < T; j++) {
                a[j] = (j <= t && (!mb || mb[j] != 0)) ? EXP(a[j] - m) : 0;
                sum += a[j];
            }
            const REAL rs = sum > 0 ? (REAL)1 / sum : 0;
            for (int64_t j = 0; j < T; j++) a[j] *= rs;
        }
        GEMM(CblasRowMajor, CblasNoTrans, CblasNoTrans, T, H, T,
             (REAL)1, S, T, V, Ck, (REAL)0, y + b * T * Cq + h * H, Cq);
    }
}

/* 逆伝播。gP = gY·V^T, gS = P*(gP - Σ_j P*gP)*scale, gQ = gS·K  ((b,q頭)ごと)
          gK = Σ_{グループ内のq頭} gS^T·Q, gV = Σ P^T·gY  ((b,kv頭)ごと) */
void attn_bwd_SFX(uint64_t pgy, uint64_t pq, uint64_t pk, uint64_t pv, uint64_t pP, uint64_t pgS,
                  uint64_t pgq, uint64_t pgk, uint64_t pgv,
                  int64_t B, int64_t T, int64_t hq, int64_t hk, int64_t H, double scale)
{
    const REAL *gy = (const REAL*)pgy, *q = (const REAL*)pq, *k = (const REAL*)pk;
    const REAL *v = (const REAL*)pv, *P = (const REAL*)pP;
    REAL *gS = (REAL*)pgS, *gq = (REAL*)pgq, *gk = (REAL*)pgk, *gv = (REAL*)pgv;
    const int64_t g = hq / hk, Cq = hq * H, Ck = hk * H;
#pragma omp parallel for collapse(2)
    for (int64_t b = 0; b < B; b++)
    for (int64_t h = 0; h < hq; h++) {
        const REAL *A = P + (b * hq + h) * T * T;
        REAL *G = gS + (b * hq + h) * T * T;
        const REAL *GY = gy + b * T * Cq + h * H;
        const REAL *K = k + b * T * Ck + (h / g) * H, *V = v + b * T * Ck + (h / g) * H;
        GEMM(CblasRowMajor, CblasNoTrans, CblasTrans, T, T, H,
             (REAL)1, GY, Cq, V, Ck, (REAL)0, G, T);
        for (int64_t t = 0; t < T; t++) {
            const REAL *a = A + t * T;
            REAL *gs = G + t * T, dsum = 0;
            for (int64_t j = 0; j < T; j++) dsum += a[j] * gs[j];
            for (int64_t j = 0; j < T; j++) gs[j] = a[j] * (gs[j] - dsum) * (REAL)scale;
        }
        GEMM(CblasRowMajor, CblasNoTrans, CblasNoTrans, T, H, T,
             (REAL)1, G, T, K, Ck, (REAL)0, gq + b * T * Cq + h * H, Cq);
    }
#pragma omp parallel for collapse(2)
    for (int64_t b = 0; b < B; b++)
    for (int64_t kh = 0; kh < hk; kh++) {
        REAL *GK = gk + b * T * Ck + kh * H, *GV = gv + b * T * Ck + kh * H;
        for (int64_t h = kh * g; h < (kh + 1) * g; h++) {
            const REAL beta = h == kh * g ? (REAL)0 : (REAL)1;
            GEMM(CblasRowMajor, CblasTrans, CblasNoTrans, T, H, T,
                 (REAL)1, gS + (b * hq + h) * T * T, T, q + b * T * Cq + h * H, Cq,
                 beta, GK, Ck);
            GEMM(CblasRowMajor, CblasTrans, CblasNoTrans, T, H, T,
                 (REAL)1, P + (b * hq + h) * T * T, T, gy + b * T * Cq + h * H, Cq,
                 beta, GV, Ck);
        }
    }
}

/* gu:(n, 2I) の前半gate, 後半up から y = SiLU(gate) * up  (n × I) */
void silu_mul_SFX(uint64_t pgu, uint64_t py, int64_t n, int64_t I)
{
    const REAL *gu = (const REAL*)pgu; REAL *y = (REAL*)py;
#pragma omp parallel for
    for (int64_t i = 0; i < n; i++) {
        const REAL *gi = gu + i * 2 * I, *ui = gi + I;
        REAL *yi = y + i * I;
        for (int64_t j = 0; j < I; j++)
            yi[j] = gi[j] * ((REAL)1 / ((REAL)1 + EXP(-gi[j]))) * ui[j];
    }
}
'''

def _nlc_dir():
    """ NLC(NEC Numeric Library Collection)の場所。環境変数NLC_HOME、無ければ/opt/nec/ve/nlcの最新版 """
    d = os.environ.get('NLC_HOME')
    if d:
        return d
    cands = sorted(glob.glob('/opt/nec/ve/nlc/*/include/cblas.h'))
    if not cands:
        raise RuntimeError('NLC (cblas.h) is not found. Set NLC_HOME.')
    return os.path.dirname(os.path.dirname(cands[-1]))


_kernels = {}
_libs = {}   # VEの関数はここに置く(VeoLibraryは属性を調べるだけで例外を出すので、
             # cf.save_parametersなどのオブジェクト探索から見えないようにする)


class VEKernels:
    """ 1つのdtype用のカーネル群。引数の配列はC順で連続していること """
    def __init__(self, np, dtype):
        sfx = {'float32': 'f32', 'float64': 'f64'}[dtype]
        real = {'float32': 'float', 'float64': 'double'}[dtype]
        src = _C_SOURCE.replace('REAL', real).replace('_SFX', '_' + sfx)
        src = src.replace('EXP', 'expf' if real == 'float' else 'exp')
        src = src.replace('GEMM', 'cblas_sgemm' if real == 'float' else 'cblas_dgemm')
        nlc = _nlc_dir()   # BLAS(NEC Numeric Library Collection)を逐次版で使う
        cflags = np.jit.get_default_cflags() + ('-I', nlc + '/include')
        ldflags = np.jit.get_default_ldflags() + (
            '-L' + nlc + '/lib', '-Wl,-rpath,' + nlc + '/lib', '-lcblas', '-lblas_sequential')
        lib = np.jit.CustomVELibrary(code=src, cflags=cflags, ldflags=ldflags)
        u, i, f = np.ve_types.uint64, np.ve_types.int64, np.ve_types.float64
        get = lambda name, args: lib.get_function(name + '_' + sfx, args_type=args)
        _libs[dtype] = dict(
            lib=lib,
            add_rms=get('add_rms', (u, u, u, u, i, i, f)),
            qkv_rope=get('qkv_rope', (u,) * 9 + (i,) * 7 + (f,)),
            attention=get('attention', (u,) * 4 + (i,) * 7 + (f,)),
            silu_mul=get('silu_mul', (u, u, i, i)),
            rope=get('rope', (u,) * 4 + (i,) * 5),
            attn_fwd=get('attn_fwd', (u,) * 6 + (i,) * 5 + (f,)),
            attn_bwd=get('attn_bwd', (u,) * 9 + (i,) * 5 + (f,)))
        self.dtype = dtype

    def _f(self, name):
        return _libs[self.dtype][name]

    @staticmethod
    def _adr(a):
        return 0 if a is None else a.ve_adr

    def add_rms(self, x, d, g, y, eps):
        n, D = x.shape
        self._f('add_rms')(x.ve_adr, self._adr(d), g.ve_adr, y.ve_adr, n, D, float(eps))

    def qkv_rope(self, qkv, bias, qg, kg, cos, sin, q, kc, vc, B, T, hq, hk, H, offset, eps):
        self._f('qkv_rope')(qkv.ve_adr, self._adr(bias), self._adr(qg), self._adr(kg),
                            cos.ve_adr, sin.ve_adr, q.ve_adr, kc.ve_adr, vc.ve_adr,
                            B, T, hq, hk, H, kc.shape[3], offset, float(eps))

    def attention(self, q, kc, vc, y, B, T, hq, hk, H, offset):
        self._f('attention')(q.ve_adr, kc.ve_adr, vc.ve_adr, y.ve_adr,
                             B, T, hq, hk, H, kc.shape[3], offset, float(H ** -0.5))

    def silu_mul(self, gu, y):
        n, I = y.shape
        self._f('silu_mul')(gu.ve_adr, y.ve_adr, n, I)

    # ---- 学習用: 因果的な注意(GQA)とRoPE。q,k,v は (B,T,h*H) ----
    def rope(self, x, y, cos, sin, h, inverse=False):
        B, T, _ = x.shape
        self._f('rope')(x.ve_adr, y.ve_adr, cos.ve_adr, sin.ve_adr,
                        B, T, h, x.shape[2] // h, -1 if inverse else 1)

    def attn_fwd(self, q, k, v, mask, P, y, hq, hk):
        B, T, C = q.shape
        H = C // hq
        self._f('attn_fwd')(q.ve_adr, k.ve_adr, v.ve_adr, self._adr(mask), P.ve_adr, y.ve_adr,
                            B, T, hq, hk, H, float(H ** -0.5))

    def attn_bwd(self, gy, q, k, v, P, gS, gq, gk, gv, hq, hk):
        B, T, C = q.shape
        H = C // hq
        self._f('attn_bwd')(gy.ve_adr, q.ve_adr, k.ve_adr, v.ve_adr, P.ve_adr, gS.ve_adr,
                            gq.ve_adr, gk.ve_adr, gv.ve_adr, B, T, hq, hk, H, float(H ** -0.5))


def get_kernels(np, dtype):
    """ NLCPyならdtype用のカーネル群を返す(初回にコンパイル)。それ以外やコンパイル失敗はNone """
    if np.__name__ != 'nlcpy':
        return None
    dtype = str(np.dtype(dtype))
    if dtype not in _kernels:
        try:
            _kernels[dtype] = VEKernels(np, dtype)
        except Exception as e:   # ncc が無いなど
            warnings.warn(f'VE kernels are not available ({e}); using array operations.')
            _kernels[dtype] = None
    return _kernels[dtype]
