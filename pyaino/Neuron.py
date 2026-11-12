# Neuron
# 20261010 A.Inoue

import copy
import warnings
import math
from pyaino.Config import *
from pyaino.nucleus import Function, CompositFunction
from pyaino import Activators
from pyaino import Optimizers
from pyaino import common_function as cf
from pyaino import LossFunctions as lf
from pyaino import Functions as F
from pyaino import safe_np as snp
from pyaino import Regularizers
from pyaino.Initializer import init_weight

class Sequential:
    """ 複数の層を積み上げて一括して扱う """
    def __init__(self, *layers, **kwargs):
        self.layers = [layer for layer in layers]
        self.error_layer = None
        self.outputshape = {}
        #print(self.layers)

    def forward(self, *x, **kwargs):
        layer = self.layers[0]
        self.error_layer = layer
        y = layer.forward(*x, **kwargs)
        self.outputshape['0'+layer.__class__.__name__] = y.shape
        for i, layer in enumerate(self.layers[1:], start=1):
            self.error_layer = layer
            y = layer.forward(y, **kwargs)
            self.outputshape[str(i)+layer.__class__.__name__] = y.shape # デバグ用
        self.error_layer = None    
        return y

    def backward(self, gy=None):
        self.error_layer = self.layers[-1]
        if gy is None:
            gx = self.layers[-1].backward()
        else:
            gx = self.layers[-1].backward(gy)
        for layer in reversed(self.layers[:-1]):
            self.error_layer = layer
            gx = layer.backward(gx)
        self.error_layer = None    
        return gx

    def update(self, eta=0.001, **kwargs):
        for layer in self.layers:
            if hasattr(layer, 'update'): 
                layer.update(eta=eta, **kwargs)
                
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def summary(self):
        for layer in self.layers:
            print(layer.__class__.__name__, end=' ')
        if hasattr(layer, 'config'):
            print(layer.config)
        else:
            print('\n')

class SequentialWithLoss:
    """
    最後の層をLinearLayerCrossEntropyまたは損失関数として、
    複数の層を積み上げて一括して扱う
    """
    def __init__(self, *layers, **kwargs):
        self.layers = [layer for layer in layers]

        if isinstance(self.layers[-1], lf.LossFunctionBase):
            self.type = 0 # 通常の損失関数
        elif isinstance(self.layers[-1], LinearLayerCrossEntropy):
            self.type = 1 # LLCE
        else:    
            raise TypeError("Last layer must be either",
                            "a subclass of LossFunctions or LinearLayerCrossEntropy",
                            f"but {type(layers[-1]).__name__}")
        self.error_layer = None
        self.outputshape = {}
        #print(self.layers)

    def forward(self, *x, **kwargs):
        # x には通常入力・res接続・正解値が位置引数として渡されることがある。
        # 本来は layer.n_inputs(未実装)で判定すべきだが、
        # 現状は res接続の有無を「2入力層」の暫定的な指標とする => TODO
        layer = self.layers[0]
        self.error_layer = layer
        t = kwargs.pop('t', None)

        # res接続の有無=>layer.n_inputs の代用
        use_residual = getattr(layer, 'use_residual', False) 

        if not use_residual and len(x) >= 2:
            # 通常層では最後の位置引数を教師値とみなして切り出す
            x, t = x[:-1], x[-1]

        y = layer.forward(*x, **kwargs)
        
        self.outputshape['0'+layer.__class__.__name__] = y.shape
        # 最終層以前まで
        for i, layer in enumerate(self.layers[1:-1], start=1): 
            self.error_layer = layer                                    # デバグ用
            y = layer.forward(y, **kwargs)
            self.outputshape[str(i)+layer.__class__.__name__] = y.shape # デバグ用
        # 以下、最終層の扱い
        self.error_layer = self.layers[-1]
        if self.type==0:  # 通常の損失関数
            if t is None:    
                return y
            l = self.layers[-1].forward(y, t)
            self.error_layer = None 
            return y, l
        if self.type==1:  # LLCE
            if t is None:
                y = self.layers[-1].forward(y)
                self.error_layer = None
                return y
            y, l = self.layers[-1](y, t)
            self.error_layer = None 
            return y, l

    def backward(self, *args, **kwargs): # 外部の勾配には未対応20260414AI
        self.error_layer = self.layers[-1]
        if self.type==0:  # 通常の損失関数
            if len(args) == 0:
                gx = self.layers[-1].backward()
            else:
                gy, = args # 仮対処20260615AI 
                gx = gy + self.layers[-1].backward()
        if self.type==1:  # LLCE
            gx = self.layers[-1].backward()
        for layer in reversed(self.layers[:-1]):
            self.error_layer = layer
            gx = layer.backward(gx)
        self.error_layer = None    
        return gx

    def update(self, eta=0.001, **kwargs):
        for layer in self.layers:
            if hasattr(layer, 'update'): 
                layer.update(eta=eta, **kwargs)
                
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def summary(self):
        for layer in self.layers:
            print(layer.__class__.__name__, end=' ')
        if hasattr(layer, 'config'):
            print(layer.config)
        else:
            print('\n')

def wrap_layer_instance(layer): # 作りかけ200260124AI　
    base_cls = layer.__class__

    if hasattr(base_cls, 'forward'):
        print('Already has forward method.')
        return layer

    # 追加機能を定義する辞書
    extra_methods = {}

    def wrapped_forward(self, *args, **kwargs):
        return super(Wrapped, self).__call__(*args, **kwargs)

    extra_methods['forward'] = wrapped_forward

    # base_cls を継承した新しいクラスを動的に作成
    Wrapped = type(
        f"Wrapped{base_cls.__name__}",
        (base_cls,),
        extra_methods
    )

    # 新しいインスタンスを作り、元の __dict__ をコピー
    new_obj = Wrapped.__new__(Wrapped)
    new_obj.__dict__ = layer.__dict__.copy()

    return new_obj

class WrapForSequential:
    ''' Sequentialの要件を満たさない層の手当をするラッパー '''
    def __init__(self, layer):
        self.layer = layer
        # layerの素性の表示
        print(f'start {layer.__class__.__name__}')
        methods = inspect.getmembers(layer, predicate=inspect.ismethod)
        self.exclude = 'train'
        for m in methods:
            #print(m[0])
            sig = inspect.signature(m[1])  
            sigdic = sig.parameters # 引数のOrderedDict
            print(f' {m[0]}{sig}')

            if m[0] in ('forward', '__call__'):  
                if self.exclude in sig.parameters:
                    self.sig_exclude = False # 除外しなくてよい
                else:
                    self.sig_exclude = True  # 除外する必要あり
                print(f' exclude {self.exclude} {self.sig_exclude}')
        print(f'end   {layer.__class__.__name__}')   

    def forward(self, x, **kwargs):
        if self.sig_exclude: # kwargsからself.excludedを除外したものを用意　
            kwargs_n = {k: v for k, v in kwargs.items() if k not in self.exclude}
        else:                # そのままkwargsを渡す 
            kwargs_n = kwargs
        if hasattr(self.layer, 'forward'):
            return self.layer.forward(x, **kwargs_n)
        elif hasattr(self.layer, '__call__'):
            return self.layer(x, **kwargs_n)

    def backward(self, gy):
        if hasattr(self.layer, 'backward'):
            return self.layer.backward()

    def update(self, eta=0.001, **kwargs):
        if hasattr(self.layer, 'update'):
            return self.layer.update(eta=eta, **kwargs)
    
class Sequential2:
    """ 複数の層を積み上げて一括して扱う """
    def __init__bkup(self, *layers, **kwargs):
        self.layers = [l for l in layers]
        print(self.layers)

    def __init__(self, *layers, **kwargs):
        self.layers = []
        for l in layers:
            if hasattr(l, 'forward') and hasattr(l, 'backward') and hasattr(l, 'update'):
                self.layers.append(l)
            else: # Sequentialに必要なメソッドを備えていない場合
                l = WrapForSequential(l) # ラッパーでごまかす
                #l = wrap_layer_instance(l)
                self.layers.append(l)
        print(self.layers)

    def forward(self, x, **kwargs):
        y = x
        for l in self.layers:
            y = l.forward(y, **kwargs)
        return y

    def backward(self, gy=None):
        if gy is None:
            gx = self.layers[-1].backward()
        else:
            gx = self.layers[-1].backward(gy)
        for l in reversed(self.layers[:-1]):
            gx = l.backward(gx)
        return gx

    def update(self, eta=0.001, **kwargs):
        for l in self.layers:
            if hasattr(l, 'update'): 
                l.update(eta=eta, **kwargs)
                
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def summary(self):
        for l in self.layers:
            print(l.__class__.__name__, end=' ')
        if hasattr(l, 'config'):
            print(l.config)
        else:
            print('\n')

#### ニューロンの基本機能 ##############################################
class Parameter(Function):
    """ 学習可能な簡易パラメタ """
    def __init__(self, *size, **kwargs):
        super().__init__()
        self.size = size
        optimize = kwargs.pop('optimize', 'SGD') 
        self.w, self.grad_w = None, None
        self.optimizer_w = cf.eval_in_module(optimize, Optimizers, **kwargs)

    def __forward__(self):
        if self.w is None:
            self.init_parameter()
        return self.w

    def __call__(self):
        return self.forward()

    def __backward__(self, gy):
        self.grad_w = gy

    def init_parameter(self):
        self.w = np.random.randn(*self.size)

    def update(self, eta=0.001, **kwargs):
        self.optimizer_w.update(self.w, self.grad_w, eta=eta, **kwargs)

#### ニューロンの基本機能 ##############################################
class WeightsAndBiases:
    """ linear変換のパラメータ管理 """
    def __init__(self, layer, **kwargs):
        print('Initialize', self.__class__.__name__)
        self.layer      = layer
        self.bias       = kwargs.pop('bias',        True) # dot_linearのbias有無
        optimize        = kwargs.get('optimize',   'SGD') # kwargsに残してBNに渡す
        self.width      = kwargs.pop('width',       None) # 重みの初期値の広がりを指定
        self.debug_mode = kwargs.pop('debug_mode', False) # 重みを一律に初期化
        self.scale      = kwargs.pop('scale',      False) # ReParameteraization

        self.w, self.b, self.gamma = None, None, None
        self.grad_w, self.grad_b, self.ggamma = None, None, None
        self.w_bkup, self.b_bkup, self.gamma_bkup = None, None, None

        self.optimizer_w = cf.eval_in_module(optimize, Optimizers, **kwargs)  # 最適化関数
        if self.bias:
            self.optimizer_b = \
                cf.eval_in_module(optimize, Optimizers, bias=True, **kwargs) # w_decayなどの対象外
        if self.scale:
            self.optimizer_g = \
                cf.eval_in_module(optimize, Optimizers, bias=True, **kwargs) # w_decayなどの対象外

    def __call__(self):
        if self.w is None:
            self.init_parameter()
        return self.w, self.b, self.gamma

    def init_parameter(self):
        m, n = self.layer.get_parameter_size()
        if m is None or n is None:
            raise Exception('Configuration is not fixed.', self.__class__.__name__)
        if hasattr(self.layer, 'activator'):
            activator = self.layer.activator
        else:
            activator = None
        self.w = init_weight((m, n),
                             width=self.width,
                             activator=activator,
                             debug_mode=self.debug_mode)        
        if self.bias:
            self.b = np.zeros(n, dtype=Config.dtype)
        if self.scale:
            self.gamma = np.array(1.0, dtype=Config.dtype)
        
    def update(self, eta=0.001, bkup=False, **kwargs):
        if bkup:
            self.backup()
        self.optimizer_w.update(self.w, self.grad_w, eta, **kwargs) # 戻り値=更新量
        if self.bias:
            self.optimizer_b.update(self.b, self.grad_b, eta, **kwargs) # 戻り値=更新量
        if self.scale:
            self.optimizer_g.update(self.gamma, self.ggamma, eta, **kwargs)
        
    def backup(self):
        if self.w_bkup is not None:
            self.w_bkup[...] = self.w
        else:
            self.w_bkup = copy.deepcopy(self.w)
        if self.bias:
            if self.b_bkup is not None:
                self.b_bkup[...] = self.b
            else:
                self.b_bkup = copy.deepcopy(self.b)
        if self.scale:        
            if self.gamma_bkup is not None:
                self.gamma_bkup[...] = self.gamma
            else:
                self.gamma_bkup = copy.deepcopy(self.gamma)
        
    def recover(self):
        self.w[...] = self.w_bkup
        if self.bias:
            self.b[...] = self.b_bkup
        if self.scale:
            self.gamma[...] = self.gamma_bkup


    def accommodate(self):
        m, n = self.layer.get_parameter_size()
        if n <= self.w.shape[1]:
            return
        print(self.__class__.__name__,
              'expand the size of w to accommodate new vocabulary.')
        xpcn = n - self.w.shape[1] # 拡張する列数
        center_w = np.mean(self.w, axis=1, keepdims=True)
        new_colums = center_w \
                   + np.random.normal(0, 0.01, size=(m, xpcn), dtype=Config.dtype)
        print('new colums of w =', new_colums.shape)
        self.w = np.concatenate([self.w, new_colums], axis=1)
        if self.bias:
            center_b = np.mean(self.b)
            new_bias = center_b \
                     + np.random.normal(0, 0.01, size=(xpcn,), dtype=Config.dtype)
            print('new bias =', new_bias.shape)
            self.b = np.concatenate([self.b, new_bias])

    def set_gradient(self, *grads, flush=True):
        if flush:
            self.grad_w = grads[0]
        else:
            self.grad_w += grads[0]
        if self.bias:
            if flush:
                self.grad_b = grads[1]
            else:
                self.grad_b += grads[1]
        if self.scale:
            if flush:
                self.ggamma = grads[-1]
            else:
                self.ggamma += grads[-1]

    def flush_gradient(self):
        self.grad_w = np.zeros_like(self.w, dtype=Config.dtype)
        if self.bias:
            self.grad_b = np.zeros_like(self.b, dtype=Config.dtype)
        if self.scale:
            self.ggamma = np.array(1.0, dtype=Config.dtype)

#### ニューロンの基本機能 ##############################################
class LinearLayer(Function):
    """ ニューロンの基本機能(Pytorch互換機能提供) """
    def __init__(self, *configuration, **kwargs):
        super().__init__()
        if   len(configuration) == 2:
            m, n = configuration
        elif len(configuration) == 1:
            m = None; n, = configuration
        else:
            m, n = None, None
        self.config = m, n                                # m:入力幅、n:ニューロン数
        print('Initialize', self.__class__.__name__, self.config)
        self.matmul     = kwargs.pop('matmul',     False) # MatMulLinearを使う
        self.bias       = kwargs.get('bias',        True) # dot_linearのbias有無
        self.scale      = kwargs.get('scale',      False) # ReParameteraization
        self.parameters = WeightsAndBiases(self, **kwargs)
        self.dot_linear = F.ScaleDotLinear(self.matmul, self.bias, self.scale)
        self.prephase   = PrePhase(self, **kwargs) 

    def update(self, eta=0.001, **kwargs):
        self.parameters.update(eta=eta, **kwargs) 
        self.prephase.update(eta=eta, **kwargs)
        
    def fix_configuration(self, shape):
        if self.matmul:
            m = shape[-1]
        else:
            m = 1
            for i in shape[1:]:                       # バッチ軸以外の積
                m *= i
        self.config = m, self.config[1]
        print(self.__class__.__name__, 'fix_configuration', shape, self.config)

    def __forward__(self, x, **kwargs):           # kwargsは使わない
        if None in self.config:
            #print(self.__class__.__name__, 'input.shape', x.shape)
            self.fix_configuration(x.shape)
        x = self.prephase.forward(x)
        w, b, gamma = self.parameters()    
        y = self.dot_linear.forward(x, w, b, gamma)
        return y 
        
    def __backward__(self, grad_y):
        grad_x, grad_w, grad_b, ggamma = self.dot_linear.backward(grad_y)
        self.parameters.set_gradient(grad_w, grad_b, ggamma) 
        grad_x = self.prephase.backward(grad_x)
        return grad_x

    def get_parameter_size(self):
        return self.config

    def accommodate(self):
        self.parameters.accommodate()
            
#### ニューロンの基本機能 ##############################################
class LinearLayerCrossEntropy(LinearLayer):
    """ Softmaxそして損失まで一気に算出するニューロンの基本機能 """
    def __init__(self, *configuration, **kwargs):
        super().__init__(*configuration, **kwargs)
        self.tile_size = kwargs.pop('tile_size', None) #
       
    def __forward__(self, x, t=None, **kwargs):       # kwargsは使わない
        self.x, self.t = x, t
        if None in self.config:
            #print(self.__class__.__name__, 'input.shape', x.shape)
            self.fix_configuration(x.shape)
        x = self.prephase.forward(x)
        w, b, gamma = self.parameters()

        m, n = self.config
        if self.tile_size is None:
            self.tile_size = n

        leading_shape = x.shape[:-1] # (B,)や(B,T)
        self.leading_size = 1.0      # leading_shapeの積
        for i in leading_shape:
            self.leading_size *= i 
        
        # 全体の最大logits値とその位置の初期値(バッチサイズ分並べる)
        max_logit = np.full(leading_shape, -np.inf, dtype=x.dtype) # 現時点の最大logit
        max_index = np.full(leading_shape, -1)                     # その語彙ID
        sum_exp = np.zeros(leading_shape, dtype=x.dtype)           # 逐次 exp 累積
        if t is not None:
            zt = np.zeros(leading_shape, dtype=x.dtype)            # 正解値の指すlogit

        for start in range(0, n, self.tile_size):
            # タイル毎にlogitsを算出
            end = min(start + self.tile_size, n)
            tile_w = w[:, start:end]
            tile_b = b[start:end]
            tile_z = self.dot_linear.forward(x, tile_w, tile_b, gamma)      # (B, Vt)

            last_max_logit = max_logit.copy() # 更新前のmax_logit  

            # タイル内の最大位置を求め、そのlogits値を得る
            tile_max_index = np.argmax(tile_z, axis=-1)            # (B,)
            tile_max_logit = (
              np.take_along_axis(tile_z,tile_max_index[...,None],axis=-1).squeeze(-1))

            # 全体最大を更新(タイルの最大が全体の最大より大きいものについて処理)
            mask = tile_max_logit > max_logit
            if mask.any():
                max_logit[mask] = tile_max_logit[mask]
                max_index[mask] = start + tile_max_index[mask]  # 全体での位置=語彙ID 
            # 最新と以前の最大値の補正をしながらsum_exp を更新
            sum_exp = (sum_exp * np.exp(last_max_logit - max_logit) # 補正項　
                    + np.sum(np.exp(tile_z - max_logit[..., None]), axis=-1)) # 更新値
            # tがtileに含まれる場合だけztを更新(zt:正解値tの指すlogit)
            if t is not None:
                t_in_tile = (start <= t) & (t < end)
                if t_in_tile.any():
                    tile_t = np.where(t_in_tile, t - start, 0).astype(np.int32)
                    tile_zt = F.take_along_axis(
                        tile_z, tile_t[..., None], axis=-1).squeeze(-1)
                    zt = np.where(t_in_tile, tile_zt, zt)

        # 予測だけ欲しい（推論）場合
        if t is None:
            return max_index, max_logit

        # 逆伝播用に保存
        self.sum_exp = sum_exp
        self.max_logit = max_logit

        # 確定した値で損失計算
        log_sum_exp = np.log(sum_exp) + max_logit     # 補正項
        loss = log_sum_exp - zt                       # CrossEntropy算出
        loss = np.sum(loss) / self.leading_size       # 平均
        # 学習時の返りは慣習的に (pred, loss) にしておく
        return max_index, loss
        
    def __backward__(self, *args): # argsは使わない
        m, n = self.config
        x, t = self.x, self.t
        w, b, gamma = self.parameters()

        grad_w = np.zeros_like(w)
        grad_b = np.zeros_like(b)
        grad_x = np.zeros_like(x)
        ggamma = np.zeros_like(gamma) if self.scale else None

        for start in range(0, n, self.tile_size):
            # タイル毎にlogitsを算出
            end = min(start + self.tile_size, n)
            tile_w = w[:, start:end]
            tile_b = b[start:end]
            tile_z = self.dot_linear.forward(x, tile_w, tile_b, gamma) # (B, Vt)

            # Softmaxでlogit->確率 
            tile_y = np.exp(tile_z - self.max_logit[...,None]) / self.sum_exp[...,None]

            # targetがtileに含まれる位置へ-1を加える
            t_in_tile = (start <= t) & (t < end)
            tile_gz = tile_y
            if t_in_tile.any():
                tile_t = np.where(t_in_tile, t - start, 0).astype(np.int32)
                correction = -t_in_tile.astype(Config.dtype)[..., None]
                tile_gz = F.scatter_add_along_axis(
                    correction, tile_t[..., None], tile_y.shape, axis=-1)
                tile_gz += tile_y
            tile_gz /= self.leading_size    # 順伝播のloss/leading_sizeに合わせる
              
            # dot_linearの逆伝播
            tile_gx, tile_gw, tile_gb, tile_gg = self.dot_linear.backward(tile_gz)    
            grad_x += tile_gx
            grad_w[:, start:end] = tile_gw
            if self.bias:
                grad_b[start:end] = tile_gb
            if self.scale:
                ggamma += tile_gg
                
        self.parameters.set_gradient(grad_w, grad_b, ggamma)        
        grad_x = self.prephase.backward(grad_x)
        return grad_x


#### ニューロン関連共通部分 ##############################################

class BaseLayer(Function):
    """ ニューロンの基本機能 """
    
    # 派生クラスの分類一覧を文字列で与える
    category_names = {
        0: ("NeuronLayer",),
        1: ("Conv1dLayer", "Conv1dTransposeLayer", "DeConv1dLayer"),
        2: ("Conv2dLayer", "Conv2dTransposeLayer", "ConvLayer",
            "DeConvLayer", "DeConv2dLayer", "MaskedExpansionLayer"),
    }
    categories = {} # 実際のクラスオブジェクトの格納場所

    @classmethod
    def resolve_categories(cls, namespace):
        """ 文字列 → クラスオブジェクトに変換 """
        for typeid, names in cls.category_names.items():
            cls.categories[typeid] = tuple(
                namespace[name] for name in names if name in namespace
                )

    def __init__(self, **kwargs):
        super().__init__()
        print('Initialize', self.__class__.__name__, self.config)
        self.matmul       = kwargs.pop('matmul',          False) # MatMulLinearを使う 
        self.bias         = kwargs.get('bias',             True) # dot_linearのbias有無
        self.scale        = kwargs.get('scale',           False) # ReParameteraization
        self.full_cnnt    = kwargs.pop('full_connection', False) # 全結合層を明示
        pre_activation    = kwargs.pop('pre_activation',  False) # pre/post切替
        activate          = kwargs.pop('activate',         None) # pre/post phaseに渡す
        self.use_residual = kwargs.pop('residual',        False) # 残差接続の有無
                                                                 # postphaseにも渡す
        self.parameters = WeightsAndBiases(self, **kwargs)
        self.dot_linear = F.ScaleDotLinear(self.matmul, self.bias, self.scale)

        # カテゴリ辞書はここで作る
        if not BaseLayer.categories:
            BaseLayer.resolve_categories(globals())

        # 自分がどの分類に属するか判定
        for typeid, classes in BaseLayer.categories.items():
            if isinstance(self, classes):
                self.typeid = typeid
                print(f"{self.__class__.__name__} is a subclass of BaseLayer",
                      f"and typeid={typeid}.")
                break
        else:
            raise Exception("Unsupported class uses BaseLayer.")

        # 自分のtypeid確定後にインスタンス化
        activate_pre  = activate if pre_activation else None 
        activate_post = None if pre_activation else activate 
        self.prephase   = PrePhase(self,  activate=activate_pre,  **kwargs) 
        self.postphase  = PostPhase(self, activate=activate_post, **kwargs)

        # サブクラス側で確定したself.configによる
        if self.typeid == 0:
            shape = self.config[:1]
        elif self.typeid == 1:
            shape = self.config[:2]
        elif self.typeid == 2:
            shape = self.config[:3]
        else:
            raise ValueError(f'Bad typeid {self.typeid}')
        # この層が内部で扱うxの形状を指定した場合にその情報を保持
        self.canonical_x_shape = (-1,) + shape if None not in shape else None

        self.original_x_shape  = None # この層が受け取るxの元の形状
        self.canonical_y_shape = None # この層で処理したままのyの内部的な形状
        self.final_y_shape     = None # この層が外部に出す最終的なyの形状
        self.did_reshape_x = None
        self.did_reshape_y = None
        self.feature_axis_preserved = None # 特徴次元を保っているかを示す

    def fix_configuration(self, shape):
        raise NotImplementedError('fix_configuration method for BaseLayer')
        
    def update(self, eta=0.001, **kwargs):
        self.prephase.update(eta=eta, **kwargs)
        self.parameters.update(eta=eta, **kwargs)
        self.postphase.update(eta=eta, **kwargs)
        
    def flush_gradient(self):
        self.parameters.flush_gradient()

    def backup(self):
        self.parameters.backup()
        
    def recover(self):
        self.parameters.recover()
        
    def align_config_and_input(self, x):
        self.original_x_shape = x.shape
        if None in self.config:
            self.fix_configuration(x.shape)

        if self.canonical_x_shape is None \
           or x.shape[1:] == self.canonical_x_shape[1:]:
            self.did_reshape_x = False
            self.feature_axis_preserved = True
            return x

        self.did_reshape_x = True
        self.feature_axis_preserved = (x.shape[-1]==self.config[0]) # True/False
        return x.reshape(*self.canonical_x_shape)   # (-1,m)

    def align_output(self, y):
        # 仮設定(そのままor後で上書き)
        self.final_y_shape     = y.shape 
        self.canonical_y_shape = y.shape
        self.did_reshape_y = False
        # Convなどは出力は整形しない(そのまま)
        if self.typeid != 0:
            return y
        
        # 以下はtypeid==0(NeuronLayer)の場合 
        if self.full_cnnt or not self.did_reshape_x: # 整形をしない場合
            return y
        # 元のxの形状に応じて最終的なyの形状を決める
        if self.did_reshape_x and self.feature_axis_preserved:
            m, n = self.config
            self.final_y_shape = (-1,) + self.original_x_shape[1:-1] + (n,)
            self.did_reshape_y = True
            return y.reshape(self.final_y_shape)
        
        raise ValueError(
          f"x.shape={self.original_x_shape} bad. May need to set 'full_connection=True'.")
            

    def __forward__(self, x, residual=None, *, train=False, dropout=0.0):
        # 残差接続の有無と入力の対応チャック
        if self.use_residual:
            if residual is None:
                raise ValueError("use_residual=True, but residual is None.")
        else:
            if residual is not None:
                raise ValueError(f"use_residual=False, but residual={residual.shape} is given.")

        # 前処理
        x = self.align_config_and_input(x)
        x = self.prephase.forward(x)
        # コアの順伝播
        y = self._forward(x)
        # 後処理
        y = self.postphase.forward(y, residual=residual, train=train, dropout=dropout)
        y = self.align_output(y)
        return y    
            

    def __backward__(self, grad_y, **kwargs):
        if self.did_reshape_y: 
            m, n = self.config
            grad_y = grad_y.reshape(-1, n)

        # 後処理の逆伝播
        grad_y, grad_r = self.postphase.backward(grad_y, **kwargs)
        # コアの逆伝播
        grad_x = self._backward(grad_y)
        # 前処理の逆伝播
        grad_x = self.prephase.backward(grad_x)
        
        if self.did_reshape_x:
            grad_x = grad_x.reshape(*self.original_x_shape)
            
        if not self.use_residual:
            return grad_x
        return grad_x, grad_r

    def get_parameter_size(self):    
        raise Exception('Invalid configuration')
    
class PrePhase(Function):
    def __init__(self, layer=None, **kwargs):
        super().__init__()
        #print(self.__class__.__name__, 'layer =', layer, 'kwargs =', kwargs)
        self.layer      = layer
        batchnorm       = kwargs.pop('batchnorm',  False) # バッチ正規化の適用有無
        layernorm       = kwargs.pop('layernorm',  False) # 層正規化の適用有無
        normdim         = kwargs.pop('normdim',     None) # 正規化の軸指定(バイアス2d)
        normaffine      = kwargs.pop('normaffine', False) # 正規化のスケール＆バイアス
        activate        = kwargs.pop('activate',    None) # Pre-activation
        norm_option     = kwargs.copy()                   # 残りは正規化のオプション
        activate_option = kwargs.copy()                   # 残りは活性化のオプション

        use_batchnorm = batchnorm is True and activate is not None
        use_layernorm = (layernorm == 'pre') \
                     or (layernorm is True and activate is not None)

        if layer is None:
            self.Norm = None

        elif use_batchnorm:
            if layer.typeid == 2 or normdim == '2d': 
                self.Norm = BatchNorm2d(scale_and_bias=normaffine, **norm_option)
            elif layer.typeid == 1 or normdim == '1d':
                self.Norm = BatchNorm1d(scale_and_bias=normaffine, **norm_option)
            else: # layer.typeid == 0 or normdim == '0d') 
                self.Norm = BatchNormalization(scale_and_bias=normaffine, **norm_option)

        elif use_layernorm:
            if layer.typeid == 2 or normdim == '2d': 
                self.Norm = LayerNorm2d(scale_and_bias=normaffine, **norm_option)
            elif layer.typeid == 1 or normdim == '1d':
                self.Norm = LayerNorm1d(scale_and_bias=normaffine, **norm_option)
            else: # layer.typeid == 0 or normdim == '0d') 
                self.Norm = LayerNormalization(scale_and_bias=normaffine, **norm_option)

        else:
            self.Norm = None
            
        self.activator = cf.eval_in_module(activate, Activators, **activate_option) \
                         if activate is not None else None
        
    def __forward__(self, x, *, train=False):
        if self.Norm:
            x = self.Norm.forward(x, train=train)      # バッチor層ノーマライゼーション
        if self.activator:
            x = self.activator.forward(x)
        return x
        
    def __backward__(self, grad_x, **kwargs):
        if self.activator:
            grad_x = self.activator.backward(grad_x)
        if self.Norm:
            grad_x = self.Norm.backward(grad_x)        # バッチor層ノーマライゼーション
        return grad_x

    def update(self, eta=0.001, **kwargs):
        if self.Norm:
            self.Norm.update(eta=eta, **kwargs)

class PostPhase(Function):  
    def __init__(self, layer=None, **kwargs):
        super().__init__()
        #print(self.__class__.__name__, 'layer =', layer, 'kwargs =', kwargs)
        activate          = kwargs.pop('activate',    None) # Post-activation
        dropout           = kwargs.pop('dropout',    False) # ドロップアウト可否(forwardで指定)
        batchnorm         = kwargs.pop('batchnorm',  False) # バッチ正規化の適用有無
        layernorm         = kwargs.pop('layernorm',  False) # 層正規化の適用有無
        normdim           = kwargs.pop('normdim',     None) # 正規化の軸指定(2d)
        normaffine        = kwargs.pop('normaffine', False) # 正規化のスケール＆バイアス
        activate_option   = kwargs.copy()                   # 残りは活性化のオプション
        norm_option       = kwargs.copy()                   # 残りは正規化のオプション

        self.activator = cf.eval_in_module(activate, Activators, **activate_option) \
                         if activate is not None else None

        use_batchnorm = batchnorm is True and activate is not None
        use_layernorm = layernorm is True and activate is not None

        if layer is None:
            self.Norm = None
            
        elif use_batchnorm:
            if layer.typeid == 2 or normdim == '2d': 
                self.Norm = BatchNorm2d(scale_and_bias=normaffine, **norm_option)
            elif layer.typeid == 1 or normdim == '1d':
                self.Norm = BatchNorm1d(scale_and_bias=normaffine, **norm_option)
            else: # layer.typeid == 0 or normdim == '0d') 
                self.Norm = BatchNormalization(scale_and_bias=normaffine, **norm_option)
                
        elif use_layernorm:
            if layer.typeid == 2 or normdim == '2d': 
                self.Norm = LayerNorm2d(scale_and_bias=normaffine, **norm_option)
            elif layer.typeid == 1 or normdim == '1d':
                self.Norm = LayerNorm1d(scale_and_bias=normaffine, **norm_option)
            else: # layer.typeid == 0 or normdim == '0d') 
                self.Norm = LayerNormalization(scale_and_bias=normaffine, **norm_option)
               
        else:
            self.Norm = None

        self.DO = Dropout() if dropout else None

    def update(self, eta=0.001, **kwargs):
        if self.Norm:
            self.Norm.update(eta=eta, **kwargs)
        
    def __forward__(self, y, residual=None, *, train=False, dropout=0.0):
        if self.Norm:
            y = self.Norm.forward(y, train=train)   # バッチor層ノーマライゼーション

        if residual is not None:                       
            y = y + residual # 加算点（与えられたものはそのまま加える）
            
        if self.activator: 
            y = self.activator.forward(y)           # 活性化関数

        if self.DO:
            y = self.DO.forward(y, dropout=dropout) # ドロップアウト
        return y    
        
    def __backward__(self, grad_y, **kwargs):
        if self.DO:
            grad_y = self.DO.backward(grad_y)       # ドロップアウト
        if self.activator:    
            grad_y = self.activator.backward(grad_y, **kwargs) # 活性化関数
            
        grad_r = grad_y      # 加算点のもう一方の枝への局所勾配をそのまま返す

        if self.Norm:
            grad_y = self.Norm.backward(grad_y)     # バッチor層ノーマライゼーション
        return grad_y, grad_r

#### 基本的な Neuron層 ##############################################
# m:上流のニューロン数、n:自身のニューロン数、activate:活性化関数、optimize:最適化、
# eta:学習係数、width:広がり係数、loss_f:損失関数、w_decay:L2正則化項の係数
class NeuronLayer(BaseLayer): # ニューロンの基本機能 
    def __init__(self, *configuration, **kwargs):
        if   len(configuration) == 2:
            m, n = configuration
        elif len(configuration) == 1:
            m = None; n, = configuration
        else:
            m, n = None, None
        self.config = m, n
        super().__init__(**kwargs)

    def fix_configuration(self, shape):
        if self.matmul:
            m = shape[-1]
        else:
            m = 1
            for i in shape[1:]:                       # バッチ軸以外の積
                m *= i
        self.config = m, self.config[1]
        print(self.__class__.__name__, 'fix_configuration', shape, self.config)

    def get_parameter_size(self):
        m, n = self.config
        return m, n

    def _forward(self, x):
        w, b, gamma = self.parameters()
        y = self.dot_linear.forward(x, w, b, gamma)
        return y 
        
    def _backward(self, grad_y, flush=True):
        grad_x, grad_w, grad_b, ggamma = self.dot_linear.backward(grad_y)
        self.parameters.set_gradient(grad_w, grad_b, ggamma, flush=flush)
        return grad_x

    def accommodate(self):
        self.parameters.accommodate()
    



### 畳み込み層 #####################################################
class Conv1dLayer(BaseLayer):
    # B:バッチサイズ, C:入力チャンネル数, Iw:入力画像幅
    # M:フィルタ数, Fw:フィルタ幅
    # stride:ストライド幅, pad:パディング幅
    # 出力チャンネル数=フィルタ数M, Ow:出力幅
    # w_decay:L2正則化項の係数
    
    def __init__(self, *configuration, **kwargs):
        C, Iw, M, Fw, stride, pad, Ow = None, None, None, 3, 1, 1, None
        if len(configuration) == 6:
            C, Iw, M, Fw, stride, pad = configuration
        if len(configuration) == 4:
            M, Fw, stride, pad = configuration
        if len(configuration) == 3:
            M, Fw, pad = configuration
        if len(configuration) == 2:
            M, Fw = configuration
        if len(configuration) == 1:
            M, = configuration
        self.config = C, Iw, M, Fw, stride, pad, Ow
        self.vc = None
        super().__init__(**kwargs)
        
    def fix_configuration(self, shape):
        C, Iw, M, Fw, stride, pad, Ow = self.config
        if len(shape) >= 2:
            Iw = shape[-1] 
            C = shape[1] if len(shape)==3 else 1
        elif C is None or Iw is None:
            raise Exception(self.__class__.__name__ + ' cannot fix configuration.')
           
        Ow = (Iw - Fw + 2*pad) // stride + 1   # 出力幅
        self.config = C, Iw, M, Fw, stride, pad, Ow
        self.vc = VecCol(Fw, stride)

    def get_parameter_size(self):
        C, Iw, M, Fw, stride, pad, Ow = self.config
        m = C*Fw  # 入力チャネル数とフィルタサイズ
        n = M     # フィルタ数
        return m, n

    def _forward(self, x):
        B, C, Iw = x.shape
        self.x_shape = x.shape
        w, b, gamma = self.parameters()    
        #C, Iw, M, Fw, stride, pad, Ow = self.config
        _, _, M, Fw, stride, pad, _ = self.config
        Ow = (Iw - Fw + 2*pad) // stride + 1   # 出力幅
        #x = x.reshape(-1, C, Iw)    # (B,C,Iw)  
        # '0'パディング B軸    C軸  Iw左Iw右
        x = np.pad(x, [(0,0),(0,0),(pad,pad)])
        self.vec_shape = x.shape # パディング後の形状
        # 入力画像を行列に変換 (B,C,Iw+2*pad)->(C*Fw,B*Ow)
        cols = self.vc.vec2col(x)
        # linear変換: (B*Ow,C*Fw)×(C*Fw,M)->(B*Ow,M)
        y = self.dot_linear.forward(cols, w, b, gamma)
        y = y.reshape(B, Ow, M).transpose(0, 2, 1)       # u.shape=(B,M,Ow) 
        return y
    
    def _backward(self, grad_y, flush=True):
        _, _, M, Fw, stride, pad, _ = self.config
        B, C, Iw = self.x_shape 
        #grad_y = grad_y.reshape(-1, M, Ow)               # grad_y.shape=(B,M,Ow)
        grad_y = grad_y.transpose(0, 2, 1).reshape(-1, M) #grad_y.shape=(B*Ow,M)
        # linearの逆伝播 grad_cols.shape=(B*Ow,C*Fw)
        grad_cols, grad_w, grad_b, ggamma = self.dot_linear.backward(grad_y)
        # 行列を画像に変換 (B*Ow,C*Fw)->(B,C,Iw)
        grad_x = self.vc.col2vec(grad_cols)
        # パディング分を外して元の画像データに戻す        
        grad_x = grad_x[:,:,pad:pad+Iw]
        #grad_x = grad_x.reshape(self.inputs[0].shape)
        self.parameters.set_gradient(grad_w, grad_b, ggamma, flush=flush)
        return grad_x

### 転置畳込み層 #####################################################
class Conv1dTransposeLayer(BaseLayer):
    # B:バッチサイズ, C:入力チャンネル数, Ih:入力画像高さ, Iw:入力画像幅
    # M:フィルタ数, Fh:フィルタ高さ, Fw:フィルタ幅
    # stride:ストライド幅, pad:パディング幅
    # 出力チャンネル数=フィルタ数M, Oh:出力高さ, Ow:出力幅
    # w_decay:L2正則化項の係数
    
    def __init__(self, *configuration, **kwargs):
        C, Iw, M, Fw, stride, pad, Ow = None, None, None, 4, 2, 1, None 
        if len(configuration) == 6:
            C, Iw, M, Fw, stride, pad = configuration
        if len(configuration) == 4:
            M, Fw, stride, pad = configuration
        if len(configuration) == 3:
            M, Fw, pad = configuration
        if len(configuration) == 2:
            M, Fw = configuration
        if len(configuration) == 1:
            M, = configuration
        self.config = C, Iw, M, Fw, stride, pad, Ow
        self.vc = None
        super().__init__(**kwargs)
        

    def fix_configuration(self, shape):
        C, Iw, M, Fw, stride, pad, Ow = self.config
        if len(shape) >= 2:
            Iw = shape[-1] 
            C  = shape[1] if len(shape)==3 else 1
        elif C is None or Iw is None:
            raise Exception(self.__class__.__name__ + ' cannot fix configuration.')
        Ow = (Iw - 1) * stride + Fw - 2 * pad  # 出力幅
        self.config = C, Iw, M, Fw, stride, pad, Ow
        self.vc = VecCol(Fw, stride)
        print(self.__class__.__name__, 'fix_configuration', shape, self.config)


    def get_parameter_size(self):
        C, Iw, M, Fw, stride, pad, Ow = self.config
        m = C              # 入力チャネル数「要注意」
        n = M*Fw           # フィルタ数とフィルタサイズ「要注意」
        return m, n

    def _forward(self, x):
        B, C, Iw = x.shape
        self.x_shape = x.shape
        w, b, gamma = self.parameters()    
        _, _, M, Fw, stride, pad, _ = self.config
        Ow = (Iw - 1) * stride + Fw - 2 * pad  # 出力幅
        x = x.transpose(0, 2, 1).reshape(-1, C) # (B*Iw,C)  
        # linear変換 (B*Iw,C)×(C,M*Fw)->(B*Iw,M*Fw)   
        cols = self.dot_linear.forward(x, w, b, gamma)
        # 行列を画像に変換 cols.T:(M*Fw,B*Iw)->(B,M,Ow)
        y = self.vc.col2vec(cols, B)
        # 画像調整 トリミング
        y = y[:,:,pad:pad+Ow]                     # y.shape=(B,M,Ow)
        return y


    def _backward(self, grad_y, flush=True):
        _, _, M, Fw, stride, pad, _ = self.config
        B, C, Iw = self.x_shape 
        #  '0'パディング
        grad_y = np.pad(grad_y, [(0,0), (0,0), (pad, pad)])
        # 画像の勾配を行列に変換 grad_y.shape=(M*Fw,B*Iw)に変換
        grad_y = self.vc.vec2col(grad_y)
        # linearの逆伝播
        grad_x, grad_w, grad_b, ggamma = self.dot_linear.backward(grad_y)
        grad_x = grad_x.reshape(B, Iw, C).transpose(0, 2, 1) # (B,C,Iw)
        self.parameters.set_gradient(grad_w, grad_b, ggamma, flush=flush)
        return grad_x


class DeConv1dLayer(Conv1dTransposeLayer):
    def __init__(self, *args, **kwargs):
        msg = (
        'What is called Deconvolution is actually ConvTranspose, '
        'which is now considered a misuse of the term.'
        )
        print(msg)
        super().__init__(*args, **kwargs)

class VecCol:
    """
    vec.shape = (B, C, W)
        <->
    col.shape = (B*N, C*Fw)

    W  : vec側の幅
    N  : 窓の個数
    Fw : フィルタ幅
    """

    def __init__(self, Fw, stride=1):
        self.Fw = Fw
        self.stride = stride

        # 直前に扱った形状
        self.B = None
        self.C = None
        self.W = None
        self.N = None

    def vec2col(self, vec):
        B, C, W = vec.shape
        Fw = self.Fw
        stride = self.stride

        # 窓の個数
        N = (W - Fw) // stride + 1

        # 形状を記憶
        self.B = B
        self.C = C
        self.W = W
        self.N = N

        col = np.empty((B, C, Fw, N), dtype=Config.dtype)

        # vecからstride毎に窓を取り出す
        for fw in range(Fw):
            w_lim = fw + stride * N
            col[:, :, fw, :] = vec[:, :, fw:w_lim:stride]

        # (B,C,Fw,N) -> (B,N,C,Fw) -> (B*N,C*Fw)
        col = col.transpose(0, 3, 1, 2).reshape(B * N, C * Fw)

        return col

    def col2vec(self, col, B=None):
        Fw = self.Fw
        stride = self.stride

        # vec2colを先に通っていれば、そのBを使える
        if B is None:
            B = self.B

        # col2vecを先に使う場合はBだけ必要
        if B is None:
            raise Exception(
                self.__class__.__name__
                + '.col2vec requires B when called first.')

        # col.shape = (B*N, C*Fw) から N,C を求める
        N = col.shape[0] // B
        C = col.shape[1] // Fw

        # vec側の幅
        W = (N - 1) * stride + Fw

        # 形状を記憶
        self.B = B
        self.C = C
        self.W = W
        self.N = N

        # (B*N,C*Fw) -> (B,N,C,Fw) -> (B,C,Fw,N)
        col = col.reshape(B, N, C, Fw).transpose(0, 2, 3, 1)

        vec = np.zeros((B, C, W), dtype=Config.dtype)

        # 重なる部分は加算して戻す
        for fw in range(Fw):
            w_lim = fw + stride * N
            vec[:, :, fw:w_lim:stride] += col[:, :, fw, :]

        return vec

class Vec2col:
    """ vec.shape = (B, C, Iw) -> col.shape = (B*Ow, C*Fw) """
    def __init__(self, C, Iw, Fw, stride, Ow=None):
        # 出力画像のサイズ
        if Ow is None:
            Ow = (Iw - Fw) // stride + 1        # 出力幅
        # パラメータをまとめる(class内での変数受渡しのため)
        self.config = (C, Iw, Fw, stride, Ow)

    def __call__(self, vec):
        C, Iw, Fw, stride, Ow = self.config
        B = vec.size // (C*Iw)
        col = np.empty((B, C, Fw, Ow), dtype=Config.dtype)
                                                 # メモリ節約のためzerosでなくempty 
        # vecからstride毎のデータを取ってきて、colsにOwになるまで並べる
        # それをFh,Fwを満たすまで繰返す
        for fw in range(Fw):
            w_lim = fw + stride*Ow
            col[:,:,fw,:] = vec[:,:,fw:w_lim:stride]
        # 軸の入替と変形     B  Ow C  Fw 
        col = col.transpose(0, 3, 1, 2).reshape(B*Ow, C*Fw)
        return col

class Col2vec:
    """ col.shape = (B*Iw, C*Fw) -> vec.shape = (B, C, Ow)  """
    def __init__(self, C, Iw, Fw, stride, Ow=None):
        # 出力画像のサイズ
        if Ow is None:
            Ow = (Iw - 1) * stride + Fw 
        # パラメータをまとめる(class内での変数受渡しのため)
        self.config = (C, Iw, Fw, stride, Ow)

    def __call__(self, col):
        C, Iw, Fw, stride, Ow = self.config
        B = col.size // (C*Fw*Iw)
        col = col.reshape(B,Iw,C,Fw).transpose(0,2,3,1) # col.shape=(B,C,Fw,Iw)
        vec = np.zeros((B, C, Ow), dtype=Config.dtype)
        # colからstride,Ow個のデータを取ってきて、vecにstride毎に並べる
        # それをFh,Fwを満たすまで繰返す
        for fw in range(Fw):
            w_lim = fw + stride*Iw
            vec[:,:,fw:w_lim:stride] += col[:,:,fw,:]
        return vec


### 畳み込み層 #####################################################
class Conv2dLayer(BaseLayer):
    """ 二次元畳込み層 """
    # B:バッチサイズ,
    # C:入力チャンネル数, Ih:入力画像高さ, Iw:入力画像幅
    # M:フィルタ数, Fh:フィルタ高さ, Fw:フィルタ幅
    # Sh/_w:ストライド高さ/幅, pad:パディング幅
    # 出力チャンネル数=フィルタ数M, Oh:出力高さ, Ow:出力幅
    # w_decay:L2正則化項の係数
    
    def __init__(self, *configuration, **kwargs):
        C, image_size, M, kernel_size, stride, pad, Oh, Ow \
            = None, None, None, 3, 1, 1, None, None
        if len(configuration) == 6:
            C, image_size, M, kernel_size, stride, pad = configuration
        if len(configuration) == 4:
            M, kernel_size, stride, pad = configuration
        if len(configuration) == 3:
            M, kernel_size, pad = configuration
        if len(configuration) == 2:
            M, kernel_size = configuration
        if len(configuration) == 1:
            M, = configuration
        Ih, Iw = image_size if isinstance(image_size, (tuple, list)) \
                            else (image_size, image_size)
        Fh, Fw = kernel_size if isinstance(kernel_size, (tuple, list)) \
                             else (kernel_size, kernel_size)
        Sh, Sw = stride if isinstance(stride, (tuple, list)) else (stride, stride)
        self.config = C, Ih, Iw, M, Fh, Fw, Sh, Sw, pad, Oh, Ow
        #self.im2col = None
        #self.col2im = None
        self.imcol = None
        super().__init__(**kwargs)
        
    def fix_configuration(self, shape):
        C, Ih, Iw, M, Fh, Fw, Sh, Sw, pad, Oh, Ow = self.config
        if len(shape) >= 3:
            Ih = shape[-2] 
            Iw = shape[-1] 
            C = shape[1] if len(shape)==4 else 1
        elif C is None or Ih is None or Iw is None:
            raise Exception(self.__class__.__name__ + ' cannot fix configuration.')
            
        Oh = (Ih - Fh + 2*pad) // Sh + 1   # 出力高さ
        Ow = (Iw - Fw + 2*pad) // Sw + 1   # 出力幅
        self.config = C, Ih, Iw, M, Fh, Fw, Sh, Sw, pad, Oh, Ow
        self.ic = ImCol(Fh, Fw, Sh, Sw)
        print(self.__class__.__name__, 'fix_configuration', shape, self.config)

    def get_parameter_size(self):
        C, Ih, Iw, M, Fh, Fw, Sh, Sw, pad, Oh, Ow = self.config
        m = C*Fh*Fw  # 入力チャネル数とフィルタサイズ
        n = M        # フィルタ数
        return m, n

    def _forward(self, x):
        B, C, Ih, Iw = x.shape
        self.x_shape = x.shape
        w, b, gamma = self.parameters()    
        _, _, _, M, Fh, Fw, Sh, Sw, pad, _, _ = self.config
        Oh = (Ih - Fh + 2*pad) // Sh + 1   # 出力高さ
        Ow = (Iw - Fw + 2*pad) // Sw + 1   # 出力幅
        # '0'パディング
        x = np.pad(x, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
        # 入力画像を行列に変換 (B,C,Ih+2*pad,Iw+2*pad)->(C*Fh*Fw,B*Oh*Ow) 
        cols = self.ic.im2col(x)
        # linear変換: (B*Oh*Ow,C*Fh*Fw)×(C*Fh*Fw,M)->(B*Oh*Ow,M)
        y = self.dot_linear.forward(cols, w, b, gamma)
        y = y.reshape(B, Oh, Ow, M).transpose(0, 3, 1, 2) # u.shape=(B,M,Oh,Ow) 
        return y
    
    def _backward(self, grad_y, flush=True):
        _, _, _, M, Fh, Fw, Sh, Sw, pad, _, _ = self.config
        B, C, Ih, Iw = self.x_shape
        grad_y = grad_y.transpose(0, 2, 3, 1).reshape(-1, M) # grad_y.shape=(B*Oh*Ow,M)
        # linearの逆伝播 grad_cols.shape=(B*Oh*Ow,C*Fh*Fw)
        grad_cols, grad_w, grad_b, ggamma = self.dot_linear.backward(grad_y)
        # 行列を画像に変換 (B*Oh*Ow,C*Fh*Fw)->(B,C,Ih,Iw)
        grad_x = self.ic.col2im(grad_cols)
        # パディング分を外して元の画像データに戻す
        grad_x = grad_x[:,:,pad:pad+Ih,pad:pad+Iw]
        self.parameters.set_gradient(grad_w, grad_b, ggamma, flush=flush)
        return grad_x

class ConvLayer(Conv2dLayer):
    pass

### 転置畳み込み層 #####################################################
class Conv2dTransposeLayer(BaseLayer):
    """ 二次元転置畳込み層 """
    # B:バッチサイズ, C:入力チャンネル数, Ih:入力画像高さ, Iw:入力画像幅
    # M:フィルタ数, Fh:フィルタ高さ, Fw:フィルタ幅
    # Sh/_w:ストライド高さ/幅, pad:パディング幅
    # 出力チャンネル数=フィルタ数M, Oh:出力高さ, Ow:出力幅
    # w_decay:L2正則化項の係数
    
    def __init__(self, *configuration, **kwargs):
        C, image_size, M, kernel_size, stride, pad, Oh, Ow \
            = None, None, None, 4, 2, 1, None, None 
        if len(configuration) == 6:
            C, image_size, M, kernel_size, stride, pad = configuration
        if len(configuration) == 4:
            M, kernel_size, stride, pad = configuration
        if len(configuration) == 3:
            M, kernel_size, pad = configuration
        if len(configuration) == 2:
            M, kernel_size = configuration
        if len(configuration) == 1:
            M, = configuration
        Ih, Iw = image_size if isinstance(image_size, (tuple, list)) \
                            else (image_size, image_size)
        Fh, Fw = kernel_size if isinstance(kernel_size, (tuple, list)) \
                             else (kernel_size, kernel_size)
        Sh, Sw = stride if isinstance(stride, (tuple, list)) else (stride, stride)
        self.config = C, Ih, Iw, M, Fh, Fw, Sh, Sw, pad, Oh, Ow
        self.ic = None
        super().__init__(**kwargs)

    def fix_configuration(self, shape):
        C, Ih, Iw, M, Fh, Fw, Sh, Sw, pad, Oh, Ow = self.config
        if len(shape) >= 3:
            Ih = shape[-2] 
            Iw = shape[-1] 
            C  = shape[1] if len(shape)==4 else 1
        elif C is None or Ih is None or Iw is None:
            raise Exception(self.__class__.__name__ + ' cannot fix configuration.')
        Oh = (Ih - 1) * Sh + Fh - 2 * pad  # 出力高さ
        Ow = (Iw - 1) * Sw + Fw - 2 * pad  # 出力幅
        self.config = C, Ih, Iw, M, Fh, Fw, Sh, Sw, pad, Oh, Ow
        self.ic = ImCol(Fh, Fw, Sh, Sw)
        print(self.__class__.__name__, 'fix_configuration', shape, self.config)

    def get_parameter_size(self):
        C, Ih, Iw, M, Fh, Fw, Sh, Sw, pad, Oh, Ow = self.config
        m = C              # 入力チャネル数「要注意」
        n = M*Fh*Fw        # フィルタ数とフィルタサイズ「要注意」
        return m, n

    def _forward(self, x):
        B, C, Ih, Iw = x.shape
        self.x_shape = x.shape
        w, b, gamma = self.parameters()    
        _, _, _, M, Fh, Fw, Sh, Sw, pad, _, _ = self.config
        Oh = (Ih - 1) * Sh + Fh - 2 * pad  # 出力高さ
        Ow = (Iw - 1) * Sw + Fw - 2 * pad  # 出力幅
        x = x.transpose(0, 2, 3, 1).reshape(-1, C) # (B,C,Ih,Iw)->(B*Ih*Iw,C)  
        # linear変換 (B*Ih*Iw,C)×(C,M*Fh*Fw)->(B*Ih*Iw,M*Fh*Fw)
        cols = self.dot_linear.forward(x, w, b, gamma)
        # 行列を画像に変換 cols.T:(M*Fh*Fw,B*Ih*Iw)->(B,M,Oh,Ow)  　
        y = self.ic.col2im(cols, (B, Ih, Iw)) # 20251107AI
        # 画像調整 トリミング
        y = y[:,:,pad:pad+Oh,pad:pad+Ow]              # y.shape=(B,M,Oh,Ow)
        return y

    def _backward(self, grad_y, flush=True):
        _, _, _, M, Fh, Fw, Sh, Sw, pad, _, _ = self.config
        B, C, Ih, Iw = self.x_shape
        #  '0'パディング
        grad_y = np.pad(grad_y, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
        # 画像の勾配を行列に変換 grad_y.shape=(M*Fh*Fw,B*Ih*Iw)に変換
        grad_y = self.ic.im2col(grad_y)
        # linearの逆伝播
        grad_x, grad_w, grad_b, ggamma = self.dot_linear.backward(grad_y)
        grad_x = grad_x.reshape(B,Ih,Iw,C).transpose(0,3,1,2) # (B,C,Ih,Iw)
        self.parameters.set_gradient(grad_w, grad_b, ggamma, flush=flush)
        return grad_x

class DeConv2dLayer(Conv2dTransposeLayer):
    def __init__(self, *args, **kwargs):
        msg = (
        'What is called Deconvolution is actually ConvTranspose, '
        'which is now considered a misuse of the term.'
        )
        print(msg)
        super().__init__(*args, **kwargs)

class DeConvLayer(DeConv2dLayer):
    pass


class ImCol:
    """ img.shape = (B, C, H, W) <-> col.shape = (B*Nh*Nw, C*Fh*Fw)

    H, W   : img側の高さ・幅
    Nh, Nw : 窓の配置数
    Fh, Fw : フィルタ高さ・幅
    Sh, Sw : stride
    """

    def __init__(self, Fh, Fw, Sh=1, Sw=1):
        self.Fh = Fh
        self.Fw = Fw
        self.Sh = Sh
        self.Sw = Sw
        self.B = None
        self.C = None
        self.H = None
        self.W = None
        self.Nh = None
        self.Nw = None

    def im2col(self, img):
        B, C, H, W = img.shape
        Fh, Fw = self.Fh, self.Fw
        Sh, Sw = self.Sh, self.Sw

        # 窓の配置数
        Nh = (H - Fh) // Sh + 1
        Nw = (W - Fw) // Sw + 1

        # 形状を記憶
        self.B = B
        self.C = C
        self.H = H
        self.W = W
        self.Nh = Nh
        self.Nw = Nw

        col = np.empty((B, C, Fh, Fw, Nh, Nw), dtype=Config.dtype)

        # imgから各窓位置のデータを取り出す
        for fh in range(Fh):
            h_lim = fh + Sh * Nh
            for fw in range(Fw):
                w_lim = fw + Sw * Nw
                col[:, :, fh, fw, :, :] = img[:, :, fh:h_lim:Sh, fw:w_lim:Sw]

        # (B,C,Fh,Fw,Nh,Nw)
        #      -> (B,Nh,Nw,C,Fh,Fw)
        #      -> (B*Nh*Nw,C*Fh*Fw)
        col = col.transpose(0, 4, 5, 1, 2, 3).reshape(B * Nh * Nw, C * Fh * Fw)

        return col

    def col2im(self, col, grid=None):
        Fh, Fw = self.Fh, self.Fw
        Sh, Sw = self.Sh, self.Sw

        # im2colを先に通っていれば、その配置情報を使える
        if grid is None:
            B = self.B
            Nh = self.Nh
            Nw = self.Nw

            if B is None or Nh is None or Nw is None:
                raise Exception(self.__class__.__name__
                    + '.col2im requires grid=(B, Nh, Nw) when called first.')

            # im2col時の元画像サイズをそのまま使う
            H = self.H
            W = self.W

        else:
            # col2imを先に呼ぶ場合
            # col.shape[0] = B*Nh*Nw の分解情報
            B, Nh, Nw = grid

            # 復元される自然な画像サイズ
            H = (Nh - 1) * Sh + Fh
            W = (Nw - 1) * Sw + Fw

        # col.shape[1] = C*Fh*Fw から C を求める
        C = col.shape[1] // (Fh * Fw)

        if col.shape[0] != B * Nh * Nw:
            raise Exception(self.__class__.__name__ + ' col shape mismatch.')

        # 形状を記憶
        self.B = B
        self.C = C
        self.H = H
        self.W = W
        self.Nh = Nh
        self.Nw = Nw

        # (B*Nh*Nw,C*Fh*Fw)
        #      -> (B,Nh,Nw,C,Fh,Fw)
        #      -> (B,C,Fh,Fw,Nh,Nw)
        col = col.reshape(B, Nh, Nw, C, Fh, Fw).transpose(0, 3, 4, 5, 1, 2)
        img = np.zeros((B, C, H, W), dtype=Config.dtype)

        # 重なる部分は加算して戻す
        for fh in range(Fh):
            h_lim = fh + Sh * Nh
            for fw in range(Fw):
                w_lim = fw + Sw * Nw
                img[:, :, fh:h_lim:Sh, fw:w_lim:Sw] += col[:, :, fh, fw, :, :]

        return img
    
class Im2col:
    """
    img.shape=(B,C,Ih,Iw) → cols.shape=(B,C,Fh,Fw,Oh,Ow) -> (B*Oh*Ow, C*Fh*Fw)

    """
    def __init__(self, C, Ih, Iw, Fh, Fw, Sh, Sw, Oh=None, Ow=None):
        # 出力画像のサイズ
        if Oh is None: 
            Oh = (Ih - Fh) // Sh + 1        # 出力高さ
        if Ow is None:    
            Ow = (Iw - Fw) // Sw + 1        # 出力幅
        # パラメータをまとめる(class内での変数受渡しのため)
        self.config = C, Ih, Iw, Fh, Fw, Sh, Sw, Oh, Ow

    def __call__(self, img):
        C, Ih, Iw, Fh, Fw, Sh, Sw, Oh, Ow = self.config
        B = img.size // (C*Ih*Iw)
        col = np.empty((B, C, Fh, Fw, Oh, Ow), dtype=Config.dtype)
                                                  # メモリ節約のためzerosでなくempty 
        # imgからstride毎のデータを取ってきて、colsにOh,Owになるまで並べる
        # それをFh,Fwを満たすまで繰返す
        for fh in range(Fh):
            h_lim = fh + Sh*Oh
            for fw in range(Fw):
                w_lim = fw + Sw*Ow
                col[:,:,fh,fw,:,:] = img[:,:,fh:h_lim:Sh,fw:w_lim:Sw]
        # 軸の入替と変形     B  Oh Ow C Fh Fw 
        col = col.transpose(0, 4, 5, 1, 2, 3).reshape(B*Oh*Ow, C*Fh*Fw)
        return col

class Col2im:
    """
    col.shape=(B*Ih*Iw, C*Fh*Fw)->(B,Ih,Iw,C,Fh,Fw)->(B,C,Fh,Fw,Ih,Iw)
             →img.shape=(B,C,Oh,Ow)
    """
    def __init__(self, C, Ih, Iw, Fh, Fw, Sh, Sw, Oh=None, Ow=None):
        # Im2col 側の Oh,Ow (= ここでの Ih,Iw) から復元後の画像サイズを計算
        if Oh is None:
            Oh = (Ih - 1) * Sh + Fh
        if Ow is None:    
            Ow = (Iw - 1) * Sw + Fw
        self.config = C, Ih, Iw, Fh, Fw, Sh, Sw, Oh, Ow

    def __call__(self, col):
        C, Ih, Iw, Fh, Fw, Sh, Sw, Oh, Ow = self.config
        B = col.size // (C*Ih*Iw*Fh*Fw)
        col = col.reshape(B, Ih, Iw, C, Fh, Fw).transpose(0, 3, 4, 5, 1, 2)
        img = np.zeros((B, C, Oh, Ow), dtype=Config.dtype)
        # colからstride*Ih,Ow個のデータを取ってきて、imgにstride毎に並べる
        # それをFh,Fwを満たすまで繰返す
        for fh in range(Fh):
            h_lim = fh + Sh * Ih
            for fw in range(Fw):
                w_lim = fw + Sw * Iw
                img[:, :, fh:h_lim:Sh, fw:w_lim:Sw] += col[:, :, fh, fw, :, :]
        return img


### プーリング層 ####################################################
class Pooling1dLayer(Function):  
    # B:バッチサイズ, C:入力チャンネル数, Iw:入力幅
    # pool:プーリング領域のサイズ, pad:パディング幅
    # C:出力チャンネル数, Ow:出力幅
    def __init__(self, *configuration, **kwargs):
        super().__init__()
        C, Iw, pool, pad, method, Ow = None, None, 2, 0, None, None
        if len(configuration) == 5:
            C, Iw, pool, pad, method = configuration
        if len(configuration) == 4:
            C, Iw, pool, pad = configuration
        if len(configuration) == 3:
            pool, pad, method = configuration
        if len(configuration) == 2:
            pool, pad = configuration
        if len(configuration) == 1:
            pool, = configuration
        self.config = C, Iw, pool, pad, Ow
        self.method = kwargs.pop('method', 'max') if method is None else method
        print('Initialize', self.__class__.__name__, self.config, self.method)
        self.max_index = None
        self.DO = Dropout() if kwargs.pop('dropout', False) else None
        self.pooling = None
        self.unpooling = None
        
    def fix_configuration(self, shape):
        C, Iw, pool, pad, Ow = self.config
        B  = shape[0]
        if len(shape) >= 2:
            Iw = shape[-1] 
            C = shape[1] if len(shape)==3 else 1
        elif C is None or Iw is None:
            raise Exception(self.__class__.__name__ + ' cannot fix configuration.')
        Ow = (Iw + 2 * pad + pool - 1) // pool # 端数は切捨て
        self.config = C, Iw, pool, pad, Ow
        self.pooling = Pooling1d(pool, Ow, self.method)
        self.unpooling = UnPooling1d(pool, self.method)
        print(self.__class__.__name__, 'fix_configuration', shape, self.config)
            
    def __forward__(self, x, *, train=False, dropout=0.0):
        self.x = x
        if None in self.config:
            #print(self.__class__.__name__, 'input.shape', x.shape)
            self.fix_configuration(x.shape)
        C, Iw, pool, pad, Ow = self.config
        #B = x.size // (C*Ih*Iw)
        x = x.reshape(-1, C, Iw)               # 入力の形状 ex. (C, Iw)に対応   
        pdw = Ow * pool - Iw - pad             # サイズの端数に対応
        # 画像調整            B      C      Iw左 Iw右　ゼロパディング   
        img_pad = np.pad(x, [(0,0), (0,0), (pad, pdw)], 'constant')
        y, self.max_index = self.pooling(img_pad)
        if self.DO:
            y = self.DO.forward(y, dropout=dropout)  # 形状は(B,C,Oh,Ow)
        return y

    def __backward__(self, grad_y):
        C, Iw, pool, pad, Ow = self.config  
        B = grad_y.size // (C*Ow)              # B = grad_y.shape[0] = len(grad_y)
        grad_y = grad_y.reshape(B, C, Ow)      #ドロップアウトへの入力形状は順伝播時と同じ
        if self.DO:
            grad_y = self.DO.backward(grad_y)  # ドロップアウト
        grad_x = self.unpooling(grad_y, self.max_index)
        # 画像調整 トリミング
        grad_x = grad_x[:, :, pad:pad+Iw]      # grad_x.shape=(B,C,Iw) 
        grad_x = grad_x.reshape(self.x.shape)
        return grad_x

### 逆プーリング層 ####################################################
class UnPooling1dLayer(Function):  
    # B:バッチサイズ, C:入力チャンネル数, Iw:入力幅
    # pool:プーリング領域のサイズ, pad:パディング幅
    # C:出力チャンネル数, Ow:出力幅
    def __init__(self, *configuration, **kwargs):
        super().__init__()
        C, Iw, pool, pad, method, Ow = None, None, 2, 0, None, None
        if len(configuration) == 5:
            C, Iw, pool, pad, method = configuration
        if len(configuration) == 4:
            C, Iw, pool, pad = configuration
        if len(configuration) == 3:
            pool, pad, method = configuration
        if len(configuration) == 2:
            pool, pad = configuration
        if len(configuration) == 1:
            pool, = configuration
        self.config = C, Iw, pool, pad, Ow
        self.method = kwargs.pop('method', 'max') if method is None else method
        print('Initialize', self.__class__.__name__, self.config, self.method)
        self.max_index = None
        self.DO = Dropout() if kwargs.pop('dropout', False) else None
        self.unpooling = None
        self.pooling = None
        
    def fix_configuration(self, shape):
        C, Iw, pool, pad, Ow = self.config
        B  = shape[0]
        if len(shape) >= 2:
            Iw = shape[-1] 
            C = shape[1] if len(shape)==3 else 1
        elif C is None or Iw is None:
            raise Exception(self.__class__.__name__ + ' cannot fix configuration.')
        Ow = Iw * pool - 2 * pad
        self.config = C, Iw, pool, pad, Ow
        self.unpooling = UnPooling1d(pool, self.method)
        self.pooling = Pooling1d(pool, Iw, self.method)
        print(self.__class__.__name__, 'fix_configuration', shape, self.config)
            
    def __forward__(self, x, *, train=False, dropout=0.0, max_index=None):
        self.x = x
        if None in self.config:
            #print(self.__class__.__name__, 'input.shape', x.shape)
            self.fix_configuration(x.shape)
        C, Iw, pool, pad, Ow = self.config
        #B = x.size // (C*Iw)
        x = x.reshape(-1, C, Iw)               # 入力の形状 ex. (C,Ih*Iw)に対応   
        #print('img_pad', img_pad.shape, self.config)
        y = self.unpooling(x, max_index)
        # 画像調整 トリミング
        y = y[:, :, pad:pad+Ow]                # y.shape=(B,C,Oh,Ow) 
        if self.DO:
            y = self.DO.forward(y, dropout=dropout)  # 形状は(B,C,Oh,Ow)
        return y

    def __backward__(self, grad_y):
        C, Iw, pool, pad, Ow = self.config     # パラメタ
        B = grad_y.size // (C*Ow)              # B = grad_y.shape[0] = len(grad_y)
        grad_y = grad_y.reshape(B, C, Ow)      # ドロップアウトへの入力形状は順伝播時と同じ
        if self.DO:
            grad_y = self.DO.backward(grad_y)  # ドロップアウト
        pdw = Iw*pool - Ow - pad               # 画像サイズの端数を調整
        # 画像調整                 B      C     Iw左 Iw右　 ゼロパディング   
        grad_y = np.pad(grad_y, [(0,0), (0,0), (pad, pdw)], 'constant')
        grad_x, _ = self.pooling(grad_y)
        grad_x = grad_x.reshape(self.x.shape)
        return grad_x

class Pooling1d:  
    def __init__(self, pool, Ow, method):
        self.config = pool, Ow, method
            
    def __call__(self, x):
        pool, Ow, method = self.config
        B, C, Iw = x.shape
                        # (0,  1, 2,  3   )
        quarry = x.reshape(-1, C, Ow, pool)          # poolはaxis=3   
                        
        if method == 'average': # averageプーリング　
            y = np.mean(quarry, axis=3)              # poolの軸で平均値
            max_index = None
        else:     # 'max': 通常は Maxプーリング
            y = np.max (quarry, axis=3)              # poolの軸で最大値
            max_index = np.argmax(quarry, axis=3)    # インデクス記録
        return y, max_index

class UnPooling1d:  
    def __init__(self, pool, method=None):
        self.config = pool, method

    def __call__(self, x, max_index=None):
        pool, method = self.config   # パラメタ
        B, C, Iw = x.shape
        quarry = np.zeros((B*C*Iw, pool), dtype='f4') # 初期値0
        # 各行に勾配を入れる
        if method == 'average' or max_index is None: 
            quarry[np.arange(B*C*Iw)] \
                =  np.repeat((x/pool).reshape(-1, 1), pool, axis=1)
        else:     #  'max':    各行の最大値であった列の要素にのみ出力の勾配を入れる
            max_index = max_index.reshape(-1)
            quarry[np.arange(B*C*Iw), max_index] = x.reshape(-1)
                      #     (B*C*Iw,  pool)
        y = quarry.reshape  (B, C, Iw*pool)
        return y


### プーリング層 ####################################################
class Pooling2dLayer(Function):
    """ 二次元プーリング層 """
    # B:バッチサイズ, C:入力チャンネル数, Ih:入力画像高さ, Iw:入力画像幅
    # pool:プーリング領域のサイズ, pad:パディング幅
    # C:出力チャンネル数, Oh:出力高さ, Ow:出力幅
    def __init__(self, *configuration, **kwargs):
        super().__init__()
        C, image_size, pool, pad, method, Oh, Ow = None, None, 2, 0, None, None, None
        if len(configuration) == 5:
            C, image_size, pool, pad, method = configuration
        if len(configuration) == 4:
            C, image_size, pool, pad = configuration
        if len(configuration) == 3 and isinstance(configuration[-1], str):
            pool, pad, method = configuration
        elif len(configuration) == 3:
            C, image_size, pool = configuration
        if len(configuration) == 2 and isinstance(configuration[-1], str):
            pool, method = configuration
        elif len(configuration) == 2:    
            pool, pad = configuration
        if len(configuration) == 1:
            pool, = configuration
        Ih, Iw = image_size if isinstance(image_size, (tuple, list)) \
                            else (image_size, image_size)
        pool_h, pool_w = pool if isinstance(pool, (tuple, list)) else (pool, pool)
        self.config = C, Ih, Iw, pool_h, pool_w, pad, Oh, Ow
        self.method = kwargs.pop('method', 'max') if method is None else method
        print('Initialize', self.__class__.__name__, self.config, self.method)
        self.max_index = None
        self.DO = Dropout() if kwargs.pop('dropout', False) else None
        self.pooling = None
        self.unpooling = None
        
    def fix_configuration(self, shape):
        C, Ih, Iw, pool_h, pool_w, pad, Oh, Ow = self.config
        B  = shape[0]
        if len(shape) >= 3:
            Ih = shape[-2] 
            Iw = shape[-1] 
            C = shape[1] if len(shape)==4 else 1
        elif C is None or Ih is None or Iw is None:
            raise Exception(self.__class__.__name__, 'cannot fix configuration.')
        Oh = (Ih + 2 * pad + pool_h - 1) // pool_h # 端数は切捨て
        Ow = (Iw + 2 * pad + pool_w - 1) // pool_w # 端数は切捨て
        self.config = C, Ih, Iw, pool_h, pool_w, pad, Oh, Ow
        self.pooling = Pooling2d(pool_h, pool_w, Oh, Ow, self.method)
        self.unpooling = UnPooling2d(pool_h, pool_w, self.method)
        print(self.__class__.__name__, 'fix_configuration', shape, self.config)
      
            
    def __forward__(self, x, *, train=False, dropout=0.0):
        self.x = x
        if None in self.config:
            #print(self.__class__.__name__, 'input.shape', x.shape)
            self.fix_configuration(x.shape)
        C, Ih, Iw, pool_h, pool_w, pad, Oh, Ow = self.config
        #B = x.size // (C*Ih*Iw)
        x = x.reshape(-1, C, Ih, Iw)           # 入力の形状 ex. (C,Ih*Iw)に対応   
        pdh = Oh * pool_h - Ih - pad           # 画像サイズの端数に対応
        pdw = Ow * pool_w - Iw - pad           # 画像サイズの端数に対応
        # 画像調整            B      C     Ih上　Ih下   Iw左 Iw右　ゼロパディング   
        img_pad = np.pad(x, [(0,0), (0,0), (pad, pdh), (pad, pdw)], 'constant')
        y, self.max_index = self.pooling(img_pad)
        if self.DO:
            y = self.DO.forward(y, dropout=dropout)  # 形状は(B,C,Oh,Ow)
        return y

    def __backward__(self, grad_y):
        C, Ih, Iw, pool_h, pool_w, pad, Oh, Ow = self.config   # パラメタ
        B = grad_y.size // (C*Oh*Ow)           # B = grad_y.shape[0] = len(grad_y)
        grad_y = grad_y.reshape(B, C, Oh, Ow)  # ドロップアウトへの入力形状は順伝播時と同じ
        if self.DO:
            grad_y = self.DO.backward(grad_y)  # ドロップアウト
        grad_x = self.unpooling(grad_y, self.max_index)
        # 画像調整 トリミング
        grad_x = grad_x[:, :, pad:pad+Ih, pad:pad+Iw] # grad_x.shape=(B,C,Ih,Iw) 
        grad_x = grad_x.reshape(self.x.shape)
        return grad_x

class PoolingLayer(Pooling2dLayer):
    pass

### 逆プーリング層 ####################################################
class UnPooling2dLayer(Function):  
    # B:バッチサイズ, C:入力チャンネル数, Ih:入力画像高さ, Iw:入力画像幅
    # pool:プーリング領域のサイズ, pad:パディング幅
    # C:出力チャンネル数, Oh:出力高さ, Ow:出力幅
    def __init__(self, *configuration, **kwargs):
        super().__init__()
        C, image_size, pool, pad, method, Oh, Ow = None, None, 2, 0, None, None, None
        if len(configuration) == 5:
            C, image_size, pool, pad, method = configuration
        if len(configuration) == 4:
            C, image_size, pool, pad = configuration
        if len(configuration) == 3 and isinstance(configuration[-1], str):
            pool, pad, method = configuration
        elif len(configuration) == 3:
            C, image_size, pool = configuration
        if len(configuration) == 2 and isinstance(configuration[-1], str):
            pool, method = configuration
        elif len(configuration) == 2:    
            pool, pad = configuration
        if len(configuration) == 1:
            pool, = configuration
        Ih, Iw = image_size if isinstance(image_size, (tuple, list)) \
                            else (image_size, image_size)
        pool_h, pool_w = pool if isinstance(pool, (tuple, list)) else (pool, pool)
        self.config = C, Ih, Iw, pool_h, pool_w, pad, Oh, Ow
        self.method = kwargs.pop('method', 'max') if method is None else method
        print('Initialize', self.__class__.__name__, self.config, self.method)
        self.max_index = None
        self.DO = Dropout() if kwargs.pop('dropout', False) else None
        self.unpooling = None
        self.pooling = None
        
    def fix_configuration(self, shape):
        C, Ih, Iw, pool_h, pool_w, pad, Oh, Ow = self.config
        B  = shape[0]
        if len(shape) >= 3:
            Ih = shape[-2] 
            Iw = shape[-1] 
            C = shape[1] if len(shape)==4 else 1
        elif C is None or Ih is None or Iw is None:
            raise Exception(self.__class__.__name__, 'cannot fix configuration.')

        Oh = Ih * pool_h - 2 * pad
        Ow = Iw * pool_w - 2 * pad
        self.config = C, Ih, Iw, pool_h, pool_w, pad, Oh, Ow
        self.unpooling = UnPooling2d(pool_h, pool_w, self.method)
        self.pooling = Pooling2d(pool_h, pool_w, Ih, Iw, self.method)
        print(self.__class__.__name__, 'fix_configuration', shape, self.config)
            
    def __forward__(self, x, *, train=False, dropout=0.0, max_index=None):
        self.x = x
        if None in self.config:
            #print(self.__class__.__name__, 'input.shape', x.shape)
            self.fix_configuration(x.shape)
        C, Ih, Iw, pool_h, pool_w, pad, Oh, Ow = self.config
        #B = x.size // (C*Ih*Iw)
        x = x.reshape(-1, C, Ih, Iw)           # 入力の形状 ex. (C,Ih*Iw)に対応   
        #print('img_pad', img_pad.shape, self.config)
        y = self.unpooling(x, max_index)
        # 画像調整 トリミング
        y = y[:, :, pad:pad+Oh, pad:pad+Ow]    # y.shape=(B,C,Oh,Ow) 
        if self.DO:
            y = self.DO.forward(y, dropout=dropout)  # 形状は(B,C,Oh,Ow)
        return y

    def __backward__(self, grad_y):
        C, Ih, Iw, pool_h, pool_w, pad, Oh, Ow = self.config   # パラメタ
        B = grad_y.size // (C*Oh*Ow)           # B = grad_y.shape[0] = len(grad_y)
        grad_y = grad_y.reshape(B, C, Oh, Ow)  # ドロップアウトへの入力形状は順伝播時と同じ
        if self.DO:
            grad_y = self.DO.backward(grad_y)  # ドロップアウト
        pdh = Ih*pool_h - Oh - pad             # 画像サイズの端数を調整
        pdw = Iw*pool_w - Ow - pad             # 画像サイズの端数を調整
        # 画像調整            B      C     Ih上　Ih下   Iw左 Iw右　ゼロパディング   
        grad_y = np.pad(grad_y, [(0,0), (0,0), (pad, pdh), (pad, pdw)], 'constant')
        grad_x, _ = self.pooling(grad_y)
        grad_x = grad_x.reshape(self.x.shape)
        return grad_x

class UnPoolingLayer(UnPooling2dLayer):
    pass

class Pooling2d:  
    def __init__(self, pool_h, pool_w, Oh, Ow, method):
        self.config = pool_h, pool_w, Oh, Ow, method
            
    def __call__(self, x):
        pool_h, pool_w, Oh, Ow, method = self.config
        B, C, Ih, Iw = x.shape
                        #   (0,  1, 2,  3,    4,   5   )
        quarry = x.reshape  (-1, C, Oh, pool_h, Ow,  pool_w) \
                  .transpose(0,  1, 2,  4,      3,   5)      \
                  .reshape  (-1, C, Oh, Ow,   pool_h*pool_w)
                        #   (0, 1, 2, 3,    4)  pool_h*pool_wはaxis=4 
        if method == 'average': # averageプーリング　
            y = np.mean(quarry, axis=4)   # pool*poolの軸で平均値
            max_index = None
        else:     # 'max': 通常は Maxプーリング
            y = np.max (quarry, axis=4)   # pool*poolの軸で最大値
            max_index = np.argmax(quarry, axis=4)   # インデクス記録
        return y, max_index

class UnPooling2d:  
    def __init__(self, pool_h, pool_w, method=None):
        self.config = pool_h, pool_w, method

    def __call__(self, x, max_index=None):
        pool_h, pool_w, method = self.config   # パラメタ
        B, C, Ih, Iw = x.shape
        quarry = np.zeros((B*C*Ih*Iw,pool_h*pool_w), dtype='f4')  # 初期値0
        # 各行に勾配を入れる
        if method == 'average' or max_index is None: 
            quarry[np.arange(B*C*Ih*Iw)] \
                =  np.repeat((x/(pool_h*pool_w)).reshape(-1, 1), pool_h*pool_w, axis=1)
        else:     #  'max':    各行の最大値であった列の要素にのみ出力の勾配を入れる
            max_index = max_index.reshape(-1)
            quarry[np.arange(B*C*Ih*Iw), max_index] = x.reshape(-1)
                      #     (B*C*Ih*Iw,    pool*pool)
        y = quarry.reshape  (B*C, Ih, Iw,  pool_h, pool_w)   \
                  .transpose(0,   1,  3,   2,      4)      \
                  .reshape  (B,C, Ih*pool_h, Iw*pool_w) 
                      #     (B,C, Oh,        Ow) 
        return y

### globalAveragePooling層 ####################################################
class GlobalAveragePooling(Function):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.config = None
        print('Initialize', self.__class__.__name__)
        self.DO = Dropout() if kwargs.pop('dropout', False) else None

    def __forward__(self, x, *, train=False, dropout=0.0):
        self.x = x
        y = np.mean(x, axis=(2, 3))
        if self.DO:
            y = self.DO.forward(y, dropout=dropout)       
        return y
        
    def __backward__(self, grad_y):
        x = self.x
        B, C, Ih, Iw = x.shape
        if self.DO:
            grad_y = self.DO.backward(grad_y) 
        grad_x = grad_y / (Ih*Iw)
        grad_x = np.broadcast_to(grad_x.reshape(B, C, 1, 1), (B, C, Ih, Iw))
        return grad_x

### 2次元補完層 ############################################################
class Interpolate2d(Function):
    """ 統合インターフェース """

    def __init__(self, scale_factor=None, size=None,
                 mode="nearest", align="legacy"):
        super().__init__()

        if scale_factor is None and size is None:
            raise ValueError(
                f"{self.__class__.__name__}"
                f" scale_factor か size のどちらかを指定してください。"
                )

        if scale_factor is None:
            self.scale_factor = None
        elif isinstance(scale_factor, (tuple, list)) and len(scale_factor)==2:
            self.scale_factor = scale_factor
        elif (isinstance(scale_factor, (int, float, np.integer, np.floating))
            and scale_factor > 0):
            self.scale_factor = scale_factor, scale_factor
        else:
            raise ValueError(
                f"{self.__class__.__name__}"
                f" invalid scale_factor specified {scale_factor}"
            )

        if size is None:
            self.size = None
        elif (isinstance(size, (tuple, list))
              and len(size) == 2
              and size[0] > 0 and size[1] > 0):
            self.size = tuple(size)
        else:
            raise ValueError(
                f"{self.__class__.__name__}: invalid size {size}"
            )

        if mode not in ('nearest', 'bilinear'):
            raise NotImplementedError(
                f"{self.__class__.__name__}"
                f" mode={mode}, size={self.size}, "
                f"scale_factor={self.scale_factor} はまだ未実装です。"
            )
            
        self.impl = None
        Ih, Iw, Oh, Ow = None, None, None, None
        self.config = Ih, Iw, Oh, Ow, mode, align


    def fix_configuration(self, shape):
        """共通の ndim チェックと Ih,Iw,Oh,Ow の決定までを行う。"""
        _, _, _, _, mode, align = self.config 
        if len(shape) < 2:
            raise ValueError(
                f"{self.__class__.__name__}: "
                f"input ndim must be >= 2 (got {shape}). "
                f"最後の2軸を (H,W) として扱います。"
            )
        B = shape[0]
        *prefix, Ih, Iw = shape[1:]

        if self.size is None:
            Oh = int(Ih * self.scale_factor[0])
            Ow = int(Iw * self.scale_factor[1])
            if Oh < 1 or Ow < 1:
                raise ValueError(
                    f"{self.__class__.__name__}: "
                    f"scale_factor {self.scale_factor} too small for input "
                    f"shape ({Ih},{Iw}) → ({Oh},{Ow})"
                )
            self.size = Oh, Ow
        Oh, Ow = self.size
        self.config = Ih, Iw, Oh, Ow, mode, align

        # ここで「どの実装を使うか」を決める
        if mode == "nearest" and Oh%Ih==0 and Ow%Iw==0:
            self.impl = Interpolate2dNearestSimple(self.config)

        else: 
            self.impl = Interpolate2dGeneral(self.config)
        print(self.impl.__class__.__name__, self.config)

    def __forward__(self, x):
        if None in self.config:
            #print(self.__class__.__name__, 'input.shape', x.shape)
            self.fix_configuration(x.shape)
        y = self.impl.forward(x)
        return y

    def __backward__(self, gy):
        return self.impl.backward(gy)

class Interpolate2dLayer(Interpolate2d):
    def __forward__(self, x, *args, **kwargs):
        return super().__forward__(x)
    

class Interpolate2dNearestSimple:
    """ 整数スケールの最近傍アップサンプリング """
    def __init__(self, config):
        self.config = config

    def forward(self, x):
        Ih, Iw, Oh, Ow, _, _ = self.config
        y = x.repeat(Ow//Iw, axis=-1) # 横方向のupsampling
        y = y.repeat(Oh//Ih, axis=-2) # 縦方向のupsampling
        self.x = x
        return y

    def backward(self, gy):
        Ih, Iw, Oh, Ow, _, _ = self.config
        gx = gy.reshape(self.x.shape[:-2]+(Ih, Oh//Ih, Iw, Ow//Iw)).sum(axis=(-3, -1))
        return gx


class Interpolate2dGeneral:
    """ 補間行列ABを用いた汎用的な補間ロジック """

    def __init__(self, config):
        self.config = config
        self.AB = None
        
    def forward(self, x):
        """ y = A @ x @ B.T 但しxは多次元配列 """
        if self.AB is None:
            self.build_AB()
        A, B = self.AB    
        tmp = np.tensordot(x, B.T, axes=([-1], [0]))  # (..., Ih, Ow)
        y   = np.tensordot(A, tmp, axes=([1], [-2]))  # (Oh, B, C, Ow)
        y   = np.moveaxis(y, 0, -2)                   # (B, C, Oh, Ow)
        return y

    def backward(self, gy):
        """ gx = A.T @ gy @ B 但しgyは多次元配列 """
        A, B = self.AB
        # まず行方向 (Oh→Ih)： gy(...,Oh,Ow) × A(Oh,Ih) → tmp(...,Ow,Ih)
        tmp = np.tensordot(gy, A, axes=([-2], [0]))   # (..., Ow, Ih)
        # 軸順を (..., Ih, Ow) に揃える
        tmp = np.moveaxis(tmp, -1, -2)                # (..., Ih, Ow)
        # 次に列方向 (Ow→Iw)： tmp(...,Ih,Ow) × B(Ow,Iw) → gx(...,Ih,Iw)
        gx = np.tensordot(tmp, B, axes=([-1], [0]))   # (..., Ih, Iw)
        return gx

    def build_AB(self):
        Ih, Iw, Oh, Ow, mode, align = self.config

        if align=='corners': # 端を揃える
            if Oh > 1:
                scale_h = (Ih - 1) / (Oh - 1)
                pos_h = np.arange(Oh, dtype=Config.dtype) * scale_h
            else:
                scale_h = 0
                pos_h = np.zeros(1, dtype=Config.dtype)

            if Ow > 1:
                scale_w = (Iw - 1) / (Ow - 1)
                pos_w = np.arange(Ow, dtype=Config.dtype) * scale_w
            else:
                scale_w = 0
                pos_w = np.zeros(1, dtype=Config.dtype)

        elif align=='center': # ピクセル中心を揃える
            scale_h = Ih / Oh
            scale_w = Iw / Ow
            pos_h = (np.arange(Oh, dtype=Config.dtype) + 0.5) * scale_h - 0.5
            pos_w = (np.arange(Ow, dtype=Config.dtype) + 0.5) * scale_w - 0.5
            pos_h = np.clip(pos_h, 0, Ih - 1)
            pos_w = np.clip(pos_w, 0, Iw - 1)
        else: # align=='legacy' 上端と左端を合せて下端と右端は成り行き
            scale_h = Ih / Oh
            scale_w = Iw / Ow
            pos_h = np.arange(Oh, dtype=Config.dtype) * scale_h  # (Oh,)
            pos_w = np.arange(Ow, dtype=Config.dtype) * scale_w  # (Ow,)
               
        pos_hi = np.floor(pos_h)  
        pos_wi = np.floor(pos_w)  
        h0 = pos_hi.astype(np.int32)
        w0 = pos_wi.astype(np.int32)
        h1 = h0 + 1
        w1 = w0 + 1
        h0 = np.clip(h0, 0, Ih - 1)
        w0 = np.clip(w0, 0, Iw - 1) 
        h1 = np.clip(h1, 0, Ih - 1)
        w1 = np.clip(w1, 0, Iw - 1) 
        th = pos_h - pos_hi # 端数=bilinearの混ぜ具合 
        tw = pos_w - pos_wi # 端数=bilinearの混ぜ具合

        # 行方向 (H) の補間行列 A、列方向 (W) の補間行列 B を作る
        A = np.zeros((Oh, Ih), dtype=Config.dtype)
        B = np.zeros((Ow, Iw), dtype=Config.dtype)
        if mode == 'bilinear':
            Ih, Iw, Oh, Ow, mode, align = self.config
            A[np.arange(Oh), h0] += (1 - th)
            A[np.arange(Oh), h1] += th
            B[np.arange(Ow), w0] += (1 - tw)
            B[np.arange(Ow), w1] += tw
        else:     # nearest
            Ih, Iw, Oh, Ow, mode, align = self.config
            A[np.arange(Oh), h0] = 1
            B[np.arange(Ow), w0] = 1
        self.AB = A, B
        
class Interpolate2dGeneral2: 
    """ 汎用的な補間ロジック(補間行列を作らない) """

    def __init__(self, config):
        self.config = config

    def forward(self, x):
        Ih, Iw, Oh, Ow, mode, align = self.config

        if align=='corners': # 端を揃える
            if Oh > 1:
                scale_h = (Ih - 1) / (Oh - 1)
                pos_h = np.arange(Oh, dtype=Config.dtype) * scale_h
            else:
                scale_h = 0
                pos_h = np.zeros(1, dtype=Config.dtype)

            if Ow > 1:
                scale_w = (Iw - 1) / (Ow - 1)
                pos_w = np.arange(Ow, dtype=Config.dtype) * scale_w
            else:
                scale_w = 0
                pos_w = np.zeros(1, dtype=Config.dtype)

        elif align=='center': # ピクセル中心を揃える
            scale_h = Ih / Oh
            scale_w = Iw / Ow
            pos_h = (np.arange(Oh, dtype=Config.dtype) + 0.5) * scale_h - 0.5
            pos_w = (np.arange(Ow, dtype=Config.dtype) + 0.5) * scale_w - 0.5
            pos_h = np.clip(pos_h, 0, Ih - 1)
            pos_w = np.clip(pos_w, 0, Iw - 1)
        else: # legacy 上端と左端を合せて下端と右端は成り行き
            pos_h = np.arange(Oh, dtype=Config.dtype) * scale_h  # (Oh,)
            pos_w = np.arange(Ow, dtype=Config.dtype) * scale_w  # (Ow,)
               
               
        pos_hi = np.floor(pos_h)  
        pos_wi = np.floor(pos_w)  
        h0 = pos_hi.astype(np.int32)
        w0 = pos_wi.astype(np.int32)
        h1 = h0 + 1
        w1 = w0 + 1
        h0 = np.clip(h0, 0, Ih - 1)
        w0 = np.clip(w0, 0, Iw - 1) 
        h1 = np.clip(h1, 0, Ih - 1)
        w1 = np.clip(w1, 0, Iw - 1) 
        th = pos_h - pos_hi # 端数=bilinearの混ぜ具合 
        tw = pos_w - pos_wi # 端数=bilinearの混ぜ具合

        # 保存（重要）
        self.h0, self.h1 = h0, h1
        self.w0, self.w1 = w0, w1
        self.th = th.reshape(Oh, 1)
        self.tw = tw.reshape(1, Ow)
        self.mode = mode

        if mode == 'nearest':
            return x[..., h0[:, None], w0[None, :]]

        th = th.reshape(Oh, 1)
        tw = tw.reshape(1, Ow)

        # W方向
        x0 = x[..., :, w0]
        x1 = x[..., :, w1]
        tmp = (1 - tw) * x0 + tw * x1

        # H方向
        tmp0 = tmp[..., h0, :]
        tmp1 = tmp[..., h1, :]
        y = (1 - th) * tmp0 + th * tmp1

        return y

    def backward(self, gy):
        Ih, Iw, Oh, Ow, mode, align = self.config

        h0, h1 = self.h0, self.h1
        w0, w1 = self.w0, self.w1
        th, tw = self.th, self.tw

        gx = np.zeros((*gy.shape[:-2], Ih, Iw), dtype=gy.dtype)

        if mode == 'nearest':
            for i in range(Oh):
                for j in range(Ow):
                    gx[..., h0[i], w0[j]] += gy[..., i, j]
            return gx
    
        # --- H方向（修正） ---
        tmp = np.zeros((*gy.shape[:-2], Ih, Ow), dtype=gy.dtype)

        for i in range(Oh):
            tmp[..., h0[i], :] += gy[..., i, :] * (1 - th[i, 0])
            tmp[..., h1[i], :] += gy[..., i, :] * th[i, 0]

        # --- W方向（既に正しい） ---
        for j in range(Ow):
            gx[..., :, w0[j]] += tmp[..., :, j] * (1 - tw[0, j])
            gx[..., :, w1[j]] += tmp[..., :, j] * tw[0, j]

        return gx

class Interpolate2dGeneral_bkup:
    """ 汎用的な補間ロジック """

    def __init__(self, config):
        self.config = config
        
    def forward(self, x):
        Ih, Iw, Oh, Ow, mode, align = self.config

        scale_h = Ih / Oh
        scale_w = Iw / Ow

        pos_h = np.arange(Oh, dtype=Config.dtype) * scale_h  # (Oh,)
        pos_w = np.arange(Ow, dtype=Config.dtype) * scale_w  # (Ow,)
        pos_hi = np.floor(pos_h)  
        pos_wi = np.floor(pos_w)  
        h0 = pos_hi.astype(np.int32)
        w0 = pos_wi.astype(np.int32)
        h1 = h0 + 1
        w1 = w0 + 1
        h0 = np.clip(h0, 0, Ih - 1)
        w0 = np.clip(w0, 0, Iw - 1) 
        h1 = np.clip(h1, 0, Ih - 1)
        w1 = np.clip(w1, 0, Iw - 1) 
        th = pos_h - pos_hi # 端数=bilinearの混ぜ具合 
        tw = pos_w - pos_wi # 端数=bilinearの混ぜ具合

        # 行方向 (H) の補間行列 A、列方向 (W) の補間行列 B を作る

        A = np.zeros((Oh, Ih), dtype=Config.dtype)
        B = np.zeros((Ow, Iw), dtype=Config.dtype)

        if mode == 'bilinear':
            self.bilinear_AB(A, B, h0, h1, w0, w1, th, tw)
        else:
            self.nearest_AB(A, B, h0, w0)

        # forward: Y = A · X · B^T
        BT = B.T  # (Iw, Ow)
        tmp = np.tensordot(x, BT, axes=([-1], [0]))   # (..., Ih, Ow)
        y   = np.tensordot(A, tmp, axes=([1], [-2]))  # (Oh, B, C, Ow)
        y   = np.moveaxis(y, 0, -2)                   # (B, C, Oh, Ow)
        # backward 用に保存
        self.A = A
        self.B = B

        return y

    def backward(self, gy):

        A = self.A  # (Oh, Ih)
        B = self.B  # (Ow, Iw)

        # dX = A^T · dY · B
        # まず行方向 (Oh→Ih)： gy(...,Oh,Ow) × A(Oh,Ih) → tmp(...,Ow,Ih)
        tmp = np.tensordot(gy, A, axes=([-2], [0]))   # (..., Ow, Ih)

        # 軸順を (..., Ih, Ow) に揃える
        tmp = np.moveaxis(tmp, -1, -2)                # (..., Ih, Ow)

        # 次に列方向 (Ow→Iw)： tmp(...,Ih,Ow) × B(Ow,Iw) → gx(...,Ih,Iw)
        gx = np.tensordot(tmp, B, axes=([-1], [0]))   # (..., Ih, Iw)

        return gx

    def nearest_AB(self, A, B, h0, w0):
        Ih, Iw, Oh, Ow, mode, align = self.config

        A[np.arange(Oh), h0] = 1
        B[np.arange(Ow), w0] = 1

    def bilinear_AB(self, A, B, h0, h1, w0, w1, th, tw):
        Ih, Iw, Oh, Ow, mode, align = self.config

        A[np.arange(Oh), h0] += (1 - th)
        A[np.arange(Oh), h1] += th
        B[np.arange(Ow), w0] += (1 - tw)
        B[np.arange(Ow), w1] += tw

        
class Interpolate2dNearest(Interpolate2d):
    """ 最近傍アップサンプリング別ルート実装 """

    def __forward__(self, x):
        self.x = x
        if None in self.config:
            #print(self.__class__.__name__, 'input.shape', x.shape)
            self.fix_configuration(x.shape)
        Ih, Iw, Oh, Ow, mode, align = self.config

        # 出力座標 0..Oh-1, 0..Ow-1
        # 入力側の連続座標 → 最近傍の整数インデックス
        # ここではシンプルに Ih/Oh, Iw/Ow でスケーリング
        scale_h = Ih / Oh
        scale_w = Iw / Ow
        
        h0 = (np.arange(Oh) * scale_h).astype(np.int32)  # h0.shape=(Oh,)
        w0 = (np.arange(Ow) * scale_w).astype(np.int32)  # w0.shape=(Ow,)
        h0 = np.clip(h0, 0, Ih - 1)
        w0 = np.clip(w0, 0, Iw - 1)

        # forward: ベクトル化参照
        # x[..., Ih, Iw] に対し、最後の2軸に (Oh,Ow) のインデックスを食わせる
        y = x[..., h0[:, None], w0[None, :]]

        # backward 用に保存
        self.h0 = h0
        self.w0 = w0

        return y

    def __backward__(self, gy):
        x = self.x
        B = x.shape[0]
        prefix = x.shape[1:-2] # バッチ軸 B と末尾の空間軸 (H, W) の間にある軸
        Ih, Iw, Oh, Ow, mode, align = self.config

        # gx をゼロ初期化
        gx = np.zeros_like(x) # xの形と型を継承

        # 先頭軸を 1 次元にまとめて扱う: N = prod(prefix)
        N = math.prod(prefix) if prefix else 1
        gx_flat = gx.reshape(B*N, Ih, Iw)
        gy_flat = gy.reshape(B*N, Oh, Ow)

        ih2 = self.h0[:, None]  # (Oh, 1)
        iw2 = self.w0[None, :]  # (1,  Ow)

        # N 軸だけ Python ループ、H/W は add.at に任せる
        for n in range(B*N):
            #gx_flat[n][h0, w0] += gy_flat[n]
            snp.add_at(gx_flat[n], (ih2, iw2), gy_flat[n])

        return gx


### マスク展開層 = 逆畳込みもどき ###########################################
class MaskedExpansionLayer(BaseLayer):
    """
    入力をマスクM分繰返して広げたのちに線形変換してフィルタFh*Fw分の拡大
    入力のチャネルは独立に扱う
    """
    # B:バッチサイズ, C:入力チャンネル数, Ih:入力画像高さ, Iw:入力画像幅
    # M:フィルタ数, Fh:フィルタ高さ, Fw:フィルタ幅
    # 出力チャンネル数=入力チャネル数, Oh:出力高さ, Ow:出力幅
    
    def __init__(self, *configuration, **kwargs):
        if len(configuration) == 6:
            C, Ih, Iw, M, Fh, Fw = configuration
        if len(configuration) == 5:
            C, Ih, Iw, Fh, Fw = configuration; M = 1
        if len(configuration) == 3:
            C = None; Ih = None; Iw = None; M, Fh, Fw = configuration
        if len(configuration) == 2:
            C = None; Ih = None; Iw = None; M = 1; Fh, Fw = configuration
        # stride, pad は未対応
        Oh = Ow = None    
        self.config = (C, Ih, Iw, M, Fh, Fw, Oh, Ow)
        super().__init__(**kwargs)

    def fix_configuration(self, shape):
        C, Ih, Iw, M, Fh, Fw, Oh, Ow = self.config
        if len(shape) >= 3:
            Ih = shape[-2] 
            Iw = shape[-1] 
            C = shape[1] if len(shape)==4 else 1
        elif C is None or Ih is None:
            raise Exception('MaskedExpansionLayer cannot fix configuration.')
            
        self.Oh = Oh = Ih * Fh
        self.Ow = Ow = Iw * Fw
        self.config = C, Ih, Iw, M, Fh, Fw, Oh, Ow
        print(self.__class__.__name__, 'fix_configuration', shape, self.config)
            
    def get_parameter_size(self):
        C, Ih, Iw, M, Fh, Fw, Oh, Ow = self.config
        m = M              # 入力チャネル数とフィルタサイズ
        n = Fh*Fw          # フィルタ数
        return m, n

    def _forward(self, x):
        self.x = x
        C, Ih, Iw, M, Fh, Fw, Oh, Ow = self.config
        B = x.size // (C*Ih*Iw)             # B = x.shape[0] = len(x)
        w, b, gamma = self.parameters()     # w.shape=(M,C*Fh*Fw)
        z = np.tile(x.reshape(B*C*Ih*Iw,1), (1, M)) # 同じ要素をフィルタ数分繰返す(B*C*Ih*Iw,M)
        y = self.dot_linear(z, w, b, gamma) # (B*C*Ih*Iw, Fh*Fw)
        y = y.reshape(B,C,Ih,Iw,Fh,Fw).transpose(0,1,2,4,3,5).reshape(B,C,Oh,Ow)
        return y
    
    def _backward(self, grad_y, flush=True): 
        x = self.x
        C, Ih, Iw, M, Fh, Fw, Oh, Ow = self.config
        B = grad_y.size // (C*Oh*Ow)        # B = grad_y.shape[0] = len(grad_y)
        grad_y = (grad_y.reshape(B,C,Ih,Fh,Iw,Fw)
                  .transpose(0,1,2,4,3,5).reshape(B*C*Ih*Iw,Fh*Fw))
        grad_z, grad_w, grad_b, ggamma = self.dot_linear.backward(grad_y)
        grad_x = np.sum(grad_z, axis=-1).reshape(B,C,Ih,Iw)
        self.parameters.set_gradient(grad_w, grad_b, ggamma, flush=flush)
        return grad_x

# -- 潜在変数をサンプリングする層 -- 20240701
# -- 潜在変数をサンプリングする層 -- 20240701
class LatentSampling(Function):
    def __init__(self, rate=1.0, kld=None, mil=None,
                 axis=-1,          # mu, log_var の分割軸
                 vectorize=True,   # Trueなら入力を(B, -1)に潰して扱う
                 mode='sum', free_bits=None,
                 **kwargs):
        super().__init__()
        print('Initialize', self.__class__.__name__)

        self.sampling = MuVarSampling()
        self.rate = rate
        self.axis = axis
        self.vectorize = vectorize

        if kld and kld > 0:
            self.r_kld = kld
            self.kld = KullbackLeiblerDivergenceNormal(
                mode=mode, free_bits=free_bits,
            )
        else:
            self.kld = None

        if mil and mil > 0:
            self.r_mil = mil
            self.mil = MutualInformationLoss()
        else:
            self.mil = None

    def __forward__(self, x, *, epsilon=None):
        # backwardで元形状に戻すため保存
        self.original_shape = x.shape

        # 従来LatentSampling互換:
        # (B, C, H, W)などを (B, -1) に潰して vector latent として扱う
        if self.vectorize:
            x = x.reshape(len(x), -1)
            axis = -1
        else:
            axis = self.axis

        self.forward_shape = x.shape
        self.forward_axis = axis

        # 指定軸を mu/log_var に2分割
        size = x.shape[axis]
        if size % 2 != 0:
            raise ValueError(
                f"LatentSampling: split axis size must be even, "
                f"but got shape={x.shape}, axis={axis}"
            )

        mu, log_var = np.split(x, 2, axis=axis)

        z = self.sampling.forward(
            mu, log_var, epsilon=epsilon, rate=self.rate
        )

        if not (self.kld or self.mil):
            return z

        if self.kld:
            kll = self.r_kld * self.kld.forward(mu, log_var)
        else:
            kll = 0

        if self.mil:
            mi = self.r_mil * self.mil.forward(z, mu, log_var)
        else:
            mi = 0

        return z, kll, mi

    def __backward__(self, gz, gkll=1, gmi=1):
        gmu, glog_var = 0, 0

        if self.mil:
            gz0, gmu0, glog_var0 = self.mil.backward(gmi * self.r_mil)
            gz = gz + gz0
            gmu += gmu0
            glog_var += glog_var0

        if self.kld:
            gmu0, glog_var0 = self.kld.backward(gkll * self.r_kld)
            gmu += gmu0
            glog_var += glog_var0

        gmu0, glog_var0 = self.sampling.backward(gz)
        gmu += gmu0
        glog_var += glog_var0

        # forward の split に対応
        gx = np.concatenate((gmu, glog_var), axis=self.forward_axis)

        # vectorize=True の場合は入力xの元形状へ戻す
        if self.vectorize:
            gx = gx.reshape(self.original_shape)

        return gx



class LatentSampling_bkup(Function):
    def __init__(self, rate=1.0, kld=None, mil=None,
                 axis=-1, # mu,log_varの分割軸
                 mode='sum', free_bits=None, # KLDのモード
                 **kwargs):# kwargsは他の層に対する指定を無視するために必要
        super().__init__()
        print('Initialize', self.__class__.__name__)
        self.sampling = MuVarSampling()             # サンプリングの関数
        self.rate = rate                            # サンプリングの広がり
        self.axis = axis    
        if kld and kld>0:
            self.r_kld = kld                        # 混ぜ具合 
            self.kld = KullbackLeiblerDivergenceNormal(
                mode=mode, free_bits=free_bits,
                )  # カルバック・ライブラー情報量関数
        else:
            self.kld = None
        if mil and mil>0:
            self.r_mil = mil                        # 混ぜ具合
            self.mil = MutualInformationLoss()      # 相互情報量
        else:
            self.mil = None
        
    def __forward__(self, x, *, epsilon=None):
        # 指定軸を mu/log_var に2分割
        size = x.shape[self.axis]
        if size % 2 != 0:
            raise ValueError(
                f"LatentSampling: split axis size must be even, "
                f"but got shape={x.shape}, axis={self.axis}"
            )
       
        mu, log_var = np.split(x, 2, axis=self.axis)
        
        # -- サンプリングとカルバック・ライプラー --    
        z = self.sampling.forward(
            mu, log_var, epsilon=epsilon, rate=self.rate
            )

        if not (self.kld or self.mil):
            return z
        if self.kld:
            kll = self.r_kld * self.kld.forward(mu, log_var)
        else:
            kll = 0
        if self.mil:
            mi  = self.r_mil * self.mil.forward(z, mu, log_var)
        else:
            mi = 0
        return z, kll, mi
    
    def __backward__(self, gz, gkll=1, gmi=1):
        gmu, glog_var = 0, 0

        if self.mil:
            gz0, gmu0, glog_var0 = self.mil.backward(gmi * self.r_mil)
            gz = gz + gz0
            gmu += gmu0
            glog_var += glog_var0

        if self.kld:
            gmu0, glog_var0 = self.kld.backward(gkll * self.r_kld) 
            gmu += gmu0
            glog_var += glog_var0

        gmu0, glog_var0 = self.sampling.backward(gz)
        gmu += gmu0
        glog_var += glog_var0

        # forward の np.split(axis=self.axis) に対応
        gx = np.concatenate((gmu, glog_var), axis=self.axis)

        return gx
    
# -- 潜在変数をサンプリングする層 -- 2023.07.17
class LatentSamplingZ:
    def __init__(self, rate=1.0, **kwargs): # kwargsは使わないが他の層に対する指定を無視するために必要
        super().__init__()
        print('Initialize', self.__class__.__name__)
        self.sampling = MuVarSampling()         # サンプリングの関数　  
        self.kld = KullbackLeiblerDivergenceNormal()  # カルバック・ライブラー情報量関数
        self.rate = rate                        # サンプリングの広がり
        #self.r_kld = kwargs.pop('r_kld', 1.0)  # kldの混ぜ具合 
        
    def forward(self, mu, log_var, epsilon=None):
        if epsilon is None:
            epsilon = np.random.randn(*log_var.shape) * self.rate
            epsilon = epsilon.astype(Config.dtype)
        z = self.sampling.forward(mu, log_var, epsilon=epsilon)
        kll = self.kld.forward(mu, log_var)
        return z, kll
    
    def backward(self, gz, gkll):
        gmu1, glog_var1 = self.sampling.backward(gz)
        gmu2, glog_var2 = self.kld.backward(gkll)    # gkllの効果は不明
        gmu = gmu1 + gmu2
        glog_var = glog_var1 + glog_var2
        return gmu, glog_var

class MuVarSampling(Function):
    """ muとlog_varによるReparameterize """
    def __forward__(self, mu, log_var, *, epsilon=None, rate=1.0):
        self.mu, self.log_var = mu, log_var
        if epsilon is None:
            epsilon = (rate * np.random.randn(*log_var.shape)).astype(Config.dtype)
        self.epsilon = epsilon # 逆伝播に備えて覚える
        z = mu + self.epsilon * np.exp(log_var/2)
        return z

    def __backward__(self, gz=1):
        mu, log_var = self.mu, self.log_var
        epsilon = self.epsilon # 順伝播から引き継ぐ 
        gmu = np.broadcast_to(gz, mu.shape) 
        glog_var = 0.5 * gz * epsilon * np.exp(log_var/2)
        return gmu, glog_var

class MuVarSampling2(Function):
    """ mu, log_var, epsilonのすべてが位置変数のサンプリング """
    def __forward__(self, mu, log_var, epsilon):
        self.mu, self.log_var, self.epsilon = mu, log_var, epsilon
        z = mu + epsilon * np.exp(log_var/2)
        return z

    def __backward__(self, gz=1):
        mu, log_var, epsilon = self.mu, self.log_var, self.epsilon
        gmu = np.broadcast_to(gz, mu.shape) 
        gepsilon = gz * np.exp(log_var/2)
        #glog_var = 0.5 * gz * epsilon * np.exp(log_var/2)
        glog_var = 0.5 * gepsilon * epsilon
        return gmu, glog_var, gepsilon

class KullbackLeiblerDivergenceNormal(Function):
    """
    標準正規分布 N(0, I) に対する N(mu, sigma^2) のKLD

    mode:
        'sum'          : 従来互換。B軸以外をすべてsumしてbatch平均
        'spatial_mean' : 空間軸(H,W)をmeanし、channel方向をsumしてbatch平均
                         Spatial latent向け
        'mean'         : B軸以外をmeanしてbatch平均

    free_bits:
        None または 0 : なし
        正値          : per-channel KLDなどに下限を設ける
                        mode='spatial_mean' と相性が良い
    """

    def __init__(self, mode='sum', free_bits=None):
        super().__init__()
        self.mode = mode
        self.free_bits = free_bits

    def __forward__(self, mu, log_var):
        self.mu, self.log_var = mu, log_var
        # element-wise KLD
        kld_elem = -0.5 * (1 + log_var - mu**2 - np.exp(log_var))

        self.kld_elem = kld_elem

        if self.mode == 'sum':
            # 従来互換: 全潜在要素を足して batch 平均
            kll = np.sum(kld_elem)
            self.scale = 1.0 / len(mu)
            self.mask = None
            return kll * self.scale

        elif self.mode == 'mean':
            # 全潜在要素の平均
            axes = tuple(range(1, kld_elem.ndim))
            kld_sample = np.mean(kld_elem, axis=axes)
            self.scale = 1.0 / len(mu)
            self.mask = None
            return np.sum(kld_sample) * self.scale

        elif self.mode == 'spatial_mean':
            if kld_elem.ndim < 4:
                # vector latent の場合は sum と同じ扱いに近い
                kld_ch = kld_elem
            else:
                # (B, C, H, W) -> (B, C)
                kld_ch = np.mean(kld_elem, axis=(2, 3))

            if self.free_bits is not None and self.free_bits > 0:
                self.mask = (kld_ch >= self.free_bits).astype(Config.dtype)
                kld_ch = np.maximum(kld_ch, self.free_bits)
            else:
                self.mask = None

            self.kld_ch_shape = kld_ch.shape
            self.scale = 1.0 / len(mu)

            # channel方向をsumし、batch平均
            return np.sum(kld_ch) * self.scale

        else:
            raise ValueError(f"Unknown KLD mode: {self.mode}")

    def __backward__(self, gkll=1):
        mu, log_var = self.mu, self.log_var

        # element-wise derivative
        gmu = mu
        glog_var = 0.5 * (np.exp(log_var) - 1)

        if self.mode == 'sum':
            coef = gkll * self.scale
            return coef * gmu, coef * glog_var

        elif self.mode == 'mean':
            axes_size = 1
            for s in mu.shape[1:]:
                axes_size *= s
            coef = gkll * self.scale / axes_size
            return coef * gmu, coef * glog_var

        elif self.mode == 'spatial_mean':
            coef = gkll * self.scale

            if mu.ndim >= 4:
                # spatial mean の分だけ H*W で割る
                H, W = mu.shape[2], mu.shape[3]
                coef = coef / (H * W)

                if self.mask is not None:
                    # (B,C) -> (B,C,1,1)
                    mask = self.mask[:, :, np.newaxis, np.newaxis]
                    gmu = gmu * mask
                    glog_var = glog_var * mask
            else:
                if self.mask is not None:
                    gmu = gmu * self.mask
                    glog_var = glog_var * self.mask

            return coef * gmu, coef * glog_var

        else:
            raise ValueError(f"Unknown KLD mode: {self.mode}")

class KullbackLeiblerDivergenceNormalBasic(Function):
    """ 標準正規分布に対する任意の正規分布のKLDの解析解 """
    def __forward__(self, mu, log_var):
        self.mu, self.log_var = mu, log_var
        kll = -0.5 * np.sum(1 + log_var - mu**2 - np.exp(log_var))
        return kll/len(mu)

    def __backward__(self, gkll=1):
        mu, log_var = self.mu, self.log_var
        gmu = gkll * mu
        glog_var = gkll * (-0.5) * (1 - np.exp(log_var))
        return gmu/len(mu), glog_var/len(log_var)

class KullbackLeiblerDivergenceNormal2(CompositFunction):
    """ 標準正規分布に対する任意の正規分布のKLDの解析解 """
    def _forward(self, mu, log_var):
        kll = -0.5 * F.sum(1 + log_var - mu**2 - F.exp(log_var))
        return kll / len(mu)

class MutualInformationLoss(Function):
    def __forward__(self, z, mu, log_var):
        self.z, self.mu, self.log_var = z, mu, log_var
        log_qz_cond_x = self.log_normal_density(z, mu, log_var)
        log_pz = self.log_standard_normal(z)
        mi_loss = np.sum(log_qz_cond_x - log_pz, axis=-1)
        return mi_loss/len(z)

    def log_normal_density(self, z, mu, log_var):
        normalization = -0.5 * (F.log(2 * np.pi) + log_var)
        log_density = normalization - 0.5 * ((z - mu) ** 2 / F.exp(log_var))
        return np.sum(log_density, axis=-1)

    def log_standard_normal(self, z):
        log_density = -0.5 * z ** 2 - 0.5 * F.log(2 * np.pi)
        return np.sum(log_density, axis=-1)

    def __backward__(self, gmil=1):
        z, mu, log_var = self.z, self.mu, self.log_var
        var = np.exp(log_var)
        dz = gmil * (-(z - mu) / var + z)
        dmu = gmil * ((z - mu) / var)
        dlog_var = gmil * 0.5 * ((z - mu)**2 / var - 1)
        return dz/len(z), dmu/len(mu), dlog_var/len(log_var)

class MutualInformationLoss2(CompositFunction):
    def _forward(self, z, mu, log_var):
        log_qz_cond_x = self.log_normal_density(z, mu, log_var)
        log_pz = self.log_standard_normal(z)
        mi_loss = F.mean(log_qz_cond_x - log_pz, axis=-1)
        return mi_loss

    def log_normal_density(self, z, mu, log_var):
        normalization = -0.5 * (F.log(2 * np.pi) + log_var)
        log_density = normalization - 0.5 * ((z - mu) ** 2 / F.exp(log_var))
        return F.sum(log_density, axis=-1)

    def log_standard_normal(self, z):
        log_density = -0.5 * z ** 2 - 0.5 * F.log(2 * np.pi)
        return F.sum(log_density, axis=-1)


# -- 潜在変数をサンプリングする層 仮版
class LatentLayer:
    def __init__(self, *configuration, **kwargs):
        if len(configuration) == 2:
            m, n = configuration
        if len(configuration) == 1:
            m = None; n, = configuration
        self.proj = NeuronLayer(m, 2*n, **kwargs) # 線形変換
        rate      = kwargs.pop('rate', 1.0)       # サンプリングの広がり        
        r_kl_loss = kwargs.pop('r_kl_loss', 1.0)  # kl_lossの混ぜ具合
        self.sampling = LatentSampling(rate=rate, kld=r_kl_loss)

    def forward(self, x, train=False, epsilon=None):
        self.x = x
        y = self.proj(x, train=train) # 2*n
        z = self.sampling(y, epsilon=epsilon)
        return z # (z, kll, mil)の場合もある
       
    def backward(self, gz=1, gkll=1, gmi=1, flush=True):
        grad_y = self.sampling.backward(gz, gkll=gkll, gmi=gmi)
        grad_x = self.proj.backward(grad_y, flush=flush)
        return grad_x

    def update(self, eta=0.001, **kwargs):
        self.proj.update(eta=eta, **kwargs)
    

#### 時系列データをまとめて処理するRNN層(Truncated BPTT方式) ##################
# m:Vector_size(入力数)、n:hidden_size(ニューロン数)

class WeightsAndBiasesForRNN:
    """ RNNの重みやバイアスを管理する """
    def __init__(self, layer, **kwargs):
        self.layer       = layer
        self.init_method = kwargs.pop('method', 'Orthogonal') # 重みの初期化手段
        self.width       = kwargs.pop('width',       None) # 重みの初期値の広がりを指定
        self.debug_mode  = kwargs.pop('debug_mode', False) # 重みを一律に初期化
        optimize         = kwargs.pop('optimize',   'SGD') # 最適化関数の指定 

        self.optimizer_w = cf.eval_in_module(optimize, Optimizers, **kwargs)
        self.optimizer_v = cf.eval_in_module(optimize, Optimizers, **kwargs)
        self.optimizer_b = cf.eval_in_module(optimize, Optimizers, bias=True, **kwargs)

        self.w, self.v, self.b = None, None, None 
        
    def __call__(self):
        if self.w is None:
            self.init_parameter()
        return self.w, self.v, self.b    

    def init_parameter(self): 
        l, m, n = self.layer.get_parameter_size() # l:戻りパス、m:入力、n:ニューロン数
        if l is None or m is None or n is None:
            raise Exception('Configuration is not fixed.', self.__class__.__name__)

        kwargs = {'method':self.init_method, 'debug_mode':self.debug_mode}     

        if self.layer.__class__.__name__=='RNN':
            self.w = init_weight((m, n), **kwargs)
            self.v = init_weight((l, n), **kwargs)
            self.b = np.zeros(n).astype(Config.dtype)
            
        elif self.layer.__class__.__name__=='LSTM':    
            wgf = init_weight((m, n), **kwargs) # 忘却
            wgi = init_weight((m, n), **kwargs) # 入力
            wgo = init_weight((m, n), **kwargs) # 出力
            wnm = init_weight((m, n), **kwargs) # 新記憶
            vgf = init_weight((l, n), **kwargs) # 忘却
            vgi = init_weight((l, n), **kwargs) # 入力
            vgo = init_weight((l, n), **kwargs) # 出力
            vnm = init_weight((l, n), **kwargs) # 新記憶
            bgf = np.ones(n).astype(Config.dtype)
            bgi = np.zeros(n).astype(Config.dtype)
            bgo = np.zeros(n).astype(Config.dtype)
            bnm = np.zeros(n).astype(Config.dtype)
            self.w = np.concatenate([wgf, wgi, wgo, wnm], axis=1)
            self.v = np.concatenate([vgf, vgi, vgo, vnm], axis=1)
            self.b = np.concatenate([bgf, bgi, bgo, bnm])
             
        elif self.layer.__class__.__name__=='GRU':    
            wgz = init_weight((m, n), **kwargs) # 忘却
            wgr = init_weight((m, n), **kwargs) # 入力
            wnm = init_weight((m, n), **kwargs) # 新記憶
            vgz = init_weight((l, n), **kwargs) # 忘却
            vgr = init_weight((l, n), **kwargs) # 入力
            vnm = init_weight((l, n), **kwargs) # 新記憶
            bgz = np.ones(n).astype(Config.dtype)
            bgr = np.zeros(n).astype(Config.dtype)
            bnm = np.zeros(n).astype(Config.dtype)
            self.w = np.concatenate([wgz, wgr, wnm], axis=1)
            self.v = np.concatenate([vgz, vgr, vnm], axis=1)
            self.b = np.concatenate([bgz, bgr, bnm])

        else:
            raise NotImplementedError(self.layer.__class__.__name__+' is not valid.')
        
    def update(self, eta=0.001, **kwargs):
        self.optimizer_w.update(self.w, self.grad_w, eta, **kwargs)
        self.optimizer_v.update(self.v, self.grad_v, eta, **kwargs)
        self.optimizer_b.update(self.b, self.grad_b, eta, **kwargs)

    def set_gradient(self, grad_w, grad_v, grad_b, flush=True):
        if flush:
            self.grad_w = grad_w
            self.grad_v = grad_v
            self.grad_b = grad_b
        else:
            self.grad_w += grad_w
            self.grad_v += grad_v
            self.grad_b += grad_b
            
    def flush_gradient(self):
        self.grad_w = np.zeros_like(self.w, dtype=Config.dtype)
        self.grad_v = np.zeros_like(self.v, dtype=Config.dtype)
        self.grad_b = np.zeros_like(self.b, dtype=Config.dtype)
        
class RnnBaseLayer(Function):
    """
    Layerは時系列の展開を担う．重みやバイアスはWeightsAndBiasesForRNNに委託
    時系列の展開に際して機能cellをインスタンス化してLayer内に蓄積する
    いっぽう機能cellではforwardの際の入出力や内部の状態を自身と一体で保持し、
    backwardの際にLayerに蓄積されたcellを呼出せばforwardの際の変数がそのまま使える
    すなわち、Layer側では時刻と対応するcellを関係づけるのみで良く、
    他方cell側では時系列を気にせずにforwardでの変数をbackwardで呼出すだけで良い　

    """
    def __init__(self, *configuration, **kwargs):
        super().__init__()
        self.cell = None
        stateful = False
        if len(configuration) == 3:
            m, n, stateful = configuration
        elif len(configuration) == 2:
            m, n = configuration
        elif len(configuration) == 1:
            m = None; n, = configuration
        elif len(configuration) == 0:
            m = None; n = 100
        else:
            raise Exception('Wrong configuration specified.')
        stateful        = kwargs.pop('stateful', stateful)
        self.config = m, n, stateful 
        print('Initialize', self.__class__.__name__,
                            self.config[:2], 'stateful', self.config[2])
        
        self.parameters = WeightsAndBiasesForRNN(self, **kwargs)
        
        self.cell_normalization = kwargs.pop('cell_normalization', False)
                                                # セル正規化(層正規化相当)の適用有無
        self.CPT = Capture()
        self.DO  = Dropout()

        self.last_state = None
        self.r0, self.c0 = None, None
        self.grad_r0, self.grad_c0 = None, None # seq2seqなどの場合に外部から参照
        self.r0_given, self.c0_given = False, False
        self.mask = None    
        
    def fix_configuration(self, shape):
        m, n, stateful = self.config
        self.config = shape[-1], n, stateful
        print(self.__class__.__name__, 'fix_configuration', shape, self.config)

    def get_parameter_size(self):
        # 実際のサイズはCellの種類による
        m, n, stateful = self.config
        return n, m, n 

    def set_state(self, r0, c0=None):
        self.r0, self.c0 = r0, c0     # last_state
        #self.flush_gradient()
        #self.ys = []

    def reset_state(self):
        self.r0, self.c0 = None, None # last_state
        #self.flush_gradient()
        
    def flush_gradient(self):
        self.parameters.flush_gradient()
        
    def update(self, eta=0.001, **kwargs):
        self.parameters.update(eta=eta, **kwargs)

    def __forward__(self, x, r0=None, c0=None, *, mask=None, CPT=None, dropout=0.0):
        """
        順伝播で一時刻に一つcellのインスタンスを生成しlayerを形成する
        順伝播の際、入出力や内部状態はcellと一体で時刻毎にlayerに記憶される
        r0/c0は初回は零で、statefulならば2回目以降はそのまま使う
        最後の時刻のrt/ctをr0/c0に保存し、statefulならば次回の順伝播で使う

        """
        if None in self.config:
            #print(self.__class__.__name__, 'input.shape', x.shape)
            self.fix_configuration(x.shape)

        w, v, b = self.parameters()
        
        # -- r0とc0を初期化し、yの器と、cellの器layerを用意
        B, T, m = x.shape          # B:バッチサイズ、T:時系列長
        _, n, stateful = self.config
        if r0 is not None and r0.shape==(B, n):    # 外から設定
            self.r0_given = True
        elif r0 is None and (not stateful or self.r0 is None): # 初期値を設定
            r0 = np.zeros((B, n), dtype=Config.dtype)
        elif stateful and self.r0 is not None and self.r0.shape==(B, n): # 前の値を継承
            r0 = self.r0
        else:
            print(self.config, x.shape, self.r0.shape)
            raise Exception(self.__class__.__name__+' state held inconsistent. May need reset_state().')
        if c0 is not None and c0.shape==(B, n):    # 外から設定
            self.c0_given = True
        elif c0 is None and (not stateful or self.c0 is None): # 初期値を設定
            c0 = np.zeros((B, n), dtype=Config.dtype)
        elif stateful and self.c0 is not None and self.c0.shape==(B, n): # 前の値を継承
            c0 = self.c0
        else:
            print(self.config, x.shape, self.c0.shape)
            raise Exception(self.__class__.__name__+' state held inconsistent. May need reset_state().')
        y = np.empty((B, T, n), dtype=Config.dtype)
        self.layer = []

        # 仮テスト
        #mask = np.ones((B, T), dtype=bool)
 
        if mask is not None:
            if x.shape[:-1]==mask.shape:
                self.mask = mask.astype(x.dtype)
            else:
                raise ValueError(
                    f"Mask shape {mask.shape} does not match input shape {x.shape}" \
                    + self.__class__.__name__)
        
        # -- 時系列を展開して順伝播 --
        rt = r0.copy()
        ct = c0.copy()
        for t in range(T):
            cell = self.cell(self.cell_normalization) # 選択したユニットのインスタンス生成
            xt = x[:, t, :]
            rtt, ctt = cell.forward(xt, rt, ct, w, v, b) # rt,ct上書き
            if mask is None:
                rt, ct = rtt, ctt
            else:
                mt = self.mask[:, t, None]
                rt = mt * rtt + (1 - mt) * rt
                ct = mt * ctt + (1 - mt) * ct
            y[:, t, :] = rt
            self.layer.append(cell)

        self.r0 = rt                       # 最後の時刻の出力　
        self.c0 = ct                       # 最後の時刻の出力
        #self.r0[...] = rt                 # 属性継承して値を最後の時刻の出力で更新　
        #self.c0[...] = ct                 # 属性継承して値を最後の時刻の出力で更新
        
        y = self.CPT.forward(y, width=CPT) # 一部出力のみの使用に対応
        # CPT指定した場合はドロップアウトしない
        y = self.DO.forward(y, dropout=dropout if CPT is None else 0.0)
        # 出力はドロップアウト対象で隠れ状態は非対象
        return y

    def __backward__(self, grad_y, flush=True): # grad_yは下流から受け取る勾配
        """
        順伝播の際にlayerに時刻毎のcellと変数が保存されているから、
        それを順伝播とは逆順に呼出して順伝播に準じた手順で逆伝播を行う
        xの勾配は器を用意して対応する時刻にはめ込んでいくいっぽう、
        w,v,bの勾配は、時系列に亘り算出した値を累積して、それを保存する
        最後に算出したrt/ctの勾配はr0/c0の勾配として保存する

        """
        # ドロップアウトした出力の勾配は逆伝播しないが隠れ状態からの勾配は逆伝播する
        grad_y = self.DO.backward(grad_y) 
        # 一部の時刻をキャプチャした場合の時系列長の調整はCaptureクラスが担う
        grad_y = self.CPT.backward(grad_y)     # 出力(下流)から内部へ遡上する勾配
        B, T, n = grad_y.shape                 # 時系列長Tが欲しい
        m, _, _ = self.config

        w, v, b = self.parameters()

        # -- rとcの勾配を初期化し、xの勾配の器を用意
        if flush:
            self.parameters.flush_gradient()

        grad_x = np.empty((B, T, m), dtype=Config.dtype)
        grad_rt = np.zeros_like(self.r0, dtype=Config.dtype)
        grad_ct = np.zeros_like(self.c0, dtype=Config.dtype)

        # -- 時系列を呼出して逆伝播 --
        for t in reversed(range(T)):
            cell = self.layer[t]
            grad_yt = grad_y[:, t, :] + grad_rt  # 出力からとリカレントを合算

            if self.mask is None:
                grad_xt, grad_rt, grad_ct, grad_wt, grad_vt, grad_bt = \
                    cell.backward(grad_yt, grad_ct, w, v, b) # grad_rt,grad_ct上書き 　

            else:
                mt = self.mask[:, t, None]  # shape: (B, 1)
                # maskが1の位置だけに対応する勾配を渡す
                grad_xt, grad_rtt, grad_ctt, grad_wt, grad_vt, grad_bt = \
                    cell.backward(mt*grad_yt, mt*grad_ct, w, v, b)
                # maskが1なら新しい勾配を、0ならそのまま伝搬
                grad_rt = grad_rtt + (1 - mt) * grad_rt
                                             # grad_yt? # mt==0に対応する位置はgrad_rtt==0
                grad_ct = grad_ctt + (1 - mt) * grad_ct # mt==0に対応する位置はgrad_ctt==0

            grad_x[:, t, :] = grad_xt
            # 展開中は勾配をparametersに直接蓄積する(flushしてはいけない)
            self.parameters.set_gradient(grad_wt, grad_vt, grad_bt, flush=False)

        self.grad_r0 = grad_rt # 最後に算出するgrad_rtはr0の勾配
        self.grad_c0 = grad_ct # c0の勾配は使わないが念のため

        if not self.r0_given:
            return grad_x
        if not self.c0_given:
            self.r0_given = False
            return grad_x, self.grad_r0
        self.c0_given = False
        return grad_x, self.grad_r0, self.grad_c0

    def step_and_stack(self, x):
        """ １時刻ずつデータを処理して状態を蓄積 """
        m, n, stateful = self.config  # B, m = x.shape; B, n = t.shape
        if x.ndim==2:
            B, _ = x.shape
        else:
            B = 1
            x = x.reshape(B, m)

        w, v, b = self.parameters()

        # -- r0とc0を初期化 --
        if self.r0 is None:
            self.r0 = np.zeros((B, n), dtype=Config.dtype)
        if self.c0 is None:
            self.c0 = np.zeros((B, n), dtype=Config.dtype)
        
        # -- リカレントにr0をセットし、cellを起こして順伝播 --
        r = self.r0
        c = self.c0
        cell = self.cell() # 選択したユニットのインスタンス生成
        y, c = cell.forward(x, r, c, w, v, b)
        self.layer.append(cell)
        self.r0 = y
        self.c0 = c

        return y

    def init_parameter(self):    
        raise Exception('Invalid configuration')

#### 時系列データをまとめて処理するN層(Truncated BPTT方式) ##################
"""
 m:Vector_size(入力数)、n:hidden_size(ニューロン数)
 cellのインスタンス化はforwardで時系列展開に際して行うため、
 cell内でnormalizationの有無に応じた処理分けが出来ない．
 やむなくnormalization無しと有りに分けたcellを用意して呼び分ける

"""
class RNN(RnnBaseLayer):
    def __init__(self, *configuration, **kwargs):
        super().__init__(*configuration, **kwargs)
        self.cell = RNN_Cell

class LSTM(RnnBaseLayer):
    def __init__(self, *configuration, **kwargs):
        super().__init__(*configuration, **kwargs)
        self.cell = LSTM_Cell

class GRU(RnnBaseLayer):
    def __init__(self, *configuration, **kwargs):
        super().__init__(*configuration, **kwargs)
        self.cell = GRU_Cell

               
#### RNN各種機能ユニット ###################################################
#### w,v,bは時系列共通、layer側で保持
        
class RNN_Cell:
    def __init__(self, normalize=False):
        self.dual_dot_linear = F.DualDotLinear()
        if normalize:
            self.norm = F.Normalize(axis=-1)
        else:
            self.norm = None
        self.normalize = normalize
        self.tanh = F.Tanh()
        
    def forward(self, x, r, c, w, v, b):   # cは使わない
        u = self.dual_dot_linear.forward(x, r, w, v, b)
        if self.normalize:
            u = self.norm.forward(u)       # 正規化
        y = self.tanh.forward(u)                   
        self.state = x, r, y               # rは前時刻のy
        return y, c                        # 出力,cはそのまま

    def backward(self, grad_y, grad_c, w, v, b): # c,g関連は使わない
        x, r, y = self.state
        delta = self.tanh.backward(grad_y)
        if self.normalize:
            delta = self.norm.backward(delta)  
        grad_x, grad_r, grad_w, grad_v, grad_b = self.dual_dot_linear.backward(delta)
        return grad_x, grad_r, grad_c, grad_w, grad_v, grad_b 

class LSTM_Cell:
    def __init__(self, normalize=False):
        self.dual_dot_linear = F.DualDotLinear()
        if normalize:
            self.norm_g = F.Normalize(axis=-1)
            self.norm_c = F.Normalize(axis=-1)
        else:
            self.norm_g = None
            self.norm_c = None
        self.normalize = normalize
        
    def forward(self, x, r, cp, w, v, b):   # 入力、前時刻状態
        B, n = r.shape
        u = self.dual_dot_linear.forward(x, r, w, v, b)
        if self.normalize:
            u = self.norm_g.forward(u)      # 諸ゲートの正規化
        gz = 1 / (1 + np.exp(-u[:, :3*n]))  # sigmoid 諸ゲート　
        gm = np.tanh(u[:, 3*n:])            # tanh  新しい記憶
        gf = gz[:, :n]                      # 忘却ゲート
        gi = gz[:, n:2*n]                   # 入力ゲート
        go = gz[:, 2*n:]                    # 出力ゲート
        cn = cp * gf + gm * gi              # 旧記憶＊忘却ゲート＋新記憶＊入力ゲート
        if self.normalize:
            cn = self.norm_c.forward(cn)    # 記憶の正規化
        y = np.tanh(cn) * go                # 記憶＊出力ゲート
        g = np.hstack((gz, gm))             # 内部状態
        self.state = x, r, cp, y, cn, g 
        return y, cn                        # 出力

    def backward(self, grad_y, grad_cn, w, v, b): # 引数＝下流からの勾配
        x, r, cp, y, cn, g = self.state
        B, n = r.shape
        gz = g[:,:3*n]                      # 忘却ゲート、入力ゲート、出力ゲート             
        gf = g[:,:n]                        # 忘却ゲート
        gi = g[:,n:2*n]                     # 入力ゲート
        go = g[:,2*n:3*n]                   # 出力ゲート
        gm = g[:,3*n:]                      # 新しい記憶
        tanh_c = np.tanh(cn)
        gcn = grad_cn + (grad_y * go) * (1 - tanh_c ** 2)
        if self.normalize:
            gcn = self.norm_c.backward(gcn) # 記憶の正規化の逆伝播
        
        dgm = gcn * gi                      # 新しい記憶の勾配

        # 諸ゲートの勾配： 忘却 dgf  入力 dgi  出力 dgo　 　　　　　　　　　　　　　
        dgz = np.hstack((gcn * cp, gcn * gm, grad_y * tanh_c))

        # 諸ゲート sigmoidの微分 と 新しい記憶  tanhの微分
        delta = np.hstack((dgz * gz * (1 - gz), dgm * (1 - gm ** 2)))
        if self.normalize:
            delta = self.norm_g.backward(delta) # 諸ゲートの正規化の逆伝播
            
        grad_cp = gcn * gf                    
        grad_x, grad_r, grad_w, grad_v, grad_b \
                            = self.dual_dot_linear.backward(delta)
        return grad_x, grad_r, grad_cp, grad_w, grad_v, grad_b 

class GRU_Cell:
    def __init__(self, normalize=False):
        self.dual_dot_linear_g = F.DualDotLinear()
        self.dual_dot_linear_m = F.DualDotLinear()
        if normalize:
            self.norm_g = F.Normalize(axis=-1)
            self.norm_u = F.Normalize(axis=-1)
        self.normalize = normalize
        
    def forward(self, x, r, c, w, v, b): # cは使わない
        B, n = r.shape
        wg, wm = w[:, :2*n], w[:, 2*n:]
        vg, vm = v[:, :2*n], v[:, 2*n:]
        bg, bm = b[:2*n],    b[2*n:]
   
        # 更新ゲートとリセットゲート
        #gu = np.dot(x, wg) + np.dot(r, vg) + bg
        gu = self.dual_dot_linear_g.forward(x, r, wg, vg, bg)
        if self.normalize:
            gu = self.norm_g.forward(gu)   # 諸ゲート正規化
        
        g = 1 / (1 + np.exp(-gu))          # sigmoid
        gz, gr = g[:, :n], g[:, n:]        # 更新ゲートとリセットゲート
        # 新しい記憶
        #u = np.dot(x, wm) + np.dot(gr * r, vm) + bm
        u = self.dual_dot_linear_m.forward(x, gr*r, wm, vm, bm)
        if self.normalize:
            u = self.norm_u.forward(u)     # 記憶正規化
        
        y = (1 - gz) * r + gz * np.tanh(u)
        self.state = x, r, y, g 
        return y, c                        # 出力,cはそのまま
    
    def backward(self, grad_y, grad_c, w, v, b): # c関連は使わない 
        x, r, y, g = self.state
        B, n = r.shape
        wg, wm = w[:, :2*n], w[:, 2*n:]
        vg, vm = v[:, :2*n], v[:, 2*n:]
        gz, gr = g[:, :n],   g[:, n:]
        
        # y算出の逆伝播
        tanh_u  = (y - (1 - gz) * r) / gz
        grad_r  = grad_y * (1 - gz) 
        grad_gz = grad_y * (tanh_u - r) ## 修正
        
        # 新しい記憶　
        delta_m = grad_y * gz * (1 - tanh_u ** 2) # 修正
        if self.normalize:
            delta_m = self.norm_u.backward(delta_m) 

        grad_x, grad_rm, grad_wm, grad_vm, grad_bm \
                        = self.dual_dot_linear_m.backward(delta_m)
        # gr * r の逆伝播 
        grad_r += gr * grad_rm
        grad_gr = grad_rm * r 

        # 更新ゲートとリセットゲート
        delta_g = np.hstack((grad_gz, grad_gr)) * g * (1 - g) # sigmoidの微分
        if self.normalize:
            delta_g = self.norm_g.backward(delta_g)
        
        grad_xm, grad_rm, grad_wg, grad_vg, grad_bg \
                        = self.dual_dot_linear_g.backward(delta_g)
        grad_r += grad_rm
        grad_x += grad_xm
        grad_w  = np.hstack((grad_wg, grad_wm))
        grad_v  = np.hstack((grad_vg, grad_vm))
        grad_b  = np.hstack((grad_bg, grad_bm))
        
        return grad_x, grad_r, grad_c, grad_w, grad_v, grad_b    


#### Embedding層用のparameter管理クラス #######################
# w:その行に対応する語のベクトルを各行が示す(行数m=語彙数、列数n=語ベクトル長)
#   全体で単語の分散表現
class ParametersForEmbedding:
    def __init__(self, layer, **kwargs):
        self.layer      = layer
        self.method     = kwargs.pop('method', 'uniform')
        self.width      = kwargs.pop('width',       None)
        self.debug_mode = kwargs.pop('debug_mode', False) # 重みを一律に初期化
        optimize        = kwargs.pop('optimize',   'SGD') 
        self.w, self.grad_w = None, None
        self.optimizer_w = cf.eval_in_module(optimize, Optimizers, **kwargs)

    def __call__(self):
        if self.w is None:
            self.init_parameter()
        return self.w    

    def init_parameter(self):
        # 通常のニューロンとは違い、出力の次元数に応じた一様乱数で初期化(u(-√1/D,√1/D))
        m, n = self.layer.get_parameter_size()
        if self.method == 'uniform':
            width = np.sqrt(1/n) if self.width is None else self.width
            self.w = init_weight((m, n),
                                 distribution='uniform',
                                 width=width,
                                 debug_mode=self.debug_mode)
        elif self.method == 'Orthogonal':
            #width = np.sqrt(1/n) if self.width is None else self.width
            self.w = init_weight((m, n), method='Orthogonal', debug_mode=self.debug_mode)
        else:
            raise Exception('Invalid method for ' + self.__class__.__name__+' specified.')

    def update(self, eta=0.001, **kwargs):
        self.optimizer_w.update(self.w, self.grad_w, eta=eta, **kwargs)

    def accommodate(self):
        m, n = self.layer.get_parameter_size()
        if m <= self.w.shape[0]:
            return
        print(self.__class__.__name__,
                                  'expand the size of w to accommodate new vocabulary.')
        xpcn = m - self.w.shape[0] # 拡張する行数
        center = np.mean(self.w, axis=0)
        new_rows = center + np.random.normal(0, 0.01, size=(xpcn, n), dtype=Config.dtype)
        print('new_rows =', new_rows.shape)
        self.w = np.concatenate([self.w, new_rows], axis=0)
        
    def set_gradient(self, x, gy, flush=True): # 未
        if flush:
            self.grad_w = np.zeros_like(self.w, dtype=Config.dtype)
        elif self.grad_w.shape == self.w.shape:
            pass
        snp.add_at(self.grad_w, x, gy)
           
            
#### 時系列データをまとめて処理する Embedding層 #######################
# m:vocab_size(語彙数)、n:wordvec_size(語ベクトル長)
class Embedding(Function):
    def __init__(self, *configuration, **kwargs):
        super().__init__()
        if len(configuration) == 2:
            m, n = configuration
        if len(configuration) == 1:
            m = 10000; n, = configuration
        if len(configuration) == 0:
            m = 10000; n = 100
        self.config = m, n    
        print('Initialize', self.__class__.__name__, self.config)
        self.parameters  = ParametersForEmbedding(self, **kwargs)
        self.mask = None            

    def get_parameter_size(self):
        return self.config

    def update(self, eta=0.001, **kwargs):
        self.parameters.update(eta=eta, **kwargs)

    def accommodate(self):
        self.parameters.accommodate()
        
    def __forward__(self, x, *, mask=None, dropout=0.0, **kwargs):
        """
        入力 x は w のどの行を抽出するかを示し
        yはxの指すwの行、xの形状(B,T)に対し、yの形状は(B, T, n)
        即ち長さnのベクトルがバッチ数×展開時間だけ並ぶ
        kwargsは使わないが、他の層と併せて呼ばれる際の引数対応
        
        """
        self.x = x
        w = self.parameters()
        y = w[x]
       
        if mask is None:
            self.mask = None
            return y
        if x.shape==mask.shape:
            self.mask = mask[..., None].astype(y.dtype)
            return y * self.mask 
        raise ValueError(f"Mask shape {mask.shape} does not match input shape {x.shape}" \
                           + self.__class__.__name__)

    def __backward__(self, gy):
        x = self.x
        gx = gy if self.mask is None else gy * self.mask
        self.parameters.set_gradient(x, gx)
            
        
#### 位置符号化 ####################################################　   
class PositionalEmbedding: 
    """ 入力の値に対する埋め込みと、その位置インデクスに対する埋め込みを合せて出力する """
    def __init__(self, vocab_size=10000, block_size=500, dimension=64, noise=0, **kwargs):
        self.token_embedding = Embedding(vocab_size, dimension, **kwargs)
        self.position_embedding = Embedding(block_size, dimension, **kwargs)
        self.block_size = block_size
        self.broadcast_to = None # forwardで形状が決まってから設定
        self.noise = noise
        
    def forward(self, x):
        """ 順伝播：positionはblock_sizeの範囲の0から始まる値でxと同形状 """
        token_vector = self.token_embedding(x)
        position = np.arange(x.shape[-1]).reshape(1, -1) % self.block_size
        position_vector = self.position_embedding(np.broadcast_to(position, x.shape))
        y = token_vector + position_vector
                                  # 出力をインプレース更新で作るとbacktrace失敗0250316AI
        if self.noise > 0:
            y += np.random.randn(*y.shape) * self.noise
        self.y = y    
        return y 

    def backward(self, gy=None):
        """ 逆伝播：入力へ勾配は伝搬しないから形状や剰余などの対応不要 """
        if gy is None:
            gy = np.ones_like(self.y)
        self.position_embedding.backward(gy) 
        self.token_embedding.backward(gy)
        
    def update(self, **kwargs):
        self.token_embedding.update(**kwargs)
        self.position_embedding.update(**kwargs)

    def accommodate(self, mode='token'):
        if mode in ('token', 'both'):
            self.token_embedding.accommodate() 
        if mode in ('position', 'both'):
            self.position_embedding.accommodate()
        else:
            ValueError(f'Invalid mode {mode}')

class PositionalEncoding:
    def __init__(self, sequence_length=10000, dimension=2):
        if dimension % 2 != 0:
            raise ValueError(self.__class__.__name__ 
                + f": dimension must be even (got {dimension}) "
                   "because we use (sin, cos) pairs."
                )
        self.dimension = dimension
        self.sequence_length = sequence_length
        self.division \
            = np.exp(np.arange(0, dimension, 2) * -(np.log(sequence_length) / dimension))
        # 上記はdivisionを決める際にsequence_lengthを見ているが元は下記
        #   = np.exp(np.arange(0, dimension, 2) * -(np.log(10000) / dimension))

        print('シーケンス長', sequence_length, '次元数', dimension, '分割', self.division)

    def __call__(self, positions):
        positions_shape = positions.shape
        positions = positions.reshape(-1, 1)
        pe = np.zeros((positions.shape[0],) + (self.dimension,)) # 末尾の次元を入替
        pe[:, 0::2] = np.sin(positions * self.division)
                                                  # 偶数次元に対するサイン関数の適用
        pe[:, 1::2] = np.cos(positions * self.division)
                                                  # 奇数次元に対するコサイン関数の適用
        pe = pe.reshape(positions_shape + (self.dimension,)) # 元の次元に末尾を加えた形状
        return pe

    def forward(self, positions, **kwargs): # kwargsは使わない
        return self.__call__(positions)


class PositionalEmbedding2: # 逆伝播が書けない
    """ 入力の値に対する埋め込みと、その位置インデクスに対する埋め込みを合せて出力する """
    def __init__(self, vocab_size=10000, block_size=500, dimension=64, **kwargs):
        self.token_embedding    = Embedding(vocab_size, dimension, **kwargs)
        self.position_embedding = Embedding(block_size, dimension, **kwargs)
        self.block_size = block_size
        
    def forward(self, x):
        token_vactor = self.token_embedding(x)
        position = np.arange(x.shape[-1]) % self.block_size
                                                # 位置インデクスのブロック長の範囲の値
        position_vector = self.position_embedding(position)
        y = token_vactor + position_vector
        return y

    def update(self, **kwargs):
        self.token_embedding.update(**kwargs)
        self.position_embedding.update(**kwargs)

class PositionalEmbedding_bkup: # broadcast_toはおかしいが、取りあえず残しておく20250315AI
    """ 入力の値に対する埋め込みと、その位置インデクスに対する埋め込みを合せて出力する """
    def __init__(self, vocab_size=10000, block_size=500, dimension=64, **kwargs):
        self.token_embedding = Embedding(vocab_size, dimension, **kwargs)
        self.position_embedding = Embedding(block_size, dimension, **kwargs)
        self.block_size = block_size
        self.broadcast_to = None # forwardで形状が決まってから設定
        
    def forward(self, x):
        tok_emb = self.token_embedding(x)
        position = np.arange(x.shape[-1]) % self.block_size
                                       # 位置インデクスのブロック長の範囲の値
        pos_emb = self.position_embedding(position)
        #print('###debug', self.__class__.__name__, x.shape,tok_emb.shape,pos_emb.shape)
        self.broadcast_to = F.BroadcastTo(tok_emb.shape)     # バッチ軸を拡張
        pos_emb = self.broadcast_to(pos_emb)
        #print('###', pos_emb.shape)
        y = tok_emb + pos_emb
        self.y = y
        return y

    def backward(self, gy=None):
        if gy is None:
            gy = np.ones_like(self.y)
        #gz = np.sum(gy, axis=0)
        gz = self.broadcast_to.backward(gy)
        self.position_embedding.backward(gz)
                                      # 入力へ勾配は伝搬しないから剰余などの対応不要
        self.token_embedding.backward(gy)    # こちらも入力へ勾配は伝搬しない

    def update(self, **kwargs):
        self.token_embedding.update(**kwargs)
        self.position_embedding.update(**kwargs)


class PatchEmbedding(Conv2dLayer):
    """
    画像からトークンへの変換を行う．　
    すなわち、画像を小さなパッチに切り分けて、
    各パッチをベクトルにフラット化して、
    線形変換によって埋め込みベクトルに変換する．

    """
    def __init__(self, dimensionality=128, patch_size=2, pad=0, **kwargs):
        #                M,              kernel_size, stride,     pad
        super().__init__(dimensionality, patch_size,  patch_size, pad, **kwargs)

    def _forward(self, x):
        C,Ih,Iw,M,Fh,Fw,Sh,Sw,pad,Oh,Ow = self.config
        y = super()._forward(x)
        y = y.transpose(0,2,3,1).reshape(-1,Oh*Ow,M)
        return y

    def _backward(self, grad_y, flush=True):
        C,Ih,Iw,M,Fh,Fw,Sh,Sw,pad,Oh,Ow = self.config
        grad_y = grad_y.reshape(-1,Oh,Ow,M).transpose(0,3,1,2)
        grad_x = super()._backward(grad_y, flush=flush)
        return grad_x

class PatchEmbeddingSimple(Function):
    def __init__(self, dimensionality=128, patch_size=2, **kwargs):
        super().__init__()
        self.patch_size = patch_size
        self.linear = LinearLayer(dimensionality, matmul=True, **kwargs)
       
    def __forward__(self, x):
        self.x = x
        B, C, Ih, Iw = x.shape
        p = self.patch_size

        # パッチ分割 (B,C,Ih,Iw) -> (B,-1,C*P*P)
        x = snp.reshape(x, (B, C, Ih//p, p, Iw//p, p))
        x = snp.transpose(x, (0, 1, 2, 4, 3, 5))
        x = snp.reshape(x, (B, C, -1, p, p))
        x = snp.transpose(x, (0, 2, 1, 3, 4))
        x = snp.reshape(x, (B, -1, C * p * p))

        # 埋め込み
        y = self.linear(x)
        return y

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def __backward__(self, gy):
        x = self.x
        B, C, Ih, Iw = x.shape
        p = self.patch_size
        
        gx = self.linear.backward(gy)

        gx = snp.reshape(gx, (B, -1, C, p, p))
        gx = snp.transpose(gx, (0, 2, 1, 3, 4))
        gx = snp.reshape(gx, (B, C, Ih//p, Iw//p, p, p))
        gx = snp.transpose(gx, (0, 1, 2, 4, 3, 5))
        gx = snp.reshape(gx, (B, C, Ih, Iw))

        return gx

    def update(self, eta=0.001, **kwargs):
        self.linear.update(eta=eta, **kwargs)

class Unpatchfy(Function):
    def __init__(self, img_size=None, patch_size=2, **kwargs):
        super().__init__()
        self.config = img_size, patch_size
        # linearのconfigは遅延設定だが、img_sizeとpatch_sizeを与えれば決まる        
        self.linear = LinearLayer(None, matmul=True, **kwargs)

    def fix_configuration(self, shape):
        """ linearのconfigを設定 """
        # config = (img_size, patch_size) は事前に設定しておく必要がある
        if None in self.config:
            raise ValueError(f'Either img_size or patch_size is bad.')
        img_size, patch_size = self.config
        out_dim = img_size[0] * patch_size * patch_size        
        self.linear.config = shape[-1], out_dim
        print(self.__class__.__name__, 'fix_configuration', shape, self.config)

    def __forward__(self, x, **kwargs):
        self.x = x
        if None in self.linear.config: # linearの設定を確認
            self.fix_configuration(x.shape)
            
        B, T, H = x.shape
        P = self.config[1]         # patch_size
        C, Ih, Iw = self.config[0] # img_size
        m, n = Ih//P, Iw//P
        if (Ih/P)*(Iw/P)!=T:
            raise ValueError(f'Either input shape {x.shape} or config {self.config} is bad.')

        y = self.linear(x, **kwargs)        # (B,T,H) → (B,T,D) 

        _, _, D = y.shape
        if C*P*P!=D: 
            raise ValueError(f'Either config {self.linear.config} or output shape {self.img_size} is bad.')

        y = snp.reshape(y, (B, T, C, P, P))        # (B,T,D) → (B,T,C,P,P)
        y = snp.reshape(y, (B, m, n, C, P, P))     # (B,T,C,P,P) → (B,m,n,C,P,P)
        y = snp.transpose(y, (0, 3, 1, 4, 2, 5))   # (B,m,n,C,P,P) → (B,C,m,P,n,P)
        y = snp.reshape(y, (B, C, Ih, Iw))         # (B,C,m,P,n,P) → (B,C,Ih,Iw)

        return y

    def __backward__(self, gy):
        x = self.x
        B, T, H = x.shape
        P = self.config[1]         # patch_size
        C, Ih, Iw = self.config[0] # img_size
        m, n = Ih//P, Iw//P

        gx = snp.reshape(gy, (B, C, m, P, n, P))   # (B,C,Ih,Iw) → (B,C,m,P,n,P)
        gx = snp.transpose(gx, (0, 2, 4, 1, 3, 5)) # (B,C,m,P,n,P) → (B,m,n,C,P,P)
        gx = snp.reshape(gx, (B, T, C, P, P))      # (B,m,n,C,P,P) → (B,T,C,P,P) ここで T=m*n
        gx = snp.reshape(gx, (B, T, C*P*P))        # (B,T,C,P,P) → (B,T,D)  ここで D=C*P*P

        gx = self.linear.backward(gx)       # (B,T,D) → (B,T,H)

        return gx

    def update(self, eta=0.001, **kwargs):
        self.linear.update(eta=eta, **kwargs)

class RotaryTransform(Function):
    def __init__(self, cos, sin):
        super().__init__()
        self.cos = cos
        self.sin = sin
        
    def __forward__(self, x):
        cos = self.cos
        sin = self.sin
        x0 = x[..., 0::2]
        x1 = x[..., 1::2]
        y = np.empty_like(x, dtype=Config.dtype)
        y[..., 0::2] = x0 * cos - x1 * sin
        y[..., 1::2] = x0 * sin + x1 * cos
        return y

    def __backward__(self, gy):
        cos = self.cos
        sin = self.sin
        gy0 = gy[..., 0::2]
        gy1 = gy[..., 1::2]
        gx = np.empty_like(gy, dtype=Config.dtype)
        gx[..., 0::2] =  gy0 * cos + gy1 * sin
        gx[..., 1::2] = -gy0 * sin + gy1 * cos
        return gx

class RotaryTransformHalf(Function):
    def __init__(self, cos, sin):
        super().__init__()
        self.cos = cos
        self.sin = sin

    def __forward__(self, x):
        cos = self.cos
        sin = self.sin
        d = x.shape[-1] // 2
        x0 = x[..., :d]
        x1 = x[..., d:]
        y = np.empty_like(x, dtype=Config.dtype)
        y[..., :d] = x0 * cos - x1 * sin
        y[..., d:] = x0 * sin + x1 * cos
        return y

    def __backward__(self, gy):
        cos = self.cos
        sin = self.sin
        d = gy.shape[-1] // 2
        gy0 = gy[..., :d]
        gy1 = gy[..., d:]
        gx = np.empty_like(gy, dtype=Config.dtype)
        gx[..., :d] =  gy0 * cos + gy1 * sin
        gx[..., d:] = -gy0 * sin + gy1 * cos
        return gx
    
class RoPE(Function):
    """
    Rotary Positional Embedding

    q, k shape:
        (..., sequence_length, head_dim)

    head_dim must be even.
    RoPE has no trainable parameters and keeps no forward input.
    """

    def __init__(self, base=10000.0, mode='interleaved'):
        super().__init__()
        self.base = base
        self.inv_freq = None # 周期
        self.cos_sin = None
        self.mode = mode
        if   mode == 'interleaved':
            self.RT = RotaryTransform
        elif self.mode == 'half':
            self.RT = RotaryTransformHalf
        else:
            raise NotImplementedError()

    def get_cos_sin(self, x):
        T = x.shape[-2]
        H = x.shape[-1]

        if self.inv_freq is None:
            if H % 2 != 0:
                raise ValueError(
                    f"RoPE requires even head_dim, but got {H}"
                )

            i = np.arange(H // 2, dtype=x.dtype)
            self.inv_freq = self.base ** (-2.0 * i / H)

        if self.cos_sin is None or len(self.cos_sin) < T:
            position = np.arange(T, dtype=x.dtype)
            theta = position[:, None] * self.inv_freq[None, :]
            self.cos_sin = np.stack(
                (np.cos(theta), np.sin(theta)), axis=-1
            )

        cos = self.cos_sin[:T, :, 0]
        sin = self.cos_sin[:T, :, 1]

        shape = (1,) * (x.ndim - 2) + (T, H // 2)

        cos = cos.reshape(shape)
        sin = sin.reshape(shape)

        return cos, sin

    def __forward__(self, q, k):
        cos_q, sin_q = self.get_cos_sin(q)
        cos_k, sin_k = self.get_cos_sin(k)
        self.rotateq = self.RT(cos_q, sin_q)
        self.rotatek = self.RT(cos_k, sin_k)
        q = self.rotateq(q)
        k = self.rotatek(k)
        return q, k

    def __backward__(self, grad_q, grad_k):
        grad_q = self.rotateq.backward(grad_q)
        grad_k = self.rotatek.backward(grad_k)
        return grad_q, grad_k


class RoPE_bkup(Function):
    """
    Rotary Positional Embedding

    q, k shape:
        (..., sequence_length, head_dim)

    head_dim must be even.
    RoPE has no trainable parameters and keeps no forward input.
    """

    def __init__(self, base=10000.0):
        super().__init__()
        self.base = base


    def _make_cos_sin(self, x):
        seq_len = x.shape[-2]
        dim = x.shape[-1]

        if dim % 2 != 0:
            raise ValueError(
                f"RoPE requires even head_dim, but got {dim}"
            )

        # 各2次元ペアに対する回転周波数
        i = np.arange(dim // 2, dtype=x.dtype)
        inv_freq = self.base ** (-2.0 * i / dim)

        # position = 0, 1, 2, ...
        position = np.arange(seq_len, dtype=x.dtype)

        theta = position[:, None] * inv_freq[None, :]

        cos = np.cos(theta)
        sin = np.sin(theta)

        # (..., sequence_length, head_dim/2) に broadcast
        shape = (1,) * (x.ndim - 2) + (seq_len, dim // 2)

        cos = cos.reshape(shape)
        sin = sin.reshape(shape)

        return cos, sin


    def _rotate(self, x, cos, sin):
        x0 = x[..., 0::2]
        x1 = x[..., 1::2]

        y = np.empty_like(x)

        y[..., 0::2] = x0 * cos - x1 * sin
        y[..., 1::2] = x0 * sin + x1 * cos

        return y


    def _rotate_backward(self, gy, cos, sin):
        """
        forward:
            y = R x

        backward:
            gx = R^T gy = R(-theta) gy
        """
        gy0 = gy[..., 0::2]
        gy1 = gy[..., 1::2]

        gx = np.empty_like(gy)

        gx[..., 0::2] =  gy0 * cos + gy1 * sin
        gx[..., 1::2] = -gy0 * sin + gy1 * cos

        return gx


    def __forward__(self, q, k):
        if q.shape != k.shape:
            raise ValueError(
                f"RoPE requires q and k to have the same shape: "
                f"q={q.shape}, k={k.shape}"
            )

        cos, sin = self._make_cos_sin(q)

        q = self._rotate(q, cos, sin)
        k = self._rotate(k, cos, sin)

        return q, k


    def __backward__(self, grad_q, grad_k):
        if grad_q.shape != grad_k.shape:
            raise ValueError(
                f"RoPE requires grad_q and grad_k to have the same shape: "
                f"grad_q={grad_q.shape}, grad_k={grad_k.shape}"
            )

        cos, sin = self._make_cos_sin(grad_q)

        grad_q = self._rotate_backward(grad_q, cos, sin)
        grad_k = self._rotate_backward(grad_k, cos, sin)

        return grad_q, grad_k



#### Attention機構 #################################################
# v:入力 value、k:入力 key、q:入力 query、y:出力、a:attention_weight
# query に一致する key を探して、その key に対応する value を出力する
# mask:self attentionで時系列を扱う場合に自身より先の時刻を無視するなど
# scale:コンテクストベクトルが大きい場合にaが大きくなりすぎるのを抑制
# 
class AttentionUnit_bkup(Function):
    """ 汎用AttentionUnit(MultiHead対応) """
    def __init__(self, head=1, **kwargs): 
        super().__init__()
        print('Initialize', self.__class__.__name__, 'head =', head, kwargs)
        self.head = head
        causality   = kwargs.pop('causality',  False)     # 時系列の前後関係 
        self.scale  = kwargs.pop('scale',       True)
        temperature = kwargs.pop('temperature',  1.0)  
        regularizer = kwargs.pop('regularizer', None)
       
        if type(regularizer) == str:
            self.regularizer = cf.eval_in_module(regularizer, Regularizers)
        else:
            self.regularizer = copy.deepcopy(regularizer) # インスタンス分離のために必須

        self.causality = causality
        self.softmax = Activators.Softmax(temperature=temperature)
        self.DO = Dropout()
        self.iter = 0
        self.loss = 0
        self.result1 = None
        self.result2 = None
        self.result3 = None
        self.tril = None # causalityの制御のマスク
        self.mask = None # 無効トークンのマスク
       
    def __forward__(self, v, k, q, *, mask=None, dropout=0.0):
        B,Tv,C = v.shape
        B,Tk,C = k.shape
        B,Tq,C = q.shape
        h = self.head
        H = C // h
        v = v.reshape(B,Tv,h,H).transpose(0,2,1,3) # (B,Tv,C)->(B,h,Tv,H)
        k = k.reshape(B,Tk,h,H).transpose(0,2,1,3) # (B,Tk,C)->(B,h,Tk,H)
        q = q.reshape(B,Tq,h,H).transpose(0,2,1,3) # (B,Tq,C)->(B,h,Tq,H)
        a = snp.matmul(q, k.transpose(0,1,3,2))     # (B,h,Tq,H)@(B,h,H,Tk)->(B,h,Tq,Tk)
        if self.scale:
            a *= np.array(H ** -0.5, dtype=a.dtype)

        if self.causality: # 時間の前後関係の保証
            if self.tril is not None and self.tril.shape==(Tq,Tk):
                pass
            elif Tq==Tk:
                self.tril = np.tril(np.ones((Tq,Tk), dtype=bool))
            else:
                raise Exception(f"causality cannot be applied" + self.__class__.__name__)
            a[:,:,self.tril==False] = - Config.inf

        if mask is None:   # 無効トークンの処理
            self.mask = None
        elif mask.shape == (B, Tk):
            self.mask = mask.astype(bool)
            a.transpose(0,3,1,2)[self.mask==False,:,:] = - Config.inf # (B,Tk,h,Tq)
        else:
            raise ValueError(f"Mask shape {mask.shape} must be ({B}, {Tk})" \
                           + self.__class__.__name__)

        a = self.softmax(a)
        
        if self.regularizer is not None: # aのエントロピーやKLDの算出
            self.loss = self.regularizer.forward(a)
            self.result1 = self.regularizer.result1
            self.result2 = self.regularizer.result2
            self.result3 = self.regularizer.result3
        self.iter += 1 

        a = self.DO.forward(a, dropout=dropout)
        y = snp.matmul(a, v)            # (B,h,Tq,Tk)@(B,h,Tv,H)->(B,h,Tq,H), Tv=Tk
        y = y.transpose(0,2,1,3).reshape(B,Tq,C)     # (B,h,Tq,H)->(B,Tq,C)
        self.k = k                                   # key
        self.q = q                                   # query
        self.v = v                                   # value
        self.a = a                                   # attention_weight
        self.y = y
        return y                                     # (B,Tq,C)

    def __backward__(self, gy):
        k = self.k
        q = self.q
        v = self.v
        a = self.a
        B, h, Tk, H = k.shape
        B, h, Tq, H = q.shape
        B, h, Tv, H = v.shape
        C = h * H
        gy = gy.reshape(B,Tq,h,H).transpose(0,2,1,3) # (B,Tq,C)->(B,h,Tq,H)
        ga = snp.matmul(gy, v.transpose(0,1,3,2))     # (B,h,Tq,H)@(B,h,H,Tv)->(B,h,Tq,Tv) Tv=Tk
        gv = snp.matmul(a.transpose(0,1,3,2), gy)     # (B,h,Tk,Tq)@(B,h,Tq,H)->(B,h,Tk,H) tk=Tv

        ga = self.DO.backward(ga)

        if self.regularizer is not None:          
            ga2 = self.regularizer.backward()
            ga += ga2                                # 勾配加算率はregularizer側に設定
        
        ga = self.softmax.backward(ga)
        
        if self.mask is not None:
            ga.transpose(0,3,1,2)[self.mask==False,:,:] = 0
        if self.causality:    
            ga[:,:,self.tril==False] = 0
        if self.scale:
            ga *= np.array(H ** -0.5, dtype=ga.dtype)

        gq = snp.matmul(ga, k)                      # (B,h,Tq,Tk)@(B,h,Tk,H)->(B,h,Tq,H)
        #gk = snp.matmul(q.transpose(0,1,3,2), ga)  # <-これはNG 20250525AI
        gk = snp.matmul(ga.transpose(0,1,3,2), q)   # (B,h,Tk,Tq)@(B,h,Tq,H)->(B,h,Tk,H)
        gv = gv.transpose(0,2,1,3).reshape(B,Tv,C) # (B,h,Tv,H)->(B,Tv,C)
        gk = gk.transpose(0,2,1,3).reshape(B,Tk,C) # (B,h,Tk,H)->(B,Tk,C)
        gq = gq.transpose(0,2,1,3).reshape(B,Tq,C) # (B,h,Tq,H)->(B,Tq,C)
        return gv, gk, gq

class AttentionUnit(Function):
    """ 汎用AttentionUnit(MultiHead対応) """
    def __init__(self, head=1, **kwargs): 
        super().__init__()
        print('Initialize', self.__class__.__name__, 'head =', head, kwargs)

        if isinstance(head, int):
            self.head = head, head, head
        elif isinstance(head, (tuple, list)) and len(head) == 2:
            self.head = head[0], head[1], head[1] 
        elif isinstance(head, (tuple, list)) and len(head) == 3:
            self.head = head[0], head[1], head[2] 
        else:
            raise ValueError(
                "head must be int or (q_head, kv_head) or (q_head, k_head, v_head)")

        causality   = kwargs.pop('causality',  False) # 時系列の前後関係 
        rope        = kwargs.pop('rope',       False) # Rotary Positional Embedding
        self.scale  = kwargs.pop('scale',       True)
        temperature = kwargs.pop('temperature',  1.0)  
        regularizer = kwargs.pop('regularizer', None)
       
        if type(regularizer) == str:
            self.regularizer = cf.eval_in_module(regularizer, Regularizers)
        else:
            self.regularizer = copy.deepcopy(regularizer) # インスタンス分離のために必須

        self.causality = causality
        self.rope = RoPE() if rope else None
        self.softmax = Activators.Softmax(temperature=temperature)
        self.DO = Dropout()
        self.iter = 0
        self.loss = 0
        self.tril = None # causalityの制御のマスク
        self.mask = None # 無効トークンのマスク
       
    def __forward__(self, q, k, v, *, mask=None, dropout=0.0):
        self.q = q                                   # query
        self.k = k                                   # key
        self.v = v                                   # value
        B,Tq,Cq = q.shape
        B,Tk,Ck = k.shape
        B,Tv,Cv = v.shape
        hq, hk, hv = self.head
        Hq = Cq // hq
        Hk = Ck // hk
        Hv = Cv // hv
        q = q.reshape(B,Tq,hq,Hq).transpose(0,2,1,3) # (B,Tq,Cq)->(B,hq,Tq,Hq)
        k = k.reshape(B,Tk,hk,Hk).transpose(0,2,1,3) # (B,Tk,Ck)->(B,hk,Tk,Hk)
        v = v.reshape(B,Tv,hv,Hv).transpose(0,2,1,3) # (B,Tv,Cv)->(B,hv,Tv,Hv)

        if self.rope is not None:
            q, k = self.rope(q, k)  
        
        if Hq != Hk:
            raise ValueError(f"q and k head_dim must be equal: Hq={Hq}, Hk={Hk}")

        if hq % hk != 0:
            raise ValueError(
                f"q_head must be divisible by k_head: q_head={hq}, k_head={hk}")
        if hq % hv != 0:
            raise ValueError(
                f"q_head must be divisible by v_head: q_head={hq}, v_head={hv}")

        a = snp.matmul(
            q.reshape(B,hk,hq//hk,Tq,Hq), k.transpose(0,1,3,2)[:,:,None,:,:])
        a = a.reshape(B,hq,Tq,Tk)   # (B,hk,hq//hk,Tq,Tk)->(B,hq,Tq,Tk)

        if self.scale:
            a *= np.array(Hk ** -0.5, dtype=a.dtype) # Hk==Hq

        if self.causality: # 時間の前後関係の保証
            if self.tril is not None and self.tril.shape[-2:]==(Tq,Tk):
                pass
            elif Tq==Tk:
                self.tril = np.tril(np.ones((1,1,Tq,Tk), dtype=Config.dtype))
                                                   #[None, None, :, :]
            else:
                raise Exception(f"causality cannot be applied" + self.__class__.__name__)
            a += (self.tril - 1.0) * Config.inf
       
        if mask is None:   # 無効トークンの処理 
            self.mask = None
        elif mask.shape==(B, Tk):
            self.mask = mask.astype(a.dtype)[:, None, None, :]       # (B,1,1,Tk)
            a += (self.mask - 1.0) * Config.inf  # 無効個所は-inf
        else:
            raise ValueError(f"Mask shape {mask.shape} must be ({B}, {Tk})" \
                           + self.__class__.__name__)
       
        a = self.softmax(a)
        
        if self.regularizer is not None: 
            self.loss = self.regularizer.forward(a)

        self.iter += 1 

        a = self.DO.forward(a, dropout=dropout)

        y = snp.matmul(a.reshape(B,hv,hq//hv,Tq,Tk) , v[:,:,None,:,:])
        y = y.reshape(B,hq,Tq,Hv)    # (B,hv,hq//hv,Tq,Hv) -> (B,hq,Tq,Hv)

        y = y.transpose(0,2,1,3).reshape(B,Tq,hq*Hv) # (B,hq,Tq,Hv)->(B,Tq,hq*Hv)
        self.a = a                                   # attention_weight
        #self.y = y                                   # (B,Tq,hq*Hv)
        return y                                     

    def __backward__(self, gy):
        q = self.q
        k = self.k
        v = self.v
        a = self.a
        B,Tq,Cq = q.shape
        B,Tk,Ck = k.shape
        B,Tv,Cv = v.shape
        hq, hk, hv = self.head
        Hq = Cq // hq
        Hk = Ck // hk
        Hv = Cv // hv
        q = q.reshape(B,Tq,hq,Hq).transpose(0,2,1,3) # (B,Tq,Cq)->(B,hq,Tq,Hq)
        k = k.reshape(B,Tk,hk,Hk).transpose(0,2,1,3) # (B,Tk,Ck)->(B,hk,Tk,Hk)
        v = v.reshape(B,Tv,hv,Hv).transpose(0,2,1,3) # (B,Tv,Cv)->(B,hv,Tv,Hv)

        if self.rope is not None:
            q, k = self.rope(q, k)  

        gy = gy.reshape(B,Tq,hq,Hv).transpose(0,2,1,3) # (B,Tq,hq*Hv)->(B,hq,Tq,Hv)

        ga = snp.matmul(
            gy.reshape(B,hv,hq//hv,Tq,Hv), v.transpose(0,1,3,2)[:,:,None,:,:])             
        ga = ga.reshape(B,hq,Tq,Tk)   # (B,hv,hq//hv,Tq,Tk) -> (B,hq,Tq,Tk)
        
        # グループ方向(hq//hv)の加算を行列積の縮約に含める(1回の積で済む)
        #   (B,hv,Tk,g*Tq)@(B,hv,g*Tq,Hv) -> (B,hv,Tk,Hv)
        gv = snp.matmul(
            a.reshape(B,hv,hq//hv,Tq,Tk).transpose(0,1,4,2,3).reshape(B,hv,Tk,hq//hv*Tq),
            gy.reshape(B,hv,hq//hv*Tq,Hv))
 
        ga = self.DO.backward(ga)

        if self.regularizer is not None:          
            ga2 = self.regularizer.backward()
            ga += ga2                 # 勾配加算率はregularizer側に設定
        
        ga = self.softmax.backward(ga)
        if self.mask is not None:
            ga *= self.mask
        if self.causality:    
            ga *= self.tril
        if self.scale:
            ga *= np.array(Hk ** -0.5, dtype=ga.dtype)

        gq = snp.matmul(ga.reshape(B,hk,hq//hk,Tq,Tk), k[:,:,None,:,:])
        gq = gq.reshape(B,hq,Tq,Hq)   # (B,hk,hq//hk,Tq,Hq) -> (B,hq,Tq,Hq)

        # グループ方向(hq//hk)の加算を行列積の縮約に含める
        #   (B,hk,Tk,g*Tq)@(B,hk,g*Tq,Hq) -> (B,hk,Tk,Hq)
        gk = snp.matmul(
            ga.reshape(B,hk,hq//hk,Tq,Tk).transpose(0,1,4,2,3).reshape(B,hk,Tk,hq//hk*Tq),
            q.reshape(B,hk,hq//hk*Tq,Hq))

        if self.rope is not None:
            gq, gk = self.rope.backward(gq, gk)  

        gq = gq.transpose(0,2,1,3).reshape(B,Tq,Cq) # (B,hq,Tq,Hq)->(B,Tq,Cq)
        gk = gk.transpose(0,2,1,3).reshape(B,Tk,Ck) # (B,hk,Tk,Hk)->(B,Tk,Ck)
        gv = gv.transpose(0,2,1,3).reshape(B,Tv,Cv) # (B,hv,Tv,Hv)->(B,Tv,Cv)
        return gq, gk, gv


class StatelessSoftmax:
    """ QueryChunkAttention用のSoftmax """
    def __init__(self, temperature=1.0, **kwargs):
        self.temperature = temperature

    def forward(self, x):
        x = x / self.temperature   # 温度スケーリング
        max_x = np.max(x, axis=-1, keepdims=True) 
        exp_a = np.exp(x - max_x)  # オーバーフロー対策
        sum_exp_a = snp.sum(exp_a, axis=-1, keepdims=True)  
        y = exp_a / (sum_exp_a + 1e-7)
        return y

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def backward(self, gy, y): 
        gx = y * gy
        sumdx = snp.sum(gx, axis=-1, keepdims=True)
        gx -= y * sumdx
        gx = gx / self.temperature # 温度スケーリング
        return gx


class QueryChunkAttentionUnit(AttentionUnit):
    def __init__(self, head=1, **kwargs):
        self.chunk_size = kwargs.pop('chunk_size', None)
        temperature = kwargs.pop('temperature', 1.0)
        super().__init__(head, **kwargs)
        self.softmax = StatelessSoftmax(temperature=temperature)
        self.DO = StatelessDropout()
       
    def __forward__(self, q, k, v, *, mask=None, dropout=0.0):
        self.q = q                                   # query
        self.k = k                                   # key
        self.v = v                                   # value
        B,Tq,Cq = q.shape
        B,Tk,Ck = k.shape
        B,Tv,Cv = v.shape
        hq, hk, hv = self.head
        Hq = Cq // hq
        Hk = Ck // hk
        Hv = Cv // hv
        q = q.reshape(B,Tq,hq,Hq).transpose(0,2,1,3) # (B,Tq,Cq)->(B,hq,Tq,Hq)
        k = k.reshape(B,Tk,hk,Hk).transpose(0,2,1,3) # (B,Tk,Ck)->(B,hk,Tk,Hk)
        v = v.reshape(B,Tv,hv,Hv).transpose(0,2,1,3) # (B,Tv,Cv)->(B,hv,Tv,Hv)

        if self.rope is not None:
            q, k = self.rope(q, k)  

        if Hq != Hk:
            raise ValueError(f"q and k head_dim must be equal: Hq={Hq}, Hk={Hk}")

        if hq % hk != 0:
            raise ValueError(
                f"q_head must be divisible by k_head: q_head={hq}, k_head={hk}")
        if hq % hv != 0:
            raise ValueError(
                f"q_head must be divisible by v_head: q_head={hq}, v_head={hv}")

        q = q.reshape(B,hk,hq//hk,Tq,Hq)
        kt = k.transpose(0,1,3,2)[:,:,None,:,:]

        chunk_size = Tq if self.chunk_size is None else self.chunk_size
        invrootH = np.array(Hk ** -0.5, dtype=Config.dtype)

        if self.causality:
            if Tq != Tk:
                raise Exception(
                    f"causality cannot be applied" + self.__class__.__name__)
            self.tril = np.empty((1,1,Tq,Tk), dtype=Config.dtype)
    
        if mask is None:   # 無効トークンの処理
            self.mask = None
        elif mask.shape == (B, Tk):
            self.mask = mask.astype(q.dtype)[:,None,None,:] # (B,1,1,Tk)
        else:
            raise ValueError(
                f"Mask shape {mask.shape} must be ({B}, {Tk})"
                + self.__class__.__name__)

        dropout_mx = 1
        if dropout > 0:
            dropout_mx = np.random.rand(B,hq,Tq,Tk) > dropout
        self.dropout = dropout
        self.dropout_mx = dropout_mx

        if self.regularizer is not None:
            a_soft = np.empty((B,hq,Tq,Tk), dtype=Config.dtype)
        else:
            a_soft = None

        y = np.empty((B,hq,Tq,Hv), dtype=Config.dtype)
        for i in range(0, Tq, chunk_size):
            j = min(i + chunk_size, Tq)
            a_chunk = snp.matmul(q[:,:,:,i:j,:], kt)
            a_chunk = a_chunk.reshape(B,hq,j-i,Tk)

            if self.scale:
                a_chunk *= invrootH

            if self.causality:
                tril_chunk = np.tri(j-i, Tk, k=i, dtype=Config.dtype)[None, None, :, :]
                self.tril[:,:,i:j,:] = tril_chunk
                a_chunk += (tril_chunk - 1.0) * Config.inf
                
            if self.mask is not None:
                a_chunk += (self.mask - 1.0) * Config.inf # 無効個所は-inf

            a_soft_chunk = self.softmax(a_chunk)

            if self.regularizer is not None:
                a_soft[:,:,i:j,:] = a_soft_chunk

            if dropout > 0:
                dropout_mx_chunk = dropout_mx[:,:,i:j,:]
            else:
                dropout_mx_chunk = 1
                
            a_drop_chunk = self.DO.forward(
                a_soft_chunk, dropout_mx_chunk, dropout=dropout)

            y_chunk = snp.matmul(a_drop_chunk.reshape(B,hv,hq//hv,j-i,Tk), v[:,:,None,:,:])
            y_chunk = y_chunk.reshape(B,hq,j-i,Hv)

            y[:,:,i:j,:] = y_chunk

        if self.regularizer is not None:
            self.loss = self.regularizer.forward(a_soft)

        self.iter += 1
        y = y.transpose(0,2,1,3).reshape(B,Tq,hq*Hv)  # (B,hq,Tq,Hv)->(B,Tq,hq*Hv)
        #self.y = y
        return y                                      # (B,Tq,hq*Hv)

    def __backward__(self, gy):
        q = self.q
        k = self.k
        v = self.v
        #a = self.a
        B,Tq,Cq = q.shape
        B,Tk,Ck = k.shape
        B,Tv,Cv = v.shape
        hq, hk, hv = self.head
        Hq = Cq // hq
        Hk = Ck // hk
        Hv = Cv // hv
        q = q.reshape(B,Tq,hq,Hq).transpose(0,2,1,3) # (B,Tq,Cq)->(B,hq,Tq,Hq)
        k = k.reshape(B,Tk,hk,Hk).transpose(0,2,1,3) # (B,Tk,Ck)->(B,hk,Tk,Hk)
        v = v.reshape(B,Tv,hv,Hv).transpose(0,2,1,3) # (B,Tv,Cv)->(B,hv,Tv,Hv)

        if self.rope is not None:
            q, k = self.rope(q, k)  

        kt = k.transpose(0,1,3,2)[:,:,None,:,:]

        dropout = self.dropout
        dropout_mx = self.dropout_mx

        gy = gy.reshape(B,Tq,hq,Hv).transpose(0,2,1,3) # (B,Tq,hq*Hv)->(B,hq,Tq,Hv)

        invrootH = np.array(Hk ** -0.5, dtype=Config.dtype)
        chunk_size = Tq if self.chunk_size is None else self.chunk_size

        gq = np.empty_like(q)
        gk = np.zeros_like(k)
        gv = np.zeros_like(v)

        ga2 = None
        if self.regularizer is not None:
            ga2 = self.regularizer.backward()
            if getattr(ga2, 'ndim', 0) == 0:
                ga2 = None

        for i in range(0, Tq, chunk_size):
            j = min(i + chunk_size, Tq)
            # -- 順伝播再計算 --
            a_chunk = snp.matmul(q.reshape(B,hk,hq//hk,Tq,Hq)[:,:,:,i:j,:], kt)
            a_chunk = a_chunk.reshape(B,hq,j-i,Tk)

            if self.scale:
                a_chunk *= invrootH

            if self.causality:
                tril_chunk = np.tri(j-i, Tk, k=i, dtype=Config.dtype)[None,None,:,:]
                self.tril[:,:,i:j,:] = tril_chunk
                a_chunk += (tril_chunk - 1.0) * Config.inf
                
            if self.mask is not None:
                a_chunk += (self.mask - 1.0) * Config.inf # 無効個所は-inf

            a_soft_chunk = self.softmax(a_chunk)

            if dropout > 0:
                dropout_mx_chunk = dropout_mx[:,:,i:j,:]
            else:
                dropout_mx_chunk = 1

            a_drop_chunk = self.DO.forward(a_soft_chunk, dropout_mx_chunk, dropout=dropout)

            # -- 以下、逆伝播の計算 --
            gy_chunk = gy[:,:,i:j,:]
            ga_chunk = snp.matmul(
                gy_chunk.reshape(B,hv,hq//hv,j-i,Hv), v.transpose(0,1,3,2)[:,:,None,:,:])
            ga_chunk = ga_chunk.reshape(B,hq,j-i,Tk)

            gv_chunk = snp.matmul(
                a_drop_chunk.reshape(B,hv,hq//hv,j-i,Tk).transpose(0,1,2,4,3),
                gy_chunk.reshape(B,hv,hq//hv,j-i,Hv))
                                 
            gv += np.sum(gv_chunk, axis=2)

            ga_chunk = self.DO.backward(ga_chunk, dropout_mx_chunk, dropout=dropout)

            if ga2 is not None:
                ga_chunk += ga2[:,:,i:j,:]

            ga_chunk = self.softmax.backward(ga_chunk, a_soft_chunk)

            # maskとcausalityとscale
            if self.mask is not None:
                ga_chunk *= self.mask
            if self.causality:
                ga_chunk *= tril_chunk
            if self.scale:
                ga_chunk *= invrootH

            gq_chunk = snp.matmul(ga_chunk.reshape(B,hk,hq//hk,j-i,Tk), k[:,:,None,:,:])
            gq[:,:,i:j,:] = gq_chunk.reshape(B,hq,j-i,Hq)
            
            gk_chunk = snp.matmul(ga_chunk.reshape(B,hk,hq//hk,j-i,Tk).transpose(0,1,2,4,3),
                                 q.reshape(B,hk,hq//hk,Tq,Hq)[:,:,:,i:j,:])
            gk += np.sum(gk_chunk, axis=2) 

        if self.rope is not None:
            gq, gk = self.rope.backward(gq, gk)

        gq = gq.transpose(0,2,1,3).reshape(B,Tq,Cq) # (B,hq,Tq,Hq)->(B,Tq,Cq)
        gk = gk.transpose(0,2,1,3).reshape(B,Tk,Ck) # (B,hk,Tk,Hk)->(B,Tk,Ck)
        gv = gv.transpose(0,2,1,3).reshape(B,Tv,Cv) # (B,hv,Tv,Hv)->(B,Tv,Cv)
        return gq, gk, gv

#### 時系列データをまとめて処理する Attention層 ############################
# q:入力 query、x:入力 keyとvalue、y:出力、w:attention_weight
# query に一致する key を探して、その key に対応する value を出力する
# Seq2seq(sequence to sequence、時系列データ変換器)での使い方として;
#  key にエンコーダ出力を入れ、
#  デコーダの隠れ層から受け取った query に対応する
#  エンコーダ出力から選んだコンテクストベクトルを得る
# 
class SimpleAttentionLayer(Function):
    """
    入力はx:key&valueとq:query　　　　
    このLayerはAttentionUnitをそのまま使う　

    """
    def __init__(self, *configuration, **kwargs):
        super().__init__()
        self.unit = AttentionUnit()
        print('Initialize', self.__class__.__name__)
        
    def __forward__(self, x, q, *, dropout=0.0):
        return self.unit.forward(q, x, x, dropout=dropout) # keyとvalueは同じものを与える

    def __backward__(self, grad_y):       # grad_yは下流から受け取る勾配
        grad_q, grad_k, grad_v = self.unit.backward(grad_y)
        grad_x = grad_k + grad_v          # keyとvalueに同じものを与えたことに対応
        return grad_x, grad_q


class SelfAttention(Function):
    """ multiple heads of self_attention in parallel """
    def __init__(self, emb_dim=None, head_dim=None, n_head=1,
                 #scale=True, temperature=1.0, entropy_decay=True,
                 **kwargs):
        super().__init__()

        if isinstance(n_head, int):
            n_head = n_head, n_head, n_head
        elif isinstance(n_head, (tuple, list)) and len(n_head) == 2:
            n_head = n_head[0], n_head[1], n_head[1] 
        else:
            raise ValueError("n_head must be int or (q_head, kv_head)")
        hq, hk, hv = n_head
        if hq % hk != 0:
            raise ValueError(
                f"q_head must be divisible by kv_head: q_head={hq}, kv_head={hk}")

        # head_dimはfix_configurationでq_headを基準に確定する
        self.config = emb_dim, head_dim, n_head
        print('Initialize', self.__class__.__name__, self.config, kwargs)

        optimize = kwargs.pop('optimize',   'Adam')
        chunk_size = kwargs.pop('chunk_size', None)

        # linear_iとlinear_oのconfigはfix_configurationで設定
        self.linear_i = LinearLayer(matmul=True, bias=False,
                                    #scale=True,
                                    optimize=optimize,
                                    #spctrnorm=1,
                                    **kwargs)

        causality = kwargs.pop('causality', False)

        if chunk_size is None:
            self.attention = AttentionUnit(
                head=n_head, causality=causality, **kwargs)
        else:
            self.attention = QueryChunkAttentionUnit(
                head=n_head, causality=causality,
                chunk_size=chunk_size, **kwargs)
                     #scale=scale, temperature=temperature,
                     #entropy_decay=entropy_decay)

        self.linear_o = LinearLayer(matmul=True, bias=True,
                                    #scale=True,
                                    optimize=optimize,
                                    #spctrnorm=1,
                                    **kwargs)

        self.DO = Dropout()
        self.step = 0


    def fix_configuration(self, shape):
        emb_dim, head_dim, n_head = self.config
        hq, hk, hv = n_head

        if emb_dim is None:
            emb_dim = shape[-1]
        elif emb_dim != shape[-1]:
            raise Exception(
                'Data shape mismatch with configuration.',
                self.__class__.__name__
            )

        if head_dim is None:
            if emb_dim % hq != 0:
                raise ValueError(
                    f"emb_dim must be divisible by q_head: "
                    f"emb_dim={emb_dim}, q_head={hq}"
                )
            head_dim = emb_dim // hq

        elif hq * head_dim != emb_dim:
            raise ValueError(
                f"head_dim mismatch: "
                f"emb_dim={emb_dim}, q_head={hq}, head_dim={head_dim}"
            )

        self.config = emb_dim, head_dim, n_head
        self.linear_i.config = emb_dim, (hq + hk + hv) * head_dim
        self.linear_o.config = emb_dim, emb_dim

        print(self.__class__.__name__, 'fix_configuration', shape, self.config,
              'qkv =', tuple(h * head_dim for h in n_head))

    def __forward__(self, x, *, mask=None, dropout=0.0):
        if None in (*self.config, *self.linear_i.config, *self.linear_o.config):
            self.fix_configuration(x.shape)
        emb_dim, head_dim, n_head = self.config
        hq, hk, hv = n_head
        z = self.linear_i.forward(x)
        # zをhq*head_dimと(hq+hk)*head_dimの2か所で切って3分割する
        query, key, value = np.split(z, [hq*head_dim, (hq+hk)*head_dim], axis=-1) 
        y = self.attention.forward(query, key, value, mask=mask, dropout=dropout)
        y = self.linear_o.forward(y)
        y = self.DO.forward(y, dropout=dropout)
        self.y = y
        return y

    def __backward__(self, gy):
        gx = self.DO.backward(gy)
        gx = self.linear_o.backward(gx)
        gq, gk, gv = self.attention.backward(gx)
        gz = np.concatenate([gq, gk, gv], axis=-1)
        gx = self.linear_i.backward(gz)
        return gx

    def update(self, eta=0.001, **kwargs):
        self.linear_i.update(eta=eta, **kwargs)
        self.linear_o.update(eta=eta, **kwargs)

    def entropy(self):
        return self.attention.entropy


class MultiHeadSelfAttention(SelfAttention):
    """ multiple heads of self_attention in parallel """
    def __init__(self, emb_dim=None, head_dim=None, n_head=1, **kwargs):
        print(self.__class__.__name__, emb_dim, head_dim, n_head, kwargs)
        super().__init__(emb_dim, head_dim, n_head, **kwargs)
        
class SingleHeadSelfAttention(SelfAttention):
    """ single head of self_attention """
    def __init__(self, emb_dim=None, head_dim=None, **kwargs):
        print(self.__class__.__name__, emb_dim, head_dim, kwargs)
        super().__init__(emb_dim, head_dim, 1, **kwargs)
        
class MultiHeadSelfAttention2(Function):
    """ 先にhead分割し、各SingleAttentionに配る """
    def __init__(self, emb_dim=None, head_dim=None, n_head=1,
                 scale=True, temperature=1.0, optimize='Adam', **kwargs):
        super().__init__()
        if emb_dim is not None and head_dim is None:
            head_dim = emb_dim // n_head
        else:
            head_dim, head_dim = None, None
        self.config = emb_dim, head_dim, n_head
        print('Initialize', self.__class__.__name__, self.config)
        # linear_iとlinear_oのconfigはfix_configurationで設定
        self.linear_q = LinearLayer(matmul=True, bias=False, optimize=optimize, **kwargs)
        self.linear_k = LinearLayer(matmul=True, bias=False, optimize=optimize, **kwargs)
        self.linear_v = LinearLayer(matmul=True, bias=False, optimize=optimize, **kwargs)
        self.attention = [
            AttentionUnit(causality='tri', scale=scale, temperature=temperature)
            for _ in range(n_head)]
        self.linear_o = LinearLayer(matmul=True, bias=True, optimize=optimize, **kwargs)
        self.DO = Dropout()
        
    def fix_configuration(self, shape):
        emb_dim, head_dim, n_head = self.config
        if emb_dim is None:
            emb_dim = shape[-1]
        elif emb_dim != shape[-1]: # 予め与えられたemb_dimがデータと合わない
            raise Exception('Data shape mismatch with configuration.',
                                                self.__class__.__name__)
        if head_dim is None:
            head_dim = emb_dim // n_head
        assert emb_dim % n_head==0, "embedding dimension must be divisible by n_head"
        assert emb_dim // n_head == head_dim, "head_dim must be emb_dim // n_head"
        self.config = emb_dim, head_dim, n_head
        self.linear_q.config = emb_dim, emb_dim 
        self.linear_k.config = emb_dim, emb_dim 
        self.linear_v.config = emb_dim, emb_dim 
        self.linear_o.config = emb_dim, emb_dim
        print(self.__class__.__name__, 'fix_configuration', shape, self.config)

    def __forward__(self, x, *, dropout=0.0):
        if None in (*self.config,
                    *self.linear_q.config,
                    *self.linear_k.config,
                    *self.linear_v.config,
                    *self.linear_o.config):
            #print(self.__class__.__name__, 'input.shape', x.shape)
            self.fix_configuration(x.shape)
        emb_dim, head_dim, n_head = self.config    
        query = self.linear_q.forward(x)
        key   = self.linear_k.forward(x)
        value = self.linear_v.forward(x)
        # head分割
        query = np.split(query, n_head, axis=-1)
        key   = np.split(key,   n_head, axis=-1)
        value = np.split(value, n_head, axis=-1)
        # 各ヘッド独立　　　　
        y = [a.forward(q, k, v, dropout=dropout) for a, q, k, v \
                       in zip(self.attention, query, key, value)] 
        # 各ヘッド出力を併合して出力層へ
        y = np.concatenate(y, axis=-1)
        y = self.linear_o.forward(y)
        y = self.DO.forward(y, dropout=dropout)
        self.y = y
        return y
        
    def __backward__(self, gy):
        emb_dim, head_dim, n_head = self.config
        gz = self.DO.backward(gy)
        gz = self.linear_o.backward(gz)
        gz = np.split(gz, n_head, axis=-1)
        
        gqkv = [a.backward(gzi) for a, gzi in zip(self.attention, gz)]

        gq = np.concatenate([g[0] for g in gqkv], axis=-1)
        gk = np.concatenate([g[1] for g in gqkv], axis=-1)
        gv = np.concatenate([g[2] for g in gqkv], axis=-1)
               
        gq = self.linear_q.backward(gq)
        gk = self.linear_k.backward(gk)
        gv = self.linear_v.backward(gv)
        gx = gq + gk + gv
        return gx

    def update(self, eta=0.001, **kwargs):
        self.linear_q.update(eta=eta, **kwargs)
        self.linear_k.update(eta=eta, **kwargs)
        self.linear_v.update(eta=eta, **kwargs)
        self.linear_o.update(eta=eta, **kwargs)

    def entropy(self):
        return [sa.entropy for sa in self.attention]

    def entropy_std_and_range(self):
        e_head = np.array([sa.attention.entropy for sa in self.attention])
        #print(e_head.shape) 
        e_std = np.std(e_head)
        e_rng = np.max(e_head) - np.min(e_head)
        #print(f'[Entropy] std={e_std:.4f}, range={e_rng:.4f}')
        return e_std, e_rng
        
class SpatialSelfAttention(Function):
    """ 画像データを処理するAttention機構 """
    def __init__(self, n_head=1, scale=1.0, **kwargs):
        super().__init__()
        self.attention = SelfAttention(n_head=n_head, **kwargs)
        self.scale = scale

    def __forward__(self, x):
        B, C, H, W = x.shape
        y = snp.reshape(x, (B, C, H*W))
        y = snp.transpose(y, (0, 2, 1))
        y = self.attention(y)
        y = snp.transpose(y, (0, 2, 1))
        y = snp.reshape(y, (B, C, H, W))
        return x + self.scale * y

    def __backward__(self, gy):
        B, C, H, W = gy.shape
        gx = self.scale * gy           
        gx = snp.reshape(gx, (B, C, H*W))
        gx = snp.transpose(gx, (0, 2, 1))  
        gx = self.attention.backward(gx)
        gx = snp.transpose(gx, (0, 2, 1))
        gx = snp.reshape(gx, (B, C, H, W))
        gx += gy
        return gx

    def update(self, eta=0.001, **kwargs):
        self.attention.update(eta=eta, **kwargs)

class ParametersForContextualSelfAttention:
    """
    時系列入力xに対してコンテキストベクトルyを返す
    即ち、入力の時系列に並ぶものの重要なことを抽出して文脈とする
    この操作では、keyとvalueは必要だがqueryは何でも良い．　　　
    なぜならば、出力y(コンテクスト)は入力xに応じて決まれば良いのだから．
    そこでqueryは入力xによらないパラメタとして用意する．
         
    """
    def __init__(self, layer, **kwargs):
        print(self.__class__.__name__)
        self.layer   = layer 
        optimize     = kwargs.pop('optimize',   'SGD') 
        self.width   = kwargs.pop('width',       None)
        self.q_shape = kwargs.pop('q_shape', (1,1,-1))
        self.w = None; self.b = None; self.q = None
        self.optimizer_w = cf.eval_in_module(optimize, Optimizers, **kwargs)
        self.optimizer_b = cf.eval_in_module(optimize, Optimizers, bias=True, **kwargs)
        self.optimizer_q = cf.eval_in_module(optimize, Optimizers, **kwargs)
        self.debug_mode = kwargs.pop('debug_mode', False)
       
    def __call__(self):
        if self.w is None:
            self.init_parameter()
        return self.w, self.b, self.q

    def init_parameter(self):#, m, n):
        m, n = self.layer.get_parameter_size() 
        if m is None or n is None:
            raise Exception('Configuration is not fixed.', self.__class__.__name__)
        self.w = init_weight((m, n),
                             width=self.width,
                             debug_mode=self.debug_mode)
        self.b = np.zeros(n, dtype=Config.dtype)
        self.q = init_weight((1,n),
                             width=np.sqrt(1/m), # 通常とは異なるため指定が必要
                             debug_mode=self.debug_mode)
        self.q = self.q.reshape(*self.q_shape)
        #(width * np.random.randn(1,1,n)).astype(Config.dtype) 

    def update(self, eta=0.001, **kwargs):
        self.optimizer_w.update(self.w, self.grad_w, eta, **kwargs) 
        self.optimizer_b.update(self.b, self.grad_b, eta, **kwargs) 
        self.optimizer_q.update(self.q, self.grad_q, eta, **kwargs)

    def set_gradient(self, *grads): 
        self.grad_w = grads[0]
        self.grad_b = grads[1]
        self.grad_q = grads[2]
        
        
class ContextualSelfAttention(Function):
    """
    時系列入力xに対してコンテキストベクトルyを返す
    即ち、入力の時系列に並ぶものの重要なことを抽出して文脈とする
    この操作では、keyとvalueは必要だがqueryは何でも良い．　　　
    なぜならば、出力y(コンテクスト)は入力xに応じて決まれば良いのだから．
    そこでqueryは入力xによらないパラメタとして用意する．
         
    """
    def __init__(self, *configuration, n_head=1,
                 linear_v=False, linear_k=False, q_shape=(1,1,-1), **kwargs):
        super().__init__()
        if len(configuration) == 2:
            m, n = configuration
        if len(configuration) == 1:
            m = None; n, = configuration
        self.config = m, n

        # linear_iとlinear_oのconfigはfix_configurationで設定
        self.linear_v = LinearLayer(matmul=True, bias=False,**kwargs) \
            if linear_v else None
        self.linear_k = LinearLayer(matmul=True, bias=False,**kwargs) \
            if linear_k else None
        # qもfix_configurationで設定
        self.q, self.grad_q = None, None
        self.q_shape = q_shape
        optimize = kwargs.pop('optimize', 'SGD')
        self.optimizer_q = cf.eval_in_module(optimize, Optimizers, **kwargs)
        # Attentionとdebug_mode
        self.attention = AttentionUnit(head=n_head)
        self.debug_mode = kwargs.pop('debug_mode',  False)
       
    def fix_configuration(self, shape):
        self.config = shape[-1], self.config[1]
        print('config =', self.config)
        m, n = self.config

        if self.linear_v is not None:
            self.linear_v.config = m, n 
        if self.linear_k is not None:
            self.linear_k.config = m, n 

        self.q = init_weight((1, n),
                             width=np.sqrt(1/m), # 通常とは異なるため指定が必要
                             debug_mode=self.debug_mode)
        self.q = self.q.reshape(*self.q_shape)
        print(self.__class__.__name__, 'fix_configuration', shape, self.config)

    def update(self, eta=0.001, **kwargs):
        if self.linear_v is not None:
            self.linear_v.update(eta=eta, **kwargs)
        if self.linear_k is not None:
            self.linear_k.update(eta=eta, **kwargs)
        self.optimizer_q.update(self.q, self.grad_q, eta, **kwargs)

    def get_parameter_size(self):
        m, n = self.config
        return m, n

    def __forward__(self, x, dropout=0.0):
        self.x = x
        if None in self.config or self.q is None:
            #print(self.__class__.__name__, '.input.shape', x.shape)
            self.fix_configuration(x.shape)
        m, n = self.config
        B, _, _ = x.shape
        
        v = self.linear_v.forward(x) if self.linear_v is not None else x
        k = self.linear_k.forward(x) if self.linear_k is not None else x
        q = np.broadcast_to(self.q, (B, 1, n))
       
        y = self.attention.forward(q, k, v, dropout=dropout)
        y = y.reshape(B, n)
        return y

    def __backward__(self, gy):
        x = self.x
        m, n = self.config
        B, _, _ = x.shape

        gy = gy.reshape(B, 1, n)

        gq, gk, gv = self.attention.backward(gy)

        self.grad_q = np.sum(gq, axis=0, keepdims=True)
                                                # (B,1,H)->(1,1,H) forwardでのBCに対応

        gxk = self.linear_k.backward(gk) if self.linear_k is not None else gk
        gxv = self.linear_v.backward(gv) if self.linear_v is not None else gv
        gx = gxk + gxv

        return gx
    
class ContextualSelfAttention_bkup(Function):
    """
    時系列入力xに対してコンテキストベクトルyを返す
    即ち、入力の時系列に並ぶものの重要なことを抽出して文脈とする
    この操作では、keyとvalueは必要だがqueryは何でも良い．　　　
    なぜならば、出力y(コンテクスト)は入力xに応じて決まれば良いのだから．
    そこでqueryは入力xによらないパラメタとして用意する．
         
    """
    def __init__(self, *configuration, **kwargs):
        super().__init__()
        if len(configuration) == 2:
            m, n = configuration
        if len(configuration) == 1:
            m = None; n, = configuration
        self.config = m, n
        self.attention = AttentionUnit(scale=True)
        self.dot_linear = F.MatMulLinear()
        self.parameters = ParametersForContextualSelfAttention(self, **kwargs)
       
    def fix_configuration(self, shape):
        self.config = shape[-1], self.config[1]
        print(self.__class__.__name__, 'fix_configuration', shape, self.config)

    def update(self, eta=0.001, **kwargs):
        self.parameters.update(eta=eta, **kwargs)

    def get_parameter_size(self):
        m, n = self.config
        return m, n

    def __forward__(self, x):
        if None in self.config:
            #print(self.__class__.__name__, '.input.shape', x.shape)
            self.fix_configuration(x.shape)

        w, b, q = self.parameters()    
        m, n = self.config # m, n = H ベクトルサイズ
        B, T, _ = x.shape
        self.x = x
        # 入力xからkeyを生成 keyの形状 (B, T, H)
        self.k = self.dot_linear.forward(x, w, b) # (B,T,H)
        y = self.attention.forward(np.broadcast_to(q, (B, 1, n)), self.k, x)
        y = y.reshape(B, n)
        return y

    def __backward__(self, gy):
        x = self.x
        m, n = self.config
        B, _, _ = x.shape

        gy = gy.reshape(B, 1, n)
        gq, gk, gv = self.attention.backward(gy) 
        grad_q = np.sum(gq, axis=0, keepdims=True)
                                             # (B,1,H)->(1,1,H) forwardでのBCに対応

        # keyの逆伝播
        gx2, grad_w, grad_b = self.dot_linear.backward(gk)
        gx += gx2

        self.parameters.set_gradient(grad_w, grad_b, grad_q)
        return gx
    

class ContextualSelfAttentionZ1(ContextualSelfAttention):
    """
    ContextualSelfAttentionと同一の処理
    順伝播、逆伝播を展開している
         
    """
    def __init__(self, *configuration, **kwargs):
        super().__init__(*configuration, **kwargs)
        self.softmax = Activators.Softmax()

    def __forward__(self, x):
        if None in self.config:
            #print(self.__class__.__name__, '.input.shape', x.shape)
            self.fix_configuration(x.shape)

        w, b, q = self.parameters()     
        m, n = self.config # m, n = H ベクトルサイズ
        B, T, _ = x.shape
        self.x = x
        # 入力xからkeyを生成 keyの形状 (B, T, H)
        #print('### x =', x.shape, 'w =', self.w.shape)   
        self.k = snp.matmul(x, w) + b      # (B,T,H)
        #print('### k =', self.k.shape, 'q =', self.q.shape)   

        # s:scoreとa:attention_weightを算出 sとaの形状:(B,T)
        s = snp.matmul(self.q, self.k.transpose(0,2,1)) # (1,1,H)*(B,T,H)->(B,1,T)
        #print('### s =', s.shape)
        s *= m ** -0.5                              # スケーリング
        self.a = self.softmax.forward(s)
        #print('### a =', self.a.shape)
        # 入力(B, T, H)をバッチ軸(B)はそのままにして、時系列軸(T)方向に重み付け
        c = snp.matmul(self.a, x) # (B,1,T)*(B,T,H)->(B,1,H)
        # コンテキストベクトル y.shape:(B, H) 
        y = c.reshape(B, n)                         # (B,1,H) -> (B,H)
        #print('### y =', y.shape)
        #input()
        return y

    def __backward__(self, gy):
        x = self.x
        a = self.a
        m, n = self.config
        B, T, _ = x.shape
        
        w, b, q = self.parameters()     
        # 重み付け和の逆伝播
        gc = gy.reshape(B, 1, n) # (B,H)->(B,1,H)
 
        ga = snp.matmul(gc, x.transpose(0,2,1))
        gx = snp.matmul(a.transpose(0,2,1), gc)
        #print('### ga =', ga.shape, 'gx =', gx.shape)
        # attention_weightの逆伝播
        gs = self.softmax.backward(ga)
        gs *= m ** -0.5

        #print('### gs =', gs.shape, 'q =', self.q.shape)
        gk = snp.matmul(gs.transpose(0,2,1), q)     # (B,1,T)*(1,1,H)->(B,T,H)
        #print('### gk =', gk.shape)
        #print('### k =', self.k.shape, 'gs =', gs.shape) 
        gq = snp.matmul(gs, self.k)                      # (B,1,T)*(B,T,H)->(B,1,H)
        #print('### gq =', gq.shape)
        grad_q = np.sum(gq, axis=0, keepdims=True) # (B,1,H) -> (1,1,H)
        #print('### grad_q =', self.grad_q.shape)
        # keyの逆伝播
        #self.grad_w = snp.matmul(x.reshape(-1, m).T, gk.reshape(-1, n))
        grad_w = snp.matmul(x.transpose(0,2,1), gk)  # (B,H,T) * (B,T,H) -> (B,H,H)
        grad_w = np.sum(grad_w, axis=0)        # (B,H,H) -> (H,H)
        grad_b = np.sum(gk, axis=(0,1))        # (B,T,H) -> (H,)
        #print('### grad_w', self.grad_w.shape)
        gx += snp.matmul(gk, self.w.T)
        #input()
        self.parameters.set_gradient(grad_w, grad_b, grad_q)
        return gx
  
class ContextualSelfAttentionZ2(ContextualSelfAttention):
    """
    ContextualSelfAttentionと同一の処理
    クエリ―の形状が違う
         
    """
    def __init__(self, *configuration, **kwargs):
        super().__init__(*configuration, **kwargs)
        self.softmax = Activators.Softmax()

    def init_parameter(self):#, m, n):
        m, n = self.get_parameter_size() 
        if m is None or n is None:
            raise Exception('Configuration is not fixed.', self.__class__.__name__)
        if self.width is not None:
            width = self.width
        else:    
            width = np.sqrt(1/m)  # Xavierの初期値

        self.w = (width * np.random.randn(m, n)).astype(Config.dtype) 
        self.b = np.zeros(n, dtype=Config.dtype)
        self.q = (width * np.random.randn(n)).astype(Config.dtype) 
        if self.debug_mode:
            self.w[...] = 1; self.q[...] = 1 # for debug
        print(self.__class__.__name__, 'init_parameters', m, n)

    def __forward__(self, x):
        if None in self.config:
            #print(self.__class__.__name__, '.input.shape', x.shape)
            self.fix_configuration(x.shape)
        if self.w is None or self.b is None or self.q is None:
            self.init_parameter()
        m, n = self.config # m, n = H ベクトルサイズ
        B, T, _ = x.shape
        self.x = x
        # 入力xからkeyを生成 keyの形状 (B, T, H)
        #print('### x =', x.shape, 'w =', self.w.shape)   
        self.k = snp.matmul(x, self.w) + self.b      # (B,T,H)
        #print('### k =', self.k.shape, 'q =', self.q.shape)   
        # s:scoreとa:attention_weightを算出 sとaの形状:(B,T)
        q = self.q.reshape(1,n,1)                   # (H,) -> (1,H,1) 
        s = snp.matmul(self.k, q)                    # (B,T,H) * (1,H,1) -> (B,T,1)
        s = s.reshape(B, T)
        #print('### s =', s.shape)
        s *= n ** -0.5                              # スケーリング
        self.a = self.softmax.forward(s)
        #print('### a =', self.a.shape)
        # 入力(B, T, H)をバッチ軸(B)はそのままにして、時系列軸(T)方向に重み付け
        a = self.a.reshape(B, T, 1) #self.a[..., np.newaxis] (B,T) -> (B,T,1)
        c = x * a                                   # (B,T,H)*(B,T,1) -> (B,T,H)
        # コンテキストベクトル y.shape:(B, H) 
        y = np.sum(c, axis=1)                       # (B,T,H) -> (B,H)
        #print('### y =', y.shape)
        return y

    def __backward__(self, gy):
        x = self.x
        a = self.a
        m, n = self.config
        B, T, _ = x.shape
        
        # 重み付け和の逆伝播
        gc = gy.reshape(B, 1, m).repeat(T, axis=1)
        ga = np.sum(gc * x, axis=-1)#, keepdims=True)
        gx = gc * self.a[..., np.newaxis]
        #print('### ga =', ga.shape, 'gx =', gx.shape)
        # attention_weightの逆伝播
        gs = self.softmax.backward(ga)
        gs *= n ** -0.5
        gs = gs[..., np.newaxis]
        #print('### gs =', gs.shape, 'q =', self.q.shape)
        qT = self.q.reshape(1,1,n)
        gk = snp.matmul(gs, qT)     # (B,T,1) * (1,1,H) -> (B,T,H)
        #print('### gk =', gk.shape)
        #print('### k =', self.k.shape, 'gs =', gs.shape) 
        gq = snp.matmul(self.k.transpose(0,2,1), gs) # (B,T,H) * (B,T,1) -> (B,H,1)
        #print('### gq =', gq.shape)
        gq = np.sum(gq, axis=0)                     # (B,H,1) -> (H,1)
        self.grad_q = gq.reshape(n) 
        #print('### grad_q =', self.grad_q.shape)
        # keyの逆伝播
        grad_w = snp.matmul(x.transpose(0,2,1), gk)  # (B,H,T) * (B,T,H) -> (B,H,H)
        self.grad_w = np.sum(grad_w, axis=0)        # (B,H,H) -> (H,H)
        self.grad_b = np.sum(gk, axis=(0,1))        # (B,T,H) -> (H,)
        #print('### grad_w', self.grad_w.shape)
        gx += snp.matmul(gk, self.w.T)
        #input()
        return gx


class ContextualSelfAttentionZ4(ContextualSelfAttention):
    """
    ContextualSelfAttentionと同一の処理
    クエリ―の形状が違うが、
    attention weight生成の際にsoftmaxの軸が末尾となっており、
    本来の機能とならない問題あり     
         
    """
    def __init__(self, *configuration, **kwargs):
        super().__init__(*configuration, **kwargs)
        self.softmax = Activators.Softmax()

    def init_parameter(self):#, m, n):
        m, n = self.get_parameter_size() 
        if m is None or n is None:
            raise Exception('Configuration is not fixed.', self.__class__.__name__)
        if self.width is not None:
            width = self.width
        else:    
            width = np.sqrt(1/m)  # Xavierの初期値

        self.w = (width * np.random.randn(m, n)).astype(Config.dtype) 
        self.b = np.zeros(n, dtype=Config.dtype)
        self.q = (width * np.random.randn(n, 1)).astype(Config.dtype) 
        if self.debug_mode:
            self.w[...] = 1; self.q[...] = 1 # for debug
        print(self.__class__.__name__, 'init_parameters', m, n)

    def __forward__(self, x):
        if None in self.config:
            #print(self.__class__.__name__, '.input.shape', x.shape)
            self.fix_configuration(x.shape)
        if self.w is None or self.b is None or self.q is None:
            self.init_parameter()
        m, n = self.config # m, n = H ベクトルサイズ
        B, T, _ = x.shape
        self.x = x
        # 入力xからkeyを生成 keyの形状 (B, T, H)
        #print('### x =', x.shape, 'w =', self.w.shape)   
        self.k = snp.matmul(x, self.w) + self.b      # (B,T,H)
        #print('### k =', self.k.shape, 'q =', self.q.shape)   
        # s:scoreとa:attention_weightを算出 sとaの形状:(B,T)
        s = snp.matmul(self.k, self.q)               # (B,T,H) * (H,1) -> (B,T,1)
        #print('### s =', s.shape)
        #s *= T ** -0.5                              # スケーリング
        s *= m ** -0.5                              # スケーリング
        self.a = self.softmax.forward(s)
        #print('### a =', self.a.shape)
        # 入力(B, T, H)をバッチ軸(B)はそのままにして、時系列軸(T)方向に重み付け
        c = x * self.a #[..., np.newaxis]
        # コンテキストベクトル y.shape:(B, H) 
        y = np.sum(c, axis=1)                       # (B,T,H) -> (B,H)
        #print('### y =', y.shape)
        return y

    def __backward__(self, gy):
        x = self.x
        a = self.a
        m, n = self.config
        B, T, _ = x.shape
        
        # 重み付け和の逆伝播
        gc = gy.reshape(B, 1, m).repeat(T, axis=1)
        ga = np.sum(gc * x, axis=-1, keepdims=True)
        gx = gc * self.a #[..., np.newaxis]
        #print('### ga =', ga.shape, 'gx =', gx.shape)
        # attention_weightの逆伝播
        gs = self.softmax.backward(ga)
        #gs *= T ** -0.5
        gs *= m ** -0.5
        #print('### gs =', gs.shape, 'q =', self.q.shape)
        gk = snp.matmul(gs, self.q.T)                # (B,T) * (1,H) -> (B,T,H)
        #print('### gk =', gk.shape)
        #print('### k =', self.k.shape, 'gs =', gs.shape) 
        gq = snp.matmul(self.k.transpose(0,2,1), gs) # (B,H,T) * (B,T,1) -> (B,H,1)
        #print('### gq =', gq.shape)
        self.grad_q = np.sum(gq, axis=0)            # (B,H,1) -> (H,1)
        #print('### grad_q =', self.grad_q.shape)
        # keyの逆伝播
        #self.grad_w = snp.matmul(x.reshape(-1, m).T, gk.reshape(-1, n))
        grad_w = snp.matmul(x.transpose(0,2,1), gk)  # (B,H,T) * (B,T,H) -> (B,H,H)
        self.grad_w = np.sum(grad_w, axis=0)        # (B,H,H) -> (H,H)
        self.grad_b = np.sum(gk, axis=(0,1))        # (B,T,H) -> (H,)
        #print('### grad_w', self.grad_w.shape)
        gx += snp.matmul(gk, self.w.T)
        #input()
        return gx



#### ドロップアウト ###############################################　
class Dropout(Function):
    """ inverted_dropout """ 
    def __init__(self, preset=None, inplace=False):
        super().__init__()
        self.preset = preset
        self.dropout_mx = None          # はじめて伝播する際に必要
        self.dropout_ratio = None       # 直前のforwardで実際に使ったdropout率
        self.inplace = inplace          # inplace演算とするかどうか
        
    def __forward__(self, x, *, dropout=0.0): # x→y,ドロップアウト率(非学習時は0)
        if self.preset is not None and dropout==0.0:
            dropout = self.preset
        y = x if self.inplace else x.copy() # inplaceではyはxと同一
        self.dropout_ratio = dropout

        if dropout > 0.0:               # ドロップアウトする場合に残る割合で拡大
            self.dropout_mx = np.random.rand(*x.shape) > dropout # True/Falseの配列
            scale = 1 / (1 - dropout + 1e-7)
            y *= self.dropout_mx        # ニューロンをランダムに無効化(0固定
            y *= scale                  # 予めスケールを合わせておく
        else:
            self.dropout_mx = 1         # ドロップアウトしたりしなかったりに対応   
        return y                        # inplaceの場合にはx更新で返り値不要

    def __backward__(self, gy):         # 順伝播時に無効化したニューロンは逆伝播しない
        gx = gy if self.inplace else gy.copy() # inplaceではgxはgyと同一

        gx *= self.dropout_mx
        if self.dropout_ratio > 0.0:
            scale = 1 / (1 - self.dropout_ratio + 1e-7)
            gx *= scale

        return gx                       # inplaceの場合にはgy更新で返り値不要 


#### ドロップアウト ###############################################　
class Dropout2(Function):
    """ direct_dropout """
    def __init__(self, preset=None, inplace=False):
        super().__init__()
        self.preset = preset
        self.dropout_mx = None          # はじめて伝播する際に必要
        self.dropout_ratio = None       # 最後に学習時に使ったdropout率
        self.inplace = inplace          # inplace演算とするかどうか
        
    def __forward__(self, x, *, dropout=0.0): # x→y,ドロップアウト率(非学習時は0)
        if self.preset is not None and dropout==0.0:
            dropout = self.preset
        y = x if self.inplace else x.copy() # inplaceではyはxと同一
        if dropout > 0.0: 
            self.dropout_ratio = dropout   # ドロップアウト時に覚える
            self.dropout_mx = np.random.rand(*x.shape) > dropout # True/Falseの行列
            y *= self.dropout_mx  # ニューロンをランダムに無効化(0固定
        else:           # ドロップアウトしない場合にスケールを合わせる
            dropout = 0 if self.dropout_ratio is None else self.dropout_ratio
            self.dropout_mx = 1.0       # ドロップアウトしたりしなかったりに対応　
            y *= (1 - dropout)
        return y                        # inplaceの場合にはx更新で返り値不要    
            
    def __backward__(self, gy):         # 順伝播時に無効化したニューロンは逆伝播しない
        gx = gy if self.inplace else gy.copy() # inplaceではgxはgyと同一
        gx *= self.dropout_mx           # 順伝播時の情報を使う
        return gx

class StatelessDropout:
    """ QueryChunkAttention用のDropout """ 
    def __init__(self, inplace=False):
        self.inplace = inplace          # inplace演算とするかどうか
        
    def forward(self, x, dropout_mx=1, dropout=0): # x→y,ドロップアウト率(非学習時は0)
        y = x if self.inplace else x.copy() 
        if dropout > 0.0:
            scale = 1 / (1 - dropout + 1e-7)
            y *= dropout_mx             # ニューロンをランダムに無効化(0固定
            y *= scale                  # 予めスケールを合わせておく
        return y                        # inplaceの場合にはx更新で返り値不要

    def backward(self, gy, dropout_mx=1, dropout=0): # 順伝播時に無効化したニューロンは逆伝播しない
        gx = gy if self.inplace else gy.copy() # inplaceではgxはgyと同一
        if dropout > 0.0:
            scale = 1 / (1 - dropout + 1e-7)
            gx *= dropout_mx
            gx *= scale
        return gx                       # inplaceの場合にはgy更新で返り値不要 


        
#### キャプチャ #####################################################　
#  RNN などで出力の一部、例えば、最後の時刻のみを使うような場合に対応する
#  forwardとbackwardで時系列長が異なる場合、たとえば最後の出力のみを使うような場合には
#  backward の際 grad_y に foward で後続層で使用された時刻の出力範囲のみ与えられるので
#  その部分に grad_y をはめ込んで、grad_y の他の部分は 0 として受け渡す
#  注：backward では、時系列の全域にわたり、勾配を逆伝播する必要があるから、
#      forward で出力を使用しなかった範囲については backward で順次算出される
#      リカレントなパスからの勾配のみを伝播する
class Capture(Function): 
    def __forward__(self, x, *, width=None):
        self.x = x
        self.config = None, None, width
        if width is None:
            return x
        #super().__init__()
        #print('Capture', x.shape, 'width', width, end='|')
        if   x.ndim==3:
            B, Tf, m = x.shape
        elif x.ndim==2:
            B, m     = x.shape
            Tf = 1
            x = x.reshape(B, 1, m)
        else:
            print('順伝播で入力の次元数が１以下ないしは４以上のため対応できません')
        self.config = Tf, m, width
        if width is not None:
            y = x[:, -width:, :]
        if width==1:
            y = y.reshape(B, m)
        return y

    def __backward__(self, grad_y):
        Tf, m, width = self.config
        x = self.x
        if width is None:
            return grad_y
        if   grad_y.ndim==3:
            B, Tb, n = grad_y.shape
        elif grad_y.ndim==2:   
            B, n     = grad_y.shape
            Tb = 1
            grad_y = grad_y.reshape(B, 1, n) 
        else:
            print('逆伝播で入力の次元数が１以下ないしは４以上のため対応できません')
        grad_x = np.zeros((B, Tf, n), dtype='f4')
        grad_x[:, -width:, :] = grad_y
        if x.ndim==2:
            grad_x = grad_x.reshape(B, n)
        return grad_x
    
def set_axis_and_shape(shape, axis, exclude=False):
    """ shapeとaxisからexcludeに従い、新たな形状とそこから外れる軸を得る """
    # shapeから全軸を抽出し、axisの指定を正規化
    ndim = len(shape)
    all_axis = list(range(ndim))        # すべての軸
    if axis is None:                    # Noneに対応
        axis = all_axis
    if type(axis) not in (tuple, list): # 整数に対応
        axis = axis,
    axis = tuple(ndim + ax if ax<0 else ax for ax in axis) # 負値に対応

    # axisに含まれる軸と含まれない軸
    include_axis = tuple(ax for ax in all_axis if ax in axis)
    exclude_axis = tuple(ax for ax in all_axis if ax not in axis)
    # axisの指定によって抽出される形状と排除される形状
    include_shape = tuple(shape[ax] if ax in axis else 1 for ax in all_axis)
    exclude_shape = tuple(1 if ax in axis else shape[ax] for ax in all_axis) 

    # axisとexcludeの指定による新たな形状とそこから外れる軸  
    if exclude:
        return exclude_shape, include_axis
    return include_shape, exclude_axis
    

#### 正規化のクラス #######################################################
class Normalization(Function):
    """ 平均0標準偏差1にする標準化(正規化の一種) """
    def __init__(self, axis=None, ppl=False, eps=1e-12,
                 mask_enable=False, inplace=False, **kwargs):
        super().__init__()
        #print('Initialize', self.__class__.__name__, axis, ppl)#, kwargs)
        self.axis = axis
        self.ppl = ppl
        self.eps = eps
        self.mu = None
        self.sigma = None
        self.mask = None
        self.mask_enable = mask_enable # mask small variant data
        self.inplace = inplace         # インプレース処理
        #optimize = kwargs.pop('optimize', 'SGD')
        if ppl: # 非訓練時に移動平均を使用(バッチノーマライゼーション)
            self.mu_ppl = None
            self.sigma_ppl = None
            self.OFm = cf.eval_in_module('SGD', Optimizers) # 最適化関数は固定 
            self.OFs = cf.eval_in_module('SGD', Optimizers) # 最適化関数は固定
        if mask_enable and inplace:
            msg = self.__class__.__name__ \
                  + ':Both mask_enable and inplace will cause wrong result.'
            warnings.warn(msg)
   
    def init_parameters(self, shape):       
        """ 軸の指定に従いmuとsigmaの形状を決めて初期化 """
        mu_sigma_shape, _ = set_axis_and_shape(shape, self.axis, True)
        #print('muとsigmaの形状', mu_sigma_shape)
        if self.ppl:
            self.mu_ppl = np.zeros(mu_sigma_shape, dtype=Config.dtype)
                                                 # 全体平均(移動平均 moving average)
            self.sigma_ppl = np.ones(mu_sigma_shape, dtype=Config.dtype)  # 全体分散

    def update(self, *args, **kwargs):
        if self.ppl:
            self.OFm.update(self.mu_ppl,    self.mu_ppl    - self.mu,    eta=0.1)
            self.OFs.update(self.sigma_ppl, self.sigma_ppl - self.sigma, eta=0.1)

    def __forward__(self, x, *, train=False):
        self.x = x
        if self.ppl and self.mu_ppl is None:
            #print(self.__class__.__name__, 'input.shape', x.shape)
            self.init_parameters(x.shape)
        #y = x if self.inplace else x.copy() # inplaceではyはxと同一
        self.mu = np.mean(x, axis=self.axis, keepdims=True)
        mu = self.mu_ppl if self.ppl and not train else self.mu
        if self.inplace:
            y = x
            y -= mu
        else:
            y = x - mu          
        var = np.mean(y*y, axis=self.axis, keepdims=True)
        self.sigma = np.sqrt(var + self.eps) # 極小分散に対応
        sigma = self.sigma_ppl if self.ppl and not train else self.sigma
        y /= sigma
        self.y = y

        if not self.mask_enable:
            return y
        self.mask = sigma < self.eps # sigmaが極小値の場合には正規化しない
        if (self.mask==False).all():
            return y
        y *= (1 - self.mask)
        y += x * self.mask           # inplaceでは誤動作 
        return y 
   
    def __backward__(self, gy):
        x = self.x
        y = self.y                   # inplaceではxと同一
        n = x.size//self.sigma.size  # 畳んだ大きさ
        gsigma = np.sum(-gy * y, axis=self.axis, keepdims=True) / self.sigma
                                                          # gyが書き変わる前に
        if self.inplace:
            gx = gy
            gx /= self.sigma     # gyを上書き
        else:    
            gx = gy / self.sigma # gyは壊さない
        
        #print('\n# backward_1 gx\n', gx)
        gmu = np.sum(gx, axis=self.axis, keepdims=True)
        gx -= gmu / n
        #print('# backward_2 gx\n', gx, '\ny\n', y, '\ngsigma\n', gsigma)
        gx += y * (gsigma / n)
        #print('# backward_3 gx\n', gx)
        if not self.mask_enable:
            return gx
        if (self.mask==False).all():
            return gx
        gx *= (1 - self.mask)
        gx += gy * self.mask
        return gx 

    def __forward__bkup(self, x, *, train=False):
        """ 参照用の処理 """
        self.x = x
        if self.ppl and self.mu_ppl is None:
            #print(self.__class__.__name__, 'input.shape', x.shape)
            self.init_parameters(x.shape)
        mu = np.mean(x, axis=self.axis, keepdims=True)
        sigma = np.std(x, axis=self.axis, keepdims=True)
        self.sigma = sigma
        self.mu = mu
        if self.ppl and not train:
            mu = self.mu_ppl 
            sigma = self.sigma_ppl
        z = x - mu
        y = z / (sigma + self.eps)
        self.y = y
        self.mask = sigma < self.eps # sigmaが極小値の場合には正規化しない
        return y * (1 - self.mask) + x * self.mask
   
    def __backward__bkup(self, gy):
        """ 参照用の処理 """
        x = self.x
        y = self.y
        sigma = self.sigma + self.eps
        n = x.size//sigma.size   # 畳んだ大きさ
        # y = z / sigma の逆伝播 
        gz = gy / sigma
        gsigma = -gy * y / sigma   # -gy * z / sigma ** 2
        gsigma = np.sum(gsigma, axis=self.axis, keepdims=True)
        # z = x - mu の逆伝播
        gx0 = gz
        gmu = np.sum(-gz, axis=self.axis, keepdims=True)
        # mu = np.mean(x) の逆伝播
        gx1 = np.broadcast_to(gmu, x.shape) / n
        # sigma = np.std(x) の逆伝播
        gx2 = gsigma * y / n
                     # gvar * dvar_dx = (gsigma * 0.5 / sigma) * ((2/n) * (x - mu))
        gx = gx0 + gx1 + gx2
        return gx * (1 - self.mask) + gy * self.mask

#### スケーリングとバイアスを適用するクラス ###############################
class ScaleAndBias(Function):
    def __init__(self, axis=None, exclude=False, **kwargs):
        super().__init__()
        #print('Initialize', self.__class__.__name__, axis, exclude)#, kwargs)
        self.axis = axis
        self.remain_axis = None
        self.exclude = exclude   # 処理がaxisの指定に沿うのか否か
        optimize = kwargs.pop('optimize',     'AdaGrad') # 勾配降下法
        self.OFg = cf.eval_in_module(optimize, Optimizers) # 最適化関数
        self.OFb = cf.eval_in_module(optimize, Optimizers) # 最適化関数
        self.gamma = None
        self.beta  = None

    def init_parameters(self, shape):
        """ 軸の指定に従いparameterの形状を決めて初期化、指定外の軸も設定 """
        parameter_shape, remain_axis = set_axis_and_shape(shape, self.axis, self.exclude)
        #print('parameterの形状', parameter_shape)
        self.gamma = np.ones(parameter_shape, dtype=Config.dtype)  # 広がり                    
        self.beta  = np.zeros(parameter_shape, dtype=Config.dtype) # オフセット                    
        self.remain_axis = remain_axis # 逆伝播に必要

    def update(self, eta=0.001, **kwargs): # 他のパラメタの更新と呼応して更新
        self.OFg.update(self.gamma, self.ggamma, eta, **kwargs) 
        self.OFb.update(self.beta,  self.gbeta,  eta, **kwargs)  
               
    def __forward__(self, x):
        self.x = x
        if self.gamma is None:
            self.init_parameters(x.shape)
        y = x * self.gamma    # xを温存しないと逆伝播出来ない
        y += self.beta
        return y

    def __backward__(self, gy):
        x = self.x
        self.gbeta  = np.sum(gy, axis=self.remain_axis, keepdims=True)
        self.ggamma = np.sum(x * gy, axis=self.remain_axis, keepdims=True)
        gx = gy * self.gamma
        return gx 

    def __forward__bkup(self, x):
        if self.gamma is None:
            self.init_parameters(x.shape)
        return self.gamma * x + self.beta

#### スカラ値によるスケーリングを適用するクラス ###############################
class ScalarScale(Function):
    def __init__(self, **kwargs):
        super().__init__()
        optimize = kwargs.pop('optimize',     'SGD')       # 勾配降下法
        self.OFg = cf.eval_in_module(optimize, Optimizers) # 最適化関数
        self.gamma = None

    def init_parameters(self):
        self.gamma = np.array(1.0, dtype=Config.dtype)                     

    def update(self, eta=0.001, **kwargs): # 他のパラメタの更新と呼応して更新
        self.OFg.update(self.gamma, self.ggamma, eta, **kwargs) 
               
    def __forward__(self, x):
        self.x = x
        if self.gamma is None:
            self.init_parameters()
        y = x * self.gamma    # xを温存しないと逆伝播出来ない
        return y

    def __backward__(self, gy):
        x = self.x
        self.ggamma = np.sum(x * gy)
        gx = gy * self.gamma
        return gx 

class FixedScale(Function):
    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = scale

    def __forward__(self, x):
        return x * self.scale

    def __backward__(self, gy):
        return gy * self.scale

#### 正規化の汎用ベース #### 
class GeneralNormalizationBase(Function):
    def __init__(self, axis=None, ppl=False, scale_and_bias=False, exclude=False,
                 eps=1e-12, inplace=False, 
                 **kwargs):
        super().__init__()
        self.ppl = ppl
        self.sb = scale_and_bias     
        self.axis = axis
        self.eps = eps
        self.mu = None
        self.sigma = None
        self.inplace = inplace
        if ppl: # 非訓練時に移動平均を使用(バッチノーマライゼーション)
            self.mu_ppl = None
            self.sigma_ppl = None
            self.OFm = cf.eval_in_module('SGD', Optimizers) # 最適化関数は固定 
            self.OFs = cf.eval_in_module('SGD', Optimizers) # 最適化関数は固定
        if self.sb:
            self.remain_axis = None
            self.exclude = exclude   # 処理がaxisの指定に沿うのか否か
            optimize = kwargs.pop('optimize',     'AdaGrad') # 勾配降下法
            self.OFg = cf.eval_in_module(optimize, Optimizers) # 最適化関数
            self.OFb = cf.eval_in_module(optimize, Optimizers) # 最適化関数
            self.gamma = None
            self.beta  = None

    def init_parameters(self, shape):
        if self.ppl:
            """ 軸の指定に従いmuとsigmaの形状を決めて初期化 """
            mu_sigma_shape, _ = set_axis_and_shape(shape, self.axis, True)
            self.mu_ppl   = np.zeros(mu_sigma_shape, dtype=Config.dtype) # 全体平均
            self.sigma_ppl = np.ones(mu_sigma_shape, dtype=Config.dtype) # 全体分散
          
        if self.sb:
            """ 軸の指定に従いparameterの形状を決めて初期化、指定外の軸も設定 """
            parameter_shape, remain_axis \
                             = set_axis_and_shape(shape, self.axis, self.exclude)
            self.gamma = np.ones(parameter_shape, dtype=Config.dtype)    # 広がり                    
            self.beta = np.zeros(parameter_shape, dtype=Config.dtype)    # オフセット                    
            self.remain_axis = remain_axis # 逆伝播に必要

    def update(self, eta=0.001, **kwargs):
        if self.ppl:
            self.OFm.update(self.mu_ppl,    self.mu_ppl    - self.mu,    eta=0.1)
            self.OFs.update(self.sigma_ppl, self.sigma_ppl - self.sigma, eta=0.1)
        if self.sb:
            self.OFg.update(self.gamma, self.ggamma, eta, **kwargs) 
            self.OFb.update(self.beta,  self.gbeta,  eta, **kwargs)  

    def __forward__(self, x, *, train=False):
        self.x = x
        if (self.ppl and self.mu_ppl is None) or (self.sb and self.gamma is None):
            self.init_parameters(x.shape)
            
        self.mu = np.mean(x, axis=self.axis, keepdims=True)
        mu = self.mu_ppl if self.ppl and not train else self.mu
        y = x - mu                           # xは逆伝播のために温存
        var = np.mean(y*y, axis=self.axis, keepdims=True) # 分散
        self.sigma = np.sqrt(var + self.eps) # 極小分散に対応
        sigma = self.sigma_ppl if self.ppl and not train else self.sigma

        if self.sb:
            y *= (self.gamma/sigma)          # Norm/sbの除乗算を同時に
            y += self.beta
        else:
            y /= sigma
           
        return y        

    def __backward__(self, gy):
        #x, = self.inputs
        x = self.x
        #y = self.get_outputs()
        n = x.size // self.sigma.size        # 正規化対象の要素数
        z = (x - self.mu) / self.sigma       # 中間値：Norm出力=Scale&Bias入力

        if self.sb:
            self.gbeta  = np.sum(gy, axis=self.remain_axis, keepdims=True)
            self.ggamma = np.sum(gy * z, axis=self.remain_axis, keepdims=True)
            gx = gy * self.gamma
        else:
            gx = gy if self.inplace else gy.copy()

        gsigma = np.sum(-gx * z, axis=self.axis, keepdims=True) / self.sigma 
        gx /= self.sigma
        gmu = np.sum(gx, axis=self.axis, keepdims=True) 
        gx -= (gmu / n)
        gx += z * (gsigma / n)
        return gx 

    def __backward__bkup(self, gy):
        x = self.x
        #y = self.get_outputs()
        n = x.size // self.sigma.size        # 正規化対象の要素数
        z = (x - self.mu) / self.sigma       # Norm出力=Scale&Bias入力

        if self.sb:
            self.gbeta  = np.sum(gy, axis=self.remain_axis, keepdims=True)
            self.ggamma = np.sum(gy * z, axis=self.remain_axis, keepdims=True)
            gx = gy * self.gamma
        else:
            gx = gy

        sum_gx = np.sum(gx, axis=self.axis, keepdims=True)
        sum_gx_y = np.sum(gx * z, axis=self.axis, keepdims=True)
        gx = (gx - sum_gx / n - z * (sum_gx_y / n)) / self.sigma

        return gx



#### 正規化の汎用ベース #### 
class GeneralNormalizationBase2(Function):
    def __init__(self, axis=None, ppl=False, scale_and_bias=False, exclude=False,
                       inplace=False, **kwargs):
        super().__init__()
        #print('Initialize', self.__class__.__name__)
        self.ppl = ppl
        self.normalization = Normalization(axis=axis, ppl=ppl, inplace=inplace, **kwargs)
        self.sb = scale_and_bias     
        if self.sb:
            self.scale_and_bias = ScaleAndBias(axis=axis, exclude=exclude, **kwargs)

    def init_parameters(self, shape):
        if self.ppl:
            self.normalization.init_parameters(shape)
        if self.sb:
            self.scale_and_bias.init_parameters(shape)

    def update(self, eta=0.001, **kwargs):
        self.normalization.update(eta, **kwargs)
        if self.sb:
            self.scale_and_bias.update(eta, **kwargs)
               
    def __forward__(self, x, *, train=False):
        y = self.normalization.forward(x, train=train)
        if self.sb:
            y = self.scale_and_bias.forward(y)
        return y        

    def __backward__(self, gy):
        if self.sb:
            gx = self.scale_and_bias.backward(gy)
        else:
            gx = gy # gyをインプレース更新
        gx = self.normalization.backward(gx)   
        return gx 

class BatchNormalization(GeneralNormalizationBase):
    def __init__(self, scale_and_bias=False, inplace=False, **kwargs):
        # 正規化の軸はバッチ軸0だが、
        # scale_and_biasの軸は特長軸すなわち0以外の軸
        # scale_and_biasを伴う場合にはnormalizationをinplace処理に出来る
        kwargs['axis']           = 0
        kwargs['exclude']        = True
        kwargs['ppl']            = True
        kwargs['inplace']        = inplace
        kwargs['scale_and_bias'] = scale_and_bias
        super().__init__(**kwargs)

class batch_normalization(BatchNormalization):
    pass


class BatchNorm1d(GeneralNormalizationBase):
    def __init__(self, scale_and_bias=False, inplace=False, **kwargs):
        # 正規化の軸はバッチ軸0+最後の空間軸(長さ軸)-1だが、
        # scale_and_biasの軸はこれ以外の軸
        # scale_and_biasを伴う場合にはnormalizationをinplace処理に出来る
        kwargs['axis']           = 0, -1
        kwargs['exclude']        = True
        kwargs['ppl']            = True
        kwargs['inplace']        = inplace
        kwargs['scale_and_bias'] = scale_and_bias
        super().__init__(**kwargs)

class batch_norm_1d(BatchNorm1d):
    pass

class BatchNorm2d(GeneralNormalizationBase):
    def __init__(self, scale_and_bias=False, inplace=False, **kwargs):
        # 正規化の軸はバッチ軸0+Ih+Iwだが、
        # scale_and_biasの軸はこれ以外の軸すなわちチャネル軸
        # scale_and_biasを伴う場合にはnormalizationをinplace処理に出来る
        kwargs['axis']           = 0, -2, -1
        kwargs['exclude']        = True
        kwargs['ppl']            = True
        kwargs['inplace']        = inplace
        kwargs['scale_and_bias'] = scale_and_bias
        super().__init__(**kwargs)

class batch_norm_2d(BatchNorm2d):
    pass

class LayerNormalization(GeneralNormalizationBase):
    def __init__(self, axis=-1, scale_and_bias=False, inplace=False, **kwargs):
        # 正規化もscale_and_biasも同じく特長軸をaxisで指定
        # scale_and_biasを伴う場合にはnormalizationをinplace処理に出来る
        kwargs['axis']           = axis
        kwargs['exclude']        = False
        kwargs['ppl']            = False
        kwargs['inplace']        = inplace
        kwargs['scale_and_bias'] = scale_and_bias
        super().__init__(**kwargs)

class LayerNorm1d(GeneralNormalizationBase):
    def __init__(self, scale_and_bias=False, inplace=False, **kwargs):
        # 正規化もscale_and_biasも同じく特長軸をaxisで指定
        # scale_and_biasを伴う場合にはnormalizationをinplace処理に出来る
        kwargs['axis']           = -2, -1
        kwargs['exclude']        = False
        kwargs['ppl']            = False
        kwargs['inplace']        = inplace
        kwargs['scale_and_bias'] = scale_and_bias
        super().__init__(**kwargs)

class LayerNorm2d(GeneralNormalizationBase):
    def __init__(self, scale_and_bias=False, inplace=False, **kwargs):
        # 正規化もscale_and_biasも同じく特長軸をaxisで指定
        # scale_and_biasを伴う場合にはnormalizationをinplace処理に出来る
        kwargs['axis']           = -3, -2, -1
        kwargs['exclude']        = False
        kwargs['ppl']            = False
        kwargs['inplace']        = inplace
        kwargs['scale_and_bias'] = scale_and_bias
        super().__init__(**kwargs)

class InstanceNorm2d(GeneralNormalizationBase):
    def __init__(self, scale_and_bias=False, inplace=False, **kwargs):
        # 正規化の軸はH,Wのみ
        kwargs['axis']           = -2, -1 # H, W
        kwargs['exclude']        = True   # ここがγ/βの軸の決め方に効く
        kwargs['ppl']            = False  # 通常はBatchに依存しないのでrunning統計なし
        kwargs['inplace']        = inplace
        kwargs['scale_and_bias'] = scale_and_bias
        super().__init__(**kwargs)

#### RMS正規化の汎用ベース #### 
class RootMeanSquareNormalization(Function):
    def __init__(self, axis=None, ppl=False, scale_and_bias=False, exclude=False,
                 eps=1e-12, inplace=False, 
                 **kwargs):
        super().__init__()
        self.ppl = ppl
        self.sb = scale_and_bias     
        self.axis = axis
        self.eps = eps
        self.sigma = None
        self.inplace = inplace
        if ppl: # 非訓練時に移動平均を使用(バッチノーマライゼーション)
            self.sigma_ppl = None
            self.OFs = cf.eval_in_module('SGD', Optimizers) # 最適化関数は固定
        if self.sb:
            self.remain_axis = None
            self.exclude = exclude   # 処理がaxisの指定に沿うのか否か
            optimize = kwargs.pop('optimize',     'AdaGrad') # 勾配降下法
            self.OFg = cf.eval_in_module(optimize, Optimizers) # 最適化関数
            self.OFb = cf.eval_in_module(optimize, Optimizers) # 最適化関数
            self.gamma = None
            self.beta  = None

    def init_parameters(self, shape):
        if self.ppl:
            """ 軸の指定に従いmuとsigmaの形状を決めて初期化 """
            mu_sigma_shape, _ = set_axis_and_shape(shape, self.axis, True)
            self.sigma_ppl = np.ones(mu_sigma_shape, dtype=Config.dtype) # 全体分散
          
        if self.sb:
            """ 軸の指定に従いparameterの形状を決めて初期化、指定外の軸も設定 """
            parameter_shape, remain_axis \
                             = set_axis_and_shape(shape, self.axis, self.exclude)
            self.gamma = np.ones(parameter_shape, dtype=Config.dtype)    # 広がり                    
            self.beta = np.zeros(parameter_shape, dtype=Config.dtype)    # オフセット                    
            self.remain_axis = remain_axis # 逆伝播に必要

    def update(self, eta=0.001, **kwargs):
        if self.ppl:
            self.OFs.update(self.sigma_ppl, self.sigma_ppl - self.sigma, eta=0.1)

        if self.sb:
            self.OFg.update(self.gamma, self.ggamma, eta, **kwargs) 
            self.OFb.update(self.beta,  self.gbeta,  eta, **kwargs)  

    def __forward__(self, x, *, train=False):
        self.x = x
        if (self.ppl and self.sigma_ppl is None) or (self.sb and self.gamma is None):
            self.init_parameters(x.shape)
            
        meansquare = np.mean(x*x, axis=self.axis, keepdims=True) # 二乗平均
        self.sigma = np.sqrt(meansquare + self.eps) # 極小値に対応
        sigma = self.sigma_ppl if self.ppl and not train else self.sigma

        if self.sb:
            y = x * (self.gamma/sigma)          # Norm/sbの除乗算を同時に
            y += self.beta
        else:
            y = x / sigma
           
        return y        

    def __backward__(self, gy):
        x = self.x
        #y = self.get_outputs()
        n = x.size // self.sigma.size        # 正規化対象の要素数
        z = x / self.sigma                   # 中間値：Norm出力=Scale&Bias入力

        if self.sb:
            self.gbeta  = np.sum(gy, axis=self.remain_axis, keepdims=True)
            self.ggamma = np.sum(gy * z, axis=self.remain_axis, keepdims=True)
            gx = gy * self.gamma
        else:
            gx = gy if self.inplace else gy.copy()

        gsigma = np.sum(-gx * z, axis=self.axis, keepdims=True) / self.sigma 
        gx /= self.sigma
        gx += z * (gsigma / n)
        return gx 

class RMSNormalization(RootMeanSquareNormalization):
    def __init__(self, axis=-1, scale_and_bias=False, inplace=False, **kwargs):
        # 正規化もscale_and_biasも同じく特長軸をaxisで指定
        # scale_and_biasを伴う場合にはnormalizationをinplace処理に出来る
        kwargs['axis']           = axis
        kwargs['exclude']        = False
        kwargs['ppl']            = False
        kwargs['inplace']        = inplace
        kwargs['scale_and_bias'] = scale_and_bias
        super().__init__(**kwargs)

class rms_normalization(RMSNormalization):
    pass

class RMSNorm1d(RootMeanSquareNormalization):
    def __init__(self, scale_and_bias=False, inplace=False, **kwargs):
        # 正規化もscale_and_biasも同じく特長軸をaxisで指定
        # scale_and_biasを伴う場合にはnormalizationをinplace処理に出来る
        kwargs['axis']           = -2, -1
        kwargs['exclude']        = False
        kwargs['ppl']            = False
        kwargs['inplace']        = inplace
        kwargs['scale_and_bias'] = scale_and_bias
        super().__init__(**kwargs)

class rms_norm_1d(RMSNorm1d):
    pass

class RMSNorm2d(RootMeanSquareNormalization):
    def __init__(self, scale_and_bias=False, inplace=False, **kwargs):
        # 正規化もscale_and_biasも同じく特長軸をaxisで指定
        # scale_and_biasを伴う場合にはnormalizationをinplace処理に出来る
        kwargs['axis']           = -3, -2, -1
        kwargs['exclude']        = False
        kwargs['ppl']            = False
        kwargs['inplace']        = inplace
        kwargs['scale_and_bias'] = scale_and_bias
        super().__init__(**kwargs)

class rms_norm_2d(RMSNorm2d):
    pass

class RMSBatchNormalization(RootMeanSquareNormalization):
    def __init__(self, scale_and_bias=False, inplace=False, **kwargs):
        # 正規化の軸はバッチ軸0だが、
        # scale_and_biasの軸は特長軸すなわち0以外の軸
        # scale_and_biasを伴う場合にはnormalizationをinplace処理に出来る
        kwargs['axis']           = 0
        kwargs['exclude']        = True
        kwargs['ppl']            = True
        kwargs['inplace']        = inplace
        kwargs['scale_and_bias'] = scale_and_bias
        super().__init__(**kwargs)

class rms_batch_normalization(RMSBatchNormalization):
    pass

class RMSBatchNorm1d(RootMeanSquareNormalization):
    def __init__(self, scale_and_bias=False, inplace=False, **kwargs):
        # 正規化の軸はバッチ軸0+最後の空間軸(長さ軸)-1だが、
        # scale_and_biasの軸はこれ以外の軸
        # scale_and_biasを伴う場合にはnormalizationをinplace処理に出来る
        kwargs['axis']           = 0, -1
        kwargs['exclude']        = True
        kwargs['ppl']            = True
        kwargs['inplace']        = inplace
        kwargs['scale_and_bias'] = scale_and_bias
        super().__init__(**kwargs)

class rms_batch_norm_1d(RMSBatchNorm1d):
    pass

class RMSBatchNorm2d(RootMeanSquareNormalization):
    def __init__(self, scale_and_bias=False, inplace=False, **kwargs):
        # 正規化の軸はバッチ軸0+Ih+Iwだが、
        # scale_and_biasの軸はこれ以外の軸すなわちチャネル軸
        # scale_and_biasを伴う場合にはnormalizationをinplace処理に出来る
        kwargs['axis']           = 0, -2, -1
        kwargs['exclude']        = True
        kwargs['ppl']            = True
        kwargs['inplace']        = inplace
        kwargs['scale_and_bias'] = scale_and_bias
        super().__init__(**kwargs)

class rms_batch_norm_2d(RMSBatchNorm2d):
    pass

#### 層正規化 #####################################################
class LayerNormalization_bkup(Function):
    def __init__(self, axis=None, ppl=False, scale_and_bias=False, **kwargs):
        super().__init__()
        print('Initialize', self.__class__.__name__)
        self.ppl = ppl
        self.normalization = Normalization(axis, ppl, **kwargs)
        self.sb = scale_and_bias     
        if self.sb:
            self.scale_and_bias = ScaleAndBias(axis, **kwargs)

    def init_parameters(self, shape):
        if self.ppl:
            self.normalization.init_parameters(shape)
        if self.sb:
            self.scale_and_bias.init_parameters(shape)

    def update(self, eta=0.001, **kwargs):
        self.normalization.update(eta, **kwargs)
        if self.sb:
            self.scale_and_bias.update(eta, **kwargs)
               
    def __forward__(self, x, *, train=False):
        y = self.normalization.forward(x, train=train)
        if self.sb:
            y = self.scale_and_bias.forward(y)
        return y        

    def __backward__(self, gy):
        if self.sb:
            gx = self.scale_and_bias.backward(gy)
        else:
            gx = gy
        gx = self.normalization.backward(gy)   
        return gx 

#### バッチノーマライゼーションの関数 ###############################
class batch_normalization2(Function):
    def __init__(self, *n):
        super().__init__()
        print('Initialize', self.__class__.__name__)   
        self.OFg = cf.eval_in_module('AdaGrad', Optimizers) # 最適化関数
        self.OFb = cf.eval_in_module('AdaGrad', Optimizers) # 最適化関数
        self.OFm = cf.eval_in_module('SGD',     Optimizers) # 最適化関数 
        self.OFv = cf.eval_in_module('SGD',     Optimizers) # 最適化関数
        self.gamma   = None
        self.beta    = None
        self.mu_ppl  = None
        self.var_ppl = None
        self.cnt      = 0                 # 学習回数

    def init_parameters(self, n):         # nは入力の大きさ(バッチサイズではない)
        self.gamma   = np.ones(n, dtype='f4')      # 広がり                    
        self.beta    = np.zeros(n, dtype='f4')     # オフセット                    
        self.mu_ppl  = np.zeros(n, dtype='f4')     # 全体平均(移動平均 moving average)
        self.var_ppl = np.ones(n, dtype='f4')      # 全体分散

    def update(self):
        self.OFg.update(self.gamma, self.ggamma)
        self.OFb.update(self.beta,  self.gbeta)
        self.OFm.update(self.mu_ppl,  self.mu_ppl  - self.mu,  eta=0.1)#, g_clip=100)
        self.OFv.update(self.var_ppl, self.var_ppl - self.var, eta=0.1)#, g_clip=100)
               
    def __forward__(self, x, *, train=False):
        if self.mu_ppl is None:
            #print(self.__class__.__name__, 'input.shape', x.shape)
            self.init_parameters(x.shape[1:])
        '''
        画像の周辺の背景部分などではバッチ内で同一値となるのは当然起きる
        この時バッチ内で分散は極小値(理論的には0)となる
        バッチ内分散が極小の場合は標準偏差も極小となって、分母が極小のため正規化はできない
        そこで分散を値1.0に固定する(標準偏差も1.0になる)
        '''
        self.x  = x
        mu = np.mean(x, axis=0)
        var = np.var(x, axis=0, ddof=0)
        #mask = var < 1e-12      # varが極小値でmask=1
        #var += mask             # varを１に固定
        #self.mask = mask
        self.var  = var
        self.mu   = mu
        mu  = mu  if train else self.mu_ppl 
        var = var if train else self.var_ppl
        xhat = (x - mu) / (np.sqrt(var) + 1e-12)            
        return self.gamma * xhat + self.beta

    def __backward__(self, gy):
        #mask  = self.mask
        istd = 1 / np.sqrt(self.var) 
        iN   = 1 / len(gy)      # バッチサイズ
        xc   = self.x - self.mu
        xhat = xc * istd
        self.gbeta  = np.sum(gy, axis=0)
        self.ggamma = np.sum(xhat * gy, axis=0)
        gxhat  = gy * self.gamma
        gy_sum = np.sum(gxhat * xc, axis=0, keepdims=True)
        gz   = (gxhat - (xhat * gy_sum * istd * iN)) * istd
        gz_sum = np.sum(gz, axis=0, keepdims=True)
        gx   = gz - (gz_sum * iN)
        return gx 
        #return gx * (1 - mask) + gy * mask # BN非対象の場合にはgyを直にgxに伝播

#### L2ノーマライゼーションの関数 ###############################
class L2Normalize(Function):
    def __init__(self, axis=None):
        super().__init__()
        self.axis = axis
        self.config = None
    
    def __forward__(self, x):
        x = np.array(x)
        l2n = np.sum(x**2, axis=self.axis, keepdims=True)**0.5
        y = x / l2n
        self.x = x
        self.l2n = l2n
        self.y = y
        return y
   
    def __backward__(self, gy=1):
        x = self.x
        y = self.y
        l2n = self.l2n
        gx0 = gy / l2n
        gl2n = - gy * x / l2n ** 2
        gl2n = np.sum(gl2n, self.axis, keepdims=True)
        gsqsm = gl2n * 0.5 / (l2n + 1e-12)
        gx1 = 2 * x * gsqsm
        gx = gx0 + gx1
        return gx


### 平坦化 #####################################################
class Flatten(Function):
    def __init__(self):
        super().__init__()
        print("Functionsに定義されたものを使ってください")
        self.config = None

    def __forward__(self, x, *args, **kwargs):
        self.x_shape = x.shape
        return x.reshape(self.x_shape[0], -1)

    def __backward__(self, gy=1):
        if isinstance(gy, np.ndarray):
            return gy.reshape(*self.x_shape)
        else:
            return np.ones(self.x_shape, dtype=Config.dtype)
        
### 平均 #######################################################
class Mean(Function):
    def __init__(self, axis=None, keepdims=False):
        super().__init__()
        print("Functionsに定義されたものを使ってください")
        self.axis = axis
        self.keepdims=keepdims
        
    def __forward__(self, x):
        self.x_shape = x.shape
        self.x_size  = x.size
        y = np.mean(x, axis=self.axis, keepdims=self.keepdims)
        self.y_shape = y.shape
        return y

    def __backward__(self, gy=1):
        gy = gy if isinstance(gy, np.ndarray) else np.array(gy, dtype=Config.dtype) 
        gy = np.broadcast_to(gy, self.y_shape)  # 先ずはyの形状に合わせる
        if self.axis is not None:               # 畳まれた軸は1、他は元の形状
            gy_shape = self.x_shape[:self.axis] + (1,) + self.x_shape[self.axis+1:]
            n = self.x_shape[self.axis]         # 畳まれる軸内の要素数
        else:
            gy_shape = self.y_shape
            n = self.x_size
        gx = (1/n) * gy.reshape(gy_shape)
        gx = np.broadcast_to(gx, self.x_shape)
        return gx

# BaseLayerのカテゴリ辞書の登録
#BaseLayer.resolve_categories(globals())
    
#########################################
if __name__=='__main__':
    print('\n#### all cast ####')
    import inspect
    import sys
    current_module = sys.modules[__name__]
    classes = map(lambda x:x[0],inspect.getmembers(current_module,inspect.isclass))
    models = []
    for c in classes:
        print(c)

     
