# Initailizer
# 2026.06.03 A.Inoue

from pyaino.Config import *
#set_np('numpy');set_seed(1); np = Config.np

class BaseInitializer:
    def __init__(self, distribution='normal', rng=None, width=None):
        self.width = width # 外部からの指定
        self.distribution = distribution.lower()
        if rng is not None:
            self.rng = rng
        else:
            rng = np.random.default_rng(Config.seed) # dtype引数が無いので注意
            if all(hasattr(rng, attr) for attr in ('normal', 'uniform')):
                self.rng = rng
            else:
                self.rng = np.random.RandomState(Config.seed)
        debug_print('random generator is', self.rng.__class__.__name__)        

    def __call__(self, shape, scale):
        scale = float(scale) # nlcpyは0次元配列の書式指定や乱数の引数への配列を受け付けない
        debug_print(self.__class__.__name__, shape, f'scale={scale:6.4f}', self.distribution)
        if self.distribution == 'normal': 
            return self.rng.normal(0.0, scale ** 0.5, size=shape).astype(Config.dtype)
        elif self.distribution == 'uniform':
            limit = (3 * scale) ** 0.5
            return self.rng.uniform(-limit, limit, size=shape).astype(Config.dtype)
        else:
            raise ValueError(f"Unsupported distribution '{self.distribution}' in {self.__class__.__name__}")
      
     
class Xavier(BaseInitializer):
    def __call__(self, shape):
        m, n = shape
        scale = 1.0 / (m + n)
        return super().__call__(shape, scale)
        
class He(BaseInitializer):
    def __call__(self, shape):
        m, n = shape
        scale = 2.0 / m
        return super().__call__(shape, scale)
        
class LeCun(BaseInitializer):
    def __call__(self, shape):
        m, n = shape
        scale = 1.0 / m
        return super().__call__(shape, scale)

class Orthogonal(BaseInitializer):
    def __call__(self, shape):
        """ 指定形状に対しSVDに基づき列直交ないしは行直交な行列 """
        m, n = shape
        head = 'Column-wise' if m > n else 'Row-wise'
        a = self.rng.normal(0.0, 1.0, size=shape).astype(Config.dtype)
        #scale = 1.0 / (m + n) # Xavier
        #if self.distribution == 'normal': 
        #    a = self.rng.normal(0.0, np.sqrt(scale), size=shape).astype(Config.dtype)
        #elif self.distribution == 'uniform':
        #    limit = np.sqrt(3 * scale)
        #    a = self.rng.uniform(-limit, limit, size=shape).astype(Config.dtype)
        debug_print(f"{head} {self.__class__.__name__} shape={shape} {self.distribution}")
        u, _, v = np.linalg.svd(a, full_matrices=False)
        if m >= n:
            return u  # m >= n のときuは列直交
        else:
            return v  # m <  n のときvは行直交

class FixedWidth(BaseInitializer):
    def __init__(self, **kwargs):
        width = kwargs.pop('width')  
        self.width = width
        super().__init__(width=width, **kwargs)
       
    def __call__(self, shape):
        scale = self.width
        return super().__call__(shape, scale)
        
class DebugMode:
    def __call__(self, shape):
        print('debug_mode:force 0.1')
        return np.full(shape, 0.1, dtype=Config.dtype)


class Legacy:
    def __init__(self, method=None, width=None, activator=None):
        self.method    = method
        self.width     = width
        self.activator = activator

    def __call__(self, shape):
        m, n = shape # l:戻りパス、m:入力、n:ニューロン数

        if self.width is not None:
            width = self.width
            name = str(self.width)
        elif self.method=='he':
            width = np.srqt(2/m)
            name = 'he'
        else: # Xavierの初期値   
            width = np.sqrt(1/m) 
            name = 'Xavier'
        # パラメータの初期値
        weight = (width * np.random.randn(m, n)).astype(Config.dtype)
        debug_print(self.__class__.__name__, name, shape, width)
        return weight


# --- Factory function ---
class WeightInitializer:
    def __init__(self, **kwargs):

        method      = kwargs.pop('method',      None)
        width       = kwargs.get('width',       None)
        activator   = kwargs.pop('activator',   None)
        debug_mode  = kwargs.pop('debug_mode', False)
        legacy      = kwargs.pop('legacy',     False) 
        
        if debug_mode:
            self.method = DebugMode()
        elif legacy:
            self.method = Legacy(method, width, activator)
        elif width is not None:
            self.method = FixedWidth(**kwargs)
        elif method is None: # 以下、無指定で自動判別
            if activator.__class__.__name__ in ('ReLU', 'LReLU', 'ELU', 'Swish', 'Mish'): 
                self.method = He(**kwargs)
            else:                                
                self.method = Xavier(**kwargs)
        else:                # 以下、指定された場合 
            if method.lower()=='lecun':
                self.method = LeCun(**kwargs)
            elif method.lower()=='orthogonal':
                self.method = Orthogonal(**kwargs)
            else:
                raise NotImplementedError()
                
    def __call__(self, shape):
        return self.method(shape)
    
def init_weight(shape, **kwargs):
    return WeightInitializer(**kwargs)(shape)

if __name__=='__main__':

    print('WeightInitializerの基本テスト', '-'*21)
    init_w = WeightInitializer(method='lecun')
    w = init_w((3, 4))
    print(w)
    init_w = WeightInitializer(method='orthogonal')
    w = init_w((3, 4))
    print(w)

    class BaseLayer:
        def __init__(self, m, n, activator='Mish', width=None, debug_mode=False, bias=True):
            self.size  = m, n
            self.width = width
            self.activator = activator
            self.bias = bias
            self.debug_mode = debug_mode

        def init_parameter(self):#, m, n):
            """ Neuron.BaseLayerから """
            m, n = self.get_parameter_size() # m:入力幅、n:ニューロン数
            if m is None or n is None:
                raise Exception('Configuration is not fixed.', self.__class__.__name__)

            self.w = init_weight((m, n),
                                             width=self.width,
                                             activator=self.activator,
                                             debug_mode=self.debug_mode)        
          
            if self.bias:
                self.b = np.zeros(n, dtype=Config.dtype)                   

        def get_parameter_size(self):
            return self.size
            

    print('基本的な初期化のテスト', '-'*28)
    model = BaseLayer(3, 4)
    model.init_parameter()
    print('model.w\n', model.w)
    print('model.b\n', model.b)
    print('mean =', np.mean(model.w), 'std =', np.std(model.w))

    print('width指定による初期化のテスト', '-'*21)
    model = BaseLayer(3, 4, width=2.0)
    model.init_parameter()
    print('model.w\n', model.w)
    print('model.b\n', model.b)
    print('mean =', np.mean(model.w), 'std =', np.std(model.w))

    print('デバグモードでの初期化のテスト', '-'*20)
    model = BaseLayer(3, 4, debug_mode=True)
    model.init_parameter()
    print('model.w\n', model.w)
    print('model.b\n', model.b)
    print('mean =', np.mean(model.w), 'std =', np.std(model.w))

    print('Orthogonalの単体テスト', '-'*28)
    w = Orthogonal()(shape=(5, 3))
    rwe = np.linalg.norm(w @ w.T - np.eye(w.shape[0]))
    cwe = np.linalg.norm(w.T @ w - np.eye(w.shape[1]))
    print(f"orthogonality: row-wise={rwe<1e-5} column-wise={cwe<1e-5}")

    w = Orthogonal()(shape=(3, 5))
    rwe = np.linalg.norm(w @ w.T - np.eye(w.shape[0]))
    cwe = np.linalg.norm(w.T @ w - np.eye(w.shape[1]))
    print(f"orthogonality: row-wise={rwe<1e-5} column-wise={cwe<1e-5}")

    

    



