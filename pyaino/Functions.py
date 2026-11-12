# Functions 順伝播逆伝播双方に対応した関数
# 20260920 A.Inoue

from pyaino.Config import *
from pyaino.nucleus import Function, HDArray
from pyaino import safe_np as snp
from functools import reduce
import itertools

class Assign(Function):
    def __forward__(self, x):
        return x.copy()  # <要注意>入力と出力は別物にする必要がある

    def __backward__(self, gy):
        return assign(gy)
    
def assign(x):
    return Assign()(x)

class Branch(Function):
    """ 下流へ分岐する際に、下流からの勾配を順に受け取り加算する """
    def __init__(self):
        super().__init__()
        self.gx = None
        
    def __forward__(self, x):
        self.x = x
        return x.copy()
    
    def __backward__(self, gy, *, flush=True):
        x = self.x
        if self.gx is None or flush: 
            self.gx = np.zeros_like(x)
        self.gx += gy             
        return self.gx
    
def bracch(x):
    return Branch()(x)

class Neg(Function):
    def __forward__(self, x):
        return -x

    def __backward__(self, gy):
        return -gy 

def neg(x):
    return Neg()(x)

class Pow(Function):
    def __init__(self, c=1):
        super().__init__()
        self.c = c
        
    def __forward__(self, x):
        self.x = x
        y = np.power(x, self.c) # 20241019 x**c
        return y

    def __backward__(self, gy):
        x = self.x
        c = self.c
        gx = gy * c * x ** (c - 1)
        return gx

def pow(x, c):
    return Pow(c)(x)

class Square(Function):
    def __forward__(self, x):
        self.x = x
        y = np.square(x)
        return y

    def __backward__(self, gy):
        x = self.x
        gx = gy * 2 * x 
        return gx
    
def square(x):
    return Square()(x)

class Sqrt(Function):
    def __forward__(self, x):
        y = np.sqrt(x)
        self.y = y
        return y

    def __backward__(self, gy):
        y = self.y
        gx = gy * 0.5 / (y + 1e-12) # gy * 0.5 * self.x ** (-0.5) 
        return gx
    
def sqrt(x):
    return Sqrt()(x)

class Exp(Function):
    """ 指数関数(底を指定可能) """
    def __init__(self, a=None):
        super().__init__()
        if a is None:          # 底がネイピア数eの場合
            log_of_base = 1          
        else:                  # 底が指定された場合　
            log_of_base = np.log(a) # 底がeの対数,これを用いて底の交換
        self.log_of_base = log_of_base   
   
    def __forward__(self, x):
        y = np.exp(self.log_of_base * x)
        self.y = y
        return y

    def __backward__(self, gy):
        y = self.y
        gx = gy * self.log_of_base * y
        return gx

def exp(x, a=None):
    return Exp(a)(x)


class Log(Function):
    """ 対数関数(底を指定可能) """
    def __init__(self, a=None):
        super().__init__()
        if a is None:          # 自然対数
            log_of_base = 1
        elif a > 0 and a != 1: # 対数の底が指定された場合
            log_of_base = np.log(a) 
        else:
            raise Exception("Bad base is given for log")
        self.log_of_base = log_of_base   
       
    def __forward__(self, x):
        self.x = x
        y = np.log(x)/self.log_of_base
        return y

    def __backward__(self, gy):
        x = self.x
        gx = gy / (x * self.log_of_base)
        return gx

def log(x, a=None): 
    return Log(a)(x) 

class Abs(Function):
    def __forward__(self, x):
        self.x = x
        y = np.abs(x)
        return y

    def __backward__(self, gy):
        x = self.x
        gx = gy * ((x >= 0) * 2 - 1)
        return gx

def abs(x):
    return Abs()(x)
    
class Sin(Function):
    def __forward__(self, x):
        self.x = x
        y = np.sin(x)
        return y

    def __backward__(self, gy):
        x = self.x
        gx = gy * cos(x)
        return gx

def sin(x):
    return Sin()(x)

class Cos(Function):
    def __forward__(self, x):
        self.x = x
        y = np.cos(x)
        return y

    def __backward__(self, gy):
        x = self.x
        gx = gy * - sin(x)
        return gx

def cos(x):
    return Cos()(x)

class Erf(Function):
    """ 誤差関数(ガウスの誤差関数) """
    def __init__(self):
        super().__init__()

        if np.__name__ == 'cupy':
            from cupyx.scipy.special import erf
            self.erf = erf
        else:
            try:
                from scipy.special import erf
                self.erf = erf
            except ImportError:
                self.erf = self.AbramowitzStegun

    @staticmethod
    def AbramowitzStegun(x):
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429
        p  = 0.3275911
        sign = np.sign(x)
        ax = np.abs(x)
        t = 1.0 / (1.0 + p * ax)
        poly = (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t
        y = 1.0 - poly * np.exp(-ax * ax)
        return sign * y

    def __forward__(self, x):
        self.x = x
        return self.erf(x)

    def __backward__(self, gy):
        x = self.x
        return gy * (2.0 / np.sqrt(np.pi)) * np.exp(-x * x)

class Add(Function):
    def __forward__(self, x0, x1):
        self.x0, self.x1 = x0, x1
        y = x0 + x1
        self.y_shape = y.shape
        return y
    
    def __backward__(self, gy):
        x0, x1 = self.x0, self.x1
        y_shape = self.y_shape
        x0_shape, x1_shape = np.shape(x0), np.shape(x1)
        gx0 = gy if y_shape==x0_shape else SumTo(x0_shape)(gy)
        gx1 = gy if y_shape==x1_shape else SumTo(x1_shape)(gy)
        return gx0, gx1

def add(x0, x1):
    return Add()(x0, x1)

class Sub(Function):
    def __forward__(self, x0, x1):
        self.x0, self.x1 = x0, x1
        y = x0 - x1
        self.y_shape = y.shape
        return y

    def __backward__(self, gy):
        x0, x1 = self.x0, self.x1
        y_shape = self.y_shape
        x0_shape, x1_shape = np.shape(x0), np.shape(x1)
        gx0 =  gy         if y_shape==x0_shape else SumTo(x0_shape)(gy)
        gx1 = -gy         if y_shape==x1_shape else SumTo(x1_shape)(-gy)
        return gx0, gx1

def sub(x0, x1):
    return Sub()(x0, x1)

def rsub(x0, x1):
    return Sub()(x1, x0)

class Mul(Function):
    def __forward__(self, x0, x1):
        self.x0, self.x1 = x0, x1
        y = x0 * x1
        self.y_shape = y.shape
        return y
    
    def __backward__(self, gy):
        x0, x1 = self.x0, self.x1
        y_shape = self.y_shape
        x0_shape, x1_shape = np.shape(x0), np.shape(x1)
        gx0 = x1 * gy
        gx1 = x0 * gy
        gx0 = gx0 if y_shape==x0_shape else SumTo(x0_shape)(gx0)
        gx1 = gx1 if y_shape==x1_shape else SumTo(x1_shape)(gx1)
        return gx0, gx1
    
def mul(x0, x1):
    return Mul()(x0, x1)

class Div(Function):
    def __forward__(self, x0, x1):
        self.x0, self.x1 = x0, x1
        y = x0 / x1
        self.y_shape = y.shape
        return y
    
    def __backward__(self, gy):
        x0, x1 = self.x0, self.x1
        y_shape = self.y_shape
        x0_shape, x1_shape = np.shape(x0), np.shape(x1)
        gx0 = gy / x1
        gx1 = - gy * x0 / x1 ** 2
        gx0 = gx0 if y_shape==x0_shape else SumTo(x0_shape)(gx0)
        gx1 = gx1 if y_shape==x1_shape else SumTo(x1_shape)(gx1)
        return gx0, gx1

def div(x0, x1):
    return Div()(x0, x1)

def rdiv(x0, x1):
    return Div()(x1, x0)

class SumTo(Function):
    def __init__(self, shape=()):
        super().__init__()
        self.shape = shape
        self.gy_shape = None

    def __forward__(self, x):
        self.x = x
        if x.shape == self.shape:
            self.gy_shape = x.shape             # backwardで必要
            return x

        ope_shape = list(self.shape)            # 操作用にコピー
        target_shape = () 
        for i, sx in enumerate(x.shape):        # 次元数を揃えたターゲット形状を得る
            if   len(ope_shape) > i and ope_shape[i] == sx:
                target_shape += sx,
            elif len(ope_shape) > i and ope_shape[i] == 1:
                target_shape += 1,
            else:                               # ope_shape[i] != sxもこのケース
                target_shape += 1,
                ope_shape = [1] + ope_shape     # ope_shapeを1つずらす
        axis = ()
        for i in range(len(x.shape)):           # 同一次元数での比較して軸を得る　
            if target_shape[i] != x.shape[i]:
                axis += i,

        y = snp.sum(x, axis=axis, keepdims=True) # 下記の為に次元保持
        self.gy_shape = y.shape                 # backwardで必要(操作したxの形状に)
        #y = y.reshape(self.shape)
        y = snp.reshape(y, self.shape)
        return y

    def __backward__(self, gy):
        x = self.x
        #gx = gy.reshape(self.gy_shape)          # 先ずは次元数を合わせる
        gx = snp.reshape(gy, self.gy_shape)      # 先ずは次元数を合わせる
        gx = snp.broadcast_to(gx, x.shape)       # それから所望のbroadcast
        return gx

def sum_to(x, shape):
    return SumTo(shape)(x)

class BroadcastTo(Function):
    def __init__(self, shape=()):
        super().__init__()
        self.shape = shape

    def __forward__(self, x):
        self.x = x
        y = snp.broadcast_to(x, self.shape)
        return y

    def __backward__(self, gy):
        x = self.x
        gx = sum_to(gy, np.shape(x))
        return gx

def broadcast_to(x, shape):
    return BroadcastTo(shape)(x)

class SumMeanVar(Function):
    """ 配列操作を担う共通クラス """
    def __init__(self, axis=None, dtype=None, out=None, keepdims=False):
        super().__init__()
        self.keepdims = keepdims
        self.axis = axis
        self.n = None
        self.gy_shape = None
        self.dtype = dtype # 仮処置20260127AI
        self.out = out     # 仮処置20260127AI
        self.last_x_shape = None
    
    def set_axis_and_shape(self, shape):
        """ 畳まれる軸と残る軸を明らかにする """
        ndim = len(shape)
        axis = self.axis # 指定された軸
        all_axis = list(range(ndim))
        if axis is None:
            axis = all_axis
        if type(axis) not in (tuple, list):
            #print('axisをタプルに')
            axis = axis,
            
        # 畳まれる軸と畳まれない軸    
        axis = [ndim + ax if ax<0 else ax for ax in axis] # 負値に対応

        # 畳まれる軸を1、他はそのままの形状
        self.gy_shape = [1 if ax in axis else s for ax, s in zip(all_axis, shape)]

        n = 1
        for ax in axis:
            n *= shape[ax] # 畳まれる軸の形状の積=大きさ
        #print('指定された軸', self.axis)
        #print('畳まれる軸', axis)
        #print('畳まれる軸を1、他はそのままの形状', self.gy_shape)
        #print('畳まれる軸の形状の積=大きさ', n)
        self.n = n
        
    def __backward__(self, gy):
        x = self.x
        x_shape = np.shape(x)
        if self.last_x_shape != x_shape:
            self.set_axis_and_shape(x_shape)
            self.last_x_shape = x_shape
        gy = snp.reshape(gy, self.gy_shape)         # gyは次元を合わせる
        gy = snp.broadcast_to(gy, x_shape)          # 畳まれた分をbroadcastして元に戻す
        return gy


class Sum(SumMeanVar):
    """ 和 """
    def __forward__(self, x):
        self.x = x
        y = snp.sum(x, axis=self.axis, keepdims=self.keepdims)
        return y

def sum(x, axis=None, keepdims=False):
    return Sum(axis=axis, keepdims=keepdims)(x)

class Mean(SumMeanVar):
    """ 平均 """
    def __forward__(self, x):
        self.x = x
        y = snp.mean(x, axis=self.axis, keepdims=self.keepdims)
        return y
    
    def __backward__(self, gy):
        gy = super().__backward__(gy)
        gx = gy * (1/self.n) 
        return gx

def mean(x, axis=None, dtype=None, out=None, keepdims=False):
    return Mean(axis=axis, dtype=dtype, out=out, keepdims=keepdims)(x)

class Var(SumMeanVar):
    """ 分散 """
    def __forward__(self, x):
        self.x = x
        y = snp.var(x, axis=self.axis, keepdims=self.keepdims)
        return y

    def __backward__(self, gy):
        gy = super().__backward__(gy)
        x = self.x
        mu = snp.mean(x, axis=self.axis, keepdims=True)
        gx = gy * (2/self.n) * (x - mu)
        return gx

def var(x, axis=None, keepdims=False):
    return Var(axis=axis, keepdims=keepdims)(x)

class Std(SumMeanVar):
    """ 標準偏差 """
    def __forward__(self, x):
        self.x = x
        y = snp.std(x, axis=self.axis, keepdims=self.keepdims)
        self.y = y
        return y

    def __backward__(self, gy):
        gy = super().__backward__(gy)
        x = self.x
        y = self.y
        mu = snp.mean(x, axis=self.axis, keepdims=True)
        yr = snp.reshape(y, self.gy_shape)
        gvar = gy * 0.5 / (yr + 1e-12) # std = sqrt(var) の逆伝播　
        dvar_dx = (2/self.n) * (x - mu)
        gx = gvar * dvar_dx
        return gx

def std(x, axis=None, keepdims=False):
    return Std(axis=axis, keepdims=keepdims)(x)

class SquareSum(SumMeanVar):
    """ 二乗和 """
    def __forward__(self, x):
        self.x = x
        y = snp.sum(x**2, axis=self.axis, keepdims=self.keepdims)
        return y

    def __backward__(self, gy):
        gy = super().__backward__(gy)
        x = self.x
        gx = gy * 2 * x
        return gx
        
class SquareMean(SumMeanVar):
    """ 二乗平均 """
    def __forward__(self, x):
        self.x = x
        y = snp.mean(x**2, axis=self.axis, keepdims=self.keepdims)
        return y

    def __backward__(self, gy):
        gy = super().__backward__(gy)
        x = self.x
        gx = gy * (1/self.n) * 2 * x
        return gx
    
class RootSumSquare(SumMeanVar):
    """ 二乗和平方根 """
    def __forward__(self, x):
        self.x = x
        sqsm = snp.sum(x**2, axis=self.axis, keepdims=self.keepdims)
        y = np.sqrt(sqsm)
        self.y = y
        return y

    def __backward__(self, gy):
        gy = super().__backward__(gy)
        x = self.x
        y = self.y
        yr = snp.reshape(y, self.gy_shape) # gyと形状を揃える
        gsqsm = gy * 0.5 / (yr + 1e-12)    # RMS = sqrt(sqmu)の逆伝播
        gx = 2 * x * gsqsm
        return gx

class RootMeanSquare(SumMeanVar):
    """ 二乗平均平方根(RootMeanSquare) """
    def __forward__(self, x):
        self.x = x
        sqmu = snp.mean(x**2, axis=self.axis, keepdims=self.keepdims)
        y = np.sqrt(sqmu)
        self.y = y
        return y

    def __backward__(self, gy):
        gy = super().__backward__(gy)
        x = self.x
        y = self.y
        yr = snp.reshape(y, self.gy_shape) # gyと形状を揃える
        gsqmu = gy * 0.5 / (yr + 1e-12)    # RMS = sqrt(sqmu)の逆伝播
        gx = (1/self.n) * 2 * x * gsqmu
        return gx

class RMS(RootMeanSquare):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class VariadicBase:
    def __init__(self):
        self.func = None
        self.packed_in_one = None

    def forward(self, *xs): # 引数の数は不定
        if all(isinstance(x, np.ndarray) for x in xs):    
            y = self.func(*xs)
            self.packed_in_one = False
            
        elif len(xs)==1 and all(isinstance(x, (tuple, list)) for x in xs):
            xs, = xs
            y = self.func(*xs)
            self.packed_in_one = True
            
        elif len(xs)==1 and all(isinstance(x, type((i for i in []))) for x in xs):
            y = self.func(*tuple(xs[0]))
            self.packed_in_one = True
        else:
            raise Exception('Non-compliant input data.')
        return y

    def __call__(self, *xs):
        return self.forward(*xs)

    def backward(self, gy=1):
        gxs = self.func.backward(gy)
        return gxs[0] if len(gxs)==1 else gxs
    
class SumVariadicCore(Function):
    def __forward__(self, *xs):
        #print('->', self.__class__.__name__, type(xs), len(xs), '\n', xs)
        y = reduce(lambda a, b : a + b, xs)
        self.reduction = len(xs)
        return y

    def __backward__(self, gy):
        gxs = (gy,) * self.reduction
        return gxs

class SumVariadic(VariadicBase):
    def __init__(self):
        super().__init__()
        self.func = SumVariadicCore()

def sum_variadic(*xs):
    return SumVariadic()(*xs)


class MaxMin(Function):
    """ MaxとMinの共通ベース """
    def __init__(self, axis=None, keepdims=False):
        super().__init__()
        self.axis = axis
        self.keepdims = keepdims

    def condition(self, x, y, axis):
        """ 抽出されたところを示す配列を作る、併せて逆伝播に必要なものを揃える """
        self.x_shape = x.shape
        self.y_shape = y.shape
        if axis is None:
            axis = range(x.ndim)
        elif isinstance(axis, int):
            axis = axis,                      
        self.z_shape = [1 if i in axis else s for i, s in enumerate(x.shape)] #xと同次元数のyの形
        #self.cond = x == y.reshape(self.z_shape) # xとyの比較
        self.cond = x == snp.reshape(y, self.z_shape) # xとyの比較

    def __forward__(self, *args, **kwargs):
        raise NotImplementedError()

    def __backward__(self, gy):
        """ 逆伝播は共通 """
        gy = gy if isinstance(gy, np.ndarray) else np.array(gy, dtype=Config.dtype) 
        gy = snp.broadcast_to(gy, self.y_shape)  # 先ずはyの形状に合わせる
        gy = snp.reshape(gy, self.z_shape)       # 次にxに次元を揃える(keepdimsの形状)
        gx = gy * self.cond                      # yに抽出されたところにgyを入れる
        return gx

class Max(MaxMin):
    """ 最大値を抽出 """
    def __forward__(self, x):
        y = snp.max(x, axis=self.axis, keepdims=self.keepdims)
        self.condition(x, y, self.axis)
        return y

class Min(MaxMin):
    """ 最小値を抽出 """
    def __forward__(self, x):
        y = snp.min(x, axis=self.axis, keepdims=self.keepdims)
        self.condition(x, y, self.axis)
        return y

     
class GetItem(Function):
    """ 要素を添字指定により部分取り出しする """
    def __init__(self, slices):
        super().__init__()
        self.slices = slices

    def __forward__(self, x):
        self.x = x
        return x[self.slices]

    def __backward__(self, gy):
        x = self.x
        gx = np.zeros_like(x, dtype=Config.dtype)
        snp.add_at(gx, self.slices, gy)
        return gx

def getitem(x, slices):
    f = GetItem(slices)
    return f(x)


class TopKprimitive(Function):
    """
    指定軸axisに沿って、大きい順に上位k個の値を返す


    TopKの順伝播は、逆伝播の対象外であるindicesも返すため、
    primitiveではindicesを返り値から除外し、valuesのみを返す

    """
    def __init__(self, k, axis=-1):
        super().__init__()
        if k < 1:
            raise ValueError(f"k must be at least 1, but got {k}")
        self.k = k
        self.axis = axis
        self.indices = None

    def __forward__(self, x):
        self.x = x
        axis = self.axis

        # 軸axisに沿って降順でxのインデクスを並べ先頭のk個を選ぶ
        indices = snp.argsort(x, axis=axis)
        indices = np.flip(indices, axis=axis)
        indices = np.take(indices, np.arange(self.k), axis=axis)

        # indicesとaxisに従って値を選ぶ
        values = np.take_along_axis(x, indices, axis=axis)
        self.indices = indices
        return values

    def __backward__(self, gy):
        x = self.x
        gx = scatter_add_along_axis(
            gy, self.indices, output_shape=x.shape, axis=self.axis,
        )
        return gx

class TopK:
    """
    指定軸axisに沿って、大きい順に上位k個の値とindicesを返す

    逆伝播の対象となるvaluesのみを返すprimitiveをラッパーで包み、
    逆伝播の対象外であるindicesを加えて提供する

    """
    def __init__(self, k, axis=-1):
        super().__init__()
        self.primitive = TopKprimitive(k, axis)

    def forward(self, x):
        y = self.primitive(x)
        indices = self.primitive.indices # indicesは逆伝播非対象
        return y, indices

    def __call__(self, x):
        return self.forward(x)

    def backward(self, gy=1):
        return self.primitive.backward(gy)

def top_k(x, k, axis=-1):
    return TopK(k, axis)(x)


class Transpose_bkup(Function):
    def __init__(self, axes=(1, 0)): # numpyのおかしな挙動に対応20241122
        super().__init__()
        if len(axes)==1:
            self.axes, = axes
        else:    
            self.axes = axes

    def __forward__(self, x):
        y = snp.transpose(x, self.axes)
        return y

    def __backward__(self, gy):
        axes = snp.argsort(np.array(self.axes)) # cupy対応
        gx = snp.transpose(gy, axes.tolist())   # cupy対応 
        return gx

class Transpose(Function):
    def __init__(self, *axes): # axesがタプルでなくても対応
        super().__init__()
        #print('###', axes)
        if axes is None:
            self.axes = (1, 0)
        elif len(axes)==0:
            self.axes = (1, 0)
        elif len(axes) > 1:    
            self.axes = axes
        else: # タプルの中にタプル
            self.axes, = axes

    def __forward__(self, x):
        y = snp.transpose(x, self.axes)
        return y

    def __backward__(self, gy):
        axes = snp.argsort(np.array(self.axes)) # cupy対応
        gx = snp.transpose(gy, axes.tolist())   # cupy対応 
        return gx

def transpose(x, *axes):
    return Transpose(*axes)(x)

def transpose_bkup(x, axes=(1, 0)):
    return Transpose(axes)(x)

class Reshape_bkup(Function):
    def __init__(self, *shape):
        super().__init__()
        if len(shape) > 1:
            self.shape = shape
        else:
            self.shape, = shape # タプルにする
        
    def __forward__(self, x):
        self.x = x
        y = snp.reshape(x, self.shape)
        return y

    def __backward__(self, gy):
        x = self.x
        gx = snp.reshape(gy, np.shape(x))
        return gx

class Reshape(Function):
    def __init__(self, *shape):
        super().__init__()

        # 正規化：常に tuple[int,...] にする
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            self.shape = tuple(shape[0])
        else:
            self.shape = shape

    def __forward__(self, x):
        self.x = x
        return snp.reshape(x, self.shape)

    def __backward__(self, gy):
        x = self.x
        return snp.reshape(gy, np.shape(x))

def reshape(x, *shape):
    return Reshape(*shape)(x)

class Dot(Function):
    def __forward__(self, x0, x1):
        self.x0, self.x1 = x0, x1
        y = np.dot(x0, x1)
        return y

    def __backward__(self, gy):
        x0, x1 = self.x0, self.x1
        gx0 = np.dot(gy, x1.T)
        gx1 = np.dot(x0.T, gy)
        return gx0, gx1
     
def dot(x0, x1):
    return Dot()(x0, x1)

class MatMul(Function):
    def __forward__(self, x0, x1):
        self.x0, self.x1 = x0, x1
        y = np.matmul(x0, x1)
        return y

    def __backward__(self, gy):
        x0, x1 = self.x0, self.x1
        #x0T = x0.T if x0.ndim <= 2 else x0.transpose(*range(x0.ndim)[:-2], -1, -2)
        #x1T = x1.T if x1.ndim <= 2 else x1.transpose(*range(x1.ndim)[:-2], -1, -2)
        x0T = x0.T if x0.ndim <= 2 else snp.transpose(x0, (*range(x0.ndim)[:-2], -1, -2))
        x1T = x1.T if x1.ndim <= 2 else snp.transpose(x1, (*range(x1.ndim)[:-2], -1, -2))
        gx0 = np.matmul(gy, x1T)
        gx1 = np.matmul(x0T, gy)
        return gx0, gx1
     
def matmul(x0, x1):
    return MatMul()(x0, x1)

class DotLinear(Function):
    """ ニューラルネットワークで使う基本の重み付け和 """
    def __init__(self, bias=True):
        super().__init__()
        self.bias = bias
        
    def __forward__(self, x, w, b):
        self.x, self.w, self.b = x, w, b
        y = np.dot(x, w)
        if self.bias:
            y += b
        return y

    def __backward__(self, gy):
        x, w, b = self.x, self.w, self.b
        gx = dot(gy, w.T)
        gw = dot(x.T, gy)
        if self.bias:
            gb = snp.sum(gy, axis=0)
        else:
            gb = None
        return gx, gw, gb

    def __backward__bkup(self, gy):
        # ニューラルネットワークで使うことに限れば不要
        x, w, b = self.x, self.w, self.b
        w_T = transpose(w) 
        x_T = transpose(x)  
        gx = np.dot(gy, w_T)
        gw = np.dot(x_T, gy)
        if self.bias and gy.shape==b.shape:
            gb = gy
        elif self.bias:
            gb = sum_to(gy, b.shape)
        else:
            gb = None
        return gx, gw, gb

class HadamardLinear(Function):
    def __forward__(self, x, w, b):
        self.x, self.w, self.b = x, w, b
        y = x * w + b
        return y

    def __backward__(self, gy):
        x, w, b = self.x, self.w, self.b
        gx = gy * w
        gw = gy * x
        gb = gy
        return gx, gw, gb

    def __backward__bkup(self, gy):
        # ニューラルネットワークで使うことに限れば不要
        x, w, b = self.x, self.w, self.b
        gx = gy * w
        gw = gy * x
        gb = gy if gy.shape==b.shape else SumTo(b.shape)(gy)
        return gx, gw, gb
     
class MatMulLinear(Function):
    """ ニューラルネットワークで使う時系列データなど入力次元数3の重み付け和 """
    def __init__(self, bias=True):
        super().__init__()
        self.bias = bias
        
    def __forward__(self, x, w, b):
        self.x, self.w, self.b = x, w, b
        y = snp.matmul(x, w)
        if self.bias:
            y += b
        return y

    def __backward__(self, gy):
        x, w, b = self.x, self.w, self.b
        #x_T = x.T if x.ndim <= 2 else x.reshape(-1, x.shape[-1]).T
        x_T = x.T if x.ndim <= 2 else snp.reshape(x, (-1, x.shape[-1])).T
        gx = snp.matmul(gy, w.T)
        #gyf = gy.reshape(-1, gy.shape[-1])
        gyf = snp.reshape(gy, (-1, gy.shape[-1]))
        gw = np.dot(x_T, gyf)
        if self.bias:
            gb = snp.sum(gyf, axis=0)
        else:
            gb = None
        return gx, gw, gb

class MatMulLinear_bkup(Function):
    """ ニューラルネットワークで使う時系列データなど入力次元数大の重み付け和 """
    def __init__(self, bias=True):
        super().__init__()
        self.bias = bias
        
    def __forward__(self, x, w, b):
        self.x, self.w, self.b = x, w, b
        y = np.matmul(x, w)
        if self.bias:
            y += b
        return y

    def __backward__(self, gy):
        # wやbの次元数
        x, w, b = self.x, self.w, self.b
        x_T = x.T if x.ndim <= 2 else snp.transpose(x, (*range(x.ndim)[:-2], -1, -2))
        gx = np.matmul(gy, w.T)
        gw = np.matmul(x_T, gy)
        if self.bias:
            gb = snp.sum(gy, axis=0)
        else:
            gb = None
        return gx, gw, gb

class DualDotLinear(Function):
    """ ニューラルネットワークで使う２重の重み付け和 """
    def __init__(self, bias=True):
        super().__init__()
        self.bias = bias
        
    def __forward__(self, x, r, w, v, b):
        self.x, self.r, self.w, self.v, self.b = x, r, w, v, b
        y = np.dot(x, w) + np.dot(r, v) 
        if self.bias:
            y += b
        return y

    def __backward__(self, gy):
        x, r, w, v, b = self.x, self.r, self.w, self.v, self.b
        gx = np.dot(gy, w.T)
        gr = np.dot(gy, v.T)
        gw = np.dot(x.T, gy)
        gv = np.dot(r.T, gy)
        if self.bias:
            gb = snp.sum(gy, axis=0)
        else:
            gb = None
        return gx, gr, gw, gv, gb

class ScaleDotLinear(Function):
    """ ニューラルネットワークで使う基本の重み付け和、ReParameterization対応 """
    def __init__(self, matmul=False, bias=True, scale=False, eps=1e-8):
        super().__init__()
        self.matmul = matmul 
        self.bias = bias
        self.scale = scale
        self.dot = snp.matmul if matmul else np.dot # snp.matmulはNLCPyでも高速
        self.eps = eps
        
    def __forward__(self, x, w, b, g=1.0):
        self.x, self.w, self.b, self.g = x, w, b, g
        y = self.dot(x, w)
        if self.scale:
            y *= g 
        if self.bias:
            y += b
        self.y = y
        return y

    def __backward__(self, gy):
        x, w, b, g = self.x, self.w, self.b, self.g
        y = self.y
        #x_T = x.T if x.ndim <= 2 else x.reshape(-1, x.shape[-1]).T
        x_T = x.T if x.ndim <= 2 else snp.reshape(x, (-1, x.shape[-1])).T
        #gyf = gy.reshape(-1, gy.shape[-1])
        gyf = snp.reshape(gy, (-1, gy.shape[-1]))
        
        gx = self.dot(gy, w.T)
        gw = self.dot(x_T, gyf)
        if self.scale:
            gx *= g
            gw *= g
        
        if self.bias:
            gb = snp.sum(gyf, axis=0)
        else:
            gb = None

        if self.scale:
            if np.abs(g) > self.eps: # 通常
                z = y - b if self.bias else y # z=g*(x@w)
                gg = snp.sum(z * gy) / g       # gg=Σgy*(x@w)
            else:                    # gが0に近いときだけ再計算
                xw = self.dot(x, w)                    
                gg = snp.sum(gy * xw)
        else:
            gg = None
        return gx, gw, gb, gg

class Flatten(Function):
    """ 軸0はバッチとし、それ以下の軸を平坦化 """
    def __forward__(self, x):
        self.x_shape = x.shape
        #return x.reshape(self.x_shape[0], -1)
        return snp.reshape(x, (self.x_shape[0], -1))

    def __backward__(self, gy):
        #return gy.reshape(*self.x_shape)
        return snp.reshape(gy, self.x_shape)
        
class Normalize(Function):
    """ 平均0標準偏差1にする標準化(正規化の一種) """
    def __init__(self, axis=None, eps=1e-12):
        super().__init__()
        self.axis = axis
        self.eps = eps
        self.sigma = None
        self.mask = None
    
    def __forward__(self, x):
        self.x = x
        mu = snp.mean(x, axis=self.axis, keepdims=True)
        sigma = snp.std(x, axis=self.axis, keepdims=True)
        z = x - mu
        y = z / (sigma + self.eps)
        self.sigma = sigma
        self.mask = sigma < self.eps # sigmaが極小値の場合には正規化しない
        self.n = x.size//sigma.size  # 畳んだ大きさ 
        y = y * (1 - self.mask) + x * self.mask
        self.y = y
        return y
   
    def __backward__(self, gy):
        x = self.x
        y = self.y
        sigma = self.sigma + self.eps
        n = self.n
        # y = z / sigma の逆伝播 
        gz = gy / sigma
        gsigma = -gy * y / sigma   # -gy * z / sigma ** 2
        #gsigma = snp.sum(gsigma, axis=self.axis, keepdims=True)
        gsigma = sum(gsigma, axis=self.axis, keepdims=True)
        # z = x - mu の逆伝播
        gx0 = gz
        #gmu = snp.sum(-gz, axis=self.axis, keepdims=True)
        gmu = sum(-gz, axis=self.axis, keepdims=True)
        # mu = snp.mean(x) の逆伝播
        gx1 = snp.broadcast_to(gmu, x.shape) / n
        #gx1 = broadcast_to(gmu, x.shape) / n
        # sigma = snp.std(x) の逆伝播
        gx2 = gsigma * y / n   # dvar * dvar_gx = (gsigma * 0.5 / sigma) * ((2/n) * (x - mu))
        gx = gx0 + gx1 + gx2
        return gx * (1 - self.mask) + gy * self.mask

class NormalizeSimple(Function):
    """ 平均0標準偏差1にする標準化(正規化の一種) """
    def __init__(self, axis=None, eps=1e-12):
        super().__init__()
        self.axis = axis
        self.eps = eps
    
    def __forward__(self, x):
        self.x = x
        mu = snp.mean(x, axis=self.axis, keepdims=True)
        sigma = snp.std(x, axis=self.axis, keepdims=True)
        z = x - mu
        y = z / (sigma + self.eps)
        self.sigma = sigma
        self.y = y
        return y
   
    def __backward__(self, gy):
        x = self.x
        y = self.y
        sigma = self.sigma + self.eps
        n = x.size//sigma.size   # 畳んだ大きさ
        # y = z / sigma の逆伝播 
        gz = gy / sigma
        gsigma = -gy * y / sigma   # -gy * z / sigma ** 2
        gsigma = snp.sum(gsigma, axis=self.axis, keepdims=True)
        # z = x - mu の逆伝播
        gx0 = gz
        gmu = snp.sum(-gz, axis=self.axis, keepdims=True)
        # mu = snp.mean(x) の逆伝播
        gx1 = snp.broadcast_to(gmu, x.shape) / n
        # sigma = snp.std(x) の逆伝播
        gx2 = gsigma * y / n   # dvar * dvar_gx = (gsigma * 0.5 / sigma) * ((2/n) * (x - mu))
        return gx0 + gx1 + gx2

class Normalize_bkup(Function):
    """ 平均0標準偏差1にする標準化(正規化の一種) """
    def __init__(self, axis=None):
        super().__init__()
        self.mean = Mean(axis, keepdims=True)
        self.std  = Std(axis, keepdims=True)
        self.sub  = Sub()
        self.div  = Div()
    
    def __forward__(self, x):
        mu  = self.mean(x)
        std = self.std(x)
        z = self.sub(x, mu)
        y = self.div(z, std)
        return y
   
    def __backward__(self, gy):
        gz, gstd = self.div.backward(gy)
        gx0, gmu = self.sub.backward(gz)
        gx1 = self.mean.backward(gmu)
        gx2 = self.std.backward(gstd)
        gx = gx0 + gx1 + gx2
        return gx

class Standardize(Normalize):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class L2Normalize(Function):
    """ L2ノーマライゼーション """
    def __init__(self, axis=None):
        super().__init__()
        self.root_sum_square = RootSumSquare(axis=axis, keepdims=True)
        self.div = Div()
    
    def __forward__(self, x):
        l2n = self.root_sum_square.forward(x)
        y = self.div.forward(x, l2n)
        return y
   
    def __backward__(self, gy):
        gx0, gl2n = self.div.backward(gy)
        gx1 = self.root_sum_square.backward(gl2n)
        gx = gx0 + gx1
        return gx

class Normalize_bkup(Function):
    """ 平均0標準偏差1にする標準化(正規化の一種) """
    def __init__(self, axis=None):
        super().__init__()
        self.axis = axis
        
    def __forward__(self, x):
        self.x = x
        mu =  snp.mean(x, axis=self.axis, keepdims=True)
        std = snp.std(x, axis=self.axis, keepdims=True)
        self.mu   = mu
        self.std  = std
        y = (x - mu) / (std + 1e-12)
        self.y = y
        return y
    
    def __backward__(self, gy):
        istd = 1/self.std
        iN = self.mu.size / self.x.size # muおよびstdを求める際に畳んだ大きさ
        xc = self.x - self.mu
        gy_sum = snp.sum(gy * xc, axis=self.axis, keepdims=True)
        gz = (gy - (self.y * gy_sum * istd * iN)) * istd
        gz_sum = snp.sum(gz, axis=self.axis, keepdims=True)
        gx = gz - (gz_sum * iN)
        return gx

class L2Normalize_bkup(Function):
    """ L2ノーマライゼーション """
    def __init__(self, axis=None):
        super().__init__()
        self.axis = axis
    
    def __forward__(self, x):
        x = np.array(x)
        l2n = snp.sum(x**2, axis=self.axis, keepdims=True)**0.5
        y = x / l2n
        self.x = x
        self.l2n = l2n
        return y
   
    def __backward__(self, gy):
        x = self.x
        l2n = self.l2n
        gx = gy * (1 - x * x.sum(axis=self.axis, keepdims=True) / l2n**2) / l2n
        return gx


class Concatenate(Function):
    """ 複数の入力を、既存の指定軸に沿って結合する """
    def __init__(self, axis=0):
        super().__init__()
        self.axis = axis

    def __forward__(self, *xs):
        self.xs = xs
        y = snp.concatenate(xs, axis=self.axis)
        self.y_shape = y.shape
        return y

    def __backward__(self, gy):
        axis = self.axis
        if axis < 0:
            axis += len(self.y_shape)

        sections = []
        stop = 0

        for x in self.xs[:-1]:
            stop += x.shape[axis]
            sections.append(stop)

        return tuple(snp.split(gy, sections, axis=axis))

def concatenate(xs, axis=0):
    return Concatenate(axis)(*xs)

class ECat(Concatenate):
    def __init__(self, axis=0):
        super().__init__(axis=axis)
        print('Retain for the time being for compatibility. Use Concatenate.')
    
def ecat(*xs, axis=0):
    return ECat(axis)(*xs)

class Split(Function):
    """入力を、指定軸に沿って複数に分割する"""
    def __init__(self, indices_or_sections, axis=0):
        super().__init__()
        self.indices_or_sections = indices_or_sections
        self.axis = axis

    def __forward__(self, x):
        return snp.split(x, self.indices_or_sections, axis=self.axis)

    def __backward__(self, *gys):
        return snp.concatenate(gys, axis=self.axis)

def split(x, indices_or_sections, axis=0):
    return Split(indices_or_sections, axis)(x)


class Stack(Function):
    """ 複数の同形状入力を、新しい指定軸に沿って積み重ねる """
    def __init__(self, axis=0):
        super().__init__()
        self.axis = axis

    def __forward__(self, *xs):
        return snp.stack(xs, axis=self.axis)

    def __backward__(self, gy):
        return tuple(snp.moveaxis(gy, self.axis, 0))

def stack(xs, axis=0):
    return Stack(axis)(*xs)


class Unstack(Function):
    """ 入力を指定軸に沿って分解し、その軸を除いた複数の出力を返す """
    def __init__(self, axis=0):
        super().__init__()
        self.axis = axis

    def __forward__(self, x):
        return tuple(snp.moveaxis(x, self.axis, 0))

    def __backward__(self, *gys):
        return snp.stack(gys, axis=self.axis)

def unstack(x, axis=0):
    return Unstack(axis)(x)

def _along_axis_index(indices, axis, input_shape, output_shape):
    """
    AlongAxis操作のScatterAdd用index tupleを作る。

    indicesは出力形状へbroadcastする。
    input側でbroadcastされた軸の座標は0とする。
    """
    indices = snp.broadcast_to(indices, output_shape)
    ndim = len(input_shape)
    index = []

    for i, size in enumerate(output_shape):
        if i == axis:
            index.append(indices)

        elif input_shape[i] == 1:
            index.append(0)

        else:
            shape = [1] * ndim
            shape[i] = size

            index.append(np.arange(size).reshape(shape))

    return tuple(index)


class TakeAlongAxis(Function):
    """
    指定軸に沿ってindicesで指定された要素を取り出す
    
    順伝播はnp.take_along_axisそのもの．indicesとaxisは微分対象外
    要注意：axis=Noneの場合、xを一次元化して処理
    """
    def __init__(self, indices, axis=-1):
        super().__init__()
        self.indices = np.asarray(indices)
        self.axis = axis

    def __forward__(self, x):
        self.x = x
        y = np.take_along_axis(x, self.indices, axis=self.axis)
        if self.axis is not None and self.axis < 0:
            self.axis += x.ndim
        return y

    def __backward__(self, gy):
        x = self.x
        gx = np.zeros_like(x, dtype=Config.dtype)

        if self.axis is None: # axis=Noneではxを一次元に展開して処理
            gx_flat = snp.reshape(gx, -1)
            snp.add_at(gx_flat, self.indices, gy)
        else:
            index = _along_axis_index(self.indices, self.axis, x.shape, gy.shape)
            snp.add_at(gx, index, gy)
        return gx

def take_along_axis(x, indices, axis=-1):
    return TakeAlongAxis(indices, axis)(x)

class ScatterAddAlongAxis(Function):
    """
    xをindicesで指定されたaxis上の位置へ加算配置する

    重複するindexには値を累積、TakeAlongAxisの随伴演算
    要注意：整数axisのみを扱う
    """
    def __init__(self, indices, output_shape, axis=-1):
        super().__init__()
        self.indices = np.asarray(indices)
        self.output_shape = tuple(output_shape)
        self.axis = axis

    def __forward__(self, x):
        self.x = x
        y = np.zeros(self.output_shape, dtype=Config.dtype)

        if x.ndim != y.ndim:
            raise ValueError(
                "x and output must have the same number of dimensions"
            )

        axis = self.axis
        if axis < 0:
            axis += x.ndim
        if axis < 0 or axis >= x.ndim:
            raise ValueError(
                f"axis {self.axis} is out of bounds for array "
                f"of dimension {x.ndim}"
            )

        for i, size in enumerate(x.shape):
            if i != axis and y.shape[i] not in (1, size):
                raise ValueError(
                    "x and output shapes must match outside axis, "
                    "except where the output size is 1"
                )

        index = _along_axis_index(self.indices, axis, y.shape, x.shape)
        snp.add_at(y, index, x)

        self.axis = axis
        return y

    def __backward__(self, gy):
        x = self.x
        indices = snp.broadcast_to(self.indices, x.shape)
        gx = np.take_along_axis(gy, indices, axis=self.axis)
        return gx


def scatter_add_along_axis(x, indices, output_shape, axis=-1):
    return ScatterAddAlongAxis(indices, output_shape, axis)(x)



############################################ 
# 仮実装　0240812
# MultiHeadAttentionでbacktraceできるようにするために
# その仮対処
############################################
class Tile(Function):
    def __init__(self, reps):
        super().__init__()
        self.reps = reps  # 繰り返し数を指定するタプルまたは整数。
        self.x_ndim = None
        self.x_shape = None

    def __forward__(self, x):
        y = snp.tile(x, self.reps)
        self.x_ndim = x.ndim
        self.x_shape = x.shape
        return y

    def __backward__(self, gy):
        #gx = gy.reshape(self.x_shape + self.reps) # 元の形に戻すため
        gx = snp.reshape(gy, (self.x_shape + self.reps)) # 元の形に戻すため
        #gx = f.sum_to(gy, self.x_shape)          # 以下は、この動作と同じ
        for ax in self.reps:                      # repsの軸ごとに
            gx = snp.sum(gx, axis=self.x_ndim) 
        return gx

def tile(x, reps):
    return Tile(reps)(x)

class Pairwise(Function):
    """ 指定された軸に従ってペアを作る（自己ペアのマスクも可能）"""
    def __init__(self, axis=-1, broadcast=True, diagonal_mask=False):
        super().__init__()
        self.axis = axis
        self.broadcast = broadcast
        self.mask = diagonal_mask
        self.ne = None

    def __forward__(self, x):
        axis = self.axis % x.ndim
        p = snp.expand_dims(x, axis)
        q = snp.expand_dims(x, axis + 1)

        if self.broadcast:
            p, q = snp.broadcast_arrays(p, q)

        if self.mask:
            sz = x.shape[axis]
            eye = np.eye(sz, dtype=bool)
            mask = ~eye
            shape = [1] * p.ndim
            shape[axis] = sz
            shape[axis + 1] = sz
            self.ne = snp.reshape(mask, shape)

            p = snp.where(self.ne, p, 0)
            q = snp.where(self.ne, q, 0)

        return p, q

    def __backward__(self, gp, gq):
        if self.mask and self.ne is not None:
            gp = snp.where(self.ne, gp, 0)
            gq = snp.where(self.ne, gq, 0)

        ndim = gp.ndim
        axis = self.axis % (ndim - 1)

        gxp = snp.sum(gp, axis=axis)
        gxq = snp.sum(gq, axis=axis + 1)
        gx = gxp + gxq
        return gx

class Pairwise_bkup(Function):
    """ 指定された軸に従ってペアを作る """
    def __init__(self, axis=-1, broadcast=True, diagonal_mask=True):
        super().__init__()
        self.axis = axis # 後ろ(-1)から数えてペアを作る軸を指定
        self.broadcast = broadcast
        self.mask = diagonal_mask

    def __forward__(self, x):
        p = snp.expand_dims(x, self.axis)
        q = snp.expand_dims(x, self.axis-1)
        if self.broadcast:
            p, q = snp.broadcast_arrays(p, q)
        if self.mask:
            self.ne = ~np.eye(x.shape[self.axis], dtype=bool)
            p = p * self.ne
            q = q * self.ne
        return p, q

    def __backward__(self, gp, gq):
        if self.mask:
            gp = gp * self.ne
            gq = gq * self.ne
        gxp = snp.sum(gp, axis=self.axis)
        gxq = snp.sum(gq, axis=self.axis-1)
        gx = gxp + gxq
        return gx

class UpperTriangle(Function):
    def __init__(self, k=0):
        super().__init__()
        """ 末尾2軸の正方行列の上三角行列 """
        self.k = k       # 対角線からのオフセット
        self.mask = None # 

    def __forward__(self, x):
        self.x = x
        if x.shape[-1] != x.shape[-2]:
            raise ValueError("末尾2軸が正方行列である必要があります")
        N = x.shape[-1]
        triu_rows, triu_cols = snp.triu_indices(N, k=self.k)
        self.mask = np.zeros((N, N), dtype=bool)
        self.mask[triu_rows, triu_cols] = True 
        y = x[..., self.mask]
        return y

    def __backward__(self, gy):
        x = self.x
        N = x.shape[-1]
        gx = np.zeros(x.shape, dtype=gy.dtype)
        gx[..., self.mask] = gy
        return gx

class Take(Function):
    def __init__(self, axis, indices):
        super().__init__()
        self.axis = axis
        self.indices = np.array(indices)

    def __forward__(self, x):
        self.x = x
        y = snp.take(x, self.indices, axis=self.axis)
        return y

    def __backward__(self, gy):
        x = self.x
        axis = self.axis
        indices = self.indices.flatten()
        gx = np.zeros_like(x, dtype=Config.dtype)
        # 対称軸を平坦化
        #gy_trans = gy.reshape(*x.shape[:axis], -1, *x.shape[axis+1:])
        gy_trans = snp.reshape(gy, (*x.shape[:axis], -1, *x.shape[axis+1:]))
        
        # 対象軸を一番前に持ってきながら、勾配をindicesの位置に設定
        gx = np.zeros_like(x, dtype=Config.dtype)
        snp.add_at(snp.moveaxis(gx, axis, 0),
                   indices,
                   snp.moveaxis(gy_trans, axis, 0))
        return gx

class Permutations(Function):
    " 多次元配列からaxisの指定する軸で順列を作る "
    def __init__(self, axis, n=None, r=2):
        super().__init__()
        self.axis = axis
        self.r = r
        self.take = None
        if n is not None:
            self.fix_configuration(n)

    def fix_configuration(self, n):    
        indices = itertools.permutations(range(n), self.r)
        self.take = Take(self.axis, np.array(list(indices)))

    def __forward__(self, x):
        if self.take is None:
            self.fix_configuration(x.shape[self.axis])
        return self.take(x)

    def __backward__(self, gy):
        return self.take.backward(gy)

class Combinations(Function):
    " 多次元配列からaxisの指定する軸で組合せを作る "
    def __init__(self, axis, n=None, r=2):
        super().__init__()
        self.axis = axis
        self.r = r
        self.take = None
        if n is not None:
            self.fix_configuration(n)

    def fix_configuration(self, n):    
        indices = itertools.combinations(range(n), self.r)
        self.take = Take(self.axis, np.array(list(indices)))

    def __forward__(self, x):
        if self.take is None:
            self.fix_configuration(x.shape[self.axis])
        return self.take(x)

    def __backward__(self, gy):
        return self.take.backward(gy)

class TakePair(Function):
    " 多次元配列からaxisの指定する軸で順列組合わせのペアを作る "
    def __init__(self, axis, method='permutation', n=None):
        super().__init__()
        self.axis = axis # 仮設定しfix_configurationで正規化して再設定　
        self.method = method
        self.take = None
        if n is not None:
            self.fix_configuration(n)

    def fix_configuration(self, shape):
        self.axis = self.axis % len(shape) # 軸の正規化
        n = shape[self.axis]
        if   self.method[:4] == 'perm':
            indices = itertools.permutations(range(n), 2)
        elif self.method[:4] == 'comb':
            indices = itertools.combinations(range(n), 2)
        else:
            raise Exception('Bad method.')
        indices = np.array(list(indices))
        self.take = Take(self.axis, indices)

    def __forward__(self, x):
        if self.take is None:
            self.fix_configuration(x.shape)
        y = self.take(x)
        y = snp.moveaxis(y, self.axis+1, 0) 
        return y[0], y[1]
    
    def __backward__(self, gy0, gy1):
        gy = snp.stack([gy0, gy1], axis=self.axis+1)       
        gx = self.take.backward(gy)
        return gx


class TakePair2(Function):
    " 多次元配列からaxisの指定する軸で順列組合わせのペアを作る "
    def __init__(self, axis, method='permutation', n=None):
        super().__init__()
        self.axis = axis # 仮設定しfix_configurationで正規化して再設定
        self.method = method
        self.take1 = None
        self.take2 = None
        if n is not None:
            self.fix_configuration(n)

    def fix_configuration(self, shape):    
        self.axis = self.axis % len(shape) # 軸の正規化
        n = shape[self.axis]
        if   self.method[:4] == 'perm':
            indices = itertools.permutations(range(n), 2)
        elif self.method[:4] == 'comb':
            indices = itertools.combinations(range(n), 2)
        else:
            raise Exception('Bad method.')
        indices = np.array(list(indices))
        self.take1 = Take(self.axis, indices[:, 0])
        self.take2 = Take(self.axis, indices[:, 1])

    def __forward__(self, x):
        if self.take1 is None or self.take2 is None:
            self.fix_configuration(x.shape)
        return self.take1(x), self.take2(x)

    def __backward__(self, gy1, gy2):
        gx1 = self.take1.backward(gy1)
        gx2 = self.take2.backward(gy2)
        return gx1 + gx2


############################################ 
# 以下、__forward__のみの定義 
############################################

class Step(Function):
    def __init__(self, c=0):
        super().__init__()
        self.c = c
        
    def __forward__(self, x):
        return snp.where(x <= self.c, 0, 1)

def step(x, c=0):
    return Step(c)(x)

class Equal(Function):
    def __forward__(self, x0, x1):
        return x0 == x1

def equal(x0, x1):
    return Equal()(x0, x1)

class GreaterThan(Function):
    def __forward__(self, x0, x1):
        return x0 > x1

def greater_than(x0, x1):
    return GreaterThan()(x0, x1)

class GreaterThanOrEqual(Function):
    def __forward__(self, x0, x1):
        return x0 >= x1

def greater_than_or_equal(x0, x1):
    return GreaterThanOrEqual()(x0, x1)

class LessThan(Function):
    def __forward__(self, x0, x1):
        return x0 < x1

def less_than(x0, x1):
    return LessThan()(x0, x1)

class LessThanOrEqual(Function):
    def __forward__(self, x0, x1):
        return x0 <= x1

def less_than_or_equal(x0, x1):
    return LessThanOrEqual()(x0, x1)

class Argmax(Function):
    def __init__(self, axis=None, keepdims=False):
        super().__init__()
        self.axis = axis
        self.keepdims = keepdims

    def __forward__(self, x):
        return snp.argmax(x, axis=self.axis, keepdims=self.keepdims)

class Argmin(Function):
    def __init__(self, axis=None, keepdims=False):
        super().__init__()
        self.axis = axis
        self.keepdims = keepdims

    def __forward__(self, x):
        return snp.argmin(x, axis=self.axis, keepdims=self.keepdims)

class Argsort(Function):
    def __init__(self, axis=None):
        super().__init__()
        self.axis = axis

    def __forward__(self, x):
        return snp.argsort(x, axis=self.axis)

############################################ 
# 以下、Activatorsから仮移植 
############################################


class Sigmoid(Function):
    def __forward__(self, x):
        y = 1 / (1 + np.exp(-x))
        self.y = y
        return y

    def __backward__(self, gy):
        y = self.y         
        gx = y * (1 - y) * gy
        return gx

class Tanh(Function):
    def __forward__(self, x):
        y = np.tanh(x)
        self.y = y
        return y

    def __backward__(self, gy):
        y = self.y
        gx = gy * (1 - y * y)
        return gx

class Softmax(Function):
    def __forward__(self, x):
        max_x = snp.max(x, axis=-1, keepdims=True) #if dimx>1 else snp.max(x)
        exp_a = np.exp(x - max_x)  # オーバーフロー対策
        sum_exp_a = snp.sum(exp_a, axis=-1, keepdims=True) #if dimx>1 else snp.sum(exp_a) 
        y = exp_a / (sum_exp_a + 1e-7)
        self.y = y
        return y

    def __backward__(self, gy): # ソフトマックス本来の逆伝播
        y = self.y
        gx = y * gy
        sumgx = snp.sum(gx, axis=-1, keepdims=True)
        gx -= y * sumgx
        return gx


#######################################################
# OperatorOverload
#######################################################

class OverloadContents:
    def save(self):
        HDArray.original_neg  = HDArray.__neg__
        HDArray.original_add  = HDArray.__add__
        HDArray.original_radd = HDArray.__radd__
        HDArray.original_sub  = HDArray.__sub__
        HDArray.original_rsub = HDArray.__rsub__
        HDArray.original_mul  = HDArray.__mul__
        HDArray.original_rmul = HDArray.__rmul__
        HDArray.original_div  = HDArray.__truediv__
        HDArray.original_rdiv = HDArray.__rtruediv__
        HDArray.original_pow  = HDArray.__pow__
        HDArray.original_rpow = HDArray.__rpow__
        HDArray.original_getitem = HDArray.__getitem__
        HDArray.original_equal = HDArray.__eq__
        HDArray.original_gt    = HDArray.__gt__
        HDArray.original_ge    = HDArray.__gt__
        HDArray.original_lt    = HDArray.__lt__
        HDArray.original_le    = HDArray.__le__

        HDArray.original_reshape = HDArray.reshape
        HDArray.original_transpose = HDArray.transpose
        HDArray.original_mean = HDArray.mean
        HDArray.original_sum  = HDArray.sum
        HDArray.original_var  = HDArray.var
        HDArray.original_std  = HDArray.std
        

    def overload(self):
        HDArray.__neg__  = neg 
        HDArray.__add__  = add 
        HDArray.__radd__ = add 
        HDArray.__sub__  = sub 
        HDArray.__rsub__ = rsub 
        HDArray.__mul__  = mul
        HDArray.__rmul__ = mul
        HDArray.__truediv__   = div 
        HDArray.__rtruediv__  = rdiv 
        HDArray.__pow__  = pow 
        HDArray.__rpow__ = exp
        HDArray.__getitem__  = getitem 
        HDArray.__eq__   = equal 
        HDArray.__gt__  = greater_than
        HDArray.__ge__  = greater_than_or_equal
        HDArray.__lt__  = less_than
        HDArray.__le__  = less_than_or_equal

        HDArray.reshape = reshape
        HDArray.transpose = transpose
        HDArray.mean = mean
        HDArray.sum  = sum
        HDArray.var  = var
        HDArray.std  = std

    def recover(self):
        HDArray.__neg__  = HDArray.original_neg 
        HDArray.__add__  = HDArray.original_add 
        HDArray.__radd__ = HDArray.original_add
        HDArray.__sub__  = HDArray.original_sub
        HDArray.__rsub__ = HDArray.original_rsub
        HDArray.__mul__  = HDArray.original_mul
        HDArray.__rmul__ = HDArray.original_mul 
        HDArray.__truediv__  = HDArray.original_div
        HDArray.__rtruediv__ = HDArray.original_rdiv
        HDArray.__pow__  = HDArray.original_pow
        HDArray.__rpow__ = HDArray.original_rpow
        HDArray.__getitem__  = HDArray.original_getitem
        HDArray.__eq__   = HDArray.original_equal
        HDArray.__gt__   = HDArray.original_gt
        HDArray.__ge__   = HDArray.original_ge
        HDArray.__lt__   = HDArray.original_lt
        HDArray.__le__   = HDArray.original_le

        HDArray.reshape = HDArray.original_reshape
        HDArray.transpose = HDArray.original_transpose
        HDArray.mean = HDArray.original_mean
        HDArray.sum  = HDArray.original_sum
        HDArray.var  = HDArray.original_var
        HDArray.std  = HDArray.original_std

def operator_overload():
    OperatorOverload().overload()

if __name__=='__main__':
    print('\n#### all cast ####')
    import inspect
    import sys
    current_module = sys.modules[__name__]
    classes = map(lambda x:x[0],inspect.getmembers(current_module,inspect.isclass))
    classes = list(classes)
    print(classes)
    
    #
    from pyaino.nucleus import CompositFunction, HDArray
    import matplotlib.pyplot as plt


    print('基本関数のテスト')
    #set_create_graph('True')

    #"""#
    print('オペランドが１つの関数')
    functions = (Assign, Neg, Abs, Sin, Cos, Square, Sqrt, Exp, Pow, Log, Erf)
    
    x = np.linspace(-4, 4)

    for f in functions:
        func = f()
        print('test ', func.__class__.__name__)
        y = func(x)
        gx = func.backward()
        
        plt.plot(x.tolist(), y.tolist())
        plt.plot(x.tolist(), gx.tolist())
        plt.title(func.__class__.__name__)
        plt.show()

    print('オペランドが１つの関数 拡張')
    functions = (Exp, Pow, Log)
    
    x = np.linspace(-4, 4)
    a = 3.0

    for f in functions:
        func = f(a)
        print('test ', func.__class__.__name__)
        y = func(x)
        gx = func.backward()
        
        plt.plot(x.tolist(), y.tolist())
        plt.plot(x.tolist(), gx.tolist())
        plt.title(func.__class__.__name__)
        plt.show()


    print('オペランドが１つの関数 拡張2')
    functions = (Branch,)
    
    x = np.linspace(-4, 4)

    for f in functions:
        func = f()
        print('test ', func.__class__.__name__)
        y = func(x)
        gy1 = np.ones_like(y)
        gx = func.backward(gy1)
        plt.plot(x.tolist(), y.tolist())
        plt.plot(x.tolist(), gx.tolist())
        gy2 = np.ones_like(y)
        gx = func.backward(gy2, flush=False)
        plt.plot(x.tolist(), gx.tolist())
        plt.title(func.__class__.__name__)
        plt.show()


    print('オペランドが2つの関数')
    functions = (Add, Sub, Mul, Div)
    x0 = np.linspace(-4, 4) 
    x1 = np.linspace(4, -4)
    
    for f in functions:
        func = f()
        print('test ', func.__class__.__name__)
        y = func(x0, x1)
        gx0, gx1 = func.backward()

        plt.plot(x0.tolist(), label='x0')
        plt.plot(x1.tolist(), label='x1')
        plt.plot(y.tolist(), label='y')
        plt.plot(gx0.tolist(), label='gx0')
        plt.plot(gx1.tolist(), label='gx1')
        plt.legend()
        plt.title(func.__class__.__name__)
        plt.show()

    print('オペランドが複数の関数')
    functions = (SumVariadic,)
    xs = []
    xs.append(np.linspace(0, 1))
    xs.append(np.linspace(1, 0))
    xs.append(np.linspace(-1, 1))
    #xs =(np.full((50,), i+1) for i in range(3))    
    for f in functions:
        func = f()
        print('test ', func.__class__.__name__)
        y = func(*xs)
        #print('xs =', xs)
        #print('y =', y)
        gxs = func.backward()
        #print('gxs =', gxs)
        for i, x in enumerate(xs):
            plt.plot(x.tolist(), label='x'+str(i))
        plt.plot(y.tolist(), label='y')
        for i, gx in enumerate(gxs):
            plt.plot(gx.tolist(), label='gx'+str(i))
        plt.legend()
        plt.title(func.__class__.__name__)
        plt.show()

 
       
    #"""#
    #"""#
    print('基本関数の組み合わせのテスト')
    functions = (Normalize, L2Normalize)
    x = np.random.rand(10)

    for f in functions:
        func = f()
        print('test ', func.__class__.__name__)

        y = func.forward(x)
        print(x.shape, y.shape)
        gy = np.arange(0, y.size) #np.random.rand(y.data.size)
        gy = gy[::-1]
        gx = func.backward(gy)

        print(type(x), type(y), type(gy), type(gx))

        plt.plot(x.tolist(), y.tolist())
        plt.plot(x.tolist(), gx.tolist())
        plt.plot(x.tolist(), gy.tolist())
        plt.grid()
        plt.title(func.__class__.__name__)
        plt.show()

    print('基本関数の組み合わせのテスト2 backtrace')
    functions = (Normalize, L2Normalize)
    set_derivative(True)
    
    for f in functions:
        func = f()
        print('test ', func.__class__.__name__)

        y = func.forward(x)
        print(x.shape, y.shape)
        gy = np.arange(0, y.size) #np.random.rand(y.data.size)
        gy = gy[::-1]
        y.backtrace(gy)
        gx = func.inputs[0].grad

        print(type(x), type(y), type(gx))

        plt.plot(x.tolist(), y.tolist())
        plt.plot(x.tolist(), gx.tolist())
        plt.plot(x.tolist(), gy.tolist())
        plt.grid()
        plt.title(func.__class__.__name__)
        plt.show()

    set_derivative(False)

    #"""#
    #"""#
    print('そのほかの関数のテスト')
    functions = (Transpose, Flatten, Max, Min)
    x = np.arange(12).reshape(3,4)
    print(x)
    for f in functions:
        func = f()
        print('test ', func.__class__.__name__)
        y = func(x)
        print(y)
        gx = func.backward()
        print(gx)
    #"""#
    

    #'''#
    print('合成関数の検証(極めて簡単な例を取り上げる)')
    #'''#
    #'''#
    class FuncComposit(CompositFunction):
        def _forward(self, x):
            y = Add()(x, 2)
            return y

    x = np.linspace(-5, 5, 10)

    func = FuncComposit()
    y = func.forward(x)
    gx = func.backward(); print(type(gx))

    plt.plot(x.tolist(), y.tolist())
    plt.plot(x.tolist(), gx.tolist())
    plt.title(func.__class__.__name__)
    plt.show()

    #'''#
    print('合成関数の検証(例としてsigmoid関数を取り上げる)')
    #'''#
    #'''#
    class SigmoidComposit(CompositFunction):
        def _forward(self, x):
            y = Div()(1, Add()(1, Exp()(Neg()(x))))
            return y

    x = np.linspace(-5, 5, 10)

    sigmoid = SigmoidComposit()
    y = sigmoid.forward(x)
    gx = sigmoid.backward(); print(type(gx))

    plt.plot(x.tolist(), y.tolist())
    plt.plot(x.tolist(), gx.tolist())
    plt.title(sigmoid.__class__.__name__)
    plt.show()

    #'''#
    print('合成関数の検証(例としてsigmoid関数を演算子オーバーロードで)')
    #'''#
    #'''#
    class SigmoidComposit(CompositFunction):
        def _forward(self, x):
            y = 1 / (1 + np.e**(-x))            
            return y

    x = np.linspace(-5, 5, 10)

    sigmoid = SigmoidComposit()
    y = sigmoid.forward(x)
    gx = sigmoid.backward(); print(type(gx))

    plt.plot(x.tolist(), y.tolist())
    plt.plot(x.tolist(), gx.tolist())
    plt.title(sigmoid.__class__.__name__)
    plt.show()

    #'''#
    #'''#
    print('合成関数の検証(tan)')
    class TanComposit(CompositFunction):
        def _forward(self, x):
            sinx = Sin()(x)
            cosx = Cos()(x)
            y = Div()(sinx, cosx)
            return y

    x = np.linspace(-1, 1, 10)
    

    tan = TanComposit()
    y = tan.forward(x)
    gx = tan.backward(); print(type(gx))
             
    plt.plot(x.tolist(), y.tolist())
    plt.plot(x.tolist(), gx.tolist())
    plt.title(tan.__class__.__name__)
    plt.show()

    #'''#
    #'''#
    print('合成関数の検証(normalize)')
    class NormalizeComposit(CompositFunction):
        def _forward(self, x, axis=None, keepdims=False):
            mu  = Mean(axis, keepdims)(x)
            var = Var(axis, keepdims)(x)
            std = Sqrt()(var)
            std = Add()(std, 1e-12)
            y   = Div()(Sub()(x, mu), std)
            return y

    x = np.random.rand(10)

    normalize = NormalizeComposit()
    y = normalize.forward(x)
    gy = np.arange(0, y.size) #np.random.rand(y.data.size)
    gy = gy[::-1]
    gx = normalize.backward(gy); print(type(gx))

    plt.plot(x.tolist(), y.tolist())
    plt.plot(x.tolist(), gx.tolist())
    plt.plot(x.tolist(), gy.tolist())
    plt.grid()
    plt.title(normalize.__class__.__name__)
    plt.show()

    #'''#
    print('追加の検証～演算子オーバーロード')

    set_derivative(True)

    x = HDArray(np.linspace(-2, 2, 5))
    print(type(x))

    print('うまく行く')
    y = 2*x**2 + x
    gy = np.ones_like(y) # gyを明示的に与える
    y.backtrace(gy)
   
    plt.plot(x.tolist(), y.tolist(), label="y=f(x)")
    plt.plot(x.tolist(), x.grad.tolist(), label="y'")
    plt.legend()#loc='lower right')
    plt.show()

    print('うまく行く様になった')
    y = 2*x**2 + x
    y.backtrace() # gyを指定しないとうまく行かない->Assignの対処でうまく行く
    
    plt.plot(x.tolist(), y.tolist(), label="y=f(x)")
    plt.plot(x.tolist(), x.grad.tolist(), label="y'")
    plt.legend()#loc='lower right')
    plt.show()

    print('テンソル操作の関数のテスト')
    x = np.arange(24).reshape(2,3,4)
    a = (4,2,3)
    func = Reshape(a)
    print('test ', func.__class__.__name__, x.shape, '->', a)
    y = func(x)  
    gx = func.backward()
    
    print(x)
    print(y)
    print(gx)
    
    func = Reshape(4,2,3)
    print('test ', func.__class__.__name__, x.shape, '->', a)
    y = func(x)  
    gx = func.backward()
    
    print(x)
    print(y)
    print(gx)

    a = (2,0,1)
    func = Transpose(a)
    print('test ', func.__class__.__name__, x.shape, ':', a)
    y = func(x)  
    gx = func.backward()
    
    print(x)
    print(y)
    print(gx)

    func = Transpose(2,0,1)
    print('test ', func.__class__.__name__, x.shape, ':', a)
    y = func(x)  
    gx = func.backward()
    
    print(x)
    print(y)
    print(gx)

    x = x.reshape(6,4) 
    func = Transpose()
    print('test ', func.__class__.__name__, x.shape, ':', a)
    y = func(x)  
    gx = func.backward()
    
    print(x)
    print(y)
    print(gx)

    x0 = np.arange(12).reshape(3, 4)
    x1 = np.arange(12).reshape(4, 3)
    func = Dot()
    print('test ', func.__class__.__name__, x0.shape, x1.shape)
    y = func(x0, x1)
    gx0, gx1 = func.backward()
    
    print(x0)
    print(x1)
    print(y)
    print(gx0)
    print(gx1)
    
    x0 = np.arange(24).reshape(2, 3, 4)
    x1 = np.arange(24).reshape(2, 4, 3)
    func = MatMul()
    print('test ', func.__class__.__name__, x0.shape, x1.shape)
    y = func(x0, x1)
    gx0, gx1 = func.backward()
    
    print(x0)
    print(x1)
    print(y)
    print(gx0)
    print(gx1)
    
    #"""#
    print('そのほかの関数のテスト2')
    functions = (DotLinear, HadamardLinear, MatMulLinear)
    x = np.arange(4).reshape(2,2)
    w = np.arange(4).reshape(2,2)
    b = np.arange(2)
    print(x)
    print(w)
    print(b)
    for f in functions:
        func = f()
        print('test ', func.__class__.__name__)
        y = func(x, w, b)
        print(y)
        gx, gw, gb = func.backward()
        print(gx)
        print(gw)
        print(gb)
    #"""#
    #"""#
    print('そのほかの関数のテスト3')
    functions = (MatMulLinear, MatMulLinear_bkup)
    x = np.arange(24).reshape(2,3,4)
    w = np.arange(8).reshape(4,2)
    b = np.arange(2)
    print(x)
    print(w)
    print(b)
    for f in functions:
        func = f()
        print('test ', func.__class__.__name__)
        y = func(x, w, b)
        print(y)
        gx, gw, gb = func.backward()
        print(gx)
        print(gw)
        print(gb)
    #"""#
    #"""#
    print('そのほかの関数のテスト4')
    functions = (DualDotLinear,)
    x = np.arange(8).reshape(2,4)
    r = np.arange(6).reshape(2,3)
    w = np.arange(12).reshape(4,3)
    v = np.arange(9).reshape(3,3)
    b = np.arange(3)
    print(x)
    print(r)
    print(w)
    print(v)
    print(b)
    for f in functions:
        func = f()
        print('test ', func.__class__.__name__)
        y = func(x, r, w, v, b)
        print(y)
        gx, gr, gw, gv, gb = func.backward()
        print(gx)
        print(gr)
        print(gw)
        print(gv)
        print(gb)
    #"""#
    #"""#
    print('そのほかの関数のテスト3')
    functions = (ScaleDotLinear,)
    x = np.arange(24).reshape(2,3,4)
    w = np.arange(8).reshape(4,2)
    b = np.arange(2)
    g = np.array(2)
    print(x)
    print(w)
    print(b)
    print(g)
    for f in functions:
        func = f(scale=True)
        print('test ', func.__class__.__name__)
        y = func(x, w, b, g)
        print(y)
        gx, gw, gb, gg = func.backward()
        print(gx)
        print(gw)
        print(gb)
        print(gg)
    #"""#
    
    print('そのほかの関数のテスト4')
    func1 = Split(3)
    print('test ', func1.__class__.__name__)
    x = np.arange(3*2*4, dtype=np.float32).reshape(3,2,4)
    ys = func1(x)
    print(ys)
    gx = func1.backward()
    print(gx)
    func2 = Concatenate()
    print('test ', func2.__class__.__name__)
    z = func2(*ys)
    print(z)
    gys = func2.backward()
    print(gys)


    print("reshape   =", snp.reshape, getattr(snp.reshape, "__module__", None))
    print("transpose =", snp.transpose, getattr(snp.transpose, "__module__", None))
    print("broadcast_to =", snp.broadcast_to, getattr(snp.broadcast_to, "__module__", None))
    print("expand_dims  =", snp.expand_dims, getattr(snp.expand_dims, "__module__", None))

