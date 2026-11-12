# Config
# 20260912 A.Inoue
#try:
#    import cupy as np
#except:
#    import numpy as np


class Config:
    np    = None
    #dtype = 'f4'     # getattr(np, 'dtype') -> NG
    dtype = 'float32' # getattr(np, 'dtype') -> OK
    inf = 1e30   # 初めはスカラ、後でset_np()の際に np.array(inf, dtype=Config.dtype)
    seed  = None
    derivative = False
    higher_derivative = False
    create_graph = False
    operator_state = 0
    enable_debug_print = False
    preserve_weakref_obj = False
    log_function = False
    #log_file = 'log_file.txt'
    function_list = []
    backtrace_duration = False # バックトレース中かどうかを明示 20250506AI
    in_forward  = False # Function.__forward__  の実行中のみ True
    in_backward = False # Function.__backward__ の実行中のみ True

def set_inf(value):
    setattr(Config, 'inf', value)
    print('Config.inf is set to', Config.inf)

def set_dtype(value):
    #print('old_value =', getattr(Config, 'dtype'))
    setattr(Config, 'dtype', value)
    print('Config.dtype is set to', Config.dtype)
  
def set_seed(value):
    #print('old_value =', getattr(Config, 'seed'))
    setattr(Config, 'seed', value)    
    np.random.seed(seed=Config.seed)
    print('random.seed', Config.seed, 'is set for', np.__name__)

def set_np(value=None):
    global np
    old_np = getattr(Config, 'np')#; print('>>>', old_np)

    if value is None:
        try:
            import cupy as np
        except:
            import numpy as np
    elif value == 'numpy':
        import numpy as np
    elif value == 'cupy':
        import cupy as np
    elif value == 'nlcpy':      # NEC SX-Aurora TSUBASA (Vector Engine)
        import nlcpy as np
    else:
        raise Exception("Invalid library specified. Specify 'numpy', 'cupy' or 'nlcpy'.")

    if np.__name__ == 'numpy':
        np.seterr(divide='raise') # 割算例外でnanで続行せずに例外処理させる
        #np.seterr(over='raise')

    setattr(Config, 'np', np)
    if old_np is not None and old_np!=np:
        print('Config.np is replaced!', end =' ')
        print(np.__name__, np.__version__, 'is running', np.random.rand(1))
        print("command 'np=Config.np' will set __main__ to use", Config.np.__name__, "as np.")

    inf = np.array(float(Config.inf), dtype=Config.dtype)
    setattr(Config, 'inf', inf)

# 環境変数 PYAINO_NP=numpy|cupy|nlcpy で既定を選べる(未指定ならcupy、無ければnumpy)
import os as _os
set_np(_os.environ.get('PYAINO_NP') or None)

#import numpy as np

print(np.__name__, np.__version__, 'is running in', __file__, np.random.rand(1))
print('Config.dtype =', Config.dtype)
print('Config.seed =', Config.seed)
print('Config.create_graph =', Config.create_graph)
print("If you want to change np, run 'set_np('numpy', 'cupy' or 'nlcpy'); np = Config.np.'")
print("If you want to change Config.dtype, run 'set_dtype('value')'")
print("If you want to set seed for np.random, run set_seed(number)")
print("If you want to set create_graph for Higher Derivative Functions, run set_create_graph(True)")


import contextlib

@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name) 
    setattr(Config, name, value)
    try:
        yield
    finally:    
        setattr(Config, name, old_value)

def debug_print(*text, **end):
    if Config.enable_debug_print:
        print(*text, end=end.get('end'))

def set_create_graph(value=True, HDF=False):
    from pyaino import nucleus
    print('Config.operator_state =', Config.operator_state)    
    if value:
        nucleus.OperatorOverload(True, HDF)()
    else:
        nucleus.OperatorOverload(False, HDF)()
    print('Config.operator_state =', Config.operator_state)    
    setattr(Config, 'create_graph', value)
    print('Config.create_graph is set to', Config.create_graph)

def numpy_overload():
    from pyaino import nucleus
    np.hdarray=nucleus.HDArray
    np.gradient=nucleus.gradient
    print(np.hdarray.__name__, 'is available as np.hdarray')
    print(np.gradient.__name__, 'is available as np.gradient')


def set_derivative(value=True, HDF=False):
    set_create_graph(value, HDF)
    print('Config.derivative old_value', getattr(Config, 'derivative'))
    setattr(Config, 'derivative', value)
    print('Config.derivative is set to', Config.derivative)

    if value:
        numpy_overload()
    else:
        pass
    

def set_higher_derivative(value=True, HDF=True):
    set_create_graph(value, HDF)
    #print('Config.derivative old_value', getattr(Config, 'derivative'))
    setattr(Config, 'derivative', value)
    #print('Config.derivative is set to', Config.derivative)

    print('Config.higher_derivative old_value', getattr(Config, 'higher_derivative'))
    setattr(Config, 'higher_derivative', value)
    #setattr(Config, 'preserve_weakref_obj', value) # 無しで大丈夫 20241031
    print('Config.higher_derivative is set to', Config.higher_derivative)
    print('Config.preserve_weakref_obj =', Config.preserve_weakref_obj)

    if value:
        numpy_overload()
    else:
        pass

    
