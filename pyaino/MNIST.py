# MNIST のデータの操作の関数
# 2025.08.18 井上

from struct import *
import os, time
import matplotlib.pyplot as plt
from pyaino.Config import *
from pyaino import common_function as cf

path = os.path.normpath(os.path.join(os.path.dirname(__file__), '../MNIST/')) + os.sep
#path = 'C:/Python37/lib/site-packages/MNIST/'
                     
def read_image(filename):
    file = open(filename, 'rb') # ファイルオープン
    global data
    data = file.read()
    print('read data file:'+filename)
    file.close()
    headder = unpack('>4l', data[0:16])
    print('file headder is', headder)
    offset = 16
    count = headder[1]
    import numpy          # numpy行列でないとうまくいかない
    V = numpy.zeros((count, 784)) 
    for i in range(count):
        V[i][:] = unpack('>784B', data[offset:offset+784]) 
        offset += 784     # 784バイトが画像
    if np.__name__ in ('cupy', 'nlcpy'):
        V = np.array(V)   # cupy/nlcpy行列に変換
    return V

def read_label(filename):
    file = open(filename, 'rb') # ファイルオープン
    global data
    data = file.read()
    print('read data file:'+filename)
    file.close()
    headder = unpack('>2l', data[0:8])
    print('file headder is', headder)
    offset = 8
    count = headder[1]
    L = np.zeros(count)
    for i in range(count):
        L[i]    = unpack('>B', data[offset:offset+1])[0]
        offset += 1
    return L

def load_data(datapath=path):
    global x_train, t_train, x_test, t_test
    
    x_train = np.zeros((60000, 784), dtype=Config.dtype)
    t_train = np.zeros(60000, dtype=int)
    x_test  = np.zeros((10000, 784), dtype=Config.dtype)
    t_test  = np.zeros(10000, dtype=int)
    
    x_train[0:60000] = read_image(datapath + 'train-images.idx3-ubyte')
    t_train[0:60000] = read_label(datapath + 'train-labels.idx1-ubyte')
    x_test [0:10000] = read_image(datapath + 't10k-images.idx3-ubyte' )
    t_test [0:10000] = read_label(datapath + 't10k-labels.idx1-ubyte')

    x_train = x_train.reshape(60000, 28, 28) 
    t_train = t_train.reshape(60000)
    x_test  = x_test .reshape(10000, 28, 28)
    t_test  = t_test .reshape(10000)
    return (x_train, t_train),(x_test, t_test)

def get_data(**kwargs):
    start = time.time()
    (x_train, t_train),(x_test, t_test) = load_data()
    print('MNISTのデータが読み込まれました')
    n_train = len(x_train)
    n_test  = len(x_test)

    # -- データの形状 --
    ndim = kwargs.pop('ndim', None)
    if ndim is not None:
        if ndim==4:
            shape=(-1, 1, 28, 28)
        elif ndim==3:
            shape=(-1, 28, 28)
        elif ndim==2:
            shape=(-1, 28*28)
        else:
            raise Exception("Wrong specification of 'ndim'")
        x_train = x_train.reshape(shape)
        x_test  = x_test .reshape(shape)

    # -- データの標準化 --
    normalize = kwargs.pop('normalize', None)
    if normalize is not None:
        x_train = cf.normalize(x_train, normalize)
        x_test  = cf.normalize(x_test,  normalize)

    # -- 正解をone-hot表現に --
    y_train = np.zeros((n_train, 10))
    for i in range(n_train):
        y_train[i, t_train[i]] = 1.0
    # -- 正解をone-hot表現に --
    y_test = np.zeros((n_test, 10))
    for i in range(n_test):
        y_test [i, t_test[i]]  = 1.0

    print('訓練用の入力と正解値のデータが用意できました')
    print('入力データの形状', x_train.shape, '正解値の形状', y_train.shape)    
    print('データの素性 最大値 {:6.3f} 最小値 {:6.3f} 平均値 {:6.3f} 分散 {:6.3f}' \
        .format(float(np.max(x_train)), float(np.min(x_train))\
              , float(np.mean(x_train)), float(np.var(x_train))))
    print('評価用の入力と正解値のデータが用意できました')
    print('入力データの形状', x_test.shape, '正解値の形状', y_test.shape)    
    print('データの素性 最大値 {:6.3f} 最小値 {:6.3f} 平均値 {:6.3f} 分散 {:6.3f}' \
        .format(float(np.max(x_test)), float(np.min(x_test))\
              , float(np.mean(x_test)), float(np.var(x_test))))

    elapsed_time = time.time() - start
    print ('elapsed_time:{0}'.format(elapsed_time) + '[sec]')

    return x_train, y_train, t_train, x_test, y_test, t_test

# -- サンプルの表示 --
def show_sample(data, label=None):
    max_picel = np.max(data); min_picel = np.min(data) # 画素データを0～1に補正
    rdata = (data - min_picel)/(max_picel - min_picel)
    rdata = rdata.reshape(28, 28)
    plt.imshow(rdata.tolist(), cmap='gray')
    plt.title(label)
    plt.show()

# -- 複数サンプルを表示(端数にも対応) --
def show_multi_samples(data=None, target=None): # data, targetは対応する複数のもの
    data = np.array(data.tolist()) if data is not None else get_data()[0][:50]
    n_data = len(data)
    n = 50 # 一度に表示する画像数
    for j in range(0, n_data, n):   # はじめのn個、次のn個と進める
        x = data[j:]
        t = target[j:] if target is not None else None
        plt.figure(figsize=(18, 10))
        m = min(n, n_data - j)      # n個以上残っていればn個、n個に満たない時はその端数
        for i in range(m):
            plt.subplot(5, 10, i+1) # 5行10列のi+1番目
            max_picel = np.max(x[i]); min_picel = np.min(x[i])
            rx = (x[i] - min_picel)/(max_picel - min_picel)
            plt.imshow(rx.tolist(), cmap='gray')
            if target is not None:
                plt.title(int(t[i]))
            plt.axis('off')
        plt.show()        


# -- 以下は実行サンプル --
if __name__=='__main__':
    x_train, y_train, t_train, x_test, y_test, t_test = get_data()

    print('-- データの中身を確認 train --')
    while True:
        try:
            i = int(input('見たいデータの番号'))
            pick_data = x_train[i]; pick_label = t_train[i]    
        except:    
            print('-- テスト終了 --')
            break
        show_sample(pick_data, pick_label)
        
    print('-- データの中身を確認 test --')
    while True:
        try:
            i = int(input('見たいデータの番号'))
            pick_data = x_test[i]; pick_label = t_test[i]    
        except:    
            print('-- テスト終了 --')
            break
        show_sample(pick_data, pick_label)

    show_multi_samples(x_train[:50], t_train[:50])
    show_multi_samples(x_test[:50],  t_test[:50])

