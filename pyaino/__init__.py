"""
pyaino: Define-by-run style automatic differentiation framework

"""

__version__ = "0.1.0"


# 公開APIを制限（必要に応じて）
__all__ = (
    'Config',
    'Functions',
    'Activators',
    'Optimizers',
    'Initializer',
    'LossFunctions',
    'Regularizers',
    'common_function',
    'Neuron',
)

# pyaino/__init__.py

# ------------------------------------------------------------
# Current official modules
# ------------------------------------------------------------
# pyaino の現行正式モジュール一覧。
# 日付付きバックアップ、*_bkup、test_* 等は含めない。
# 外部ツールはこの一覧を pyaino の正式構成として参照できる。

CURRENT_MODULES = (
    # core
    'Config',
    'nucleus',
    'Functions',
    'HDFunctions',
    'safe_np',

    # basic neural-network components
    'Activators',
    'Optimizers',
    'Initializer',
    'LossFunctions',
    'Regularizers',
    'common_function',
    'Neuron',

    # model / network structures
    'NN_CNN',
    'RNN',
    'seq2seq',
    'skeletons',
    'stems_blocks_heads',
    'BigramLanguageModel',
    'BigramLanguageModel2',
    'RnnLanguageModel',
    'Diffuser',
    'GAN',
    'VAE',
    'UNet',
    'ResNet',
    'Markov',
    'Qwen',
    'QwenTokenizer',

    # datasets / data utilities
    'data_loader',
    'MNIST',
    'CIFAR10',
    'CIFER10',
    'STL10',
    'sklearn_datasets',
    'sklearn_digits',
    'sklearn_iris',
    'sklearn_OlivettiFaces',

    # tooling / interoperability
    'pyaino_to_ufiesia',
    'onnx_to_pyaino',
    'pyaino_to_onnx',
    'torch_bridge',
    'numpy_min',
    'exe_builder',
)

