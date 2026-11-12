# pyainoについての知識QA (READMEの内容から作成)
# 各事実について 学習用の質問2つ + 評価用(学習に使わない言い換え)の質問1つ

FACTS = [
    dict(
        answer='pyainoは、NumPy/CuPyを中核に据えた深層学習フレームワークです。'
               'define-by-run方式の自動微分とGPU計算を備えながら、'
               'モデルの順伝播を通常の配列演算のまま記述できるのが特徴です。',
        train=['pyainoとは何ですか？', 'pyainoについて教えてください。'],
        eval='pyainoってどんなソフトウェア？'),
    dict(
        answer='pyainoの基本的な設計思想は「モデルをフレームワークに閉じ込めない」ことです。'
               '学習済みモデルは、モデル定義とチェックポイントがあれば、'
               'pyainoが無くてもNumPyだけで再構成して実行できます。',
        train=['pyainoの設計思想は何ですか？', 'pyainoが最も大切にしている設計原則は？'],
        eval='pyainoの根本的な考え方を説明してください。'),
    dict(
        answer='はい、できます。pyainoで学習したモデルは、モデル定義とチェックポイント'
               '（必要ならトークナイザ）があれば、ONNXなどの中間形式に変換しなくても'
               'NumPyだけで実行できます。',
        train=['pyainoで学習したモデルをpyainoなしで動かせますか？',
               'pyainoのモデルを実行するのにpyaino本体は必須ですか？'],
        eval='pyainoがインストールされていない環境で、pyainoで作ったモデルを使えますか？'),
    dict(
        answer='pyainoは数学と同じ行列積の規約 y = xW を採用しています。'
               '重みは(入力次元, 出力次元)の形で保持するので、数式との対応が素直です。',
        train=['pyainoの行列積の規約はどうなっていますか？',
               'pyainoでは重み行列の形はどう定義されますか？'],
        eval='pyainoの線形層の重みの向きについて教えて。'),
    dict(
        answer='pyainoは高階微分を数学的な構造に従って計算します。'
               'たとえば一次関数の二階微分は、人工的な残差の計算グラフを残さずに正しく0になります。',
        train=['pyainoは高階微分をどう扱いますか？', 'pyainoで一次関数を2回微分するとどうなりますか？'],
        eval='pyainoの二階微分の特徴は？'),
    dict(
        answer='はい。pyainoはCuPyがあればGPUで、無ければNumPyでCPU上で動きます。'
               'Config.set_np(\'numpy\') または set_np(\'cupy\') で切り替えられます。',
        train=['pyainoはGPUを使えますか？', 'pyainoでCPUとGPUを切り替えるには？'],
        eval='pyainoをGPUで動かすには何が必要ですか？'),
    dict(
        answer='pyainoでは各基本演算（Function）が順伝播 __forward__ と逆伝播 __backward__ を'
               '直接定義しています。そのため不要な計算グラフを保持せず、'
               '資源の消費を抑えつつ挙動を検証しやすくなっています。',
        train=['pyainoの逆伝播はどのように実装されていますか？',
               'pyainoのFunctionクラスは何を定義しますか？'],
        eval='pyainoで自動微分がどう計算されるか説明してください。'),
    dict(
        answer='はい。pyaino.QwenでQwen2.5やQwen3の小型モデルを、'
               'HuggingFaceのsafetensorsから直接読み込んで、推論・学習・LoRAができます。',
        train=['pyainoでQwenを動かせますか？', 'pyainoでLLMのLoRA学習はできますか？'],
        eval='pyainoで小さな言語モデルをファインチューニングできますか？'),
]

SYSTEM = None  # Qwen2.5の既定systemを使う

def train_examples():
    return [[{'role': 'user', 'content': q}, {'role': 'assistant', 'content': f['answer']}]
            for f in FACTS for q in f['train']]

def eval_examples():
    return [[{'role': 'user', 'content': f['eval']}, {'role': 'assistant', 'content': f['answer']}]
            for f in FACTS]

# 一般能力の劣化を見るための質問(ベースモデル自身の回答を参照にする)
GENERAL = ['日本の首都はどこですか？', '1から10までの整数の和はいくつですか？',
           'Pythonでリストを逆順にする方法を1つ教えてください。',
           'What is the boiling point of water at sea level?']

# 忘却抑制(replay)用の一般質問。評価用 GENERAL とは重ならないようにする
REPLAY = ['富士山の高さは何メートルですか？', '2の10乗はいくつですか？',
          'Pythonで辞書のキーを列挙する方法は？', '光の速さはおよそどれくらいですか？',
          '日本で一番長い川は何ですか？', '3の倍数を判定するPythonの関数を書いてください。',
          'What is the capital of France?', '水の化学式は何ですか？']
