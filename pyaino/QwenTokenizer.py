# QwenTokenizer
# 20261112
# Qwen2/Qwen2.5/Qwen3 用 byte-level BPE トークナイザ (tokenizer.json を直接読む)
# 外部依存なし。regexモジュールがあれば厳密な前処理正規表現を使い、
# 無ければ標準reによる近似(\p{L},\p{N}の置換)で動作する。

import json
import os
import unicodedata

try:
    import regex as _re
    _HAS_REGEX = True
except ImportError:  # pragma: no cover
    import re as _re
    _HAS_REGEX = False


def bytes_to_unicode():
    """ GPT-2方式のbyte->可視unicode文字の対応表 """
    bs = list(range(ord("!"), ord("~") + 1)) \
       + list(range(ord("¡"), ord("¬") + 1)) \
       + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


# Qwen2系の前処理(Split)の正規表現。tokenizer.jsonにあればそちらを優先する
QWEN_PRETOKENIZE_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")


def resolve_model_dir(name_or_dir, revision=None):
    """
    ディレクトリならそのまま返す。'Qwen/Qwen2.5-0.5B-Instruct' のようなHFのリポジトリ名なら
    huggingface_hub で取得(キャッシュ ~/.cache/huggingface/hub/ はtransformersと共通)して
    そのパスを返す。huggingface_hub はこの場合にだけ必要
    """
    if os.path.isdir(name_or_dir):
        return name_or_dir
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError(f"'{name_or_dir}' is not a directory. To download it from "
                          "Hugging Face Hub, install huggingface_hub "
                          "(pip install huggingface_hub).") from None
    return snapshot_download(name_or_dir, revision=revision,
                             allow_patterns=['*.json', '*.safetensors', '*.txt'])


def _compile_pattern(pattern):
    if _HAS_REGEX:
        return _re.compile(pattern)
    # 標準reでの近似: \p{L}=文字(数字と_以外の\w), \p{N}=数字
    p = pattern.replace(r'[^\r\n\p{L}\p{N}]', r'[^\r\n\w]|_') # 近似
    p = p.replace(r'\p{L}+', r'[^\W\d_]+').replace(r'\p{N}', r'\d')
    p = p.replace(r'[^\s\p{L}\p{N}]', r'[^\s\w]|_')
    return _re.compile(p)


class QwenTokenizer:
    """
    Qwen系のbyte-level BPEトークナイザ

    tok = QwenTokenizer(model_dir)  # HFのリポジトリ名 'Qwen/Qwen2.5-0.5B-Instruct' も可
    ids = tok.encode('こんにちは')  # cf.Tokenizer と同じく np.array で返す(ndarray=Falseでリスト)
    text = tok.decode(ids)
    tok.vocab_size(), tok.token2id, tok.id2token も cf.Tokenizer と同じ名前
    prompt = tok.apply_chat_template([{'role':'user','content':'...'}])
    """
    def __init__(self, path, revision=None):
        if not os.path.exists(path):   # ローカルに無ければHFのリポジトリ名とみなす
            path = resolve_model_dir(path, revision)
        if os.path.isdir(path):
            model_dir = path
            path = os.path.join(path, 'tokenizer.json')
        else:
            model_dir = os.path.dirname(path)
        with open(path, encoding='utf-8') as f:
            tj = json.load(f)

        model = tj['model']
        if model.get('type') != 'BPE':
            raise ValueError('Only BPE tokenizer is supported.')
        self.token2id = dict(model['vocab'])
        merges = model['merges']
        self.bpe_ranks = {}
        for i, m in enumerate(merges):
            a, b = m.split(' ', 1) if isinstance(m, str) else m
            self.bpe_ranks[(a, b)] = i
        self.ignore_merges = bool(model.get('ignore_merges', False))

        # 追加トークン(<|im_start|>など)は正規化・前処理より先に切り出す
        self.added_tokens = {}
        self.special_ids = set()
        for t in tj.get('added_tokens', []):
            self.added_tokens[t['content']] = t['id']
            if t.get('special', False):
                self.special_ids.add(t['id'])
        self.token2id.update(self.added_tokens)
        self.id2token = {v: k for k, v in self.token2id.items()}
        if self.added_tokens:
            alts = sorted(self.added_tokens, key=len, reverse=True)
            self.added_re = _re.compile('(' + '|'.join(_re.escape(a) for a in alts) + ')')
        else:
            self.added_re = None

        # 正規化と前処理
        norm = tj.get('normalizer') or {}
        self.nfc = norm.get('type') == 'NFC'
        pattern = self._find_split_pattern(tj.get('pre_tokenizer'))
        self.pat = _compile_pattern(pattern or QWEN_PRETOKENIZE_PATTERN)

        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.cache = {}

        # tokenizer_config.json からの付帯情報
        self.eos_token = '<|im_end|>'
        self.pad_token = '<|endoftext|>'
        cfg_path = os.path.join(model_dir, 'tokenizer_config.json')
        if os.path.exists(cfg_path):
            with open(cfg_path, encoding='utf-8') as f:
                cfg = json.load(f)
            self.eos_token = cfg.get('eos_token') or self.eos_token
            self.pad_token = cfg.get('pad_token') or self.pad_token
            template = cfg.get('chat_template') or ''
        else:
            template = ''
        # Qwen3の雛形は既定system無し・<think>あり、Qwen2.5は既定system有り
        self.qwen3_style = 'enable_thinking' in template
        self.eos_token_id = self.token2id[self.eos_token]
        self.pad_token_id = self.token2id[self.pad_token]
        print(self.__class__.__name__, model_dir, 'vocab_size =', self.vocab_size())

    @staticmethod
    def _find_split_pattern(pre):
        if not pre:
            return None
        if pre.get('type') == 'Split':
            return pre['pattern'].get('Regex')
        for p in pre.get('pretokenizers', []):
            r = QwenTokenizer._find_split_pattern(p)
            if r:
                return r
        return None

    def vocab_size(self):
        return len(self.token2id)

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        if self.ignore_merges and token in self.token2id:
            self.cache[token] = [token]
            return [token]
        word = list(token)
        ranks = self.bpe_ranks
        while len(word) > 1:
            best, best_rank = None, None
            for i in range(len(word) - 1):
                r = ranks.get((word[i], word[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best, best_rank = i, r
            if best is None:
                break
            a, b = word[best], word[best + 1]
            new = []
            i = 0
            while i < len(word):   # 同じペアは左から全て併合
                if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                    new.append(a + b)
                    i += 2
                else:
                    new.append(word[i])
                    i += 1
            word = new
        self.cache[token] = word
        return word

    def _split_ordinary(self, text):
        if self.nfc:
            text = unicodedata.normalize('NFC', text)
        tokens = []
        for piece in self.pat.findall(text):
            if not piece:
                continue
            mapped = ''.join(self.byte_encoder[b] for b in piece.encode('utf-8'))
            tokens.extend(self.bpe(mapped))
        return tokens

    def split_by_default(self, text):
        """ textを追加トークン(<|im_start|>など)が出るたびに分割、返り値はリスト """
        if self.added_re is None:
            return [text]
        return [p for p in self.added_re.split(text) if p]

    def split_text_into_tokens(self, text, allow_special=True):
        """ text(文字列)をtokens(byte-level BPEのトークン文字列のリスト)に分割 """
        if text is None:
            return None
        if isinstance(text, (list, tuple)): # 分割済みと想定
            return list(text)
        if not allow_special:
            return self._split_ordinary(text)
        tokens = []
        for part in self.split_by_default(text):
            if part in self.added_tokens:
                tokens.append(part)
            else:
                tokens.extend(self._split_ordinary(part))
        return tokens

    def encode(self, text, ndarray=True, allow_special=True):
        """ textを分割してidに変換し、それをindicesで返す(ndarray=Falseならリスト) """
        tokens = self.split_text_into_tokens(text, allow_special)
        indices = [self.token2id[t] for t in tokens]
        if ndarray:
            from pyaino.Config import np
            indices = np.array(indices)
        return indices

    def decode(self, indices, skip_special_tokens=False):
        """ indicesからtokenに変換して結合し文章textを返す """
        if hasattr(indices, 'tolist'):     # numpy/cupy の配列
            indices = indices.tolist()
        if isinstance(indices, int):
            indices = [indices]
        out, buf = [], []
        for i in indices:
            i = int(i)
            if i in self.special_ids and skip_special_tokens:
                continue
            tok = self.id2token.get(i)
            if tok is None:
                continue
            if tok in self.added_tokens:
                if buf:
                    out.append(bytes(buf).decode('utf-8', errors='replace'))
                    buf = []
                out.append(tok)
            else:
                buf.extend(self.byte_decoder[c] for c in tok)
        if buf:
            out.append(bytes(buf).decode('utf-8', errors='replace'))
        return ''.join(out)

    def __call__(self, text):
        return self.encode(text)

    # ---- チャット雛形(tools無しの範囲でHFのjinja雛形と同一出力) ----
    def apply_chat_template(self, messages, add_generation_prompt=True,
                            enable_thinking=None, default_system=None):
        """
        messages: [{'role': 'system'|'user'|'assistant', 'content': str}, ...]
        enable_thinking: Qwen3のみ。Falseで空の<think></think>を付けて思考を省略
        """
        s = ''
        if messages and messages[0]['role'] == 'system':
            s += '<|im_start|>system\n' + messages[0]['content'] + '<|im_end|>\n'
        elif not self.qwen3_style:
            sys_msg = default_system or \
                'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.'
            s += '<|im_start|>system\n' + sys_msg + '<|im_end|>\n'

        last_query = len(messages) - 1
        for i in range(len(messages) - 1, -1, -1):
            if messages[i]['role'] == 'user':
                last_query = i
                break

        for i, m in enumerate(messages):
            role, content = m['role'], m.get('content') or ''
            if role == 'user' or (role == 'system' and i != 0):
                s += '<|im_start|>' + role + '\n' + content + '<|im_end|>\n'
            elif role == 'assistant':
                if self.qwen3_style:
                    reasoning = m.get('reasoning_content')
                    if reasoning is None:
                        reasoning = ''
                        if '</think>' in content:
                            reasoning = content.split('</think>')[0].rstrip('\n')\
                                               .split('<think>')[-1].lstrip('\n')
                            content = content.split('</think>')[-1].lstrip('\n')
                    if i > last_query and (i == len(messages) - 1 or reasoning):
                        s += '<|im_start|>assistant\n<think>\n' + reasoning.strip('\n') \
                           + '\n</think>\n\n' + content.lstrip('\n')
                    else:
                        s += '<|im_start|>assistant\n' + content
                    s += '<|im_end|>\n'
                else:
                    s += '<|im_start|>assistant\n' + content + '<|im_end|>\n'
        if add_generation_prompt:
            s += '<|im_start|>assistant\n'
            if self.qwen3_style and enable_thinking is False:
                s += '<think>\n\n</think>\n\n'
        return s
