# pyaino.Qwen で対話(逐次表示)。LoRAアダプタを重ねることもできる
#   OPENBLAS_NUM_THREADS=16 python experiments/qwen/chat.py
#   OPENBLAS_NUM_THREADS=16 python experiments/qwen/chat.py --lora experiments/qwen/results/lora_adapter.pkl
#   python experiments/qwen/chat.py --model Qwen/Qwen3-0.6B --no-think
import argparse, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from pyaino.Config import *
from pyaino.Qwen import QwenLanguageModel
from pyaino.QwenTokenizer import QwenTokenizer

ap = argparse.ArgumentParser()
ap.add_argument('--model', default='Qwen/Qwen2.5-0.5B-Instruct')
ap.add_argument('--lora', default=None)
ap.add_argument('--max-new-tokens', type=int, default=256)
ap.add_argument('--temperature', type=float, default=0.7, help='0で確定的(argmax)')
ap.add_argument('--no-think', action='store_true', help='Qwen3で思考を省略')
ap.add_argument('--prompt', default=None, help='1回だけ答えて終了')
args = ap.parse_args()

tok = QwenTokenizer(args.model)       # ディレクトリまたはHFのリポジトリ名
model = QwenLanguageModel.from_pretrained(args.model)
if args.lora:
    model.load_lora(args.lora)

history = []
while True:
    q = args.prompt if args.prompt is not None else input('\nuser> ')
    if not q.strip():
        break
    history.append({'role': 'user', 'content': q})
    ids = tok.encode(tok.apply_chat_template(
        history, enable_thinking=False if args.no_think else None))
    buf, shown = [], 0
    def stream(t):
        global shown
        buf.append(t)
        s = tok.decode(buf, skip_special_tokens=True)
        if not s.endswith('�'):          # UTF-8の途中で切れていなければ表示
            print(s[shown:], end='', flush=True)
            shown = len(s)
    print('assistant> ', end='')
    t0 = time.time()
    t = args.temperature
    out = model.generate(ids, max_tokens=len(ids) + args.max_new_tokens,
                         stochastic=t > 0, beta=1 / t if t > 0 else 1, # beta=1/温度
                         top_k=20, top_p=0.8, repetition_penalty=1.05,
                         end_ids=[tok.eos_token_id, tok.pad_token_id], stream=stream)
    out = out[len(ids):]                  # seedを除いた応答部分
    print(f'\n[{len(out)} tokens, {len(out)/(time.time()-t0):.1f} tok/s]')
    history.append({'role': 'assistant', 'content': tok.decode(out, skip_special_tokens=True)})
    if args.prompt is not None:
        break
