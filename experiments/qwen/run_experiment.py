# Qwen2.5-0.5B-Instruct に pyaino の知識を教える実験 (LoRA / 全パラメタ学習)
#
#   OPENBLAS_NUM_THREADS=16 python experiments/qwen/run_experiment.py --mode lora
#   OPENBLAS_NUM_THREADS=16 python experiments/qwen/run_experiment.py --mode full
#
# 測るもの
#   - 学習損失の推移
#   - 評価損失: 学習に使っていない言い換え質問に対する正解応答のNLL
#   - 一般能力の変化: 無関係な質問へのベースモデル自身の回答のNLL(上がるほど劣化)
#   - 学習前後の生成例
import argparse, json, os, random, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.dirname(__file__))

from pyaino.Config import *
from pyaino.Qwen import QwenLanguageModel, build_sft_example, collate
from pyaino import Optimizers
from pyaino.QwenTokenizer import QwenTokenizer
import pyaino_qa as qa

ap = argparse.ArgumentParser()
ap.add_argument('--model', default='Qwen/Qwen2.5-0.5B-Instruct')
ap.add_argument('--mode', choices=['lora', 'full'], default='lora')
ap.add_argument('--steps', type=int, default=24)
ap.add_argument('--batch', type=int, default=8)
ap.add_argument('--lr', type=float, default=None)
ap.add_argument('--r', type=int, default=8)
ap.add_argument('--alpha', type=float, default=16)
ap.add_argument('--replay', action='store_true',
                help='ベースモデル自身の一般応答を学習データに混ぜて忘却を抑える')
ap.add_argument('--tag', default=None, help='結果ファイル名(既定はmode)')
ap.add_argument('--out', default=os.path.join(os.path.dirname(__file__), 'results'))
args = ap.parse_args()
lr = args.lr or (2e-4 if args.mode == 'lora' else 1e-5)
os.makedirs(args.out, exist_ok=True)

d = args.model  # ディレクトリまたはHFのリポジトリ名
tok = QwenTokenizer(d)
model = QwenLanguageModel.from_pretrained(d, optimize='AdamT', w_decay=0.0)
EOS = [tok.eos_token_id, tok.pad_token_id]

def chat(q, n=80):
    ids = tok.encode(tok.apply_chat_template([{'role': 'user', 'content': q}]))
    out = model.generate(ids, max_tokens=len(ids) + n, end_ids=EOS) # seedを含む全長
    return tok.decode(out[len(ids):], skip_special_tokens=True).strip()

def mean_loss(examples, bs=8):
    """ 応答部分のトークン平均NLL """
    tot, n = 0.0, 0
    for i in range(0, len(examples), bs):
        x, t = collate(examples[i:i + bs], tok.pad_token_id)
        l = float(model.forward(x, t))
        k = int((t != -100).sum())
        tot += l * k; n += k
    return tot / n

train = [build_sft_example(tok, m) for m in qa.train_examples()]
evals = [build_sft_example(tok, m) for m in qa.eval_examples()]
print(f'train {len(train)} examples, eval {len(evals)} examples, '
      f'tokens/example ~{sum(len(i) for i, _ in train)/len(train):.0f}')

# ---- 学習前 ----
t0 = time.time()
before = {'eval_q': {f['eval']: chat(f['eval']) for f in qa.FACTS}}
before['general'] = {q: chat(q, 60) for q in qa.GENERAL}
# 一般質問へのベースモデル自身の回答を参照データにする
general_ref = [build_sft_example(tok, [{'role': 'user', 'content': q},
                                       {'role': 'assistant', 'content': a}])
               for q, a in before['general'].items()]
before['train_loss'] = mean_loss(train)
before['eval_loss'] = mean_loss(evals)
before['general_nll'] = mean_loss(general_ref)
print(f"before: train={before['train_loss']:.3f} eval={before['eval_loss']:.3f} "
      f"general={before['general_nll']:.3f} ({time.time()-t0:.0f}s)")

if args.replay:  # 学習前のモデル自身の応答を教師にする(自己蒸留的なreplay)
    replay = [build_sft_example(tok, [{'role': 'user', 'content': q},
                                      {'role': 'assistant', 'content': chat(q, 80)}])
              for q in qa.REPLAY]
    train = train + replay
    print(f'replay: +{len(replay)} examples -> train {len(train)}')

# ---- 学習 ----
if args.mode == 'lora':
    model.apply_lora(r=args.r, alpha=args.alpha, seed=0) # 基底は凍結
dropout = 0.05 if args.mode == 'lora' else 0.0               # LoRA経路のdropout
n_train = model.num_parameters(trainable_only=True)

# 学習率: 2ステップで線形に立ち上げ、その後cosineで0まで下げる(pyainoのOptimizersの関数)
#   iteration = step+1 で呼ぶ(step 0 から学習率が0にならないように)
scheduler = Optimizers.LinearGrowCosineDecay(warmup=2, decay_rate=0.0,
                                             decay_start=3, decay_end=args.steps + 1)
random.seed(0)
order = []
log = []
t_train = time.time()
for step in range(args.steps):
    if len(order) < args.batch:
        perm = list(range(len(train))); random.shuffle(perm); order += perm
    batch = [train[order.pop(0)] for _ in range(args.batch)]
    x, t = collate(batch, tok.pad_token_id)
    eta = lr * float(scheduler(step + 1))
    ts = time.time()
    loss = float(model.forward(x, t, dropout=dropout))
    model.backward()
    gnorm = model.update(eta=eta, max_grad_norm=1.0)
    rec = dict(step=step + 1, loss=loss, lr=eta, grad_norm=gnorm,
               tokens=int(x.size), sec=time.time() - ts)
    if (step + 1) % 6 == 0 or step + 1 == args.steps:
        rec['eval_loss'] = mean_loss(evals)
    log.append(rec)
    print(' '.join(f'{k}={v:.4g}' if isinstance(v, float) else f'{k}={v}'
                   for k, v in rec.items()), flush=True)
t_train = time.time() - t_train

# ---- 学習後 ----
after = {'eval_q': {f['eval']: chat(f['eval']) for f in qa.FACTS}}
after['general'] = {q: chat(q, 60) for q in qa.GENERAL}
after['train_loss'] = mean_loss(train)
after['eval_loss'] = mean_loss(evals)
after['general_nll'] = mean_loss(general_ref)
print(f"after:  train={after['train_loss']:.3f} eval={after['eval_loss']:.3f} "
      f"general={after['general_nll']:.3f}")

extra = {}
if args.mode == 'lora':
    # 保存→新しいモデルへ読み込み→logits一致、merge後もlogits一致を確認
    ad = os.path.join(args.out, f'{args.tag or "lora"}_adapter.pkl')
    model.save_lora(ad)
    model.export_peft(os.path.join(args.out, f'{args.tag or "lora"}_peft'), args.model)
    probe = collate([evals[0]], tok.pad_token_id)[0]
    z_trained = model.forward(probe)
    m2 = QwenLanguageModel.from_pretrained(d)
    m2.load_lora(ad)
    z_loaded = m2.forward(probe)
    m2.merge_lora()
    z_merged = m2.forward(probe)
    extra = dict(adapter_bytes=os.path.getsize(ad),
                 reload_max_abs_diff=float(np.abs(z_loaded - z_trained).max()),
                 merge_max_abs_diff=float(np.abs(z_merged - z_trained).max()))
    print(extra)

res = dict(args=vars(args), lr=lr, trainable_params=n_train,
           total_params=model.num_parameters(), train_seconds=t_train,
           before=before, after=after, log=log, **extra)
with open(os.path.join(args.out, f'{args.tag or args.mode}.json'), 'w') as f:
    json.dump(res, f, ensure_ascii=False, indent=1)
print('\n==== 生成例(評価用の言い換え質問) ====')
for q in before['eval_q']:
    print(f'Q: {q}\n  前: {before["eval_q"][q]}\n  後: {after["eval_q"][q]}')
print('\n==== 一般質問 ====')
for q in before['general']:
    print(f'Q: {q}\n  前: {before["general"][q]}\n  後: {after["general"][q]}')
