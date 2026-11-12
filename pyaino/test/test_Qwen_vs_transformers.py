# pyaino.Qwen を HuggingFace transformers(PyTorch, float32) と数値照合する
#   1) logits の一致          2) KVキャッシュ生成 == キャッシュ無し, HFのgreedy生成との一致
#   3) 全パラメタ学習時の勾配    4) LoRA の勾配(torch側は同じLoRAを手で挿入)
# 実行には torch, transformers が必要(pyaino.Qwen自体は numpy のみで動作)
import os, sys, glob, time
import numpy
import torch
from transformers import AutoModelForCausalLM

from pyaino.Config import *
from pyaino.Qwen import QwenLanguageModel
from pyaino.QwenTokenizer import QwenTokenizer

dirs = sys.argv[1:] or sorted(glob.glob(
    os.path.expanduser('~/.cache/huggingface/hub/models--Qwen--*/snapshots/*')))

def rel(a, b):
    a = numpy.asarray(a, dtype=numpy.float64); b = numpy.asarray(b, dtype=numpy.float64)
    return numpy.abs(a - b).max() / (numpy.abs(b).max() + 1e-12)

for d in dirs:
    print('=' * 70, '\n', d)
    tok = QwenTokenizer(d)
    model = QwenLanguageModel.from_pretrained(d)
    ref = AutoModelForCausalLM.from_pretrained(d, dtype=torch.float32,
                                               attn_implementation='eager').eval()

    text = tok.apply_chat_template([{'role': 'user', 'content': 'pyainoとは何ですか？簡潔に。'}])
    ids = numpy.array([tok.encode(text)])
    T = ids.shape[1]

    # 1) logits
    z = model.forward(ids)
    with torch.no_grad():
        zr = ref(torch.tensor(ids)).logits.numpy()
    print(f'[logits]   T={T} rel.max.err={rel(z, zr):.2e}  argmax一致率='
          f'{(z.argmax(-1) == zr.argmax(-1)).mean():.3f}')

    # 2) KVキャッシュ生成をキャッシュ無しの一括計算と、HFのgreedy生成と比べる
    t0 = time.time()
    out = model.generate(ids[0], max_tokens=T + 24)[T:].tolist() # seedを含む全長を指定
    dt = time.time() - t0
    with torch.no_grad():
        outr = ref.generate(torch.tensor(ids), max_new_tokens=24, do_sample=False,
                            top_p=None, top_k=None, temperature=None,
                            repetition_penalty=1.0  # generation_configの既定値(1.1)を打ち消す
                            )[0, T:].tolist()
    seq = list(ids[0])
    for t in out[:8]:
        assert int(model.forward(numpy.array([seq]))[0, -1].argmax()) == t, \
            'cache/no-cache mismatch'
        seq.append(t)
    n_same = next((i for i, (a, b) in enumerate(zip(out, outr)) if a != b),
                  min(len(out), len(outr)))
    print(f'[generate] {len(out)} tok in {dt:.1f}s; HF greedyと先頭{n_same}/{len(outr)}トークン一致')
    print('           pyaino:', repr(tok.decode(out)))

    # 学習用の入力(pyainoの流儀: x と 次トークン t、前半は-100で無視)
    x, t = ids[:, :-1], ids[:, 1:].copy()
    t[0, :T // 2] = -100
    lab_t = torch.tensor(numpy.concatenate([[[-100]], t], axis=1))  # HFはlabelsを内部でずらす

    # 3) 全パラメタ学習時の勾配
    loss = model.forward(x, t)
    model.backward()
    ref.zero_grad()
    out_t = ref(torch.tensor(ids), labels=lab_t)
    out_t.loss.backward()
    print(f'[loss]     pyaino={float(loss):.6f} torch={out_t.loss.item():.6f}')
    G = model.hf_state_dict(grad=True)
    R = dict(ref.named_parameters())
    worst = 0.0
    for n in G:
        e = rel(G[n], R[n].grad.numpy()); worst = max(worst, e)
    print(f'   全{len(G)}パラメタの勾配: max rel.err={worst:.2e}')

    # 4) LoRA の勾配
    model.apply_lora(r=4, alpha=8, seed=0)
    L = model.lora_parameters()
    for k, v in L.items():                     # Bも非零にして全経路を検査
        if k.endswith('lora_B'):
            v[...] = numpy.random.randn(*v.shape).astype('float32') * 0.01
    tl = {}
    for name, m in ref.named_modules():
        if isinstance(m, torch.nn.Linear) and name.split('.')[-1] in model.LORA_TARGETS:
            A_ = torch.tensor(L[name + '.lora_A'], requires_grad=True)
            B_ = torch.tensor(L[name + '.lora_B'], requires_grad=True)
            tl[name] = (A_, B_)
            m.register_forward_hook(lambda mod, inp, out, A_=A_, B_=B_:
                                    out + (inp[0] @ A_) @ B_ * 2.0)
    loss = model.forward(x, t)
    model.backward()
    for p in ref.parameters():
        p.requires_grad_(False)
    out_t = ref(torch.tensor(ids), labels=lab_t)
    out_t.loss.backward()
    print(f'[LoRA]     loss pyaino={float(loss):.6f} torch={out_t.loss.item():.6f}')
    LG = model.lora_parameters(grad=True)
    errs = []
    for name, (A_, B_) in tl.items():
        errs.append(rel(LG[name + '.lora_A'], A_.grad.numpy()))
        errs.append(rel(LG[name + '.lora_B'], B_.grad.numpy()))
    print(f'   LoRA grads: {len(errs)} tensors, max rel.err={max(errs):.2e} '
          f'median={numpy.median(errs):.2e}')
    assert rel(z, zr) < 1e-4 and n_same == len(outr)
    assert max(errs) < 1e-3 and worst < 1e-3
    del model, ref
print('ALL OK')
