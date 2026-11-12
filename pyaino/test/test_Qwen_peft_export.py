# pyaino で学習した LoRA を PEFT 形式で書き出し、HF transformers + peft で読んで logits を比較
#   python test_Qwen_peft_export.py <model_dir> <lora_adapter.pkl> <peft_dir>
import sys
import numpy
import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel

from pyaino.Config import *
from pyaino.Qwen import QwenLanguageModel
from pyaino.QwenTokenizer import QwenTokenizer

d, adapter, peft_dir = sys.argv[1:4]
tok = QwenTokenizer(d)
ids = numpy.array([tok.encode(tok.apply_chat_template(
    [{'role': 'user', 'content': 'pyainoってどんなソフトウェア？'}]))])

model = QwenLanguageModel.from_pretrained(d)
model.load_lora(adapter)
model.export_peft(peft_dir)          # 読み込んだアダプタをPEFT形式で書き出す
z = model.forward(ids)

ref = AutoModelForCausalLM.from_pretrained(d, dtype=torch.float32)
ref = PeftModel.from_pretrained(ref, peft_dir).eval()
with torch.no_grad():
    zr = ref(torch.tensor(ids)).logits.numpy()
err = numpy.abs(z - zr).max() / numpy.abs(zr).max()
print(f'pyaino LoRA vs transformers+peft: rel.max.err={err:.2e}, '
      f'argmax一致率={(z.argmax(-1) == zr.argmax(-1)).mean():.3f}')
assert err < 1e-4
print('OK')
