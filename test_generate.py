"""
Test generation from real checkpoint
Run: py -3.11 test_generate.py
"""
import os, sys, torch
sys.path.insert(0, ".")
from model.bangla_llm import BanglaLLM, Config
from tokenizers import Tokenizer

CKPT = os.environ.get("CHECKPOINT", "checkpoints/7b/best_checkpoint.pt")
TOK  = "tokenizer/tokenizer.json"

if not os.path.exists(CKPT):
    print(f"Checkpoint not found: {CKPT}")
    print("Run training first or set CHECKPOINT env var")
    sys.exit(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"Loading {CKPT}...")

ck  = torch.load(CKPT, map_location=device, weights_only=False)
cfg = Config(ck["cfg"]["size"])
m   = BanglaLLM(cfg).to(device)
if device.type == "cuda":
    m = m.to(torch.bfloat16)
m.load_state_dict(ck["model"])
m.eval()
print(f"Model loaded! size={cfg.size} | step={ck['step']:,} | loss={ck['loss']:.4f}")

tok = Tokenizer.from_file(TOK)
BOS = tok.token_to_id("<s>") or 1
EOS = tok.token_to_id("</s>") or 2

prompts = [
    "বাংলাদেশ একটি",
    "রবীন্দ্রনাথ ঠাকুর",
    "ঢাকা শহরে",
    "বাংলা ভাষা",
]

print()
for prompt in prompts:
    ids = torch.tensor([[BOS] + tok.encode(prompt).ids]).to(device)
    with torch.no_grad():
        out = m.generate(ids, max_new=100, temp=0.8, top_p=0.9, eos=EOS)
    result = tok.decode(out[0].tolist(), skip_special_tokens=True)
    print(f"Prompt: {prompt}")
    print(f"Output: {result[:300]}")
    print()

print("7B GENERATION OK")
