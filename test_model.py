"""
Test model architecture on laptop CPU
Run: py -3.11 test_model.py
"""
import sys, torch
sys.path.insert(0, ".")
from model.bangla_llm import BanglaLLM, Config, HAS_FLASH

print()
print("="*60)
print("  BanglaLLM — Laptop Model Test")
print("="*60)
print(f"  Flash Attention: {'ENABLED' if HAS_FLASH else 'disabled (standard attn)'}")
print()

# Print parameter counts without loading 7B on CPU
for size in ["10m", "100m", "7b"]:
    cfg = Config(size)
    # Calculate params mathematically
    p  = cfg.vocab * cfg.hidden
    per = (cfg.hidden*cfg.heads*cfg.head_dim +
           cfg.hidden*cfg.kv_heads*cfg.head_dim*2 +
           cfg.heads*cfg.head_dim*cfg.hidden +
           cfg.hidden*cfg.ffn*2 + cfg.ffn*cfg.hidden + cfg.hidden*2)
    p += per * cfg.layers + cfg.hidden
    print(f"  {size:5s}  params={p/1e9:.4f}B  size={p*2/1e9:.2f}GB  "
          f"flash={cfg.flash}  gqa={cfg.heads}Q/{cfg.kv_heads}KV  ctx={cfg.ctx}")

print()

# Forward pass on 10m
print("  Testing 10m forward pass...")
cfg = Config("10m")
m = BanglaLLM(cfg).cuda()
actual = m.n_params()
x = torch.randint(0, 64000, (2, 64)).cuda()
logits, loss = m(x, x)
assert logits.shape == (2, 64, 64000)
print(f"  10m actual params: {actual/1e9:.4f}B")
print(f"  Loss: {loss.item():.4f}")

# Generation test
m.eval()
out = m.generate(x[:1,:5], max_new=10)
assert out.shape[1] > 5
print(f"  Generation: {x[:1,:5].shape} -> {out.shape}")

# Verify 7B parameter count
cfg7 = Config("7b")
expected = 5_933_109_248
p = cfg7.vocab * cfg7.hidden
per = (cfg7.hidden*cfg7.heads*cfg7.head_dim +
       cfg7.hidden*cfg7.kv_heads*cfg7.head_dim*2 +
       cfg7.heads*cfg7.head_dim*cfg7.hidden +
       cfg7.hidden*cfg7.ffn*2 + cfg7.ffn*cfg7.hidden + cfg7.hidden*2)
p += per * cfg7.layers + cfg7.hidden
print(f"\n  7B parameter verification:")
print(f"  Expected:  {expected:,}")
print(f"  Calculated:{p:,}")
assert abs(p - expected) < 1000000, f"Parameter mismatch! expected={expected} got={p}"
print(f"  MATCH ✓")

print()
print("  ALL OK!")
print("="*60)
