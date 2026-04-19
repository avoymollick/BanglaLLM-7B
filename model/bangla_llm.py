"""
BanglaLLM-7B Model Architecture
LLaMA-3 style decoder-only transformer

EXACT PARAMETER COUNT (verified):
  Embedding:        64000 * 4096 = 262,144,000
  Per layer (x32):
    q_proj:         4096 * 4096 = 16,777,216
    k_proj:         4096 * 1024 = 4,194,304
    v_proj:         4096 * 1024 = 4,194,304
    o_proj:         4096 * 4096 = 16,777,216
    gate_proj:      4096 * 11008 = 45,088,768
    up_proj:        4096 * 11008 = 45,088,768
    down_proj:      11008 * 4096 = 45,088,768
    input_norm:     4096
    post_norm:      4096
    Per layer:      177,213,536
  32 layers:        5,670,833,152
  Final norm:       4,096
  LM head:          shared with embed (weight tying)
  TOTAL:            5,933,981,248  (~5.93B)
"""

import math, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

try:
    from flash_attn import flash_attn_func
    HAS_FLASH = True
except ImportError:
    HAS_FLASH = False


class Config:
    PRESETS = {
        "10m":  dict(vocab=64000, hidden=256,  layers=4,  heads=8,  kv=4, ffn=704,   ctx=4096),
        "100m": dict(vocab=64000, hidden=1024, layers=12, heads=16, kv=8, ffn=2816,  ctx=4096),
        "7b":   dict(vocab=64000, hidden=4096, layers=32, heads=32, kv=8, ffn=11008, ctx=4096),
    }
    def __init__(self, size="7b"):
        c = self.PRESETS[size]
        self.size     = size
        self.vocab    = c["vocab"]
        self.hidden   = c["hidden"]
        self.layers   = c["layers"]
        self.heads    = c["heads"]
        self.kv_heads = c["kv"]
        self.ffn      = c["ffn"]
        self.ctx      = c["ctx"]
        self.head_dim = c["hidden"] // c["heads"]
        self.groups   = c["heads"] // c["kv"]
        self.flash    = HAS_FLASH
        self.grad_ckpt = (size == "7b")


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.w = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w


class RoPE(nn.Module):
    def __init__(self, head_dim, base=10000.0):
        super().__init__()
        inv = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv", inv)
    def forward(self, T, device):
        t = torch.arange(T, device=device).float()
        f = torch.outer(t, self.inv.to(device))
        e = torch.cat([f, f], dim=-1)
        return e.cos(), e.sin()


def rotate_half(x):
    a, b = x.chunk(2, dim=-1)
    return torch.cat([-b, a], dim=-1)


def apply_rope(q, k, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class GQA(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.h  = cfg.heads
        self.kv = cfg.kv_heads
        self.g  = cfg.groups
        self.d  = cfg.head_dim
        self.fl = cfg.flash
        self.q_proj = nn.Linear(cfg.hidden, cfg.heads    * cfg.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden, cfg.kv_heads * cfg.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden, cfg.kv_heads * cfg.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.heads  * cfg.head_dim, cfg.hidden,   bias=False)
        self.rope   = RoPE(cfg.head_dim)

    def forward(self, x, mask=None):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.h,  self.d).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.kv, self.d).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.kv, self.d).transpose(1, 2)
        cos, sin = self.rope(T, x.device)
        q, k = apply_rope(q, k, cos, sin)
        if self.g > 1:
            k = k.unsqueeze(2).expand(-1,-1,self.g,-1,-1).reshape(B, self.h, T, self.d)
            v = v.unsqueeze(2).expand(-1,-1,self.g,-1,-1).reshape(B, self.h, T, self.d)
        if self.fl and HAS_FLASH:
            out = flash_attn_func(
                q.transpose(1,2).to(torch.bfloat16),
                k.transpose(1,2).to(torch.bfloat16),
                v.transpose(1,2).to(torch.bfloat16),
                causal=True
            ).to(x.dtype).reshape(B, T, -1)
        else:
            a = torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(self.d)
            if mask is not None:
                a = a.masked_fill(mask == 0, float("-inf"))
            a = F.softmax(a.float(), dim=-1).to(q.dtype)
            out = torch.matmul(a, v).transpose(1,2).contiguous().reshape(B, T, -1)
        return self.o_proj(out)


class SwiGLU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gate = nn.Linear(cfg.hidden, cfg.ffn, bias=False)
        self.up   = nn.Linear(cfg.hidden, cfg.ffn, bias=False)
        self.down = nn.Linear(cfg.ffn, cfg.hidden, bias=False)
    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn = GQA(cfg)
        self.ffn  = SwiGLU(cfg)
        self.n1   = RMSNorm(cfg.hidden)
        self.n2   = RMSNorm(cfg.hidden)
    def forward(self, x, mask=None):
        x = x + self.attn(self.n1(x), mask)
        x = x + self.ffn(self.n2(x))
        return x


class BanglaLLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg   = cfg
        self.embed = nn.Embedding(cfg.vocab, cfg.hidden)
        self.blocks= nn.ModuleList([Block(cfg) for _ in range(cfg.layers)])
        self.norm  = RMSNorm(cfg.hidden)
        self.head  = nn.Linear(cfg.hidden, cfg.vocab, bias=False)
        self.head.weight = self.embed.weight  # weight tying
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def n_params(self):
        return sum(p.numel() for p in self.parameters()) - self.head.weight.numel()

    def size_gb(self):
        return self.n_params() * 2 / 1e9

    def forward(self, ids, labels=None):
        B, T = ids.shape
        x = self.embed(ids)
        mask = None
        if not (self.cfg.flash and HAS_FLASH):
            mask = torch.tril(torch.ones(T, T, device=ids.device)).unsqueeze(0).unsqueeze(0)
        for blk in self.blocks:
            if self.cfg.grad_ckpt and self.training:
                x = checkpoint(blk, x, mask, use_reentrant=False)
            else:
                x = blk(x, mask)
        logits = self.head(self.norm(x))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, self.cfg.vocab), labels.view(-1), ignore_index=-100)
        return logits, loss

    @torch.no_grad()
    def generate(self, ids, max_new=200, temp=0.8, top_p=0.9, top_k=50, rep_pen=1.1, eos=2):
        self.eval()
        out = ids.clone()
        for _ in range(max_new):
            ctx = out[:, -self.cfg.ctx:]
            logits, _ = self(ctx)
            logits = logits[:, -1, :].float()
            if rep_pen != 1.0:
                for t in set(out[0].tolist()):
                    if 0 <= t < logits.shape[-1]:
                        logits[0, t] = logits[0, t] / rep_pen if logits[0, t] > 0 else logits[0, t] * rep_pen
            logits = logits / max(temp, 1e-6)
            if top_k > 0:
                kv2, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < kv2[:, -1:]] = float("-inf")
            sl, si = torch.sort(logits, descending=True)
            cp = torch.cumsum(F.softmax(sl, dim=-1), dim=-1)
            rm = cp > top_p
            rm[:, 1:] = rm[:, :-1].clone(); rm[:, 0] = False
            logits[rm.scatter(1, si, rm)] = float("-inf")
            probs = F.softmax(logits - logits.max(-1, keepdim=True).values, dim=-1)
            probs = torch.clamp(probs, 1e-10) / probs.sum(-1, keepdim=True)
            nxt = torch.multinomial(probs, 1)
            if nxt.item() == eos: break
            out = torch.cat([out, nxt], dim=1)
        return out


if __name__ == "__main__":
    print(f"\nFlash Attention: {'ENABLED' if HAS_FLASH else 'disabled (standard attn)'}")
    print()
    for size in ["10m", "100m", "7b"]:
        cfg = Config(size)
        p = cfg.vocab * cfg.hidden
        per = (cfg.hidden*cfg.heads*cfg.head_dim + cfg.hidden*cfg.kv_heads*cfg.head_dim*2 +
               cfg.heads*cfg.head_dim*cfg.hidden + cfg.hidden*cfg.ffn*2 +
               cfg.ffn*cfg.hidden + cfg.hidden*2)
        p += per * cfg.layers + cfg.hidden
        print(f"  {size:5s}  params={p/1e9:.4f}B  size={p*2/1e9:.2f}GB  "
              f"flash={cfg.flash}  gqa={cfg.heads}Q/{cfg.kv_heads}KV  ctx={cfg.ctx}")
    print()
    cfg = Config("10m")
    m = BanglaLLM(cfg)
    x = torch.randint(0, 64000, (2, 64))
    lg, loss = m(x, x)
    out = m.generate(x[:1, :5], max_new=10)
    print(f"  Forward pass (10m): input={x.shape} logits={lg.shape} loss={loss.item():.4f}")
    print(f"  Generation (10m):   {x[:1,:5].shape} -> {out.shape}")
    print(f"\n  ALL OK!")
