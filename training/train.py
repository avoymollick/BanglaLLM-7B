"""
BanglaLLM Training Script
Memory optimized for 7B on H100 80GB with 4096 context
"""
import os, sys, math, time, argparse, torch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch.utils.data import DataLoader
from model.bangla_llm import BanglaLLM, Config
from training.dataset import BengaliDataset

try:
    import wandb; HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


def lr_schedule(step, max_lr, min_lr, warmup, total):
    if step < warmup:
        return max_lr * step / max(warmup, 1)
    p = (step - warmup) / max(total - warmup, 1)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * p))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--size",       default="7b")
    p.add_argument("--steps",      type=int,   default=100000)
    p.add_argument("--batch",      type=int,   default=2)
    p.add_argument("--max_len",    type=int,   default=4096)
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--min_lr",     type=float, default=3e-5)
    p.add_argument("--warmup",     type=int,   default=2000)
    p.add_argument("--save_every", type=int,   default=1000)
    p.add_argument("--log_every",  type=int,   default=10)
    p.add_argument("--data",       default="data/final/train.txt")
    p.add_argument("--tokenizer",  default="tokenizer/tokenizer.json")
    p.add_argument("--ckpt_dir",   default="checkpoints/7b")
    p.add_argument("--resume",     default="")
    p.add_argument("--wandb",      action="store_true")
    p.add_argument("--run_name",   default="BanglaLLM-7B-v2")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice : {device}")
    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Load model in bfloat16 directly
    cfg = Config(args.size)
    model = BanglaLLM(cfg).to(device)
    if device.type == "cuda":
        model = model.to(torch.bfloat16)
        torch.cuda.empty_cache()
        used = torch.cuda.memory_allocated() / 1e9
        free = torch.cuda.get_device_properties(0).total_memory / 1e9 - used
        print(f"VRAM after model: {used:.1f}GB used / {free:.1f}GB free")

    params = model.n_params()
    print(f"Model  : {args.size.upper()} | {params/1e9:.4f}B params | {model.size_gb():.2f}GB")
    print(f"Flash  : {cfg.flash} | GQA {cfg.heads}Q/{cfg.kv_heads}KV | grad_ckpt={cfg.grad_ckpt}")
    print(f"Context: {args.max_len} tokens")

    # Dataset
    nw = 0 if os.name == "nt" else min(4, os.cpu_count())
    ds = BengaliDataset(args.data, args.tokenizer, max_len=args.max_len)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True,
                    num_workers=nw, pin_memory=(device.type=="cuda"),
                    persistent_workers=(nw > 0))

    # Optimizer
    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr,
        betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1
    )

    # W&B
    if args.wandb and HAS_WANDB:
        try:
            wandb.init(project="BanglaLLM", name=args.run_name,
                       config={"size": args.size, "steps": args.steps,
                               "batch": args.batch, "max_len": args.max_len,
                               "lr": args.lr, "params": params})
            print("W&B  : connected")
        except Exception as e:
            print(f"W&B  : failed ({e})")

    # Resume
    step, best_loss = 0, float("inf")
    if args.resume and os.path.isfile(args.resume):
        print(f"\nResuming: {args.resume}")
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step = ck["step"] + 1
        best_loss = ck.get("best_loss", float("inf"))
        print(f"Resumed at step {step:,} | best_loss={best_loss:.4f}")

    print(f"\n{'='*65}")
    print(f"Training {args.size.upper()} | {args.steps:,} steps | batch={args.batch} | ctx={args.max_len}")
    print(f"{'='*65}\n")

    tok_seen = 0
    t0 = time.time()
    model.train()

    for _ in range(99999):
        for inp, lbl in dl:
            inp = inp.to(device, non_blocking=True)
            lbl = lbl.to(device, non_blocking=True)

            lr = lr_schedule(step, args.lr, args.min_lr, args.warmup, args.steps)
            for g in opt.param_groups:
                g["lr"] = lr

            _, loss = model(inp, lbl)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)

            lv = loss.item()
            tok_seen += inp.numel()

            if step % args.log_every == 0:
                el = time.time() - t0
                tps = tok_seen / max(el, 1)
                eta = (args.steps - step) / max(step / max(el, 1), 1) if step > 0 else 0
                print(f"step {step:7d}/{args.steps:,} | loss {lv:.4f} | "
                      f"lr {lr:.2e} | grad {grad:.2f} | "
                      f"{tps/1000:.1f}k tok/s | "
                      f"ETA {int(eta//3600)}h{int((eta%3600)//60):02d}m")
                if args.wandb and HAS_WANDB and wandb.run:
                    try:
                        wandb.log({"loss": lv, "lr": lr, "grad_norm": grad,
                                   "tok_per_s": tps, "tokens_seen_B": tok_seen/1e9},
                                  step=step)
                    except Exception:
                        pass

            if step % args.save_every == 0 and step > 0:
                path = f"{args.ckpt_dir}/step_{step:07d}.pt"
                torch.save({"step": step, "model": model.state_dict(),
                            "opt": opt.state_dict(), "loss": lv,
                            "best_loss": best_loss,
                            "cfg": {"size": args.size, "max_len": args.max_len}}, path)
                print(f"  Saved: {path}")

            if lv < best_loss:
                best_loss = lv
                torch.save({"step": step, "model": model.state_dict(),
                            "opt": opt.state_dict(), "loss": best_loss,
                            "best_loss": best_loss,
                            "cfg": {"size": args.size, "max_len": args.max_len}},
                           f"{args.ckpt_dir}/best_checkpoint.pt")

            step += 1
            if step >= args.steps:
                print(f"\n{'='*65}")
                print(f"TRAINING COMPLETE")
                print(f"final_loss={lv:.4f} | best_loss={best_loss:.4f} | "
                      f"tokens_seen={tok_seen/1e9:.3f}B | steps={step:,}")
                print(f"{'='*65}")
                if args.wandb and HAS_WANDB and wandb.run:
                    wandb.finish()
                return


if __name__ == "__main__":
    main()
