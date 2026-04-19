"""
BanglaLLM Instruction Fine-Tuning
Converts pretrained model to Bengali instruction-following assistant
"""
import os, sys, json, math, time, argparse, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch.utils.data import Dataset, DataLoader
from model.bangla_llm import BanglaLLM, Config

try:
    import wandb; HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


PROMPT_TEMPLATE = """<|system|>
আপনি BanglaLLM, একটি সহায়ক বাংলা ভাষার AI সহকারী।
<|user|>
{instruction}{input_part}
<|assistant|>
{output}"""


class InstructDataset(Dataset):
    def __init__(self, data_path, tokenizer_path, max_len=2048):
        from tokenizers import Tokenizer
        tok_file = tokenizer_path if tokenizer_path.endswith(".json") else                    os.path.join(tokenizer_path, "tokenizer.json")
        self.tok = Tokenizer.from_file(tok_file)
        self.max_len = max_len
        self.bos = self.tok.token_to_id("<s>") or 1
        self.eos = self.tok.token_to_id("</s>") or 2
        self.samples = []

        with open(data_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    row = json.loads(line)
                    inst  = row.get("instruction", "").strip()
                    inp   = row.get("input", "").strip()
                    outp  = row.get("output", "").strip()
                    if inst and outp:
                        self.samples.append((inst, inp, outp))
                except Exception:
                    continue

        print(f"Loaded {len(self.samples):,} instruction samples")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        inst, inp, outp = self.samples[idx]
        inp_part = f"\n{inp}" if inp else ""
        prompt = PROMPT_TEMPLATE.format(
            instruction=inst, input_part=inp_part, output=outp
        )
        ids = [self.bos] + self.tok.encode(prompt).ids + [self.eos]
        ids = ids[:self.max_len]
        t = torch.tensor(ids, dtype=torch.long)
        return t[:-1], t[1:]


def collate(batch):
    inps, lbls = zip(*batch)
    max_l = max(x.size(0) for x in inps)
    padded_inp = torch.zeros(len(inps), max_l, dtype=torch.long)
    padded_lbl = torch.full((len(lbls), max_l), -100, dtype=torch.long)
    for i, (x, y) in enumerate(zip(inps, lbls)):
        padded_inp[i, :x.size(0)] = x
        padded_lbl[i, :y.size(0)] = y
    return padded_inp, padded_lbl


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_ckpt", required=True)
    p.add_argument("--data",      default="data/instruct/combined.jsonl")
    p.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    p.add_argument("--ckpt_dir",  default="checkpoints/instruct")
    p.add_argument("--steps",     type=int,   default=3000)
    p.add_argument("--batch",     type=int,   default=2)
    p.add_argument("--lr",        type=float, default=2e-5)
    p.add_argument("--wandb",     action="store_true")
    args = p.parse_args()

    os.makedirs(args.ckpt_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading base model: {args.base_ckpt}")
    ck = torch.load(args.base_ckpt, map_location=device, weights_only=False)
    cfg = Config(ck["cfg"]["size"])
    model = BanglaLLM(cfg).to(device)
    if device.type == "cuda":
        model = model.to(torch.bfloat16)
    model.load_state_dict(ck["model"])
    print(f"Base model loaded | size={cfg.size} | step={ck['step']:,} | loss={ck['loss']:.4f}")

    ds = InstructDataset(args.data, args.tokenizer)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True,
                    collate_fn=collate, num_workers=0)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                             betas=(0.9, 0.95), weight_decay=0.1)

    if args.wandb and HAS_WANDB:
        try:
            wandb.init(project="BanglaLLM", name="instruct-finetune")
        except Exception:
            pass

    step, best = 0, float("inf")
    t0 = time.time()
    model.train()
    print(f"\nInstruction fine-tuning: {args.steps} steps\n")

    for _ in range(999):
        for inp, lbl in dl:
            inp = inp.to(device); lbl = lbl.to(device)
            lr = args.lr * min(1.0, step / 100)
            for g in opt.param_groups: g["lr"] = lr
            _, loss = model(inp, lbl)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); opt.zero_grad(set_to_none=True)
            lv = loss.item()
            if step % 10 == 0:
                el = time.time() - t0
                print(f"instruct step {step:5d}/{args.steps} | loss {lv:.4f} | lr {lr:.2e}")
            if lv < best:
                best = lv
                torch.save({"step": step, "model": model.state_dict(),
                            "loss": best, "cfg": {"size": cfg.size}},
                           f"{args.ckpt_dir}/instruct_best.pt")
            step += 1
            if step >= args.steps:
                print(f"\nInstruct fine-tune done! best_loss={best:.4f}")
                if args.wandb and HAS_WANDB and wandb.run: wandb.finish()
                return

if __name__ == "__main__":
    main()
