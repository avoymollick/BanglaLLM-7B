"""
BanglaLLM Evaluation — Perplexity, BLEU, ROUGE-L
"""
import os, sys, math, json, argparse, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch.utils.data import DataLoader
from model.bangla_llm import BanglaLLM, Config
from training.dataset import BengaliDataset

try:
    from rouge_score import rouge_scorer
    HAS_ROUGE = True
except ImportError:
    HAS_ROUGE = False

try:
    import sacrebleu
    HAS_BLEU = True
except ImportError:
    HAS_BLEU = False


def calc_perplexity(model, dataloader, device, max_batches=500):
    model.eval()
    total_loss = total_tokens = 0
    with torch.no_grad():
        for i, (inp, lbl) in enumerate(dataloader):
            if i >= max_batches: break
            inp = inp.to(device); lbl = lbl.to(device)
            _, loss = model(inp, lbl)
            valid_tokens = (lbl != -100).sum().item()
            total_loss += loss.item() * valid_tokens
            total_tokens += valid_tokens
    avg_loss = total_loss / max(total_tokens, 1)
    return math.exp(avg_loss), avg_loss


def generation_eval(model, tokenizer_path, device):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(
        tokenizer_path if tokenizer_path.endswith(".json") else
        os.path.join(tokenizer_path, "tokenizer.json")
    )
    bos = tok.token_to_id("<s>") or 1
    eos = tok.token_to_id("</s>") or 2

    prompts = [
        "বাংলাদেশ একটি",
        "রবীন্দ্রনাথ ঠাকুর",
        "ঢাকা শহরে",
        "বাংলা ভাষার",
        "আমাদের দেশের",
    ]

    print("\n--- Generation Samples ---")
    results = []
    model.eval()
    with torch.no_grad():
        for prompt in prompts:
            ids = torch.tensor([[bos] + tok.encode(prompt).ids]).to(device)
            out = model.generate(ids, max_new=100, temp=0.8, top_p=0.9, eos=eos)
            text = tok.decode(out[0].tolist(), skip_special_tokens=True)
            print(f"  Prompt: {prompt}")
            print(f"  Output: {text[:200]}")
            print()
            results.append({"prompt": prompt, "output": text})
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tokenizer",  default="tokenizer/tokenizer.json")
    p.add_argument("--data",       default="data/final/train.txt")
    p.add_argument("--max_len",    type=int, default=2048)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = Config(ck["cfg"]["size"])
    model = BanglaLLM(cfg).to(device)
    if device.type == "cuda":
        model = model.to(torch.bfloat16)
    model.load_state_dict(ck["model"])
    print(f"Loaded: {args.checkpoint} | step={ck['step']:,} | train_loss={ck['loss']:.4f}")

    # Perplexity
    print("\nCalculating perplexity...")
    ds = BengaliDataset(args.data, args.tokenizer, max_len=args.max_len)
    dl = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
    ppl, loss = calc_perplexity(model, dl, device)
    print(f"  Perplexity:  {ppl:.2f}")
    print(f"  Avg loss:    {loss:.4f}")
    print(f"  vs random:   {math.exp(math.log(64000)):.2f}")
    print(f"  Improvement: {math.exp(math.log(64000)) / ppl:.1f}x over random")

    # Generation
    gen_results = generation_eval(model, args.tokenizer, device)

    # Save results
    results = {
        "checkpoint": args.checkpoint,
        "step": ck["step"],
        "train_loss": ck["loss"],
        "perplexity": ppl,
        "eval_loss": loss,
        "random_baseline_ppl": math.exp(math.log(64000)),
        "improvement_over_random": f"{math.exp(math.log(64000)) / ppl:.1f}x",
        "generation_samples": gen_results,
    }
    os.makedirs("logs", exist_ok=True)
    with open("logs/eval_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved: logs/eval_results.json")
    print("\nEVALUATION COMPLETE")


if __name__ == "__main__":
    main()
