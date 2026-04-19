"""
BanglaLLM Data Validation
Checks data quality before tokenizer training
"""
import os, random

BENGALI_START = 0x0980
BENGALI_END   = 0x09FF

def bengali_ratio(text):
    chars = [c for c in text if not c.isspace()]
    if not chars: return 0.0
    return sum(1 for c in chars if BENGALI_START <= ord(c) <= BENGALI_END) / len(chars)


def main():
    path = "data/final/train.txt"
    if not os.path.exists(path):
        print(f"ERROR: {path} not found. Run clean_data.py first.")
        return

    print(f"Analyzing: {path}")
    total = short = medium = long_ = 0
    total_bn_ratio = 0.0
    total_words = 0
    samples = []

    with open(path, encoding="utf-8") as f:
        lines = f.readlines()

    total = len(lines)
    for line in lines:
        line = line.strip()
        n = len(line)
        total_bn_ratio += bengali_ratio(line)
        total_words += len(line.split())
        if n < 50:   short += 1
        elif n < 500: medium += 1
        else:         long_ += 1

    avg_bn  = total_bn_ratio / max(total, 1)
    avg_len = sum(len(l.strip()) for l in lines) / max(total, 1)
    est_tok = total_words * 6.5  # conservative estimate at 6.5 tok/word

    print(f"Total lines        : {total:,}")
    print(f"Avg line length    : {avg_len:.0f} chars")
    print(f"Avg Bengali ratio  : {avg_bn*100:.1f}%")
    print(f"Short (<50)        : {short:,}  ({short/max(total,1)*100:.1f}%)")
    print(f"Medium (50-500)    : {medium:,}  ({medium/max(total,1)*100:.1f}%)")
    print(f"Long (>500)        : {long_:,}  ({long_/max(total,1)*100:.1f}%)")
    print(f"Word count         : {total_words:,}")
    print(f"Estimated tokens   : {est_tok/1e9:.2f}B (at 6.5 tok/word)")

    print(f"\n--- Training readiness ---")
    status_lines = "GOOD" if total > 1_000_000 else "WARNING"
    status_bn    = "GOOD" if avg_bn > 0.90 else "WARNING"
    print(f"{status_lines:<8}: {total/1e6:.1f}M lines")
    print(f"{status_bn:<8}: Bengali ratio {avg_bn*100:.1f}%")

    print(f"\n--- 5 random sample lines ---")
    sample_lines = random.sample(lines, min(5, len(lines)))
    for s in sample_lines:
        s = s.strip()
        print(f"  [{len(s)} chars] {s[:120]}")

    print(f"\nNext: python scripts/train_tokenizer.py")


if __name__ == "__main__":
    main()
