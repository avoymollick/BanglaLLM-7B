"""
BanglaLLM Data Cleaning Pipeline
Filters Bengali text for quality and language purity
"""
import os, re, unicodedata

BENGALI_START = 0x0980
BENGALI_END   = 0x09FF
MIN_LEN = 30
MAX_LEN = 2000
MIN_BN_RATIO = 0.5


def is_bengali(ch):
    return BENGALI_START <= ord(ch) <= BENGALI_END


def bengali_ratio(text):
    chars = [c for c in text if not c.isspace()]
    if not chars: return 0.0
    return sum(1 for c in chars if is_bengali(c)) / len(chars)


def clean_line(line):
    line = line.strip()
    if not line: return None
    # Remove URLs
    line = re.sub(r"http\S+|www\S+", "", line)
    # Remove HTML tags
    line = re.sub(r"<[^>]+>", "", line)
    # Remove excessive whitespace
    line = re.sub(r"\s+", " ", line).strip()
    # Length filter
    if len(line) < MIN_LEN or len(line) > MAX_LEN: return None
    # Bengali ratio filter
    if bengali_ratio(line) < MIN_BN_RATIO: return None
    # Remove lines that are mostly numbers/punctuation
    alpha = sum(1 for c in line if c.isalpha())
    if alpha < 10: return None
    return line


def main():
    raw_dir = "data/raw"
    out_path = "data/final/train.txt"
    os.makedirs("data/final", exist_ok=True)

    raw_files = [f for f in os.listdir(raw_dir) if f.endswith(".txt")]
    if not raw_files:
        print("No raw .txt files found in data/raw/")
        return

    print(f"Found {len(raw_files)} raw files: {raw_files}")
    print(f"Cleaning → {out_path}")

    total_kept = 0
    total_skipped = 0
    seen = set()  # deduplication

    with open(out_path, "w", encoding="utf-8") as fout:
        for fname in sorted(raw_files):
            path = os.path.join(raw_dir, fname)
            kept = skipped = 0
            with open(path, encoding="utf-8", errors="ignore") as fin:
                for line in fin:
                    cleaned = clean_line(line)
                    if cleaned is None:
                        skipped += 1; continue
                    # Deduplication
                    h = hash(cleaned)
                    if h in seen:
                        skipped += 1; continue
                    seen.add(h)
                    fout.write(cleaned + "\n")
                    kept += 1
            rate = kept / max(kept + skipped, 1) * 100
            print(f"  {fname:<25}  kept={kept:>9,}  skipped={skipped:>9,}  keep_rate={rate:.1f}%")
            total_kept += kept
            total_skipped += skipped

    total = total_kept + total_skipped
    print(f"\nTotal kept:    {total_kept:>12,}")
    print(f"Total skipped: {total_skipped:>12,}")
    print(f"Keep rate:     {total_kept/max(total,1)*100:.1f}%")
    print(f"Output:        {out_path}")
    print(f"\nNext: python scripts/validate_data.py")


if __name__ == "__main__":
    main()
