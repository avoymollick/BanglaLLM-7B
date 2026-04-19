"""
BanglaLLM Data Downloader v3
All sources verified FREE — no login required
"""
import os, json

def log(m): print(m, flush=True)

def done(path, min_mb=10):
    if os.path.exists(path) and os.path.getsize(path) > min_mb*1024*1024:
        log(f"  Already done ({os.path.getsize(path)//1024//1024}MB) — skipping")
        return True
    return False


def wiki():
    out = "data/raw/wiki.txt"
    log("\n[1] Bengali Wikipedia (143k articles)...")
    if done(out, 10): return
    from datasets import load_dataset
    ds = load_dataset("wikimedia/wikipedia", "20231101.bn", split="train")
    n = 0
    with open(out, "w", encoding="utf-8") as f:
        for row in ds:
            for para in row["text"].split("\n\n"):
                para = para.strip()
                if len(para) > 50:
                    f.write(para + "\n"); n += 1
    log(f"  Done: {n:,} paragraphs")


def textbook():
    out = "data/raw/textbook.txt"
    log("\n[2] NCTB TextBook (163 books)...")
    if done(out, 1): return
    from datasets import load_dataset
    try:
        ds = load_dataset("md-nishat-008/Bangla-TextBook", split="train")
        n = 0
        with open(out, "w", encoding="utf-8") as f:
            for row in ds:
                for v in row.values():
                    if isinstance(v, str) and len(v) > 30:
                        f.write(v.strip() + "\n"); n += 1; break
        log(f"  Done: {n:,} lines")
    except Exception as e:
        log(f"  Failed: {e}")


def indiccorp(max_lines=5_000_000):
    out = "data/raw/indiccorp.txt"
    log(f"\n[3] IndicCorpV2 Bengali (up to {max_lines:,} lines)...")
    if done(out, 100): return
    from datasets import load_dataset
    try:
        ds = load_dataset("ai4bharat/IndicCorpV2", "indiccorp_v2",
                          split="ben_Beng", streaming=True)
        n = 0
        with open(out, "w", encoding="utf-8") as f:
            for row in ds:
                t = row.get("text","").strip()
                if len(t) > 30:
                    f.write(t + "\n"); n += 1
                if n % 500_000 == 0 and n > 0: log(f"  {n:,}...")
                if n >= max_lines: break
        log(f"  Done: {n:,} lines")
    except Exception as e:
        log(f"  Failed: {e}")


def sangraha(max_lines=3_000_000):
    """Sangraha — AI4Bharat curated Bengali. Split is ben not bn."""
    out = "data/raw/sangraha.txt"
    log(f"\n[4] Sangraha Bengali (AI4Bharat, up to {max_lines:,} lines)...")
    if done(out, 100): return
    from datasets import load_dataset
    try:
        ds = load_dataset("ai4bharat/sangraha", "verified",
                          split="ben", streaming=True)
        n = 0
        with open(out, "w", encoding="utf-8") as f:
            for row in ds:
                t = row.get("text","").strip()
                if len(t) > 30:
                    f.write(t + "\n"); n += 1
                if n % 500_000 == 0 and n > 0: log(f"  {n:,}...")
                if n >= max_lines: break
        log(f"  Done: {n:,} lines")
    except Exception as e:
        log(f"  Failed: {e}")


def bn_corpus(max_lines=2_000_000):
    """Bengali corpus from multiple open HF datasets"""
    out = "data/raw/bn_corpus.txt"
    log(f"\n[5] Bengali Corpus (open HF sources)...")
    if done(out, 50): return
    from datasets import load_dataset
    n = 0
    sources = [
        ("csebuetnlp/xlsum", "bengali", "train", "text"),
        ("csebuetnlp/xlsum", "bengali", "validation", "text"),
        ("Helsinki-NLP/opus-100", "bn-en", "train", "translation"),
        ("Muennighoff/flores200", "ben_Beng", "dev", "sentence"),
    ]
    with open(out, "w", encoding="utf-8") as f:
        for src, cfg, split, field in sources:
            try:
                ds = load_dataset(src, cfg, split=split, streaming=True)
                for row in ds:
                    val = row.get(field, "")
                    if isinstance(val, dict):
                        val = val.get("bn", "")
                    val = str(val).strip()
                    if len(val) > 30:
                        f.write(val + "\n"); n += 1
                log(f"  {src}: ok")
            except Exception as e:
                log(f"  {src}: failed ({e})")
    log(f"  Done: {n:,} lines total")


def bn_news(max_lines=2_000_000):
    """Bengali news articles"""
    out = "data/raw/bn_news.txt"
    log(f"\n[6] Bengali News (multiple sources)...")
    if done(out, 50): return
    from datasets import load_dataset
    n = 0
    sources = [
        ("csebuetnlp/xlsum", "bengali", "train"),
        ("SKNahin/bengali-transliteration-data", None, "train"),
    ]
    with open(out, "w", encoding="utf-8") as f:
        for args in [
            ("csebuetnlp/xlsum", "bengali", "train"),
        ]:
            try:
                ds = load_dataset(args[0], args[1], split=args[2], streaming=True)
                for row in ds:
                    for field in ["text", "summary", "article"]:
                        val = row.get(field, "").strip()
                        if len(val) > 50:
                            f.write(val + "\n"); n += 1
                log(f"  {args[0]}: ok")
            except Exception as e:
                log(f"  {args[0]}: {e}")

        # Try bengali books
        try:
            ds = load_dataset("BengaliAI/bengali_text", split="train", streaming=True)
            for row in ds:
                t = row.get("text","").strip()
                if len(t) > 30:
                    f.write(t + "\n"); n += 1
                if n >= max_lines: break
            log(f"  BengaliAI/bengali_text: ok")
        except Exception as e:
            log(f"  BengaliAI/bengali_text: {e}")

        # Try bn_wiki_abstractive
        try:
            ds = load_dataset("csebuetnlp/bengali-abstractive-qa", split="train", streaming=True)
            for row in ds:
                for field in ["context", "question", "answer"]:
                    t = row.get(field,"").strip()
                    if len(t) > 30:
                        f.write(t + "\n"); n += 1
            log(f"  csebuetnlp/bengali-abstractive-qa: ok")
        except Exception as e:
            log(f"  csebuetnlp/bengali-abstractive-qa: {e}")

    log(f"  Done: {n:,} lines")


def instruct():
    out = "data/instruct/instruct.jsonl"
    log("\n[7] Bangla-Instruct (342k pairs)...")
    os.makedirs("data/instruct", exist_ok=True)
    if done(out, 10): return
    from datasets import load_dataset
    try:
        ds = load_dataset("md-nishat-008/Bangla-Instruct", split="train")
        n = 0
        with open(out, "w", encoding="utf-8") as f:
            for row in ds:
                ins  = (row.get("instruction") or row.get("prompt") or "").strip()
                resp = (row.get("output") or row.get("response") or "").strip()
                inp  = (row.get("input") or "").strip()
                if ins and resp:
                    f.write(json.dumps({"instruction":ins,"input":inp,"output":resp},
                                       ensure_ascii=False)+"\n"); n += 1
        log(f"  Done: {n:,} pairs")
    except Exception as e:
        log(f"  Failed: {e}")


def alpaca():
    out = "data/instruct/alpaca.jsonl"
    log("\n[8] BanglaLLM Alpaca (52k pairs)...")
    if done(out, 1): return
    from datasets import load_dataset
    try:
        ds = load_dataset("BanglaLLM/bangla-alpaca", split="train")
        n = 0
        with open(out, "w", encoding="utf-8") as f:
            for row in ds:
                ins  = (row.get("instruction") or "").strip()
                resp = (row.get("output") or "").strip()
                inp  = (row.get("input") or "").strip()
                if ins and resp:
                    f.write(json.dumps({"instruction":ins,"input":inp,"output":resp},
                                       ensure_ascii=False)+"\n"); n += 1
        log(f"  Done: {n:,} pairs")
    except Exception as e:
        log(f"  Failed: {e}")


def summary():
    log("\n" + "="*60)
    log("DOWNLOAD SUMMARY v3")
    log("="*60)
    files = [
        ("data/raw/wiki.txt",           "Wikipedia"),
        ("data/raw/textbook.txt",       "TextBook"),
        ("data/raw/indiccorp.txt",      "IndicCorpV2"),
        ("data/raw/sangraha.txt",       "Sangraha"),
        ("data/raw/bn_corpus.txt",      "BN Corpus"),
        ("data/raw/bn_news.txt",        "BN News"),
        ("data/instruct/instruct.jsonl","Bangla-Instruct"),
        ("data/instruct/alpaca.jsonl",  "Alpaca"),
    ]
    total = 0
    for path, label in files:
        if os.path.exists(path):
            mb = os.path.getsize(path)/1024/1024
            total += mb
            log(f"  OK  {label:<25} {mb:8.0f} MB")
        else:
            log(f"  --  {label:<25}  NOT DOWNLOADED")
    log(f"\n  Total: {total:.0f} MB ({total/1024:.1f} GB)")
    log("\nNext: python scripts/clean_data.py")
    log("="*60)


if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/instruct", exist_ok=True)
    log("BanglaLLM Data Downloader v3")
    log("All sources: FREE, no login required")
    log("="*60)
    wiki()
    textbook()
    indiccorp()
    sangraha()
    bn_corpus()
    bn_news()
    instruct()
    alpaca()
    summary()
