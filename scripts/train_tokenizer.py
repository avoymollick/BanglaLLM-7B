"""
BanglaLLM Best Bengali Tokenizer
64,000 vocab BPE trained specifically for Bengali morphology

Why this is the best possible Bengali tokenizer:
1. csebuetnlp NFKC normalizer — Bengali-specific Unicode normalization
   handles Hasanta, Nukta, conjunct consonants correctly
2. ByteLevel pre-tokenizer — handles any Unicode without OOV
3. BPE algorithm — learns Bengali morphemes naturally
4. 64,000 vocab — enough to cover Bengali morphology
5. Special tokens for instruction tuning ready
6. Trained on full Bengali corpus — not byte-fallback English tokenizer
"""
import os, sys
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers, processors, decoders

# Try to import csebuetnlp normalizer (best for Bengali)
try:
    from normalizer import normalize as bn_normalize
    HAS_BN_NORM = True
    print("csebuetnlp normalizer: LOADED (best Bengali Unicode normalization)")
except ImportError:
    HAS_BN_NORM = False
    print("csebuetnlp normalizer: not found (using Unicode NFC)")


SPECIAL_TOKENS = [
    "<unk>", "<s>", "</s>", "<pad>", "<mask>",
    "<|system|>", "<|user|>", "<|assistant|>",
]

VOCAB_SIZE = 64000


def normalize_text(text: str) -> str:
    """Best possible Bengali normalization"""
    if HAS_BN_NORM:
        try:
            return bn_normalize(text)
        except Exception:
            pass
    # Fallback: Unicode NFC normalization
    import unicodedata
    return unicodedata.normalize("NFC", text)


def prepare_corpus(data_path: str, out_path: str, max_lines: int = 8_000_000):
    """Normalize and prepare training corpus for tokenizer"""
    print(f"[1/3] Normalizing corpus (max {max_lines:,} lines)...")
    n = 0
    with open(data_path, encoding="utf-8") as fin,          open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if len(line) < 10:
                continue
            normalized = normalize_text(line)
            if normalized:
                fout.write(normalized + "\n")
                n += 1
            if n % 1_000_000 == 0 and n > 0:
                print(f"   {n:,} lines...")
            if n >= max_lines:
                break
    print(f"   Done: {n:,} lines normalized")
    return out_path


def train_tokenizer(corpus_path: str, output_dir: str):
    """Train best Bengali BPE tokenizer"""
    print(f"[2/3] Training BPE tokenizer (vocab={VOCAB_SIZE:,})...")
    os.makedirs(output_dir, exist_ok=True)

    # BPE model
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))

    # ByteLevel pre-tokenizer — handles all Unicode without OOV
    # This is the same approach used by GPT-2/LLaMA/all modern LLMs
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # ByteLevel decoder for correct decoding
    tokenizer.decoder = decoders.ByteLevel()

    # BPE trainer
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=2,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    # Train
    tokenizer.train([corpus_path], trainer=trainer)

    # Post-processor: add BOS/EOS automatically
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<s> $A </s>",
        special_tokens=[
            ("<s>",  tokenizer.token_to_id("<s>")),
            ("</s>", tokenizer.token_to_id("</s>")),
        ],
    )

    print(f"   Vocab size: {tokenizer.get_vocab_size():,}")
    return tokenizer


def save_tokenizer(tokenizer, output_dir: str):
    """Save tokenizer in HuggingFace format"""
    print(f"[3/3] Saving tokenizer to {output_dir}/...")
    os.makedirs(output_dir, exist_ok=True)
    tokenizer.save(os.path.join(output_dir, "tokenizer.json"))

    # Save config
    import json
    config = {
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
        "mask_token": "<mask>",
        "model_max_length": 4096,
        "tokenizer_class": "PreTrainedTokenizerFast",
        "vocab_size": VOCAB_SIZE,
    }
    with open(os.path.join(output_dir, "tokenizer_config.json"), "w") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    special_tokens = {
        "bos_token": {"content": "<s>", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False},
        "eos_token": {"content": "</s>", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False},
        "unk_token": {"content": "<unk>", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False},
        "pad_token": {"content": "<pad>", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False},
        "mask_token": {"content": "<mask>", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False},
    }
    with open(os.path.join(output_dir, "special_tokens_map.json"), "w") as f:
        json.dump(special_tokens, f, ensure_ascii=False, indent=2)

    print(f"   Saved: tokenizer.json, tokenizer_config.json, special_tokens_map.json")


def evaluate_tokenizer(tokenizer):
    """Evaluate tokenizer quality"""
    test_cases = [
        ("simple sentence",      "বাংলাদেশ একটি সুন্দর দেশ।"),
        ("verb inflections",     "যাচ্ছি যাচ্ছো যাচ্ছে যাচ্ছিলাম যাচ্ছিলেন"),
        ("conjuncts + noun",     "বাংলাদেশের স্বাধীনতা সংগ্রামের ইতিহাস"),
        ("literature",           "আমার সোনার বাংলা আমি তোমায় ভালোবাসি"),
        ("numbers + mixed",      "২০২৪ সালে বাংলাদেশের জনসংখ্যা ১৭ কোটি"),
        ("long sentence",        "বাংলাদেশ দক্ষিণ এশিয়ার একটি স্বাধীন সার্বভৌম রাষ্ট্র যার রাজধানী ঢাকা"),
    ]

    print("\n=== Tokenizer Evaluation ===")
    total_tok = 0
    total_words = 0
    all_pass = True

    for name, text in test_cases:
        enc = tokenizer.encode(text)
        decoded = tokenizer.decode(enc.ids, skip_special_tokens=True)

        # Roundtrip check
        rt_pass = decoded.strip() == text.strip()
        if not rt_pass:
            all_pass = False

        words = len(text.split())
        toks = len(enc.ids)
        fertility = toks / max(words, 1)
        total_tok += toks
        total_words += words

        status = "PASS" if rt_pass else "FAIL"
        print(f"  [{name:<20}]  tokens={toks:3d}  fertility={fertility:.1f}  roundtrip={status}")

    avg_fertility = total_tok / max(total_words, 1)
    print(f"\n  Average fertility: {avg_fertility:.2f} tokens/word")
    print(f"  (target: ~2.0 with full 50GB corpus | English LLMs score 3.5+ on Bengali)")
    print(f"  Roundtrip: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    print("\nTokenizer ready!")


if __name__ == "__main__":
    DATA   = sys.argv[1] if len(sys.argv) > 1 else "data/final/train.txt"
    OUTDIR = sys.argv[2] if len(sys.argv) > 2 else "tokenizer"
    NORM   = "data/final/normalized_for_tokenizer.txt"

    if not os.path.exists(DATA):
        print(f"ERROR: {DATA} not found. Run download_data.py and clean_data.py first.")
        sys.exit(1)

    prepare_corpus(DATA, NORM)
    tok = train_tokenizer(NORM, OUTDIR)
    save_tokenizer(tok, OUTDIR)
    evaluate_tokenizer(tok)

    # Cleanup temp file
    if os.path.exists(NORM):
        os.remove(NORM)
