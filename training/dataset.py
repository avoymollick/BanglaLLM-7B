"""
BanglaLLM Dataset — Bengali text loader with 4096 context
"""
import os, random, torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer


class BengaliDataset(Dataset):
    def __init__(self, data_path, tokenizer_path, max_len=4096):
        self.max_len = max_len
        tok = Tokenizer.from_file(tokenizer_path) if tokenizer_path.endswith(".json") else               Tokenizer.from_file(os.path.join(tokenizer_path, "tokenizer.json"))
        tok.enable_truncation(max_length=max_len + 1)

        print(f"Loading {data_path} ...")
        self.samples = []
        with open(data_path, encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        
        # Build token sequences of exactly max_len
        buf = []
        bos = tok.token_to_id("<s>") or 1
        eos = tok.token_to_id("</s>") or 2

        for line in lines:
            ids = [bos] + tok.encode(line).ids + [eos]
            buf.extend(ids)
            while len(buf) >= max_len + 1:
                self.samples.append(buf[:max_len + 1])
                buf = buf[max_len // 2:]  # 50% overlap for better learning

        print(f"Loaded {len(self.samples):,} samples (max_len={max_len})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq = torch.tensor(self.samples[idx], dtype=torch.long)
        return seq[:-1], seq[1:]  # input, labels
