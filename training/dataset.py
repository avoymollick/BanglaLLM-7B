import os, torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer

class BengaliDataset(Dataset):
    def __init__(self, data_path, tokenizer_path, max_len=4096, max_samples=50000):
        cache = data_path.replace('.txt', f'_cached_{max_len}.pt')
        tok_file = tokenizer_path if tokenizer_path.endswith('.json') else os.path.join(tokenizer_path,'tokenizer.json')
        tok = Tokenizer.from_file(tok_file)
        bos = tok.token_to_id('<s>') or 1
        eos = tok.token_to_id('</s>') or 2
        if os.path.exists(cache):
            print(f'Loading from cache: {cache}')
            self.samples = torch.load(cache, weights_only=False)
            print(f'Loaded {len(self.samples):,} samples instantly!')
            return
        print(f'Building dataset (max {max_samples:,} samples)...')
        self.samples = []
        buf = []
        n = 0
        with open(data_path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                buf.extend([bos] + tok.encode(line).ids + [eos])
                while len(buf) >= max_len + 1:
                    self.samples.append(buf[:max_len+1])
                    buf = buf[max_len:]
                    if len(self.samples) >= max_samples: break
                if len(self.samples) >= max_samples: break
                n += 1
                if n % 100000 == 0:
                    print(f'  {n:,} lines -> {len(self.samples):,} samples...')
        print(f'Saving cache to {cache}...')
        torch.save(self.samples, cache)
        print(f'Dataset ready: {len(self.samples):,} samples. Saved permanently.')
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        seq = torch.tensor(self.samples[idx], dtype=torch.long)
        return seq[:-1], seq[1:]
