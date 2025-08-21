

# -------------------- Dataset --------------------
from typing import List
import torch
from torch.utils.data import Dataset

from tokenizer import Tokenizer


class TextDataset(Dataset):
    """Creates training examples from raw texts using a sliding window.

    Each example is a sequence of token ids of length `seq_len`. If a text is longer than seq_len,
    we create multiple examples by sliding with stride `stride`.
    """

    def __init__(self, texts: List[str], tokenizer: Tokenizer, seq_len: int = 128, stride: int = 64):
        if not tokenizer.frozen:
            raise RuntimeError("tokenizer must have a built vocab")
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.examples = []  # list of lists (token ids)
        for t in texts:
            ids = tokenizer.encode(t)
            if len(ids) == 0:
                continue
            if len(ids) <= seq_len:
                self.examples.append(ids + [tokenizer.vocab[tokenizer.pad_token]] * (seq_len - len(ids)))
            else:
                i = 0
                while i < len(ids):
                    chunk = ids[i : i + seq_len]
                    if len(chunk) < seq_len:
                        chunk = chunk + [tokenizer.vocab[tokenizer.pad_token]] * (seq_len - len(chunk))
                    self.examples.append(chunk)
                    if i + seq_len >= len(ids):
                        break
                    i += stride

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return torch.tensor(self.examples[idx], dtype=torch.long)