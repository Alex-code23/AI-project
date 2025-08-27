"""
Lightweight example implementation of a Transformer-based LLM with a Mixture-of-Experts (MoE)
module, plus a simple tokenizer and a training method.

Features added compared to previous version:
- Simple whitespace+punctuation tokenizer with vocab building, encode/decode, save/load
- TextDataset for sliding-window example creation
- A `train_model` method on SimpleLLM for training on plain text data (with AdamW, amp support,
  gradient clipping, checkpointing)

Note: This is educational code and not production-optimized. Use small configs to test locally.
"""

from pathlib import Path
from typing import Optional, List, Iterable, Tuple
import math
import os
import re
import json
import random
from collections import Counter, defaultdict

from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from cot_helpers import ensure_special_tokens
from little_llm import SimpleLLM, parameter_summary, reward_exact_match
from dataset import TextDataset
from tokenizer import Tokenizer
from utils import load_texts_from_path


# -------------------- Example --------------------
if __name__ == "__main__":
    PATH_FOLDER = Path(r"C:/Users/alexander.zainoun/Documents/GitHub/AI-project/LLM")
    
    # tiny smoke test
    texts = load_texts_from_path(PATH_FOLDER/"my_little_book.txt", split_mode="paragraph")
    print(f"Loaded {len(texts)} paragraphs")


    tokenizer = Tokenizer(do_lower=True, vocab_mode="word")

    tokenizer.build_vocab(texts, max_vocab=100000, min_freq=1)
    tokenizer.save(PATH_FOLDER/"tokens.json")
    print(f"Size vocab : {len(tokenizer.vocab)}")
    ensure_special_tokens(tokenizer, ["<COT>", "</COT>", "<ANS>"])
    print(f"Size vocab with COT: {len(tokenizer.vocab)}")
    
    ds = TextDataset(texts, tokenizer, seq_len=32, stride=8)
    print(f"Loaded {len(ds)} examples")

    examples = [
        {"prompt": "Q: What is 12 * 9?", "cot": "12 * 9 = 12*(10-1) = 120 - 12 = 108", "answer": "108"},
        {"prompt": "Q: If you have 3 apples and get 2 more, how many?", "cot": "Start with 3, add 2 => 3 + 2 = 5", "answer": "5"},
        {"prompt": "Q: What is 7 * 8?", "cot": "7 * 8 = 7*(10-2) = 70 - 14 = 56", "answer": "56"},
        {"prompt": "Q: What is 15 divided by 3?", "cot": "15 divided by 3 => 15 / 3 = 5", "answer": "5"},
        {"prompt": "Q: What is 25% of 80?", "cot": "25% = 1/4, so 80 * 1/4 = 80 / 4 = 20", "answer": "20"},
        {"prompt": "Q: A car drives at 60 km/h for 2.5 hours. How far does it travel?", "cot": "distance = speed * time = 60 * 2.5 = 150", "answer": "150 km"},
        {"prompt": "Q: Alice has 3 red marbles and 4 blue marbles. What's the probability she draws a red marble?", "cot": "Total marbles = 3 + 4 = 7. Favorable = 3. Probability = 3/7", "answer": "3/7"},
        {"prompt": "Q: Double a number and add 3 gives 11. What is the number?", "cot": "Let x be number. 2x + 3 = 11 -> 2x = 8 -> x = 4", "answer": "4"},
        {"prompt": "Q: John is 3 years older than Mary. If Mary is 10, how old is John?", "cot": "John = Mary + 3 = 10 + 3 = 13", "answer": "13"},
        {"prompt": "Q: What is the area of a rectangle 5 by 7?", "cot": "Area = width * height = 5 * 7 = 35", "answer": "35"},
        {"prompt": "Q: If you save $20 every week for 6 weeks, how much do you have?", "cot": "20 per week * 6 weeks = 20 * 6 = 120", "answer": "120"},
        {"prompt": "Q: Convert 2 hours to minutes.", "cot": "1 hour = 60 minutes, so 2 hours = 2 * 60 = 120", "answer": "120 minutes"},
        {"prompt": "Q: A pen costs $1.50. You buy 4 pens. What is the total cost?", "cot": "Cost = 1.50 * 4 = 6.00", "answer": "6.00"},
        {"prompt": "Q: What is the average of 2, 4, 6, 8?", "cot": "Sum = 2+4+6+8 = 20. Average = 20 / 4 = 5", "answer": "5"},
        {"prompt": "Q: Next number in the sequence 2, 4, 8, 16, ...?", "cot": "Sequence doubles each time (powers of 2). Next = 16 * 2 = 32", "answer": "32"},
        {"prompt": "Q: Two fair coins are tossed. What's the probability of two heads?", "cot": "Total outcomes = 4. Only HH is two heads => 1/4", "answer": "1/4"},
        {"prompt": "Q: If a recipe needs 3 eggs per cake, how many eggs for 5 cakes?", "cot": "3 eggs * 5 cakes = 15 eggs", "answer": "15"},
        {"prompt": "Q: A car uses 8 L per 100 km. How much fuel for 250 km?", "cot": "Fuel = 8/100 * 250 = 8 * 2.5 = 20", "answer": "20 L"},
        {"prompt": "Q: You have $50 and spend 15%. How much remains?", "cot": "Spend = 50 * 0.15 = 7.5. Remaining = 50 - 7.5 = 42.5", "answer": "42.5"},
        {"prompt": "Q: Compute 9 choose 2 (combinations of 9 items taken 2).", "cot": "nC2 = n*(n-1)/2 = 9*8/2 = 72/2 = 36", "answer": "36"}
    ]

    model = SimpleLLM(vocab_size=len(tokenizer.vocab), d_model=256, n_heads=8, d_ff=512, n_layers=10, n_experts=6, moe_k=2)
    
    summary = parameter_summary(model, trainable=True, top_level=True, unique=True)
    
    model.train_model(dataset=ds, tokenizer=tokenizer, epochs=1, batch_size=64, lr=1e-4, seq_len=32, save_dir=PATH_FOLDER, print_every=10, use_amp=False)
    model.train_with_cot(examples, tokenizer, epochs=10, batch_size=4, seq_len=64, mask_rationale=True, lr=1e-4, print_every=20)
    # model.train_with_cot_rl(examples, tokenizer, reward_fn=reward_exact_match, epochs=10, batch_size=4, lr=1e-4, print_every=20)

    # Load later
    # model, vocab = SimpleLLM.load("checkpoints/simplellm.pt", vocab_size=len(tokenizer.vocab), d_model=256, n_heads=4, d_ff=512, n_layers=6, n_experts=4, moe_k=2)
    # tokenizer = SimpleTokenizer.load(PATH_FOLDER/"tokens.json")

    # generate
    prompt = "Q: What is 7 * 8?"
    seed_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
    out = model.generate_with_cot(seed_ids, tokenizer, max_new_tokens=120, stop_on_token="</COT>", temperature=0.8, top_k=40)
    print("decoded:", tokenizer.decode(out[0].tolist()))

    
