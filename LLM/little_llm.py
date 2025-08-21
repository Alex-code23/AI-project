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

from dataset import TextDataset
from tokenizer import Tokenizer
from utils import load_texts_from_path






# -------------------- Transformer & MoE --------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        batch, seq, _ = x.shape
        qkv = self.qkv_proj(x)
        qkv = qkv.view(batch, seq, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attn_mask is not None:
            scores = scores + attn_mask
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(batch, seq, self.d_model)
        out = self.out_proj(context)
        return out


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, activation=F.gelu):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class MoE(nn.Module):
    def __init__(self, d_model: int, d_ff: int, n_experts: int = 4, k: int = 2, dropout: float = 0.1, activation=F.gelu):
        super().__init__()
        assert k >= 1 and k <= n_experts, "k must be between 1 and n_experts"
        self.d_model = d_model
        self.n_experts = n_experts
        self.k = k
        self.experts = nn.ModuleList([FeedForward(d_model, d_ff, dropout, activation) for _ in range(n_experts)])
        self.gate = nn.Linear(d_model, n_experts)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, d = x.shape
        logits = self.gate(x)               # [batch, seq, n_experts]
        probs = F.softmax(logits, dim=-1)   # [batch, seq, n_experts]
        if self.k < self.n_experts:
            # Sélection des k meilleurs experts
            topk_vals, topk_idx = torch.topk(probs, self.k, dim=-1)
            mask = torch.zeros_like(probs)
            mask.scatter_(-1, topk_idx, 1.0)
            probs = probs * mask
            denom = probs.sum(dim=-1, keepdim=True)
            denom = denom + (denom == 0).float()
            probs = probs / denom
        expert_outputs = []
        for expert in self.experts:
            # Chaque expert traite l’entrée entière (même si son poids est 0).
            expert_outputs.append(expert(x))
        expert_stack = torch.stack(expert_outputs, dim=0)   # [n_experts, batch, seq, d_model]
        expert_stack = expert_stack.permute(1, 2, 0, 3)     # [batch, seq, n_experts, d_model]
        probs_expanded = probs.unsqueeze(-1)        #  [batch, seq, n_experts, 1]
        moe_out = torch.sum(probs_expanded * expert_stack, dim=2)
        moe_out = self.dropout(moe_out)
        return moe_out      #  [batch, seq, d_model]


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1, use_moe: bool = False, moe_params: Optional[dict] = None):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.use_moe = use_moe
        if use_moe:
            moe_params = moe_params or {}
            self.ff = MoE(d_model=d_model, d_ff=d_ff, **moe_params)
        else:
            self.ff = FeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.ln1(x), attn_mask=attn_mask))
        x = x + self.dropout(self.ff(self.ln2(x)))
        return x


def build_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    # returns shape (1, 1, seq_len, seq_len) with 0 for allowed and -1e9 for masked
    mask = torch.tril(torch.ones((seq_len, seq_len), device=device)).unsqueeze(0).unsqueeze(0)
    mask = (1.0 - mask) * -1e9
    return mask


class SimpleLLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 512, n_heads: int = 8, d_ff: int = 2048, n_layers: int = 12,
                 max_seq_len: int = 1024, dropout: float = 0.1, moe_every: int = 2, n_experts: int = 6, moe_k: int = 2):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model, max_len=max_seq_len)
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            use_moe = ((i % moe_every) == (moe_every - 1)) and (n_experts > 1)
            moe_params = {"n_experts": n_experts, "k": moe_k, "dropout": dropout} if use_moe else None
            blk = TransformerBlock(d_model, n_heads, d_ff, dropout, use_moe, moe_params)
            self.layers.append(blk)
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self._tie_weights()

    def _tie_weights(self):
        try:
            self.lm_head.weight = self.token_emb.weight
        except Exception:
            pass

    def forward(self, input_ids: torch.LongTensor, attn_mask: Optional[torch.Tensor] = None):
        x = self.token_emb(input_ids) * math.sqrt(self.d_model)
        x = self.pos_emb(x)
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

    @torch.no_grad()
    def generate(self, input_ids: torch.LongTensor, max_new_tokens: int = 50, eos_token_id: Optional[int] = None):
        device = input_ids.device
        for _ in range(max_new_tokens):
            seq_len = input_ids.size(1)
            attn_mask = build_causal_mask(seq_len, device=device)
            logits = self.forward(input_ids, attn_mask=attn_mask)
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            if eos_token_id is not None:
                if (next_token == eos_token_id).all():
                    break
        return input_ids

    def train_model(self,
                    texts: Optional[List[str]] = None,
                    dataset: Optional[TextDataset] = None,
                    tokenizer: Optional[Tokenizer] = None,
                    epochs: int = 3,
                    batch_size: int = 16,
                    lr: float = 5e-5,
                    device: Optional[str] = None,
                    seq_len: int = 128,
                    stride: int = 64,
                    gradient_clip: float = 1.0,
                    save_dir: Optional[str] = None,
                    save_every_steps: int = 1000,
                    use_amp: bool = True,
                    print_every: int = 50):
        """
        Train the model on raw texts or a prebuilt TextDataset.

        - If `texts` is provided, `tokenizer` must be provided and the dataset will be built.
        - `device` can be 'cpu' or 'cuda'. If None, chosen automatically.
        - Checkpoints (model + optimizer state) are saved to `save_dir` if provided.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device)
        self.to(device)

        if dataset is None:
            if texts is None or tokenizer is None:
                raise ValueError("Provide either dataset or (texts and tokenizer)")
            dataset = TextDataset(texts, tokenizer, seq_len=seq_len, stride=stride)

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        scaler = torch.amp.GradScaler(device = device ,enabled=(use_amp and device.type == "cuda"))
        global_step = 0

        # metrics to plot
        metrics = {}
        metrics["training_loss"] = []
        metrics["next_epoch"] = []

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)

        self.train()
        for epoch in range(1, epochs + 1):
            running_loss = 0.0
            for step, batch in enumerate(dataloader, start=1):
                input_ids = batch.to(device)  # (batch, seq)
                seq = input_ids.size(1)
                attn_mask = build_causal_mask(seq, device=device)
                labels = input_ids.clone()
                optimizer.zero_grad()

                with torch.amp.autocast(device_type=device.type, enabled=(use_amp and device.type == "cuda")):
                    logits = self.forward(input_ids, attn_mask=attn_mask)
                    # shift logits and labels for next-token prediction
                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_labels = labels[:, 1:].contiguous()
                    loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=tokenizer.vocab[tokenizer.pad_token] if tokenizer else -100)

                scaler.scale(loss).backward()
                # gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.parameters(), gradient_clip)
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item()
                global_step += 1

                
                if global_step % print_every == 0:
                    # add new values metrics 
                    metrics["training_loss"].append(running_loss)

                    avg = running_loss / print_every
                    print(f"Epoch {epoch} step {global_step}: avg_loss={avg:.4f}")
                    running_loss = 0  

            metrics["next_epoch"].append(global_step)
            # metrics        
            # saving plot
            self.figure_training(metrics, save_dir=save_dir, print_every=print_every)

            print("="*40)
            print(f"test epoch: {epoch}")
            seed = torch.tensor([tokenizer.encode("A ready banquet on the turf")], dtype=torch.long)
            out = model.generate(seed, max_new_tokens=50)
            print("generated ids:", out.tolist())
            print("decoded:", tokenizer.decode(out[0].tolist()))
            print("="*40)
                             

        # save 
        if save_dir is not None:
            model.save("checkpoints/simplellm.pt", tokenizer=tokenizer, step=global_step)

        print("Training complete")

    def save(self, path: str, tokenizer: Optional[Tokenizer] = None, optimizer: Optional[torch.optim.Optimizer] = None, step: Optional[int] = None):
        """
        Save model state, optionally tokenizer and optimizer.
        """
        ckpt = {
            "model_state": self.state_dict(),
        }
        if tokenizer is not None:
            ckpt["tokenizer"] = tokenizer.vocab
        if optimizer is not None:
            ckpt["optimizer_state"] = optimizer.state_dict()
        if step is not None:
            ckpt["step"] = step

        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(ckpt, path)
        print(f"Saved model checkpoint to {path}")

    @classmethod
    def load(cls, path: str, **model_kwargs):
        """
        Load a model from a checkpoint file.
        `model_kwargs` are passed to the model constructor.
        Returns: model instance, optionally tokenizer dict if present
        """
        ckpt = torch.load(path, map_location="cpu")
        model = cls(**model_kwargs)
        model.load_state_dict(ckpt["model_state"])
        tokenizer_vocab = ckpt.get("tokenizer", None)
        print(f"Loaded model from {path}")
        return model, tokenizer_vocab
    
    def count_parameters(self, trainable: bool = True) -> int:
        """Return number of parameters (trainable if trainable=True)."""
        if trainable:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
    
    def figure_training(self, metrics:dict, save_dir:str, print_every:int):
        training_loss = metrics["training_loss"]
        next_epoch = metrics["next_epoch"]

        fig, axes = plt.subplots(1, 1, figsize=(12, 8))

        ax = axes
        ax.plot(
            np.linspace(0,len(training_loss)*print_every, len(training_loss) ),
            training_loss,
            "-o",                # ligne + points
            markersize=4,
            linewidth=1,
            label="training_loss per episode"
        )

        size_window = 3
        ma = moving_average(training_loss, window=size_window)
        if ma.size > 0:
            ax.plot(np.linspace(0,len(ma)*print_every,len(ma) ) + (size_window - 1), 
                    ma, label="Moving average (10)", linewidth=2)

        for step in next_epoch:
            ax.axvline(x=step, color="red", linestyle="--", alpha=0.7, label="Epoch boundary")

        # éviter répétition de légende si plusieurs epochs
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys())

        ax.set_xlabel("Steps")
        ax.set_ylabel("training_loss")
        ax.set_yscale("log")
        ax.set_title("training_loss per step")
        ax.grid(True)

        plt.tight_layout()
        plt.savefig(save_dir/"test.png")
        plt.close()


def moving_average(x, window):
    if len(x) < 1:
        return np.array([])
    window = max(1, int(window))
    return np.convolve(x, np.ones(window)/window, mode='valid')
    
def parameter_summary(model: nn.Module, trainable: bool = True, top_level: bool = True, unique: bool = True):
    """
    Print and return a summary dict {group_name: count}.
    - top_level: group by the first name component before '.' (e.g. 'layers', 'token_emb', 'lm_head')
    - unique: avoid double-counting shared Parameter objects
    """
    seen = set()
    groups = defaultdict(int)
    total = 0

    for name, p in model.named_parameters():
        if unique:
            if id(p) in seen:
                continue
            seen.add(id(p))

        if trainable and not p.requires_grad:
            continue

        if top_level:
            group = name.split('.')[0]
        else:
            group = name  # no grouping

        n = p.numel()
        groups[group] += n
        total += n

    # sort groups by size
    sorted_groups = sorted(groups.items(), key=lambda x: -x[1])
    for g, c in sorted_groups:
        print(f"{g:20s} : {c:,} params ({c/1e6:.3f} M)")

    print("-" * 40)
    print(f"Total params (trainable={trainable}, unique={unique}): {total:,} ({total/1e6:.3f} M)")
    return dict(groups)


# -------------------- Example usage --------------------
if __name__ == "__main__":
    PATH_FOLDER = Path(r"C:/Users/alexa/Documents/GitHub/AI-project/LLM")
    
    # tiny smoke test
    texts = load_texts_from_path(PATH_FOLDER/"my_little_book.txt", split_mode="paragraph")
    print(f"Loaded {len(texts)} paragraphs")

    tokenizer = Tokenizer(do_lower=True, vocab_mode="word")

    tokenizer.build_vocab(texts, max_vocab=100000, min_freq=1)
    tokenizer.save(PATH_FOLDER/"tokens.json")
    print(f"Size vocab : {len(tokenizer.vocab)}")
    ds = TextDataset(texts, tokenizer, seq_len=32, stride=8)
    print(f"Loaded {len(ds)} examples")

    model = SimpleLLM(vocab_size=len(tokenizer.vocab), d_model=256, n_heads=8, d_ff=512, n_layers=10, n_experts=6, moe_k=2)
    
    summary = parameter_summary(model, trainable=True, top_level=True, unique=True)
    
    model.train_model(texts=texts, tokenizer=tokenizer, epochs=30, batch_size=32, lr=1e-4, seq_len=32, save_dir=PATH_FOLDER, print_every=10, use_amp=False)

    # Load later
    # model, vocab = SimpleLLM.load("checkpoints/simplellm.pt", vocab_size=len(tokenizer.vocab), d_model=256, n_heads=4, d_ff=512, n_layers=6, n_experts=4, moe_k=2)
    # tokenizer = SimpleTokenizer.load(PATH_FOLDER/"tokens.json")

    # generate
    print("="*40)
    print("Final test")
    seed = torch.tensor([tokenizer.encode("A ready banquet on the turf")], dtype=torch.long)
    out = model.generate(seed, max_new_tokens=50)
    print("generated ids:", out.tolist())
    print("decoded:", tokenizer.decode(out[0].tolist()))

    
