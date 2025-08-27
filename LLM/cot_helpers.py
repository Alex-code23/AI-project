

# ---------- Chain-of-thought helpers (add to your file) ----------

from collections.abc import Iterable
from typing import Dict

import torch


def ensure_special_tokens(tokenizer, special_tokens: Iterable[str]):
    """
    Ensure tokens are present in tokenizer.vocab. If not present, append them to vocab
    with consecutive ids. This mutates tokenizer.vocab in-place; adapt if your tokenizer
    needs a different method (e.g. tokenizer.add_special_tokens).
    """
    vocab = tokenizer.vocab
    max_id = max(vocab.values()) if len(vocab) > 0 else -1
    for tok in special_tokens:
        if tok not in vocab:
            max_id += 1
            vocab[tok] = max_id
    # If tokenizer stores reverse mapping, update it too (optional)
    if hasattr(tokenizer, "id_to_token"):
        tokenizer.id_to_token = {i: t for t, i in vocab.items()}


def format_cot_example(prompt: str, cot: str, answer: str,
                    cot_start: str = "<COT>", cot_end: str = "</COT>", ans_tok: str = "<ANS>") -> str:
    """
    Build a single training string with an explicit chain-of-thought section.
    Example output:
    "Q: ... <COT> reasoning steps ... </COT> <ANS> final answer"
    """
    return f"{prompt} {cot_start} {cot} {cot_end} {ans_tok} {answer}"


def build_cot_texts(examples: Iterable[Dict[str,str]], tokenizer,
                    cot_start="<COT>", cot_end="</COT>", ans_tok="<ANS>") -> list:
    """
    Given examples as dicts {'prompt':.., 'cot':.., 'answer':..'}, return list of formatted texts
    and ensure special tokens exist in tokenizer vocab.
    """
    ensure_special_tokens(tokenizer, [cot_start, cot_end, ans_tok])
    texts = []
    for ex in examples:
        texts.append(format_cot_example(ex["prompt"], ex["cot"], ex["answer"], cot_start, cot_end, ans_tok))
    return texts


# Add methods to SimpleLLM
def _mask_rationale_in_shift_labels(labels: torch.LongTensor,
                                    shift_labels: torch.LongTensor,
                                    tokenizer,
                                    cot_start: str = "<COT>",
                                    cot_end: str = "</COT>"):
    """
    In-place set shift_labels positions that correspond to tokens inside the CoT
    content to -100 so cross_entropy ignores them.
    labels: original input_ids (batch, seq)
    shift_labels: labels[:, 1:] (batch, seq-1) - predictions for next tokens
    """
    start_id = tokenizer.vocab.get(cot_start, None)
    end_id = tokenizer.vocab.get(cot_end, None)
    if start_id is None or end_id is None:
        return  # nothing to mask

    bsz, seq = labels.size()
    seq_shift = shift_labels.size(1)
    for i in range(bsz):
        seq_tokens = labels[i]  # shape (seq,)
        # find first occurrence in this window (may be absent)
        starts = (seq_tokens == start_id).nonzero(as_tuple=True)[0]
        ends = (seq_tokens == end_id).nonzero(as_tuple=True)[0]
        if starts.numel() == 0 or ends.numel() == 0:
            continue
        start_pos = int(starts[0].item())
        end_pos = int(ends[0].item())
        # content tokens are positions p in (start_pos, end_pos) (exclusive of markers)
        # prediction index for token at position p is p-1 in shift_labels
        mask_positions = []
        for p in range(start_pos + 1, end_pos):
            idx = p - 1
            if 0 <= idx < seq_shift:
                mask_positions.append(idx)
        if len(mask_positions) > 0:
            shift_labels[i, mask_positions] = -100