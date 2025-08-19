

# -------------------- Tokenizer --------------------
from collections import Counter
import json
import re
from typing import Iterable, List


class SimpleTokenizer:
    """A minimal tokenizer for experimentation.

    Behavior:
    - Lowercases text (configurable)
    - Splits on whitespace and punctuation
    - Builds a vocabulary from training texts with a min frequency and max vocab size
    - Provides encode/decode, save/load

    This is intentionally simple — for production use HuggingFace tokenizers or sentencepiece.
    """

    _split_re = re.compile(r"(\w+|[^\w\s])", flags=re.UNICODE)
    # \w+ → attrape une suite d’un ou plusieurs caractères « word » (lettres, chiffres, _). Avec flags=re.UNICODE ça inclut les lettres accentuées et autres lettres Unicode (ex. Café reste un seul token).

    # [^\w\s] → attrape un seul caractère qui n’est ni un caractère « word » ni un espace — en pratique c’est la ponctuation et les symboles (, . ! ? - ' " 🙂, etc.).

    # L’alternative (\w+|[^\w\s]) signifie : soit un mot entier, soit un caractère de ponctuation.

    # Les parenthèses forcent un groupe capturant, donc re.findall renverra la liste des tokens capturés (mots et ponctuation).

    # Les espaces et autres caractères de séparation ne sont pas capturés (donc ignorés).
    
    # print(_split_re.findall("Hello, world!"))     # ['Hello', ',', 'world', '!']
    # print(_split_re.findall("Café au lait."))     # ['Café', 'au', 'lait', '.']
    # print(_split_re.findall("can't"))             # ['can', "'", 't']
    # print(_split_re.findall("hello-world"))       # ['hello', '-', 'world']
    # print(_split_re.findall("Wait..."))           # ['Wait', '.', '.', '.']
    # print(_split_re.findall("😀👍"))             # ['😀', '👍']   # emojis pris comme symboles

    def __init__(self, do_lower: bool = True, unk_token: str = "<unk>", pad_token: str = "<pad>"):
        self.do_lower = do_lower
        self.unk_token = unk_token
        self.pad_token = pad_token

        self.vocab = {}            # token -> id
        self.id_to_token = {}      # id -> token
        self.frozen = False

    def tokenize(self, text: str) -> List[str]:
        if self.do_lower:
            text = text.lower()
        tokens = self._split_re.findall(text)
        return tokens

    def build_vocab(self, texts: Iterable[str], max_vocab: int = 30000, min_freq: int = 2):
        if self.frozen:
            raise RuntimeError("vocab already built/frozen")
        counter = Counter()
        for t in texts:     # paragraph, sentence
            counter.update(self.tokenize(t))
        # keep tokens above min_freq
        items = [(tok, freq) for tok, freq in counter.items() if freq >= min_freq]
        items.sort(key=lambda x: (-x[1], x[0]))
        # reserve ids for special tokens
        vocab_list = [self.pad_token, self.unk_token] + [tok for tok, _ in items[: max_vocab - 2]]
        self.vocab = {tok: i for i, tok in enumerate(vocab_list)}
        self.id_to_token = {i: tok for tok, i in self.vocab.items()}
        self.frozen = True

    def encode(self, text: str, add_eos: bool = False) -> List[int]:
        tokens = self.tokenize(text)
        ids = [self.vocab.get(t, self.vocab.get(self.unk_token)) for t in tokens]
        if add_eos:
            ids.append(self.vocab.get(self.unk_token))
        return ids

    def decode(self, ids: List[int]) -> str:
        tokens = [self.id_to_token.get(i, self.unk_token) for i in ids]
        return "".join(self._reconstruct_spacing(tokens))

    def _reconstruct_spacing(self, tokens: List[str]) -> List[str]:
        # join tokens inserting spaces between alphanumeric tokens and where appropriate
        out = []
        prev_was_word = False
        for t in tokens:
            if re.match(r"^\w+$", t):
                if prev_was_word:
                    out.append(" ")
                out.append(t)
                prev_was_word = True
            else:
                out.append(t)
                prev_was_word = False
        return out

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "do_lower": self.do_lower,
                "unk_token": self.unk_token,
                "pad_token": self.pad_token,
                "vocab": self.vocab,
            }, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str):
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        tk = cls(do_lower=obj.get("do_lower", True), unk_token=obj.get("unk_token", "<unk>"), pad_token=obj.get("pad_token", "<pad>"))
        tk.vocab = obj["vocab"]
        tk.id_to_token = {int(i): t for t, i in tk.vocab.items()}
        tk.frozen = True
        return tk
