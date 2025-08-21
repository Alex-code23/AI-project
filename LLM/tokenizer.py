# advanced_tokenizer.py
import json
import re
import unicodedata
from collections import Counter, defaultdict
from typing import Iterable, List, Tuple, Dict, Optional


def _strip_accents(text: str) -> str:
    '''
    reduce token sparsity and vocabulary size (so the model sees café and cafe as the same), but you lose information
    '''
    # Remove accents by decomposing characters and dropping combining marks.
    text = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in text if unicodedata.category(ch) != "Mn")


class Tokenizer:
    """
    AdvancedTokenizer: improved tokenizer supporting word-level and BPE subword vocabularies.

    Key features:
    - Unicode normalization (NFKC) and optional accent stripping
    - URL/email/emoji-aware splitting + fallback to word / punctuation tokens
    - Optional lowercasing
    - Small cache for tokenization and encoding to speed repeated calls
    - Two vocab modes:
        * 'word' : tokens are words/punct (like SimpleTokenizer)
        * 'bpe'  : train Byte-Pair Encoding merges and build subword vocab
    - Save / load preserving merges, vocab, and config
    """

    # Precompiled regex parts
    _url_re = r"https?://[^\s]+|www\.[^\s]+"
    _email_re = r"\b\S+@\S+\.\S+\b"
    # Unicode emoji block heuristic (will match most emoji characters)
    _emoji_re = r"[\U0001F300-\U0001F6FF\U0001F900-\U0001F9FF\U0001FA70-\U0001FAFF]"
    # core token regex: words (\w+), apostrophes inside words, or single punctuation/symbols
    _word_re = r"[A-Za-zÀ-ÖØ-öø-ÿ0-9_]+(?:'[A-Za-zÀ-ÖØ-öø-ÿ0-9_]+)?"
    _punct_re = r"[^\w\s]"
    _token_re = re.compile(
        f"({_url_re}|{_email_re}|{_emoji_re}|{_word_re}|{_punct_re})",
        flags=re.UNICODE,
    )

    def __init__(
        self,
        do_lower: bool = True,
        unk_token: str = "<unk>",
        pad_token: str = "<pad>",
        strip_accents: bool = False,
        vocab_mode: str = "bpe",
        debug: bool = False,
    ):
        assert vocab_mode in ("word", "bpe")
        self.do_lower = do_lower
        self.strip_accents = strip_accents
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.vocab_mode = vocab_mode
        self.debug = debug

        self.vocab: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.frozen = False

        self.bpe_merges: List[Tuple[str, str]] = []
        self.bpe_vocab_counts: Counter = Counter()

        self._tokenize_cache: Dict[str, Tuple[str, ...]] = {}
        self._encode_cache: Dict[Tuple[str, bool, str], Tuple[int, ...]] = {}

    def _normalize(self, text: str) -> str:
        text = unicodedata.normalize("NFKC", text)
        if self.strip_accents:
            text = _strip_accents(text)
        if self.do_lower:
            text = text.lower()
        return text

    def tokenize(self, text: str) -> List[str]:
        if text in self._tokenize_cache:
            return list(self._tokenize_cache[text])

        norm = self._normalize(text)
        toks = self._token_re.findall(norm)
        tokens = []
        for m in toks:
            if isinstance(m, tuple):
                token = next((x for x in m if x), "")
            else:
                token = m
            if token:
                tokens.append(token)
        self._tokenize_cache[text] = tuple(tokens)
        return tokens

    def _build_word_vocab(self, texts: Iterable[str], max_vocab: int, min_freq: int):
        counter = Counter()
        for t in texts:
            counter.update(self.tokenize(t))
        items = [(tok, freq) for tok, freq in counter.items() if freq >= min_freq]
        items.sort(key=lambda x: (-x[1], x[0]))
        vocab_list = [self.pad_token, self.unk_token] + [tok for tok, _ in items[: max_vocab - 2]]
        self.vocab = {tok: i for i, tok in enumerate(vocab_list)}
        self.id_to_token = {i: tok for tok, i in self.vocab.items()}
        self.frozen = True
        if self.debug:
            print("WORD VOCAB built: size=", len(self.vocab))

    @staticmethod
    def _get_pairs(word_symbols: Tuple[str, ...]) -> Counter:
        pairs = Counter()
        prev = word_symbols[0]
        for sym in word_symbols[1:]:
            pairs[(prev, sym)] += 1
            prev = sym
        return pairs

    def train_bpe(self, texts: Iterable[str], vocab_size: int = 30000, min_freq: int = 2, num_merges: Optional[int] = None):
        if self.frozen:
            raise RuntimeError("vocab already built/frozen")

        # word frequency (surface tokens)
        word_freqs = Counter()
        for t in texts:
            for w in self.tokenize(t):
                word_freqs[w] += 1

        # filter low-frequency words
        word_freqs = Counter({w: f for w, f in word_freqs.items() if f >= min_freq})
        if not word_freqs:
            raise ValueError("No words above min_freq; lower min_freq or provide more data.")

        # represent words as character sequences with end-of-word marker
        words = {w: tuple(list(w) + ["</w>"]) for w in word_freqs}

        # initial pair counts scaled by word frequency
        pairs = Counter()
        for w, symbols in words.items():
            pc = self._get_pairs(symbols)
            freq = word_freqs[w]
            for pair, cnt in pc.items():
                pairs[pair] += cnt * freq

        merges: List[Tuple[str, str]] = []

        if num_merges is None:
            target_merges = max(100, min(20000, vocab_size // 2))
        else:
            target_merges = num_merges

        for i in range(target_merges):
            if not pairs:
                break
            (a, b), freq = pairs.most_common(1)[0]
            if freq < 1:
                break
            merges.append((a, b))

            # apply this merge across all words
            new_words = {}
            new_pairs = Counter()
            for w, symbols in words.items():
                # merge adjacent a,b
                new_sym = []
                j = 0
                while j < len(symbols):
                    if j < len(symbols) - 1 and symbols[j] == a and symbols[j + 1] == b:
                        new_sym.append(a + b)
                        j += 2
                    else:
                        new_sym.append(symbols[j])
                        j += 1
                new_symbols = tuple(new_sym)
                new_words[w] = new_symbols
                # count pairs in this updated word and scale by word frequency
                pc = self._get_pairs(new_symbols)
                freq_w = word_freqs[w]
                for pair, cnt in pc.items():
                    new_pairs[pair] += cnt * freq_w
            words = new_words
            pairs = new_pairs

        # final subword counts: count occurrences of each symbol in words scaled by freq
        subword_counts = Counter()
        for w, symbols in words.items():
            symbol_counts = Counter(symbols)  # counts of each symbol in this word
            freq_w = word_freqs[w]
            for sym, cnt in symbol_counts.items():
                subword_counts[sym] += cnt * freq_w

        # ensure special tokens present
        subword_counts[self.pad_token] += 0
        subword_counts[self.unk_token] += 0

        if not subword_counts:
            raise RuntimeError("BPE training produced no subword tokens (empty corpus?).")

        most_common_subwords = [tok for tok, _ in subword_counts.most_common(max(0, vocab_size - 2))]
        vocab_list = [self.pad_token, self.unk_token] + most_common_subwords
        self.vocab = {tok: i for i, tok in enumerate(vocab_list)}
        self.id_to_token = {i: tok for tok, i in self.vocab.items()}

        self.bpe_merges = merges
        self.bpe_vocab_counts = subword_counts
        self.frozen = True

        if self.debug:
            print("BPE training done:")
            print("  merges =", len(merges))
            print("  subword types =", len(subword_counts))
            print("  final vocab size =", len(self.vocab))

    def build_vocab(self, texts: Iterable[str], max_vocab: int = 30000, min_freq: int = 2, **kwargs):
        if self.vocab_mode == "word":
            self._build_word_vocab(texts, max_vocab=max_vocab, min_freq=min_freq)
        else:
            self.train_bpe(texts, vocab_size=max_vocab, min_freq=min_freq, **kwargs)

    def _apply_bpe_to_word(self, word: str) -> List[str]:
        symbols = list(word) + ["</w>"]
        if not self.bpe_merges:
            return symbols
        for a, b in self.bpe_merges:
            merged = a + b
            j = 0
            new_sym = []
            while j < len(symbols):
                if j < len(symbols) - 1 and symbols[j] == a and symbols[j + 1] == b:
                    new_sym.append(merged)
                    j += 2
                else:
                    new_sym.append(symbols[j])
                    j += 1
            symbols = new_sym
        return symbols

    def encode(self, text: str, add_eos: bool = False) -> List[int]:
        cache_key = (text, add_eos, self.vocab_mode)
        if cache_key in self._encode_cache:
            return list(self._encode_cache[cache_key])

        tokens = self.tokenize(text)
        ids = []
        unk_id = self.vocab.get(self.unk_token, None)
        if self.vocab_mode == "word":
            for t in tokens:
                ids.append(self.vocab.get(t, unk_id))
        else:
            for t in tokens:
                pieces = self._apply_bpe_to_word(t)
                for p in pieces:
                    ids.append(self.vocab.get(p, unk_id))
        if add_eos:
            ids.append(unk_id)
        self._encode_cache[cache_key] = tuple(ids)
        return ids

    def decode(self, ids: List[int]) -> str:
        tokens = [self.id_to_token.get(i, self.unk_token) for i in ids]
        if self.vocab_mode == "bpe":
            out_words = []
            cur = []
            for t in tokens:
                if t == self.pad_token or t == self.unk_token:
                    cur.append(t)
                    continue
                if t.endswith("</w>"):
                    if t == "</w>":
                        out_words.append("".join(cur))
                        cur = []
                    else:
                        cur.append(t.replace("</w>", ""))
                        out_words.append("".join(cur))
                        cur = []
                elif t == "</w>":
                    out_words.append("".join(cur))
                    cur = []
                else:
                    cur.append(t)
            if cur:
                out_words.append("".join(cur))
            return " ".join(out_words)
        else:
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
            return "".join(out)

    def save(self, path: str):
        data = {
            "do_lower": self.do_lower,
            "strip_accents": self.strip_accents,
            "unk_token": self.unk_token,
            "pad_token": self.pad_token,
            "vocab_mode": self.vocab_mode,
            "vocab": self.vocab,
            "bpe_merges": ["\t".join(pair) for pair in self.bpe_merges],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str):
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        tk = cls(
            do_lower=obj.get("do_lower", True),
            unk_token=obj.get("unk_token", "<unk>"),
            pad_token=obj.get("pad_token", "<pad>"),
            strip_accents=obj.get("strip_accents", False),
            vocab_mode=obj.get("vocab_mode", "bpe"),
        )
        tk.vocab = {tok: int(i) for tok, i in obj["vocab"].items()}
        tk.id_to_token = {i: tok for tok, i in tk.vocab.items()}
        tk.bpe_merges = [tuple(x.split("\t")) for x in obj.get("bpe_merges", [])]
        tk.frozen = True
        return tk


if __name__ == "__main__":
    texts = [
        "Hello, world! Visit https://example.com or mail me at test@example.org.",
        "Café au lait — can't stop. emojis: 😀👍",
        "Tokenization is fun. Tokenization is useful."
    ]

    tok = Tokenizer(do_lower=True, strip_accents=False, vocab_mode="bpe", debug=True)
    tok.build_vocab(texts, max_vocab=200, min_freq=1, num_merges=20)
    print("Vocab size:", len(tok.vocab))
    print(f"Vocabulary : {tok.vocab}")
    s = "Hello, café!"
    enc = tok.encode(s)
    print("Encoded:", enc)
    print("Decoded:", tok.decode(enc))

    
    
    
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