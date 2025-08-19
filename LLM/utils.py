from pathlib import Path
import re
from typing import List, Optional


def load_texts_from_path(path: str, encoding: str = 'utf-8', split_mode: str = 'paragraph', *, min_chars: int = 1, max_chars: Optional[int] = None, recursive: bool = False) -> List[str]:
    """
    Load text(s) from a file path, directory, or glob pattern and return a list of text segments.

    Parameters
    ----------
    path : str
        Path to a single .txt file, a directory containing .txt files, or a glob pattern (e.g. "data/*.txt").
    encoding : str
        File encoding used to read files (default 'utf-8').
    split_mode : str
        How to split file contents into texts. Options:
        - 'paragraph' : split on blank lines (default)
        - 'line'      : split on individual lines
        - 'sentence'  : split on sentence boundaries using punctuation
        - 'whole'     : return the entire file contents as one string
    min_chars : int
        Minimum number of characters for a text segment to be kept.
    max_chars : Optional[int]
        If provided, long segments will be chunked into pieces of at most max_chars characters.
    recursive : bool
        When `path` is a directory or a glob, search files recursively if True.

    Returns
    -------
    List[str]
        A list of text strings ready to be passed to the tokenizer or dataset builder.
    """
    p = Path(path)
    files: List[Path] = []
    if p.exists() and p.is_file():
        files = [p]
    elif p.exists() and p.is_dir():
        files = list(p.rglob("*.txt")) if recursive else list(p.glob("*.txt"))
    else:
        # treat as glob pattern
        import glob
        matches = glob.glob(path, recursive=recursive)
        files = [Path(m) for m in matches]

    texts: List[str] = []
    for fp in files:
        try:
            raw = fp.read_text(encoding=encoding)
        except Exception:
            # skip files we can't read
            continue
        # normalize newlines
        raw = raw.replace("\r\n", "\n").replace("\r", "\n").strip()
        if not raw:
            continue

        if split_mode == 'line':
            parts = [ln.strip() for ln in raw.split('\n') if ln.strip()]
        elif split_mode == 'paragraph':
            parts = [p.strip() for p in re.split(r"\n\s*\n", raw) if p.strip()]
        elif split_mode == 'sentence':
            parts = [s.strip() for s in re.split(r'(?<=[.!?])\s+', raw) if s.strip()]
        elif split_mode == 'whole':
            parts = [raw]
        else:
            raise ValueError(f"Unknown split_mode: {split_mode}")

        for part in parts:
            if max_chars is not None and len(part) > max_chars:
                # naive chunking: cut into fixed-size pieces (preserves characters, not tokens)
                start = 0
                while start < len(part):
                    chunk = part[start : start + max_chars].strip()
                    if len(chunk) >= min_chars:
                        texts.append(chunk)
                    start += max_chars
            else:
                if len(part) >= min_chars:
                    texts.append(part)

    return texts
