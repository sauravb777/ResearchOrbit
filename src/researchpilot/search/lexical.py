from __future__ import annotations
import math
import re
from collections import Counter
from ..textutils import tokenize
_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-']+")
_K1 = 1.4
_B = 0.72
def _terms(text: str) -> list[str]:
    stop = tokenize(text)
    return [w for w in (m.lower() for m in _WORD.findall(text or "")) if w in stop]
def bm25_scores(query: str, docs: list[str], title_boost: list[str] | None = None) -> list[float]:
    """Return one BM25 score per document (title text is counted twice)."""
    q_terms = list(dict.fromkeys(_terms(query)))
    if not q_terms or not docs:
        return [0.0] * len(docs)
    tokenised = []
    for i, d in enumerate(docs):
        t = _terms(d)
        if title_boost:
            t = _terms(title_boost[i]) * 2 + t
        tokenised.append(t)
    n = len(tokenised)
    avg = (sum(len(t) for t in tokenised) / n) or 1.0
    df: Counter[str] = Counter()
    for t in tokenised:
        for w in set(t) & set(q_terms):
            df[w] += 1
    out: list[float] = []
    for t in tokenised:
        tf = Counter(t)
        dl = len(t) or 1
        s = 0.0
        for w in q_terms:
            f = tf.get(w, 0)
            if not f:
                continue
            idf = math.log(1.0 + (n - df[w] + 0.5) / (df[w] + 0.5))
            s += idf * (f * (_K1 + 1)) / (f + _K1 * (1 - _B + _B * dl / avg))
        out.append(s)
    return out
def saturate(score: float, query: str) -> float:
    """Map an unbounded BM25 score to 0..1 without per-pool max-normalising,
    so a weak pool is not inflated to look strong."""
    n = max(1, len(set(_terms(query))))
    scale = 2.2 * n
    return max(0.0, min(1.0, score / (score + scale)))
