from __future__ import annotations
import re
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "how", "in", "into", "is", "it", "its", "of", "on", "or", "our", "so",
    "such", "that", "the", "their", "then", "there", "these", "this", "to",
    "was", "were", "what", "when", "where", "which", "who", "why", "will",
    "with", "you", "your", "we", "does", "do", "did", "can", "could",
    "should", "would", "recent", "recently", "approach", "approaches",
    "research", "question", "paper", "papers", "study", "studies",
}
_WORD_RE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9\-]{1,}")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
def tokenize(text: str) -> set[str]:
    """Lowercase word tokens with stopwords removed."""
    words = (w.lower() for w in _WORD_RE.findall(text or ""))
    return {w for w in words if w not in _STOPWORDS and len(w) > 2}
def split_sentences(text: str) -> list[str]:
    """A pragmatic sentence splitter (no NLTK download required)."""
    if not text:
        return []
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    parts = _SENT_SPLIT_RE.split(text)
    return [p.strip() for p in parts if len(p.strip()) > 0]
def keyword_overlap_score(query: str, *fields: str) -> float:
    """A 0..1 relevance score based on token overlap between the query and
    one or more text fields (e.g. title, abstract). Title matches are
    weighted more heavily than body matches.
    """
    q_tokens = tokenize(query)
    if not q_tokens:
        return 0.0
    combined = tokenize(" ".join(fields))
    if not combined:
        return 0.0
    overlap = len(q_tokens & combined) / len(q_tokens)
    return round(min(1.0, overlap), 4)
def title_match_score(query: str, title: str) -> float:
    q_tokens = tokenize(query)
    t_tokens = tokenize(title)
    if not q_tokens or not t_tokens:
        return 0.0
    return round(min(1.0, len(q_tokens & t_tokens) / len(q_tokens)), 4)
def recency_score(year: str) -> float:
    try:
        y = int(str(year)[:4])
    except (ValueError, TypeError):
        return 0.0
    return 0.1 if y >= 2023 else 0.0
def truncate(text: str, max_chars: int) -> str:
    text = text or ""
    return text if len(text) <= max_chars else text[: max_chars - 1].rstrip() + "…"
