from __future__ import annotations
import math
import re
from collections import Counter
from dataclasses import dataclass
from ..schemas import Source
from ..textutils import tokenize
_SECTION_RE = re.compile(
    r"(?im)^(?:\s*(?:\d+(?:\.\d+)*[.)]?|[IVXLC]+[.)]?)\s+)?"
    r"(abstract|introduction|background|related work|method(?:s|ology)?|approach|"
    r"architecture|experiments?|results?|evaluation|ablation(?:s)?|analysis|discussion|"
    r"limitations?|conclusion|future work|appendix(?:\s+[A-Z])?)\s*$"
)
@dataclass(frozen=True)
class Passage:
    source_id: str
    text: str
    start_char: int
    end_char: int
    section: str = ""
    score: float = 0.0
    page: int = 0
def _page_at(text: str, position: int, passage: str = "") -> int:
    local = list(re.finditer(r"\[PAGE\s+(\d+)\]", passage or "", flags=re.I))
    if local:
        return int(local[-1].group(1))
    prefix = text[:max(0, position)]
    matches = list(re.finditer(r"\[PAGE\s+(\d+)\]", prefix, flags=re.I))
    return int(matches[-1].group(1)) if matches else 0
def _section_at(text: str, position: int) -> str:
    prefix = text[max(0, position - 3500):position]
    matches = list(_SECTION_RE.finditer(prefix))
    return matches[-1].group(1).title() if matches else ""
def chunk_text(text: str, chunk_chars: int = 2600, overlap_chars: int = 350) -> list[tuple[str, int, int, str]]:
    """Create paragraph-aware overlapping chunks over the complete text."""
    text = re.sub(r"\r\n?", "\n", text or "").strip()
    if not text:
        return []
    chunk_chars = max(800, int(chunk_chars))
    overlap_chars = max(0, min(int(overlap_chars), chunk_chars // 2))
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    chunks: list[tuple[str, int, int, str]] = []
    cursor = 0
    current = ""
    current_start = 0
    for para in paragraphs:
        pos = text.find(para, cursor)
        if pos < 0:
            pos = cursor
        cursor = pos + len(para)
        candidate = (current + "\n\n" + para).strip() if current else para
        if current and len(candidate) > chunk_chars:
            end = current_start + len(current)
            chunks.append((current.strip(), current_start, end, _section_at(text, current_start)))
            overlap = current[-overlap_chars:] if overlap_chars else ""
            current_start = max(0, end - len(overlap))
            current = (overlap + "\n\n" + para).strip() if overlap else para
        else:
            if not current:
                current_start = pos
            current = candidate
        while len(current) > chunk_chars * 1.35:
            cut = current.rfind(" ", 0, chunk_chars)
            cut = cut if cut >= chunk_chars // 2 else chunk_chars
            piece = current[:cut].strip()
            piece_start = current_start
            piece_end = piece_start + len(piece)
            chunks.append((piece, piece_start, piece_end, _section_at(text, piece_start)))
            overlap = piece[-overlap_chars:] if overlap_chars else ""
            current_start = max(piece_start, piece_end - len(overlap))
            current = (overlap + " " + current[cut:]).strip()
    if current.strip():
        chunks.append((current.strip(), current_start, current_start + len(current), _section_at(text, current_start)))
    return chunks
def _bm25(query_tokens: list[str], doc_tokens: list[str], avgdl: float, k1: float = 1.35, b: float = 0.72) -> float:
    if not query_tokens or not doc_tokens:
        return 0.0
    q = set(query_tokens)
    tf = Counter(doc_tokens)
    dl = len(doc_tokens)
    score = 0.0
    for token in q:
        f = tf.get(token, 0)
        if f:
            score += (f * (k1 + 1.0)) / (f + k1 * (1.0 - b + b * dl / max(avgdl, 1.0)))
    return score
def _semantic_scores(query: str, texts: list[str]) -> list[float] | None:
    try:
        import numpy as np
        from .semantic_rerank import _load_model
        from ..config import get_settings
        settings = get_settings()
        if not settings.enable_semantic_rerank or not texts:
            return None
        model = _load_model(settings.semantic_rerank_model)
        embeddings = model.encode([query] + texts, normalize_embeddings=True, show_progress_bar=False)
        return [float(x) for x in np.dot(embeddings[1:], embeddings[0])]
    except Exception:
        return None
def retrieve_passages(
    query: str,
    sources: list[Source],
    top_k: int = 12,
    per_source_cap: int = 3,
    chunk_chars: int = 2600,
    overlap_chars: int = 350,
) -> list[Passage]:
    """Retrieve passages from the complete available text of each source."""
    q_tokens = list(tokenize(query))
    if not q_tokens:
        return []
    candidates: list[tuple[Passage, list[str]]] = []
    for source in sources:
        text = source.full_text or source.abstract
        if not text:
            continue
        for chunk, start, end, section in chunk_text(text, chunk_chars, overlap_chars):
            tokens = list(tokenize(chunk))
            if not tokens:
                continue
            candidates.append((Passage(source.id, chunk, start, end, section, 0.0, _page_at(text, start, chunk)), tokens))
    if not candidates:
        return []
    n = len(candidates)
    df = Counter()
    for _, tokens in candidates:
        for token in set(tokens):
            df[token] += 1
    avgdl = sum(len(tokens) for _, tokens in candidates) / max(n, 1)
    scored: list[Passage] = []
    for passage, tokens in candidates:
        lexical = 0.0
        tf = Counter(tokens)
        dl = len(tokens)
        for token in q_tokens:
            f = tf.get(token, 0)
            if not f:
                continue
            idf = math.log(1.0 + (n - df.get(token, 0) + 0.5) / (df.get(token, 0) + 0.5))
            lexical += idf * (f * 2.35) / (f + 1.35 * (1.0 - 0.72 + 0.72 * dl / max(avgdl, 1.0)))
        structural = 0.08 if passage.section.lower() in {
            "results", "experiments", "evaluation", "ablation", "analysis", "limitations", "discussion", "appendix"
        } else 0.0
        scored.append(Passage(passage.source_id, passage.text, passage.start_char, passage.end_char, passage.section, lexical + structural, passage.page))
    scored.sort(key=lambda p: p.score, reverse=True)
    semantic = _semantic_scores(query, [p.text for p in scored[: min(len(scored), 80)]])
    if semantic is not None:
        rescored = []
        for p, sim in zip(scored[: len(semantic)], semantic):
            bonus = max(0.0, min(1.0, (sim + 1.0) / 2.0)) * 0.9
            rescored.append(Passage(p.source_id, p.text, p.start_char, p.end_char, p.section, p.score + bonus, p.page))
        scored = rescored + scored[len(semantic):]
        scored.sort(key=lambda p: p.score, reverse=True)
    selected: list[Passage] = []
    counts: Counter[str] = Counter()
    for p in scored:
        if counts[p.source_id] >= per_source_cap:
            continue
        selected.append(p)
        counts[p.source_id] += 1
        if len(selected) >= top_k:
            break
    return selected
