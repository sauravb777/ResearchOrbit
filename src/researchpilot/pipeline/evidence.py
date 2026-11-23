from __future__ import annotations
import re
from ..schemas import Evidence, Source
from .retrieval import retrieve_passages
from ..textutils import keyword_overlap_score, split_sentences
_KIND_KEYWORDS = {
    "ablation": ["ablation", "ablation study", "when removed", "without the", "removing the",
                 "we remove", "we ablate", "isolate the effect", "component is removed"],
    "limitation": ["limitation", "challenge", "however", "fail", "weakness", "additional latency",
                   "computational cost", "computational overhead", "at the cost of", "high cost",
                   "costly", "expensive", "future work", "does not", "cannot", "drawback",
                   "trade-off", "tradeoff"],
    "result": ["result", "improve", "outperform", "accuracy", "f1 score", "recall", "precision",
               "bleu", "rouge", "benchmark result", "state-of-the-art", "sota", "evaluation shows"],
    "dataset": ["dataset", "corpus", "benchmark dataset", "collected", "annotated"],
    "method": ["method", "approach", "framework", "architecture", "algorithm", "propose", "model"],
}
_MIN_SENTENCE_LEN = 45
_INJECTION_RE = re.compile(
    r"\b(ignore (all |the )?(previous|prior|above) instructions?|disregard (the |all )?(above|previous)|"
    r"you are now|act as (?:if you were|a) |reveal your (?:system )?prompt|system prompt is|"
    r"new instructions?:|override your instructions?|jailbreak|do anything now|\bDAN\b|"
    r"forget (?:everything|all) (?:you were told|previous)|"
    r"cite (?:a |the )?(?:fake|false|fabricated|nonexistent|invented) (?:evidence|citation|reference|source)|"
    r"(?:fabricate|invent|insert) (?:a |an )?(?:fake |false )?(?:citation|evidence|reference|result|quote))\b",
    re.IGNORECASE,
)
def _looks_like_injection(sentence: str) -> bool:
    return bool(_INJECTION_RE.search(sentence))
def _source_locator(text: str, sentence: str, fallback: str) -> str:
    """Attach a stable page/section locator when page anchors are available."""
    pos = text.find(sentence)
    if pos < 0:
        return fallback
    prefix = text[:pos]
    pages = list(re.finditer(r"\[PAGE\s+(\d+)\]", prefix, flags=re.IGNORECASE))
    page = pages[-1].group(1) if pages else ""
    return f"page {page}, {fallback}" if page else fallback
def _compile_keyword_patterns() -> dict[str, list[re.Pattern]]:
    """Regex per keyword phrase, tuned to avoid two opposite failure modes:
    - A bare `\\bword\\b` (full boundary) misses legitimate inflections —
      "results", "improves", "frameworks" wouldn't match "result",
      "improve", "framework".
    - A bare substring check (no boundary at all) false-positives on
      unrelated words containing the same letters — "accosted" would
      match "cost", "methods" would always outrank "method"-adjacent
      words regardless of context.
    The middle ground used here: multi-word phrases get a full boundary on
    both sides (they're specific enough not to need inflection tolerance).
    Single-word keywords get a left boundary only, so "result" still
    matches "results"/"resulting" but "accosted" still doesn't match
    "cost".
    """
    compiled: dict[str, list[re.Pattern]] = {}
    for kind, phrases in _KIND_KEYWORDS.items():
        patterns = []
        for phrase in phrases:
            escaped = re.escape(phrase)
            pattern = rf"\b{escaped}\b" if " " in phrase else rf"\b{escaped}"
            patterns.append(re.compile(pattern))
        compiled[kind] = patterns
    return compiled
_KIND_PATTERNS = _compile_keyword_patterns()
def _classify(sentence: str) -> str:
    low = sentence.lower()
    for kind, patterns in _KIND_PATTERNS.items():
        if any(p.search(low) for p in patterns):
            return kind
    return "context"
def extract_evidence_for_source(question: str, source: Source, max_items: int = 6) -> list[Evidence]:
    """Retrieve relevant passages first, then extract evidence sentences.
    This avoids the old failure mode where a keyword-heavy introduction or
    related-work section crowded out the results/limitations later in the
    paper. The retriever works over the whole available text and keeps each
    source isolated.
    """
    text = source.full_text or source.abstract
    if not text:
        return []
    location = "full text" if source.full_text else "abstract"
    passages = retrieve_passages(
        question,
        [source],
        top_k=max(6, max_items * 2),
        per_source_cap=max(3, max_items),
    )
    if not passages:
        passages = []
    candidates: list[tuple[float, str, str, int]] = []
    seen: set[str] = set()
    for passage in passages:
        for idx, sentence in enumerate(split_sentences(passage.text)):
            clean = " ".join(re.sub(r"\[PAGE\s+\d+\]", "", sentence, flags=re.I).split())
            if len(clean) < _MIN_SENTENCE_LEN or clean.lower().startswith("references"):
                continue
            if _looks_like_injection(clean):
                continue
            key = clean.lower()[:220]
            if key in seen:
                continue
            seen.add(key)
            relevance = keyword_overlap_score(question, clean)
            relevance = max(relevance, 0.15)
            score = min(1.0, 0.55 * relevance + 0.45 * min(1.0, passage.score / 8.0))
            kind = _classify(clean)
            if passage.section.lower() in {"results", "experiments", "evaluation", "ablation", "limitations", "discussion"}:
                score = min(1.0, score + 0.06)
            candidates.append((score, clean, kind, idx))
    if len(candidates) < max_items:
        for idx, sentence in enumerate(split_sentences(text)):
            clean = " ".join(re.sub(r"\[PAGE\s+\d+\]", "", sentence, flags=re.I).split())
            if len(clean) < _MIN_SENTENCE_LEN:
                continue
            if _looks_like_injection(clean):
                continue
            key = clean.lower()[:220]
            if key in seen:
                continue
            seen.add(key)
            relevance = max(0.15, keyword_overlap_score(question, clean))
            candidates.append((relevance, clean, _classify(clean), idx))
    candidates.sort(key=lambda c: c[0], reverse=True)
    out: list[Evidence] = []
    for relevance, sentence, kind, idx in candidates[:max_items]:
        out.append(
            Evidence(
                id="",
                source_id=source.id,
                text=sentence,
                locator=_source_locator(text, sentence, f"{location}, retrieved passage sentence {idx + 1}"),
                source_span=sentence,
                kind=kind,
                relevance=round(relevance, 4),
            )
        )
    return out
def extract_all(question: str, sources: list[Source], max_total: int = 40) -> list[Evidence]:
    per_source_cap = max(3, max_total // max(1, len(sources) or 1))
    pooled: list[Evidence] = []
    for source in sources:
        pooled.extend(extract_evidence_for_source(question, source, max_items=per_source_cap))
    pooled.sort(key=lambda e: e.relevance, reverse=True)
    trimmed = pooled[:max_total]
    for i, ev in enumerate(trimmed, start=1):
        ev.id = f"E{i}"
    return trimmed
def merge_evidence(pools: list[list[Evidence]], max_total: int = 60) -> list[Evidence]:
    """Combine multiple already-extracted evidence pools (typically the
    heuristic pool from `extract_all` plus verified quotes from
    `deep_read.analyze_papers`), drop near-duplicate text, and assign the
    final sequential [E#] ids used everywhere downstream (claims, report,
    follow-up Q&A, claim verification).
    Deep-read evidence is created with a relevance floor of 0.85 (see
    `deep_read.py`), so a plain relevance-sorted merge already prefers it
    over lower-scoring heuristic sentences without any special-casing here
    -- but nothing heuristic is discarded outright, it's just outranked.
    """
    seen: set[str] = set()
    pooled: list[Evidence] = []
    for pool in pools:
        for ev in pool:
            key = " ".join(ev.text.lower().split())[:160]
            if not key or key in seen:
                continue
            seen.add(key)
            pooled.append(ev)
    pooled.sort(key=lambda e: e.relevance, reverse=True)
    trimmed = pooled[:max_total]
    for i, ev in enumerate(trimmed, start=1):
        ev.id = f"E{i}"
    return trimmed
