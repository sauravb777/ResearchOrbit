from __future__ import annotations
from dataclasses import dataclass, field
from ..schemas import ResearchPlan, Source
from .ranker import keyword_overlap_score, title_match_score
from .relevance_engine import recency_score, publication_signal
@dataclass
class RelevanceScore:
    source_id: str
    keyword: float
    retrieval: float
    intent: float
    method: float
    evidence_availability: float
    quality: float
    total: float
    reasons: list[str] = field(default_factory=list)
def _semantic_fallback(question: str, source: Source) -> float:
    """Conservative title+abstract semantic proxy for offline operation."""
    overlap = keyword_overlap_score(question, source.title, source.abstract)
    title = title_match_score(question, source.title)
    q = " ".join(question.lower().replace("-", " ").split())
    paper = " ".join(f"{source.title} {source.abstract}".lower().replace("-", " ").split())
    phrase_bonus = 0.25 if "retrieval augmented generation" in q and "retrieval augmented generation" in paper else 0.0
    return max(0.0, min(1.0, max(overlap, title) + phrase_bonus))
def score_paper_relevance(question: str, plan: ResearchPlan, source: Source) -> RelevanceScore:
    semantic = _semantic_fallback(question, source)
    recency = float(recency_score(source.year) or 0.0)
    publication = publication_signal(source)
    total = 0.70 * semantic + 0.15 * recency + 0.15 * publication
    return RelevanceScore(
        source_id=source.id,
        keyword=semantic,
        retrieval=semantic,
        intent=semantic,
        method=semantic,
        evidence_availability=1.0 if source.full_text.strip() else 0.5,
        quality=publication,
        total=round(total, 4),
        reasons=[
            f"Semantic relevance (offline fallback): {semantic:.0%}",
            f"Recency: {recency:.0%}",
            f"Publication signal: {publication:.0%}",
        ],
    )
def score_all(question: str, plan: ResearchPlan, sources: list[Source]) -> dict[str, RelevanceScore]:
    return {s.id: score_paper_relevance(question, plan, s) for s in sources}
