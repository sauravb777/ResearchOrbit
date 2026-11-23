from __future__ import annotations
from dataclasses import dataclass, field
from ..schemas import Source
from .relevance import RelevanceScore
from .relevance_engine import AUTO_RELEVANCE_THRESHOLD, AUTO_PRIORITY_THRESHOLD, AUTO_MIN_PAPERS, AUTO_FALLBACK_PAPERS, AUTO_MAX_PAPERS, automatic_selection_target
@dataclass
class SelectionResult:
    deep_read: list[Source]
    no_read: list[Source] = field(default_factory=list)
    rejected: list[Source] = field(default_factory=list)
def _family(source: Source) -> str:
    text = f"{source.title} {source.abstract}".lower()
    for key in ("survey", "benchmark", "long-context", "hybrid", "sparse", "dense", "adaptive", "rerank", "graph", "transformer", "llm", "agent", "generation", "retrieval"):
        if key in text:
            return key
    return "general"
def select_evidence_corpus(sources: list[Source], scores: dict[str, RelevanceScore], max_papers: int | None = None) -> SelectionResult:
    """Select the default autonomous RED corpus.
    Automatic selection is evidence-driven: use the relevance thresholds and
    configured corpus ceiling to determine a useful set. There is no fixed
    ranking-position quota. User-selected papers remain authoritative outside
    this automatic utility.
    """
    ranked = sorted(sources, key=lambda s: scores.get(s.id).total if s.id in scores else s.score, reverse=True)
    eligible = [s for s in ranked if (scores.get(s.id).total if s.id in scores else s.score) >= AUTO_RELEVANCE_THRESHOLD]
    configured_cap = max_papers if max_papers is not None else AUTO_MAX_PAPERS
    cap = max(1, min(int(configured_cap), AUTO_MAX_PAPERS))
    target = min(automatic_selection_target(eligible, score_getter=lambda s: scores.get(s.id).total if s.id in scores else s.score), cap)
    priority = [s for s in eligible if (scores.get(s.id).total if s.id in scores else s.score) > AUTO_PRIORITY_THRESHOLD]
    selected = priority[:target]
    if len(selected) < target:
        selected_ids = {s.id for s in selected}
        selected.extend(s for s in eligible if s.id not in selected_ids)
        selected = selected[:target]
    selected_ids = {s.id for s in selected}
    for source in ranked:
        source.role = "deep_read" if source.id in selected_ids else "no_read"
    rejected = [s for s in ranked if s.id not in selected_ids]
    return SelectionResult(deep_read=selected, no_read=rejected, rejected=rejected)
