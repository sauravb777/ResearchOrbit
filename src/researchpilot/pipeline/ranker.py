from __future__ import annotations
from ..schemas import Source
from ..textutils import keyword_overlap_score, recency_score, title_match_score
def score_source(query: str, source: Source) -> float:
    """Return a retrieval-only lexical score, never the production relevance score.
    This helper is used before the authoritative 70/15/15 relevance judge. It
    therefore must not inject recency, citation counts, venue prestige, author
    reputation or any other hidden ranking signal.
    """
    overlap = keyword_overlap_score(query, source.title, source.abstract, source.venue)
    title = title_match_score(query, source.title)
    return round(min(1.0, 0.75 * overlap + 0.25 * title), 4)
def has_readable_content(source: Source) -> bool:
    """A source with no abstract and no full text can't honestly support
    any claim later in the pipeline — better to drop it at the source than
    let it occupy a selection slot and produce zero evidence silently."""
    return bool((source.abstract or "").strip() or (source.full_text or "").strip())
def rank_sources(query: str, sources: list[Source]) -> list[Source]:
    for source in sources:
        source.score = score_source(query, source)
    return sorted(sources, key=lambda s: s.score, reverse=True)
def select_diverse(
    sources: list[Source], max_sources: int, query: str = "", min_per_provider_pass: int = 4
) -> list[Source]:
    """Pick the top sources while trying to represent more than one
    provider early on, so a single flaky/narrow provider can't dominate.
    If `query` is given and any source hasn't been scored yet (score==0,
    the default), scores are computed against that query first. If `query`
    is omitted, the function trusts whatever `.score` the caller already
    set (e.g. after semantic reranking) rather than silently overwriting
    it — this is what lets `semantic_rerank()`'s blended scores survive
    into selection instead of being recomputed from scratch.
    """
    if query and any(s.score == 0 for s in sources):
        rank_sources(query, sources)
    ranked = sorted(sources, key=lambda s: s.score, reverse=True)
    selected: list[Source] = []
    seen_titles: set[str] = set()
    seen_providers: set[str] = set()
    for source in ranked:
        key = " ".join(source.title.lower().split())
        if not key or key in seen_titles:
            continue
        if len(selected) >= max_sources:
            break
        if len(selected) < min_per_provider_pass and source.provider in seen_providers and len(seen_providers) < 2:
            continue
        selected.append(source)
        seen_titles.add(key)
        seen_providers.add(source.provider)
    if len(selected) < max_sources:
        for source in ranked:
            key = " ".join(source.title.lower().split())
            if key in seen_titles:
                continue
            selected.append(source)
            seen_titles.add(key)
            if len(selected) >= max_sources:
                break
    return selected
