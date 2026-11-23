from __future__ import annotations
from ..config import get_settings
from ..schemas import RunEvent, Source
from .arxiv import ArxivProvider
from .base import SourceProvider
class SearchAggregator:
    """arXiv-only scholarly discovery boundary.
    ResearchPilot deliberately searches arXiv only. The provider list is kept
    as an interface (so tests can inject fixed providers), but the production
    registry contains just arXiv; any other configured name is ignored.
    Deduplication happens before ranking so a paper returned by several
    queries becomes one research object.
    """
    def __init__(self, providers: list[SourceProvider] | None = None):
        if providers is not None:
            self.providers = list(providers)
            self._production_registry = False
            return
        settings = get_settings()
        self.providers = [ArxivProvider()]
        self._production_registry = True
    @staticmethod
    def _identity_keys(source: Source) -> set[str]:
        title = " ".join((source.title or "").lower().split())
        keys = set()
        if source.doi:
            keys.add("doi:" + source.doi.lower().strip().removeprefix("https://doi.org/").removeprefix("doi:"))
        if title: keys.add("title:" + title)
        if source.id: keys.add("id:" + source.id.lower())
        return keys
    @staticmethod
    def _merge_sources(primary: Source, incoming: Source) -> Source:
        for field in ("title", "authors", "year", "venue", "doi", "url", "pdf_url", "abstract", "full_text"):
            current, candidate = getattr(primary, field), getattr(incoming, field)
            if (not current or len(str(current)) < len(str(candidate))) and candidate:
                setattr(primary, field, candidate)
        if incoming.citation_count is not None:
            primary.citation_count = max(primary.citation_count or 0, incoming.citation_count)
        primary.score = max(primary.score, incoming.score)
        primary.relevance_reasons = list(dict.fromkeys(primary.relevance_reasons + incoming.relevance_reasons))[:8]
        return primary
    def run(
        self,
        queries: list[str],
        *,
        deadline: float | None = None,
        max_results: int | None = None,
        starts: list[int] | None = None,
    ) -> tuple[list[Source], list[RunEvent]]:
        """Search queries while respecting one absolute wall-clock deadline.
        arXiv requests are serialised on purpose: arXiv's terms allow roughly one
        request per 3 s on one connection, and a thread pool would only queue
        behind that limiter (and risk HTTP 429 bans). Speed comes instead from
        (a) large result pages (up to 100 records / request), (b) a shared
        process-wide throttle, (c) a 1 h response cache, and (d) overlapping
        LLM / ranking work with search (see AnytimeRetrievalController).
        """
        settings = get_settings()
        events: list[RunEvent] = []
        found_by_key: dict[str, Source] = {}
        index: dict[str, str] = {}
        production_arxiv_only = bool(getattr(self, "_production_registry", False))
        limit = max_results
        if limit is None:
            limit = (
                getattr(settings, "v79_results_per_query", settings.results_per_query)
                if getattr(settings, "enable_v79_architecture", False)
                else settings.results_per_query
            )
        limit = max(1, min(int(limit), 100))
        for provider in self.providers:
            for query_index, query in enumerate(queries):
                start = int(starts[query_index]) if starts and query_index < len(starts) else 0
                if deadline is not None and __import__("time").monotonic() >= deadline:
                    events.append(RunEvent(
                        stage="search_deadline",
                        detail=f"retrieval deadline reached before query '{query[:100]}'",
                        ok=False,
                    ))
                    return list(found_by_key.values()), events
                try:
                    bounded_search = getattr(provider, "search_bounded", None)
                    if callable(bounded_search) and deadline is not None:
                        try:
                            found = bounded_search(query, limit, deadline=deadline, start=start)
                        except TypeError:
                            found = bounded_search(query, limit, deadline=deadline)
                    else:
                        found = provider.search(query, limit)
                    if production_arxiv_only:
                        found = [s for s in found if (s.provider or provider.name).lower() == "arxiv"]
                    for source in found:
                        keys = self._identity_keys(source)
                        existing = next((index[k] for k in keys if k in index), None)
                        if existing is None:
                            key = source.dedupe_key()
                            found_by_key[key] = source
                            for k in keys:
                                index[k] = key
                        else:
                            self._merge_sources(found_by_key[existing], source)
                    events.append(RunEvent(
                        stage=f"search:{provider.name}",
                        detail=f"'{query}' [start={start}] -> {len(found)} results",
                    ))
                except TimeoutError as exc:
                    events.append(RunEvent(
                        stage="search_deadline",
                        detail=f"'{query[:100]}' stopped at retrieval deadline: {exc}",
                        ok=False,
                    ))
                    return list(found_by_key.values()), events
                except Exception as exc:
                    events.append(RunEvent(
                        stage=f"search:{provider.name}",
                        detail=f"'{query}' failed: {exc}",
                        ok=False,
                    ))
        return list(found_by_key.values()), events
