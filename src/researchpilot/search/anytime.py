from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Iterable
import time
from ..schemas import RunEvent, Source
from ..textutils import keyword_overlap_score, title_match_score
from .lexical import bm25_scores, saturate
class RetrievalDeadlineExceeded(TimeoutError):
    """Raised internally when a bounded retrieval operation reaches its wall clock."""
def _norm_query(query: str) -> str:
    return " ".join(str(query or "").split()).strip().lower()
@dataclass
class RetrievalSnapshot:
    candidates: list[Source]
    elapsed_seconds: float
    rounds: int
    queries_attempted: int
    deadline_reached: bool
    stable_rounds: int
    retrieval_pool: list[Source] = field(default_factory=list)
    deep_read_ids: list[str] = field(default_factory=list)
    trace: list[dict] = field(default_factory=list)
class AnytimeRetrievalController:
    """Progressive Top-K arXiv discovery with a hard wall-clock budget."""
    def __init__(self, aggregator):
        self.aggregator = aggregator
    def run(
        self,
        question: str,
        initial_queries: Iterable[str],
        *,
        deadline_seconds: float = 600.0,
        max_papers: int = 15,
        results_per_query: int = 10,
        batch_size: int = 4,
        followup_factory: Callable[[], Iterable[str]] | None = None,
        min_stable_rounds: int = 2,
        candidate_pool_size: int = 240,
        min_stability_seconds: float = 20.0,
        cancel_check: Callable[[], bool] | None = None,
    ) -> tuple[RetrievalSnapshot, list[RunEvent]]:
        start = time.monotonic()
        deadline = start + max(1.0, float(deadline_seconds))
        max_papers = max(1, int(max_papers))
        candidate_pool_size = max(max_papers, min(500, int(candidate_pool_size)))
        batch_size = max(1, min(6, int(batch_size)))
        results_per_query = max(1, min(100, int(results_per_query)))
        max_pages_per_query = 5 if bool(getattr(self.aggregator, "_production_registry", False)) else 1
        pending: list[tuple[str, int]] = [(str(q).strip(), 0) for q in initial_queries if str(q).strip()]
        seen_query_pages: set[tuple[str, int]] = set()
        exhausted_queries: set[str] = set()
        query_text_by_key: dict[str, str] = {}
        pool: dict[str, Source] = {}
        events: list[RunEvent] = []
        trace: list[dict] = []
        rounds = 0
        attempted = 0
        stable_rounds = 0
        previous_ids: tuple[str, ...] = ()
        previous_scores: dict[str, float] = {}
        followups_used = False
        deadline_reached = False
        def remaining() -> float:
            return deadline - time.monotonic()
        def ranked_pool() -> list[Source]:
            """Score the WHOLE reservoir (cheap) and return it best-first."""
            sources = list(pool.values())
            if not sources:
                return []
            docs = [f"{s.title}. {s.abstract or ''} {s.venue or ''}" for s in sources]
            raw_bm25 = bm25_scores(question, docs, title_boost=[s.title for s in sources])
            ranked = []
            for source, bm in zip(sources, raw_bm25):
                overlap = keyword_overlap_score(question, source.title, source.abstract, source.venue)
                title = title_match_score(question, source.title)
                lexical = 0.75 * overlap + 0.25 * title
                source.score = round(min(1.0, 0.6 * lexical + 0.4 * saturate(bm, question)), 4)
                ranked.append(source)
            ranked.sort(key=lambda s: (float(getattr(s, "score", 0.0)), s.title.lower()), reverse=True)
            return ranked
        def top_pool() -> list[Source]:
            return ranked_pool()[:max_papers]
        def trim_reservoir() -> None:
            if len(pool) <= candidate_pool_size:
                return
            keep = {s.dedupe_key() for s in ranked_pool()[:candidate_pool_size]}
            for k in [k for k in pool if k not in keep]:
                pool.pop(k, None)
        while remaining() > 0:
            if cancel_check is not None and cancel_check():
                events.append(RunEvent(stage="retrieval_cancelled", detail="retrieval cancelled by user", ok=False))
                raise RuntimeError("Job cancelled")
            batch: list[tuple[str, int]] = []
            while pending and len(batch) < batch_size:
                q, page = pending.pop(0)
                key = _norm_query(q)
                marker = (key, int(page))
                if key and marker not in seen_query_pages:
                    seen_query_pages.add(marker)
                    query_text_by_key[key] = q
                    batch.append((q, int(page)))
            if not batch:
                generated: list[str] = []
                if followup_factory is not None:
                    try:
                        generated = [str(q).strip() for q in followup_factory() if str(q).strip()]
                    except Exception as exc:
                        events.append(RunEvent(
                            stage="retrieval_followup",
                            detail=f"follow-up query planning failed: {exc}",
                            ok=False,
                        ))
                fresh_generated = []
                for q in generated:
                    key = _norm_query(q)
                    if key and key not in exhausted_queries and (key, 0) not in seen_query_pages:
                        fresh_generated.append((q, 0))
                pending.extend(fresh_generated)
                if not pending:
                    for marker_key, marker_page in sorted(seen_query_pages, key=lambda item: item[1]):
                        if marker_page + 1 < max_pages_per_query and marker_key not in exhausted_queries:
                            pending.append((query_text_by_key.get(marker_key, marker_key), marker_page + 1))
                            break
                if pending:
                    continue
                break
            if remaining() <= 0:
                deadline_reached = True
                break
            rounds += 1
            attempted += len(batch)
            before_ids = tuple(s.id for s in top_pool()[:max_papers])
            batch_queries = [q for q, _page in batch]
            batch_starts = [page * results_per_query for _q, page in batch]
            try:
                if cancel_check is not None and cancel_check():
                    raise RuntimeError("Job cancelled")
                try:
                    found, search_events = self.aggregator.run(
                        batch_queries, deadline=deadline, max_results=results_per_query, starts=batch_starts
                    )
                except TypeError:
                    found, search_events = self.aggregator.run(
                        batch_queries, deadline=deadline, max_results=results_per_query
                    )
                events.extend(search_events)
            except TimeoutError:
                found = []
                deadline_reached = True
            except RuntimeError as exc:
                if str(exc) == "Job cancelled":
                    raise
                raise
            for query_text, page in batch:
                if not found:
                    exhausted_queries.add(_norm_query(query_text))
                elif page + 1 < max_pages_per_query:
                    pending.append((query_text, page + 1))
            for source in found:
                if not source or not getattr(source, "id", None):
                    continue
                key = source.dedupe_key()
                current = pool.get(key)
                if current is None:
                    pool[key] = source
                else:
                    for field in ("title", "authors", "year", "abstract", "full_text", "url", "pdf_url"):
                        incoming = getattr(source, field, "")
                        existing = getattr(current, field, "")
                        if incoming and len(str(incoming)) > len(str(existing)):
                            setattr(current, field, incoming)
            trim_reservoir()
            top = top_pool()[:max_papers]
            after_ids = tuple(s.id for s in top)
            after_scores = {s.id: float(getattr(s, "score", 0.0) or 0.0) for s in top}
            membership_changed = after_ids != before_ids
            score_improved = any(after_scores.get(sid, 0.0) > previous_scores.get(sid, 0.0) + 0.01 for sid in after_ids)
            if after_ids == previous_ids and not score_improved:
                stable_rounds += 1
            else:
                stable_rounds = 0
            previous_ids = after_ids
            previous_scores = after_scores
            elapsed = time.monotonic() - start
            trace.append({
                "round": rounds,
                "queries": len(batch),
                "new_results": len(found),
                "active_top_k": len(top),
                "top_ids": list(after_ids),
                "stable_rounds": stable_rounds,
                "elapsed_seconds": round(elapsed, 2),
                "remaining_seconds": round(max(0.0, deadline - time.monotonic()), 2),
            })
            events.append(RunEvent(
                stage="retrieval_anytime",
                detail=(
                    f"round {rounds}: searched {len(batch)} query/page(s), "
                    f"received {len(found)} result(s), reservoir={len(pool)}, "
                    f"Top-{min(max_papers, len(top))} (BM25+lexical over full reservoir), "
                    f"elapsed={elapsed:.1f}s, remaining={max(0.0, deadline-__import__('time').monotonic()):.1f}s"
                ),
            ))
            if deadline - time.monotonic() <= 0:
                deadline_reached = True
                break
            if (stable_rounds >= min_stable_rounds
                    and len(pool) >= min(candidate_pool_size, max(60, max_papers * 6))
                    and elapsed >= float(min_stability_seconds)):
                events.append(RunEvent(
                    stage="retrieval_stable",
                    detail=f"Top-{max_papers} stabilized after {elapsed:.1f}s with {len(pool)} candidates; ending discovery early",
                ))
                break
        if remaining() <= 0:
            deadline_reached = True
        final_top = top_pool()[:max_papers]
        deep_read_target = min(5, len(final_top))
        deep_read_ids = [s.id for s in final_top[:deep_read_target]]
        for source in final_top:
            source.role = "deep_read" if source.id in deep_read_ids else "no_read"
        elapsed = time.monotonic() - start
        events.append(RunEvent(
            stage="discovery_complete",
            detail=(
                f"anytime retrieval complete: {len(final_top)} active candidates, "
                f"{rounds} rounds, {attempted} queries, elapsed={elapsed:.1f}s, "
                f"deadline_reached={deadline_reached}"
            ),
            ok=True,
        ))
        return RetrievalSnapshot(
            candidates=final_top,
            retrieval_pool=ranked_pool()[:candidate_pool_size],
            elapsed_seconds=elapsed,
            rounds=rounds,
            queries_attempted=attempted,
            deadline_reached=deadline_reached,
            stable_rounds=stable_rounds,
            deep_read_ids=deep_read_ids,
            trace=trace,
        ), events
