from __future__ import annotations
import threading
import time
import xml.etree.ElementTree as ET
import re
import requests
from ..config import get_settings
from ..schemas import Source
from .base import SourceProvider
_ATOM_NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
_BASE_URLS = ("https://export.arxiv.org/api/query", "https://arxiv.org/api/query")
_TRANSIENT_STATUS = {429, 500, 502, 503, 504}
_MIN_REQUEST_INTERVAL = 3.05
_MAX_PAGE = 100
_GLOBAL_THROTTLE_LOCK = threading.Lock()
_GLOBAL_LAST_REQUEST = [0.0]
_CACHE_TTL_SECONDS = 3600.0
_SEARCH_CACHE: dict[tuple[str, int, int], tuple[float, list[Source]]] = {}
_SEARCH_CACHE_LOCK = threading.Lock()
class ArxivProvider(SourceProvider):
    name = "arxiv"
    def __init__(self) -> None:
        self._lock = _GLOBAL_THROTTLE_LOCK
    def _throttle(self, deadline: float | None = None) -> None:
        with _GLOBAL_THROTTLE_LOCK:
            now = time.monotonic()
            wait = _MIN_REQUEST_INTERVAL - (now - _GLOBAL_LAST_REQUEST[0])
            if wait > 0:
                if deadline is not None and time.monotonic() + wait >= deadline:
                    raise TimeoutError("arXiv throttle would exceed retrieval deadline")
                time.sleep(wait)
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("arXiv retrieval deadline exceeded")
            _GLOBAL_LAST_REQUEST[0] = time.monotonic()
    @staticmethod
    def _query_text(query: str) -> str:
        clean = " ".join((query or "").split())
        clean = clean.replace("\x00", " ")
        return clean[:420]
    def _request(self, base_url: str, params: dict, timeout: int, deadline: float | None = None) -> requests.Response:
        last_exc: Exception | None = None
        for attempt in range(3):
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0.2:
                    raise TimeoutError("arXiv retrieval deadline exceeded")
                request_timeout = max(0.2, min(float(timeout), remaining))
            else:
                request_timeout = timeout
            self._throttle(deadline)
            try:
                response = requests.get(
                    base_url,
                    params=params,
                    timeout=request_timeout,
                    headers={
                        "User-Agent": "ResearchPilot/79.12 (academic research client)",
                        "Accept": "application/atom+xml, application/xml;q=0.9, */*;q=0.1",
                    },
                )
                if response.status_code in _TRANSIENT_STATUS:
                    retry_after = response.headers.get("Retry-After", "")
                    try:
                        delay = min(15.0, max(3.2, float(retry_after))) if retry_after else min(15.0, 3.2 * (attempt + 1))
                    except ValueError:
                        delay = min(15.0, 3.2 * (attempt + 1))
                    last_exc = RuntimeError(f"arXiv returned HTTP {response.status_code}")
                    if attempt < 2:
                        if deadline is not None and time.monotonic() + delay >= deadline:
                            raise TimeoutError(
                                f"arXiv retry deferred by deadline after HTTP {response.status_code}"
                            )
                        time.sleep(delay)
                        continue
                    raise last_exc
                response.raise_for_status()
                ET.fromstring(response.text)
                return response
            except TimeoutError:
                raise
            except (requests.RequestException, ET.ParseError, RuntimeError) as exc:
                last_exc = exc
                if attempt < 2:
                    delay = min(12.0, 2.0 * (attempt + 1))
                    if deadline is not None and time.monotonic() + delay >= deadline:
                        raise TimeoutError("arXiv retry budget exhausted by retrieval deadline") from exc
                    time.sleep(delay)
                    continue
                raise
        raise last_exc or RuntimeError("arXiv request failed")
    def search_bounded(self, query: str, limit: int = 8, *, deadline: float | None = None, start: int = 0) -> list[Source]:
        """Search arXiv without allowing throttle/retry/network work past deadline."""
        settings = get_settings()
        clean_query = self._query_text(query)
        if not clean_query:
            return []
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("arXiv retrieval deadline exceeded")
        cache_key = (clean_query.lower(), max(0, int(start)), max(1, min(int(limit), _MAX_PAGE)))
        now = time.monotonic()
        with _SEARCH_CACHE_LOCK:
            cached = _SEARCH_CACHE.get(cache_key)
            if cached and now - cached[0] < _CACHE_TTL_SECONDS:
                return [Source.model_validate(s.model_dump(mode="json")) for s in cached[1]]
        params = {
            "search_query": clean_query if re.search(r"(?:^|\s)(?:cat|ti|abs|all|au|co|jr|id):", clean_query, re.I) else f"all:{clean_query}",
            "start": max(0, int(start)),
            "max_results": max(1, min(int(limit), _MAX_PAGE)),
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        errors: list[str] = []
        response: requests.Response | None = None
        for base_url in _BASE_URLS:
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("arXiv retrieval deadline exceeded")
            try:
                response = self._request(base_url, params, settings.search_timeout, deadline=deadline)
                break
            except TimeoutError:
                raise
            except Exception as exc:
                errors.append(f"{base_url}: {exc}")
        if response is None:
            raise RuntimeError("; ".join(errors) or "arXiv API request failed")
        root = ET.fromstring(response.text)
        sources: list[Source] = []
        for entry in root.findall("a:entry", _ATOM_NS):
            arxiv_id = (entry.findtext("a:id", default="", namespaces=_ATOM_NS) or "").strip()
            title = " ".join((entry.findtext("a:title", default="", namespaces=_ATOM_NS) or "").split())
            summary = " ".join((entry.findtext("a:summary", default="", namespaces=_ATOM_NS) or "").split())
            published = (entry.findtext("a:published", default="", namespaces=_ATOM_NS) or "")[:4]
            journal_ref = " ".join((entry.findtext("arxiv:journal_ref", default="", namespaces=_ATOM_NS) or "").split())
            doi = (entry.findtext("arxiv:doi", default="", namespaces=_ATOM_NS) or "").strip()
            comment = " ".join((entry.findtext("arxiv:comment", default="", namespaces=_ATOM_NS) or "").split())
            authors = ", ".join(
                (a.findtext("a:name", default="", namespaces=_ATOM_NS) or "").strip()
                for a in entry.findall("a:author", _ATOM_NS)
            )
            pdf_url = ""
            for link in entry.findall("a:link", _ATOM_NS):
                if link.attrib.get("title") == "pdf":
                    pdf_url = (link.attrib.get("href") or "").strip()
                    break
            if not title or not arxiv_id:
                continue
            arxiv_key = arxiv_id.rstrip("/").rsplit("/", 1)[-1]
            if not pdf_url:
                pdf_url = f"https://arxiv.org/pdf/{arxiv_key}.pdf"
            sources.append(
                Source(
                    id=arxiv_key,
                    title=title,
                    authors=authors,
                    year=published,
                    venue=journal_ref or "arXiv",
                    journal_ref=journal_ref,
                    publication_status="published_elsewhere" if journal_ref else "preprint",
                    doi=doi,
                    url=arxiv_id,
                    pdf_url=pdf_url,
                    abstract=(
                        summary
                        + (f"\nPublication metadata: {journal_ref}." if journal_ref else "")
                        + (f"\n{comment}" if comment else "")
                    ).strip(),
                    provider=self.name,
                )
            )
        with _SEARCH_CACHE_LOCK:
            _SEARCH_CACHE[cache_key] = (time.monotonic(), [Source.model_validate(s.model_dump(mode="json")) for s in sources])
            if len(_SEARCH_CACHE) > 512:
                oldest = sorted(_SEARCH_CACHE.items(), key=lambda item: item[1][0])[:64]
                for key, _ in oldest:
                    _SEARCH_CACHE.pop(key, None)
        return sources
    def search(self, query: str, limit: int = 8) -> list[Source]:
        """Backward-compatible unbounded API used by non-v79 callers/tests."""
        return self.search_bounded(query, limit, deadline=None)
