from __future__ import annotations
import io
import ipaddress
import re
import socket
import uuid
from urllib.parse import urljoin, urlparse
from html.parser import HTMLParser
class _ArxivHTMLParser(HTMLParser):
    """Small dependency-free extractor for arXiv HTML."""
    def __init__(self):
        super().__init__()
        self.parts=[]; self.skip=0; self.in_heading=False; self.table_count=0; self.figure_count=0
    def handle_starttag(self, tag, attrs):
        tag=tag.lower()
        if tag in {"script","style","nav","noscript"}: self.skip += 1
        if tag in {"h1","h2","h3","h4","h5","h6"}: self.in_heading=True
        if tag=="table": self.table_count += 1
        if tag=="figure": self.figure_count += 1
    def handle_endtag(self, tag):
        tag=tag.lower()
        if tag in {"script","style","nav","noscript"}: self.skip=max(0,self.skip-1)
        if tag in {"h1","h2","h3","h4","h5","h6"}: self.in_heading=False
        if tag in {"p","div","li","tr","h1","h2","h3","h4","h5","h6","figcaption"}: self.parts.append("\n")
    def handle_data(self, data):
        if not self.skip and data.strip(): self.parts.append(data.strip()+" ")
def _extract_arxiv_html(url: str, max_chars: int, timeout: int) -> tuple[str, dict]:
    """Fetch structured arXiv HTML when available; return clean text + artifact counts."""
    response=requests.get(url, timeout=timeout, headers={"User-Agent":"ResearchPilot/79.0 (academic research client)","Accept":"text/html"})
    if response.status_code == 404: raise FileNotFoundError("arXiv HTML unavailable")
    response.raise_for_status()
    parser=_ArxivHTMLParser(); parser.feed(response.text)
    text=re.sub(r"[ \t]+", " ", "".join(parser.parts))
    text=re.sub(r"\n{3,}", "\n\n", text).strip()[:max_chars]
    if len(text) < 5000: raise ValueError("arXiv HTML contained insufficient readable text")
    return text, {"tables":parser.table_count,"figures":parser.figure_count,"captions":[] ,"notes":["source=arxiv_html"]}
import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from ..config import get_settings
from ..schemas import RunEvent, Source
def _validate_remote_url(url: str) -> None:
    """Reject local/reserved destinations before fetching a provider PDF URL.
    Search metadata is external input. In a public deployment, blindly following
    a provider-supplied PDF URL can otherwise become an SSRF primitive. Hostnames
    are resolved before the request and private, loopback, link-local, multicast,
    reserved, or unspecified addresses are rejected. Redirects are validated again
    by the downloader below.
    """
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("unsupported PDF URL scheme")
    host = parsed.hostname.strip().lower()
    try:
        addresses = {ipaddress.ip_address(host)}
    except ValueError:
        try:
            infos = socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80), type=socket.SOCK_STREAM)
            addresses = {ipaddress.ip_address(info[4][0]) for info in infos}
        except (OSError, ValueError):
            raise ValueError("could not resolve PDF host safely") from None
    if not addresses or any(
        (not addr.is_global) or addr.is_private or addr.is_loopback or addr.is_link_local
        or addr.is_multicast or addr.is_reserved or addr.is_unspecified
        for addr in addresses
    ):
        raise ValueError("PDF URL resolves to a private or reserved network address")
def _download_pdf_bytes(url: str, max_bytes: int, timeout: int) -> bytes:
    """Download a bounded PDF payload without buffering an unbounded response.
    Redirects are followed manually so every destination is subject to the same
    SSRF check rather than trusting a redirect from an otherwise public host.
    """
    current_url = url
    for _ in range(5):
        _validate_remote_url(current_url)
        with requests.get(
            current_url, timeout=timeout, headers={"User-Agent": "ResearchPilot/39.0.0 (academic research client)"},
            stream=True, allow_redirects=False,
        ) as resp:
            if resp.status_code in {301, 302, 303, 307, 308}:
                location = resp.headers.get("Location")
                if not location:
                    raise ValueError("PDF host returned a redirect without a destination")
                current_url = urljoin(current_url, location)
                continue
            resp.raise_for_status()
            content_type = (resp.headers.get("Content-Type") or "").lower()
            if content_type and "pdf" not in content_type and "octet-stream" not in content_type:
                raise ValueError(f"expected PDF response, got Content-Type {content_type}")
            content_length = resp.headers.get("Content-Length")
            if content_length:
                try:
                    if int(content_length) > max_bytes:
                        raise ValueError(f"PDF exceeds the {max_bytes // 1_000_000} MB download limit")
                except ValueError as exc:
                    if "download limit" in str(exc):
                        raise
            chunks: list[bytes] = []
            total = 0
            for chunk in resp.iter_content(chunk_size=64 * 1024):
                if not chunk:
                    continue
                total += len(chunk)
                if total > max_bytes:
                    raise ValueError(f"PDF exceeds the {max_bytes // 1_000_000} MB download limit")
                chunks.append(chunk)
            payload = b"".join(chunks)
            if not payload.startswith(b"%PDF-"):
                raise ValueError("downloaded response is not a PDF")
            return payload
    raise ValueError("too many PDF redirects")
def _extract_pdf_text(content: bytes, max_chars: int) -> tuple[str, int]:
    """Extract page-aware text with a quality fallback.
    pypdf is lightweight and remains the first reader, but academic PDFs often
    contain two-column layouts, ligatures and positioned text that pypdf can
    flatten poorly. PyMuPDF is used as a fallback when the first extraction is
    suspiciously sparse or structurally broken. Neither path performs OCR, so
    scanned-only papers are explicitly left unreadable instead of being
    hallucinated into a deep read.
    """
    from pypdf import PdfReader
    settings = get_settings()
    try:
        reader = PdfReader(io.BytesIO(content))
        page_count = len(reader.pages)
    except Exception:
        return "", 0
    if page_count > settings.max_pdf_pages:
        raise ValueError(f"PDF exceeds the {settings.max_pdf_pages}-page processing limit")
    def pypdf_extract() -> str:
        chunks: list[str] = []
        total = 0
        for page_number, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception:
                continue
            text = re.sub(r"[ \t]+", " ", text).strip()
            if not text:
                continue
            page_block = f"[PAGE {page_number}]\n{text}"
            chunks.append(page_block)
            total += len(page_block)
            if total >= max_chars:
                break
        return "\n".join(chunks)[:max_chars]
    primary = pypdf_extract()
    plain = re.sub(r"\[PAGE\s+\d+\]", "", primary, flags=re.I)
    sparse = len(plain.strip()) < min(5000, max_chars // 5) or len(re.findall(r"[.!?]", plain)) < 12
    if not sparse:
        return primary, page_count
    try:
        import fitz
        doc = fitz.open(stream=content, filetype="pdf")
        chunks = []
        total = 0
        for page_number, page in enumerate(doc, start=1):
            text = page.get_text("text", sort=True) or ""
            text = re.sub(r"[ \t]+", " ", text).strip()
            if not text:
                continue
            block = f"[PAGE {page_number}]\n{text}"
            chunks.append(block)
            total += len(block)
            if total >= max_chars:
                break
        doc.close()
        fallback = "\n".join(chunks)[:max_chars]
        if len(re.sub(r"\[PAGE\s+\d+\]", "", fallback, flags=re.I).strip()) > len(plain.strip()):
            return fallback, page_count
    except Exception:
        pass
    return primary, page_count
def _extract_pdf_artifacts(content: bytes) -> tuple[int, int, list[str], list[str]]:
    """Extract layout-level research artifacts without using them for ranking."""
    try:
        import fitz
        doc = fitz.open(stream=content, filetype="pdf")
        table_count = figure_count = 0
        captions: list[str] = []
        notes: list[str] = []
        for page_number, page in enumerate(doc, start=1):
            text = page.get_text("text", sort=True) or ""
            for line in text.splitlines():
                line = re.sub(r"\s+", " ", line).strip()
                if re.match(r"^(table|tab\.)\s*\d+\b", line, re.I):
                    table_count += 1
                    captions.append(f"[PAGE {page_number}] {line[:800]}")
                elif re.match(r"^(figure|fig\.)\s*\d+\b", line, re.I):
                    figure_count += 1
                    captions.append(f"[PAGE {page_number}] {line[:800]}")
            low = text.lower()
            if "table" in low and any(ch.isdigit() for ch in low):
                notes.append(f"page {page_number}: table-like text detected")
            if "figure" in low or "fig." in low:
                notes.append(f"page {page_number}: figure/caption text detected")
        doc.close()
        return table_count, figure_count, list(dict.fromkeys(captions))[:200], list(dict.fromkeys(notes))[:200]
    except Exception:
        return 0, 0, [], []
def _extraction_quality(text: str, page_count: int, max_chars: int) -> tuple[float, float, bool]:
    """Assess whether parsed text is complete enough for deep research.
    A paper is RED-eligible only when the extraction covers nearly all pages,
    has substantial text, and was not cut off by the configured character cap.
    This rejects scanned/partially parsed PDFs instead of silently deep-reading
    fragments.
    """
    plain = re.sub(r"\[PAGE\s+\d+\]", "", text or "", flags=re.I).strip()
    page_blocks = re.findall(
        r"\[PAGE\s+(\d+)\]\s*\n(.*?)(?=\n\[PAGE\s+\d+\]|\Z)",
        text or "",
        flags=re.I | re.S,
    )
    if page_count <= 0:
        return 0.0, 0.0, False
    readable_pages = sum(1 for _, body in page_blocks if len(body.strip()) >= 80)
    coverage = readable_pages / page_count
    density = min(1.0, len(plain) / max(2500.0, page_count * 500.0))
    not_truncated = len(text or "") < max_chars
    quality = max(0.0, min(1.0, 0.70 * coverage + 0.30 * density))
    complete = bool(
        len(plain) >= 2500
        and coverage >= 0.85
        and density >= 0.55
        and (not_truncated or quality >= 0.78)
    )
    return round(quality, 4), round(coverage, 4), complete
def enrich_with_full_text(sources: list[Source], max_fetch: int | None = None, cancel_check=None) -> list[RunEvent]:
    """Acquire verified arXiv full text with bounded per-paper parallelism.
    Search itself remains sequential because arXiv asks clients to throttle API
    requests. Full-text HTML/PDF downloads are independent resources, so a small
    worker pool prevents one slow paper from serialising the entire 15-paper
    report. Every worker still has its own absolute paper deadline.
    """
    settings = get_settings()
    if not settings.fetch_full_text:
        return []
    fetch_cap = settings.max_full_text_sources if max_fetch is None else max(0, int(max_fetch))
    candidates = [
        s for s in sources
        if s.pdf_url and (not s.full_text or not getattr(s, "full_text_complete", False))
    ][:fetch_cap]
    for source in sources:
        if source.full_text.strip() and not source.full_text_parsed:
            source.full_text_parsed = True
            source.full_text_chars = len(source.full_text)
            source.full_text_pages = len(re.findall(r"\[PAGE\s+\d+\]", source.full_text, flags=re.I))
            source.full_text_quality, source.full_text_page_coverage, source.full_text_complete = _extraction_quality(
                source.full_text, source.full_text_pages, settings.full_text_max_chars
            )
            source.reading_basis = "full_text" if source.full_text_complete else "abstract_only"
            source.pdf_acquired = source.pdf_acquired or bool(source.pdf_url)
    def process(source: Source) -> list[RunEvent]:
        if cancel_check is not None and cancel_check():
            raise RuntimeError("Job cancelled")
        events: list[RunEvent] = []
        try:
            paper_deadline = time.perf_counter() + max(10, int(getattr(settings, "full_text_paper_timeout", 60)))
            def remaining_timeout() -> int:
                remaining = int(paper_deadline - time.perf_counter())
                return max(1, min(int(settings.search_timeout), remaining))
            urls = [source.pdf_url]
            if source.provider == "arxiv" and source.url:
                arxiv_key = source.url.rstrip("/").rsplit("/", 1)[-1]
                urls.extend([
                    f"https://arxiv.org/pdf/{arxiv_key}.pdf",
                    f"https://export.arxiv.org/pdf/{arxiv_key}.pdf",
                    f"https://arxiv.org/pdf/{arxiv_key}",
                ])
            last_error = None
            if source.provider == "arxiv" and source.url:
                arxiv_key = source.url.rstrip("/").rsplit("/", 1)[-1]
                html_urls = [f"https://arxiv.org/html/{arxiv_key}", f"https://export.arxiv.org/html/{arxiv_key}"]
                for html_url in html_urls:
                    if cancel_check is not None and cancel_check():
                        raise RuntimeError("Job cancelled")
                    if time.perf_counter() >= paper_deadline:
                        break
                    try:
                        html_text, meta = _extract_arxiv_html(html_url, settings.full_text_max_chars, remaining_timeout())
                        source.full_text = html_text
                        source.full_text_parsed = True
                        source.full_text_chars = len(html_text)
                        source.full_text_pages = max(1, len(re.findall(r"(?im)^\s*(?:section|\d+(?:\.\d+)*)\b", html_text)))
                        source.full_text_quality = 1.0
                        source.full_text_page_coverage = 1.0
                        source.full_text_complete = len(html_text) < settings.full_text_max_chars or len(html_text) >= 5000
                        source.reading_basis = "full_text" if source.full_text_complete else "abstract_only"
                        source.pdf_table_count = meta["tables"]
                        source.pdf_figure_count = meta["figures"]
                        source.pdf_caption_text = meta["captions"]
                        source.pdf_artifact_notes = meta["notes"]
                        source.full_text_error = ""
                        if source.full_text_complete:
                            events.append(RunEvent(stage="fulltext", detail=f"arXiv HTML acquired '{source.title[:60]}' ({source.full_text_chars} chars, tables={source.pdf_table_count}, figures={source.pdf_figure_count})"))
                            return events
                    except Exception as exc:
                        last_error = exc
            for pdf_url in dict.fromkeys(urls):
                if cancel_check is not None and cancel_check():
                    raise RuntimeError("Job cancelled")
                if time.perf_counter() >= paper_deadline:
                    last_error = TimeoutError("per-paper full-text extraction budget exhausted")
                    break
                try:
                    content = _download_pdf_bytes(pdf_url, settings.max_pdf_bytes, remaining_timeout())
                    source.pdf_acquired = True
                    text, page_count = _extract_pdf_text(content, settings.full_text_max_chars)
                    if text.strip() and len(re.sub(r"\[PAGE\s+\d+\]", "", text, flags=re.I).strip()) >= 800:
                        quality, coverage, complete = _extraction_quality(text, page_count, settings.full_text_max_chars)
                        tables, figures, captions, artifact_notes = _extract_pdf_artifacts(content)
                        if complete:
                            source.full_text = text
                            source.full_text_parsed = True
                            source.full_text_chars = len(text)
                            source.full_text_pages = len(re.findall(r"\[PAGE\s+\d+\]", text, flags=re.I))
                            source.full_text_quality = quality
                            source.full_text_page_coverage = coverage
                            source.pdf_table_count = tables
                            source.pdf_figure_count = figures
                            source.pdf_caption_text = captions
                            source.pdf_artifact_notes = artifact_notes
                            source.full_text_complete = True
                            source.reading_basis = "full_text"
                            source.full_text_error = ""
                            events.append(RunEvent(stage="fulltext", detail=f"PDF acquired and parsed '{source.title[:60]}' ({source.full_text_pages} pages, {source.full_text_chars} chars)"))
                            return events
                        last_error = ValueError(f"PDF text extraction was incomplete (quality={quality:.2f}, page coverage={coverage:.0%})")
                    else:
                        last_error = ValueError("PDF parsed but contained insufficient extractable text")
                except Exception as exc:
                    last_error = exc
            source.full_text_complete = False
            source.full_text_error = str(last_error or "PDF could not be parsed")[:500]
            source.reading_basis = "abstract_only" if source.abstract else "unavailable"
            events.append(RunEvent(stage="fulltext", detail=f"PDF read failed for '{source.title[:60]}': {source.full_text_error}", ok=False))
            return events
        except Exception as exc:
            if str(exc) == "Job cancelled":
                raise
            source.full_text_error = str(exc)[:500]
            source.reading_basis = "abstract_only" if source.abstract else "unavailable"
            return [RunEvent(stage="fulltext", detail=f"PDF acquisition failed for '{source.title[:60]}': {exc}", ok=False)]
    if not candidates:
        return []
    workers = min(4, max(1, len(candidates)))
    events: list[RunEvent] = []
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="researchpilot-fulltext") as pool:
        futures = {pool.submit(process, source): source for source in candidates}
        for future in as_completed(futures):
            if cancel_check is not None and cancel_check():
                for pending in futures:
                    pending.cancel()
                raise RuntimeError("Job cancelled")
            try:
                events.extend(future.result())
            except Exception as exc:
                if str(exc) == "Job cancelled":
                    raise
                source = futures[future]
                source.full_text_error = str(exc)[:500]
                events.append(RunEvent(stage="fulltext", detail=f"Full-text worker failed for '{source.title[:60]}': {exc}", ok=False))
    return events
def source_from_uploaded_pdf(filename: str, content: bytes, max_chars: int | None = None) -> Source:
    """Build a first-class Source from a user PDF with metadata-aware parsing.
    The uploader is a deliberate researcher input, so the extracted title,
    authors, DOI and abstract should be useful to ranking and citation output
    instead of simply mirroring the filename.
    """
    settings = get_settings()
    if len(content) > settings.max_pdf_bytes:
        raise ValueError(f"PDF exceeds the {settings.max_pdf_bytes // 1_000_000} MB upload limit")
    max_chars = settings.full_text_max_chars if max_chars is None else max(0, int(max_chars))
    text, page_count = _extract_pdf_text(content, max_chars)
    tables, figures, captions, artifact_notes = _extract_pdf_artifacts(content)
    filename_title = filename.rsplit(".", 1)[0].replace("_", " ").replace("-", " ").strip() or "Uploaded paper"
    title, authors = filename_title, ""
    doi = ""
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(content))
        metadata = reader.metadata or {}
        meta_title = str(getattr(metadata, "title", "") or "").strip()
        meta_author = str(getattr(metadata, "author", "") or "").strip()
        if 8 < len(meta_title) < 240 and meta_title.lower() not in {"untitled", "microsoft word"}:
            title = meta_title
        if meta_author and len(meta_author) < 300:
            authors = meta_author.replace(";", ", ")
    except Exception:
        pass
    lines = [" ".join(line.split()) for line in (text or "").splitlines() if line.strip()]
    if title == filename_title:
        for idx, line in enumerate(lines[:45]):
            low = line.lower()
            if not (20 <= len(line) <= 220):
                continue
            if low.startswith(("abstract", "keywords", "arxiv", "doi:", "copyright")):
                continue
            if re.search(r"@|\b(university|department|school|laboratory|institute)\b", low):
                continue
            if re.fullmatch(r"[A-Z][A-Za-z .'-]+(?:, [A-Z][A-Za-z .'-]+){1,}", line):
                continue
            title = line
            if idx + 1 < len(lines) and not authors:
                candidate_author = lines[idx + 1]
                if 5 < len(candidate_author) < 260 and not candidate_author.lower().startswith("abstract"):
                    authors = candidate_author
            break
    doi_match = re.search(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+", text or "", flags=re.I)
    if doi_match:
        doi = doi_match.group(0).rstrip(".,;)")
    abstract = ""
    for idx, line in enumerate(lines):
        if line.lower() in {"abstract", "abstract."}:
            abstract = " ".join(lines[idx + 1:idx + 8])[:1800]
            break
    abstract = abstract or text[:1800]
    return Source(
        id=f"upload-{uuid.uuid4().hex[:10]}",
        title=title,
        authors=authors,
        doi=doi,
        provider="user_upload",
        abstract=abstract,
        full_text=text,
        pdf_acquired=True,
        full_text_parsed=bool(text.strip()),
        full_text_pages=len(re.findall(r"\[PAGE\s+\d+\]", text, flags=re.I)),
        full_text_chars=len(text),
        full_text_quality=_extraction_quality(text, page_count, max_chars)[0],
        full_text_page_coverage=_extraction_quality(text, page_count, max_chars)[1],
        full_text_complete=_extraction_quality(text, page_count, max_chars)[2],
        pdf_table_count=tables,
        pdf_figure_count=figures,
        pdf_caption_text=captions,
        pdf_artifact_notes=artifact_notes,
        reading_basis="full_text" if _extraction_quality(text, page_count, max_chars)[2] else ("abstract_only" if abstract else "unavailable"),
        url="",
        role="no_read",
    )
