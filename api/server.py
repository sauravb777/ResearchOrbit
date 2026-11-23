from __future__ import annotations
import hmac
import json
import logging
import os
import re
import sqlite3
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field, field_validator
from api.jobs import JobManager
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from researchpilot import __version__ as RESEARCHPILOT_VERSION
from researchpilot.config import get_settings
from researchpilot.export import (content_disposition_header, markdown_to_docx,
                                  markdown_to_latex, markdown_to_pdf,
                                  safe_filename_stem, sources_to_bibtex)
from researchpilot.pipeline import intelligence as intelligence_mod
from researchpilot.pipeline import relevance as relevance_mod
from researchpilot.pipeline.fulltext import (enrich_with_full_text,
                                             source_from_uploaded_pdf)
from researchpilot.pipeline.orchestrator import ResearchAgent
from researchpilot.pipeline.ranker import rank_sources
from researchpilot.schemas import (ResearchPlan, ResearchResult, RunEvent,
                                   Source)
from researchpilot.storage import ResultStore
logger = logging.getLogger("researchpilot.api")
settings = get_settings()
store = ResultStore(settings.db_path)
agent = ResearchAgent(dossier_cache=store)
_background_enrichment = ThreadPoolExecutor(max_workers=1, thread_name_prefix="researchpilot-background")
jobs = JobManager(
    workers=max(1, int(os.getenv("RESEARCHPILOT_JOB_WORKERS", "2"))),
    db_path=os.getenv("RESEARCHPILOT_JOB_DB", "data/jobs.db"),
)
DISCOVERY_JOB_TIMEOUT_SECONDS = max(60, int(os.getenv("DISCOVERY_JOB_TIMEOUT_SECONDS", "600")))
REPORT_JOB_TIMEOUT_SECONDS = None
MAX_UPLOAD_BYTES = int(os.getenv("RESEARCHPILOT_MAX_UPLOAD_BYTES", str(15 * 1024 * 1024)))
MAX_UPLOAD_FILES = max(1, int(os.getenv("RESEARCHPILOT_MAX_UPLOAD_FILES", "16")))
MAX_TOTAL_UPLOAD_BYTES = max(MAX_UPLOAD_BYTES, int(os.getenv("RESEARCHPILOT_MAX_TOTAL_UPLOAD_BYTES", str(64 * 1024 * 1024))))
API_KEY = os.getenv("RESEARCHPILOT_API_KEY", "").strip()
_WORKSPACE_ID: ContextVar[str | None] = ContextVar("researchpilot_workspace_id", default=None)
_WORKSPACE_DB = os.getenv("RESEARCHPILOT_WORKSPACE_DB", "data/workspaces.db")
Path(_WORKSPACE_DB).parent.mkdir(parents=True, exist_ok=True)
with sqlite3.connect(_WORKSPACE_DB) as _db:
    _db.execute("CREATE TABLE IF NOT EXISTS workspace_runs (run_id TEXT PRIMARY KEY, owner_id TEXT NOT NULL)")
    _db.execute("CREATE TABLE IF NOT EXISTS workspace_jobs (job_id TEXT PRIMARY KEY, owner_id TEXT NOT NULL)")
def _workspace_id() -> str:
    value = _WORKSPACE_ID.get()
    if not value:
        raise HTTPException(400, "Missing browser workspace. Refresh the page and try again.")
    return value
def _bind_job(job_id: str, owner_id: str | None = None) -> None:
    owner = owner_id or _workspace_id()
    with sqlite3.connect(_WORKSPACE_DB) as db:
        db.execute("INSERT OR REPLACE INTO workspace_jobs(job_id, owner_id) VALUES (?, ?)", (job_id, owner))
def _job_owner(job_id: str) -> str | None:
    job = jobs.get(job_id)
    if job and job.owner_id:
        return job.owner_id
    with sqlite3.connect(_WORKSPACE_DB) as db:
        row = db.execute("SELECT owner_id FROM workspace_jobs WHERE job_id = ?", (job_id,)).fetchone()
    return row[0] if row else None
def _bind_run(run_id: str, owner_id: str | None = None) -> None:
    owner = owner_id or _workspace_id()
    with sqlite3.connect(_WORKSPACE_DB) as db:
        db.execute("INSERT OR REPLACE INTO workspace_runs(run_id, owner_id) VALUES (?, ?)", (run_id, owner))
def _run_owner(run_id: str) -> str | None:
    with sqlite3.connect(_WORKSPACE_DB) as db:
        row = db.execute("SELECT owner_id FROM workspace_runs WHERE run_id = ?", (run_id,)).fetchone()
    return row[0] if row else None
def _assert_job_owner(job_id: str) -> None:
    if _job_owner(job_id) != _workspace_id():
        raise HTTPException(404, "Research job not found")
def _background_creator_pack(run_id: str) -> None:
    """Build secondary Creator Pack material after report publication.
    This deliberately runs outside the user-visible report critical path. Chat
    requests use a higher scheduler priority, so opening Chat immediately after
    publication does not have to wait behind this optional artifact.
    """
    try:
        current = store.load(run_id)
        if current is None or current.report_type != "standard_report" or not agent.llm.available:
            return
        from researchpilot.pipeline.creator_pack import build_creator_pack
        pack, used = build_creator_pack(current.question, current.report_markdown, current.evidence, current.paper_analyses, agent.llm)
        if used:
            current.creator_pack = pack
            store.save(current)
    except Exception:
        logger.exception("Background creator pack failed for %s", run_id)
def _schedule_background_enrichment(result: ResearchResult) -> None:
    if result.report_type == "standard_report" and agent.llm.available:
        _background_enrichment.submit(_background_creator_pack, result.run_id)
_chat_cancel_events: dict[str, threading.Event] = {}
_chat_cancel_lock = threading.Lock()
app = FastAPI(title="ResearchOrbit API", version=RESEARCHPILOT_VERSION)
@app.middleware("http")
async def workspace_isolation(request: Request, call_next):
    path = request.url.path
    if not path.startswith("/api/") or path in {"/api/health", "/api/readiness"} or request.method == "OPTIONS":
        return await call_next(request)
    workspace = request.headers.get("x-workspace-id", "").strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]{32,80}", workspace):
        return Response(content="Missing or invalid browser workspace", status_code=400)
    token = _WORKSPACE_ID.set(workspace)
    try:
        parts = path.strip("/").split("/")
        if len(parts) >= 3 and parts[1] == "jobs":
            if _job_owner(parts[2]) != workspace:
                return Response(content="Research job not found", status_code=404)
        if len(parts) >= 3 and parts[1] == "research" and parts[2] not in {"discover", "run"}:
            if _run_owner(parts[2]) != workspace:
                return Response(content="Research case not found", status_code=404)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store, private"
        response.headers["Vary"] = "Origin, X-Workspace-ID"
        return response
    finally:
        _WORKSPACE_ID.reset(token)
@app.middleware("http")
async def api_key_guard(request, call_next):
    if API_KEY and request.url.path not in {"/api/health", "/api/readiness"}:
        if not hmac.compare_digest(request.headers.get("x-api-key", ""), API_KEY):
            return Response(content="Unauthorized", status_code=401)
    return await call_next(request)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(dict.fromkeys(
        x.strip() for x in (
            os.getenv(
                "RESEARCHPILOT_CORS_ORIGINS",
                "http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,http://127.0.0.1:5173,http://localhost:8080",
            ).split(",") + [
                "https://researchorbit.vercel.app",
            ]
        ) if x.strip()
    )),
    allow_credentials=False,
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key", "X-Workspace-ID"],
    expose_headers=["Content-Disposition", "Content-Length"],
)
class ResearchRequest(BaseModel):
    question: str = Field(min_length=3, max_length=12000)
    title: str = Field(min_length=1, max_length=240)
    report_type: Literal[
        "standard_report", "literature_review", "state_of_art", "gap_analysis", "comparison",
        "evidence_review", "research_proposal", "paper_analysis"
    ] = "standard_report"
    depth: Literal["standard", "deep"] = "deep"
    discovery_job_id: str = ""
    selected_source_ids: list[str] = Field(default_factory=list, max_length=5)
    selection_overridden: bool | None = None
    analysis_selection: dict[str, str] = Field(default_factory=dict)
    extra_sources: list[dict[str, Any]] = Field(default_factory=list, max_length=30)
    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        value = value.strip()
        if len(value) < 3:
            raise ValueError("question must contain at least 3 non-whitespace characters")
        return value
    @field_validator("title")
    @classmethod
    def validate_title(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("report title is required")
        return value
class DiscoveryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=12000)
    title: str = Field(min_length=1, max_length=240)
    report_type: Literal[
        "standard_report", "literature_review", "state_of_art", "gap_analysis", "comparison",
        "evidence_review", "research_proposal", "paper_analysis"
    ] = "standard_report"
    depth: Literal["standard", "deep"] = "deep"
    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        value = value.strip()
        if len(value) < 3:
            raise ValueError("question must contain at least 3 non-whitespace characters")
        return value
    @field_validator("title")
    @classmethod
    def validate_title(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("report title is required")
        return value
class ChatRequest(BaseModel):
    run_id: str
    question: str = Field(min_length=1, max_length=12000)
    request_id: str = Field(default="", max_length=120)
    mode: Literal["Ask", "Verify", "ask", "verify"] = "Ask"
    depth: Literal["one_line", "compact", "elaborate", "eli5"] = "compact"
    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question must not be blank")
        return value
class SelectedTextRequest(BaseModel):
    run_id: str
    selected_text: str = Field(min_length=1, max_length=20000)
    question: str = Field(default="Explain this passage in the context of the report.", max_length=4000)
    @field_validator("selected_text")
    @classmethod
    def validate_selected_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("selected_text must not be blank")
        return value
class DiscoveryChatRequest(BaseModel):
    question: str = Field(min_length=2, max_length=6000)
class RebuildRequest(BaseModel):
    mode: Literal["new", "combine"]
    arxiv_ids: list[str] = Field(min_length=1, max_length=3)
    add_supporting: bool = False
class CaseUpdateRequest(BaseModel):
    name: str | None = Field(default=None, max_length=240)
    pinned: bool | None = None
    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("case name must not be blank")
        return value
def _reduced_depth_warning(metrics: dict[str, Any] | None) -> str | None:
    """Return a user-facing depth caveat only when the production reading contract was not met."""
    metrics = metrics or {}
    reading_regime = str(metrics.get("reading_regime", ""))
    deep_read_complete = int(metrics.get("papers_deep_read", 0) or 0) >= 1
    council_required = reading_regime != "feynman_multistage_full_research"
    if deep_read_complete and (not council_required or metrics.get("council_used", False)):
        return None
    return (
        "Delivered in reduced-depth mode: full-text deep reading or the required research "
        "reasoning stage did not fully complete (check LLM key/quota). Treat conclusions as provisional."
    )
def _sanitize_text(value: str) -> str:
    return "".join("\ufffd" if 0xD800 <= ord(ch) <= 0xDFFF else ch for ch in value)
def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump(mode="python"))
    if isinstance(value, str):
        return _sanitize_text(value)
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {_sanitize_text(str(k)): _jsonable(v) for k, v in value.items()}
    return value
def _source_summary(source: Any) -> dict[str, Any]:
    data = _jsonable(source)
    return {
        "id": data.get("id", ""),
        "title": data.get("title", "Untitled source"),
        "authors": data.get("authors", ""),
        "year": data.get("year", ""),
        "venue": data.get("venue", ""),
        "url": data.get("url", ""),
        "pdf_url": data.get("pdf_url", ""),
        "abstract": data.get("abstract", ""),
        "role": data.get("role", "core"),
        "score": data.get("score"),
        "relevance_score": data.get("relevance_score", data.get("score", 0) * 100 if data.get("score") is not None else 0),
        "importance_score": data.get("importance_score", 0),
        "corpus_value_score": data.get("corpus_value_score", 0),
        "ranking_confidence": data.get("ranking_confidence", 0),
        "ranking_components": data.get("ranking_components", {}),
        "missing_ranking_signals": data.get("missing_ranking_signals", []),
        "doi": data.get("doi", ""),
        "provider": data.get("provider", ""),
        "citation_count": data.get("citation_count"),
        "relevance_reasons": data.get("relevance_reasons", []),
        "citation_tag": data.get("citation_tag", ""),
    }
def _result_payload(result: ResearchResult) -> dict[str, Any]:
    data = _jsonable(result)
    data["sources"] = [_source_summary(source) for source in result.sources]
    data["metrics"] = _jsonable(result.metrics)
    data["report_markdown"] = result.report_markdown or ""
    data["run_id"] = result.run_id
    return data
def _candidate_payload(job_result: dict[str, Any]) -> dict[str, Any]:
    plan = job_result["plan"]
    return {
        "plan": _jsonable(plan),
        "candidates": [_source_summary(s) for s in job_result["candidates"]],
        "events": _jsonable(job_result.get("events", [])),
        "usage": _jsonable(job_result.get("usage", {})),
    }
def _require_result(run_id: str) -> ResearchResult:
    result = store.load(run_id)
    if result is None or _run_owner(run_id) != _workspace_id():
        raise HTTPException(404, "Research case not found")
    return result
@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "researchorbit", "version": app.version}
@app.get("/api/readiness")
def readiness() -> dict[str, Any]:
    return {
        "status": "ready",
        "database": "configured" if settings.db_path else "missing",
        "workers": "available",
        "llm_available": agent.llm.available,
        "version": app.version,
    }
@app.get("/api/llm/health")
def llm_health() -> dict[str, Any]:
    """Expose provider/model diagnostics without ever returning API keys."""
    return {"version": app.version, **agent.llm.health()}
@app.get("/api/metrics")
def metrics() -> dict[str, Any]:
    active = completed = failed = 0
    for job_id in list(jobs._jobs):
        item = jobs.get(job_id)
        if not item:
            continue
        if item.status in {"queued", "running"}:
            active += 1
        elif item.status == "completed":
            completed += 1
        elif item.status in {"failed", "timed_out"}:
            failed += 1
    return {"active_jobs": active, "completed_jobs": completed, "failed_jobs": failed}
@app.get("/api/history")
def history() -> list[dict[str, Any]]:
    owner = _workspace_id()
    return _jsonable([item for item in store.list_cases(limit=200) if _run_owner(item.get("run_id", "")) == owner])
@app.post("/api/research/discover")
def discover(payload: DiscoveryRequest) -> dict[str, Any]:
    """Create a reviewable candidate-ranking job before expensive deep reading."""
    def execute(job):
        if job.cancel_event.is_set():
            raise RuntimeError("Job cancelled")
        jobs.update(job.job_id, stage="planning", progress=10, message="Planning search strategy")
        discover_kwargs = {
            "report_type": payload.report_type,
            "report_title": payload.title.strip(),
            "candidate_limit": 15,
        }
        import inspect
        if "cancel_check" in inspect.signature(agent.discover_candidates).parameters:
            discover_kwargs["cancel_check"] = job.cancel_event.is_set
        plan, candidates, events, _, usage = agent.discover_candidates(
            payload.question.strip(), **discover_kwargs
        )
        if job.cancel_event.is_set():
            raise RuntimeError("Job cancelled")
        if not candidates:
            failures = [e.detail for e in events if getattr(e, "ok", True) is False and getattr(e, "detail", "")]
            if failures:
                raise RuntimeError("No usable arXiv papers were found. Search errors: " + " | ".join(failures[:3]))
            raise RuntimeError("No usable arXiv papers were found for this research question. Try a more specific research question.")
        jobs.update(job.job_id, stage="screening", progress=90, message=f"Ranked {len(candidates)} candidate papers")
        return {"plan": plan, "candidates": candidates, "events": events, "usage": usage}
    job = jobs.submit(execute, timeout_seconds=DISCOVERY_JOB_TIMEOUT_SECONDS, owner_id=_workspace_id())
    _bind_job(job.job_id, job.owner_id)
    return jobs.public(job.job_id)
@app.post("/api/research/discover/upload")
async def discover_upload(discovery_job_id: str, files: list[UploadFile] = File(...)) -> dict[str, Any]:
    """Compatibility endpoint for synchronous API clients.
    The React application uses ``/discover/upload/job`` so the upload analysis and
    ranking itself has a durable, pollable backend job.
    """
    payload = await _prepare_upload_batch(files, max_files=MAX_UPLOAD_FILES)
    job = _submit_upload_ranking_job(discovery_job_id, payload)
    done = await _wait_for_job_completion(job.job_id)
    return done.get("result", {"candidates": []})
async def _prepare_upload_batch(files: list[UploadFile], *, max_files: int) -> list[tuple[str, bytes]]:
    if not files:
        raise HTTPException(400, "Upload at least one PDF")
    if len(files) > max_files:
        raise HTTPException(413, f"Upload at most {max_files} PDFs at once")
    payload: list[tuple[str, bytes]] = []
    total = 0
    for upload in files:
        filename = (upload.filename or "uploaded.pdf").strip() or "uploaded.pdf"
        body = await upload.read(MAX_UPLOAD_BYTES + 1)
        if len(body) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"{filename}: uploaded file exceeds {MAX_UPLOAD_BYTES} bytes")
        if not filename.lower().endswith(".pdf") or not body.startswith(b"%PDF"):
            raise HTTPException(415, f"{filename}: only valid PDF uploads are supported")
        total += len(body)
        if total > MAX_TOTAL_UPLOAD_BYTES:
            raise HTTPException(413, f"Combined upload batch exceeds {MAX_TOTAL_UPLOAD_BYTES} bytes")
        payload.append((filename, body))
    return payload
def _submit_upload_ranking_job(discovery_job_id: str, payload: list[tuple[str, bytes]]):
    _assert_job_owner(discovery_job_id)
    discovery = jobs.get(discovery_job_id)
    if discovery is None or discovery.status != "completed" or not isinstance(discovery.result, dict):
        raise HTTPException(409, "Candidate screening job is not complete")
    plan_raw = discovery.result.get("plan")
    plan = plan_raw if isinstance(plan_raw, ResearchPlan) else ResearchPlan.model_validate(plan_raw)
    def execute(job):
        jobs.update(job.job_id, stage="upload_reading", progress=15, message="Reading uploaded papers")
        parsed: list[Source] = []
        total = len(payload)
        for index, (filename, body) in enumerate(payload, start=1):
            if job.cancel_event.is_set():
                raise RuntimeError("Job cancelled")
            source = source_from_uploaded_pdf(filename, body)
            source.role = "no_read"
            if source.full_text or source.abstract:
                parsed.append(source)
            jobs.update(job.job_id, stage="upload_reading", progress=15 + int(35 * index / max(1, total)), message=f"Read {index} of {total} uploaded paper(s)")
        jobs.update(job.job_id, stage="upload_ranking", progress=55, message="Analyzing and ranking uploaded papers")
        if parsed:
            if semantic_rerank(plan.question, parsed):
                jobs.update(job.job_id, stage="upload_ranking", progress=62, message="Applying semantic relevance ranking")
            else:
                rank_sources(plan.question, parsed)
            intent, _, _ = intelligence_mod.build_intent(plan.question, plan.report_type, "standard", agent.llm)
            assessments, _ = intelligence_mod.assess_relevance(plan.question, plan, parsed, agent.llm)
            relevance_scores = relevance_mod.score_all(plan.question, plan, parsed)
            for index, source in enumerate(parsed, start=1):
                if job.cancel_event.is_set():
                    raise RuntimeError("Job cancelled")
                score = relevance_scores[source.id]
                assessment = assessments.get(source.id)
                if assessment:
                    from researchpilot.pipeline.relevance_engine import (
                        apply_ranking, score_assessment)
                    assessment = score_assessment(assessment, source, plan.report_type)
                    source.score = assessment.overall
                else:
                    source.score = score.total
                    source.relevance_score = round(source.score * 100, 2)
                source.relevance_reasons = list(dict.fromkeys((assessment.reasons if assessment else []) + score.reasons))[:6]
                if assessment:
                    source.relevance_score = round(assessment.overall * 100, 2)
                    source.importance_score = round(assessment.importance * 100, 2)
                    source.corpus_value_score = round(assessment.coverage_value * 100, 2)
                    source.ranking_confidence = round(assessment.ranking_confidence * 100, 2)
                jobs.update(job.job_id, stage="upload_ranking", progress=62 + int(28 * index / max(1, len(parsed))), message=f"Ranked {index} of {len(parsed)} uploaded paper(s)")
            _, decisions = intelligence_mod.build_corpus(parsed, {k:v.total for k,v in relevance_scores.items()}, assessments, plan.report_type, "standard", settings.corpus_safety_ceiling)
            state = {d.source_id: d.state for d in decisions}
            for source in parsed:
                source.role = state.get(source.id, "no_read")
        parsed.sort(key=lambda source: (-int(source.role == "deep_read"), -source.score, source.title.lower()))
        jobs.update(job.job_id, stage="upload_complete", progress=95, message=f"Finished ranking {len(parsed)} uploaded paper(s)")
        return {"candidates": [_jsonable(source) for source in parsed]}
    job = jobs.submit(execute, owner_id=_workspace_id())
    _bind_job(job.job_id, job.owner_id)
    return job
@app.post("/api/research/discover/upload/job")
async def discover_upload_job(discovery_job_id: str, files: list[UploadFile] = File(...)) -> dict[str, Any]:
    """Queue uploaded-paper parsing and relevance ranking as a real backend job."""
    payload = await _prepare_upload_batch(files, max_files=MAX_UPLOAD_FILES)
    job = _submit_upload_ranking_job(discovery_job_id, payload)
    return jobs.public(job.job_id)
async def _wait_for_job_completion(job_id: str) -> dict[str, Any]:
    for _ in range(1800):
        status = jobs.public(job_id) or {}
        if status.get("status") == "completed":
            full = get_job(job_id)
            return full
        if status.get("status") == "failed":
            raise HTTPException(500, status.get("error") or "Upload processing failed")
        if status.get("status") == "cancelled":
            raise HTTPException(409, "Upload processing was cancelled")
        await __import__("asyncio").sleep(0.1)
    raise HTTPException(504, "Upload processing timed out")
@app.post("/api/research/run")
def run_research(payload: ResearchRequest) -> dict[str, Any]:
    """Queue the final report run using the exact researcher-selected corpus when supplied."""
    selected_sources: list[Source] | None = None
    if payload.discovery_job_id:
        _assert_job_owner(payload.discovery_job_id)
        discovery = jobs.get(payload.discovery_job_id)
        if discovery is None or discovery.status != "completed" or not isinstance(discovery.result, dict):
            raise HTTPException(409, "Candidate screening job is not complete or is no longer available")
        discovery_plan = discovery.result.get("plan")
        if discovery_plan is not None:
            discovered_question = getattr(discovery_plan, "question", None) or (discovery_plan.get("question") if isinstance(discovery_plan, dict) else None)
            discovered_type = getattr(discovery_plan, "report_type", None) or (discovery_plan.get("report_type") if isinstance(discovery_plan, dict) else None)
            type_compatible = discovered_type == payload.report_type or {discovered_type, payload.report_type} <= {"standard_report", "literature_review"}
            if discovered_question != payload.question.strip() or not type_compatible:
                raise HTTPException(409, "Candidate screening context does not match the requested research run")
        candidate_items = discovery.result.get("candidates", [])
        candidates = [item if isinstance(item, Source) else Source.model_validate(item) for item in candidate_items]
        requested_ids = list(dict.fromkeys(payload.selected_source_ids or [])) if (payload.selection_overridden is True or payload.selection_overridden is None) else []
        if len(requested_ids) > 5:
            raise HTTPException(422, "Deep Read supports a maximum of 5 papers. Select 5 or fewer papers.")
        candidate_by_id = {s.id: s for s in candidates}
        if payload.selection_overridden is True and not requested_ids:
            raise HTTPException(422, "The explicit deep-read selection is empty. Select at least one paper and try again.")
        if requested_ids:
            missing_ids = [sid for sid in requested_ids if sid not in candidate_by_id]
            if missing_ids:
                raise HTTPException(
                    422,
                    "The selected paper IDs no longer match the discovery results: " + ", ".join(missing_ids[:5])
                    + ". Refresh the paper list and select again.",
                )
            selected_sources = [candidate_by_id[sid] for sid in requested_ids]
            if len(selected_sources) != len(requested_ids):
                raise HTTPException(422, "The backend could not resolve every explicitly selected paper. Refresh and try again.")
        else:
            selected_sources = [s for s in candidates if s.role == "deep_read"]
        for source in candidates:
            source.role = "deep_read" if source in selected_sources else "no_read"
        if not selected_sources:
            raise HTTPException(422, "Select at least one paper for deep reading before generating the report")
        requested_set = set(requested_ids)
        normalized_uploads = []
        for raw in payload.extra_sources:
            source = Source.model_validate(raw)
            if requested_ids:
                source.role = "deep_read" if source.id in requested_set else "no_read"
            normalized_uploads.append(source)
        payload.extra_sources = [source.model_dump() for source in normalized_uploads]
        seen = set(); deduped = []
        for source in selected_sources:
            if source.dedupe_key() in seen: continue
            seen.add(source.dedupe_key()); deduped.append(source)
        selected_sources = deduped
    def execute(job):
        if job.cancel_event.is_set():
            raise RuntimeError("Job cancelled")
        jobs.update(job.job_id, stage="retrieval", progress=20, message="Preparing the selected research corpus")
        if job.cancel_event.is_set():
            raise RuntimeError("Job cancelled")
        uploaded_sources = [Source.model_validate(raw) for raw in payload.extra_sources]
        result = agent.run(
            question=payload.question,
            report_title=payload.title,
            report_type=payload.report_type,
            analysis_depth=payload.depth,
            preselected_sources=selected_sources if payload.selection_overridden is not False else None,
            automatic_candidates=candidates if payload.selection_overridden is False else None,
            extra_sources=uploaded_sources,
            cancel_check=job.cancel_event.is_set,
        )
        metrics = result.metrics or {}
        note = _reduced_depth_warning(metrics)
        if note:
            try:
                if note not in (result.warnings or []):
                    result.warnings = list(result.warnings or []) + [note]
            except Exception:
                pass
        if job.cancel_event.is_set():
            raise RuntimeError("Job cancelled")
        jobs.update(job.job_id, stage="grounding", progress=75, message="Extracting evidence and grounding claims")
        store.save(result)
        _schedule_background_enrichment(result)
        job.run_id = result.run_id
        jobs.update(job.job_id, run_id=result.run_id, stage="synthesis", progress=90, message="Preparing report and exports")
        return result
    job = jobs.submit(execute, timeout_seconds=None, owner_id=_workspace_id())
    _bind_job(job.job_id, job.owner_id)
    return jobs.public(job.job_id)
@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict[str, Any]:
    _assert_job_owner(job_id)
    if not jobs.cancel(job_id):
        raise HTTPException(409, "Job cannot be cancelled")
    return jobs.public(job_id) or {"job_id": job_id, "status": "cancelled"}
@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    payload = jobs.public(job_id)
    if payload is None:
        raise HTTPException(404, "Job not found")
    job = jobs.get(job_id)
    if job and job.owner_id != _workspace_id():
        raise HTTPException(404, "Research job not found")
    if job and job.status == "completed" and job.result is not None:
        if isinstance(job.result, ResearchResult):
            _bind_run(job.result.run_id, job.owner_id)
            for version_item in store.versions(job.result, limit=100):
                _bind_run(version_item.run_id, job.owner_id)
            payload["result"] = _result_payload(job.result)
        elif isinstance(job.result, dict) and "candidates" in job.result:
            try:
                restored = {
                    "plan": ResearchPlan.model_validate(job.result["plan"]),
                    "candidates": [Source.model_validate(item) for item in job.result.get("candidates", [])],
                    "events": [RunEvent.model_validate(item) for item in job.result.get("events", [])],
                    "usage": job.result.get("usage", {}),
                }
                payload["result"] = _candidate_payload(restored)
            except Exception:
                payload["result"] = job.result
        elif isinstance(job.result, dict) and "run_id" in job.result:
            try:
                restored = ResearchResult.model_validate(job.result)
                _bind_run(restored.run_id, job.owner_id)
                for version_item in store.versions(restored, limit=100):
                    _bind_run(version_item.run_id, job.owner_id)
                payload["result"] = _result_payload(restored)
            except Exception:
                payload["result"] = job.result
    return payload
@app.get("/api/jobs/{job_id}/events")
def job_events(job_id: str) -> StreamingResponse:
    _assert_job_owner(job_id)
    if jobs.get(job_id) is None:
        raise HTTPException(404, "Job not found")
    def stream():
        last = 0
        for _ in range(900):
            current = jobs.public(job_id) or {}
            events = current.get("events", [])
            for event in events[last:]:
                yield f"data: {json.dumps(_jsonable(event), ensure_ascii=True)}\n\n"
            last = len(events)
            if current.get("status") in {"completed", "failed", "cancelled", "timed_out"}:
                yield f"event: done\ndata: {json.dumps(_jsonable(current), ensure_ascii=True)}\n\n"
                return
            time.sleep(1)
    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
@app.get("/api/research/{run_id}")
def get_research(run_id: str) -> dict[str, Any]:
    return _result_payload(_require_result(run_id))
@app.get("/api/research/{run_id}/versions")
def versions(run_id: str) -> list[dict[str, Any]]:
    result = _require_result(run_id)
    return [
        {
            "run_id": item.run_id,
            "version": item.version,
            "parent_run_id": item.parent_run_id,
            "build_mode": item.build_mode,
            "created_at": item.metrics.get("created_at", ""),
            "report_title": item.report_title,
            "case_name": item.case_name,
            "report_type": item.report_type,
            "sources_selected": item.metrics.get("sources_selected", len(item.sources)),
            "papers_deep_read": item.metrics.get("papers_deep_read", len(item.paper_analyses)),
        }
        for item in store.versions(result, limit=50)
    ]
@app.patch("/api/research/{run_id}/case")
def update_case(run_id: str, payload: CaseUpdateRequest) -> dict[str, Any]:
    result = _require_result(run_id)
    if payload.name is not None:
        updated = store.rename_case(result.run_id, payload.name)
        if updated is None:
            raise HTTPException(404, "Research case not found")
    if payload.pinned is not None:
        updated = store.set_pinned(result.run_id, payload.pinned)
        if updated is None:
            raise HTTPException(404, "Research case not found")
    latest = store.load(result.run_id)
    return _result_payload(latest or result)
@app.delete("/api/research/{run_id}/case")
def delete_case(run_id: str) -> dict[str, str]:
    _require_result(run_id)
    store.delete_case(run_id)
    return {"status": "deleted", "run_id": run_id}
@app.post("/api/research/{run_id}/discover")
def chat_discover(run_id: str, payload: DiscoveryChatRequest) -> dict[str, Any]:
    """Discover up to three additional real arXiv papers for the current report."""
    result = _require_result(run_id)
    try:
        found, events = agent.aggregator.run([payload.question.strip(), f"{result.question} {payload.question.strip()}"])
        existing={s.id for s in result.sources}
        candidates=[s for s in found if s.id not in existing]
        enrich_with_full_text(candidates[:20], max_fetch=min(20,len(candidates)))
        from researchpilot.pipeline.v79_retrieval import run_v79_retrieval
        v79=run_v79_retrieval(payload.question.strip(), result.report_type, candidates[:20], agent.llm, max_papers=3)
        cards=[]
        for source in v79.selected[:3]:
            cards.append({**_source_summary(source), "reason": " ".join(source.relevance_reasons[:2]), "adds_to_corpus": [c.get("facet") for c in v79.coverage if source.id in c.get("source_ids", [])]})
        return {"candidates":cards,"message": "No additional paper cleared the relevance bar." if not cards else "Found additional arXiv evidence."}
    except Exception:
        logger.exception("Paper discovery failed for %s", run_id)
        raise HTTPException(500,"Paper discovery failed. Check the server logs for details.") from None
@app.post("/api/research/{run_id}/rebuild")
def rebuild_report(run_id: str, payload: RebuildRequest) -> dict[str, Any]:
    """Queue an immutable New/Combine rebuild as a long-running research job."""
    result = _require_result(run_id)
    wanted = set(payload.arxiv_ids)
    selected = [s for s in result.sources if s.id in wanted]
    missing = wanted - {s.id for s in selected}
    if missing:
        found, _ = agent.aggregator.run(["id:" + x for x in sorted(missing)])
        enrich_with_full_text(found, max_fetch=len(found))
        selected.extend([s for s in found if s.id in missing])
    selected = [s for s in selected if s.full_text_complete]
    if not selected:
        raise HTTPException(409, "None of the selected arXiv papers has verified readable full text.")
    if payload.add_supporting:
        seed_terms = " ".join(s.title for s in selected)[:700]
        supporting, _ = agent.aggregator.run([result.question, seed_terms])
        supporting = [s for s in supporting if s.id not in wanted and s.id not in {x.id for x in result.sources}]
        enrich_with_full_text(supporting[:20], max_fetch=min(20, len(supporting)))
        from researchpilot.pipeline.v79_retrieval import run_v79_retrieval
        expanded = run_v79_retrieval(
            result.question, result.report_type, supporting[:20], agent.llm, max_papers=3
        ).selected
        selected.extend(expanded)
    if payload.mode == "new":
        corpus = list(dict((s.dedupe_key(), s) for s in selected).values())
        for source in corpus:
            source.citation_tag = ""
    else:
        corpus = list(dict((s.dedupe_key(), s) for s in result.sources + selected).values())
    corpus = corpus[:14]
    if not corpus:
        raise HTTPException(409, "The rebuild corpus is empty.")
    def execute(job):
        if job.cancel_event.is_set():
            raise RuntimeError("Job cancelled")
        jobs.update(job.job_id, stage="reading", progress=20, message="Deep-reading the rebuilt evidence corpus")
        updated = agent.run(
            result.question,
            max_sources=max(10, len(corpus)),
            report_type=result.report_type,
            report_title=result.report_title,
            version=result.version + 1,
            parent_run_id=result.run_id,
            build_mode=payload.mode,
            preselected_sources=corpus,
            extra_sources=[],
        )
        if job.cancel_event.is_set():
            raise RuntimeError("Job cancelled")
        store.save(updated)
        jobs.update(job.job_id, run_id=updated.run_id, stage="synthesis", progress=90, message="Preparing the rebuilt report")
        return updated
    job = jobs.submit(execute, owner_id=_workspace_id())
    _bind_job(job.job_id, job.owner_id)
    return jobs.public(job.job_id) or {"job_id": job.job_id, "status": "queued"}
@app.post("/api/research/{run_id}/chat")
def chat(run_id: str, payload: ChatRequest) -> dict[str, Any]:
    if run_id != payload.run_id:
        raise HTTPException(400, "run_id mismatch")
    result = _require_result(run_id)
    request_id = (payload.request_id or str(uuid4())).strip()
    cancel_event = threading.Event()
    with _chat_cancel_lock:
        if request_id in _chat_cancel_events:
            raise HTTPException(409, "A chat request with this request_id is already running.")
        _chat_cancel_events[request_id] = cancel_event
    try:
        if cancel_event.is_set():
            raise RuntimeError("Chat cancelled")
        if str(payload.mode).lower() == "verify":
            try:
                verification = agent.verify_claim(payload.question, result=result, allow_new_search=False, cancel_check=cancel_event.is_set)
            except TypeError:
                verification = agent.verify_claim(payload.question, result=result, allow_new_search=False)
            source_list = list(result.sources + [s for s in (verification.new_sources or []) if s.id not in {x.id for x in result.sources}])
            source_map = {s.id: s for s in source_list}
            tag_map = {source.id: f"S{idx}" for idx, source in enumerate(source_list, start=1)}
            evidence_map = {e.id: e for e in list(result.evidence) + list(verification.new_evidence or [])}
            refs = []
            for eid in verification.evidence_ids[:12]:
                ev = evidence_map.get(eid)
                if not ev:
                    continue
                src = source_map.get(ev.source_id)
                refs.append({
                    "tag": tag_map.get(ev.source_id, ""),
                    "title": src.title if src else "Research evidence",
                    "kind_label": "Verification evidence",
                    "location": ev.locator or "Paper evidence",
                    "excerpt": ev.text[:900],
                })
            store.save(result)
            return {
                "answer": verification.reasoning,
                "reasoning": verification.reasoning,
                "verdict": verification.verdict,
                "evidence_ids": verification.evidence_ids,
                "source_ids": [e.source_id for e in (verification.new_evidence or []) if e.id in verification.evidence_ids],
                "references": refs,
                "llm_used": verification.llm_used,
                "mode": "verify",
            }
        try:
            answer = agent.ask_followup(result, payload.question, depth=payload.depth, cancel_check=cancel_event.is_set)
        except TypeError:
            try:
                answer = agent.ask_followup(result, payload.question, depth=payload.depth)
            except TypeError:
                answer = agent.ask_followup(result, payload.question)
        if cancel_event.is_set():
            raise RuntimeError("Chat cancelled")
        store.save(result)
        return _jsonable(answer)
    except RuntimeError as exc:
        if "cancel" in str(exc).lower() or cancel_event.is_set():
            raise HTTPException(499, "Chat cancelled") from None
        logger.exception("Chat failed for research case %s", run_id)
        raise HTTPException(500, "Chat failed. Check the server logs for details.") from None
    except Exception:
        logger.exception("Chat failed for research case %s", run_id)
        raise HTTPException(500, "Chat failed. Check the server logs for details.") from None
    finally:
        with _chat_cancel_lock:
            _chat_cancel_events.pop(request_id, None)
@app.post("/api/research/{run_id}/chat/cancel")
def cancel_chat(run_id: str, request_id: str) -> dict[str, Any]:
    _require_result(run_id)
    with _chat_cancel_lock:
        event = _chat_cancel_events.get(request_id)
        if event is None:
            return {"request_id": request_id, "status": "not_found"}
        event.set()
    return {"request_id": request_id, "status": "cancel_requested"}
@app.post("/api/research/{run_id}/explain-selection")
def explain_selection(run_id: str, payload: SelectedTextRequest) -> dict[str, Any]:
    if run_id != payload.run_id:
        raise HTTPException(400, "run_id mismatch")
    result = _require_result(run_id)
    prompt = f"{payload.question}\n\nSelected report passage:\n{payload.selected_text}"
    try:
        answer = agent.ask_followup(result, prompt)
        store.save(result)
        return _jsonable(answer)
    except Exception:
        logger.exception("Selection explanation failed for research case %s", run_id)
        raise HTTPException(500, "Selection explanation failed. Check the server logs for details.") from None
async def _read_upload(upload: UploadFile) -> tuple[Source, int]:
    filename = Path(upload.filename or "paper.pdf").name
    suffix = Path(filename).suffix.lower()
    if suffix != ".pdf":
        raise HTTPException(415, f"{filename}: only PDF uploads are supported")
    body = await upload.read(MAX_UPLOAD_BYTES + 1)
    if len(body) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"{filename}: uploaded file exceeds {MAX_UPLOAD_BYTES} bytes")
    if not body.startswith(b"%PDF"):
        raise HTTPException(415, f"{filename}: file does not appear to be a PDF")
    try:
        source = source_from_uploaded_pdf(filename, body)
        source.role = "no_read"
        return source, len(body)
    except Exception as exc:
        raise HTTPException(400, f"Could not read {filename}: {exc}") from exc
@app.post("/api/research/{run_id}/upload")
async def upload_paper(run_id: str, file: UploadFile = File(...)) -> dict[str, Any]:
    """Compatibility endpoint: parse one PDF without changing the immutable report."""
    _require_result(run_id)
    source, _size = await _read_upload(file)
    return _source_summary(source)
@app.post("/api/research/{run_id}/reanalyze/job")
async def reanalyze_job(run_id: str, files: list[UploadFile] = File(...)) -> dict[str, Any]:
    """Queue post-report paper additions as a real, pollable research job."""
    result = _require_result(run_id)
    if not files:
        raise HTTPException(400, "Upload at least one PDF")
    if len(files) > 2:
        raise HTTPException(413, "Add at most 2 papers after the report is generated")
    existing_deep = int(result.metrics.get("papers_deep_read", len(result.paper_analyses)))
    if existing_deep + len(files) > 10:
        raise HTTPException(409, f"This research case already has {existing_deep} deep reads; at most {10 - existing_deep} more can be added")
    payload = await _prepare_upload_batch(files, max_files=2)
    def execute(job):
        jobs.update(job.job_id, stage="upload_reading", progress=15, message="Reading added papers")
        parsed: list[Source] = []
        seen = {s.dedupe_key() for s in result.sources}
        for index, (filename, body) in enumerate(payload, start=1):
            if job.cancel_event.is_set():
                raise RuntimeError("Job cancelled")
            source = source_from_uploaded_pdf(filename, body)
            source.role = "no_read"
            if source.dedupe_key() in seen:
                continue
            seen.add(source.dedupe_key())
            parsed.append(source)
            jobs.update(job.job_id, stage="upload_reading", progress=15 + int(25 * index / len(payload)), message=f"Read {index} of {len(payload)} added paper(s)")
        if not parsed:
            raise RuntimeError("All uploaded papers are already in this research case")
        jobs.update(job.job_id, stage="deep_analysis", progress=45, message=f"Deep analyzing {len(parsed)} added paper(s)")
        updated = agent.add_sources_and_reanalyze(result, parsed, deep_read_top_k=len(parsed))
        if job.cancel_event.is_set():
            raise RuntimeError("Job cancelled")
        store.save(updated)
        jobs.update(job.job_id, run_id=updated.run_id, stage="synthesis", progress=90, message="Updating the research report and evidence")
        return {**_result_payload(updated), "previous_run_id": result.run_id, "added_count": len(parsed)}
    job = jobs.submit(execute, owner_id=_workspace_id())
    _bind_job(job.job_id, job.owner_id)
    return jobs.public(job.job_id)
@app.post("/api/research/{run_id}/reanalyze")
async def reanalyze(run_id: str, files: list[UploadFile] = File(...)) -> dict[str, Any]:
    """Add one or more PDFs and create a new immutable report version."""
    result = _require_result(run_id)
    if not files:
        raise HTTPException(400, "Upload at least one PDF")
    if len(files) > 2:
        raise HTTPException(413, "Add at most 2 papers after the report is generated")
    existing_deep = int(result.metrics.get("papers_deep_read", len(result.paper_analyses)))
    if existing_deep + len(files) > 10:
        raise HTTPException(409, f"This research case already has {existing_deep} deep reads; at most {10 - existing_deep} more can be added")
    parsed: list[Source] = []
    seen = {s.dedupe_key() for s in result.sources}
    total_upload_bytes = 0
    for upload in files:
        source, size = await _read_upload(upload)
        total_upload_bytes += size
        if total_upload_bytes > MAX_TOTAL_UPLOAD_BYTES:
            raise HTTPException(413, f"Combined upload batch exceeds {MAX_TOTAL_UPLOAD_BYTES} bytes")
        if source.dedupe_key() in seen:
            continue
        seen.add(source.dedupe_key())
        parsed.append(source)
    if not parsed:
        raise HTTPException(409, "All uploaded papers are already in this research case")
    try:
        updated = agent.add_sources_and_reanalyze(result, parsed, deep_read_top_k=None)
        store.save(updated)
        _bind_run(updated.run_id, _workspace_id())
        payload = _result_payload(updated)
        payload["previous_run_id"] = result.run_id
        payload["added_count"] = len(parsed)
        return payload
    except Exception:
        logger.exception("Re-analysis failed for research case %s", run_id)
        raise HTTPException(500, "Re-analysis failed. Check the server logs for details.") from None
@app.get("/api/research/{run_id}/download/{kind}")
def download(run_id: str, kind: str) -> Response:
    result = _require_result(run_id)
    stem = safe_filename_stem(result.report_title, run_id)
    report_markdown = _sanitize_text(result.report_markdown)
    if kind == "bib":
        body = sources_to_bibtex(result.sources)
        return Response(body, media_type="application/x-bibtex", headers={"Content-Disposition": content_disposition_header(stem, "bib"), "Cache-Control": "no-store"})
    if kind == "pdf":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            out_path = Path(tmp.name)
        try:
            path = markdown_to_pdf(report_markdown, out_path)
            body = path.read_bytes()
            if not body.startswith(b"%PDF"):
                raise RuntimeError("PDF exporter returned an invalid file")
        except Exception as exc:
            raise HTTPException(500, f"PDF export failed: {exc}") from exc
        finally:
            out_path.unlink(missing_ok=True)
        return Response(body, media_type="application/pdf", headers={"Content-Disposition": content_disposition_header(stem, "pdf"), "Cache-Control": "no-store"})
    if kind == "latex":
        try:
            body = markdown_to_latex(report_markdown)
            if not body.strip().endswith(r"\end{document}"):
                raise RuntimeError("LaTeX exporter returned an incomplete document")
        except Exception as exc:
            raise HTTPException(500, f"LaTeX export failed: {exc}") from exc
        return Response(body, media_type="application/x-tex", headers={"Content-Disposition": content_disposition_header(stem, "tex"), "Cache-Control": "no-store"})
    if kind == "docx":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
            out_path = Path(tmp.name)
        try:
            path = markdown_to_docx(report_markdown, out_path)
            body = path.read_bytes()
            if len(body) < 1000:
                raise RuntimeError("DOCX exporter returned an empty or invalid file")
        except Exception as exc:
            raise HTTPException(500, f"DOCX export failed: {exc}") from exc
        finally:
            out_path.unlink(missing_ok=True)
        return Response(body, media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document", headers={"Content-Disposition": content_disposition_header(stem, "docx"), "Cache-Control": "no-store"})
    raise HTTPException(404, "Unknown download format")
