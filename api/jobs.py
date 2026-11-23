from __future__ import annotations
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Event, Lock, Timer
from typing import Any, Callable
from uuid import uuid4
def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
def _safe_text(value: str) -> str:
    return "".join("\ufffd" if 0xD800 <= ord(ch) <= 0xDFFF else ch for ch in value)
def _safe_json(value: Any) -> Any:
    if isinstance(value, str): return _safe_text(value)
    if isinstance(value, list): return [_safe_json(v) for v in value]
    if isinstance(value, tuple): return [_safe_json(v) for v in value]
    if isinstance(value, dict): return {_safe_text(str(k)): _safe_json(v) for k, v in value.items()}
    return value
@dataclass
class Job:
    job_id: str
    owner_id: str | None = None
    status: str = "queued"
    created_at: str = field(default_factory=utc_now)
    started_at: str | None = None
    finished_at: str | None = None
    progress: int = 0
    stage: str = "queued"
    message: str = "Queued"
    result: Any = None
    error: str | None = None
    run_id: str | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    cancel_event: Event = field(default_factory=Event, repr=False)
    timeout_seconds: float | None = field(default=None, repr=False)
class JobStore:
    def __init__(self, path: str = "data/jobs.db"):
        self.path = path
        self._lock = Lock()
        Path = __import__("pathlib").Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as db:
            db.execute("CREATE TABLE IF NOT EXISTS jobs (job_id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
    def load_all(self) -> list[dict[str, Any]]:
        with self._lock, sqlite3.connect(self.path) as db:
            rows = db.execute("SELECT payload FROM jobs").fetchall()
        return [json.loads(row[0]) for row in rows]
    def save(self, job: Job) -> None:
        def jsonable(value):
            if hasattr(value, "model_dump"):
                return value.model_dump(mode="json")
            if isinstance(value, list):
                return [jsonable(item) for item in value]
            if isinstance(value, tuple):
                return [jsonable(item) for item in value]
            if isinstance(value, dict):
                return {key: jsonable(item) for key, item in value.items()}
            return value
        result = jsonable(job.result)
        result_kind = "research" if hasattr(job.result, "run_id") else ("discovery" if isinstance(job.result, dict) and "candidates" in job.result else "generic")
        payload = {"job_id": job.job_id, "owner_id": job.owner_id, "status": job.status, "created_at": job.created_at,
                   "started_at": job.started_at, "finished_at": job.finished_at, "progress": job.progress,
                   "stage": job.stage, "message": job.message, "error": job.error, "run_id": job.run_id,
                   "result_kind": result_kind, "result": result, "events": job.events[-100:]}
        payload = _safe_json(payload)
        with self._lock, sqlite3.connect(self.path) as db:
            db.execute("INSERT OR REPLACE INTO jobs(job_id, payload) VALUES (?, ?)", (job.job_id, json.dumps(payload, ensure_ascii=True)))
class JobManager:
    def __init__(self, workers: int = 2, db_path: str = "data/jobs.db"):
        self._jobs: dict[str, Job] = {}
        self._lock = Lock()
        self._store = JobStore(db_path)
        self._restore()
        self._executor = ThreadPoolExecutor(max_workers=max(1, workers), thread_name_prefix="researchpilot")
    def _restore(self) -> None:
        for payload in self._store.load_all():
            status = payload.get("status", "failed")
            if status in {"queued", "running"}:
                status = "failed"
                payload["error"] = "Process restarted before job completion"
                payload["message"] = "Job interrupted by process restart"
            job = Job(job_id=payload["job_id"], owner_id=payload.get("owner_id"), status=status, created_at=payload.get("created_at", utc_now()),
                      started_at=payload.get("started_at"), finished_at=payload.get("finished_at"),
                      progress=payload.get("progress", 0), stage=payload.get("stage", status),
                      message=payload.get("message", "Restored"), error=payload.get("error"),
                      run_id=payload.get("run_id"), result=payload.get("result"), events=payload.get("events", []))
            self._jobs[job.job_id] = job
            if status == "failed": self._store.save(job)
    def submit(self, fn: Callable[[Job], Any], *, timeout_seconds: float | None = None, owner_id: str | None = None) -> Job:
        job = Job(job_id=str(uuid4()), owner_id=owner_id, timeout_seconds=timeout_seconds)
        with self._lock:
            self._jobs[job.job_id] = job
            self._record(job, status="queued", stage="queued", progress=0, message="Queued")
        self._executor.submit(self._run, job.job_id, fn)
        if timeout_seconds is not None and timeout_seconds > 0:
            timer = Timer(float(timeout_seconds), self._timeout_job, args=(job.job_id,))
            timer.daemon = True
            timer.start()
        return job
    def _timeout_job(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job or job.status in {"completed", "failed", "cancelled", "timed_out"}:
                return
            job.cancel_event.set()
            self._record(job, status="timed_out", finished_at=utc_now(), stage="timed_out", progress=100,
                         message="Job exceeded its maximum runtime; the result was discarded.",
                         error="Job runtime limit exceeded")
    def _run(self, job_id: str, fn: Callable[[Job], Any]) -> None:
        job = self.get(job_id)
        if not job: return
        self.update(job_id, status="running", started_at=utc_now(), stage="planning", progress=5, message="Planning research strategy")
        try:
            if job.cancel_event.is_set(): raise RuntimeError("Job cancelled")
            result = fn(job)
            with self._lock:
                current = self._jobs.get(job_id)
                if not current or current.status in {"cancelled", "timed_out"} or current.cancel_event.is_set():
                    if current:
                        current.result = None
                        self._store.save(current)
                    return
                job.result = result
                self._record(job, status="completed", finished_at=utc_now(), stage="complete",
                             progress=100, message="Research completed")
        except Exception as exc:
            with self._lock:
                current = self._jobs.get(job_id)
                terminal_cancel = bool(current and current.status in {"cancelled", "timed_out"})
                if terminal_cancel and current:
                    current.result = None
                    self._store.save(current)
            if terminal_cancel:
                return
            status = "cancelled" if (current and current.cancel_event.is_set()) or str(exc) == "Job cancelled" else "failed"
            self.update(job_id, status=status, finished_at=utc_now(), stage=status, progress=100, message=f"Research {status}", error=None if status == "cancelled" else str(exc))
    def _record(self, job: Job, **changes: Any) -> None:
        event = {"timestamp": utc_now(), **changes}
        job.events.append(event)
        job.events = job.events[-100:]
        for key, value in changes.items():
            if hasattr(job, key): setattr(job, key, value)
        self._store.save(job)
    def update(self, job_id: str, **changes: Any) -> Job | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job: self._record(job, **changes)
            return job
    def cancel(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job or job.status in {"completed", "failed", "cancelled", "timed_out"}:
                return False
            job.cancel_event.set()
            self._record(job, status="cancelled", finished_at=utc_now(), stage="cancelled", progress=100,
                         message="Research cancelled", error=None)
            return True
    def get(self, job_id: str) -> Job | None:
        with self._lock: return self._jobs.get(job_id)
    def public(self, job_id: str) -> dict[str, Any] | None:
        job = self.get(job_id)
        if not job: return None
        return {"job_id": job.job_id, "owner_id": job.owner_id, "status": job.status, "created_at": job.created_at, "started_at": job.started_at,
                "finished_at": job.finished_at, "progress": job.progress, "stage": job.stage, "message": job.message,
                "run_id": job.run_id, "error": job.error, "events": list(job.events[-50:])}
