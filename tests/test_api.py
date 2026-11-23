from __future__ import annotations
import time
from pathlib import Path
from fastapi.testclient import TestClient
import api.server as server
from api.jobs import JobManager
from researchpilot.schemas import (FollowupQA, ResearchPlan, ResearchResult,
                                   Source)
from researchpilot.storage import ResultStore
def _source(i: int, role: str = "core") -> Source:
    return Source(id=f"s{i}", title=f"Paper {i}", abstract=f"Abstract {i}", authors="Author", year="2026", role=role, provider="test", score=1.0 - i / 100)
def _result(run_id: str = "r1") -> ResearchResult:
    plan = ResearchPlan(question="test question", mode="literature_review", report_type="literature_review", report_title="Test report", search_queries=["test"])
    return ResearchResult(run_id=run_id, question="test question", report_title="Test report", case_name="Test report", plan=plan, sources=[_source(1)], report_markdown="# Test report\n\nEvidence-backed result.", metrics={"sources_selected": 1, "papers_deep_read": 1})
class FakeAgent:
    def __init__(self):
        self.base = _result()
    def discover_candidates(self, question, report_type=None, report_title="", candidate_limit=18):
        candidates = [_source(i) for i in range(1, 7)]
        plan = ResearchPlan(question=question, mode="literature_review", report_type=report_type or "literature_review", report_title=report_title or "Test report", search_queries=["test"])
        return plan, candidates, [], False, {"llm_calls": 0}
    def run(self, **kwargs):
        selected = kwargs.get("preselected_sources")
        if selected is None:
            automatic = kwargs.get("automatic_candidates") or []
            selected = automatic or [self.base.sources[0]]
        return self.base.model_copy(update={"run_id": "run-final", "sources": selected, "metrics": {"sources_selected": len(selected)}})
    def ask_followup(self, result, question):
        return FollowupQA(question=question, answer="Grounded answer", evidence_ids=[], source_ids=[s.id for s in result.sources])
    def verify_claim(self, claim, result=None, allow_new_search=True):
        from researchpilot.schemas import ClaimVerification
        return ClaimVerification(claim=claim, verdict="well_supported", reasoning="Verified against case evidence", evidence_ids=[], new_sources=[])
    def add_sources_and_reanalyze(self, result, new_sources, deep_read_top_k=None):
        return result.model_copy(update={
            "run_id": "run-v2",
            "version": result.version + 1,
            "parent_run_id": result.run_id,
            "sources": result.sources + new_sources,
            "added_source_ids": [s.id for s in new_sources],
            "report_markdown": result.report_markdown + "\n\nAdded analysis.",
        })
def wait_for(client, job_id):
    for _ in range(100):
        data = client.get(f"/api/jobs/{job_id}").json()
        if data["status"] in {"completed", "failed", "cancelled"}:
            return data
        time.sleep(0.01)
    raise AssertionError("job did not finish")
def test_react_api_workflow(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "agent", FakeAgent())
    monkeypatch.setattr(server, "store", ResultStore(str(tmp_path / "research.db")))
    monkeypatch.setattr(server, "jobs", JobManager(workers=1, db_path=str(tmp_path / "jobs.db")))
    monkeypatch.setattr(server, "source_from_uploaded_pdf", lambda filename, body: _source(99, "uploaded").model_copy(update={"title": filename}))
    with TestClient(server.app, headers={"X-Workspace-ID": "test-workspace-000000000000000000000000000000000000"}) as client:
        discovery = client.post("/api/research/discover", json={"question": "test question", "title": "Test report", "max_papers": 2})
        assert discovery.status_code == 200
        d = wait_for(client, discovery.json()["job_id"])
        assert d["status"] == "completed"
        assert len(d["result"]["candidates"]) == 6
        run = client.post("/api/research/run", json={
            "question": "test question", "title": "Test report", "max_papers": 2,
            "discovery_job_id": discovery.json()["job_id"], "selected_source_ids": ["s2", "s4"],
        })
        assert run.status_code == 200
        r = wait_for(client, run.json()["job_id"])
        assert r["status"] == "completed"
        assert [s["id"] for s in r["result"]["sources"]] == ["s2", "s4"]
        assert all(s["role"] == "deep_read" for s in r["result"]["sources"])
        loaded = client.get("/api/research/run-final")
        assert loaded.status_code == 200
        chat = client.post("/api/research/run-final/chat", json={"run_id": "run-final", "question": "What does it say?", "mode": "Ask"})
        assert chat.status_code == 200
        assert chat.json()["answer"] == "Grounded answer"
        verify = client.post("/api/research/run-final/chat", json={"run_id": "run-final", "question": "Claim X", "mode": "Verify"})
        assert verify.status_code == 200
        assert verify.json()["answer"] == "Verified against case evidence"
        assert verify.json()["verdict"] == "well_supported"
        re = client.post("/api/research/run-final/reanalyze", files={"files": ("new.pdf", b"%PDF-fake", "application/pdf")})
        assert re.status_code == 200
        assert re.json()["run_id"] == "run-v2"
        assert re.json()["version"] == 2
        assert re.json()["added_count"] == 1
        history = client.get("/api/history")
        assert history.status_code == 200
        assert history.json()[0]["version_count"] == 2
        patch = client.patch("/api/research/run-v2/case", json={"name": "Renamed case", "pinned": True})
        assert patch.status_code == 200
        assert patch.json()["case_name"] == "Renamed case"
        versions = client.get("/api/research/run-v2/versions")
        assert versions.status_code == 200
        assert {v["version"] for v in versions.json()} == {1, 2}
def test_api_uses_autonomous_corpus_and_matches_discovery_context(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "agent", FakeAgent())
    monkeypatch.setattr(server, "store", ResultStore(str(tmp_path / "research.db")))
    monkeypatch.setattr(server, "jobs", JobManager(workers=1, db_path=str(tmp_path / "jobs.db")))
    with TestClient(server.app, headers={"X-Workspace-ID": "test-workspace-000000000000000000000000000000000000"}) as client:
        discovery = client.post("/api/research/discover", json={"question": "test question", "title": "Test report", "report_type": "literature_review", "max_papers": 2})
        assert discovery.status_code == 200
        d = wait_for(client, discovery.json()["job_id"])
        assert d["status"] == "completed"
        automatic = client.post("/api/research/run", json={
            "question": "test question", "title": "Test report", "report_type": "literature_review", "max_papers": 2,
            "discovery_job_id": discovery.json()["job_id"], "selected_source_ids": []
        })
        assert automatic.status_code == 200
        mismatch = client.post("/api/research/run", json={
            "question": "different question", "title": "Test report", "report_type": "literature_review", "max_papers": 2,
            "discovery_job_id": discovery.json()["job_id"], "selected_source_ids": ["s1"]
        })
        assert mismatch.status_code == 409
def test_api_preserves_automatic_corpus_when_selection_was_not_overridden(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "agent", FakeAgent())
    monkeypatch.setattr(server, "store", ResultStore(str(tmp_path / "research.db")))
    monkeypatch.setattr(server, "jobs", JobManager(workers=1, db_path=str(tmp_path / "jobs.db")))
    with TestClient(server.app, headers={"X-Workspace-ID": "test-workspace-000000000000000000000000000000000000"}) as client:
        discovery = client.post("/api/research/discover", json={"question": "test question", "title": "Test report", "report_type": "literature_review"})
        assert discovery.status_code == 200
        d = wait_for(client, discovery.json()["job_id"])
        assert d["status"] == "completed"
        run = client.post("/api/research/run", json={
            "question": "test question", "title": "Test report", "report_type": "literature_review",
            "discovery_job_id": discovery.json()["job_id"],
            "selected_source_ids": ["s2"],
            "selection_overridden": False,
        })
        assert run.status_code == 200
        r = wait_for(client, run.json()["job_id"])
        assert r["status"] == "completed"
        assert len(r["result"]["sources"]) == 6
def test_completed_job_result_survives_job_manager_restart(tmp_path):
    from api.jobs import JobManager
    from researchpilot.schemas import ResearchPlan, ResearchResult
    db = tmp_path / "jobs.db"
    first = JobManager(workers=1, db_path=str(db))
    job = first.submit(lambda j: ResearchResult(
        run_id="persisted-run", question="q",
        plan=ResearchPlan(question="q", mode="literature_review"), report_markdown="# R"
    ))
    for _ in range(100):
        current = first.get(job.job_id)
        if current and current.status == "completed":
            break
        import time; time.sleep(0.01)
    assert first.get(job.job_id).status == "completed"
    second = JobManager(workers=1, db_path=str(db))
    restored = second.get(job.job_id)
    assert restored is not None
    assert restored.status == "completed"
    assert isinstance(restored.result, dict)
    assert restored.result["run_id"] == "persisted-run"
def test_api_automatic_false_override_passes_snapshot_as_automatic_candidates(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "agent", FakeAgent())
    monkeypatch.setattr(server, "store", ResultStore(str(tmp_path / "research.db")))
    monkeypatch.setattr(server, "jobs", JobManager(workers=1, db_path=str(tmp_path / "jobs.db")))
    with TestClient(server.app, headers={"X-Workspace-ID": "test-workspace-000000000000000000000000000000000000"}) as client:
        discovery = client.post("/api/research/discover", json={
            "question": "test question", "title": "Test report", "report_type": "literature_review"
        })
        assert discovery.status_code == 200
        d = wait_for(client, discovery.json()["job_id"])
        assert d["status"] == "completed"
        run = client.post("/api/research/run", json={
            "question": "test question", "title": "Test report", "report_type": "literature_review",
            "discovery_job_id": discovery.json()["job_id"],
            "selected_source_ids": ["s6"],
            "selection_overridden": False,
        })
        assert run.status_code == 200
        r = wait_for(client, run.json()["job_id"])
        assert r["status"] == "completed"
        assert [s["id"] for s in r["result"]["sources"]] == ["s1", "s2", "s3", "s4", "s5", "s6"]
def test_late_cancellation_is_terminal_and_discards_late_worker_result(tmp_path):
    """Cancellation is terminal even if the worker was already running."""
    from threading import Event
    from api.jobs import JobManager
    db = tmp_path / "jobs.db"
    manager = JobManager(workers=1, db_path=str(db))
    started = Event()
    release = Event()
    def work(job):
        started.set()
        assert release.wait(2)
        return {"run_id": "finished-report", "report_markdown": "# Completed"}
    job = manager.submit(work)
    assert started.wait(2)
    assert manager.cancel(job.job_id) is True
    release.set()
    for _ in range(200):
        current = manager.get(job.job_id)
        if current and current.status in {"completed", "failed", "cancelled"}:
            break
        time.sleep(0.01)
    current = manager.get(job.job_id)
    assert current is not None
    assert current.status == "cancelled"
    assert current.result is None
def test_job_cancel_is_immediately_terminal_and_late_worker_result_is_discarded(tmp_path):
    import threading
    import time
    from api.jobs import JobManager
    manager = JobManager(workers=1, db_path=str(tmp_path / "jobs.db"))
    started = threading.Event()
    release = threading.Event()
    def slow(job):
        started.set()
        release.wait(2)
        return {"should": "be discarded"}
    job = manager.submit(slow, timeout_seconds=10)
    assert started.wait(1)
    assert manager.cancel(job.job_id) is True
    status = manager.public(job.job_id)
    assert status["status"] == "cancelled"
    assert status["finished_at"]
    release.set()
    time.sleep(0.1)
    status = manager.public(job.job_id)
    assert status["status"] == "cancelled"
    assert manager.get(job.job_id).result is None
def test_api_rejects_stale_explicit_selection_instead_of_silently_dropping_papers(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "agent", FakeAgent())
    monkeypatch.setattr(server, "store", ResultStore(str(tmp_path / "research.db")))
    monkeypatch.setattr(server, "jobs", JobManager(workers=1, db_path=str(tmp_path / "jobs.db")))
    with TestClient(server.app, headers={"X-Workspace-ID": "test-workspace-000000000000000000000000000000000000"}) as client:
        discovery = client.post("/api/research/discover", json={"question": "test question", "title": "Test report"})
        assert discovery.status_code == 200
        done = wait_for(client, discovery.json()["job_id"])
        assert done["status"] == "completed"
        response = client.post("/api/research/run", json={
            "question": "test question", "title": "Test report",
            "discovery_job_id": discovery.json()["job_id"],
            "selected_source_ids": ["s1", "stale-paper-id", "s3"],
            "selection_overridden": True,
        })
        assert response.status_code == 422
        assert "stale-paper-id" in response.json()["detail"]
def test_api_rejects_empty_explicit_selection_instead_of_falling_back_to_defaults(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "agent", FakeAgent())
    monkeypatch.setattr(server, "store", ResultStore(str(tmp_path / "research.db")))
    monkeypatch.setattr(server, "jobs", JobManager(workers=1, db_path=str(tmp_path / "jobs.db")))
    with TestClient(server.app, headers={"X-Workspace-ID": "test-workspace-000000000000000000000000000000000000"}) as client:
        discovery = client.post("/api/research/discover", json={"question": "test question", "title": "Test report"})
        assert discovery.status_code == 200
        done = wait_for(client, discovery.json()["job_id"])
        assert done["status"] == "completed"
        response = client.post("/api/research/run", json={
            "question": "test question", "title": "Test report",
            "discovery_job_id": discovery.json()["job_id"],
            "selected_source_ids": [], "selection_overridden": True,
        })
        assert response.status_code == 422
        assert "selection is empty" in response.json()["detail"]
def test_workspace_isolation_blocks_cross_visitor_report_and_job(tmp_path, monkeypatch):
    from uuid import uuid4
    owner_a = "workspace-a-" + uuid4().hex
    owner_b = "workspace-b-" + uuid4().hex
    run_id = "private-run-" + uuid4().hex
    monkeypatch.setattr(server, "store", ResultStore(str(tmp_path / "research.db")))
    monkeypatch.setattr(server, "jobs", JobManager(workers=1, db_path=str(tmp_path / "jobs.db")))
    result = _result(run_id)
    server.store.save(result)
    server._bind_run(run_id, owner_a)
    job = server.jobs.submit(lambda j: {"ok": True}, owner_id=owner_a)
    server._bind_job(job.job_id, owner_a)
    with TestClient(server.app, headers={"X-Workspace-ID": owner_a}) as client_a, TestClient(server.app, headers={"X-Workspace-ID": owner_b}) as client_b:
        assert client_a.get(f"/api/research/{run_id}").status_code == 200
        assert client_b.get(f"/api/research/{run_id}").status_code == 404
        assert client_a.get(f"/api/jobs/{job.job_id}").status_code == 200
        assert client_b.get(f"/api/jobs/{job.job_id}").status_code == 404
        assert all(item["run_id"] != run_id for item in client_b.get("/api/history").json())
def test_production_cors_allows_workspace_header():
    with TestClient(server.app) as client:
        response = client.options(
            "/api/research/discover",
            headers={
                "Origin": "https://researchorbit.vercel.app",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type,x-workspace-id",
            },
        )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "https://researchorbit.vercel.app"
    assert "x-workspace-id" in response.headers.get("access-control-allow-headers", "").lower()
