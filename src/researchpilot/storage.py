from __future__ import annotations
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from .schemas import ResearchResult
def _sanitize_json_value(value):
    """Return a JSON-safe structure with isolated UTF-16 surrogates removed."""
    if isinstance(value, str):
        return "".join("\ufffd" if 0xD800 <= ord(ch) <= 0xDFFF else ch for ch in value)
    if isinstance(value, dict):
        return {_sanitize_json_value(k): _sanitize_json_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_json_value(v) for v in value]
    if isinstance(value, tuple):
        return [_sanitize_json_value(v) for v in value]
    return value
_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    question TEXT NOT NULL,
    created_at TEXT NOT NULL,
    metrics_json TEXT NOT NULL,
    result_json TEXT NOT NULL,
    parent_run_id TEXT NOT NULL DEFAULT '',
    version INTEGER NOT NULL DEFAULT 1,
    report_title TEXT NOT NULL DEFAULT '',
    report_type TEXT NOT NULL DEFAULT 'literature_review',
    case_name TEXT NOT NULL DEFAULT '',
    pinned INTEGER NOT NULL DEFAULT 0
);
"""
class ResultStore:
    def __init__(self, db_path: str = "data/researchpilot.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(_SCHEMA)
            self._migrate(conn)
    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)
    @staticmethod
    def _migrate(conn: sqlite3.Connection) -> None:
        existing = {row[1] for row in conn.execute("PRAGMA table_info(runs)").fetchall()}
        additions = {
            "parent_run_id": "TEXT NOT NULL DEFAULT ''",
            "version": "INTEGER NOT NULL DEFAULT 1",
            "report_title": "TEXT NOT NULL DEFAULT ''",
            "report_type": "TEXT NOT NULL DEFAULT 'literature_review'",
            "case_name": "TEXT NOT NULL DEFAULT ''",
            "pinned": "INTEGER NOT NULL DEFAULT 0",
        }
        for name, definition in additions.items():
            if name not in existing:
                conn.execute(f"ALTER TABLE runs ADD COLUMN {name} {definition}")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS paper_dossiers (
                cache_key TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                pipeline_version TEXT NOT NULL,
                created_at TEXT NOT NULL,
                payload_json TEXT NOT NULL
            )
        """)
    def save_dossier(self, cache_key: str, source_id: str, pipeline_version: str, payload: dict) -> None:
        now = datetime.now(timezone.utc).isoformat()
        body = json.dumps(_sanitize_json_value(payload), ensure_ascii=False)
        with self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO paper_dossiers
                   (cache_key, source_id, pipeline_version, created_at, payload_json)
                   VALUES (?, ?, ?, ?, ?)""",
                (cache_key, source_id, pipeline_version, now, body),
            )
    def load_dossier(self, cache_key: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload_json FROM paper_dossiers WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
        if not row:
            return None
        try:
            value = json.loads(row[0])
            return value if isinstance(value, dict) else None
        except (TypeError, json.JSONDecodeError):
            return None
    def save(self, result: ResearchResult) -> None:
        with self._connect() as conn:
            existing_row = conn.execute("SELECT created_at FROM runs WHERE run_id = ?", (result.run_id,)).fetchone()
            existing = existing_row is not None
            created_at = existing_row[0] if existing_row else datetime.now(timezone.utc).isoformat()
            if not existing and result.parent_run_id:
                rows = conn.execute("SELECT run_id, parent_run_id, version FROM runs").fetchall()
                parent = {r[0]: r[1] for r in rows}
                versions = {r[0]: int(r[2] or 1) for r in rows}
                root = self._root_for(result.parent_run_id, parent)
                lineage_max = max((versions[rid] for rid in versions if self._root_for(rid, parent) == root), default=0)
                if result.version <= lineage_max:
                    result.version = lineage_max + 1
            conn.execute(
                "INSERT OR REPLACE INTO runs "
                "(run_id, question, created_at, metrics_json, result_json, parent_run_id, version, report_title, report_type, case_name, pinned) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    result.run_id,
                    result.question,
                    created_at,
                    json.dumps(_sanitize_json_value(result.metrics), ensure_ascii=True),
                    json.dumps(_sanitize_json_value(result.model_dump(mode="json")), ensure_ascii=True),
                    result.parent_run_id,
                    result.version,
                    result.report_title,
                    result.report_type,
                    result.case_name or result.report_title or result.question[:80],
                    int(result.pinned),
                ),
            )
    def list_runs(self, limit: int = 25) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT run_id, question, created_at, metrics_json, parent_run_id, version, report_title, report_type, case_name, pinned "
                "FROM runs ORDER BY pinned DESC, created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {
                "run_id": r[0], "question": r[1], "created_at": r[2], "metrics": json.loads(r[3]),
                "parent_run_id": r[4], "version": r[5], "report_title": r[6], "report_type": r[7],
                "case_name": r[8], "pinned": bool(r[9]),
            }
            for r in rows
        ]
    def list_cases(self, limit: int = 30) -> list[dict]:
        """Return one sidebar entry per complete research-case lineage.
        Unlike the earlier implementation, this does not inspect only the
        newest N rows. That shortcut could split an old case if its parent
        version fell outside the metadata window. Only lightweight metadata is
        loaded here; the potentially large result JSON stays untouched.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT run_id, question, created_at, metrics_json, parent_run_id, version, report_title, report_type, case_name, pinned "
                "FROM runs ORDER BY created_at DESC"
            ).fetchall()
        items = [
            {
                "run_id": r[0], "question": r[1], "created_at": r[2], "metrics": json.loads(r[3]),
                "parent_run_id": r[4], "version": r[5], "report_title": r[6], "report_type": r[7],
                "case_name": r[8], "pinned": bool(r[9]),
            }
            for r in rows
        ]
        by_id = {x["run_id"]: x for x in items}
        parent = {k: v.get("parent_run_id", "") for k, v in by_id.items()}
        roots: dict[str, str] = {}
        for item in items:
            roots[item["run_id"]] = self._root_for(item["run_id"], parent)
        grouped: dict[str, list[dict]] = {}
        for item in items:
            grouped.setdefault(roots[item["run_id"]], []).append(item)
        latest = []
        for versions in grouped.values():
            current = max(versions, key=lambda x: (x.get("version", 1), x.get("created_at", "")))
            current = dict(current)
            current["version_count"] = len(versions)
            current["pinned"] = any(v.get("pinned") for v in versions)
            latest.append(current)
        latest.sort(key=lambda x: (x.get("pinned", False), x.get("created_at", "")), reverse=True)
        return latest[:limit]
    def delete_case(self, run_id: str) -> None:
        """Delete every stored version belonging to the same case lineage."""
        with self._connect() as conn:
            rows = conn.execute("SELECT run_id, parent_run_id FROM runs").fetchall()
            parent = {r[0]: r[1] for r in rows}
            all_ids = set(parent)
            root = run_id
            seen = set()
            while parent.get(root) and parent[root] in all_ids and root not in seen:
                seen.add(root)
                root = parent[root]
            to_delete = {rid for rid in all_ids if self._root_for(rid, parent) == root}
            if not to_delete:
                to_delete = {run_id}
            conn.executemany("DELETE FROM runs WHERE run_id = ?", [(rid,) for rid in to_delete])
    @staticmethod
    def _root_for(run_id: str, parent: dict[str, str]) -> str:
        current = run_id
        seen = set()
        while parent.get(current) and parent[current] in parent and current not in seen:
            seen.add(current)
            current = parent[current]
        return current
    def delete(self, run_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
    def same_case(self, run_id_a: str, run_id_b: str) -> bool:
        """Return True when two snapshots belong to the same research case lineage.
        The UI can have an older snapshot open while the sidebar represents the
        latest snapshot of that case. Case-level actions such as rename, pin and
        delete must therefore be lineage-aware rather than comparing run ids.
        """
        if not run_id_a or not run_id_b:
            return False
        with self._connect() as conn:
            rows = conn.execute("SELECT run_id, parent_run_id FROM runs").fetchall()
        parent = {r[0]: r[1] for r in rows}
        if run_id_a not in parent or run_id_b not in parent:
            return False
        return self._root_for(run_id_a, parent) == self._root_for(run_id_b, parent)
    def _lineage_ids(self, run_id: str) -> list[str]:
        """Return every version in the same case lineage, oldest-to-newest.
        Case metadata such as the display name and pin state is a property of
        the case, not of one immutable report version. Updating only the current
        row caused an older opened version to resurrect the old sidebar name.
        """
        with self._connect() as conn:
            rows = conn.execute("SELECT run_id, parent_run_id FROM runs").fetchall()
        parent = {r[0]: r[1] for r in rows}
        if run_id not in parent:
            return []
        root = self._root_for(run_id, parent)
        return [rid for rid in parent if self._root_for(rid, parent) == root]
    def rename_case(self, run_id: str, name: str) -> ResearchResult | None:
        result = self.load(run_id)
        if result is None:
            return None
        clean = (name or "").strip()
        ids = self._lineage_ids(run_id) or [run_id]
        with self._connect() as conn:
            rows = conn.execute("SELECT run_id, result_json FROM runs WHERE run_id IN ({})".format(",".join("?" for _ in ids)), ids).fetchall()
            for rid, raw in rows:
                stored = ResearchResult.model_validate_json(raw)
                conn.execute(
                    "UPDATE runs SET case_name = ?, result_json = ? WHERE run_id = ?",
                    (clean, stored.model_copy(update={"case_name": clean}).model_dump_json(), rid),
                )
        result.case_name = clean
        return result
    def set_pinned(self, run_id: str, pinned: bool) -> ResearchResult | None:
        result = self.load(run_id)
        if result is None:
            return None
        ids = self._lineage_ids(run_id) or [run_id]
        with self._connect() as conn:
            rows = conn.execute("SELECT run_id, result_json FROM runs WHERE run_id IN ({})".format(",".join("?" for _ in ids)), ids).fetchall()
            for rid, raw in rows:
                stored = ResearchResult.model_validate_json(raw)
                conn.execute(
                    "UPDATE runs SET pinned = ?, result_json = ? WHERE run_id = ?",
                    (int(bool(pinned)), stored.model_copy(update={"pinned": bool(pinned)}).model_dump_json(), rid),
                )
        result.pinned = bool(pinned)
        return result
    def load(self, run_id: str) -> ResearchResult | None:
        with self._connect() as conn:
            row = conn.execute("SELECT result_json FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if not row:
            return None
        raw = json.loads(row[0])
        legacy_map = {"technical_brief": "literature_review", "multi_paper_analysis": "paper_analysis", "technical_deep_dive": "paper_analysis"}
        if raw.get("report_type") in legacy_map:
            raw["report_type"] = legacy_map[raw["report_type"]]
        plan = raw.get("plan") or {}
        if plan.get("report_type") in legacy_map:
            plan["report_type"] = legacy_map[plan["report_type"]]
        if plan.get("mode") not in {"comparison", "landscape", "explanation", "design", "literature_review"}:
            plan["mode"] = "literature_review"
        raw["plan"] = plan
        return ResearchResult.model_validate(raw)
    def versions(self, result: ResearchResult, limit: int = 20) -> list[ResearchResult]:
        """Return every snapshot in the same case lineage, newest first.
        A user can intentionally reopen an older version and branch from it.
        Parent-chain-only traversal would hide the other branch from Research
        History, so the case is resolved by root lineage and all snapshots are
        returned in version/creation order.
        """
        with self._connect() as conn:
            rows = conn.execute("SELECT run_id, parent_run_id, version, created_at FROM runs ORDER BY created_at DESC").fetchall()
        parent = {r[0]: r[1] for r in rows}
        if result.run_id not in parent:
            return [result]
        root = self._root_for(result.run_id, parent)
        ids = [r[0] for r in rows if self._root_for(r[0], parent) == root]
        loaded: list[ResearchResult] = []
        for rid in ids:
            item = self.load(rid)
            if item is not None:
                loaded.append(item)
        loaded.sort(key=lambda x: (x.version, x.run_id), reverse=True)
        return loaded[:limit]
