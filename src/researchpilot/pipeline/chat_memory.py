from __future__ import annotations
import re
from typing import Any
from ..schemas import Evidence, PaperAnalysis, ResearchResult, Source
from .feynman_report import paper_card, _clean
from .synthesizer import citation_map
def _report_sections(markdown: str) -> list[dict[str, str]]:
    if not markdown:
        return []
    out: list[dict[str, str]] = []
    title = "Report"
    buf: list[str] = []
    for line in markdown.splitlines():
        if line.startswith("## "):
            if buf:
                body = "\n".join(buf).strip()
                if title.lower() != "references" and body:
                    out.append({"title": title, "text": body})
            title, buf = line[3:].strip(), []
        else:
            buf.append(line)
    if buf:
        body = "\n".join(buf).strip()
        if title.lower() != "references" and body:
            out.append({"title": title, "text": body})
    return out
def build_research_chat_memory(result: ResearchResult) -> dict[str, Any]:
    """Build the reusable chat/verification memory for a completed case."""
    sources = list(result.sources or [])
    cmap = citation_map(sources)
    analyses = {a.source_id: a for a in (result.paper_analyses or [])}
    by_tag = {cmap[s.id]: s for s in sources if s.id in cmap}
    evidence_by_tag: dict[str, list[dict[str, Any]]] = {tag: [] for tag in by_tag}
    for e in result.evidence or []:
        tag = cmap.get(e.source_id)
        if not tag:
            continue
        evidence_by_tag.setdefault(tag, []).append({
            "id": e.id, "text": e.text, "locator": e.locator, "kind": e.kind,
            "relevance": e.relevance, "conditions": e.conditions,
            "evidence_level": e.evidence_level,
        })
    papers: dict[str, dict[str, Any]] = {}
    for tag, source in by_tag.items():
        a = analyses.get(source.id)
        papers[tag] = {
            "source_id": source.id,
            "title": source.title,
            "authors": source.authors,
            "year": source.year,
            "abstract": source.abstract,
            "full_text": source.full_text,
            "full_text_complete": source.full_text_complete,
            "dossier": paper_card(tag, source, a, limit=6200),
            "analysis": a.model_dump(mode="json") if a is not None else {},
            "evidence": evidence_by_tag.get(tag, [])[:80],
        }
    memory = {
        "version": 1,
        "run_id": result.run_id,
        "research_question": result.question,
        "report_title": result.report_title,
        "report_type": result.report_type,
        "source_order": [cmap[s.id] for s in sources if s.id in cmap],
        "papers": papers,
        "report_sections": _report_sections(result.report_markdown),
        "knowledge_model": result.research_knowledge_model or {},
        "evidence_count": len(result.evidence or []),
        "paper_count": len(sources),
    }
    return memory
def memory_for_tags(memory: dict[str, Any], tags: list[str]) -> dict[str, Any]:
    papers = memory.get("papers") or {}
    wanted = {str(t).upper() for t in tags}
    return {tag: papers[tag] for tag in memory.get("source_order", []) if tag in wanted and tag in papers}
