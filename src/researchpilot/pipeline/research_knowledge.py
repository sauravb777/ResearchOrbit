from __future__ import annotations
import json
from typing import Any
from ..llm import LLMClient
from ..schemas import Evidence, PaperAnalysis, Source
_KNOWLEDGE_SYSTEM = """You are the internal research synthesis engine for ResearchOrbit.
Build a compact, evidence-grounded knowledge model from the supplied full-paper reading records.
Do not add outside knowledge. Do not infer a relationship merely because two papers discuss related topics.
Distinguish direct findings from cross-paper synthesis and unknowns. Preserve conditions and contradictions.
The model is an internal reasoning aid, not a report about the research process.
Return JSON only with keys: question_components, themes, relationships, established_knowledge,
unresolved_knowledge, evidence_gaps, important_conditions, comparison_matrix, conclusion_logic, knowledge_boundaries, targeted_reread_requests.
A targeted_reread_requests item must be {"source_tag":"S1","query":"...","reason":"..."}. Request rereads only when a missing or ambiguous detail could materially change the synthesis. Never request a new paper.
Each item should contain source_tags where applicable. Keep the result compact and information-dense."""
def _analysis_blob(a: PaperAnalysis, tag: str) -> dict[str, Any]:
    return {
        "source_tag": tag,
        "problem": a.problem,
        "objective": a.research_objective,
        "core_idea": a.core_idea,
        "method": a.method_summary,
        "mechanism": a.mechanism,
        "setup": a.experimental_setup,
        "datasets": a.dataset_details,
        "baselines": a.baselines,
        "metrics": a.metrics,
        "results": a.key_results,
        "quantitative_results": [q.model_dump() for q in a.quantitative_results],
        "limitations": a.limitations,
        "established": a.established,
        "not_established": a.not_established,
        "assumptions": a.assumptions,
        "failure_modes": a.failure_modes,
        "generalization": a.generalization,
        "interpretation": a.result_interpretation,
        "open_questions": a.open_questions,
        "evidence_inventory": (a.evidence_inventory or [])[:14],
    }
def _deterministic(question: str, sources: list[Source], analyses: list[PaperAnalysis]) -> dict[str, Any]:
    tag_by_id = {s.id: f"S{i}" for i, s in enumerate(sources, 1)}
    established, unresolved, conditions, themes = [], [], [], []
    for a in analyses:
        tag = tag_by_id.get(a.source_id, "S?")
        if a.established:
            established.append({"statement": a.established, "source_tags": [tag], "basis": "paper_record"})
        elif a.key_results:
            established.append({"statement": a.key_results, "source_tags": [tag], "basis": "paper_result"})
        if a.not_established or a.open_questions:
            unresolved.append({"statement": a.not_established or a.open_questions, "source_tags": [tag]})
        if a.limitations or a.assumptions:
            conditions.append({"statement": a.limitations or a.assumptions, "source_tags": [tag]})
        idea = a.core_idea or a.method_summary or a.problem
        if idea:
            themes.append({"theme": a.method_category or "paper contribution", "statement": idea, "source_tags": [tag]})
    return {
        "research_question": question,
        "question_components": [question],
        "paper_roles": [{"source_tag": tag_by_id.get(a.source_id, "S?"), "role": a.method_category or "evidence contributor"} for a in analyses],
        "conceptual_model": "",
        "themes": themes,
        "relationships": [],
        "established_knowledge": established,
        "unresolved_knowledge": unresolved,
        "evidence_gaps": unresolved,
        "important_conditions": conditions,
        "comparison_matrix": [],
        "conclusion_logic": [],
        "knowledge_boundaries": ["Only claims established by the supplied paper records are represented."],
        "targeted_reread_requests": [],
    }
def build_research_knowledge(question: str, sources: list[Source], analyses: list[PaperAnalysis], evidence: list[Evidence] | None = None, llm: LLMClient | None = None) -> tuple[dict[str, Any], bool]:
    base = _deterministic(question, sources, analyses)
    llm = llm or LLMClient()
    if not llm.available or not analyses:
        return base, False
    tags = {s.id: f"S{i}" for i, s in enumerate(sources, 1)}
    payload = {
        "research_question": question,
        "papers": [_analysis_blob(a, tags.get(a.source_id, "S?")) for a in analyses],
        "high_value_evidence": [{"source_tag": tags.get(e.source_id, "S?"), "locator": e.locator, "text": e.text, "kind": e.kind, "conditions": e.conditions, "level": e.evidence_level} for e in (evidence or [])[:60]],
    }
    try:
        raw = llm.complete_json(_KNOWLEDGE_SYSTEM, json.dumps(payload, ensure_ascii=False)[:42000], temperature=0.12, max_tokens=3000)
    except Exception:
        raw = None
    if not isinstance(raw, dict):
        return base, False
    raw["research_question"] = question
    raw.setdefault("paper_roles", base["paper_roles"])
    raw.setdefault("targeted_reread_requests", [])
    raw.setdefault("knowledge_boundaries", base["knowledge_boundaries"])
    raw["source_tags"] = sorted(tags.values(), key=lambda x: int(x[1:]))
    requests = raw.get("targeted_reread_requests") or []
    if requests and evidence is not None:
        from .retrieval import retrieve_passages
        by_tag = {tag: s for sid, tag in tags.items() for s in sources if s.id == sid}
        reread_blocks = []
        for req in requests[:4]:
            if not isinstance(req, dict):
                continue
            tag = str(req.get("source_tag") or "").upper()
            src = by_tag.get(tag)
            query = str(req.get("query") or "").strip()
            if not src or not query:
                continue
            passages = retrieve_passages(query, [src], top_k=2, per_source_cap=2, chunk_chars=1800, overlap_chars=200)
            for passage in passages:
                reread_blocks.append({
                    "source_tag": tag,
                    "locator": f"page {passage.page}" if passage.page else (passage.section or "paper text"),
                    "text": passage.text,
                    "reason": str(req.get("reason") or ""),
                })
        if reread_blocks:
            try:
                refinement_system = _KNOWLEDGE_SYSTEM + "\nThis is a refinement pass. Correct only what the targeted literal passages justify. Preserve all supported prior knowledge and do not add outside facts."
                refined_payload = {
                    "research_question": question,
                    "prior_knowledge_model": raw,
                    "targeted_rereads": reread_blocks,
                }
                refined = llm.complete_json(refinement_system, json.dumps(refined_payload, ensure_ascii=False)[:26000], temperature=0.08, max_tokens=2600)
                if isinstance(refined, dict):
                    refined["research_question"] = question
                    refined["paper_roles"] = raw.get("paper_roles", base["paper_roles"])
                    refined["targeted_reread_requests"] = raw.get("targeted_reread_requests", [])
                    refined["targeted_reread_evidence"] = reread_blocks
                    refined["knowledge_boundaries"] = refined.get("knowledge_boundaries") or raw.get("knowledge_boundaries", base["knowledge_boundaries"])
                    refined["source_tags"] = raw["source_tags"]
                    raw = refined
            except Exception:
                raw["targeted_reread_evidence"] = reread_blocks
    return raw, True
