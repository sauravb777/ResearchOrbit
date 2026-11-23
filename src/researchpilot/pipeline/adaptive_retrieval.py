from __future__ import annotations
import json
import re
from typing import Any
from ..llm import LLMClient
from ..schemas import PaperAnalysis, ResearchIntent
GAP_SEARCH_SYSTEM = """You are the retrieval strategist for a rigorous academic research agent.
The first retrieval and deep-reading pass is complete. Do NOT suggest generic future-work queries.
Instead, identify evidence that is genuinely missing from the current corpus and could materially
change, qualify, contradict, or strengthen the answer to the user's research question.
Look for:
1. important subquestions with no direct evidence,
2. claims supported by only one paper when independent evidence is needed,
3. contradictions or tension that need a paper under the relevant conditions,
4. untested boundary conditions, populations, datasets, settings, or failure modes,
5. missing baselines, controls, replications, or evaluation protocols,
6. an important method/mechanism that was mentioned but not actually examined.
Search queries must be scholarly and concrete. Prefer the missing concept + the research domain +
method/condition over simply repeating the original question. Avoid author names unless necessary.
Return JSON only:
{"gaps":[{"gap":"...","importance":0.0,"reason":"...","evidence_needed":"..."}],
 "queries":[{"query":"...","gap":"...","priority":0.0}]}.
At most 4 queries. Do not invent a factual gap that is already answered by the supplied evidence."""
def _json_call(llm: LLMClient, system: str, payload: dict[str, Any], max_tokens: int = 2600):
    if not llm or not llm.available:
        return None
    try:
        return llm.complete_json(system, json.dumps(payload, ensure_ascii=False), max_tokens=max_tokens)
    except Exception:
        return None
def _clean_query(query: str) -> str:
    query = re.sub(r"\s+", " ", str(query or "")).strip()
    query = query.strip("\"'")
    return query[:420]
def _fallback_queries(question: str, analyses: list[PaperAnalysis]) -> list[dict[str, Any]]:
    """Generate conservative gap searches without an LLM.
    This is intentionally narrow. It uses explicit 'not established' and
    failure/limitation language rather than guessing what the literature lacks.
    """
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for analysis in analyses:
        candidates = [
            ("not_established", getattr(analysis, "not_established", ""), 0.88),
            ("failure_modes", getattr(analysis, "failure_modes", ""), 0.82),
            ("open_questions", getattr(analysis, "open_questions", ""), 0.78),
        ]
        for kind, text, priority in candidates:
            text = _clean_query(text)
            if not text or text.lower() in {"not reported", "none", "n/a"}:
                continue
            query = _clean_query(f"{question} {text}")
            key = query.lower()
            if len(query) < 25 or key in seen:
                continue
            seen.add(key)
            out.append({"query": query, "gap": f"Explicit {kind.replace('_', ' ')}: {text}", "priority": priority})
            if len(out) >= 4:
                return out
    return out
def identify_evidence_gaps(
    question: str,
    report_type: str,
    intent: ResearchIntent | None,
    analyses: list[PaperAnalysis],
    evidence: list[Any],
    llm: LLMClient,
) -> tuple[dict[str, Any], bool]:
    """Return high-value missing-evidence hypotheses and targeted search queries."""
    if not analyses:
        return {"gaps": [], "queries": []}, False
    payload = {
        "question": question,
        "report_type": report_type,
        "intent": {
            "subquestions": list(getattr(intent, "subquestions", []) or []),
            "evidence_requirements": list(getattr(intent, "evidence_requirements", []) or []),
            "inclusion_criteria": list(getattr(intent, "inclusion_criteria", []) or []),
        },
        "papers": [
            {
                "source_id": a.source_id,
                "problem": a.problem,
                "core_idea": a.core_idea,
                "established": a.established,
                "not_established": a.not_established,
                "limitations": a.limitations,
                "failure_modes": a.failure_modes,
                "open_questions": a.open_questions,
                "generalization": a.generalization,
                "transfer_conditions": a.transfer_conditions,
                "negative_results": a.negative_results,
                "evidence_conflicts": a.evidence_conflicts,
                "experiment_records": a.experiment_records[:12],
                "research_relevance_map": a.research_relevance_map,
                "unmapped_observations": a.unmapped_observations[:20],
            }
            for a in analyses
        ],
        "evidence_inventory": [
            {"id": e.id, "source_id": e.source_id, "kind": e.kind, "text": e.text, "conditions": e.conditions}
            for e in evidence[:100]
        ],
    }
    data = _json_call(llm, GAP_SEARCH_SYSTEM, payload)
    if isinstance(data, dict):
        queries = []
        seen: set[str] = set()
        for item in data.get("queries", []) if isinstance(data.get("queries"), list) else []:
            if not isinstance(item, dict):
                continue
            q = _clean_query(item.get("query", ""))
            if len(q) < 15 or q.lower() in seen:
                continue
            seen.add(q.lower())
            queries.append({"query": q, "gap": _clean_query(item.get("gap", ""))[:500], "priority": float(item.get("priority", 0.5) or 0.5)})
        queries.sort(key=lambda x: x["priority"], reverse=True)
        data["queries"] = queries[:4]
        data.setdefault("gaps", [])
        return data, True
    return {"gaps": [], "queries": _fallback_queries(question, analyses)}, False
