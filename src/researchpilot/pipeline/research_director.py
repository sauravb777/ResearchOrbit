from __future__ import annotations
import json
from typing import Any
from ..llm import LLMClient
ROLE_CATALOG: dict[str, str] = {
    "evidence_skeptic": "stress-test claims, evidence strength, overclaiming and unsupported conclusions",
    "methodologist": "audit datasets, controls, baselines, metrics, protocols, uncertainty and comparability",
    "field_synthesizer": "discover themes, intellectual structure, consensus, tensions and extensions",
    "contradiction_hunter": "find genuine disagreements and test whether conditions explain the disagreement",
    "gap_strategist": "identify high-value unanswered questions and what evidence would change the answer",
    "mechanism_analyst": "trace why methods work, which components matter and what mechanisms are actually supported",
    "transfer_analyst": "test generalization, boundary conditions, populations, domains and transfer assumptions",
    "reproducibility_auditor": "inspect implementation detail, data availability, ablations, compute and reproducibility risks",
    "proposal_designer": "turn evidence-supported gaps into defensible research questions and evaluation designs",
}
DIRECTOR_PROMPT = """You are the research director coordinating a small internal academic research team.
You are NOT writing the final report and you are NOT required to use every specialist.
Choose only the specialists that are justified by the actual question and evidence.
Core principle: the report type is a user preference for presentation, not a rigid checklist.
The team must adapt to what the corpus contains. If the corpus is sparse, prioritize evidence
quality and answer boundaries. If it is rich, spend calls on disagreement, mechanisms, transfer,
reproducibility or other high-value distinctions. Never create a specialist assignment merely
because its name sounds relevant.
Return JSON only:
{
  "team": [{"role":"...", "priority":0.0, "reason":"...", "questions":[]}],
  "research_strategy": "...",
  "must_preserve": [],
  "likely_report_shape": [],
  "stop_conditions": []
}
Rules:
- Select 2 to 5 roles.
- evidence_skeptic is usually selected unless the corpus has no usable evidence.
- methodologist is selected when empirical comparison/testing matters.
- field_synthesizer is selected for multi-paper synthesis.
- contradiction_hunter only when there are signs of tension or multiple approaches.
- gap_strategist when important uncertainty remains.
- mechanism_analyst for technical/method questions.
- transfer_analyst when generalization matters or evidence comes from narrow settings.
- reproducibility_auditor when implementation/evaluation transparency matters.
- proposal_designer only for design/proposal-oriented questions.
- Do not invent facts about papers. Base assignments on the supplied snapshot.
"""
def _call(llm: LLMClient, payload: dict[str, Any]) -> dict[str, Any]:
    if not llm or not llm.available:
        return {}
    try:
        result = llm.complete_json(DIRECTOR_PROMPT, json.dumps(payload, ensure_ascii=False), max_tokens=2600, tier="S", priority="report")
        return result if isinstance(result, dict) else {}
    except TypeError:
        try:
            result = llm.complete_json(DIRECTOR_PROMPT, json.dumps(payload, ensure_ascii=False), max_tokens=2600)
            return result if isinstance(result, dict) else {}
        except Exception:
            return {}
    except Exception:
        return {}
def _heuristic_team(question: str, report_type: str, analyses: list[Any], findings: list[Any]) -> dict[str, Any]:
    text = " ".join(
        str(x)
        for a in analyses
        for x in (
            getattr(a, "not_established", ""), getattr(a, "limitations", ""),
            getattr(a, "evidence_conflicts", ""), getattr(a, "failure_modes", ""),
            getattr(a, "transfer_conditions", ""), getattr(a, "open_questions", ""),
        )
    ).lower()
    team: list[dict[str, Any]] = [
        {"role": "evidence_skeptic", "priority": 1.0, "reason": "Every synthesis benefits from an explicit evidence-boundary check.", "questions": ["Which conclusions are directly supported?"]},
    ]
    if len(analyses) >= 2:
        team.append({"role": "field_synthesizer", "priority": 0.95, "reason": "Multiple papers require cross-paper structure rather than paper-by-paper summary.", "questions": ["What does the corpus collectively establish?"]})
    if report_type in {"comparison", "evidence_review", "state_of_art"} or any(k in question.lower() for k in ("compare", "benchmark", "evaluation", "evidence")):
        team.append({"role": "methodologist", "priority": 0.92, "reason": "The question depends on evaluation or comparison conditions.", "questions": ["Which comparisons are actually valid?"]})
    if any(k in text for k in ("contradict", "disagree", "conflict", "mixed", "inconsistent")) or len(findings) >= 8:
        team.append({"role": "contradiction_hunter", "priority": 0.88, "reason": "The evidence snapshot contains signals of disagreement or competing findings.", "questions": ["Are disagreements substantive or condition-dependent?"]})
    if any(k in text for k in ("not established", "open question", "unknown", "future", "limitation", "gap")):
        team.append({"role": "gap_strategist", "priority": 0.84, "reason": "The corpus exposes unresolved areas that may matter to the answer.", "questions": ["What missing evidence could change the conclusion?"]})
    if any(k in question.lower() for k in ("how does", "how do", "mechanism", "architecture", "method", "algorithm")):
        team.append({"role": "mechanism_analyst", "priority": 0.82, "reason": "The question asks for method-level explanation.", "questions": ["What mechanism is actually demonstrated?"]})
    if any(k in text for k in ("generaliz", "transfer", "domain", "population", "distribution")):
        team.append({"role": "transfer_analyst", "priority": 0.78, "reason": "Generalization boundaries are material to the available evidence.", "questions": ["Where can these findings safely transfer?"]})
    if report_type == "research_proposal":
        team.append({"role": "proposal_designer", "priority": 0.80, "reason": "A proposal requires evidence-grounded gap-to-evaluation design.", "questions": ["What design follows from the evidence rather than from template fields?"]})
    seen: set[str] = set()
    clean = []
    for item in sorted(team, key=lambda x: float(x.get("priority", 0)), reverse=True):
        role = item.get("role")
        if role in ROLE_CATALOG and role not in seen:
            seen.add(role)
            clean.append(item)
        if len(clean) >= 5:
            break
    return {
        "team": clean,
        "research_strategy": "Use the selected specialists as independent lenses, then let the chair integrate only evidence-supported conclusions.",
        "must_preserve": ["source/evidence provenance", "uncertainty", "non-comparable results", "answer boundary"],
        "likely_report_shape": [],
        "stop_conditions": ["Do not add sections merely to satisfy the requested report type.", "Do not resolve contradictions without evidence."]
    }
def direct_research_team(question: str, report_type: str, analyses: list[Any], findings: list[Any], evidence: list[Any], llm: LLMClient) -> tuple[dict[str, Any], bool]:
    if not analyses:
        return {"team": []}, False
    payload = {
        "question": question,
        "report_type": report_type,
        "role_catalog": ROLE_CATALOG,
        "paper_snapshot": [
            {
                "source_id": a.source_id,
                "problem": a.problem,
                "objective": getattr(a, "research_objective", ""),
                "method": a.method_summary,
                "results": a.key_results,
                "established": a.established,
                "not_established": a.not_established,
                "limitations": a.limitations,
                "failure_modes": a.failure_modes,
                "generalization": a.generalization,
                "transfer_conditions": a.transfer_conditions,
                "evidence_conflicts": a.evidence_conflicts,
                "open_questions": a.open_questions,
            }
            for a in analyses
        ],
        "cross_findings": [
            {"id": f.id, "type": f.finding_type, "text": f.text, "source_ids": f.source_ids, "evidence_ids": f.evidence_ids}
            for f in findings[:100]
        ],
        "evidence_count": len(evidence),
    }
    data = _call(llm, payload)
    if data:
        clean: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in data.get("team", []) if isinstance(data.get("team"), list) else []:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role", "")).strip()
            if role not in ROLE_CATALOG or role in seen:
                continue
            seen.add(role)
            clean.append({
                "role": role,
                "priority": max(0.0, min(1.0, float(item.get("priority", 0.5) or 0.5))),
                "reason": str(item.get("reason", ""))[:500],
                "questions": [str(q)[:300] for q in (item.get("questions") or [])[:5]],
            })
        clean.sort(key=lambda x: x["priority"], reverse=True)
        data["team"] = clean[:5]
        return data, True
    return _heuristic_team(question, report_type, analyses, findings), False
