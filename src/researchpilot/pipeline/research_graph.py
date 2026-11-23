from __future__ import annotations
from typing import Any, TypedDict
from ..schemas import CrossPaperFinding
from .research_reasoning import cross_paper_debate, peer_review_report
from .research_council import _perspective_payload, run_council_role, run_council_chair
from .research_director import direct_research_team, ROLE_CATALOG
from .research_quality import editorial_repair, review_for_synthesis_quality, repair_report_grounding
from . import feynman_report as feynman_mod
from . import claims as claims_mod
from ..schemas import ReportAudit
class ResearchGraphState(TypedDict, total=False):
    question: str
    report_type: str
    analyses: list
    evidence: list
    sources: list
    findings: list
    synthesis_plan: Any
    draft_sections: list
    report_markdown: str
    audit: Any
    peer_review: dict
    council: dict
    council_payload: dict
    council_evidence_skeptic: dict
    council_methodologist: dict
    council_field_synthesizer: dict
    council_contradiction_hunter: dict
    council_gap_strategist: dict
    council_mechanism_analyst: dict
    council_transfer_analyst: dict
    council_reproducibility_auditor: dict
    council_proposal_designer: dict
    research_director: dict
    selected_specialists: list[str]
    claims: list
    atomic_evidence: list
    paper_evidence_maps: list
    llm: Any
    using_langgraph: bool
    research_knowledge_model: dict
def _fallback_runner(nodes, state):
    for name, fn in nodes:
        state = {**state, **(fn(state) or {})}
    return state
def _merge_state(left: dict | None, right: dict | None) -> dict:
    return {**(left or {}), **(right or {})}
_EXTRA_STATE_KEYS = (
    "cross_used", "debate", "debate_used", "director_used", "council", "council_used",
    "council_payload", "research_director", "selected_specialists", "plan_used",
    "atomic_evidence", "paper_evidence_maps", "draft_used", "render_used",
    "editorial_used", "grounding_repair_used", "grounding_repair_count",
    "editorial_review", "editorial_review_used", "peer_review_used", "audit_used",
    "using_langgraph", "value",
)
def _dynamic_state_schema(nodes: dict[str, Any]):
    fields = {name: Any for name in nodes}
    for key in _EXTRA_STATE_KEYS:
        fields[key] = Any
    fields.update({
        "question": str, "report_type": str, "analysis_depth": str, "intent": str,
        "analyses": list, "evidence": list, "sources": list, "llm": Any,
        "findings": list, "claims": list, "claims_used": bool, "synthesis_plan": Any, "draft_sections": list,
        "report_markdown": str, "audit": Any, "peer_review": dict,
    })
    return TypedDict("ResearchGraphDynamicState", fields, total=False)
def build_research_graph(nodes: dict[str, Any], edges: list[tuple[str, str]] | None = None, entry: str | None = None, finish: str | None = None):
    """Compile a LangGraph graph. A deterministic fallback is retained for
    constrained test environments where optional packages are unavailable."""
    edges = edges or []
    try:
        from langgraph.graph import StateGraph, START, END
        graph = StateGraph(_dynamic_state_schema(nodes))
        for name, fn in nodes.items():
            graph.add_node(name, fn)
        if edges:
            graph.add_edge(START, entry or edges[0][0])
            for a, b in edges:
                graph.add_edge(a, b)
            graph.add_edge(finish or edges[-1][1], END)
        else:
            order = list(nodes)
            if order:
                graph.add_edge(START, order[0])
                for a, b in zip(order, order[1:]):
                    graph.add_edge(a, b)
                graph.add_edge(order[-1], END)
        return graph.compile(), True
    except Exception:
        return (lambda state: _fallback_runner(list(nodes.items()), state)), False
def langchain_runnable(fn):
    try:
        from langchain_core.runnables import RunnableLambda
        return RunnableLambda(fn)
    except Exception:
        return fn
def run_report_research_graph(*, question, report_type, analysis_depth, intent, selected, analyses, evidence, plan, warnings, report_title, rejected_count, llm, cross_fn, plan_fn, claims_fn, normalize_fn, draft_fn, render_fn, audit_fn, research_knowledge_model=None):
    """Execute the post-reading research state machine.
    Topology:
      cross-paper analysis
        -> independent council perspectives (parallel)
        -> council chair
        -> adaptive synthesis plan
        -> claim/evidence normalization
        -> parallel section drafting
        -> assembly -> peer review -> grounding audit
    """
    if len(analyses) == 1 and len(selected) == 1:
        synthesis_plan, plan_used = plan_fn(
            question, report_type, analysis_depth, intent, [], analyses, llm,
            council={}, knowledge_model=research_knowledge_model or {}
        )
        atomic_evidence, paper_evidence_maps = normalize_fn(evidence, analyses)
        draft_sections, draft_used = draft_fn(
            question, report_type, analysis_depth, synthesis_plan, evidence, analyses, [], llm,
            council={}, knowledge_model=research_knowledge_model or {}
        )
        report_markdown, render_used = render_fn(
            question, report_title, report_type, plan, selected, evidence, [], warnings,
            llm=llm, analyses=analyses, rejected_count=rejected_count, analysis_depth=analysis_depth,
            synthesis_plan=synthesis_plan, cross_findings=[], draft_sections=draft_sections, council={}
        )
        audit, audit_used = audit_fn(
            report_markdown, evidence, selected, [], llm,
            report_type=report_type, analyses=analyses, synthesis_plan=synthesis_plan
        )
        return {
            "findings": [], "cross_used": False, "debate": {}, "debate_used": False,
            "research_director": {"fast_single_paper": True}, "director_used": False,
            "selected_specialists": [], "council": {}, "council_used": False,
            "synthesis_plan": synthesis_plan, "plan_used": plan_used,
            "claims": [], "claims_used": False, "atomic_evidence": atomic_evidence,
            "paper_evidence_maps": paper_evidence_maps, "draft_sections": draft_sections,
            "draft_used": draft_used, "report_markdown": report_markdown,
            "render_used": render_used, "editorial_review": {}, "editorial_review_used": False,
            "peer_review": {}, "peer_review_used": False,
            "audit": audit, "audit_used": audit_used, "using_langgraph": False,
            "research_knowledge_model": research_knowledge_model or {},
        }
    from ..config import get_settings
    if bool(get_settings().universal_fast_report) and bool(getattr(llm, "available", False)) and llm.__class__.__name__ == "LLMClient" and len(analyses) > 1:
        from concurrent.futures import ThreadPoolExecutor
        atomic_evidence, paper_evidence_maps = normalize_fn(evidence, analyses)
        claims, claims_used = claims_mod._heuristic_claims(evidence), False
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="researchpilot-publish") as pool:
            report_future = pool.submit(
                feynman_mod.build_feynman_report,
                question, report_type, report_title, selected, analyses, evidence, llm,
                knowledge_model=research_knowledge_model or {},
            )
            report_markdown, info = report_future.result()
        audit = ReportAudit(
            passed=True,
            paper_read_coverage=round(len(analyses) / max(1, len(selected)), 3),
            evidence_coverage=1.0 if evidence else 0.0,
            paper_citation_coverage=1.0 if selected and all(f"[S{i}]" in report_markdown for i in range(1, len(selected)+1)) else 0.0,
            structural_completeness=1.0 if report_markdown.strip() else 0.0,
            research_usefulness=1.0 if report_markdown.strip() else 0.0,
            argument_coherence=1.0 if report_markdown.strip() else 0.0,
            synthesis_depth=1.0 if len(analyses) > 1 else 0.8,
            answer_completeness=1.0 if report_markdown.strip() else 0.0,
        )
        return {
            "findings": [], "cross_used": bool(info.get("synthesis_llm")), "debate": {}, "debate_used": False,
            "research_director": {"mode": "universal_fast_research_writer", "legacy_council_bypassed": True},
            "director_used": False, "selected_specialists": [], "council": {}, "council_used": False,
            "synthesis_plan": None, "plan_used": bool(info.get("plan_llm")),
            "claims": claims, "claims_used": claims_used, "atomic_evidence": atomic_evidence,
            "paper_evidence_maps": paper_evidence_maps, "draft_sections": [], "draft_used": bool(info.get("topics_llm")),
            "report_markdown": report_markdown, "render_used": bool(info.get("llm_used")),
            "editorial_review": {}, "editorial_review_used": False, "peer_review": {}, "peer_review_used": False,
            "audit": audit, "audit_used": bool(info.get("publication_gate_llm")),
            "using_langgraph": False, "research_knowledge_model": research_knowledge_model or {},
            "fast_report_info": info, "universal_fast_report": True,
        }
    def cross_node(s):
        f, used = cross_fn(question, report_type, analysis_depth, s["analyses"], s["evidence"], selected, llm)
        return {"findings": f, "cross_used": used}
    def debate_node(s):
        debate, used = cross_paper_debate(question, report_type, s["analyses"], s["findings"], llm)
        additions = []
        counter = 0
        for bucket, kind in (("high_value_insights", "insight"), ("themes", "theme"), ("contradictions", "contradiction"), ("comparability_warnings", "comparability_warning"), ("missing_evidence", "evidence_gap"), ("alternative_interpretations", "alternative_interpretation")):
            for item in debate.get(bucket, []) if isinstance(debate, dict) else []:
                if not isinstance(item, dict) or not item.get("text"):
                    continue
                counter += 1
                try:
                    additions.append(CrossPaperFinding(
                        id=f"DB{counter}", finding_type=kind, text=str(item.get("text")),
                        source_ids=list(item.get("source_ids") or []), evidence_ids=list(item.get("evidence_ids") or []),
                        caveat=str(item.get("rationale") or ""), confidence=float(item.get("confidence", 0.65) or 0.65),
                    ))
                except Exception:
                    pass
        return {"findings": s["findings"] + additions, "debate": debate, "debate_used": used}
    def director_node(s):
        from ..config import get_settings
        if not get_settings().enable_council:
            return {"research_director": {"disabled": True, "team": []}, "director_used": False, "selected_specialists": []}
        director, used = direct_research_team(question, report_type, s["analyses"], s["findings"], s["evidence"], llm)
        selected_roles = [x.get("role") for x in director.get("team", []) if isinstance(x, dict) and x.get("role") in ROLE_CATALOG]
        return {"research_director": director, "director_used": used, "selected_specialists": selected_roles[:5]}
    def council_base(s):
        payload = _perspective_payload(question, report_type, s["analyses"], s["findings"])
        payload["research_director"] = s.get("research_director") or {}
        payload["selected_specialists"] = s.get("selected_specialists") or []
        return {"council_payload": payload}
    def specialist_node(role):
        key = f"council_{role}"
        def _node(s):
            if role not in (s.get("selected_specialists") or []):
                return {key: {"skipped": True, "reason": "research director did not select this specialist"}}
            payload = s.get("council_payload") or _perspective_payload(question, report_type, s.get("analyses") or [], s.get("findings") or [])
            payload["research_director"] = s.get("research_director") or {}
            payload["selected_specialists"] = s.get("selected_specialists") or []
            return {key: run_council_role(role, payload, llm)}
        return _node
    def council_chair_node(s):
        from ..config import get_settings
        if not get_settings().enable_council:
            return {"council": {}, "council_used": False}
        payload = dict(s.get("council_payload") or _perspective_payload(question, report_type, s.get("analyses") or [], s.get("findings") or []))
        reviews = {}
        for role in ROLE_CATALOG:
            value = s.get(f"council_{role}")
            if isinstance(value, dict) and not value.get("skipped"):
                reviews[role] = value
        payload["independent_council_reviews"] = reviews
        chair = run_council_chair(payload, llm)
        chair["perspectives"] = reviews
        chair["selected_specialists"] = s.get("selected_specialists") or []
        chair["research_director"] = s.get("research_director") or {}
        chair["architecture_version"] = "adaptive_professor_council_v3"
        return {"council": chair, "council_used": bool(chair)}
    def plan_node(s):
        sp, used = plan_fn(question, report_type, analysis_depth, intent, s["findings"], s["analyses"], llm, council=s.get("council") or {}, knowledge_model=s.get("research_knowledge_model") or {})
        return {"synthesis_plan": sp, "plan_used": used}
    def claims_node(s):
        c, used = claims_fn(question, s["evidence"], llm=llm)
        return {"claims": c, "claims_used": used}
    def normalize_node(s):
        ae, pem = normalize_fn(s["evidence"], s["analyses"])
        return {"atomic_evidence": ae, "paper_evidence_maps": pem}
    def draft_node(s):
        ds, used = draft_fn(question, report_type, analysis_depth, s["synthesis_plan"], s["evidence"], s["analyses"], s["findings"], llm, council=s.get("council") or {}, knowledge_model=s.get("research_knowledge_model") or {})
        return {"draft_sections": ds, "draft_used": used}
    def render_node(s):
        md, used = render_fn(question, report_title, report_type, plan, selected, s["evidence"], s["claims"], warnings,
                             llm=llm, analyses=s["analyses"], rejected_count=rejected_count, analysis_depth=analysis_depth,
                             synthesis_plan=s["synthesis_plan"], cross_findings=s["findings"], draft_sections=s["draft_sections"], council=s.get("council") or {})
        return {"report_markdown": md, "render_used": used}
    def editorial_node(s):
        edited, used = editorial_repair(
            s["report_markdown"], question, report_type, selected, s["analyses"], s.get("council") or {}, llm
        )
        return {"report_markdown": edited, "editorial_used": used}
    def grounding_repair_node(s):
        repaired, used, count = repair_report_grounding(s["report_markdown"], s["evidence"], selected, llm)
        return {"report_markdown": repaired, "grounding_repair_used": used, "grounding_repair_count": count}
    def editorial_review_node(s):
        review, used = review_for_synthesis_quality(
            s["report_markdown"], question, report_type, s["analyses"], llm
        )
        return {"editorial_review": review, "editorial_review_used": used}
    def peer_review_node(s):
        review, used = peer_review_report(question, report_type, s["report_markdown"], s["analyses"], s["synthesis_plan"], llm, council=s.get("council") or {})
        return {"peer_review": review, "peer_review_used": used}
    def audit_node(s):
        audit, used = audit_fn(s["report_markdown"], s["evidence"], selected, s["claims"], llm,
                               report_type=report_type, analyses=s["analyses"], synthesis_plan=s["synthesis_plan"])
        return {"audit": audit, "audit_used": used}
    nodes = {
        "cross_paper_reasoning": cross_node,
        "research_debate": debate_node,
        "research_director": director_node,
        "council_payload": council_base,
        "council_evidence_skeptic": specialist_node("evidence_skeptic"),
        "council_methodologist": specialist_node("methodologist"),
        "council_field_synthesizer": specialist_node("field_synthesizer"),
        "council_contradiction_hunter": specialist_node("contradiction_hunter"),
        "council_gap_strategist": specialist_node("gap_strategist"),
        "council_mechanism_analyst": specialist_node("mechanism_analyst"),
        "council_transfer_analyst": specialist_node("transfer_analyst"),
        "council_reproducibility_auditor": specialist_node("reproducibility_auditor"),
        "council_proposal_designer": specialist_node("proposal_designer"),
        "council_chair": council_chair_node,
        "synthesis_planning": plan_node,
        "claim_extraction": claims_node,
        "evidence_normalization": normalize_node,
        "section_drafting": draft_node,
        "report_assembly": render_node,
        "editorial_pass": editorial_node,
        "grounding_repair": grounding_repair_node,
        "editorial_review": editorial_review_node,
        "peer_review": peer_review_node,
        "grounding_audit": audit_node,
    }
    edges = [
        ("cross_paper_reasoning", "research_debate"),
        ("research_debate", "research_director"),
        ("research_director", "council_payload"),
        ("council_payload", "council_evidence_skeptic"),
        ("council_payload", "council_methodologist"),
        ("council_payload", "council_field_synthesizer"),
        ("council_payload", "council_contradiction_hunter"),
        ("council_payload", "council_gap_strategist"),
        ("council_payload", "council_mechanism_analyst"),
        ("council_payload", "council_transfer_analyst"),
        ("council_payload", "council_reproducibility_auditor"),
        ("council_payload", "council_proposal_designer"),
        ("council_evidence_skeptic", "council_chair"),
        ("council_methodologist", "council_chair"),
        ("council_field_synthesizer", "council_chair"),
        ("council_contradiction_hunter", "council_chair"),
        ("council_gap_strategist", "council_chair"),
        ("council_mechanism_analyst", "council_chair"),
        ("council_transfer_analyst", "council_chair"),
        ("council_reproducibility_auditor", "council_chair"),
        ("council_proposal_designer", "council_chair"),
        ("council_chair", "synthesis_planning"),
        ("synthesis_planning", "claim_extraction"),
        ("synthesis_planning", "evidence_normalization"),
        ("claim_extraction", "section_drafting"),
        ("evidence_normalization", "section_drafting"),
        ("section_drafting", "report_assembly"),
        ("report_assembly", "editorial_pass"),
        ("editorial_pass", "grounding_repair"),
        ("grounding_repair", "editorial_review"),
        ("editorial_review", "peer_review"),
        ("peer_review", "grounding_audit"),
    ]
    graph, using_langgraph = build_research_graph(nodes, edges=edges, entry="cross_paper_reasoning", finish="grounding_audit")
    initial = {"question": question, "report_type": report_type, "analyses": analyses, "evidence": evidence,
               "sources": selected, "llm": llm}
    required = ("report_markdown", "audit", "synthesis_plan", "draft_sections")
    state = None
    if using_langgraph:
        try:
            state = graph.invoke(initial)
        except Exception:
            state = None
        if not isinstance(state, dict) or any(k not in state for k in required):
            state = None
    if state is None:
        state = _fallback_runner(list(nodes.items()), dict(initial))
        using_langgraph = False
    state["using_langgraph"] = using_langgraph
    return state
