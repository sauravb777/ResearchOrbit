from __future__ import annotations
from dataclasses import dataclass
from pydantic import ValidationError
from ..llm import LLMClient
from ..schemas import ResearchPlan, canonical_report_type
_SYSTEM_PROMPT = (
    "You are a research strategist designing the retrieval and analytical plan for a serious academic deliverable. "
    "Given a research question AND a specified report type, propose non-redundant, compact technical search queries "
    "using established terminology, synonyms, alternate task names, and relevant datasets/method families. Queries "
    "must work as academic search phrases, not long paraphrases of the user's prose. Also propose precise "
    "subquestions that serve that report purpose. Do not optimize for generic topic mentions: prioritize papers "
    "that can provide primary evidence, comparative experiments, mechanisms, limitations, evaluations, or gaps "
    "appropriate to the report type. Respond with ONLY a JSON object matching this schema: "
    '{"mode": "comparison|landscape|explanation|design|literature_review", "subquestions": ["..."], "search_queries": ["..."], "analysis_axes": ["..."]}. '
    "No prose, no markdown fences."
)
def _classify_mode(question: str) -> str:
    q = question.lower()
    if any(k in q for k in ("compare", "versus", " vs ", "trade-off", "which is better", "benchmark")):
        return "comparison"
    if any(k in q for k in ("gap analysis", "research gap", "gaps in", "underexplored", "open problem")):
        return "landscape"
    if any(k in q for k in ("latest", "recent", "state of the art", "trend", "what approaches")):
        return "landscape"
    if any(k in q for k in ("how does", "explain", "what is", "why does", "why do")):
        return "explanation"
    if any(k in q for k in ("design", "build", "propose", "architecture", "system for")):
        return "design"
    return "literature_review"
_AXES_BY_MODE = {
    "comparison": ["problem", "method", "data/evaluation", "results", "cost", "limitations"],
    "landscape": ["major approaches", "recent developments", "evaluation", "open problems"],
    "explanation": ["definition", "mechanism", "assumptions", "evidence", "limitations"],
    "design": ["requirements", "architecture options", "trade-offs", "failure modes", "evaluation"],
    "literature_review": ["problem", "methods", "datasets", "metrics", "results", "gaps"],
    "evidence_review": ["inclusion criteria", "study quality", "effect direction", "consistency across studies", "certainty of evidence"],
    "paper_analysis": ["claims", "method", "evidence for each claim", "limitations", "relevance to the question"],
}
_AXES_BY_REPORT_TYPE = {
    "standard_report": ["problem framing", "literature landscape", "methods and technical approaches", "datasets and evaluation", "results", "comparison and trade-offs", "limitations", "research gaps", "integrated synthesis"],
    "literature_review": ["problem framing", "method families", "datasets and evaluation", "results", "agreement/disagreement", "limitations", "research gaps"],
    "state_of_art": ["approach taxonomy", "chronological development", "benchmark landscape", "design patterns", "robustness/generalization", "bottlenecks", "open problems"],
    "gap_analysis": ["current state", "established vs unresolved", "gap taxonomy", "assumptions and boundaries", "why gaps persist", "priority research questions", "evidence needed to close gaps"],
    "comparison": ["problem alignment", "method/mechanism", "datasets and protocols", "results", "ablations", "cost", "robustness and failure modes"],
    "evidence_review": ["search and screening scope", "study characteristics", "evidence extraction", "study quality", "consistency", "bias/limitations", "evidence synthesis"],
    "research_proposal": ["evidence-backed problem", "unresolved gap", "research questions", "hypotheses", "baselines", "ablations", "evaluation", "risks", "reproducibility"],
    "paper_analysis": ["problem/objective", "novelty", "method", "mechanism", "experimental setup", "results", "ablations", "robustness", "limitations", "what is established"],
}
_SUBQUESTIONS_BY_REPORT_TYPE = {
    "standard_report": ["What does the retained literature establish?", "What are the main approaches and how do they work?", "What does the experimental evidence show?", "Where do studies agree, disagree or fail to be directly comparable?", "What limitations and research gaps remain?"],
    "literature_review": ["What does the retained literature establish?", "How do the main approaches differ?", "Under what conditions do reported results hold?", "Where do studies agree or disagree?", "What remains unresolved?"],
    "state_of_art": ["What are the major approach families?", "What has changed in recent work?", "How is progress currently evaluated?", "Which bottlenecks recur across studies?", "What technical problems remain open?"],
    "gap_analysis": ["What is already well established?", "Which findings or capabilities remain uncertain?", "What kinds of gaps are present?", "Why have existing studies not resolved them?", "What evidence or experiments would resolve the highest-priority gaps?"],
    "comparison": ["What exactly is being compared and are the settings aligned?", "How does each method work?", "Which results are genuinely comparable?", "What trade-offs emerge in cost, robustness and performance?", "Which conditions change the interpretation of the comparison?"],
    "evidence_review": ["What was searched and retained?", "What evidence does each study contribute?", "How consistent is the evidence?", "Where are quality or coverage limitations?", "What conclusion is justified by the evidence as a whole?"],
    "research_proposal": ["What evidence establishes the problem?", "What specific gap justifies new work?", "What research questions follow from that gap?", "What experiment would distinguish competing explanations?", "How should success, robustness and failure be evaluated?"],
    "paper_analysis": ["What is the paper actually trying to establish?", "What is novel and how does the mechanism work?", "How convincing are the experiments and ablations?", "What assumptions and failure modes matter?", "What does the paper establish and what does it leave open?"],
}
@dataclass(frozen=True)
class ReportBlueprint:
    """Research contract bound before retrieval and paper reading.
    The blueprint is deliberately data driven so report type changes the
    questions asked of every paper and the cross-paper synthesis contract,
    rather than only changing the final headings.
    """
    report_type: str
    lens_description: str
    per_paper_lens_fields: tuple[str, ...]
    synthesis_strategy: str
    section_template: tuple[str, ...]
_REPORT_BLUEPRINTS = {
    "standard_report": ReportBlueprint(
        "standard_report",
        "balanced understanding of problem, methods, evidence, limitations and gaps",
        ("method_summary", "experimental_setup", "key_results", "limitations", "gaps"),
        "thematic synthesis with evidence-weighted comparison and gap analysis",
        ("Executive Summary", "Background and Scope", "Literature Landscape", "Technical and Methodological Analysis", "Evidence and Findings", "Comparative Analysis", "Limitations", "Research Gaps", "Conclusion", "References"),
    ),
    "literature_review": ReportBlueprint(
        "literature_review",
        "thematic synthesis of what the literature establishes, where it agrees and where it conflicts",
        ("stance", "theoretical_framework", "key_results", "agreement", "contradiction", "limitations"),
        "theme clustering plus agreement, contradiction and extension-chain analysis",
        ("Scope and Question", "Thematic Map", "Thematic Findings", "Cross-cutting Methodological Patterns", "Consensus", "Contention", "Literature Limitations", "Open Questions", "References"),
    ),
    "state_of_art": ReportBlueprint(
        "state_of_art",
        "field cartography across approach families, chronology, benchmarks and current bottlenecks",
        ("timeline", "approach_family", "benchmark", "evaluation_protocol", "bottlenecks"),
        "thematic clustering plus taxonomy and chronological synthesis",
        ("Field Definition", "Approach Taxonomy", "Historical Evolution", "Current Paradigms", "Evaluation Landscape", "Bottlenecks", "Emerging Directions", "Outlook", "References"),
    ),
    "gap_analysis": ReportBlueprint(
        "gap_analysis",
        "confidence-stratified identification of established, contested and unexplored areas",
        ("author_gaps", "implicit_gaps", "evidence_strength", "assumptions", "failure_boundaries"),
        "confidence stratification with explicit, inferred, contradiction, evaluation and methodological gap classes",
        ("Question and Why It Matters", "Established Findings", "Contested Findings", "Unexplored Areas", "Gap Catalog", "Why Gaps Persist", "Priority Research Questions", "References"),
    ),
    "comparison": ReportBlueprint(
        "comparison",
        "apples-to-apples methodological comparison with explicit comparability boundaries",
        ("dataset", "split", "task_formulation", "metrics", "baselines", "compute", "robustness"),
        "comparability alignment first, then mechanism, performance and trade-off analysis",
        ("Methods Under Review", "Comparability Matrix", "Mechanism-level Comparison", "Performance Comparison", "Trade-offs and Robustness", "Where Comparison Breaks Down", "References"),
    ),
    "evidence_review": ReportBlueprint(
        "evidence_review",
        "process-transparent evidence synthesis with study quality and uncertainty",
        ("study_design", "sample_or_dataset", "controls", "bias_indicators", "replication", "uncertainty"),
        "study characterization, quality grading, consistency and evidence certainty synthesis",
        ("Question", "Screening and Inclusion", "Study Characteristics", "Evidence Quality", "Consistency and Heterogeneity", "Synthesized Findings", "What Evidence Can and Cannot Support", "Uncertainty and Caveats", "References"),
    ),
    "research_proposal": ReportBlueprint(
        "research_proposal",
        "forward-looking gap-to-question-to-method design grounded in existing evidence",
        ("established_baselines", "evaluation_protocols", "gap", "confounders", "ablations", "risks"),
        "gap-to-research-question chain followed by baseline and evaluation design synthesis",
        ("Problem Motivation", "Gap Justification", "Research Questions and Hypotheses", "Proposed Approach", "Baselines", "Evaluation Plan", "Risks and Limitations", "Expected Contribution", "References"),
    ),
    "paper_analysis": ReportBlueprint(
        "paper_analysis",
        "paper-preserving deep analysis of mechanism, evidence, ablations, failure boundaries and reproducibility",
        ("mechanism", "ablations", "failure_boundaries", "reproducibility", "claim_ledger", "experiment_records"),
        "paper-preserving deep dives followed by cross-paper roll-up",
        ("Corpus Overview", "Paper Deep Dives", "Cross-paper Synthesis", "Reproducibility Scorecard", "Consolidated Takeaways", "References"),
    ),
}
def get_report_blueprint(report_type: str) -> ReportBlueprint:
    canonical = canonical_report_type(report_type)
    return _REPORT_BLUEPRINTS.get(canonical, _REPORT_BLUEPRINTS["literature_review"])
VALID_REPORT_TYPES = {
    "standard_report", "literature_review", "comparison", "state_of_art", "gap_analysis",
    "evidence_review", "research_proposal", "paper_analysis",
}
REPORT_TYPE_LABELS = {
    "standard_report": "Standard Report",
    "literature_review": "Literature Review",
    "comparison": "Comparative Analysis",
    "state_of_art": "State of the Art / Research Landscape",
    "gap_analysis": "Research Gap Analysis",
    "evidence_review": "Systematic / Evidence Review",
    "research_proposal": "Research Design / Proposal",
    "paper_analysis": "Paper / Corpus Analysis",
}
def report_type_for_question(question: str) -> str:
    """Map a raw question to a sensible default report type."""
    mode = _classify_mode(question)
    return {
        "comparison": "comparison",
        "landscape": "state_of_art",
        "explanation": "paper_analysis",
        "design": "research_proposal",
        "literature_review": "literature_review",
    }[mode]
def suggest_report_title(question: str, report_type: str = "literature_review") -> str:
    """Create a concise academic title without copying a question verbatim."""
    q = " ".join((question or "").strip().rstrip("?").split())
    if not q:
        return "Research Report"
    replacements = [
        ("what are recent approaches to ", ""), ("what are the recent approaches to ", ""),
        ("compare ", ""), ("compare the ", ""), ("how does ", ""), ("how do ", ""), ("why does ", ""), ("why do ", ""),
        ("does ", ""), ("can ", ""), ("what is ", ""), ("what are ", ""),
    ]
    lower = q.lower()
    for prefix, replacement in replacements:
        if lower.startswith(prefix):
            q = q[len(prefix):]
            break
    q = q[:1].upper() + q[1:]
    suffix = {
        "standard_report": "", "literature_review": "", "comparison": "", "state_of_art": "",
        "gap_analysis": "", "evidence_review": "",
        "research_proposal": "", "paper_analysis": "",
    }.get(report_type, "")
    title = q[:110].rstrip(" .")
    return title + suffix
_SEARCH_SUFFIXES_BY_REPORT_TYPE = {
    "standard_report": ["comprehensive literature review", "methods technical evidence evaluation", "comparison limitations research gaps"],
    "literature_review": ["survey literature review", "methods datasets evaluation", "limitations open problems"],
    "state_of_art": ["state of the art recent advances", "taxonomy benchmark comparison", "limitations bottlenecks open problems"],
    "gap_analysis": ["research gaps limitations", "underexplored unresolved problems", "benchmark limitations evaluation gaps", "future work open questions"],
    "comparison": ["comparative study benchmark", "head-to-head evaluation ablation", "tradeoffs efficiency robustness"],
    "evidence_review": ["systematic review evidence", "study design outcomes limitations", "evidence synthesis quality bias"],
    "research_proposal": ["research gap proposed method", "baseline ablation evaluation", "open problem future research"],
    "paper_analysis": ["paper method experiments ablation", "paper limitations robustness reproducibility", "paper results discussion failure modes"],
}
def _heuristic_plan(question: str, report_type: str | None = None, report_title: str = "") -> ResearchPlan:
    mode = _classify_mode(question)
    report_type = canonical_report_type(report_type or report_type_for_question(question))
    report_type = report_type if report_type in VALID_REPORT_TYPES else report_type_for_question(question)
    mode = {
        "state_of_art": "landscape", "gap_analysis": "landscape",
        "research_proposal": "design", "comparison": "comparison",
        "evidence_review": "literature_review", "paper_analysis": "literature_review",
    }.get(report_type, mode)
    suffixes = _SEARCH_SUFFIXES_BY_REPORT_TYPE.get(report_type, _SEARCH_SUFFIXES_BY_REPORT_TYPE["literature_review"])
    queries = [question] + [f"{question} {suffix}" for suffix in suffixes[:4]]
    queries.append(f"{question} 2024 2025 2026 recent advances")
    return ResearchPlan(
        question=question,
        mode=mode,
        report_type=report_type,
        report_title=report_title,
        subquestions=list(_SUBQUESTIONS_BY_REPORT_TYPE.get(report_type, [
            "What is the best-supported answer given available evidence?",
            "Which methods or findings matter most?",
            "Where do sources agree or disagree?",
            "What gaps remain?",
        ])),
        search_queries=queries,
        analysis_axes=list(_AXES_BY_REPORT_TYPE.get(report_type, _AXES_BY_MODE[mode])),
    )
def make_plan(question: str, llm: LLMClient | None = None, report_type: str | None = None, report_title: str = "") -> tuple[ResearchPlan, bool]:
    """Returns (plan, llm_was_used)."""
    heuristic = _heuristic_plan(question, report_type=report_type, report_title=report_title)
    llm = llm or LLMClient()
    if not llm.available:
        return heuristic, False
    try:
        data = llm.complete_json(_SYSTEM_PROMPT, f"QUESTION: {question}\nREPORT TYPE: {heuristic.report_type}\nREQUIRED ANALYSIS AXES: {heuristic.analysis_axes}", max_tokens=900, priority="retrieval", tier="F")
    except Exception:
        return heuristic, False
    if not isinstance(data, dict):
        return heuristic, False
    try:
        merged = ResearchPlan(
            question=question,
            mode=data.get("mode", heuristic.mode),
            report_type=report_type or heuristic.report_type,
            report_title=report_title or heuristic.report_title,
            subquestions=list(dict.fromkeys(list(data.get("subquestions") or []) + heuristic.subquestions))[:8],
            search_queries=list(dict.fromkeys(list(data.get("search_queries") or []) + heuristic.search_queries))[:10] or heuristic.search_queries,
            analysis_axes=list(dict.fromkeys(heuristic.analysis_axes + list(data.get("analysis_axes") or [])))[:12],
        )
        return merged, True
    except ValidationError:
        return heuristic, False
