from __future__ import annotations
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
from ..llm import LLMClient
FAST_COUNCIL_ROLES = {
    "evidence_skeptic": """Interrogate the supplied paper dossiers for scientific overclaiming. Identify what is directly demonstrated, what is author interpretation, what is inference, important caveats, missing controls, benchmark dependence, invalid comparisons and evidence boundaries. Focus on the research question. Return compact JSON with: strongest_evidence, cautions, unsupported_or_weak_claims, evidence_gaps, priority_questions.""",
    "methodologist": """Act as a senior experimental methodologist. Compare datasets, populations, tasks, models, baselines, metrics, splits, ablations, protocols and deployment conditions. Explain which results can and cannot be compared and why. Identify methodological patterns that explain agreement or disagreement. Return compact JSON with: comparable_results, non_comparable_results, methodological_patterns, confounders, priority_questions.""",
    "mechanism_analyst": """Act as a mechanism and systems scientist. Determine how the important approaches work, why their components should matter, and which mechanisms are actually supported by experiments or ablations. Separate demonstrated mechanisms from plausible interpretations. Identify important quantitative or implementation details. Return compact JSON with: mechanisms, supporting_evidence, alternative_explanations, important_details, priority_questions.""",
    "boundary_inquiry": """Act as the research-inquiry specialist. Derive concrete unresolved questions from limitations, failure cases, missing evaluations, contradictions, boundary conditions, untested assumptions and reproducibility issues. Distinguish author-stated open questions from gaps discovered by cross-paper comparison. For each major gap state what evidence or experiment could resolve it. Return compact JSON with: established_boundaries, unresolved_questions, evidence_backed_gaps, author_open_questions, proposed_tests, priority_questions.""",
}
INDEPENDENT_SYNTHESIS = """You are an independent senior researcher performing a cross-paper synthesis before seeing any other reviewers' conclusions.
Build the intellectual map of the supplied evidence for the research question. Find consensus, meaningful tensions, complementary contributions, conditional findings, important comparisons, surprising connections and the strongest answer the evidence permits. Do not write polished report prose. Preserve source tags and distinguish evidence from interpretation. Return compact JSON with: central_answer, conceptual_model, themes, cross_paper_relationships, consensus, disagreements, conditional_findings, evidence_gaps, high_value_details, comparison_dimensions, report_requirements, conclusion_logic."""
CHIEF_PROMPT = """You are the chief research scientist and Feynman-style lead author.
You receive the immutable paper dossiers plus independent parallel council reviews and an independent cross-paper synthesis. Your job is to perform the ONE decisive reasoning pass that turns these views into a coherent research argument and report blueprint.
Think like an exceptional scientist teaching another scientist: first establish the simple mental model, then mechanism, then evidence, then interpretation, then limits and unresolved boundaries. Do not merely concatenate the council outputs. Resolve disagreements only when the evidence supports resolution; otherwise preserve the condition that explains the disagreement.
The blueprint must be evidence-led and report-type aware. It must produce an ADAPTIVE set of substantive sections, normally 4-7, chosen from what the deep-read evidence actually contains. Do not force a fixed template. Never create a section merely because a report type usually contains it. Never produce a paper-summary list unless the requested report type is paper analysis or the evidence genuinely requires paper-preserving treatment. The architecture should decide which of the following components are useful and how they should be ordered: conceptual foundation, problem framing, approach taxonomy, mechanism deep dive, design rationale, experimental evidence, quantitative comparison, ablations and causal evidence, robustness/generalization, disagreements and alternative explanations, limitations/boundaries, research gaps, implications, reproducibility, deployment/efficiency trade-offs, or research directions. For each selected section explain why it belongs in the report and what evidence it must contain. At least one section must explicitly address unresolved boundaries for report types where gaps are scientifically relevant. Include an executive summary, key takeaways and open questions in the JSON so no extra conclusion call is needed.
For every major claim in the blueprint, keep source tags. Do not invent facts, numbers, papers or experiments.
Return ONLY JSON:
{
  "central_answer": str,
  "executive_summary": str,
  "key_takeaways": [str],
  "open_questions": [str],
  "answer_boundary": {"supported":[],"not_supported":[],"needed_evidence":[]},
  "themes": [],
  "cross_paper_relationships": [],
  "comparison_dimensions": [],
  "gaps": [{"gap":str,"source_tags":[],"evidence_basis":str,"why_it_matters":str,"resolving_test":str}],
  "title": str,
  "topics": [{"title":str,"goal":str,"role":str,"section_type":str,"why_in_report":str,"required_evidence":[],"source_tags":[],"subtopics":[{"title":str,"what_to_explain":str,"evidence_focus":[]}]}]
}
"""
def _safe_json_call(llm: LLMClient, system: str, payload: dict, max_tokens: int, tier: str = "F") -> dict:
    if not llm or not llm.available:
        return {}
    try:
        result = llm.complete_json(system, json.dumps(payload, ensure_ascii=False), temperature=0.12, max_tokens=max_tokens, priority="report", tier=tier)
        return result if isinstance(result, dict) else {}
    except TypeError:
        try:
            result = llm.complete_json(system, json.dumps(payload, ensure_ascii=False), temperature=0.12, max_tokens=max_tokens)
            return result if isinstance(result, dict) else {}
        except Exception:
            return {}
    except Exception:
        return {}
def run_parallel_council(question: str, report_type: str, cards: dict[str, str], analyses: dict[str, Any], llm: LLMClient) -> tuple[dict, bool]:
    """Run four independent council lenses and one independent synthesis concurrently."""
    if not llm or not llm.available or not cards:
        return {}, False
    dossier = {
        "research_question": question,
        "report_type": report_type,
        "papers": cards,
        "analysis_memory": {
            tag: {
                "question_answer": getattr(a, "question_answer", "") if a else "",
                "established": getattr(a, "established", "") if a else "",
                "not_established": getattr(a, "not_established", "") if a else "",
                "limitations": getattr(a, "limitations_author_stated", "") if a else "",
                "open_questions": getattr(a, "open_questions", "") if a else "",
            }
            for tag, a in analyses.items()
        },
    }
    jobs = [(name, prompt) for name, prompt in FAST_COUNCIL_ROLES.items()]
    jobs.append(("independent_synthesis", INDEPENDENT_SYNTHESIS))
    outputs: dict[str, dict] = {}
    workers = min(5, len(jobs))
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="researchpilot-fast-council") as pool:
        futures = {pool.submit(_safe_json_call, llm, prompt, dossier, 2600, "F"): name for name, prompt in jobs}
        for future in as_completed(futures):
            name = futures[future]
            try:
                value = future.result()
                if value:
                    outputs[name] = value
            except Exception:
                outputs[name] = {}
    return outputs, bool(outputs)
def run_chief_synthesis(question: str, report_type: str, cards: dict[str, str], council: dict, llm: LLMClient) -> tuple[dict, bool]:
    """One strong join pass over all independent views."""
    if not llm or not llm.available or not cards:
        return {}, False
    payload = {
        "research_question": question,
        "report_type": report_type,
        "paper_dossiers": cards,
        "parallel_council": council,
    }
    result = _safe_json_call(llm, CHIEF_PROMPT, payload, 5600, "M")
    return result, bool(result.get("topics") or result.get("central_answer"))
COMPACT_COUNCIL_PROMPT = """You are ResearchOrbit's chief professor-level research council and lead author. Perform the council's evidence, methodology, mechanism, alternative-explanation and research-gap reasoning IN ONE decisive pass over the supplied immutable paper dossiers. The output is a research architecture, not a short summary.
For each paper, internally test: what is directly demonstrated, how the method works, whether the experimental design supports the claimed mechanism, what can actually be compared, and what remains unresolved. Then synthesize across papers. Do not produce paper-by-paper summaries. Preserve disagreements when evidence does not resolve them. Derive research gaps only from concrete limitations, missing evaluations, contradictions, boundary conditions, untested assumptions or evidence failures. For each gap state why it matters and what evidence/experiment would resolve it.
Use a Feynman progression: simple mental model -> mechanism -> evidence -> interpretation -> boundary. Keep source tags on every substantive claim. Return ONLY JSON:
{
  "central_answer": str,
  "executive_summary": str,
  "key_takeaways": [str],
  "open_questions": [str],
  "answer_boundary": {"supported":[],"not_supported":[],"needed_evidence":[]},
  "themes": [],
  "cross_paper_relationships": [],
  "comparison_dimensions": [],
  "gaps": [{"gap":str,"source_tags":[],"evidence_basis":str,"why_it_matters":str,"resolving_test":str}],
  "title": str,
  "topics": [{"title":str,"goal":str,"role":str,"section_type":str,"why_in_report":str,"required_evidence":[],"source_tags":[],"subtopics":[{"title":str,"what_to_explain":str,"evidence_focus":[]}]}]
}
"""
def run_compact_professor_council(question: str, report_type: str, cards: dict[str, str], llm: LLMClient) -> tuple[dict, bool]:
    """Single high-capacity council pass used by the v84 bounded hot path.
    It replaces five fan-out council calls plus a chief join. The intellectual
    roles are still explicitly present in the prompt, but the provider sees the
    corpus once and returns one joined research blueprint.
    """
    if not llm or not llm.available or not cards:
        return {}, False
    payload = {"research_question": question, "report_type": report_type, "paper_dossiers": cards}
    result = _safe_json_call(llm, COMPACT_COUNCIL_PROMPT, payload, int(getattr(getattr(llm, "settings", None), "performance_council_max_tokens", 1800) or 1800), "M")
    return result, bool(result.get("topics") or result.get("central_answer"))
