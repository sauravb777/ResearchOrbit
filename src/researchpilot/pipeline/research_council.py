from __future__ import annotations
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
from ..llm import LLMClient
ROLE_PROMPTS = {
    "evidence_skeptic": """You are the evidence skeptic on a senior academic research council.
Do not write a report. Interrogate the supplied cross-paper evidence for overclaiming.
Identify which findings are directly demonstrated, which are only author-reported,
which are synthesis/inference, which are genuinely contested, and which conclusions
would require evidence that the corpus does not contain. Look for hidden conditions,
benchmark dependence, missing controls, weak baselines, selective reporting and
numbers that are not comparable. Preserve uncertainty rather than resolving it by
intuition. Return JSON only with findings, cautions, required_evidence, and priority_questions.
""",
    "methodologist": """You are a senior experimental methodologist on a research council.
Do not write a report. Examine how the papers were tested and whether their results
can legitimately be compared or synthesized. Track datasets, splits, populations,
models, baselines, metrics, evaluation protocols, ablations, uncertainty, compute
and failure conditions. Identify apples-to-oranges comparisons and explain why they
break. Also identify methodological patterns that explain apparent disagreement.
Return JSON only with valid_comparisons, invalid_comparisons, methodological_patterns,
confounders, and priority_questions.
""",
    "field_synthesizer": """You are a senior professor synthesizing a literature corpus.
Do not produce polished prose. Determine the intellectual structure of the field from
the supplied evidence. Find themes that emerge from the papers rather than from a
predefined template, extensions, competing explanations, consensus, disagreement,
underexplored areas, and surprising connections. Ask what the corpus collectively
teaches that no single paper establishes. Keep every proposed synthesis tied to
source/evidence IDs. Return JSON only with themes, consensus, tensions, extensions,
gaps, surprising_connections, and priority_questions.
""",
    "contradiction_hunter": """You are a contradiction hunter on a research council. Find apparent disagreements, negative results, or incompatible claims across the supplied papers. Do not assume the literature is contradictory merely because numbers differ. Check task, dataset, split, population, model, metric, protocol, and conditions. Return JSON only with contradictions, condition_explanations, unresolved_disagreements, and priority_questions.""",
    "gap_strategist": """You are a research-gap strategist. Identify only gaps that are justified by the supplied evidence, not generic future-work boilerplate. Separate absence of evidence from evidence of absence. For each gap explain what evidence would reduce the uncertainty and whether the gap is central or peripheral. Return JSON only with gaps, evidence_needed, gap_type, and priority_questions.""",
    "mechanism_analyst": """You are a mechanism analyst. Explain which components, design choices, interventions, or causal mechanisms are actually supported by experiments or ablations. Separate demonstrated mechanism from author interpretation and plausible inference. Return JSON only with mechanisms, supporting_evidence, alternative_explanations, unsupported_mechanisms, and priority_questions.""",
    "transfer_analyst": """You are a transfer and boundary-condition analyst. Examine where findings have and have not been tested, including domain, population, dataset, distribution, scale, task formulation, and deployment conditions. Do not extrapolate beyond evidence. Return JSON only with supported_transfer, boundary_conditions, untested_transfer, risks, and priority_questions.""",
    "reproducibility_auditor": """You are a reproducibility auditor. Inspect the corpus for data availability, implementation detail, training/evaluation protocol, randomization, ablations, compute, dependencies, and missing information that affects reproducibility. Do not score papers by prestige. Return JSON only with reproducibility_strengths, risks, missing_details, and priority_questions.""",
    "proposal_designer": """You are a research-design specialist. Only when the question is forward-looking, convert evidence-supported gaps into research questions, hypotheses, baselines, controls, and evaluation designs. Do not invent a contribution that the evidence does not motivate. Return JSON only with candidate_questions, hypotheses, baselines, evaluations, risks, and rationale.""",
}
CHAIR_PROMPT = """You are the chair of a senior academic research council.
You receive independent reviews from a dynamically selected internal research team plus the underlying paper memories. Your task is to construct the
research argument, not the final prose.
Build an evidence-led hierarchy for the user's question. Decide which findings are
central, supporting, contextual, contested, or should be excluded. Resolve only
where the evidence justifies resolution. Otherwise preserve disagreement and explain
its likely source. Create a flexible report architecture whose sections emerge from
the evidence. A report type is a style constraint, not a list of blanks to fill.
Every major proposed section must have evidence/source IDs and a reason it is needed.
Include an explicit 'answer boundary': what the corpus can answer, cannot answer,
and what evidence would change the conclusion.
Return JSON only:
{
  "central_thesis": "...",
  "answer_boundary": {"supported":[],"not_supported":[],"needed_evidence":[]},
  "argument_threads":[{"title":"...","question":"...","role":"central|supporting|contextual|contested","source_ids":[],"evidence_ids":[],"reason":"..."}],
  "themes":[{"title":"...","source_ids":[],"evidence_ids":[],"relationship":"consensus|tension|extension|gap|context","reason":"..."}],
  "comparison_rules":[],
  "contradictions":[{"issue":"...","source_ids":[],"evidence_ids":[],"possible_explanations":[]}],
  "gaps":[{"gap":"...","evidence_ids":[],"why_it_is_a_gap":"...","confidence":0.0}],
  "outline":[{"title":"...","purpose":"...","argument":"...","source_ids":[],"evidence_ids":[],"analytical_tasks":[],"must_answer":[]}],
  "excluded_material":[{"item":"...","reason":"..."}]
}
"""
def _call(llm: LLMClient, system: str, payload: Any, max_tokens: int = 3000) -> dict:
    if not llm or not llm.available:
        return {}
    try:
        result = llm.complete_json(system, json.dumps(payload, ensure_ascii=False), max_tokens=max_tokens, tier="S", priority="report")
        return result if isinstance(result, dict) else {}
    except TypeError:
        try:
            result = llm.complete_json(system, json.dumps(payload, ensure_ascii=False), max_tokens=max_tokens)
            return result if isinstance(result, dict) else {}
        except Exception:
            return {}
    except Exception:
        return {}
def _perspective_payload(question: str, report_type: str, analyses: list[Any], findings: list[Any]) -> dict:
    return {
        "question": question,
        "report_type": report_type,
        "paper_memories": [
            {
                "source_id": a.source_id,
                "title": getattr(a, "problem", ""),
                "problem": a.problem,
                "method": a.method_summary,
                "results": a.key_results,
                "established": a.established,
                "not_established": a.not_established,
                "limitations": a.limitations,
                "inventory": (getattr(a, "evidence_inventory", []) or [])[:80],
                "relevance": getattr(a, "research_relevance_map", {}) or {},
                "experiments": (getattr(a, "experiment_records", []) or [])[:30],
                "quantitative_results": [q.model_dump() for q in getattr(a, "quantitative_results", [])],
            }
            for a in analyses
        ],
        "cross_findings": [
            {"id": f.id, "type": f.finding_type, "text": f.text,
             "source_ids": f.source_ids, "evidence_ids": f.evidence_ids, "caveat": f.caveat}
            for f in findings
        ],
    }
def run_research_council(question: str, report_type: str, analyses: list[Any], findings: list[Any], llm: LLMClient) -> tuple[dict, bool]:
    """Run three independent senior perspectives, then a chair synthesis.
    The perspectives are intentionally independent. They do not share one
    model's conclusions, which reduces premature convergence and makes the
    final architecture a reasoned join rather than a single chain of thought.
    """
    if not llm.available or not analyses:
        return {}, False
    payload = _perspective_payload(question, report_type, analyses, findings)
    legacy_roles = ("evidence_skeptic", "methodologist", "field_synthesizer")
    roles = [(name, ROLE_PROMPTS[name]) for name in legacy_roles]
    outputs: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=min(3, len(roles)), thread_name_prefix="researchpilot-council") as pool:
        futures = {pool.submit(_call, llm, prompt, payload, 3000): name for name, prompt in roles}
        for future in as_completed(futures):
            name = futures[future]
            try:
                outputs[name] = future.result()
            except Exception:
                outputs[name] = {}
    chair_payload = {**payload, "independent_council_reviews": outputs}
    chair = _call(llm, CHAIR_PROMPT, chair_payload, 4200)
    if not chair:
        return {"perspectives": outputs}, bool(outputs)
    chair["perspectives"] = outputs
    chair["architecture_version"] = "professor_council_v1"
    return chair, True
def runnable_role(role: str, llm: LLMClient):
    """Return a LangChain Core RunnableLambda for a council role when installed."""
    prompt = ROLE_PROMPTS[role]
    try:
        from langchain_core.runnables import RunnableLambda
        return RunnableLambda(lambda payload: _call(llm, prompt, payload, 3000))
    except Exception:
        return lambda payload: _call(llm, prompt, payload, 3000)
def run_council_role(role: str, payload: dict, llm: LLMClient) -> dict:
    runnable = runnable_role(role, llm)
    try:
        result = runnable.invoke(payload) if hasattr(runnable, "invoke") else runnable(payload)
        return result if isinstance(result, dict) else {}
    except Exception:
        return {}
def run_council_chair(payload: dict, llm: LLMClient) -> dict:
    try:
        from langchain_core.runnables import RunnableLambda
        result = RunnableLambda(lambda x: _call(llm, CHAIR_PROMPT, x, 4200)).invoke(payload)
        return result if isinstance(result, dict) else {}
    except Exception:
        return _call(llm, CHAIR_PROMPT, payload, 4200)
