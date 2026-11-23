from __future__ import annotations
import json
from typing import Any
from ..llm import LLMClient
from ..schemas import PaperAnalysis
def _call(llm: LLMClient, system: str, payload: Any, max_tokens: int):
    if not llm or not llm.available:
        return None
    try:
        return llm.complete_json(system, json.dumps(payload, ensure_ascii=False), max_tokens=max_tokens, tier="S", priority="report")
    except TypeError:
        try:
            return llm.complete_json(system, json.dumps(payload, ensure_ascii=False), max_tokens=max_tokens)
        except Exception:
            return None
    except Exception:
        return None
QUESTION_MAP_SYSTEM = """You are the second reader of an academic paper. The first reader has already
read the supplied paper text and created an intentionally broad research memory. Your task is NOT to
summarize the paper and NOT to fill a report template. Instead, interrogate the memory against the
actual research question.
Find: (1) observations directly useful for answering the question, (2) observations that become important
only when combined with another paper, (3) evidence that limits or qualifies an apparent answer, (4) surprising
or non-obvious details a template-driven reader could miss, (5) evidence that is absent and therefore should not
be claimed, and (6) useful questions this paper raises. Then build a compact research-memory layer: the paper's
actual answer to the question, the strongest evidence chain from design/method to result to conclusion,
decision-relevant insights, method trade-offs, replication requirements and research implications.
Preserve the distinction between what the paper demonstrates, what its authors claim, and what you infer.
Do not invent facts. Return JSON only with:
{"direct_answer":[],"supporting_observations":[],"qualifiers":[],"cross_paper_hooks":[],"surprises":[],"not_established":[],"open_questions":[],"importance_rationale":"...","question_answer":"...","key_evidence_chain":[],"decision_relevant_insights":[],"method_tradeoffs":[],"replication_requirements":[],"research_implications":[]}
Observation items should contain {"text":"...","evidence_ids":[],"importance":0.0,"reason":"..."}.
Evidence-chain items should contain {"step":"...","evidence_quote_or_id":"...","why_it_supports":"..."}.
Decision-insight and trade-off items should contain {"text":"...","importance":0.0,"evidence_quote_or_id":"..."}."""
def map_paper_to_question(question: str, analysis: PaperAnalysis, llm: LLMClient) -> tuple[dict, bool]:
    inventory = getattr(analysis, "evidence_inventory", []) or []
    payload = {
        "question": question,
        "paper": {
            "source_id": analysis.source_id,
            "problem": analysis.problem,
            "objective": analysis.research_objective,
            "method": analysis.method_summary,
            "mechanism": analysis.mechanism,
            "experiments": analysis.experiment_records[:30],
            "claims": analysis.claim_ledger[:30],
            "results": analysis.key_results,
            "quantitative_results": [q.model_dump() for q in analysis.quantitative_results],
            "limitations": analysis.limitations,
            "established": analysis.established,
            "not_established": analysis.not_established,
            "failure_modes": analysis.failure_modes,
            "generalization": analysis.generalization,
            "transfer_conditions": analysis.transfer_conditions,
            "negative_results": analysis.negative_results,
            "conflicts": analysis.evidence_conflicts,
            "inventory": inventory[:80],
            "unmapped_observations": analysis.unmapped_observations[:50],
        },
    }
    data = _call(llm, QUESTION_MAP_SYSTEM, payload, 2800)
    if isinstance(data, dict):
        data["relevance_stage"] = "question_aware_second_reader"
        return data, True
    return {"relevance_stage": "unavailable"}, False
CROSS_DEBATE_SYSTEM = """You are a senior academic researcher conducting a cross-paper debate before a report is
written. Do not produce polished prose. Challenge the apparent synthesis.
For every important theme ask: what evidence supports it, what evidence qualifies it, are the studies
actually comparable, is there a plausible alternative explanation, and what would a skeptical professor
object to? Identify consensus, conditional consensus, disagreement, contradiction, extension, evidence
gaps, and apparent contradictions caused by different conditions. Give special attention to observations
that were not part of standard paper summaries.
Return JSON only: {"themes":[],"contradictions":[],"comparability_warnings":[],"missing_evidence":[],"alternative_interpretations":[],"high_value_insights":[]}. Each item must contain text, source_ids, evidence_ids, confidence, and rationale."""
def cross_paper_debate(question: str, report_type: str, analyses: list[PaperAnalysis], findings: list | None, llm: LLMClient) -> tuple[dict, bool]:
    payload = {
        "question": question,
        "report_type": report_type,
        "papers": [
            {
                "source_id": a.source_id,
                "core": a.core_idea,
                "results": a.key_results,
                "established": a.established,
                "not_established": a.not_established,
                "limitations": a.limitations,
                "experiment_records": a.experiment_records[:20],
                "claim_ledger": a.claim_ledger[:20],
                "evidence_inventory": a.evidence_inventory[:60],
                "relevance": a.research_relevance_map,
                "unmapped": a.unmapped_observations[:30],
                "question_answer": getattr(a, "question_answer", ""),
                "evidence_chain": getattr(a, "key_evidence_chain", [])[:20],
                "decision_insights": getattr(a, "decision_relevant_insights", [])[:30],
                "tradeoffs": getattr(a, "method_tradeoffs", [])[:20],
                "replication_requirements": getattr(a, "replication_requirements", [])[:20],
                "research_implications": getattr(a, "research_implications", [])[:20],
            }
            for a in analyses
        ],
        "existing_findings": [
            {"id": getattr(f, "id", ""), "text": getattr(f, "text", ""), "evidence_ids": getattr(f, "evidence_ids", [])}
            for f in (findings or [])
        ],
    }
    data = _call(llm, CROSS_DEBATE_SYSTEM, payload, 4200)
    if isinstance(data, dict):
        return data, True
    return {}, False
PEER_REVIEW_SYSTEM = """You are a demanding professor reviewing a research report draft before publication.
Judge the reasoning, not the prose polish. The report must genuinely answer the research question and
must not merely enumerate papers.
Look for: shallow paper summaries, template-filling, important evidence omitted, unique high-value knowledge
silently dropped, themes that are merely lists, unsupported causal language, numerical comparisons under
mismatched conditions, missing negative results, failure to distinguish author claims from demonstrated
findings, redundant sections, weak transitions, conclusions stronger than the corpus, unexplained
contradictions, and research gaps that are just generic future-work statements. Treat omission of
question-relevant knowledge as a substantive defect. Check whether the report preserves each paper's
mechanism, decisive experimental conditions, quantitative findings, controls/ablations, negative or
failure results, transfer boundaries, important trade-offs, reproducibility implications, surprising
findings, and research implications when those items exist in the supplied paper memory. The goal is to
compress repeated information, not silently discard unique high-value information.
Return JSON only:
{"overall_assessment":"...","critical_issues":[],"missing_insights":[],"weak_sections":[],"redundancies":[],"unsupported_or_overstated":[],"comparison_problems":[],"important_findings_not_integrated":[],"required_revisions":[]}
Each issue should contain {"section":"...","problem":"...","why_it_matters":"...","evidence_ids":[],"source_ids":[],"revision":"...","severity":"critical|major|minor"}."""
def peer_review_report(question: str, report_type: str, report_markdown: str, analyses: list[PaperAnalysis], synthesis_plan: Any, llm: LLMClient, council: dict | None = None) -> tuple[dict, bool]:
    council = council or {}
    payload = {
        "question": question,
        "report_type": report_type,
        "report": report_markdown,
        "professor_council": council,
        "planned_sections": [getattr(x, "title", "") if not isinstance(x, dict) else x.get("title", "") for x in getattr(synthesis_plan, "sections", [])],
        "paper_memory": [
            {
                "source_id": a.source_id,
                "relevance": a.research_relevance_map,
                "problem": a.problem,
                "objective": a.research_objective,
                "core_idea": a.core_idea,
                "mechanism": a.mechanism,
                "experimental_setup": a.experimental_setup,
                "datasets": a.dataset_details,
                "baselines": a.baselines,
                "metrics": a.metrics,
                "results": a.key_results,
                "quantitative_results": [q.model_dump() for q in (a.quantitative_results or [])[:20]],
                "ablations": a.ablations,
                "robustness": a.robustness,
                "computational_cost": a.computational_cost,
                "reproducibility": a.reproducibility,
                "surprises": [x for x in a.evidence_inventory if x.get("surprising_or_nonobvious")][:15],
                "high_value_inventory": [
                    x for x in a.evidence_inventory
                    if isinstance(x, dict) and (
                        float(x.get("importance", 0) or 0) >= 0.7
                        or x.get("surprising_or_nonobvious")
                        or x.get("decision_relevant")
                        or x.get("failure_mode")
                    )
                ][:35],
                "negative_results": a.negative_results,
                "not_established": a.not_established,
                "failure_modes": a.failure_modes,
                "transfer_conditions": a.transfer_conditions,
                "evidence_conflicts": a.evidence_conflicts,
                "decision_insights": a.decision_relevant_insights[:20],
                "tradeoffs": a.method_tradeoffs[:20],
                "replication_requirements": a.replication_requirements[:20],
                "research_implications": a.research_implications[:20],
                "open_questions": a.open_questions,
                "unmapped_observations": a.unmapped_observations[:20],
            }
            for a in analyses
        ],
    }
    data = _call(llm, PEER_REVIEW_SYSTEM, payload, 4200)
    if isinstance(data, dict):
        return data, True
    return {}, False
