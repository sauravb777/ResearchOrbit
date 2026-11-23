from __future__ import annotations
from researchpilot.pipeline.deep_read import _quote_is_verified
from researchpilot.pipeline.synthesizer import build_report
from researchpilot.pipeline.planner import _heuristic_plan
from researchpilot.schemas import Evidence, PaperAnalysis, Source
def test_quote_verification_rejects_a_quote_from_a_different_paper():
    text_a = "Our method improves accuracy by 15.4% over the baseline on the QA benchmark."
    quote_from_b = "Our approach reduces hallucination by 23.7% compared to standard GPT-3.5."
    assert _quote_is_verified("Our method improves accuracy by 15.4%", text_a) is True
    assert _quote_is_verified(quote_from_b, text_a) is False
def test_report_never_mixes_one_papers_evidence_into_another_papers_deep_analysis():
    """Two synthetic core papers with intentionally different, specific
    quantitative results. Paper A's subsection in 'Deep Analysis of Core
    Papers' must contain ONLY Paper A's number, never Paper B's -- this is
    the exact failure mode named in the spec ("Paper A reports 15.4%...
    the final synthesis must never attribute 15.4% to Paper B")."""
    source_a = Source(
        id="a", title="Paper A: RAG-RLHF", authors="A. One", year="2025", venue="X",
        abstract="RAG-RLHF improves factual accuracy by 15.4% over standard GPT-3.5.",
        full_text="RAG-RLHF improves factual accuracy by 15.4% over standard GPT-3.5.",
        provider="fixed", role="deep_read",
    )
    source_b = Source(
        id="b", title="Paper B: Self-Reflective RAG", authors="B. Two", year="2025", venue="Y",
        abstract="Self-reflective RAG reduces hallucination to 5.8% on patient vignettes.",
        full_text="Self-reflective RAG reduces hallucination to 5.8% on patient vignettes.",
        provider="fixed", role="deep_read",
    )
    sources = [source_a, source_b]
    evidence = [
        Evidence(id="E1", source_id="a", text="RAG-RLHF improves factual accuracy by 15.4%",
                  locator="full text, deep-read", kind="result", relevance=0.9),
        Evidence(id="E2", source_id="b", text="Self-reflective RAG reduces hallucination to 5.8%",
                  locator="full text, deep-read", kind="result", relevance=0.9),
    ]
    analyses = [
        PaperAnalysis(source_id="a", depth="full_text", core_idea="RAG-RLHF", key_results="Improves accuracy by 15.4%."),
        PaperAnalysis(source_id="b", depth="full_text", core_idea="Self-reflective RAG", key_results="Reduces hallucination to 5.8%."),
    ]
    plan = _heuristic_plan("What are recent approaches to reducing hallucination in RAG?")
    report, _ = build_report(plan.question, plan, sources, evidence, claims=[], warnings=[], analyses=analyses)
    section_a_start = report.index("### S1 -- Paper A: RAG-RLHF")
    section_b_start = report.index("### S2 -- Paper B: Self-Reflective RAG")
    comparison_start = report.index("## Comparison of Sources")
    if section_a_start < section_b_start:
        section_a = report[section_a_start:section_b_start]
        section_b = report[section_b_start:comparison_start]
    else:
        section_b = report[section_b_start:section_a_start]
        section_a = report[section_a_start:comparison_start]
    assert "15.4%" in section_a and "5.8%" not in section_a
    assert "5.8%" in section_b and "15.4%" not in section_b
def test_supporting_paper_evidence_never_appears_in_core_deep_analysis_section():
    core = Source(id="c1", title="Core Paper", provider="fixed", role="deep_read",
                  abstract="Core paper abstract about method X improving results.",
                  full_text="Core paper abstract about method X improving results.")
    supporting = Source(id="s1", title="Supporting Paper", provider="fixed", role="no_read",
                        abstract="A supporting paper mentioning an unrelated statistic of 42%.")
    sources = [core, supporting]
    evidence = [
        Evidence(id="E1", source_id="c1", text="method X improving results", locator="full text, deep-read", kind="method", relevance=0.9),
        Evidence(id="E2", source_id="s1", text="unrelated statistic of 42%", locator="abstract", kind="context", relevance=0.5),
    ]
    analyses = [PaperAnalysis(source_id="c1", depth="full_text", core_idea="Method X", key_results="Improves results.")]
    plan = _heuristic_plan("What methods exist for X?")
    report, _ = build_report(plan.question, plan, sources, evidence, claims=[], warnings=[], analyses=analyses)
    deep_section_start = report.index("## Deep Analysis of Core Papers")
    deep_section = report[deep_section_start:]
    assert "42%" not in deep_section
    assert "Supporting Literature" not in report
