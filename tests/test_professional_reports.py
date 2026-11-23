from researchpilot.pipeline.synthesizer import build_professional_report, build_evidence_audit_report
from researchpilot.pipeline.planner import _heuristic_plan
from researchpilot.schemas import Evidence, ResearchResult, Source
def _case(report_type):
    plan = _heuristic_plan("Compare retrieval methods for question answering", report_type=report_type, report_title="Retrieval Methods")
    sources = [
        Source(id="a", title="Method A", provider="fixed", role="deep_read", abstract="Method A improves QA accuracy by 10%.", full_text="Method A improves QA accuracy by 10%.", score=.9),
        Source(id="b", title="Method B", provider="fixed", role="no_read", abstract="Method B reports lower latency.", full_text="Method B reports lower latency.", score=.6),
    ]
    evidence = [
        Evidence(id="E1", source_id="a", text="Method A improves QA accuracy by 10%.", locator="full text", kind="result", relevance=.9),
        Evidence(id="E2", source_id="b", text="Method B reports lower latency.", locator="full text", kind="result", relevance=.7),
    ]
    return plan, sources, evidence
def test_professional_report_is_clean_and_type_specific():
    plan, sources, evidence = _case("comparison")
    report, _ = build_professional_report("Compare retrieval methods for question answering", "Retrieval Methods", "comparison", plan, sources, evidence, [], [])
    assert report.startswith("# Retrieval Methods")
    assert "## Executive Summary" in report
    assert "## Comparison of Approaches" in report
    assert "## Evidence Base" not in report
    assert "[S1]" in report
def test_audit_is_separate_from_clean_report():
    plan, sources, evidence = _case("literature_review")
    report, _ = build_professional_report(plan.question, "Retrieval Methods", "literature_review", plan, sources, evidence, [], [])
    result = ResearchResult(run_id="r1", question=plan.question, report_title="Retrieval Methods", report_type="literature_review", plan=plan, sources=sources, evidence=evidence, report_markdown=report)
    audit = build_evidence_audit_report(result)
    assert "## Source Register" in audit
    assert "## Evidence Register" in audit
    assert "## Evidence Base" not in report
import pytest
@pytest.mark.parametrize("report_type", [
    "literature_review", "comparison", "state_of_art",
    "research_proposal", "evidence_review", "paper_analysis",
])
def test_every_report_type_renders_without_error(report_type):
    plan, sources, evidence = _case(report_type)
    report, _ = build_professional_report(plan.question, "Retrieval Methods", report_type, plan, sources, evidence, [], [])
    assert report.startswith("# Retrieval Methods")
    assert "## Executive Summary" in report
    assert "## References" in report
def test_audit_includes_inspectable_verification_evidence():
    from researchpilot.schemas import ClaimVerification, Evidence, ResearchResult, Source
    plan = _heuristic_plan("Check whether retrieval improves QA", report_type="evidence_review", report_title="Verification Audit")
    source = Source(id="s1", title="Core Study", role="deep_read", abstract="retrieval improves QA")
    evidence = [Evidence(id="E1", source_id="s1", text="Retrieval improves QA.", kind="result", relevance=.9)]
    result = ResearchResult(
        run_id="r1", question=plan.question, report_title="Verification Audit", report_type="evidence_review",
        plan=plan, sources=[source], evidence=evidence, report_markdown="# Verification Audit",
        verifications=[ClaimVerification(
            claim="Retrieval improves QA", verdict="partially_supported", reasoning="Checked [V1].",
            evidence_ids=["V1"], new_sources=[Source(id="v1", title="Verification Study", role="no_read", abstract="independent evidence")],
            new_evidence=[Evidence(id="V1", source_id="v1", text="Independent verification evidence.", locator="verification", kind="result")],
        )],
    )
    audit = build_evidence_audit_report(result)
    assert "## Verification Evidence Register" in audit
    assert "**V1**" in audit
    assert "Independent verification evidence." in audit
    assert "Verification Study" in audit
