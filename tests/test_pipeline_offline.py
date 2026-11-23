from __future__ import annotations
import re
from researchpilot.pipeline.orchestrator import ResearchAgent
from researchpilot.schemas import Source
from researchpilot.search.aggregator import SearchAggregator
from researchpilot.search.base import SourceProvider
_CITATION_RE = re.compile(r"\[(E\d+|S\d+)\]")
class _FakeProvider(SourceProvider):
    name = "arxiv"
    def __init__(self, sources: list[Source]):
        self._sources = sources
    def search(self, query: str, limit: int = 8) -> list[Source]:
        return list(self._sources)
def _fake_agent(sources: list[Source]) -> ResearchAgent:
    aggregator = SearchAggregator(providers=[_FakeProvider(sources)])
    return ResearchAgent(aggregator=aggregator)
def test_offline_pipeline_produces_a_complete_report(sample_sources):
    agent = _fake_agent(sample_sources)
    result = agent.run("What are recent approaches to retrieval augmented generation?")
    assert result.report_markdown.startswith("# ")
    assert "## Executive Summary" in result.report_markdown
    assert "## References" in result.report_markdown
    assert result.llm_used is False
    assert result.metrics["evidence_count"] == len(result.evidence)
    assert result.metrics["claim_count"] == len(result.claims)
    assert len(result.sources) > 0
    assert len(result.evidence) > 0
def test_offline_pipeline_reports_timing_and_usage_metrics(sample_sources):
    agent = _fake_agent(sample_sources)
    result = agent.run("What are recent approaches to retrieval augmented generation?")
    assert result.metrics["total_runtime_seconds"] >= 0
    assert "stage_seconds_plan" in result.metrics
    assert "stage_seconds_search" in result.metrics
    assert "stage_seconds_synthesize" in result.metrics
    assert result.metrics["llm_calls"] == 0
    assert result.metrics["estimated_cost_usd"] == 0.0
def test_offline_pipeline_has_no_dangling_citations(sample_sources):
    """Every [E#]/[S#] tag in the report must map to a real evidence or
    source id — the report and the audit trail must never diverge."""
    agent = _fake_agent(sample_sources)
    result = agent.run("Compare retrieval methods for question answering")
    valid_evidence_ids = {e.id for e in result.evidence}
    valid_source_labels = {f"S{i}" for i in range(1, len(result.sources) + 1)}
    valid_tags = valid_evidence_ids | valid_source_labels
    used_tags = set(_CITATION_RE.findall(result.report_markdown))
    dangling = used_tags - valid_tags
    assert not dangling, f"Report cites tags with no backing evidence/source: {dangling}"
def test_offline_pipeline_handles_zero_sources_gracefully():
    agent = _fake_agent([])
    result = agent.run("A question with no retrievable sources at all")
    assert result.report_markdown.startswith("# ")
    assert "## Executive Summary" in result.report_markdown
    assert any("No sources were retrieved" in w for w in result.warnings)
    assert result.metrics["sources_selected"] == 0
def test_offline_pipeline_rejects_below_threshold_sources_without_readable_evidence():
    empty_source = Source(id="empty", title="A Paper With No Abstract", provider="arxiv")
    real_source = Source(
        id="real", title="A Paper With Real Content", provider="arxiv",
        abstract="This paper studies retrieval augmented generation and reports a clear improvement in accuracy.",
    )
    agent = _fake_agent([empty_source, real_source])
    result = agent.run("retrieval augmented generation")
    selected_ids = {s.id for s in result.sources}
    assert "empty" not in selected_ids
    assert "real" not in selected_ids
    assert any("No sources were retrieved or retained as RED" in w for w in result.warnings)
def test_offline_pipeline_empty_question_raises():
    agent = _fake_agent([])
    try:
        agent.run("   ")
        assert False, "expected ValueError for empty question"
    except ValueError:
        pass
