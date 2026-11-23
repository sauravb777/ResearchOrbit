from __future__ import annotations
import pytest
from researchpilot.config import reset_settings_cache
from researchpilot.schemas import Source
@pytest.fixture(autouse=True)
def _clean_settings(monkeypatch):
    """Ensure each test starts with no LLM keys set unless it opts in."""
    monkeypatch.setenv("REQUIRE_RESEARCH_GRADE_REPORT", "false")
    monkeypatch.setenv("REQUIRE_VERIFIED_DEEP_REPORT", "false")
    monkeypatch.setenv("REPORT_STYLE", "survey")
    for var in ("GROQ_API_KEY", "OPENAI_API_KEY", "SEMANTIC_SCHOLAR_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    reset_settings_cache()
    from researchpilot.search import arxiv as _arxiv
    _arxiv._GLOBAL_LAST_REQUEST[0] = 0.0
    yield
    reset_settings_cache()
@pytest.fixture
def sample_sources() -> list[Source]:
    sources = [
        Source(
            id="s1",
            title="Retrieval-Augmented Generation for Domain-Specific Question Answering",
            authors="A. Author, B. Author",
            year="2025",
            venue="arXiv",
            abstract=(
                "We propose a retrieval-augmented generation framework that combines a "
                "dense retriever with a fine-tuned generator. Our method improves exact "
                "match accuracy by 8.4 points over the baseline on a domain-specific "
                "benchmark. However, the approach incurs additional latency due to the "
                "retrieval step, which limits deployment in real-time settings."
            ),
            provider="arxiv",
            score=0.0,
        ),
        Source(
            id="s2",
            title="A Survey of Adaptive Retrieval Methods for Question Answering",
            authors="C. Author",
            year="2024",
            venue="ACL",
            abstract=(
                "This survey reviews adaptive retrieval methods that adjust retrieval "
                "depth based on query difficulty. Several benchmark datasets are "
                "compared, including one collected specifically for multi-hop reasoning. "
                "A key limitation across methods is poor generalization to unseen domains."
            ),
            provider="arxiv",
            score=0.0,
        ),
    ]
    for source in sources:
        source.full_text = "[PAGE 1]\n" + (source.abstract + " ") * 60
        source.full_text_parsed = True
        source.full_text_complete = True
        source.full_text_pages = 1
        source.full_text_chars = len(source.full_text)
    return sources
