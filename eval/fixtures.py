from __future__ import annotations
from dataclasses import dataclass, field
from researchpilot.schemas import Source
from researchpilot.search.base import SourceProvider
class FixedProvider(SourceProvider):
    name = "arxiv"
    def __init__(self, sources: list[Source]):
        self._sources = sources
    def search(self, query: str, limit: int = 8) -> list[Source]:
        return [source.model_copy(update={"provider": "arxiv"}) for source in self._sources]
@dataclass
class Scenario:
    name: str
    question: str
    sources: list[Source]
    description: str = ""
    max_sources: int = 10
    max_iterations: int = 1
    expect_min_sources_selected: int = 0
    expect_max_sources_selected: int | None = None
    expect_warning_contains: list[str] = field(default_factory=list)
    expect_report_contains: list[str] = field(default_factory=list)
def _src(id: str, title: str, abstract: str, year: str = "2024", venue: str = "arXiv",
         provider: str = "fixed", authors: str = "A. Author") -> Source:
    return Source(id=id, title=title, authors=authors, year=year, venue=venue, abstract=abstract, provider=provider)
RICH_EVIDENCE_SOURCES = [
    _src(
        "r1", "Retrieval-Augmented Generation for Domain-Specific QA",
        "We propose a retrieval-augmented generation framework combining a dense retriever with a "
        "fine-tuned generator. Our method improves exact match accuracy by 8.4 points over the "
        "baseline on a domain-specific benchmark dataset. However, the approach incurs additional "
        "latency due to the retrieval step, which limits deployment in real-time settings.",
        year="2025",
    ),
    _src(
        "r2", "A Survey of Adaptive Retrieval Methods for QA",
        "This survey reviews adaptive retrieval methods that adjust retrieval depth based on query "
        "difficulty. Several benchmark datasets are compared, including one collected specifically "
        "for multi-hop reasoning. A key limitation across methods is poor generalization to unseen "
        "domains.",
        year="2024", venue="ACL", provider="semantic_scholar",
    ),
    _src(
        "r3", "Efficient Dense Retrieval with Learned Sparse Representations",
        "We introduce a hybrid retrieval method combining dense embeddings with learned sparse "
        "representations. Results show a 5 point improvement in recall at low latency cost compared "
        "to pure dense retrieval baselines on standard QA benchmarks.",
        year="2023", venue="EMNLP", provider="semantic_scholar",
    ),
    _src(
        "r4", "Long-Context Retrieval for Scientific Literature Review",
        "Our approach processes full documents rather than short passages, using a hierarchical "
        "retriever. We collect a new dataset of 10,000 annotated scientific paper pairs for "
        "evaluation. The method fails to scale beyond 50-page documents due to memory constraints.",
        year="2025", venue="arXiv", provider="arxiv",
    ),
    _src(
        "r5", "Benchmarking Retrieval-Augmented Language Models",
        "We benchmark six retrieval-augmented language models across four QA datasets. The best "
        "performing model achieves state-of-the-art accuracy, outperforming non-retrieval baselines "
        "by a wide margin. Cost per query is not analyzed in this work.",
        year="2024", venue="NeurIPS", provider="arxiv",
    ),
]
SPARSE_EVIDENCE_SOURCES = [
    _src("s1", "A Short Note on Retrieval", "Retrieval helps.", year="2020"),
    _src("s2", "Another Brief Mention", "We used retrieval in our pipeline.", year="2019"),
]
SINGLE_PROVIDER_SOURCES = [
    _src(f"sp{i}", f"Paper {i} on Retrieval Methods",
         f"This paper studies retrieval method variant {i} and reports incremental improvements.",
         provider="arxiv")
    for i in range(1, 6)
]
LARGE_CORPUS_SOURCES = [
    _src(f"lc{i}", f"Paper {i}: Retrieval Augmented Generation Study",
         f"Study {i} examines retrieval augmented generation and reports method M{i} improves "
         f"accuracy by {i} points on a benchmark dataset.",
         provider="arxiv" if i % 2 == 0 else "semantic_scholar")
    for i in range(1, 13)
]
NO_LIMITATION_OR_DATASET_SOURCES = [
    _src(
        "nl1", "A Clean Method Paper",
        "We propose a new architecture for the task. The architecture improves accuracy substantially "
        "over prior work on the standard leaderboard.",
    ),
    _src(
        "nl2", "Another Clean Method Paper",
        "Our framework combines two existing techniques. Experimental results show consistent gains "
        "across all evaluated settings.",
    ),
]
for _collection in (RICH_EVIDENCE_SOURCES, LARGE_CORPUS_SOURCES, NO_LIMITATION_OR_DATASET_SOURCES):
    for _source in _collection:
        _source.full_text = "[PAGE 1]\n" + (_source.abstract + " ") * 60
        _source.full_text_parsed = True
        _source.full_text_complete = True
        _source.full_text_pages = 1
        _source.full_text_chars = len(_source.full_text)
SCENARIOS: list[Scenario] = [
    Scenario(
        name="rich_evidence_literature_review",
        question="What are recent approaches to retrieval augmented generation?",
        sources=RICH_EVIDENCE_SOURCES,
        description="A well-covered topic; the strict 40% raw-relevance gate keeps only the clearly relevant paper and the report must say the corpus is thin.",
        expect_min_sources_selected=1,
        expect_warning_contains=[],
        expect_report_contains=["## Executive Summary", "## Literature Landscape and Major Themes", "## References"],
    ),
    Scenario(
        name="comparison_mode",
        question="Compare dense retrieval versus sparse retrieval methods for QA",
        sources=RICH_EVIDENCE_SOURCES,
        description="A comparison-style question should trigger the comparison table section.",
        expect_min_sources_selected=1,
        expect_report_contains=["## Comparative Findings and Trade-offs"],
    ),
    Scenario(
        name="sparse_evidence",
        question="What is known about a very narrow, under-studied retrieval technique?",
        sources=SPARSE_EVIDENCE_SOURCES,
        description="Only 2 thin sources — the agent must flag this rather than overstate confidence.",
        expect_max_sources_selected=2,
        expect_warning_contains=["No sources were retrieved or retained as RED deep-read evidence"],
    ),
    Scenario(
        name="zero_sources",
        question="A question with no retrievable sources at all xyzzy123",
        sources=[],
        description="No sources returned by any provider — must still produce a complete report.",
        expect_max_sources_selected=0,
        expect_warning_contains=["No sources were retrieved"],
        expect_report_contains=["# ", "## Executive Summary", "## References"],
    ),
    Scenario(
        name="single_provider_only",
        question="What are retrieval method variants?",
        sources=SINGLE_PROVIDER_SOURCES,
        description="All sources come from one provider — the agent should flag limited source diversity.",
        expect_warning_contains=["No sources were retrieved or retained as RED deep-read evidence"],
    ),
    Scenario(
        name="large_corpus_coverage_selection",
        question="Retrieval augmented generation studies",
        sources=LARGE_CORPUS_SOURCES,
        description="A larger candidate set tests evidence-coverage selection without a user-facing paper-count cap.",
        expect_min_sources_selected=3,
        expect_max_sources_selected=12,
    ),
    Scenario(
        name="no_limitation_or_dataset_language",
        question="What methods exist for the task?",
        sources=NO_LIMITATION_OR_DATASET_SOURCES,
        description="Sources with no limitation/dataset language — the report must not invent a gap and should still conclude cleanly.",
        expect_report_contains=["## Conclusion"],
    ),
]
