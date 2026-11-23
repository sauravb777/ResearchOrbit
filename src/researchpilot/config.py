from __future__ import annotations
import os
from dataclasses import dataclass, field
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass
def _bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}
def _int(name: str, default: int) -> int:
    val = os.getenv(name)
    try:
        return int(val) if val else default
    except ValueError:
        return default
def _float(name: str, default: float) -> float:
    val = os.getenv(name)
    try:
        return float(val) if val else default
    except ValueError:
        return default
def _list(name: str, default: str) -> list[str]:
    raw = os.getenv(name, default)
    return [x.strip() for x in raw.split(",") if x.strip()]
@dataclass(frozen=True)
class Settings:
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    groq_base_url: str = field(
        default_factory=lambda: os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    )
    groq_models: list[str] = field(
        default_factory=lambda: _list(
            "GROQ_MODELS", "openai/gpt-oss-120b,openai/gpt-oss-20b"
        )
    )
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_base_url: str = field(
        default_factory=lambda: os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    )
    openai_models: list[str] = field(
        default_factory=lambda: _list("OPENAI_MODELS", "gpt-4o-mini,gpt-4.1-mini")
    )
    llm_timeout: int = field(default_factory=lambda: _int("LLM_TIMEOUT", 45))
    llm_retries: int = field(default_factory=lambda: _int("LLM_RETRIES", 0))
    llm_max_concurrency: int = field(default_factory=lambda: _int("LLM_MAX_CONCURRENCY", 5))
    v88_retrieval_min_stable_rounds: int = field(default_factory=lambda: _int("V88_RETRIEVAL_MIN_STABLE_ROUNDS", 2))
    v88_retrieval_min_stability_seconds: float = field(default_factory=lambda: _float("V88_RETRIEVAL_MIN_STABILITY_SECONDS", 6.0))
    paper_read_concurrency: int = field(default_factory=lambda: _int("PAPER_READ_CONCURRENCY", 5))
    deep_read_max_selected_regions: int = field(default_factory=lambda: _int("DEEP_READ_MAX_SELECTED_REGIONS", 8))
    llm_temperature: float = field(default_factory=lambda: _float("LLM_TEMPERATURE", 0.2))
    max_run_cost_usd: float = field(default_factory=lambda: _float("MAX_RUN_COST_USD", 5.0))
    max_llm_calls: int = field(default_factory=lambda: _int("MAX_LLM_CALLS", 800))
    enable_council: bool = field(default_factory=lambda: _bool("ENABLE_COUNCIL", True))
    enable_v79_architecture: bool = field(default_factory=lambda: _bool("ENABLE_V79_ARCHITECTURE", bool(os.getenv("GROQ_API_KEY"))))
    v79_max_papers: int = field(default_factory=lambda: _int("V79_MAX_PAPERS", 15))
    v79_max_candidates: int = field(default_factory=lambda: _int("V79_MAX_CANDIDATES", 15))
    v79_discovery_budget_seconds: int = field(default_factory=lambda: _int("V79_DISCOVERY_BUDGET_SECONDS", 180))
    v79_extraction_max_papers: int = field(default_factory=lambda: _int("V79_EXTRACTION_MAX_PAPERS", 15))
    v79_results_per_query: int = field(default_factory=lambda: _int("V79_RESULTS_PER_QUERY", 10))
    v80_arxiv_page_size: int = field(default_factory=lambda: _int("V80_ARXIV_PAGE_SIZE", 60))
    v80_candidate_reservoir: int = field(default_factory=lambda: _int("V80_CANDIDATE_RESERVOIR", 240))
    v80_min_stability_seconds: int = field(default_factory=lambda: _int("V80_MIN_STABILITY_SECONDS", 12))
    v80_llm_rerank_pool: int = field(default_factory=lambda: _int("V80_LLM_RERANK_POOL", 40))
    v81_lens_passes: bool = field(default_factory=lambda: _bool("V81_LENS_PASSES", True))
    fast_deep_read: bool = field(default_factory=lambda: _bool("FAST_DEEP_READ", True))
    fast_deep_read_max_chars: int = field(default_factory=lambda: _int("FAST_DEEP_READ_MAX_CHARS", 140000))
    fast_deep_read_chunk_chars: int = field(default_factory=lambda: _int("FAST_DEEP_READ_CHUNK_CHARS", 105000))
    fast_deep_read_overlap_chars: int = field(default_factory=lambda: _int("FAST_DEEP_READ_OVERLAP_CHARS", 6000))
    fast_report_pipeline: bool = field(default_factory=lambda: _bool("FAST_REPORT_PIPELINE", True))
    performance_input_chars_per_paper: int = field(default_factory=lambda: _int("PERFORMANCE_INPUT_CHARS_PER_PAPER", 18000))
    performance_reader_max_tokens: int = field(default_factory=lambda: _int("PERFORMANCE_READER_MAX_TOKENS", 2600))
    performance_council_card_chars: int = field(default_factory=lambda: _int("PERFORMANCE_COUNCIL_CARD_CHARS", 5200))
    performance_council_max_tokens: int = field(default_factory=lambda: _int("PERFORMANCE_COUNCIL_MAX_TOKENS", 3000))
    performance_writer_max_tokens: int = field(default_factory=lambda: _int("PERFORMANCE_WRITER_MAX_TOKENS", 2600))
    performance_writer_passages: int = field(default_factory=lambda: _int("PERFORMANCE_WRITER_PASSAGES", 4))
    performance_max_report_topics: int = field(default_factory=lambda: _int("PERFORMANCE_MAX_REPORT_TOPICS", 7))
    quality_first: bool = field(default_factory=lambda: _bool("QUALITY_FIRST", True))
    quality_deep_read_region_chars: int = field(default_factory=lambda: _int("QUALITY_DEEP_READ_REGION_CHARS", 24000))
    quality_deep_read_bundle_chars: int = field(default_factory=lambda: _int("QUALITY_DEEP_READ_BUNDLE_CHARS", 70000))
    quality_deep_read_bundle_tokens: int = field(default_factory=lambda: _int("QUALITY_DEEP_READ_BUNDLE_TOKENS", 4200))
    quality_deep_read_integration_tokens: int = field(default_factory=lambda: _int("QUALITY_DEEP_READ_INTEGRATION_TOKENS", 6200))
    quality_deep_read_max_bundles: int = field(default_factory=lambda: _int("QUALITY_DEEP_READ_MAX_BUNDLES", 12))
    quality_report_min_topics: int = field(default_factory=lambda: _int("QUALITY_REPORT_MIN_TOPICS", 4))
    quality_report_max_topics: int = field(default_factory=lambda: _int("QUALITY_REPORT_MAX_TOPICS", 6))
    quality_report_writer_chars: int = field(default_factory=lambda: _int("QUALITY_REPORT_WRITER_CHARS", 22000))
    groq_tpm_hint: int = field(default_factory=lambda: _int("GROQ_TPM_HINT", 8000))
    universal_fast_report: bool = field(default_factory=lambda: _bool("UNIVERSAL_FAST_REPORT", True))
    research_run_deadline_seconds: int = field(default_factory=lambda: _int("RESEARCH_RUN_DEADLINE_SECONDS", 540))
    compact_report_reasoning: bool = field(default_factory=lambda: _bool("COMPACT_REPORT_REASONING", True))
    ultra_fast_report: bool = field(default_factory=lambda: _bool("ULTRA_FAST_REPORT", False))
    ultra_fast_report_deadline_seconds: int = field(default_factory=lambda: _int("ULTRA_FAST_REPORT_DEADLINE_SECONDS", 600))
    ultra_fast_max_topics: int = field(default_factory=lambda: _int("ULTRA_FAST_MAX_TOPICS", 3))
    ultra_fast_card_chars: int = field(default_factory=lambda: _int("ULTRA_FAST_CARD_CHARS", 13000))
    fast_skip_relevance_enrichment: bool = field(default_factory=lambda: _bool("FAST_SKIP_RELEVANCE_ENRICHMENT", True))
    beast_deep_reader: bool = field(default_factory=lambda: _bool("BEAST_DEEP_READER", True))
    beast_llm_skim: bool = field(default_factory=lambda: _bool("BEAST_LLM_SKIM", False))
    beast_region_chars: int = field(default_factory=lambda: _int("BEAST_REGION_CHARS", 17000))
    beast_bundle_chars: int = field(default_factory=lambda: _int("BEAST_BUNDLE_CHARS", 70000))
    beast_overlap_chars: int = field(default_factory=lambda: _int("BEAST_OVERLAP_CHARS", 2200))
    beast_integration_chars: int = field(default_factory=lambda: _int("BEAST_INTEGRATION_CHARS", 85000))
    beast_whole_paper_tier: str = field(default_factory=lambda: os.getenv("BEAST_WHOLE_PAPER_TIER", "M").strip().upper() or "F")
    beast_whole_paper_tokens: int = field(default_factory=lambda: _int("BEAST_WHOLE_PAPER_TOKENS", 6000))
    selective_deep_read: bool = field(default_factory=lambda: _bool("SELECTIVE_DEEP_READ", True))
    selective_cartographer_threshold_chars: int = field(default_factory=lambda: _int("SELECTIVE_CARTOGRAPHER_THRESHOLD_CHARS", 90000))
    selective_read_section_chars: int = field(default_factory=lambda: _int("SELECTIVE_READ_SECTION_CHARS", 28000))
    selective_read_overlap_chars: int = field(default_factory=lambda: _int("SELECTIVE_READ_OVERLAP_CHARS", 2500))
    selective_read_max_regions: int = field(default_factory=lambda: _int("SELECTIVE_READ_MAX_REGIONS", 12))
    selective_read_max_jobs: int = field(default_factory=lambda: _int("SELECTIVE_READ_MAX_JOBS", 12))
    selective_read_bundle_chars: int = field(default_factory=lambda: _int("SELECTIVE_READ_BUNDLE_CHARS", 52000))
    selective_read_bundle_sections: int = field(default_factory=lambda: _int("SELECTIVE_READ_BUNDLE_SECTIONS", 4))
    selective_integration_chars: int = field(default_factory=lambda: _int("SELECTIVE_INTEGRATION_CHARS", 120000))
    v81_lens_max_chars: int = field(default_factory=lambda: _int("V81_LENS_MAX_CHARS", 90000))
    v81_max_report_topics: int = field(default_factory=lambda: _int("V81_MAX_REPORT_TOPICS", 4))
    v80_retrieval_llm_workers: int = field(default_factory=lambda: _int("V80_RETRIEVAL_LLM_WORKERS", 4))
    v79_gap_search_budget_seconds: int = field(default_factory=lambda: _int("V79_GAP_SEARCH_BUDGET_SECONDS", 90))
    v79_max_read_passes: int = field(default_factory=lambda: _int("V79_MAX_READ_PASSES", 2))
    v79_fail_closed: bool = field(default_factory=lambda: _bool("V79_FAIL_CLOSED", True))
    search_providers: list[str] = field(default_factory=lambda: ["arxiv"])
    max_sources: int = field(default_factory=lambda: _int("MAX_SOURCES", 30))
    max_evidence: int = field(default_factory=lambda: _int("MAX_EVIDENCE", 40))
    max_iterations: int = field(default_factory=lambda: _int("MAX_ITERATIONS", 2))
    results_per_query: int = field(default_factory=lambda: _int("RESULTS_PER_QUERY", 8))
    search_timeout: int = field(default_factory=lambda: _int("SEARCH_TIMEOUT", 20))
    fetch_full_text: bool = field(default_factory=lambda: _bool("FETCH_FULL_TEXT", True))
    max_full_text_sources: int = field(
        default_factory=lambda: _int("MAX_FULL_TEXT_SOURCES", 15)
    )
    full_text_paper_timeout: int = field(default_factory=lambda: _int("FULL_TEXT_PAPER_TIMEOUT", 60))
    full_text_max_chars: int = field(default_factory=lambda: _int("FULL_TEXT_MAX_CHARS", 1500000))
    max_pdf_bytes: int = field(default_factory=lambda: _int("MAX_PDF_BYTES", 50_000_000))
    max_pdf_pages: int = field(default_factory=lambda: _int("MAX_PDF_PAGES", 250))
    enable_semantic_rerank: bool = field(default_factory=lambda: _bool("ENABLE_SEMANTIC_RERANK", False))
    semantic_rerank_model: str = field(
        default_factory=lambda: os.getenv("SEMANTIC_RERANK_MODEL", "all-MiniLM-L6-v2")
    )
    deep_read_max_papers: int = field(default_factory=lambda: 5)
    enable_deep_read: bool = field(default_factory=lambda: _bool("ENABLE_DEEP_READ", True))
    corpus_safety_ceiling: int = field(default_factory=lambda: _int("CORPUS_SAFETY_CEILING", 30))
    auto_selection_max: int = field(default_factory=lambda: _int("AUTO_SELECTION_MAX", 15))
    require_verified_deep_report: bool = field(default_factory=lambda: _bool("REQUIRE_VERIFIED_DEEP_REPORT", False))
    require_research_grade_report: bool = field(default_factory=lambda: _bool("REQUIRE_RESEARCH_GRADE_REPORT", False))
    enable_adaptive_gap_search: bool = field(default_factory=lambda: _bool("ENABLE_ADAPTIVE_GAP_SEARCH", False))
    adaptive_gap_max_queries: int = field(default_factory=lambda: _int("ADAPTIVE_GAP_MAX_QUERIES", 4))
    final_repair_cycles: int = field(default_factory=lambda: _int("FINAL_REPAIR_CYCLES", 2))
    report_style: str = field(default_factory=lambda: os.getenv("REPORT_STYLE", "feynman").strip().lower() or "survey")
    deep_read_max_segments: int = field(default_factory=lambda: _int("DEEP_READ_MAX_SEGMENTS", 24))
    deep_read_segment_chars: int = field(default_factory=lambda: _int("DEEP_READ_SEGMENT_CHARS", 24000))
    deep_read_grounding_chars: int = field(default_factory=lambda: _int("DEEP_READ_GROUNDING_CHARS", 90000))
    verify_max_new_sources: int = field(default_factory=lambda: _int("VERIFY_MAX_NEW_SOURCES", 6))
    data_dir: str = field(default_factory=lambda: os.getenv("DATA_DIR", "data"))
    reports_dir: str = field(default_factory=lambda: os.getenv("REPORTS_DIR", "reports"))
    db_path: str = field(default_factory=lambda: os.getenv("DB_PATH", "data/researchpilot.db"))
    def __post_init__(self) -> None:
        """Normalize malformed environment values into safe runtime bounds.
        Environment variables are external configuration, not trusted typed
        input. Without this normalization a value such as MAX_SOURCES=-3 or
        LLM_TIMEOUT=-1 can produce invalid UI bounds or invalid HTTP timeouts.
        Keep intentionally useful zero values (for example, disabling full-text
        fetches or verification) while preventing values that can only be
        nonsensical at runtime.
        """
        positive_ints = (
            "llm_timeout", "search_timeout", "max_sources", "llm_max_concurrency", "paper_read_concurrency",
            "results_per_query", "full_text_max_chars", "max_pdf_bytes", "max_pdf_pages",
        )
        nonnegative_ints = (
            "llm_retries", "max_evidence", "max_full_text_sources",
            "max_iterations",
            "corpus_safety_ceiling", "auto_selection_max", "verify_max_new_sources", "adaptive_gap_max_queries", "max_llm_calls", "final_repair_cycles", "deep_read_max_segments", "deep_read_segment_chars", "deep_read_grounding_chars", "require_research_grade_report", "v79_max_papers", "v79_max_candidates", "v79_results_per_query", "v80_arxiv_page_size", "v80_candidate_reservoir", "v80_min_stability_seconds", "v80_llm_rerank_pool", "v80_retrieval_llm_workers", "v81_lens_max_chars", "v81_max_report_topics", "v79_gap_search_budget_seconds", "v79_max_read_passes", "fast_deep_read_max_chars", "fast_deep_read_chunk_chars", "fast_deep_read_overlap_chars", "performance_input_chars_per_paper", "performance_reader_max_tokens", "performance_council_card_chars", "performance_council_max_tokens", "performance_writer_max_tokens", "performance_writer_passages", "performance_max_report_topics", "quality_first", "quality_deep_read_region_chars", "quality_deep_read_bundle_chars", "quality_deep_read_bundle_tokens", "quality_deep_read_integration_tokens", "quality_deep_read_max_bundles", "quality_report_min_topics", "quality_report_max_topics", "quality_report_writer_chars", "groq_tpm_hint", "beast_region_chars", "beast_whole_paper_tokens", "fast_skip_relevance_enrichment", "universal_fast_report", "research_run_deadline_seconds", "compact_report_reasoning", "ultra_fast_report_deadline_seconds", "ultra_fast_max_topics", "ultra_fast_card_chars", "beast_bundle_chars", "beast_llm_skim", "beast_overlap_chars", "beast_integration_chars", "selective_cartographer_threshold_chars", "selective_read_section_chars", "selective_read_overlap_chars", "selective_read_max_regions", "selective_read_max_jobs", "selective_read_bundle_chars", "selective_read_bundle_sections", "selective_integration_chars",
        )
        for name in positive_ints:
            object.__setattr__(self, name, max(1, int(getattr(self, name))))
        for name in nonnegative_ints:
            object.__setattr__(self, name, max(0, int(getattr(self, name))))
        object.__setattr__(self, "max_run_cost_usd", max(0.0, float(self.max_run_cost_usd)))
        object.__setattr__(self, "llm_temperature", min(2.0, max(0.0, float(self.llm_temperature))))
        if not self.groq_models:
            object.__setattr__(self, "groq_models", _list("__MISSING_GROQ_MODELS__", "openai/gpt-oss-120b,openai/gpt-oss-20b"))
        if not self.openai_models:
            object.__setattr__(self, "openai_models", _list("__MISSING_OPENAI_MODELS__", "gpt-4o-mini,gpt-4.1-mini"))
    @property
    def llm_available(self) -> bool:
        if self.enable_v79_architecture:
            return bool(self.groq_api_key)
        return bool(self.groq_api_key or self.openai_api_key)
_settings: Settings | None = None
def get_settings() -> Settings:
    """Return a process-wide cached Settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
def reset_settings_cache() -> None:
    """Used by tests to force settings to be re-read from the environment."""
    global _settings
    _settings = None
