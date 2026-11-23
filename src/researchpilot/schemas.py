from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field, model_validator
REPORT_TYPES = (
    "standard_report", "literature_review", "state_of_art", "gap_analysis", "comparison",
    "evidence_review", "research_proposal", "paper_analysis",
)
LEGACY_REPORT_TYPE_MAP = {
    "technical_deep_dive": "paper_analysis",
    "technical_brief": "literature_review",
    "multi_paper_analysis": "paper_analysis",
}
def canonical_report_type(value: str | None) -> str:
    value = (value or "literature_review").strip().lower()
    return LEGACY_REPORT_TYPE_MAP.get(value, value if value in REPORT_TYPES else "literature_review")
class ResearchIntent(BaseModel):
    """Machine-readable specification of what evidence the research question requires."""
    normalized_question: str
    question_type: str = "descriptive"
    scope: list[str] = Field(default_factory=list)
    exclusions: list[str] = Field(default_factory=list)
    key_terms: list[str] = Field(default_factory=list)
    subquestions: list[str] = Field(default_factory=list)
    evidence_requirements: list[str] = Field(default_factory=list)
    inclusion_criteria: list[str] = Field(default_factory=list)
    exclusion_criteria: list[str] = Field(default_factory=list)
    ambiguity_notes: list[str] = Field(default_factory=list)
class ResearchBlueprint(BaseModel):
    """Adaptive, question-specific specification of what a useful report must teach.
    This is deliberately not a fixed report template. It is an evolving research
    model used to steer retrieval, ranking, reading and final synthesis.
    """
    question: str = ""
    report_type: str = "literature_review"
    purpose: str = ""
    answer_axes: list[str] = Field(default_factory=list)
    must_cover: list[str] = Field(default_factory=list)
    priority_questions: list[str] = Field(default_factory=list)
    evidence_needs: list[str] = Field(default_factory=list)
    comparison_axes: list[str] = Field(default_factory=list)
    boundary_checks: list[str] = Field(default_factory=list)
    exclusions: list[str] = Field(default_factory=list)
    retrieval_targets: list[str] = Field(default_factory=list)
    completeness_tests: list[str] = Field(default_factory=list)
    report_shape_hypotheses: list[str] = Field(default_factory=list)
    rationale: list[str] = Field(default_factory=list)
    revision_notes: list[str] = Field(default_factory=list)
    version: int = 1
class EvidenceRequirement(BaseModel):
    id: str
    description: str
    evidence_type: str
    importance: float = 0.5
    report_sections: list[str] = Field(default_factory=list)
class RelevanceAssessment(BaseModel):
    source_id: str
    semantic_relevance: float = 0.0
    direct_relevance: float = 0.0
    question_alignment: float = 0.0
    methodological_relevance: float = 0.0
    evidence_relevance: float = 0.0
    report_type_relevance: float = 0.0
    complementarity: float = 0.0
    redundancy: float = 0.0
    coverage_value: float = 0.0
    recency: float | None = None
    venue_quality: float | None = None
    publication_signal: float | None = None
    citation_influence: float | None = None
    foundational_importance: float | None = None
    overall: float = 0.0
    importance: float = 0.0
    corpus_value: float = 0.0
    available_weight: float = 0.0
    ranking_confidence: float = 0.0
    missing_signals: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
class CorpusDecision(BaseModel):
    source_id: str
    state: Literal["deep_read", "no_read"] = "no_read"
    rationale: str = ""
    coverage_contribution: list[str] = Field(default_factory=list)
class CrossPaperFinding(BaseModel):
    id: str
    finding_type: Literal["theme", "agreement", "contradiction", "comparison", "gap", "limitation", "trend", "uncertainty"] = "theme"
    text: str
    source_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    caveat: str = ""
    confidence: float = 0.0
class SynthesisPlan(BaseModel):
    report_type: str
    depth: Literal["standard", "deep"] = "standard"
    purpose: str = ""
    sections: list[dict] = Field(default_factory=list)
    required_findings: list[str] = Field(default_factory=list)
    comparisons: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    tables: list[dict] = Field(default_factory=list)
    exclusions: list[str] = Field(default_factory=list)
    writer_brief: dict = Field(default_factory=dict)
class ClaimAudit(BaseModel):
    claim: str
    citation_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    status: Literal["supported", "partial", "unsupported", "overstated", "not_comparable"] = "unsupported"
    correction: str = ""
class ReportAudit(BaseModel):
    claim_audits: list[ClaimAudit] = Field(default_factory=list)
    unsupported_claims: int = 0
    citation_errors: int = 0
    comparison_errors: int = 0
    gap_errors: int = 0
    structural_issues: list[str] = Field(default_factory=list)
    citation_coverage: float = 0.0
    evidence_coverage: float = 0.0
    paper_read_coverage: float = 0.0
    comparison_validity: float = 1.0
    gap_support: float = 1.0
    structural_completeness: float = 0.0
    paper_citation_coverage: float = 0.0
    section_coverage: float = 0.0
    evidence_density: float = 0.0
    quantitative_coverage: float = 0.0
    research_usefulness: float = 0.0
    argument_coherence: float = 0.0
    synthesis_depth: float = 0.0
    answer_completeness: float = 0.0
    evidence_sufficiency: float = 0.0
    quality_gate_reason: str = ""
    revision_cycles: int = 0
    passed: bool = False
class RetrievalPlan(BaseModel):
    """Evidence-driven retrieval strategy produced after research intent."""
    query_families: dict[str, list[str]] = Field(default_factory=dict)
    priorities: list[str] = Field(default_factory=list)
    stopping_signals: list[str] = Field(default_factory=list)
    ranking_weights: dict[str, float] = Field(default_factory=dict)
    metadata_targets: list[str] = Field(default_factory=list)
class PaperReadingPlan(BaseModel):
    """Report-aware reading priorities for one retained paper."""
    source_id: str
    priority_sections: list[str] = Field(default_factory=list)
    evidence_targets: list[str] = Field(default_factory=list)
    unresolved_checks: list[str] = Field(default_factory=list)
class AtomicEvidence(BaseModel):
    """A normalized, source-located evidence unit."""
    id: str
    source_id: str
    evidence_type: str = "context"
    claim: str = ""
    conditions: str = ""
    source_locator: str = ""
    verbatim_span: str = ""
    confidence: float = 0.0
class PaperEvidenceMap(BaseModel):
    """Evidence map for one RED paper, keeping evidence classes explicit."""
    paper_id: str
    problem_evidence: list[str] = Field(default_factory=list)
    method_evidence: list[str] = Field(default_factory=list)
    mechanism_evidence: list[str] = Field(default_factory=list)
    dataset_evidence: list[str] = Field(default_factory=list)
    experiment_evidence: list[str] = Field(default_factory=list)
    result_evidence: list[str] = Field(default_factory=list)
    ablation_evidence: list[str] = Field(default_factory=list)
    limitation_evidence: list[str] = Field(default_factory=list)
    failure_evidence: list[str] = Field(default_factory=list)
    generalization_evidence: list[str] = Field(default_factory=list)
    reproducibility_evidence: list[str] = Field(default_factory=list)
    claim_evidence: list[str] = Field(default_factory=list)
class ComparisonRecord(BaseModel):
    id: str
    source_ids: list[str] = Field(default_factory=list)
    dimensions: dict[str, str] = Field(default_factory=dict)
    comparable: bool = False
    rationale: str = ""
    evidence_ids: list[str] = Field(default_factory=list)
class GapRecord(BaseModel):
    id: str
    category: Literal["explicit_gap", "inferred_gap", "insufficient_evidence", "contradiction"]
    statement: str
    source_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    evidence_needed: list[str] = Field(default_factory=list)
    caveat: str = ""
class DraftSection(BaseModel):
    id: str
    title: str
    purpose: str = ""
    markdown: str = ""
    evidence_ids: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    claim_audits: list[ClaimAudit] = Field(default_factory=list)
    status: Literal["drafted", "revised", "failed", "fallback"] = "drafted"
    revision_count: int = 0
class ResearchArtifact(BaseModel):
    """Stable conceptual artifact for completed research runs."""
    @model_validator(mode="before")
    @classmethod
    def canonicalize_legacy_report_type(cls, obj):
        if isinstance(obj, dict):
            obj = dict(obj)
            obj["report_type"] = canonical_report_type(obj.get("report_type"))
        return obj
    run_id: str
    question: str
    title: str = ""
    report_type: str
    depth: Literal["standard", "deep"]
    warnings: list[str] = Field(default_factory=list)
    metrics: dict[str, int | float | str] = Field(default_factory=dict)
class Source(BaseModel):
    """A single retrieved paper / document."""
    id: str
    title: str
    authors: str = ""
    year: str = ""
    venue: str = ""
    journal_ref: str = ""
    publication_status: str = "preprint"
    venue_quality: float = 0.0
    doi: str = ""
    url: str = ""
    pdf_url: str = ""
    abstract: str = ""
    provider: str = ""
    citation_count: int | None = None
    full_text: str = ""
    pdf_acquired: bool = False
    full_text_parsed: bool = False
    full_text_pages: int = 0
    full_text_chars: int = 0
    full_text_error: str = ""
    full_text_quality: float = 0.0
    full_text_page_coverage: float = 0.0
    full_text_complete: bool = False
    pdf_table_count: int = 0
    pdf_figure_count: int = 0
    pdf_caption_text: list[str] = Field(default_factory=list)
    pdf_artifact_notes: list[str] = Field(default_factory=list)
    reading_basis: Literal["full_text", "abstract_only", "unavailable", "not_attempted"] = "not_attempted"
    score: float = 0.0
    relevance_score: float = 0.0
    importance_score: float = 0.0
    corpus_value_score: float = 0.0
    ranking_confidence: float = 0.0
    ranking_components: dict[str, float | None] = Field(default_factory=dict)
    missing_ranking_signals: list[str] = Field(default_factory=list)
    role: Literal["deep_read", "no_read"] = "no_read"
    relevance_reasons: list[str] = Field(default_factory=list)
    citation_tag: str = ""
    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_role(cls, obj):
        if isinstance(obj, dict):
            obj = dict(obj)
            legacy = obj.get("role")
            if legacy in {"core", "uploaded", "deep"}: obj["role"] = "deep_read"
            elif legacy in {"supporting", "skip", "green"}: obj["role"] = "no_read"
            elif legacy is None: obj["role"] = "no_read"
            if obj.get("full_text"):
                obj.setdefault("full_text_parsed", True)
                obj.setdefault("full_text_chars", len(str(obj.get("full_text") or "")))
                obj.setdefault("full_text_complete", True)
                obj.setdefault("reading_basis", "full_text")
            elif obj.get("abstract"):
                obj.setdefault("reading_basis", "abstract_only")
        return obj
    def dedupe_key(self) -> str:
        if self.doi:
            doi = self.doi.lower().strip()
            doi = doi.removeprefix("https://doi.org/").removeprefix("http://doi.org/")
            doi = doi.removeprefix("https://dx.doi.org/").removeprefix("http://dx.doi.org/")
            doi = doi.removeprefix("doi:").strip().rstrip(".,;)")
            return f"doi:{doi}"
        return "title:" + " ".join(self.title.lower().split())
class Evidence(BaseModel):
    """A short, quotable snippet extracted from one source."""
    id: str
    source_id: str
    text: str
    locator: str = ""
    kind: str = "context"
    relevance: float = 0.0
    conditions: str = ""
    evidence_level: Literal["literal", "paper_interpretation", "cross_paper", "inference", "proposal"] = "literal"
    source_span: str = ""
class Claim(BaseModel):
    """A synthesized finding that must be traceable to evidence ids."""
    id: str
    text: str
    evidence_ids: list[str] = Field(default_factory=list)
    status: Literal["supported", "qualified", "contested", "unverified"] = "unverified"
    confidence: float = 0.0
    evidence_status: Literal["demonstrated", "author_reported", "inferred", "not_established"] = "author_reported"
class ResearchPlan(BaseModel):
    question: str
    mode: Literal["comparison", "landscape", "explanation", "design", "literature_review"]
    report_type: Literal[
        "standard_report", "literature_review", "comparison", "state_of_art", "gap_analysis",
        "evidence_review", "research_proposal", "paper_analysis"
    ] = "literature_review"
    report_title: str = ""
    subquestions: list[str] = Field(default_factory=list)
    search_queries: list[str] = Field(default_factory=list)
    analysis_axes: list[str] = Field(default_factory=list)
    @model_validator(mode="before")
    @classmethod
    def canonicalize_report_type(cls, obj):
        if isinstance(obj, dict):
            obj = dict(obj)
            obj["report_type"] = canonical_report_type(obj.get("report_type"))
        return obj
class ConversationTurn(BaseModel):
    """One chronological turn in the case research conversation."""
    mode: Literal["ask", "verify"] = "ask"
    question: str
    answer: str
    evidence_ids: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    llm_used: bool = False
    references: list[dict] = Field(default_factory=list)
    needs_clarification: bool = False
    clarification_options: list[str] = Field(default_factory=list)
class FollowupQA(BaseModel):
    """One grounded follow-up question/answer pair over a run's evidence."""
    question: str
    answer: str
    evidence_ids: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    llm_used: bool = False
    references: list[dict] = Field(default_factory=list)
    needs_clarification: bool = False
    clarification_options: list[str] = Field(default_factory=list)
class QuantResult(BaseModel):
    """One structured quantitative result, kept as baseline/proposed/delta
    rather than flattened into prose -- this is the direct fix for
    "reduce results to 'the method performs better'". Any field the source
    text didn't state is left empty rather than guessed."""
    dataset: str = ""
    model: str = ""
    metric: str = ""
    baseline: str = ""
    proposed: str = ""
    absolute_delta: str = ""
    relative_delta: str = ""
    evidence_id: str = ""
class PaperAnalysis(BaseModel):
    """A structured, LLM-produced deep reading of ONE paper.
    This is the direct answer to "you only look at the abstract": when an
    LLM is available and full text was fetched (or, failing that, the
    abstract), the paper is read specifically for its problem, method,
    experiments, ablations, and limitations rather than being reduced to a
    handful of independently-ranked sentences.
    Design note on trust: the free-text fields below are genuine paraphrase
    (that is unavoidable — real understanding cannot be a literal substring
    of the source), so they do NOT get the same hard "must be a literal
    quote" guarantee as `Evidence`. What they DO get: they are only ever
    produced from real retrieved text (never from the model's unaided
    memory of the paper), every field is optional and left blank rather
    than padded when the source text doesn't support it, and each analysis
    is displayed in the report next to the paper's *literal, verified*
    evidence quotes (see `Evidence` with locator containing "deep-read") so
    a reader can spot-check the paraphrase against a real sentence from the
    paper. `depth` tells the reader plainly which regime produced this.
    Only ever built for a paper's OWN source text and never mixes evidence
    from another paper. Provenance checks enforce this boundary.
    """
    @model_validator(mode="before")
    @classmethod
    def _coerce_llm_shapes(cls, obj):
        """One mis-shaped field must not discard an otherwise good dossier."""
        if not isinstance(obj, dict):
            return obj
        obj = dict(obj)
        for name, f in cls.model_fields.items():
            if name not in obj or obj[name] is None:
                continue
            v, ann = obj[name], str(f.annotation)
            if ann == "<class 'str'>" and not isinstance(v, str):
                if isinstance(v, list):
                    v = "; ".join(
                        str(x.get("observation") or x.get("text") or x) if isinstance(x, dict) else str(x) for x in v
                    )
                elif isinstance(v, dict):
                    v = "; ".join(f"{k}: {x}" for k, x in v.items())
                else:
                    v = str(v)
                obj[name] = v
            elif ann.startswith("list[dict]") and not isinstance(v, list):
                obj[name] = [v] if isinstance(v, dict) else ([{"observation": str(v)}] if v else [])
            elif ann.startswith("list[dict]") and isinstance(v, list):
                obj[name] = [x if isinstance(x, dict) else {"observation": str(x)} for x in v]
        return obj
    source_id: str
    depth: Literal["full_text", "abstract_only"] = "abstract_only"
    method_category: str = ""
    problem: str = ""
    research_objective: str = ""
    intervention_point: str = ""
    core_idea: str = ""
    method_summary: str = ""
    mechanism: str = ""
    experimental_setup: str = ""
    dataset_details: str = ""
    baselines: str = ""
    metrics: str = ""
    training_protocol: str = ""
    statistical_uncertainty: str = ""
    error_analysis: str = ""
    key_results: str = ""
    quantitative_results: list[QuantResult] = Field(default_factory=list)
    ablations: str = ""
    robustness: str = ""
    computational_cost: str = ""
    reproducibility: str = ""
    limitations_author_stated: str = ""
    limitations_inferred: str = ""
    established: str = ""
    not_established: str = ""
    caveat: str = ""
    assumptions: str = ""
    novelty_claim: str = ""
    strengths: str = ""
    failure_modes: str = ""
    generalization: str = ""
    implementation_details: str = ""
    dependency_risks: str = ""
    result_interpretation: str = ""
    open_questions: str = ""
    section_map: list[dict] = Field(default_factory=list)
    claim_ledger: list[dict] = Field(default_factory=list)
    experiment_records: list[dict] = Field(default_factory=list)
    negative_results: str = ""
    causal_claims: str = ""
    transfer_conditions: str = ""
    evidence_conflicts: str = ""
    citation_context: str = ""
    evidence_quality_notes: str = ""
    question_answer: str = ""
    key_evidence_chain: list[dict] = Field(default_factory=list)
    decision_relevant_insights: list[dict] = Field(default_factory=list)
    method_tradeoffs: list[dict] = Field(default_factory=list)
    replication_requirements: list[str] = Field(default_factory=list)
    research_implications: list[str] = Field(default_factory=list)
    evidence_inventory: list[dict] = Field(default_factory=list)
    research_relevance_map: dict = Field(default_factory=dict)
    research_memory: list[dict] = Field(default_factory=list)
    unmapped_observations: list[dict] = Field(default_factory=list)
    field_evidence: dict[str, list[str]] = Field(default_factory=dict)
    first_principles: str = ""
    equation_explanations: list[dict] = Field(default_factory=list)
    design_rationale: list[dict] = Field(default_factory=list)
    experiment_chain: list[dict] = Field(default_factory=list)
    ablation_analysis: list[dict] = Field(default_factory=list)
    baseline_fairness: str = ""
    claim_evidence_audit: list[dict] = Field(default_factory=list)
    section_consistency: str = ""
    hidden_assumptions: list[dict] = Field(default_factory=list)
    novelty_breakdown: str = ""
    significance: str = ""
    concept_glossary: list[dict] = Field(default_factory=list)
    reading_questions: list[dict] = Field(default_factory=list)
    feynman_explanation: str = ""
    feynman_analogy: str = ""
    feynman_caveat: str = ""
    reading_passes: int = 0
    reading_coverage: str = ""
    llm_used: bool = True
    @property
    def limitations(self) -> str:
        """Back-compat convenience for callers written against the
        earlier single-field version (report rendering, qa.py notes)."""
        parts = [p for p in (self.limitations_author_stated, self.limitations_inferred) if p]
        return " ".join(parts)
class ClaimVerification(BaseModel):
    """Result of checking a user-supplied claim against evidence."""
    claim: str
    verdict: Literal[
        "well_supported", "partially_supported", "mixed", "contradicted", "insufficient_evidence"
    ] = "insufficient_evidence"
    reasoning: str = ""
    evidence_ids: list[str] = Field(default_factory=list)
    sources_checked: int = 0
    new_sources: list[Source] = Field(default_factory=list)
    new_evidence: list[Evidence] = Field(default_factory=list)
    llm_used: bool = False
class RunEvent(BaseModel):
    """One entry in the run's audit trail (for transparency in the UI)."""
    stage: str
    detail: str
    ok: bool = True
class ResearchReadingStrategy(BaseModel):
    """Question-derived reading protocol created before any paper is deeply read."""
    research_question: str = ""
    report_type: str = ""
    central_answer_target: str = ""
    priority_dimensions: list[str] = Field(default_factory=list)
    evidence_questions: list[str] = Field(default_factory=list)
    evidence_targets: list[str] = Field(default_factory=list)
    comparison_dimensions: list[str] = Field(default_factory=list)
    contradiction_checks: list[str] = Field(default_factory=list)
    limitation_checks: list[str] = Field(default_factory=list)
    unexpected_evidence_prompt: str = ""
    inclusion_boundary: str = ""
    exclusion_boundary: str = ""
    generated_by_llm: bool = False
class ResearchResult(BaseModel):
    """The complete output of one research run.
    Legacy report names are accepted on load and immediately canonicalized so
    old saved cases remain readable without exposing obsolete modes in the UI.
    """
    @model_validator(mode="before")
    @classmethod
    def canonicalize_legacy_report_type(cls, obj):
        if isinstance(obj, dict):
            obj = dict(obj)
            obj["report_type"] = canonical_report_type(obj.get("report_type"))
        return obj
    run_id: str
    question: str
    report_title: str = ""
    case_name: str = ""
    pinned: bool = False
    report_type: Literal[
        "standard_report", "literature_review", "comparison", "state_of_art", "gap_analysis",
        "evidence_review", "research_proposal", "paper_analysis"
    ] = "literature_review"
    version: int = 1
    parent_run_id: str = ""
    build_mode: Literal["initial", "new", "combine"] = "initial"
    added_source_ids: list[str] = Field(default_factory=list)
    plan: ResearchPlan
    sources: list[Source] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    events: list[RunEvent] = Field(default_factory=list)
    report_markdown: str = ""
    llm_used: bool = False
    metrics: dict[str, int | float | str] = Field(default_factory=dict)
    followups: list[FollowupQA] = Field(default_factory=list)
    conversation: list[ConversationTurn] = Field(default_factory=list)
    paper_analyses: list[PaperAnalysis] = Field(default_factory=list)
    paper_read_status: list[dict] = Field(default_factory=list)
    verifications: list[ClaimVerification] = Field(default_factory=list)
    research_intent: ResearchIntent | None = None
    research_knowledge_model: dict = Field(default_factory=dict)
    research_chat_memory: dict = Field(default_factory=dict)
    report_type_analysis: dict = Field(default_factory=dict)
    research_blueprint: ResearchBlueprint | None = None
    evidence_requirements: list[EvidenceRequirement] = Field(default_factory=list)
    cross_paper_findings: list[CrossPaperFinding] = Field(default_factory=list)
    research_council: dict = Field(default_factory=dict)
    research_director: dict = Field(default_factory=dict)
    synthesis_plan: SynthesisPlan | None = None
    report_audit: ReportAudit | None = None
    retrieval_plan: RetrievalPlan | None = None
    paper_reading_plans: list[PaperReadingPlan] = Field(default_factory=list)
    reading_strategy: ResearchReadingStrategy | None = None
    paper_evidence_maps: list[PaperEvidenceMap] = Field(default_factory=list)
    atomic_evidence: list[AtomicEvidence] = Field(default_factory=list)
    comparison_records: list[ComparisonRecord] = Field(default_factory=list)
    gap_records: list[GapRecord] = Field(default_factory=list)
    corpus_decisions: list[CorpusDecision] = Field(default_factory=list)
    retrieval_trace: list[dict] = Field(default_factory=list)
    coverage_table: list[dict] = Field(default_factory=list)
    atomic_fact_ledger: list[dict] = Field(default_factory=list)
    creator_pack: dict = Field(default_factory=dict)
    draft_sections: list[DraftSection] = Field(default_factory=list)
