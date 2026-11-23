from __future__ import annotations
import time
import uuid
import json
import re
from ..llm import LLMClient
from ..schemas import canonical_report_type, ClaimVerification, ConversationTurn, FollowupQA, PaperAnalysis, ResearchPlan, ResearchIntent, EvidenceRequirement, ResearchResult, ResearchReadingStrategy, RunEvent, Source
from ..search.aggregator import SearchAggregator
from ..search.anytime import AnytimeRetrievalController
from . import claims as claims_mod
from . import intelligence as intelligence_mod
from .research_reasoning import peer_review_report
from . import deep_read as deep_read_mod
from . import reading_resilience as reading_rr
from . import evidence as evidence_mod
from . import evidence_normalization as evidence_norm_mod
from . import report_drafting as drafting_mod
from . import planner
from . import relevance as relevance_mod
from . import selection as selection_mod
from . import synthesizer
from . import feynman_report as feynman_mod
from ..schemas import ReportAudit
from . import targeting as targeting_mod
from . import research_quality as research_quality_mod
from . import adaptive_retrieval as adaptive_retrieval_mod
from .research_graph import run_report_research_graph
from . import verify as verify_mod
from .fulltext import enrich_with_full_text
from .qa import answer_followup
from .research_knowledge import build_research_knowledge
from .chat_memory import build_research_chat_memory
from .ranker import has_readable_content, rank_sources
from .relevance_engine import AUTO_RELEVANCE_THRESHOLD, AUTO_PRIORITY_THRESHOLD, AUTO_MIN_PAPERS, AUTO_FALLBACK_PAPERS, AUTO_MAX_PAPERS, automatic_selection_target
from .corpus_integrity import validate_sources, validate_analyses, rescue_analysis_corpus, rescue_source_corpus
from .semantic_rerank import semantic_rerank
from .v79_retrieval import build_facets, build_query_families, run_v79_retrieval
def generate_followup_queries(question: str, plan: ResearchPlan, already_tried: list[str]) -> list[str]:
    """Generate additional search queries for a second/third round,
    targeting the plan's subquestions and analysis axes so widening the
    search is purposeful rather than repeating the same terms. Standalone
    (not a method) so both `ResearchAgent` and any future execution path
    can share it without duplicating logic.
    """
    tried_lower = {q.lower() for q in already_tried}
    candidates: list[str] = []
    for sub in plan.subquestions:
        candidates.append(sub)
    for axis in plan.analysis_axes:
        candidates.append(f"{question} {axis}")
    fresh = [c for c in candidates if c.lower() not in tried_lower]
    return fresh[:4]
class ResearchAgent:
    def __init__(self, aggregator: SearchAggregator | None = None, llm: LLMClient | None = None, dossier_cache=None):
        self.aggregator = aggregator or SearchAggregator()
        self.llm = llm or LLMClient()
        self.chat_llm = LLMClient(settings=self.llm.settings) if type(self.llm) is LLMClient else self.llm
        self.dossier_cache = dossier_cache
    @staticmethod
    def _chat_context(result: ResearchResult) -> tuple[list[Source], list]:
        """Return the report corpus plus verification material with canonical source ids.
        Verification searches can rediscover a paper already in the report. In
        that case the verification evidence may carry a fresh provider id; map
        it back to the existing case source so citation tags never become S?.
        """
        sources = list(result.sources)
        dedupe_to_id = {s.dedupe_key(): s.id for s in sources}
        id_to_source = {s.id: s for s in sources}
        evidence = list(result.evidence)
        seen_evidence = {e.id for e in evidence}
        for verification in result.verifications:
            local_id_map: dict[str, str] = {}
            for source in verification.new_sources:
                canonical_id = dedupe_to_id.get(source.dedupe_key())
                if canonical_id is None:
                    sources.append(source)
                    dedupe_to_id[source.dedupe_key()] = source.id
                    id_to_source[source.id] = source
                    canonical_id = source.id
                local_id_map[source.id] = canonical_id
            for item in verification.new_evidence:
                canonical_source_id = local_id_map.get(item.source_id)
                if canonical_source_id is None and item.source_id in id_to_source:
                    canonical_source_id = dedupe_to_id.get(id_to_source[item.source_id].dedupe_key(), item.source_id)
                canonical_source_id = canonical_source_id or item.source_id
                if item.id in seen_evidence:
                    continue
                if canonical_source_id != item.source_id:
                    item = item.model_copy(update={"source_id": canonical_source_id})
                evidence.append(item)
                seen_evidence.add(item.id)
        return sources, evidence
    def run(self, question: str, *args, **kwargs) -> ResearchResult:
        """Run the research pipeline with explicit v79 publication gating.
        Legacy compatibility modes may degrade. When v79 fail-closed execution
        is enabled, internal failures propagate instead of publishing an
        unverified deterministic report.
        """
        import dataclasses, logging
        log = logging.getLogger("researchpilot")
        try:
            return self._run_full(question, *args, **kwargs)
        except ValueError:
            raise
        except Exception as exc:
            if getattr(self.llm, "ultra_fast_override", False):
                self.llm.clear_run_deadline()
                self.llm.ultra_fast_override = False
            if "cancel" in str(exc).lower():
                raise
            if getattr(self.llm.settings, "enable_v79_architecture", False) and getattr(self.llm.settings, "v79_fail_closed", True):
                log.exception("v79 research-grade pipeline stopped: %s", exc)
                raise
            log.exception("Full pipeline failed; retrying in reduced mode: %s", exc)
            first_error = str(exc)
        original = self.llm
        try:
            return self._degraded_run(original, first_error, question, *args, **kwargs)
        finally:
            self.llm = original
    def _degraded_run(self, original, first_error, question, *args, **kwargs) -> ResearchResult:
        import dataclasses, logging
        log = logging.getLogger("researchpilot")
        try:
            reduced = dataclasses.replace(original.settings, enable_council=False, final_repair_cycles=0,
                                          require_research_grade_report=False, require_verified_deep_report=False)
            self.llm = LLMClient(settings=reduced, usage=original.usage)
            result = self._run_full(question, *args, **kwargs)
            result.warnings = list(result.warnings or []) + [f"Reduced-depth mode used after an internal error: {first_error[:200]}"]
            return result
        except ValueError:
            raise
        except Exception as exc2:
            if "cancel" in str(exc2).lower():
                raise
            log.exception("Reduced mode failed; producing deterministic report: %s", exc2)
            second_error = str(exc2)
        try:
            no_llm = dataclasses.replace(original.settings, groq_api_key="", openai_api_key="", enable_council=False,
                                         final_repair_cycles=0, require_research_grade_report=False,
                                         require_verified_deep_report=False)
            self.llm = LLMClient(settings=no_llm)
            result = self._run_full(question, *args, **kwargs)
            result.warnings = list(result.warnings or []) + [
                "Offline evidence-based report (LLM stages unavailable). Check API key/quota and regenerate for full depth. "
                f"Cause: {second_error[:200]}"]
            return result
        finally:
            self.llm = original
    def _run_full(
        self,
        question: str,
        max_sources: int = 10,
        max_evidence: int = 40,
        max_iterations: int | None = None,
        deep_read_top_k: int | None = None,
        extra_sources: list[Source] | None = None,
        report_type: str | None = None,
        report_title: str = "",
        version: int = 1,
        parent_run_id: str = "",
        build_mode: str = "initial",
        preselected_sources: list[Source] | None = None,
        automatic_candidates: list[Source] | None = None,
        analysis_depth: str = "standard",
        precomputed_plan: ResearchPlan | None = None,
        precomputed_events: list[RunEvent] | None = None,
        precomputed_usage: dict[str, float | int] | None = None,
        cancel_check=None,
    ) -> ResearchResult:
        """Run the full pipeline. `extra_sources` (optional) are user-supplied
        papers (see `fulltext.source_from_uploaded_pdf`) that are merged into
        the retrieved pool before ranking, so a paper the user already knows
        is relevant is never at the mercy of the search APIs surfacing it."""
        question = (question or "").strip()
        explicit_report_type = report_type
        report_type = canonical_report_type(report_type) if report_type else None
        analysis_depth = "deep"
        run_id = uuid.uuid4().hex[:10]
        events: list[RunEvent] = list(precomputed_events or [])
        warnings: list[str] = []
        v79_retrieval_trace: list[dict] = []
        v79_coverage: list[dict] = []
        creator_pack: dict = {}
        settings = self.llm.settings
        explicit_selection = preselected_sources is not None
        ultra_fast_requested = bool(getattr(settings, "ultra_fast_report", False))
        self.llm.ultra_fast_override = bool(ultra_fast_requested)
        self.llm.set_run_deadline(
            getattr(settings, "ultra_fast_report_deadline_seconds", 600)
            if ultra_fast_requested
            else getattr(settings, "research_run_deadline_seconds", 540)
        )
        max_iterations = max(0, max_iterations if max_iterations is not None else settings.max_iterations)
        stage_seconds: dict[str, float] = {}
        run_started = time.perf_counter()
        if not question:
            raise ValueError("question must not be empty")
        if getattr(settings, "enable_v79_architecture", False) and getattr(settings, "v79_fail_closed", True) and not self.llm.available:
            raise RuntimeError("ResearchOrbit v79 requires a configured GROQ_API_KEY; no LLM was available, so the run was stopped instead of generating a fallback report.")
        t0 = time.perf_counter()
        research_blueprint = None
        blueprint_llm_used = False
        if explicit_selection:
            mode = "comparison" if any(k in question.lower() for k in ("compare", " versus ", " vs ")) else "literature_review"
            report_type = canonical_report_type(report_type or "literature_review")
            report_title = report_title or question
            plan = ResearchPlan(
                question=question, mode=mode, report_type=report_type, report_title=report_title,
                subquestions=["What does the retained literature establish?", "How do the approaches work and differ?", "What evidence supports the findings?", "Where do findings agree or disagree?", "What remains unresolved and why?"],
                search_queries=[question],
                analysis_axes=["problem", "methods", "mechanisms", "evidence", "results", "limitations", "research gaps"],
            )
            intent = ResearchIntent(
                normalized_question=question,
                question_type=mode,
                subquestions=list(plan.subquestions),
                evidence_requirements=["methods", "mechanisms", "experimental evidence", "limitations", "research gaps"],
            )
            evidence_requirements = [EvidenceRequirement(id=f"ER{i}", description=d, evidence_type="research_evidence", importance=0.8) for i, d in enumerate(intent.evidence_requirements, 1)]
            intent_llm_used = False
            plan_llm_used = False
            retrieval_plan = None
            retrieval_llm_used = False
        else:
            if precomputed_plan is not None:
                plan = precomputed_plan.model_copy(deep=True)
                if plan.question.strip() != question:
                    raise ValueError("precomputed_plan.question must match question")
                report_type = plan.report_type
                report_title = plan.report_title or report_title
                plan_llm_used = False
            else:
                requested_type = report_type or "literature_review"
                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=2, thread_name_prefix="researchpilot-plan") as pool:
                    intent_future = pool.submit(
                        intelligence_mod.build_intent, question, requested_type, analysis_depth, self.llm
                    )
                    plan_future = pool.submit(
                        planner.make_plan, question, llm=self.llm, report_type=requested_type, report_title=report_title
                    )
                    intent, evidence_requirements, intent_llm_used = intent_future.result()
                    plan, plan_llm_used = plan_future.result()
                research_blueprint, blueprint_llm_used = intelligence_mod.build_research_blueprint(
                    question, plan.report_type, intent, self.llm
                )
                retrieval_plan, retrieval_llm_used = intelligence_mod.build_retrieval_plan(
                    intent, plan.report_type, self.llm, blueprint=research_blueprint
                )
                plan.search_queries = intelligence_mod.plan_queries(retrieval_plan, plan)
                report_type = plan.report_type
                report_title = plan.report_title or planner.suggest_report_title(question, report_type)
                plan.report_title = report_title
            if precomputed_plan is not None:
                intent, evidence_requirements, intent_llm_used = intelligence_mod.build_intent(question, report_type, analysis_depth, self.llm)
                research_blueprint, blueprint_llm_used = intelligence_mod.build_research_blueprint(question, report_type, intent, self.llm)
                retrieval_plan, retrieval_llm_used = intelligence_mod.build_retrieval_plan(intent, report_type, self.llm, blueprint=research_blueprint)
                if retrieval_plan.query_families:
                    plan.search_queries = intelligence_mod.plan_queries(retrieval_plan, plan)
        report_title = report_title or planner.suggest_report_title(question, report_type)
        plan.report_title = report_title
        stage_seconds["plan"] = round(time.perf_counter() - t0, 3)
        events.append(RunEvent(stage="intent", detail=f"research intent specified; evidence requirements={len(evidence_requirements)}, llm_used={intent_llm_used}"))
        if research_blueprint is not None:
            events.append(RunEvent(stage="research_blueprint", detail=f"adaptive report blueprint v{research_blueprint.version} created; axes={len(research_blueprint.answer_axes)}, evidence needs={len(research_blueprint.evidence_needs)}, llm_used={blueprint_llm_used}"))
        events.append(RunEvent(stage="plan", detail=f"mode={plan.mode}, report_type={report_type}, llm_used={plan_llm_used}"))
        t0 = time.perf_counter()
        raw_sources: dict[str, Source] = {}
        queries_tried: list[str] = []
        preselected = list(preselected_sources or [])
        automatic_snapshot = list(automatic_candidates or [])
        uploads = list(extra_sources or [])
        for src in preselected:
            if has_readable_content(src):
                src.role = "deep_read"
                raw_sources[src.dedupe_key()] = src
        for src in uploads:
            if has_readable_content(src):
                if src.role != "deep_read":
                    src.role = "no_read"
                raw_sources.setdefault(src.dedupe_key(), src)
        if preselected_sources is None and automatic_snapshot:
            for src in automatic_snapshot:
                if has_readable_content(src):
                    raw_sources.setdefault(src.dedupe_key(), src)
            events.append(RunEvent(
                stage="discovery_snapshot",
                detail=f"using {len(raw_sources)} ranked candidates from the completed autonomous screening snapshot"
            ))
        if preselected_sources is None and not automatic_snapshot and max_iterations > 0:
            discovery_budget = max(30, int(getattr(settings, "v79_discovery_budget_seconds", 600)))
            base_queries = list(plan.search_queries or [question])
            v79_plan = None
            v79_plan_used = False
            if getattr(settings, "enable_v79_architecture", False) and self.llm.available:
                try:
                    v79_plan, v79_plan_used = build_facets(
                        question, report_type or "literature_review", self.llm
                    )
                    base_queries.extend(build_query_families(question, v79_plan))
                except Exception as exc:
                    events.append(RunEvent(
                        stage="retrieval_R1",
                        detail=f"facet query planning unavailable; using base queries: {exc}",
                        ok=False,
                    ))
            seen_initial: set[str] = set()
            initial_queries = []
            for q in base_queries:
                key = " ".join(str(q).lower().split())
                if key and key not in seen_initial:
                    seen_initial.add(key)
                    initial_queries.append(str(q).strip())
            controller = AnytimeRetrievalController(self.aggregator)
            def _followups():
                fresh = self._followup_queries(question, plan, list(seen_initial))[:8]
                seen_initial.update(" ".join(str(q).lower().split()) for q in fresh)
                return fresh
            snapshot, retrieval_events = controller.run(
                question,
                initial_queries,
                deadline_seconds=discovery_budget,
                max_papers=min(15, int(getattr(settings, "v79_max_papers", 15))),
                results_per_query=min(100, int(getattr(settings, "v80_arxiv_page_size", 60))),
                batch_size=4,
                followup_factory=_followups,
                min_stable_rounds=max(2, int(getattr(settings, "v88_retrieval_min_stable_rounds", 2))),
                candidate_pool_size=int(getattr(settings, "v80_candidate_reservoir", 240)),
                min_stability_seconds=float(getattr(settings, "v88_retrieval_min_stability_seconds", 6)),
            )
            events.extend(retrieval_events)
            queries_tried.extend(initial_queries)
            raw_sources = {
                src.dedupe_key(): src
                for src in snapshot.candidates
                if has_readable_content(src)
            }
            provisional_red = set(snapshot.deep_read_ids)
            for source in raw_sources.values():
                source.role = "deep_read" if source.id in provisional_red else "no_read"
            events.append(RunEvent(
                stage="discovery_snapshot",
                detail=(
                    f"bounded Top-{len(raw_sources)} retained; "
                    f"provisional deep-read={len(provisional_red)}; "
                    f"rounds={snapshot.rounds}; queries={snapshot.queries_attempted}; "
                    f"elapsed={snapshot.elapsed_seconds:.1f}s; "
                    f"deadline_reached={snapshot.deadline_reached}; planner_llm={v79_plan_used}"
                ),
            ))
            if snapshot.deadline_reached:
                events.append(RunEvent(
                    stage="retrieval_budget",
                    detail=f"retrieval window reached {discovery_budget}s; proceeding with best-known Top-{len(raw_sources)}",
                ))
        candidates = list(raw_sources.values())
        snapshot_selected = [s for s in automatic_snapshot if s.role == "deep_read"] if automatic_snapshot and preselected_sources is None else []
        already_red = {s.id for s in preselected} | {s.id for s in uploads if s.role == "deep_read"}
        decision_candidates = [s for s in candidates if s.id not in already_red]
        if snapshot_selected:
            decision_candidates = []
            events.append(RunEvent(stage="snapshot_corpus_preserved", detail=f"preserved {len(snapshot_selected)} RED paper(s) from completed autonomous screening"))
        decisions: list = []
        relevance_llm_used = False
        if decision_candidates and not explicit_selection:
            if semantic_rerank(question, decision_candidates):
                events.append(RunEvent(stage="rerank", detail="semantic retrieval signal applied"))
            else:
                rank_sources(question, decision_candidates)
            llm_assessments, relevance_llm_used = intelligence_mod.assess_relevance(
                question, plan, decision_candidates, self.llm, intent=intent, blueprint=research_blueprint
            )
            scores = relevance_mod.score_all(question, plan, decision_candidates)
            for source in decision_candidates:
                a = llm_assessments.get(source.id)
                if a:
                    source.score = round(a.overall, 4)
                    source.relevance_reasons = list(dict.fromkeys(a.reasons))[:6]
                else:
                    source.score = scores[source.id].total
                    source.relevance_score = round(source.score * 100, 2)
                    source.relevance_reasons = scores[source.id].reasons[:6]
            from .relevance_engine import apply_ranking
            apply_ranking(decision_candidates, llm_assessments, plan.report_type)
            raw_gate_pool = [
                s for s in decision_candidates
                if float(getattr(s, "score", 0.0) or 0.0) >= AUTO_RELEVANCE_THRESHOLD
            ]
            raw_gate_pool.sort(key=lambda s: float(getattr(s, "score", 0.0) or 0.0), reverse=True)
            extraction_ceiling = max(1, int(getattr(settings, "v79_extraction_max_papers", 15)))
            r6_reserve = min(2, max(0, extraction_ceiling - 1)) if getattr(settings, "enable_adaptive_gap_search", False) else 0
            raw_gate_pool = raw_gate_pool[:max(1, extraction_ceiling - r6_reserve)]
            events.append(RunEvent(
                stage="relevance_gate",
                detail=f"{len(raw_gate_pool)} candidate(s) crossed raw {AUTO_RELEVANCE_THRESHOLD:.0%}; "
                       f"verifying readable full text before corpus selection"
            ))
            events.extend(enrich_with_full_text(raw_gate_pool, max_fetch=len(raw_gate_pool), cancel_check=cancel_check))
            readable_gate_pool = [
                s for s in raw_gate_pool
                if bool(s.full_text.strip()) and bool(s.full_text_parsed) and bool(getattr(s, "full_text_complete", True))
            ]
            unreadable_gate_pool = [s for s in raw_gate_pool if s not in readable_gate_pool]
            for source in unreadable_gate_pool:
                source.role = "no_read"
                events.append(RunEvent(
                    stage="fulltext_gate",
                    detail=f"excluded '{source.title[:60]}' before corpus selection: verified readable full text unavailable",
                    ok=False,
                ))
            events.append(RunEvent(
                stage="fulltext_preselection",
                detail=f"{len(readable_gate_pool)} of {len(raw_gate_pool)} raw-gate candidates have verified readable full text"
            ))
            v79_selected = []
            v79_decisions = []
            if getattr(settings, "enable_v79_architecture", False) and self.llm.available and readable_gate_pool:
                extraction_budget_used = len(raw_gate_pool)
                extraction_ceiling = max(1, int(getattr(settings, "v79_extraction_max_papers", 15)))
                def _v79_gap_search(queries):
                    nonlocal extraction_budget_used
                    remaining_budget = max(0, extraction_ceiling - extraction_budget_used)
                    if remaining_budget <= 0:
                        events.append(RunEvent(stage="retrieval_R6", detail="targeted gap search found no remaining full-text extraction budget; keeping Top-15 extraction ceiling"))
                        return []
                    gap_deadline = time.monotonic() + max(
                        15, int(getattr(settings, "v79_gap_search_budget_seconds", 90))
                    )
                    found, gap_events = self.aggregator.run(
                        queries,
                        deadline=gap_deadline,
                        max_results=min(10, int(getattr(settings, "v79_results_per_query", 10))),
                    )
                    events.extend(gap_events)
                    batch = found[:remaining_budget]
                    extraction_budget_used += len(batch)
                    enrich_with_full_text(batch, max_fetch=len(batch), cancel_check=cancel_check)
                    return [x for x in batch if x.full_text_complete]
                v79 = run_v79_retrieval(
                    question, report_type,
                    readable_gate_pool[: getattr(settings, "v79_max_papers", 15)],
                    self.llm, max_papers=getattr(settings, "v79_max_papers", 15),
                    search_fn=_v79_gap_search,
                )
                events.append(RunEvent(stage="retrieval_v79", detail=f"R0-R6 selected {len(v79.selected)} papers; facets={len(v79.facets)}; coverage={sum(1 for c in v79.coverage if c.get('covered'))}/{len(v79.coverage)}; llm_used={v79.llm_used}"))
                v79_retrieval_trace = list(v79.trace)
                v79_coverage = list(v79.coverage)
                for warning in v79.warnings:
                    warnings.append(warning)
                for item in v79.trace:
                    events.append(RunEvent(stage=f"retrieval_{item.get('stage','v79')}", detail=json.dumps(item, ensure_ascii=False)))
                retrieval_ceiling = max(1, int(getattr(settings, "v79_max_papers", 15)))
                eligible_v79 = [s for s in v79.selected if float(getattr(s, "score", 0.0) or 0.0) >= AUTO_RELEVANCE_THRESHOLD]
                target_v79 = automatic_selection_target(eligible_v79 or list(v79.selected), score_getter=lambda s: float(getattr(s, "score", 0.0) or 0.0))
                target_v79 = min(target_v79, int(getattr(settings, "auto_selection_max", AUTO_MAX_PAPERS) or AUTO_MAX_PAPERS), len(v79.selected))
                v79_selected = list(v79.selected[:target_v79])
                selected_ids = {s.id for s in v79_selected}
                for extra in v79.selected:
                    extra.role = "deep_read" if extra.id in selected_ids else "no_read"
                events.append(RunEvent(
                    stage="deep_read_budget",
                    detail=f"retrieval corpus={len(v79.selected)}; automatic deep-read recommendation={len(v79_selected)}; user may adjust the selection",
                ))
                from ..schemas import CorpusDecision
                selected_ids = {s.id for s in v79_selected}
                for rank, src in enumerate(readable_gate_pool, 1):
                    v79_decisions.append(CorpusDecision(
                        source_id=src.id, state="deep_read" if src.id in selected_ids else "no_read",
                        rationale=("v79 facet/verification selection" if src.id in selected_ids else "v79 retrieval candidate not admitted to RED corpus"),
                        coverage_contribution=[],
                    ))
                added_red, decisions = v79_selected, v79_decisions
            else:
                added_red, decisions = intelligence_mod.build_corpus(
                    readable_gate_pool,
                    {k: v.total for k, v in scores.items()},
                    llm_assessments,
                    report_type,
                    analysis_depth,
                    settings.corpus_safety_ceiling,
                    semantic_screening_used=relevance_llm_used,
                    auto_selection_max=settings.auto_selection_max,
                )
            selected = (
                (preselected + [s for s in uploads if s.role == "deep_read"])
                if preselected_sources is not None
                else ((snapshot_selected + [s for s in uploads if s.role == "deep_read"]) if snapshot_selected else ([s for s in uploads if s.role == "deep_read"] + added_red))
            )
        else:
            selected = preselected + [s for s in uploads if s.role == "deep_read"]
        selection_before_fulltext = list(dict((s.dedupe_key(), s) for s in selected).values())
        explicit_pool = [s for s in selected if not (s.full_text.strip() and s.full_text_parsed)]
        if explicit_pool:
            events.extend(enrich_with_full_text(explicit_pool, max_fetch=len(explicit_pool), cancel_check=cancel_check))
        selected = [
            s for s in selected
            if s.full_text.strip() and s.full_text_parsed and bool(getattr(s, "full_text_complete", True))
        ]
        readable_ids = {s.id for s in selected}
        unreadable_selected = [s for s in selection_before_fulltext if s.id not in readable_ids]
        for source in unreadable_selected:
            reason = source.full_text_error or "full text could not be acquired and verified as readable"
            events.append(RunEvent(
                stage="fulltext_gate",
                detail=f"explicitly selected paper excluded before deep reading: '{source.title[:80]}' ({reason[:180]})",
                ok=False,
            ))
        selected = list(dict((s.dedupe_key(), s) for s in selected).values())
        deep_read_cap = 5
        if len(selected) > deep_read_cap:
            if preselected_sources is not None:
                selected = selected[:deep_read_cap]
                warnings.append("Deep Read is limited to 5 papers; only the first 5 explicit selections were retained.")
            else:
                selected = sorted(selected, key=lambda s: (-float(getattr(s, "score", 0.0) or 0.0), s.title.lower()))[:deep_read_cap]
                warnings.append("Deep Read is limited to 5 papers; the five highest-ranked readable papers were retained.")
            events.append(RunEvent(stage="deep_read_budget", detail=f"hard deep-read ceiling applied: {len(selected)}/{deep_read_cap} paper(s)"))
        for source in selected:
            source.role = "deep_read"
        for source in candidates:
            if source not in selected:
                source.role = "no_read"
        sel = selection_mod.SelectionResult(
            deep_read=selected,
            no_read=[s for s in candidates if s not in selected],
            rejected=[s for s in candidates if s not in selected],
        )
        corpus_label = "explicit corpus accepted as RED deep-read evidence" if preselected_sources is not None else "autonomous corpus"
        events.append(RunEvent(
            stage="corpus",
            detail=f"{corpus_label}: {len(selected)} RED deep-read / {len(candidates)-len(selected)} GREEN no-read; "
                   f"raw eligibility={AUTO_RELEVANCE_THRESHOLD:.0%}; priority=>{AUTO_PRIORITY_THRESHOLD:.0%}; full-text verified before selection"
        ))
        events.append(RunEvent(stage="corpus_complete", detail=f"autonomous corpus: {len(selected)} RED deep-read / {len(candidates)-len(selected)} GREEN no-read"))
        stage_seconds["search"] = round(time.perf_counter() - t0, 3)
        stage_seconds["select"] = stage_seconds["search"]
        if not selected:
            warnings.append("No sources were retrieved or retained as RED deep-read evidence with both raw relevance >= 39% and verified readable full text.")
        elif len(selected) < 3:
            warnings.append("Fewer than 3 RED papers were retained; conclusions may be provisional.")
        incomplete_selected = [
            s for s in selected
            if not (s.full_text.strip() and s.full_text_parsed and bool(getattr(s, "full_text_complete", True)))
        ]
        if incomplete_selected:
            for source in incomplete_selected:
                source.role = "no_read"
            selected = [s for s in selected if s not in incomplete_selected]
            warnings.append(f"{len(incomplete_selected)} paper(s) without verified full text were excluded from deep reading.")
        full_text_selected = len(selected)
        events.append(RunEvent(stage="fulltext_gate", detail=f"verified full-text RED corpus: {len(selected)} papers; raw eligibility={AUTO_RELEVANCE_THRESHOLD:.0%}; priority=>{AUTO_PRIORITY_THRESHOLD:.0%}; automatic target={min(AUTO_MAX_PAPERS, len(selected))}, hard maximum={AUTO_MAX_PAPERS}"))
        alignment = [] if explicit_selection else validate_sources(question, selected, self.llm, research_blueprint=research_blueprint)
        mismatched = [a for a in alignment if not a.aligned]
        if explicit_selection:
            events.append(RunEvent(stage="corpus_integrity", detail=f"explicit selection locked: {len(selected)} selected paper(s); semantic replacement gate skipped on the user-selected corpus", ok=True))
        elif mismatched and preselected_sources is not None:
            events.append(RunEvent(
                stage="corpus_integrity",
                detail=f"explicit selection locked: {len(selected)} selected paper(s) retained; {len(mismatched)} alignment warning(s) recorded for reader/reporter context",
                ok=True,
            ))
            warnings.append(
                "Explicit deep-read selection is locked. Topical alignment warnings are retained for the final synthesis rather than silently replacing the researcher's chosen papers."
            )
        elif mismatched:
            mismatch_ids = {a.source_id for a in mismatched}
            replacements = [s for s in candidates if s.id not in {x.id for x in selected} and s.id not in mismatch_ids and float(getattr(s, "score", 0.0) or 0.0) >= AUTO_RELEVANCE_THRESHOLD]
            if replacements:
                events.extend(enrich_with_full_text(replacements[:settings.auto_selection_max], max_fetch=min(len(replacements), settings.auto_selection_max), cancel_check=cancel_check))
                replacements = [s for s in replacements if s.full_text.strip() and s.full_text_parsed and bool(getattr(s, "full_text_complete", True))]
                replacement_alignment = validate_sources(question, replacements, self.llm, research_blueprint=research_blueprint)
                good = [s for s, a in zip(replacements, replacement_alignment) if a.aligned]
                selected = [s for s in selected if s.id not in mismatch_ids] + good[:len(mismatched)]
                for s in selected:
                    s.role = "deep_read"
                events.append(RunEvent(stage="corpus_integrity", detail=f"automatic corpus repaired {len(mismatched)} alignment issue(s) with {min(len(mismatched), len(good))} replacement(s)", ok=True))
            else:
                events.append(RunEvent(stage="corpus_integrity", detail=f"automatic corpus retained {len(selected)} verified-full-text paper(s); no aligned replacement was available", ok=True))
        else:
            events.append(RunEvent(stage="corpus_integrity", detail=f"all {len(selected)} selected papers passed topical coherence validation"))
        if not selected and (preselected_sources is not None or candidates):
            recovery_pool = [s for s in (preselected or candidates) if s.full_text.strip() and s.full_text_parsed and bool(getattr(s, "full_text_complete", True))]
            if recovery_pool:
                recovery_pool.sort(key=lambda s: (-float(getattr(s, "score", 0.0) or 0.0), s.title.lower()))
                selected = recovery_pool[:max(1, min(len(recovery_pool), settings.auto_selection_max))]
                for source in selected:
                    source.role = "deep_read"
                warnings.append("Semantic corpus validation removed the initial set, so the strongest verified full-text papers were recovered for evidence-first deep reading; final report delivery remains gated by source grounding.")
                events.append(RunEvent(stage="corpus_recovery", detail=f"recovered {len(selected)} verified full-text paper(s) after semantic false-negative", ok=True))
            else:
                warnings.append("No selected papers remain after research-question coherence validation.")
        if getattr(settings, "enable_v79_architecture", False):
            reading_strategy = {"evidence_questions": [], "mode": "v79_integrated"}
            reading_strategy_used = False
        else:
            reading_strategy, reading_strategy_used = intelligence_mod.build_reading_strategy(
                question, report_type, intent, self.llm
            )
        events.append(RunEvent(
            stage="reading_strategy",
            detail=f"pre-reading evidence agenda ready; {len(reading_strategy.get('evidence_questions', []))} evidence questions; llm_used={reading_strategy_used}"
        ))
        from concurrent.futures import ThreadPoolExecutor
        deep_budget = len(selected)
        paper_read_statuses: list = [
            reading_rr.excluded_status(
                source.id, source.title,
                source.full_text_error or "full text could not be acquired and verified as readable",
                diagnosis=reading_rr.NO_TEXT,
            )
            for source in unreadable_selected
        ]
        def _extract_heuristic_evidence():
            started = time.perf_counter()
            value = evidence_mod.extract_all(question, selected, max_total=max_evidence)
            return value, round(time.perf_counter() - started, 3)
        def _run_deep_read():
            started = time.perf_counter()
            value = deep_read_mod.analyze_papers(
                question, selected, llm=self.llm, top_k=deep_budget, analysis_mode=analysis_depth,
                report_type=report_type, reading_strategy=reading_strategy, dossier_cache=self.dossier_cache,
                cancel_check=cancel_check, status_sink=paper_read_statuses,
            )
            return value, round(time.perf_counter() - started, 3)
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="researchpilot-read-prep") as pool:
            evidence_future = pool.submit(_extract_heuristic_evidence)
            deep_future = pool.submit(_run_deep_read)
            heuristic_evidence, evidence_seconds = evidence_future.result()
            (analyses, deep_evidence, deep_events), deep_seconds = deep_future.result()
        stage_seconds["evidence"] = evidence_seconds
        events.extend(deep_events)
        if not analyses and selected and self.llm.available and getattr(settings, "enable_v79_architecture", False):
            why = "; ".join(f"'{st.title[:70]}': {st.failure_class or 'unknown'} - {st.reason[:120]}" for st in paper_read_statuses[:5]) or "no paper produced a verified dossier"
            raise RuntimeError(
                f"None of the {len(selected)} selected paper(s) produced a verified dossier, so there is no evidence to build a report from. Details: {why}"
            )
        if not analyses and settings.require_verified_deep_report:
            raise RuntimeError(
                "No paper-level deep reading completed. Check the configured LLM provider/API key and extraction events."
            )
        stage_seconds["deep_read"] = deep_seconds
        evidence = evidence_mod.merge_evidence([heuristic_evidence, deep_evidence], max_total=max_evidence + len(deep_evidence))
        completed_read_ids = {a.source_id for a in analyses}
        failed_read_sources = [s for s in selected if s.id not in completed_read_ids]
        if failed_read_sources and self.llm.available:
            for source in failed_read_sources:
                source.role = "no_read"
            selected = [s for s in selected if s.id in completed_read_ids]
            evidence = [e for e in evidence if e.source_id in completed_read_ids]
            warnings.append(
                f"{len(failed_read_sources)} selected paper(s) were excluded from synthesis because whole-paper deep reading did not complete; no abstract-only claims were used as deep-read evidence."
            )
            for st in paper_read_statuses:
                if not st.verified:
                    warnings.append(f"Not analysed: '{st.title[:100]}' ({st.failure_class or 'unknown'}: {st.reason[:160]}).")
            events.append(RunEvent(
                stage="deep_read_corpus_finalized",
                detail=f"retained {len(selected)} evidence-bearing deep-read paper(s); excluded {len(failed_read_sources)} failed deep-read paper(s)",
                ok=bool(selected),
            ))
        analysis_alignment = [] if explicit_selection else validate_analyses(question, selected, analyses, self.llm)
        bad_analysis_ids = {a.source_id for a in analysis_alignment if not a.aligned}
        if bad_analysis_ids:
            _orig_analyses, _orig_selected, _orig_evidence = list(analyses), list(selected), list(evidence)
            rescued_analyses, rescued_sources, rescued_ids = rescue_analysis_corpus(
                question, selected, analyses, bad_analysis_ids
            )
            rescued_set = set(rescued_ids)
            analyses = [a for a in rescued_analyses if a.source_id in {x.source_id for x in rescued_analyses}]
            admitted_ids = {a.source_id for a in analyses}
            evidence = [e for e in evidence if e.source_id in admitted_ids]
            selected = [s for s in selected if s.id in admitted_ids]
            dropped_ids = bad_analysis_ids - rescued_set
            events.append(RunEvent(
                stage="deep_read_integrity",
                detail=(f"post-reading coherence review flagged {len(bad_analysis_ids)} dossier(s); "
                        f"rescued {len(rescued_ids)} on independent relevance/evidence signals; "
                        f"removed {len(dropped_ids)} with no corroborating alignment signal"),
                ok=bool(analyses),
            ))
            if rescued_ids:
                warnings.append(
                    f"{len(rescued_ids)} deep-read paper(s) received a conservative post-reading alignment false-negative "
                    "but were retained because their corpus relevance and paper-level evidence independently supported the research question."
                )
            if not analyses:
                analyses, selected, evidence = _orig_analyses, _orig_selected, _orig_evidence
                warnings.append("Post-reading alignment review found weak topical support; the report is provisional and should be read with caution.")
        _admitted_ids = {a.source_id for a in analyses}
        _titles = {src.id: src.title for src in selected}
        for _st in paper_read_statuses:
            if _st.verified and _st.source_id not in _admitted_ids:
                _st.verified, _st.level, _st.status = False, reading_rr.LEVEL_EXCLUDED, "excluded"
                _st.reason = "dossier was removed by the post-reading topical alignment review"
                _st.failure_class = "alignment_rejected"
        deep_read_mod.attach_quantitative_evidence_ids(analyses, evidence)
        deep_read_mod.attach_structured_evidence_ids(analyses, evidence)
        events.append(
            RunEvent(
                stage="evidence",
                detail=f"{len(evidence)} snippets total ({len(deep_evidence)} from deep-read, {len(analyses)} papers deep-read)",
            )
        )
        explicit_selection = preselected_sources is not None
        if research_blueprint is not None and analyses and self.llm.available and not (explicit_selection and len(analyses) == 1):
            revised_blueprint, revised_blueprint_used = intelligence_mod.revise_research_blueprint(
                question, report_type, research_blueprint, analyses, [], self.llm
            )
            if revised_blueprint_used:
                research_blueprint = revised_blueprint
                blueprint_llm_used = True
                events.append(RunEvent(stage="research_blueprint_revision", detail=f"blueprint revised from first-pass evidence to v{research_blueprint.version}; axes={len(research_blueprint.answer_axes)}, must_cover={len(research_blueprint.must_cover)}"))
        adaptive_gap_count = 0
        adaptive_added_count = 0
        adaptive_queries: list[str] = []
        adaptive_plan: dict = {}
        adaptive_used = False
        if settings.enable_adaptive_gap_search and not explicit_selection and selected and max_iterations >= 2 and self.llm.available:
            gap_plan, gap_llm_used = adaptive_retrieval_mod.identify_evidence_gaps(
                question, report_type, intent, analyses, evidence, self.llm
            )
            adaptive_plan = gap_plan if isinstance(gap_plan, dict) else {}
            adaptive_used = bool(gap_llm_used)
            query_items = adaptive_plan.get("queries", []) if isinstance(adaptive_plan, dict) else []
            query_items = query_items[: max(1, settings.adaptive_gap_max_queries)]
            adaptive_queries = [str(item.get("query", "")).strip() for item in query_items if isinstance(item, dict) and str(item.get("query", "")).strip()]
            adaptive_queries = list(dict.fromkeys(adaptive_queries))[:settings.adaptive_gap_max_queries]
            if adaptive_queries:
                events.append(RunEvent(stage="adaptive_retrieval", detail=f"evidence-gap search proposed {len(adaptive_queries)} targeted queries; llm_used={gap_llm_used}"))
                queries_tried.extend(adaptive_queries)
                adaptive_deadline = time.monotonic() + max(
                    15, int(getattr(settings, "v79_gap_search_budget_seconds", 90))
                )
                found, search_events = self.aggregator.run(
                    adaptive_queries,
                    deadline=adaptive_deadline,
                    max_results=min(10, int(getattr(settings, "v79_results_per_query", 10))),
                )
                events.extend(search_events)
                existing_keys = {s.dedupe_key() for s in candidates}
                new_candidates = []
                for src in found:
                    if has_readable_content(src) and src.dedupe_key() not in existing_keys:
                        existing_keys.add(src.dedupe_key())
                        candidates.append(src)
                        new_candidates.append(src)
                adaptive_gap_count = len(new_candidates)
                if new_candidates:
                    if semantic_rerank(question, new_candidates):
                        events.append(RunEvent(stage="adaptive_rerank", detail="semantic retrieval signal applied to gap-search candidates"))
                    else:
                        rank_sources(question, new_candidates)
                    gap_assessments, gap_relevance_used = intelligence_mod.assess_relevance(
                        question, plan, new_candidates, self.llm, intent=intent, blueprint=research_blueprint
                    )
                    gap_scores = relevance_mod.score_all(question, plan, new_candidates)
                    for source in new_candidates:
                        assessment = gap_assessments.get(source.id)
                        if assessment:
                            source.score = round(assessment.overall, 4)
                            source.relevance_score = round(assessment.overall * 100, 2)
                            source.relevance_reasons = list(dict.fromkeys(assessment.reasons))[:6]
                        else:
                            source.score = gap_scores[source.id].total
                            source.relevance_score = round(source.score * 100, 2)
                            source.relevance_reasons = gap_scores[source.id].reasons[:6]
                    from .relevance_engine import apply_ranking
                    apply_ranking(new_candidates, gap_assessments, plan.report_type)
                    remaining_auto_slots = max(0, AUTO_MAX_PAPERS - len(selected))
                    gap_gate_pool = [
                        s for s in new_candidates
                        if float(getattr(s, "score", 0.0) or 0.0) >= AUTO_RELEVANCE_THRESHOLD
                    ]
                    events.extend(enrich_with_full_text(gap_gate_pool, max_fetch=len(gap_gate_pool), cancel_check=cancel_check))
                    readable_gap_pool = [
                        s for s in gap_gate_pool
                        if s.full_text.strip() and s.full_text_parsed and bool(getattr(s, "full_text_complete", True))
                    ]
                    added_red, gap_decisions = intelligence_mod.build_corpus(
                        readable_gap_pool,
                        {k: v.total for k, v in gap_scores.items()},
                        gap_assessments,
                        report_type,
                        analysis_depth,
                        settings.corpus_safety_ceiling,
                        semantic_screening_used=gap_relevance_used,
                        auto_selection_max=remaining_auto_slots,
                    ) if remaining_auto_slots else ([], [])
                    decisions.extend(gap_decisions)
                    for source in added_red:
                        source.role = "deep_read"
                    if added_red:
                        adaptive_added_count = len(added_red)
                        events.append(RunEvent(stage="adaptive_corpus", detail=f"gap search added {len(added_red)} RED deep-read paper(s) from {len(new_candidates)} new candidates"))
                        t_adapt = time.perf_counter()
                        adaptive_analyses, adaptive_deep_evidence, adaptive_events = deep_read_mod.analyze_papers(
                            question, added_red, llm=self.llm, top_k=len(added_red), analysis_mode=analysis_depth,
                            report_type=report_type, reading_strategy=reading_strategy, dossier_cache=self.dossier_cache
                        )
                        events.extend(adaptive_events)
                        stage_seconds["adaptive_deep_read"] = round(time.perf_counter() - t_adapt, 3)
                        heuristic_adaptive = evidence_mod.extract_all(question, added_red, max_total=max_evidence)
                        readable_added = {a.source_id for a in adaptive_analyses}
                        readable_sources = [s for s in added_red if s.id in readable_added]
                        for source in added_red:
                            if source.id not in readable_added:
                                for decision in reversed(decisions):
                                    if decision.source_id == source.id:
                                        decision.state = "no_read"
                                        decision.rationale = "Gap-search candidate crossed the relevance gate but was not admitted because a verified full-text deep read was unavailable."
                                        break
                        selected.extend(readable_sources)
                        selected = list(dict((s.dedupe_key(), s) for s in selected).values())
                        adaptive_added_count = len(readable_sources)
                        analyses.extend(adaptive_analyses)
                        evidence = evidence_mod.merge_evidence(
                            [evidence, heuristic_adaptive, adaptive_deep_evidence],
                            max_total=max_evidence + len(deep_evidence) + len(adaptive_deep_evidence),
                        )
                        deep_read_mod.attach_quantitative_evidence_ids(analyses, evidence)
                        deep_read_mod.attach_structured_evidence_ids(analyses, evidence)
                        events.append(RunEvent(stage="adaptive_read_complete", detail=f"{len(adaptive_analyses)} gap-search paper(s) deep-read and merged into the research memory"))
        if research_blueprint is not None and adaptive_added_count and self.llm.available:
            revised_blueprint, revised_blueprint_used = intelligence_mod.revise_research_blueprint(
                question, report_type, research_blueprint, analyses, [], self.llm
            )
            if revised_blueprint_used:
                research_blueprint = revised_blueprint
                blueprint_llm_used = True
                events.append(RunEvent(stage="research_blueprint_revision", detail=f"blueprint revised after adaptive evidence to v{research_blueprint.version}"))
        for source in candidates:
            source.role = "deep_read" if any(source.dedupe_key() == kept.dedupe_key() for kept in selected) else "no_read"
        sel = selection_mod.SelectionResult(
            deep_read=selected,
            no_read=[s for s in candidates if s not in selected],
            rejected=[s for s in candidates if s not in selected],
        )
        raw_sources_list = candidates
        incomplete_selected = [s for s in selected if not (s.full_text.strip() and s.full_text_parsed)]
        if incomplete_selected and not any("not verified as full-text readable" in w for w in warnings):
            warnings.append(f"{len(incomplete_selected)} RED paper(s) could not be verified as full-text readable. They are not counted as deep reads; any abstract evidence is explicitly treated as incomplete.")
        fast_multi_report = bool(getattr(settings, "universal_fast_report", False)) and len(selected) > 1 and bool(self.llm.available)
        if settings.report_style == "feynman" or fast_multi_report or (len(selected) == 1 and len(analyses) == 1):
            from .research_knowledge import _deterministic as _deterministic_knowledge
            research_knowledge_model = _deterministic_knowledge(question, selected, analyses)
            knowledge_used = False
        else:
            research_knowledge_model, knowledge_used = build_research_knowledge(
                question, selected, analyses, evidence, self.llm
            )
        events.append(RunEvent(
            stage="research_knowledge",
            detail=f"shared knowledge model built; llm={knowledge_used}; targeted_rereads={len(research_knowledge_model.get('targeted_reread_evidence', []))}",
            ok=bool(research_knowledge_model),
        ))
        graph_started = time.perf_counter()
        feynman_mode = settings.report_style == "feynman" or ultra_fast_requested
        if feynman_mode:
            graph_state = self._feynman_post_reading(
                question, report_type, report_title, selected, analyses, evidence,
                knowledge_model=research_knowledge_model,
            )
        else:
          graph_state = run_report_research_graph(
              question=question, report_type=report_type, analysis_depth=analysis_depth,
              intent=intent, selected=selected, analyses=analyses, evidence=evidence,
              plan=plan, warnings=warnings, report_title=report_title, rejected_count=len(sel.rejected),
              llm=self.llm,
              cross_fn=intelligence_mod.cross_paper_reasoning,
              plan_fn=lambda q, rt, d, it, findings, ans, llm, council=None, knowledge_model=None: intelligence_mod.build_synthesis_plan(q, rt, d, it, findings, ans, llm, council=council, blueprint=research_blueprint, knowledge_model=knowledge_model),
              claims_fn=claims_mod.make_claims,
              normalize_fn=evidence_norm_mod.normalize_evidence,
              draft_fn=drafting_mod.draft_sections,
              render_fn=synthesizer.build_professional_report,
              audit_fn=intelligence_mod.audit_report,
              research_knowledge_model=research_knowledge_model,
          )
        universal_fast_report = bool(graph_state.get("universal_fast_report"))
        cross_findings = graph_state.get("findings", [])
        cross_llm_used = bool(graph_state.get("cross_used"))
        synthesis_plan = graph_state.get("synthesis_plan")
        synthesis_plan_llm_used = bool(graph_state.get("plan_used"))
        claims = graph_state.get("claims", [])
        claims_llm_used = bool(graph_state.get("claims_used"))
        atomic_evidence = graph_state.get("atomic_evidence", [])
        paper_evidence_maps = graph_state.get("paper_evidence_maps", [])
        draft_sections = graph_state.get("draft_sections", [])
        drafting_llm_used = bool(graph_state.get("draft_used"))
        report_md = graph_state.get("report_markdown", "")
        research_knowledge_model = graph_state.get("research_knowledge_model") or research_knowledge_model or {}
        if feynman_mode or universal_fast_report:
            grounding_repair_used, grounding_repair_count = False, 0
        else:
            report_md, grounding_repair_used, grounding_repair_count = research_quality_mod.repair_report_grounding(
                report_md, evidence, selected, self.llm
            )
        if grounding_repair_used:
            events.append(RunEvent(stage="grounding_repair", detail=f"restored explicit evidence grounding in {grounding_repair_count} substantive paragraph(s)", ok=True))
        synth_llm_used = bool(graph_state.get("render_used")) or grounding_repair_used
        report_audit = graph_state.get("audit")
        audit_llm_used = bool(graph_state.get("audit_used"))
        peer_review = graph_state.get("peer_review") or {}
        editorial_review = graph_state.get("editorial_review") or {}
        research_council = graph_state.get("council") or {}
        research_director = graph_state.get("research_director") or {}
        selected_specialists = list(graph_state.get("selected_specialists") or research_council.get("selected_specialists") or [])
        director_used = bool(graph_state.get("director_used"))
        peer_review_used = bool(graph_state.get("peer_review_used"))
        peer_issues = []
        if isinstance(peer_review, dict):
            for bucket in ("critical_issues", "missing_insights", "weak_sections", "redundancies", "unsupported_or_overstated", "comparison_problems", "required_revisions", "important_findings_not_integrated"):
                peer_issues.extend(peer_review.get(bucket) or [])
        if isinstance(editorial_review, dict):
            for bucket in ("synthesis_problems", "unsupported_or_overstated", "important_findings_not_integrated", "paper_summary_sections", "repetitions", "weak_transitions", "comparison_problems", "conclusion_problems"):
                peer_issues.extend(editorial_review.get(bucket) or [])
        if report_audit is not None and peer_issues:
            major_peer = [x for x in peer_issues if isinstance(x, dict) and str(x.get("severity", "major")).lower() in {"critical", "major"}]
            if major_peer:
                report_audit.structural_issues.append(f"Academic peer review identified {len(major_peer)} major/critical reasoning issue(s); targeted revision required.")
                report_audit.passed = False
        paper_reading_plans = intelligence_mod.build_paper_reading_plans(report_type, selected, intent)
        comparison_records = intelligence_mod.build_comparison_records(selected, analyses, evidence)
        gap_records = intelligence_mod.build_gap_records(analyses, cross_findings)
        graph_elapsed = round(time.perf_counter() - graph_started, 3)
        fast_timing = (graph_state.get("fast_report_info") or {}).get("timing") if isinstance(graph_state, dict) else None
        if isinstance(fast_timing, dict):
            stage_seconds["report_reasoning"] = float(fast_timing.get("reasoning_seconds", 0.0) or 0.0)
            stage_seconds["report_writers"] = float(fast_timing.get("writer_seconds", 0.0) or 0.0)
            stage_seconds["report_assembly"] = max(0.0, round(graph_elapsed - stage_seconds["report_reasoning"] - stage_seconds["report_writers"], 3))
        else:
            stage_seconds["report_graph"] = graph_elapsed
        stage_seconds["synthesize"] = graph_elapsed
        events.append(RunEvent(stage="research_graph", detail=f"post-reading research graph completed with {len(cross_findings)} cross-paper findings, {len(draft_sections)} section drafts, langgraph={graph_state.get('using_langgraph', False)}"))
        revision_cycles = 0
        revision_llm_used = False
        peer_has_issues = (
            any(bool(peer_review.get(k)) for k in ("critical_issues", "missing_insights", "weak_sections", "redundancies", "unsupported_or_overstated", "comparison_problems", "required_revisions", "important_findings_not_integrated")) if isinstance(peer_review, dict) else False
        ) or (
            any(bool(editorial_review.get(k)) for k in ("synthesis_problems", "unsupported_or_overstated", "important_findings_not_integrated", "paper_summary_sections", "repetitions", "weak_transitions", "comparison_problems", "conclusion_problems")) if isinstance(editorial_review, dict) else False
        )
        if draft_sections and (not report_audit.passed or peer_has_issues):
            audit_items: dict[str, list[str]] = {}
            for issue_bucket in ("critical_issues", "missing_insights", "weak_sections", "redundancies", "unsupported_or_overstated", "comparison_problems", "required_revisions", "synthesis_problems", "important_findings_not_integrated", "paper_summary_sections", "repetitions", "weak_transitions", "conclusion_problems"):
                for issue in peer_review.get(issue_bucket, []) if isinstance(peer_review, dict) else []:
                    if not isinstance(issue, dict) or not issue.get("revision"):
                        continue
                    title = str(issue.get("section") or "")
                    for section in draft_sections:
                        if title and title.lower() not in section.title.lower():
                            continue
                        audit_items.setdefault(section.id, []).append("PEER REVIEW: " + str(issue.get("revision")))
                        break
            for item in report_audit.claim_audits:
                if item.status in {"unsupported", "overstated", "partial", "not_comparable"} and item.correction:
                    for section in draft_sections:
                        if item.claim[:80] in section.markdown or not audit_items:
                            audit_items.setdefault(section.id, []).append(item.correction)
                            break
            revised_sections, revision_cycles, revision_llm_used = drafting_mod.revise_sections(draft_sections, audit_items, evidence, self.llm, max_cycles=2)
            if revision_llm_used:
                report_md, synth2 = synthesizer.build_professional_report(
                    question, report_title, report_type, plan, selected, evidence, claims, warnings,
                    llm=self.llm, analyses=analyses, rejected_count=len(sel.rejected), analysis_depth=analysis_depth,
                    synthesis_plan=synthesis_plan, cross_findings=cross_findings, draft_sections=revised_sections,
                )
                report_md, repair2_used, repair2_count = research_quality_mod.repair_report_grounding(
                    report_md, evidence, selected, self.llm
                )
                if repair2_used:
                    events.append(RunEvent(stage="grounding_repair", detail=f"restored explicit evidence grounding in {repair2_count} revised paragraph(s)", ok=True))
                draft_sections = revised_sections
                report_audit, audit2 = intelligence_mod.audit_report(report_md, evidence, selected, claims, self.llm, report_type=report_type, analyses=analyses, synthesis_plan=synthesis_plan)
                audit_llm_used = audit_llm_used or audit2
                final_peer_review, final_peer_used = peer_review_report(question, report_type, report_md, analyses, synthesis_plan, self.llm)
                peer_review_used = peer_review_used or final_peer_used
                final_major = []
                if isinstance(final_peer_review, dict):
                    for bucket in ("critical_issues", "missing_insights", "weak_sections", "redundancies", "unsupported_or_overstated", "comparison_problems", "required_revisions", "important_findings_not_integrated"):
                        final_major.extend([x for x in (final_peer_review.get(bucket) or []) if isinstance(x, dict) and str(x.get("severity", "major")).lower() in {"critical", "major"}])
                if final_major:
                    report_audit.structural_issues.append(f"Final academic peer review still reports {len(final_major)} major/critical issue(s).")
                    report_audit.passed = False
                else:
                    peer_review = final_peer_review or peer_review
                events.append(RunEvent(stage="revision_complete", detail=f"targeted revision cycles={revision_cycles}, passed={report_audit.passed}, peer_review_issues={len(final_major)}", ok=report_audit.passed))
        if feynman_mode or universal_fast_report:
            final_report_md, final_repair_used, final_repair_count = report_md, False, 0
        else:
            final_report_md, final_repair_used, final_repair_count = research_quality_mod.repair_report_grounding(
                report_md, evidence, selected, self.llm
            )
        if final_repair_used:
            report_md = final_report_md
            events.append(RunEvent(stage="grounding_repair_final", detail=f"final provenance pass repaired {final_repair_count} substantive paragraph(s)", ok=True))
        if feynman_mode or universal_fast_report:
            final_audit, final_audit_used = report_audit, False
        else:
            final_audit, final_audit_used = intelligence_mod.audit_report(
                report_md, evidence, selected, claims, self.llm,
                report_type=report_type, analyses=analyses, synthesis_plan=synthesis_plan
            )
        report_audit = final_audit
        audit_llm_used = audit_llm_used or final_audit_used
        report_audit.revision_cycles = revision_cycles
        report_audit.paper_read_coverage = round(len(analyses) / len(selected), 3) if selected else 0.0
        report_audit.comparison_validity = round(max(0.0, 1.0 - report_audit.comparison_errors / max(1, len(report_audit.claim_audits))), 3)
        gap_findings = [f for f in cross_findings if f.finding_type == "gap"]
        report_audit.gap_support = 1.0 if not gap_findings or all(f.evidence_ids or f.source_ids for f in gap_findings) else 0.0
        if incomplete_selected and not feynman_mode:
            report_audit.structural_issues.append(f"{len(incomplete_selected)} RED source(s) were not verified as full-text readable")
            report_audit.passed = False
        final_repair_cycles = 0
        while not report_audit.passed and draft_sections and final_repair_cycles < settings.final_repair_cycles:
            final_repair_cycles += 1
            repair_items: dict[str, list[str]] = {}
            for item in report_audit.claim_audits:
                if item.status in {"unsupported", "overstated", "partial", "not_comparable"} and item.correction:
                    target = next((section for section in draft_sections if item.claim[:80] in section.markdown), draft_sections[0])
                    repair_items.setdefault(target.id, []).append(f"FINAL AUDIT: {item.correction}")
            if report_audit.structural_issues:
                repair_items.setdefault(draft_sections[0].id, []).extend(f"FINAL STRUCTURE: {x}" for x in report_audit.structural_issues[:8])
            revised, used = drafting_mod.revise_sections(draft_sections, repair_items, evidence, self.llm, max_cycles=1)
            if not used:
                break
            draft_sections = revised
            report_md, _ = synthesizer.build_professional_report(
                question, report_title, report_type, plan, selected, evidence, claims, warnings,
                llm=self.llm, analyses=analyses, rejected_count=len(sel.rejected), analysis_depth=analysis_depth,
                synthesis_plan=synthesis_plan, cross_findings=cross_findings, draft_sections=draft_sections,
            )
            report_md, repair_used, repair_count = research_quality_mod.repair_report_grounding(report_md, evidence, selected, self.llm)
            if repair_used:
                events.append(RunEvent(stage="final_repair", detail=f"final scholarly repair cycle {final_repair_cycles}: restored {repair_count} evidence links", ok=True))
            final_audit, final_audit_used = intelligence_mod.audit_report(
                report_md, evidence, selected, claims, self.llm, report_type=report_type, analyses=analyses, synthesis_plan=synthesis_plan
            )
            audit_llm_used = audit_llm_used or final_audit_used
            report_audit = final_audit
        report_audit.revision_cycles = revision_cycles + final_repair_cycles
        if settings.require_research_grade_report and not report_audit.passed:
            reason = report_audit.quality_gate_reason or "; ".join(report_audit.structural_issues[:3]) or "final scholarly quality audit failed"
            warnings.append(f"Final scholarly audit flagged issues (report delivered with caveats): {reason}")
        coverage_note = reading_rr.coverage_statement(paper_read_statuses)
        if coverage_note and report_md.strip() and "Evidence coverage" not in report_md.split("\n## ", 1)[0][:4000]:
            report_md = reading_rr.insert_coverage_note(report_md, coverage_note)
            warnings.append(coverage_note.splitlines()[0])
        if paper_read_statuses:
            events.append(RunEvent(stage="reading_coverage", detail="; ".join(f"{st.status}/L{st.level}:{st.source_id}" for st in paper_read_statuses), ok=all(st.verified for st in paper_read_statuses)))
        if getattr(settings, "enable_v79_architecture", False) and getattr(settings, "v79_fail_closed", True):
            budget_reason = self.llm.budget_exhausted()
            if budget_reason:
                warnings.append(f"LLM budget warning: {budget_reason}")
            if not selected:
                raise RuntimeError("No verified full-text papers entered the research corpus, so no evidence-grounded report can be generated.")
            if not analyses:
                raise RuntimeError("No paper-level deep reading completed, so ResearchOrbit could not build an evidence-grounded research report.")
            if not report_md.strip():
                raise RuntimeError("Report generation produced no deliverable.")
            if len(analyses) < len(selected):
                warnings.append(f"{len(selected) - len(analyses)} selected paper(s) did not complete deep reading and were excluded from synthesis.")
            if not bool(graph_state.get("render_used")) and not feynman_mode:
                warnings.append("Primary report renderer reported incomplete telemetry; the final markdown artifact was still produced and audited.")
            if not bool(audit_llm_used) and not feynman_mode:
                warnings.append("LLM publication audit was unavailable; deterministic evidence/provenance checks remain in effect.")
            if not report_audit.passed:
                reason = report_audit.quality_gate_reason or "; ".join(report_audit.structural_issues[:3]) or "final scholarly quality audit reported non-fatal issues"
                warnings.append(f"Scholarly quality audit reported caveats: {reason}")
        if report_type == "standard_report" and getattr(settings, "enable_v79_architecture", False) and self.llm.available:
            events.append(RunEvent(stage="creator_pack", detail="deferred to background after report publication", ok=True))
        atomic_fact_ledger = []
        for analysis in analyses:
            for item in getattr(analysis, "evidence_inventory", []) or []:
                if isinstance(item, dict):
                    atomic_fact_ledger.append({**item, "source_id": analysis.source_id})
        strength_label, strength_score, _ = synthesizer.evidence_strength(selected, evidence, analyses)
        stage_seconds["total"] = round(time.perf_counter() - run_started, 3)
        llm_used_overall = bool(plan_llm_used or intent_llm_used or cross_llm_used or synthesis_plan_llm_used or claims_llm_used or synth_llm_used or audit_llm_used or drafting_llm_used or revision_llm_used or adaptive_used or analyses or (precomputed_usage or {}).get("llm_calls", 0))
        usage_summary = self.llm.usage.summary()
        if precomputed_usage:
            usage_summary = {
                key: usage_summary.get(key, 0) + precomputed_usage.get(key, 0)
                for key in set(usage_summary) | set(precomputed_usage)
            }
        try:
            from .survey_frame import quality_metrics as _survey_metrics, self_challenge as _self_challenge
            _survey = _survey_metrics(report_md, selected, analyses)
            _survey.update(_self_challenge(report_md, question, plan, selected, analyses, cross_findings, report_audit,
                                           len(raw_sources_list), draft_sections))
        except Exception:
            _survey = {}
        metrics = {
            **_survey,
            "sources_found": len(raw_sources_list),
            "sources_selected": len(selected),
            "sources_explicitly_selected": len(selection_before_fulltext),
            "sources_selected_readable": len(selected),
            "sources_core": len(selected),
            "sources_deep_read": len(selected),
            "sources_no_read": max(0, len(raw_sources_list) - len(selected)),
            "sources_rejected": len(sel.rejected),
            "auto_selected_threshold": AUTO_RELEVANCE_THRESHOLD,
            "auto_selected_count": len(selected),
            "evidence_count": len(evidence),
            "claim_count": len(claims),
            "atomic_evidence_count": len(atomic_evidence),
            "paper_evidence_map_count": len(paper_evidence_maps),
            "paper_reading_plan_count": len(paper_reading_plans),
            "comparison_record_count": len(comparison_records),
            "gap_record_count": len(gap_records),
            "draft_section_count": len(draft_sections),
            "verification_failures": report_audit.unsupported_claims + report_audit.citation_errors,
            "revision_cycles": revision_cycles + final_repair_cycles,
            "citation_coverage": report_audit.citation_coverage,
            "evidence_coverage": report_audit.evidence_coverage,
            "paper_read_coverage": report_audit.paper_read_coverage,
            "comparison_validity": report_audit.comparison_validity,
            "gap_support": report_audit.gap_support,
            "structural_completeness": report_audit.structural_completeness,
            "paper_citation_coverage": report_audit.paper_citation_coverage,
            "section_coverage": report_audit.section_coverage,
            "evidence_density": report_audit.evidence_density,
            "quantitative_coverage": report_audit.quantitative_coverage,
            "research_usefulness": report_audit.research_usefulness,
            "papers_deep_read": len(analyses),
            "analysis_depth": analysis_depth,
            "papers_full_text": sum(1 for a in analyses if a.depth == "full_text"),
            "papers_pdf_acquired": sum(1 for s in selected if s.pdf_acquired),
            "papers_full_text_parsed": full_text_selected,
            "papers_incomplete_read": len(incomplete_selected),
            "reading_regime": "universal_fast_research_writer" if universal_fast_report else ("feynman_multistage_full_research" if feynman_mode else "professor_council_full_research"),
            "council_used": bool(research_council),
            "research_director_used": director_used,
            "specialist_agent_count": len(selected_specialists),
            "specialist_agents": ",".join(selected_specialists),
            "providers_used": len({s.provider for s in selected}),
            "search_queries_used": len(queries_tried),
            "adaptive_gap_search_used": adaptive_used,
            "adaptive_gap_count": adaptive_gap_count,
            "adaptive_added_count": adaptive_added_count,
            "adaptive_query_count": len(adaptive_queries),
            "v79_architecture": bool(getattr(settings, "enable_v79_architecture", False)),
            "v79_selected_papers": len(selected),
            "v79_covered_facets": sum(1 for c in v79_coverage if c.get("covered")),
            "v79_total_facets": len(v79_coverage),
            "research_blueprint_version": int(research_blueprint.version) if research_blueprint else 0,
            "research_blueprint_llm_used": blueprint_llm_used,
            "llm_available": self.llm.available,
            "llm_used": llm_used_overall,
            "evidence_strength": strength_label,
            "evidence_strength_score": strength_score,
            "total_runtime_seconds": stage_seconds["total"],
            **{f"stage_seconds_{k}": v for k, v in stage_seconds.items() if k != "total"},
            **usage_summary,
        }
        if ultra_fast_requested:
            self.llm.clear_run_deadline()
            self.llm.ultra_fast_override = False
        chat_memory_result = ResearchResult(
            run_id=run_id, question=question, report_title=report_title, case_name=report_title,
            report_type=report_type, version=version, parent_run_id=parent_run_id,
            build_mode=build_mode if build_mode in {"initial", "new", "combine"} else "initial",
            plan=plan, sources=selected, evidence=evidence, report_markdown=report_md,
            paper_analyses=analyses, research_knowledge_model=research_knowledge_model or {},
        )
        research_chat_memory = build_research_chat_memory(chat_memory_result)
        return ResearchResult(
            run_id=run_id,
            question=question,
            report_title=report_title,
            case_name=report_title,
            report_type=report_type,
            version=version,
            parent_run_id=parent_run_id,
            build_mode=build_mode if build_mode in {"initial", "new", "combine"} else "initial",
            added_source_ids=[s.id for s in (extra_sources or []) if any(s.dedupe_key() == kept.dedupe_key() for kept in selected)],
            plan=plan,
            sources=selected,
            evidence=evidence,
            claims=claims,
            warnings=warnings,
            events=events,
            report_markdown=report_md,
            llm_used=llm_used_overall,
            metrics=metrics,
            paper_analyses=analyses,
            paper_read_status=[st.to_dict() for st in paper_read_statuses],
            research_intent=intent,
            research_knowledge_model=research_knowledge_model,
            report_type_analysis={"report_type": report_type, "lens": report_type},
            research_blueprint=research_blueprint,
            evidence_requirements=evidence_requirements,
            cross_paper_findings=cross_findings,
            research_council=research_council,
            research_director=research_director,
            synthesis_plan=synthesis_plan,
            report_audit=report_audit,
            retrieval_plan=retrieval_plan,
            paper_reading_plans=paper_reading_plans,
            reading_strategy=ResearchReadingStrategy.model_validate(reading_strategy),
            paper_evidence_maps=paper_evidence_maps,
            atomic_evidence=atomic_evidence,
            comparison_records=comparison_records,
            gap_records=gap_records,
            corpus_decisions=decisions,
            retrieval_trace=v79_retrieval_trace,
            coverage_table=v79_coverage,
            atomic_fact_ledger=atomic_fact_ledger,
            creator_pack=creator_pack,
            draft_sections=draft_sections,
        )
    def discover_candidates(
        self,
        question: str,
        max_iterations: int | None = None,
        candidate_limit: int = 15,
        report_type: str = "literature_review",
        report_title: str = "",
        cancel_check=None,
    ):
        """Run the reliable, bounded paper-discovery phase.
        Important product boundary:
        discovery must be able to produce a real arXiv corpus even when Groq is
        unavailable, rate-limited or temporarily unhealthy. The older v79 path
        spent several LLM calls *before the first arXiv request*, then mixed
        discovery with expensive verification. That made a provider problem look
        like a retrieval hang.
        This phase therefore uses the deterministic planner as its R0/R1 contract:
        the original question is always a recall anchor and report-type-specific
        query variants provide additional directions. arXiv is searched under one
        absolute wall-clock deadline. Ranking is deterministic and bounded here;
        Groq-based triage/reranking and whole-paper verification remain downstream
        in the actual research run where they improve quality without blocking the
        candidate list.
        """
        from .planner import make_plan
        from ..search.anytime import AnytimeRetrievalController
        question = (question or "").strip()
        if not question:
            raise ValueError("question must not be empty")
        requested_type = report_type or "literature_review"
        settings = self.llm.settings
        events: list[RunEvent] = []
        plan, planner_used = make_plan(
            question, llm=self.llm, report_type=requested_type, report_title=report_title
        )
        from ..search.query_strategy import build_discovery_queries
        plan.search_queries = build_discovery_queries(question, plan.search_queries, limit=10)
        from ..search.query_strategy import is_cross_view_geolocalization_query
        if is_cross_view_geolocalization_query(question):
            suffixes = []
        else:
            suffixes = {
                "standard_report": ["methods", "experimental evaluation", "limitations", "research gaps"],
                "literature_review": ["methods", "empirical evaluation", "limitations", "research gaps"],
                "state_of_art": ["recent advances", "benchmark evaluation", "approaches", "limitations"],
                "gap_analysis": ["limitations", "failure cases", "unresolved problems", "future work"],
                "comparison": ["method comparison", "benchmark evaluation", "ablation", "trade-offs"],
                "evidence_review": ["empirical evidence", "evaluation", "replication", "limitations"],
                "research_proposal": ["baseline", "evaluation", "failure modes", "open problems"],
                "paper_analysis": ["method", "experiments", "ablations", "limitations"],
            }.get(requested_type, ["methods", "evaluation", "limitations", "research gaps"])
        for suffix in suffixes:
            q = f"{question} {suffix}"[:420]
            if q.lower() not in {x.lower() for x in plan.search_queries}:
                plan.search_queries.append(q)
        events.append(RunEvent(
            stage="retrieval_R0_R1",
            detail=f"deterministic intent/query plan ready: {len(plan.search_queries)} query directions; Groq not required",
        ))
        tried: list[str] = []
        controller = AnytimeRetrievalController(self.aggregator)
        def _discover_followups():
            fresh = self._followup_queries(question, plan, tried)[:8]
            tried.extend(fresh)
            return fresh
        retrieval_budget = max(30, int(getattr(settings, "v79_discovery_budget_seconds", 600)))
        snapshot, retrieval_events = controller.run(
            question,
            plan.search_queries,
            deadline_seconds=retrieval_budget,
            max_papers=min(15, max(1, int(candidate_limit))),
            results_per_query=min(100, int(getattr(settings, "v80_arxiv_page_size", 60))),
            batch_size=4,
            followup_factory=_discover_followups,
            min_stable_rounds=2,
            candidate_pool_size=max(60, int(getattr(settings, "v80_candidate_reservoir", 240))),
            min_stability_seconds=float(getattr(settings, "v80_min_stability_seconds", 20)),
            cancel_check=cancel_check,
        )
        events.extend(retrieval_events)
        tried.extend(plan.search_queries)
        candidates = list(getattr(snapshot, "retrieval_pool", None) or snapshot.candidates)
        if candidates:
            try:
                from .ranker import rank_sources
                rank_sources(question, candidates)
            except Exception:
                candidates.sort(key=lambda s: (-float(getattr(s, "score", 0.0) or 0.0), s.title.lower()))
        geo_weak_ids: set[str] = set()
        try:
            from ..search.query_strategy import geo_localization_relevance, is_cross_view_geolocalization_query
            geo_intent = is_cross_view_geolocalization_query(question)
            for source in candidates:
                bonus, reasons = geo_localization_relevance(question, source.title, source.abstract)
                source.score = round(max(0.0, min(1.0, float(source.score or 0.0) + bonus)), 4)
                if reasons:
                    source.relevance_reasons = list(dict.fromkeys((source.relevance_reasons or []) + reasons))[:8]
                    if geo_intent and any(str(reason).startswith("Weak topical match") for reason in reasons):
                        geo_weak_ids.add(str(source.id))
        except Exception:
            geo_intent = False
        candidates = [s for s in candidates if has_readable_content(s)]
        if not candidates:
            failures = [e.detail for e in events if getattr(e, "ok", True) is False and getattr(e, "detail", "")]
            if failures:
                events.append(RunEvent(
                    stage="retrieval_empty",
                    detail="No usable arXiv papers were retained. " + " | ".join(failures[:3]),
                    ok=False,
                ))
            else:
                events.append(RunEvent(stage="retrieval_empty", detail="No usable arXiv papers were retained.", ok=False))
            return plan, [], events, False, self.llm.usage.summary()
        ranked_candidates = sorted(
            candidates, key=lambda s: (-float(getattr(s, "score", 0.0) or 0.0), s.title.lower())
        )
        try:
            from .semantic_rerank import semantic_rerank as _sem
            _slice = ranked_candidates[:100]
            if _sem(question, _slice):
                ranked_candidates = sorted(
                    candidates, key=lambda s: (-float(getattr(s, "score", 0.0) or 0.0), s.title.lower())
                )
                events.append(RunEvent(stage="retrieval_embedding", detail=f"embedding similarity applied to top {len(_slice)} of {len(candidates)} reservoir candidates"))
        except Exception:
            pass
        strict_relevance_gate_applied = False
        if self.llm.available and candidates:
            rerank_pool = ranked_candidates[:min(30, max(15, int(getattr(settings, "v80_llm_rerank_pool", 30))), len(ranked_candidates))]
            rerank_payload = {
                "question": question,
                "report_type": requested_type,
                "papers": [
                    {"source_id": s.id, "title": s.title, "abstract": (s.abstract or "")[:1800],
                     "retrieval_score": round(float(getattr(s, "score", 0.0) or 0.0), 4)}
                    for s in rerank_pool
                ],
            }
            rerank_system = (
                "You are a strict academic paper relevance judge, not a general topic matcher. "
                "Rank each supplied paper against the exact research question and report type using only its title and abstract. "
                "Score relevance_score from 0 to 100: 90-100 direct match to the task/method; 70-89 strongly useful adjacent evidence; "
                "50-69 partial background; below 50 weak or unrelated. Do not treat shared generic words as relevance. "
                "For cross-view geo-localization, ground-to-aerial/satellite matching, visual localization, video geolocation, "
                "distinguish direct matching/localization methods from generic computer vision, generic video understanding, "
                "or unrelated image classification. Exclude irrelevant papers instead of filling the list. "
                "Return JSON only as {\"ranking\":[{\"source_id\":\"...\",\"relevance_score\":0,\"exclude\":false,\"reason\":\"short evidence-based reason\"}]}. "
                "Use every supplied id at most once; do not invent ids."
            )
            try:
                ranked_json = self.llm.complete_json(
                    rerank_system, json.dumps(rerank_payload, ensure_ascii=False),
                    max_tokens=2500, tier="S", priority="retrieval"
                )
            except Exception:
                ranked_json = None
            if isinstance(ranked_json, dict):
                ranking_items = [x for x in (ranked_json.get("ranking") or []) if isinstance(x, dict)]
                by_id = {s.id: s for s in rerank_pool}
                valid_items = [x for x in ranking_items if x.get("source_id") in by_id]
                has_scores = bool(valid_items) and all(isinstance(x.get("relevance_score"), (int, float)) for x in valid_items)
                if has_scores:
                    strict_relevance_gate_applied = True
                    accepted = [x for x in valid_items if not x.get("exclude", False) and float(x["relevance_score"]) >= 48]
                    accepted_ids = [x["source_id"] for x in sorted(accepted, key=lambda x: float(x["relevance_score"]), reverse=True)]
                    if accepted_ids:
                        ranked_candidates = [by_id[i] for i in accepted_ids]
                    else:
                        ranked_candidates = []
                    accepted_by_id = {x["source_id"]: x for x in accepted}
                    for source in ranked_candidates:
                        item = accepted_by_id.get(source.id, {})
                        source.relevance_score = float(item.get("relevance_score", 0))
                        source.relevance_reasons = [str(item.get("reason", "Direct topical relevance assessed from title and abstract"))][:1]
                    events.append(RunEvent(stage="retrieval_llm_relevance_gate", detail=f"strict relevance screening retained {len(ranked_candidates)} of {len(rerank_pool)} candidates (threshold 48/100)"))
                else:
                    ids = [x.get("source_id") for x in valid_items]
                    ordered = [by_id[i] for i in ids if i in by_id]
                    ordered.extend(s for s in rerank_pool if s.id not in {x.id for x in ordered})
                    if ordered:
                        rank_pos = {s.id: i for i, s in enumerate(ordered)}
                        ranked_candidates = sorted(candidates, key=lambda s: (rank_pos.get(s.id, 999), -float(getattr(s, "score", 0.0) or 0.0), s.title.lower()))
                        for source in ranked_candidates[:15]:
                            source.relevance_reasons = [
                                str(x.get("reason", "LLM reranked for direct research relevance"))
                                for x in valid_items if x.get("source_id") == source.id
                            ][:1] or source.relevance_reasons
                        events.append(RunEvent(stage="retrieval_llm_rerank", detail=f"reranked top {len(rerank_pool)} metadata candidates with one bounded scholarly listwise pass"))
        if geo_intent and not strict_relevance_gate_applied:
            before_gate = len(ranked_candidates)
            ranked_candidates = [s for s in ranked_candidates if str(s.id) not in geo_weak_ids]
            events.append(RunEvent(
                stage="retrieval_domain_fallback_gate",
                detail=f"provider scoring unavailable; kept {len(ranked_candidates)} of {before_gate} candidates with a direct domain signal",
                ok=bool(ranked_candidates),
            ))
        eligible = [s for s in ranked_candidates if float(getattr(s, "score", 0.0) or 0.0) > 0.0]
        top15 = eligible[:15]
        auto_target = automatic_selection_target(top15, score_getter=lambda s: float(getattr(s, "score", 0.0) or 0.0))
        auto_target = min(auto_target, int(getattr(settings, "auto_selection_max", AUTO_MAX_PAPERS) or AUTO_MAX_PAPERS), len(top15))
        default_deep_ids = {s.id for s in top15[:auto_target]}
        reservoir_size = len(candidates)
        candidates = list(top15) if top15 else list(ranked_candidates[:max(1, min(15, int(candidate_limit)))])
        for source in candidates:
            source.role = "deep_read" if source.id in default_deep_ids else "no_read"
            raw_score = float(getattr(source, "score", 0.0) or 0.0)
            if not float(getattr(source, "relevance_score", 0.0) or 0.0):
                source.relevance_score = round(max(0.0, min(100.0, raw_score * 100.0)), 2)
            if not source.relevance_reasons:
                source.relevance_reasons = [
                    "Strong lexical alignment with the research question" if raw_score >= 0.5
                    else "Retrieved as a complementary evidence candidate"
                ]
        elapsed = snapshot.elapsed_seconds
        events.append(RunEvent(
            stage="retrieval_complete",
            detail=(
                f"arXiv discovery complete: Top-{len(candidates)} from a {reservoir_size}-paper reservoir, "
                f"{len(default_deep_ids)} default deep-read recommendations from Top-{len(top15)}, {snapshot.queries_attempted} query/page requests, "
                f"elapsed={elapsed:.1f}s, deadline_reached={snapshot.deadline_reached}"
            ),
        ))
        return plan, candidates, events, planner_used, self.llm.usage.summary()
    def _followup_queries(self, question: str, plan: ResearchPlan, already_tried: list[str]) -> list[str]:
        """Generate fresh retrieval directions for the anytime search window.
        The retrieval controller may call this repeatedly until its hard
        deadline. Each call therefore derives a different small batch from the
        plan instead of acting as a one-shot second search round.
        """
        tried = {" ".join(str(q).lower().split()) for q in already_tried}
        base = generate_followup_queries(question, plan, already_tried)
        from ..search.query_strategy import is_cross_view_geolocalization_query, build_discovery_queries
        if is_cross_view_geolocalization_query(question):
            expansions = build_discovery_queries(question, [], limit=10)[:-1]
            expansions += [
                'all:"cross-view geolocation" AND all:dataset',
                'all:"video localization" AND all:satellite',
                'all:"camera pose" AND all:"ground-to-satellite"',
            ]
        else:
            expansions = [
                "mechanism " + question,
                "evidence " + question,
                "empirical results " + question,
                "limitations " + question,
                "comparison " + question,
                "benchmark evaluation " + question,
                "failure cases " + question,
                "generalization " + question,
                "uncertainty " + question,
                "state of the art " + question,
                "systematic review " + question,
                "recent advances " + question,
            ]
        for axis in list(plan.analysis_axes or [])[:12]:
            expansions.append(f"{question} {axis}")
        candidates = base + expansions
        fresh = []
        for q in candidates:
            q = " ".join(str(q).split()).strip()[:420]
            key = q.lower()
            if q and key not in tried and key not in {x.lower() for x in fresh}:
                fresh.append(q)
        return fresh[:8]
    def _feynman_post_reading(self, question, report_type, report_title, selected, analyses, evidence, knowledge_model=None) -> dict:
        """Lean post-reading stage for the Feynman report style.
        The legacy graph runs a council, debate, peer review and several audits over a
        survey-style draft. The Feynman report needs none of that: it writes plain-language
        topics straight from the papers' own notes and passages. Claims and normalised
        evidence are still produced because chat and claim-verification use them.
        """
        ultra = bool(getattr(self.llm, "ultra_fast_override", False) or getattr(getattr(self.llm, "settings", None), "ultra_fast_report", False))
        knowledge_used = bool(knowledge_model)
        knowledge_model = knowledge_model or {}
        atomic_evidence, paper_evidence_maps = evidence_norm_mod.normalize_evidence(evidence, analyses)
        report_md = ""
        info = {}
        claims, claims_used = ([], False)
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="researchpilot-postread") as pool:
            report_future = pool.submit(
                feynman_mod.build_feynman_report,
                question, report_type, report_title, selected, analyses, evidence, self.llm,
                knowledge_model=knowledge_model,
            )
            if not ultra:
                claims_future = pool.submit(claims_mod.make_claims, question, evidence, llm=self.llm)
            else:
                claims_future = None
            report_md, info = report_future.result()
            if claims_future is not None:
                claims, claims_used = claims_future.result()
        return {
            "findings": [], "cross_used": False, "synthesis_plan": None, "plan_used": False,
            "claims": claims, "claims_used": claims_used, "atomic_evidence": atomic_evidence, "research_knowledge_model": knowledge_model, "knowledge_used": knowledge_used,
            "paper_evidence_maps": paper_evidence_maps, "draft_sections": [], "draft_used": False,
            "report_markdown": report_md, "render_used": bool(info.get("llm_used")),
            "audit": ReportAudit(passed=True), "audit_used": False, "peer_review": {}, "peer_review_used": False,
            "editorial_review": {}, "council": {}, "research_director": {}, "selected_specialists": [],
            "director_used": False, "using_langgraph": False,
        }
    def ask_followup(self, result: ResearchResult, question: str, depth: str = "compact", cancel_check=None) -> FollowupQA:
        """Answer a follow-up from the current research case.
        Verification is only activated when the user explicitly asks for it.
        General-literature verification uses a fresh bounded search; corpus
        verification stays inside the selected research papers.
        """
        q = question.strip()
        if cancel_check is not None and cancel_check():
            raise RuntimeError("Chat cancelled")
        chat_sources, chat_evidence = self._chat_context(result)
        chat_llm = self.chat_llm
        try:
            chat_llm.research_case = result
        except Exception:
            pass
        qa = answer_followup(q, chat_evidence, chat_sources, llm=chat_llm, analyses=result.paper_analyses, claims=result.claims, report_markdown=result.report_markdown, prior_followups=result.followups, prior_verifications=result.verifications, depth=depth, report_title=result.report_title, cancel_check=cancel_check, chat_memory=result.research_chat_memory)
        result.followups.append(qa)
        result.conversation.append(ConversationTurn(mode="ask", question=qa.question, answer=qa.answer, evidence_ids=qa.evidence_ids, source_ids=qa.source_ids, llm_used=qa.llm_used, references=qa.references, needs_clarification=qa.needs_clarification, clarification_options=qa.clarification_options))
        return qa
    def ask_about_upload(
        self,
        result: ResearchResult,
        question: str,
        uploaded_source: Source,
        cached_analysis: PaperAnalysis | None = None,
    ) -> tuple[FollowupQA, PaperAnalysis | None]:
        """Targeted routing for a follow-up that clearly refers to an
        uploaded paper ("analyze this paper", "compare this with CRR") --
        see `pipeline/targeting.py` for why this needs to bypass the
        generic grounded-QA path entirely rather than treating the upload
        as just more evidence to keyword-search over. `cached_analysis`,
        when supplied (e.g. from a previous call's second return value),
        skips re-deep-reading the same upload on every follow-up turn.
        Appends the Q&A to `result.followups` like `ask_followup` does."""
        named = targeting_mod.find_named_source(question, result.sources)
        qa, analysis = targeting_mod.answer_about_uploaded_paper(
            question, uploaded_source, named, result.sources, result.evidence,
            llm=self.llm, existing_analysis=cached_analysis,
        )
        result.followups.append(qa)
        result.conversation.append(ConversationTurn(mode="ask", question=qa.question, answer=qa.answer, evidence_ids=qa.evidence_ids, llm_used=qa.llm_used))
        return qa, analysis
    def verify_claim(
        self,
        claim: str,
        result: ResearchResult | None = None,
        allow_new_search: bool = True,
        cancel_check=None,
    ) -> ClaimVerification:
        """Check a claim against `result`'s evidence (if given) and,
        when that's thin, a fresh bounded search. See `pipeline/verify.py`
        for the full policy. Appends the outcome to `result.verifications`
        when a result is supplied, so it's visible in that run's history."""
        existing_sources = result.sources if result else []
        existing_evidence = result.evidence if result else []
        verification = verify_mod.verify_claim(
            claim,
            existing_sources=existing_sources,
            existing_evidence=existing_evidence,
            llm=self.chat_llm,
            aggregator=self.aggregator,
            allow_new_search=allow_new_search,
            analyses=result.paper_analyses if result else [],
            verification_evidence_offset=(
                sum(len(v.new_evidence) for v in result.verifications) if result else 0
            ),
            cancel_check=cancel_check,
        )
        if result is not None:
            result.verifications.append(verification)
            result.conversation.append(ConversationTurn(mode="verify", question=claim, answer=verification.reasoning, evidence_ids=verification.evidence_ids, source_ids=[s.id for s in verification.new_sources], llm_used=verification.llm_used))
        return verification
    def add_sources_and_reanalyze(
        self, result: ResearchResult, new_sources: list[Source], deep_read_top_k: int | None = None
    ) -> ResearchResult:
        """"I extracted the top 5 papers, now let the user add more and
        re-process" -- merges `new_sources` (e.g. from
        `fulltext.source_from_uploaded_pdf`) into an existing result's
        source pool, deep-reads the new ones, and rebuilds evidence/claims/
        the report from the combined pool. Returns a NEW `ResearchResult`
        (fresh `run_id`) so the original run stays intact in history;
        `sources_found`/search-related metrics are carried over from the
        original run since no new search is performed here.
        """
        settings = self.llm.settings
        events: list[RunEvent] = list(result.events)
        existing_ids = {s.dedupe_key() for s in result.sources}
        genuinely_new: list[Source] = []
        for source in new_sources:
            key = source.dedupe_key()
            if key in existing_ids:
                continue
            existing_ids.add(key)
            genuinely_new.append(source)
        combined_sources = result.sources + genuinely_new
        t0 = time.perf_counter()
        events.extend(enrich_with_full_text(genuinely_new))
        readable_new = [
            s for s in genuinely_new
            if s.full_text.strip() and s.full_text_parsed and bool(getattr(s, "full_text_complete", True))
        ]
        unreadable_new = [s for s in genuinely_new if s not in readable_new]
        if unreadable_new:
            events.append(RunEvent(stage="reanalyze_fulltext_gate", detail=f"{len(unreadable_new)} uploaded paper(s) were excluded from the new deep-read corpus because their PDF text was not fully readable", ok=False))
        for source in unreadable_new:
            source.role = "no_read"
            events.append(RunEvent(stage="reanalyze_fulltext_gate", detail=f"excluded uploaded paper '{source.title[:60]}' because verified complete text extraction failed", ok=False))
        combined_sources = result.sources + readable_new
        heuristic_evidence = evidence_mod.extract_all(result.question, combined_sources, max_total=settings.max_evidence)
        existing_analyses = list(result.paper_analyses)
        effective_deep_read_top_k = len(readable_new) if deep_read_top_k is None else max(0, int(deep_read_top_k))
        reading_strategy = result.reading_strategy.model_dump() if result.reading_strategy else intelligence_mod.build_reading_strategy(result.question, result.report_type, result.research_intent, self.llm)[0]
        new_analyses, new_deep_evidence, deep_events = deep_read_mod.analyze_papers(
            result.question, readable_new, llm=self.llm, top_k=effective_deep_read_top_k, force=True, analysis_mode="deep",
            report_type=result.report_type, reading_strategy=reading_strategy, dossier_cache=self.dossier_cache
        )
        events.extend(deep_events)
        analyses = existing_analyses + new_analyses
        old_deep_evidence = [e for e in result.evidence if "deep-read" in e.locator]
        evidence = evidence_mod.merge_evidence(
            [heuristic_evidence, old_deep_evidence, new_deep_evidence],
            max_total=settings.max_evidence + len(old_deep_evidence) + len(new_deep_evidence),
        )
        analysis_alignment = validate_analyses(result.question, combined_sources, analyses, self.llm)
        bad_analysis_ids = {a.source_id for a in analysis_alignment if not a.aligned}
        if bad_analysis_ids:
            rescued_analyses, rescued_sources, rescued_ids = rescue_analysis_corpus(
                result.question, combined_sources, analyses, bad_analysis_ids
            )
            analyses = rescued_analyses
            admitted_ids = {a.source_id for a in analyses}
            evidence = [e for e in evidence if e.source_id in admitted_ids]
            selected = [s for s in combined_sources if s.id in admitted_ids]
            dropped_ids = bad_analysis_ids - set(rescued_ids)
            events.append(RunEvent(
                stage="deep_read_integrity",
                detail=(f"post-reading coherence review flagged {len(bad_analysis_ids)} dossier(s); "
                        f"rescued {len(rescued_ids)} on independent relevance/evidence signals; "
                        f"removed {len(dropped_ids)} without corroborating alignment"),
                ok=bool(analyses),
            ))
            if rescued_ids:
                warnings = list(result.warnings) + [
                    f"{len(rescued_ids)} re-analyzed paper(s) were retained after conservative post-reading alignment review because independent corpus relevance and evidence supported inclusion."
                ]
            if not analyses:
                raise ValueError(
                    "Deep reading produced no independently supported paper-level evidence for the research question. "
                    "Broaden the corpus before generating the report."
                )
        deep_read_mod.attach_quantitative_evidence_ids(analyses, evidence)
        deep_read_mod.attach_structured_evidence_ids(analyses, evidence)
        events.append(RunEvent(stage="reanalyze", detail=f"added {len(genuinely_new)} user-supplied source(s); {len(evidence)} evidence total"))
        depth_mode = "deep" if result.metrics.get("analysis_depth") == "deep" else "standard"
        intent = result.research_intent or intelligence_mod.build_intent(result.question, result.report_type, depth_mode, self.llm)[0]
        _fey = settings.report_style == "feynman"
        cross_findings, cross_llm_used = ([], False) if _fey else intelligence_mod.cross_paper_reasoning(result.question, result.report_type, depth_mode, analyses, evidence, combined_sources, self.llm)
        research_council = {}
        synthesis_plan, synthesis_plan_llm_used = (None, False) if _fey else intelligence_mod.build_synthesis_plan(result.question, result.report_type, depth_mode, intent, cross_findings, analyses, self.llm)
        paper_reading_plans = intelligence_mod.build_paper_reading_plans(result.report_type, combined_sources, intent)
        comparison_records = intelligence_mod.build_comparison_records(combined_sources, analyses, evidence)
        gap_records = intelligence_mod.build_gap_records(analyses, cross_findings)
        claims, claims_llm_used = claims_mod.make_claims(result.question, evidence, llm=self.llm)
        atomic_evidence, paper_evidence_maps = evidence_norm_mod.normalize_evidence(evidence, analyses)
        knowledge_model, _knowledge_used = build_research_knowledge(result.question, combined_sources, analyses, evidence, self.llm)
        research_knowledge_model = knowledge_model
        if _fey:
            draft_sections, drafting_llm_used = [], False
            report_md, _finfo = feynman_mod.build_feynman_report(result.question, result.report_type, result.report_title, combined_sources, analyses, evidence, self.llm, knowledge_model=knowledge_model)
            synth_llm_used = bool(_finfo.get("llm_used"))
            report_audit, audit_llm_used = ReportAudit(passed=True), False
        else:
            draft_sections, drafting_llm_used = drafting_mod.draft_sections(
                result.question, result.report_type, depth_mode, synthesis_plan, evidence, analyses, cross_findings, self.llm
            )
            report_md, synth_llm_used = synthesizer.build_professional_report(
                result.question, result.report_title, result.report_type, result.plan, combined_sources, evidence, claims,
                result.warnings, llm=self.llm, analyses=analyses, analysis_depth=depth_mode, synthesis_plan=synthesis_plan, cross_findings=cross_findings, draft_sections=draft_sections
            )
            report_audit, audit_llm_used = intelligence_mod.audit_report(report_md, evidence, combined_sources, claims, self.llm, report_type=result.report_type, analyses=analyses, synthesis_plan=synthesis_plan)
        revision_cycles = 0
        revision_llm_used = False
        if draft_sections and not report_audit.passed:
            audit_items = {}
            for item in report_audit.claim_audits:
                if item.status in {"unsupported", "overstated", "partial", "not_comparable"} and item.correction:
                    for section in draft_sections:
                        if item.claim[:80] in section.markdown or not audit_items:
                            audit_items.setdefault(section.id, []).append(item.correction)
                            break
            draft_sections, revision_cycles, revision_llm_used = drafting_mod.revise_sections(draft_sections, audit_items, evidence, self.llm, max_cycles=2)
            if revision_llm_used:
                report_md, synth2 = synthesizer.build_professional_report(
                    result.question, result.report_title, result.report_type, result.plan, combined_sources, evidence, claims,
                    result.warnings, llm=self.llm, analyses=analyses, analysis_depth=depth_mode, synthesis_plan=synthesis_plan, cross_findings=cross_findings, draft_sections=draft_sections
                )
                report_audit, audit2 = intelligence_mod.audit_report(report_md, evidence, combined_sources, claims, self.llm, report_type=result.report_type, analyses=analyses, synthesis_plan=synthesis_plan)
                audit_llm_used = audit_llm_used or audit2
        report_audit.revision_cycles = revision_cycles
        red_count = sum(1 for source in combined_sources if source.role == "deep_read")
        incomplete_selected = [source for source in combined_sources if source.role == "deep_read" and not (source.full_text.strip() and source.full_text_parsed and bool(getattr(source, "full_text_complete", True)))]
        report_audit.paper_read_coverage = round(len(analyses) / red_count, 3) if red_count else 0.0
        report_audit.comparison_validity = round(max(0.0, 1.0 - report_audit.comparison_errors / max(1, len(report_audit.claim_audits))), 3)
        gap_findings = [f for f in cross_findings if f.finding_type == "gap"]
        report_audit.gap_support = 1.0 if not gap_findings or all(f.evidence_ids or f.source_ids for f in gap_findings) else 0.0
        if incomplete_selected and not _fey:
            report_audit.structural_issues.append(f"{len(incomplete_selected)} RED source(s) were not verified as full-text readable")
            report_audit.passed = False
        strength_label, strength_score, _ = synthesizer.evidence_strength(combined_sources, evidence, analyses)
        full_text_selected = sum(1 for source in combined_sources if source.full_text.strip() and source.full_text_parsed and bool(getattr(source, "full_text_complete", True)))
        incomplete_selected = [source for source in combined_sources if source.role == "deep_read" and not (source.full_text.strip() and source.full_text_parsed and bool(getattr(source, "full_text_complete", True)))]
        elapsed = round(time.perf_counter() - t0, 3)
        research_council = result.research_council
        research_director = result.research_director
        metrics = dict(result.metrics)
        metrics.update(
            {
                "sources_selected": len(combined_sources),
                "evidence_count": len(evidence),
                "claim_count": len(claims),
                "atomic_evidence_count": len(atomic_evidence),
                "paper_evidence_map_count": len(paper_evidence_maps),
                "paper_reading_plan_count": len(paper_reading_plans),
                "comparison_record_count": len(comparison_records),
                "gap_record_count": len(gap_records),
                "draft_section_count": len(draft_sections),
                "verification_failures": report_audit.unsupported_claims + report_audit.citation_errors,
                "revision_cycles": revision_cycles,
                "papers_deep_read": len(analyses),
                "analysis_depth": "deep" if result.metrics.get("analysis_depth") == "deep" else "standard",
                "papers_full_text": sum(1 for a in analyses if a.depth == "full_text"),
            "papers_pdf_acquired": sum(1 for s in combined_sources if s.pdf_acquired),
            "papers_full_text_parsed": full_text_selected,
            "papers_incomplete_read": len(incomplete_selected),
            "reading_regime": "feynman_multistage_full_research" if _fey else "professor_council_full_research",
            "council_used": bool(research_council),
            "research_director_used": result.metrics.get("research_director_used", False),
            "specialist_agent_count": result.metrics.get("specialist_agent_count", 0),
            "specialist_agents": result.metrics.get("specialist_agents", ""),
                "evidence_strength": strength_label,
                "evidence_strength_score": strength_score,
                "stage_seconds_reanalyze": elapsed,
                **self.llm.usage.summary(),
            }
        )
        reanalysis_memory_seed = ResearchResult(
            run_id=uuid.uuid4().hex[:10], question=result.question, report_title=result.report_title,
            report_type=result.report_type, plan=result.plan, sources=combined_sources, evidence=evidence,
            report_markdown=report_md, paper_analyses=analyses, research_knowledge_model=research_knowledge_model or {},
        )
        research_chat_memory = build_research_chat_memory(reanalysis_memory_seed)
        if getattr(settings, "enable_v79_architecture", False) and getattr(settings, "v79_fail_closed", True):
            if not combined_sources or not analyses or not report_md.strip():
                raise RuntimeError("Reanalysis could not produce an evidence-grounded deliverable because no complete deep-read evidence survived the pipeline.")
            if len(analyses) < len(combined_sources):
                result.warnings.append(f"{len(combined_sources) - len(analyses)} combined paper(s) were excluded after deep-reading failures.")
            if not report_audit.passed:
                result.warnings.append("Reanalysis scholarly audit reported caveats; the grounded artifact was still produced.")
        return ResearchResult(
            run_id=uuid.uuid4().hex[:10],
            question=result.question,
            report_title=result.report_title,
            case_name=result.case_name or result.report_title,
            pinned=result.pinned,
            report_type=result.report_type,
            version=result.version + 1,
            parent_run_id=result.run_id,
            build_mode="combine",
            added_source_ids=[s.id for s in readable_new],
            plan=result.plan,
            sources=combined_sources,
            evidence=evidence,
            claims=claims,
            warnings=result.warnings,
            events=events,
            report_markdown=report_md,
            llm_used=result.llm_used or claims_llm_used or synth_llm_used or cross_llm_used or synthesis_plan_llm_used or audit_llm_used or drafting_llm_used or revision_llm_used or bool(new_analyses),
            metrics=metrics,
            paper_analyses=analyses,
            followups=list(result.followups),
            conversation=list(result.conversation),
            verifications=list(result.verifications),
            research_intent=intent,
            research_knowledge_model=research_knowledge_model,
            research_chat_memory=research_chat_memory,
            report_type_analysis={"report_type": result.report_type, "lens": result.report_type},
            evidence_requirements=list(result.evidence_requirements),
            cross_paper_findings=cross_findings,
            research_council=research_council,
            research_director=research_director,
            synthesis_plan=synthesis_plan,
            report_audit=report_audit,
            retrieval_plan=result.retrieval_plan,
            paper_reading_plans=paper_reading_plans,
            reading_strategy=ResearchReadingStrategy.model_validate(reading_strategy),
            paper_evidence_maps=paper_evidence_maps,
            atomic_evidence=atomic_evidence,
            comparison_records=comparison_records,
            gap_records=gap_records,
            corpus_decisions=list(getattr(result, "corpus_decisions", [])),
            draft_sections=draft_sections,
        )
