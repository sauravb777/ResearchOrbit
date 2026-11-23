from __future__ import annotations
import hashlib
import html
import sys
import tempfile
import uuid
from pathlib import Path
_SRC = Path(__file__).parent / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
import streamlit as st
from researchpilot.config import get_settings
from researchpilot.export import (markdown_to_docx, markdown_to_pdf,
                                  sources_to_bibtex)
from researchpilot.pipeline.fulltext import source_from_uploaded_pdf
from researchpilot.pipeline.grounder import Grounder
from researchpilot.pipeline.orchestrator import ResearchAgent
from researchpilot.pipeline.planner import (REPORT_TYPE_LABELS,
                                            report_type_for_question,
                                            suggest_report_title)
from researchpilot.pipeline.synthesizer import build_evidence_audit_report
from researchpilot.pipeline.targeting import question_targets_upload
from researchpilot.storage import ResultStore
st.set_page_config(page_title="ResearchOrbit", page_icon="🪐", layout="wide", initial_sidebar_state="expanded")
settings = get_settings()
store = ResultStore(settings.db_path)
NEW_REPORT_TYPES = [
    "literature_review",
    "state_of_art",
    "gap_analysis",
    "comparison",
    "evidence_review",
    "research_proposal",
    "paper_analysis",
]
REPORT_DESCRIPTIONS = {
    "literature_review": "A broad, evidence-led synthesis of what the field knows, how approaches differ, and what remains unresolved.",
    "state_of_art": "A current map of the field: major research directions, representative methods, benchmarks, advances, and open problems.",
    "gap_analysis": "A structured investigation of where the literature is incomplete, contradictory, under-evaluated, or methodologically underexplored.",
    "comparison": "A controlled comparison of methods or studies across aligned datasets, metrics, assumptions, results, costs, and limitations.",
    "evidence_review": "A review-style document emphasizing search scope, screening, evidence quality, consistency, uncertainty, and traceability.",
    "research_proposal": "An evidence-derived research plan connecting the literature, a defensible gap, research questions, experiments, baselines, and evaluation.",
    "paper_analysis": "A deep reading of one or more selected papers, covering problem, method, experiments, results, ablations, limitations, and what is actually established.",
}
def _init_state() -> None:
    defaults = {
        "result": None,
        "loaded_run_id": None,
        "uploaded_sources": [],
        "uploaded_analysis_cache": {},
        "chat_mode": "Ask",
        "chat_panel": "split",
        "research_stage": "setup",
        "candidate_plan": None,
        "candidate_sources": [],
        "candidate_events": [],
        "candidate_usage": {},
        "candidate_selected_ids": [],
        "candidate_generation_id": "",
        "candidate_question": "",
        "candidate_title": "",
        "candidate_report_type": "literature_review",
        "candidate_depth": "Deep",
        "candidate_max_sources": min(settings.max_sources, 10),
        "show_all_candidates": False,
        "report_updated_notice": "",
        "sidebar_rename_open": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
_init_state()
st.markdown(
    """
    <style>
    .block-container { max-width: 1380px; padding-top: 1.35rem; padding-bottom: 4rem; }
    .rp-eyebrow { font-size:.68rem; letter-spacing:.13em; text-transform:uppercase; color:#737373; font-weight:700; }
    .rp-subtle { color:#737373; }
    .rp-rule { height:1px; background:#e5e5e5; margin:1rem 0; }
    .rp-case-meta { color:#737373; font-size:.84rem; }
    .rp-chip { display:inline-block; border:1px solid #dedede; border-radius:999px; padding:3px 8px; margin-right:4px; color:#555; background:#fafafa; font-size:.73rem; }
    .rp-paper-card { min-height:132px; }
    .rp-paper-title { font-size:.88rem; font-weight:650; line-height:1.25; }
    .rp-paper-meta { color:#737373; font-size:.72rem; line-height:1.35; margin-top:4px; }
    .rp-paper-score { color:#333; font-size:.75rem; font-weight:600; }
    .rp-chat-head { padding-bottom:8px; border-bottom:1px solid #e5e5e5; margin-bottom:8px; }
    .rp-chat-context { font-size:.75rem; color:#737373; line-height:1.4; }
    .rp-sidebar-title { white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
    div[data-testid="stSidebar"] .stButton button { white-space:nowrap; overflow:hidden; text-overflow:ellipsis; text-align:left; }
    div[data-testid="stVerticalBlockBorderWrapper"] { border-color:#e5e5e5 !important; }
    div[data-testid="stChatMessage"] { border-radius:10px; }
    </style>
    """,
    unsafe_allow_html=True,
)
def _short_case_name(item: dict, width: int = 34) -> str:
    name = (item.get("case_name") or item.get("report_title") or item.get("question") or "Untitled research").strip()
    if len(name) <= width:
        return name
    return name[: width - 1].rstrip() + "…"
def _reset_to_new_case() -> None:
    st.session_state.result = None
    st.session_state.loaded_run_id = None
    st.session_state.uploaded_sources = []
    st.session_state.uploaded_analysis_cache = {}
    st.session_state.research_stage = "setup"
    st.session_state.candidate_plan = None
    st.session_state.candidate_sources = []
    st.session_state.candidate_events = []
    st.session_state.candidate_usage = {}
    st.session_state.candidate_selected_ids = []
    st.session_state.candidate_generation_id = ""
    st.session_state.show_all_candidates = False
    st.session_state.candidate_question = ""
    st.session_state.candidate_title = ""
    st.session_state.candidate_report_type = "literature_review"
    st.session_state.candidate_depth = "Deep"
    st.session_state.candidate_max_sources = min(settings.max_sources, 10)
    st.session_state.report_updated_notice = ""
    st.session_state.sidebar_rename_open = False
    st.session_state.chat_panel = "split"
    st.session_state.chat_mode = "Ask"
with st.sidebar:
    top = st.columns([3, 1])
    with top[0]:
        st.markdown("# ResearchOrbit")
    with top[1]:
        if st.button("+", help="New research case", key="new_case_top"):
            _reset_to_new_case()
            st.rerun()
    st.caption("Evidence-grounded research workspace")
    st.divider()
    history = store.list_cases(limit=30)
    st.markdown("**Recent research cases**")
    if history:
        for item in history:
            case_id = item["run_id"]
            cols = st.columns([5, 1])
            with cols[0]:
                label = ("📌 " if item.get("pinned") else "") + _short_case_name(item)
                if st.button(label, key=f"case_{case_id}", use_container_width=True):
                    st.session_state.loaded_run_id = case_id
                    st.rerun()
            with cols[1]:
                with st.popover("⋯", use_container_width=True):
                    st.caption(item.get("report_type", "Research case").replace("_", " ").title())
                    new_name = st.text_input("Case name", value=item.get("case_name") or item.get("report_title") or "", key=f"rename_{case_id}")
                    if st.button("Rename", key=f"do_rename_{case_id}", use_container_width=True):
                        updated = store.rename_case(case_id, new_name)
                        if st.session_state.result and store.same_case(st.session_state.result.run_id, case_id):
                            st.session_state.result = store.load(st.session_state.result.run_id) or updated
                        st.rerun()
                    pin_label = "Unpin" if item.get("pinned") else "Pin case"
                    if st.button(pin_label, key=f"pin_{case_id}", use_container_width=True):
                        updated = store.set_pinned(case_id, not item.get("pinned", False))
                        if st.session_state.result and store.same_case(st.session_state.result.run_id, case_id):
                            st.session_state.result = store.load(st.session_state.result.run_id) or updated
                        st.rerun()
                    st.divider()
                    if st.button("Delete case", key=f"delete_{case_id}", use_container_width=True):
                        active_belongs = bool(st.session_state.result and store.same_case(st.session_state.result.run_id, case_id))
                        store.delete_case(case_id)
                        if active_belongs:
                            _reset_to_new_case()
                        st.rerun()
    else:
        st.caption("Your completed research cases will appear here.")
    st.divider()
    with st.expander("Research settings"):
        max_evidence = settings.max_evidence
        st.caption("Core corpus size is chosen in the main research setup so paper selection stays visible and deliberate.")
        st.caption(f"Evidence retrieval budget: {max_evidence} high-value snippets per report. This is managed automatically rather than exposed as another slider.")
        st.caption(f"Remote PDF downloads are capped at {settings.max_pdf_bytes // 1_000_000} MB per paper.")
        st.caption("Deep mode reads every selected core paper. Uploaded PDFs are always deep-read when an LLM is available.")
        if settings.llm_available:
            st.success("LLM configured", icon="🧠")
        else:
            st.info("Deterministic mode is available. Add an LLM key for deep paper analysis and research-grade prose.")
    with st.expander("Grounding policy"):
        st.caption(
            "Follow-ups stay attached to this case's selected papers, verified evidence, deep paper analyses, "
            "claims, and report context. When the corpus cannot answer a question, ResearchOrbit says so rather "
            "than silently filling the gap from general model knowledge."
        )
if st.session_state.loaded_run_id:
    loaded = store.load(st.session_state.loaded_run_id)
    if loaded:
        st.session_state.result = loaded
        st.session_state.uploaded_sources = [s for s in loaded.sources if s.role == "uploaded"]
        st.session_state.uploaded_analysis_cache = {}
        st.session_state.research_stage = "setup"
    st.session_state.loaded_run_id = None
result = st.session_state.result
if result is None:
    st.markdown('<div class="rp-eyebrow">New research case</div>', unsafe_allow_html=True)
    st.title("Start a research case")
    st.caption("Define the research task, inspect the literature set, then generate from the papers you actually want in the corpus.")
    screening_locked = st.session_state.research_stage == "screening"
    question = st.text_area(
        "Research question",
        value=st.session_state.candidate_question,
        placeholder="e.g. What are the main research gaps in low-resource visual localization from ground imagery to satellite imagery?",
        height=104,
        key="new_question",
        disabled=screening_locked,
    )
    st.session_state.candidate_question = question
    c1, c2, c3 = st.columns([1.45, 1.1, .85], gap="medium")
    with c1:
        report_title = st.text_input(
            "Report title (optional)",
            value=st.session_state.candidate_title,
            placeholder="Concise academic title",
            help="This controls the deliverable title. The research question remains the analytical target.",
            key="new_report_title",
            disabled=screening_locked,
        )
        st.session_state.candidate_title = report_title
    with c2:
        recommended = report_type_for_question(question) if question.strip() else "literature_review"
        if recommended not in NEW_REPORT_TYPES:
            recommended = "literature_review"
        current_type = st.session_state.candidate_report_type if st.session_state.candidate_report_type in NEW_REPORT_TYPES else recommended
        selected_type = st.selectbox("Report type", NEW_REPORT_TYPES, index=NEW_REPORT_TYPES.index(current_type), format_func=lambda x: REPORT_TYPE_LABELS[x], key="new_report_type", disabled=screening_locked)
        st.session_state.candidate_report_type = selected_type
    with c3:
        st.markdown("**Reading regime**")
        st.info("Full research reading. Every RED paper is verified for PDF acquisition and parsed full text before it can count as a deep read.")
    st.caption(REPORT_DESCRIPTIONS[selected_type])
    st.caption("Corpus size is selected automatically from relevance, evidence coverage, complementarity, redundancy, and the research question. You can override individual RED/GREEN paper decisions during screening.")
    if st.session_state.research_stage == "setup":
        if st.button("Find and rank relevant papers →", type="primary", use_container_width=True, disabled=not question.strip()):
            effective_title = report_title.strip() or suggest_report_title(question, selected_type)
            with st.status("Searching, reranking and screening the literature…", expanded=True) as status:
                try:
                    plan, candidates, events, _, usage = ResearchAgent().discover_candidates(
                        question.strip(), report_type=selected_type, report_title=effective_title, candidate_limit=max(18, selected_corpus_size + 6)
                    )
                    st.session_state.candidate_plan = plan
                    st.session_state.candidate_sources = candidates
                    st.session_state.candidate_events = events
                    st.session_state.candidate_usage = usage
                    st.session_state.candidate_selected_ids = [s.id for s in candidates[:selected_corpus_size]]
                    st.session_state.candidate_generation_id = uuid.uuid4().hex[:10]
                    st.session_state.show_all_candidates = False
                    st.session_state.research_stage = "screening"
                    status.update(label=f"{len(candidates)} candidate papers ranked", state="complete")
                    st.rerun()
                except Exception as exc:
                    status.update(label="Literature search failed", state="error")
                    st.exception(exc)
    else:
        candidates = st.session_state.candidate_sources
        st.markdown("<div class='rp-rule'></div>", unsafe_allow_html=True)
        h1, h2, h3 = st.columns([2.5, 1, 1])
        with h1:
            st.markdown('<div class="rp-eyebrow">Corpus selection</div>', unsafe_allow_html=True)
            st.markdown(f"**{len(st.session_state.candidate_selected_ids)} selected** · {len(candidates)} ranked candidates · core corpus {selected_corpus_size}")
        candidate_widget_prefix = st.session_state.candidate_generation_id or hashlib.sha1(question.strip().encode("utf-8")).hexdigest()[:10]
        with h2:
            if st.button("Select top", use_container_width=True):
                selected = [s.id for s in candidates[:selected_corpus_size]]
                st.session_state.candidate_selected_ids = selected
                for source in candidates:
                    st.session_state[f"cand_{candidate_widget_prefix}_{source.id}"] = source.id in set(selected)
                st.rerun()
        with h3:
            if st.button("Clear", use_container_width=True):
                st.session_state.candidate_selected_ids = []
                for source in candidates:
                    st.session_state[f"cand_{candidate_widget_prefix}_{source.id}"] = False
                st.rerun()
        st.caption("The ranking is an auditable recommendation. Your selection is the actual corpus used for the report and deep analysis.")
        if st.button("Change research setup", use_container_width=True):
            st.session_state.research_stage = "setup"
            st.session_state.candidate_sources = []
            st.session_state.candidate_events = []
            st.session_state.candidate_selected_ids = []
            st.session_state.show_all_candidates = False
            st.rerun()
        selected_ids = set(st.session_state.candidate_selected_ids)
        visible = candidates if st.session_state.show_all_candidates else candidates[:12]
        for group_start in range(0, len(visible), 3):
            row = visible[group_start:group_start + 3]
            cards = st.columns(3, gap="medium")
            for offset, source in enumerate(row):
                with cards[offset]:
                    with st.container(border=True):
                        checked = source.id in selected_ids
                        widget_key = f"cand_{candidate_widget_prefix}_{source.id}"
                        new_checked = st.checkbox("Include in corpus", value=checked, key=widget_key)
                        if new_checked:
                            selected_ids.add(source.id)
                        else:
                            selected_ids.discard(source.id)
                        st.markdown(f'<div class="rp-paper-title">{group_start + offset + 1}. {html.escape(source.title)}</div>', unsafe_allow_html=True)
                        meta = " · ".join(x for x in [source.authors, source.year, source.venue] if x)
                        st.markdown(f'<div class="rp-paper-meta">{html.escape(meta or "Metadata limited")}</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="rp-paper-score">Relevance {source.score:.2f}</div>', unsafe_allow_html=True)
                        if source.relevance_reasons:
                            st.caption("; ".join(source.relevance_reasons[:2]))
                        with st.expander("Abstract & source details"):
                            st.write(source.abstract or "No abstract was returned by the search provider.")
                            st.caption("Full text fetched for screening" if source.full_text else "Abstract-only screening")
                            if source.url or source.pdf_url:
                                st.link_button("Open paper / source", source.url or source.pdf_url, use_container_width=True)
        if len(candidates) > 12:
            if st.button("Show all ranked candidates" if not st.session_state.show_all_candidates else "Show top 12", use_container_width=True):
                st.session_state.show_all_candidates = not st.session_state.show_all_candidates
                st.rerun()
        st.session_state.candidate_selected_ids = [
            source.id for source in candidates if source.id in selected_ids
        ]
        if len(st.session_state.candidate_selected_ids) > selected_corpus_size:
            st.warning(
                f"You selected {len(st.session_state.candidate_selected_ids)} papers, but the core corpus is capped at {selected_corpus_size}. "
                "Remove a selection before generating the report."
            )
        st.markdown("<div class='rp-rule'></div>", unsafe_allow_html=True)
        b1, b2 = st.columns([1, 3])
        with b1:
            if st.button("← Change question", use_container_width=True):
                st.session_state.research_stage = "setup"
                st.rerun()
        with b2:
            selected_ids = st.session_state.candidate_selected_ids
            if st.button("Generate report from selected papers →", type="primary", use_container_width=True, disabled=not selected_ids or len(selected_ids) > selected_corpus_size):
                effective_title = report_title.strip() or suggest_report_title(question, selected_type)
                selected_sources = [s for s in candidates if s.id in selected_ids]
                effective_deep_k = len(selected_sources)
                with st.status("Reading the selected corpus and building the report…", expanded=True) as status:
                    try:
                        agent = ResearchAgent()
                        status.write(f"Selected {len(selected_sources)} paper(s). Verifying PDFs and close-reading the selected full-text corpus.")
                        if not settings.llm_available:
                            status.write("No LLM is configured, so structured deep reading cannot run; the report will state that limitation.")
                        new_result = agent.run(
                            question.strip(), max_sources=selected_corpus_size, max_evidence=max_evidence,
                            deep_read_top_k=effective_deep_k, analysis_depth="deep", report_type=selected_type,
                            report_title=effective_title, version=1, preselected_sources=selected_sources,
                            precomputed_plan=st.session_state.candidate_plan, precomputed_events=st.session_state.candidate_events,
                            precomputed_usage=st.session_state.candidate_usage,
                        )
                        new_result.case_name = report_title.strip() or effective_title
                        store.save(new_result)
                        st.session_state.result = new_result
                        st.session_state.uploaded_sources = []
                        st.session_state.uploaded_analysis_cache = {}
                        st.session_state.research_stage = "setup"
                        status.update(label=f"Report generated · {len(new_result.paper_analyses)} deep reads", state="complete")
                        st.rerun()
                    except Exception as exc:
                        status.update(label="Report generation failed", state="error")
                        st.exception(exc)
    st.stop()
case_display = result.case_name or result.report_title or result.question
st.markdown('<div class="rp-eyebrow">Research case</div>', unsafe_allow_html=True)
header_cols = st.columns([7, 1, 1])
with header_cols[0]:
    st.title(result.report_title or result.question)
    st.markdown(
        f'<div class="rp-case-meta">{REPORT_TYPE_LABELS.get(result.report_type, result.report_type)} · Version {result.version} · '
        f'{len(result.sources)} papers · {len(result.paper_analyses)} deep reads</div>', unsafe_allow_html=True
    )
with header_cols[1]:
    if st.button("New research", use_container_width=True):
        _reset_to_new_case()
        st.rerun()
with header_cols[2]:
    if st.button("Rename", use_container_width=True):
        st.session_state.sidebar_rename_open = True
if getattr(st.session_state, "sidebar_rename_open", False):
    rename = st.text_input("Case name", value=case_display, key="active_case_rename")
    rc1, rc2 = st.columns([1, 5])
    if rc1.button("Save name", type="primary"):
        renamed = store.rename_case(result.run_id, rename.strip() or result.report_title)
        if renamed is not None:
            result = renamed
            st.session_state.result = renamed
        st.session_state.sidebar_rename_open = False
        st.rerun()
    if rc2.button("Cancel"):
        st.session_state.sidebar_rename_open = False
        st.rerun()
if st.session_state.get("report_updated_notice"):
    st.success(st.session_state.report_updated_notice, icon="🔄")
    st.session_state.report_updated_notice = ""
if result.warnings:
    with st.expander(f"{len(result.warnings)} research limitation(s) to note"):
        for warning in result.warnings:
            st.warning(warning)
m = result.metrics
facts = st.columns(5)
facts[0].caption(f"**{m.get('sources_selected', len(result.sources))}** papers")
facts[1].caption(f"**{m.get('sources_core', len([s for s in result.sources if s.role in ('core','uploaded')]))}** core")
facts[2].caption(f"**{m.get('evidence_count', len(result.evidence))}** evidence")
facts[3].caption(f"**{m.get('claim_count', len(result.claims))}** claims")
facts[4].caption(f"**{m.get('papers_deep_read', len(result.paper_analyses))}** deep reads")
with st.expander("Run details", expanded=False):
    detail_cols = st.columns(4)
    detail_cols[0].metric("Runtime", f"{float(m.get('total_runtime_seconds', 0) or 0):.1f}s")
    detail_cols[1].metric("LLM calls", int(m.get("llm_calls", 0) or 0))
    detail_cols[2].metric("Estimated cost", f"${float(m.get('estimated_cost_usd', 0) or 0):.4f}")
    detail_cols[3].metric("Evidence strength", str(m.get("evidence_strength", "unknown")).title())
    st.caption(
        f"Search queries: {m.get('search_queries_used', 0)} · Providers represented: {m.get('providers_used', 0)} · "
        f"Full-text reads: {m.get('papers_full_text', 0)} · Prompt/completion tokens: "
        f"{m.get('prompt_tokens', 0):,}/{m.get('completion_tokens', 0):,}"
    )
    stage_keys = [
        ("plan", "Planning"), ("search", "Search"), ("select", "Selection"),
        ("fulltext", "Full text"), ("evidence", "Evidence"), ("deep_read", "Deep reading"),
        ("claims", "Claims"), ("synthesize", "Synthesis"),
    ]
    stage_rows = [(label, float(m.get(f"stage_seconds_{key}", 0) or 0)) for key, label in stage_keys if f"stage_seconds_{key}" in m]
    if stage_rows:
        st.caption("Stage timings")
        st.dataframe({"Stage": [x[0] for x in stage_rows], "Seconds": [round(x[1], 3) for x in stage_rows]}, hide_index=True, use_container_width=True)
def _export_temp_path(suffix: str) -> Path:
    return Path(tempfile.gettempdir()) / f"researchpilot_{result.run_id}{suffix}"
def _safe_filename(name: str, fallback: str = "research_report") -> str:
    """Return a portable download filename safe on Windows, macOS and Linux."""
    import re
    clean = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', " ", name or "").strip().rstrip(".")
    clean = re.sub(r"\s+", " ", clean)
    return (clean or fallback)[:100]
def render_report() -> None:
    st.markdown('<div class="rp-eyebrow">Current report · Version {}</div>'.format(result.version), unsafe_allow_html=True)
    st.markdown(result.report_markdown)
    with st.expander("Export deliverables", expanded=False):
        st.caption("DOCX is the source-of-truth formatted document. PDF is rendered from that DOCX when Word or LibreOffice is available.")
        dc = st.columns(3)
        safe_name = _safe_filename(result.report_title or "research_report")
        dc[0].download_button("Markdown", result.report_markdown, file_name=f"{safe_name}.md", use_container_width=True)
        dc[1].download_button("BibTeX", sources_to_bibtex(result.sources), file_name=f"{safe_name}.bib", use_container_width=True)
        if dc[2].button("Prepare Word / PDF", key=f"prepare_exports_{result.run_id}", use_container_width=True):
            try:
                docx_path = _export_temp_path(".docx")
                pdf_path = _export_temp_path(".pdf")
                markdown_to_docx(result.report_markdown, docx_path)
                st.session_state[f"docx_bytes_{result.run_id}"] = docx_path.read_bytes()
                try:
                    markdown_to_pdf(result.report_markdown, pdf_path)
                    st.session_state[f"pdf_bytes_{result.run_id}"] = pdf_path.read_bytes()
                except Exception as pdf_exc:
                    st.session_state[f"pdf_bytes_{result.run_id}"] = None
                    st.session_state[f"pdf_error_{result.run_id}"] = str(pdf_exc)
                st.rerun()
            except Exception as exc:
                st.error(f"Couldn't prepare the formatted deliverables: {exc}")
        prepared_docx = st.session_state.get(f"docx_bytes_{result.run_id}")
        prepared_pdf = st.session_state.get(f"pdf_bytes_{result.run_id}")
        if prepared_docx:
            d1, d2 = st.columns(2)
            d1.download_button("Download DOCX", prepared_docx, file_name=f"{safe_name}.docx", key=f"download_docx_{result.run_id}", use_container_width=True)
            if prepared_pdf:
                d2.download_button("Download PDF", prepared_pdf, file_name=f"{safe_name}.pdf", key=f"download_pdf_{result.run_id}", use_container_width=True)
            else:
                d2.caption(st.session_state.get(f"pdf_error_{result.run_id}", "PDF conversion was not available on this machine."))
def _ensure_conversation(r) -> None:
    """Migrate older saved cases that predate the chronological conversation field."""
    if getattr(r, "conversation", None):
        return
    turns = []
    for qa in r.followups:
        turns.append({"mode": "ask", "question": qa.question, "answer": qa.answer, "evidence_ids": qa.evidence_ids, "source_ids": getattr(qa, "source_ids", []), "llm_used": qa.llm_used})
    for verification in r.verifications:
        turns.append({"mode": "verify", "question": verification.claim, "answer": verification.reasoning, "evidence_ids": verification.evidence_ids, "source_ids": [s.id for s in verification.new_sources], "llm_used": verification.llm_used})
    if turns:
        from researchpilot.schemas import ConversationTurn
        r.conversation = [ConversationTurn.model_validate(x) for x in turns]
def _run_chat_question(text: str, mode: str) -> None:
    if not text.strip():
        return
    agent = ResearchAgent()
    if mode == "Verify claim":
        verification = agent.verify_claim(text.strip(), result=result, allow_new_search=True)
    else:
        grounder = Grounder(agent.llm)
        if question_targets_upload(text, has_recent_upload=bool(st.session_state.uploaded_sources)):
            source = st.session_state.uploaded_sources[-1]
            cached = st.session_state.uploaded_analysis_cache.get(source.id)
            if cached is None:
                cached = next((a for a in result.paper_analyses if a.source_id == source.id), None)
            qa, analysis = grounder.answer_about_source(result, text.strip(), source, cached)
            if analysis:
                st.session_state.uploaded_analysis_cache[source.id] = analysis
        else:
            qa = grounder.answer(result, text.strip())
        result.followups.append(qa)
        from researchpilot.schemas import ConversationTurn
        result.conversation.append(ConversationTurn(mode="ask", question=qa.question, answer=qa.answer, evidence_ids=qa.evidence_ids, source_ids=getattr(qa, "source_ids", []), llm_used=qa.llm_used))
    store.save(result)
    st.rerun()
def render_chat() -> None:
    _ensure_conversation(result)
    st.markdown('<div class="rp-eyebrow">Research conversation</div>', unsafe_allow_html=True)
    st.markdown('<div class="rp-chat-head"><b>Grounded research colleague</b><br><span class="rp-chat-context">Answers use the current report, selected papers, deep readings, verified evidence, added PDFs and prior turns. The assistant stays concise and flags missing evidence.</span></div>', unsafe_allow_html=True)
    controls = st.columns(3)
    if controls[0].button("↔ Split", use_container_width=True):
        st.session_state.chat_panel = "split"; st.rerun()
    if controls[1].button("↗ Expand", use_container_width=True):
        st.session_state.chat_panel = "full"; st.rerun()
    if controls[2].button("− Minimize", use_container_width=True):
        st.session_state.chat_panel = "hidden"; st.rerun()
    st.caption(f"{len(result.sources)} papers · {len(result.paper_analyses)} deep reads · {len(result.evidence)} evidence items")
    chat_mode = st.radio("Mode", ["Ask", "Verify claim"], horizontal=True, key="chat_mode", label_visibility="collapsed")
    with st.container(height=590, border=False):
        if result.conversation:
            for turn in result.conversation:
                with st.chat_message("user"):
                    st.write(("Verify: " if turn.mode == "verify" else "") + turn.question)
                with st.chat_message("assistant"):
                    st.write(turn.answer)
                    if turn.evidence_ids or getattr(turn, "source_ids", None):
                        bits = []
                        if turn.evidence_ids:
                            bits.append("Evidence: " + ", ".join(turn.evidence_ids))
                        if getattr(turn, "source_ids", None):
                            chat_sources = list(result.sources)
                            seen = {s.dedupe_key() for s in chat_sources}
                            for verification in result.verifications:
                                for source in verification.new_sources:
                                    if source.dedupe_key() not in seen:
                                        chat_sources.append(source)
                                        seen.add(source.dedupe_key())
                            cmap = {source.id: f"S{i}" for i, source in enumerate(chat_sources, start=1)}
                            source_tags = [cmap.get(sid, "S?") for sid in turn.source_ids]
                            bits.append("Sources: " + ", ".join(dict.fromkeys(source_tags)))
                        st.caption(" · ".join(bits))
        else:
            st.caption("Ask about a method, result, ablation, contradiction, limitation, dataset, gap or next experiment.")
            suggestions = [
                "What are the strongest findings, and what evidence supports them?",
                "Where do the papers disagree or become difficult to compare?",
                "What important research gap remains after reading this corpus?",
            ]
            for suggestion in suggestions:
                if st.button(suggestion, key=f"suggest_{result.run_id}_{suggestion[:12]}", use_container_width=True):
                    _run_chat_question(suggestion, chat_mode)
    placeholder = "Ask about this research…" if chat_mode == "Ask" else "Enter a claim to verify against the evidence…"
    user_msg = st.chat_input(placeholder)
    if user_msg:
        _run_chat_question(user_msg, chat_mode)
st.markdown("<div class='rp-rule'></div>", unsafe_allow_html=True)
if st.session_state.chat_panel == "hidden":
    c1, c2 = st.columns([1, 7])
    with c1:
        if st.button("Open chat", type="secondary", use_container_width=True):
            st.session_state.chat_panel = "split"
            st.rerun()
    with c2:
        render_report()
elif st.session_state.chat_panel == "full":
    c1, c2 = st.columns([1, 7])
    with c1:
        if st.button("← Report", use_container_width=True):
            st.session_state.chat_panel = "split"
            st.rerun()
    with c2:
        render_chat()
else:
    left, right = st.columns([1, 1], gap="large")
    with left:
        render_report()
    with right:
        render_chat()
st.markdown("<div class='rp-rule'></div>", unsafe_allow_html=True)
st.markdown('<div class="rp-eyebrow">Expand the research corpus</div>', unsafe_allow_html=True)
with st.expander("Add papers and re-analyze", expanded=False):
    st.write(
        "Add papers that the search missed or that you specifically want considered. "
        "Each added paper is treated as a core paper and sent through Deep analysis when an LLM is available. "
        "Re-analysis creates a new report version while preserving the case conversation and previous report."
    )
    uploaded_files = st.file_uploader(
        "Drop one or more research PDFs here",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"paper_uploader_{result.run_id}",
        label_visibility="collapsed",
    )
    if uploaded_files:
        st.caption(f"{len(uploaded_files)} paper(s) ready · every new paper will be deeply analyzed")
        cols = st.columns(min(4, len(uploaded_files)))
        for i, f in enumerate(uploaded_files):
            cols[i % len(cols)].caption("📄 " + f.name)
        if st.button("Re-analyze and create new report version →", type="primary", key=f"reanalyze_{result.run_id}", use_container_width=True):
            new_sources = []
            with st.status("Deep-reading the added papers and rebuilding the report…", expanded=True) as status:
                for f in uploaded_files:
                    try:
                        new_sources.append(source_from_uploaded_pdf(f.name, f.getvalue()))
                    except Exception as exc:
                        st.error(f"Couldn't read {f.name}: {exc}")
                if new_sources:
                    existing_keys = {s.dedupe_key() for s in result.sources}
                    unique_new_sources = []
                    seen_upload_keys = set(existing_keys)
                    for source in new_sources:
                        key = source.dedupe_key()
                        if key in seen_upload_keys:
                            st.warning(f"Already in this case: {source.title}")
                            continue
                        seen_upload_keys.add(key)
                        unique_new_sources.append(source)
                    if not unique_new_sources:
                        status.update(label="No new papers to add", state="complete")
                        st.stop()
                    try:
                        updated = ResearchAgent().add_sources_and_reanalyze(result, unique_new_sources, deep_read_top_k=len(unique_new_sources))
                        updated.case_name = result.case_name or result.report_title
                        updated.pinned = result.pinned
                        store.save(updated)
                        st.session_state.result = updated
                        st.session_state.uploaded_sources = [s for s in updated.sources if s.role == "uploaded"]
                        st.session_state.uploaded_analysis_cache = {}
                        added_ids = set(updated.added_source_ids)
                        deep_added = sum(1 for a in updated.paper_analyses if a.source_id in added_ids)
                        if deep_added == len(unique_new_sources):
                            analysis_note = f"{deep_added} new paper(s) were incorporated and deeply analyzed."
                        elif not settings.llm_available:
                            analysis_note = f"{len(unique_new_sources)} new paper(s) were incorporated; structured deep analysis requires an LLM key."
                        else:
                            analysis_note = f"{len(unique_new_sources)} new paper(s) were incorporated; {deep_added} received structured deep analysis. See the report limitations for any unavailable reads."
                        st.session_state.report_updated_notice = (
                            f"Report replaced with Version {updated.version}. "
                            + analysis_note + " The previous report remains in Research history."
                        )
                        status.update(label=f"New report version {updated.version} is ready", state="complete")
                        st.rerun()
                    except Exception as exc:
                        status.update(label="Re-analysis failed", state="error")
                        st.exception(exc)
if st.session_state.uploaded_sources:
    with st.expander(f"Added papers in this case ({len(st.session_state.uploaded_sources)})"):
        for source in st.session_state.uploaded_sources:
            analysis = next((a for a in result.paper_analyses if a.source_id == source.id), None)
            deep_label = "deep analyzed" if analysis else ("deep analysis unavailable" if not settings.llm_available else "deep analysis failed")
            st.write(f"**{source.title}**")
            st.caption(f"{source.authors or 'Authors not extracted'} · {source.year or 'Year not extracted'} · {deep_label}")
st.markdown("<div class='rp-rule'></div>", unsafe_allow_html=True)
st.subheader("Research history")
versions = store.versions(result)
if versions:
    for version_result in versions:
        is_current = version_result.run_id == result.run_id
        label = f"Version {version_result.version}{' · Current' if is_current else ''} · {len(version_result.sources)} papers · {len(version_result.paper_analyses)} deep reads"
        with st.expander(label, expanded=is_current):
            if not is_current:
                st.caption(version_result.report_title or version_result.question)
                if st.button("Open this version", key=f"open_version_{version_result.run_id}"):
                    st.session_state.result = version_result
                    st.session_state.uploaded_sources = [s for s in version_result.sources if s.role == "uploaded"]
                    st.rerun()
            else:
                st.caption("Current report. Previous versions remain immutable so you can compare how the research changed after adding papers.")
with st.expander("Evidence & research audit"):
    st.caption("The main report stays clean. This companion audit preserves source selection, evidence snippets, claim mappings, processing events, and traceability.")
    audit = build_evidence_audit_report(result)
    ac = st.columns(3)
    safe_name = _safe_filename(result.report_title or "research_report")
    ac[0].download_button("Audit Markdown", audit, file_name=f"{safe_name}_audit.md")
    if ac[1].button("Prepare audit PDF", key=f"audit_pdf_{result.run_id}"):
        try:
            path = markdown_to_pdf(audit, _export_temp_path("_audit.pdf"))
            ac[1].download_button("Download audit PDF", path.read_bytes(), file_name=f"{safe_name}_audit.pdf", key=f"dl_audit_{result.run_id}")
        except Exception as exc:
            st.error(f"Couldn't build the audit PDF: {exc}")
    ac[2].caption("Every source-level citation in the report can be traced back to this audit.")
