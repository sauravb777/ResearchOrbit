from __future__ import annotations
import re
from ..llm import LLMClient
from ..schemas import Evidence, FollowupQA, PaperAnalysis, Source
from . import deep_read as deep_read_mod
from .grounding import safe_grounded_text
from .synthesizer import citation_map
_REFERRING_PHRASES = [
    "this paper", "this pdf", "this document", "the uploaded paper",
    "the paper i uploaded", "the attached paper", "this study",
    "the paper i attached", "the paper i just uploaded", "my uploaded paper",
]
_SYSTEM_PROMPT = (
    "You are a research assistant. The user uploaded a paper and is asking about it "
    "specifically -- treat the UPLOADED PAPER ANALYSIS below as the primary subject, not "
    "background material. If an EXISTING SOURCE is also given, the user wants a comparison "
    "or wants to know how it relates to the existing literature -- use its evidence for that "
    "purpose only, never let it override or get mixed into facts about the uploaded paper "
    "itself. Answer the user's question using ONLY the material given below. Every specific "
    "fact must end with its citation tag, e.g. [E3]. Never invent a citation tag not given "
    "to you, and never introduce a fact about either paper that isn't in the material below."
)
def question_targets_upload(question: str, has_recent_upload: bool) -> bool:
    """True if `question` clearly refers to a just-uploaded paper (and one
    exists in this session) -- the signal for routing to the targeted
    analysis path instead of generic grounded QA."""
    if not has_recent_upload or not question:
        return False
    q = question.lower()
    return any(phrase in q for phrase in _REFERRING_PHRASES)
def find_named_source(question: str, sources: list[Source]) -> Source | None:
    """If the question names an existing source directly -- by its [S#]
    report tag ("compare with S2") or by a distinctive acronym/word from
    its title ("compare this with CRR") -- find it. Returns None if no
    source is confidently named; callers should treat that as "just
    analyze the uploaded paper on its own", not an error."""
    if not sources:
        return None
    cmap = citation_map(sources)
    tag_by_source = {v: k for k, v in cmap.items()}
    m = re.search(r"\bs(\d+)\b", question.lower())
    if m and f"S{m.group(1)}" in tag_by_source:
        source_id = tag_by_source[f"S{m.group(1)}"]
        return next((s for s in sources if s.id == source_id), None)
    for word in re.findall(r"\b[A-Za-z][A-Za-z0-9\-]{2,}\b", question):
        if word.lower() in {"this", "paper", "with", "compare", "the", "and", "for"}:
            continue
        for s in sources:
            if len(word) >= 3 and word.lower() in s.title.lower():
                return s
    return None
def answer_about_uploaded_paper(
    question: str,
    uploaded_source: Source,
    named_source: Source | None,
    all_sources: list[Source],
    all_evidence: list[Evidence],
    llm: LLMClient | None = None,
    existing_analysis: PaperAnalysis | None = None,
    report_type: str = "paper_analysis",
) -> tuple[FollowupQA, PaperAnalysis | None]:
    """Returns (answer, analysis_of_the_uploaded_paper). `analysis` is
    returned so callers (the orchestrator/app) can cache it -- re-running a
    full deep-read on every follow-up about the same uploaded paper would
    be wasteful and slower than necessary."""
    llm = llm or LLMClient()
    analysis = existing_analysis
    verified_evidence: list[Evidence] = []
    if analysis is None:
        analysis, verified_evidence = deep_read_mod.analyze_paper(
            question, uploaded_source, llm=llm, analysis_mode="deep", report_type=report_type
        )
    if analysis is None:
        return (
            FollowupQA(
                question=question,
                answer=(
                    "I couldn't produce a structured analysis of the uploaded paper "
                    "(no LLM is configured, or no readable text could be extracted from the "
                    "file). Try enabling an LLM key, or check that the PDF actually contains "
                    "extractable text (not a scanned image)."
                ),
                evidence_ids=[],
                source_ids=[uploaded_source.id],
                llm_used=False,
            ),
            None,
        )
    upload_tag = citation_map(all_sources).get(uploaded_source.id, "S?")
    bullets = [
        f"[UPLOADED PAPER {upload_tag}: {uploaded_source.title}] problem: {analysis.problem}",
        f"[UPLOADED PAPER {upload_tag}] core idea: {analysis.core_idea}",
        f"[UPLOADED PAPER {upload_tag}] method: {analysis.method_summary}",
        f"[UPLOADED PAPER {upload_tag}] results: {analysis.key_results}",
        f"[UPLOADED PAPER {upload_tag}] ablations: {analysis.ablations}",
        f"[UPLOADED PAPER {upload_tag}] limitations: {analysis.limitations}",
        f"[UPLOADED PAPER {upload_tag}] established: {analysis.established} | not established: {analysis.not_established}",
    ]
    bullets = [b for b in bullets if b.split(": ", 1)[-1].strip()]
    for e in verified_evidence:
        bullets.append(f"{e.text} [{e.id}]")
    allowed = {e.id for e in verified_evidence} | {upload_tag}
    if named_source is not None:
        cmap = citation_map(all_sources)
        named_tag = cmap.get(named_source.id, "S?")
        named_evidence = sorted(
            [e for e in all_evidence if e.source_id == named_source.id], key=lambda e: e.relevance, reverse=True
        )[:8]
        bullets.append(f"[EXISTING SOURCE {named_tag}: {named_source.title}] -- evidence below is about THIS paper, not the upload:")
        for e in named_evidence:
            bullets.append(f"{e.text} [{e.id}][{named_tag}]")
        allowed |= {e.id for e in named_evidence} | {named_tag}
    text, used = safe_grounded_text(_SYSTEM_PROMPT, bullets, allowed, llm, max_tokens=550, preamble=f"USER QUESTION: {question}")
    if text:
        return FollowupQA(question=question, answer=text, evidence_ids=[e.id for e in verified_evidence if e.id in text], source_ids=list(dict.fromkeys([uploaded_source.id] + ([named_source.id] if named_source else []))), llm_used=used), analysis
    fallback_parts = [f"Analysis of the uploaded paper ({uploaded_source.title}):"]
    for label, value in [
        ("Problem", analysis.problem), ("Core idea", analysis.core_idea),
        ("Method", analysis.method_summary), ("Results", analysis.key_results),
        ("Limitations", analysis.limitations),
    ]:
        if value:
            fallback_parts.append(f"- {label}: {value}")
    return (
        FollowupQA(question=question, answer="\n".join(fallback_parts), evidence_ids=[e.id for e in verified_evidence], source_ids=list(dict.fromkeys([uploaded_source.id] + ([named_source.id] if named_source else []))), llm_used=False),
        analysis,
    )
