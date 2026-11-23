from __future__ import annotations
import json
import re
from typing import Any
from ..llm import LLMClient
_EDITOR_SYSTEM = """You are the senior scientific editor of a literature-synthesis report.
Your job is to turn an already evidence-grounded draft into a coherent research paper-like
report. Do NOT add facts. Do NOT invent citations. Do NOT turn internal extraction metadata
into prose. Do NOT write a checklist of what individual papers failed to report.
The report should read like a strong human-written survey/research synthesis: establish the
question and intellectual structure, explain how approaches differ, synthesize mechanisms and
evidence across studies, reconcile apparent disagreements by experimental conditions, surface
important quantitative and negative findings, and end with a bounded synthesis. Individual papers
should appear as evidence within an argument, not as isolated mini-summaries unless the requested
report type is explicitly paper_analysis.
If the corpus does not support a proposed statement, remove or soften the statement. Only mention
an evidence limitation when it materially changes the scientific interpretation. Do not say things
like 'paper X did not report Y' merely because an extraction field is empty. Never mention RED/GREEN,
agent, pipeline, extraction, chunk, prompt, missing field, unavailable paper, or internal scoring.
Preserve all valid source citations supplied in the draft. Any rewritten factual paragraph must retain
or improve its supporting [S#] / [E#] citations. Return only the edited report markdown."""
def _citation_tags(text: str) -> set[str]:
    return set(re.findall(r"\[[A-Z]\d+(?:,\s*p\.\s*\d+)?\]", text or ""))
def _internal_language_flags(text: str) -> list[str]:
    patterns = [
        r"\bRED\b", r"\bGREEN\b", r"deep[- ]read", r"abstract[- ]only",
        r"extraction", r"pipeline", r"LLM", r"agent", r"prompt", r"chunk",
        r"relevance score", r"39% gate", r"not reported", r"not found",
        r"could not find", r"missing field", r"unavailable source",
    ]
    return [p for p in patterns if re.search(p, text or "", re.I)]
def _citation_tag_re() -> re.Pattern:
    return re.compile(r"\[([ES]\d+)(?:,\s*p\.\s*\d+)?\]")
def _is_grounding_exempt_paragraph(paragraph: str, section_title: str) -> bool:
    """Return True for document framing that is not an externally verifiable claim.
    The quality gate must not punish the report for stating the user's research
    question or describing the document's own scope. Analytical prose, including
    the conclusion and executive summary, remains evidence-grounded.
    """
    title = (section_title or "").strip().lower()
    p = re.sub(r"\s+", " ", paragraph or "").strip()
    if title == "research question":
        return True
    if p.startswith("*") and p.endswith("*") and len(p) < 320:
        return True
    return False
def _report_section_paragraphs(report: str):
    """Yield (section_title, paragraph) pairs from markdown without losing body text."""
    body = (report or "").split("## References", 1)[0]
    chunks = re.split(r"(?=^#{1,6}\s+.+$)", body, flags=re.M)
    current = ""
    for chunk in chunks:
        lines = chunk.splitlines()
        if not lines:
            continue
        heading = None
        start = 0
        if re.match(r"^#{1,6}\s+", lines[0].strip()):
            heading = re.sub(r"^#{1,6}\s+", "", lines[0].strip()).strip()
            current = heading
            start = 1
        content = "\n".join(lines[start:])
        for raw in re.split(r"\n\s*\n", content):
            paragraph = raw.strip()
            if paragraph:
                yield current, paragraph
def grounding_eligible_paragraphs(report: str) -> list[str]:
    """Return claim-bearing paragraphs that the grounding gate should measure."""
    out = []
    for section_title, paragraph in _report_section_paragraphs(report):
        if paragraph.startswith("#") or paragraph.startswith("|"):
            continue
        if len(paragraph) < 80 or _is_grounding_exempt_paragraph(paragraph, section_title):
            continue
        out.append(paragraph)
    return out
def _evidence_for_paragraph(paragraph: str, evidence: list[Any], limit: int = 12) -> list[Any]:
    """Select compact, relevant grounding context for a repair call."""
    tokens = set(re.findall(r"[a-zA-Z0-9]{4,}", paragraph.lower()))
    scored = []
    for e in evidence:
        text = str(getattr(e, "text", "") or "")
        et = set(re.findall(r"[a-zA-Z0-9]{4,}", text.lower()))
        overlap = len(tokens & et) / max(1, len(tokens))
        score = 0.65 * overlap + 0.35 * float(getattr(e, "relevance", 0.0) or 0.0)
        scored.append((score, e))
    return [e for _, e in sorted(scored, key=lambda x: x[0], reverse=True)[:limit]]
def repair_report_grounding(report: str, evidence: list[Any], sources: list[Any], llm: LLMClient) -> tuple[str, bool, int]:
    """Repair provenance at the public-report boundary without weakening prose.
    The renderer has already converted internal E# evidence ids into public S#
    source citations. This function therefore repairs with *public* source tags
    only. Evidence is still supplied behind each tag so the model can choose the
    closest support, but internal evidence ids can never leak into the delivered
    report.
    The repair is deliberately claim-preserving: it may attach citations or
    qualify an overstatement, but it must not invent a new factual proposition.
    It works in bounded batches so a long report does not trigger an unbounded
    rewrite loop.
    """
    if not llm.available or not report.strip() or not evidence or not sources:
        return report, False, 0
    targets: list[tuple[str, str]] = []
    for section_title, paragraph in _report_section_paragraphs(report):
        if paragraph.startswith("#") or paragraph.startswith("|"):
            continue
        if len(paragraph) < 80 or _is_grounding_exempt_paragraph(paragraph, section_title):
            continue
        if not _citation_tag_re().search(paragraph):
            targets.append((section_title, paragraph))
    if not targets:
        return report, False, 0
    source_tag_by_id = {s.id: f"S{i+1}" for i, s in enumerate(sources)}
    evidence_payload = []
    for e in evidence:
        tag = source_tag_by_id.get(getattr(e, "source_id", ""))
        if not tag:
            continue
        evidence_payload.append({
            "evidence_id": str(getattr(e, "id", "")),
            "public_source_tag": tag,
            "text": str(getattr(e, "text", "") or ""),
            "locator": str(getattr(e, "locator", "") or ""),
        })
    if not evidence_payload:
        return report, False, 0
    max_targets = min(len(targets), 32)
    targets = targets[:max_targets]
    repaired_by_index: dict[int, str] = {}
    batch_size = 8
    system = """You are the provenance editor for a scholarly research report.
You are NOT rewriting the report. For each supplied paragraph, preserve its useful
information, ordering, density, and academic wording. Do not add a new factual claim.
Your only normal operation is to attach the closest existing PUBLIC source citation
such as [S1] or [S2] to each externally verifiable claim-bearing sentence. If a
paragraph overstates what the supplied evidence supports, make the smallest possible
qualification while preserving the useful information. Cross-paper interpretations
may use multiple source tags. Do not cite a source merely because it exists.
CRITICAL: return ONLY public [S#] citations. Never output internal evidence ids such
as [E1], [C1], [V1], or any other internal identifier. Only use public_source_tag
values supplied in the evidence payload.
Return JSON only:
{"repairs":[{"index":0,"text":"repaired paragraph"}]}"""
    for offset in range(0, len(targets), batch_size):
        batch = targets[offset:offset + batch_size]
        payload = []
        for local_idx, (section, paragraph) in enumerate(batch):
            ev = _evidence_for_paragraph(paragraph, evidence, limit=10)
            payload.append({
                "index": offset + local_idx,
                "section": section,
                "paragraph": paragraph,
                "candidate_evidence": [
                    {
                        "public_source_tag": source_tag_by_id.get(e.source_id),
                        "text": e.text,
                        "locator": getattr(e, "locator", ""),
                    }
                    for e in ev if source_tag_by_id.get(e.source_id)
                ],
            })
        try:
            out = llm.complete_json(
                system,
                json.dumps({"paragraphs": payload}, ensure_ascii=False),
                max_tokens=3800,
            )
        except Exception:
            continue
        repairs = out.get("repairs") if isinstance(out, dict) else None
        if not isinstance(repairs, list):
            continue
        for item in repairs:
            if not isinstance(item, dict):
                continue
            try:
                idx = int(item.get("index", -1))
            except Exception:
                continue
            text = str(item.get("text", "")).strip()
            if not (0 <= idx < len(targets) and text):
                continue
            tags = set(_citation_tag_re().findall(text))
            if not tags or any(not tag.startswith("S") for tag in tags):
                continue
            allowed_public = set(source_tag_by_id.values())
            if not tags.issubset(allowed_public):
                continue
            if not _citation_tag_re().search(text):
                continue
            repaired_by_index[idx] = text
    if not repaired_by_index:
        return report, False, 0
    body = report.split("## References", 1)[0]
    for idx in sorted(repaired_by_index, reverse=True):
        old = targets[idx][1]
        body = body.replace(old, repaired_by_index[idx], 1)
    suffix = report[len(report.split("## References", 1)[0]):]
    repaired = body + suffix
    return repaired, True, len(repaired_by_index)
def deterministic_quality(report: str, sources: list[Any], evidence: list[Any], analyses: list[Any], report_type: str) -> dict:
    """Compute machine checks used as routing signals, not a public ranking."""
    body = (report or "").split("## References", 1)[0]
    paragraphs = grounding_eligible_paragraphs(report)
    cited = [p for p in paragraphs if _citation_tags(p)]
    source_tags = {f"S{i+1}" for i in range(len(sources))}
    cited_tags = set()
    for p in cited:
        cited_tags |= {x.split(',')[0][1:] for x in _citation_tags(p) if x.startswith('[S')}
    multi_source = sum(1 for p in paragraphs if len({x.split(',')[0] for x in _citation_tags(p) if x.startswith('[S')}) >= 2)
    title_like = 0
    if report_type != "paper_analysis":
        for a in analyses:
            title = getattr(a, "source_id", "")
            if title and re.search(rf"\b{re.escape(title)}\b", body, re.I):
                title_like += 1
    internal = _internal_language_flags(body)
    return {
        "substantive_paragraphs": len(paragraphs),
        "cited_paragraphs": len(cited),
        "citation_density": round(len(cited) / len(paragraphs), 3) if paragraphs else 0.0,
        "multi_source_synthesis_paragraphs": multi_source,
        "selected_sources_cited": len(cited_tags & source_tags),
        "selected_source_count": len(source_tags),
        "source_coverage": round(len(cited_tags & source_tags) / len(source_tags), 3) if source_tags else 1.0,
        "internal_language_flags": internal,
        "paper_summary_risk": title_like,
    }
def editorial_repair(report: str, question: str, report_type: str, sources: list[Any], analyses: list[Any], council: dict, llm: LLMClient) -> tuple[str, bool]:
    if not llm.available or not report.strip():
        return report, False
    payload = {
        "question": question,
        "report_type": report_type,
        "draft": report,
        "editorial_checks": deterministic_quality(report, sources, [], analyses, report_type),
        "council_thesis": (council or {}).get("central_thesis", "") if isinstance(council, dict) else "",
        "argument_threads": (council or {}).get("argument_threads", []) if isinstance(council, dict) else [],
    }
    try:
        edited = llm.complete(_EDITOR_SYSTEM, json.dumps(payload, ensure_ascii=False), max_tokens=6500, tier="S", priority="report")
    except Exception:
        return report, False
    if not edited or not edited.strip():
        return report, False
    if not _citation_tags(edited).issubset(_citation_tags(report)):
        return report, False
    return edited.strip(), True
def review_for_synthesis_quality(report: str, question: str, report_type: str, analyses: list[Any], llm: LLMClient) -> tuple[dict, bool]:
    """A final editorial review focused on synthesis quality rather than citation syntax."""
    if not llm.available or not report.strip():
        return {}, False
    paper_memory = []
    for a in analyses:
        paper_memory.append({
            "source_id": a.source_id,
            "question_answer": getattr(a, "question_answer", ""),
            "surprises": [x for x in (getattr(a, "evidence_inventory", []) or []) if x.get("surprising_or_nonobvious")][:8],
            "tradeoffs": (getattr(a, "method_tradeoffs", []) or [])[:8],
            "boundaries": getattr(a, "generalization", ""),
        })
    prompt = f"""QUESTION: {question}\nREPORT TYPE: {report_type}\nREPORT:\n{report}\n\nPAPER MEMORIES:\n{json.dumps(paper_memory, ensure_ascii=False)}\n
Assess the report as a senior scientific editor. Do not rewrite it. Return JSON only:
{{"synthesis_problems":[],"unsupported_or_overstated":[],"important_findings_not_integrated":[],"paper_summary_sections":[],"repetitions":[],"weak_transitions":[],"comparison_problems":[],"conclusion_problems":[],"overall":"..."}}
Each issue should contain severity, section, problem, why_it_matters and revision. Do not flag a topic merely because a paper did not report it. Flag only problems that affect the scientific usefulness of the report."""
    try:
        out = llm.complete_json("You are a demanding journal editor.", prompt, max_tokens=4200, tier="S", priority="report")
        return out if isinstance(out, dict) else {}, True
    except Exception:
        return {}, False
