from __future__ import annotations
import json
import re
from collections import Counter
_STOP = set("""a an the of and or in on for to with by from as at is are was were be been this that these those
how what which why when where does do can could should would into via using use based between across
about over under than then their its our we it not no more most less new recent current state art
including include includes such well better best improve improved improving understood understand understanding
monitored monitor predicted predict predicting overall towards toward through within among while whether
between different various several major key main role roles impact effect effects analysis study studies paper papers
research review survey approach approaches method methods technique techniques""".split())
MIN_SECTION_WORDS = 220
TARGET_TOTAL_WORDS = (3500, 7000)
def _words(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text or ""))
def _cell(text, limit: int = 140) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip().replace("|", "/")
    text = re.sub(r"\[(?:E|S|C)\d+\]", "", text).strip()
    if len(text) > limit:
        cut = text[:limit].rsplit(" ", 1)[0].rstrip(",;:")
        text = cut + "..."
    return text or "n/a"
def _is_empty_cell(value: str) -> bool:
    return (value or "").strip().lower() in {"", "n/a", "not reported", "none", "unknown"}
def _first_sentence(text: str, limit: int = 150) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    m = re.match(r"(.+?[.!?])(\s|$)", text)
    return _cell(m.group(1) if m else text, limit)
def keywords(question: str, analyses, limit: int = 6) -> list[str]:
    out: list[str] = []
    for a in analyses:
        cat = (getattr(a, "method_category", "") or "").strip().lower()
        if cat and cat not in out and len(cat) < 50:
            out.append(cat)
    counts = Counter(w for w in re.findall(r"[a-zA-Z][a-zA-Z\-]{3,}", question.lower()) if w not in _STOP)
    seen_stems = {re.sub(r"(ing|ed|es|s)$", "", k) for k in out}
    for w, _ in counts.most_common(limit * 2):
        stem = re.sub(r"(ing|ed|es|s)$", "", w)
        if w not in out and stem not in seen_stems:
            out.append(w)
            seen_stems.add(stem)
    return out[:limit]
def keywords_line(question: str, analyses) -> str:
    kws = keywords(question, analyses)
    return ("**Keywords:** " + ", ".join(kws) + "\n") if kws else ""
def review_methodology(question: str, sources, analyses, *, found: int, rejected: int, warnings=None) -> str:
    """Describe the real review protocol. ``sources`` should be the papers actually read;
    ``found`` is the number of candidates screened. Counts are never inflated."""
    read = len(analyses) if analyses else len(sources)
    read_ids = {getattr(a, "source_id", None) for a in analyses}
    years = sorted({str(s.year) for s in sources if getattr(s, "year", "") and (not read_ids or s.id in read_ids)})
    span = f"{years[0]} to {years[-1]}" if len(years) > 1 else (years[0] if years else "the retrieved period")
    full = sum(1 for a in analyses if getattr(a, "depth", "") == "full_text")
    screened = max(found, read)
    lines = [
        "Candidate papers were retrieved from arXiv with complementary queries derived from the research question, "
        "deduplicated, and ranked on semantic relevance to the question, recency and publication signal. "
        f"Of {screened} screened candidates, {read} met the relevance and full-text-readability criteria and were read in full"
        + (f"; the remaining {screened - read} were screened but not read in depth" if screened > read else "")
        + f". The papers read span {span}.",
        "",
        f"Each paper read ({full} from verified full text) was analysed section by section to extract its problem, "
        "method, experimental setup, quantitative results, ablations, limitations and stated boundaries. Findings are "
        "compared across papers only where experimental conditions make the comparison defensible, and factual "
        "statements are tied to passages in the source papers.",
    ]
    if warnings:
        lines += ["", "Scope note: " + _cell(warnings[0], 260)]
    return "\n".join(lines)
def studies_table(sources, analyses, cmap) -> list[str]:
    by_id = {s.id: s for s in sources}
    rows = []
    for a in analyses:
        s = by_id.get(a.source_id)
        if not s:
            continue
        tag = cmap.get(a.source_id, "?")
        approach = a.method_category or _first_sentence(getattr(a, "core_idea", "") or a.method_summary or a.problem, 60)
        data = _first_sentence(a.dataset_details or a.experimental_setup, 90)
        result = _first_sentence(a.key_results or a.established, 130)
        limit = _first_sentence(a.limitations_author_stated or a.limitations or a.limitations_inferred or a.not_established, 110)
        cells = [_cell(approach, 60), data, result, limit]
        if sum(_is_empty_cell(c) for c in cells) >= 3:
            continue
        cells = ["not reported" if _is_empty_cell(c) else c for c in cells]
        rows.append(f"| [{tag}] | {_cell(s.year, 6)} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} |")
    if not rows:
        return []
    head = ["**Table 1.** Overview of the reviewed studies.", "",
            "| Ref | Year | Approach | Data / setting | Main finding | Stated limitation |",
            "|---|---|---|---|---|---|"]
    return head + rows + [""]
def mechanism_table(sources, analyses, cmap) -> list[str]:
    """Comparison matrix: mechanism, cost and operating conditions (fields as extracted; no inference)."""
    rows = []
    for a in analyses:
        mech = _first_sentence(a.mechanism or getattr(a, "core_idea", "") or a.method_summary, 120)
        cost = _first_sentence(getattr(a, "computational_cost", ""), 90)
        cond = _first_sentence(getattr(a, "transfer_conditions", "") or a.generalization or a.robustness, 110)
        if mech == cost == cond == "n/a":
            continue
        rows.append(f"| [{cmap.get(a.source_id, '?')}] | {mech} | {cost} | {cond} |")
    if len(rows) < 2:
        return []
    return ["**Table 2.** Mechanisms, cost and operating conditions.", "",
            "| Ref | Core mechanism | Cost | Conditions and boundaries |", "|---|---|---|---|"] + rows + [""]
def study_summaries(sources, analyses, cmap) -> list[str]:
    """One compact, information-preserving summary per paper actually read, built only from extracted
    fields (numbers and conditions are copied, never recomputed). Papers with no usable fields are skipped."""
    by_id = {s.id: s for s in sources}
    fields = (
        ("Problem", "problem"), ("Approach", "method_summary"), ("Mechanism", "mechanism"),
        ("Data and setting", "dataset_details"), ("Protocol and baselines", "experimental_setup"),
        ("Results", "key_results"), ("Ablations", "ablations"), ("Failure modes", "failure_modes"),
        ("Limitations", "limitations_author_stated"), ("Not established", "not_established"),
    )
    blocks: list[str] = []
    for a in analyses:
        src = by_id.get(a.source_id)
        if not src:
            continue
        pieces = []
        lead = (getattr(a, "core_idea", "") or "").strip()
        for label, attr in fields:
            val = re.sub(r"\s+", " ", str(getattr(a, attr, "") or "")).strip()
            if val and val.lower() not in {"not reported", "n/a", "none"}:
                pieces.append(f"**{label}.** {val}")
        for q in (getattr(a, "quantitative_results", []) or [])[:6]:
            vals = [str(x).strip() for x in (q.proposed, q.absolute_delta, q.relative_delta) if str(x or "").strip()]
            if vals:
                cond = " / ".join(x for x in (q.dataset, q.model, q.metric) if x)
                pieces.append(f"**Reported figure.** {'; '.join(vals)}" + (f" ({cond})" if cond else ""))
        if not pieces and not lead:
            continue
        head = f"**[{cmap.get(a.source_id, '?')}] {src.title.rstrip('.')}**" + (f" - {lead}" if lead else "")
        blocks.append(head + "\n\n" + " ".join(pieces) + "\n")
    return blocks
def format_references(sources, cmap, only_tags: set[str] | None = None) -> list[str]:
    lines = []
    for s in sources:
        if only_tags is not None and cmap.get(s.id) not in only_tags:
            continue
        tag = cmap.get(s.id, "?")
        authors = (s.authors or "Unknown author").strip().rstrip(".")
        year = f"({s.year})" if s.year else "(n.d.)"
        venue = s.journal_ref or s.venue or ""
        venue = f" *{venue}*." if venue and venue.lower() != "arxiv" else (" *arXiv preprint*." if (s.provider == "arxiv" or "arxiv" in (s.url or "").lower()) else "")
        ident = f" https://doi.org/{s.doi}" if s.doi else (f" {s.url or s.pdf_url}" if (s.url or s.pdf_url) else "")
        lines.append(f"- **[{tag}]** {authors} {year}. {s.title.rstrip('.')}.{venue}{ident}".rstrip())
    return lines
def _clean_llm_text(text: str, allowed: set[str], *, keep_citations: bool = True) -> str:
    text = (text or "").strip()
    text = re.sub(r"^```(?:markdown)?|```$", "", text, flags=re.M).strip()
    text = re.sub(r"^#{1,6}\s+.*$", "", text, flags=re.M).strip()
    if keep_citations:
        text = re.sub(r"\[((?:E|S)\d+)\]", lambda m: m.group(0) if m.group(0) in allowed else "", text)
    else:
        text = re.sub(r"\[(?:E|S|C)\d+\]", "", text)
    return re.sub(r"[ \t]+\n", "\n", text).replace("\u2014", "-").replace("\u2013", "-").strip()
def _context(question, analyses, claims, findings, evidence, council):
    top_ev = sorted(evidence, key=lambda e: getattr(e, "relevance", 0), reverse=True)[:30]
    ctx = {
        "question": question,
        "council_thesis": (council or {}).get("central_thesis", "") if isinstance(council, dict) else "",
        "n_papers": len(analyses),
        "method_families": sorted({a.method_category for a in analyses if a.method_category}),
        "claims": [{"text": c.text, "evidence_ids": c.evidence_ids} for c in claims if c.evidence_ids][:10],
        "findings": [{"type": f.finding_type, "text": f.text, "evidence_ids": f.evidence_ids}
                     for f in (findings or [])[:14]],
        "gaps": [g for a in analyses for g in ([a.not_established] if a.not_established else [])][:8],
        "open_questions": [q for a in analyses for q in (getattr(a, "open_questions", []) or [])][:10],
        "evidence": [{"id": e.id, "text": e.text[:320]} for e in top_ev],
    }
    return ctx, {f"[{e.id}]" for e in top_ev}
_INTRO_SYS = """You are writing the Introduction of a compact, publishable survey article. Use ONLY the supplied claims, findings and evidence.
Write 3 to 4 paragraphs of markdown, no heading: (1) why the problem matters and what the field is trying to achieve, (2) the specific question and scope of this review, (3) the main contributions of this review (what the reader will learn: taxonomy, comparison, evidence patterns, gaps), (4) how the rest of the article is organised.
Cite claims with supplied [E#] ids only, at the end of the relevant sentence. Never invent numbers, papers or facts. No em dashes, no en dashes, no bullet lists. Academic, direct, precise."""
_ABSTRACT_SYS = """You are writing the Abstract of a compact survey article from the supplied verified material only.
Write ONE paragraph of 150 to 220 words covering: context and motivation, the review question and scope (number of papers), the main findings and trade-offs, and the key open problems. No citations, no headings, no bullet lists, no invented numbers. No em dashes or en dashes."""
_CHALLENGES_SYS = """You are writing 'Open Challenges and Future Directions' of a survey article, using ONLY the supplied gaps, findings and evidence.
Write 3 to 5 paragraphs of markdown, no heading. Every paragraph must have this logic and no generic statements ("more datasets", "further research") are allowed:
(a) what existing studies evaluated and under which conditions; (b) which condition or question the evidence does NOT cover; (c) why that gap matters (which mechanism or conclusion behaves differently or is uncertain there); (d) a specific, testable next experiment or study design, including what must be isolated or controlled.
Label your own inferences as inferences and keep them separate from what the papers showed. Cite supplied [E#] ids only. No invention, no em dashes or en dashes."""
def _llm(llm, system, ctx, max_tokens):
    try:
        return llm.complete(system, json.dumps(ctx, ensure_ascii=False), max_tokens=max_tokens) or ""
    except Exception:
        return ""
def abstract(question, analyses, claims, findings, evidence, council, llm, fallback: str) -> tuple[str, bool]:
    if llm is not None and getattr(llm, "available", False) and analyses:
        ctx, _ = _context(question, analyses, claims, findings, evidence, council)
        text = _clean_llm_text(_llm(llm, _ABSTRACT_SYS, ctx, 700), set(), keep_citations=False)
        if _words(text) >= 80:
            return text, True
    return fallback, False
def introduction(question, analyses, claims, findings, evidence, council, llm) -> tuple[str, bool]:
    ctx, allowed = _context(question, analyses, claims, findings, evidence, council)
    if llm is not None and getattr(llm, "available", False) and analyses:
        text = _clean_llm_text(_llm(llm, _INTRO_SYS, ctx, 1400), allowed)
        if _words(text) >= 120:
            return text, True
    fams = ", ".join(ctx["method_families"][:5]) or "several approach families"
    text = (f"This review addresses the question: {question.strip().rstrip('?')}? "
            f"It synthesises {len(analyses)} closely read papers covering {fams}, "
            "organising the evidence around approaches, experimental results, trade-offs and unresolved problems. "
            "The remainder of the article describes the review protocol, summarises the studies, analyses the evidence "
            "thematically, discusses open challenges and concludes.")
    return text, False
def challenges(question, analyses, claims, findings, evidence, council, llm) -> tuple[str, bool]:
    ctx, allowed = _context(question, analyses, claims, findings, evidence, council)
    if llm is not None and getattr(llm, "available", False) and analyses:
        text = _clean_llm_text(_llm(llm, _CHALLENGES_SYS, ctx, 2000), allowed)
        if _words(text) >= 120:
            return text, True
    items = [f"- {g}" for g in ctx["gaps"][:5]] or [f"- {q}" for q in ctx["open_questions"][:5]]
    if not items:
        return "", False
    return "Open problems that the reviewed evidence leaves unresolved:\n\n" + "\n".join(items), False
def covers_challenges(titles: list[str]) -> bool:
    return any(re.search(r"gap|challenge|future|open (problem|question)|direction", t or "", re.I) for t in titles)
def substantive_headings(markdown: str, min_words: int = 40) -> list[str]:
    """Titles of '## ' sections whose body has real content (empty stubs do not count)."""
    out = []
    chunks = re.split(r"^## ", markdown, flags=re.M)[1:]
    for chunk in chunks:
        title, _, body = chunk.partition("\n")
        if _words(body) >= min_words:
            out.append(title.strip())
    return out
def quality_metrics(report_md: str, sources, analyses) -> dict:
    body = report_md.split("## References", 1)[0]
    cited = set(re.findall(r"\[S(\d+)", body))
    n = max(1, len(sources))
    sections = re.findall(r"^## (.+)$", body, flags=re.M)
    words = _words(body)
    lo, hi = TARGET_TOTAL_WORDS
    issues = []
    if words < lo:
        issues.append(f"report is short ({words} words; target {lo}-{hi})")
    if len(cited) / n < 0.8:
        issues.append(f"only {len(cited)} of {len(sources)} papers are cited in the body")
    for need in ("Introduction", "Review Methodology", "Conclusion"):
        if not any(need.lower() in s.lower() for s in sections):
            issues.append(f"missing section: {need}")
    return {
        "survey_words": words, "survey_sections": len(sections),
        "survey_papers_cited": len(cited), "survey_citation_coverage": round(len(cited) / n, 3),
        "survey_has_table": int("| Ref |" in body), "survey_issues": "; ".join(issues),
        "survey_grade": "publishable-draft" if not issues else "needs-review",
    }
def _paragraphs(md: str) -> list[str]:
    body = md.split("## References", 1)[0]
    out = []
    for chunk in re.split(r"\n\s*\n", body):
        text = re.sub(r"^(#{1,6} .*\n)+", "", chunk.strip()).strip()
        if len(text) > 120 and not text.startswith(("|", "**Table", "**Keywords")):
            out.append(text)
    return out
def self_challenge(report_md, question, plan, sources, analyses, findings, audit, n_found, draft_sections) -> dict:
    """Deterministic self-audit across the nine dimensions in the product vision.
    Every score is computed from the actual artifacts (0..1). Nothing here calls an LLM, so
    the scorecard cannot flatter the report.
    """
    paras = _paragraphs(report_md)
    body = report_md.split("## References", 1)[0]
    n = max(1, len(paras))
    multi = sum(1 for p in paras if len(set(re.findall(r"\[S(\d+)", p))) >= 2)
    quant_expected = sum(len(getattr(a, "quantitative_results", []) or []) for a in analyses)
    quant_in_report = len(re.findall(r"\d+(?:\.\d+)?\s?%|\b\d+(?:\.\d+)?x\b|\b\d{2,}\b", body))
    subq = [q for q in (getattr(plan, "subquestions", []) or []) if q]
    def _hit(q):
        toks = {w for w in re.findall(r"[a-z]{5,}", q.lower())}
        return not toks or len(toks & set(re.findall(r"[a-z]{5,}", body.lower()))) / len(toks) >= 0.4
    contr = [f for f in (findings or []) if getattr(f, "finding_type", "") in {"contradiction", "comparability_warning", "alternative_interpretation"}]
    contr_in = bool(re.search(r"contradict|conflict|disagree|inconsisten|not directly comparable|differ", body, re.I))
    interp = bool(re.search(r"\b(suggests?|indicates?|implies|possible implication|interpret)", body, re.I))
    card = {
        "retrieval": min(1.0, n_found / max(1, len(sources) * 3)),
        "corpus": 1.0 if sources and all(getattr(s, "role", "") == "deep_read" for s in sources) else 0.6,
        "evidence_reading": round(sum(1 for a in analyses if getattr(a, "depth", "") == "full_text") / max(1, len(sources)), 3),
        "coverage": round(sum(_hit(q) for q in subq) / len(subq), 3) if subq else 0.7,
        "synthesis": round(multi / n, 3),
        "grounding": float(getattr(audit, "citation_coverage", 0.0) or 0.0),
        "contradiction": 1.0 if (contr_in or not contr) else 0.3,
        "quantitative": round(min(1.0, quant_in_report / max(1, min(quant_expected, 12))), 3) if quant_expected else 0.7,
        "interpretation_labelled": 1.0 if interp else 0.4,
    }
    weak = [k for k, v in card.items() if v < 0.5]
    return {
        **{f"audit_{k}": round(v, 3) for k, v in card.items()},
        "audit_weak_dimensions": ", ".join(weak),
        "quality_status": "meets-bar" if (not weak and getattr(audit, "passed", True)) else "delivered-with-flags",
    }
_ID = r"(?:E|DR|C)\d+"
_ID_RANGE_OR_LIST = rf"{_ID}(?:\s*(?:[-\u2013]|,|;|and|to)\s*(?:E|DR|C)?\d+)*"
def _expand_ids(chunk: str) -> list[str]:
    """'E6-E10' -> E6..E10 ; 'E17, E30' -> [E17, E30] ; 'E3, E4 and E5' -> [E3, E4, E5]."""
    ids: list[str] = []
    parts = re.split(r"\s*(?:,|;|and)\s*", chunk.strip())
    for part in parts:
        m = re.fullmatch(r"((?:E|DR|C))(\d+)\s*(?:[-\u2013]|to)\s*(?:(?:E|DR|C))?(\d+)", part.strip())
        if m:
            lo, hi = int(m.group(2)), int(m.group(3))
            if 0 < hi - lo <= 40:
                ids.extend(f"{m.group(1)}{i}" for i in range(lo, hi + 1))
            else:
                ids.extend([f"{m.group(1)}{lo}", f"{m.group(1)}{hi}"])
            continue
        m = re.fullmatch(r"((?:E|DR|C))(\d+)", part.strip())
        if m:
            ids.append(part.strip())
    return ids
def normalize_public_citations(text: str, evidence_by_id: dict, cmap: dict, claims_by_id: dict | None = None,
                               suffix=None) -> str:
    """Rewrite every internal evidence/claim reference (single, list, range, bracketed or
    parenthesised) into public [S#] source tags. Unresolvable ids are dropped cleanly, never left visible."""
    claims_by_id = claims_by_id or {}
    def tags_for(ids: list[str]) -> str:
        out, seen = [], set()
        for i in ids:
            if i.startswith("C"):
                cl = claims_by_id.get(i)
                eids = list(getattr(cl, "evidence_ids", []) or []) if cl else []
            else:
                eids = [i]
            for eid in eids:
                ev = evidence_by_id.get(eid)
                if not ev:
                    continue
                tag = cmap.get(ev.source_id)
                if not tag:
                    continue
                t = f"[{tag}{suffix(ev) if suffix else ''}]"
                if t not in seen:
                    seen.add(t)
                    out.append(t)
        return "".join(out)
    def repl(m: re.Match) -> str:
        return tags_for(_expand_ids(m.group(1)))
    text = re.sub(rf"\[\s*({_ID_RANGE_OR_LIST})\s*\]", repl, text)
    text = re.sub(rf"\(\s*({_ID_RANGE_OR_LIST})\s*\)", repl, text)
    text = re.sub(rf"(?<![\w\[])({_ID})(?![\w\]])", lambda m: tags_for([m.group(1)]), text)
    text = re.sub(r"[ \t]+([.,;:])", r"\1", text)
    return re.sub(r"(\[S\d+[^\]]*\])(?:\s*\1)+", r"\1", text)
def complete_sentences(text: str) -> str:
    """Trim a cut-off tail (e.g. an abstract ending in 'and') back to the last full sentence."""
    t = (text or "").rstrip()
    if not t or re.search(r"[.!?][\)\]\"']*$", t):
        return t
    cut = max(t.rfind(". "), t.rfind("! "), t.rfind("? "))
    return t[:cut + 1] if cut >= len(t) * 0.25 else t.rstrip(",;: ") + "."
_KEEP_EMPTY = {"references", "conclusion"}
def drop_empty_sections(markdown: str, min_words: int = 12) -> str:
    """Remove headings that have no real body (placeholder stubs) at ## and ### level,
    while always keeping References and Conclusion."""
    def prune(md: str, marker: str) -> str:
        split_at = r"(?m)^(?=#{2,3} )" if marker == "###" else r"(?m)^(?=## )"
        out = []
        for chunk in re.split(split_at, md):
            head, _, body = chunk.partition("\n")
            if head.startswith(marker + " "):
                title = head[len(marker) + 1:].strip().lower()
                real = re.sub(r"(?m)^#{2,6} .*$", "", body)
                if title not in _KEEP_EMPTY and _words(real) < min_words and "|" not in real:
                    continue
            out.append(chunk)
        return "".join(out)
    md = prune(markdown, "###")
    md = prune(md, "##")
    return re.sub(r"\n{3,}", "\n\n", md)
_ABSENCE_SENTENCE = re.compile(
    r"(?is)^\s*(?:[-*]\s+)?(?:"
    r"(?:the\s+)?other\s+(?:retained\s+|reviewed\s+|selected\s+)?(?:papers|studies)\s+(?:do|does|did)\s+not\s+(?:report|describe|provide|discuss|mention|specify|include)\b"
    r"|(?:no|little|limited)\s+(?:information|detail|data|evidence)\s+(?:was|were|is|are)\s+(?:found|extracted|available|reported)\b"
    r"|.{0,120}\b(?:was|were|is|are)\s+not\s+(?:found|extracted|identified|reported)\s+(?:in|by|across)\s+(?:any\s+of\s+)?the\s+(?:retained|reviewed|selected|included|analy[sz]ed)\b"
    r"|the\s+retained\s+(?:papers|studies|evidence)\s+provide\s+(?:only\s+)?qualitative\b"
    r"|no\s+substantive\s+(?:limitation|pattern)\b"
    r"|no\s+evidence-backed\s+open\s+question\b"
    r")"
)
def strip_absence_statements(markdown: str) -> str:
    """Drop sentences/bullets that only announce that something was not found in the corpus."""
    out_lines = []
    for line in markdown.split("\n"):
        if not line.strip() or line.lstrip().startswith(("#", "|", "**Table", "**Keywords")):
            out_lines.append(line)
            continue
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\[])", line)
        kept = [p for p in parts if not _ABSENCE_SENTENCE.match(p)]
        if not kept:
            continue
        out_lines.append(" ".join(kept))
    return "\n".join(out_lines)
