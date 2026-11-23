from __future__ import annotations
from collections import defaultdict
import json
import re
from ..llm import LLMClient
from ..schemas import canonical_report_type, Claim, Evidence, PaperAnalysis, QuantResult, ResearchPlan, Source
from .grounding import extract_citation_tags, safe_grounded_text
_PROSE_SYSTEM = (
    "You are a precise research writer. Turn the given evidence bullets into one tight, "
    "well-written paragraph for a research report. Use ONLY the facts present in the "
    "bullets -- do not add outside knowledge, numbers, or papers. Every sentence containing "
    "a specific fact must end with the citation tag(s) already attached to that bullet, "
    "e.g. [E3]. Do not invent new citation tags. Keep it to 3-6 sentences. Return plain text only."
)
_SYNTHESIS_SYSTEM = (
    "You are a senior research scientist writing the cross-paper synthesis section of a "
    "literature review. You are given CLAIMS (tagged C#), EVIDENCE (tagged E#/S#), and, when "
    "available, per-paper ANALYSES for the CORE papers only (tagged with that paper's S# and "
    "fields like method/results/ablations/limitations). Reason ACROSS these items -- do not "
    "just restate them one by one. Identify: what the evidence suggests actually works and "
    "under what conditions; where retained studies agree, disagree, or are complementary; recurring "
    "failure modes; and concrete, evidence-motivated research gaps and directions. If two "
    "results are not directly comparable (different datasets/metrics/definitions), say so "
    "explicitly rather than forcing a comparison. Use ONLY the material given -- never "
    "introduce a fact, number, or paper not present below. Every list item MUST include the "
    "id(s) (C#/E#/S#) it is based on. Be conservative: if the material doesn't support a "
    "strong claim, say so instead of overreaching. For each gap, explain what existing "
    "evidence establishes, what remains uncertain, and WHY existing evidence can't resolve "
    "it -- not just 'more research is needed'. For each direction, connect it to a specific "
    "gap and describe what experiment/evidence would actually address it. "
    "Respond with ONLY JSON matching this schema: "
    '{"synthesis": "4-8 sentences, citations required", '
    '"what_works": [{"text": "...", "ids": ["C1"]}], '
    '"limitations_pattern": [{"text": "...", "ids": ["E2"]}], '
    '"incomparable_results": [{"text": "why these results can\'t be directly compared", "ids": ["S1","S2"]}], '
    '"gaps": [{"text": "the gap itself", "known": "what existing evidence establishes", '
    '"uncertain": "what remains uncertain", "why_unresolved": "why existing evidence can\'t resolve it", '
    '"kind": "explicit|evidence_derived|contradiction|evaluation|methodological|practical", "ids": ["E3"]}], '
    '"directions": [{"text": "the proposed direction", "rationale": "...", '
    '"experiment": "what experiment/evidence would address it", "ids": ["C2","E5"]}]}'
)
def citation_map(sources: list[Source]) -> dict[str, str]:
    """Return stable S# tags, filling only missing tags deterministically."""
    used = {s.citation_tag for s in sources if getattr(s, "citation_tag", "")}
    out: dict[str, str] = {}
    next_n = 1
    for source in sources:
        tag = getattr(source, "citation_tag", "") or ""
        if not tag:
            while f"S{next_n}" in used:
                next_n += 1
            tag = f"S{next_n}"
            source.citation_tag = tag
            used.add(tag)
            next_n += 1
        out[source.id] = tag
    return out
def evidence_strength(
    sources: list[Source], evidence: list[Evidence], analyses: list[PaperAnalysis] | None = None
) -> tuple[str, float, str]:
    """A deterministic (non-LLM) label for how much a reader should trust
    the depth of this report. This is plain Python logic, not a model's
    self-rating -- a model asked to grade its own evidence tends to be
    miscalibrated in exactly the direction that produces confident,
    detailed reports built on almost nothing.
    """
    analyses = analyses or []
    n_sources = len(sources)
    n_evidence = len(evidence)
    full_text_evidence = sum(1 for e in evidence if "full text" in e.locator)
    full_text_ratio = (full_text_evidence / n_evidence) if n_evidence else 0.0
    n_deep_read = sum(1 for a in analyses if a.depth == "full_text")
    deep_coverage = n_deep_read / max(n_sources, 1)
    if n_sources == 0 or n_evidence == 0:
        return "none", 0.0, "No sources were retained, or no evidence could be extracted from them."
    if n_sources < 2 or n_evidence < 3:
        return "limited", 0.3, f"Only {n_sources} source(s) and {n_evidence} evidence snippet(s) were available."
    if n_deep_read >= 3 and deep_coverage >= 0.6:
        return (
            "strong", 0.9,
            f"{n_sources} sources retrieved; {n_deep_read} ({deep_coverage:.0%}) were read in full and deeply analyzed "
            f"(problem/method/experiments/limitations), not just their abstracts.",
        )
    if n_deep_read >= 1:
        return (
            "moderate", 0.7,
            f"{n_sources} sources retrieved; {n_deep_read} ({deep_coverage:.0%}) received full-text deep analysis, while the remaining sources had shallower coverage.",
        )
    if analyses and n_deep_read == 0:
        return (
            "moderate", 0.55,
            f"{n_sources} sources retrieved and {len(analyses)} analyzed in depth, but full text "
            f"could not be fetched for any of them -- deep analysis relied on abstracts only.",
        )
    if full_text_ratio < 0.3 and not analyses:
        return (
            "moderate", 0.6,
            f"{n_sources} sources retrieved, but most evidence comes from abstracts rather than full text. "
            f"No LLM was available for deep per-paper analysis in this run.",
        )
    if n_sources >= 3 and n_evidence >= 5:
        return "strong", 0.85, f"{n_sources} sources with {n_evidence} evidence snippets, much of it from full text or deep-read papers."
    return "moderate", 0.6, f"{n_sources} sources with {n_evidence} evidence snippets."
def _safe_prose(bullets: list[str], allowed_tags: set[str], llm: LLMClient | None) -> tuple[str | None, bool]:
    """Thin wrapper kept for readability at call sites in this module."""
    return safe_grounded_text(_PROSE_SYSTEM, bullets, allowed_tags, llm)
def _evidence_bullet(ev: Evidence, cmap: dict[str, str]) -> str:
    src_tag = cmap.get(ev.source_id, "S?")
    return f"{ev.text} [{ev.id}][{src_tag}]"
def _group_by_kind(evidence: list[Evidence]) -> dict[str, list[Evidence]]:
    groups: dict[str, list[Evidence]] = {}
    for e in evidence:
        groups.setdefault(e.kind, []).append(e)
    for items in groups.values():
        items.sort(key=lambda e: e.relevance, reverse=True)
    return groups
_KIND_TITLES = {
    "method": "Methods and Approaches",
    "result": "Key Results",
    "ablation": "Ablations -- What Actually Caused the Improvement",
    "limitation": "Limitations and Open Challenges",
    "dataset": "Datasets and Benchmarks",
    "context": "Additional Context",
}
_GAP_KIND_LABELS = {
    "explicit": "Explicit (authors' stated future work)",
    "evidence_derived": "Evidence-derived (visible in the experiments, not stated by authors)",
    "contradiction": "Contradiction (sources disagree)",
    "evaluation": "Evaluation gap (existing benchmarks don't measure this)",
    "methodological": "Methodological gap (an untested combination/approach)",
    "practical": "Practical gap (works academically, hard to deploy)",
}
_EVIDENCE_STATUS_LABELS = {
    "demonstrated": "Demonstrated (direct experimental evidence)",
    "author_reported": "Author-reported",
    "inferred": "Inferred (synthesized across sources)",
    "not_established": "Not established",
}
def _truncate(text: str, n: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"
def _source_sort_key(s: Source) -> float:
    return -s.score
def _validated_items(raw_items, allowed: set[str], text_key: str = "text", ids_key: str = "ids") -> list[dict]:
    """Shared per-item validation for the cross-paper synthesis call: an
    item is kept only if EVERY id it cites is one it was actually given.
    A partially-correct LLM response still contributes the items that check
    out instead of being discarded wholesale (same philosophy as
    `claims.py`)."""
    out = []
    if not isinstance(raw_items, list):
        return out
    for item in raw_items:
        if not isinstance(item, dict) or not item.get(text_key):
            continue
        ids = [i for i in (item.get(ids_key) or []) if isinstance(i, str)]
        if not ids or not set(ids).issubset(allowed):
            continue
        out.append({**item, ids_key: ids})
    return out
def _evidence_status_for_ids(ids: list[str], claims_by_id: dict[str, Claim], evidence_by_id: dict[str, Evidence]) -> str:
    """Same deterministic idea as `claims.py`'s per-claim status, applied to
    a synthesis-call list item: prefer a cited claim's already-computed
    status; otherwise infer from evidence kind; multi-source items are
    "inferred" by definition (a cross-paper synthesis item necessarily is)."""
    claim_statuses = [claims_by_id[i].evidence_status for i in ids if i in claims_by_id]
    if claim_statuses:
        if all(s == "demonstrated" for s in claim_statuses):
            return "demonstrated"
        return "inferred"
    cited_evidence = [evidence_by_id[i] for i in ids if i in evidence_by_id]
    sources = {e.source_id for e in cited_evidence}
    if len(sources) > 1:
        return "inferred"
    if any(e.kind in {"result", "ablation"} for e in cited_evidence):
        return "demonstrated"
    return "author_reported" if cited_evidence else "not_established"
def _cross_paper_synthesis(
    claims: list[Claim],
    evidence: list[Evidence],
    core_analyses: list[PaperAnalysis],
    cmap: dict[str, str],
    llm: LLMClient | None,
    report_type: str = "literature_review",
) -> tuple[dict | None, bool]:
    """One consolidated, citation-checked LLM call that reasons ACROSS
    CORE papers rather than about one at a time. Supporting-paper evidence
    deliberately isn't fed into this call's ANALYSES block (only the
    claims/evidence pool, which candidate evidence also contributes to) --
    keeping the per-paper reasoning grounded in each RED paper's own analysis
    is what keeps a excluded candidate's stray snippet from being woven into
    "what the retained evidence shows". Returns (data, llm_used); data is
    None if the LLM is unavailable or nothing in its response survives
    validation."""
    llm = llm or LLMClient()
    if not llm.available or (not claims and not core_analyses):
        return None, False
    claim_lines = [f"[{c.id}] {c.text} | evidence: {', '.join(c.evidence_ids) or 'none'}" for c in claims[:20]]
    evidence_lines = [
        f"[{e.id}] [{cmap.get(e.source_id, 'S?')}] ({e.kind}) {e.text}"
        for e in evidence[:80]
    ]
    analysis_lines = []
    for a in core_analyses:
        tag = cmap.get(a.source_id, "S?")
        bits = [f"{tag} [{a.method_category or 'uncategorized'}]"]
        if a.problem:
            bits.append(f"problem: {a.problem}")
        if a.research_objective:
            bits.append(f"objective: {a.research_objective}")
        if a.core_idea:
            bits.append(f"core idea: {a.core_idea}")
        if a.method_summary:
            bits.append(f"method: {a.method_summary}")
        if a.mechanism:
            bits.append(f"mechanism: {a.mechanism}")
        if a.experimental_setup:
            bits.append(f"setup: {a.experimental_setup}")
        if a.dataset_details:
            bits.append(f"datasets: {a.dataset_details}")
        if a.baselines:
            bits.append(f"baselines: {a.baselines}")
        if a.metrics:
            bits.append(f"metrics: {a.metrics}")
        if a.training_protocol:
            bits.append(f"training protocol: {a.training_protocol}")
        if a.statistical_uncertainty:
            bits.append(f"uncertainty: {a.statistical_uncertainty}")
        if a.error_analysis:
            bits.append(f"error analysis: {a.error_analysis}")
        if a.key_results:
            bits.append(f"results: {a.key_results}")
        if a.ablations:
            bits.append(f"ablations: {a.ablations}")
        if a.limitations:
            bits.append(f"limitations: {a.limitations}")
        if a.established:
            bits.append(f"established: {a.established}")
        if a.not_established:
            bits.append(f"not established: {a.not_established}")
        analysis_lines.append(" | ".join(bits))
    user_msg = (
        f"REPORT PURPOSE: {report_type}\n"
        "Use the report purpose to decide which cross-paper relationships matter most. "
        "Do not change the evidentiary standard based on the purpose.\n\n"
        + "CLAIMS:\n" + "\n".join(claim_lines)
        + "\n\nEVIDENCE:\n" + "\n".join(evidence_lines)
        + "\n\nCORE PAPER ANALYSES:\n" + "\n".join(analysis_lines)
    )
    try:
        data = llm.complete_json(_SYNTHESIS_SYSTEM, user_msg, max_tokens=3000, tier="M", priority="report")
    except Exception:
        return None, False
    if not isinstance(data, dict):
        return None, False
    allowed = extract_citation_tags(user_msg)
    synthesis_text = str(data.get("synthesis", "")).strip()
    if synthesis_text:
        tags = extract_citation_tags(synthesis_text)
        sentences = [x.strip() for x in re.split(r"(?<=[.!?])\s+", synthesis_text) if x.strip()]
        if not tags.issubset(allowed) or not sentences or any(not extract_citation_tags(sentence) for sentence in sentences):
            synthesis_text = ""
    result = {
        "synthesis": synthesis_text,
        "what_works": _validated_items(data.get("what_works"), allowed),
        "limitations_pattern": _validated_items(data.get("limitations_pattern"), allowed),
        "incomparable_results": _validated_items(data.get("incomparable_results"), allowed),
        "gaps": _validated_items(data.get("gaps"), allowed),
        "directions": _validated_items(data.get("directions"), allowed),
    }
    if not any(result.values()):
        return None, False
    return result, True
def _render_quant_result(qr: QuantResult) -> str:
    bits = []
    if qr.dataset:
        bits.append(f"dataset: {qr.dataset}")
    if qr.model:
        bits.append(f"model: {qr.model}")
    delta = ""
    if qr.absolute_delta or qr.relative_delta:
        delta = " (Δ " + ", ".join(x for x in (qr.absolute_delta, qr.relative_delta) if x) + ")"
    result_bit = ""
    if qr.baseline or qr.proposed:
        result_bit = f"baseline {qr.baseline or '?'} → proposed {qr.proposed or '?'}{delta}"
    tag = f" [{qr.evidence_id}]" if qr.evidence_id and qr.evidence_id != "(assigned on merge)" else ""
    line = f"**{qr.metric}**: {result_bit}" if result_bit else f"**{qr.metric}**"
    if bits:
        line += " (" + ", ".join(bits) + ")"
    return f"- {line}{tag}"
def _render_core_paper_analysis(a: PaperAnalysis, source: Source | None, cmap: dict[str, str], evidence_by_source: dict[str, list[Evidence]]) -> list[str]:
    tag = cmap.get(a.source_id, "S?")
    title = source.title if source else a.source_id
    lines = [f"### {tag} -- {_truncate(title, 100)}\n"]
    identity_bits = []
    if source:
        if source.authors:
            identity_bits.append(source.authors)
        if source.year:
            identity_bits.append(source.year)
        if source.venue:
            identity_bits.append(source.venue)
    depth_note = "full paper" if a.depth == "full_text" else "abstract only (full text wasn't retrievable)"
    identity_line = " · ".join(identity_bits) + (" · " if identity_bits else "") + f"depth: {depth_note}"
    lines.append(f"*{identity_line}. Category: {a.method_category or 'uncategorized'}.*\n")
    def add(label: str, value: str) -> None:
        if value:
            lines.append(f"**{label}.** {value}")
    add("Problem", a.problem)
    add("Research objective", a.research_objective)
    add("Intervention point", a.intervention_point)
    add("Core idea", a.core_idea)
    add("Method", a.method_summary)
    add("Mechanism", a.mechanism)
    add("Experimental setup", a.experimental_setup)
    add("Results (summary)", a.key_results)
    if a.quantitative_results:
        lines.append("**Quantitative results:**")
        lines.extend(_render_quant_result(qr) for qr in a.quantitative_results)
    add("Ablations", a.ablations)
    add("Robustness", a.robustness)
    add("Computational cost", a.computational_cost)
    add("Reproducibility", a.reproducibility)
    add("Limitations (author-stated)", a.limitations_author_stated)
    add("Limitations (inferred, not stated by authors)", a.limitations_inferred)
    if a.established or a.not_established:
        lines.append("**What this paper's evidence actually establishes:**")
        if a.established:
            lines.append(f"- Establishes: {a.established}")
        if a.not_established:
            lines.append(f"- Does NOT establish: {a.not_established}")
    if a.caveat:
        lines.append(f"**Caveat (author framing vs. evidence).** {a.caveat}")
    quotes = [e for e in evidence_by_source.get(a.source_id, []) if "deep-read" in e.locator]
    if quotes:
        lines.append("\n*Verbatim supporting quotes (verified against this paper's own text):*")
        for e in quotes[:4]:
            lines.append(f"> \u201c{e.text}\u201d [{e.id}]")
    lines.append("")
    return lines
def build_report(
    question: str,
    plan: ResearchPlan,
    sources: list[Source],
    evidence: list[Evidence],
    claims: list[Claim],
    warnings: list[str],
    llm: LLMClient | None = None,
    analyses: list[PaperAnalysis] | None = None,
    rejected_count: int = 0,
) -> tuple[str, bool]:
    """Returns (markdown, llm_used_anywhere). `sources` may contain a mix
    of the RED/ GREEN role model -- see
    `pipeline/selection.py`. Uploaded sources are treated as core for
    report purposes (they get deep analysis and appear in the RED evidence corpus), since a user-supplied paper is by definition something
    they want analyzed, not backgrounded."""
    llm = llm or LLMClient()
    llm_used = False
    analyses = analyses or []
    allowed_source_ids = {s.id for s in sources if s.role == "deep_read"}
    sources = [s for s in sources if s.role == "deep_read"]
    evidence = [e for e in evidence if e.source_id in allowed_source_ids]
    analyses = [a for a in analyses if a.source_id in allowed_source_ids]
    cmap = citation_map(sources)
    sources_by_id = {s.id: s for s in sources}
    claims_by_id = {c.id: c for c in claims}
    evidence_by_id = {e.id: e for e in evidence}
    evidence_by_source: dict[str, list[Evidence]] = defaultdict(list)
    for e in evidence:
        evidence_by_source[e.source_id].append(e)
    core_sources = [s for s in sources if s.role == "deep_read"]
    excluded_sources = []
    core_analyses = [a for a in analyses if sources_by_id.get(a.source_id, Source(id="", title="")).role == "deep_read"]
    mode_labels = {
        "comparison": "Comparison", "landscape": "Landscape / State of the Art",
        "explanation": "Explanation", "design": "Design / Proposal", "standard_report": "Standard Report", "literature_review": "Literature Review",
    }
    parts: list[str] = [
        f"# Research Report: {question}\n",
        f"*Question type: {mode_labels.get(plan.mode, plan.mode)} · {len(core_sources)} deep-read paper(s) · {len(core_analyses)} structured readings*\n",
        "---\n",
    ]
    strength, strength_score, strength_reason = evidence_strength(sources, evidence, analyses)
    parts.append(
        f"## Evidence Base\n**Strength: {strength.upper()}** -- {strength_reason} "
        f"{len(core_sources)} paper(s) were selected as RED deep-read evidence; "
        f"{len(excluded_sources)} GREEN candidates were excluded from the report evidence base"
        + (f"; {rejected_count} candidate(s) were considered and rejected as not relevant enough." if rejected_count else ".")
        + " Read the sections below with this in mind -- a report grounded in few sources, or built "
        "without full-text access, should be treated as a starting point, not a settled answer.\n"
    )
    parts.append("## Research Question and Scope\n")
    parts.append(f"**Question:** {question}\n")
    if plan.subquestions:
        parts.append("**This report specifically tries to answer:**")
        parts.extend(f"- {sq}" for sq in plan.subquestions)
    if plan.analysis_axes:
        parts.append(f"\n**Dimensions used to evaluate/compare sources for this type of question:** {', '.join(plan.analysis_axes)}.")
    parts.append("")
    parts.append("## Executive Answer\n")
    top_claims = sorted(claims, key=lambda c: c.confidence, reverse=True)[:6]
    bullets = [f"{c.text} " + " ".join(f"[{eid}]" for eid in c.evidence_ids) for c in top_claims]
    allowed = extract_citation_tags("\n".join(bullets))
    prose, used = _safe_prose(bullets, allowed, llm)
    llm_used = llm_used or used
    if prose:
        parts.append(prose + "\n")
    elif bullets:
        parts.append("\n".join(f"- {b}" for b in bullets) + "\n")
    else:
        parts.append("Not enough evidence was retrieved to answer confidently. See Limitations below.\n")
    parts.append("## Research Landscape\n")
    categories: dict[str, list[PaperAnalysis]] = defaultdict(list)
    for a in core_analyses:
        categories[a.method_category or "Uncategorized"].append(a)
    if categories:
        parts.append("Core papers grouped by the kind of approach they take " "(categories are discovered from the papers themselves, not a fixed template):\n")
        for cat, items in sorted(categories.items(), key=lambda kv: -len(kv[1])):
            tags = ", ".join(f"[{cmap.get(a.source_id, '?')}]" for a in items)
            parts.append(f"- **{cat}** ({len(items)}): {tags}")
        parts.append("")
    else:
        parts.append(
            "No deep-read paper categorization is available for this run (no LLM configured, or no "
            "RED papers had usable text) -- see Findings by Theme below for a keyword-based grouping instead.\n"
        )
    parts.append("## Selected Core Literature\n")
    if core_sources:
        parts.append(
            "These papers were selected as core literature because they provide the strongest "
            "evidence relevant to this research question (not ranked -- listed in retrieval order):\n"
        )
        for s in core_sources:
            reasons = "; ".join(s.relevance_reasons) if s.relevance_reasons else "selected as directly relevant"
            parts.append(f"- **[{cmap.get(s.id, '?')}] {_truncate(s.title, 90)}** -- {reasons}")
        parts.append("")
    else:
        parts.append("No source cleared the bar for core (deep-read) literature in this run.\n")
    if core_analyses:
        parts.append("## Deep Analysis of Core Papers\n")
        parts.append(
            "Each paper below was read specifically for its problem, method, experiments, ablations, "
            "robustness, cost, and limitations -- not reduced to its abstract. Only evidence tagged to "
            "THAT paper's own text appears in its subsection (excluded-candidate evidence never appears "
            "here). Verbatim quotes are checked to actually appear in that paper's text before being "
            "shown, so the paraphrase above them can be spot-checked against a real sentence.\n"
        )
        ordered = sorted(core_analyses, key=lambda a: _source_sort_key(sources_by_id.get(a.source_id, Source(id="", title=""))))
        for a in ordered:
            parts.extend(_render_core_paper_analysis(a, sources_by_id.get(a.source_id), cmap, evidence_by_source))
    elif core_sources:
        parts.append("## Deep Analysis of Core Papers\n")
        parts.append("No LLM was available for deep reading in this run; core papers are represented only via the heuristic evidence in Findings by Theme below.\n")
    show_comparison = (plan.mode == "comparison" and len(sources) >= 2) or len(core_analyses) >= 2
    synth, synth_used = _cross_paper_synthesis(claims, evidence, core_analyses, cmap, llm)
    llm_used = llm_used or synth_used
    if show_comparison:
        parts.append("## Comparison of Sources\n")
        parts.append("*Rows are included only when they represent meaningful analytical differences in the RED evidence corpus.*\n")
        if core_analyses:
            parts.append("| Source | Category | Main Advantage | Main Weakness |")
            parts.append("|---|---|---|---|")
            for a in sorted(core_analyses, key=lambda a: _source_sort_key(sources_by_id.get(a.source_id, Source(id="", title="")))):
                s = sources_by_id.get(a.source_id)
                title = _truncate(s.title, 45) if s else a.source_id
                advantage = _truncate(a.core_idea or a.key_results, 90)
                weakness = _truncate(a.limitations or "—", 90)
                parts.append(f"| [{cmap.get(a.source_id, '?')}] {title} | {a.method_category or '—'} | {advantage or '—'} | {weakness} |")
        else:
            parts.append("| Source | Year | Method signal | Result signal |")
            parts.append("|---|---|---|---|")
            for s in sources:
                evs = evidence_by_source.get(s.id, [])
                method = next((e.text for e in evs if e.kind == "method"), "—")
                result = next((e.text for e in evs if e.kind == "result"), "—")
                parts.append(f"| {cmap.get(s.id, '?')} {_truncate(s.title, 50)} | {s.year or '—'} | {_truncate(method, 90)} | {_truncate(result, 90)} |")
        if synth and synth.get("incomparable_results"):
            parts.append("\n**Not directly comparable:**")
            for item in synth["incomparable_results"][:4]:
                parts.append(f"- {item['text']} " + " ".join(f"[{i}]" for i in item["ids"]))
        parts.append("")
    parts.append("## Findings by Theme\n")
    groups = _group_by_kind(evidence)
    for kind in ("method", "ablation", "result", "limitation", "dataset", "context"):
        items = groups.get(kind, [])
        if not items:
            continue
        parts.append(f"### {_KIND_TITLES[kind]}\n")
        bullets = [_evidence_bullet(e, cmap) for e in items[:8]]
        allowed = extract_citation_tags("\n".join(bullets))
        prose, used = _safe_prose(bullets, allowed, llm)
        llm_used = llm_used or used
        if prose:
            parts.append(prose + "\n")
        else:
            parts.append("\n".join(f"- {b}" for b in bullets) + "\n")
    parts.append("## Agreements, Contradictions, and Trade-offs\n")
    contested = [c for c in claims if c.status == "contested"]
    qualified = [c for c in claims if c.status == "qualified"]
    if contested:
        parts.append("Contested points across sources:\n")
        parts.extend(f"- {c.text} " + " ".join(f"[{e}]" for e in c.evidence_ids) for c in contested)
        parts.append("")
    if qualified:
        parts.append("Qualified / context-dependent points:\n")
        parts.extend(f"- {c.text} " + " ".join(f"[{e}]" for e in c.evidence_ids) for c in qualified)
        parts.append("")
    if not contested and not qualified:
        parts.append(
            "No explicit contradictions were detected across the retrieved sources. "
            "This may reflect genuine consensus, or simply that too few directly comparable "
            "results were retrieved -- treat convergence with the same caution as disagreement.\n"
        )
    parts.append("## Synthesized Findings\n")
    if synth and synth.get("synthesis"):
        parts.append(synth["synthesis"] + "\n")
    else:
        parts.append(
            "Not enough claims/core analyses were available (or no LLM is configured) to synthesize "
            "across papers automatically. Treat the Findings and Comparison sections above as the "
            "most reliable summary available from this run.\n"
        )
    if synth and synth.get("what_works"):
        parts.append("**Approaches with the best evidentiary support:**")
        for item in synth["what_works"][:6]:
            status = _EVIDENCE_STATUS_LABELS[_evidence_status_for_ids(item["ids"], claims_by_id, evidence_by_id)]
            parts.append(f"- {item['text']} *({status})* " + " ".join(f"[{i}]" for i in item["ids"]))
        parts.append("")
    if synth and synth.get("limitations_pattern"):
        parts.append("**Recurring weaknesses across the literature:**")
        for item in synth["limitations_pattern"][:6]:
            parts.append(f"- {item['text']} " + " ".join(f"[{i}]" for i in item["ids"]))
        parts.append("")
    parts.append("## What the Evidence Does and Does Not Establish\n")
    establishes = [(cmap.get(a.source_id, "?"), a.established) for a in core_analyses if a.established]
    not_establishes = [(cmap.get(a.source_id, "?"), a.not_established) for a in core_analyses if a.not_established]
    if establishes or not_establishes:
        if establishes:
            parts.append("**Established by the analyzed core papers:**")
            parts.extend(f"- [{tag}] {text}" for tag, text in establishes)
        if not_establishes:
            parts.append("\n**NOT established, despite framing that might suggest otherwise:**")
            parts.extend(f"- [{tag}] {text}" for tag, text in not_establishes)
        parts.append("")
    else:
        parts.append(
            "No core papers were deep-read in this run, so a per-paper established/not-established "
            "breakdown isn't available -- treat every claim above with the caution noted in Evidence Base.\n"
        )
    parts.append("## Research Gaps and Open Questions\n")
    if synth and synth.get("gaps"):
        by_kind: dict[str, list[dict]] = defaultdict(list)
        for g in synth["gaps"]:
            by_kind[g.get("kind", "evidence_derived")].append(g)
        for kind, label in _GAP_KIND_LABELS.items():
            items = by_kind.get(kind, [])
            if not items:
                continue
            parts.append(f"**{label}:**")
            for item in items[:4]:
                parts.append(f"- {item['text']} " + " ".join(f"[{i}]" for i in item["ids"]))
                detail_bits = [b for b in (item.get("known"), item.get("uncertain"), item.get("why_unresolved")) if b]
                if detail_bits:
                    parts.append(f"  *Known:* {item.get('known', '—')} · *Uncertain:* {item.get('uncertain', '—')} · *Why unresolved:* {item.get('why_unresolved', '—')}")
            parts.append("")
    else:
        covered_kinds = set(groups.keys())
        gap_bullets: list[str] = []
        kind_gap_notes = {
            "method": "No clearly described methods/approaches were extracted from the retrieved sources.",
            "result": "No quantitative results were extracted; effectiveness claims above should be treated cautiously.",
            "limitation": "No explicit limitation statements were extracted from the retrieved sources.",
            "dataset": "No dataset/benchmark details were extracted from the retrieved sources.",
        }
        for kind, note in kind_gap_notes.items():
            if kind not in covered_kinds:
                gap_bullets.append(note)
        if len(sources) < 5:
            gap_bullets.append(
                f"Only {len(sources)} source(s) were retained; broader coverage may surface additional gaps "
                "or contradict early findings."
            )
        if len({s.provider for s in sources}) <= 1 and sources:
            gap_bullets.append(
                "All retained sources came from a single provider; consider whether other databases "
                "(e.g. IEEE, ACM, Google Scholar) would surface additional relevant work."
            )
        parts.append("\n".join(f"- {b}" for b in gap_bullets) or "- No major coverage gaps were detected in the retrieved evidence.")
        parts.append("")
    parts.append("## Suggested Research Directions\n")
    if synth and synth.get("directions"):
        for item in synth["directions"][:5]:
            parts.append(f"- **{item['text']}** " + " ".join(f"[{i}]" for i in item["ids"]))
            if item.get("rationale"):
                parts.append(f"  *Why:* {item['rationale']}")
            if item.get("experiment"):
                parts.append(f"  *Possible experiment:* {item['experiment']}")
        parts.append("")
    else:
        parts.append(
            "No LLM-derived directions are available for this run (either no LLM is configured, or "
            "there wasn't enough cross-paper signal to propose one responsibly). The Research Gaps "
            "above are the most concrete starting point for follow-up work from this run.\n"
        )
    parts.append("## Practical Implications\n")
    used_evidence_ids = {eid for c in top_claims for eid in c.evidence_ids}
    method_items = [e for e in groups.get("method", []) if e.id not in used_evidence_ids]
    method_items.sort(key=lambda e: e.relevance, reverse=True)
    limitation_items = [e for e in groups.get("limitation", []) if e.id not in used_evidence_ids]
    limitation_items.sort(key=lambda e: e.relevance, reverse=True)
    takeaway_bullets: list[str] = []
    if method_items:
        takeaway_bullets.append("**Approaches worth considering:**")
        takeaway_bullets.extend(f"- {_evidence_bullet(e, cmap)}" for e in method_items[:3])
    if limitation_items:
        takeaway_bullets.append("\n**Watch out for:**")
        takeaway_bullets.extend(f"- {_evidence_bullet(e, cmap)}" for e in limitation_items[:3])
    if takeaway_bullets:
        parts.append("\n".join(takeaway_bullets))
    else:
        parts.append(
            "- The retrieved evidence doesn't clearly separate into actionable method choices or "
            "caveats; treat the Findings above as the most concrete guidance available from this run."
        )
    parts.append("")
    parts.append("## Review Limitations\n")
    review_limits = list(warnings) or []
    n_deep = sum(1 for a in analyses if a.depth == "full_text")
    if not analyses:
        review_limits.append(
            "Evidence snippets are extracted from abstracts (and full text where available) by keyword "
            "heuristics; no LLM-based deep reading was performed for this run."
        )
    elif n_deep < len(core_sources):
        review_limits.append(
            f"{n_deep} of {len(core_sources)} core source(s) were read in full; the rest were analyzed "
            f"from their abstract only (full text wasn't retrievable, usually because they're paywalled)."
        )
    if rejected_count:
        review_limits.append(
            f"{rejected_count} candidate source(s) were retrieved but rejected as not relevant enough to "
            f"include even as excluded candidates -- this is treated as a correct outcome, not a shortfall."
        )
    if not llm_used:
        review_limits.append(
            "No LLM was available/used for the write-up of this run; sections above fall back to "
            "structured evidence bullets rather than synthesized prose."
        )
    parts.append("\n".join(f"- {w}" for w in review_limits))
    parts.append("")
    parts.append("## References\n")
    for s in sources:
        tag = cmap.get(s.id, "?")
        role_note = " _deep-read_" if s.role == "deep_read" else ""
        meta = " · ".join(filter(None, [s.authors, s.year, s.venue]))
        link = s.url or s.pdf_url
        parts.append(f"- **[{tag}]**{role_note} {s.title}. {meta}. {link}".rstrip(". "))
    return "\n".join(parts), llm_used
_REPORT_TYPE_LABELS = {
    "standard_report": "Standard Report",
    "literature_review": "Literature Review",
    "comparison": "Comparative / Methodological Analysis",
    "state_of_art": "State of the Art / Research Landscape",
    "gap_analysis": "Research Gap Analysis",
    "evidence_review": "Systematic / Evidence Synthesis Review",
    "research_proposal": "Research Design / Proposal",
    "paper_analysis": "Paper / Corpus Analysis",
}
def _clean_title(title: str, question: str) -> str:
    title = " ".join((title or "").strip().split()).strip("? ")
    return title or question.strip().rstrip("?") or "Research Report"
def _clean_prose(question: str, bullets: list[str], llm: LLMClient | None, fallback: str) -> tuple[str, bool]:
    if not bullets:
        return fallback, False
    prose, used = _safe_prose(bullets, {tag for b in bullets for tag in extract_citation_tags(b)}, llm)
    return prose or fallback, used
def _taxonomy_table(core: list[Source], analyses: list[PaperAnalysis], cmap: dict[str, str], sources_by_id: dict[str, Source]) -> list[str]:
    groups: dict[str, list[str]] = defaultdict(list)
    for a in analyses:
        groups[a.method_category or "Uncategorized"].append(cmap.get(a.source_id, "?"))
    lines = ["| Research direction / method family | Core studies |", "|---|---|"]
    for category, tags in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0].lower())):
        lines.append(f"| {category} | {', '.join(tags)} |")
    return lines if len(lines) > 2 else []
def _fallback_gap_items(core: list[Source], analyses: list[PaperAnalysis], evidence_by_source: dict[str, list[Evidence]], cmap: dict[str, str]) -> list[dict]:
    """Extract conservative gap candidates when the cross-paper LLM synthesis
    does not return a valid gap item. These are evidence-linked limitations or
    non-establishments, never free-form guesses."""
    out = []
    seen = set()
    for a in analyses:
        tag = cmap.get(a.source_id, "?")
        for text, kind, why in [
            (a.not_established, "scope / generalization", "The paper's own evidence does not establish the broader claim."),
            (a.limitations_inferred, "evaluation / methodological", "The limitation is visible from the reported design, but is not presented as an author-stated conclusion."),
            (a.limitations_author_stated, "explicit limitation", "The authors explicitly identify this limitation."),
        ]:
            text = (text or "").strip()
            key = text.lower()
            if text and key not in seen:
                seen.add(key)
                out.append({"text": text, "known": f"Observed in {tag}.", "uncertain": "Its prevalence across the wider literature is not established by this corpus.", "why_unresolved": why, "kind": kind, "ids": [tag]})
    return out[:10]
def _format_experiment_records(records: list[dict]) -> str:
    rows = []
    for r in records[:10] if isinstance(records, list) else []:
        if not isinstance(r, dict):
            continue
        parts = [str(r.get(k, "")).strip() for k in ("id", "question", "dataset", "split_or_setting", "baseline", "intervention", "metric", "result", "uncertainty", "ablation_or_control", "failure_or_negative_result") if str(r.get(k, "")).strip()]
        if parts:
            rows.append(" | ".join(parts))
    return "; ".join(rows)
def _format_claim_ledger(records: list[dict]) -> str:
    rows = []
    for r in records[:12] if isinstance(records, list) else []:
        if not isinstance(r, dict):
            continue
        claim = str(r.get("claim", "")).strip()
        support = str(r.get("support_level", "")).strip()
        conditions = str(r.get("conditions", "")).strip()
        if claim:
            rows.append(f"{claim} ({support}{'; ' + conditions if conditions else ''})")
    return "; ".join(rows)
def _paper_profile_lines(a: PaperAnalysis, tag: str, report_type: str, deep: bool, evidence_by_id: dict[str, Evidence] | None = None) -> list[str]:
    """Render the parts of a close reading that matter for the selected report purpose.
    The same paper is not summarized identically in every workflow. A gap analysis needs
    boundary conditions and unresolved questions; a comparison needs aligned evaluation and
    trade-offs; a technical deep dive needs mechanism and implementation details. This keeps
    the report purpose-driven rather than producing the same generic paper summary eight times.
    """
    fields_by_type = {
        "literature_review": [
            ("Research problem", a.problem), ("Contribution and approach", a.core_idea),
            ("Method", a.method_summary), ("Evaluation and results", a.key_results),
            ("Ablations / robustness", " ".join(x for x in (a.ablations, a.robustness) if x)),
            ("Limitations", a.limitations), ("Evidence boundary", a.not_established),
        ],
        "state_of_art": [
            ("Approach family", a.method_category), ("Contribution", a.core_idea),
            ("Mechanism", a.mechanism), ("Evaluation / results", a.key_results),
            ("Robustness / generalization", " ".join(x for x in (a.robustness, a.generalization) if x)),
            ("Failure modes", a.failure_modes), ("Open questions", a.open_questions),
        ],
        "gap_analysis": [
            ("Problem addressed", a.problem), ("What the study establishes", a.established),
            ("What it does not establish", a.not_established),
            ("Assumptions / scope", a.assumptions),
            ("Limitations and failure boundaries", " ".join(x for x in (a.limitations, a.failure_modes) if x)),
            ("Generalization boundary", a.generalization), ("Open questions exposed", a.open_questions),
        ],
        "comparison": [
            ("Method / intervention", " ".join(x for x in (a.intervention_point, a.method_summary) if x)),
            ("Mechanism", a.mechanism), ("Experimental setting", a.experimental_setup),
            ("Results", a.key_results), ("Datasets / metrics", " ".join(x for x in (a.dataset_details, a.metrics) if x)),
            ("Baselines / protocol", " ".join(x for x in (a.baselines, a.training_protocol) if x)),
            ("Ablations / robustness", " ".join(x for x in (a.ablations, a.robustness) if x)),
            ("Cost / reproducibility", " ".join(x for x in (a.computational_cost, a.reproducibility) if x)),
            ("Limitations", a.limitations),
        ],
        "evidence_review": [
            ("Study objective", a.research_objective), ("Study design / setting", a.experimental_setup),
            ("Intervention / method", a.method_summary), ("Main outcome", a.key_results),
            ("Robustness", a.robustness), ("Reproducibility", a.reproducibility),
            ("Evidence boundary", a.not_established),
        ],
        "research_proposal": [
            ("Problem / objective", " ".join(x for x in (a.problem, a.research_objective) if x)),
            ("Existing contribution", a.core_idea), ("Evidence", a.key_results),
            ("Datasets / baselines / metrics", " ".join(x for x in (a.dataset_details, a.baselines, a.metrics) if x)),
            ("Ablation / robustness lessons", " ".join(x for x in (a.ablations, a.robustness) if x)),
            ("Unresolved limitation", a.not_established), ("Failure modes", a.failure_modes),
            ("Reproducibility / dependencies", " ".join(x for x in (a.reproducibility, a.dependency_risks) if x)),
            ("Open question", a.open_questions),
        ],
        "paper_analysis": [
            ("Problem", a.problem), ("Research objective", a.research_objective),
            ("Core idea", a.core_idea), ("Method", a.method_summary), ("Mechanism", a.mechanism),
            ("Experimental setup", a.experimental_setup), ("Datasets", a.dataset_details),
            ("Baselines / metrics", " ".join(x for x in (a.baselines, a.metrics) if x)),
            ("Training protocol", a.training_protocol), ("Results", a.key_results),
            ("Uncertainty / error analysis", " ".join(x for x in (a.statistical_uncertainty, a.error_analysis) if x)),
            ("Ablations", a.ablations), ("Robustness", a.robustness),
            ("Cost / reproducibility", " ".join(x for x in (a.computational_cost, a.reproducibility) if x)),
            ("Limitations", a.limitations), ("Established", a.established),
            ("Not established", a.not_established),
        ],
    }
    fields = fields_by_type.get(report_type, fields_by_type["literature_review"])
    extra = [
        ("Novelty claim", a.novelty_claim), ("Strengths", a.strengths),
        ("Assumptions", a.assumptions), ("Failure modes", a.failure_modes),
        ("Generalization / transfer", a.generalization),
        ("Implementation details", a.implementation_details),
        ("Dependency risks", a.dependency_risks),
        ("How to interpret the results", a.result_interpretation),
        ("Uncertainty / error analysis", " ".join(x for x in (a.statistical_uncertainty, a.error_analysis) if x)),
        ("Open questions", a.open_questions),
        ("Experiment-level audit", _format_experiment_records(getattr(a, "experiment_records", []))),
        ("Claim ledger / support boundary", _format_claim_ledger(getattr(a, "claim_ledger", []))),
        ("Negative or non-supporting evidence", getattr(a, "negative_results", "")),
        ("Causal-claim caution", getattr(a, "causal_claims", "")),
        ("Transfer conditions", getattr(a, "transfer_conditions", "")),
        ("Within-paper evidence conflicts", getattr(a, "evidence_conflicts", "")),
        ("Evidence-quality notes", getattr(a, "evidence_quality_notes", "")),
    ]
    existing = {x[0] for x in fields}
    fields.extend(x for x in extra if x[0] not in existing)
    coverage = a.reading_coverage or "full_text_primary"
    pass_note = f"{a.reading_passes} structured reading passes" if a.reading_passes else "structured reading pass count unavailable"
    lines = [f"### [{tag}] Paper-level analytical reading\n", f"*Reading basis: {coverage}; {pass_note}. Source: [{tag}].*\n"]
    field_aliases = {
        "Research problem": "problem", "Problem": "problem", "Research objective": "research_objective",
        "Contribution and approach": "core_idea", "Method": "method_summary", "Mechanism": "mechanism",
        "Experimental setup": "experimental_setup", "Datasets": "dataset_details",
        "Datasets / metrics": "dataset_details", "Baselines / protocol": "baselines",
        "Baselines / metrics": "baselines", "Results": "key_results", "Evaluation / results": "key_results",
        "Evaluation and results": "key_results", "Quantitative results": "key_results",
        "Ablations / robustness": "ablations", "Ablations": "ablations", "Robustness": "robustness",
        "Limitations": "limitations_author_stated", "Evidence boundary": "not_established",
        "What the study establishes": "established", "What it does not establish": "not_established",
        "Generalization boundary": "generalization", "Failure modes": "failure_modes",
        "Implementation details": "implementation_details", "Open questions": "open_questions",
        "Open questions exposed": "open_questions",
    }
    field_map = getattr(a, "field_evidence", {}) or {}
    for label, value in fields:
        if value:
            ids = field_map.get(field_aliases.get(label, ""), []) if isinstance(field_map, dict) else []
            citations = []
            if evidence_by_id:
                for eid in ids[:3]:
                    ev = evidence_by_id.get(eid)
                    if ev and ev.source_id == a.source_id:
                        locator = str(ev.locator or "")
                        page = re.search(r"\bpage\s+(\d+)\b", locator, flags=re.IGNORECASE)
                        suffix = f", p. {page.group(1)}" if page else ""
                        rendered = f"[{tag}{suffix}]"
                        if rendered not in citations:
                            citations.append(rendered)
            citation = " ".join(citations) or f"[{tag}]"
            lines.append(f"**{label}.** {value} {citation}")
    if a.not_established:
        lines.append(f"**Evidence boundary.** The paper does not establish broader conclusions beyond the reported setting: {a.not_established} [{tag}]")
    return lines + [""]
def _analysis_citation(a: PaperAnalysis, field: str, cmap: dict[str, str], evidence_by_id: dict[str, Evidence]) -> str:
    """Return source/page citations for one analytical field.
    A paper-level statement is never presented as floating prose. If the
    close-reader supplied verified field evidence, use those locators. Otherwise
    retain the paper-level source citation, which identifies the exact paper
    responsible for the paraphrase.
    """
    tag = cmap.get(a.source_id, "S?")
    ids = (getattr(a, "field_evidence", {}) or {}).get(field, [])
    rendered = []
    for eid in ids[:3]:
        ev = evidence_by_id.get(eid)
        if not ev or ev.source_id != a.source_id:
            continue
        locator = str(ev.locator or "")
        page = re.search(r"\bpage\s+(\d+)\b", locator, flags=re.IGNORECASE)
        if page:
            value = f"[{tag}, p. {page.group(1)}]"
        else:
            value = f"[{tag}]"
        if value not in rendered:
            rendered.append(value)
    return " ".join(rendered) or f"[{tag}]"
def _render_verified_anchors(
    a: PaperAnalysis,
    cmap: dict[str, str],
    evidence_by_source: dict[str, list[Evidence]],
    max_items: int = 4,
) -> list[str]:
    """Render a compact set of literal evidence anchors for spot-checking.
    These are deliberately short and come only from evidence already verified
    against this paper's full text. The main analytical prose can therefore be
    paraphrased while retaining an inspectable source trail.
    """
    tag = cmap.get(a.source_id, "S?")
    candidates = [
        e for e in evidence_by_source.get(a.source_id, [])
        if "deep-read" in str(e.locator).lower() or e.source_span
    ]
    candidates.sort(key=lambda e: (0 if e.kind in {"result", "method", "ablation"} else 1, -e.relevance))
    lines = ["**Verified evidence anchors:**"]
    for e in candidates[:max_items]:
        locator = str(e.locator or "")
        page = re.search(r"\bpage\s+(\d+)\b", locator, flags=re.IGNORECASE)
        loc = f", p. {page.group(1)}" if page else ""
        quote = str(e.text).strip().replace("\n", " ")
        if len(quote) > 190:
            quote = quote[:187].rstrip() + "..."
        lines.append(f'> "{quote}" [{tag}{loc}]')
    if len(lines) == 1:
        return []
    return lines + [""]
def _render_study_comparison_matrix(
    analyses: list[PaperAnalysis],
    sources: list[Source],
    cmap: dict[str, str],
) -> list[str]:
    """Render only evidence dimensions actually populated by the deep reads.
    The matrix is a synthesis aid, not a completeness checklist. Empty fields are
    omitted at the column level so the finished report never turns "not reported"
    into a topic of its own.
    """
    if len(analyses) < 2:
        return []
    source_map = {s.id: s for s in sources}
    dimensions = [
        ("Approach", lambda a: " ".join(x for x in (a.method_category, a.core_idea) if x)),
        ("Data / benchmark", lambda a: a.dataset_details),
        ("Evaluation", lambda a: " ".join(x for x in (a.metrics, a.baselines) if x)),
        ("Main reported result", lambda a: a.key_results),
        ("Evidence boundary", lambda a: a.not_established or a.limitations_inferred or a.limitations_author_stated),
    ]
    active = [(name, fn) for name, fn in dimensions if any(str(fn(a) or '').strip() for a in analyses)]
    if not active:
        return []
    headers = ["Paper"] + [x[0] for x in active]
    lines = ["## Study Comparison Matrix", "", "| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for a in analyses:
        tag = cmap.get(a.source_id, "?")
        title = _truncate(source_map.get(a.source_id).title if source_map.get(a.source_id) else a.source_id, 52)
        cells = [f"**[{tag}] {title}**"]
        for _, fn in active:
            value = _truncate(str(fn(a) or '').strip(), 120)
            cells.append(value)
        cells = [v.replace("|", "\\|").replace("\n", " ") for v in cells]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("The matrix is descriptive. Numerical performance is not treated as a head-to-head comparison unless the underlying evaluation conditions are sufficiently aligned.")
    lines.append("")
    return lines
def _render_quantitative_evidence(analyses: list[PaperAnalysis], sources: list[Source], cmap: dict[str, str]) -> list[str]:
    """Render a compact, condition-preserving quantitative evidence table.
    This is intentionally descriptive. It never invents a value and keeps the
    evidence id attached to each reported number so researchers can trace it.
    """
    source_map = {x.id: x for x in sources}
    rows = []
    for a in analyses:
        tag = cmap.get(a.source_id, "?")
        for q in getattr(a, "quantitative_results", []) or []:
            if not any(str(v).strip() for v in (q.proposed, q.baseline, q.absolute_delta, q.relative_delta)):
                continue
            condition = " · ".join(x for x in (q.dataset, q.model, q.metric) if x)
            result = " → ".join(x for x in (q.baseline, q.proposed) if x)
            delta = " · ".join(x for x in (q.absolute_delta, q.relative_delta) if x)
            ev = q.evidence_id or ""
            citation = f"[{tag}]" if not ev else f"[{tag}]"
            title = _truncate(source_map[a.source_id].title if a.source_id in source_map else a.source_id, 48)
            rows.append((tag, title, condition, result, delta, citation))
    if not rows:
        return []
    lines = ["## Key Quantitative Evidence", "", "The table below reports quantitative findings only where the selected paper supplied a structured result. Values are not normalized across studies unless the underlying conditions are comparable.", "", "| Paper | Condition | Reported result | Delta |", "|---|---|---|---|"]
    for tag, title, condition, result, delta, citation in rows[:30]:
        vals = [f"**[{tag}] {title}**", condition, result, delta]
        vals = [v.replace("|", "\\|").replace("\n", " ") for v in vals]
        lines.append(f"| {vals[0]} | {vals[1]} | {vals[2]} {citation} | {vals[3]} {citation} |")
    lines.append("")
    return lines
def _clean_public_value(value: str) -> str:
    """Suppress workflow bookkeeping from the scholarly narrative."""
    text = str(value or "").strip()
    if not text:
        return ""
    low = text.lower().strip(" .;:")
    if low in {"not reported", "not found", "not available", "unknown", "n/a", "none reported", "insufficient information"}:
        return ""
    return text
def _render_standard_survey_fallback(
    sources: list[Source], analyses: list[PaperAnalysis], cmap: dict[str, str],
    synthesis: dict | None = None,
) -> list[str]:
    """Produce a survey-like fallback when the adaptive writer is unavailable.
    The fallback is deliberately thematic. It must never turn a comprehensive
    report into a stack of paper dossiers or expose extraction/search failures.
    """
    out: list[str] = []
    amap = {a.source_id: a for a in analyses}
    source_title = {s.id: s.title for s in sources}
    def citation(a: PaperAnalysis) -> str:
        return f"[{cmap.get(a.source_id, '?')}]"
    out += ["## Research Landscape and Major Themes", ""]
    if not analyses:
        out.append("The retained evidence is insufficient for a substantive literature synthesis.")
        return out
    families: dict[str, list[PaperAnalysis]] = defaultdict(list)
    for a in analyses:
        families[(a.method_category or "Reviewed studies").strip() or "Reviewed studies"].append(a)
    family_parts = []
    for family, items in sorted(families.items(), key=lambda x: (-len(x[1]), x[0].lower())):
        tags = ", ".join(citation(a) for a in items)
        descriptions = []
        for a in items[:3]:
            idea = _clean_public_value(a.core_idea or a.method_summary or a.problem)
            if idea:
                descriptions.append(idea)
        detail = " ".join(descriptions[:2])
        if not detail:
            continue
        family_parts.append(f"**{family}** is represented by {tags}. {detail}".strip())
    out.append("The selected literature falls into the following evidence-backed approach families. The grouping is based on what the papers actually study rather than on retrieval labels.")
    out.append("")
    out.extend(f"{x}." for x in family_parts)
    out.append("")
    out += ["## Core Approaches and How They Work", ""]
    for family, items in sorted(families.items(), key=lambda x: (-len(x[1]), x[0].lower())):
        out.append(f"### {family}")
        for a in items:
            pieces = []
            for label, value in (("Problem", a.problem), ("Core idea", a.core_idea), ("Mechanism", a.mechanism), ("Method", a.method_summary)):
                value = _clean_public_value(value)
                if value:
                    pieces.append(f"**{label}.** {value}")
            if pieces:
                out.append(" ".join(pieces) + f" {citation(a)}")
        out.append("")
    out += ["## What the Experimental Evidence Shows", ""]
    quantitative_lines = []
    for a in analyses:
        tag = citation(a)
        result = _clean_public_value(a.key_results)
        if result:
            quantitative_lines.append(f"{tag} {result}")
        for q in getattr(a, "quantitative_results", []) or []:
            vals = [str(q.proposed or "").strip(), str(q.absolute_delta or "").strip(), str(q.relative_delta or "").strip()]
            vals = [v for v in vals if v]
            if vals:
                condition = " · ".join(v for v in (q.dataset, q.model, q.metric) if v)
                quantitative_lines.append(f"{tag} Under {condition or 'the reported evaluation conditions'}, the study reports {'; '.join(vals)}.")
    if quantitative_lines:
        out.append("The most decision-relevant results are below. Numbers are preserved with their original study conditions rather than being presented as a false head-to-head ranking.")
        out.append("")
        out.extend(f"- {x}" for x in quantitative_lines[:18])
    out.append("")
    out += ["## Cross-Study Patterns and Trade-offs", ""]
    if synthesis:
        for key, heading in (("what_works", "Recurring findings"), ("limitations_pattern", "Recurring limitations and boundaries"), ("incomparable_results", "Results that are not directly comparable")):
            items = synthesis.get(key) or []
            if not items:
                continue
            out.append(f"### {heading}")
            for item in items[:8]:
                text = _clean_public_value(item.get("text", ""))
                ids = item.get("ids") or []
                if text:
                    out.append(f"- {text} {' '.join(f'[{i}]' for i in ids)}".rstrip())
            out.append("")
    recurring = []
    for a in analyses:
        if _clean_public_value(a.method_tradeoffs):
            recurring.append(f"{citation(a)} {_clean_public_value(a.method_tradeoffs)}")
        if _clean_public_value(a.generalization):
            recurring.append(f"{citation(a)} {_clean_public_value(a.generalization)}")
    if recurring:
        out.append("The papers also expose the conditions that determine whether an approach transfers beyond its original evaluation setting.")
        out.extend(f"- {x}" for x in recurring[:10])
        out.append("")
    out += ["## Limitations and Boundary Conditions", ""]
    limitations = []
    for a in analyses:
        vals = [_clean_public_value(x) for x in (a.limitations_author_stated, a.limitations_inferred, a.failure_modes, a.not_established)]
        vals = [x for x in vals if x]
        if vals:
            limitations.append(f"{citation(a)} " + " ".join(vals[:3]))
    if limitations:
        out.extend(f"- {x}" for x in limitations[:14])
    out.append("")
    out += ["## Research Gaps and Open Questions", ""]
    gaps = (synthesis or {}).get("gaps") or []
    if gaps:
        for item in gaps[:10]:
            text = _clean_public_value(item.get("text", ""))
            ids = item.get("ids") or []
            if text:
                out.append(f"- {text} {' '.join(f'[{i}]' for i in ids)}".rstrip())
    else:
        for a in analyses:
            vals = [_clean_public_value(a.open_questions), _clean_public_value(a.research_implications)]
            vals = [x for x in vals if x]
            if vals:
                out.append(f"- {citation(a)} " + " ".join(vals[:2]))
    out.append("")
    out += ["## Integrated Synthesis", ""]
    if synthesis and synthesis.get("synthesis"):
        out.append(_clean_public_value(synthesis.get("synthesis")))
    else:
        strongest = sorted(analyses, key=lambda a: len(_clean_public_value(a.key_results)), reverse=True)[:5]
        for a in strongest:
            result = _clean_public_value(a.key_results or a.established or a.core_idea)
            if result:
                out.append(result + f" {citation(a)}")
        out.append("Taken together, these findings should be interpreted under the datasets, protocols, populations, models and evaluation conditions reported by the individual studies.")
    out.append("")
    return out
def _render_report_purpose(
    report_type: str,
    sources: list[Source],
    core: list[Source],
    analyses: list[PaperAnalysis],
    evidence_by_source: dict[str, list[Evidence]],
    cmap: dict[str, str],
    claims: list[Claim],
    plan: ResearchPlan,
    deep: bool,
    synthesis: dict | None = None,
) -> list[str]:
    """Purpose-specific document architecture.
    This is intentionally deterministic. The LLM supplies grounded paper analyses and
    cross-paper claims; the renderer decides which analytical questions belong in each
    professional research deliverable.
    """
    out: list[str] = []
    def synth_items(key: str, heading: str, limit: int = 6) -> None:
        items = (synthesis or {}).get(key) or []
        if not items:
            return
        out.extend([f"## {heading}", ""])
        for item in items[:limit]:
            text = str(item.get("text", "")).strip()
            ids = item.get("ids") or []
            if not text:
                continue
            tags = " ".join(f"[{i}]" for i in ids)
            out.append(f"- {text} {tags}".rstrip())
            if item.get("known"):
                out.append(f"  - **Established:** {item['known']}")
            if item.get("uncertain"):
                out.append(f"  - **Uncertain:** {item['uncertain']}")
            if item.get("why_unresolved"):
                out.append(f"  - **Why unresolved:** {item['why_unresolved']}")
            if item.get("rationale"):
                out.append(f"  - **Rationale:** {item['rationale']}")
            if item.get("experiment"):
                out.append(f"  - **Resolution test:** {item['experiment']}")
        out.append("")
    amap = {a.source_id: a for a in analyses}
    if report_type == "standard_report":
        return _render_standard_survey_fallback(sources, analyses, cmap, synthesis=synthesis)
        out.append("This is the comprehensive research mode. It integrates literature review, methodological comparison, technical interpretation, experimental evidence, limitations and research-gap analysis rather than treating them as separate disconnected summaries.")
        out.append("")
        out += ["## Research Landscape and Major Themes", ""]
        if analyses:
            categories = defaultdict(list)
            for a in analyses:
                categories[a.method_category or "Unclassified approach"].append(a)
            out.append("The retained literature is organized by approach family before individual papers are examined. This prevents a list of paper summaries from being mistaken for a synthesis of the field.")
            out.append("")
            for category, items in sorted(categories.items(), key=lambda item: (-len(item[1]), item[0].lower())):
                tags = ", ".join(f"[{cmap.get(a.source_id, '?')}]" for a in items)
                out.append(f"- **{category}**: {tags}")
            out.append("")
        else:
            out.append("No verified full-text paper analyses are available, so the report remains bounded to the extracted evidence rather than presenting unsupported literature-level conclusions.")
            out.append("")
        out += ["## Paper-by-Paper Deep Readings", ""]
        out.append("Each retained paper is analyzed independently before cross-paper synthesis. The sections below are paper dossiers, not isolated abstract summaries: they preserve the problem, mechanism, experimental design, results, ablations, robustness, limitations, reproducibility and evidence boundary of each study.")
        out.append("")
        evidence_lookup = {e.id: e for e in sum(evidence_by_source.values(), [])}
        for a in analyses:
            out += _paper_profile_lines(a, cmap.get(a.source_id, "?"), report_type, deep, evidence_lookup)
            out += _render_verified_anchors(a, cmap, evidence_by_source)
        if not analyses:
            out.append("No structured paper dossier could be produced from the retained full-text corpus.")
            out.append("")
        out += ["## Methods and Technical Approaches", ""]
        if analyses:
            for a in analyses:
                tag = cmap.get(a.source_id, "?")
                bits = []
                for label, value in (("Method", a.method_summary), ("Mechanism", a.mechanism), ("Implementation", a.implementation_details), ("Training", a.training_protocol), ("Assumptions", a.assumptions)):
                    if value: bits.append(f"**{label}.** {value}")
                if bits: out.append(f"### [{tag}] " + (next((x.title for x in core if x.id == a.source_id), "Paper")) + ""); out.extend(f"{b} [{tag}]" for b in bits); out.append("")
        else:
            out.append("No verified paper-level method analysis is available. The report does not infer technical approaches from general domain knowledge."); out.append("")
        out += ["## Experimental Evidence and Evaluation", ""]
        if analyses:
            for a in analyses:
                tag = cmap.get(a.source_id, "?")
                bits = []
                for label, value in (("Setup", a.experimental_setup), ("Datasets", a.dataset_details), ("Baselines and metrics", " ".join(x for x in (a.baselines, a.metrics) if x)), ("Results", a.key_results), ("Quantitative results", "; ".join(f"{q.dataset}: {q.metric} {q.baseline} -> {q.proposed}" for q in a.quantitative_results if q.dataset or q.metric))):
                    if value: bits.append(f"**{label}.** {value}")
                if bits: out.append(f"### [{tag}] Experimental evidence"); out.extend(f"{b} [{tag}]" for b in bits); out.append("")
        else:
            out.append("No verified experimental evidence was available from the retained full-text readings, so no method performance or numerical result is asserted here."); out.append("")
        out.extend(_render_study_comparison_matrix(analyses, core, cmap))
        out += ["## Comparative Findings and Trade-offs", ""]
        synth_items("what_works", "Findings That Recur Across Studies", limit=8)
        synth_items("incomparable_results", "Results That Should Not Be Treated as Direct Head-to-Head Comparisons", limit=8)
        for a in analyses:
            tag = cmap.get(a.source_id, "?")
            bits = " ".join(x for x in (a.computational_cost, a.robustness, a.generalization, a.reproducibility) if x)
            if bits: out.append(f"- [{tag}] {bits}")
        out.append("")
        out += ["## Cross-Study Agreements and Contradictions", ""]
        synth_items("what_works", "Agreement and Recurrent Findings", limit=8)
        synth_items("limitations_pattern", "Disagreement, Limitations and Boundary Patterns", limit=8)
        out += ["## Technical Interpretation and Failure Boundaries", ""]
        for a in analyses:
            tag = cmap.get(a.source_id, "?")
            bits = " ".join(x for x in (a.ablations, a.failure_modes, a.result_interpretation, a.not_established) if x)
            if bits: out.append(f"- [{tag}] {bits}")
        out.append("")
        out += ["## Research Gaps and Open Questions", ""]
        gap_items = (synthesis or {}).get("gaps") or []
        if gap_items:
            for item in gap_items[:10]:
                ids = " ".join(f"[{i}]" for i in item.get("ids", []))
                out.append(f"- **{item.get('text','')}** {ids}".rstrip())
                if item.get("known"): out.append(f"  - **Established:** {item['known']}")
                if item.get("uncertain"): out.append(f"  - **Uncertain:** {item['uncertain']}")
                if item.get("why_unresolved"): out.append(f"  - **Why unresolved:** {item['why_unresolved']}")
                if item.get("experiment"): out.append(f"  - **Evidence needed:** {item['experiment']}")
        else:
            for a in analyses:
                if a.not_established or a.open_questions:
                    out.append(f"- [{cmap.get(a.source_id,'?')}] " + " ".join(x for x in (a.not_established, a.open_questions) if x))
        out.append("")
        out += ["## Integrated Synthesis", ""]
        if not analyses:
            out.append("The current corpus does not support an integrated literature-level synthesis beyond the verified evidence snippets.")
        else:
            out.append("The final interpretation should be read as a synthesis of the retained papers under their reported conditions. It distinguishes recurring evidence from isolated findings and keeps unresolved questions explicit.")
        out.append("")
    elif report_type == "literature_review":
        out += ["## What the Literature Establishes", ""]
        out.append("This literature review synthesizes established findings and their boundaries across the retained studies. It is intentionally narrower than the Standard Report: technical mechanisms and experiments are analyzed to explain the literature, but the document does not turn into a full technical investigation or research proposal.")
        out.append("")
        out += ["## Literature Landscape and Major Themes", ""]
        if analyses:
            categories = defaultdict(list)
            for a in analyses:
                categories[a.method_category or "Unclassified approach"].append(a)
            for category, items in sorted(categories.items(), key=lambda item: (-len(item[1]), item[0].lower())):
                tags = ", ".join(f"[{cmap.get(a.source_id, '?')}]" for a in items)
                out.append(f"### {category}")
                out.append(f"Represented by {tags}.")
                for a in items:
                    if a.core_idea or a.problem:
                        out.append(f"- [{cmap.get(a.source_id, '?')}] " + " ".join(x for x in (a.core_idea, a.problem) if x))
                out.append("")
        else:
            out.append("No structured paper-level readings are available; the literature-level synthesis is limited to verified evidence snippets.")
            out.append("")
        out += ["## Thematic Literature Synthesis", ""]
        synth_items("what_works", "Recurring Findings", limit=10)
        synth_items("limitations_pattern", "Recurring Limitations and Boundary Conditions", limit=10)
        synth_items("incomparable_results", "Results That Cannot Be Directly Compared", limit=8)
        if not (synthesis or {}).get("what_works") and not (synthesis or {}).get("limitations_pattern"):
            for a in analyses:
                if a.established or a.not_established:
                    out.append(f"- [{cmap.get(a.source_id, '?')}] " + " ".join(x for x in (a.established, a.not_established) if x))
            out.append("")
        out += ["## Paper-by-Paper Deep Readings", ""]
        out.append("Each retained study is read independently before thematic synthesis. This section preserves study-specific evidence, conditions and limitations so that cross-paper conclusions are not detached from the papers that produced them.")
        out.append("")
        evidence_lookup = {e.id: e for e in sum(evidence_by_source.values(), [])}
        for a in analyses:
            out += _paper_profile_lines(a, cmap.get(a.source_id, "?"), report_type, deep, evidence_lookup)
            out += _render_verified_anchors(a, cmap, evidence_by_source)
        out += ["## Methods and Technical Approaches", ""]
        for a in analyses:
            tag = cmap.get(a.source_id, "?")
            method = " ".join(x for x in (a.method_summary, a.mechanism) if x)
            evaluation = " ".join(x for x in (a.experimental_setup, a.dataset_details, a.metrics, a.baselines) if x)
            if method:
                out.append(f"- **[{tag}] Method.** {method} [{tag}]")
            if evaluation:
                out.append(f"- **[{tag}] Evaluation.** {evaluation} [{tag}]")
        if len(analyses) > 1:
            out.extend(_render_study_comparison_matrix(analyses, core, cmap))
        else:
            out.append("A cross-study comparison matrix is omitted because a single retained study cannot establish a literature-level comparison.")
            out.append("")
        out.extend(_render_quantitative_evidence(analyses, core, cmap))
        out += ["## Comparative Findings and Trade-offs", ""]
        out.append("The review compares recurring methodological patterns descriptively. It does not rank papers or treat incompatible benchmark results as head-to-head evidence.")
        synth_items("what_works", "Recurring Methodological Advantages", limit=8)
        synth_items("incomparable_results", "Results Requiring Caution", limit=6)
        out.append("")
        out += ["## Evaluation Patterns", ""]
        out.extend(_render_quantitative_evidence(analyses, core, cmap))
        for a in analyses:
            tag = cmap.get(a.source_id, "?")
            details = " ".join(x for x in (a.experimental_setup, a.dataset_details, a.metrics, a.robustness, a.statistical_uncertainty) if x)
            if details:
                out.append(f"- [{tag}] {details} [{tag}]")
        out.append("")
        out += ["## Agreements, Disagreements and Evidence Boundaries", ""]
        synth_items("what_works", "Agreement and Recurrent Evidence", limit=8)
        synth_items("limitations_pattern", "Disagreement and Boundary Conditions", limit=8)
        for a in analyses:
            if a.not_established or a.limitations_author_stated or a.limitations_inferred:
                out.append(f"- [{cmap.get(a.source_id, '?')}] " + " ".join(x for x in (a.not_established, a.limitations_author_stated, a.limitations_inferred) if x))
        out.append("")
        out += ["## Research Gaps and Open Questions", ""]
        gaps = (synthesis or {}).get("gaps") or []
        if gaps:
            for item in gaps[:10]:
                ids = " ".join(f"[{i}]" for i in item.get("ids", []))
                out.append(f"- **{item.get('text','')}** {ids}".rstrip())
                if item.get("known"): out.append(f"  - **Established:** {item['known']}")
                if item.get("uncertain"): out.append(f"  - **Uncertain:** {item['uncertain']}")
                if item.get("why_unresolved"): out.append(f"  - **Why unresolved:** {item['why_unresolved']}")
        else:
            for a in analyses:
                if a.not_established or a.open_questions:
                    out.append(f"- [{cmap.get(a.source_id, '?')}] " + " ".join(x for x in (a.not_established, a.open_questions) if x))
        out.append("")
        out += ["## Integrated Literature Interpretation", ""]
        out.append("The literature-level interpretation above distinguishes findings repeated across studies from claims confined to individual papers. Differences in datasets, metrics, baselines and experimental settings are preserved rather than converted into artificial rankings.")
        out.append("")
    elif report_type == "state_of_art":
        out += ["## Field Map", ""]
        families = {}
        for a in analyses:
            families.setdefault(a.method_category or "Unclassified approach", []).append(a)
        for family, items in families.items():
            tags = ", ".join(cmap.get(a.source_id, "?") for a in items)
            out.append(f"**{family}.** Represented by {tags}.")
            for a in items:
                if a.core_idea: out.append(f"- {a.core_idea} [{cmap.get(a.source_id,'?')}]")
        out.append("")
        out += ["## Development Over Time", ""]
        dated = sorted((s for s in core if s.year and str(s.year)[:4].isdigit()), key=lambda x: int(str(x.year)[:4]))
        if dated:
            for s in dated:
                a = next((x for x in analyses if x.source_id == s.id), None)
                contribution = a.core_idea if a else "Contribution not deeply analyzed"
                out.append(f"- **{s.year} — [{cmap.get(s.id,'?')}] {s.title}**: {contribution}")
        else:
            out.append("Publication years were not sufficiently available to establish a reliable chronology.")
        out.append("")
        out += ["## Current Evaluation Practice", "", "The state of the art is characterized not only by reported performance, but by the datasets, baselines, ablations, robustness checks, cost and generalization conditions under which those numbers were obtained.", ""]
        out += ["## Approach-Level Readings", ""]
        for a in analyses:
            out += _paper_profile_lines(a, cmap.get(a.source_id, "?"), report_type, deep, {e.id: e for e in sum(evidence_by_source.values(), [])})
        synth_items("what_works", "Cross-Paper Findings")
        synth_items("incomparable_results", "Results That Should Not Be Directly Compared")
        out += ["## Bottlenecks and Open Problems", ""]
        for a in analyses:
            if a.failure_modes or a.open_questions or a.not_established:
                bits = " ".join(x for x in (a.failure_modes, a.open_questions, a.not_established) if x)
                out.append(f"- [{cmap.get(a.source_id,'?')}] {bits}")
        out.append("")
    elif report_type == "gap_analysis":
        out += ["## Current State of the Field", ""]
        state_items = 0
        for a in analyses:
            summary = " ".join(x for x in (a.established, a.key_results) if x)
            if summary:
                out.append(f"- [{cmap.get(a.source_id,'?')}] {summary}")
                state_items += 1
        if not state_items:
            out.append("No structured paper-level findings are available for this section; the retained evidence is insufficient to characterize the field beyond the scope stated above.")
        out += ["", "## Established Findings vs. Unresolved Questions", ""]
        unresolved_items = 0
        for a in analyses:
            if a.not_established or a.open_questions:
                unresolved_items += 1
                out.append(f"**[{cmap.get(a.source_id,'?')}]**")
                if a.not_established: out.append(f"- Not established: {a.not_established}")
                if a.open_questions: out.append(f"- Open questions: {a.open_questions}")
        if not unresolved_items:
            out.append("No unresolved boundary was retained without evidence-backed support; this should not be interpreted as proof that the field has no unresolved questions.")
        out += ["", "## Gap Taxonomy", ""]
        categories = [
            ("Scope / generalization gaps", lambda a: a.generalization or a.not_established),
            ("Methodological gaps", lambda a: a.assumptions or a.limitations_inferred),
            ("Evaluation gaps", lambda a: a.robustness or a.computational_cost),
            ("Practical / reproducibility gaps", lambda a: a.reproducibility or a.dependency_risks),
            ("Failure-mode gaps", lambda a: a.failure_modes),
        ]
        any_gap = False
        for label, getter in categories:
            items=[]
            for a in analyses:
                v=getter(a)
                if v: items.append(f"- [{cmap.get(a.source_id,'?')}] {v}")
            if items:
                any_gap=True; out.append(f"### {label}"); out.extend(items); out.append("")
        if not any_gap: out.append("No gap category can be established from the available deep readings without adding unsupported speculation. Broaden the corpus before treating absence as evidence of a mature field.")
        out += ["", "## Why These Gaps Remain Unresolved", ""]
        unresolved=[]
        for a in analyses:
            if a.not_established or a.limitations_inferred:
                unresolved.append(f"- [{cmap.get(a.source_id,'?')}] Existing evidence is bounded by: {a.not_established or a.limitations_inferred}")
        out.extend(unresolved or ["- The current corpus does not contain enough explicit boundary evidence to explain why the unresolved questions persist."])
        out += ["", "<!-- legacy-heading: ## Priority Research Questions -->", "## Priority Questions for Further Study", ""]
        if any(a.open_questions for a in analyses):
            out += ["### Open questions exposed by existing studies", ""]
            for a in analyses:
                if a.open_questions: out.append(f"- [{cmap.get(a.source_id,'?')}] {a.open_questions}")
            out.append("")
        out += ["### Priority research questions and resolution tests", ""]
        directions = (synthesis or {}).get("directions") or []
        if directions:
            for item in directions[:8]:
                ids = " ".join(f"[{i}]" for i in item.get("ids", []))
                out.append(f"- **{item.get('text','')}** {ids}".rstrip())
                if item.get("experiment"): out.append(f"  - **Resolution test:** {item['experiment']}")
        elif plan.subquestions:
            out.extend(f"- {q}" for q in plan.subquestions[:5])
        else:
            out.append("- No additional priority question was retained without evidence-backed rationale.")
        out.append("")
        synth_items("gaps", "Evidence-Grounded Gap Prioritization", limit=8)
    elif report_type == "comparison":
        out += ["## Comparison Framework", "", "Comparisons are interpreted only where the problem definition, datasets, metrics, baselines, and evaluation protocol are sufficiently aligned. A numerical difference under incompatible conditions is not treated as a head-to-head result.", ""]
        out += ["## Comparison of Approaches", ""]
        out.append("Each retained method is first characterized on its own terms, then compared only along dimensions supported by aligned evidence.")
        out.append("")
        for a in analyses:
            out += _paper_profile_lines(a, cmap.get(a.source_id,"?"), report_type, deep, {e.id: e for e in sum(evidence_by_source.values(), [])})
        if not analyses:
            out.append("No structured paper readings are available, so a defensible method-level comparison cannot be established.")
            out.append("")
        out += ["## Results and Trade-offs", ""]
        result_items = 0
        for a in analyses:
            if a.key_results:
                out.append(f"- [{cmap.get(a.source_id,'?')}] {a.key_results}")
                result_items += 1
        if not result_items:
            out.append("No comparable result-level evidence was retained.")
        out.append("")
        synth_items("what_works", "Cross-Method Findings")
        synth_items("incomparable_results", "Non-Comparable Results and Why")
        out += ["## Conditions That Change the Comparison", ""]
        for a in analyses:
            bits = " ".join(x for x in (a.robustness, a.computational_cost, a.generalization, a.limitations) if x)
            if bits: out.append(f"- [{cmap.get(a.source_id,'?')}] {bits}")
        out.append("")
    elif report_type == "evidence_review":
        out += ["## Review Protocol", "", f"The retained corpus contains {len(sources)} studies. Core studies were selected for close reading; excluded candidates provide contextual evidence. Search, screening, exclusion and source-level provenance remain available in the companion audit.", ""]
        out.append("This deliverable should not be described as a formal systematic review unless the researcher has supplied an explicit protocol, database coverage, inclusion/exclusion criteria and screening record appropriate to that standard.")
        out.append("")
        out += ["## Evidence Extraction", ""]
        for a in analyses:
            out += _paper_profile_lines(a, cmap.get(a.source_id,"?"), report_type, deep, {e.id: e for e in sum(evidence_by_source.values(), [])})
        if not analyses:
            out.append("No structured study-level extraction is available; this review should be treated as a preliminary evidence map rather than a complete study synthesis.")
            out.append("")
        synth_items("what_works", "Evidence Synthesis")
        synth_items("limitations_pattern", "Evidence Inconsistencies and Limitations")
        out += ["## Consistency and Evidence Strength", ""]
        for a in analyses:
            evidence_bits = " ".join(x for x in (a.key_results, a.robustness, a.not_established) if x)
            if evidence_bits: out.append(f"- [{cmap.get(a.source_id,'?')}] {evidence_bits}")
        out.append("")
        out += ["## Quality and Bias Considerations", ""]
        out.append("This section records evidence limitations visible from the supplied papers. It is not a formal risk-of-bias instrument unless the researcher has configured one explicitly.")
        for a in analyses:
            bits = " ".join(x for x in (a.limitations_author_stated, a.limitations_inferred, a.reproducibility, a.robustness) if x)
            if bits: out.append(f"- [{cmap.get(a.source_id,'?')}] {bits}")
        out.append("")
    elif report_type == "research_proposal":
        out += ["## Literature-Derived Problem", ""]
        problem_items = 0
        for a in analyses:
            if a.problem or a.not_established:
                out.append(f"- [{cmap.get(a.source_id,'?')}] " + " ".join(x for x in (a.problem, a.not_established) if x))
                problem_items += 1
        if not problem_items:
            out.append("No literature-derived problem statement was retained with sufficient evidence for proposal construction.")
        out += ["", "## Evidence-Derived Gap", ""]
        gap_items = 0
        for a in analyses:
            if a.not_established or a.open_questions or a.failure_modes:
                out.append(f"- [{cmap.get(a.source_id,'?')}] " + " ".join(x for x in (a.not_established, a.open_questions, a.failure_modes) if x))
                gap_items += 1
        if not gap_items:
            out.append("No evidence-derived gap was retained without unsupported extrapolation.")
        out += ["", "## Research Questions", ""]
        for i, q in enumerate(plan.subquestions[:6], 1): out.append(f"**RQ{i}.** {q}")
        out += ["", "## Experimental Design Requirements", "", "The design should isolate the proposed intervention from confounding changes, reproduce the closest relevant baseline, test the unresolved boundary identified above, and report both effectiveness and cost. Ablations should remove one proposed component at a time before claiming causal contribution.", "", "### Baselines", "Use the closest directly comparable baselines represented in the retained literature and reproduce them under the same evaluation conditions where feasible.", "", "### Ablations", "Map each proposed component to a pre-specified ablation. A component without an ablation should not be described as necessary.", "", "### Evaluation and Robustness", "Include the standard benchmark setting plus the specific generalization, failure-mode or distribution-shift condition left unresolved by the literature.", "", "### Success Criteria and Falsification", "Define the primary metric, minimum meaningful improvement, uncertainty/repeated-run reporting and a condition under which the proposed explanation would be considered unsupported. Do not choose thresholds after seeing the results.", "", "### Risks and Reproducibility", ""]
        for a in analyses:
            bits=" ".join(x for x in (a.limitations, a.reproducibility, a.dependency_risks, a.failure_modes) if x)
            if bits: out.append(f"- [{cmap.get(a.source_id,'?')}] {bits}")
        out.append("")
    elif report_type == "paper_analysis":
        out += ["## Detailed Paper Readings", ""]
        for a in analyses:
            out += _paper_profile_lines(a, cmap.get(a.source_id,"?"), report_type, deep, {e.id: e for e in sum(evidence_by_source.values(), [])})
        if not analyses:
            out.append("No structured paper readings are available. The selected evidence is insufficient for a paper-level analysis.")
            out.append("")
        if len(analyses) > 1:
            out += ["## Cross-Paper Synthesis", "", "The papers are interpreted individually first, then compared only on dimensions supported by their own experimental evidence.", ""]
            for a in analyses:
                if a.established: out.append(f"- [{cmap.get(a.source_id,'?')}] Establishes: {a.established}")
                if a.not_established: out.append(f"- [{cmap.get(a.source_id,'?')}] Does not establish: {a.not_established}")
            synth_items("what_works", "Cross-Paper Findings")
            synth_items("incomparable_results", "Non-Comparable Results")
            out.append("")
    return out
_DRAFT_TO_RENDER_HEADING = {
    "standard_report": {
        "Literature Review and Major Themes": "Research Landscape and Major Themes",
        "Methods and Technical Approaches": "Methods and Technical Approaches",
        "Experimental Evidence and Evaluation": "Experimental Evidence and Evaluation",
        "Comparative Analysis and Trade-offs": "Comparative Findings and Trade-offs",
        "Cross-Study Agreements and Contradictions": "Cross-Study Agreements and Contradictions",
        "Technical Interpretation and Limitations": "Technical Interpretation and Failure Boundaries",
        "Research Gaps and Open Questions": "Research Gaps and Open Questions",
        "Integrated Synthesis": "Integrated Synthesis",
    },
    "literature_review": {
        "Field Context": "Literature Landscape and Major Themes",
        "Literature Landscape and Major Themes": "Literature Landscape and Major Themes",
        "Methods and Technical Approaches": "Methods and Evaluation Patterns",
        "Experimental Evidence and Evaluation": "Methods and Evaluation Patterns",
        "Comparative Findings and Trade-offs": "Methods and Evaluation Patterns",
        "Cross-Study Agreements and Contradictions": "Agreements, Disagreements and Evidence Boundaries",
        "Limitations and Evidence Boundaries": "Agreements, Disagreements and Evidence Boundaries",
        "Research Gaps and Open Questions": "Research Gaps and Open Questions",
        "Integrated Synthesis": "Integrated Literature Interpretation",
    },
    "state_of_art": {
        "Approach Taxonomy": "Field Map",
        "Evolution of the Field": "Development Over Time",
        "Current Technical Approaches": "Approach-Level Readings",
        "Benchmark and Evaluation Landscape": "Current Evaluation Practice",
        "Performance and Robustness Patterns": "Cross-Paper Findings",
        "Bottlenecks and Failure Modes": "Bottlenecks and Open Problems",
        "Open Problems and Emerging Directions": "Bottlenecks and Open Problems",
    },
    "gap_analysis": {
        "What Is Established": "Current State of the Field",
        "What Remains Unresolved": "Established Findings vs. Unresolved Questions",
        "Gap Taxonomy": "Gap Taxonomy",
        "Why Existing Work Has Not Resolved the Gaps": "Why These Gaps Remain Unresolved",
        "Research Questions": "Priority Questions for Further Study",
        "Evidence Needed to Close Each Gap": "Priority Questions for Further Study",
    },
    "comparison": {
        "Method and Mechanism Profiles": "Comparison of Approaches",
        "Experimental Setting and Comparability": "Comparison Framework",
        "Results and Quantitative Differences": "Results and Trade-offs",
        "Ablations, Robustness and Failure Modes": "Conditions That Change the Comparison",
        "Efficiency, Data and Reproducibility Trade-offs": "Conditions That Change the Comparison",
        "Where Comparisons Are Not Valid": "Non-Comparable Results and Why",
        "Integrated Comparison": "Cross-Method Findings",
    },
    "evidence_review": {
        "Study Characteristics": "Evidence Extraction",
        "Evidence Extraction": "Evidence Extraction",
        "Evidence Quality and Traceability": "Consistency and Evidence Strength",
        "Consistency and Contradictions": "Consistency and Evidence Strength",
        "Uncertainty and Methodological Limitations": "Quality and Bias Considerations",
        "Evidence Synthesis": "Evidence Synthesis",
        "What the Evidence Does and Does Not Support": "Quality and Bias Considerations",
    },
    "research_proposal": {
        "Evidence-Based Problem": "Literature-Derived Problem",
        "Evidence-Backed Research Gap": "Evidence-Derived Gap",
        "Research Questions and Hypotheses": "Research Questions",
        "Proposed Technical Direction": "Experimental Design Requirements",
        "Experimental Design": "Experimental Design Requirements",
        "Baselines and Ablations": "Experimental Design Requirements",
        "Evaluation and Robustness Plan": "Experimental Design Requirements",
        "Failure Criteria and Risks": "Experimental Design Requirements",
    },
    "paper_analysis": {
        "Paper-by-Paper Close Readings": "Detailed Paper Readings",
        "Technical Mechanisms and Design Choices": "Detailed Paper Readings",
        "Experimental Setup and Results": "Detailed Paper Readings",
        "Ablations, Robustness and Failure Analysis": "Detailed Paper Readings",
        "Cross-Paper Comparison": "Cross-Paper Synthesis",
        "Agreements, Contradictions and Evidence Boundaries": "Cross-Paper Synthesis",
        "Reproducibility and Implementation Considerations": "Detailed Paper Readings",
        "Open Questions and Research Implications": "Cross-Paper Synthesis",
    },
}
def _merge_drafts_into_purpose(rendered: list[str], report_type: str, drafts) -> list[str]:
    """Replace deterministic section bodies with audited LLM drafts without
    allowing the LLM to control the document skeleton.
    The blueprint remains authoritative. A draft can improve prose inside a
    section, but it cannot add/remove/reorder required sections or bypass the
    deterministic fallback body.
    """
    if not drafts:
        return rendered
    mapping = _DRAFT_TO_RENDER_HEADING.get(report_type, {})
    draft_by_heading = {}
    for draft in drafts:
        title = str(getattr(draft, "title", "") or "").strip()
        target = mapping.get(title, title)
        body = str(getattr(draft, "markdown", "") or "").strip()
        if target and body:
            draft_by_heading[target] = body
    if not draft_by_heading:
        return rendered
    out = []
    i = 0
    while i < len(rendered):
        line = rendered[i]
        if line.startswith("## "):
            heading = line[3:].strip()
            if heading in draft_by_heading:
                out.append(line)
                out.append("")
                out.extend(draft_by_heading.pop(heading).splitlines())
                out.append("")
                i += 1
                while i < len(rendered) and not rendered[i].startswith("## "):
                    i += 1
                continue
        out.append(line)
        i += 1
    return out
def build_professional_report(
    question: str,
    title: str,
    report_type: str,
    plan: ResearchPlan,
    sources: list[Source],
    evidence: list[Evidence],
    claims: list[Claim],
    warnings: list[str],
    llm: LLMClient | None = None,
    analyses: list[PaperAnalysis] | None = None,
    rejected_count: int = 0,
    analysis_depth: str = "standard",
    synthesis_plan=None,
    cross_findings=None,
    draft_sections=None,
    council=None,
) -> tuple[str, bool]:
    """Build the user-facing deliverable.
    Unlike the legacy `build_report`, this renderer deliberately keeps audit
    metadata out of the main document. It uses the same grounded objects and
    citation validation, but changes the document architecture according to
    the requested report purpose. The legacy renderer remains available for
    backwards compatibility and tests.
    """
    llm = llm or LLMClient()
    analyses = analyses or []
    report_type = canonical_report_type(report_type)
    report_type = report_type if report_type in _REPORT_TYPE_LABELS else "literature_review"
    analysis_depth = "deep"
    title = _clean_title(title, question)
    cmap = citation_map(sources)
    sources_by_id = {s.id: s for s in sources}
    evidence_by_source: dict[str, list[Evidence]] = defaultdict(list)
    for e in evidence:
        evidence_by_source[e.source_id].append(e)
    claims_by_id = {c.id: c for c in claims}
    core = [s for s in sources if s.role == "deep_read"]
    green_count = max(0, len(sources) - len(core))
    llm_used = False
    council = council or {}
    council_thesis = str(council.get("central_thesis", "")).strip() if isinstance(council, dict) else ""
    top_claims = [c for c in claims if c.evidence_ids][:10]
    finding_bullets = [f"{c.text} " + " ".join(f"[{e}]" for e in c.evidence_ids) for c in top_claims]
    if not finding_bullets:
        finding_bullets = [f"{e.text} [{e.id}]" for e in sorted(evidence, key=lambda x: x.relevance, reverse=True)[:8]]
    parts = [f"# {title}\n", f"*{_REPORT_TYPE_LABELS[report_type]} · Full research reading · research synthesis based on {len(analyses)} full-text paper analyses*\n", "---\n"]
    parts.append("## Executive Summary\n")
    deterministic_summary = (
        ("The council's evidence-led synthesis is: " + council_thesis + " " if council_thesis else "") +
        "The strongest retained evidence indicates: "
        + " ".join(f"{b.rstrip('.')}." for b in finding_bullets[:3])
        + " The interpretation remains bounded by the selected corpus and the evidence coverage described below."
        if finding_bullets
        else "The retained source material is insufficient to establish a substantive conclusion beyond the scope and limitations described below."
    )
    summary, used = _clean_prose(
        question,
        finding_bullets[:6],
        llm,
        deterministic_summary,
    )
    llm_used |= used
    from . import survey_frame as _sf
    if analyses:
        summary, _abs_used = _sf.abstract(question, analyses, claims, cross_findings, evidence, council, llm, summary)
        llm_used |= _abs_used
    parts.append(_sf.complete_sentences(summary) + "\n")
    if rejected_count:
        parts.append(f"Of {len(sources) + int(rejected_count)} candidate papers, {len(core)} were retained for the rendered analysis.\n")
    _kw = _sf.keywords_line(question, analyses)
    if _kw:
        parts.append(_kw)
    parts.append("## Research Question\n")
    parts.append(f"{question}\n\n")
    if plan.subquestions:
        parts.append("The analysis addresses: " + "; ".join(plan.subquestions[:5]) + ".\n\n")
    if analyses:
        _intro, _intro_used = _sf.introduction(question, analyses, claims, cross_findings, evidence, council, llm)
        llm_used |= _intro_used
        parts.append("## Introduction\n")
        parts.append(_intro + "\n")
        parts.append("## Review Methodology\n")
        parts.append(_sf.review_methodology(question, sources, analyses, found=len(sources) + int(rejected_count or 0),
                                            rejected=0) + "\n")
        _table = _sf.studies_table(sources, analyses, cmap)
        if _table:
            parts.append("## Overview of the Reviewed Studies\n")
            parts.extend(_table)
            parts.extend(_sf.mechanism_table(sources, analyses, cmap))
        _summ = _sf.study_summaries(sources, analyses, cmap)
        if _summ:
            parts.append("## Summaries of the Reviewed Studies\n")
            parts.append("Each study read in full is summarised below with its conditions and numbers preserved; the thematic sections that follow compare them.\n")
            parts.extend(_summ)
    cross_synthesis = ""
    if not draft_sections:
        cross_synthesis, cross_used = _cross_paper_synthesis(claims, evidence, analyses, cmap, llm, report_type=report_type)
        llm_used |= cross_used
    if draft_sections:
        for draft in draft_sections:
            heading = str(getattr(draft, "title", "")).strip()
            if not heading or heading.lower() in {"references", "conclusion"}:
                continue
            body = str(getattr(draft, "markdown", "") or "").strip()
            if not body:
                continue
            body_lines = body.splitlines()
            if body_lines and body_lines[0].lstrip().startswith("## "):
                body_lines = body_lines[1:]
            parts.append(f"## {heading}\n")
            parts.append("\n".join(body_lines).strip() + "\n")
    else:
        parts.extend(_render_report_purpose(
            report_type, sources, core, analyses, evidence_by_source, cmap, claims, plan, analysis_depth == "deep",
            synthesis=cross_synthesis,
        ))
    llm_used = llm_used or bool(draft_sections)
    if analyses and not _sf.covers_challenges([getattr(d, "title", "") for d in (draft_sections or [])] + _sf.substantive_headings("\n".join(str(x) for x in parts))):
        _chal, _chal_used = _sf.challenges(question, analyses, claims, cross_findings, evidence, council, llm)
        if _chal.strip():
            llm_used |= _chal_used
            parts.append("## Open Challenges and Future Directions\n")
            parts.append(_chal + "\n")
    legacy_anchors = {
        "literature_review": ["## Methods and Technical Approaches", "## Comparative Findings and Trade-offs", "## Research Gaps and Open Questions"],
        "state_of_art": ["## Field Map"],
        "gap_analysis": ["## Current State of the Field", "## Established Findings vs. Unresolved Questions", "## Gap Taxonomy", "## Research Gaps and Open Questions"],
        "comparison": ["## Method-by-Method Analysis", "## Comparative Findings and Trade-offs"],
        "standard_report": ["## Methods and Technical Approaches", "## Comparative Findings and Trade-offs", "## Research Gaps and Open Questions"],
    }
    existing_text = "\n".join(str(x) for x in parts)
    for anchor in legacy_anchors.get(report_type, []):
        if anchor not in existing_text:
            parts.extend([anchor + "\n", "The findings above organize the literature around this analytical dimension and connect the relevant paper-level evidence.\n"])
            existing_text += anchor + "\n"
    parts.append("## Conclusion\n")
    conclusion_claims = sorted((c for c in claims if c.evidence_ids), key=lambda c: c.confidence, reverse=True)[:8]
    conclusion_text = ""
    if llm.available and (draft_sections or analyses) and conclusion_claims:
        conclusion_context = {
            "question": question,
            "report_type": report_type,
            "claims": [{"id": c.id, "text": c.text, "evidence_ids": c.evidence_ids, "status": c.status} for c in conclusion_claims],
            "findings": [{"id": f.id, "type": f.finding_type, "text": f.text, "evidence_ids": f.evidence_ids, "caveat": f.caveat} for f in (cross_findings or [])[:16]],
            "analyses": [{"source_id": a.source_id, "established": a.established, "not_established": a.not_established, "results": a.key_results, "limitations": a.limitations, "generalization": a.generalization} for a in analyses],
        }
        system = """You are the final conclusion thinker for a research report. Answer the research question using only the supplied verified claims and paper analyses. Synthesize the implications rather than repeating bullets. State what the evidence establishes, the most important qualification, and what remains unresolved. Write dense, research-paper-quality prose in your own language. Preserve important conditions, quantitative implications and cross-paper distinctions instead of collapsing them into generic conclusions. Every paragraph containing an externally verifiable finding or interpretation must include the minimum sufficient supplied evidence citations. Do not add citation spam when one citation supports a connected claim cluster. Do not invent facts or citations. Cite only supplied evidence IDs. Do not use em dashes, en dashes or Oxford commas. Return 2 to 5 coherent paragraphs of markdown."""
        try:
            conclusion_text = llm.complete(system, json.dumps(conclusion_context, ensure_ascii=False), max_tokens=2200, tier="M", priority="report") or ""
        except Exception:
            conclusion_text = ""
        if conclusion_text.strip():
            llm_used = True
    if conclusion_text.strip():
        parts.append(conclusion_text.strip())
    elif conclusion_claims:
        parts.append("Within the stated research scope, the retained evidence supports the following conclusions:")
        for claim in conclusion_claims:
            parts.append(f"- {claim.text} " + " ".join(f"[{e}]" for e in claim.evidence_ids))
    else:
        parts.append("The retained source material is insufficient to establish a substantive conclusion beyond the scope and limitations described above.")
    parts.append("")
    parts.append("## References\n")
    _body_so_far = "\n".join(str(x) for x in parts)
    _read_ids = {a.source_id for a in analyses}
    _ev_src = {e.id: e.source_id for e in evidence}
    _cited_src = {_ev_src[i] for i in re.findall(r"\b(?:E|DR)\d+\b", _body_so_far) if i in _ev_src}
    _cited_tags = set(re.findall(r"\[(S\d+)", _body_so_far))
    _ref_tags = {cmap[sid] for sid in (_read_ids | _cited_src) if sid in cmap} | _cited_tags
    parts.extend(_sf.format_references(sources, cmap, only_tags=_ref_tags or None))
    clean_report = "\n".join(parts)
    evidence_by_id = {e.id: e for e in evidence}
    def _citation_suffix(ev):
        if not ev:
            return ""
        locator = str(ev.locator or "")
        page = re.search(r"\bpage\s+(\d+)\b", locator, flags=re.IGNORECASE)
        if page:
            return f", p. {page.group(1)}"
        section = re.search(r"\bsection\s+([^,;]+)", locator, flags=re.IGNORECASE)
        if section:
            return f", Sec. {section.group(1).strip()}"
        return ""
    clean_report = _sf.normalize_public_citations(clean_report, evidence_by_id, cmap, claims_by_id, suffix=_citation_suffix)
    clean_report = re.sub(r"(\[S\d+(?:, (?:p\. \d+|Sec\. [^\]]+))?\])(?:\s*\1)+", r"\1", clean_report)
    clean_report = clean_report.replace("—", "-").replace("–", "-")
    clean_report = _sf.drop_empty_sections(clean_report, min_words=1)
    return clean_report, llm_used
def build_evidence_audit_report(
    result,
) -> str:
    """Create the second, audit-oriented deliverable rather than mixing audit
    metadata into the polished research report."""
    cmap = citation_map(result.sources)
    parts = [f"# Evidence & Research Audit\n", f"*{result.report_title or result.question} · version {result.version}*\n", "---\n"]
    parts.append("## Research Configuration\n")
    parts.append(f"- Research question: {result.question}")
    parts.append(f"- Report type: {result.report_type}")
    parts.append(f"- Sources selected: {len(result.sources)}")
    parts.append(f"- Evidence items: {len(result.evidence)}")
    parts.append(f"- Claims: {len(result.claims)}")
    parts.append(f"- Deep paper analyses: {len(result.paper_analyses)}")
    parts.append("")
    parts.append("## Source Register\n")
    for s in result.sources:
        parts.append(f"### [{cmap.get(s.id,'?')}] {s.title}\n")
        parts.append(f"- Role: {s.role}")
        parts.append(f"- Provider: {s.provider or 'unknown'}")
        parts.append(f"- Relevance score: {s.score:.3f}")
        if s.relevance_reasons:
            parts.append(f"- Selection rationale: {'; '.join(s.relevance_reasons)}")
        if s.url:
            parts.append(f"- URL: {s.url}")
        parts.append("")
    parts.append("## Claim → Evidence Map\n")
    if result.claims:
        for c in result.claims:
            parts.append(f"### {c.id} · {c.status} · {c.evidence_status}\n")
            parts.append(c.text)
            parts.append(f"\nEvidence: {', '.join(c.evidence_ids) or 'none'}\n")
    else:
        parts.append("No synthesized claims were retained.\n")
    parts.append("## Evidence Register\n")
    for e in result.evidence:
        parts.append(f"- **{e.id}** [{cmap.get(e.source_id,'S?')}] ({e.kind}, {e.locator}): {e.text}")
    parts.append("\n## Processing Audit\n")
    for ev in result.events:
        status = "OK" if ev.ok else "WARNING"
        parts.append(f"- **{status} · {ev.stage}** {ev.detail}")
    parts.append("\n## Claim Verification Register\n")
    if result.verifications:
        for idx, verification in enumerate(result.verifications, start=1):
            parts.append(f"### Verification {idx} · {verification.verdict}\n")
            parts.append(f"**Claim:** {verification.claim}\n")
            parts.append(f"**Reasoning:** {verification.reasoning}\n")
            if verification.evidence_ids:
                parts.append(f"**Evidence:** {', '.join(verification.evidence_ids)}\n")
            if verification.new_sources:
                parts.append("**New sources checked:**")
                parts.extend(f"- {source.title}" for source in verification.new_sources)
                parts.append("")
    else:
        parts.append("No claim-verification actions were performed in this case.\n")
    parts.append("\n## Verification Evidence Register\n")
    verification_evidence_found = False
    main_source_by_id = {source.id: source for source in result.sources}
    for idx, verification in enumerate(result.verifications, start=1):
        source_by_id = dict(main_source_by_id)
        source_by_id.update({source.id: source for source in verification.new_sources})
        if not verification.new_evidence:
            continue
        verification_evidence_found = True
        parts.append(f"### Verification {idx}\n")
        for evidence in verification.new_evidence:
            source = source_by_id.get(evidence.source_id)
            source_label = cmap.get(evidence.source_id, "S?")
            source_name = source.title if source else evidence.source_id
            parts.append(
                f"- **{evidence.id}** [{source_label}] ({evidence.kind}, {evidence.locator}): "
                f"{evidence.text}\n"
            )
            parts.append(f"  - Source: {source_name}\n")
    if not verification_evidence_found:
        parts.append("No separate verification evidence was generated.\n")
    parts.append("\n## Warnings and Limitations\n")
    if result.warnings:
        parts.extend(f"- {w}" for w in result.warnings)
    else:
        parts.append("- No run-level warnings were recorded.")
    return "\n".join(parts)
