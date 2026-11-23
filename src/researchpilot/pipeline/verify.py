from __future__ import annotations
import re
from ..config import get_settings
from ..llm import LLMClient
from ..schemas import ClaimVerification, Evidence, Source
from ..search.aggregator import SearchAggregator
from ..textutils import keyword_overlap_score
from . import evidence as evidence_mod
from . import deep_read as deep_read_mod
from .fulltext import enrich_with_full_text
from .grounding import extract_citation_tags
from .ranker import has_readable_content, select_diverse
from .synthesizer import citation_map
_TARGET_RE = re.compile(r"(?<![A-Za-z0-9])\[?S(\d+)\]?(?![A-Za-z0-9])", re.I)
def _active_tags(sources: list[Source]) -> dict[str, Source]:
    return {f"S{i}": source for i, source in enumerate(sources, start=1)}
def _requested_tags(claim: str, sources: list[Source]) -> list[str]:
    by_tag = _active_tags(sources)
    out = []
    for m in _TARGET_RE.finditer(claim or ""):
        tag = f"S{m.group(1)}"
        if tag in by_tag and tag not in out:
            out.append(tag)
    words = {
        "first":"S1", "1st":"S1", "second":"S2", "2nd":"S2",
        "third":"S3", "3rd":"S3", "fourth":"S4", "4th":"S4",
        "fifth":"S5", "5th":"S5",
    }
    if not out:
        q = (claim or "").lower()
        for word, tag in words.items():
            if re.search(rf"\b(?:paper\s+)?{re.escape(word)}\b", q) and tag in by_tag:
                out.append(tag); break
    return out
def _case_affinity(claim: str, sources: list[Source], analyses) -> float:
    from ..textutils import keyword_overlap_score
    corpus = " ".join([claim] + [f"{s.title} {s.abstract}" for s in sources] + [
        " ".join(str(x) for x in [getattr(a, "problem", ""), getattr(a, "core_idea", ""), getattr(a, "method_summary", ""), getattr(a, "key_results", "")])
        for a in (analyses or [])
    ])
    case = " ".join([s.title + " " + s.abstract for s in sources] + [
        " ".join(str(x) for x in [getattr(a, "problem", ""), getattr(a, "core_idea", ""), getattr(a, "method_summary", ""), getattr(a, "key_results", "")])
        for a in (analyses or [])
    ])
    return keyword_overlap_score(claim, case)
_SYSTEM_PROMPT = (
    "You are a skeptical research assistant checking a specific CLAIM against a set of "
    "evidence snippets pulled from real papers. Do not assume the claim is true or false "
    "going in -- weigh only what the evidence below actually says. Some evidence may support "
    "the claim, some may contradict it, and some may be irrelevant (ignore irrelevant items). "
    "Respond with ONLY JSON: "
    '{"verdict": "well_supported|partially_supported|mixed|contradicted|insufficient_evidence", '
    '"reasoning": "1-4 compact sentences citing tags like [E3] for every specific point", '
    '"evidence_ids": ["E3", "E7"]}. '
    "verdict meanings: well_supported = multiple independent sources directly support it with "
    "no contradiction found here; partially_supported = supported under some conditions/caveats "
    "the evidence states; mixed = real disagreement between sources; contradicted = the "
    "evidence directly contradicts the claim; insufficient_evidence = the evidence given doesn't "
    "actually address the claim either way. Never invent an evidence id that wasn't given to you."
)
def _heuristic_verdict(claim: str, evidence: list[Evidence]) -> tuple[str, str]:
    """A crude, transparent fallback used only when no LLM is available:
    just says how much relevant evidence there is and lets the human read
    it themselves, rather than pretending to adjudicate the claim."""
    if not evidence:
        return "insufficient_evidence", "No LLM is configured and no relevant evidence was found to check this claim against."
    return (
        "insufficient_evidence",
        f"No LLM is configured to weigh the evidence, so a verdict can't be computed automatically. "
        f"{len(evidence)} potentially relevant snippet(s) are listed below for you to judge directly.",
    )
def _gather_new_evidence(
    claim: str, aggregator: SearchAggregator | None, max_new_sources: int
) -> tuple[list[Source], list[Evidence]]:
    aggregator = aggregator or SearchAggregator()
    queries = [claim, f"{claim} evidence study", f"{claim} experiment results"]
    found, _events = aggregator.run(queries)
    readable = [s for s in found if has_readable_content(s)]
    selected = select_diverse(readable, max_sources=max_new_sources, query=claim)
    new_evidence = evidence_mod.extract_all(claim, selected, max_total=max_new_sources * 4)
    for i, ev in enumerate(new_evidence, start=1):
        ev.id = f"V{i}"
    return selected, new_evidence
def verify_claim(
    claim: str,
    existing_sources: list[Source] | None = None,
    existing_evidence: list[Evidence] | None = None,
    llm: LLMClient | None = None,
    aggregator: SearchAggregator | None = None,
    allow_new_search: bool = True,
    relevance_floor: float = 0.12,
    analyses=None,
    verification_evidence_offset: int = 0,
    cancel_check=None,
    target_tags: list[str] | None = None,
    corpus_only: bool = False,
) -> ClaimVerification:
    """Check `claim` against `existing_evidence` (if given), optionally
    widening with a fresh bounded search when the existing pool looks thin
    or irrelevant. Returns a `ClaimVerification`; `new_sources` on the
    result should be merged into the caller's session state if the caller
    wants the newly-found papers to stick around."""
    claim = (claim or "").strip()
    if cancel_check is not None and cancel_check():
        raise RuntimeError("Chat cancelled")
    llm = llm or LLMClient()
    settings = get_settings()
    existing_sources = existing_sources or []
    existing_evidence = existing_evidence or []
    original_sources = list(existing_sources)
    requested = [str(x).upper() for x in (target_tags or _requested_tags(claim, original_sources))]
    if requested:
        tag_to_source = _active_tags(original_sources)
        allowed_ids = {tag_to_source[tag].id for tag in requested if tag in tag_to_source}
        existing_sources = [s for s in existing_sources if s.id in allowed_ids]
        existing_evidence = [e for e in existing_evidence if e.source_id in allowed_ids]
    elif corpus_only and existing_sources and _case_affinity(claim, existing_sources, analyses) < 0.18:
        return ClaimVerification(
            claim=claim, verdict="insufficient_evidence",
            reasoning="This question is outside the selected research corpus, so the research case does not contain evidence that can verify it.",
            sources_checked=len(existing_sources), llm_used=False,
        )
    if not claim:
        return ClaimVerification(claim=claim, verdict="insufficient_evidence", reasoning="No claim text was given.")
    scored_existing = sorted(
        existing_evidence, key=lambda e: keyword_overlap_score(claim, e.text), reverse=True
    )
    best_existing_score = keyword_overlap_score(claim, scored_existing[0].text) if scored_existing else 0.0
    new_sources: list[Source] = []
    if requested:
        pool_evidence = scored_existing[:20]
    else:
        pool_evidence = [e for e in scored_existing if keyword_overlap_score(claim, e.text) >= relevance_floor][:15]
    pool_sources = list(existing_sources)
    if allow_new_search and not requested and not corpus_only and (best_existing_score < 0.25 or len(pool_evidence) < 3):
        if cancel_check is not None and cancel_check():
            raise RuntimeError("Chat cancelled")
        new_sources, new_evidence = _gather_new_evidence(claim, aggregator, settings.verify_max_new_sources)
        existing_keys = {s.dedupe_key() for s in existing_sources}
        new_sources = [s for s in new_sources if s.dedupe_key() not in existing_keys]
        new_source_ids = {s.id for s in new_sources}
        new_evidence = [e for e in new_evidence if e.source_id in new_source_ids]
        for i, ev in enumerate(new_evidence, start=verification_evidence_offset + 1):
            ev.id = f"V{i}"
        if new_sources and llm.available:
            if cancel_check is not None and cancel_check():
                raise RuntimeError("Chat cancelled")
            enrich_with_full_text(new_sources, max_fetch=len(new_sources), cancel_check=cancel_check)
            if cancel_check is not None and cancel_check():
                raise RuntimeError("Chat cancelled")
            _, deep_evidence, _ = deep_read_mod.analyze_papers(claim, new_sources, llm=llm, top_k=len(new_sources), analysis_mode="deep", report_type="evidence_review")
            new_evidence.extend(deep_evidence)
            if cancel_check is not None and cancel_check():
                raise RuntimeError("Chat cancelled")
            for i, ev in enumerate(new_evidence, start=verification_evidence_offset + 1):
                ev.id = f"V{i}"
        pool_sources = existing_sources + new_sources
        combined = pool_evidence + new_evidence
        pool_evidence = sorted(combined, key=lambda e: keyword_overlap_score(claim, e.text) + 0.15 * e.relevance, reverse=True)[:20]
    if not pool_evidence:
        return ClaimVerification(
            claim=claim,
            verdict="insufficient_evidence",
            reasoning="No relevant evidence could be found for this claim, even after searching.",
            sources_checked=len(pool_sources),
            new_sources=new_sources,
            new_evidence=new_evidence if "new_evidence" in locals() else [],
        )
    cmap = {source.id: f"S{i}" for i, source in enumerate(original_sources, start=1)}
    next_tag = len(cmap) + 1
    for source in pool_sources:
        if source.id not in cmap:
            cmap[source.id] = f"S{next_tag}"
            next_tag += 1
    bullets = [f"{e.text} [{e.id}][{cmap.get(e.source_id, 'S?')}]" for e in pool_evidence]
    if analyses:
        allowed_analysis_ids = {s.id for s in pool_sources}
        for a in analyses[:10]:
            if a.source_id not in allowed_analysis_ids:
                continue
            inventory_text = " ".join(str(x.get("observation", "")) for x in (getattr(a, "evidence_inventory", []) or []))
            relevance_text = " ".join(str(x) for x in ((getattr(a, "research_relevance_map", {}) or {}).get("relevance_threads", []) if isinstance(getattr(a, "research_relevance_map", {}), dict) else []))
            candidate_blob = " ".join((a.problem, a.core_idea, a.key_results, a.limitations, inventory_text, relevance_text))
            if any(keyword_overlap_score(claim, text) > 0.10 for text in (candidate_blob, a.result_interpretation, a.not_established)):
                tag = cmap.get(a.source_id, "S?")
                obs = sorted(getattr(a, "evidence_inventory", []) or [], key=lambda x: float(x.get("importance", 0.0) or 0.0), reverse=True)[:10]
                obs_text = " | ".join(str(x.get("observation", "")) for x in obs if x.get("observation"))
                rel = getattr(a, "research_relevance_map", {}) or {}
                direct = rel.get("direct_answer", "") if isinstance(rel, dict) else ""
                bullets.append(f"DEEP PAPER RESEARCH MEMORY [{tag}]: problem={a.problem}; method={a.core_idea}; results={a.key_results}; limitations={a.limitations}; important observations={obs_text}; question relevance={direct}")
    prompt_tags = extract_citation_tags("\n".join(bullets))
    allowed = {e.id for e in pool_evidence} | prompt_tags
    allowed_evidence_ids = {e.id for e in pool_evidence}
    if llm.available:
        try:
            data = llm.complete_json(
                _SYSTEM_PROMPT,
                f"CLAIM: {claim}\n\nEVIDENCE:\n" + "\n".join(f"- {b}" for b in bullets),
                max_tokens=600,
            )
        except Exception:
            data = None
        if isinstance(data, dict) and data.get("verdict") in {
            "well_supported", "partially_supported", "mixed", "contradicted", "insufficient_evidence"
        }:
            reasoning = str(data.get("reasoning", ""))
            cited = extract_citation_tags(reasoning)
            submitted_ids = {i for i in (data.get("evidence_ids") or []) if isinstance(i, str)}
            if cited.issubset(allowed) and submitted_ids.issubset(allowed_evidence_ids):
                ids = [i for i in (data.get("evidence_ids") or []) if i in allowed_evidence_ids]
                verdict = data["verdict"]
                cited_source_ids = {e.source_id for e in pool_evidence if e.id in ids}
                if verdict == "well_supported" and len(cited_source_ids) < 2 and not requested:
                    verdict = "partially_supported" if cited_source_ids else "insufficient_evidence"
                return ClaimVerification(
                    claim=claim,
                    verdict=verdict,
                    reasoning=reasoning,
                    evidence_ids=ids or [e.id for e in pool_evidence[:5]],
                    sources_checked=len(pool_sources),
                    new_sources=new_sources,
                    new_evidence=new_evidence if "new_evidence" in locals() else [],
                    llm_used=True,
                )
    verdict, reasoning = _heuristic_verdict(claim, pool_evidence)
    return ClaimVerification(
        claim=claim,
        verdict=verdict,
        reasoning=reasoning + "\n\nTop matching evidence:\n" + "\n".join(f"- {b}" for b in bullets[:8]),
        evidence_ids=[e.id for e in pool_evidence[:8]],
        sources_checked=len(pool_sources),
        new_sources=new_sources,
        new_evidence=new_evidence if "new_evidence" in locals() else [],
        llm_used=False,
    )
