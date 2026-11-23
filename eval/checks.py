from __future__ import annotations
import re
from researchpilot.schemas import ResearchResult
_CITATION_RE = re.compile(r"\[(E\d+|S\d+)\]")
def no_dangling_citations(result: ResearchResult) -> tuple[bool, str]:
    valid_evidence_ids = {e.id for e in result.evidence}
    valid_source_labels = {f"S{i}" for i in range(1, len(result.sources) + 1)}
    valid = valid_evidence_ids | valid_source_labels
    used = set(_CITATION_RE.findall(result.report_markdown))
    dangling = used - valid
    return (not dangling, f"dangling citations: {sorted(dangling)}" if dangling else "all citations traceable")
def evidence_is_literal_substring_of_source(result: ResearchResult) -> tuple[bool, str]:
    sources_by_id = {s.id: s for s in result.sources}
    bad = []
    for e in result.evidence:
        src = sources_by_id.get(e.source_id)
        if src is None:
            bad.append(e.id)
            continue
        haystack = src.full_text or src.abstract
        normalize = lambda x: " ".join(str(x or "").split())
        if normalize(e.text) not in normalize(haystack):
            bad.append(e.id)
    return (not bad, f"evidence not found verbatim in source: {bad}" if bad else "all evidence verbatim")
def claims_reference_valid_evidence(result: ResearchResult) -> tuple[bool, str]:
    valid_ids = {e.id for e in result.evidence}
    bad = [c.id for c in result.claims if not set(c.evidence_ids).issubset(valid_ids)]
    return (not bad, f"claims citing unknown evidence: {bad}" if bad else "all claims traceable")
def metrics_match_actual_counts(result: ResearchResult) -> tuple[bool, str]:
    checks = {
        "sources_selected": len(result.sources) == result.metrics.get("sources_selected"),
        "evidence_count": len(result.evidence) == result.metrics.get("evidence_count"),
        "claim_count": len(result.claims) == result.metrics.get("claim_count"),
    }
    failed = [k for k, ok in checks.items() if not ok]
    return (not failed, f"metrics mismatch: {failed}" if failed else "metrics consistent with actual counts")
def report_is_nonempty_and_well_formed(result: ResearchResult) -> tuple[bool, str]:
    md = result.report_markdown
    ok = md.startswith("# ") and "## Executive Summary" in md and "## References" in md and len(md) > 100 and "## Evidence Base" not in md
    return (ok, "well-formed" if ok else "report missing expected structure")
ALL_CHECKS = [
    no_dangling_citations,
    evidence_is_literal_substring_of_source,
    claims_reference_valid_evidence,
    metrics_match_actual_counts,
    report_is_nonempty_and_well_formed,
]
