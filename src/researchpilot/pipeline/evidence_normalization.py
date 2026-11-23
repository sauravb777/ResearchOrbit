from __future__ import annotations
from collections import defaultdict
from ..schemas import AtomicEvidence, Evidence, PaperAnalysis, PaperEvidenceMap
def normalize_evidence(evidence: list[Evidence], analyses: list[PaperAnalysis]) -> tuple[list[AtomicEvidence], list[PaperEvidenceMap]]:
    by_source: dict[str, list[Evidence]] = defaultdict(list)
    for item in evidence:
        if item.text.strip():
            by_source[item.source_id].append(item)
    atomic: list[AtomicEvidence] = []
    maps: list[PaperEvidenceMap] = []
    kind_map = {
        "problem": "problem_evidence",
        "method": "method_evidence",
        "dataset": "dataset_evidence",
        "result": "result_evidence",
        "ablation": "ablation_evidence",
        "limitation": "limitation_evidence",
        "failure": "failure_evidence",
        "generalization": "generalization_evidence",
        "reproducibility": "reproducibility_evidence",
    }
    for source_id, items in by_source.items():
        pem = PaperEvidenceMap(paper_id=source_id)
        for item in items:
            atom = AtomicEvidence(
                id=item.id,
                source_id=item.source_id,
                evidence_type=item.kind,
                claim=item.text,
                conditions=item.conditions,
                source_locator=item.locator,
                verbatim_span=item.source_span or item.text,
                confidence=max(0.0, min(1.0, item.relevance)),
            )
            atomic.append(atom)
            field = kind_map.get(item.kind, "claim_evidence")
            getattr(pem, field).append(item.id)
        maps.append(pem)
    existing = {m.paper_id for m in maps}
    for analysis in analyses:
        if analysis.source_id not in existing:
            maps.append(PaperEvidenceMap(paper_id=analysis.source_id))
    return atomic, maps
