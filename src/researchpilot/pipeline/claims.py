from __future__ import annotations
import re
from ..llm import LLMClient
from ..schemas import Claim, Evidence
_SYSTEM_PROMPT = (
    "You are an evidence-first research analyst. You will be given a numbered list of "
    "evidence snippets, each already tagged with an id like E1, E2. Synthesize 6-10 "
    "high-value claims (comparisons, results, limitations, agreements, contradictions) "
    "using ONLY the supplied evidence. Every claim MUST cite the id(s) of the evidence "
    "it is based on. Never invent facts, numbers, or papers not present in the evidence. "
    'Respond with ONLY JSON: {"claims": [{"text": "...", "evidence_ids": ["E1"], '
    '"status": "supported|qualified|contested", "confidence": 0.0}]}'
)
_NUMBER_RE = re.compile(r"(?:\d+(?:\.\d+)?%|\b\d+(?:\.\d+)?\s*(?:points?|ms|s|sec|seconds?|minutes?|hours?|GB|MB|KB|M|B|x)\b|\bfrom\s+\d+(?:\.\d+)?\s+(?:to|->)\s+\d+(?:\.\d+)?\b)", re.I)
def _infer_evidence_status(evidence_ids: list[str], evidence_by_id: dict[str, Evidence], confidence: float) -> str:
    """Deterministic evidence-strength label for one claim:
    - "not_established": no valid evidence behind it, or confidence too low
      to present it with any weight.
    - "demonstrated": backed by a result/ablation-kind snippet that actually
      contains a number -- i.e. an experimental measurement, not just prose.
    - "inferred": the claim draws on evidence from more than one source --
      by definition a synthesis, not something any single paper stated.
    - "author_reported": the default -- backed by real evidence, but not a
      quantitative result and not a cross-paper synthesis.
    """
    cited = [evidence_by_id[eid] for eid in evidence_ids if eid in evidence_by_id]
    if not cited or confidence < 0.3:
        return "not_established"
    distinct_sources = {e.source_id for e in cited}
    if len(distinct_sources) > 1:
        return "inferred"
    if any(e.kind in {"result", "ablation"} and _NUMBER_RE.search(e.text) for e in cited):
        return "demonstrated"
    return "author_reported"
def _heuristic_claims(evidence: list[Evidence], max_claims: int = 10) -> list[Claim]:
    evidence_by_id = {e.id: e for e in evidence}
    ranked = sorted(evidence, key=lambda e: e.relevance, reverse=True)
    claims = []
    for i, ev in enumerate(ranked[:max_claims], start=1):
        confidence = round(min(0.95, 0.5 + 0.4 * ev.relevance), 2)
        claims.append(
            Claim(
                id=f"C{i}",
                text=ev.text,
                evidence_ids=[ev.id],
                status="supported",
                confidence=confidence,
                evidence_status=_infer_evidence_status([ev.id], evidence_by_id, confidence),
            )
        )
    return claims
def make_claims(question: str, evidence: list[Evidence], llm: LLMClient | None = None) -> tuple[list[Claim], bool]:
    """Returns (claims, llm_was_used)."""
    if not evidence:
        return [], False
    heuristic = _heuristic_claims(evidence)
    llm = llm or LLMClient()
    if not llm.available:
        return heuristic, False
    valid_ids = {e.id for e in evidence}
    evidence_by_id = {e.id: e for e in evidence}
    evidence_block = "\n".join(f"{e.id} ({e.kind}): {e.text}" for e in evidence)
    try:
        data = llm.complete_json(
            _SYSTEM_PROMPT,
            f"RESEARCH QUESTION: {question}\n\nEVIDENCE:\n{evidence_block}",
            max_tokens=1800,
            priority="report", tier="F",
        )
    except Exception:
        return heuristic, False
    if not isinstance(data, dict) or not isinstance(data.get("claims"), list):
        return heuristic, False
    accepted: list[Claim] = []
    for i, raw in enumerate(data["claims"], start=1):
        try:
            ids = [eid for eid in (raw.get("evidence_ids") or []) if eid in valid_ids]
            if not ids or not raw.get("text"):
                continue
            confidence = max(0.0, min(1.0, float(raw.get("confidence", 0.6))))
            claim = Claim(
                id=f"C{i}",
                text=str(raw["text"]).strip(),
                evidence_ids=ids,
                status=raw.get("status", "supported") if raw.get("status") in
                {"supported", "qualified", "contested", "unverified"} else "supported",
                confidence=confidence,
                evidence_status=_infer_evidence_status(ids, evidence_by_id, confidence),
            )
            accepted.append(claim)
        except (KeyError, TypeError, ValueError):
            continue
    if len(accepted) < 3:
        return heuristic, False
    return accepted, True
